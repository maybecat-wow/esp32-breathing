/*
 * SPDX-FileCopyrightText: 2026
 * SPDX-License-Identifier: Apache-2.0
 *
 * Wired environment sensors for the CSI firmware:
 *   - LDR (CdS) on an ADC1 channel  (ADC2 is unusable while Wi-Fi is on — V3)
 *   - AM2302 / DHT22 on a 1-wire GPIO, captured with the RMT RX peripheral
 *     instead of a bit-bang loop so FreeRTOS jitter can't corrupt the
 *     timing-critical 40-bit frame (C3).
 *
 * Each reading is packed into a csi_env_t and sent by app_main's env_task as
 * a MSG_ENV message. This file never touches the TCP socket.
 */
#include "env_sensors.h"

#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_rom_sys.h"

#if CONFIG_ENV_LDR_ENABLE
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#endif

#if CONFIG_ENV_AM2302_ENABLE
#include "driver/gpio.h"
#include "driver/rmt_rx.h"
#endif

static const char *TAG = "env_sensors";

/* AM2302 must not be polled faster than this (datasheet minimum). */
#define AM2302_MIN_READ_INTERVAL_US   2000000   /* 2 s */

static uint32_t s_seq = 0;

/* ── LDR (ADC1) ──────────────────────────────────────────────────────────── */
#if CONFIG_ENV_LDR_ENABLE
static adc_oneshot_unit_handle_t s_adc       = NULL;
static adc_cali_handle_t         s_adc_cali  = NULL;
static bool                      s_cali_ok   = false;

static bool ldr_cali_init(void)
{
    adc_atten_t atten = ADC_ATTEN_DB_12;
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    adc_cali_curve_fitting_config_t cfg = {
        .unit_id  = ADC_UNIT_1,
        .chan     = CONFIG_ENV_LDR_ADC1_CHANNEL,
        .atten    = atten,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    if (adc_cali_create_scheme_curve_fitting(&cfg, &s_adc_cali) == ESP_OK) {
        return true;
    }
#endif
#if ADC_CALI_SCHEME_LINE_FITTING_SUPPORTED
    adc_cali_line_fitting_config_t lcfg = {
        .unit_id  = ADC_UNIT_1,
        .atten    = atten,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    if (adc_cali_create_scheme_line_fitting(&lcfg, &s_adc_cali) == ESP_OK) {
        return true;
    }
#endif
    return false;
}

static void ldr_init(void)
{
    adc_oneshot_unit_init_cfg_t unit_cfg = { .unit_id = ADC_UNIT_1 };
    if (adc_oneshot_new_unit(&unit_cfg, &s_adc) != ESP_OK) {
        ESP_LOGE(TAG, "ADC1 unit init failed — LDR disabled");
        s_adc = NULL;
        return;
    }
    adc_oneshot_chan_cfg_t chan_cfg = {
        .atten    = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(
        s_adc, CONFIG_ENV_LDR_ADC1_CHANNEL, &chan_cfg));
    s_cali_ok = ldr_cali_init();
    ESP_LOGI(TAG, "LDR on ADC1 ch%d (cal %s)",
             CONFIG_ENV_LDR_ADC1_CHANNEL, s_cali_ok ? "on" : "off");
}

/* Reads raw + (if calibrated) millivolts into the env struct. */
static void ldr_read(csi_env_t *out)
{
    if (!s_adc) return;
    int raw = 0;
    if (adc_oneshot_read(s_adc, CONFIG_ENV_LDR_ADC1_CHANNEL, &raw) != ESP_OK) {
        return;
    }
    out->ldr_raw = (uint16_t)raw;
    int mv = 0;
    if (s_cali_ok && adc_cali_raw_to_voltage(s_adc_cali, raw, &mv) == ESP_OK) {
        out->ldr_mv = (uint16_t)mv;
    }
}
#endif /* CONFIG_ENV_LDR_ENABLE */

/* ── AM2302 (DHT22) over RMT ─────────────────────────────────────────────── */
#if CONFIG_ENV_AM2302_ENABLE
static rmt_channel_handle_t s_rmt_rx = NULL;
static QueueHandle_t        s_rmt_q  = NULL;
/* 1 ack symbol + 40 data bits, with headroom for an EOF symbol and noise. */
static rmt_symbol_word_t    s_syms[64];

/* Cached last reading — served between hardware reads and held across a CRC
 * failure so the stream is never poisoned with garbage (V5). */
static int16_t  s_temp_c_x10 = 0;
static uint16_t s_rh_x10     = 0;
static uint8_t  s_am_status  = 3;   /* 3 = not_present until first success */
static int64_t  s_last_read_us = 0;

static bool IRAM_ATTR rmt_rx_done(rmt_channel_handle_t ch,
                                  const rmt_rx_done_event_data_t *edata,
                                  void *user_ctx)
{
    BaseType_t hp = pdFALSE;
    QueueHandle_t q = (QueueHandle_t)user_ctx;
    xQueueSendFromISR(q, edata, &hp);
    return hp == pdTRUE;
}

static void am2302_init(void)
{
    s_rmt_q = xQueueCreate(1, sizeof(rmt_rx_done_event_data_t));
    if (!s_rmt_q) {
        ESP_LOGE(TAG, "RMT queue alloc failed — AM2302 disabled");
        return;
    }
    rmt_rx_channel_config_t cfg = {
        .gpio_num         = CONFIG_ENV_AM2302_GPIO,
        .clk_src          = RMT_CLK_SRC_DEFAULT,
        .resolution_hz    = 1000000,   /* 1 MHz → 1 µs per tick */
        .mem_block_symbols = 64,
    };
    if (rmt_new_rx_channel(&cfg, &s_rmt_rx) != ESP_OK) {
        ESP_LOGE(TAG, "RMT RX channel init failed — AM2302 disabled");
        s_rmt_rx = NULL;
        return;
    }
    rmt_rx_event_callbacks_t cbs = { .on_recv_done = rmt_rx_done };
    ESP_ERROR_CHECK(rmt_rx_register_event_callbacks(s_rmt_rx, &cbs, s_rmt_q));
    ESP_ERROR_CHECK(rmt_enable(s_rmt_rx));
    ESP_LOGI(TAG, "AM2302 on GPIO%d via RMT", CONFIG_ENV_AM2302_GPIO);
}

/* Pull the data line low ≥1 ms then release — the AM2302 start request. */
static void am2302_send_start(void)
{
    gpio_set_direction(CONFIG_ENV_AM2302_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level(CONFIG_ENV_AM2302_GPIO, 0);
    esp_rom_delay_us(1500);
    gpio_set_level(CONFIG_ENV_AM2302_GPIO, 1);
    esp_rom_delay_us(30);
    gpio_set_direction(CONFIG_ENV_AM2302_GPIO, GPIO_MODE_INPUT);
}

/* Decode the captured RMT symbols into 5 bytes and validate the checksum.
 * Each data bit is a ~50 µs low followed by a high whose width encodes the
 * bit: ~26 µs = 0, ~70 µs = 1. We threshold the high (level==1) duration. */
static bool am2302_decode(const rmt_symbol_word_t *syms, size_t n)
{
    /* Need the ack symbol + 40 data bits. */
    if (n < 41) return false;

    /* Data bits start right after the sensor's ack (low+high). */
    const rmt_symbol_word_t *bits = syms + (n - 40);
    uint8_t b[5] = {0};
    for (int i = 0; i < 40; i++) {
        rmt_symbol_word_t s = bits[i];
        uint32_t high_us = s.level0 ? s.duration0 : s.duration1;
        int bit = (high_us > 45) ? 1 : 0;
        b[i / 8] = (uint8_t)((b[i / 8] << 1) | bit);
    }
    uint8_t sum = (uint8_t)(b[0] + b[1] + b[2] + b[3]);
    if (sum != b[4]) return false;

    uint16_t rh   = (uint16_t)((b[0] << 8) | b[1]);
    uint16_t rawt = (uint16_t)((b[2] << 8) | b[3]);
    int16_t  temp = (rawt & 0x8000) ? -(int16_t)(rawt & 0x7FFF) : (int16_t)rawt;

    s_rh_x10     = rh;
    s_temp_c_x10 = temp;
    return true;
}

/* Perform one hardware read, updating the cache + status. On any failure the
 * cached temp/RH are left untouched (V5); only the status changes. */
static void am2302_read_hw(void)
{
    if (!s_rmt_rx) { s_am_status = 3; return; }

    rmt_receive_config_t rxc = {
        .signal_range_min_ns = 1000,      /* <1 µs = glitch */
        .signal_range_max_ns = 200000,    /* >200 µs ends the frame */
    };
    am2302_send_start();
    if (rmt_receive(s_rmt_rx, s_syms, sizeof(s_syms), &rxc) != ESP_OK) {
        s_am_status = 2;   /* timeout / transport */
        return;
    }
    rmt_rx_done_event_data_t ev;
    if (xQueueReceive(s_rmt_q, &ev, pdMS_TO_TICKS(50)) != pdTRUE) {
        s_am_status = 2;   /* no response */
        return;
    }
    if (!am2302_decode(ev.received_symbols, ev.num_symbols)) {
        s_am_status = 1;   /* crc / frame error — keep last good */
        return;
    }
    s_am_status = 0;
}

static void am2302_fill(csi_env_t *out)
{
    int64_t now = esp_timer_get_time();
    if (s_last_read_us == 0 || now - s_last_read_us >= AM2302_MIN_READ_INTERVAL_US) {
        am2302_read_hw();
        s_last_read_us = now;
    }
    out->temp_c_x10    = s_temp_c_x10;
    out->rh_x10        = s_rh_x10;
    out->am2302_status = s_am_status;
}
#endif /* CONFIG_ENV_AM2302_ENABLE */

/* ── Public API ──────────────────────────────────────────────────────────── */

void env_sensors_init(void)
{
#if CONFIG_ENV_LDR_ENABLE
    ldr_init();
#endif
#if CONFIG_ENV_AM2302_ENABLE
    am2302_init();
#endif
}

uint8_t env_sensor_flags(void)
{
    uint8_t f = 0;
#if CONFIG_ENV_LDR_ENABLE
    f |= SENSOR_FLAG_LDR;
#endif
#if CONFIG_ENV_AM2302_ENABLE
    f |= SENSOR_FLAG_AM2302;
#endif
    return f;
}

bool env_read(csi_env_t *out)
{
    memset(out, 0, sizeof(*out));
    out->esp_time_us   = (uint64_t)esp_timer_get_time();
    out->seq           = s_seq++;
    out->am2302_status = 3;   /* default: not present */

    bool any = false;
#if CONFIG_ENV_LDR_ENABLE
    ldr_read(out);
    any = true;
#endif
#if CONFIG_ENV_AM2302_ENABLE
    am2302_fill(out);
    any = true;
#endif
    return any;
}
