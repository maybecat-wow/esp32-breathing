/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/* Get recv router csi — TCP output edition

   CSI frames are sent as newline-terminated CSV over a persistent TCP
   connection to a host running capture.py.

   Menuconfig (Component config → CSI TCP output):
     CONFIG_CSI_TCP_HOST  — host IP of the capture machine
     CONFIG_CSI_TCP_PORT  — TCP port (default 3490)

   Or set the defaults below directly if not using menuconfig.

   Wi-Fi provisioning:
     First boot: device enters ESPTouch mode. Use the Espressif EspTouch
     phone app to send credentials. They are saved to NVS and reused on
     subsequent boots.
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"

#include "nvs_flash.h"
#include "nvs.h"

#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_now.h"
#include "esp_smartconfig.h"
#include "esp_timer.h"

#include "lwip/inet.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "lwip/err.h"
#include "ping/ping_sock.h"

#include "esp_csi_gain_ctrl.h"

/* ── Configuration ──────────────────────────────────────────────────────── */

#define CONFIG_SEND_FREQUENCY   100

#ifndef CONFIG_CSI_TCP_HOST
#define CONFIG_CSI_TCP_HOST     "192.168.0.100"   /* ← set to your Mac's IP */
#endif

#ifndef CONFIG_CSI_TCP_PORT
#define CONFIG_CSI_TCP_PORT     3490
#endif

/* How large a single CSV line can be.
   Worst case: header (~120 B) + 128 CSI bytes as "%d," (~640 B) + overhead */
#define TCP_TX_BUF_SIZE         1024

/* Exponential backoff for TCP reconnection (milliseconds). Starts at
   TCP_RECONNECT_BACKOFF_MIN_MS after a failure, doubles on each failure, caps
   at TCP_RECONNECT_BACKOFF_MAX_MS. Resets to MIN on a successful connect. */
#define TCP_RECONNECT_BACKOFF_MIN_MS    500
#define TCP_RECONNECT_BACKOFF_MAX_MS    30000

/* Poll interval while the TCP connection is up (used to drive the 1 Hz
   heartbeat when no CSI has been sent recently). */
#define TCP_CONNECTED_POLL_MS           500

/* Heartbeat line is emitted when this long has passed since the last send. */
#define TCP_HEARTBEAT_INTERVAL_US       1000000  /* 1 s */

/* Wi-Fi reconnect backoff (milliseconds). Never erases NVS once we have
   successfully associated at least once — transient AP glitches overnight
   must not wipe credentials. */
#define WIFI_RECONNECT_BACKOFF_MIN_MS   500
#define WIFI_RECONNECT_BACKOFF_MAX_MS   30000

/* Auth-class failures allowed *before first successful association only*
   before NVS is erased and ESPTouch re-provisioning is triggered. */
#define WIFI_AUTH_FAIL_THRESHOLD        5

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
#define CSI_FORCE_LLTF          0
#endif

#define CONFIG_FORCE_GAIN       0

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || \
    CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6  || \
    CONFIG_IDF_TARGET_ESP32C61
#define CONFIG_GAIN_CONTROL     1
#endif

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const char *TAG = "csi_recv_router";

/* ── TCP + Wi-Fi state ──────────────────────────────────────────────────── */

static int          s_tcp_sock   = -1;          /* socket fd, -1 = disconnected */
static SemaphoreHandle_t s_tcp_mutex = NULL;    /* guards all fields below      */
static bool         s_header_sent = false;      /* have we sent the CSV header?  */
static int64_t      s_last_tx_us  = 0;          /* esp_timer_get_time() of last successful send */

/* Observability fields for the periodic heartbeat line. Updated from the
   Wi-Fi event handler; read from tcp_reconnect_task under s_tcp_mutex. */
static uint32_t     s_reconnect_count  = 0;     /* Wi-Fi associations since boot */
static int          s_last_disc_reason = 0;     /* last WIFI_EVENT_STA_DISCONNECTED reason */

/* ── TCP helpers ─────────────────────────────────────────────────────────── */

/**
 * Open a blocking TCP connection to the capture host.
 * Returns the socket fd on success, -1 on failure.
 */
static int tcp_connect(void)
{
    struct sockaddr_in dest = {
        .sin_family = AF_INET,
        .sin_port   = htons(CONFIG_CSI_TCP_PORT),
    };

    if (inet_pton(AF_INET, CONFIG_CSI_TCP_HOST, &dest.sin_addr) != 1) {
        ESP_LOGE(TAG, "Invalid host IP: %s", CONFIG_CSI_TCP_HOST);
        return -1;
    }

    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket() failed: errno %d", errno);
        return -1;
    }

    /* Keep-alive so the OS detects a dead connection.
       The default lwIP idle (~2 h) is useless for an overnight 100 Hz stream
       where the capture host could suspend its Wi-Fi at any moment. Tight
       values guarantee we notice a dead peer within ~idle + intvl * cnt
       = 5 + 2*3 = 11 s and trigger reconnect instead of silently piling
       bytes into kernel send buffers. */
    int keep = 1;
    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &keep, sizeof(keep));
    int keep_idle  = 5;   /* seconds of idle before probes start */
    int keep_intvl = 2;   /* seconds between probes */
    int keep_cnt   = 3;   /* dead after this many missed probes */
    setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE,  &keep_idle,  sizeof(keep_idle));
    setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, &keep_intvl, sizeof(keep_intvl));
    setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT,   &keep_cnt,   sizeof(keep_cnt));

    /* Send buffer — lwIP default is fine, but be explicit */
    int sndbuf = TCP_TX_BUF_SIZE * 4;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    ESP_LOGI(TAG, "Connecting to %s:%d …", CONFIG_CSI_TCP_HOST, CONFIG_CSI_TCP_PORT);
    if (connect(sock, (struct sockaddr *)&dest, sizeof(dest)) != 0) {
        ESP_LOGE(TAG, "connect() failed: errno %d", errno);
        close(sock);
        return -1;
    }

    ESP_LOGI(TAG, "TCP connected to %s:%d", CONFIG_CSI_TCP_HOST, CONFIG_CSI_TCP_PORT);
    return sock;
}

/**
 * Send exactly `len` bytes.  Closes the socket on error so the
 * reconnect task will re-establish the connection.
 * Must be called with s_tcp_mutex held.
 */
static void tcp_send_locked(const char *buf, int len)
{
    if (s_tcp_sock < 0 || len <= 0) return;

    int sent = 0;
    while (sent < len) {
        int n = send(s_tcp_sock, buf + sent, len - sent, 0);
        if (n < 0) {
            ESP_LOGW(TAG, "send() failed (errno %d) — closing socket", errno);
            close(s_tcp_sock);
            s_tcp_sock    = -1;
            s_header_sent = false;
            return;
        }
        sent += n;
    }
    /* Drives the 1 Hz heartbeat in tcp_reconnect_task. */
    s_last_tx_us = esp_timer_get_time();
}

/**
 * Emit a one-line heartbeat so the capture host can detect a silent link
 * even during long periods with no CSI (e.g. AP channel switch). Called
 * from tcp_reconnect_task when no send has happened in the last second.
 * Must be called with s_tcp_mutex held.
 */
static void tcp_send_heartbeat_locked(void)
{
    if (s_tcp_sock < 0 || !s_header_sent) return;

    wifi_ap_record_t ap = {0};
    int rssi = 0;
    if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) rssi = ap.rssi;

    int64_t now_us    = esp_timer_get_time();
    uint32_t uptime_s = (uint32_t)(now_us / 1000000);

    char line[96];
    int pos = snprintf(line, sizeof(line),
        "HEARTBEAT,%lld,%d,%lu,%lu,%d\n",
        (long long)now_us, rssi,
        (unsigned long)uptime_s,
        (unsigned long)s_reconnect_count,
        s_last_disc_reason);
    if (pos > 0 && pos < (int)sizeof(line)) {
        tcp_send_locked(line, pos);
    }
}

/**
 * Background task: keeps the TCP connection alive, reconnecting whenever
 * it drops, and emitting a 1 Hz heartbeat while connected.
 *
 * Reconnect uses exponential backoff (500 ms → 30 s) so a long host outage
 * doesn't hammer the network, but a quick bounce recovers within ~500 ms.
 */
static void tcp_reconnect_task(void *arg)
{
    uint32_t backoff_ms = TCP_RECONNECT_BACKOFF_MIN_MS;

    while (true) {
        xSemaphoreTake(s_tcp_mutex, portMAX_DELAY);
        bool connected = (s_tcp_sock >= 0);
        xSemaphoreGive(s_tcp_mutex);

        if (!connected) {
            int sock = tcp_connect();
            xSemaphoreTake(s_tcp_mutex, portMAX_DELAY);
            s_tcp_sock    = sock;
            s_header_sent = false;   /* re-send header on new connection */
            if (sock >= 0) {
                s_last_tx_us = esp_timer_get_time();
            }
            xSemaphoreGive(s_tcp_mutex);

            if (sock < 0) {
                vTaskDelay(pdMS_TO_TICKS(backoff_ms));
                backoff_ms <<= 1;
                if (backoff_ms > TCP_RECONNECT_BACKOFF_MAX_MS) {
                    backoff_ms = TCP_RECONNECT_BACKOFF_MAX_MS;
                }
            } else {
                backoff_ms = TCP_RECONNECT_BACKOFF_MIN_MS;
            }
        } else {
            /* Connected: poll slowly and emit a heartbeat if the CSI path
               has been quiet for ≥1 s. */
            xSemaphoreTake(s_tcp_mutex, portMAX_DELAY);
            int64_t idle_us = esp_timer_get_time() - s_last_tx_us;
            if (s_tcp_sock >= 0 && idle_us >= TCP_HEARTBEAT_INTERVAL_US) {
                tcp_send_heartbeat_locked();
            }
            xSemaphoreGive(s_tcp_mutex);

            vTaskDelay(pdMS_TO_TICKS(TCP_CONNECTED_POLL_MS));
        }
    }
}

/* ── CSI callback ────────────────────────────────────────────────────────── */

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG));
        return;
    }

    if (memcmp(info->mac, ctx, 6)) {
        return;
    }

    /* Drop frame if not yet connected */
    xSemaphoreTake(s_tcp_mutex, portMAX_DELAY);
    if (s_tcp_sock < 0) {
        xSemaphoreGive(s_tcp_mutex);
        return;
    }

    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    static int s_count = 0;
    float compensate_gain = 1.0f;
    static uint8_t agc_gain = 0;
    static int8_t  fft_gain = 0;

#if CONFIG_GAIN_CONTROL
    static uint8_t agc_gain_baseline = 0;
    static int8_t  fft_gain_baseline = 0;
    esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
    if (s_count < 100) {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
    } else if (s_count == 100) {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_gain_baseline, &fft_gain_baseline);
#if CONFIG_FORCE_GAIN
        esp_csi_gain_ctrl_set_rx_force_gain(agc_gain_baseline, fft_gain_baseline);
        ESP_LOGI(TAG, "fft_force %d, agc_force %d", fft_gain_baseline, agc_gain_baseline);
#endif
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);
    ESP_LOGD(TAG, "compensate_gain %f, agc_gain %d, fft_gain %d",
             compensate_gain, agc_gain, fft_gain);
#endif

    /* Build the CSV line into a stack buffer */
    char buf[TCP_TX_BUF_SIZE];
    int  pos = 0;

    /* ── CSV header (sent once per connection) ── */
    if (!s_header_sent) {
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
        pos = snprintf(buf, sizeof(buf),
            "type,id,mac,rssi,rate,noise_floor,fft_gain,agc_gain,"
            "channel,local_timestamp,sig_len,rx_state,len,first_word,data\n");
#else
        pos = snprintf(buf, sizeof(buf),
            "type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,"
            "not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,"
            "ampdu_cnt,channel,secondary_channel,local_timestamp,ant,"
            "sig_len,rx_state,len,first_word,data\n");
#endif
        tcp_send_locked(buf, pos);
        s_header_sent = true;
        pos = 0;
    }

    /* ── CSI_DATA row — metadata fields ── */
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
    pos += snprintf(buf + pos, sizeof(buf) - pos,
        "CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d",
        s_count, MAC2STR(info->mac),
        rx_ctrl->rssi, rx_ctrl->rate,
        rx_ctrl->noise_floor, fft_gain, agc_gain,
        rx_ctrl->channel, rx_ctrl->timestamp,
        rx_ctrl->sig_len, rx_ctrl->rx_state);
#else
    pos += snprintf(buf + pos, sizeof(buf) - pos,
        "CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
        s_count, MAC2STR(info->mac),
        rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
        rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing,
        rx_ctrl->not_sounding, rx_ctrl->aggregation,
        rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
        rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt,
        rx_ctrl->channel, rx_ctrl->secondary_channel,
        rx_ctrl->timestamp, rx_ctrl->ant,
        rx_ctrl->sig_len, rx_ctrl->rx_state);
#endif

    /* ── CSI data array ── */
#if (CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61) && CSI_FORCE_LLTF
    int16_t csi_val = (int16_t)(((((uint16_t)info->buf[1]) << 8) | info->buf[0]) << 4) >> 4;
    pos += snprintf(buf + pos, sizeof(buf) - pos,
        ",%d,%d,\"[%d",
        (info->len - 2) / 2, info->first_word_invalid,
        (int16_t)(compensate_gain * csi_val));

    /* Flush the header + first element before the loop */
    tcp_send_locked(buf, pos);
    pos = 0;

    for (int i = 2; i < (info->len - 2); i += 2) {
        csi_val = (int16_t)(((((uint16_t)info->buf[i+1]) << 8) | info->buf[i]) << 4) >> 4;
        pos += snprintf(buf + pos, sizeof(buf) - pos,
            ",%d", (int16_t)(compensate_gain * csi_val));
        /* Flush when approaching buffer limit */
        if (pos > TCP_TX_BUF_SIZE - 32) {
            tcp_send_locked(buf, pos);
            pos = 0;
        }
    }
#else
    pos += snprintf(buf + pos, sizeof(buf) - pos,
        ",%d,%d,\"[%d",
        info->len, info->first_word_invalid,
        (int16_t)(compensate_gain * info->buf[0]));

    /* Flush header + first element */
    tcp_send_locked(buf, pos);
    pos = 0;

    for (int i = 1; i < info->len; i++) {
        pos += snprintf(buf + pos, sizeof(buf) - pos,
            ",%d", (int16_t)(compensate_gain * info->buf[i]));
        if (pos > TCP_TX_BUF_SIZE - 32) {
            tcp_send_locked(buf, pos);
            pos = 0;
        }
    }
#endif

    /* ── Close the array and row ── */
    pos += snprintf(buf + pos, sizeof(buf) - pos, "]\"\n");
    tcp_send_locked(buf, pos);

    xSemaphoreGive(s_tcp_mutex);
    s_count++;
}

/* ── Wi-Fi CSI init ──────────────────────────────────────────────────────── */

static void wifi_csi_init(void)
{
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
    wifi_csi_config_t csi_config = {
        .enable                   = true,
        .acquire_csi_legacy       = true,
        .acquire_csi_force_lltf   = CSI_FORCE_LLTF,
        .acquire_csi_ht20         = true,
        .acquire_csi_ht40         = true,
        .acquire_csi_vht          = false,
        .acquire_csi_su           = false,
        .acquire_csi_mu           = false,
        .acquire_csi_dcm          = false,
        .acquire_csi_beamformed   = false,
        .acquire_csi_he_stbc_mode = 2,
        .val_scale_cfg            = 0,
        .dump_ack_en              = false,
        .reserved                 = false
    };
#elif CONFIG_IDF_TARGET_ESP32C6
    wifi_csi_config_t csi_config = {
        .enable                 = true,
        .acquire_csi_legacy     = true,
        .acquire_csi_ht20       = true,
        .acquire_csi_ht40       = true,
        .acquire_csi_su         = false,
        .acquire_csi_mu         = false,
        .acquire_csi_dcm        = false,
        .acquire_csi_beamformed = false,
        .acquire_csi_he_stbc    = 2,
        .val_scale_cfg          = false,
        .dump_ack_en            = false,
        .reserved               = false
    };
#else
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = false,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = true,
        .shift             = true,
    };
#endif

    static wifi_ap_record_t s_ap_info = {0};
    ESP_ERROR_CHECK(esp_wifi_sta_get_ap_info(&s_ap_info));
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, s_ap_info.bssid));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

/* ── Ping router (generates the CSI traffic) ─────────────────────────────── */

static esp_err_t wifi_ping_router_start(void)
{
    static esp_ping_handle_t ping_handle = NULL;

    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
    ping_config.count           = 0;
    ping_config.interval_ms     = 1000 / CONFIG_SEND_FREQUENCY;
    ping_config.task_stack_size = 3072;
    ping_config.data_size       = 1;

    esp_netif_ip_info_t local_ip;
    esp_netif_get_ip_info(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"), &local_ip);
    ESP_LOGI(TAG, "got ip:" IPSTR ", gw: " IPSTR,
             IP2STR(&local_ip.ip), IP2STR(&local_ip.gw));

    ping_config.target_addr.u_addr.ip4.addr = ip4_addr_get_u32(&local_ip.gw);
    ping_config.target_addr.type = ESP_IPADDR_TYPE_V4;

    esp_ping_callbacks_t cbs = {0};
    esp_ping_new_session(&ping_config, &cbs, &ping_handle);
    esp_ping_start(ping_handle);

    return ESP_OK;
}

/* ── Wi-Fi provisioning (ESPTouch) ──────────────────────────────────────── */

#define NVS_NAMESPACE   "wifi_creds"
#define NVS_KEY_SSID    "ssid"
#define NVS_KEY_PASS    "password"

#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1
#define ESPTOUCH_DONE_BIT   BIT2

static EventGroupHandle_t s_wifi_event_group;
static bool s_has_stored_creds  = false;
/* Latched true on first successful IP_EVENT_STA_GOT_IP and never cleared.
   Used to prevent NVS credential erasure after a mid-night disconnect —
   overnight glitches must not disable the sensor until morning. */
static bool s_ever_connected    = false;
static int  s_auth_fail_count   = 0;   /* consecutive auth-class failures before first association */
static int  s_reconnect_attempt = 0;   /* exponential-backoff counter, reset on GOT_IP */
static esp_timer_handle_t s_wifi_reconnect_timer = NULL;

/* Helpers for the reason-aware disconnect policy. */
static bool wifi_reason_is_auth(int reason)
{
    switch (reason) {
        case WIFI_REASON_AUTH_FAIL:
        case WIFI_REASON_AUTH_EXPIRE:
        case WIFI_REASON_HANDSHAKE_TIMEOUT:
        case WIFI_REASON_MIC_FAILURE:
        case WIFI_REASON_4WAY_HANDSHAKE_TIMEOUT:
            return true;
        default:
            return false;
    }
}

static void wifi_reconnect_timer_cb(void *arg)
{
    esp_err_t err = esp_wifi_connect();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_connect() in backoff cb: %s", esp_err_to_name(err));
    }
}

/* Schedule the next esp_wifi_connect() after `delay_ms` without blocking
   the default event loop task. Creates the one-shot timer on first use. */
static void wifi_schedule_reconnect(uint32_t delay_ms)
{
    if (s_wifi_reconnect_timer == NULL) {
        const esp_timer_create_args_t args = {
            .callback = &wifi_reconnect_timer_cb,
            .name     = "wifi_reconn",
        };
        if (esp_timer_create(&args, &s_wifi_reconnect_timer) != ESP_OK) {
            /* Fall back to an immediate connect; worst case the next
               disconnect schedules it again. */
            esp_wifi_connect();
            return;
        }
    }
    esp_timer_stop(s_wifi_reconnect_timer);   /* harmless if not armed */
    esp_timer_start_once(s_wifi_reconnect_timer, (uint64_t)delay_ms * 1000);
}

static bool load_wifi_credentials(char *ssid, size_t ssid_len,
                                   char *password, size_t pass_len)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &h) != ESP_OK) return false;
    bool ok = (nvs_get_str(h, NVS_KEY_SSID, ssid, &ssid_len) == ESP_OK &&
               nvs_get_str(h, NVS_KEY_PASS, password, &pass_len) == ESP_OK);
    nvs_close(h);
    return ok;
}

static void save_wifi_credentials(const char *ssid, const char *password)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_str(h, NVS_KEY_SSID, ssid);
    nvs_set_str(h, NVS_KEY_PASS, password);
    nvs_commit(h);
    nvs_close(h);
}

static void erase_wifi_credentials(void)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) return;
    nvs_erase_key(h, NVS_KEY_SSID);
    nvs_erase_key(h, NVS_KEY_PASS);
    nvs_commit(h);
    nvs_close(h);
}

static void smartconfig_task(void *parm)
{
    ESP_LOGI(TAG, "Starting ESPTouch provisioning — open the EspTouch app now");
    ESP_ERROR_CHECK(esp_smartconfig_set_type(SC_TYPE_ESPTOUCH));
    smartconfig_start_config_t cfg = SMARTCONFIG_START_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_smartconfig_start(&cfg));

    xEventGroupWaitBits(s_wifi_event_group,
                        ESPTOUCH_DONE_BIT | WIFI_FAIL_BIT,
                        true, false, portMAX_DELAY);

    ESP_LOGI(TAG, "ESPTouch provisioning complete");
    esp_smartconfig_stop();
    vTaskDelete(NULL);
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        if (s_has_stored_creds) {
            esp_wifi_connect();
        } else {
            xTaskCreate(smartconfig_task, "smartconfig", 4096, NULL, 3, NULL);
        }

    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t *ev =
            (wifi_event_sta_disconnected_t *)event_data;
        int reason = ev ? ev->reason : 0;

        /* Publish the reason so the heartbeat line can report it. The 64-bit
           s_last_tx_us and this 32-bit field are both read under s_tcp_mutex
           from the reconnect task, but the write here is a single-word store
           that any reader will either see old or new — either is fine. */
        s_last_disc_reason = reason;

        /* Compute exponential backoff: 500 ms, 1 s, 2 s, … capped at 30 s.
           Reset to zero on GOT_IP. */
        uint32_t delay_ms = WIFI_RECONNECT_BACKOFF_MIN_MS
                            << (s_reconnect_attempt < 6 ? s_reconnect_attempt : 6);
        if (delay_ms > WIFI_RECONNECT_BACKOFF_MAX_MS) {
            delay_ms = WIFI_RECONNECT_BACKOFF_MAX_MS;
        }
        s_reconnect_attempt++;

        if (s_ever_connected) {
            /* Once we've been online, NEVER erase credentials and NEVER give
               up — unattended overnight use must survive AP reboots, DFS
               radar switches, channel hops, host laptop sleeping its Wi-Fi,
               etc. Just keep retrying with backoff forever. */
            ESP_LOGW(TAG, "WiFi disconnected (reason %d) — retry in %lu ms",
                     reason, (unsigned long)delay_ms);
            wifi_schedule_reconnect(delay_ms);
        } else if (wifi_reason_is_auth(reason)) {
            /* Auth-class failures during the *initial* association are the
               only situation where we erase NVS and fall back to ESPTouch —
               this catches truly-bad stored credentials. Requires several in
               a row so a one-off handshake glitch doesn't trigger it. */
            s_auth_fail_count++;
            ESP_LOGW(TAG, "WiFi auth failure %d/%d (reason %d) during initial connect",
                     s_auth_fail_count, WIFI_AUTH_FAIL_THRESHOLD, reason);
            if (s_auth_fail_count >= WIFI_AUTH_FAIL_THRESHOLD) {
                ESP_LOGE(TAG, "Stored credentials rejected — will erase and re-provision");
                xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
            } else {
                wifi_schedule_reconnect(delay_ms);
            }
        } else {
            /* Non-auth failures during the initial connect (AP not found,
               beacon timeout, …) — keep trying forever. The AP might just be
               slow to come up after a power cycle. */
            ESP_LOGW(TAG, "WiFi initial connect failed (reason %d) — retry in %lu ms",
                     reason, (unsigned long)delay_ms);
            wifi_schedule_reconnect(delay_ms);
        }

    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_reconnect_attempt = 0;
        s_auth_fail_count   = 0;
        s_ever_connected    = true;
        s_reconnect_count++;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);

    } else if (event_base == SC_EVENT && event_id == SC_EVENT_SCAN_DONE) {
        ESP_LOGI(TAG, "ESPTouch: scan done");

    } else if (event_base == SC_EVENT && event_id == SC_EVENT_FOUND_CHANNEL) {
        ESP_LOGI(TAG, "ESPTouch: found channel");

    } else if (event_base == SC_EVENT && event_id == SC_EVENT_GOT_SSID_PSWD) {
        smartconfig_event_got_ssid_pswd_t *evt =
            (smartconfig_event_got_ssid_pswd_t *)event_data;

        wifi_config_t wifi_config = {0};
        memcpy(wifi_config.sta.ssid,     evt->ssid,     sizeof(wifi_config.sta.ssid));
        memcpy(wifi_config.sta.password, evt->password, sizeof(wifi_config.sta.password));
        if (evt->bssid_set) {
            wifi_config.sta.bssid_set = true;
            memcpy(wifi_config.sta.bssid, evt->bssid, sizeof(wifi_config.sta.bssid));
        }

        ESP_LOGI(TAG, "ESPTouch: got SSID \"%s\"", (char *)evt->ssid);
        save_wifi_credentials((char *)evt->ssid, (char *)evt->password);

        ESP_ERROR_CHECK(esp_wifi_disconnect());
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
        esp_wifi_connect();

    } else if (event_base == SC_EVENT && event_id == SC_EVENT_SEND_ACK_DONE) {
        xEventGroupSetBits(s_wifi_event_group, ESPTOUCH_DONE_BIT);
    }
}

static void wifi_init(void)
{
    s_wifi_event_group = xEventGroupCreate();

    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                               &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                               &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(SC_EVENT, ESP_EVENT_ANY_ID,
                                               &wifi_event_handler, NULL));

    /* Load stored credentials if available */
    char ssid[32] = {0}, password[64] = {0};
    if (load_wifi_credentials(ssid, sizeof(ssid), password, sizeof(password))) {
        ESP_LOGI(TAG, "Found stored credentials for \"%s\", connecting…", ssid);
        wifi_config_t wifi_config = {0};
        memcpy(wifi_config.sta.ssid,     ssid,     sizeof(wifi_config.sta.ssid));
        memcpy(wifi_config.sta.password, password, sizeof(wifi_config.sta.password));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
        s_has_stored_creds = true;
    } else {
        ESP_LOGI(TAG, "No stored credentials — will use ESPTouch");
    }

    ESP_ERROR_CHECK(esp_wifi_start());

    /* Block until we get an IP, or stored credentials are rejected as bad.
       WIFI_FAIL_BIT is now set ONLY after WIFI_AUTH_FAIL_THRESHOLD consecutive
       auth-class failures before the first successful association — so a
       flaky AP that eventually comes back won't trigger credential erasure. */
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                           false, false, portMAX_DELAY);

    if (bits & WIFI_FAIL_BIT) {
        ESP_LOGE(TAG, "Stored credentials look genuinely bad — erasing and restarting");
        erase_wifi_credentials();
        esp_restart();
    }
}

/* ── app_main ────────────────────────────────────────────────────────────── */

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init();

    /* Create the mutex before the CSI callback can fire */
    s_tcp_mutex = xSemaphoreCreateMutex();
    configASSERT(s_tcp_mutex);

    /* Reconnect task — runs at low priority, wakes every 200 ms */
    xTaskCreate(tcp_reconnect_task, "tcp_reconnect", 4096, NULL, 3, NULL);

    /* Give the reconnect task time to establish the first connection
       before CSI frames start arriving */
    ESP_LOGI(TAG, "Waiting for TCP connection to %s:%d …",
             CONFIG_CSI_TCP_HOST, CONFIG_CSI_TCP_PORT);
    for (int i = 0; i < 20; i++) {
        xSemaphoreTake(s_tcp_mutex, portMAX_DELAY);
        bool ok = (s_tcp_sock >= 0);
        xSemaphoreGive(s_tcp_mutex);
        if (ok) break;
        vTaskDelay(pdMS_TO_TICKS(500));
    }

    wifi_csi_init();
    wifi_ping_router_start();
}
