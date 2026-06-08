/*
 * plug_relay — see plug_relay.h.
 */
#include "plug_relay.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"

static const char *TAG = "plug_relay";

#define WATCHDOG_US     ((int64_t)CONFIG_PLUG_WATCHDOG_S * 1000000)
/* How often the watchdog timer checks for expiry. Far finer than the window. */
#define WATCHDOG_CHECK_US   250000   /* 250 ms */

static SemaphoreHandle_t s_mutex = NULL;
static esp_timer_handle_t s_wd_timer = NULL;

/* Timestamp (esp_timer_get_time, microseconds) of the last /relay|/heartbeat. */
static int64_t s_last_cmd_us = 0;

/* GPIO level that means "relay energised" (ON), honouring drive polarity. */
#if CONFIG_PLUG_RELAY_ACTIVE_HIGH
#define LEVEL_ON   1
#define LEVEL_OFF  0
#else
#define LEVEL_ON   0
#define LEVEL_OFF  1
#endif

static inline int on_to_level(bool on) { return on ? LEVEL_ON : LEVEL_OFF; }
static inline bool level_to_on(int level) { return level == LEVEL_ON; }

/* Drive + read-back. Caller MUST hold s_mutex. */
static bool drive_locked(bool on)
{
    gpio_set_level(CONFIG_PLUG_RELAY_GPIO, on_to_level(on));
    return level_to_on(gpio_get_level(CONFIG_PLUG_RELAY_GPIO));
}

static void watchdog_cb(void *arg)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    if ((esp_timer_get_time() - s_last_cmd_us) >= WATCHDOG_US) {
        /* Force OFF. Leave s_last_cmd_us untouched so we stay expired (and
           don't spam): the relay simply remains de-energised until the next
           command arrives. */
        if (level_to_on(gpio_get_level(CONFIG_PLUG_RELAY_GPIO))) {
            drive_locked(false);
            ESP_LOGW(TAG, "watchdog fired — relay forced OFF");
        }
    }
    xSemaphoreGive(s_mutex);
}

void plug_relay_init(void)
{
    s_mutex = xSemaphoreCreateMutex();
    configASSERT(s_mutex);

    /* INPUT_OUTPUT so the /relay handler can read the driven level back (V51). */
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << CONFIG_PLUG_RELAY_GPIO,
        .mode         = GPIO_MODE_INPUT_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io));

    /* V49: fail-closed boot — drive OFF before anything else. */
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    drive_locked(false);
    s_last_cmd_us = esp_timer_get_time();
    xSemaphoreGive(s_mutex);

    const esp_timer_create_args_t args = {
        .callback = &watchdog_cb,
        .name     = "plug_wd",
    };
    ESP_ERROR_CHECK(esp_timer_create(&args, &s_wd_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(s_wd_timer, WATCHDOG_CHECK_US));

    ESP_LOGI(TAG, "relay on GPIO %d (active-%s), watchdog %d s",
             CONFIG_PLUG_RELAY_GPIO,
             LEVEL_ON ? "high" : "low",
             CONFIG_PLUG_WATCHDOG_S);
}

bool plug_relay_set(bool on, int64_t *ts_ms_out)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    bool actual = drive_locked(on);
    s_last_cmd_us = esp_timer_get_time();
    if (ts_ms_out) {
        *ts_ms_out = s_last_cmd_us / 1000;
    }
    xSemaphoreGive(s_mutex);
    return actual;
}

void plug_relay_heartbeat(void)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    s_last_cmd_us = esp_timer_get_time();   /* watchdog reset only (V52) */
    xSemaphoreGive(s_mutex);
}

bool plug_relay_get(void)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    bool on = level_to_on(gpio_get_level(CONFIG_PLUG_RELAY_GPIO));
    xSemaphoreGive(s_mutex);
    return on;
}

int64_t plug_relay_last_cmd_age_ms(void)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    int64_t age = (esp_timer_get_time() - s_last_cmd_us) / 1000;
    xSemaphoreGive(s_mutex);
    return age < 0 ? 0 : age;
}

int64_t plug_relay_watchdog_remaining_ms(void)
{
    int64_t wd_ms  = WATCHDOG_US / 1000;
    int64_t rem    = wd_ms - plug_relay_last_cmd_age_ms();
    if (rem < 0)     rem = 0;
    if (rem > wd_ms) rem = wd_ms;
    return rem;
}
