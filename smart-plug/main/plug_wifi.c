/*
 * plug_wifi — see plug_wifi.h. Adapted from the CSI firmware's Wi-Fi +
 * ESPTouch provisioning (../../main/app_main.c).
 */
#include "plug_wifi.h"

#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "nvs.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_smartconfig.h"
#include "esp_timer.h"
#include "esp_log.h"

static const char *TAG = "plug_wifi";

#define NVS_NAMESPACE   "wifi_creds"
#define NVS_KEY_SSID    "ssid"
#define NVS_KEY_PASS    "password"

#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1
#define ESPTOUCH_DONE_BIT   BIT2

#define WIFI_RECONNECT_BACKOFF_MIN_MS   500
#define WIFI_RECONNECT_BACKOFF_MAX_MS   30000

/* Auth-class failures tolerated before the first ever association before NVS
   is erased and ESPTouch is re-triggered. */
#define WIFI_AUTH_FAIL_THRESHOLD        5

static EventGroupHandle_t s_wifi_event_group;
static bool s_has_stored_creds  = false;
static bool s_ever_connected    = false;   /* latched on first GOT_IP */
static int  s_auth_fail_count   = 0;
static int  s_reconnect_attempt = 0;
static esp_timer_handle_t s_reconnect_timer = NULL;

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

static void reconnect_timer_cb(void *arg)
{
    esp_err_t err = esp_wifi_connect();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_connect() in backoff cb: %s", esp_err_to_name(err));
    }
}

static void schedule_reconnect(uint32_t delay_ms)
{
    if (s_reconnect_timer == NULL) {
        const esp_timer_create_args_t args = {
            .callback = &reconnect_timer_cb,
            .name     = "wifi_reconn",
        };
        if (esp_timer_create(&args, &s_reconnect_timer) != ESP_OK) {
            esp_wifi_connect();
            return;
        }
    }
    esp_timer_stop(s_reconnect_timer);   /* harmless if not armed */
    esp_timer_start_once(s_reconnect_timer, (uint64_t)delay_ms * 1000);
}

static bool load_credentials(char *ssid, size_t ssid_len,
                             char *password, size_t pass_len)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &h) != ESP_OK) return false;
    bool ok = (nvs_get_str(h, NVS_KEY_SSID, ssid, &ssid_len) == ESP_OK &&
               nvs_get_str(h, NVS_KEY_PASS, password, &pass_len) == ESP_OK);
    nvs_close(h);
    return ok;
}

static void save_credentials(const char *ssid, const char *password)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_str(h, NVS_KEY_SSID, ssid);
    nvs_set_str(h, NVS_KEY_PASS, password);
    nvs_commit(h);
    nvs_close(h);
}

static void erase_credentials(void)
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

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        if (s_has_stored_creds) {
            esp_wifi_connect();
        } else {
            xTaskCreate(smartconfig_task, "smartconfig", 4096, NULL, 3, NULL);
        }

    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t *ev = (wifi_event_sta_disconnected_t *)data;
        int reason = ev ? ev->reason : 0;

        uint32_t delay_ms = WIFI_RECONNECT_BACKOFF_MIN_MS
                            << (s_reconnect_attempt < 6 ? s_reconnect_attempt : 6);
        if (delay_ms > WIFI_RECONNECT_BACKOFF_MAX_MS) {
            delay_ms = WIFI_RECONNECT_BACKOFF_MAX_MS;
        }
        s_reconnect_attempt++;

        if (s_ever_connected) {
            /* Once online, never erase credentials and never give up. */
            ESP_LOGW(TAG, "WiFi disconnected (reason %d) — retry in %lu ms",
                     reason, (unsigned long)delay_ms);
            schedule_reconnect(delay_ms);
        } else if (wifi_reason_is_auth(reason)) {
            s_auth_fail_count++;
            ESP_LOGW(TAG, "WiFi auth failure %d/%d (reason %d) during initial connect",
                     s_auth_fail_count, WIFI_AUTH_FAIL_THRESHOLD, reason);
            if (s_auth_fail_count >= WIFI_AUTH_FAIL_THRESHOLD) {
                ESP_LOGE(TAG, "Stored credentials rejected — will erase and re-provision");
                xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
            } else {
                schedule_reconnect(delay_ms);
            }
        } else {
            ESP_LOGW(TAG, "WiFi initial connect failed (reason %d) — retry in %lu ms",
                     reason, (unsigned long)delay_ms);
            schedule_reconnect(delay_ms);
        }

    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_reconnect_attempt = 0;
        s_auth_fail_count   = 0;
        s_ever_connected    = true;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);

    } else if (base == SC_EVENT && id == SC_EVENT_GOT_SSID_PSWD) {
        smartconfig_event_got_ssid_pswd_t *evt =
            (smartconfig_event_got_ssid_pswd_t *)data;

        wifi_config_t wifi_config = {0};
        memcpy(wifi_config.sta.ssid,     evt->ssid,     sizeof(wifi_config.sta.ssid));
        memcpy(wifi_config.sta.password, evt->password, sizeof(wifi_config.sta.password));
        if (evt->bssid_set) {
            wifi_config.sta.bssid_set = true;
            memcpy(wifi_config.sta.bssid, evt->bssid, sizeof(wifi_config.sta.bssid));
        }

        ESP_LOGI(TAG, "ESPTouch: got SSID \"%s\"", (char *)evt->ssid);
        save_credentials((char *)evt->ssid, (char *)evt->password);

        ESP_ERROR_CHECK(esp_wifi_disconnect());
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
        esp_wifi_connect();

    } else if (base == SC_EVENT && id == SC_EVENT_SEND_ACK_DONE) {
        xEventGroupSetBits(s_wifi_event_group, ESPTOUCH_DONE_BIT);
    }
}

void plug_wifi_init(void)
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

    char ssid[32] = {0}, password[64] = {0};
    if (load_credentials(ssid, sizeof(ssid), password, sizeof(password))) {
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

    /* Block until we get an IP, or stored credentials are rejected as bad. */
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                           false, false, portMAX_DELAY);

    if (bits & WIFI_FAIL_BIT) {
        ESP_LOGE(TAG, "Stored credentials look genuinely bad — erasing and restarting");
        erase_credentials();
        esp_restart();
    }
}
