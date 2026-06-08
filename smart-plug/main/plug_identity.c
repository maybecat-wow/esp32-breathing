/*
 * plug_identity — see plug_identity.h.
 */
#include "plug_identity.h"

#include <stdio.h>
#include <string.h>

#include "esp_mac.h"
#include "esp_log.h"
#include "nvs.h"

static const char *TAG = "plug_id";

/* NVS store for plug config that must survive reboots (the user-visible name).
   Relay state is NEVER stored here (V49). */
#define NVS_NAMESPACE   "plug_cfg"
#define NVS_KEY_NAME    "name"

static char s_id[7]    = "000000";          /* 6 hex chars + NUL          */
static char s_name[64] = CONFIG_PLUG_NAME;  /* user label; NVS-backed     */

#if CONFIG_IDF_TARGET_ESP32S3
static const char *s_hw = "esp32-s3";
#else
static const char *s_hw = "esp32-wroom";
#endif

static void load_name(void)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &h) == ESP_OK) {
        size_t len = sizeof(s_name);
        esp_err_t err = nvs_get_str(h, NVS_KEY_NAME, s_name, &len);
        nvs_close(h);
        if (err == ESP_OK) {
            return;  /* stored name wins */
        }
    }

    /* No stored name yet — seed NVS from CONFIG_PLUG_NAME so a later rename
       persists and the default survives a config change. */
    strlcpy(s_name, CONFIG_PLUG_NAME, sizeof(s_name));
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h) == ESP_OK) {
        nvs_set_str(h, NVS_KEY_NAME, s_name);
        nvs_commit(h);
        nvs_close(h);
    }
}

void plug_identity_init(void)
{
    uint8_t mac[6] = {0};
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(s_id, sizeof(s_id), "%02x%02x%02x", mac[3], mac[4], mac[5]);

    load_name();

    ESP_LOGI(TAG, "id=%s name=\"%s\" fw=%s hw=%s",
             s_id, s_name, PLUG_FW_VERSION, s_hw);
}

const char *plug_id(void)     { return s_id; }
const char *plug_name(void)   { return s_name; }
const char *plug_fw_ver(void) { return PLUG_FW_VERSION; }
const char *plug_hw(void)     { return s_hw; }
