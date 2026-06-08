/*
 * Mecha smart-plug firmware — entry point.
 *
 * Implements the LAN HTTP + mDNS contract the Flutter app's Plug subsystem
 * speaks (see SPEC.md and csi-app/FIRMWARE.md). Controls a mains relay via a
 * GPIO whose pin number is set in menuconfig (CONFIG_PLUG_RELAY_GPIO), with a
 * fail-closed watchdog so the load can never be left energised by a crashed
 * app or a dead network.
 *
 * Bring-up order is safety-first: the relay is configured and forced OFF
 * before any network service can accept a command.
 *
 *   1. NVS
 *   2. relay init  → GPIO forced OFF, watchdog timer started   (V49, V50)
 *   3. identity    → id (MAC suffix), name (NVS), fw, hw
 *   4. Wi-Fi       → STA + ESPTouch provisioning, blocks until associated
 *   5. mDNS        → advertise _mechaplug._tcp                  (V48)
 *   6. HTTP server → GET / , GET /state , POST /relay , POST /heartbeat
 *
 * Menuconfig (idf.py menuconfig → Mecha Smart Plug):
 *   CONFIG_PLUG_RELAY_GPIO        relay control pin (default 21)
 *   CONFIG_PLUG_RELAY_ACTIVE_HIGH relay drive polarity
 *   CONFIG_PLUG_HTTP_PORT         REST API port (default 80)
 *   CONFIG_PLUG_NAME              initial plug name
 *   CONFIG_PLUG_WATCHDOG_S        fail-closed window (default 30)
 */
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_log.h"

#include "plug_relay.h"
#include "plug_identity.h"
#include "plug_wifi.h"
#include "plug_mdns.h"
#include "plug_http.h"

static const char *TAG = "smart_plug";

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

    /* Safety first: relay OFF and watchdog running before the network is up. */
    plug_relay_init();

    plug_identity_init();

    plug_wifi_init();   /* blocks until associated (or re-provisions) */

    plug_mdns_start();
    plug_http_start();

    ESP_LOGI(TAG, "smart-plug ready");
}
