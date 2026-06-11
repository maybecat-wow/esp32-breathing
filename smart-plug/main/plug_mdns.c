/*
 * plug_mdns — see plug_mdns.h.
 *
 * Service:  _mechaplug._tcp  on  CONFIG_PLUG_HTTP_PORT
 * TXT:      id, name, fw      (V48 — keys the Flutter app's discovery expects)
 * Hostname: mechaplug-<id>    (unique so multiple plugs coexist on one LAN)
 */
#include "plug_mdns.h"
#include "plug_identity.h"

#include <stdio.h>

#include "mdns.h"
#include "esp_log.h"

static const char *TAG = "plug_mdns";

void plug_mdns_start(void)
{
    ESP_ERROR_CHECK(mdns_init());

    char hostname[32];
    snprintf(hostname, sizeof(hostname), "mechaplug-%s", plug_id());
    ESP_ERROR_CHECK(mdns_hostname_set(hostname));
    ESP_ERROR_CHECK(mdns_instance_name_set(plug_name()));

    /* Note the key difference vs HTTP JSON: TXT uses `fw`, JSON uses `fw_ver`. */
    mdns_txt_item_t txt[] = {
        {"id",   (char *)plug_id()},
        {"name", (char *)plug_name()},
        {"fw",   (char *)plug_fw_ver()},
    };

    ESP_ERROR_CHECK(mdns_service_add(NULL, "_mechaplug", "_tcp",
                                     CONFIG_PLUG_HTTP_PORT,
                                     txt, sizeof(txt) / sizeof(txt[0])));

    ESP_LOGI(TAG, "advertising %s._mechaplug._tcp on port %d",
             hostname, CONFIG_PLUG_HTTP_PORT);
}
