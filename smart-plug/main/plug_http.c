/*
 * plug_http — see plug_http.h.
 *
 * Field names below are the EXACT snake_case wire names the app's
 * plug_client.dart typedefs expect (fw_ver, uptime_s, last_cmd_age_ms,
 * watchdog_remaining_ms, ts_ms, hw). Do not rename.
 *
 * Safety mapping:
 *   POST /relay     422 on missing/non-bool `on`; 5xx on internal fault; the
 *                   response `on` is read back from the GPIO, never echoed
 *                   (V51, V53). Relay/watchdog untouched on any non-2xx.
 *   POST /heartbeat resets the watchdog only, ignores the body (V52).
 */
#include "plug_http.h"
#include "plug_identity.h"
#include "plug_relay.h"

#include <stdlib.h>
#include <string.h>

#include "esp_http_server.h"
#include "esp_wifi.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "cJSON.h"

static const char *TAG = "plug_http";

/* Largest /relay body we will buffer; {"on":true} is tiny. */
#define RELAY_BODY_MAX  128

static int current_rssi(void)
{
    wifi_ap_record_t ap;
    return (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) ? ap.rssi : 0;
}

static int64_t uptime_s(void)
{
    return esp_timer_get_time() / 1000000;
}

/* Serialise `root`, send it as application/json, and free both. Falls back to
   500 if serialisation fails. Always consumes `root`. */
static esp_err_t send_json(httpd_req_t *req, cJSON *root)
{
    char *body = root ? cJSON_PrintUnformatted(root) : NULL;
    cJSON_Delete(root);
    if (!body) {
        httpd_resp_set_status(req, "500 Internal Server Error");
        httpd_resp_set_type(req, "application/json");
        httpd_resp_sendstr(req, "{\"error\":\"oom\"}");
        return ESP_OK;
    }
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, body);
    free(body);
    return ESP_OK;
}

static esp_err_t send_status(httpd_req_t *req, const char *status, const char *body)
{
    httpd_resp_set_status(req, status);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, body);
    return ESP_OK;
}

/* GET / — Info */
static esp_err_t info_handler(httpd_req_t *req)
{
    cJSON *root = cJSON_CreateObject();
    if (!root) return send_status(req, "500 Internal Server Error", "{\"error\":\"oom\"}");
    cJSON_AddStringToObject(root, "id", plug_id());
    cJSON_AddStringToObject(root, "name", plug_name());
    cJSON_AddStringToObject(root, "fw_ver", plug_fw_ver());
    cJSON_AddBoolToObject(root, "on", plug_relay_get());
    cJSON_AddNumberToObject(root, "uptime_s", (double)uptime_s());
    cJSON_AddNumberToObject(root, "rssi", current_rssi());
    cJSON_AddStringToObject(root, "hw", plug_hw());
    return send_json(req, root);
}

/* GET /state — Live state */
static esp_err_t state_handler(httpd_req_t *req)
{
    cJSON *root = cJSON_CreateObject();
    if (!root) return send_status(req, "500 Internal Server Error", "{\"error\":\"oom\"}");
    cJSON_AddBoolToObject(root, "on", plug_relay_get());
    cJSON_AddNumberToObject(root, "uptime_s", (double)uptime_s());
    cJSON_AddNumberToObject(root, "rssi", current_rssi());
    cJSON_AddNumberToObject(root, "last_cmd_age_ms", (double)plug_relay_last_cmd_age_ms());
    cJSON_AddNumberToObject(root, "watchdog_remaining_ms",
                            (double)plug_relay_watchdog_remaining_ms());
    return send_json(req, root);
}

/* POST /relay — atomic toggle */
static esp_err_t relay_handler(httpd_req_t *req)
{
    /* Reject oversized bodies as bad input (422) before touching any state. */
    if (req->content_len > RELAY_BODY_MAX) {
        return send_status(req, "422 Unprocessable Entity",
                           "{\"error\":\"body too large\"}");
    }

    char buf[RELAY_BODY_MAX + 1];
    int received = 0;
    while (received < req->content_len) {
        int r = httpd_req_recv(req, buf + received, req->content_len - received);
        if (r == HTTPD_SOCK_ERR_TIMEOUT) continue;
        if (r <= 0) {
            /* Transport fault — relay/watchdog untouched (V53). */
            return send_status(req, "500 Internal Server Error",
                               "{\"error\":\"recv\"}");
        }
        received += r;
    }
    buf[received] = '\0';

    cJSON *root = cJSON_Parse(buf);
    if (!root) {
        return send_status(req, "422 Unprocessable Entity",
                           "{\"error\":\"invalid json\"}");
    }
    cJSON *on = cJSON_GetObjectItemCaseSensitive(root, "on");
    if (!cJSON_IsBool(on)) {
        cJSON_Delete(root);
        /* Missing or not a JSON boolean — relay/watchdog untouched (V53). */
        return send_status(req, "422 Unprocessable Entity",
                           "{\"error\":\"on must be a boolean\"}");
    }
    bool want = cJSON_IsTrue(on);
    cJSON_Delete(root);

    int64_t ts_ms = 0;
    bool actual = plug_relay_set(want, &ts_ms);   /* atomic toggle + wd reset */

    cJSON *resp = cJSON_CreateObject();
    if (!resp) return send_status(req, "500 Internal Server Error", "{\"error\":\"oom\"}");
    cJSON_AddBoolToObject(resp, "on", actual);     /* read-back, not request */
    cJSON_AddNumberToObject(resp, "ts_ms", (double)ts_ms);
    return send_json(req, resp);
}

/* POST /heartbeat — watchdog reset only. Body intentionally ignored (V52). */
static esp_err_t heartbeat_handler(httpd_req_t *req)
{
    plug_relay_heartbeat();
    cJSON *root = cJSON_CreateObject();
    if (!root) return send_status(req, "500 Internal Server Error", "{\"error\":\"oom\"}");
    cJSON_AddNumberToObject(root, "watchdog_remaining_ms",
                            (double)plug_relay_watchdog_remaining_ms());
    return send_json(req, root);
}

void plug_http_start(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port    = CONFIG_PLUG_HTTP_PORT;
    config.lru_purge_enable = true;

    httpd_handle_t server = NULL;
    ESP_ERROR_CHECK(httpd_start(&server, &config));

    const httpd_uri_t uris[] = {
        { .uri = "/",          .method = HTTP_GET,  .handler = info_handler },
        { .uri = "/state",     .method = HTTP_GET,  .handler = state_handler },
        { .uri = "/relay",     .method = HTTP_POST, .handler = relay_handler },
        { .uri = "/heartbeat", .method = HTTP_POST, .handler = heartbeat_handler },
    };
    for (size_t i = 0; i < sizeof(uris) / sizeof(uris[0]); i++) {
        ESP_ERROR_CHECK(httpd_register_uri_handler(server, &uris[i]));
    }

    ESP_LOGI(TAG, "HTTP server up on port %d", CONFIG_PLUG_HTTP_PORT);
}
