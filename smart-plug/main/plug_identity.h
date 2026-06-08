/*
 * plug_identity — stable device identity for the smart-plug.
 *
 * `id`   : last 3 bytes of the STA MAC as 6 lowercase hex chars. Stable across
 *          reboots (pure function of the efuse MAC). Used for mDNS TXT `id`,
 *          the mDNS hostname, and JSON `id`.
 * `name` : user-friendly label, persisted in NVS, seeded from CONFIG_PLUG_NAME.
 * `fw`   : firmware semver (mDNS TXT `fw` / JSON `fw_ver`).
 * `hw`   : compile-time hardware string (JSON `hw`).
 */
#pragma once

#define PLUG_FW_VERSION "1.0.0"

/* Compute id from the efuse MAC and load name from NVS. Safe to call before
   Wi-Fi is started (esp_read_mac reads efuse directly). */
void plug_identity_init(void);

const char *plug_id(void);       /* e.g. "a3f1c2"      */
const char *plug_name(void);     /* e.g. "Kitchen"     */
const char *plug_fw_ver(void);   /* PLUG_FW_VERSION    */
const char *plug_hw(void);       /* "esp32-wroom" / "esp32-s3" */
