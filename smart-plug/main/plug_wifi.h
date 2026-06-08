/*
 * plug_wifi — STA Wi-Fi with NVS credentials + ESPTouch provisioning.
 *
 * Lifted from the CSI firmware (../../main/app_main.c: wifi_init /
 * smartconfig_task / wifi_event_handler), trimmed of CSI/TCP specifics.
 * First boot with no stored credentials launches ESPTouch (use the official
 * Espressif EspTouch app). Credentials are stored in NVS and reused; they are
 * erased (→ re-provision) only after repeated auth failures during the very
 * first association. Relay state is never stored (V49).
 *
 * Blocks until an IP is obtained.
 */
#pragma once

void plug_wifi_init(void);
