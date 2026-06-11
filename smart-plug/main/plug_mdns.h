/*
 * plug_mdns — advertise the plug on `_mechaplug._tcp.local` (V48).
 * Call after Wi-Fi association and after plug_identity_init().
 */
#pragma once

void plug_mdns_start(void);
