/*
 * plug_http — REST API the Flutter app's PlugClient speaks (V51-V53).
 *   GET  /            info
 *   GET  /state       live state
 *   POST /relay       atomic toggle
 *   POST /heartbeat   watchdog reset
 * Call after Wi-Fi association.
 */
#pragma once

void plug_http_start(void);
