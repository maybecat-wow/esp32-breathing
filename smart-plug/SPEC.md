# Mecha Smart-Plug — Firmware Spec

ESP-IDF firmware for an ESP32-based mains smart-plug. It implements the LAN
HTTP + mDNS contract that the **Flutter app's Plug subsystem already speaks**,
controls a mains relay through a single GPIO whose pin number is set in
`menuconfig`, and enforces a fail-closed watchdog so the load can never be left
energised by a crashed app or a dead network.

This document is the firmware author's source of truth. It is the smart-plug
analogue of `../docs/superpowers/specs/2026-05-14-binary-csi-protocol-design.md`
(the CSI firmware's design spec).

> **Canonical contract.** The app-facing wire contract is defined verbatim in
> `csi-app/FIRMWARE.md` (which in turn quotes `csi-app/SPEC.md` §I and
> §V48–V54). Where this spec restates JSON shapes, field names, status codes,
> mDNS keys, or invariants, `csi-app/FIRMWARE.md` is canonical and wins on any
> discrepancy. This spec adds the *implementation* design on top of it.

---

## 1. Overview & scope

- **Purpose.** Turn an ESP32 into a discoverable LAN smart-plug that the
  existing Flutter app can list, poll, and toggle. No cloud, no TLS, no auth —
  LAN-trust only, exactly as the app expects.
- **Target chips.** Classic ESP32 (`esp32-wroom`) and ESP32-S3, matching the
  chip support of the CSI firmware in this repo.
- **ESP-IDF.** v6.0 (same toolchain as the CSI firmware:
  `source ~/.espressif/tools/activate_idf_v6.0.sh`). Components used:
  `esp_http_server`, `json`/cJSON, `esp_wifi`, `esp_netif`, `esp_event`,
  `nvs_flash`, `esp_timer`, `driver` (all bundled), plus the managed
  `espressif/mdns` component (extracted from the IDF core in 5.x+).
- **Out of scope.** The CSI breathing pipeline (this is an independent
  firmware), in-app Wi-Fi provisioning (handled via the official Espressif
  EspTouch app — see §7), and any persistence of relay state (forbidden by
  V49).
- **Relationship to the CSI firmware.** This is a *separate* firmware project
  living in its own subdirectory; it does not share a build with the CSI app.
  It deliberately *reuses* the CSI firmware's Wi-Fi + ESPTouch + NVS code (§7).

---

## 2. Source tree

```
esp32-breathing/smart-plug/
├── CMakeLists.txt              # top-level IDF project (project(mecha_smart_plug))
├── sdkconfig.defaults          # see §9
├── main/
│   ├── app_main.c              # bring-up: NVS → relay → identity → wifi → mdns → http
│   ├── plug_relay.c / .h       # GPIO drive + fail-closed watchdog (safety core)
│   ├── plug_http.c / .h        # esp_http_server URI handlers + cJSON
│   ├── plug_mdns.c / .h        # mDNS advertise + TXT records
│   ├── plug_identity.c / .h    # id (MAC suffix), name (NVS), fw version, hw
│   ├── plug_wifi.c / .h        # STA + NVS creds + ESPTouch (lifted from CSI fw)
│   ├── Kconfig.projbuild       # menuconfig options (§4)
│   ├── CMakeLists.txt          # idf_component_register(SRCS ... REQUIRES ...)
│   └── idf_component.yml       # dependencies: idf + espressif/mdns
└── SPEC.md                     # this file
```

`esp_http_server` and `json` (cJSON) ship inside ESP-IDF. **mDNS** was extracted
from the IDF core into a managed component in IDF 5.x+, so it is pulled in via
`idf_component.yml` (`espressif/mdns`). No other third-party dependencies.

---

## 3. mDNS discovery

- **Service:** `_mechaplug._tcp.local` (V48). This string must match the app's
  `PlugDiscoveryService` browse target exactly.
- **Timing:** advertised *after* Wi-Fi association completes (an IP is bound),
  and re-advertised on reconnect.
- **SRV port:** `CONFIG_PLUG_HTTP_PORT` (default 80). The app reads the host
  from the mDNS A record and the port from the SRV record.
- **Instance/hostname:** derive a unique hostname from the plug `id`, e.g.
  `mechaplug-<id>` so multiple plugs coexist on one LAN.

### TXT records (all required)

| key    | example   | source                                            |
|--------|-----------|---------------------------------------------------|
| `id`   | `a3f1c2`  | stable plug id — last 3 bytes of the STA MAC, hex  |
| `name` | `Kitchen` | user-friendly label, read from NVS (§7), seeded from `CONFIG_PLUG_NAME` |
| `fw`   | `1.0.3`   | firmware semver constant                          |

`id` is computed once at boot via `esp_read_mac(..., ESP_MAC_WIFI_STA)` and
must be **stable across reboots** (it is a pure function of the MAC). `fw` here
is the same value returned as `fw_ver` over HTTP (note the key differs: `fw`
in TXT, `fw_ver` in JSON).

---

## 4. Kconfig / menuconfig options

Declared in `main/Kconfig.projbuild`, following the exact style of the CSI
firmware's `main/Kconfig.projbuild`. All are referenced in C as `CONFIG_PLUG_*`.

```kconfig
menu "Mecha Smart Plug"

    config PLUG_RELAY_GPIO
        int "Relay control GPIO pin"
        default 21
        range 0 48
        help
            GPIO pin that drives the low-current signal-relay coil (see the
            hardware topology in §6 / V54). 21 is a safe default on both
            ESP32 and ESP32-S3. Avoid input-only pins (ESP32 GPIO 34-39),
            strapping pins (ESP32 0/2/12/15; S3 0/3/45/46), and any pin wired
            to flash/PSRAM. Range 0-48 covers ESP32 (0-39) and ESP32-S3 (0-48).

    config PLUG_RELAY_ACTIVE_HIGH
        bool "Relay is active-high"
        default y
        help
            y: GPIO HIGH energises the relay (ON), LOW de-energises (OFF).
            n: inverted, for boards with an active-low relay driver. The
            fail-closed OFF level is always the de-energised level, whichever
            polarity this selects.

    config PLUG_HTTP_PORT
        int "HTTP server port"
        default 80
        range 1 65535
        help
            Port for the REST API. The app discovers this via the mDNS SRV
            record, so non-default ports work, but 80 matches FIRMWARE.md.

    config PLUG_NAME
        string "Initial plug name"
        default "Mecha Plug"
        help
            Seed value for the `name` mDNS TXT record and the JSON `name`
            field on first boot. Persisted to NVS thereafter (§7).

    config PLUG_WATCHDOG_S
        int "Fail-closed watchdog (seconds)"
        default 30
        range 5 600
        help
            If no POST /relay or POST /heartbeat arrives within this window,
            the relay is forced OFF (V50). Must match the app's WATCHDOG_S
            (30 s); changing it changes the safety margin the app assumes.

endmenu
```

The **relay pin is customizable in menuconfig** via `CONFIG_PLUG_RELAY_GPIO`,
which is the central requirement of this task. It is consumed in C as e.g.:

```c
gpio_config_t io = {
    .pin_bit_mask = 1ULL << CONFIG_PLUG_RELAY_GPIO,
    .mode = GPIO_MODE_INPUT_OUTPUT,   /* INPUT_OUTPUT so we can read back (V51) */
};
```

`GPIO_MODE_INPUT_OUTPUT` lets the `/relay` handler read the post-toggle level
back from the same pin to build a truthful response (V51).

---

## 5. HTTP endpoints

LAN-trust, no TLS, no auth. Served by `esp_http_server` on
`CONFIG_PLUG_HTTP_PORT`. JSON is parsed and built with the IDF-bundled cJSON
(`json` component). Field names below are the **exact snake_case wire names**
the app's `plug_client.dart` typedefs expect — do not rename.

Helper sources used by the handlers:
- `rssi` — `esp_wifi_sta_get_ap_info(&ap).rssi`.
- `uptime_s` — `esp_timer_get_time() / 1000000`.
- `ts_ms`, `last_cmd_age_ms`, `watchdog_remaining_ms` — from the relay/watchdog
  module's monotonic timestamps (§6).
- `id`, `name`, `fw_ver`, `hw` — from the identity module (§3).

### `GET /` — Info

Response `200`:

```json
{
  "id": "a3f1c2",
  "name": "Kitchen",
  "fw_ver": "1.0.3",
  "on": false,
  "uptime_s": 1234,
  "rssi": -58,
  "hw": "esp32-wroom"
}
```

- `hw` is a compile-time string per target (`"esp32-wroom"` / `"esp32-s3"`).
- `on` is read back from the relay GPIO, not a cached intent.

### `GET /state` — Live state

Response `200`:

```json
{
  "on": false,
  "uptime_s": 1234,
  "rssi": -58,
  "last_cmd_age_ms": 12450,
  "watchdog_remaining_ms": 17550
}
```

- `last_cmd_age_ms` = now − last successful `/relay`|`/heartbeat` timestamp.
- `watchdog_remaining_ms` = `CONFIG_PLUG_WATCHDOG_S*1000 − last_cmd_age_ms`,
  clamped to `[0, CONFIG_PLUG_WATCHDOG_S*1000]`.
- Polled by the app every 2 s while the Plugs tab is focused.

### `POST /relay` — Toggle relay

Request:

```json
{ "on": true }
```

Response `200`:

```json
{ "on": true, "ts_ms": 1234567 }
```

- **Atomic (V51):** GPIO write, watchdog reset, and bookkeeping happen in one
  critical section (§6). The response `on` is the **post-toggle level read
  back from the GPIO**, never the echoed request value.
- **`422` (V53)** when `on` is missing, or present but not a JSON boolean
  (reject numbers/strings like `1` or `"true"`). On `422` the relay and the
  watchdog are unchanged.
- **`5xx`** on internal fault (e.g. cJSON/heap failure). On any non-2xx the
  relay and watchdog are unchanged.
- `ts_ms` = `esp_timer_get_time()/1000` at the moment the toggle executed.

### `POST /heartbeat` — Reset watchdog

Empty body (the handler must not require or read one).

Response `200`:

```json
{ "watchdog_remaining_ms": 30000 }
```

- **State-preserving (V52):** resets the watchdog timer only; the relay is
  never touched.
- Sent by the app every 10 s while any plug is `on`.

---

## 6. Relay control + fail-closed watchdog (`plug_relay.c`)

This module is the safety core. All relay/watchdog state lives behind a single
FreeRTOS mutex (or `portMUX_TYPE` spinlock); HTTP handlers and the watchdog
timer are the only callers.

### State

- `s_relay_on` — last-written intent (sanity only; the truth is the GPIO).
- `s_last_cmd_us` — `esp_timer` timestamp of the last `/relay` or `/heartbeat`.
- A periodic `esp_timer` (e.g. 250 ms) that checks for watchdog expiry.

### Boot (V49 — fail-closed boot)

1. Configure `CONFIG_PLUG_RELAY_GPIO` as `GPIO_MODE_INPUT_OUTPUT`.
2. Drive it to the **OFF (de-energised)** level *before* mDNS or the HTTP
   server start, honouring `CONFIG_PLUG_RELAY_ACTIVE_HIGH` for polarity.
3. **Never** read or write relay state in NVS — power-loss recovery must leave
   the load de-energised.

### Toggle (V51 — atomic relay toggle)

Inside one critical section: write the GPIO to the requested level, read it
back, reset `s_last_cmd_us = now`, update `s_relay_on`. The handler returns the
read-back level as `on`.

### Watchdog (V50)

- `/relay` (any `on` value) and `/heartbeat` both set `s_last_cmd_us = now`.
- The `esp_timer` callback forces the relay OFF when
  `now − s_last_cmd_us ≥ CONFIG_PLUG_WATCHDOG_S` seconds. Forcing OFF also
  happens inside the critical section.
- Default window is 30 s, matching the app's `WATCHDOG_S`.

### State-preservation (V52, V53)

- `/heartbeat` resets `s_last_cmd_us` only; it must not enter the GPIO-write
  path.
- Any handler that returns non-2xx must do so *before* mutating GPIO or
  `s_last_cmd_us`, so a rejected/failed request leaves relay + watchdog intact.

### Hardware topology (V54)

```
3.3V GPIO  ──►  signal-relay coil
                signal-relay NO contact  ──►  contactor coil  ──►  220V AC load
```

- 3.3V logic MUST be galvanically isolated from the contactor coil supply.
- The ESP32 GPIO MUST NOT switch mains directly — always through the
  low-current signal relay → contactor chain.
- The signal-relay coil should be flyback-diode protected; the contactor coil
  typically runs at 5V/12V from an isolated supply.

---

## 7. Wi-Fi + provisioning

Reuse the CSI firmware's proven flow (`../main/app_main.c`: `wifi_init()` and
`smartconfig_task()`). It can be lifted near-verbatim:

- **STA mode**, event handlers registered, power save can stay default (no CSI
  load here).
- **NVS credentials** under a namespace (the CSI firmware uses `"wifi_creds"`
  with keys `"ssid"`/`"password"`). Loaded on boot; on success the firmware
  proceeds to bring up the services.
- **ESPTouch (SmartConfig)** on first boot when no credentials are stored. The
  user sends credentials with the official Espressif EspTouch phone app; they
  are written to NVS and reused on subsequent boots.
- **Re-provision** after repeated auth failures (the CSI firmware uses a retry
  threshold of 5 before erasing stored credentials and restarting into
  ESPTouch), so a moved/retired AP doesn't brick the plug.

### NVS usage rules for this firmware

- ✅ Persist Wi-Fi credentials (as above).
- ✅ Persist the plug `name` (so a future rename survives reboot). Seeded from
  `CONFIG_PLUG_NAME` when absent.
- ❌ **Never** persist relay state (V49).

> Note: `csi-app/FIRMWARE.md` lists provisioning as "out of scope for the
> firmware HTTP contract" — meaning the *app* offers no in-app provisioning
> surface. The firmware still needs a provisioning path, and reusing ESPTouch
> keeps it consistent with the EspTouch flow the app directs users to.

---

## 8. Task / bring-up layout

`app_main()` sequences bring-up so the relay is safe before the network is
reachable:

1. `nvs_flash_init()`.
2. **Relay init → forced OFF** (§6 boot) — before anything network-facing.
3. Identity init: compute `id` from MAC, load `name` from NVS, set `fw_ver`.
4. Wi-Fi init + provisioning; **block until associated** (§7).
5. Start the watchdog `esp_timer`.
6. Advertise mDNS (§3).
7. Start `esp_http_server` and register the four URI handlers (§5).

Concurrency:
- HTTP URI handlers execute on the `esp_http_server` task.
- The watchdog runs in an `esp_timer` callback.
- All relay/watchdog state is guarded by the single mutex in `plug_relay.c`, so
  a toggle racing the watchdog callback is serialised.

---

## 9. `sdkconfig.defaults`

Minimal, smart-plug-appropriate (the CSI-specific tuning does not apply):

- `CONFIG_ESP32_DEFAULT_CPU_FREQ_240=y` (and the S3 equivalent).
- Default console/monitor baud is fine (no high-rate streaming here); 921600
  optional for parity with the CSI firmware.
- `esp_http_server` defaults are sufficient (handful of URIs, tiny bodies).
- Leave Wi-Fi power save at default (`WIFI_PS_MIN_MODEM`).

---

## 10. Invariant traceability (V48–V54)

| Inv. | Requirement | Enforced by |
|------|-------------|-------------|
| V48 | mDNS service `_mechaplug._tcp.local`, TXT `id`/`name`/`fw` | §3 `plug_mdns.c`; service string + TXT built from identity module |
| V49 | Fail-closed boot; relay OFF; no NVS persistence of relay state | §6 boot step 2 (forced OFF before services); §7 NVS rules (relay state never written) |
| V50 | 30 s watchdog; `/relay` and `/heartbeat` reset it; force OFF on expiry | §6 watchdog `esp_timer` + `s_last_cmd_us`; `CONFIG_PLUG_WATCHDOG_S` |
| V51 | Atomic GPIO toggle + watchdog reset; response reflects read-back GPIO | §6 toggle critical section; `GPIO_MODE_INPUT_OUTPUT` read-back; §5 `/relay` |
| V52 | `/heartbeat` resets watchdog only, never relay | §5 `/heartbeat`; §6 heartbeat path excludes GPIO write |
| V53 | Any non-2xx leaves relay + watchdog unchanged; `422` on bad `on` | §5 `/relay` validation; §6 "validate/return before mutating" rule |
| V54 | GPIO → signal relay → contactor → mains; galvanic isolation; no direct mains | §6 hardware topology; documented for the hardware build |

---

## 11. Verification (for the later implementation pass)

Contract conformance is the bar. When the firmware is built:

1. **Field-name parity.** Every JSON field name/type and status code in §5
   must match `csi-app/FIRMWARE.md` and the typedefs in
   `csi-app/lib/plug/services/plug_client.dart` byte-for-byte (snake_case).
2. **End-to-end with the app.** Flash, provision via EspTouch, confirm the plug
   appears in the app's Plugs tab (mDNS), toggles, and that the switch reverts
   if `/relay` fails.
3. **Watchdog.** Turn the plug ON, kill the app / block heartbeats, confirm the
   relay forces OFF after `CONFIG_PLUG_WATCHDOG_S` and `/state` reports
   `watchdog_remaining_ms` counting down to 0.
4. **Manual probes** (no app needed):
   ```sh
   curl http://<plug-ip>/                                  # info
   curl http://<plug-ip>/state                             # state
   curl -X POST -d '{"on":true}'  http://<plug-ip>/relay   # -> {"on":true,...}
   curl -X POST -d '{"on":"yes"}' http://<plug-ip>/relay   # -> 422, relay unchanged
   curl -X POST http://<plug-ip>/heartbeat                 # watchdog reset, relay unchanged
   ```
5. **Pin config.** Change `CONFIG_PLUG_RELAY_GPIO` in `idf.py menuconfig`
   (*Mecha Smart Plug → Relay control GPIO pin*), rebuild, and confirm the new
   pin drives the relay.
