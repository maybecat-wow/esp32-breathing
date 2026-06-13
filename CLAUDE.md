# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project uses Wi-Fi Channel State Information (CSI) from an ESP32 to detect breathing rates. The ESP32 firmware pings the router at 100 Hz, captures the resulting CSI frames, and streams them as length-prefixed binary messages over a persistent TCP connection to a host machine. The host saves frames to a binary capture file, which is then processed offline to extract breathing rate estimates.

## Development Workflow

### Firmware (ESP-IDF)

Activate the IDF environment before any firmware commands:
```sh
source ~/.espressif/tools/activate_idf_v6.0.sh
```

Build and flash:
```sh
idf.py build
idf.py flash
idf.py monitor   # 921600 baud, configured in sdkconfig.defaults
```

Or combined:
```sh
idf.py build flash monitor
```

### Host-side capture

Start the TCP server (listens on port 3490 by default):
```sh
python capture.py -s csi_data.bin
python capture.py -s csi_data.bin -p 3490 -l csi_data_log.txt
```

Captures are written as `.bin` files (length-prefixed binary protocol) and can be replayed in Python via `csi_breathing.load_binary(path)`.

### Offline analysis

```sh
pip install numpy scipy matplotlib PyWavelets
python csi_breathing.py csi_data.bin --output-dir results
```

## Configuration Before Use

Two settings **must** be changed before flashing:

1. **Host IP** — edit `CONFIG_CSI_TCP_HOST` in `main/app_main.c` (line 55) or set it via `idf.py menuconfig` under *Component config → CSI TCP output*. The default `192.168.0.100` is a placeholder for the capture machine's IP.

2. **Wi-Fi provisioning** — first boot launches ESPTouch mode. Use the Espressif EspTouch phone app to send Wi-Fi credentials. They are stored in NVS and reused on subsequent boots. If stored credentials fail after `WIFI_MAX_RETRY` (5) attempts, the device erases them and restarts into ESPTouch mode again.

## Architecture

### Data flow

```
ESP32 (app_main.c)
  └─ pings router at 100 Hz  →  CSI frames via wifi_csi_rx_cb()
       └─ packed as binary messages  →  TCP stream to capture.py
            └─ csi_data.bin  →  csi_breathing.py  →  breathing rate plot
```

### Firmware (`main/app_main.c`)

- **`wifi_init()`** — initialises STA mode, registers event handlers, loads NVS credentials or falls back to ESPTouch (`smartconfig_task`), blocks until connected.
- **`tcp_reconnect_task()`** — FreeRTOS task that maintains the outbound TCP connection; reconnects automatically when it drops. Guards `s_tcp_sock` with `s_tcp_mutex`.
- **`wifi_csi_rx_cb()`** — CSI callback registered via `esp_wifi_set_csi_rx_cb()`. Filters frames by the AP's BSSID, builds a binary `CSI_FRAME` message (header + meta + raw CSI bytes) into a stack buffer, and enqueues it atomically via `xStreamBufferSend` for the writer task. No `snprintf`, no logging on the success path.
- **`tcp_send_session_info_locked()`** — emits one `SESSION_INFO` message per (re)connect from the reconnect task while holding `s_tcp_mutex`. Establishes `boot_id` so the host can tell a TCP reconnect from an ESP32 reboot.
- **`tcp_send_heartbeat_locked()`** — emits a binary `HEARTBEAT` at 1 Hz when no CSI has gone out for ≥1 s.
- **`wifi_ping_router_start()`** — starts a continuous ping to the gateway at `CONFIG_SEND_FREQUENCY` (100) Hz, which is the traffic source that generates CSI frames.
- **`env_task()`** (`main/env_sensors.c`, only when `CONFIG_ENV_ENABLE=y`) — samples the wired LDR + AM2302 sensors and emits a low-rate `MSG_ENV`. The (blocking) sensor read happens off `s_tcp_mutex`; the send holds it. Independent of the CSI path.

### Wire format

The ESP32 streams a sequence of length-prefixed binary messages
(little-endian, `u8 type | u16 length | payload`). Four message types:
SESSION_INFO (sent once per TCP (re)connect), CSI_FRAME (one per CSI
capture), HEARTBEAT (1 Hz idle), ENV (low-rate wired-sensor reading —
light + temp/humidity; 22-byte fixed payload, see Environment sensors).
The wire format is defined in `csi_protocol.py` and `main/csi_protocol.h`;
the two MUST stay in sync — the Python side pins struct sizes via unit
tests, and the firmware side pins them via `_Static_assert`.

SESSION_INFO carries a `sensor_flags` byte (bit0=LDR, bit1=AM2302) so
the host knows which env sensors are present. The old `reserved` byte
was repurposed for this; the struct stays 26 bytes.

Supported chips: classic ESP32 and ESP32-S3, LLTF-only (lltf_en=true,
htltf_en=false), 64 subcarriers × I/Q = 128 CSI bytes per frame.
Each CSI byte pair is (imag, real) signed int8 — same byte order as the
old CSV `data` array.

### Host capture (`capture.py`)

Single-connection TCP server. Acts as a near-dumb byte-pipe:
1. Reads each message's 3-byte header, then exactly `length` payload bytes. The first message MUST be `SESSION_INFO` or the socket is closed and logged.
2. Appends raw framed bytes (`type | length | payload`) verbatim to the output `.bin` file. The file on disk is byte-identical to the wire, so a capture can be replayed by feeding it back to `csi_breathing.load_binary`.
3. Peeks each message's meta via `csi_protocol.decode_*` to update `stats.json` (`frames_written`, `gap_count`, `total_gap_seconds`, etc.) — the raw CSI payload bytes are never parsed by capture.py itself.
4. `MAX_PAYLOAD_BYTES` cap (4096) closes the socket on any oversized length field; protocol-level offenses go to the sidecar log file.

### Analysis (`csi_breathing.py`)

Implements three approaches (default: ratio):
- **Ratio**: cross-subcarrier conjugate product `H[t,m] * conj(H[t,ref])` against the highest-SNR subcarrier; cancels common-mode hardware phase noise; BoI picks the best pair.
- **Amplitude**: Band-of-Interest (BoI) subcarrier selection based on amplitude variance.
- **Phase**: phase-difference approach with linear detrending.

Key constants at the top of the file control the breathing frequency band (default 0.1–0.5 Hz / 6–30 BPM), BoI selection, pilot/null subcarrier masking, and Hampel filter parameters for DC removal.

`MSG_ENV` samples land on a separate `CSIDataset.env` list (never in `frames`), exposed via `env_temps_c` / `env_rh` / `env_ldr_raw` / `env_lux` arrays. Temp/RH are decoded once on the wire (×10 ints, host just ÷10); `ldr_lux_estimate()` turns the LDR divider reading into a rough lux value via a CdS power-law — estimate only, never authoritative. Env rides the `esp_timer` clock, not the CSI `rx_ctrl` clock, so align by host wall-clock rather than subtracting timestamps.

### Environment sensors (`main/env_sensors.c`, optional)

Enabled by `CONFIG_ENV_ENABLE`. Wiring + design:
- **LDR (CdS light)** — divider `3V3 — LDR — node — R_fix(10k) — GND`, node into an **ADC1** channel (`CONFIG_ENV_LDR_ADC1_CHANNEL`). ADC1 is mandatory — ADC2 is dead while Wi-Fi is on. Firmware sends raw + calibrated mV.
- **AM2302 / DHT22 (temp/RH)** — 1-wire DATA on `CONFIG_ENV_AM2302_GPIO` with a 4.7k–10k pull-up to 3V3. Read via the **RMT** peripheral (not bit-bang) to survive FreeRTOS jitter, at most once per 2 s (sensor minimum) with last-good caching. Firmware validates the 5-byte checksum and emits °C/%RH ×10; on CRC failure it keeps the last good values and sets `am2302_status`.

Menuconfig: *CSI Breathing Monitor → Environment sensors* — `CONFIG_ENV_ENABLE`, `CONFIG_ENV_EMIT_HZ`, `CONFIG_ENV_LDR_*` (incl. `CONFIG_ENV_LDR_RFIX_OHM`, keep in sync with `LDR_R_FIX_OHM` in `csi_breathing.py`), `CONFIG_ENV_AM2302_*`.

## Component Dependencies

Declared in `main/idf_component.yml`:
- `idf >= 4.4.1`

`sdkconfig.defaults` sets: CSI enabled, 240 MHz CPU, 1000 Hz FreeRTOS tick, performance compiler optimisation, 921600 baud UART, 32/128 TX/RX Wi-Fi buffers.
