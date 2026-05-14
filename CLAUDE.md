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
- **`wifi_csi_rx_cb()`** — CSI callback registered via `esp_wifi_set_csi_rx_cb()`. Filters frames by the AP's BSSID, applies gain compensation (`esp_csi_gain_ctrl`), and emits a `SESSION_INFO` message once per TCP (re)connect followed by one `CSI_FRAME` message per captured frame. Buffer management flushes before `TCP_TX_BUF_SIZE - 32` bytes to avoid overflow.
- **`wifi_ping_router_start()`** — starts a continuous ping to the gateway at `CONFIG_SEND_FREQUENCY` (100) Hz, which is the traffic source that generates CSI frames.
- Gain control is enabled on ESP32-S3 (and other supported gain-library targets); the first 100 frames are used to establish a gain baseline.

### Wire format

The ESP32 streams a sequence of length-prefixed binary messages
(little-endian, `u8 type | u16 length | payload`). Three message types:
SESSION_INFO (sent once per TCP (re)connect), CSI_FRAME (one per CSI
capture), HEARTBEAT (1 Hz idle). The wire format is defined in
`csi_protocol.py` and `main/csi_protocol.h`; the two MUST stay in sync —
the Python side pins struct sizes via unit tests, and the firmware side
pins them via `_Static_assert`.

Supported chips: classic ESP32 and ESP32-S3, LLTF-only (lltf_en=true,
htltf_en=false), 64 subcarriers × I/Q = 128 CSI bytes per frame.
Each CSI byte pair is (imag, real) signed int8 — same byte order as the
old CSV `data` array.

### Host capture (`capture.py`)

Single-connection TCP server. On each new connection it:
1. Reads the next `SESSION_INFO` message and logs the firmware build / session metadata.
2. Appends raw framed bytes (`type | length | payload`) to the output `.bin` file as they arrive, so the file remains a valid concatenation of binary messages across reconnects.
3. `CSI_FRAME` messages are validated by `csi_protocol.decode_*` (length, declared vs actual payload size) before being written; malformed or non-CSI traffic goes to the log file.

### Analysis (`csi_breathing.py`)

Implements three approaches (default: ratio):
- **Ratio**: cross-subcarrier conjugate product `H[t,m] * conj(H[t,ref])` against the highest-SNR subcarrier; cancels common-mode hardware phase noise; BoI picks the best pair.
- **Amplitude**: Band-of-Interest (BoI) subcarrier selection based on amplitude variance.
- **Phase**: phase-difference approach with linear detrending.

Key constants at the top of the file control the breathing frequency band (default 0.1–0.5 Hz / 6–30 BPM), BoI selection, pilot/null subcarrier masking, and Hampel filter parameters for DC removal.

## Component Dependencies

Declared in `main/idf_component.yml`:
- `idf >= 4.4.1`
- `esp_csi_gain_ctrl >= 0.1.4` (from the IDF Component Registry)

`sdkconfig.defaults` sets: CSI enabled, 240 MHz CPU, 1000 Hz FreeRTOS tick, performance compiler optimisation, 921600 baud UART, 32/128 TX/RX Wi-Fi buffers.
