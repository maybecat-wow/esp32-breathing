# esp32-breathing

Contactless breathing rate detection using Wi-Fi Channel State Information (CSI) on an ESP32.

The firmware pings the router at 100 Hz and captures the resulting CSI frames, which encode how the wireless channel changes over time. Chest movement during breathing produces subtle, periodic perturbations in the channel — this project extracts that signal and estimates breathing rate in breaths per minute (BPM).

> **This repo also contains a second, independent firmware:** a Wi-Fi **smart plug** under [`smart-plug/`](smart-plug/) that controls a mains relay over a LAN HTTP + mDNS API and is driven by the Mecha Flutter app. See [Smart Plug firmware](#smart-plug-firmware) below and [`smart-plug/SPEC.md`](smart-plug/SPEC.md) for the full contract.

### Firmware architecture

```
Wi-Fi task (highest priority)
  └─ wifi_csi_rx_cb()    ← called per ping by the Wi-Fi driver
        │  builds one binary CSI_FRAME message into a stack buffer
        │  (no mutex, no send, no snprintf)
        └─ xStreamBufferSend()  ← non-blocking; drops frame if buffer full
               │
               ▼  StreamBuffer (8 KB, ~57 frames)
tcp_writer_task (priority 5)
  └─ xStreamBufferReceive() → send()
       (only place a blocking send() can occur)

tcp_reconnect_task (priority 3)
  └─ reconnects on drop, sends SESSION_INFO once per connect,
     emits 1 Hz binary HEARTBEAT, logs CSI frame counters every 5 s
```

The Wi-Fi task never touches the TCP socket directly. If the network stalls, the stream buffer absorbs short bursts; frames beyond ~57 are dropped (counted) rather than blocking the Wi-Fi task and risking beacon loss.

### Host pipeline

```
capture.py  ──► csi_data.bin  ──► csi_breathing.py  ──► BPM + plots
```

`csi_data.bin` is byte-identical to the wire — the host is a near-dumb pipe that writes every received message verbatim and peeks the headers for `stats.json`. Captures can be replayed by feeding the `.bin` back into `csi_breathing.load_binary(path)`.

---

## Requirements

### Hardware

- ESP32 (classic) or ESP32-S3 — LLTF-only CSI (64 subcarriers × I/Q, 128 bytes per frame)
- A 2.4 GHz Wi-Fi access point (the router the ESP32 pings)
- A host machine (Linux/macOS/Windows) on the same network

### Firmware

- [ESP-IDF v6.0](https://docs.espressif.com/projects/esp-idf/en/v6.0/esp32/get-started/) (activate with `source ~/.espressif/tools/activate_idf_v6.0.sh`)

### Host / Python

```sh
pip install numpy scipy matplotlib PyWavelets
```

PyWavelets is optional — if absent, DWT-based filtering is skipped and the tool falls back to a Butterworth bandpass filter.

---

## Quick Start

### 1. Configure the host IP

Run menuconfig and set the IP under **CSI Breathing Monitor → Capture machine IP address**:

```sh
idf.py menuconfig
# navigate to: CSI Breathing Monitor → Capture machine IP address
```

Or edit the default directly in `main/Kconfig.projbuild` and rebuild — no need to touch the C source:

```
config CSI_TCP_HOST
    string "Capture machine IP address"
    default "192.168.x.x"   # <- set this to your machine's IP
```

### 2. Flash the firmware

```sh
source ~/.espressif/tools/activate_idf_v6.0.sh
idf.py build flash monitor
```

The serial monitor runs at 921600 baud (configured in `sdkconfig.defaults`).

### 3. Provision Wi-Fi (first boot only)

On first boot the device enters **ESPTouch v1** (SmartConfig) mode. Use the [Espressif EspTouch app](https://www.espressif.com/en/support/download/apps) (Android / iOS) to push your Wi-Fi credentials to the device. Credentials are saved to NVS and reused on all subsequent boots.

Credentials are only erased automatically if the device fails to authenticate **before the first ever successful connection** (wrong password entered during provisioning). After the first successful association, NVS is never erased automatically — transient AP glitches during overnight captures will not trigger re-provisioning. To manually reset credentials: `idf.py erase-flash`.

### 4. Start the capture server

On your host machine, before or just after the ESP32 boots:

```sh
python capture.py -s csi_data.bin
# optional flags:
#   -p 3490          TCP port (default 3490)
#   -l capture.log   sidecar log file for protocol errors (default csi_data_log.txt)
```

The server listens on `0.0.0.0:<port>`, accepts the ESP32 connection, and appends every binary message verbatim to `csi_data.bin`. Message headers are peeked to update a `csi_data.bin.stats.json` sidecar (frame count, gap profile, last heartbeat). When the connection drops it waits for the next reconnection automatically.

Collect at least **30 seconds** of data (~3000 frames at 100 Hz) for reliable breathing rate estimation. More is better — 60–120 s is typical for a clean measurement.

### 5. Run the analysis

```sh
python csi_breathing.py csi_data.bin --output-dir results
```

Eight PNG plots are saved to `results/`. Breathing rate estimates from three independent methods are printed to stdout.

---

## Analysis Methods

All three methods operate on the same LLTF subcarrier data. After parsing, subcarriers are reordered from ESP32 storage order (positive → negative frequency) to standard FFT order, invalid/null/pilot subcarriers are masked out, and the signal is resampled to a uniform time grid (20–50 Hz).

### Ratio (default)

```
R[t, m] = H[t, m] x conj(H[t, ref])
```

The conjugate product of each subcarrier with a high-SNR reference cancels the common per-packet phase rotation that is constant across subcarriers (CFO, random phase offset). The Band-of-Interest (BoI) score selects the subcarrier pair with the most energy in the breathing band (0.1–0.5 Hz). The differential phase of `R` is then used as the breathing waveform.

> **Note:** With a single-antenna ESP32, the SFO-induced slope term `(m - m_ref) x delta` is *not* cancelled. Pairs far from the reference subcarrier may carry a residual bias; the BoI selection automatically deprioritises them.

### Amplitude

Selects the subcarrier with the highest BoI score on raw amplitude and uses its mean-subtracted amplitude envelope as the breathing waveform.

### Phase

Unwraps the phase of each subcarrier, removes the linear trend per subcarrier (detrending reduces phase drift unrelated to breathing), and picks the subcarrier with the highest variance in the breathing band.

### Rate estimation

Each method's signal is bandpass-filtered (Butterworth 4th-order, 0.1–0.5 Hz) and then fed into three estimators:

| Estimator | How it works |
|-----------|-------------|
| **PSD peak** | Zero-padded FFT (16x zero pad), find peak in breathing band → BPM |
| **Autocorrelation** | Find first peak in ACF at lags corresponding to 6–30 BPM |
| **Peak counting** | `scipy.signal.find_peaks` with minimum spacing = shortest expected breath period |

Results from all three estimators are printed side-by-side so you can cross-check them.

---

## Output Plots

| File | Contents |
|------|----------|
| `01_csi_overview.png` | CSI amplitude heatmap, phase heatmap, RSSI over time, per-subcarrier mean amplitude |
| `02_cir_analysis.png` | Channel Impulse Response (IFFT of CSI) heatmap and mean power delay profile |
| `03_subcarrier_selection.png` | BoI scores per subcarrier for each method; selected subcarrier highlighted |
| `04_breathing_signals.png` | Raw and bandpass-filtered breathing waveform for each method |
| `05_rate_estimation.png` | PSD and autocorrelation plots with peak markers; estimated BPM annotated |
| `06_complex_plane.png` | I-Q scatter for the selected subcarrier (circular arc = pure phase modulation) |
| `07_multi_subcarrier.png` | Amplitude and phase time series for the top-5 BoI subcarriers |
| `08_breathing_waveform.png` | Best-method waveform with detected peaks overlaid |

---

## Wire Format

The ESP32 streams length-prefixed binary messages over TCP. Every message:

```
u8  type        // 0x01 SESSION_INFO, 0x02 CSI_FRAME, 0x03 HEARTBEAT, 0x04 ENV
u16 length      // payload length (little-endian, NOT including this 3-byte header)
u8  payload[length]
```

All multi-byte integers are little-endian. Four message types:

| Type | Name | When emitted | Payload |
|------|------|--------------|---------|
| `0x01` | `SESSION_INFO` | Once per TCP (re)connect, before any other message | chip_id, csi_format, csi_bytes, MAC, channel, `sensor_flags` (bit0=LDR, bit1=AM2302), sample_rate_hz, `boot_id` (random per boot), esp_time_us — 26 bytes |
| `0x02` | `CSI_FRAME` | Once per CSI capture (~100 Hz) | local_timestamp_us, seq, rssi, noise_floor, rate, first_word_invalid, len, followed by `len` raw CSI bytes — 14 byte meta + 128 byte CSI = 142 bytes |
| `0x03` | `HEARTBEAT` | 1 Hz idle (no CSI sent for ≥1 s) | esp_time_us, rssi, channel, uptime_s, reconnect_count, last_disc_reason — 19 bytes |
| `0x04` | `ENV` | Low-rate wired-sensor reading (`CONFIG_ENV_EMIT_HZ`, default 1 Hz) | esp_time_us, seq, ldr_raw, ldr_mv, temp_c_x10, rh_x10, am2302_status — fixed 22 bytes |

The wire format is defined in two places that **MUST stay in sync**:

- `csi_protocol.py` — Python source of truth. Struct sizes are pinned by unit tests so accidental field additions fail loudly.
- `main/csi_protocol.h` — ESP32 mirror. Each struct has a `_Static_assert` on its size so a drift fails the firmware build.

The CSI byte payload is 64 complex subcarriers stored as interleaved `[imag0, real0, imag1, real1, ...]` `int8` pairs. The first two subcarriers (indices 0–1) are hardware artifacts and are masked out during analysis. `local_timestamp_us` is the ESP32's internal microsecond timer (`esp_timer_get_time()`); the host loader unwraps the u32 across the ~71 min boundary into a monotonic 64-bit `logical_us` before resampling.

The host loader (`csi_breathing.load_binary`) walks the binary stream and rebuilds the same `CSIDataset` shape the legacy CSV loader produced, so the downstream DSP pipeline is unchanged. `MSG_ENV` samples are collected on a separate `CSIDataset.env` list and never mixed into the CSI frames.

---

## Environment Sensors (optional)

Set `CONFIG_ENV_ENABLE=y` to attach a wired light sensor and a temperature/humidity sensor. A dedicated firmware task (`main/env_sensors.c`) samples them and emits one `MSG_ENV` message per `CONFIG_ENV_EMIT_HZ` tick — completely independent of the CSI breathing path.

### Wiring

All sensors run off the ESP32 **3V3** rail and a common **GND** (not 5 V).

**LDR (CdS light) — analog divider into ADC1:**

```
3V3 ──[ LDR ]──┬──[ R_fix (10k) ]── GND
               └── ADC1 GPIO  (CONFIG_ENV_LDR_ADC1_CHANNEL)
```

Must be an **ADC1** channel — ADC2 is unusable while Wi-Fi is on. Classic ESP32: ch0–7 = GPIO36,37,38,39,32,33,34,35. ESP32-S3: ch0–9 = GPIO1–10. The firmware sends raw + calibrated mV; the host derives a rough lux estimate (CdS power-law) that is an estimate only, never authoritative.

**AM2302 / DHT22 (temp/humidity) — 1-wire digital:**

```
AM2302 DATA ──┬── GPIO  (CONFIG_ENV_AM2302_GPIO)
              └── R 4.7k–10k ── 3V3   (mandatory pull-up)
```

Captured via the **RMT** peripheral (not bit-bang) so FreeRTOS jitter can't corrupt the timing-critical 40-bit frame. Read at most once every 2 s (sensor minimum) with last-good caching; the 5-byte checksum is validated and a CRC failure keeps the last good reading and flags `am2302_status`. Pick a normal GPIO — avoid strapping pins and the input-only GPIO34–39 on the classic ESP32.

### Menuconfig options

Under **CSI Breathing Monitor → Environment sensors**:

| Option | Default | Meaning |
|--------|---------|---------|
| `CONFIG_ENV_ENABLE` | `n` | Master switch for the env sensors |
| `CONFIG_ENV_EMIT_HZ` | `1` | MSG_ENV send rate (AM2302 only updates ~0.5 Hz) |
| `CONFIG_ENV_LDR_ENABLE` | `y` | Enable the LDR |
| `CONFIG_ENV_LDR_ADC1_CHANNEL` | `6` | ADC1 channel for the LDR node |
| `CONFIG_ENV_LDR_RFIX_OHM` | `10000` | Divider fixed resistor (keep in sync with `LDR_R_FIX_OHM` in `csi_breathing.py`) |
| `CONFIG_ENV_AM2302_ENABLE` | `y` | Enable the AM2302 |
| `CONFIG_ENV_AM2302_GPIO` | `4` | AM2302 DATA GPIO |

---

## Firmware Configuration

### Menuconfig options

Configurable via `idf.py menuconfig` under **CSI Breathing Monitor** (defined in `main/Kconfig.projbuild`):

| Symbol | Default | Description |
|--------|---------|-------------|
| `CONFIG_CSI_TCP_HOST` | `"192.168.0.100"` | IP of the capture machine |
| `CONFIG_CSI_TCP_PORT` | `3490` | TCP port |
| `CONFIG_SEND_FREQUENCY` | `100` | Ping rate in Hz (= CSI frame rate) |

### Compile-time constants (`app_main.c`)

| Constant | Default | Description |
|----------|---------|-------------|
| `TCP_TX_BUF_SIZE` | `256` | Writer task drain buffer size (bytes) — sized for one binary frame |
| `CSI_STREAM_BUF_BYTES` | `8192` | Stream buffer between CSI callback and writer task (~57 frames) |
| `TCP_RECONNECT_BACKOFF_MIN_MS` | `500` | Initial TCP reconnect delay |
| `TCP_RECONNECT_BACKOFF_MAX_MS` | `30000` | Maximum TCP reconnect delay (exponential backoff cap) |
| `TCP_HEARTBEAT_INTERVAL_US` | `1000000` | Heartbeat period when no CSI is being sent (1 s) |
| `WIFI_AUTH_FAIL_THRESHOLD` | `5` | Wrong-password failures before NVS erase (pre-first-assoc only) |

### `sdkconfig.defaults`

Sets at build time: CSI enabled, 240 MHz CPU, 1000 Hz FreeRTOS tick, performance compiler optimisation, 921600 baud UART, 32 TX / 128 RX Wi-Fi buffers, 32 Wi-Fi management buffers (`CONFIG_ESP32_WIFI_MGMT_SBUF_NUM`).

Wi-Fi power save (`WIFI_PS_NONE`) and STA inactive time (30 s beacon-loss watchdog) are configured at runtime in `wifi_init()` via `esp_wifi_set_ps()` / `esp_wifi_set_inactive_time()` — these are not Kconfig-settable in IDF 6.0.

---

## Project Structure

```
esp32-breathing/
├── main/
│   ├── app_main.c            # CSI firmware: Wi-Fi, CSI capture, TCP streaming, env_task
│   ├── env_sensors.c/.h      # optional LDR (ADC1) + AM2302 (RMT) sensors → MSG_ENV
│   ├── csi_protocol.h        # ESP32 mirror of the wire format
│   ├── Kconfig.projbuild     # menuconfig options (host IP, port, ping rate, env sensors)
│   ├── CMakeLists.txt
│   └── idf_component.yml     # IDF component manager manifest
├── capture.py                # Host TCP server — writes raw binary stream to .bin
├── csi_protocol.py           # Python wire-format source of truth
├── csi_breathing.py          # Offline analysis — breathing rate estimation + plots
├── tests/                    # pytest suite for protocol + loader
│   ├── test_binary_protocol.py
│   ├── test_binary_loader.py
│   └── test_capture_pipe.py
├── CMakeLists.txt            # Top-level IDF project CMake file (CSI firmware)
├── sdkconfig.defaults        # Non-interactive IDF config overrides
├── smart-plug/               # Independent smart-plug firmware (see section above)
│   ├── main/                 # app_main + plug_relay / plug_http / plug_mdns / plug_wifi / plug_identity
│   ├── SPEC.md               # authoritative HTTP + mDNS contract
│   ├── CMakeLists.txt
│   └── sdkconfig.defaults
└── CLAUDE.md                 # AI assistant context for this repo
```

---

## Troubleshooting

**ESP32 is not connecting to Wi-Fi**
- Ensure you ran ESPTouch v1 provisioning on first boot.
- If you need to re-provision, erase flash with `idf.py erase-flash` and reflash — the device will enter ESPTouch mode again on first boot.
- Credentials are NOT automatically erased after the first successful connection. Only a wrong-password failure before any successful association triggers auto-erase.

**Periodic disconnections during overnight capture**
- The firmware disables Wi-Fi power save (`WIFI_PS_NONE`) and sets the beacon-loss watchdog to 30 s (`esp_wifi_set_inactive_time`). If you still see disconnects, check your AP's client idle timeout setting.
- The serial monitor logs `CSI stats` every 5 s showing `total`, `accepted`, and `dropped` frame counts. A rising `dropped` count means TCP back-pressure is filling the stream buffer — check the capture machine's network load.
- `listen_interval=1` is set in the STA config so the AP treats the ESP32 as always-listening. Some APs deprioritize STAs with higher listen intervals even when power save is off.

**`capture.py` receives no data**
- Confirm the host IP in `main/Kconfig.projbuild` (or set via `idf.py menuconfig`) matches the machine running `capture.py`.
- Check that port 3490 is not blocked: `nc -l 3490` should accept a connection from the ESP32.
- The serial monitor will log TCP connection attempts and errno values on failure.
- A binary `HEARTBEAT` message is sent every 1 s even when no CSI arrives — `capture.py` updates `last_heartbeat_utc` in the stats sidecar. If you see heartbeats but `frames_written` stays at zero, the ping path to the router may be blocked.

**Estimated breathing rate looks wrong**
- Collect more data — 60+ seconds is recommended.
- Stay still during capture; body motion other than breathing dominates the CSI signal.
- Try all three methods (`--methods ratio amplitude phase`) and compare; the ratio method is generally most robust.
- Check `01_csi_overview.png` — if the amplitude heatmap is mostly flat, the device may be too far from the subject or the environment has too much static multipath.

**`PyWavelets not installed` warning**
- The DWT filter path is disabled but everything else works. Install with `pip install PyWavelets` to enable it.

---

## Smart Plug firmware

A separate ESP-IDF firmware under [`smart-plug/`](smart-plug/) turns an ESP32 into a LAN-controlled mains smart plug. It is unrelated to the CSI pipeline above — it shares only the repo and the ESPTouch/NVS Wi-Fi provisioning approach. It implements the exact HTTP + mDNS contract the Mecha Flutter app's Plug subsystem speaks; the authoritative spec is [`smart-plug/SPEC.md`](smart-plug/SPEC.md).

### What it does

- Advertises itself on mDNS as `_mechaplug._tcp` with TXT records `id` (MAC suffix), `name`, and `fw`, so the app can discover it automatically.
- Serves a small JSON REST API for the app to read state and toggle the relay.
- Drives a relay through **one GPIO pin set in menuconfig**, via a low-current signal relay → contactor chain (3.3 V logic is galvanically isolated from mains — the GPIO never switches mains directly).
- **Fails closed:** the relay boots OFF, is never persisted to NVS, and a watchdog forces it OFF if no command arrives within `CONFIG_PLUG_WATCHDOG_S` (default 30 s). The app sends a heartbeat every 10 s to hold it on.

### HTTP endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Info — `id`, `name`, `fw_ver`, `on`, `uptime_s`, `rssi`, `hw` |
| `GET` | `/state` | Live state — `on`, `uptime_s`, `rssi`, `last_cmd_age_ms`, `watchdog_remaining_ms` |
| `POST` | `/relay` | Body `{"on":true}` → atomic toggle; response `on` is read back from the GPIO. `422` on bad input; relay/watchdog unchanged on any non-2xx |
| `POST` | `/heartbeat` | Reset the watchdog only (never changes relay state); empty body |

### Build and flash

```sh
source ~/.espressif/tools/activate_idf_v6.0.sh
cd smart-plug
idf.py set-target esp32     # or esp32s3
idf.py build flash monitor
```

The component manager pulls in `espressif/mdns` and `espressif/cjson` on first build. Wi-Fi provisioning is identical to the CSI firmware: first boot enters ESPTouch, credentials are stored in NVS and reused.

### Menuconfig options

Under **Mecha Smart Plug** (`idf.py menuconfig`, defined in `smart-plug/main/Kconfig.projbuild`):

| Symbol | Default | Description |
|--------|---------|-------------|
| `CONFIG_PLUG_RELAY_GPIO` | `21` | GPIO pin driving the relay signal |
| `CONFIG_PLUG_RELAY_ACTIVE_HIGH` | `y` | Relay drive polarity (off level is always de-energised) |
| `CONFIG_PLUG_HTTP_PORT` | `80` | REST API port (also advertised via mDNS SRV) |
| `CONFIG_PLUG_NAME` | `"Mecha Plug"` | Initial plug name (persisted to NVS) |
| `CONFIG_PLUG_WATCHDOG_S` | `30` | Fail-closed watchdog window; must match the app's `WATCHDOG_S` |

---

## References

- Espressif [esp-csi](https://github.com/espressif/esp-csi) — official CSI examples
- Espressif [ESP-IDF Programming Guide](https://docs.espressif.com/projects/esp-idf/en/latest/)
- Wang et al., "WiFi CSI-based passive human activity recognition" — BoI subcarrier selection approach
- PhaseBeat / IndoTrack — phase-difference breathing / motion sensing techniques
