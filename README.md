# esp32-breathing

Contactless breathing rate detection using Wi-Fi Channel State Information (CSI) on an ESP32.

The firmware pings the router at 100 Hz and captures the resulting CSI frames, which encode how the wireless channel changes over time. Chest movement during breathing produces subtle, periodic perturbations in the channel — this project extracts that signal and estimates breathing rate in breaths per minute (BPM).

### Firmware architecture

```
Wi-Fi task (highest priority)
  └─ wifi_csi_rx_cb()    ← called per ping by the Wi-Fi driver
        │  builds one CSV row into a stack buffer (no mutex, no send)
        └─ xStreamBufferSend()  ← non-blocking; drops frame if buffer full
               │
               ▼  StreamBuffer (8 KB, ~10 rows)
tcp_writer_task (priority 5)
  └─ xStreamBufferReceive() → send()
       (only place a blocking send() can occur)

tcp_reconnect_task (priority 3)
  └─ reconnects on drop, sends CSV header, emits 1 Hz heartbeat,
     logs CSI frame counters (total / accepted / dropped) every 5 s
```

The Wi-Fi task never touches the TCP socket directly. If the network stalls, the stream buffer absorbs short bursts; frames beyond ~10 rows are dropped (counted) rather than blocking the Wi-Fi task and risking beacon loss.

### Host pipeline

```
capture.py  ──► csi_data.csv  ──► csi_breathing.py  ──► BPM + plots
```

---

## Requirements

### Hardware

- Any ESP32-family board with Wi-Fi: ESP32, ESP32-S3, ESP32-C3, ESP32-C5, ESP32-C6, or ESP32-C61
- A 2.4 GHz Wi-Fi access point (the router the ESP32 pings)
- A host machine (Linux/macOS/Windows) on the same network

### Firmware

- [ESP-IDF v6.0](https://docs.espressif.com/projects/esp-idf/en/v6.0/esp32/get-started/) (activate with `source ~/.espressif/tools/activate_idf_v6.0.sh`)
- IDF Component: `esp_csi_gain_ctrl >= 0.1.4` (fetched automatically by the component manager)

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
python capture.py -s csi_data.csv
# optional flags:
#   -p 3490          TCP port (default 3490)
#   -l capture.log   log file for invalid rows (default csi_data_log.txt)
```

The server listens on `0.0.0.0:<port>`, accepts the ESP32 connection, and appends validated CSI rows to `csi_data.csv`. When the connection drops (e.g. device reset) it waits for the next reconnection automatically.

Collect at least **30 seconds** of data (~3000 frames at 100 Hz) for reliable breathing rate estimation. More is better — 60–120 s is typical for a clean measurement.

### 5. Run the analysis

```sh
python csi_breathing.py csi_data.csv --output-dir results
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

## CSV Format

The ESP32 sends a CSV header row on every new TCP connection. Two layouts exist depending on target chip:

**ESP32 / S3 / C3 (25 columns)**

```
type, id, mac, rssi, rate, sig_mode, mcs, bandwidth, smoothing, not_sounding,
aggregation, stbc, fec_coding, sgi, noise_floor, ampdu_cnt, channel,
secondary_channel, local_timestamp, ant, sig_len, rx_state, len, first_word, data
```

**C5 / C6 / C61 (15 columns)**

```
type, id, mac, rssi, rate, noise_floor, fft_gain, agc_gain,
channel, local_timestamp, sig_len, rx_state, len, first_word, data
```

The `data` field is a JSON integer array — 128 bytes representing 64 complex subcarriers stored as interleaved `[imag0, real0, imag1, real1, ...]` pairs. The first two subcarriers (indices 0–1) are hardware artifacts and are masked out during analysis.

`local_timestamp` is the ESP32's internal microsecond timer (`esp_timer_get_time()`). Timestamps are non-uniform due to FreeRTOS tick quantization and ping scheduling jitter; the analysis script resamples to a uniform grid before spectral processing.

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
| `TCP_TX_BUF_SIZE` | `1024` | Per-row CSV buffer size (bytes) |
| `CSI_STREAM_BUF_BYTES` | `8192` | Stream buffer between CSI callback and writer task (~10 rows) |
| `TCP_RECONNECT_BACKOFF_MIN_MS` | `500` | Initial TCP reconnect delay |
| `TCP_RECONNECT_BACKOFF_MAX_MS` | `30000` | Maximum TCP reconnect delay (exponential backoff cap) |
| `TCP_HEARTBEAT_INTERVAL_US` | `1000000` | Heartbeat period when no CSI is being sent (1 s) |
| `WIFI_AUTH_FAIL_THRESHOLD` | `5` | Wrong-password failures before NVS erase (pre-first-assoc only) |
| `CONFIG_GAIN_CONTROL` | `1` (C3/C5/C6/S3) | Enable AGC/FFT gain compensation |
| `CONFIG_FORCE_GAIN` | `0` | Lock gain to baseline after 100 frames |

### `sdkconfig.defaults`

Sets at build time: CSI enabled, 240 MHz CPU, 1000 Hz FreeRTOS tick, performance compiler optimisation, 921600 baud UART, 32 TX / 128 RX Wi-Fi buffers, 32 Wi-Fi management buffers (`CONFIG_ESP32_WIFI_MGMT_SBUF_NUM`).

Wi-Fi power save (`WIFI_PS_NONE`) and STA inactive time (30 s beacon-loss watchdog) are configured at runtime in `wifi_init()` via `esp_wifi_set_ps()` / `esp_wifi_set_inactive_time()` — these are not Kconfig-settable in IDF 6.0.

---

## Project Structure

```
esp32-breathing/
├── main/
│   ├── app_main.c            # ESP32 firmware: Wi-Fi, CSI capture, TCP streaming
│   ├── Kconfig.projbuild     # menuconfig options (host IP, port, ping rate)
│   ├── CMakeLists.txt
│   └── idf_component.yml     # IDF component manager manifest
├── capture.py                # Host TCP server — writes CSI data to CSV
├── csi_breathing.py          # Offline analysis — breathing rate estimation + plots
├── CMakeLists.txt            # Top-level IDF project CMake file
├── sdkconfig.defaults        # Non-interactive IDF config overrides
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
- A heartbeat line (`HEARTBEAT,...`) is sent every 1 s even when no CSI arrives — if you see heartbeats but no `CSI_DATA` rows, the ping path to the router may be blocked.

**Estimated breathing rate looks wrong**
- Collect more data — 60+ seconds is recommended.
- Stay still during capture; body motion other than breathing dominates the CSI signal.
- Try all three methods (`--methods ratio amplitude phase`) and compare; the ratio method is generally most robust.
- Check `01_csi_overview.png` — if the amplitude heatmap is mostly flat, the device may be too far from the subject or the environment has too much static multipath.

**`PyWavelets not installed` warning**
- The DWT filter path is disabled but everything else works. Install with `pip install PyWavelets` to enable it.

---

## References

- Espressif [esp-csi](https://github.com/espressif/esp-csi) — official CSI examples and gain control component
- Espressif [ESP-IDF Programming Guide](https://docs.espressif.com/projects/esp-idf/en/latest/)
- Wang et al., "WiFi CSI-based passive human activity recognition" — BoI subcarrier selection approach
- PhaseBeat / IndoTrack — phase-difference breathing / motion sensing techniques
