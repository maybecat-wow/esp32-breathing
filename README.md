# esp32-breathing

Contactless breathing rate detection using Wi-Fi Channel State Information (CSI) on an ESP32.

The firmware pings the router at 100 Hz and captures the resulting CSI frames, which encode how the wireless channel changes over time. Chest movement during breathing produces subtle, periodic perturbations in the channel — this project extracts that signal and estimates breathing rate in breaths per minute (BPM).

```
ESP32 ──pings router 100 Hz──► CSI frames
         │
         ▼
   wifi_csi_rx_cb()
         │  CSV over TCP
         ▼
   capture.py  ──► csi_data.csv
         │
         ▼
   csi_breathing.py  ──► breathing rate + plots
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

Edit `main/app_main.c` line 55 and set `CONFIG_CSI_TCP_HOST` to the IP address of the machine that will run `capture.py`:

```c
#define CONFIG_CSI_TCP_HOST  "192.168.x.x"   // ← your machine's IP
```

Alternatively, set it through menuconfig:

```sh
idf.py menuconfig   # Component config → CSI TCP output → Host IP
```

### 2. Flash the firmware

```sh
source ~/.espressif/tools/activate_idf_v6.0.sh
idf.py build flash monitor
```

The serial monitor runs at 921600 baud (configured in `sdkconfig.defaults`).

### 3. Provision Wi-Fi (first boot only)

On first boot the device enters **ESPTouch** mode. Use the [Espressif EspTouch app](https://www.espressif.com/en/support/download/apps) (Android / iOS) to push your Wi-Fi credentials to the device. Credentials are saved to NVS and reused on all subsequent boots. If saved credentials fail five times in a row the device erases them and re-enters ESPTouch mode.

### 4. Start the capture server

On your host machine, before or just after the ESP32 boots:

```sh
python capture.py -s csi_data.csv
# optional flags:
#   -p 3490          TCP port (default 3490)
#   -l capture.log   log file for invalid rows (default csi_data_log.txt)
```

The server listens on `0.0.0.0:<port>`, accepts the ESP32 connection, and appends validated CSI rows to `csi_data.csv`. When the connection drops (e.g. device reset) it waits for the next reconnection automatically.

Collect at least **30 seconds** of data (≈ 3000 frames at 100 Hz) for reliable breathing rate estimation. More is better — 60–120 s is typical for a clean measurement.

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
R[t, m] = H[t, m] × conj(H[t, ref])
```

The conjugate product of each subcarrier with a high-SNR reference cancels the common per-packet phase rotation that is constant across subcarriers (CFO, random phase offset). The Band-of-Interest (BoI) score selects the subcarrier pair with the most energy in the breathing band (0.1–0.5 Hz). The differential phase of `R` is then used as the breathing waveform.

> **Note:** With a single-antenna ESP32, the SFO-induced slope term `(m − m_ref) × δ` is *not* cancelled. Pairs far from the reference subcarrier may carry a residual bias; the BoI selection automatically deprioritises them.

### Amplitude

Selects the subcarrier with the highest BoI score on raw amplitude and uses its mean-subtracted amplitude envelope as the breathing waveform.

### Phase

Unwraps the phase of each subcarrier, removes the linear trend per subcarrier (detrending reduces phase drift unrelated to breathing), and picks the subcarrier with the highest variance in the breathing band.

### Rate estimation

Each method's signal is bandpass-filtered (Butterworth 4th-order, 0.1–0.5 Hz) and then fed into three estimators:

| Estimator | How it works |
|-----------|-------------|
| **PSD peak** | Zero-padded FFT (16× zero pad), find peak in breathing band → BPM |
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

The `data` field is a JSON integer array — 128 bytes representing 64 complex subcarriers stored as interleaved `[imag₀, real₀, imag₁, real₁, …]` pairs. The first two subcarriers (indices 0–1) are hardware artifacts and are masked out during analysis.

`local_timestamp` is the ESP32's internal microsecond timer (`esp_timer_get_time()`). Timestamps are non-uniform due to FreeRTOS tick quantization and ping scheduling jitter; the analysis script resamples to a uniform grid before spectral processing.

---

## Firmware Configuration

Key parameters in `main/app_main.c`:

| Symbol | Default | Description |
|--------|---------|-------------|
| `CONFIG_CSI_TCP_HOST` | `"192.168.0.100"` | IP of the capture machine |
| `CONFIG_CSI_TCP_PORT` | `3490` | TCP port |
| `CONFIG_SEND_FREQUENCY` | `100` | Ping rate in Hz (= CSI frame rate) |
| `TCP_TX_BUF_SIZE` | `1024` | Per-row transmit buffer (bytes) |
| `TCP_RECONNECT_DELAY_MS` | `1000` | Retry delay after a dropped connection |
| `WIFI_MAX_RETRY` | `5` | Failed connection attempts before erasing NVS credentials |
| `CONFIG_GAIN_CONTROL` | `1` (C3/C5/C6/S3) | Enable AGC/FFT gain compensation |
| `CONFIG_FORCE_GAIN` | `0` | Lock gain to the baseline after 100 frames |

`sdkconfig.defaults` sets: CSI enabled, 240 MHz CPU, 1000 Hz FreeRTOS tick, performance compiler optimisation, 921600 baud UART, 32 TX / 128 RX Wi-Fi buffers.

---

## Project Structure

```
esp32-breathing/
├── main/
│   ├── app_main.c          # ESP32 firmware: Wi-Fi, CSI capture, TCP streaming
│   ├── CMakeLists.txt
│   └── idf_component.yml   # IDF component manager manifest
├── capture.py              # Host TCP server — writes CSI data to CSV
├── csi_breathing.py        # Offline analysis — breathing rate estimation + plots
├── CMakeLists.txt          # Top-level IDF project CMake file
├── sdkconfig.defaults      # Non-interactive IDF config overrides
└── CLAUDE.md               # AI assistant context for this repo
```

---

## Troubleshooting

**ESP32 is not connecting to Wi-Fi**
- Ensure you ran ESPTouch provisioning on first boot. If credentials were previously saved but are now wrong, power-cycle the device five times or erase NVS with `idf.py erase-flash`.

**`capture.py` receives no data**
- Confirm the host IP in `app_main.c` matches the machine running `capture.py`.
- Check that port 3490 is not blocked by a firewall: `nc -l 3490` should accept a connection from the ESP32.
- The ESP32 serial monitor (`idf.py monitor`) will log connection attempts and errors.

**Estimated breathing rate looks wrong**
- Collect more data — 60+ seconds is recommended.
- Stay still during capture; body motion other than breathing dominates the CSI signal.
- Try all three methods (`--methods ratio amplitude phase`) and compare; the ratio method is generally most robust.
- Check `01_csi_overview.png` — if the CSI amplitude heatmap is mostly flat (no variation), the device may be too far from the subject or the room has too much multipath clutter.

**`PyWavelets not installed` warning**
- The DWT filter path is disabled but everything else works. Install with `pip install PyWavelets` to enable it.

---

## References

- Espressif [esp-csi](https://github.com/espressif/esp-csi) — official CSI examples and gain control component
- Espressif [ESP-IDF Programming Guide](https://docs.espressif.com/projects/esp-idf/en/latest/)
- Wang et al., "WiFi CSI-based passive human activity recognition" — BoI subcarrier selection approach
- PhaseBeat / IndoTrack — phase-difference breathing / motion sensing techniques
