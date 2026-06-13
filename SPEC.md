# SPEC — env sensors on CSI firmware

LDR (CdS cell) + AM2302 (DHT22) wired to main ESP32 CSI sensor.
New protocol message carry light + temp/humidity to host.

## §G — goal

Add two wired env sensors to `main/app_main.c` firmware. Stream
their data to host over existing TCP protocol via new low-rate
message. CSI breathing path untouched.

## §C — constraints

- C1: ESP32 ADC2 dead while Wi-Fi on. LDR ADC MUST use ADC1.
- C2: AM2302 min 2 s between reads (sensor spec). Emit rate may be
  faster — cache last good reading.
- C3: AM2302 single-wire bus, timing-critical. Bit-bang risks jitter
  under FreeRTOS. Prefer RMT peripheral capture.
- C4: env sampling MUST NOT block `wifi_csi_rx_cb()` nor TCP writer
  hot path. Own FreeRTOS task.
- C5: wire protocol pinned both sides — `main/csi_protocol.h`
  `_Static_assert` + `csi_protocol.py` struct-size unit test. Stay sync.
- C6: socket writes serialize on `s_tcp_mutex` (existing).
- C7: chips ESP32 classic + ESP32-S3. Pins customizable in menuconfig.

## §I — external surfaces

- I.proto — `main/csi_protocol.h` + `csi_protocol.py` wire format.
  New `MSG_ENV 0x04`. SESSION_INFO `reserved` byte → `sensor_flags`.
- I.menuconfig — new `CONFIG_ENV_*` Kconfig: enable, LDR ADC1 channel,
  AM2302 GPIO, emit rate.
- I.capture — `capture.py` decode + peek MSG_ENV into `stats.json`.
- I.loader — `csi_breathing.load_binary()` surface env stream.
- I.wiring — AM2302 data pin + 4.7k–10k pullup to 3V3; LDR in divider
  with fixed R to an ADC1 GPIO.

## §V — invariants

- V1: env data ride own `MSG_ENV 0x04` message. CSI_FRAME meta NEVER
  widened — protect 100 Hz path.
- V2: `csi_protocol.h` struct sizes == `csi_protocol.py` struct sizes.
  `_Static_assert` C side, unit test Python side.
- V3: LDR sampled on ADC1 only (ADC2 + Wi-Fi conflict).
- V4: AM2302 hardware read interval ≥ 2 s. Emit may exceed — serve
  cached last-good between reads.
- V5: AM2302 5-byte checksum validated. On CRC fail / timeout set
  `am2302_status` non-zero, keep last-good values, never poison stream.
- V6: env sample + send in dedicated task. Never block CSI rx cb or
  writer task.
- V7: SESSION_INFO `sensor_flags` bitmap advertise present sensors
  (bit0=LDR, bit1=AM2302). Host ignore fields for absent sensors.
- V8: every MSG_ENV send holds `s_tcp_mutex`.
- V9: `MSG_ENV` payload fixed 22 bytes. `len` field absent — type
  implies size.
- V10: host route by `type`. Unknown type skipped, socket stays open
  (forward-compat). Raw framed bytes still appended verbatim to `.bin`.
- V11: env decode-to-units done ONCE — firmware emits engineering units
  (°C, %RH ×10). Host only divides by 10. LDR raw→lux is host-side
  estimate, never authoritative.
- V12: env time base = `esp_time_us` (esp_timer). CSI time base =
  `local_timestamp_us` (rx_ctrl clock). Different origins — host align
  env-to-CSI by wall-clock interpolation, NOT raw subtraction.
- V13: env NEVER feeds breathing estimator. Separate stream; correlate
  only, no contamination of CSI pipeline.

## §T — tasks

| id | status | task | cites |
|----|--------|------|-------|
| T1 | x | add `MSG_ENV 0x04` + `csi_env_t` struct both sides; `_Static_assert(sizeof==22)` C; py struct + size test | V1,V2,V9 |
| T2 | x | SESSION_INFO `reserved`→`sensor_flags` both sides; size stay 26 | V7,V2 |
| T3 | x | Kconfig.projbuild `CONFIG_ENV_ENABLE`, `CONFIG_ENV_LDR_ADC1_CHANNEL`, `CONFIG_ENV_AM2302_GPIO`, `CONFIG_ENV_EMIT_HZ` | C7,V3 |
| T4 | x | LDR read: ADC1 oneshot (+ cal mV if avail) | V3 |
| T5 | x | AM2302 driver via RMT capture; parse 40-bit frame; checksum | C3,V4,V5 |
| T6 | x | `env_task`: read LDR each tick, AM2302 ≥2 s cache, build `csi_env_t`, `tcp_send_env_locked()` at `CONFIG_ENV_EMIT_HZ` | V4,V5,V6,V8 |
| T7 | x | set `sensor_flags` at session build from enabled CONFIG | V7 |
| T8 | x | `capture.py`: decode MSG_ENV, peek into `stats.json` (env_frames, last temp/rh/ldr) | I.capture,V1 |
| T9 | x | `csi_breathing.load_binary()`: expose env samples alongside CSI | I.loader,V1 |
| T10 | . | docs: README + CLAUDE.md wire-format + wiring | I.wiring |
| T11 | x | host: route by `type`, skip unknown, surface env stream from `load_binary()`; expose `temp_c`, `rh`, `ldr_raw`, `ldr_lux?` arrays | V10,V11,V13 |
| T12 | x | host: LDR raw→resistance→lux estimate (divider math, CdS power-law); align env↔CSI by wall-clock | V11,V12 |

## §I.proto — proposed layout

```
#define MSG_ENV 0x04

/* MSG_ENV payload (22 bytes). */
typedef struct __attribute__((packed)) {
    uint64_t esp_time_us;     /* esp_timer at read                       */
    uint32_t seq;             /* env sample counter                      */
    uint16_t ldr_raw;         /* ADC1 raw 0..4095 (12-bit)               */
    uint16_t ldr_mv;          /* calibrated millivolts, 0 if no cal      */
    int16_t  temp_c_x10;      /* AM2302 °C ×10, signed (-400..800)       */
    uint16_t rh_x10;          /* AM2302 %RH ×10 (0..1000)                */
    uint8_t  am2302_status;   /* 0=ok 1=crc_fail 2=timeout 3=not_present */
    uint8_t  reserved;        /* pad → 22                                */
} csi_env_t;
_Static_assert(sizeof(csi_env_t) == 22, "csi_env_t size drift");
```

Python mirror: `ENV = struct.Struct("<QIHHhHBB")` (size 22).

SESSION_INFO change: `uint8_t reserved` → `uint8_t sensor_flags`
(bit0 LDR, bit1 AM2302). Size 26 unchanged, asserts hold.

## §I.wiring — hardware

All sensors share ESP32 **3V3** rail + common **GND**. NOT 5V (ADC +
GPIO are 3.3 V parts).

### AM2302 (DHT22) — 1-wire digital

```
  AM2302                       ESP32
  ┌─────────┐
  │ 1 VDD   ├──────────────── 3V3
  │ 2 DATA  ├───────┬──────── GPIO  (CONFIG_ENV_AM2302_GPIO)
  │ 3 NC    │       │
  │ 4 GND   ├────┐  R 4.7k–10k
  └─────────┘    │  │
                 │  └───────── 3V3   (pull-up)
                GND
```
- DATA pull-up 4.7k–10k to 3V3 mandatory (open-drain bus).
- 100 nF decoupling VDD↔GND near sensor advised.
- Pick non-strapping, non-input-only GPIO. AVOID GPIO0/45/46 (esp32s3
  strap), GPIO34–39 (classic input-only — no output, bus needs drive).

### LDR (CdS) — analog divider into ADC1

```
  3V3 ──[ LDR ]──┬──[ R_fix ]── GND
                 │
                 └── ADC1 GPIO  (CONFIG_ENV_LDR_ADC1_CHANNEL)
```
- `R_fix` ≈ 10k (sets mid-light node near mid-scale). Tune to cell.
- Bright → LDR low ohm → node HIGH. Dark → node LOW.
- ADC1 channels: classic GPIO32–39; S3 GPIO1–10. MUST be ADC1 (V3).
- 12 dB atten = ~0..3.1 V range. Keep node ≤ 3.1 V (divider does).
- Optional 100 nF node↔GND to debounce flicker.

## §D — data processing

### LDR (host-side, V11)

Firmware emits `ldr_raw` (0..4095) + `ldr_mv` (cal mV, 0 if uncal).
Host derives resistance + lux estimate:

```
V_node = ldr_mv / 1000           # volts (else raw/4095*3.1)
R_ldr  = R_fix * (3.3 - V_node) / V_node     # divider solve
lux    ≈ A * R_ldr^B             # CdS power-law, A/B per-cell cal
```
- `R_fix`, `A`, `B` host consts — defaults approximate, NOT calibrated.
- lux flagged estimate, never used for control or breathing.

### AM2302 (firmware-side)

RMT captures 40-bit frame: `RH_hi RH_lo T_hi T_lo CRC`.
```
rh_x10   = (RH_hi<<8)|RH_lo          # already ×10
raw_t    = (T_hi<<8)|T_lo
temp_c_x10 = (raw_t & 0x8000) ? -(raw_t & 0x7FFF) : raw_t   # sign bit15
crc_ok   = ((RH_hi+RH_lo+T_hi+T_lo) & 0xFF) == CRC
```
- crc fail → `am2302_status=1`, hold last-good (V5).
- Host just `/10.0` → °C, %RH. No re-decode (V11).

### Env↔CSI alignment (host)

Different clocks (V12). Host build env time series on `esp_time_us`,
CSI on `local_timestamp_us`; map both to receive wall-clock at host,
interpolate env onto CSI grid for correlation plots only.

## §B — bug ledger

| id | date | cause | fix |
|----|------|-------|-----|
