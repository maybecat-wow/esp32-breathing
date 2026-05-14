# Binary CSI Protocol — Design

**Date:** 2026-05-14
**Scope:** ESP32 firmware (`main/app_main.c`) + Python host (`capture.py`, `csi_breathing.py`)
**Out of scope:** Flutter app (`csi_app/`) — no changes in this work.

## Motivation

The current protocol streams CSI frames as text CSV at ~700 B/frame. At the
default 100 Hz that is ~70 KB/s plus ~150–250 µs per frame spent in `snprintf`
inside the WiFi task callback. Source code already notes radio jitter under
load (the `esp_wifi_set_inactive_time(WIFI_IF_STA, 30)` bump exists because
"CSI at 50+ Hz the radio occasionally misses a short run of beacons due to
TX/RX contention"). Switching to a length-prefixed binary protocol cuts wire
bytes ~5× and frees WiFi-task CPU budget, leaving headroom for future work
(e.g. Matter on ESP32-S3) without changing the DSP pipeline or the Flutter
app.

## Goals

- Cut bytes/sec on TCP from ~70 KB/s to ~14 KB/s at 100 Hz.
- Remove `snprintf` work from the WiFi task hot path.
- Keep the existing reconnect, heartbeat, fsync, and stats.json behavior.
- Same `BreathingResult` output from the analysis pipeline (no DSP changes).
- Clean break: no CSV path in firmware or host after this work.

## Non-goals

- Flutter app changes. `CaptureProvider` continues to consume the ESP32 TCP
  stream live; that connection is independent of the host capture script and
  will be updated in a follow-up.
- Backward-compat reader for existing `.csv` files. Old captures stay
  readable by the unmodified pre-change `csi_breathing.py` at the previous
  git revision; a one-shot offline converter is not part of this work.
- C5/C6/C61 support. Firmware code paths for those chips are removed. Target
  is classic ESP32 or ESP32-S3 only.
- Timestamp wraparound handling (`u32 µs` wraps at ~71 min). Current CSV
  loader also does not handle this; deferred.

## Architecture

```
ESP32 (S3 or classic, LLTF only)
  wifi_csi_rx_cb       packs binary CSI_FRAME → StreamBuffer
  tcp_writer_task      drains StreamBuffer → send()
  tcp_reconnect_task   maintains socket, sends SESSION_INFO on (re)connect,
                       sends HEARTBEAT 1 Hz idle

Host
  capture.py           accepts TCP, peeks msg types for stats, writes raw
                       bytes verbatim to <store>.bin
  csi_breathing.py     mmap .bin, walk messages, build CsiDataset, run DSP
```

`capture.py` is intentionally a near-dumb pipe on the hot path — it does not
parse CSI sample bytes, only peeks the 3-byte message header to drive
`stats.json`. The on-disk file is the same byte stream as the wire, so
replaying a capture is a matter of re-feeding the `.bin` to the loader.

## Wire format

All values little-endian. Every message:

```
u8  type
u16 length        (length of payload only, not including this 3-byte header)
u8  payload[length]
```

### Message types

| Type | Name         | When |
|------|--------------|------|
| 0x01 | SESSION_INFO | once per TCP (re)connect, before any other message |
| 0x02 | CSI_FRAME    | one per CSI capture (~100 Hz) |
| 0x03 | HEARTBEAT    | 1 Hz when no CSI has been sent for ≥1 s |

No on-wire gap marker. Gaps are derived host-side from: timestamp jumps,
connection drops (`recv` returns 0 or timeout), and SESSION_INFO boundaries
between frames.

### 0x01 SESSION_INFO (~24 B)

```
u8   chip_id          1 = classic ESP32, 2 = ESP32-S3
u8   csi_format       0 = legacy HT LLTF (lltf_en=1, ltf_merge_en=1)
u16  csi_bytes        128 for 64-subcarrier × I/Q × 1 B each
u8   mac[6]
u8   channel
u8   reserved         0
u16  sample_rate_hz   currently 100 (from CONFIG_SEND_FREQUENCY)
u32  boot_id          esp_random() captured once at boot; used to
                      distinguish "reboot" from "TCP reconnect"
u64  esp_time_us      esp_timer_get_time() at send
```

### 0x02 CSI_FRAME (~140 B at csi_bytes=128)

```
u32  local_timestamp_us   rx_ctrl->timestamp; monotonic between reboots
u32  seq                  the firmware's s_count counter
i8   rssi                 rx_ctrl->rssi
i8   noise_floor          rx_ctrl->noise_floor
u8   rate                 rx_ctrl->rate
u8   first_word_invalid   info->first_word_invalid
u16  len                  info->len; equal to SESSION_INFO.csi_bytes
u8   data[len]            raw bytes of info->buf — no gain compensation
                          baked in (the classic ESP32 path has no gain ctrl
                          anyway; S3 path skips it to keep frames cheap)
```

### 0x03 HEARTBEAT (~16 B)

```
u64  esp_time_us
i8   rssi
u8   channel
u32  uptime_s
u32  reconnect_count
u8   last_disc_reason
```

### Bytes per second

At 100 Hz with csi_bytes=128:
- CSI_FRAME: 3 + 12 + 128 = 143 B/frame → 14.3 KB/s
- HEARTBEAT/SESSION_INFO: negligible

vs. current CSV ~70 KB/s. ~5× reduction.

## ESP32 firmware changes

**Files touched:** `main/app_main.c`, new header `main/csi_protocol.h`.

### Add `main/csi_protocol.h`

- `#define MSG_SESSION_INFO 0x01`, `MSG_CSI_FRAME 0x02`, `MSG_HEARTBEAT 0x03`.
- Packed structs for each payload, `__attribute__((packed))`.
- `_Static_assert(sizeof(csi_frame_meta_t) == 12, "...")` etc. to catch
  accidental padding from future field additions.

### Modify `app_main.c`

Remove:
- All CSV `snprintf` work in `wifi_csi_rx_cb` (lines 442–496).
- `tcp_send_header_locked` and `tcp_send_heartbeat_locked` (replaced).
- All `#if CONFIG_IDF_TARGET_ESP32C5 || …C6 || …C61` branches.
- `CSI_FORCE_LLTF`, `CONFIG_GAIN_CONTROL` blocks. Keep `esp_csi_gain_ctrl`
  out of the binary frame builder for simplicity.

Add/change:
- `wifi_csi_rx_cb`: write 3-byte msg header into stack buf, populate packed
  `csi_frame_meta_t`, memcpy `info->buf` bytes after, single
  `xStreamBufferSend`.
- `tcp_reconnect_task`: on each successful connect, build and send a
  SESSION_INFO message before anything else; reset `s_header_sent`-style
  flag to indicate session is open. Drop the call to `tcp_send_header_locked`.
- Idle heartbeat path: build binary HEARTBEAT msg in `tcp_send_heartbeat_locked`
  replacement.
- `boot_id`: static `uint32_t s_boot_id = esp_random()` captured once at
  `app_main` start.

Tune:
- `TCP_TX_BUF_SIZE` → 256 (binary frame ≈ 143 B; was 1024 for worst-case CSV).
- `CSI_STREAM_BUF_BYTES` → keep 8192 (now absorbs ~57 frames; previously ~10).

Unchanged:
- `tcp_writer_task` stays a byte-pipe.
- Reconnect/backoff logic.
- WiFi event handler, ESPTouch, NVS credential handling.
- `esp_wifi_set_ps(WIFI_PS_NONE)`, `esp_wifi_set_inactive_time(30)`,
  `listen_interval=1`.

## Python host changes

**Files touched:** `capture.py`, `csi_breathing.py`. Also `CLAUDE.md`,
`.gitignore`.

### `capture.py` — dumb pipe

Drop:
- `csv` import, `DATA_COLUMNS_NAMES*`, `VALID_COLUMN_COUNTS`.
- `CaptureSession.ingest_csi_row`, `emit_gap`, `_recv_lines` (the
  newline-splitting generator).
- All CSV header parsing on (re)connect.

Keep:
- TCP server loop, accept/reconnect, `StallError`, fsync cadence,
  stats.json format and cadence.

Change:
- Output filename: `<store>.bin`. Default `./csi_data.bin`. CLI flag stays
  `-s`/`--store` for compatibility with muscle memory.
- Per-connection flow:
  1. `recv` exactly enough bytes to assemble one message: read 3-byte
     header, read `length` bytes for payload.
  2. Validate first message is `MSG_SESSION_INFO`. If not, log, close.
  3. Append every received message verbatim (header + payload) to `.bin`.
     The file is the same byte stream as the wire, including SESSION_INFO
     at every reconnect.
  4. While appending, peek `type` + first few bytes per message to update
     stats: `frames_written` (for CSI_FRAME), per-gap derivation between
     SESSION_INFO instances or recv stalls. Do not deserialize the full
     CSI byte payload — only the meta header is read.
- On `length > MAX_PAYLOAD_BYTES` (define = 4096): log, close socket, wait
  for next connect. Defensive against runaway/garbage data.

stats.json: same shape as today. `frames_written`, `gap_count`,
`total_gap_seconds`, `longest_segment_seconds`, `last_heartbeat_utc`,
`gaps[]` all populated from the peek-only inline parser.

### `csi_breathing.py` — binary loader

Add `load_binary(path) -> CsiDataset`:
- `mmap` the file.
- Walk: `type, length = struct.unpack_from('<BH', mm, off)`; advance.
- Dispatch:
  - `MSG_SESSION_INFO`: update current session state (csi_bytes,
    sample_rate, mac, channel, boot_id). If a CSI_FRAME has already been
    seen and the new `boot_id` differs from the previous → hard gap +
    reset session state. Same `boot_id` → soft gap (TCP reconnect, no
    reboot).
  - `MSG_CSI_FRAME`: build one row matching today's per-frame dict. CSI
    bytes go into the same numpy buffer the existing pipeline expects
    after subcarrier reorder.
  - `MSG_HEARTBEAT`: record in metadata; do not produce a frame.
- Duplicate `local_timestamp_us` within a session → drop (parity with
  today's CSV path).
- Δt > 500 ms within a session → gap.

Output type: existing `CsiDataset` (frames + gap indices). The downstream
DSP pipeline does not change.

Remove the CSV loader path. Old `.csv` files are not readable by post-change
`csi_breathing.py`.

### CLAUDE.md updates

- Update "Data flow" §1 to describe binary message stream.
- Note: classic ESP32 / S3 only; C5/C6 paths removed.
- Note: gap reasons (TIMEOUT/CLOSE/OSERR/MONOTONIC/REBOOT) are now host-side
  derivations; wire has no gap message type.

### .gitignore

- Add `*.bin` (if not already covered).

## Error handling

### ESP32 side

| Condition | Behavior |
|-----------|----------|
| StreamBuffer full | drop frame, `s_csi_frames_dropped++` (unchanged) |
| `send()` returns -1 | close socket, reconnect task reopens, new SESSION_INFO |
| `info->len` > buffer cap | log warning, drop frame (no torn writes possible) |

### Host side

| Condition | Behavior |
|-----------|----------|
| First message not SESSION_INFO | close, log; wait for next connect |
| `length` > 4096 | close, log; protects against garbage stream |
| recv timeout (`RECV_TIMEOUT_S=3`) | record TIMEOUT in stats, close, wait |
| Unknown `type` byte | skip `length` bytes, log once per session |
| File truncated mid-message | loader stops at last complete msg, logs unparsed-byte count |
| Same `boot_id` SESSION_INFO mid-stream | soft gap (TCP reconnect) |
| Different `boot_id` SESSION_INFO | hard gap, reset session state (REBOOT) |
| Duplicate `local_timestamp_us` | drop (same as today) |

## Testing

### New file `test_binary_protocol.py`

1. **Round-trip parser.** Hand-construct SESSION_INFO + 3 CSI_FRAMEs +
   HEARTBEAT + 2 CSI_FRAMEs. Loader returns 5 frames; metadata fields and
   CSI byte arrays byte-equal to constructed input.
2. **TCP-reconnect gap.** Two SESSION_INFOs with same `boot_id`, one frame
   between and one after. Loader emits one soft gap.
3. **Reboot gap.** Two SESSION_INFOs with different `boot_id`. Hard gap;
   session state replaced.
4. **Truncated tail.** Round-trip bytes, truncate last 5 bytes mid-CSI.
   Loader parses all preceding messages, stops cleanly, no exception.
5. **Bad type byte.** Inject `\xFF\x00\x00` between valid messages. Loader
   advances past zero-length unknown message; frame count correct.
6. **Length overflow.** Inject `type=0x02, length=0xFFFF`. Loader bails at
   injection point, logs, stops walking that session.
7. **Duplicate timestamp.** Two CSI_FRAMEs with identical
   `local_timestamp_us`. First kept; second dropped.
8. **>500 ms timestamp jump.** Δt > 500 ms triggers gap insertion.

### ESP32 manual smoke

- Flash, observe `s_count`/`s_csi_frames_dropped` counters via serial log.
- Run `capture.py`, confirm `.bin` grows at ~14 KB/s.
- Run `csi_breathing.py` against `.bin`, confirm breathing rate output is
  produced.

### Validation gate before merging

- Overnight side-by-side capture: binary build for one night. Compare
  `stats.json` frame count + gap profile to the last CSV-era overnight run
  from the same room. ±5% on frame count, gap count in the same ballpark =
  pass.

## Open questions

None. All design decisions resolved during brainstorming. (Wraparound of
`u32 local_timestamp_us` at ~71 min is a known limitation inherited from
the current CSV path and is explicitly out of scope.)
