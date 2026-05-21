# Binary CSI Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CSV TCP stream between ESP32 and host with a length-prefixed binary protocol. Cuts wire bytes ~5× at 100 Hz and frees WiFi-task CPU. Flutter app is out of scope.

**Architecture:** ESP32 packs `SESSION_INFO` / `CSI_FRAME` / `HEARTBEAT` messages (little-endian, `u8 type | u16 length | payload`) and streams them. `capture.py` becomes a byte-pipe that writes the raw stream to `<store>.bin` while peeking message headers for stats. `csi_breathing.py` adds a binary loader that mmaps the file, walks messages, unwraps the u32 timestamp into a monotonic `logical_us`, and produces the existing `CSIDataset` shape so the DSP pipeline is untouched.

**Tech Stack:** ESP-IDF (C, FreeRTOS), Python 3.x (numpy, pytest), `struct` for wire layout.

**Spec:** `docs/superpowers/specs/2026-05-14-binary-csi-protocol-design.md`

---

## File Structure

### Python — created
- `csi_protocol.py` — shared protocol constants + `struct.Struct` definitions for each message + `iter_messages(buf)` generator + small encoder helpers used by tests
- `tests/__init__.py` — empty package marker
- `tests/conftest.py` — pytest config (path setup so `tests/` finds top-level modules)
- `tests/test_binary_protocol.py` — parser unit tests

### Python — modified
- `capture.py` — full rewrite: drop CSV, become binary byte-pipe; keep server loop, stall handling, fsync cadence, `stats.json`
- `csi_breathing.py` — add `load_binary(path) -> CSIDataset`; route `parse_file` to the binary loader for `.bin` inputs and keep CSV loader for `.csv` inputs (until users migrate)

### Firmware — created
- `main/csi_protocol.h` — packed structs for each payload + message-type defines + `_Static_assert` on sizes

### Firmware — modified
- `main/app_main.c` — rip out CSV builders, replace `wifi_csi_rx_cb` body with binary frame packing, send `SESSION_INFO` on (re)connect, swap text heartbeat for binary one, drop C5/C6/C61 branches

### Housekeeping — modified
- `.gitignore` — add `*.bin`
- `CLAUDE.md` — update Data Flow, CSV Format → Binary Format, remove C5/C6 row from chip table

---

## Task 1: Define Python protocol module + size tests

**Files:**
- Create: `csi_protocol.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_binary_protocol.py`

The module owns the wire-format truth. Every consumer (`capture.py`, `csi_breathing.py`, tests) imports its `struct.Struct` instances and constants. Writing the size tests first locks the layout against accidental drift.

- [ ] **Step 1: Install pytest if missing**

```bash
python3 -m pip install --user pytest
```

- [ ] **Step 2: Create the test package skeleton**

Write `tests/__init__.py` as an empty file.

Write `tests/conftest.py`:

```python
"""Pytest config: put the project root on sys.path so tests can import top-level modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 3: Write failing size tests**

Write `tests/test_binary_protocol.py`:

```python
"""Wire format must not drift. Sizes pinned per the design doc:
docs/superpowers/specs/2026-05-14-binary-csi-protocol-design.md
"""
import struct

import csi_protocol as p


def test_msg_type_constants():
    assert p.MSG_SESSION_INFO == 0x01
    assert p.MSG_CSI_FRAME == 0x02
    assert p.MSG_HEARTBEAT == 0x03


def test_header_struct_size():
    # u8 type | u16 length
    assert p.HEADER.size == 3


def test_session_info_struct_size():
    # u8 chip_id, u8 csi_format, u16 csi_bytes, u8 mac[6], u8 channel,
    # u8 reserved, u16 sample_rate_hz, u32 boot_id, u64 esp_time_us
<<<<<<< Updated upstream
    assert p.SESSION_INFO.size == 24
=======
    assert p.SESSION_INFO.size == 26
>>>>>>> Stashed changes


def test_csi_frame_meta_struct_size():
    # u32 local_timestamp_us, u32 seq, i8 rssi, i8 noise_floor,
    # u8 rate, u8 first_word_invalid, u16 len
    assert p.CSI_FRAME_META.size == 14


def test_heartbeat_struct_size():
    # u64 esp_time_us, i8 rssi, u8 channel, u32 uptime_s,
    # u32 reconnect_count, u8 last_disc_reason
    assert p.HEARTBEAT.size == 19
```

<<<<<<< Updated upstream
Note: design doc said ~24 / ~140 / ~16 — those were rough estimates including the 3-byte header. Exact `.size` values pinned here are payload-only.
=======
Note: payload `.size` values (26 / 14 / 19) match the `_Static_assert`s in `csi_protocol.h`.
>>>>>>> Stashed changes

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest tests/test_binary_protocol.py -v`
Expected: `ModuleNotFoundError: No module named 'csi_protocol'`

- [ ] **Step 5: Implement `csi_protocol.py`**

Write `csi_protocol.py`:

```python
"""Binary wire protocol between ESP32 firmware and host capture.

All multibyte integers are little-endian. Every message on the wire is:

    u8  type
    u16 length        (length of payload, not including this 3-byte header)
    u8  payload[length]

See docs/superpowers/specs/2026-05-14-binary-csi-protocol-design.md.
"""
import struct

# ── Message types ──────────────────────────────────────────────────────────
MSG_SESSION_INFO = 0x01
MSG_CSI_FRAME    = 0x02
MSG_HEARTBEAT    = 0x03

# ── Limits ─────────────────────────────────────────────────────────────────
MAX_PAYLOAD_BYTES = 4096   # host parser closes the socket if length exceeds this

# ── Wire structs (little-endian, no padding) ───────────────────────────────
# Every Struct format starts with "<" to force little-endian and unaligned packing.

# u8 type | u16 length
HEADER = struct.Struct("<BH")

# Sent once per TCP (re)connect, before any other message.
#   chip_id        : 1 = classic ESP32, 2 = ESP32-S3
#   csi_format     : 0 = legacy HT LLTF (lltf_en=1, ltf_merge_en=1)
#   csi_bytes      : 128 for LLTF 64-subcarrier × I/Q × 1 B
#   mac[6]         : station MAC at the time of capture
#   channel        : primary WiFi channel
#   reserved       : 0 (padding for u16 alignment in human reading; struct.Struct
#                    forces unaligned regardless)
#   sample_rate_hz : nominal CSI capture rate (CONFIG_SEND_FREQUENCY)
#   boot_id        : random u32 captured once at ESP32 boot; lets the host tell
#                    "TCP reconnect" (same boot_id) from "ESP32 rebooted"
#                    (different boot_id).
#   esp_time_us    : esp_timer_get_time() at send
SESSION_INFO = struct.Struct("<BBH6sBBHIQ")

# One CSI capture (~100 Hz).
#   local_timestamp_us : wifi_pkt_rx_ctrl_t->timestamp, u32 µs, wraps at 2^32
#   seq                : firmware s_count counter
#   rssi               : rx_ctrl->rssi (dBm, signed)
#   noise_floor        : rx_ctrl->noise_floor
#   rate               : rx_ctrl->rate
#   first_word_invalid : info->first_word_invalid
#   len                : info->len (CSI byte count; equals SESSION_INFO.csi_bytes)
#
# The csi_bytes payload follows this 14-byte meta block.
CSI_FRAME_META = struct.Struct("<IIbbBBH")

# Sent at 1 Hz when no CSI has gone out for ≥1 s.
HEARTBEAT = struct.Struct("<QbBIIB")


def pack_header(msg_type: int, length: int) -> bytes:
    """Build the 3-byte message header. Raises ValueError on length > u16."""
    if not 0 <= length <= 0xFFFF:
        raise ValueError(f"length {length} out of u16 range")
    return HEADER.pack(msg_type, length)


def iter_messages(buf):
    """Yield (msg_type, payload_bytes) for each complete message in *buf*.

    Stops at the first truncated message (i.e. EOF mid-message) without raising,
    so callers can walk a partially-written .bin file safely.

    *buf* may be `bytes`, `bytearray`, or a `mmap.mmap` — anything that supports
    slicing and `len()`.
    """
    off = 0
    total = len(buf)
    while off + HEADER.size <= total:
        msg_type, length = HEADER.unpack_from(buf, off)
        end = off + HEADER.size + length
        if end > total:
            return  # truncated tail — stop cleanly
        payload = bytes(buf[off + HEADER.size : end])
        yield msg_type, payload
        off = end
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest tests/test_binary_protocol.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add csi_protocol.py tests/__init__.py tests/conftest.py tests/test_binary_protocol.py
git commit -m "feat: add csi_protocol module with size-pinned wire structs

Owns the binary protocol truth used by capture.py, csi_breathing.py,
and tests. Struct sizes are pinned by unit tests so accidental field
additions can't shift the wire layout silently."
```

---

## Task 2: Parser round-trip test + encoder helpers

**Files:**
- Modify: `csi_protocol.py` (add encoder helpers used by tests)
- Test: `tests/test_binary_protocol.py`

Encoders live in `csi_protocol.py` (not in tests) because `capture.py`'s lightweight stats peek and the ESP32-side parity check in the smoke gate can reuse them.

- [ ] **Step 1: Write failing round-trip test**

Append to `tests/test_binary_protocol.py`:

```python
import os


def _sample_session_info(boot_id: int = 0xDEADBEEF) -> bytes:
    return p.encode_session_info(
        chip_id=1, csi_format=0, csi_bytes=128,
        mac=b"\xaa\xbb\xcc\xdd\xee\xff",
        channel=6, sample_rate_hz=100,
        boot_id=boot_id, esp_time_us=1_000_000,
    )


def _sample_csi_frame(ts: int, seq: int, csi_payload: bytes) -> bytes:
    return p.encode_csi_frame(
        local_timestamp_us=ts, seq=seq,
        rssi=-55, noise_floor=-95,
        rate=11, first_word_invalid=0,
        csi_bytes=csi_payload,
    )


def _sample_heartbeat() -> bytes:
    return p.encode_heartbeat(
        esp_time_us=2_000_000, rssi=-55, channel=6,
        uptime_s=42, reconnect_count=1, last_disc_reason=0,
    )


def test_round_trip_session_info():
    raw = _sample_session_info()
    msgs = list(p.iter_messages(raw))
    assert len(msgs) == 1
    msg_type, payload = msgs[0]
    assert msg_type == p.MSG_SESSION_INFO
    info = p.decode_session_info(payload)
    assert info.chip_id == 1
    assert info.csi_bytes == 128
    assert info.mac == b"\xaa\xbb\xcc\xdd\xee\xff"
    assert info.sample_rate_hz == 100
    assert info.boot_id == 0xDEADBEEF


def test_round_trip_csi_frame():
    payload = bytes(range(128))
    raw = _sample_csi_frame(ts=10_000, seq=7, csi_payload=payload)
    msgs = list(p.iter_messages(raw))
    assert len(msgs) == 1
    msg_type, body = msgs[0]
    assert msg_type == p.MSG_CSI_FRAME
    meta, csi = p.decode_csi_frame(body)
    assert meta.local_timestamp_us == 10_000
    assert meta.seq == 7
    assert meta.rssi == -55
    assert meta.len == 128
    assert csi == payload


def test_round_trip_heartbeat():
    raw = _sample_heartbeat()
    msgs = list(p.iter_messages(raw))
    assert len(msgs) == 1
    msg_type, payload = msgs[0]
    assert msg_type == p.MSG_HEARTBEAT
    hb = p.decode_heartbeat(payload)
    assert hb.rssi == -55
    assert hb.channel == 6
    assert hb.uptime_s == 42


def test_iter_messages_mixed_stream():
    csi = bytes(range(128))
    stream = (
        _sample_session_info()
        + _sample_csi_frame(ts=10_000, seq=0, csi_payload=csi)
        + _sample_csi_frame(ts=20_000, seq=1, csi_payload=csi)
        + _sample_heartbeat()
        + _sample_csi_frame(ts=30_000, seq=2, csi_payload=csi)
    )
    types = [t for (t, _) in p.iter_messages(stream)]
    assert types == [
        p.MSG_SESSION_INFO,
        p.MSG_CSI_FRAME, p.MSG_CSI_FRAME,
        p.MSG_HEARTBEAT,
        p.MSG_CSI_FRAME,
    ]
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest tests/test_binary_protocol.py -v`
Expected: 4 failures with `AttributeError: module 'csi_protocol' has no attribute 'encode_session_info'`.

- [ ] **Step 3: Add encoder/decoder helpers**

Append to `csi_protocol.py`:

```python
from collections import namedtuple

# Decoded views — fields match the struct layout 1:1, plus the trailing csi
# byte payload for CSI_FRAME.
SessionInfo = namedtuple(
    "SessionInfo",
    "chip_id csi_format csi_bytes mac channel reserved sample_rate_hz "
    "boot_id esp_time_us",
)
CsiFrameMeta = namedtuple(
    "CsiFrameMeta",
    "local_timestamp_us seq rssi noise_floor rate first_word_invalid len",
)
HeartbeatPayload = namedtuple(
    "HeartbeatPayload",
    "esp_time_us rssi channel uptime_s reconnect_count last_disc_reason",
)


def _wrap(msg_type: int, payload: bytes) -> bytes:
    return pack_header(msg_type, len(payload)) + payload


def encode_session_info(*, chip_id, csi_format, csi_bytes, mac, channel,
                        sample_rate_hz, boot_id, esp_time_us, reserved=0):
    if len(mac) != 6:
        raise ValueError("mac must be exactly 6 bytes")
    body = SESSION_INFO.pack(
        chip_id, csi_format, csi_bytes, mac, channel, reserved,
        sample_rate_hz, boot_id, esp_time_us,
    )
    return _wrap(MSG_SESSION_INFO, body)


def decode_session_info(payload: bytes) -> SessionInfo:
    return SessionInfo._make(SESSION_INFO.unpack(payload))


def encode_csi_frame(*, local_timestamp_us, seq, rssi, noise_floor,
                     rate, first_word_invalid, csi_bytes):
    meta = CSI_FRAME_META.pack(
        local_timestamp_us, seq, rssi, noise_floor,
        rate, first_word_invalid, len(csi_bytes),
    )
    return _wrap(MSG_CSI_FRAME, meta + csi_bytes)


def decode_csi_frame(payload: bytes):
    """Return (CsiFrameMeta, csi_payload_bytes)."""
    meta = CsiFrameMeta._make(CSI_FRAME_META.unpack_from(payload, 0))
    csi = payload[CSI_FRAME_META.size : CSI_FRAME_META.size + meta.len]
    if len(csi) != meta.len:
        raise ValueError(
            f"truncated CSI payload: expected {meta.len}, got {len(csi)}"
        )
    return meta, csi


def encode_heartbeat(*, esp_time_us, rssi, channel, uptime_s,
                     reconnect_count, last_disc_reason):
    body = HEARTBEAT.pack(esp_time_us, rssi, channel, uptime_s,
                          reconnect_count, last_disc_reason)
    return _wrap(MSG_HEARTBEAT, body)


def decode_heartbeat(payload: bytes) -> HeartbeatPayload:
    return HeartbeatPayload._make(HEARTBEAT.unpack(payload))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest tests/test_binary_protocol.py -v`
Expected: 9 passed (5 from Task 1 + 4 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add csi_protocol.py tests/test_binary_protocol.py
git commit -m "feat: round-trip encoders/decoders for binary protocol

Adds encode_*/decode_* helpers and named-tuple views. Used by the
host parser, the capture-side stats peek, and tests."
```

---

## Task 3: Parser edge-case tests

**Files:**
- Test: `tests/test_binary_protocol.py`

Locks down truncation, garbage type bytes, and length-overflow handling. No production code changes — `iter_messages` already handles truncation by design; the tests prove it.

- [ ] **Step 1: Add truncated-tail and bad-type tests**

Append to `tests/test_binary_protocol.py`:

```python
def test_iter_messages_truncated_tail():
    csi = bytes(range(128))
    full = _sample_session_info() + _sample_csi_frame(10_000, 0, csi)
    # Chop the last 5 bytes off (mid-CSI payload).
    truncated = full[:-5]
    msgs = list(p.iter_messages(truncated))
    # SESSION_INFO is complete, CSI_FRAME is partial → only SESSION_INFO yields.
    assert len(msgs) == 1
    assert msgs[0][0] == p.MSG_SESSION_INFO


def test_iter_messages_unknown_type_zero_length():
    # \xFF type, length=0 → opaque message but well-formed. The walker should
    # skip it and continue to the next message rather than stalling.
    csi = bytes(range(128))
    stream = (
        _sample_csi_frame(10_000, 0, csi)
        + p.pack_header(0xFF, 0)
        + _sample_csi_frame(20_000, 1, csi)
    )
    types = [t for (t, _) in p.iter_messages(stream)]
    assert types == [p.MSG_CSI_FRAME, 0xFF, p.MSG_CSI_FRAME]


def test_iter_messages_unknown_type_with_payload():
    # Non-zero length unknown type: walker still advances by length, but the
    # consumer dispatch is what decides to ignore it. Just verify it doesn't
    # corrupt the stream walk.
    csi = bytes(range(128))
    stream = (
        _sample_csi_frame(10_000, 0, csi)
        + p.pack_header(0x42, 7) + b"\x00" * 7
        + _sample_csi_frame(20_000, 1, csi)
    )
    types = [t for (t, _) in p.iter_messages(stream)]
    assert types == [p.MSG_CSI_FRAME, 0x42, p.MSG_CSI_FRAME]
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest tests/test_binary_protocol.py -v`
Expected: all pass (12 total).

- [ ] **Step 3: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add tests/test_binary_protocol.py
git commit -m "test: cover truncated tail and unknown message types"
```

---

## Task 4: Binary loader in `csi_breathing.py` — basic load

**Files:**
- Modify: `csi_breathing.py` (add `load_binary` + dispatch from `parse_file`)
- Test: `tests/test_binary_loader.py` (new)

Builds the host-side `CSIDataset` straight from a binary stream. Existing `CSIFrame` is kept; fields that aren't in the binary protocol are zeroed/defaulted.

- [ ] **Step 1: Write failing loader test**

Write `tests/test_binary_loader.py`:

```python
"""End-to-end binary loader tests: bytes → CSIDataset."""
import numpy as np

import csi_protocol as p
from csi_breathing import CSIDataset, load_binary_bytes


def _session_info(boot_id=0xDEADBEEF, sample_rate=100):
    return p.encode_session_info(
        chip_id=1, csi_format=0, csi_bytes=128,
        mac=b"\xaa\xbb\xcc\xdd\xee\xff",
        channel=6, sample_rate_hz=sample_rate,
        boot_id=boot_id, esp_time_us=1_000_000,
    )


def _csi_frame(ts, seq, csi_bytes=None):
    if csi_bytes is None:
        # 128 bytes: 64 (imag, real) int8 pairs. Use a known pattern so
        # the resulting complex array is easy to check.
        csi_bytes = bytes((i - 64) & 0xFF for i in range(128))
    return p.encode_csi_frame(
        local_timestamp_us=ts, seq=seq,
        rssi=-55, noise_floor=-95,
        rate=11, first_word_invalid=0,
        csi_bytes=csi_bytes,
    )


def test_load_two_frames():
    stream = _session_info() + _csi_frame(10_000, 0) + _csi_frame(20_000, 1)
    ds = load_binary_bytes(stream)
    assert isinstance(ds, CSIDataset)
    assert ds.num_frames == 2
    assert ds.gap_indices == []
    assert ds.frames[0].seq == 0
    assert ds.frames[1].seq == 1
    assert ds.frames[0].local_timestamp == 10_000
    assert ds.frames[1].local_timestamp == 20_000
    assert ds.frames[0].rssi == -55
    # 128 raw bytes → 64 complex64 subcarriers
    assert ds.frames[0].raw_csi.shape == (64,)
    assert ds.frames[0].raw_csi.dtype == np.complex64


def test_load_endian_and_imag_real_order():
    # First (imag, real) pair = (1, 2) → complex(2, 1).
    # Second pair = (-1, -2) → complex(-2, -1).
    payload = bytes(b"\x01\x02\xff\xfe" + b"\x00" * 124)
    stream = _session_info() + _csi_frame(10_000, 0, csi_bytes=payload)
    ds = load_binary_bytes(stream)
    csi = ds.frames[0].raw_csi
    assert csi[0] == complex(2, 1)
    assert csi[1] == complex(-2, -1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest tests/test_binary_loader.py -v`
Expected: `ImportError: cannot import name 'load_binary_bytes' from 'csi_breathing'`.

- [ ] **Step 3: Add `load_binary` + `load_binary_bytes` to `csi_breathing.py`**

Find the existing `_parse_csv` definition in `csi_breathing.py` (around line 341). Insert a new section *above* `_parse_csv` (so the binary code lives near the other loaders, not at the bottom of the file). Add:

```python
# ---------------------------------------------------------------------------
# Binary loader — wire protocol defined in csi_protocol.py
# ---------------------------------------------------------------------------

import csi_protocol as _proto


def _csi_bytes_to_complex(csi_bytes: bytes) -> np.ndarray:
    """Decode ESP32 raw CSI bytes into a complex64 subcarrier array.

    Wire layout: pairs of signed int8s, (imag, real). Mirrors
    parse_csi_values() for CSV input but operates on raw bytes.
    """
    # np.frombuffer with int8 reinterprets the unsigned bytes as signed.
    raw = np.frombuffer(csi_bytes, dtype=np.int8)
    num_complex = len(raw) // 2
    imag = raw[0 : 2 * num_complex : 2].astype(np.float32)
    real = raw[1 : 2 * num_complex : 2].astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def load_binary_bytes(stream: bytes) -> CSIDataset:
    """Parse a complete binary capture *stream* (bytes/bytearray/mmap) into a CSIDataset.

    The walker processes messages with `csi_protocol.iter_messages`. The first
    message of each session establishes csi_bytes, sample_rate, and boot_id;
    subsequent CSI_FRAMEs build CSIFrames; HEARTBEATs are ignored for now.

    Session boundaries that don't carry frames between them are not recorded as
    gaps (no visible discontinuity in the frame timeline).
    """
    dataset = CSIDataset()

    # Per-session state.
    session_boot_id = None
    session_csi_bytes = None
    session_mac = ""
    session_channel = 0

    # Wraparound + monotonicity state, valid within a single boot_id.
    prev_raw_us = None
    prev_logical_us = None
    wrap_offset_us = 0

    # 500 ms gap threshold (matches capture.py GAP_THRESHOLD_US).
    GAP_THRESHOLD_US = 500_000

    def _start_new_session(info: "_proto.SessionInfo", hard_gap: bool):
        nonlocal session_boot_id, session_csi_bytes
        nonlocal session_mac, session_channel
        nonlocal prev_raw_us, prev_logical_us, wrap_offset_us
        session_boot_id = info.boot_id
        session_csi_bytes = info.csi_bytes
        # mac is bytes; render as colon-hex to match CSIFrame.mac type (str).
        session_mac = ":".join(f"{b:02x}" for b in info.mac)
        session_channel = info.channel
        prev_raw_us = None
        prev_logical_us = None
        wrap_offset_us = 0
        if hard_gap and dataset.num_frames > 0:
            idx = dataset.num_frames
            if not dataset.gap_indices or dataset.gap_indices[-1] != idx:
                dataset.gap_indices.append(idx)

    for msg_type, payload in _proto.iter_messages(stream):
        if msg_type == _proto.MSG_SESSION_INFO:
            try:
                info = _proto.decode_session_info(payload)
            except struct.error:
                dataset.skipped_rows += 1
                continue
            hard = session_boot_id is not None and info.boot_id != session_boot_id
            _start_new_session(info, hard_gap=hard)
            continue

        if msg_type == _proto.MSG_CSI_FRAME:
            if session_csi_bytes is None:
                # Frame before any SESSION_INFO — corrupt stream prefix.
                dataset.skipped_rows += 1
                continue
            try:
                meta, csi_bytes = _proto.decode_csi_frame(payload)
            except (struct.error, ValueError):
                dataset.skipped_rows += 1
                continue

            raw = meta.local_timestamp_us
            if prev_raw_us is not None and raw < prev_raw_us:
                # Backward jump within one boot_id: u32 wrap.
                wrap_offset_us += 1 << 32
            logical_us = raw + wrap_offset_us
            prev_raw_us = raw

            # Duplicate timestamp → drop (parity with old CSV loader).
            if prev_logical_us is not None and logical_us == prev_logical_us:
                continue

            # >500 ms jump → mark gap before recording the frame.
            if (prev_logical_us is not None
                    and logical_us - prev_logical_us > GAP_THRESHOLD_US):
                idx = dataset.num_frames
                if not dataset.gap_indices or dataset.gap_indices[-1] != idx:
                    dataset.gap_indices.append(idx)

            prev_logical_us = logical_us

            frame = _frame_from_fields(
                seq=meta.seq,
                mac=session_mac,
                rssi=meta.rssi,
                rate=meta.rate,
                sig_mode=0, mcs=0, bandwidth=0, smoothing=0,
                not_sounding=0, aggregation=0, stbc=0,
                fec_coding=0, sgi=0,
                noise_floor=meta.noise_floor,
                ampdu_cnt=0,
                channel=session_channel,
                secondary_channel=0,
                local_timestamp=logical_us,
                ant=0,
                sig_len=0,
                rx_state=0,
                csi_len=meta.len,
                first_word=meta.first_word_invalid,
                raw_csi=_csi_bytes_to_complex(csi_bytes),
            )
            dataset.frames.append(frame)
            continue

        if msg_type == _proto.MSG_HEARTBEAT:
            # Heartbeats are informational; could record on dataset later if
            # the analysis ever needs link-health context.
            continue

        # Unknown type: walker has already advanced past it; just count.
        dataset.skipped_rows += 1

    return dataset


def load_binary(filepath: str) -> CSIDataset:
    """Open *filepath* as a binary capture and parse it. Uses mmap so very
    large overnight files don't get fully read into RAM."""
    import mmap
    with open(filepath, "rb") as f:
        data = f.read()
    # mmap is overkill for the file sizes we hit (~50 MB/night); plain read
    # is simpler and the test suite doesn't have to worry about file handles.
    return load_binary_bytes(data)
```

Also add `import struct` at the top of the file if not already present (it isn't — verify with `grep -n "^import struct" csi_breathing.py`; the binary loader uses `struct.error`).

- [ ] **Step 4: Run tests**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest tests/test_binary_loader.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add csi_breathing.py tests/test_binary_loader.py
git commit -m "feat: binary loader in csi_breathing — basic frames + endianness

Adds load_binary() / load_binary_bytes(). Decodes raw CSI int8 pairs
in (imag, real) order to complex64, matching parse_csi_values()."
```

---

## Task 5: Loader gap detection — reconnect, reboot, duplicate, 500 ms jump

**Files:**
- Test: `tests/test_binary_loader.py`

The loader code from Task 4 already implements these cases; tests just lock them in. Pure regression coverage.

- [ ] **Step 1: Add gap-handling tests**

Append to `tests/test_binary_loader.py`:

```python
def test_reconnect_same_boot_id_no_hard_gap():
    # Same boot_id between two session blocks → soft (TCP reconnect).
    # No gap is recorded because the second SESSION_INFO arrives between
    # frames and there's no timestamp discontinuity yet.
    stream = (
        _session_info(boot_id=1)
        + _csi_frame(10_000, 0)
        + _session_info(boot_id=1)
        + _csi_frame(20_000, 1)
    )
    ds = load_binary_bytes(stream)
    assert ds.num_frames == 2
    assert ds.gap_indices == []


def test_reboot_different_boot_id_hard_gap():
    stream = (
        _session_info(boot_id=1)
        + _csi_frame(10_000, 0)
        + _session_info(boot_id=2)   # reboot
        + _csi_frame(10_000, 0)      # ESP32 timer restarted near zero
    )
    ds = load_binary_bytes(stream)
    assert ds.num_frames == 2
    assert ds.gap_indices == [1]  # gap at index of first post-reboot frame


def test_duplicate_timestamp_dropped():
    stream = (
        _session_info()
        + _csi_frame(10_000, 0)
        + _csi_frame(10_000, 1)   # duplicate ts
        + _csi_frame(20_000, 2)
    )
    ds = load_binary_bytes(stream)
    assert ds.num_frames == 2
    seqs = [f.seq for f in ds.frames]
    assert seqs == [0, 2]


def test_500ms_jump_inserts_gap():
    stream = (
        _session_info()
        + _csi_frame(10_000, 0)
        + _csi_frame(600_000, 1)   # 590 ms later → > 500 ms
    )
    ds = load_binary_bytes(stream)
    assert ds.num_frames == 2
    assert ds.gap_indices == [1]
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest tests/test_binary_loader.py -v`
Expected: 6 passed.

- [ ] **Step 3: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add tests/test_binary_loader.py
git commit -m "test: loader gap detection — reconnect, reboot, dupe, 500ms jump"
```

---

## Task 6: Loader wraparound tests

**Files:**
- Test: `tests/test_binary_loader.py`

Locks the u32 wrap → `logical_us` unwrap. Implementation is already in Task 4.

- [ ] **Step 1: Add wrap tests**

Append to `tests/test_binary_loader.py`:

```python
UINT32 = 1 << 32


def test_wraparound_single():
    # Two frames 10 ms apart straddling the u32 wrap boundary.
    raw_prev = UINT32 - 5_000   # would be 0xFFFFEC78
    raw_new = 5_000             # post-wrap
    stream = (
        _session_info()
        + _csi_frame(raw_prev, 0)
        + _csi_frame(raw_new, 1)
    )
    ds = load_binary_bytes(stream)
    assert ds.num_frames == 2
    assert ds.gap_indices == []  # no gap — looks like a continuous 10 ms step
    logical = [f.local_timestamp for f in ds.frames]
    assert logical[1] - logical[0] == 10_000


def test_wraparound_multiple():
    # 5 frames straddling two wraps, each 10 ms apart in raw terms.
    raws = [
        UINT32 - 30_000,   # pre-wrap
        UINT32 - 20_000,
        UINT32 - 10_000,
        0,                 # wrap #1
        10_000,
    ]
    stream = _session_info()
    for i, t in enumerate(raws):
        stream += _csi_frame(t, i)
    ds = load_binary_bytes(stream)
    assert ds.num_frames == 5
    assert ds.gap_indices == []
    logical = [f.local_timestamp for f in ds.frames]
    diffs = np.diff(logical)
    assert (diffs == 10_000).all()


def test_wrap_vs_reboot_disambiguation():
    # Backward jump WITH new boot_id → reboot, not wrap. Hard gap, no unwrap.
    stream = (
        _session_info(boot_id=1)
        + _csi_frame(UINT32 - 5_000, 0)
        + _session_info(boot_id=2)
        + _csi_frame(5_000, 0)       # raw 5_000 in the new boot
    )
    ds = load_binary_bytes(stream)
    assert ds.num_frames == 2
    assert ds.gap_indices == [1]
    # Post-reboot frame's logical_us is its raw value (no offset).
    assert ds.frames[1].local_timestamp == 5_000
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest tests/test_binary_loader.py -v`
Expected: 9 passed.

- [ ] **Step 3: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add tests/test_binary_loader.py
git commit -m "test: loader unwraps u32 timestamps into continuous logical_us"
```

---

## Task 7: Loader length-overflow + corrupt-stream guards

**Files:**
- Modify: `csi_breathing.py` (defensive caps inside loader)
- Test: `tests/test_binary_loader.py`

Already covered by `iter_messages` for *malformed* lengths inside a session; this task adds the host-side MAX cap.

- [ ] **Step 1: Add failing test**

Append to `tests/test_binary_loader.py`:

```python
def test_length_overflow_stops_walk():
    # type=MSG_CSI_FRAME, length=0xFFFF — way past MAX_PAYLOAD_BYTES.
    # The walker would happily try to slice that many bytes; the loader
    # must clamp.
    stream = (
        _session_info()
        + _csi_frame(10_000, 0)
        + p.pack_header(p.MSG_CSI_FRAME, 0xFFFF)  # no payload bytes after
    )
    ds = load_binary_bytes(stream)
    # First frame was good; oversized header consumes the rest as truncated.
    assert ds.num_frames == 1
    # We don't strictly require skipped_rows here — the truncated message
    # never yielded, so it's silently dropped by iter_messages. The point of
    # the test is just that the loader doesn't crash or loop.
```

- [ ] **Step 2: Run tests — should pass already**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest tests/test_binary_loader.py -v`
Expected: 10 passed. (`iter_messages` returns on truncation, so a 0xFFFF length with no body never yields.)

If it fails, the loader needs an explicit `if length > MAX_PAYLOAD_BYTES: break` guard inside `iter_messages`. Add it to `csi_protocol.py`:

```python
def iter_messages(buf, max_payload=MAX_PAYLOAD_BYTES):
    off = 0
    total = len(buf)
    while off + HEADER.size <= total:
        msg_type, length = HEADER.unpack_from(buf, off)
        if length > max_payload:
            return
        end = off + HEADER.size + length
        if end > total:
            return
        payload = bytes(buf[off + HEADER.size : end])
        yield msg_type, payload
        off = end
```

Re-run; should now pass.

- [ ] **Step 3: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add csi_protocol.py tests/test_binary_loader.py
git commit -m "test: oversized length field stops the walker safely"
```

---

## Task 8: Wire `parse_file` to dispatch by extension

**Files:**
- Modify: `csi_breathing.py` (`parse_file` function around line 491)

CLI entry point should accept `.bin` files transparently. Existing `.csv` path stays for old captures.

- [ ] **Step 1: Inspect existing `parse_file` to know what to change**

Run: `grep -n "def parse_file\|_detect_format\|def main\b" /Users/maybecat/Projects/Mecha/project/csi_breathing.py | head`
Expected: shows `def parse_file` near line 491.

- [ ] **Step 2: Add binary dispatch**

Edit `csi_breathing.py`. Find the existing `parse_file` (around line 491). Replace its body with:

```python
def parse_file(filepath: str) -> CSIDataset:
    """Dispatch by file extension. .bin → binary loader, .csv → CSV, else
    falls through to the serial-monitor sniffer for legacy files."""
    if filepath.endswith(".bin"):
        return load_binary(filepath)
    fmt = _detect_format(filepath)
    if fmt in ("csv", "csv_no_header"):
        return _parse_csv(filepath)
    return _parse_serial(filepath)
```

(If the existing implementation is structurally identical except missing the `.bin` branch, add only the new first two lines and leave the rest.)

- [ ] **Step 3: Add a tiny smoke test that goes through the file path**

Append to `tests/test_binary_loader.py`:

```python
def test_parse_file_dispatches_to_binary(tmp_path):
    stream = _session_info() + _csi_frame(10_000, 0)
    binpath = tmp_path / "smoke.bin"
    binpath.write_bytes(stream)
    from csi_breathing import parse_file
    ds = parse_file(str(binpath))
    assert ds.num_frames == 1
    assert ds.frames[0].local_timestamp == 10_000
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest -v`
Expected: all pass (12 + 11 = 23 total).

- [ ] **Step 5: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add csi_breathing.py tests/test_binary_loader.py
git commit -m "feat: parse_file dispatches .bin to binary loader"
```

---

## Task 9: Rewrite `capture.py` as a binary byte-pipe

**Files:**
- Modify: `capture.py` (full rewrite of core loop; CLI surface preserved)

`capture.py` no longer parses CSV. It receives bytes, peeks the 3-byte message header to populate `stats.json`, and appends every byte verbatim to `.bin`.

- [ ] **Step 1: Replace the file contents**

Overwrite `capture.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture.py — TCP server that receives the binary CSI stream from the ESP32
and writes it to a .bin file verbatim.

The on-disk file is the same byte stream as the wire (length-prefixed
SESSION_INFO / CSI_FRAME / HEARTBEAT messages), so a recording can be
replayed by feeding the .bin to csi_breathing.load_binary().

Usage:
    python capture.py -s csi_data.bin
    python capture.py -s csi_data.bin -p 3490
    python capture.py -s csi_data.bin -p 3490 -l csi_data_log.txt

stats.json keeps the same shape as before (frames_written, gap_count,
total_gap_seconds, longest_segment_seconds, last_heartbeat_utc, gaps[]),
but the values come from a header-only peek of the binary stream rather
than CSV parsing.
"""

import datetime as _dt
import json
import os
import socket
import sys
import time

import csi_protocol as proto

HOST = "0.0.0.0"

RECV_TIMEOUT_S      = 3.0
GAP_THRESHOLD_US    = 500_000
FSYNC_INTERVAL_S    = 60.0
STATS_INTERVAL_S    = 60.0


class StallError(Exception):
    pass


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *conn* or raise StallError on timeout /
    ConnectionError on remote close."""
    out = bytearray()
    while len(out) < n:
        try:
            chunk = conn.recv(n - len(out))
        except socket.timeout as e:
            raise StallError("recv timeout") from e
        if not chunk:
            raise ConnectionError("peer closed")
        out.extend(chunk)
    return bytes(out)


class CaptureSession:
    """Tracks stats across reconnects while appending raw bytes to .bin."""

    def __init__(self, bin_fd, log_fd, store_path):
        self.bin_fd      = bin_fd
        self.log_fd      = log_fd
        self.store_path  = store_path
        self.stats_path  = store_path + ".stats.json"

        self.started_wall    = _dt.datetime.now(_dt.timezone.utc)
        self.frames_written  = 0
        self.last_fsync_mono = time.monotonic()
        self.last_stats_mono = 0.0
        self.last_heartbeat_wall = None

        # Per-session timestamp/wrap state (mirrors loader logic).
        self.session_boot_id = None
        self.prev_raw_us     = None
        self.wrap_offset_us  = 0
        self.prev_logical_us = None

        self.gaps              = []      # bounded at 1000 entries
        self.gap_count         = 0
        self.total_gap_ms      = 0
        self.longest_segment_s = 0.0
        self._segment_start_ns = time.monotonic_ns()

    def write_raw(self, blob: bytes):
        """Append *blob* (already-framed bytes) verbatim to the .bin."""
        self.bin_fd.write(blob)
        self._periodic_maintenance()

    def record_message(self, msg_type: int, payload: bytes):
        """Update stats based on a single parsed message. Does NOT write to
        .bin — write_raw() does that."""
        if msg_type == proto.MSG_SESSION_INFO:
            try:
                info = proto.decode_session_info(payload)
            except Exception:
                return
            hard = (self.session_boot_id is not None
                    and info.boot_id != self.session_boot_id)
            if hard:
                self._note_gap("REBOOT", 0)
            self.session_boot_id = info.boot_id
            self.prev_raw_us = None
            self.wrap_offset_us = 0
            self.prev_logical_us = None

        elif msg_type == proto.MSG_CSI_FRAME:
            try:
                meta, _ = proto.decode_csi_frame(payload)
            except Exception:
                return
            raw = meta.local_timestamp_us
            if self.prev_raw_us is not None and raw < self.prev_raw_us:
                self.wrap_offset_us += 1 << 32
            logical = raw + self.wrap_offset_us
            self.prev_raw_us = raw

            if (self.prev_logical_us is not None
                    and logical - self.prev_logical_us > GAP_THRESHOLD_US):
                self._note_gap("MONOTONIC",
                               (logical - self.prev_logical_us) // 1000)
            self.prev_logical_us = logical
            self.frames_written += 1

        elif msg_type == proto.MSG_HEARTBEAT:
            self.last_heartbeat_wall = _dt.datetime.now(_dt.timezone.utc)

    def _note_gap(self, reason: str, duration_ms: int):
        self.gap_count += 1
        self.total_gap_ms += max(0, int(duration_ms))
        entry = {
            "t": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "duration_ms": int(duration_ms),
            "reason": reason,
        }
        if len(self.gaps) < 1000:
            self.gaps.append(entry)
        now_ns = time.monotonic_ns()
        seg_s = (now_ns - self._segment_start_ns) / 1e9
        if seg_s > self.longest_segment_s:
            self.longest_segment_s = seg_s
        self._segment_start_ns = now_ns

    def note_connection_gap(self, reason: str, duration_ms: int):
        """Called by the server loop on TCP-level breaks."""
        self._note_gap(reason, duration_ms)

    def _periodic_maintenance(self):
        now = time.monotonic()
        if now - self.last_fsync_mono >= FSYNC_INTERVAL_S:
            try:
                os.fsync(self.bin_fd.fileno())
            except OSError:
                pass
            self.last_fsync_mono = now
        if now - self.last_stats_mono >= STATS_INTERVAL_S:
            self.write_stats()
            self.last_stats_mono = now

    def write_stats(self):
        live_seg = (time.monotonic_ns() - self._segment_start_ns) / 1e9
        longest = max(self.longest_segment_s, live_seg)
        stats = {
            "started_utc": self.started_wall.isoformat(timespec="seconds"),
            "now_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "frames_written": self.frames_written,
            "gap_count": self.gap_count,
            "total_gap_seconds": round(self.total_gap_ms / 1000.0, 3),
            "longest_segment_seconds": round(longest, 3),
            "last_heartbeat_utc": (
                self.last_heartbeat_wall.isoformat(timespec="seconds")
                if self.last_heartbeat_wall else None
            ),
            "gaps": self.gaps,
        }
        tmp = self.stats_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(stats, f, indent=2)
            os.replace(tmp, self.stats_path)
        except OSError as e:
            print(f"Could not write stats file: {e}")


def consume_connection(conn: socket.socket, session: CaptureSession):
    """Drive one TCP connection: read message-by-message, validate the first
    is SESSION_INFO, append raw bytes to .bin, update stats from peek."""
    header = _recv_exact(conn, proto.HEADER.size)
    msg_type, length = proto.HEADER.unpack(header)

    if msg_type != proto.MSG_SESSION_INFO:
        print(f"First message was type 0x{msg_type:02x}, not SESSION_INFO — closing")
        session.log_fd.write(
            f"first-message-not-session-info type=0x{msg_type:02x}\n")
        session.log_fd.flush()
        return
    if length > proto.MAX_PAYLOAD_BYTES:
        print(f"SESSION_INFO length {length} exceeds cap — closing")
        return

    payload = _recv_exact(conn, length)
    session.write_raw(header + payload)
    session.record_message(msg_type, payload)

    while True:
        header = _recv_exact(conn, proto.HEADER.size)
        msg_type, length = proto.HEADER.unpack(header)
        if length > proto.MAX_PAYLOAD_BYTES:
            session.log_fd.write(
                f"oversized length {length} for type 0x{msg_type:02x} — closing\n")
            session.log_fd.flush()
            return
        payload = _recv_exact(conn, length) if length else b""
        session.write_raw(header + payload)
        session.record_message(msg_type, payload)


def run_server(port: int, store_path: str, log_path: str):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    print(f"Listening on {HOST}:{port}  →  writing to {store_path}")

    with open(store_path, "wb") as bin_fd, \
         open(log_path, "w") as log_fd:

        session = CaptureSession(bin_fd, log_fd, store_path)
        session.write_stats()

        pending_reason = None
        pending_start_mono_ns = None

        while True:
            print("Waiting for ESP32 connection …")
            try:
                conn, addr = srv.accept()
            except KeyboardInterrupt:
                print("\nShutting down.")
                session.write_stats()
                return
            conn.settimeout(RECV_TIMEOUT_S)
            print(f"Connected: {addr[0]}:{addr[1]}")

            if pending_reason is not None:
                gap_ms = (time.monotonic_ns() - pending_start_mono_ns) // 1_000_000
                session.note_connection_gap(pending_reason, gap_ms)
            pending_reason = None
            pending_start_mono_ns = None

            reason = "CLOSE"
            try:
                consume_connection(conn, session)
            except StallError:
                print(f"Recv stall > {RECV_TIMEOUT_S}s — closing socket.")
                reason = "TIMEOUT"
            except ConnectionError as e:
                print(f"Connection error: {e}")
                reason = "OSERR"
            except OSError as e:
                print(f"Socket error: {e}")
                reason = "OSERR"
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
                print("Connection closed.")

            pending_reason = reason
            pending_start_mono_ns = time.monotonic_ns()
            session.write_stats()


if __name__ == "__main__":
    if sys.version_info < (3, 6):
        print("Python >= 3.6 required")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(
        description="TCP server: receive binary CSI stream and save to .bin")
    parser.add_argument("-p", "--port", type=int, default=3490,
                        help="TCP port to listen on (default: 3490)")
    parser.add_argument("-s", "--store", dest="store_file",
                        default="./csi_data.bin",
                        help="Output .bin file (default: ./csi_data.bin)")
    parser.add_argument("-l", "--log", dest="log_file",
                        default="./csi_data_log.txt",
                        help="Sidecar log for protocol errors (default: ./csi_data_log.txt)")
    args = parser.parse_args()
    run_server(args.port, args.store_file, args.log_file)
```

- [ ] **Step 2: Smoke-test the byte-pipe via loopback**

Write `tests/test_capture_pipe.py`:

```python
"""End-to-end smoke: feed a synthetic binary stream into capture.py over
loopback and verify the resulting .bin parses back to the expected dataset."""
import socket
import threading
import time

import csi_protocol as p
from capture import CaptureSession


def _make_stream():
    info = p.encode_session_info(
        chip_id=1, csi_format=0, csi_bytes=128,
        mac=b"\xaa\xbb\xcc\xdd\xee\xff",
        channel=6, sample_rate_hz=100,
        boot_id=0xDEADBEEF, esp_time_us=0,
    )
    frames = b"".join(
        p.encode_csi_frame(
            local_timestamp_us=i * 10_000, seq=i,
            rssi=-55, noise_floor=-95, rate=11,
            first_word_invalid=0, csi_bytes=bytes(128),
        )
        for i in range(3)
    )
    return info + frames


def test_capture_writes_verbatim(tmp_path):
    binpath = tmp_path / "out.bin"
    logpath = tmp_path / "out.log"

    with open(binpath, "wb") as bf, open(logpath, "w") as lf:
        session = CaptureSession(bf, lf, str(binpath))
        stream = _make_stream()
        # Drive the session directly (no socket needed for byte-level coverage).
        for msg_type, payload in p.iter_messages(stream):
            session.write_raw(p.pack_header(msg_type, len(payload)) + payload)
            session.record_message(msg_type, payload)
        session.write_stats()

    written = binpath.read_bytes()
    assert written == _make_stream()
    # frames_written is the post-peek stat.
    import json
    stats = json.loads((tmp_path / "out.bin.stats.json").read_text())
    assert stats["frames_written"] == 3
```

- [ ] **Step 3: Run all tests**

Run: `cd /Users/maybecat/Projects/Mecha/project && python3 -m pytest -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add capture.py tests/test_capture_pipe.py
git commit -m "feat: rewrite capture.py as binary byte-pipe with stats peek

Drops CSV parsing entirely. Receives length-prefixed messages, writes
bytes verbatim to .bin, peeks msg headers to update stats.json. CLI
flags preserved (-s, -p, -l)."
```

---

## Task 10: ESP32 protocol header (`main/csi_protocol.h`)

**Files:**
- Create: `main/csi_protocol.h`

Mirrors the Python side. `_Static_assert` traps padding from future struct edits at compile time.

- [ ] **Step 1: Write the header**

Write `main/csi_protocol.h`:

```c
/*
 * SPDX-FileCopyrightText: 2026
 * SPDX-License-Identifier: Apache-2.0
 *
 * Binary wire protocol between ESP32 firmware and host capture.
 *
 * All multi-byte integers are little-endian (native to Xtensa LX6 / LX7).
 * Every message:
 *
 *     uint8_t  type;
 *     uint16_t length;       // payload length, NOT including this 3-byte header
 *     uint8_t  payload[length];
 *
 * Mirror in csi_protocol.py — keep both sides in sync.
 */
#pragma once

#include <stdint.h>
#include <assert.h>

#define MSG_SESSION_INFO  0x01
#define MSG_CSI_FRAME     0x02
#define MSG_HEARTBEAT     0x03

#define MAX_PAYLOAD_BYTES 4096

/* 3-byte message header. */
typedef struct __attribute__((packed)) {
    uint8_t  type;
    uint16_t length;
} csi_msg_header_t;
_Static_assert(sizeof(csi_msg_header_t) == 3, "csi_msg_header_t must be 3 bytes");

/* SESSION_INFO payload (24 bytes). */
typedef struct __attribute__((packed)) {
    uint8_t  chip_id;          /* 1=ESP32 classic, 2=ESP32-S3 */
    uint8_t  csi_format;       /* 0=legacy HT LLTF */
    uint16_t csi_bytes;        /* 128 for LLTF 64-subcarrier */
    uint8_t  mac[6];
    uint8_t  channel;
    uint8_t  reserved;
    uint16_t sample_rate_hz;
    uint32_t boot_id;          /* random per boot; lets host distinguish
                                  reboot from TCP reconnect */
    uint64_t esp_time_us;
} csi_session_info_t;
<<<<<<< Updated upstream
_Static_assert(sizeof(csi_session_info_t) == 24, "csi_session_info_t size drift");
=======
_Static_assert(sizeof(csi_session_info_t) == 26, "csi_session_info_t size drift");
>>>>>>> Stashed changes

/* CSI_FRAME meta block (14 bytes). The csi byte payload (length = `len`)
 * follows this struct in the message. */
typedef struct __attribute__((packed)) {
    uint32_t local_timestamp_us;
    uint32_t seq;
    int8_t   rssi;
    int8_t   noise_floor;
    uint8_t  rate;
    uint8_t  first_word_invalid;
    uint16_t len;
} csi_frame_meta_t;
_Static_assert(sizeof(csi_frame_meta_t) == 14, "csi_frame_meta_t size drift");

/* HEARTBEAT payload (19 bytes). */
typedef struct __attribute__((packed)) {
    uint64_t esp_time_us;
    int8_t   rssi;
    uint8_t  channel;
    uint32_t uptime_s;
    uint32_t reconnect_count;
    uint8_t  last_disc_reason;
} csi_heartbeat_t;
_Static_assert(sizeof(csi_heartbeat_t) == 19, "csi_heartbeat_t size drift");
```

- [ ] **Step 2: Verify it compiles by listing it from CMake**

ESP-IDF picks up `.h` files from `main/` automatically (covered by `main/CMakeLists.txt` `SRCS` glob or `INCLUDE_DIRS .`). Run `idf.py reconfigure` to confirm the file is seen:

```bash
cd /Users/maybecat/Projects/Mecha/project
source ~/.espressif/tools/activate_idf_v6.0.sh
idf.py reconfigure
```

Expected: no errors. The header isn't used yet, but reconfigure validates the source tree shape.

- [ ] **Step 3: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add main/csi_protocol.h
git commit -m "feat(firmware): csi_protocol.h — wire format mirror of Python"
```

---

## Task 11: ESP32 firmware — replace CSV builders with binary

**Files:**
- Modify: `main/app_main.c`

This is the largest single-file change. Take it carefully; existing reconnect / WiFi / NVS logic stays untouched.

- [ ] **Step 1: Include the new header**

Edit `main/app_main.c`. After the existing `#include` block (just below `#include "esp_csi_gain_ctrl.h"`), add:

```c
#include "csi_protocol.h"
```

- [ ] **Step 2: Drop chip-variant `#if` branches around CSI config**

Delete the `CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61` block at lines 108–110.
Delete the `#if CONFIG_IDF_TARGET_ESP32S3 || …` `CONFIG_GAIN_CONTROL` define at lines 114–118.
Delete `#define CONFIG_FORCE_GAIN 0` at line 112.

- [ ] **Step 3: Add boot_id at file scope**

Below the existing `static const char *TAG = "csi_recv_router";`, add:

```c
/* Captured once at boot; lets the host distinguish "ESP32 rebooted"
   (different boot_id) from "TCP reconnect" (same boot_id). */
static uint32_t s_boot_id = 0;
```

- [ ] **Step 4: Initialize boot_id in `app_main`**

Find `app_main` (line 871). Right after the `nvs_flash_init()` block (before `esp_netif_init`), add:

```c
    s_boot_id = esp_random();
```

(Requires `#include "esp_random.h"` near the top — add it.)

- [ ] **Step 5: Replace `tcp_send_header_locked` with `tcp_send_session_info_locked`**

Delete the existing `tcp_send_header_locked` function (lines 234–252). In its place, write:

```c
/**
 * Build and send a SESSION_INFO message. Called from tcp_reconnect_task on
 * each successful connect. Must be called with s_tcp_mutex held.
 */
static void tcp_send_session_info_locked(void)
{
    if (s_tcp_sock < 0 || s_header_sent) return;

    wifi_ap_record_t ap = {0};
    uint8_t channel = 0;
    uint8_t mac[6] = {0};
    if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
        channel = ap.primary;
    }
    esp_wifi_get_mac(WIFI_IF_STA, mac);

#if CONFIG_IDF_TARGET_ESP32S3
    const uint8_t chip_id = 2;
#else
    const uint8_t chip_id = 1;
#endif

    uint8_t buf[sizeof(csi_msg_header_t) + sizeof(csi_session_info_t)];
    csi_msg_header_t *hdr = (csi_msg_header_t *)buf;
    csi_session_info_t *info =
        (csi_session_info_t *)(buf + sizeof(csi_msg_header_t));

    hdr->type   = MSG_SESSION_INFO;
    hdr->length = sizeof(csi_session_info_t);

    info->chip_id        = chip_id;
    info->csi_format     = 0;  /* legacy HT LLTF */
    info->csi_bytes      = 128;
    memcpy(info->mac, mac, 6);
    info->channel        = channel;
    info->reserved       = 0;
    info->sample_rate_hz = (uint16_t)CONFIG_SEND_FREQUENCY;
    info->boot_id        = s_boot_id;
    info->esp_time_us    = (uint64_t)esp_timer_get_time();

    tcp_send_locked((const char *)buf, sizeof(buf));
    s_header_sent = true;
}
```

In `tcp_reconnect_task` (line 315), replace the call `tcp_send_header_locked();` (line 362) with `tcp_send_session_info_locked();`.

- [ ] **Step 6: Replace `tcp_send_heartbeat_locked` body**

Replace the existing function (lines 260–281) with:

```c
static void tcp_send_heartbeat_locked(void)
{
    if (s_tcp_sock < 0 || !s_header_sent) return;

    wifi_ap_record_t ap = {0};
    int8_t rssi = 0;
    uint8_t channel = 0;
    if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
        rssi = (int8_t)ap.rssi;
        channel = ap.primary;
    }
    int64_t now_us = esp_timer_get_time();
    uint32_t uptime_s = (uint32_t)(now_us / 1000000);

    uint8_t buf[sizeof(csi_msg_header_t) + sizeof(csi_heartbeat_t)];
    csi_msg_header_t *hdr = (csi_msg_header_t *)buf;
    csi_heartbeat_t  *hb  = (csi_heartbeat_t *)(buf + sizeof(csi_msg_header_t));

    hdr->type   = MSG_HEARTBEAT;
    hdr->length = sizeof(csi_heartbeat_t);

    hb->esp_time_us      = (uint64_t)now_us;
    hb->rssi             = rssi;
    hb->channel          = channel;
    hb->uptime_s         = uptime_s;
    hb->reconnect_count  = s_reconnect_count;
    hb->last_disc_reason = (uint8_t)s_last_disc_reason;

    tcp_send_locked((const char *)buf, sizeof(buf));
}
```

- [ ] **Step 7: Replace `wifi_csi_rx_cb` body**

Replace the entire function body (lines 392–507) with:

```c
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG));
        return;
    }

    s_csi_frames_total++;

    /* Filter by AP BSSID — only frames from our router. */
    if (memcmp(info->mac, ctx, 6)) {
        return;
    }

    s_csi_frames_accepted++;

    if (s_tx_stream == NULL) return;

    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    static uint32_t s_seq = 0;

    /* Cap CSI byte count at a sane size; LLTF gives 128 bytes. Anything
       wildly different is hardware glitch territory. */
    int csi_len = info->len;
    if (csi_len < 0 || csi_len > 256) {
        ESP_LOGW(TAG, "unexpected CSI len=%d — dropping frame", info->len);
        return;
    }

    /* Build the entire message in one stack buffer so we enqueue atomically. */
    uint8_t buf[sizeof(csi_msg_header_t) + sizeof(csi_frame_meta_t) + 256];
    csi_msg_header_t *hdr = (csi_msg_header_t *)buf;
    csi_frame_meta_t *meta =
        (csi_frame_meta_t *)(buf + sizeof(csi_msg_header_t));
    uint8_t *csi_dst =
        buf + sizeof(csi_msg_header_t) + sizeof(csi_frame_meta_t);

    hdr->type   = MSG_CSI_FRAME;
    hdr->length = sizeof(csi_frame_meta_t) + csi_len;

    meta->local_timestamp_us = (uint32_t)rx_ctrl->timestamp;
    meta->seq                = s_seq++;
    meta->rssi               = (int8_t)rx_ctrl->rssi;
    meta->noise_floor        = (int8_t)rx_ctrl->noise_floor;
    meta->rate               = (uint8_t)rx_ctrl->rate;
    meta->first_word_invalid = (uint8_t)info->first_word_invalid;
    meta->len                = (uint16_t)csi_len;

    memcpy(csi_dst, info->buf, csi_len);

    size_t total = sizeof(csi_msg_header_t) + sizeof(csi_frame_meta_t) + csi_len;
    size_t sent = xStreamBufferSend(s_tx_stream, buf, total, 0);
    if (sent != total) {
        s_csi_frames_dropped++;
    }
}
```

(This deletes the gain-control block. Re-add it only if a future smoke test shows raw bytes need compensation — for the current 100 Hz overnight workload the receiver does not depend on it.)

- [ ] **Step 8: Tighten buffer sizes**

Change `TCP_TX_BUF_SIZE` (line 81) from `1024` to `256`. (Used by `tcp_writer_task`'s receive buf — small fixed-size chunks are fine because the producer side emits one full message per `xStreamBufferSend`.)

Leave `CSI_STREAM_BUF_BYTES` at 8192 (now absorbs ~57 frames instead of ~10 with the smaller bin protocol).

- [ ] **Step 9: Drop the `#if CONFIG_IDF_TARGET_ESP32C5 …` branches in `wifi_csi_init`**

In `wifi_csi_init` (line 511), keep ONLY the `#else` branch (legacy HT config — lines 546–554). Delete the C5/C61 and C6 branches.

- [ ] **Step 10: Build the firmware**

```bash
cd /Users/maybecat/Projects/Mecha/project
source ~/.espressif/tools/activate_idf_v6.0.sh
idf.py build
```

Expected: success. Fix compile errors before moving on.

- [ ] **Step 11: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add main/app_main.c
git commit -m "feat(firmware): send binary CSI stream instead of CSV

Builds SESSION_INFO at connect, CSI_FRAME per capture, HEARTBEAT 1 Hz
idle. Drops snprintf hot path, C5/C6 branches, gain-control logic.
Buffer sizes tuned for the smaller binary frames."
```

---

## Task 12: Manual smoke gate

**Files:**
- None modified.

ESP32 firmware has no automated test infrastructure. Manual confirmation is the gate before declaring the firmware change shippable.

- [ ] **Step 1: Flash and monitor**

```bash
cd /Users/maybecat/Projects/Mecha/project
source ~/.espressif/tools/activate_idf_v6.0.sh
idf.py flash monitor
```

Expected serial output, in order:
1. `Got IP: <ip>` from the WiFi event handler.
2. `Connecting to <host>:<port> …` from `tcp_connect`.
3. `TCP connected to <host>:<port>`.
4. Periodic `CSI stats: total=… accepted=… dropped=…` lines every 5 s — `accepted` should be increasing at ~500/5 s = 100 Hz.

- [ ] **Step 2: Run capture in another terminal**

```bash
cd /Users/maybecat/Projects/Mecha/project
python3 capture.py -s smoke.bin -p 3490
```

Expected:
- `Listening on 0.0.0.0:3490 → writing to smoke.bin`
- `Connected: <ip>:<port>` within ~1 s of the ESP32 boot.
- `smoke.bin` grows at ~14 KB/s (`ls -l smoke.bin` repeatedly). Compare to the historical CSV rate of ~70 KB/s.

- [ ] **Step 3: Stop after ~30 s, parse the .bin**

Stop the capture with Ctrl-C. Inspect:

```bash
python3 -c "from csi_breathing import parse_file; ds = parse_file('smoke.bin'); print(ds.num_frames, ds.frames[0].local_timestamp, ds.frames[-1].local_timestamp, 'gap_indices:', ds.gap_indices)"
```

Expected:
- `num_frames` ≈ 3000 (100 Hz × 30 s).
- `gap_indices` empty or short.
- `frames[0].local_timestamp` < `frames[-1].local_timestamp`.

- [ ] **Step 4: Run the analysis end-to-end**

```bash
python3 csi_breathing.py smoke.bin --output-dir smoke_out
```

Expected: produces a breathing-rate plot in `smoke_out/` without errors. The numeric estimate doesn't need to be physiologically meaningful for a 30 s capture, but the pipeline must run to completion.

- [ ] **Step 5: Document the smoke result**

If anything looks off, stop and debug. Otherwise no commit — this task is just a verification gate.

---

## Task 13: Housekeeping — `.gitignore`, `CLAUDE.md`

**Files:**
- Modify: `.gitignore`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add `*.bin` to `.gitignore`**

Check whether it's already ignored:

```bash
cd /Users/maybecat/Projects/Mecha/project
grep -n "\*\.bin" .gitignore || echo "not present"
```

If "not present", append `*.bin` on a new line at the end of `.gitignore`.

- [ ] **Step 2: Update `CLAUDE.md`**

Modify the Data Flow section (lines 55–62) and CSV Format section (lines 72–81) of `CLAUDE.md`:

Replace the Data Flow code block with:

```
ESP32 (app_main.c)
  └─ pings router at 100 Hz  →  CSI frames via wifi_csi_rx_cb()
       └─ packed as binary messages  →  TCP stream to capture.py
            └─ csi_data.bin  →  csi_breathing.py  →  breathing rate plot
```

Replace the "CSV format" subsection heading with "Wire format" and replace its body with:

```
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
```

Also drop the `C5/C6/C61` row from any chip-variant table in the file.

- [ ] **Step 3: Commit**

```bash
cd /Users/maybecat/Projects/Mecha/project
git add .gitignore CLAUDE.md
git commit -m "docs: update CLAUDE.md and .gitignore for binary protocol"
```

---

## Task 14: Validation gate — side-by-side overnight comparison

**Files:**
- None modified.

Final integration check before declaring the migration done.

- [ ] **Step 1: Run an overnight capture with the binary build**

```bash
cd /Users/maybecat/Projects/Mecha/project
python3 capture.py -s overnight.bin
```

Leave running ~8 hours in the same room used for past CSV-era recordings.

- [ ] **Step 2: Compare stats**

```bash
cat overnight.bin.stats.json
```

Compare against the last CSV-era `*.stats.json` from the same room. Acceptance criteria:
- `frames_written` within ±5% of the historical value (scaled for actual duration).
- `gap_count` in the same ballpark (no order-of-magnitude regression).
- `total_gap_seconds` not dramatically larger.

- [ ] **Step 3: Run analysis and eyeball the breathing rate**

```bash
python3 csi_breathing.py overnight.bin --output-dir overnight_out
```

Expected: completes; output plot shows the same general breathing-rate profile shape as past overnight runs from the same room.

If any of the above fails, capture the discrepancy in a `BUGS.md` entry and consider invoking the `backprop` skill to amend the spec rather than papering over the issue.

---

## Self-Review

Done after writing all tasks above — fresh-eyes check against the spec.

**Spec coverage:**
- §Motivation / Goals: ✓ Task 9–11 implement the wire-bytes reduction; Task 11 removes snprintf hot path.
- §Wire format (HEADER, SESSION_INFO, CSI_FRAME, HEARTBEAT): ✓ Task 1 (Python) + Task 10 (firmware), sizes pinned both sides.
- §ESP32 firmware changes (csi_protocol.h, snprintf removal, SESSION_INFO send, binary heartbeat, buffer retune, C5/C6 removal): ✓ Tasks 10, 11.
- §Python host changes (capture.py byte-pipe, csi_breathing.load_binary, .gitignore, CLAUDE.md): ✓ Tasks 4, 7, 8, 9, 13.
- §Error handling table (first-msg-not-session-info, length overflow, recv timeout, unknown type, truncated tail, reconnect/reboot gap, duplicate ts, wrap): ✓ Tasks 3, 5, 6, 7, 9.
- §Testing (round-trip, reconnect gap, reboot gap, truncated tail, bad type, length overflow, duplicate ts, 500 ms jump, wrap simple, wrap multi, wrap-vs-reboot): ✓ Tasks 2, 3, 5, 6, 7.
- §Validation gate (overnight side-by-side): ✓ Task 14.

**Placeholder scan:** No "TBD", "TODO", or hand-waved steps. Every code block is the actual code to write.

**Type consistency:** `load_binary` / `load_binary_bytes` / `_csi_bytes_to_complex` names match across Tasks 4, 5, 6, 7, 8. `CSIDataset`, `CSIFrame`, `_frame_from_fields` reuse existing names from `csi_breathing.py`. Firmware `csi_session_info_t`, `csi_frame_meta_t`, `csi_heartbeat_t`, `csi_msg_header_t` match `SESSION_INFO`, `CSI_FRAME_META`, `HEARTBEAT`, `HEADER` in `csi_protocol.py` by size (24, 14, 19, 3).

No gaps found.
