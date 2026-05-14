"""Binary wire protocol between ESP32 firmware and host capture.

All multibyte integers are little-endian. Every message on the wire is:

    u8  type
    u16 length        (length of payload, not including this 3-byte header)
    u8  payload[length]

See docs/superpowers/specs/2026-05-14-binary-csi-protocol-design.md.
"""
import struct
from collections import namedtuple

# ── Message types ──────────────────────────────────────────────────────────
MSG_SESSION_INFO = 0x01
MSG_CSI_FRAME    = 0x02
MSG_HEARTBEAT    = 0x03

# ── Limits ─────────────────────────────────────────────────────────────────
MAX_PAYLOAD_BYTES = 4096   # host parser closes the socket if length exceeds this

# ── Gap detection (shared between live capture.py and offline loader) ──────
# An inter-frame gap in `local_timestamp_us` greater than this is treated as
# a discontinuity (CSI_GAP marker live; gap_indices entry offline).
GAP_THRESHOLD_US = 500_000   # 500 ms

# u32 wrap boundary for ESP32's local_timestamp_us (~71.6 min).
U32_WRAP = 1 << 32

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
