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
