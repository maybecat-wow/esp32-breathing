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
    assert p.SESSION_INFO.size == 26


def test_csi_frame_meta_struct_size():
    # u32 local_timestamp_us, u32 seq, i8 rssi, i8 noise_floor,
    # u8 rate, u8 first_word_invalid, u16 len
    assert p.CSI_FRAME_META.size == 14


def test_heartbeat_struct_size():
    # u64 esp_time_us, i8 rssi, u8 channel, u32 uptime_s,
    # u32 reconnect_count, u8 last_disc_reason
    assert p.HEARTBEAT.size == 19
