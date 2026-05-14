"""Wire format must not drift. Sizes pinned per the design doc:
docs/superpowers/specs/2026-05-14-binary-csi-protocol-design.md
"""
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
