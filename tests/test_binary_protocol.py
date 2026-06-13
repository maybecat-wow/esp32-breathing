"""Wire format must not drift. Sizes pinned per the design doc:
docs/superpowers/specs/2026-05-14-binary-csi-protocol-design.md
"""
import csi_protocol as p


def test_msg_type_constants():
    assert p.MSG_SESSION_INFO == 0x01
    assert p.MSG_CSI_FRAME == 0x02
    assert p.MSG_HEARTBEAT == 0x03
    assert p.MSG_ENV == 0x04


def test_header_struct_size():
    # u8 type | u16 length
    assert p.HEADER.size == 3


def test_session_info_struct_size():
    # u8 chip_id, u8 csi_format, u16 csi_bytes, u8 mac[6], u8 channel,
    # u8 sensor_flags, u16 sample_rate_hz, u32 boot_id, u64 esp_time_us
    # Repurposing reserved→sensor_flags must NOT change the 26-byte size (V2).
    assert p.SESSION_INFO.size == 26


def test_session_info_sensor_flags_round_trip():
    raw = p.encode_session_info(
        chip_id=2, csi_format=0, csi_bytes=128,
        mac=b"\x11\x22\x33\x44\x55\x66",
        channel=11, sample_rate_hz=100,
        boot_id=1, esp_time_us=5,
        sensor_flags=0b11,   # LDR + AM2302 present
    )
    info = p.decode_session_info(list(p.iter_messages(raw))[0][1])
    assert info.sensor_flags == 0b11


def test_csi_frame_meta_struct_size():
    # u32 local_timestamp_us, u32 seq, i8 rssi, i8 noise_floor,
    # u8 rate, u8 first_word_invalid, u16 len
    assert p.CSI_FRAME_META.size == 14


def test_heartbeat_struct_size():
    # u64 esp_time_us, i8 rssi, u8 channel, u32 uptime_s,
    # u32 reconnect_count, u8 last_disc_reason
    assert p.HEARTBEAT.size == 19


def test_env_struct_size():
    # u64 esp_time_us, u32 seq, u16 ldr_raw, u16 ldr_mv, i16 temp_c_x10,
    # u16 rh_x10, u8 am2302_status, u8 reserved  →  fixed 22 bytes (V9)
    assert p.ENV.size == 22


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


def test_round_trip_env():
    raw = p.encode_env(
        esp_time_us=3_000_000, seq=9,
        ldr_raw=2048, ldr_mv=1500,
        temp_c_x10=235, rh_x10=487,
        am2302_status=0,
    )
    msgs = list(p.iter_messages(raw))
    assert len(msgs) == 1
    msg_type, payload = msgs[0]
    assert msg_type == p.MSG_ENV
    assert len(payload) == 22
    env = p.decode_env(payload)
    assert env.esp_time_us == 3_000_000
    assert env.seq == 9
    assert env.ldr_raw == 2048
    assert env.ldr_mv == 1500
    assert env.temp_c_x10 == 235
    assert env.rh_x10 == 487
    assert env.am2302_status == 0


def test_round_trip_env_negative_temp():
    # Sub-zero temperature must survive the signed i16 field.
    raw = p.encode_env(
        esp_time_us=1, seq=0, ldr_raw=0, ldr_mv=0,
        temp_c_x10=-115, rh_x10=900, am2302_status=1,
    )
    env = p.decode_env(list(p.iter_messages(raw))[0][1])
    assert env.temp_c_x10 == -115
    assert env.am2302_status == 1


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


def test_iter_messages_truncated_tail():
    csi = bytes(range(128))
    session_bytes = _sample_session_info()
    full = session_bytes + _sample_csi_frame(10_000, 0, csi)
    # Chop the last 5 bytes off (mid-CSI payload).
    truncated = full[:-5]
    msgs = list(p.iter_messages(truncated))
    # SESSION_INFO is complete, CSI_FRAME is partial → only SESSION_INFO yields.
    assert len(msgs) == 1
    assert msgs[0][0] == p.MSG_SESSION_INFO
    # Payload bytes match — locks down offset advancement, not just type byte.
    assert msgs[0][1] == session_bytes[p.HEADER.size:]


def test_iter_messages_unknown_type_zero_length():
    # \xFF type, length=0 → opaque message but well-formed. The walker should
    # skip it and continue to the next message rather than stalling.
    csi = bytes(range(128))
    stream = (
        _sample_csi_frame(10_000, 0, csi)
        + p.pack_header(0xFF, 0)
        + _sample_csi_frame(20_000, 1, csi)
    )
    msgs = list(p.iter_messages(stream))
    assert [t for (t, _) in msgs] == [p.MSG_CSI_FRAME, 0xFF, p.MSG_CSI_FRAME]
    assert msgs[1][1] == b""


def test_iter_messages_unknown_type_with_payload():
    # Non-zero length unknown type: walker still advances by length, but the
    # consumer dispatch is what decides to ignore it. Just verify it doesn't
    # corrupt the stream walk.
    csi = bytes(range(128))
    unknown_payload = b"\x01\x02\x03\x04\x05\x06\x07"
    stream = (
        _sample_csi_frame(10_000, 0, csi)
        + p.pack_header(0x42, len(unknown_payload)) + unknown_payload
        + _sample_csi_frame(20_000, 1, csi)
    )
    msgs = list(p.iter_messages(stream))
    assert [t for (t, _) in msgs] == [p.MSG_CSI_FRAME, 0x42, p.MSG_CSI_FRAME]
    assert msgs[1][1] == unknown_payload
