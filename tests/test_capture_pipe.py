"""End-to-end smoke: feed a synthetic binary stream through CaptureSession
and verify the resulting .bin parses back to the expected dataset."""
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

    import json
    stats = json.loads((tmp_path / "out.bin.stats.json").read_text())
    assert stats["frames_written"] == 3


def test_capture_peeks_env(tmp_path):
    binpath = tmp_path / "env.bin"
    logpath = tmp_path / "env.log"

    info = p.encode_session_info(
        chip_id=2, csi_format=0, csi_bytes=128,
        mac=b"\xaa\xbb\xcc\xdd\xee\xff",
        channel=6, sample_rate_hz=100,
        boot_id=1, esp_time_us=0, sensor_flags=0b11,
    )
    env = p.encode_env(
        esp_time_us=1_000_000, seq=0, ldr_raw=1234, ldr_mv=900,
        temp_c_x10=236, rh_x10=455, am2302_status=0,
    )
    stream = info + env + env  # two env samples, no CSI frames

    with open(binpath, "wb") as bf, open(logpath, "w") as lf:
        session = CaptureSession(bf, lf, str(binpath))
        for msg_type, payload in p.iter_messages(stream):
            session.write_raw(p.pack_header(msg_type, len(payload)) + payload)
            session.record_message(msg_type, payload)
        session.write_stats()

    # Bytes still verbatim on disk regardless of message type.
    assert binpath.read_bytes() == stream

    import json
    stats = json.loads((tmp_path / "env.bin.stats.json").read_text())
    assert stats["frames_written"] == 0          # env never counts as CSI
    assert stats["env_frames"] == 2
    assert stats["last_env"]["temp_c"] == 23.6    # host only /10 (V11)
    assert stats["last_env"]["rh"] == 45.5
    assert stats["last_env"]["ldr_raw"] == 1234
