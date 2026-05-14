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
