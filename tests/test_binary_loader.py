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


def test_length_overflow_stops_walk():
    # type=MSG_CSI_FRAME, length=0xFFFF — way past MAX_PAYLOAD_BYTES.
    # Trailing garbage bytes ensure the walker isn't merely stopped by EOF;
    # the oversized-length cap in iter_messages must be what halts it.
    garbage = b"\x00" * 0xFFFF
    stream = (
        _session_info()
        + _csi_frame(10_000, 0)
        + p.pack_header(p.MSG_CSI_FRAME, 0xFFFF)
        + garbage
    )
    ds = load_binary_bytes(stream)
    # First frame is kept; oversized header halts walk before garbage is read.
    assert ds.num_frames == 1


def _env(seq, *, ldr_raw=2048, ldr_mv=1500, temp_c_x10=235, rh_x10=487,
         status=0, esp_time_us=None):
    return p.encode_env(
        esp_time_us=esp_time_us if esp_time_us is not None else seq * 1_000_000,
        seq=seq, ldr_raw=ldr_raw, ldr_mv=ldr_mv,
        temp_c_x10=temp_c_x10, rh_x10=rh_x10, am2302_status=status,
    )


def test_env_stream_separate_from_frames():
    # Two CSI frames + two env samples interleaved. Env must land on ds.env,
    # NOT on ds.frames, and must not create gaps (V13).
    stream = (
        _session_info()
        + _csi_frame(10_000, 0)
        + _env(0)
        + _csi_frame(20_000, 1)
        + _env(1)
    )
    ds = load_binary_bytes(stream)
    assert ds.num_frames == 2          # env never counted as a CSI frame
    assert ds.gap_indices == []
    assert len(ds.env) == 2
    assert ds.skipped_rows == 0        # env is known, not "unknown" (V10)


def test_env_decode_units_and_validity():
    stream = (
        _session_info()
        + _env(0, temp_c_x10=236, rh_x10=455, status=0)
        + _env(1, temp_c_x10=-115, rh_x10=900, status=1)   # crc_fail
    )
    ds = load_binary_bytes(stream)
    assert len(ds.env) == 2
    # Host only divides by 10 (V11).
    assert ds.env[0].temp_c == 23.6
    assert ds.env[0].rh == 45.5
    assert ds.env[0].temp_valid is True
    assert ds.env[1].temp_valid is False
    # env_temps_c excludes the crc_fail sample.
    assert list(ds.env_temps_c) == [23.6]
    assert list(ds.env_rh) == [45.5]


def test_env_lux_estimate_monotonic():
    # Brighter (lower LDR resistance → higher node voltage) → more lux.
    bright = load_binary_bytes(_session_info() + _env(0, ldr_mv=2500)).env[0]
    dark = load_binary_bytes(_session_info() + _env(0, ldr_mv=300)).env[0]
    assert bright.ldr_lux > dark.ldr_lux > 0.0


def test_parse_file_dispatches_to_binary(tmp_path):
    stream = _session_info() + _csi_frame(10_000, 0)
    binpath = tmp_path / "smoke.bin"
    binpath.write_bytes(stream)
    from csi_breathing import parse_file
    ds = parse_file(str(binpath))
    assert ds.num_frames == 1
    assert ds.frames[0].local_timestamp == 10_000
