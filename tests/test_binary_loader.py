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
