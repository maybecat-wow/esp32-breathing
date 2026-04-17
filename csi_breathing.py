#!/usr/bin/env python3
"""
CSI Breathing Rate Analysis Tool
=================================
Processes ESP32 Wi-Fi CSI data for breathing rate estimation.

Implements three methods:
  - Ratio (default): cross-subcarrier conjugate product against a high-SNR
    reference cancels common-mode hardware phase noise; BoI selects the pair.
  - Amplitude: Band-of-Interest subcarrier selection on amplitude variance.
  - Phase: phase-based approach with linear detrending.

Data format: ESP32 CSI serial output from esp-csi/csi_recv_router.
Each CSI frame = 128 bytes = 64 subcarriers, stored as [imag, real] pairs.
First 4 bytes (2 subcarriers) are invalid due to ESP32 hardware limitation.

Usage:
    python csi_breathing.py <csi_data_file> [--output-dir <dir>]
    python csi_breathing.py CSI_DATA.txt --output-dir results

Dependencies: numpy, scipy, matplotlib, pywt (PyWavelets)
    pip install numpy scipy matplotlib PyWavelets
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import signal as sig

try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print(
        "[WARN] PyWavelets not installed. DWT filtering disabled. Install: pip install PyWavelets"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ESP32 CSI: first 4 bytes (2 complex values) are invalid hardware artifact
INVALID_SUBCARRIERS = 2

# LLTF 20 MHz: subcarrier indices 0..31, -32..-1 → 64 total
# Data subcarriers: ±1..±26 (52 usable), pilots at ±7, ±21
NUM_SUBCARRIERS = 64
PILOT_INDICES_LLTF = {-21, -7, 7, 21}  # relative to center

# Breathing frequency band (Hz) — 6 to 30 BPM
BREATH_FREQ_LO = 0.1  # 6 BPM lower bound
BREATH_FREQ_HI = 0.5  # 30 BPM upper bound

# BoI score computation band
BOI_FREQ_LO = 0.15
BOI_FREQ_HI = 0.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CSIFrame:
    """Single parsed CSI frame from ESP32."""

    line_number: int
    timestamp_str: str  # serial monitor timestamp e.g. "20:04:16.919"
    seq: int           # frame sequence number (wraps at 65535)
    mac: str           # source MAC address of the transmitting device
    rssi: int          # received signal strength (dBm); used for SNR context
    rate: int          # PHY data rate index
    sig_mode: int      # 0=non-HT (legacy), 1=HT, 3=VHT
    mcs: int           # modulation and coding scheme index
    bandwidth: int     # channel bandwidth in MHz (0=20, 1=40)
    smoothing: int     # whether channel smoothing was applied by hardware
    not_sounding: int  # 0=sounding frame (beamforming probe), 1=normal
    aggregation: int   # MPDU aggregation flag
    stbc: int          # space-time block coding flag
    fec_coding: int    # 0=BCC, 1=LDPC
    sgi: int           # short guard interval (400ns vs 800ns)
    noise_floor: int   # estimated noise floor (dBm)
    ampdu_cnt: int     # A-MPDU sub-frame count
    channel: int       # primary Wi-Fi channel (1–14)
    secondary_channel: int  # secondary channel offset for 40 MHz (0/1/2)
    local_timestamp: int  # ESP32 microsecond timer (esp_timer_get_time)
    ant: int           # receive antenna index
    sig_len: int        # MPDU length in bytes
    rx_state: int       # hardware receive state (0=normal)
    csi_len: int        # raw CSI byte count (= 2 × num_subcarriers)
    first_word: int     # first-word validity flag; non-zero means first 2 subcarriers are corrupt
    raw_csi: np.ndarray  # complex64 array of subcarrier values, length = csi_len / 2


@dataclass
class CSIDataset:
    """Collection of parsed CSI frames."""

    frames: list = field(default_factory=list)
    skipped_rows: int = 0  # lines that failed to parse (malformed/incomplete)

    # Frame indices at which a new contiguous segment begins — populated by
    # _parse_csv when it encounters `CSI_GAP` marker rows produced by
    # capture.py during overnight dropouts. Each entry is `len(frames)` at
    # the moment the gap row was seen, i.e. the index of the first CSI_DATA
    # frame after the gap.
    gap_indices: list = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def segment_spans(self) -> list:
        """Return [(start, end), …] frame-index ranges for each gap-free span.

        End indices are exclusive, half-open Python slice style. Empty
        gap_indices yields a single span covering the whole dataset.
        Spans with zero length (repeated gaps with no frames between them)
        are filtered out.
        """
        n = self.num_frames
        if n == 0:
            return []
        bounds = [0] + [g for g in self.gap_indices if 0 < g < n] + [n]
        return [(bounds[i], bounds[i + 1])
                for i in range(len(bounds) - 1)
                if bounds[i + 1] > bounds[i]]

    def longest_segment(self) -> tuple:
        """Return the (start, end) of the longest gap-free span."""
        spans = self.segment_spans()
        if not spans:
            return (0, 0)
        return max(spans, key=lambda s: s[1] - s[0])

    def slice(self, start: int, end: int) -> "CSIDataset":
        """Return a new CSIDataset containing only frames[start:end].
        gap_indices is empty in the slice since segments are gap-free."""
        out = CSIDataset(frames=self.frames[start:end])
        out.skipped_rows = self.skipped_rows
        return out

    @property
    def timestamps_us(self) -> np.ndarray:
        """ESP32 local timestamps in microseconds."""
        return np.array([f.local_timestamp for f in self.frames], dtype=np.float64)

    @property
    def timestamps_s(self) -> np.ndarray:
        """Timestamps in seconds (relative to first frame)."""
        ts = self.timestamps_us
        # Subtract first timestamp so t=0 at recording start
        return (ts - ts[0]) / 1e6

    @property
    def estimated_fs(self) -> float:
        """Estimated sampling rate in Hz."""
        ts = self.timestamps_s
        if len(ts) < 2:
            return 100.0  # default
        dt = np.diff(ts)
        dt = dt[dt > 0]
        if len(dt) == 0:
            return 100.0
        # Median is robust to burst gaps and duplicated timestamps
        return 1.0 / np.median(dt)

    @property
    def rssi(self) -> np.ndarray:
        # Shape (T,) — useful for flagging low-quality frames
        return np.array([f.rssi for f in self.frames])

    def csi_matrix(self) -> np.ndarray:
        """Return (num_frames, num_subcarriers) complex matrix."""
        return np.array([f.raw_csi for f in self.frames])

    @property
    def duration_s(self) -> float:
        # Total recording span in seconds; used to assess signal length for PSD
        ts = self.timestamps_s
        return ts[-1] - ts[0] if len(ts) > 1 else 0.0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Regex for serial monitor format: "<timestamp> -> CSI_DATA,<fields>"
SERIAL_RE = re.compile(
    r"^\s*(?:\d+\t)?"  # optional line number + tab
    r"([\d:.]+)\s*->\s*"  # timestamp
    r"CSI_DATA,"  # marker
    r"(.+)$"  # rest of fields + CSI data
)

# CSV header columns in the standard esp-csi format
CSV_COLUMNS = [
    "type",
    "id",
    "mac",
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "local_timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
    "data",
]


def parse_csi_values(csi_str: str) -> np.ndarray:
    """Parse CSI data string '[imag0,real0,imag1,real1,...]' into complex array.

    ESP32 format: each subcarrier stored as (imaginary, real) byte pair.
    Returns complex64 array of length num_subcarriers.
    """
    csi_str = csi_str.strip().strip('"').strip("[]")
    values = [int(x.strip()) for x in csi_str.split(",")]

    num_complex = len(values) // 2
    csi = np.zeros(num_complex, dtype=np.complex64)
    for i in range(num_complex):
        # ESP32 stores each complex sample as two consecutive int8 bytes in
        # (imaginary, real) order — the opposite of the usual (real, imag) convention.
        # Byte layout: [..., imag_k, real_k, imag_{k+1}, real_{k+1}, ...]
        # So index 2*i is imaginary and 2*i+1 is real for subcarrier i.
        imag = values[2 * i]      # even offset → imaginary component
        real = values[2 * i + 1]  # odd offset  → real component
        csi[i] = complex(real, imag)

    return csi


def _detect_format(filepath: str) -> str:
    """Detect whether file is CSV-with-header or serial-monitor format."""
    # Two distinct formats exist in the wild:
    #   Serial: lines like "HH:MM:SS.mmm -> CSI_DATA,field1,...,[imag,real,...]"
    #           produced by the Arduino/IDF serial monitor.
    #   CSV: lines like "CSI_DATA,field1,...,\"[imag,real,...]\""
    #        produced by esp-csi's logger or exported tools.
    # We sniff the first line to decide which parser to invoke.
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()
    if first_line.startswith("type,") or first_line.startswith("type\t"):
        return "csv"
    if "-> CSI_DATA" in first_line or "CSI_DATA," in first_line:
        return "serial"
    # Try second line (first might be a comment)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        second_line = f.readline().strip()
    if second_line.startswith("CSI_DATA"):
        return "csv_no_header"
    return "serial"  # fallback


def _frame_from_fields(
    seq: int,
    mac: str,
    rssi: int,
    rate: int,
    sig_mode: int,
    mcs: int,
    bandwidth: int,
    smoothing: int,
    not_sounding: int,
    aggregation: int,
    stbc: int,
    fec_coding: int,
    sgi: int,
    noise_floor: int,
    ampdu_cnt: int,
    channel: int,
    secondary_channel: int,
    local_timestamp: int,
    ant: int,
    sig_len: int,
    rx_state: int,
    csi_len: int,
    first_word: int,
    raw_csi: np.ndarray,
    timestamp_str: str = "",
) -> CSIFrame:
    return CSIFrame(
        line_number=0,
        timestamp_str=timestamp_str,
        seq=seq,
        mac=mac,
        rssi=rssi,
        rate=rate,
        sig_mode=sig_mode,
        mcs=mcs,
        bandwidth=bandwidth,
        smoothing=smoothing,
        not_sounding=not_sounding,
        aggregation=aggregation,
        stbc=stbc,
        fec_coding=fec_coding,
        sgi=sgi,
        noise_floor=noise_floor,
        ampdu_cnt=ampdu_cnt,
        channel=channel,
        secondary_channel=secondary_channel,
        local_timestamp=local_timestamp,
        ant=ant,
        sig_len=sig_len,
        rx_state=rx_state,
        csi_len=csi_len,
        first_word=first_word,
        raw_csi=raw_csi,
    )


def _parse_csv(filepath: str) -> CSIDataset:
    """Parse CSV format (with or without header): type,id,mac,...,data."""
    import csv

    dataset = CSIDataset()
    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        # Sniff for header
        first_line = f.readline()
        f.seek(0)

        has_header = first_line.strip().startswith("type,")
        reader = csv.reader(f)

        if has_header:
            header = next(reader)
        else:
            header = CSV_COLUMNS

        col_idx = {name.strip(): i for i, name in enumerate(header)}

        for row in reader:
            if not row:
                continue
            row_type = row[0].strip()
            if row_type == "CSI_GAP":
                # Mark a segment boundary at the next CSI_DATA index.
                # Dedupe consecutive gaps (they can only be produced by a
                # future change to capture.py; the current one already
                # dedupes, but belt-and-braces).
                idx = len(dataset.frames)
                if not dataset.gap_indices or dataset.gap_indices[-1] != idx:
                    dataset.gap_indices.append(idx)
                continue
            if row_type != "CSI_DATA":
                continue

            try:

                def g(name, default="0"):
                    idx = col_idx.get(name)
                    return (
                        row[idx].strip()
                        if idx is not None and idx < len(row)
                        else default
                    )

                csi_str = g("data", "")
                if not csi_str:
                    continue
                raw_csi = parse_csi_values(csi_str)

                frame = _frame_from_fields(
                    seq=int(g("id", g("seq", "0"))),
                    mac=g("mac", ""),
                    rssi=int(g("rssi", "0")),
                    rate=int(g("rate", "0")),
                    sig_mode=int(g("sig_mode", "0")),
                    mcs=int(g("mcs", "0")),
                    bandwidth=int(g("bandwidth", "0")),
                    smoothing=int(g("smoothing", "0")),
                    not_sounding=int(g("not_sounding", "0")),
                    aggregation=int(g("aggregation", "0")),
                    stbc=int(g("stbc", "0")),
                    fec_coding=int(g("fec_coding", "0")),
                    sgi=int(g("sgi", "0")),
                    noise_floor=int(g("noise_floor", "0")),
                    ampdu_cnt=int(g("ampdu_cnt", "0")),
                    channel=int(g("channel", "0")),
                    secondary_channel=int(g("secondary_channel", "0")),
                    local_timestamp=int(g("local_timestamp", "0")),
                    ant=int(g("ant", "0")),
                    sig_len=int(g("sig_len", "0")),
                    rx_state=int(g("rx_state", "0")),
                    csi_len=int(g("len", str(len(raw_csi) * 2))),
                    first_word=int(g("first_word", "0")),
                    raw_csi=raw_csi,
                )
                dataset.frames.append(frame)
            except (ValueError, IndexError):
                dataset.skipped_rows += 1
                continue

    return dataset


def _parse_serial(filepath: str) -> CSIDataset:
    """Parse serial monitor format: <timestamp> -> CSI_DATA,<fields>."""
    dataset = CSIDataset()

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            m = SERIAL_RE.match(line)
            if not m:
                # Also try bare CSV lines starting with CSI_DATA
                if line.strip().startswith("CSI_DATA,"):
                    pass  # will be handled by CSV parser
                continue

            timestamp_str = m.group(1)
            rest = m.group(2)

            csi_match = re.search(r'"(\[.+?\])"', rest)
            if not csi_match:
                continue

            csi_str = csi_match.group(1)
            fields_str = rest[: csi_match.start()].rstrip(",").strip()
            fields = [x.strip() for x in fields_str.split(",")]

            if len(fields) < 21:
                continue

            try:
                raw_csi = parse_csi_values(csi_str)
            except (ValueError, IndexError):
                continue

            frame = _frame_from_fields(
                seq=int(fields[0]),
                mac=fields[1],
                rssi=int(fields[2]),
                rate=int(fields[3]),
                sig_mode=int(fields[4]),
                mcs=int(fields[5]),
                bandwidth=int(fields[6]),
                smoothing=int(fields[7]),
                not_sounding=int(fields[8]),
                aggregation=int(fields[9]),
                stbc=int(fields[10]),
                fec_coding=int(fields[11]),
                sgi=int(fields[12]),
                noise_floor=int(fields[13]),
                ampdu_cnt=int(fields[14]),
                channel=int(fields[15]),
                secondary_channel=int(fields[16]),
                local_timestamp=int(fields[17]),
                ant=int(fields[18]),
                sig_len=int(fields[19]),
                rx_state=int(fields[20]),
                csi_len=int(fields[21]) if len(fields) > 21 else len(raw_csi) * 2,
                first_word=int(fields[22]) if len(fields) > 22 else 0,
                raw_csi=raw_csi,
                timestamp_str=timestamp_str,
            )
            dataset.frames.append(frame)

    return dataset


def parse_file(filepath: str) -> CSIDataset:
    """Auto-detect format and parse ESP32 CSI data file into CSIDataset."""
    fmt = _detect_format(filepath)
    if fmt in ("csv", "csv_no_header"):
        return _parse_csv(filepath)
    else:
        ds = _parse_serial(filepath)
        # Fallback: if serial parser found nothing, try CSV parser
        if ds.num_frames == 0:
            return _parse_csv(filepath)
        return ds


# ---------------------------------------------------------------------------
# Subcarrier reordering
# ---------------------------------------------------------------------------


def reorder_subcarriers(csi_matrix: np.ndarray) -> np.ndarray:
    """Reorder LLTF subcarriers from storage order to frequency order.

    ESP32 LLTF storage: indices 0..31 (positive freq), 32..63 (negative freq)
    Reorder to: -32..-1, 0..31 (standard FFT shift)

    Also invalidates first 2 subcarriers (hardware artifact).
    """
    N = csi_matrix.shape[1]
    if N != 64:
        return csi_matrix  # only reorder for standard 64-subcarrier LLTF

    reordered = np.zeros_like(csi_matrix)
    # FFT-shift rationale: the ESP32 firmware reports positive-frequency subcarriers
    # first (indices 0..31) followed by negative-frequency subcarriers (indices 32..63).
    # Standard signal-processing convention places negative frequencies before positive
    # (cf. np.fft.fftshift), so we swap the two halves to obtain the intuitive layout:
    # index 0 → subcarrier –32 (most negative), index 32 → DC, index 63 → subcarrier +31.
    reordered[:, :32] = csi_matrix[:, 32:]  # negative freq subcarriers → left half
    reordered[:, 32:] = csi_matrix[:, :32]  # positive freq subcarriers → right half

    # After the shift, storage indices 0 and 1 (positive side) end up at reordered
    # indices 32 and 33.  The ESP32 hardware marks these two subcarriers as invalid
    # via the first_word field; zero them unconditionally to prevent them from
    # contaminating subcarrier selection or BoI scoring.
    reordered[:, 32] = 0  # was storage index 0 — hardware-invalid first word
    reordered[:, 33] = 0  # was storage index 1 — hardware-invalid first word

    return reordered


def get_valid_subcarrier_mask(n_sub: int = 64) -> np.ndarray:
    """Return boolean mask for valid data subcarriers (excluding null/pilot/invalid)."""
    mask = np.ones(n_sub, dtype=bool)

    # After reordering: index 32 = DC (subcarrier 0)
    # Guard bands: indices 0..5 and 59..63 roughly (depends on mode)
    # Null subcarriers near DC: 32 ± 1
    # Pilot subcarriers at specific positions

    # For LLTF 20MHz: data on subcarriers -26..-1, 1..26
    # After reorder, that's indices 6..31 (neg side) and 33..58 (pos side)
    # But first 2 storage positions are invalid → indices 32, 33 already zeroed

    # Guard bands (LLTF 20 MHz): subcarriers ±27..±32 are guard/null tones that
    # carry no data and have near-zero energy; exclude to avoid polluting BoI.
    mask[:6] = False   # indices 0..5  → subcarriers −32..−27 (guard band)
    mask[59:] = False  # indices 59..63 → subcarriers +27..+31 (guard band)

    # DC subcarrier (index 32) is always zero in OFDM; the two adjacent indices
    # (32, 33) are also zeroed because they correspond to the hardware-invalid
    # first_word subcarriers (see reorder_subcarriers).
    mask[32] = False   # DC subcarrier — no channel information
    mask[33] = False   # hardware-invalid (first_word artifact)

    # Pilot subcarriers (after reorder, center at 32):
    # LLTF pilots at relative offsets ±7 and ±21 carry known BPSK symbols used
    # for phase tracking by the receiver; their amplitude variation is dominated
    # by AGC/phase corrections rather than the channel, so exclude them.
    # subcarrier -21 → index 32-21=11, -7 → 25, +7 → 39, +21 → 53
    for p in [11, 25, 39, 53]:
        if 0 <= p < n_sub:
            mask[p] = False

    return mask


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def compute_boi_scores(
    csi_matrix: np.ndarray,
    fs: float,
    freq_lo: float = BOI_FREQ_LO,
    freq_hi: float = BOI_FREQ_HI,
) -> np.ndarray:
    """Compute Band-of-Interest score for each subcarrier.

    BoI = (power in breathing band) / (power above breathing band)
    Higher score → subcarrier is more sensitive to breathing.
    """
    T, M = csi_matrix.shape
    scores = np.zeros(M)

    for m in range(M):
        x = csi_matrix[:, m]
        if np.all(x == 0):
            scores[m] = 0
            continue

        # Work on the mean-subtracted amplitude envelope;
        # this separates breathing-driven fluctuations from the carrier level.
        amp_signal = np.abs(x) - np.mean(np.abs(x))

        if T < 8:
            # Too few samples for Welch — fall back to a zero-padded FFT.
            # Zero-padding interpolates the DFT grid but does not add spectral
            # resolution; it is only used here to get a usable frequency axis.
            X = np.fft.fft(amp_signal, n=max(256, T * 8))
            freqs = np.fft.fftfreq(len(X), d=1.0 / fs)
            psd = np.abs(X) ** 2
        else:
            # Welch’s method splits the signal into overlapping segments and
            # averages their periodograms, reducing spectral variance by ~nperseg/2
            # compared to a single FFT — critical for short, noisy CSI records.
            nperseg = min(T, max(8, T // 2))
            freqs, psd = sig.welch(
                amp_signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
            )

        # BoI score = in-band power / out-of-band power.
        # A high ratio means the subcarrier’s amplitude variation is concentrated
        # in the breathing band rather than in higher-frequency noise.
        breath_mask = (np.abs(freqs) >= freq_lo) & (np.abs(freqs) <= freq_hi)
        above_mask = np.abs(freqs) > freq_hi

        breath_power = np.sum(psd[breath_mask])
        noise_power = np.sum(psd[above_mask])

        if noise_power > 0:
            scores[m] = breath_power / noise_power
        elif breath_power > 0:
            scores[m] = breath_power
        else:
            scores[m] = 0

    return scores


def resample_uniform(
    csi_matrix: np.ndarray, timestamps_s: np.ndarray, target_fs: float
) -> tuple:
    """Resample non-uniformly sampled CSI to a uniform grid via linear interpolation.

    Returns: (resampled_matrix, uniform_timestamps, actual_fs)
    """
    # ESP32 timestamps are non-uniform: the Wi-Fi driver schedules pings at a
    # nominal rate but FreeRTOS tick quantization (1 ms default) and driver
    # jitter cause variable inter-frame spacing.  All spectral estimators (FFT,
    # Welch, autocorrelation) require a uniform time axis, so we resample first.
    duration = timestamps_s[-1] - timestamps_s[0]
    n_samples = int(duration * target_fs) + 1
    t_uniform = np.linspace(timestamps_s[0], timestamps_s[-1], n_samples)
    actual_fs = (n_samples - 1) / duration if duration > 0 else target_fs

    M = csi_matrix.shape[1]
    resampled = np.zeros((n_samples, M), dtype=np.complex64)

    for m in range(M):
        # Interpolate real and imaginary parts independently: numpy’s np.interp
        # operates on real-valued arrays, and there is no standard definition of
        # linear interpolation for complex numbers (the "straight line" between
        # two phasors depends on whether you interpolate magnitude+phase or I+Q).
        # Interpolating I and Q separately preserves linearity in Cartesian space.
        real_interp = np.interp(t_uniform, timestamps_s, np.real(csi_matrix[:, m]))
        imag_interp = np.interp(t_uniform, timestamps_s, np.imag(csi_matrix[:, m]))
        resampled[:, m] = real_interp + 1j * imag_interp

    return resampled, t_uniform, actual_fs


def cross_subcarrier_ratio(
    csi_matrix: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple:
    """Compute conjugate product of every subcarrier against a reference.

    For each valid subcarrier m:
        R[t, m] = H[t, m] * conj(H[t, ref])

    The product cancels common-mode hardware phase noise components that are
    constant across subcarriers (CFO, random per-packet phase offset), leaving
    primarily the differential channel variation caused by motion/breathing.

    Limitation: SFO and phase-based distortion (PBD) introduce a slope across
    subcarrier indices that does NOT cancel intra-antenna. Full cancellation of
    the slope term requires two physical antennas (inter-antenna ratio). With a
    single-antenna ESP32, a residual term (ξ_d + ξ_s)(m - m_ref) remains.

    The reference subcarrier is the valid subcarrier with the highest mean
    amplitude (best SNR).

    Returns:
        ratio:   (T, M) complex array of conjugate products
        ref_idx: index of the reference subcarrier
    """
    mean_amp = np.mean(np.abs(csi_matrix), axis=0)
    mean_amp_valid = mean_amp.copy()
    mean_amp_valid[~valid_mask] = 0
    # Choose the valid subcarrier with the highest mean amplitude as the reference;
    # highest amplitude → best SNR → least noise injected into every ratio.
    ref_idx = int(np.argmax(mean_amp_valid))

    ref = csi_matrix[:, ref_idx]          # (T,)
    # H[t,m] * conj(H[t,ref]) eliminates the per-packet common phase rotation
    # θ(t) that the ESP32 hardware applies identically across all subcarriers
    # (due to CFO correction residual and random initial phase).  After the
    # product only differential channel variation — caused by motion or breathing
    # — and the SFO slope term remain.
    # NOTE: with a single antenna the SFO/PBD slope (m - m_ref)*δ is NOT
    # cancelled by this product; it introduces a frequency-dependent phase ramp
    # that can bias the breathing estimate for subcarrier pairs far from the
    # reference.  Inter-antenna ratios (two physical RX chains) would cancel it.
    ratio = csi_matrix * np.conj(ref[:, np.newaxis])  # (T, M)
    return ratio, ref_idx


def extract_breathing_signal(
    csi_matrix: np.ndarray,
    fs: float,
    method: str = "ratio",
    valid_mask: Optional[np.ndarray] = None,
) -> tuple:
    """Extract breathing signal from CSI matrix.

    Args:
        csi_matrix: (T, M) complex CSI matrix
        fs: sampling rate in Hz
        method: 'ratio' | 'amplitude' | 'phase'
        valid_mask: boolean mask for valid subcarriers

    Returns:
        (breathing_signal, selected_subcarrier_idx, boi_scores)
    """
    T, M = csi_matrix.shape

    if valid_mask is None:
        valid_mask = get_valid_subcarrier_mask(M)

    if method == "ratio":
        # RATIO branch: form the cross-subcarrier conjugate product to remove the
        # common per-packet phase rotation, then extract the differential phase of
        # the best-scoring subcarrier pair.  Phase of the ratio is a robust proxy
        # for the breathing-induced path-length change (modulo the SFO residual).
        ratio, ref_idx = cross_subcarrier_ratio(csi_matrix, valid_mask)

        boi = compute_boi_scores(ratio, fs, freq_lo=BOI_FREQ_LO, freq_hi=BOI_FREQ_HI)
        boi[~valid_mask] = 0
        boi[ref_idx] = 0  # ratio of reference with itself is trivially zero

        best_idx = int(np.argmax(boi))

        # Phase of the ratio is the differential phase — robust to hardware offsets
        phase_ratio = np.unwrap(np.angle(ratio[:, best_idx]))
        breath_raw = phase_ratio - np.mean(phase_ratio)

    elif method == "amplitude":
        # AMPLITUDE branch: breathing moves the scatterer (chest), changing the
        # multipath interference at each subcarrier and thus its amplitude.  We
        # select the subcarrier whose amplitude PSD has the highest in-band
        # (BoI) score, then use its mean-subtracted amplitude as the breathing signal.
        boi = compute_boi_scores(csi_matrix, fs, freq_lo=BOI_FREQ_LO, freq_hi=BOI_FREQ_HI)
        boi[~valid_mask] = 0

        amplitudes = np.abs(csi_matrix)
        best_idx = np.argmax(boi)
        breath_raw = amplitudes[:, best_idx] - np.mean(amplitudes[:, best_idx])

    elif method == "phase":
        # PHASE branch: breathing causes a time-varying phase shift on each
        # subcarrier proportional to the path-length change.  We unwrap the raw
        # phase to remove 2π discontinuities, then remove the linear drift
        # (SFO + static phase offset) with a first-order polynomial fit, and
        # select the subcarrier with the highest residual phase variance.
        phases = np.angle(csi_matrix)
        # Use phase directly (single antenna — no inter-antenna difference available)
        # Remove linear trend per subcarrier
        phase_detrended = np.zeros_like(phases)
        for m in range(M):
            p = np.unwrap(phases[:, m])
            # Fit and subtract a line (degree-1 polynomial) to remove the SFO
            # frequency ramp and any constant hardware phase offset.
            p = p - np.polyval(np.polyfit(np.arange(T), p, 1), np.arange(T))
            phase_detrended[:, m] = p

        variances = np.var(phase_detrended, axis=0)
        variances[~valid_mask] = 0

        best_idx = np.argmax(variances)
        breath_raw = phase_detrended[:, best_idx]
        boi = variances

    else:
        raise ValueError(f"Unknown method: {method}")

    return breath_raw, best_idx, boi


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def bandpass_filter(
    x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4
) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    nyq = fs / 2.0
    if hi >= nyq:
        hi = nyq * 0.95
    if lo <= 0:
        lo = 0.01
    if lo >= hi:
        return x

    try:
        b, a = sig.butter(order, [lo / nyq, hi / nyq], btype="band")
        # filtfilt applies the filter forward and backward, achieving zero-phase
        # filtering with no group-delay distortion — essential for preserving the
        # timing of breathing peaks used by the autocorrelation estimator.
        # padlen = 3 × filter order is the scipy default and ensures the edge
        # transients die out before the actual data; smaller values cause ringing.
        return sig.filtfilt(b, a, x, padlen=min(3 * max(len(b), len(a)), len(x) - 1))
    except ValueError as e:
        print(f"[WARN] Butterworth filter failed ({e}); falling back to FFT bandpass")
        return fft_bandpass(x, fs, lo, hi)


def fft_bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """FFT-based bandpass filter (works for short signals)."""
    N = len(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    mask = (np.abs(freqs) >= lo) & (np.abs(freqs) <= hi)
    X[~mask] = 0
    return np.real(np.fft.ifft(X))


def dwt_filter(
    x: np.ndarray, fs: float, max_freq: float = 0.5, wavelet: str = "db4"
) -> np.ndarray:
    """Discrete wavelet transform filter — keep components below max_freq.

    Used in PhaseBeat for DC removal and high-frequency noise suppression.
    """
    if not HAS_PYWT:
        return x

    # Determine max decomposition level
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet))
    if max_level < 1:
        return x

    # Each DWT level halves the bandwidth of the approximation subband.
    # At level L the approximation covers [0, fs / 2^(L+1)].  We solve for L
    # such that fs / 2^(L+1) ≈ max_freq, i.e. L ≈ log2(fs / (2*max_freq)).
    # This ensures we keep exactly the sub-band that contains the breathing signal.
    target_level = max(1, int(np.log2(fs / (2 * max_freq))))
    level = min(target_level, max_level)

    # Decompose
    coeffs = pywt.wavedec(x, wavelet, level=level)

    # Zero out all detail coefficients (high-frequency subbands) and keep only
    # the level-L approximation (coeffs[0]).  The approximation represents the
    # low-pass filtered signal; detail bands contain progressively higher
    # frequencies that are above the breathing band and are discarded as noise.
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])

    return pywt.waverec(coeffs, wavelet)[: len(x)]


# ---------------------------------------------------------------------------
# Breathing rate estimation
# ---------------------------------------------------------------------------


def estimate_breathing_rate_psd(
    x: np.ndarray,
    fs: float,
    freq_lo: float = BREATH_FREQ_LO,
    freq_hi: float = BREATH_FREQ_HI,
    zero_pad_factor: int = 16,
) -> tuple:
    """Estimate breathing rate from PSD peak detection.

    Returns: (rate_bpm, frequency, psd_freqs, psd_values, reason)
    reason is an empty string on success, or a short failure description.
    """
    N = len(x)
    # Zero-padding to nfft > N interpolates the DFT grid, giving a finer
    # frequency resolution in the output — useful for resolving the breathing
    # peak when N is small.  It does NOT increase true spectral resolution
    # (which is bounded by 1/duration), but it reduces picket-fence bias.
    nfft = max(1024, N * zero_pad_factor)

    X = np.fft.fft(x, n=nfft)
    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)

    # Restrict to positive frequencies: the input x is real-valued so the
    # spectrum is Hermitian-symmetric and the negative half is redundant.
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    psd = np.abs(X[pos_mask]) ** 2

    # Restrict to breathing band
    breath_mask = (freqs_pos >= freq_lo) & (freqs_pos <= freq_hi)

    if not np.any(breath_mask):
        return 0.0, 0.0, freqs_pos, psd, "no energy in breathing band"

    psd_breath = psd.copy()
    psd_breath[~breath_mask] = 0

    peak_idx = np.argmax(psd_breath)
    peak_freq = freqs_pos[peak_idx]
    rate_bpm = peak_freq * 60.0

    return rate_bpm, peak_freq, freqs_pos, psd, ""


def estimate_breathing_rate_autocorr(
    x: np.ndarray,
    fs: float,
    freq_lo: float = BREATH_FREQ_LO,
    freq_hi: float = BREATH_FREQ_HI,
) -> tuple:
    """Estimate breathing rate from autocorrelation peak.

    Returns: (rate_bpm, peak_lag_seconds, lags, autocorr, reason)
    reason is an empty string on success, or a short failure description.
    """
    N = len(x)
    x_norm = x - np.mean(x)

    # Full autocorrelation
    acf = np.correlate(x_norm, x_norm, mode="full")
    acf = acf[N - 1 :]  # keep positive lags (lag 0 and above)
    # Normalize by the zero-lag value (= signal power) so the ACF is in [-1, 1].
    # The 1e-12 guard prevents division-by-zero for all-zero signals.
    acf = acf / (acf[0] + 1e-12)

    lags_samples = np.arange(len(acf))
    lags_seconds = lags_samples / fs

    # Bound the search window to the physically plausible breath-period range.
    # min_lag = fs / freq_hi corresponds to the shortest possible breath period
    # (fastest breathing, freq_hi = 0.5 Hz → 2 s period).  max_lag = fs / freq_lo
    # corresponds to the slowest (freq_lo = 0.1 Hz → 10 s period).  Searching
    # outside this window picks up harmonics or DC-drift artifacts instead.
    min_lag = int(fs / freq_hi) if freq_hi > 0 else 1
    max_lag = int(fs / freq_lo) if freq_lo > 0 else len(acf)
    max_lag = min(max_lag, len(acf) - 1)

    if min_lag >= max_lag or min_lag >= len(acf):
        return 0.0, 0.0, lags_seconds, acf, "search range empty"

    acf_search = acf[min_lag : max_lag + 1]
    if len(acf_search) == 0:
        return 0.0, 0.0, lags_seconds, acf, "search range empty"

    peak_local = np.argmax(acf_search)
    peak_lag = min_lag + peak_local
    peak_period = peak_lag / fs

    if peak_period > 0:
        rate_bpm = 60.0 / peak_period
    else:
        rate_bpm = 0.0
        return rate_bpm, peak_period, lags_seconds, acf, "zero-lag peak"

    return rate_bpm, peak_period, lags_seconds, acf, ""


def estimate_breathing_rate_peaks(
    x: np.ndarray,
    fs: float,
    freq_lo: float = BREATH_FREQ_LO,
    freq_hi: float = BREATH_FREQ_HI,
) -> tuple:
    """Estimate breathing rate from time-domain peak counting.

    Returns: (rate_bpm, peak_times, peak_indices, reason)
    reason is an empty string on success, or a short failure description.
    """
    # distance=int(fs/freq_hi) enforces a minimum separation between detected
    # peaks equal to the shortest expected breath period (at freq_hi = 0.5 Hz,
    # that is 2 s → fs*2 samples).  Without this constraint, noise spikes within
    # a single breath cycle would be counted as separate breaths.
    min_distance = int(fs / freq_hi) if freq_hi > 0 else 1
    min_distance = max(1, min_distance)

    peaks, properties = sig.find_peaks(x, distance=min_distance, prominence=0)

    if len(peaks) < 2:
        return 0.0, np.array([]) / fs if len(peaks) > 0 else np.array([]), peaks, f"<2 peaks ({len(peaks)} found)"

    peak_times = peaks / fs
    intervals = np.diff(peak_times)

    if len(intervals) == 0:
        return 0.0, peak_times, peaks, "<2 peaks"

    mean_interval = np.mean(intervals)
    if mean_interval > 0:
        rate_bpm = 60.0 / mean_interval
    else:
        rate_bpm = 0.0
        return rate_bpm, peak_times, peaks, "zero mean interval"

    return rate_bpm, peak_times, peaks, ""


# ---------------------------------------------------------------------------
# CIR analysis
# ---------------------------------------------------------------------------


def compute_cir_matrix(csi_matrix: np.ndarray) -> np.ndarray:
    """Compute Channel Impulse Response via IFFT for each frame."""
    # The CSI matrix H[t, m] is the channel transfer function sampled at
    # discrete subcarrier frequencies.  The IFFT along the frequency axis
    # transforms it to the delay (time-of-flight) domain, yielding the channel
    # impulse response h[t, τ].  Each tap τ corresponds to a multipath delay
    # of τ / bandwidth seconds (tap spacing = 1 / (N × subcarrier spacing)).
    # Breathing-induced chest movement shifts the dominant multipath tap
    # amplitude over time, making the CIR a useful complementary view.
    return np.fft.ifft(csi_matrix, axis=1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_comprehensive_analysis(
    dataset: CSIDataset,
    output_dir: str,
    methods: list = None,
):
    """Generate comprehensive visualization of CSI breathing analysis.

    If the dataset contains gap markers (i.e. the overnight capture.py
    detected one or more TCP/Wi-Fi dropouts), the analysis below is run on
    the longest gap-free segment only — interpolating across an outage would
    corrupt the FFT/autocorrelation of breathing variability. Use the
    `--sliding` CLI mode to get a per-segment breathing-rate-vs-time view
    across the whole night.
    """

    if methods is None:
        methods = ["ratio", "amplitude", "phase"]

    if dataset.gap_indices:
        start, end = dataset.longest_segment()
        total = dataset.num_frames
        print(
            f"  [gap] {len(dataset.gap_indices)} dropout(s) detected — "
            f"analysing longest segment ({start}:{end}, "
            f"{end - start}/{total} frames)"
        )
        dataset = dataset.slice(start, end)
        if dataset.num_frames == 0:
            print("  [gap] Longest segment is empty; nothing to analyse.")
            return

    csi_raw = dataset.csi_matrix()
    T_raw, M = csi_raw.shape
    ts_raw = dataset.timestamps_s
    fs_raw = dataset.estimated_fs

    # Both steps below are required before any spectral analysis:
    # 1. reorder_subcarriers maps storage order to frequency order so that
    #    subcarrier indices are monotonically increasing in frequency and the
    #    valid-mask indices align correctly.
    # 2. resample_uniform converts the non-uniform ESP32 timestamps to a
    #    regular grid; without this, FFT and autocorrelation results are
    #    meaningless because they assume equal time spacing.
    csi_ordered_raw = reorder_subcarriers(csi_raw)
    valid_mask = get_valid_subcarrier_mask(M)

    # Nyquist reasoning: the breathing band tops at 0.5 Hz, so any rate above
    # ~1 Hz satisfies Nyquist; 20 Hz gives a comfortable 40× oversampling
    # margin.  We cap at 50 Hz to avoid allocating a huge resampled array for
    # devices that report CSI at 100+ Hz.
    target_fs = min(max(20.0, fs_raw), 50.0)
    csi_ordered, ts, fs = resample_uniform(csi_ordered_raw, ts_raw, target_fs)
    T = len(ts)

    print(f"\n{'='*60}")
    print(f"  CSI Breathing Rate Analysis")
    print(f"{'='*60}")
    print(f"  Raw frames:        {T_raw}")
    print(f"  After resampling:  {T} samples @ {fs:.1f} Hz")
    print(f"  Subcarriers:       {M} ({np.sum(valid_mask)} valid)")
    print(f"  Duration:          {dataset.duration_s:.2f} s")
    print(f"  Raw sample rate:   {fs_raw:.1f} Hz (non-uniform)")
    print(f"  RSSI range:        {dataset.rssi.min()} to {dataset.rssi.max()} dBm")
    print(f"  Channel:           {dataset.frames[0].channel}")
    print(f"{'='*60}\n")

    short_data = T < 50
    if short_data:
        print(f"  [NOTE] Only {T} samples — too few for reliable breathing estimation.")
        print(f"  [NOTE] Showing signal processing pipeline and subcarrier analysis.\n")

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================================
    # Figure 1: Raw CSI Overview
    # Subplot layout:
    #   [0,0] CSI amplitude heatmap — shows which subcarriers are active and
    #         whether amplitude varies with time (breathing modulates this).
    #   [0,1] CSI phase heatmap — reveals phase wrapping patterns and the
    #         SFO-induced frequency slope across subcarriers.
    #   [1,0] RSSI time series — overall link quality; sharp drops may flag
    #         corrupted frames that should be excluded.
    #   [1,1] Mean subcarrier amplitude bar chart — green bars are valid data
    #         subcarriers; red are guard/pilot/DC tones excluded from analysis.
    # =====================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle("CSI Data Overview", fontsize=14, fontweight="bold")

    amp_matrix = np.abs(csi_ordered)
    im = axes1[0, 0].imshow(
        amp_matrix.T,
        aspect="auto",
        origin="lower",
        extent=[ts[0], ts[-1], 0, M],
        cmap="viridis",
    )
    axes1[0, 0].set_xlabel("Time (s)")
    axes1[0, 0].set_ylabel("Subcarrier index")
    axes1[0, 0].set_title("CSI Amplitude")
    fig1.colorbar(im, ax=axes1[0, 0], label="Amplitude (arb.)")

    phase_matrix = np.angle(csi_ordered)
    im2 = axes1[0, 1].imshow(
        phase_matrix.T,
        aspect="auto",
        origin="lower",
        extent=[ts[0], ts[-1], 0, M],
        cmap="twilight",
    )
    axes1[0, 1].set_xlabel("Time (s)")
    axes1[0, 1].set_ylabel("Subcarrier index")
    axes1[0, 1].set_title("CSI Phase")
    fig1.colorbar(im2, ax=axes1[0, 1], label="Phase (rad)")

    axes1[1, 0].plot(ts_raw, dataset.rssi, linewidth=0.8, color="#2196F3", alpha=0.7)
    axes1[1, 0].set_xlabel("Time (s)")
    axes1[1, 0].set_ylabel("RSSI (dBm)")
    axes1[1, 0].set_title("RSSI")
    axes1[1, 0].grid(True, alpha=0.3)

    mean_amp = np.mean(amp_matrix, axis=0)
    colors = ["#4CAF50" if valid_mask[i] else "#F44336" for i in range(M)]
    axes1[1, 1].bar(range(M), mean_amp, color=colors, width=1.0, alpha=0.7)
    axes1[1, 1].set_xlabel("Subcarrier index (freq-ordered)")
    axes1[1, 1].set_ylabel("Mean amplitude (arb.)")
    axes1[1, 1].set_title("Subcarrier Amplitudes (green=valid, red=null/pilot)")
    axes1[1, 1].grid(True, alpha=0.3, axis="y")

    fig1.tight_layout()
    fig1.savefig(
        os.path.join(output_dir, "01_csi_overview.png"), dpi=150, bbox_inches="tight"
    )
    print(f"  Saved: 01_csi_overview.png")

    # =====================================================================
    # Figure 2: CIR Analysis
    # The delay-domain (CIR) view separates multipath components by their
    # time-of-flight.  A breathing subject slightly shifts the path length
    # of one or more dominant taps, causing their amplitude to oscillate at
    # the breathing rate.  This can be easier to detect than the frequency-
    # domain CSI when the target is spatially isolated in the delay profile.
    # =====================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Channel Impulse Response Analysis", fontsize=14, fontweight="bold")

    cir_matrix = compute_cir_matrix(csi_ordered)
    cir_amp = np.abs(cir_matrix)

    im3 = axes2[0].imshow(
        cir_amp.T,
        aspect="auto",
        origin="lower",
        extent=[ts[0], ts[-1], 0, M],
        cmap="magma",
    )
    axes2[0].set_xlabel("Time (s)")
    axes2[0].set_ylabel("Delay tap")
    axes2[0].set_title("CIR Amplitude")
    fig2.colorbar(im3, ax=axes2[0], label="Amplitude (arb.)")

    mean_cir = np.mean(cir_amp, axis=0)
    axes2[1].stem(
        range(min(32, M)),
        mean_cir[: min(32, M)],
        linefmt="-",
        markerfmt="o",
        basefmt="k-",
    )
    axes2[1].set_xlabel("Delay tap")
    axes2[1].set_ylabel("Mean amplitude (arb.)")
    axes2[1].set_title("Power Delay Profile (mean)")
    axes2[1].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(
        os.path.join(output_dir, "02_cir_analysis.png"), dpi=150, bbox_inches="tight"
    )
    print(f"  Saved: 02_cir_analysis.png")

    # =====================================================================
    # Figure 3: Subcarrier Selection (BoI Scores)
    # =====================================================================
    fig3, axes3 = plt.subplots(1, len(methods), figsize=(8 * len(methods), 5))
    if len(methods) == 1:
        axes3 = [axes3]
    fig3.suptitle(
        "Subcarrier Selection (BoI score for ratio/amplitude; variance for phase)",
        fontsize=14,
        fontweight="bold",
    )

    method_colors = {
        "ratio": "#2196F3",
        "amplitude": "#FF9800",
        "phase": "#9C27B0",
    }

    for i, method in enumerate(methods):
        _, best_idx, boi_scores = extract_breathing_signal(
            csi_ordered, fs, method=method, valid_mask=valid_mask
        )
        bars = axes3[i].bar(range(M), boi_scores, width=1.0, color="#BDBDBD", alpha=0.6)
        if best_idx < len(boi_scores):
            bars[best_idx].set_color(method_colors.get(method, "#2196F3"))
            bars[best_idx].set_alpha(1.0)
        axes3[i].set_xlabel("Subcarrier index")
        axes3[i].set_ylabel("BoI score" if method == "amplitude" else "Variance")
        axes3[i].set_title(f"{method.capitalize()} — best: SC {best_idx}")
        axes3[i].grid(True, alpha=0.3, axis="y")

    fig3.tight_layout()
    fig3.savefig(
        os.path.join(output_dir, "03_subcarrier_selection.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  Saved: 03_subcarrier_selection.png")

    # =====================================================================
    # Figure 4: Extracted Breathing Signals
    # =====================================================================
    n_methods = len(methods)
    fig4, axes4 = plt.subplots(n_methods, 1, figsize=(14, 4 * n_methods))
    if n_methods == 1:
        axes4 = [axes4]
    fig4.suptitle("Extracted Breathing Signals", fontsize=14, fontweight="bold")

    results = {}
    for i, method in enumerate(methods):
        breath_raw, best_idx, _ = extract_breathing_signal(
            csi_ordered, fs, method=method, valid_mask=valid_mask
        )

        if len(breath_raw) > 12 and not short_data:
            try:
                breath_filtered = bandpass_filter(
                    breath_raw, fs, BREATH_FREQ_LO, BREATH_FREQ_HI
                )
            except Exception as e:
                print(f"[WARN] Bandpass filter failed for {method} ({e}); falling back to FFT bandpass")
                breath_filtered = fft_bandpass(
                    breath_raw, fs, BREATH_FREQ_LO, BREATH_FREQ_HI
                )
        else:
            breath_filtered = breath_raw - np.mean(breath_raw)

        results[method] = {
            "raw": breath_raw,
            "filtered": breath_filtered,
            "best_sc": best_idx,
        }

        color = method_colors.get(method, "#2196F3")
        axes4[i].plot(
            ts, breath_raw, alpha=0.3, color=color, linewidth=0.8, label="Raw"
        )
        axes4[i].plot(ts, breath_filtered, color=color, linewidth=1.8, label="Filtered")
        axes4[i].set_xlabel("Time (s)")
        axes4[i].set_ylabel("Amplitude (arb.)")
        axes4[i].set_title(f"{method.capitalize()} — Subcarrier {best_idx}")
        axes4[i].legend(loc="upper right")
        axes4[i].grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig(
        os.path.join(output_dir, "04_breathing_signals.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  Saved: 04_breathing_signals.png")

    # =====================================================================
    # Figure 5: Breathing Rate Estimation (PSD + Autocorrelation)
    # =====================================================================
    fig5, axes5 = plt.subplots(n_methods, 2, figsize=(14, 4 * n_methods))
    if n_methods == 1:
        axes5 = axes5.reshape(1, -1)
    fig5.suptitle("Breathing Rate Estimation", fontsize=14, fontweight="bold")

    print(f"\n  Breathing Rate Estimates:")
    print(
        f"  {'Method':<15} {'PSD (BPM)':<25} {'Autocorr (BPM)':<25} {'Peaks (BPM)':<25}"
    )
    print(f"  {'-'*88}")

    for i, method in enumerate(methods):
        x = results[method]["filtered"]
        color = method_colors.get(method, "#2196F3")

        rate_psd, peak_freq, freqs_psd, psd_vals, reason_psd = estimate_breathing_rate_psd(x, fs)

        breath_band = (freqs_psd >= BREATH_FREQ_LO) & (freqs_psd <= BREATH_FREQ_HI)
        axes5[i, 0].semilogy(
            freqs_psd, psd_vals, color="#757575", alpha=0.5, linewidth=0.8
        )
        axes5[i, 0].fill_between(
            freqs_psd,
            psd_vals,
            where=breath_band,
            alpha=0.3,
            color=color,
            label=f"Breathing band ({BREATH_FREQ_LO}-{BREATH_FREQ_HI} Hz)",
        )
        if peak_freq > 0:
            axes5[i, 0].axvline(
                peak_freq,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Peak: {rate_psd:.1f} BPM ({peak_freq:.3f} Hz)",
            )
        axes5[i, 0].set_xlabel("Frequency (Hz)")
        axes5[i, 0].set_ylabel("Normalized PSD")
        axes5[i, 0].set_title(f"{method.capitalize()} — Power Spectral Density")
        axes5[i, 0].set_xlim(0, min(2.0, fs / 2))
        axes5[i, 0].legend(fontsize=8)
        axes5[i, 0].grid(True, alpha=0.3)

        rate_ac, peak_period, lags, acf, reason_ac = estimate_breathing_rate_autocorr(x, fs)
        max_lag_plot = min(len(lags), int(fs * 15))
        axes5[i, 1].plot(
            lags[:max_lag_plot], acf[:max_lag_plot], color=color, linewidth=1.2
        )
        if peak_period > 0:
            axes5[i, 1].axvline(
                peak_period,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Peak: {rate_ac:.1f} BPM ({peak_period:.2f} s)",
            )
        if BREATH_FREQ_HI > 0 and BREATH_FREQ_LO > 0:
            axes5[i, 1].axvspan(
                1.0 / BREATH_FREQ_HI,
                1.0 / BREATH_FREQ_LO,
                alpha=0.1,
                color=color,
                label="Breathing range",
            )
        axes5[i, 1].set_xlabel("Lag (s)")
        axes5[i, 1].set_ylabel("Autocorrelation")
        axes5[i, 1].set_title(f"{method.capitalize()} — Autocorrelation")
        axes5[i, 1].legend(fontsize=8)
        axes5[i, 1].grid(True, alpha=0.3)

        rate_pk, peak_times_m, peak_idx_m, reason_pk = estimate_breathing_rate_peaks(x, fs)

        psd_str = f"{rate_psd:.1f}" if rate_psd > 0 else f"N/A ({reason_psd})"
        ac_str = f"{rate_ac:.1f}" if rate_ac > 0 else f"N/A ({reason_ac})"
        pk_str = f"{rate_pk:.1f}" if rate_pk > 0 else f"N/A ({reason_pk})"
        print(f"  {method:<15} {psd_str:<25} {ac_str:<25} {pk_str:<25}")

    fig5.tight_layout()
    fig5.savefig(
        os.path.join(output_dir, "05_rate_estimation.png"), dpi=150, bbox_inches="tight"
    )
    print(f"\n  Saved: 05_rate_estimation.png")

    # =====================================================================
    # Figure 6: Complex Plane / Constellation
    # If breathing is the dominant channel variation, the CSI phasor for the
    # selected subcarrier traces a circular arc in the I-Q plane: the amplitude
    # (radius) is roughly constant while the phase (angle) oscillates.  A
    # well-defined arc is a visual indicator of pure phase modulation from
    # breathing.  Scattered or irregular patterns suggest noise or body motion.
    # =====================================================================
    fig6, axes6 = plt.subplots(1, len(methods), figsize=(8 * len(methods), 5))
    if len(methods) == 1:
        axes6 = [axes6]
    fig6.suptitle(
        "CSI Complex Plane (Selected Subcarriers)", fontsize=14, fontweight="bold"
    )

    for i, method in enumerate(methods):
        best_sc = results[method]["best_sc"]
        csi_sc = csi_ordered[:, best_sc]
        color = method_colors.get(method, "#2196F3")

        sc = axes6[i].scatter(
            np.real(csi_sc),
            np.imag(csi_sc),
            c=np.arange(T),
            cmap="coolwarm",
            s=12,
            edgecolors="k",
            linewidths=0.2,
            zorder=2,
        )
        axes6[i].plot(
            np.real(csi_sc),
            np.imag(csi_sc),
            color=color,
            alpha=0.12,
            linewidth=0.4,
            zorder=1,
        )
        axes6[i].set_xlabel("Real (I)")
        axes6[i].set_ylabel("Imaginary (Q)")
        axes6[i].set_title(f"{method.capitalize()} — SC {best_sc}")
        axes6[i].grid(True, alpha=0.3)
        axes6[i].set_aspect("equal")
        fig6.colorbar(sc, ax=axes6[i], label="Sample index")

    fig6.tight_layout()
    fig6.savefig(
        os.path.join(output_dir, "06_complex_plane.png"), dpi=150, bbox_inches="tight"
    )
    print(f"  Saved: 06_complex_plane.png")

    # =====================================================================
    # Figure 7: Multi-subcarrier time series
    # =====================================================================
    fig7, axes7 = plt.subplots(2, 1, figsize=(14, 8))
    fig7.suptitle(
        "Multi-Subcarrier Amplitude & Phase Variation", fontsize=14, fontweight="bold"
    )

    _, _, boi_all = extract_breathing_signal(
        csi_ordered, fs, method="amplitude", valid_mask=valid_mask
    )
    boi_masked = boi_all.copy()
    boi_masked[~valid_mask] = 0
    top_indices = np.argsort(boi_masked)[-5:][::-1]

    cmap_sc = plt.colormaps.get_cmap("tab10")
    for j, sc_idx in enumerate(top_indices):
        amp_ts = np.abs(csi_ordered[:, sc_idx])
        amp_ts = amp_ts - np.mean(amp_ts)
        axes7[0].plot(
            ts, amp_ts, color=cmap_sc(j), label=f"SC {sc_idx}", linewidth=1.0, alpha=0.8
        )

        phase_ts = np.unwrap(np.angle(csi_ordered[:, sc_idx]))
        phase_ts = phase_ts - np.mean(phase_ts)
        axes7[1].plot(
            ts,
            phase_ts,
            color=cmap_sc(j),
            label=f"SC {sc_idx}",
            linewidth=1.0,
            alpha=0.8,
        )

    axes7[0].set_xlabel("Time (s)")
    axes7[0].set_ylabel("Amplitude, DC removed (arb.)")
    axes7[0].set_title("Top-5 Subcarrier Amplitude Variation")
    axes7[0].legend(fontsize=8, ncol=5, loc="upper right")
    axes7[0].grid(True, alpha=0.3)

    axes7[1].set_xlabel("Time (s)")
    axes7[1].set_ylabel("Phase (rad, unwrapped)")
    axes7[1].set_title("Top-5 Subcarrier Phase Variation")
    axes7[1].legend(fontsize=8, ncol=5, loc="upper right")
    axes7[1].grid(True, alpha=0.3)

    fig7.tight_layout()
    fig7.savefig(
        os.path.join(output_dir, "07_multi_subcarrier.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  Saved: 07_multi_subcarrier.png")

    # =====================================================================
    # Figure 8: Breathing waveform with detected peaks (best method)
    # =====================================================================
    if not short_data:
        # Select the method whose filtered signal has the largest peak-to-peak
        # amplitude as a proxy for SNR: a higher amplitude relative to other
        # methods means breathing modulates that feature more strongly, giving
        # more reliable peak detection and rate estimation.
        best_method = max(
            results.keys(), key=lambda m: np.max(np.abs(results[m]["filtered"]))
        )
        x_best = results[best_method]["filtered"]
        rate_best, peak_times_best, peak_idx_best, _ = estimate_breathing_rate_peaks(
            x_best, fs
        )

        fig8, ax8 = plt.subplots(1, 1, figsize=(14, 5))
        ax8.plot(
            ts,
            x_best,
            color=method_colors.get(best_method, "#2196F3"),
            linewidth=1.5,
            label=f"{best_method.capitalize()} filtered",
        )
        if len(peak_idx_best) > 0:
            ax8.plot(
                ts[peak_idx_best],
                x_best[peak_idx_best],
                "rv",
                markersize=8,
                label=f"Peaks ({len(peak_idx_best)} detected)",
            )
        ax8.set_xlabel("Time (s)")
        ax8.set_ylabel("Amplitude (arb.)")
        rate_str = f"{rate_best:.1f} BPM" if rate_best > 0 else "N/A"
        ax8.set_title(
            f"Breathing Waveform — {best_method.capitalize()} (strongest signal) — Est. Rate: {rate_str}"
        )
        ax8.legend(loc="upper right")
        ax8.grid(True, alpha=0.3)

        fig8.tight_layout()
        fig8.savefig(
            os.path.join(output_dir, "08_breathing_waveform.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print(f"  Saved: 08_breathing_waveform.png")

    plt.close("all")

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'='*60}")
    if short_data:
        print(f"  Data contains only {T} samples (~{dataset.duration_s:.2f}s).")
        print(f"  For reliable breathing estimation, collect at least 30 seconds")
        print(f"  of continuous data (~100 Hz -> ~3000 frames).")
    else:
        print(f"  Analysis complete. All figures saved to: {output_dir}/")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _estimate_rate_for_window(dataset: CSIDataset, method: str) -> float:
    """Run the core breathing-rate estimator on a short window and return the
    estimated rate in BPM, or NaN if the window is unusable."""
    n = dataset.num_frames
    if n < 50:
        return float("nan")
    csi_raw = dataset.csi_matrix()
    _, M = csi_raw.shape
    csi_ordered_raw = reorder_subcarriers(csi_raw)
    valid_mask = get_valid_subcarrier_mask(M)
    target_fs = min(max(20.0, dataset.estimated_fs), 50.0)
    try:
        csi_ordered, _, fs = resample_uniform(
            csi_ordered_raw, dataset.timestamps_s, target_fs
        )
        breath_raw, _, _ = extract_breathing_signal(
            csi_ordered, fs, method=method, valid_mask=valid_mask
        )
        filtered = bandpass_filter(breath_raw, fs, BREATH_FREQ_LO, BREATH_FREQ_HI)
        rate, _, _, _, reason = estimate_breathing_rate_psd(filtered, fs)
        if reason:
            return float("nan")
        return float(rate)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return float("nan")


def _run_sliding(dataset: CSIDataset, output_dir: str, methods: list,
                 window_s: float, stride_s: float) -> None:
    """Overnight breathing-rate vs time plot.

    Segments the recording on `CSI_GAP` markers from capture.py, then runs
    the per-method estimator on sliding windows (window_s wide, stride_s
    apart) inside each segment. Produces `<output_dir>/sliding_breathing.png`
    — the actual sleep-quality signal: breathing rate trajectory through the
    night, with blank bands at dropouts instead of smoothed bridges.
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    spans = dataset.segment_spans()
    if not spans:
        print("  [sliding] No frames to analyse.")
        return

    t0_us = float(dataset.frames[0].local_timestamp)

    print(f"  [sliding] {len(spans)} segment(s); "
          f"window={window_s}s stride={stride_s}s; methods={methods}")

    per_method_times = {m: [] for m in methods}
    per_method_rates = {m: [] for m in methods}

    for span_i, (start, end) in enumerate(spans):
        seg = dataset.slice(start, end)
        fs_est = seg.estimated_fs
        if seg.num_frames < 50 or fs_est <= 0:
            continue
        seg_duration = seg.duration_s
        if seg_duration < window_s:
            # Still run the whole segment as one window so short segments
            # aren't lost entirely.
            t_center = (
                seg.frames[0].local_timestamp
                + (seg.frames[-1].local_timestamp - seg.frames[0].local_timestamp) / 2.0
            )
            t_center_s = (t_center - t0_us) / 1e6
            for m in methods:
                rate = _estimate_rate_for_window(seg, m)
                per_method_times[m].append(t_center_s)
                per_method_rates[m].append(rate)
            continue

        n_window = int(round(window_s * fs_est))
        n_stride = max(1, int(round(stride_s * fs_est)))
        for i in range(0, seg.num_frames - n_window + 1, n_stride):
            win = seg.slice(i, i + n_window)
            t_center_us = (
                win.frames[0].local_timestamp
                + (win.frames[-1].local_timestamp - win.frames[0].local_timestamp) / 2.0
            )
            t_center_s = (t_center_us - t0_us) / 1e6
            for m in methods:
                rate = _estimate_rate_for_window(win, m)
                per_method_times[m].append(t_center_s)
                per_method_rates[m].append(rate)

        print(f"  [sliding] segment {span_i}: {seg.num_frames} frames, "
              f"{seg_duration:.1f}s")

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"ratio": "tab:blue", "amplitude": "tab:orange", "phase": "tab:green"}
    any_points = False
    for m in methods:
        ts = np.array(per_method_times[m])
        rs = np.array(per_method_rates[m])
        if len(ts) == 0:
            continue
        any_points = True
        ax.plot(ts / 3600.0, rs, ".-", label=m, color=colors.get(m),
                markersize=3, linewidth=1)
    if any_points:
        for gap_idx in dataset.gap_indices:
            if 0 < gap_idx < dataset.num_frames:
                t_gap = (dataset.frames[gap_idx].local_timestamp - t0_us) / 1e6 / 3600.0
                ax.axvline(t_gap, color="red", alpha=0.2, linewidth=1)
        ax.set_xlabel("Time since start (hours)")
        ax.set_ylabel("Breathing rate (BPM)")
        ax.set_ylim(BREATH_FREQ_LO * 60 - 2, BREATH_FREQ_HI * 60 + 2)
        ax.set_title(f"Breathing rate vs time  "
                     f"(window={window_s:.0f}s, stride={stride_s:.0f}s, "
                     f"{len(dataset.gap_indices)} gaps)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out_path = os.path.join(output_dir, "sliding_breathing.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  [sliding] wrote {out_path}")
    else:
        plt.close(fig)
        print("  [sliding] no windows produced a valid rate estimate.")


def main():
    parser = argparse.ArgumentParser(
        description="ESP32 CSI Breathing Rate Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python csi_breathing.py CSI_DATA.txt
    python csi_breathing.py CSI_DATA.txt --output-dir results
    python csi_breathing.py CSI_DATA.txt --methods ratio amplitude phase
    python csi_breathing.py night.csv --sliding --window 60 --stride 10

Methods:
    ratio        - Cross-subcarrier ratio: conjugate product against a reference
                   subcarrier cancels hardware phase noise; BoI selects the best pair
    amplitude    - Band-of-Interest subcarrier selection on amplitude
    phase        - Phase-based with linear detrending

Sleep-quality mode:
    --sliding    - Plot breathing rate vs time across the whole overnight
                   recording. Segments on CSI_GAP markers emitted by capture.py
                   so dropouts appear as blank bands, not smoothed bridges.

References:
    [1] Espressif ESP-CSI: https://github.com/espressif/esp-csi
        """,
    )
    parser.add_argument("input", help="Path to CSI data file")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="csi_results",
        help="Output directory for plots (default: csi_results)",
    )
    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        default=["ratio", "amplitude", "phase"],
        choices=["ratio", "amplitude", "phase"],
        help="Analysis methods to run",
    )
    parser.add_argument(
        "--sliding",
        action="store_true",
        help="Run sliding-window breathing-rate-vs-time analysis across the "
             "whole recording (designed for overnight sleep-quality captures)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=60.0,
        help="Sliding window width in seconds (default: 60)",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=10.0,
        help="Sliding window stride in seconds (default: 10)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    print(f"  Parsing: {args.input}")
    dataset = parse_file(args.input)

    if dataset.num_frames == 0:
        print("Error: No valid CSI frames found in file.")
        sys.exit(1)

    skipped_info = f" ({dataset.skipped_rows} malformed rows skipped)" if dataset.skipped_rows > 0 else ""
    gap_info = f" ({len(dataset.gap_indices)} dropout(s) marked)" if dataset.gap_indices else ""
    print(f"  Parsed {dataset.num_frames} CSI frames{skipped_info}{gap_info}")

    if args.sliding:
        _run_sliding(dataset, args.output_dir, args.methods,
                     args.window, args.stride)
    else:
        plot_comprehensive_analysis(dataset, args.output_dir, methods=args.methods)


if __name__ == "__main__":
    main()
