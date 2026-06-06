"""Unit tests for the per-epoch RR + confidence model (sliding-window mode)."""

import math

import numpy as np

import csi_breathing as cb


def _rec(full=True, **method_vals):
    """Build a minimal sliding record with the given <method>_bpm/_snr keys."""
    rec = {"is_full_window": 1 if full else 0}
    rec.update(method_vals)
    return rec


def test_no_finite_rate_is_garbage():
    nan = float("nan")
    rec = _rec(ratio_bpm=nan, ratio_snr=nan)
    rr, conf = cb._epoch_rr_confidence(rec, ["ratio"])
    assert math.isnan(rr)
    assert conf == 0.0


def test_snr_weighted_consensus():
    # Two methods agree-ish; high-SNR one dominates the weighted mean.
    rec = _rec(ratio_bpm=15.0, ratio_snr=20.0,
               amplitude_bpm=12.0, amplitude_snr=2.0)
    rr, conf = cb._epoch_rr_confidence(rec, ["ratio", "amplitude"])
    plain = (15.0 + 12.0) / 2
    weighted = (15.0 * 20.0 + 12.0 * 2.0) / 22.0
    assert rr == weighted
    assert rr > plain  # pulled toward the high-SNR estimate
    assert 0.0 <= conf <= 1.0


def test_agreement_raises_confidence_over_disagreement():
    agree = _rec(ratio_bpm=15.0, ratio_snr=20.0,
                 amplitude_bpm=15.2, amplitude_snr=20.0)
    disagree = _rec(ratio_bpm=8.0, ratio_snr=20.0,
                    amplitude_bpm=24.0, amplitude_snr=20.0)
    _, c_agree = cb._epoch_rr_confidence(agree, ["ratio", "amplitude"])
    _, c_dis = cb._epoch_rr_confidence(disagree, ["ratio", "amplitude"])
    assert c_agree > c_dis
    assert c_agree >= cb.SLIDING_CONF_MIN
    assert c_dis < cb.SLIDING_CONF_MIN  # spread >> tol -> garbage


def test_high_snr_agreeing_epoch_is_good():
    rec = _rec(ratio_bpm=16.0, ratio_snr=30.0,
               amplitude_bpm=16.1, amplitude_snr=30.0,
               phase_bpm=15.9, phase_snr=30.0)
    rr, conf = cb._epoch_rr_confidence(rec, ["ratio", "amplitude", "phase"])
    assert abs(rr - 16.0) < 0.2
    assert conf >= cb.SLIDING_CONF_MIN


def test_partial_window_penalized():
    full = _rec(True, ratio_bpm=16.0, ratio_snr=30.0,
                amplitude_bpm=16.0, amplitude_snr=30.0)
    part = _rec(False, ratio_bpm=16.0, ratio_snr=30.0,
                amplitude_bpm=16.0, amplitude_snr=30.0)
    _, c_full = cb._epoch_rr_confidence(full, ["ratio", "amplitude"])
    _, c_part = cb._epoch_rr_confidence(part, ["ratio", "amplitude"])
    assert c_part == c_full * cb.SLIDING_PARTIAL_WINDOW_CONF


def test_single_method_capped():
    rec = _rec(ratio_bpm=16.0, ratio_snr=1e9)  # huge SNR, one method
    _, conf = cb._epoch_rr_confidence(rec, ["ratio"])
    # snr_conf -> ~1, agree capped at SINGLE_METHOD_CONF, full coverage.
    assert conf <= cb.SLIDING_SINGLE_METHOD_CONF + 1e-9


def test_low_snr_drags_confidence_down():
    rec = _rec(ratio_bpm=16.0, ratio_snr=1.0,
               amplitude_bpm=16.0, amplitude_snr=1.0)
    _, conf = cb._epoch_rr_confidence(rec, ["ratio", "amplitude"])
    # snr_conf = 1/(1+SNR_HALF) ~ 0.2 even with perfect agreement.
    assert conf < cb.SLIDING_CONF_MIN


def test_finite_rate_nonfinite_snr_still_consensus():
    nan = float("nan")
    rec = _rec(ratio_bpm=14.0, ratio_snr=nan,
               amplitude_bpm=14.0, amplitude_snr=10.0)
    rr, conf = cb._epoch_rr_confidence(rec, ["ratio", "amplitude"])
    assert np.isfinite(rr)
    assert abs(rr - 14.0) < 1e-9
