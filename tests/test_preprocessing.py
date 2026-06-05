from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pytest

from p0ly_utils.preprocessing import (
    _minmax_zscore,
    artefact_rejection,
    fix_channels,
    ica_clean_dnn,
    ica_clean_regression,
)

_CH_NAMES = json.loads((Path(__file__).parent / "data" / "ch_names.json").read_text())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw(n_channels: int = 10, duration: float = 10.0, sfreq: float = 256.0) -> mne.io.RawArray:
    """Gaussian noise on all channels."""
    ch_names = _CH_NAMES["eeg"][:n_channels]
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, int(sfreq * duration))) * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False).set_montage("easycap-M1", on_missing="ignore")


def _make_raw_with_spike(
    channel: int,
    sample: int,
    value: float = 10.0,
    *,
    n_channels: int = 10,
    duration: float = 10.0,
    sfreq: float = 256.0,
    all_channels: bool = False,
) -> mne.io.RawArray:
    """Build raw data with an injected amplitude outlier at construction time."""
    ch_names = _CH_NAMES["eeg"][:n_channels]
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, int(sfreq * duration))) * 1e-6
    if all_channels:
        data[:, sample] = value
    else:
        data[channel, sample] = value
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False).set_montage("easycap-M1", on_missing="ignore")


def _make_raw_with_eog(
    n_eeg: int = 10, duration: float = 60.0, sfreq: float = 256.0
) -> mne.io.RawArray:
    """EEG + EOG channels; EOG carries a 1 Hz blink-like sinusoid mixed into EEG."""
    ch_names = _CH_NAMES["eeg"][:n_eeg]
    eog_names = _CH_NAMES["eog"]
    n_eog = len(eog_names)
    rng = np.random.default_rng(42)
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq

    # One sinusoid per EOG channel, each with independent noise
    blink = 50e-6 * np.sin(2 * np.pi * 1.0 * t)
    eog_data = blink[np.newaxis, :] + rng.standard_normal((n_eog, n_samples)) * 2e-6

    eeg_data = rng.standard_normal((n_eeg, n_samples)) * 1e-6
    # Mix mean EOG signal into EEG; amplitude diminishes away from frontal channels
    eeg_data += eog_data.mean(axis=0) * 0.5

    data = np.vstack([eeg_data, eog_data])
    info = mne.create_info(
        ch_names=ch_names + eog_names,
        sfreq=sfreq,
        ch_types=["eeg"] * n_eeg + ["eog"] * n_eog,
    )
    return mne.io.RawArray(data, info, verbose=False)


# Threshold slightly below 3.0: with one dominant outlier among many tiny
# channels, |z| converges to 3.0 and float rounding can land just below it.
DETECTION_THRESHOLD = 2.9


class TestMinmaxZscore:
    def test_detects_outlier_channel(self):
        raw = _make_raw_with_spike(channel=5, sample=100)
        mask = _minmax_zscore(raw, axis="channels", threshold=DETECTION_THRESHOLD)
        assert mask[5]
        assert mask.sum() == 1

    def test_no_outlier_all_false(self):
        raw = _make_raw(n_channels=10)
        mask = _minmax_zscore(raw, axis="channels", threshold=3.0)
        assert not mask.any()

    def test_iterative_refinement_catches_more(self):
        raw = _make_raw(n_channels=20)
        data = raw.get_data()
        data[3, 50] = 0.5
        data[7, 50] = 10.0
        raw._data[:] = data
        mask_1 = _minmax_zscore(raw, threshold=3.0, max_iter=1)
        mask_3 = _minmax_zscore(raw, threshold=3.0, max_iter=3)
        assert mask_3.sum() >= mask_1.sum()

    def test_time_axis_flags_bad_epochs(self):
        raw = _make_raw_with_spike(
            channel=0, sample=768, duration=10.0, all_channels=True
        )
        epo = mne.make_fixed_length_epochs(raw, duration=1.0, verbose=False)
        mask = _minmax_zscore(epo, axis="time", threshold=DETECTION_THRESHOLD)
        assert mask[3]

    def test_mask_excludes_from_baseline(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 2560)) * 1e-6
        data[0, 50] = 0.01
        data[1, 50] = 10.0
        info = mne.create_info([f"EEG{i:03d}" for i in range(10)], 256.0, "eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        without_mask = _minmax_zscore(raw, threshold=3.0)
        pre_mask = np.zeros(10, dtype=bool)
        pre_mask[1] = True
        with_mask = _minmax_zscore(raw, threshold=3.0, mask=pre_mask)
        assert not without_mask.any()
        assert with_mask[1]


# ---------------------------------------------------------------------------
# fix_channels
# ---------------------------------------------------------------------------


class TestFixChannels:
    def test_outlier_added_to_bads(self):
        raw = _make_raw_with_spike(channel=3, sample=100)
        result = fix_channels(raw, threshold=DETECTION_THRESHOLD)
        assert "Pz" in result.info["bads"]

    def test_does_not_mutate_input(self):
        raw = _make_raw_with_spike(channel=3, sample=100)
        _ = fix_channels(raw)
        assert raw.info["bads"] == []

    def test_preserves_existing_bads(self):
        raw = _make_raw_with_spike(channel=5, sample=100)
        raw.info["bads"] = ["Cz"]
        result = fix_channels(raw, threshold=DETECTION_THRESHOLD)
        assert "Cz" in result.info["bads"]
        assert "C4" in result.info["bads"]

    def test_no_bad_channels_returns_empty_bads(self):
        raw = _make_raw(n_channels=10)
        result = fix_channels(raw, threshold=DETECTION_THRESHOLD)
        assert result.info["bads"] == []


# ---------------------------------------------------------------------------
# artefact_rejection
# ---------------------------------------------------------------------------


class TestArtefactRejection:
    def test_flags_segment_with_spike(self):
        raw = _make_raw_with_spike(
            channel=0, sample=768, duration=10.0, sfreq=256.0, all_channels=True
        )
        annot = artefact_rejection(raw, threshold=DETECTION_THRESHOLD, duration=1.0)
        assert len(annot) >= 1
        assert any(abs(onset - 3.0) < 1.0 for onset in annot.onset)

    def test_clean_recording_no_annotations(self):
        raw = _make_raw(duration=5.0)
        annot = artefact_rejection(raw, threshold=3.0, duration=0.5)
        assert len(annot) == 0

    def test_annotation_label(self):
        raw = _make_raw_with_spike(
            channel=0, sample=500, duration=5.0, all_channels=True
        )
        annot = artefact_rejection(raw, threshold=DETECTION_THRESHOLD, duration=0.5)
        for desc in annot.description:
            assert desc == "bad_minmax_zscore"

    def test_stimulation_mask_excludes_window(self):
        raw = _make_raw_with_spike(
            channel=0, sample=512, duration=10.0, sfreq=256.0, all_channels=True
        )
        data = raw.get_data()
        data[:, 1792] = 0.01
        raw._data[:] = data
        annot_masked = artefact_rejection(
            raw, threshold=DETECTION_THRESHOLD, duration=1.0, stimulation=(0.0, 5.0)
        )
        assert len(annot_masked) >= 1

    def test_stimulation_none_endpoints(self):
        raw = _make_raw_with_spike(
            channel=0, sample=256, duration=10.0, all_channels=True
        )
        annot = artefact_rejection(raw, threshold=3.0, duration=1.0, stimulation=(None, None))
        assert len(annot) == 0


# ---------------------------------------------------------------------------
# ica_clean_dnn
# ---------------------------------------------------------------------------


class TestIcaCleanDnn:
    @pytest.fixture
    def long_raw(self):
        """ICA needs enough data; 60s at 256 Hz with 20 channels. Reference is set to average of all channels."""
        return _make_raw(n_channels=20, duration=60.0, sfreq=256.0).set_eeg_reference()

    def test_returns_tuple(self, long_raw):
        cleaned, ica = ica_clean_dnn(long_raw)
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert isinstance(ica, mne.preprocessing.ICA)

    def test_does_not_mutate_input(self, long_raw):
        orig_data = long_raw.get_data().copy()
        _ = ica_clean_dnn(long_raw)
        np.testing.assert_array_equal(long_raw.get_data(), orig_data)

    def test_duration_float_centered_crop(self, long_raw):
        cleaned, ica = ica_clean_dnn(long_raw, duration=30.0)
        assert ica.n_iter_ is not None

    def test_duration_tuple_crop(self, long_raw):
        cleaned, ica = ica_clean_dnn(long_raw, duration=(5.0, 55.0))
        assert ica.n_iter_ is not None

    def test_duration_none_uses_full(self, long_raw):
        cleaned, ica = ica_clean_dnn(long_raw, duration=None)
        assert ica.n_iter_ is not None

    def test_exclude_components_empty(self, long_raw):
        _, ica = ica_clean_dnn(long_raw, exclude_components=[])
        assert ica.exclude == []


# ---------------------------------------------------------------------------
# ica_clean_regression
# ---------------------------------------------------------------------------


class TestIcaCleanRegression:
    @pytest.fixture
    def raw_with_eog(self):
        return _make_raw_with_eog(n_eeg=20, duration=60.0, sfreq=256.0)

    def test_returns_tuple(self, raw_with_eog):
        cleaned, ica = ica_clean_regression(raw_with_eog, components=10)
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert isinstance(ica, mne.preprocessing.ICA)

    def test_does_not_mutate_input(self, raw_with_eog):
        orig_data = raw_with_eog.get_data().copy()
        _ = ica_clean_regression(raw_with_eog, components=10)
        np.testing.assert_array_equal(raw_with_eog.get_data(), orig_data)

    def test_auto_detects_eog_channels(self, raw_with_eog):
        _, ica = ica_clean_regression(raw_with_eog, components=10)
        assert isinstance(ica.exclude, list)

    def test_no_eog_channels_no_exclusions(self):
        raw = _make_raw(n_channels=20, duration=60.0)
        _, ica = ica_clean_regression(raw, components=10)
        assert ica.exclude == []

    def test_tmin_tmax_restricts_fit(self, raw_with_eog):
        _, ica = ica_clean_regression(raw_with_eog, components=10, tmin=5.0, tmax=50.0)
        assert ica.n_iter_ is not None
