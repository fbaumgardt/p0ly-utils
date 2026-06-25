from __future__ import annotations

import mne
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from p0ly_utils.epoching import _match_onsets, align_epochs_metadata, epoch_with_metadata
from p0ly_utils.metadata import dmss, events_from_raw, igt, parse_metadata

sorted_onsets = st.lists(
    st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=50,
).map(sorted)


def _as_ms(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


ALIGNMENT_CASES = [
    pytest.param([1, 2, 3], [1, 2, 3], [0, 1, 2], [0, 1, 2], id="perfect_1_to_1"),
    pytest.param([1, 2, 3], [2, 3], [1, 2], [0, 1], id="missing_first_metadata"),
    pytest.param([1, 2, 3], [1, 2], [0, 1], [0, 1], id="missing_last_metadata"),
    pytest.param([2, 3], [1, 2, 3], [0, 1], [1, 2], id="missing_first_epochs"),
    pytest.param([1, 2], [1, 2, 3], [0, 1], [0, 1], id="missing_last_epochs"),
    pytest.param([1, 2, 5, 6], [1, 2, 3, 4, 5, 6], [0, 1, 2, 3], [0, 1, 4, 5], id="gap_in_epochs"),
    pytest.param([1, 2, 3, 4, 5, 6], [1, 2, 5, 6], [0, 1, 4, 5], [0, 1, 2, 3], id="gap_in_metadata"),
    pytest.param([1, 3, 5], [1, 2, 3, 4, 5], [0, 1, 2], [0, 2, 4], id="gaps_on_both_sides"),
    pytest.param([5], [5], [0], [0], id="single_element_match"),
    pytest.param([1, 2], [100, 101], [], [], id="no_overlap_rejected_by_tolerance"),
    pytest.param([], [1, 2], [], [], id="empty_epochs"),
    pytest.param([1, 2], [], [], [], id="empty_metadata"),
    pytest.param([], [], [], [], id="both_empty"),
    pytest.param([1.001, 2.002], [1.0, 2.0], [0, 1], [0, 1], id="jitter_within_tolerance"),
    pytest.param([1, 50], [1, 2, 3], [0], [0], id="one_pair_within_tolerance"),
]


class TestMatchOnsetsParametrized:
    @pytest.mark.parametrize(
        ("epoch_onsets", "meta_onsets", "expected_epoch_idx", "expected_meta_idx"),
        ALIGNMENT_CASES,
    )
    def test_alignment_cases(
        self,
        epoch_onsets: list[float],
        meta_onsets: list[float],
        expected_epoch_idx: list[int],
        expected_meta_idx: list[int],
    ) -> None:
        matched_epoch_idx, matched_meta_idx = _match_onsets(
            _as_ms(epoch_onsets),
            _as_ms(meta_onsets),
        )
        np.testing.assert_array_equal(matched_epoch_idx, np.asarray(expected_epoch_idx, dtype=int))
        np.testing.assert_array_equal(matched_meta_idx, np.asarray(expected_meta_idx, dtype=int))


class TestMatchOnsetsProperties:
    @given(epoch_onsets=sorted_onsets, meta_onsets=sorted_onsets)
    @settings(max_examples=500)
    def test_alignment_invariants(
        self,
        epoch_onsets: list[float],
        meta_onsets: list[float],
    ) -> None:
        epoch_arr = _as_ms(epoch_onsets)
        meta_arr = _as_ms(meta_onsets)
        matched_epoch_idx, matched_meta_idx = _match_onsets(epoch_arr, meta_arr)

        assert len(matched_epoch_idx) == len(matched_meta_idx)
        if len(matched_epoch_idx) == 0:
            return

        assert len(set(matched_epoch_idx)) == len(matched_epoch_idx)
        assert len(set(matched_meta_idx)) == len(matched_meta_idx)
        assert list(matched_epoch_idx) == sorted(matched_epoch_idx)
        assert list(matched_meta_idx) == sorted(matched_meta_idx)
        assert matched_epoch_idx.min() >= 0
        assert matched_epoch_idx.max() < len(epoch_arr)
        assert matched_meta_idx.min() >= 0
        assert matched_meta_idx.max() < len(meta_arr)

        if len(meta_arr) >= 2:
            tolerance_ms = float(np.mean(np.diff(meta_arr)))
        else:
            tolerance_ms = float("inf")

        diffs = np.abs(epoch_arr[matched_epoch_idx] - meta_arr[matched_meta_idx])
        assert np.all(diffs < tolerance_ms)


class TestMatchOnsetsIdempotency:
    def test_uniform_spacing_is_idempotent(self) -> None:
        epoch_arr = _as_ms([1, 2, 3])
        meta_arr = _as_ms([1, 2, 3])
        matched_epoch_idx, matched_meta_idx = _match_onsets(epoch_arr, meta_arr)
        rematched_epoch_idx, rematched_meta_idx = _match_onsets(
            epoch_arr[matched_epoch_idx],
            meta_arr[matched_meta_idx],
        )
        np.testing.assert_array_equal(
            rematched_epoch_idx,
            np.arange(len(matched_epoch_idx)),
        )
        np.testing.assert_array_equal(
            rematched_meta_idx,
            np.arange(len(matched_meta_idx)),
        )


# ---------------------------------------------------------------------------
# epoch_with_metadata: synthetic raw + spec + metadata scenarios (US-017 AC #5)
# ---------------------------------------------------------------------------

_DMSS_BASELINE = (-0.2, 0.0)


def _raw_with_annotations(
    rows: list[tuple[float, str]],
    sfreq: float = 100.0,
    duration: float | None = None,
    durations: list[float] | None = None,
) -> mne.io.Raw:
    """Tiny Raw carrying Psychtoolbox marker annotations (and optional bad anns)."""
    onsets = [t for t, _ in rows]
    descriptions = [d for _, d in rows]
    if duration is None:
        duration = (max(onsets) if onsets else 0.0) + 2.0
    if durations is None:
        durations = [0.0] * len(rows)
    info = mne.create_info(["Cz"], sfreq, ch_types=["eeg"])
    data = np.zeros((1, int(sfreq * duration)))
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    )
    return raw


# Two dmss trials in one block (onsets chosen so stim epochs fit a >=0 raw).
_DMSS_ROWS = [
    (0.0, "Stim/S  3"),   # block start
    (0.1, "Stim/S  5"),   # trial 1 start
    (0.2, "Stim/S 11"),   # stim (size1)
    (0.5, "Stim/S 57"),   # RT start
    (1.0, "Stim/S 64"),   # Correct
    (1.2, "Stim/S  6"),   # trial 1 end
    (1.3, "Stim/S  5"),   # trial 2 start
    (1.4, "Stim/S 25"),   # stim (size2)
    (1.5, "Stim/S 57"),
    (1.8, "Stim/S 60"),
    (1.9, "Stim/S  6"),
    (2.0, "Stim/S  4"),   # block end
]


def _dmss_epochs(rows: list[tuple[float, str]]):
    raw = _raw_with_annotations(rows)
    events = events_from_raw(raw)
    metadata = parse_metadata(dmss.spec, events)
    return epoch_with_metadata(raw, dmss.spec, "stim", metadata, _DMSS_BASELINE)


class TestEpochWithMetadataClean:
    def test_clean_one_to_one_alignment(self):
        epochs, excluded = _dmss_epochs(_DMSS_ROWS)
        assert len(epochs) == 2
        assert list(epochs.metadata["Trial"]) == [1, 2]
        assert list(epochs.metadata["Block"]) == [1, 1]
        assert excluded.empty


class TestEpochWithMetadataExtraEpoch:
    def test_duplicate_stim_retained_with_nan_and_logged(self):
        # Inject a duplicate Stim/S 11 into trial 1: 3 stim epochs, 2 metadata
        # rows; the duplicate has no matching metadata row (1:1 onset match).
        rows = _DMSS_ROWS[:4] + [(0.25, "Stim/S 11")] + _DMSS_ROWS[4:]
        epochs, excluded = _dmss_epochs(rows)
        assert len(epochs) == 3
        nan_mask = epochs.metadata["Trial"].isna()
        assert nan_mask.sum() == 1
        trials = sorted(epochs.metadata.loc[~nan_mask, "Trial"].tolist())
        assert trials == [1, 2]
        assert len(excluded) == 1
        assert list(excluded["reason"]) == ["extra_epoch"]


class TestEpochWithMetadataMissingEpoch:
    def test_missing_stim_metadata_row_excluded(self):
        # Drop trial 2's stim marker -> trial 2 metadata row has no epoch.
        rows = [r for r in _DMSS_ROWS if r != (1.4, "Stim/S 25")]
        epochs, excluded = _dmss_epochs(rows)
        assert len(epochs) == 1
        assert list(epochs.metadata["Trial"]) == [1]
        assert len(excluded) == 1
        assert excluded.iloc[0]["Trial"] == 2
        assert excluded.iloc[0]["reason"] == "no_timelock_event"


class TestEpochWithMetadataExpansion:
    def test_expand_on_event_aligns_by_num_sel(self):
        # IGT trial with two selections, shifted so tmin=-1.2 fits a >=0 raw.
        rows = [
            (1.5, "Stim/S 20"),   # block
            (1.6, "Stim/S 30"),   # trial
            (1.8, "Stim/S 40"),   # select 1
            (1.9, "Stim/S 41"),
            (2.0, "Stim/S 45"),
            (2.3, "Stim/S 40"),   # select 2
            (2.4, "Stim/S 42"),
            (2.5, "Stim/S 46"),
            (2.7, "Stim/S 50"),   # submit
            (2.9, "Stim/S 61"),
            (3.0, "Stim/S 31"),
            (3.1, "Stim/S 21"),
        ]
        raw = _raw_with_annotations(rows, duration=5.0)
        events = events_from_raw(raw)
        metadata = parse_metadata(igt.spec, events, expand_trials=True)
        assert len(metadata) == 2
        epochs, excluded = epoch_with_metadata(
            raw, igt.spec, "select", metadata, _DMSS_BASELINE
        )
        assert len(epochs) == 2
        assert list(epochs.metadata["Num_Sel"]) == [0, 1]
        assert list(epochs.metadata["Trial"]) == [1, 1]
        assert excluded.empty


class TestEpochWithMetadataBadInterval:
    def test_bad_annotation_drops_epoch_and_logs(self):
        # US-008-style bad-interval annotation overlapping trial 2's stim epoch
        # (window [1.2, 2.6]) but not trial 1's ([0.0, 1.4]).
        rows = _DMSS_ROWS + [(1.5, "bad_minmax_zscore")]
        raw = _raw_with_annotations(rows, durations=([0.0] * len(_DMSS_ROWS)) + [0.3])
        events = events_from_raw(raw)
        metadata = parse_metadata(dmss.spec, events)
        epochs, excluded = epoch_with_metadata(
            raw, dmss.spec, "stim", metadata, _DMSS_BASELINE
        )
        assert len(epochs) == 1
        assert list(epochs.metadata["Trial"]) == [1]
        assert len(excluded) == 1
        assert excluded.iloc[0]["Trial"] == 2
        assert excluded.iloc[0]["reason"] == "bad_interval"


class TestEpochWithMetadataEmpty:
    def test_no_timelock_events_excludes_all_metadata(self):
        # dmss raw with no stim markers -> 0 epochs, all metadata rows excluded.
        rows = [
            (0.0, "Stim/S  3"),
            (0.1, "Stim/S  5"),
            (0.5, "Stim/S 57"),
            (1.0, "Stim/S 64"),
            (1.2, "Stim/S  6"),
            (2.0, "Stim/S  4"),
        ]
        epochs, excluded = _dmss_epochs(rows)
        assert len(epochs) == 0
        assert len(excluded) == 1
        assert excluded.iloc[0]["reason"] == "no_timelock_event"


class TestAlignEpochsMetadataSeconds:
    def test_seconds_onset_alignment(self):
        raw = _raw_with_annotations(_DMSS_ROWS)
        events = events_from_raw(raw)
        metadata = parse_metadata(dmss.spec, events)  # Onset in seconds
        ev, ev_id = mne.events_from_annotations(raw)
        ev_id = {k.replace("Stimulus", "Stim"): v for k, v in ev_id.items()}
        stim_ids = [ev_id[c] for c in dmss.spec.timelocks["stim"].values() if c in ev_id]
        stim_ev = ev[np.isin(ev[:, 2], stim_ids)]
        tmin, tmax = dmss.spec.intervals["stim"]
        evt_id = {k: ev_id[v] for k, v in dmss.spec.timelocks["stim"].items() if v in ev_id}
        epochs = mne.Epochs(
            raw, stim_ev, evt_id,
            tmin=tmin, tmax=tmax, baseline=_DMSS_BASELINE, preload=True, verbose=False,
        )
        aligned = align_epochs_metadata(epochs, metadata)
        assert len(aligned) == 2
        assert list(aligned.metadata["Trial"]) == [1, 2]
