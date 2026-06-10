from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from p0ly_utils.epoching import _match_onsets

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
