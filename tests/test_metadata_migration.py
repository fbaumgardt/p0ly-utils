import numpy as np
import pandas as pd

from p0ly_utils.metadata import dmss, dotprobe, mgsearch
from p0ly_utils.metadata._legacy import (
    dmss_get_metadata,
    dotprobe_get_metadata,
    mgsearch_get_metadata,
)
from p0ly_utils.metadata.core import (
    BoolPresence,
    CodeLookup,
    ExperimentSpec,
    IntSum,
    ListCollect,
    PairedMarkers,
)


def _codes_from_spec(spec: ExperimentSpec) -> list[str]:
    codes: set[str] = set()
    for strategy in (spec.block_strategy, spec.trial_strategy):
        if isinstance(strategy, PairedMarkers):
            codes.update((strategy.start, strategy.end))
    for extractor in spec.columns.values():
        if isinstance(extractor, CodeLookup):
            codes.update(extractor.code_map.values())
        elif isinstance(extractor, (IntSum, ListCollect)):
            codes.update(extractor.code_map.keys())
        elif isinstance(extractor, BoolPresence):
            codes.add(extractor.code)
    for rt_def in spec.rt_defs:
        codes.update(rt_def.start)
        codes.update(rt_def.end)
    return sorted(codes)


def _full_ids_for_spec(spec: ExperimentSpec) -> dict[str, int]:
    return {code: index + 1 for index, code in enumerate(_codes_from_spec(spec))}


def _evt(rows: list[tuple[float, int]]) -> np.ndarray:
    return np.array([[time, 0, code] for time, code in rows], dtype=float)


def _assert_frames_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_like=True)


class TestDotprobeMetadata:
    def test_matches_legacy(self):
        ids = _full_ids_for_spec(dotprobe.spec)
        ids["Stim/S 31"] = max(ids.values()) + 1
        code = ids["Stim/S  9"]
        end_code = ids["Stim/S 10"]
        evt = _evt(
            [
                (0.0, code),
                (0.1, ids["Stim/S 11"]),
                (0.2, ids["Stim/S 16"]),
                (0.5, ids["Stim/S 14"]),
                (1.0, ids["Stim/S 37"]),
                (1.2, ids["Stim/S 34"]),
                (1.3, ids["Stim/S 30"]),
                (1.4, end_code),
                (1.5, code),
                (1.6, ids["Stim/S 11"]),
                (1.7, ids["Stim/S 14"]),
                (2.0, ids["Stim/S 30"]),
                (2.1, end_code),
            ]
        )
        legacy = dotprobe_get_metadata(evt, ids)
        current = dotprobe.get_metadata(evt, ids)
        _assert_frames_equal(current, legacy)

    def test_negative_rt_becomes_nan(self):
        ids = _full_ids_for_spec(dotprobe.spec)
        evt = _evt(
            [
                (0.0, ids["Stim/S  9"]),
                (0.5, ids["Stim/S 14"]),
                (1.5, ids["Stim/S 10"]),
            ]
        )
        result = dotprobe.get_metadata(evt, ids)
        assert np.isnan(result["RT"].iloc[0])


class TestMgsearchMetadata:
    def test_matches_legacy(self):
        ids = _full_ids_for_spec(mgsearch.spec)
        evt = _evt(
            [
                (0.0, 999),
                (0.1, ids["Stim/S182"]),
                (0.2, ids["Stim/S 10"]),
                (0.3, ids["Stim/S201"]),
                (0.4, ids["Stim/S121"]),
                (0.5, ids["Stim/S141"]),
                (0.6, ids["Stim/S160"]),
                (0.7, ids["Stim/S110"]),
                (1.0, ids["Stim/S192"]),
                (1.2, ids["Stim/S189"]),
                (1.3, ids["Stim/S183"]),
                (1.4, ids["Stim/S182"]),
                (1.5, ids["Stim/S 11"]),
                (1.6, ids["Stim/S111"]),
                (1.9, ids["Stim/S190"]),
                (2.0, ids["Stim/S183"]),
            ]
        )
        legacy = mgsearch_get_metadata(evt, ids)
        current = mgsearch.get_metadata(evt, ids)
        _assert_frames_equal(current, legacy)

    def test_infers_blocks_from_cue_side(self):
        ids = _full_ids_for_spec(mgsearch.spec)
        evt = _evt(
            [
                (0.0, 999),
                (0.1, ids["Stim/S182"]),
                (0.2, ids["Stim/S 10"]),
                (0.5, ids["Stim/S110"]),
                (0.8, ids["Stim/S189"]),
                (0.9, ids["Stim/S183"]),
                (1.0, ids["Stim/S182"]),
                (1.1, ids["Stim/S 10"]),
                (1.4, ids["Stim/S110"]),
                (1.7, ids["Stim/S189"]),
                (1.8, ids["Stim/S183"]),
                (1.9, ids["Stim/S182"]),
                (2.0, ids["Stim/S 11"]),
                (2.3, ids["Stim/S110"]),
                (2.6, ids["Stim/S189"]),
                (2.7, ids["Stim/S183"]),
                (2.8, 999),
            ]
        )
        result = mgsearch.get_metadata(evt, ids)
        assert list(result["Block"]) == [1, 1, 2]


class TestDmssMetadata:
    def test_matches_legacy(self):
        ids = _full_ids_for_spec(dmss.spec)
        evt = _evt(
            [
                (0.0, ids["Stim/S  3"]),
                (0.1, ids["Stim/S  5"]),
                (0.2, ids["Stim/S 11"]),
                (0.5, ids["Stim/S 57"]),
                (1.0, ids["Stim/S 64"]),
                (1.2, ids["Stim/S  6"]),
                (1.3, ids["Stim/S  5"]),
                (1.4, ids["Stim/S 57"]),
                (1.7, ids["Stim/S 60"]),
                (1.9, ids["Stim/S  6"]),
                (2.0, ids["Stim/S  4"]),
            ]
        )
        legacy = dmss_get_metadata(evt, ids)
        current = dmss.get_metadata(evt, ids)
        _assert_frames_equal(current, legacy)

    def test_allows_negative_rt(self):
        ids = _full_ids_for_spec(dmss.spec)
        evt = _evt(
            [
                (0.0, ids["Stim/S  3"]),
                (0.1, ids["Stim/S  5"]),
                (0.3, ids["Stim/S 60"]),
                (1.0, ids["Stim/S 57"]),
                (1.2, ids["Stim/S  6"]),
                (1.3, ids["Stim/S  4"]),
            ]
        )
        result = dmss.get_metadata(evt, ids)
        assert result["RT"].iloc[0] < 0
