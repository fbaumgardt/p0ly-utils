from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class SegmentStrategy(ABC):
    @abstractmethod
    def segment(self, evt: np.ndarray, ids: dict[str, int]) -> list[tuple[int, int]]:
        """Return (start, end) index pairs for slicing ``evt`` (end exclusive)."""


@dataclass
class PairedMarkers(SegmentStrategy):
    start: str
    end: str
    offset: tuple[int, int] = (1, 0)

    def segment(self, evt: np.ndarray, ids: dict[str, int]) -> list[tuple[int, int]]:
        if self.start not in ids or self.end not in ids:
            return []
        starts = np.where(evt[:, 2] == ids[self.start])[0]
        ends = np.where(evt[:, 2] == ids[self.end])[0]
        start_off, end_off = self.offset
        return [(int(s) + start_off, int(e) + end_off) for s, e in zip(starts, ends)]


@dataclass
class WholeRecording(SegmentStrategy):
    start: int = 0
    end: int | None = None

    def segment(self, evt: np.ndarray, ids: dict[str, int]) -> list[tuple[int, int]]:
        del ids
        end = len(evt) if self.end is None else self.end
        return [(self.start, end)]


@dataclass
class InferFromColumn(SegmentStrategy):
    """Sentinel strategy: parser assigns block numbers post-hoc from a column."""

    column: str
    start: int = 0
    end: int | None = None

    def segment(self, evt: np.ndarray, ids: dict[str, int]) -> list[tuple[int, int]]:
        del ids
        end = len(evt) if self.end is None else self.end
        return [(self.start, end)]


class ColumnExtractor(ABC):
    @abstractmethod
    def extract(self, evt_t: np.ndarray, ids: dict[str, int]) -> Any:
        pass


@dataclass
class CodeLookup(ColumnExtractor):
    code_map: dict[str, str]
    agg: str = "join"

    def extract(self, evt_t: np.ndarray, ids: dict[str, int]) -> str:
        resolved = {
            ids[code]: label for label, code in self.code_map.items() if code in ids
        }
        if self.agg == "first":
            for event_id in evt_t[:, 2]:
                if event_id in resolved:
                    return resolved[event_id]
            return ""
        return "".join(resolved.get(event_id, "") for event_id in evt_t[:, 2])


@dataclass
class BoolPresence(ColumnExtractor):
    code: str

    def extract(self, evt_t: np.ndarray, ids: dict[str, int]) -> bool:
        if self.code not in ids:
            return False
        code_id = ids[self.code]
        return any(event_id == code_id for event_id in evt_t[:, 2])


@dataclass
class IntSum(ColumnExtractor):
    code_map: dict[str, int]

    def extract(self, evt_t: np.ndarray, ids: dict[str, int]) -> int:
        resolved = {
            ids[code]: value for code, value in self.code_map.items() if code in ids
        }
        return sum(resolved.get(event_id, 0) for event_id in evt_t[:, 2])


@dataclass
class ListCollect(ColumnExtractor):
    code_map: dict[str, Any]

    def extract(self, evt_t: np.ndarray, ids: dict[str, int]) -> list[Any]:
        resolved = {
            ids[code]: value
            for value, code in self.code_map.items()
            if code in ids
        }
        return [resolved[event_id] for event_id in evt_t[:, 2] if event_id in resolved]


@dataclass
class RTMeasure:
    name: str
    start: list[str]
    end: list[str]
    nan_if_negative: bool = True

    def extract(self, evt_t: np.ndarray, ids: dict[str, int]) -> float:
        start_ids = {ids[code] for code in self.start if code in ids}
        end_ids = {ids[code] for code in self.end if code in ids}
        begin = sum(
            evt_t[row, 0] for row, event_id in enumerate(evt_t[:, 2]) if event_id in start_ids
        )
        end = sum(
            evt_t[row, 0] for row, event_id in enumerate(evt_t[:, 2]) if event_id in end_ids
        )
        rt = end - begin
        if self.nan_if_negative and rt < 0:
            return float("nan")
        return float(rt)


@dataclass
class ExperimentSpec:
    name: str
    timelocks: dict[str, dict[str, str]]
    intervals: dict[str, tuple[float, float]]
    block_strategy: SegmentStrategy
    trial_strategy: PairedMarkers
    columns: dict[str, ColumnExtractor]
    rt_defs: list[RTMeasure] = field(default_factory=list)
