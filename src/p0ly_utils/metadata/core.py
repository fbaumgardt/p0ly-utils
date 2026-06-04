from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


class ColumnExtractor(ABC):
    @abstractmethod
    def extract(self, group: pd.DataFrame) -> Any:
        """Compute one column's value from a single trial's event rows."""

    @abstractmethod
    def codes(self) -> set[str]:
        """Event codes this extractor consumes; used to keep relevant rows."""


@dataclass
class CodeLookup(ColumnExtractor):
    """Map present event codes back to human labels.

    `agg="first"` returns the label of the first matching code (use when exactly
    one of the codes is expected); the default concatenates all matches in event
    order (non-matching rows contribute "").
    """

    code_map: dict[str, str]
    agg: str = "join"

    def extract(self, group: pd.DataFrame) -> str:
        # code_map is label->code; invert because the group gives us codes.
        reverse = {code: label for label, code in self.code_map.items()}
        if self.agg == "first":
            for desc in group["description"]:
                if desc in reverse:
                    return reverse[desc]
            return ""
        return "".join(reverse.get(desc, "") for desc in group["description"])

    def codes(self) -> set[str]:
        return set(self.code_map.values())


@dataclass
class BoolPresence(ColumnExtractor):
    code: str

    def extract(self, group: pd.DataFrame) -> bool:
        return self.code in group["description"].values

    def codes(self) -> set[str]:
        return {self.code}


@dataclass
class IntSum(ColumnExtractor):
    """Sum the integer weights of any matching codes (codes absent => 0).

    Useful for score-like columns where each marker contributes points.
    """

    code_map: dict[str, int]

    def extract(self, group: pd.DataFrame) -> int:
        return int(group["description"].map(self.code_map).fillna(0).sum())

    def codes(self) -> set[str]:
        return set(self.code_map.keys())


@dataclass
class ListCollect(ColumnExtractor):
    """Collect every matching code's value, preserving event (chronological) order.

    Unlike CodeLookup this keeps each occurrence, so it feeds the per-selection
    lists consumed by ExpandOnEvent.
    """

    code_map: dict[str, Any]

    def extract(self, group: pd.DataFrame) -> list[Any]:
        reverse = {code: value for value, code in self.code_map.items()}
        return group["description"].map(reverse).dropna().tolist()

    def codes(self) -> set[str]:
        return set(self.code_map.values())


@dataclass
class DerivedColumn(ColumnExtractor):
    """Column computed from other already-extracted columns, not from raw events.

    The parser detects this type and evaluates it in a final pass (see
    _extract_row), passing the partially-built row instead of the event group.
    """

    depends_on: list[str]
    fn: Callable[[dict[str, Any]], Any]

    def extract(self, group: pd.DataFrame) -> Any:
        # Never called: extract() operates on events, but a derived value needs
        # the assembled row. The parser routes DerivedColumns through derive().
        del group
        raise NotImplementedError("DerivedColumn uses derive()")

    def derive(self, row: dict[str, Any]) -> Any:
        return self.fn({key: row[key] for key in self.depends_on})

    def codes(self) -> set[str]:
        # Consumes columns, not raw event codes, so it contributes none.
        return set()


@dataclass
class ExpandOnEvent:
    """Turn a trial with repeated `event_code` markers into one row per occurrence.

    per_event_columns are extractors (typically ListCollect) whose results are
    indexed positionally against the selections — see parser._expand_trial_rows.
    """

    event_code: str
    per_event_columns: dict[str, ColumnExtractor]


@dataclass
class RTMeasure:
    """Reaction time as onset(end marker) - onset(start marker) within a trial."""

    name: str
    start: list[str]
    end: list[str]
    nan_if_negative: bool = True

    def extract(self, group: pd.DataFrame) -> float:
        # sum() collapses the (expected single) matching onset to a scalar and
        # yields 0 when a marker is missing.
        begin = group.loc[group["description"].isin(self.start), "onset"].sum()
        end = group.loc[group["description"].isin(self.end), "onset"].sum()
        rt = float(end - begin)
        # A negative RT means the markers are missing or out of order; report NaN
        # rather than a nonsensical negative latency.
        if self.nan_if_negative and rt < 0:
            return float("nan")
        return rt


@dataclass
class ExperimentSpec:
    """Declarative description of one experiment's metadata extraction.

    Fields:
      timelocks / intervals: epoching config consumed downstream of the parser.
      trial_codes / block_codes: markers that delimit trials and blocks.
      columns: output column name -> extractor.
      trial_end: see parser._assign_blocks_trials (shift amount, 0 or 1).
      rt_defs: reaction-time columns.
      trial_expander: optional one-row-per-selection expansion.
      csv_columns: output col -> (csv col, transform); positionally merged.
      infer_block_from: derive a Block column from an output column post-hoc.
    """

    name: str
    timelocks: dict[str, dict[str, str]]
    intervals: dict[str, tuple[float, float]]
    trial_codes: list[str]
    columns: dict[str, ColumnExtractor]
    block_codes: list[str] = field(default_factory=list)
    trial_end: bool = False
    rt_defs: list[RTMeasure] = field(default_factory=list)
    trial_expander: ExpandOnEvent | None = None
    csv_columns: dict[str, tuple[str, Callable[[Any], Any]]] | None = None
    infer_block_from: str | None = None

    def column_codes(self) -> set[str]:
        # Union of every code any extractor/RT/expander needs: this is what the
        # parser uses to decide which blocks/trials carry real data.
        codes: set[str] = set()
        for tl in self.timelocks.values():
            codes.update(tl.values())
        for extractor in self.columns.values():
            codes |= extractor.codes()
        for rt_def in self.rt_defs:
            codes.update(rt_def.start)
            codes.update(rt_def.end)
        if self.trial_expander is not None:
            codes.add(self.trial_expander.event_code)
            for extractor in self.trial_expander.per_event_columns.values():
                codes |= extractor.codes()
        return codes

    def event_codes(self) -> set[str]:
        # Full marker allowlist for the initial _select_events filter: column
        # codes plus the structural trial/block delimiters.
        return set(self.trial_codes) | set(self.block_codes) | self.column_codes()
