from __future__ import annotations

import pandas as pd

from p0ly_utils.metadata.core import ExperimentSpec, InferFromColumn


def _infer_blocks(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty:
        return df
    blocks = [1]
    for row in range(1, len(df)):
        if df[column].iloc[row - 1] != df[column].iloc[row]:
            blocks.append(blocks[-1] + 1)
        else:
            blocks.append(blocks[-1])
    df = df.copy()
    df["Block"] = blocks
    return df


def parse_metadata(
    spec: ExperimentSpec,
    evt,
    ids: dict[str, int],
    csv_path: str | None = None,
) -> pd.DataFrame:
    del csv_path  # reserved for future CSV merge support

    rows: list[dict] = []
    blocks = spec.block_strategy.segment(evt, ids)
    for block_idx, (block_start, block_end) in enumerate(blocks):
        evt_b = evt[block_start:block_end, :]
        trials = spec.trial_strategy.segment(evt_b, ids)
        for trial_idx, (trial_start, trial_end) in enumerate(trials):
            evt_t = evt_b[trial_start:trial_end, :]
            row: dict = {"Block": block_idx + 1, "Trial": trial_idx + 1}
            for col_name, extractor in spec.columns.items():
                row[col_name] = extractor.extract(evt_t, ids)
            for rt_def in spec.rt_defs:
                row[rt_def.name] = rt_def.extract(evt_t, ids)
            rows.append(row)

    df = pd.DataFrame(rows)
    if isinstance(spec.block_strategy, InferFromColumn):
        df = _infer_blocks(df, spec.block_strategy.column)
    return df
