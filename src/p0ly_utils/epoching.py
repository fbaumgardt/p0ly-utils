from __future__ import annotations

from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from p0ly_utils.metadata.core import ExperimentSpec


def _match_onsets(
    epoch_onsets_ms: np.ndarray,
    meta_onsets_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align epoch and metadata trial lists by reciprocal nearest onset match."""
    n_epochs = len(epoch_onsets_ms)
    n_meta = len(meta_onsets_ms)
    if n_epochs == 0 or n_meta == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    arr = epoch_onsets_ms[:, np.newaxis] - meta_onsets_ms[np.newaxis, :]
    df_match_to_evt = np.sum(arr >= 0, axis=1) - 1
    evt_match_to_df = np.sum(arr < 0, axis=0)

    epoch_idx = np.arange(n_epochs)
    meta_idx = df_match_to_evt
    in_bounds = (meta_idx >= 0) & (meta_idx < n_meta)
    mutual = in_bounds & (evt_match_to_df[meta_idx] == epoch_idx)
    matched_epoch_idx = epoch_idx[mutual]
    matched_meta_idx = meta_idx[mutual]

    if len(matched_epoch_idx) == 0:
        return matched_epoch_idx, matched_meta_idx

    if n_meta >= 2:
        tolerance_ms = float(np.mean(np.diff(meta_onsets_ms)))
    else:
        tolerance_ms = float("inf")

    close_enough = np.abs(arr[matched_epoch_idx, matched_meta_idx]) < tolerance_ms
    return matched_epoch_idx[close_enough], matched_meta_idx[close_enough]


def align_epochs_metadata(epochs: mne.BaseEpochs, metadata: pd.DataFrame) -> mne.BaseEpochs:
    """Align events and metadata by their onset times (seconds).

    Thin Path-A onset matcher: subsets ``epochs`` to those whose event onset
    reciprocal-nearest-matches a metadata ``Onset`` and attaches the matched
    rows as ``epochs.metadata``. Mismatches are silently dropped here — use
    :func:`epoch_with_metadata` for the explicit mismatch bookkeeping
    (``excluded_trials.csv``) required by the pipeline epoch rule.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs whose event onsets are matched to metadata. Event onsets are
        read from ``epochs.events[:, 0]`` (samples) and converted to seconds
        via ``epochs.info["sfreq"]``.
    metadata : pd.DataFrame
        DataFrame with an ``Onset`` column in **seconds** (SCHEMA §2), e.g. the
        US-016 ``events_from_raw`` → ``parse_metadata`` output.

    Returns
    -------
    mne.BaseEpochs
        Epochs subset aligned to matching metadata rows, with
        ``.metadata`` attached.
    """
    epoch_onsets_s = epochs.events[:, 0] / epochs.info["sfreq"]
    meta_onsets_s = metadata["Onset"].to_numpy()
    matched_epoch_idx, matched_meta_idx = _match_onsets(epoch_onsets_s, meta_onsets_s)

    epochs = epochs[matched_epoch_idx]
    epochs.metadata = metadata.iloc[matched_meta_idx].reset_index(drop=True)
    return epochs


# Columns written to ``excluded_trials.csv`` (SCHEMA §6). ``subject`` /
# ``timelock`` scope a row to one (subject, timelock) epoch run; ``Block`` /
# ``Trial`` / ``Onset`` identify the trial; ``reason`` classifies the mismatch.
_EXCLUDED_COLUMNS = ["subject", "timelock", "Block", "Trial", "Onset", "reason"]


def _empty_excluded(subject: str | None, timelock: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in _EXCLUDED_COLUMNS})


def _excluded_rows(
    block,
    trial,
    onset,
    reason,
    subject: str | None,
    timelock: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": subject,
            "timelock": timelock,
            "Block": block,
            "Trial": trial,
            "Onset": onset,
            "reason": reason,
        }
    )


def _drop_reasons(drop_log: tuple[tuple[str, ...], ...]) -> np.ndarray:
    """Classify each drop_log entry as ``bad_interval`` or ``dropped``.

    ``mne.Epochs(reject_by_annotation=True)`` records the overlapping
    annotation description (e.g. ``bad_minmax_zscore`` from US-008) in the
    drop log; anything starting with ``bad`` (case-insensitive) is a
    bad-interval reject.
    """
    reasons: list[str] = []
    for entry in drop_log:
        if any(str(tag).lower().startswith("bad") for tag in entry):
            reasons.append("bad_interval")
        else:
            reasons.append("dropped")
    return np.asarray(reasons, dtype=object)


def _trial_lookup(trial_meta: pd.DataFrame, trial_onsets_s: np.ndarray):
    """Return (sorted_onsets, order) so searchsorted maps onsets to trial rows."""
    order = np.argsort(trial_onsets_s, kind="stable")
    return trial_onsets_s[order], order


def epoch_with_metadata(
    raw: mne.io.BaseRaw,
    spec: ExperimentSpec,
    timelock: str,
    metadata: pd.DataFrame,
    baseline: list[float] | tuple[float, float] | None,
    *,
    subject: str | None = None,
) -> tuple[mne.BaseEpochs, pd.DataFrame]:
    """Segment ``raw`` into epochs for one timelock and align trial metadata.

    Cuts ``mne.Epochs`` from the timelock's annotation events using
    ``spec.intervals[timelock]`` → ``(tmin, tmax)`` and the shared ``baseline``
    window, then reconciles the by-trial ``metadata`` (US-016 output) with the
    cut epochs **explicitly** — mismatches are retained/logged, never silently
    dropped:

    * epochs with no matching metadata row → **retained with NaN metadata**
      and logged to the excluded frame as ``extra_epoch``;
    * metadata rows with no matching epoch → excluded frame, reason
      ``no_timelock_event`` (or ``bad_interval`` when that trial's timelock
      epoch was dropped by a US-008 bad-interval annotation);
    * epochs overlapping a ``bad*`` annotation → dropped by
      ``mne.Epochs(reject_by_annotation=True)`` and logged as ``bad_interval``.

    Alignment is key-based on ``(Block, Trial)`` for one-epoch-per-trial
    timelocks, and on ``(Block, Trial, Num_Sel)`` for the experiment's
    ``ExpandOnEvent`` timelock (e.g. IGT ``select``). Trial identity is
    assigned to each epoch by onset proximity to the metadata ``Onset`` column
    (seconds, SCHEMA §2). No per-channel/per-epoch loops, no ``iterrows``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Cleaned continuous raw (US-008 output) carrying event-marker
        ``Annotations`` and bad-interval ``Annotations``. Underlying data
        shape ``(n_channels, n_times)``.
    spec : ExperimentSpec
        Experiment spec providing ``timelocks`` / ``intervals`` / optional
        ``trial_expander``.
    timelock : str
        Key into ``spec.timelocks`` / ``spec.intervals``.
    metadata : pd.DataFrame
        By-trial metadata (US-016 ``parse_metadata`` output) with ``Block``,
        ``Trial``, ``Onset`` (seconds) and, when ``expand_trials`` was set,
        ``Num_Sel``. One row per trial (or per selection for expanded rows).
    baseline : list[float] | tuple[float, float] | None
        Baseline window ``[start, end]`` in seconds (shared across timelocks,
        from ``config.yaml`` ``epoching.baseline``); ``None`` disables.
    subject : str | None
        Subject id stamped onto excluded-frame rows for ``excluded_trials.csv``
        (SCHEMA §6). Pure label — no filesystem or signal use.

    Returns
    -------
    epochs : mne.BaseEpochs
        Epochs for this timelock (shape ``(n_epochs, n_channels, n_times)``)
        with ``.metadata`` attached; extra epochs carry NaN metadata rows.
    excluded : pd.DataFrame
        ``excluded_trials.csv`` content (columns ``subject, timelock, Block,
        Trial, Onset, reason``). One row per mismatched metadata row or
        dropped/extra epoch.
    """
    # ---- events from annotations ----
    events, event_id = mne.events_from_annotations(raw)
    # Recordings may label markers "Stimulus/..."; specs use "Stim/...".
    event_id = {k.replace("Stimulus", "Stim"): v for k, v in event_id.items()}

    tl_map: dict[str, str] = spec.timelocks[timelock]
    evt_id: dict[str, int] = {
        label: event_id[code] for label, code in tl_map.items() if code in event_id
    }
    tmin, tmax = spec.intervals[timelock]

    target_ids = np.fromiter(evt_id.values(), dtype=int)
    sel = np.isin(events[:, 2], target_ids) if len(target_ids) else np.zeros(len(events), dtype=bool)
    tl_events = events[sel]

    expander_code = spec.trial_expander.event_code if spec.trial_expander is not None else None
    is_expander = expander_code is not None and expander_code in set(tl_map.values())

    meta_cols = list(metadata.columns)

    # ---- empty-events guard: nothing to segment ----
    if len(tl_events) == 0:
        epochs = _empty_epochs(raw, tmin, tmax, baseline)
        excluded = _excluded_rows(
            metadata["Block"].to_numpy(),
            metadata["Trial"].to_numpy(),
            metadata["Onset"].to_numpy(),
            np.full(len(metadata), "no_timelock_event", dtype=object),
            subject,
            timelock,
        )
        return epochs, excluded

    # ---- segment ----
    epochs = mne.Epochs(
        raw,
        tl_events,
        evt_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject_by_annotation=True,
        preload=True,
        verbose=False,
    )
    sfreq = float(epochs.info["sfreq"])

    # Candidate timelock-event onsets (all, in seconds) -> kept vs dropped.
    cand_onsets_s = tl_events[:, 0] / sfreq
    kept_mask = np.zeros(len(tl_events), dtype=bool)
    kept_mask[epochs.selection] = True
    survived_onsets_s = epochs.events[:, 0] / sfreq
    n_kept = len(epochs)

    # Dropped epochs (bad-interval rejects) — onsets + reasons.
    dropped_idx = np.flatnonzero(~kept_mask)
    dropped_onsets_s = cand_onsets_s[dropped_idx]
    dropped_reasons = _drop_reasons(epochs.drop_log)[dropped_idx]

    # ---- trial-level metadata view + trial onsets (seconds) ----
    has_num_sel = "Num_Sel" in metadata.columns
    if is_expander and has_num_sel:
        meta_exp = metadata.reset_index(drop=True)
        trial_meta = meta_exp.groupby(["Block", "Trial"], as_index=False).first()
    else:
        # Collapse to one row per (Block, Trial). For expanded metadata used
        # against a non-expander timelock (e.g. IGT submit/fdb), the first row
        # per trial carries the trial-level columns shared across selections.
        if has_num_sel:
            trial_meta = metadata.groupby(["Block", "Trial"], as_index=False).first()
        else:
            trial_meta = metadata.reset_index(drop=True)
        meta_exp = None

    trial_onsets_s = trial_meta["Onset"].to_numpy(dtype=float)
    sorted_onsets, order = _trial_lookup(trial_meta, trial_onsets_s)
    bt = trial_meta[["Block", "Trial"]].to_numpy()

    # Dropped epochs → nearest preceding trial (searchsorted, seconds) for
    # the bad_interval cross-reference. -1 ⇒ before the first trial.
    def _assign_trials(onsets_s: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(sorted_onsets, onsets_s, side="right") - 1
        trial_row = np.where(idx >= 0, order[np.clip(idx, 0, None)], -1)
        return trial_row

    dropped_trial_row = _assign_trials(dropped_onsets_s) if len(dropped_onsets_s) else np.array([], dtype=int)
    dropped_trials = set(
        tuple(bt[r]) for r in dropped_trial_row if r >= 0
    )

    excluded_frames: list[pd.DataFrame] = []

    if is_expander and meta_exp is not None:
        # ---- Path B: N epochs per trial, key-join on (Block, Trial, Num_Sel) ----
        ep = pd.DataFrame(
            {
                "ep_pos": np.arange(n_kept),
                "Onset_s": survived_onsets_s,
            }
        )
        ep_trial_row = _assign_trials(survived_onsets_s)
        valid = ep_trial_row >= 0
        valid_rows = ep_trial_row.clip(0, None)
        ep["Block"] = np.where(valid, bt[valid_rows, 0], np.nan)
        ep["Trial"] = np.where(valid, bt[valid_rows, 1], np.nan)

        # Rank epochs within (Block, Trial) by onset → Num_Sel (vectorized
        # groupby cumcount on the sorted-by-onset order).
        ep_sorted = ep.loc[valid].sort_values(["Block", "Trial", "Onset_s"])
        ep_sorted["Num_Sel"] = ep_sorted.groupby(["Block", "Trial"]).cumcount()
        ep["Num_Sel"] = np.nan
        ep.loc[ep_sorted.index, "Num_Sel"] = ep_sorted["Num_Sel"].to_numpy()

        merged = ep.merge(
            meta_exp, on=["Block", "Trial", "Num_Sel"], how="left", sort=False
        )
        aligned = merged[meta_cols].reset_index(drop=True)
        epochs.metadata = aligned

        # Excluded metadata rows: expanded rows with no matching epoch.
        matched_keys = (
            merged.loc[merged["Num_Sel"].notna(), ["Block", "Trial", "Num_Sel"]]
            .drop_duplicates()
        )
        unmatched = meta_exp.merge(
            matched_keys, on=["Block", "Trial", "Num_Sel"], how="left", indicator=True
        )
        unmatched = unmatched[unmatched["_merge"] == "left_only"].drop(columns="_merge")
        if len(unmatched):
            ubt = unmatched[["Block", "Trial"]].to_numpy()
            is_bad = np.array([tuple(r) in dropped_trials for r in ubt])
            reasons = np.where(is_bad, "bad_interval", "no_timelock_event")
            excluded_frames.append(
                _excluded_rows(
                    unmatched["Block"].to_numpy(),
                    unmatched["Trial"].to_numpy(),
                    unmatched["Onset"].to_numpy(),
                    reasons,
                    subject,
                    timelock,
                )
            )
        # Extra epochs (no trial or no matching Num_Sel row): retained NaN,
        # logged as extra_epoch.
        extra_mask = merged["Num_Sel"].isna() if "Num_Sel" in merged else np.ones(n_kept, dtype=bool)
        extra_mask = aligned["Block"].isna().to_numpy() if "Block" in aligned else extra_mask.to_numpy()
        if extra_mask.any():
            extra_pos = np.flatnonzero(extra_mask)
            extra_onsets = survived_onsets_s[extra_pos]
            extra_trial_row = ep_trial_row[extra_pos]
            extra_block = np.where(
                extra_trial_row >= 0, bt[extra_trial_row.clip(0, None), 0], np.nan
            )
            extra_trial = np.where(
                extra_trial_row >= 0, bt[extra_trial_row.clip(0, None), 1], np.nan
            )
            excluded_frames.append(
                _excluded_rows(
                    extra_block,
                    extra_trial,
                    extra_onsets,
                    np.full(len(extra_pos), "extra_epoch", dtype=object),
                    subject,
                    timelock,
                )
            )
    else:
        # ---- Path A: 1 epoch per trial, reciprocal onset match on seconds ----
        matched_ep, matched_meta = _match_onsets(survived_onsets_s, trial_onsets_s)
        aligned = pd.DataFrame(np.nan, index=np.arange(n_kept), columns=meta_cols)
        if len(matched_ep):
            aligned.iloc[matched_ep] = trial_meta.iloc[matched_meta][meta_cols].to_numpy()
        epochs.metadata = aligned.reset_index(drop=True)

        # Excluded metadata rows (trials with no surviving epoch).
        matched_meta_set = set(matched_meta.tolist()) if len(matched_meta) else set()
        all_rows = np.arange(len(trial_meta))
        unmatched_rows = np.setdiff1d(all_rows, np.asarray(list(matched_meta_set), dtype=int)) \
            if matched_meta_set else all_rows
        if len(unmatched_rows):
            ubt = bt[unmatched_rows]
            is_bad = np.array([tuple(r) in dropped_trials for r in ubt])
            reasons = np.where(is_bad, "bad_interval", "no_timelock_event")
            excluded_frames.append(
                _excluded_rows(
                    bt[unmatched_rows, 0],
                    bt[unmatched_rows, 1],
                    trial_onsets_s[unmatched_rows],
                    reasons,
                    subject,
                    timelock,
                )
            )
        # Extra epochs (survived but no metadata match): retained NaN, logged.
        extra_pos = np.setdiff1d(np.arange(n_kept), matched_ep)
        if len(extra_pos):
            extra_onsets = survived_onsets_s[extra_pos]
            extra_trial_row = _assign_trials(extra_onsets)
            extra_block = np.where(
                extra_trial_row >= 0, bt[extra_trial_row.clip(0, None), 0], np.nan
            )
            extra_trial = np.where(
                extra_trial_row >= 0, bt[extra_trial_row.clip(0, None), 1], np.nan
            )
            excluded_frames.append(
                _excluded_rows(
                    extra_block,
                    extra_trial,
                    extra_onsets,
                    np.full(len(extra_pos), "extra_epoch", dtype=object),
                    subject,
                    timelock,
                )
            )

    # Bad-interval-dropped epochs that did NOT map to any metadata trial still
    # get a drop log entry (reason bad_interval / dropped), with NaN trial id.
    if len(dropped_onsets_s):
        dropped_no_trial = dropped_trial_row < 0
        if dropped_no_trial.any():
            pos = np.flatnonzero(dropped_no_trial)
            excluded_frames.append(
                _excluded_rows(
                    np.full(pos.size, np.nan),
                    np.full(pos.size, np.nan),
                    dropped_onsets_s[pos],
                    dropped_reasons[pos],
                    subject,
                    timelock,
                )
            )

    excluded = (
        pd.concat(excluded_frames, ignore_index=True)
        if excluded_frames
        else _empty_excluded(subject, timelock)
    )
    return epochs, excluded


def _empty_epochs(
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
    baseline: list[float] | tuple[float, float] | None,
) -> mne.BaseEpochs:
    """Construct a zero-epoch ``mne.Epochs`` matching ``raw``'s info/timing."""
    sfreq = float(raw.info["sfreq"])
    n_times = int(round((tmax - tmin) * sfreq))
    data = np.zeros((0, len(raw.info["ch_names"]), max(n_times, 1)), dtype=np.float64)
    # Explicit ``drop_log=()`` avoids MNE's default drop-log computation, which
    # calls ``max(self.selection)`` and raises on an empty (zero-epoch) selection.
    return mne.EpochsArray(
        data, raw.info, tmin=tmin, baseline=baseline, drop_log=(), verbose=False
    )
