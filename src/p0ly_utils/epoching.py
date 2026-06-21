from __future__ import annotations

import mne
import numpy as np
import pandas as pd


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
    """Align events and metadata by their onset times.

    Args:
        epochs: MNE epochs object whose event onsets are matched to metadata.
        metadata: DataFrame containing an ``Onset`` column in milliseconds.

    Returns:
        Epochs subset aligned to matching metadata rows.
    """
    epoch_onsets_ms = epochs.events[:, 0] * 1000 / epochs.info["sfreq"]
    meta_onsets_ms = metadata.Onset.to_numpy()
    matched_epoch_idx, matched_meta_idx = _match_onsets(epoch_onsets_ms, meta_onsets_ms)

    epochs = epochs[matched_epoch_idx]
    epochs.metadata = metadata.iloc[matched_meta_idx]
    return epochs
