from __future__ import annotations

import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components


def _minmax_zscore(
    inst: mne.io.BaseRaw | mne.BaseEpochs,
    axis: str = "channels",
    threshold: float = 3.0,
    max_iter: int = 1,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean mask flagging outlier channels or epochs.

    Peak-to-peak amplitude is computed per channel (axis='channels') or per
    epoch (axis='time'), then outliers (|z| > threshold) are identified
    iteratively. At each step the mean and std are re-estimated on the unmasked
    subset, so earlier outliers don't inflate the baseline.
    """
    data = inst.get_data()
    if axis == "time":
        # Average over channels first so that X reflects per-epoch amplitude.
        data = np.mean(data, axis=1)
    X: np.ndarray = np.max(data, axis=-1) - np.min(data, axis=-1)
    current_mask: np.ndarray = mask if mask is not None else np.zeros(len(X), dtype=bool)
    for _ in range(max_iter):
        Y = np.ma.masked_array(X, current_mask)
        mn = float(np.mean(Y))
        sd = float(np.std(Y))
        current_mask = np.abs((X - mn) / sd) >= threshold
    return current_mask


def fix_channels(
    raw: mne.io.BaseRaw,
    threshold: float = 3.0,
    max_iter: int = 1,
) -> mne.io.BaseRaw:
    """Return a copy of raw with statistically extreme channels added to bads."""
    r = raw.copy()
    bad_mask = _minmax_zscore(r, threshold=threshold, max_iter=max_iter)
    new_bads = np.array(r.ch_names)[bad_mask].tolist()
    r.info["bads"] = np.unique(
        np.concatenate((np.array(r.info["bads"]), np.array(new_bads)))
    ).tolist()
    return r


def artefact_rejection(
    raw: mne.io.BaseRaw,
    threshold: float = 3.0,
    max_iter: int = 3,
    duration: float = 0.5,
    stimulation: tuple[float | None, float | None] | None = None,
) -> mne.Annotations:
    """Identify bad time segments via peak-to-peak z-score on fixed-length epochs.

    Returns Annotations marking each bad segment onset with label
    'bad_minmax_zscore'.

    stimulation: optional (start, end) window in seconds to exclude from the
    z-score baseline (e.g. a stimulation artefact window). None endpoints mean
    the beginning / end of the recording. Pass None to skip masking entirely.
    """
    epo = mne.make_fixed_length_epochs(raw, duration=duration, reject_by_annotation=False)

    stim_mask: np.ndarray | None = None
    if stimulation is not None:
        # Mark epochs whose onset falls within the stimulation window so they
        # don't inflate the baseline mean/std used for z-score thresholding.
        epoch_onsets = epo.events[:, 0] / epo.info["sfreq"]
        stim_start = stimulation[0] if stimulation[0] is not None else raw.times[0]
        stim_end = stimulation[1] if stimulation[1] is not None else raw.times[-1]
        stim_mask = (epoch_onsets >= stim_start) & (epoch_onsets <= stim_end)

    bad_mask = _minmax_zscore(
        epo, axis="time", threshold=threshold, max_iter=max_iter, mask=stim_mask
    )
    bad_starts = epo.events[bad_mask, 0] / epo.info["sfreq"]
    return mne.Annotations(bad_starts, duration, "bad_minmax_zscore", orig_time=raw.annotations.orig_time)


def ica_clean_dnn(
    raw: mne.io.BaseRaw,
    exclude_components: list[str] = ["eye blink","muscle artifact"],
    bandpass: tuple[float, float] = (1.0, 100.0),
    duration: float | tuple[float, float] | None = None,
    n_jobs: int = 1,
) -> tuple[mne.io.BaseRaw, ICA]:
    """Remove artefact ICA components identified by ICLabel (DNN-based).

    duration controls how much of the recording is used to fit the ICA:
      - None: use the full recording.
      - (tmin, tmax): crop to that explicit window.
      - float: crop a centred window of that length around the recording midpoint.

    The input raw is never mutated; a cleaned copy is returned together with the
    fitted ICA object for inspection.
    """

    r_ic = raw.copy().filter(*bandpass, n_jobs=n_jobs)

    if duration is not None:
        if isinstance(duration, tuple):
            tmin, tmax = duration
        else:
            # Centre the window around the recording midpoint.
            midpoint = (raw.times[0] + raw.times[-1]) / 2.0
            tmin = midpoint - duration / 2.0
            tmax = midpoint + duration / 2.0
        r_ic = r_ic.crop(tmin=tmin, tmax=tmax)

    ica = ICA(
        n_components=None,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=97,
        max_iter="auto",
    )
    ica.fit(r_ic)
    labels = label_components(r_ic, ica, method="iclabel")
    ica.exclude = [i for i, label in enumerate(labels["labels"]) if label in exclude_components]

    cleaned = raw.copy()
    ica.apply(cleaned)
    return cleaned, ica


def ica_clean_regression(
    raw: mne.io.BaseRaw,
    bandpass: tuple[float, float] = (1.0, 40.0),
    decim: int = 1,
    components: int = 20,
    tmin: float = 0.0,
    tmax: float | None = None,
) -> tuple[mne.io.BaseRaw, ICA]:
    """Remove EOG artefact components via regression-based ICA.

    EOG channels are detected automatically from raw.info. The search window
    for find_bads_eog is capped at 1000 s from tmin to keep it tractable on
    long recordings. The input raw is never mutated.
    """
    r_ic = raw.copy().crop(tmin=tmin, tmax=tmax).filter(*bandpass)
    ica = ICA(n_components=components).fit(r_ic, decim=decim)

    eog_indices = mne.pick_types(raw.info, meg=False, eeg=False, eog=True)
    eog_channels = [raw.ch_names[i] for i in eog_indices]

    # Limit the EOG correlation search to a 1000 s window for efficiency.
    # Consider replacing tmin+1000.0 with tmax
    search_start = max(tmin, raw.times[0])
    search_stop = min(tmin + 1000.0, raw.times[-1])

    bad_components: list[int] = []
    for ch_name in eog_channels:
        found, _ = ica.find_bads_eog(raw, ch_name=ch_name, start=search_start, stop=search_stop)
        bad_components.extend(found)
    ica.exclude = list(set(bad_components))

    cleaned = raw.copy()
    ica.apply(cleaned)
    return cleaned, ica
