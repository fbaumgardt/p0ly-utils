"""Raw EEG ingestion dispatcher.

Single entry point :func:`load_raw` selects a per-format reader based on the
``fmt`` argument. BrainVision recordings are read with
:func:`mne.io.read_raw_brainvision`; BIDS datasets are read with
:func:`mne_bids.read_raw_bids`. EOG channels are marked and the ``montage``
argument is applied only when the recording has no montage set.

This module is stateless and decoupled from Snakemake: it reads from the
filesystem and returns an :class:`mne.io.BaseRaw` but never writes derivative
files. Pipeline scripts are responsible for saving.
"""

from __future__ import annotations

from pathlib import Path

import mne
from mne_bids import BIDSPath, read_raw_bids

__all__ = ["load_raw"]


def _read_brainvision(
    data_dir: Path,
    subject: str,
    task: str,
    montage: str,
) -> mne.io.BaseRaw:
    """Read a BrainVision ``.vhdr`` (with ``.eeg``/``.vmrk`` siblings) and
    set channel types and montage (if missing).

    Expected layout: ``{data_dir}/sub-{subject}/sub-{subject}_task-{task}.vhdr``.
    """
    vhdr = data_dir / f"sub-{subject}" / f"sub-{subject}_task-{task}.vhdr"
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    raw.set_channel_types({k:'eog' for k in raw.ch_names if "EOG" in k.upper()})
    if raw.get_montage() is None:
        raw.set_montage(montage, on_missing="ignore")
    return raw


def _read_bids(
    data_dir: Path,
    subject: str,
    task: str,
    montage: str,
    datatype: str,
) -> mne.io.BaseRaw:
    """Read a BIDS EEG recording via :func:`mne_bids.read_raw_bids`.

    ``data_dir`` is the BIDS dataset root. ``montage`` is applied only when the
    sidecar did not provide one.
    """
    bids_path = BIDSPath(
        subject=subject,
        task=task,
        datatype=datatype,
        root=str(data_dir),
    )
    raw = read_raw_bids(bids_path=bids_path, verbose="ERROR")
    raw.set_channel_types({k:'eog' for k in raw.ch_names if "EOG" in k.upper()})
    if raw.get_montage() is None:
        raw.set_montage(montage, on_missing="ignore")
    return raw


def load_raw(
    fmt: str,
    subject: str,
    data_dir: str | Path,
    task: str,
    montage: str = "easycap-M1",
    datatype: str = "eeg",
) -> mne.io.BaseRaw:
    """Dispatch to a per-format raw reader and return a montage-set ``Raw``.

    Parameters
    ----------
    fmt
        Input format: ``"BrainVision"`` or ``"BIDS"``.
    subject
        Bare subject id, e.g. ``"001"`` (referenced as ``sub-{subject}`` in paths).
    data_dir
        Dataset root. For BrainVision this is the directory containing
        ``sub-{subject}/`` folders; for BIDS it is the BIDS dataset root.
    task
        Task label, embedded in BrainVision filenames and used as the BIDS
        ``task`` entity.
    montage
        MNE builtin montage name applied to the BrainVision recording (and to
        the BIDS recording only when the sidecar lacks one). Default
        ``"easycap-M1"``.
    datatype
        BIDS datatype entity (default ``"eeg"``). Ignored for BrainVision.

    Returns
    -------
    raw : mne.io.BaseRaw
        Raw object with montage set. Shape of ``raw.get_data()`` is
        ``(n_channels, n_times)`` (see SCHEMA §1).

    Raises
    ------
    ValueError
        If ``fmt`` is not a supported format.
    """
    data_dir = Path(data_dir)
    match fmt:
        case "BrainVision":
            return _read_brainvision(data_dir, subject, task, montage)
        case "BIDS":
            return _read_bids(data_dir, subject, task, montage, datatype)
        case _:
            raise ValueError(f"Unsupported input_format {fmt!r}. Expected 'BrainVision' or 'BIDS'.")
