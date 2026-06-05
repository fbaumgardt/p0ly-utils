# p0ly-utils

EEG utilities library (`p0ly_utils`) for preprocessing, epoching, and metadata parsing.

![CI](https://github.com/fbaumgardt/p0ly-utils/actions/workflows/ci.yml/badge.svg?branch=main)

This library provides functionality that supports EEG analysis pipeline development. In particular it is used in p0ly-eeg -- a set of analysis workflows used for research at Boston University's Reinhart Lab for Computational Cognitive Neuroscience.

Part of the library is a module (`metadata`) that parses EEG event markers into structured metadata for detailed trial characterization. The abstract layer `core` and `parser` is instantiated with concrete experiment specs in the remaining files in the module. The experiments were chosen to showcase a broad range of different block- and trial-structures, and different levels of verbosity in event code definition (e.g. block and trial boundaries coded or implicit).

The `preprocessing` overlaps with a number of different Python packages that implement similar functionality, and perhaps more robustly (pyprep, autoreject, mne-faster). The purpose of maintaining our in-house solutions is the flexibility to handle edge cases of data corruption (e.g. from stimulation) and target highly specific signal-to-noise compromises (e.g. for very small or very large sample sizes). If you don't have specific technical requirements, please refer to these published and peer-reviewed packages.



## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Editable install for development:

```bash
uv sync --group dev
```

## Usage

```python
from p0ly_utils.metadata import dotprobe, parse_metadata

# evt: MNE events array; ids: Stim/ code -> numeric id map
df = dotprobe.get_metadata(evt, ids)
# or: parse_metadata(dotprobe.spec, evt, ids)
```

Experiment specs live under `p0ly_utils.metadata` (`dotprobe`, `mgsearch`, `dmss`, etc.), each exposing `spec`, `timelocks`, `intervals`, and `get_metadata`.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run mypy
uv run pre-commit install   # optional local hooks before commit
```

## CI

GitHub Actions on `main` runs ruff, scoped mypy (migrated metadata modules + tests), and pytest with coverage. Enable Actions in the repo settings on first push.

## See also

- [docs/DEFERRED_CI_AUTOMATION.md](docs/DEFERRED_CI_AUTOMATION.md) — automation to enable after legacy metadata and `dump_*` code are removed
