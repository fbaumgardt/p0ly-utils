# p0ly-utils

EEG utilities library (`p0ly_utils`) for preprocessing, epoching, and metadata parsing.

![CI](https://github.com/fbaumgardt/p0ly-utils/actions/workflows/ci.yml/badge.svg?branch=main)


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
