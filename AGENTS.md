# Agent Directives: Data Analysis Pipeline

## 1. Core Philosophy & Abstraction Hierarchy

Your primary goal is to write elegant, performant, and highly abstract data pipelines. You must exhaust all built-in library methods before dropping down to lower levels of abstraction.

Follow this strict hierarchy of execution:
1. **MNE-Python Native:** Always use MNE's built-in methods (e.g., `raw.filter()`, `epochs.average()`, `mne.compute_covariance()`) before extracting raw arrays.
2. **Pandas Vectorized Operations:** For tabular event and metadata logic, compose pandas methods idiomatically — `.isin()`, `.cumsum()`, `.groupby().transform()`, boolean Series indexing. Pandas sits above NumPy because its labeled, higher-level operations should be preferred for structured data before dropping to raw arrays.
3. **NumPy Vectorization/Broadcasting:** If neither MNE nor Pandas provides the right abstraction, extract data via `get_data()` or `.to_numpy()` and use purely vectorized NumPy operations.
4. **Standard Python:** Use standard Python data structures (lists, dicts, for-loops) ONLY for file I/O, configuration management, or when libraries explicitly require it.

## 2. Negative Constraints (Strictly Banned Patterns)

* **No `for` loops over channels or epochs:** Never iterate over EEG channels or epochs manually.
* **No premature data extraction:** Do not call `get_data()` on MNE objects unless you are immediately passing the output to a vectorized NumPy function or scikit-learn pipeline. Keep data in MNE objects to preserve metadata for as long as possible.
* **No row-wise DataFrame iteration:** Never use `pandas.iterrows()`, or `pandas.apply()` with a Python lambda that operates row-by-row. These are loop-in-disguise patterns. Use vectorized Series operations or `groupby(...).transform(...)` instead.
* **Prefer inline composition for linear transforms:** For sequential transformations with no branching, compose operations inline (e.g. `df["x"].isin(codes).shift(1).cumsum()`). Named intermediates are acceptable when a value is reused, logic branches, or readability clearly benefits.

## 3. Translation Anchors (Few-Shot Examples)

Use the following table to calibrate your definition of elegant code.

| Library | Novice Approach (Banned) | Elegant Approach (Required) |
| :--- | :--- | :--- |
| **MNE** | Looping `epochs` to average: `[e.mean() for e in epochs]` | Native method: `epochs.average()` |
| **MNE** | Modifying channels via loop | Native mapping: `raw.rename_channels(mapping_dict)` |
| **MNE / NumPy** | Looping to apply a custom function | Vectorized injection: `raw.apply_function(np.log10)` |
| **NumPy** | Nested loops for matrix math | `np.einsum()` or native broadcasting |
| **NumPy** | Masking with loops and `if` statements | Boolean indexing: `data[data > threshold] = 0` |
| **Pandas** | Sequential mutations: `df['a'] = 1`, `df['b'] = 2` | Method chaining: `df.assign(a=1, b=2).query(...)` |
| **Pandas** | Loop to flag trial boundaries: `for i, row in df.iterrows(): ...` | Vectorized composition: `df["description"].isin(codes).cumsum()` |

## 4. Metadata Architecture

This project uses a declarative spec pattern for EEG metadata extraction. When working in `src/p0ly_utils/metadata/`:

* Each experiment has its own spec module (e.g. `metadata/dmss.py`, `metadata/dotprobe.py`) that defines an `ExperimentSpec` dataclass instance.
* Column extraction is entirely declarative: use the provided `ColumnExtractor` subclasses — `CodeLookup`, `IntSum`, `BoolPresence`, `ListCollect`, `DerivedColumn` — to describe what to extract from each trial group.
* The generic parser (`metadata/parser.py`) drives all experiments via `parse_metadata(spec, df)`. It is experiment-agnostic.
* **Do not modify parser logic for experiment-specific behavior.** New experiments require only a new spec file. If a new extraction pattern is needed, add a new `ColumnExtractor` subclass in `core.py`.

## 5. Testing Conventions

* Always run `uv run pytest` before considering any change complete.
* But only run time-consuming ICA-related tests after changes to `preprocessing.py`.
* `tests/data/` contains real-world examples that should be used to construct valid tests.

## 6. Type Annotations and Dataclass Conventions

* All modules use `from __future__ import annotations` at the top.
* Data containers are `@dataclass` classes with fully typed fields.
* Use `field(default_factory=list)` or `field(default_factory=dict)` for mutable defaults — never bare `[]` or `{}` as default values.
* Prefer `X | None` union syntax over `Optional[X]`.
