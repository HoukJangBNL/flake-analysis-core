# flake-analysis-core

Shared algorithm core for 2D material flake microscopy analysis, extracted from
the [Qpress](https://github.com/HoukJangBNL) distributed hardware control system.

## What it is

A pip-installable Python package containing the algorithmic primitives used by
the flake analysis pipeline:

- Background generation (median / mean over raw images)
- Per-domain color and area statistics
- Pairwise domain distance + union-find flake construction
- 5-metric bidirectional selector
- Manual seed-group GMM clustering (with per-cluster probability thresholds)
- COCO + RLE annotation IO

The package is UI-free, DB-free, and ZMQ-free. It exposes thin functional
wrappers (`flake_core.pipeline.*`) that take plain arguments and return plain
dict outputs.

## Used by

- **Qpress** — the original analyzer module imports from `flake_core` after the
  M2 migration.
- **stand-alone-analyzer** — a Streamlit single-user desktop app that depends on
  `flake_core` and adds disk-SSOT manifest, Plotly visualization, and a
  6-step pipeline UI.

This package is **not intended for direct end-users**. Use one of the consumer
applications above.

## Install

```bash
# Development (editable) install
pip install -e .

# With dev tools (pytest, ruff)
pip install -e ".[dev]"

# Future: published install (not yet on PyPI)
pip install flake-analysis-core
```

## License

MIT — see [LICENSE](LICENSE).

## Status

**Alpha (M0 skeleton)** — version `0.1.0a0`. Algorithm modules are extracted in
M1; Qpress migration lands in M2; standalone Streamlit app and parity validation
follow in M3–M4.

See `plan_v1.md` (in the Qpress workspace) for the full roadmap.
