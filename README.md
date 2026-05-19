# flake-analysis-core (DEPRECATED)

**This repository has been merged into [stand-alone-analyzer](https://github.com/HoukJangBNL/stand-alone-analyzer).**

The algorithm core that used to live here is now at
`stand-alone-analyzer/src/flake_analysis/core/` as of stand-alone-analyzer
v0.2.0 (2026-05-18).

## Why the merge?

Original split was based on plan v1 r1-r6's "(c) shared package" strategy
where Qpress and the standalone tool would both import this package.
Plan r8 reverted that decision ("(a) plain copy" — Qpress unchanged).
A separate package was no longer necessary.

## Migration

```python
# OLD
from flake_core.pipeline.background import run_background

# NEW
from flake_analysis.core.pipeline.background import run_background
```

## What if you depend on this package?

Install the new package:

```bash
git clone https://github.com/HoukJangBNL/stand-alone-analyzer
pip install -e stand-alone-analyzer
```

The last release of this standalone package was `v0.2.0`. No further
releases will be published from this repo.
