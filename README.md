# flake-analysis-core

Algorithm core for 2D material flake microscopy analysis.

Extracted from the [Qpress](https://github.com/HoukJangBNL/Qpress) hardware-control system into a standalone, reusable package. Used by [stand-alone-analyzer](https://github.com/HoukJangBNL/stand-alone-analyzer) (Streamlit GUI).

## Status

`v0.1.0` — alpha. APIs may change between minor versions.

## What it provides

- **Annotations IO** — COCO + RLE flake mask loading (`flake_core.annotations`)
- **Image processing** — median background generation (with reproducibility seed) + mask pair distance + union-find flake construction (`flake_core.image_processing`)
- **Color classification** — per-domain color stats computation (`flake_core.color_classification`)
- **Manual seed-group GMM clustering** (`flake_core.clustering`)
- **Function-style pipeline wrappers** — `flake_core.pipeline.{background, domain_stats, domain_proximity, selector, clustering}`

No DB, no SSH, no GUI dependencies. Pure Python + numpy + scipy + sklearn + opencv + pycocotools + Pillow.

## Install

```bash
git clone https://github.com/HoukJangBNL/flake-analysis-core.git
cd flake-analysis-core
pip install -e ".[dev]"
pytest -v
```

## Quick example

```python
from flake_core.pipeline.background import run_background
from flake_core.pipeline.domain_stats import run_domain_stats

# Generate median background from raw images (reproducible with seed)
result = run_background(
    raw_images_dir="/path/to/raw_images",
    output_path="/path/to/analysis/01_background/background.npy",
    seed=0,
    max_images=100,
)

# Compute per-domain RGB stats
stats = run_domain_stats(
    annotations_path="/path/to/annotations.json",
    raw_images_dir="/path/to/raw_images",
    background_path="/path/to/analysis/01_background/background.npy",
    analysis_folder="/path/to/analysis",
)
```

See [stand-alone-analyzer](https://github.com/HoukJangBNL/stand-alone-analyzer) for a full Streamlit GUI on top of these functions.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Algorithms adapted from the Qpress analyzer module (BNL/CFN). Credit to the Qpress contributors.
