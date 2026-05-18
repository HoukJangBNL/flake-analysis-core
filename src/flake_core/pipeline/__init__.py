"""Function-style pipeline wrappers (no Operation class)."""
from flake_core.pipeline.background import run_background
from flake_core.pipeline.clustering import run_clustering
from flake_core.pipeline.domain_proximity import run_domain_proximity
from flake_core.pipeline.domain_stats import run_domain_stats
from flake_core.pipeline.selector import run_selector

__all__ = [
    "run_background",
    "run_clustering",
    "run_domain_proximity",
    "run_domain_stats",
    "run_selector",
]
