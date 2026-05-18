"""Function-style pipeline wrappers (no Operation class)."""
from flake_core.pipeline.background import run_background
from flake_core.pipeline.domain_proximity import run_domain_proximity

__all__ = ["run_background", "run_domain_proximity"]
