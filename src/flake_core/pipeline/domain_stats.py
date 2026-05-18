"""Thin wrapper for per-domain color stats compute.

Replaces Qpress's ``flake_stats_operation.py`` logic without DB / Operation /
Context. Loads annotations + raw images (+ optional pre-computed background)
and writes a single ``stats.npz`` artifact whose schema matches Qpress's
``flake_stats_<bg_mode>_<repr_mode>.npz`` exactly:

  * ``repr_rgbs``  shape (N, 3) float64
  * ``std_pcts``   shape (N, 3) float64
  * ``areas``      shape (N,)   int32
  * ``flake_ids``  shape (N,)   int64

Output is reordered to match the input flakes list (deterministic for a
given annotations + raw image set).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from flake_core._compat import msg
from flake_core.annotations import AnnotationsCache, load_flakes_from_annotations
from flake_core.color_classification.loader import compute_and_cache_stats_from_flakes


def _hash_params(params: Dict[str, Any]) -> str:
    """SHA-256 of canonical JSON for the params dict."""
    payload = json.dumps(
        params, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def run_domain_stats(
    annotations_path: Union[str, Path],
    raw_images_dir: Union[str, Path],
    *,
    output_path: Union[str, Path],
    background_path: Optional[Union[str, Path]] = None,
    bg_mode: str = "median",
    repr_mode: str = "median",
    raw_ext: str = ".png",
) -> Dict[str, Any]:
    """Compute per-domain color stats and write a single NPZ.

    Parameters
    ----------
    annotations_path : str | Path
        Path to ``annotations.json`` (COCO-style). Either a file path or the
        directory containing the file is accepted (pursues the file inside
        ``segmentation/`` when a parent dir is given).
    raw_images_dir : str | Path
        Directory containing raw PNGs whose stems match
        ``annotations.images[].file_name`` stems.
    output_path : str | Path
        Destination NPZ file. Parent directory is created if missing.
    background_path : str | Path | None, optional
        Pre-computed background image (PNG or NPY) to use for vignetting
        correction. If provided, ``bg_mode`` is forced to ``"median"`` and
        the supplied image overrides on-disk lookups.
    bg_mode : str, optional
        ``"median"`` (vignetting correction) or ``"raw"`` (no correction).
        Default ``"median"``.
    repr_mode : str, optional
        ``"median"`` (default, robust) or ``"mean"``.
    raw_ext : str, optional
        File extension of raw images. Default ``".png"``.

    Returns
    -------
    dict
        Summary including ``output_path``, ``num_flakes``, ``params``,
        ``params_hash``.
    """
    if bg_mode not in ("raw", "median"):
        raise ValueError(f"bg_mode must be 'raw' or 'median', got {bg_mode!r}")
    if repr_mode not in ("median", "mean"):
        raise ValueError(f"repr_mode must be 'median' or 'mean', got {repr_mode!r}")

    annotations_path = Path(annotations_path)
    raw_images_dir = Path(raw_images_dir)
    output_path = Path(output_path)

    msg.info(
        f"[pipeline.domain_stats] start ann={annotations_path} "
        f"raw={raw_images_dir} out={output_path} "
        f"bg_mode={bg_mode} repr_mode={repr_mode}"
    )

    # --- Resolve annotations layout --------------------------------------
    # AnnotationsCache.load() expects:
    #   annotations.json at: {analysis_dir}/{analysis_type}/annotations.json
    # We accept either a direct file path or any of its ancestors and
    # reconstruct the (analysis_dir, analysis_type) split.
    if annotations_path.is_file():
        ann_file = annotations_path
    elif annotations_path.is_dir() and (annotations_path / "annotations.json").exists():
        ann_file = annotations_path / "annotations.json"
    else:
        raise FileNotFoundError(f"annotations.json not found at {annotations_path}")

    analysis_type_dir = ann_file.parent
    analysis_type = analysis_type_dir.name or "segmentation"
    analysis_dir = analysis_type_dir.parent
    scan_folder = analysis_dir.parent if analysis_dir != analysis_type_dir else analysis_dir

    cache = AnnotationsCache()
    loaded = cache.load(
        scan_folder=scan_folder,
        analysis_dir=analysis_dir,
        analysis_type=analysis_type,
    )
    if not loaded:
        raise RuntimeError(
            f"Failed to load annotations from {ann_file} "
            f"(analysis_dir={analysis_dir}, analysis_type={analysis_type})"
        )

    flakes = load_flakes_from_annotations(cache, raw_images_dir, raw_ext=raw_ext)
    msg.info(f"[pipeline.domain_stats] loaded {len(flakes)} flakes")

    # --- Optional explicit background image ------------------------------
    background_image: Optional[np.ndarray] = None
    if background_path is not None:
        bg_path = Path(background_path)
        if not bg_path.exists():
            raise FileNotFoundError(f"background_path not found: {bg_path}")
        if bg_path.suffix.lower() == ".npy":
            background_image = np.load(bg_path).astype(np.float64)
        else:
            background_image = np.array(Image.open(bg_path)).astype(np.float64)
        msg.info(f"[pipeline.domain_stats] using background from {bg_path}")
        # Per Qpress flake_stats_operation.py: explicit background forces
        # bg_mode=median (so the per-pixel division branch runs).
        bg_mode = "median"

    # --- Compute (writes Qpress-compatible NPZ inside cache_dir) ---------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path.parent

    raw_image_folder = raw_images_dir if bg_mode == "median" and background_image is None else raw_images_dir
    result = compute_and_cache_stats_from_flakes(
        flakes=flakes,
        cache_dir=cache_dir,
        raw_image_folder=raw_image_folder,
        background_mode=bg_mode,
        representative_mode=repr_mode,
        force_recompute=True,  # wrapper always writes fresh artifact
        raw_ext=raw_ext,
        background_image=background_image,
    )

    # ``compute_and_cache_stats_from_flakes`` writes
    #   cache_dir/flake_stats_<bg>_<repr>.npz
    # which may not match the user's requested output_path. Resolve by
    # writing the canonical artifact at output_path explicitly so the
    # caller-specified filename is honored.
    flake_ids = np.array([f.flake_id for f in flakes], dtype=np.int64)
    np.savez(
        output_path,
        repr_rgbs=result["repr_rgbs"],
        std_pcts=result["std_pcts"],
        areas=result["areas"],
        flake_ids=flake_ids,
    )
    msg.info(
        f"[pipeline.domain_stats] wrote {len(flake_ids)} domain rows to {output_path}"
    )

    params: Dict[str, Any] = {
        "annotations_path": str(annotations_path),
        "raw_images_dir": str(raw_images_dir),
        "background_path": str(background_path) if background_path else None,
        "bg_mode": bg_mode,
        "repr_mode": repr_mode,
        "raw_ext": raw_ext,
    }
    return {
        "output_path": output_path,
        "num_flakes": int(len(flake_ids)),
        "params": params,
        "params_hash": _hash_params(params),
    }
