"""Thin wrapper for manual seed-group GMM clustering.

Wraps ``flake_core.clustering.engine.InteractiveClusteringEngine`` with file
I/O — loads the per-domain stats NPZ + selector parquet, narrows to the
selected domains, fits a seed-initialized GMM, and persists the result as:

  * ``labels.json``        — top-level metadata (counts, n_clusters, params)
  * ``assignments.parquet`` — per-domain cluster assignment + max posterior
  * ``gmm_model.pkl``       — pickled ``InteractiveClusterResult``
                              (means, covariances, weights, thresholds)

Plan v1 r6 D6.1: ``random_state=42`` is hard-baked in
``InteractiveClusteringEngine.fit``.

Plan v1 r6 D6.2 (positional-index adapter):
    The caller passes ``seed_groups`` keyed by ``domain_id`` (the
    user-facing identifier). The engine, however, indexes ``repr_rgbs`` by
    *positional* row index in the selector-narrowed array. This wrapper
    converts ``domain_ids`` -> positional indices before calling
    ``engine.fit``. Domain ids missing from the selected subset are
    skipped with a warning rather than aborting the run.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pandas as pd

from flake_core._compat import msg
from flake_core.clustering.engine import (
    InteractiveClusterResult,
    InteractiveClusteringEngine,
)


def _hash_params(params: Dict[str, Any]) -> str:
    payload = json.dumps(
        params, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _build_positional_seed_groups(
    seed_groups: Sequence[Dict[str, Any]],
    selected_domain_ids: np.ndarray,
) -> List[Dict[str, Any]]:
    """Convert domain_id-based seed groups to positional indices.

    ``selected_domain_ids`` is the array (already filtered to ``selected==True``)
    in the same order as the rows fed to the engine. We map each
    ``domain_id`` -> its positional row.
    """
    id_to_pos = {int(did): int(pos) for pos, did in enumerate(selected_domain_ids)}

    out: List[Dict[str, Any]] = []
    for grp in seed_groups:
        name = grp.get("name", f"group_{len(out)}")
        domain_ids = grp.get("domain_ids", grp.get("indices", []))
        positions: List[int] = []
        missing: List[int] = []
        for did in domain_ids:
            pos = id_to_pos.get(int(did))
            if pos is None:
                missing.append(int(did))
            else:
                positions.append(pos)
        if missing:
            msg.warning(
                f"[pipeline.clustering] seed group '{name}': "
                f"{len(missing)} domain_id(s) not in selected subset, skipping"
            )
        if not positions:
            raise ValueError(
                f"seed group '{name}' has no valid domain_ids in the "
                f"selected subset"
            )
        out.append({"name": name, "indices": positions, "domain_ids": list(domain_ids)})
    return out


def run_clustering(
    stats_npz_path: Union[str, Path],
    selection_parquet_path: Union[str, Path],
    seed_groups: Sequence[Dict[str, Any]],
    *,
    output_dir: Union[str, Path],
    rgb_threshold: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """Fit GMM with manual seed groups and persist labels + model.

    Parameters
    ----------
    stats_npz_path : str | Path
        NPZ produced by ``run_domain_stats`` (must contain
        ``repr_rgbs`` and ``flake_ids``).
    selection_parquet_path : str | Path
        Parquet produced by ``run_selector`` (columns:
        ``domain_id``, ``selected``).
    seed_groups : sequence of dict
        Each entry must have:
          * ``name`` (str, optional — auto-named when absent),
          * ``domain_ids`` (list[int]) — domain ids assigned by user.
        ``indices`` is accepted as an alias for ``domain_ids`` for
        backward compatibility with the engine's pre-extraction API.
    output_dir : str | Path
        Directory to receive ``labels.json``, ``assignments.parquet``,
        ``gmm_model.pkl``.
    rgb_threshold : float, optional
        Posterior probability cutoff for Filter 1, broadcast to all
        clusters. Default ``0.5``.
    max_iter, tol : float
        EM hyperparameters forwarded to ``GaussianMixture``.

    Returns
    -------
    dict
        Summary including output paths, ``n_clusters``, ``n_assigned``,
        ``n_unassigned``, ``params``, ``params_hash``.
    """
    stats_npz_path = Path(stats_npz_path)
    selection_parquet_path = Path(selection_parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not seed_groups:
        raise ValueError("seed_groups must be non-empty")

    msg.info(
        f"[pipeline.clustering] start npz={stats_npz_path} "
        f"selection={selection_parquet_path} n_groups={len(seed_groups)} "
        f"random_state=42"
    )

    # --- Load stats + selection ------------------------------------------
    npz = np.load(stats_npz_path, allow_pickle=False)
    if "repr_rgbs" not in npz.files or "flake_ids" not in npz.files:
        raise KeyError(
            f"stats NPZ missing required keys (have: {npz.files})"
        )

    repr_rgbs_all = npz["repr_rgbs"]
    flake_ids_all = npz["flake_ids"].astype(np.int64)

    selection_df = pd.read_parquet(selection_parquet_path)
    if not {"domain_id", "selected"}.issubset(selection_df.columns):
        raise KeyError(
            f"selection parquet missing required columns "
            f"(have: {list(selection_df.columns)})"
        )

    # Build domain_id -> position-in-NPZ map, then narrow to selected.
    id_to_npz_pos = {int(did): pos for pos, did in enumerate(flake_ids_all)}
    selected_mask = selection_df["selected"].astype(bool).to_numpy()
    selected_domain_ids = selection_df.loc[selected_mask, "domain_id"].astype(int).to_numpy()

    if selected_domain_ids.size == 0:
        raise ValueError("selection parquet contains zero selected domains")

    npz_positions = np.array(
        [id_to_npz_pos[int(did)] for did in selected_domain_ids if int(did) in id_to_npz_pos],
        dtype=np.int64,
    )
    if npz_positions.size != selected_domain_ids.size:
        # Some selected domain_ids were not present in the stats NPZ.
        # Drop them and keep the bookkeeping consistent.
        kept = np.array(
            [int(did) in id_to_npz_pos for did in selected_domain_ids], dtype=bool
        )
        msg.warning(
            f"[pipeline.clustering] {(~kept).sum()} selected domain_id(s) "
            f"missing from stats NPZ; dropping"
        )
        selected_domain_ids = selected_domain_ids[kept]

    repr_rgbs_sel = repr_rgbs_all[npz_positions]
    msg.info(
        f"[pipeline.clustering] selected subset: {repr_rgbs_sel.shape[0]} domains"
    )

    # --- Positional-index adapter (D6.2) ---------------------------------
    positional_groups = _build_positional_seed_groups(seed_groups, selected_domain_ids)
    seed_indices_only = [grp["indices"] for grp in positional_groups]

    # --- Fit GMM ---------------------------------------------------------
    engine = InteractiveClusteringEngine()
    result: InteractiveClusterResult = engine.fit(
        repr_rgbs_sel,
        seed_indices_only,
        rgb_threshold=rgb_threshold,
        max_iter=max_iter,
        tol=tol,
    )

    # --- Persist outputs -------------------------------------------------
    assignments_df = pd.DataFrame(
        {
            "domain_id": selected_domain_ids.astype(np.int64),
            "cluster_label": result.labels.astype(np.int64),
            "max_posterior": result.probabilities.astype(np.float64),
        }
    )
    assignments_path = output_dir / "assignments.parquet"
    assignments_df.to_parquet(assignments_path, engine="pyarrow", index=False)

    n_assigned = int((result.labels >= 0).sum())
    n_unassigned = int((result.labels == -1).sum())

    labels_payload = {
        "n_clusters": int(result.n_clusters),
        "n_selected": int(repr_rgbs_sel.shape[0]),
        "n_assigned": n_assigned,
        "n_unassigned": n_unassigned,
        "rgb_threshold": float(rgb_threshold),
        "thresholds": list(result.thresholds) if result.thresholds is not None else None,
        "cluster_centers": result.cluster_centers.tolist(),
        "groups": [
            {"name": grp["name"], "domain_ids": grp["domain_ids"]}
            for grp in positional_groups
        ],
    }
    labels_path = output_dir / "labels.json"
    labels_path.write_text(json.dumps(labels_payload, indent=2))

    gmm_path = output_dir / "gmm_model.pkl"
    with open(gmm_path, "wb") as f:
        pickle.dump(result, f)

    msg.info(
        f"[pipeline.clustering] wrote labels={labels_path.name} "
        f"assignments={assignments_path.name} gmm={gmm_path.name} "
        f"(assigned={n_assigned}, unassigned={n_unassigned})"
    )

    params: Dict[str, Any] = {
        "stats_npz_path": str(stats_npz_path),
        "selection_parquet_path": str(selection_parquet_path),
        "n_groups": len(positional_groups),
        "rgb_threshold": rgb_threshold,
        "max_iter": max_iter,
        "tol": tol,
        "random_state": 42,
    }
    return {
        "labels_path": labels_path,
        "assignments_path": assignments_path,
        "gmm_model_path": gmm_path,
        "n_clusters": int(result.n_clusters),
        "n_assigned": n_assigned,
        "n_unassigned": n_unassigned,
        "params": params,
        "params_hash": _hash_params(params),
    }
