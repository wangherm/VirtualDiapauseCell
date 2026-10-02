"""Explicit expression-to-programme scoring; no silent ID or scale conversion."""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata


def normalise_expression(x: np.ndarray, mask: np.ndarray, scale: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if x.ndim != 2 or x.shape != mask.shape or not np.isfinite(x[mask]).all():
        raise ValueError("Invalid expression/mask")
    if scale not in {"counts", "estimated_counts", "log_expression", "linear_abundance"}:
        raise ValueError("Declare a supported expression scale explicitly")
    if scale in {"counts", "estimated_counts"}:
        if np.any(x[mask] < 0) or (scale == "counts" and not np.allclose(x[mask], np.round(x[mask]))):
            raise ValueError("counts require nonnegative integers; do not infer counts from logs")
        total = np.where(mask, x, 0).sum(1, keepdims=True)
        if (total <= 0).any():
            raise ValueError("Zero-library sample")
        y = np.log1p(np.where(mask, x, 0) / total * 10000)
    elif scale == "linear_abundance":
        if np.any(x[mask] < 0):
            raise ValueError("linear abundance must be nonnegative")
        y = np.log1p(np.where(mask, x, 0))
    else:
        y = x.copy()
    return np.where(mask, y, 0)


def score_programmes(x: np.ndarray, mask: np.ndarray, genes: list[str], programmes: list[dict],
                     min_genes: int = 3, min_coverage: float = 0.5) -> dict:
    """Mean centred within-sample rank, optionally signed. Not a pathway-activity assay.

    programmes: [{'id': '...', 'members': {'gene': +1.0, ...}, 'source_ref': '...'}].
    Membership/identifier mapping is supplied, never guessed here.
    """
    x = np.asarray(x, dtype=float); mask = np.asarray(mask, dtype=bool)
    if x.ndim != 2 or x.shape != mask.shape or len(genes) != x.shape[1] or len(set(genes)) != len(genes):
        raise ValueError("Invalid or duplicate gene IDs / shapes")
    if not np.isfinite(x[mask]).all() or min_genes < 1 or not 0 <= min_coverage <= 1:
        raise ValueError("Invalid observation or scoring threshold")
    index = {g: j for j, g in enumerate(genes)}
    ranked = np.zeros_like(x)
    for i in range(x.shape[0]):
        pos = np.flatnonzero(mask[i])
        if len(pos) < 2:
            continue
        ranked[i, pos] = (rankdata(x[i, pos], method="average") - 1) / (len(pos) - 1) - 0.5
    pids = [p["id"] for p in programmes]
    if len(set(pids)) != len(pids):
        raise ValueError("Duplicate programme IDs")
    values = np.zeros((len(x), len(programmes)))
    coverage = np.zeros_like(values); available = np.zeros_like(values, dtype=bool)
    for k, p in enumerate(programmes):
        members = p["members"]
        if not members or not p.get("source_ref"):
            raise ValueError("Membership and source_ref are required")
        if any(not np.isfinite(v) or v == 0 for v in members.values()):
            raise ValueError("Member weights must be finite and nonzero")
        mapped = [(index[g], float(w)) for g, w in members.items() if g in index]
        for i in range(len(x)):
            used = [(j, w) for j, w in mapped if mask[i, j]]
            coverage[i, k] = len(used) / len(members)
            if len(used) >= min_genes and coverage[i, k] >= min_coverage:
                values[i, k] = sum(ranked[i, j] * w for j, w in used) / sum(abs(w) for _, w in used)
                available[i, k] = True
    return {"values": values.astype(np.float32), "mask": available,
            "coverage": coverage.astype(np.float32), "feature_ids": pids,
            "definition": "centred_within_sample_rank_v1_not_pathway_activity"}
