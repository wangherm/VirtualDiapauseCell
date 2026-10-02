from __future__ import annotations
import numpy as np
from .io import finite_or_none


def regression_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    pred = np.asarray(pred); target = np.asarray(target); mask = np.asarray(mask, dtype=bool)
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Metric shapes differ")
    if not mask.any():
        return {"status": "unavailable", "n_observed": 0, "mse": None, "mae": None, "r2": None}
    if not np.isfinite(pred[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Non-finite evaluated values")
    y, yhat = target[mask].astype(float), pred[mask].astype(float)
    err = yhat - y
    denom = np.square(y - y.mean()).sum()
    return {"status": "computed", "n_observed": int(mask.sum()),
            "mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err))),
            "r2": finite_or_none(1 - np.square(err).sum() / denom) if denom > 0 else None}


def grouped_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, rows: list[dict]) -> dict:
    per = {}
    for study in sorted({r["study_family"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["study_family"] == study]
        per[study] = regression_metrics(pred[idx], target[idx], mask[idx])
        per[study]["biological_units"] = len({rows[i]["biological_unit"] for i in idx})
    mses = [v["mse"] for v in per.values() if v["mse"] is not None]
    return {"by_study": per, "macro_study_mse": float(np.mean(mses)) if mses else None,
            "pooled": regression_metrics(pred, target, mask)}


def study_unit_weights(rows: list[dict]) -> np.ndarray:
    """Equal study -> equal biological unit -> equal row weight, not equal cell weight."""
    studies = sorted({r["study_family"] for r in rows})
    weights = np.zeros(len(rows), dtype=np.float32)
    for s in studies:
        units = sorted({r["biological_unit"] for r in rows if r["study_family"] == s})
        for u in units:
            idx = [i for i, r in enumerate(rows) if r["study_family"] == s and r["biological_unit"] == u]
            weights[idx] = 1 / (len(studies) * len(units) * len(idx))
    return weights
