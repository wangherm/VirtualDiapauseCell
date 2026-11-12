"""Learn a regularized apricot response anchored to persistence."""
import json
from pathlib import Path

import numpy as np


def initialize(data, config, parent):
    if parent is not None:
        raise ValueError("Use explicit structural-refit lineage")
    return {"step": 0}


def validate_x(x, genes):
    if (x.ndim != 2 or x.shape[1] != 2 * genes + 4 or not np.isfinite(x).all()
            or not np.all(x[:, 2 * genes + 1:] == [100., 400., 800.])):
        raise ValueError("Expected native100/400 history and fixed 800h request")


def step(state, data, config):
    genes = config["features"]
    n = config["source_count"]
    if state["step"] != 0 or set(data) != {"x", "y", "unit_code"} or n not in (2, 3):
        raise ValueError("Wrong source step or arrays")
    x, y = data["x"], data["y"]
    validate_x(x, genes)
    if x.shape[0] != n or y.shape != (n, genes) or not np.isfinite(y).all() or not np.array_equal(data["unit_code"], np.arange(n)):
        raise ValueError("Wrong source cultivar counts")
    current = x[:, genes:2 * genes]
    early = current - x[:, :genes]
    late = y - current
    mc, me, ml = current.mean(axis=0), early.mean(axis=0), late.mean(axis=0)
    design = np.stack(((current - mc).ravel(), (early - me).ravel()), axis=1)
    response = (late - ml).ravel()
    # Standardized unit ridge penalty. Zero-variance components contribute zero.
    gram = design.T @ design
    cross = design.T @ response
    keep = np.diag(gram) > 0
    beta = np.zeros(2)
    if keep.any():
        kept = gram[np.ix_(keep, keep)]
        beta[keep] = np.linalg.solve(kept + np.diag(np.diag(kept)), cross[keep])
    beta_current = float(cross[0] / (2 * gram[0, 0])) if keep[0] else 0.
    if not np.isfinite(beta).all() or not np.isfinite(beta_current):
        raise ValueError("Nonfinite fit")
    return {"step": 1, "features": genes, "mean_current": mc, "mean_early": me, "mean_late": ml,
            "history_coef": beta, "current_coef": beta_current, "half_mean_late": 0.5,
            "active_covariates": int(keep.sum()), "source_count": n, "rng": None}


def save(state, directory):
    folder = Path(directory)
    keys = ("mean_current", "mean_early", "mean_late", "history_coef")
    np.savez_compressed(folder / "parameters.npz", **{k: state[k] for k in keys})
    (folder / "state.json").write_text(json.dumps({k: v for k, v in state.items() if k not in keys}, allow_nan=False), encoding="utf-8")


def load(directory):
    folder = Path(directory)
    state = json.loads((folder / "state.json").read_bytes())
    with np.load(folder / "parameters.npz", allow_pickle=False) as values:
        if set(values.files) != {"mean_current", "mean_early", "mean_late", "history_coef"}:
            raise ValueError("Unexpected saved arrays")
        state.update({k: values[k] for k in values.files})
    return state


def predict(state, data, config):
    genes = state["features"]
    if set(data) != {"x"} or state["step"] != 1 or genes != config["features"]:
        raise ValueError("Wrong request or state")
    x = data["x"]
    validate_x(x, genes)
    current = x[:, genes:2 * genes]
    early = current - x[:, :genes]
    dc, de = current - state["mean_current"], early - state["mean_early"]
    base = current + state["half_mean_late"] * state["mean_late"]
    rule = config.get("rule", "history")
    if rule == "history":
        a, b = state["history_coef"]
        return base + a * dc + b * de
    if rule == "current":
        return base + state["current_coef"] * dc
    if rule == "half_source_drift":
        return base
    raise ValueError("Unknown rule")
