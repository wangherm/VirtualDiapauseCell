"""Closed-form source-only shrinkage on the frozen deposited RMA axis."""
import json
from pathlib import Path

import numpy as np


def initialize(data, config, parent):
    if parent is not None:
        raise ValueError("This one-step backend uses independent source fits")
    return {"step": 0, "features": config["features"]}


def step(state, data, config):
    if state["step"] != 0 or set(data) != {"x", "y"}:
        raise ValueError("Expected a single closed-form fit on source x and y")
    x, y = data["x"], data["y"]
    shape = (4, config["features"])
    if x.shape != shape or y.shape != shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Invalid four-source cohort means")
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    dx, dy = x - mean_x, y - mean_y
    numerator = float(np.sum(dx * dy))
    denominator = float(np.sum(dx * dx))
    raw = numerator / denominator if denominator else 0.0
    return {"step": 1, "features": config["features"], "mean_x": mean_x, "mean_y": mean_y,
            "alpha": float(np.clip(raw, 0.0, 1.0)), "alpha_unclipped": raw,
            "numerator": numerator, "denominator": denominator}


def save(state, directory):
    directory = Path(directory)
    np.savez_compressed(directory / "parameters.npz", mean_x=state["mean_x"], mean_y=state["mean_y"])
    fields = {key: value for key, value in state.items() if key not in ("mean_x", "mean_y")}
    fields.update(optimizer="closed_form_one_step", rng=None)
    (directory / "state.json").write_text(json.dumps(fields, sort_keys=True, allow_nan=False), encoding="utf-8")


def load(directory):
    directory = Path(directory)
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    with np.load(directory / "parameters.npz", allow_pickle=False) as values:
        state.update(mean_x=values["mean_x"], mean_y=values["mean_y"])
    return state


def predict(state, data, config):
    x = data["x"]
    if (set(data) != {"x"} or x.ndim != 2 or x.shape[1] != state["features"] or not len(x)
            or state["features"] != config["features"] or not np.isfinite(x).all() or state["step"] != 1):
        raise ValueError("Expected finite FR requests on the fitted native axis")
    return state["mean_y"] + state["alpha"] * (x - state["mean_x"])
