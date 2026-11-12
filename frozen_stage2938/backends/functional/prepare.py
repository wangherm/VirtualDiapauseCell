"""Fit one physically isolated fold using source means and the frozen schema."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np

from chilling_functional_model import canonical_ids, design, panel, sigmoid


OUT = Path("/out")


def pin(b):
    return {"bytes": len(b), "sha256": sha256(b).hexdigest()}


def write(name, value):
    b = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    (OUT / name).write_bytes(b)
    return pin(b)


def fractional_fit(x, y):
    n, p = x.shape
    interior = (y > 0.0) & (y < 1.0)
    interior_rank = int(np.linalg.matrix_rank(x[interior])) if np.any(interior) else 0
    if interior_rank != p:
        return None, {"status": "unavailable", "reason": "Interior rank guard not met",
                      "interior_rows": int(interior.sum()), "interior_rank": interior_rank}

    def objective(b):
        eta = x @ b
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta))

    b = np.zeros(p, dtype=np.float64)
    tolerance = 1e-10
    for iteration in range(201):
        pred = sigmoid(x @ b)
        grad = x.T @ (pred - y) / n
        grad_inf = float(np.linalg.norm(grad, ord=np.inf))
        if grad_inf <= tolerance:
            return b.tolist(), {"status": "fitted", "iterations": iteration,
                                "gradient_inf_norm": grad_inf, "tolerance": tolerance,
                                "objective": objective(b), "interior_rows": int(interior.sum()),
                                "interior_rank": interior_rank}
        assert iteration < 200, "Fractional optimizer failed stationarity"
        wt = pred * (1.0 - pred)
        hess = (x.T * wt) @ x / n
        direction = np.linalg.solve(hess, grad)
        decrease = float(grad @ direction)
        assert np.isfinite(decrease) and decrease > 0
        loss = objective(b)
        step = 1.0
        for _ in range(64):
            trial = b - step * direction
            trial_loss = objective(trial)
            if trial_loss <= loss - 1e-4 * step * decrease:
                break
            if trial_loss <= loss + 8 * np.finfo(float).eps and decrease < 1e-14:
                trial_norm = np.linalg.norm(x.T @ (sigmoid(x @ trial) - y) / n, ord=np.inf)
                if trial_norm < grad_inf:
                    break
            step *= 0.5
        else:
            raise ValueError("Fractional optimizer line search failed")
        b = trial
    raise AssertionError("Unreachable optimizer state")


config_raw = Path("/code/fit_config.json").read_bytes()
cfg = json.loads(config_raw)
code = Path("/code")
assert pin((code / "chilling_functional_model.py").read_bytes()) == cfg["predictor_code_pin"]
mounts = json.loads(Path("/input/artifacts.json").read_bytes())
by_ref = {tuple(sorted(v["ref"].items())): Path(v["container_path"]) for v in mounts.values()}
assert len(by_ref) == len(mounts) == 2 and set(cfg["inputs"]) == {"schema", "source"}
assert set(by_ref) == {tuple(sorted(v["ref"].items())) for v in cfg["inputs"].values()}
loaded = {}
for key, spec in cfg["inputs"].items():
    raw = by_ref[tuple(sorted(spec["ref"].items()))].read_bytes()
    assert pin(raw) == spec["pin"]
    loaded[key] = json.loads(raw)
schema, packet = loaded["schema"], loaded["source"]
assert schema["protocol_id"] == packet["protocol_id"] == cfg["protocol_id"]
assert schema["protocol_pin"] == cfg["protocol_pin"]
assert packet["schema_pin"] == cfg["inputs"]["schema"]["pin"]
assert packet["role"] == "source" and packet["fold"] == cfg["fold"]
assert set(packet) == {"protocol_id", "schema_pin", "fold", "role", "rows"}
source_rows = packet["rows"]
assert [r["row_id"] for r in source_rows] == canonical_ids(schema, cfg["fold"], False)
assert len(source_rows) == 18 and len(canonical_ids(schema, cfg["fold"], True)) == 6
assert all(type(r["germinated_fraction"]) in (int, float) for r in source_rows)
x = design(schema, source_rows, source=True)
y = np.asarray([r["germinated_fraction"] for r in source_rows], dtype=np.float64)
assert x.shape == (18, 9) and y.shape == (18,)
assert np.isfinite(y).all() and ((0.0 <= y) & (y <= 1.0)).all()
ols, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
assert rank == 9 and np.isfinite(ols).all()
frac, solver = fractional_fit(x, y)
lines, arms = panel(schema)


def med(field, levels):
    return {level: float(np.median([r["germinated_fraction"] for r in source_rows if r[field] == level]))
            for level in levels}


model = {
    "model_version": "chilling-germination-calibration-candidate.v1",
    "protocol_id": cfg["protocol_id"], "protocol_pin": cfg["protocol_pin"],
    "fold": cfg["fold"], "schema_pin": cfg["inputs"]["schema"]["pin"],
    "source_pin": cfg["inputs"]["source"]["pin"], "predictor_code_pin": cfg["predictor_code_pin"],
    "feature_columns": schema["feature_design"]["columns"],
    "fractional_logit_coefficients": frac, "two_way_ols_coefficients": ols.tolist(),
    "condition_medians": med("condition_id", arms), "line_medians": med("genotype", lines),
    "grand_median": float(np.median(y)),
    "scope": "Known line and treatment arm, this 2014 experiment only, source-cell calibration required.",
    "target": schema["target"], "method": cfg["method"], "scientific_approval": False,
}
mpin = write("functional_model.json", model)
write("fit_receipt.json", {
    "status": "completed", "fold": cfg["fold"], "protocol_id": cfg["protocol_id"],
    "protocol_pin": cfg["protocol_pin"], "config": pin(config_raw), "mounted_inputs": cfg["inputs"],
    "model": mpin, "source_rows": 18, "additive_rank": int(rank), "solver": solver,
    "source_cell_weights": "equal", "source_mean_endpoints_0_or_1": int(((y == 0) | (y == 1)).sum()),
    "query_labels_mounted": 0, "query_features_mounted": 0, "prediction_calls": 0,
    "score_calls": 0, "new_backgrounds": 0, "scientific_approval": False,
})
write("metadata.json", {"status": "fit_executed", "fold": cfg["fold"],
                        "model": "functional_model.json", "receipt": "fit_receipt.json",
                        "registered_engine_task": False, "task_ready": False})
