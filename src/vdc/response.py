"""Small response baselines with distinct endpoint, transition and functional contracts.

The interfaces accept programme vectors or frozen state vectors with a representation hash.
They do not invent paired cells, acute KO events, functional targets, or outcome expression.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .contracts import audit_rows
from .io import read_json, write_json, save_npz, sha256
from .metrics import study_unit_weights


def matched_contrast(control: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """One group-level contrast, not every control x treatment Cartesian pair."""
    control, treated = np.asarray(control, float), np.asarray(treated, float)
    if control.ndim != 2 or treated.ndim != 2 or control.shape[1] != treated.shape[1]:
        raise ValueError("Matched group matrices require the same feature order")
    if len(control) < 1 or len(treated) < 1 or not np.isfinite(control).all() or not np.isfinite(treated).all():
        raise ValueError("Empty or invalid group")
    return treated.mean(0) - control.mean(0)


@dataclass
class ResponseDataset:
    current: np.ndarray
    action: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    rows: list[dict]
    mode: str
    representation_id: str
    input_names: list[str]
    action_names: list[str]
    target_names: list[str]
    context_id: str
    elapsed: np.ndarray | None = None
    time_unit: str | None = None
    endpoint: str | None = None
    protocol_id: str | None = None
    split_policy: str = "unit_holdout"

    def validate(self) -> "ResponseDataset":
        self.current = np.asarray(self.current, float); self.action = np.asarray(self.action, float)
        self.target = np.asarray(self.target, float); self.target_mask = np.asarray(self.target_mask, bool)
        n = len(self.rows)
        if self.current.shape != (n, len(self.input_names)) or self.action.shape != (n, len(self.action_names)):
            raise ValueError("Response input shape/order mismatch")
        if self.target.shape != (n, len(self.target_names)) or self.target_mask.shape != self.target.shape:
            raise ValueError("Response target shape/order mismatch")
        if any(len(x) != len(set(x)) for x in (self.input_names, self.action_names, self.target_names)):
            raise ValueError("Duplicate feature names")
        if not np.isfinite(self.current).all() or not np.isfinite(self.action).all() or not np.isfinite(self.target[self.target_mask]).all():
            raise ValueError("Finite prepared response inputs and observed targets required")
        if self.mode not in {"endpoint", "transition", "functional"}:
            raise ValueError("Declare endpoint, transition or functional mode")
        if not self.representation_id or not self.context_id:
            raise ValueError("Input representation checksum and scope are required")
        audit_rows(self.rows, self.split_policy)
        for r in self.rows:
            if not r.get("control_or_initial_ids") or not r.get("outcome_ids"):
                raise ValueError("Control/initial and outcome source IDs are required")
            linked = set(r["link_ids"])
            if not set(r["control_or_initial_ids"] + r["outcome_ids"]).issubset(linked):
                raise ValueError("All reused controls/initial/outcome IDs must appear in global link_ids")
            if r.get("input_contains_future_information", False):
                raise ValueError("Future expression/clock/results are prohibited prediction inputs")
            if r.get("new_acute_ko", False) and r.get("same_genotype_already_in_initial", False):
                raise ValueError("Constitutive genotype cannot be reapplied as a new acute KO")
        if self.mode == "transition":
            self.elapsed = np.asarray(self.elapsed, float) if self.elapsed is not None else None
            if self.elapsed is None or self.elapsed.shape != (n,) or not np.isfinite(self.elapsed).all():
                raise ValueError("Timed transition requires observed elapsed times")
            if (self.elapsed < 0).any() or self.time_unit not in {"hours", "days", "minutes"}:
                raise ValueError("Known nonnegative elapsed time and explicit units required")
            if self.input_names != self.target_names:
                raise ValueError("Minimal transition baseline predicts the same state representation")
        if self.mode == "functional" and (not self.endpoint or not self.protocol_id):
            raise ValueError("A functional outcome requires its measured endpoint and protocol")
        return self

    @property
    def scope(self) -> dict:
        return {k: getattr(self, k) for k in ("mode", "representation_id", "input_names", "action_names",
                                             "target_names", "context_id", "time_unit", "endpoint", "protocol_id")}

    def save(self, directory: str | Path) -> None:
        self.validate(); p = Path(directory)
        arrays = {k: getattr(self, k) for k in ("current", "action", "target", "target_mask")}
        if self.elapsed is not None: arrays["elapsed"] = self.elapsed
        save_npz(p / "response.npz", **arrays)
        write_json(p / "response.json", {**self.scope, "rows": self.rows, "split_policy": self.split_policy,
                   "arrays_sha256": sha256(p / "response.npz")})

    @classmethod
    def load(cls, directory: str | Path) -> "ResponseDataset":
        p = Path(directory); m = read_json(p / "response.json")
        if m.pop("arrays_sha256", None) != sha256(p / "response.npz"):
            raise ValueError("Response array checksum mismatch")
        with np.load(p / "response.npz", allow_pickle=False) as a:
            return cls(**m, **{k: a[k].copy() for k in a.files}).validate()


class ResponseRegressor:
    def __init__(self, alpha: float = 1.0):
        if alpha <= 0: raise ValueError("Positive regularisation required")
        self.alpha = alpha
        self.fitted = False

    @staticmethod
    def design(current: np.ndarray, action: np.ndarray, mode: str, elapsed: np.ndarray | None) -> np.ndarray:
        # Explicit low-capacity interactions are a baseline, not a mechanistic model.
        interaction = (current[:, :, None] * action[:, None, :]).reshape(len(current), -1)
        parts = [current, action, interaction]
        if mode == "transition":
            if elapsed is None or not np.isfinite(elapsed).all() or (elapsed < 0).any():
                raise ValueError("Missing/invalid elapsed time")
            parts.append(np.log1p(elapsed)[:, None])
        return np.concatenate(parts, axis=1)

    def fit(self, dataset: ResponseDataset) -> "ResponseRegressor":
        d = dataset.validate()
        train = np.array([i for i, r in enumerate(d.rows) if r["split"] == "train"])
        if len(train) < 2: raise ValueError("At least two train records required")
        x = self.design(d.current, d.action, d.mode, d.elapsed)
        w = study_unit_weights([d.rows[i] for i in train])
        self.mean = np.average(x[train], axis=0, weights=w)
        self.scale = np.maximum(np.sqrt(np.average((x[train]-self.mean)**2, axis=0, weights=w)), 1e-3)
        xx = np.column_stack((np.ones(len(x)), (x-self.mean)/self.scale))
        y = np.where(d.target_mask, d.target, 0).copy()
        mask = d.target_mask.copy()
        if d.mode == "transition":
            zero = d.elapsed == 0
            if np.any(mask[zero] & ~np.isclose(d.target[zero], d.current[zero], atol=1e-6)):
                raise ValueError("Zero-duration continuous transition is not identity; use endpoint mode")
            mask[zero] = False
            y = (y - d.current) / np.where(zero, 1, d.elapsed)[:, None]
        self.coefficients = np.zeros((xx.shape[1], y.shape[1]))
        self.target_available = np.zeros(y.shape[1], bool)
        penalty = np.eye(xx.shape[1]) * self.alpha; penalty[0, 0] = 0
        for j in range(y.shape[1]):
            use = mask[train, j]
            if use.sum() < 2: continue
            weights = np.sqrt(w[use] / w[use].sum() * use.sum())
            tx = xx[train[use]] * weights[:, None]; ty = y[train[use], j] * weights
            self.coefficients[:, j] = np.linalg.solve(tx.T @ tx + penalty, tx.T @ ty)
            self.target_available[j] = True
        if not self.target_available.any():
            raise ValueError("No measured targets to fit; unavailable is not a zero label")
        self.scope = d.scope
        self.is_synthetic = any(r["origin"] == "synthetic" for r in d.rows)
        self.fit_ids = [d.rows[i]["observation_id"] for i in train]
        self.fitted = True
        return self

    def predict(self, current: np.ndarray, action: np.ndarray, scope: dict,
                elapsed: np.ndarray | None = None) -> np.ndarray:
        if not self.fitted: raise RuntimeError("Fit/load the response model first")
        if scope != self.scope: raise ValueError("Response scope or input encoder representation changed")
        current, action = np.asarray(current, float), np.asarray(action, float)
        if current.ndim != 2 or current.shape[1] != len(scope["input_names"]) or action.shape != (len(current), len(scope["action_names"])):
            raise ValueError("Response inference shape/order mismatch")
        if not np.isfinite(current).all() or not np.isfinite(action).all(): raise ValueError("Invalid inputs")
        elapsed = None if elapsed is None else np.asarray(elapsed, float)
        if scope['mode'] == 'transition' and (elapsed is None or elapsed.shape != (len(current),)):
            raise ValueError('One elapsed time per input row required')
        x = self.design(current, action, scope["mode"], elapsed)
        y = np.column_stack((np.ones(len(x)), (x-self.mean)/self.scale)) @ self.coefficients
        if scope["mode"] == "transition":
            y = current + elapsed[:, None] * y
        return np.where(self.target_available[None], y, np.nan)

    def save(self, directory: str | Path) -> None:
        if not self.fitted: raise RuntimeError("Nothing fitted")
        p = Path(directory)
        save_npz(p / "response_model.npz", coefficients=self.coefficients, mean=self.mean,
                 scale=self.scale, target_available=self.target_available)
        write_json(p / "response_model.json", {"scope": self.scope, "alpha": self.alpha,
                   "is_synthetic": self.is_synthetic, "fit_ids": self.fit_ids,
                   "science_status": "unvalidated", "uncertainty_status": "not_calibrated"})

    @classmethod
    def load(cls, directory: str | Path) -> "ResponseRegressor":
        p = Path(directory); m = read_json(p / "response_model.json"); obj = cls(m["alpha"])
        with np.load(p / "response_model.npz", allow_pickle=False) as a:
            for k in a.files: setattr(obj, k, a[k].copy())
        obj.scope, obj.is_synthetic, obj.fit_ids = m["scope"], m["is_synthetic"], m["fit_ids"]
        obj.fitted = True
        return obj
