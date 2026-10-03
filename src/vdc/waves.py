"""DC03/DC04 share a curve implementation; gene IDs and programme IDs stay distinct."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .io import save_npz, read_json, write_json
from .contracts import audit_rows


@dataclass
class WaveReference:
    coefficients: np.ndarray
    fitted: np.ndarray
    feature_ids: list[str]
    coordinate_id: str
    feature_kind: str
    context_id: str
    lower: np.ndarray
    upper: np.ndarray
    centre: float
    scale: float
    degree: int

    @classmethod
    def fit(cls, coordinate: np.ndarray, values: np.ndarray, mask: np.ndarray,
            feature_ids: list[str], rows: list[dict], coordinate_id: str,
            context_id: str, feature_kind: str = "programme", degree: int = 1,
            alpha: float = 0.001, split_policy: str = "unit_holdout") -> "WaveReference":
        audit_rows(rows, split_policy)
        from .admission import audit_internal_task
        audit_internal_task(rows,'waves')
        t, y, mask = np.asarray(coordinate, float), np.asarray(values, float), np.asarray(mask, bool)
        if y.ndim != 2 or y.shape != mask.shape or len(t) != len(y) or len(rows) != len(t):
            raise ValueError("Invalid curve data shapes")
        if y.shape[1] != len(feature_ids) or len(set(feature_ids)) != len(feature_ids):
            raise ValueError("Feature IDs differ from curve data")
        if degree not in (1, 2, 3) or alpha < 0 or not coordinate_id or not context_id:
            raise ValueError("Explicit coordinate/context and degree 1..3 required")
        if feature_kind not in {"gene", "TF", "programme", "regulon"}:
            raise ValueError("Unknown wave feature kind")
        idx = np.array([i for i, r in enumerate(rows) if r["split"] == "train" and r.get("reference_eligible", False)])
        if len(idx) < degree + 2 or not np.isfinite(t[idx]).all():
            raise ValueError("Too few explicitly eligible train reference observations")
        if len(np.unique(t[idx])) < degree + 2:
            raise ValueError("Curve complexity exceeds distinct observed coordinates")
        centre, scale = float(t[idx].mean()), max(float(t[idx].std()), 1e-6)
        basis = np.vander((t[idx] - centre) / scale, N=degree + 1, increasing=True)
        co = np.zeros((degree + 1, y.shape[1])); fitted = np.zeros(y.shape[1], bool)
        lower, upper = np.full(y.shape[1], np.nan), np.full(y.shape[1], np.nan)
        penalty = np.eye(degree + 1) * alpha; penalty[0, 0] = 0
        for j in range(y.shape[1]):
            use = mask[idx, j]
            if use.sum() < degree + 2 or len(np.unique(t[idx][use])) < degree + 2:
                continue
            if not np.isfinite(y[idx[use], j]).all():
                raise ValueError("Non-finite observed reference data")
            x = basis[use]
            co[:, j] = np.linalg.solve(x.T @ x + penalty, x.T @ y[idx[use], j])
            fitted[j] = True; lower[j], upper[j] = t[idx[use]].min(), t[idx[use]].max()
        if not fitted.any():
            raise ValueError("No feature has enough reference observations")
        return cls(co, fitted, feature_ids, coordinate_id, feature_kind, context_id,
                   lower, upper, centre, scale, degree)

    def predict(self, coordinate: np.ndarray, coordinate_id: str, context_id: str,
                allow_extrapolation: bool = False) -> dict:
        if coordinate_id != self.coordinate_id or context_id != self.context_id:
            raise ValueError("Reference coordinate/context differs")
        t = np.asarray(coordinate, float)
        if t.ndim != 1 or not np.isfinite(t).all():
            raise ValueError("Invalid coordinate")
        mean = np.vander((t - self.centre) / self.scale, N=self.degree + 1, increasing=True) @ self.coefficients
        supported = self.fitted[None] & (t[:, None] >= self.lower) & (t[:, None] <= self.upper)
        emit = np.broadcast_to(self.fitted, mean.shape) if allow_extrapolation else supported
        return {"expected": np.where(emit, mean, np.nan), "within_observed_range": supported,
                "is_extrapolation": np.broadcast_to(self.fitted, mean.shape) & ~supported,
                "uncertainty_status": "not_calibrated"}

    def residual(self, coordinate: np.ndarray, observed: np.ndarray, mask: np.ndarray,
                 coordinate_id: str, context_id: str) -> dict:
        out = self.predict(coordinate, coordinate_id, context_id)
        if observed.shape != out["expected"].shape or mask.shape != observed.shape:
            raise ValueError("Residual observation shape differs")
        valid = np.asarray(mask, bool) & out["within_observed_range"]
        out["residual"] = np.where(valid, observed - out["expected"], np.nan)
        out["residual_mask"] = valid
        return out

    def save(self, directory: str | Path) -> None:
        p = Path(directory)
        save_npz(p / "reference.npz", coefficients=self.coefficients, fitted=self.fitted,
                 lower=self.lower, upper=self.upper)
        write_json(p / "reference.json", {"feature_ids": self.feature_ids, "coordinate_id": self.coordinate_id,
                   "feature_kind": self.feature_kind, "context_id": self.context_id, "centre": self.centre,
                   "scale": self.scale, "degree": self.degree, "uncertainty_status": "not_calibrated"})

    @classmethod
    def load(cls, directory: str | Path) -> "WaveReference":
        p = Path(directory); m = read_json(p / "reference.json"); m.pop("uncertainty_status")
        with np.load(p / "reference.npz", allow_pickle=False) as a:
            return cls(**m, **{k: a[k].copy() for k in ("coefficients", "fitted", "lower", "upper")})


def describe_wave(coordinate: np.ndarray, raw_values: np.ndarray,
                  flat_range: float = 0.05, minimum_correlation: float = 0.6) -> dict:
    """Descriptive templates only; not a measured activation clock or the exact thesis code.

    Flat is assessed on raw amplitude BEFORE standardisation. Unknown/complex curves remain legal.
    """
    t, y = np.asarray(coordinate, float), np.asarray(raw_values, float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 3 or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError("Finite coordinate/curve vectors of equal length >=3 required")
    if np.any(np.diff(t) <= 0) or flat_range < 0:
        raise ValueError("Coordinates must be strictly increasing")
    amplitude = float(np.ptp(y))
    if amplitude <= flat_range:
        return {"shape": "Flat", "raw_amplitude": amplitude, "shape_correlation": None}
    u = (t - t[0]) / (t[-1] - t[0])
    templates = {"Up": u, "Down": 1-u,
                 "EarlyUp": np.minimum(2*u, 1), "LateUp": np.maximum(2*u-1, 0),
                 "EarlyDown": 1-np.minimum(2*u, 1), "LateDown": 1-np.maximum(2*u-1, 0)}
    for name, centre in (("Early", .2), ("Mid", .5), ("Late", .8)):
        v = np.exp(-.5 * ((u-centre)/.08)**2)
        templates[name+"Peak"] = v; templates[name+"Valley"] = -v
    corr = {name: float(np.corrcoef(y, v)[0, 1]) for name, v in templates.items()}
    best = max(corr, key=corr.get)
    return {"shape": best if corr[best] >= minimum_correlation else "Unclassified",
            "raw_amplitude": amplitude, "shape_correlation": corr[best],
            "relative_coordinate_only": True, "fine_peak_timing_validated": False}
