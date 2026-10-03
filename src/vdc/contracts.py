"""Prediction-time data contracts. Targets never enter model inputs implicitly."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .io import object_hash, read_json, sha256, write_json, save_npz

SPLITS = {"train", "validation", "test", "locked_test"}


def audit_rows(rows: list[dict], split_policy: str) -> dict:
    if split_policy not in {"study_holdout", "unit_holdout"}:
        raise ValueError("Declare study_holdout or unit_holdout, do not infer a split policy.")
    seen_ids: set[str] = set()
    links: dict[str, str] = {}
    for row in rows:
        required = ("observation_id", "study_family", "biological_unit", "split", "origin", "link_ids")
        if any(k not in row for k in required):
            raise ValueError(f"Missing row fields: {required}")
        if row["split"] not in SPLITS:
            raise ValueError("Unknown split")
        if row["origin"] not in {"public", "internal", "synthetic"}:
            raise ValueError("origin must be public, internal or synthetic")
        if not all(isinstance(row[k], str) and row[k].strip()
                   for k in ("observation_id", "study_family", "biological_unit")):
            raise ValueError("IDs must be nonempty strings")
        if row["observation_id"] in seen_ids:
            raise ValueError("Duplicate observation ID")
        seen_ids.add(row["observation_id"])
        if row["origin"] == "internal" and row["split"] != "locked_test":
            from .admission import validate_internal_row
            validate_internal_row(row)
        if not isinstance(row["link_ids"], list) or any(not isinstance(x, str) or not x for x in row["link_ids"]):
            raise ValueError("link_ids must be globally namespaced nonempty strings")
        # A single shared control / parent / animal / modality link cannot cross splits.
        keys = [f"unit:{row['study_family']}:{row['biological_unit']}"] + row["link_ids"]
        if split_policy == "study_holdout":
            keys.append(f"study:{row['study_family']}")
        for key in keys:
            if key in links and links[key] != row["split"]:
                raise ValueError(f"Leakage: {key} crosses {links[key]} and {row['split']}")
            links[key] = row["split"]
    return {s: sum(r["split"] == s for r in rows) for s in sorted(SPLITS)}


@dataclass
class ObservationBundle:
    values: np.ndarray
    mask: np.ndarray
    coverage: np.ndarray
    feature_ids: list[str]
    rows: list[dict]
    context: dict
    split_policy: str
    clock: np.ndarray
    clock_mask: np.ndarray
    clock_reference_id: str | None
    schema_version: str = "observation-v1"

    def validate(self) -> "ObservationBundle":
        self.values = np.asarray(self.values, dtype=np.float32)
        self.mask = np.asarray(self.mask, dtype=bool)
        self.coverage = np.asarray(self.coverage, dtype=np.float32)
        self.clock = np.asarray(self.clock, dtype=np.float32)
        self.clock_mask = np.asarray(self.clock_mask, dtype=bool)
        if self.values.ndim != 2 or min(self.values.shape) < 1:
            raise ValueError("values must be a nonempty samples-by-programmes matrix")
        n, p = self.values.shape
        if self.mask.shape != (n, p) or self.coverage.shape != (n, p):
            raise ValueError("mask/coverage must match values")
        if len(self.rows) != n or len(self.feature_ids) != p or len(set(self.feature_ids)) != p:
            raise ValueError("Row count or feature order is invalid")
        if any(not isinstance(x, str) or not x for x in self.feature_ids):
            raise ValueError("feature IDs must be nonempty strings")
        if not np.isfinite(self.values[self.mask]).all():
            raise ValueError("Observed values must be finite")
        if not np.isfinite(self.coverage).all() or (self.coverage < 0).any() or (self.coverage > 1).any():
            raise ValueError("coverage must be finite in [0, 1]")
        if np.any(self.mask & (self.coverage <= 0)) or np.any(self.mask.sum(1) == 0):
            raise ValueError("Observed entries need positive coverage; all-missing samples are unavailable")
        if self.clock.shape != (n,) or self.clock_mask.shape != (n,):
            raise ValueError("clock and clock_mask must have shape (samples,)")
        if not np.isfinite(self.clock[self.clock_mask]).all():
            raise ValueError("Observed clock targets must be finite")
        if self.clock_mask.any() and not self.clock_reference_id:
            raise ValueError("Clock supervision requires a named molecular coordinate, not bare hours")
        for key in ("species", "system", "material", "assay", "programme_definition_id", "preprocessing_id"):
            if not isinstance(self.context.get(key), str) or not self.context[key]:
                raise ValueError(f"One declared context per bundle is required: {key}")
        audit_rows(self.rows, self.split_policy)
        # Missing is stored as zero plus mask for transport, never as measured zero.
        self.values = np.where(self.mask, self.values, 0).astype(np.float32)
        self.clock = np.where(self.clock_mask, self.clock, 0).astype(np.float32)
        return self

    def indices(self, split: str) -> np.ndarray:
        return np.array([i for i, row in enumerate(self.rows) if row["split"] == split], dtype=int)

    def subset(self, indices) -> "ObservationBundle":
        """Select rows before normalisation, corruption or a model forward call."""
        import copy
        idx = np.asarray(indices, dtype=int)
        return ObservationBundle(self.values[idx].copy(), self.mask[idx].copy(),
            self.coverage[idx].copy(), self.feature_ids.copy(),
            [copy.deepcopy(self.rows[i]) for i in idx], copy.deepcopy(self.context),
            self.split_policy, self.clock[idx].copy(), self.clock_mask[idx].copy(),
            self.clock_reference_id).validate()

    @property
    def scope(self) -> dict:
        return {"context": self.context, "feature_ids": self.feature_ids,
                "clock_reference_id": self.clock_reference_id}

    @property
    def fingerprint(self) -> str:
        import hashlib
        h = hashlib.sha256(object_hash({"scope": self.scope, "rows": self.rows,
                                       "split_policy": self.split_policy}).encode())
        for a in (self.values, self.mask, self.coverage, self.clock, self.clock_mask):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def save(self, directory: str | Path) -> None:
        self.validate()
        directory = Path(directory)
        save_npz(directory / "observations.npz", values=self.values, mask=self.mask,
                 coverage=self.coverage, clock=self.clock, clock_mask=self.clock_mask)
        write_json(directory / "manifest.json", {
            "schema_version": self.schema_version, "feature_ids": self.feature_ids,
            "rows": self.rows, "context": self.context, "split_policy": self.split_policy,
            "clock_reference_id": self.clock_reference_id,
            "arrays_sha256": sha256(directory / "observations.npz")})

    @classmethod
    def load(cls, directory: str | Path) -> "ObservationBundle":
        directory = Path(directory)
        m = read_json(directory / "manifest.json")
        if m.get("schema_version") != "observation-v1":
            raise ValueError("Unsupported schema; convert data explicitly")
        if sha256(directory / "observations.npz") != m["arrays_sha256"]:
            raise ValueError("Array checksum differs from manifest")
        with np.load(directory / "observations.npz", allow_pickle=False) as a:
            return cls(**{k: a[k].copy() for k in ("values", "mask", "coverage", "clock", "clock_mask")},
                       **{k: m[k] for k in ("feature_ids", "rows", "context", "split_policy", "clock_reference_id")}).validate()
