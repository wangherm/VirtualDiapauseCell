"""Explicit held-out evaluation; fitting commands never call this on test data."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from .contracts import ObservationBundle, audit_rows
from .state import StatePredictor, corrupt
from .metrics import grouped_metrics
from .io import sha256, write_json, save_npz


def evaluate_state(run_dir: str | Path, bundle: ObservationBundle, out_dir: str | Path,
                   split: str = "test", allow_locked_test: bool = False, seed: int = 1007) -> dict:
    if split not in {"validation", "test", "locked_test"}: raise ValueError("Explicit held-out split required")
    if split == "locked_test" and not allow_locked_test:
        raise ValueError("Locked test requires an explicitly authorised final evaluation")
    b = bundle.validate(); idx = b.indices(split)
    if not len(idx): raise ValueError("No observations in the requested split")
    # Never forward test/locked_test while requesting validation.
    b = b.subset(idx)
    idx = np.arange(len(b.rows))
    p = StatePredictor.load(run_dir)
    fit_ids = set(p.manifest["fit_ids"])
    if fit_ids.intersection(b.rows[i]["observation_id"] for i in idx):
        raise ValueError("Evaluation overlaps fitted observation IDs")
    if "fit_rows" not in p.manifest:
        raise ValueError("Checkpoint lacks fit lineage; rebuild with audited training metadata")
    audit_rows(p.manifest["fit_rows"] + [b.rows[i] for i in idx], p.manifest["split_policy"])
    cfg = p.model.config
    q = copy.deepcopy(b)
    scaled = torch.as_tensor(np.where(b.mask, (b.values-p.mean)/p.scale, 0), dtype=torch.float32)
    xx, mm, cc, hidden = corrupt(scaled, torch.tensor(b.mask), torch.tensor(b.coverage),
                                cfg.corrupt_fraction, cfg.additive_noise, seed)
    q.values = np.where(mm.numpy(), xx.numpy()*p.scale+p.mean, 0).astype(np.float32)
    q.mask, q.coverage = mm.numpy(), cc.numpy()
    # Target clock is kept outside the predictor query.
    q.clock[:], q.clock_mask[:] = 0, False
    pred = p.predict(q)
    base = np.where(q.mask, q.values, p.mean)
    report = {"checkpoint_sha256": sha256(Path(run_dir)/"best.pt"),
              "data_kind": "synthetic" if any(b.rows[i]["origin"] == "synthetic" for i in idx) else "real",
              "evaluation_scope": {"split": split, "split_policy": b.split_policy, "context": b.context,
                         "bundle_fingerprint": b.fingerprint, "test_observation_ids": [b.rows[i]["observation_id"] for i in idx],
                         "task": "controlled_additional_measurement_corruption_not_latent_biological_truth"},
              "baseline_comparisons": {}, "approval": "requires_scientific_review",
              "uncertainty_status": "not_calibrated"}
    rows = [b.rows[i] for i in idx]
    for name, values in (("model", pred["programme"]), ("input_with_train_mean_for_missing", base),
                         ("training_mean", np.broadcast_to(p.mean, b.values.shape))):
        report["baseline_comparisons"][name] = {
             "full_originally_observed": grouped_metrics(values[idx], b.values[idx], b.mask[idx], rows),
             "additional_hidden": grouped_metrics(values[idx], b.values[idx], hidden.numpy()[idx], rows)}
    report["clock"] = grouped_metrics(pred["clock"][idx], b.clock[idx], b.clock_mask[idx], rows) if pred["clock"] is not None else {"status": "unavailable"}
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError("Choose a new evaluation directory")
    write_json(out / "evaluation.json", report)
    save_npz(out / "predictions.npz", programme=pred["programme"][idx], target=b.values[idx],
             target_mask=b.mask[idx], additional_hidden=hidden.numpy()[idx])
    return report
