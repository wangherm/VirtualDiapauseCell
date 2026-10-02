"""Synthetic software fixtures. Never used as biological training or validation data."""
import numpy as np
from vdc.contracts import ObservationBundle
from vdc.response import ResponseDataset


def observation_fixture(seed=1, n=36, p=12):
    rng = np.random.default_rng(seed)
    clock = np.tile(np.linspace(0, 1, 12), 3).astype(np.float32)[:n]
    residual = rng.normal(size=(n, 2))
    slopes = np.linspace(-1, 1, p)
    values = clock[:, None] * slopes + residual @ rng.normal(size=(2, p)) * .1
    values = values.astype(np.float32)
    mask = rng.random((n, p)) > .07
    mask[:, :2] = True
    rows = []
    for i in range(n):
        split = "train" if i < 24 else "validation" if i < 30 else "locked_test"
        rows.append({"observation_id": f"synthetic-{i}", "study_family": "synthetic-study",
                     "biological_unit": f"synthetic-unit-{i}", "split": split, "origin": "synthetic",
                     "link_ids": [f"synthetic-parent-{i}"], "reference_eligible": True})
    context = {"species": "synthetic_not_an_organism", "system": "software_test_only",
               "material": "synthetic_vectors", "assay": "synthetic",
               "programme_definition_id": "fixture-v1", "preprocessing_id": "fixture-linear-v1"}
    return ObservationBundle(values, mask, mask.astype(np.float32), [f"P{i:03}" for i in range(p)],
             rows, context, "unit_holdout", clock, np.ones(n, bool), "synthetic_ordinal_axis_v1").validate()


def response_fixture(mode="endpoint", seed=2):
    rng = np.random.default_rng(seed); n, p, a = 24, 4, 2
    current = rng.normal(size=(n, p)); action = rng.integers(0, 2, size=(n, a)).astype(float)
    change = .2 + action @ rng.normal(size=(a, p))
    elapsed = np.tile([0., 1., 2., 4.], 6) if mode == "transition" else None
    target = current + elapsed[:, None] * change if mode == "transition" else change
    if mode == "functional": target = (current[:, :1] + action[:, :1]) * .2
    rows = []
    for i in range(n):
        split = "train" if i < 18 else "validation"
        initial, outcome = f"synthetic-initial-{i}", f"synthetic-outcome-{i}"
        rows.append({"observation_id": f"synthetic-response-{i}", "study_family": "synthetic-response-study",
                     "biological_unit": f"unit-{i}", "split": split, "origin": "synthetic",
                     "control_or_initial_ids": [initial], "outcome_ids": [outcome],
                     "link_ids": [initial, outcome]})
    return ResponseDataset(current, action, target, np.ones_like(target, bool), rows, mode,
           "synthetic-representation-v1", [f"p{i}" for i in range(p)], ["action0", "action1"],
           [f"p{i}" for i in range(target.shape[1])], "synthetic-context-v1", elapsed,
           "hours" if mode == "transition" else None,
           "synthetic_continuous_assay" if mode == "functional" else None,
           "synthetic_protocol" if mode == "functional" else None).validate()
