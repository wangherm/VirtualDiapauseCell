import copy
import numpy as np
import pytest
from .fixtures import observation_fixture
from vdc.contracts import audit_rows, ObservationBundle
from vdc.observation import normalise_expression, score_programmes
from vdc.metrics import study_unit_weights, regression_metrics


def test_bundle_roundtrip(tmp_path):
    b = observation_fixture(); b.save(tmp_path)
    restored = ObservationBundle.load(tmp_path)
    assert restored.fingerprint == b.fingerprint


def test_array_checksum_detects_change(tmp_path):
    b = observation_fixture(); b.save(tmp_path)
    with (tmp_path / "observations.npz").open("ab") as f: f.write(b"changed")
    with pytest.raises(ValueError, match="checksum"): ObservationBundle.load(tmp_path)


def test_shared_parent_cannot_cross_split():
    b = observation_fixture(); b.rows[24]["link_ids"] = b.rows[0]["link_ids"]
    with pytest.raises(ValueError, match="Leakage"): b.validate()


def test_biological_unit_cannot_cross_split():
    b = observation_fixture(); b.rows[24]["biological_unit"] = b.rows[0]["biological_unit"]
    with pytest.raises(ValueError, match="Leakage"): b.validate()


def test_study_holdout_is_explicit():
    b = observation_fixture(); b.split_policy = "study_holdout"
    with pytest.raises(ValueError, match="Leakage"): b.validate()


def test_internal_train_is_blocked():
    b = observation_fixture(); b.rows[0]["origin"] = "internal"
    with pytest.raises(ValueError, match="internal"): b.validate()


def test_clock_without_reference_rejected():
    b = observation_fixture(); b.clock_reference_id = None
    with pytest.raises(ValueError, match="coordinate",): b.validate()


def test_nan_only_allowed_when_masked():
    b = observation_fixture(); b.values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"): b.validate()
    b.mask[0, 0] = False; b.validate(); assert b.values[0, 0] == 0


def test_all_missing_observation_is_not_a_zero_sample():
    b = observation_fixture(); b.mask[0] = False
    with pytest.raises(ValueError, match="all-missing"): b.validate()


def test_duplicate_ids_fail():
    b = observation_fixture(); b.feature_ids[0] = b.feature_ids[1]
    with pytest.raises(ValueError): b.validate()


def test_log_is_not_counts():
    x = np.array([[.2, .4, .6]])
    with pytest.raises(ValueError, match="integers"): normalise_expression(x, np.ones_like(x, bool), "counts")


def test_library_normalization():
    x = np.array([[1, 2, 3], [2, 4, 6]])
    out = normalise_expression(x, np.ones_like(x, bool), "counts")
    np.testing.assert_allclose(out[0], out[1])


def test_absent_programme_is_missing_not_measured_zero():
    x = np.arange(8).reshape(2, 4)
    p = [{"id": "known", "members": {"a": 1, "b": 1}, "source_ref": "fixture"},
         {"id": "absent", "members": {"not_measured": 1}, "source_ref": "fixture"}]
    r = score_programmes(x, np.ones_like(x, bool), ["a", "b", "c", "d"], p, min_genes=1)
    assert r["mask"][:, 0].all() and not r["mask"][:, 1].any()
    assert (r["coverage"][:, 1] == 0).all()


def test_rank_score_is_invariant_to_positive_rescaling():
    x = np.array([[1., 5., 8., 3.]])
    p = [{"id": "p", "members": {"b": 1, "c": 1}, "source_ref": "fixture"}]
    a = score_programmes(x, np.ones_like(x, bool), ["a", "b", "c", "d"], p, min_genes=2)
    b = score_programmes(x*10, np.ones_like(x, bool), ["a", "b", "c", "d"], p, min_genes=2)
    np.testing.assert_array_equal(a["values"], b["values"])


def test_study_weight_not_dominated_by_row_count():
    rows = [{"study_family": "A", "biological_unit": "a"} for _ in range(20)]
    rows += [{"study_family": "B", "biological_unit": "b"}]
    w = study_unit_weights(rows)
    assert np.isclose(w[:20].sum(), .5) and np.isclose(w[-1], .5)


def test_unavailable_metric_is_not_zero_error():
    r = regression_metrics(np.zeros(2), np.zeros(2), np.zeros(2, bool))
    assert r["mse"] is None and r["status"] == "unavailable"
