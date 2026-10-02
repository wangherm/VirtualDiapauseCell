import copy
import numpy as np
import pytest
from .fixtures import observation_fixture, response_fixture
from vdc.waves import WaveReference, describe_wave
from vdc.response import ResponseRegressor, matched_contrast, ResponseDataset


def fitted_wave():
    b = observation_fixture()
    return b, WaveReference.fit(b.clock, b.values, b.mask, b.feature_ids, b.rows,
                               b.clock_reference_id, "ctx")


def test_wave_roundtrip(tmp_path):
    b, w = fitted_wave(); w.save(tmp_path); other = WaveReference.load(tmp_path)
    np.testing.assert_allclose(w.predict(b.clock, b.clock_reference_id, "ctx")["expected"],
                              other.predict(b.clock, b.clock_reference_id, "ctx")["expected"], equal_nan=True)


def test_outside_wave_range_is_explicit():
    b, w = fitted_wave(); out = w.predict(np.array([5.]), b.clock_reference_id, "ctx")
    assert np.isnan(out["expected"]).all() and out["is_extrapolation"].all()


def test_wave_coordinate_identity_enforced():
    b, w = fitted_wave()
    with pytest.raises(ValueError): w.predict(b.clock, "hours_are_not_biotime", "ctx")


def test_sparse_time_grid_cannot_claim_complex_curve():
    b = observation_fixture(); b.clock = np.round(b.clock*2)/2
    with pytest.raises(ValueError, match="complexity"):
        WaveReference.fit(b.clock, b.values, b.mask, b.feature_ids, b.rows, b.clock_reference_id, "ctx", degree=3)


def test_wave_preserves_residual_not_mean_only():
    b, w = fitted_wave(); expected = w.predict(b.clock, b.clock_reference_id, "ctx")["expected"]
    shifted = expected + .4
    out = w.residual(b.clock, shifted, np.ones_like(b.mask), b.clock_reference_id, "ctx")
    np.testing.assert_allclose(out["residual"][out["residual_mask"]], .4)


def test_flat_checked_before_shape_standardisation():
    out = describe_wave(np.linspace(0, 1, 20), np.linspace(0, .01, 20), flat_range=.05)
    assert out["shape"] == "Flat"


def test_monotone_wave():
    t = np.linspace(0, 1, 50); out = describe_wave(t, 2*t)
    assert out["shape"] == "Up" and out["raw_amplitude"] == 2


def test_observed_response_group_not_cartesian_product():
    y = matched_contrast(np.zeros((3, 4)), np.ones((5, 4)))
    assert y.shape == (4,) and np.all(y == 1)


@pytest.mark.parametrize("mode", ["endpoint", "transition", "functional"])
def test_response_fit_save_reload(mode, tmp_path):
    d = response_fixture(mode); m = ResponseRegressor().fit(d); m.save(tmp_path)
    other = ResponseRegressor.load(tmp_path)
    a = m.predict(d.current, d.action, d.scope, d.elapsed)
    z = other.predict(d.current, d.action, d.scope, d.elapsed)
    np.testing.assert_array_equal(a, z)


def test_transition_zero_time_identity():
    d = response_fixture("transition"); m = ResponseRegressor().fit(d)
    y = m.predict(d.current, d.action, d.scope, np.zeros(len(d.rows)))
    np.testing.assert_allclose(y, d.current)


def test_no_extra_action_still_has_natural_drift():
    d = response_fixture("transition"); m = ResponseRegressor(alpha=.01).fit(d)
    y = m.predict(d.current, np.zeros_like(d.action), d.scope, np.ones(len(d.rows)))
    assert np.mean(np.abs(y - d.current)) > .05


def test_missing_elapsed_not_zero_time():
    d = response_fixture("transition"); d.elapsed = None
    with pytest.raises(ValueError, match="elapsed"): d.validate()


def test_existing_genotype_not_reapplied():
    d = response_fixture("transition")
    d.rows[0].update(new_acute_ko=True, same_genotype_already_in_initial=True)
    with pytest.raises(ValueError, match="Constitutive"): d.validate()


def test_future_info_is_prohibited():
    d = response_fixture(); d.rows[0]["input_contains_future_information"] = True
    with pytest.raises(ValueError, match="Future"): d.validate()


def test_no_functional_label_not_filled_from_clock():
    d = response_fixture("functional"); d.target_mask[:] = False
    with pytest.raises(ValueError, match="No measured"): ResponseRegressor().fit(d)


def test_functional_protocol_required():
    d = response_fixture("functional"); d.protocol_id = None
    with pytest.raises(ValueError, match="protocol"): d.validate()


def test_response_encoder_change_invalidates_scope():
    d = response_fixture(); m = ResponseRegressor().fit(d)
    scope = {**d.scope, "representation_id": "new-encoder-checksum"}
    with pytest.raises(ValueError, match="changed"): m.predict(d.current, d.action, scope)


def test_shared_control_leakage_is_detected():
    d = response_fixture(); control = d.rows[0]["control_or_initial_ids"][0]
    d.rows[-1]["control_or_initial_ids"] = [control]; d.rows[-1]["link_ids"].append(control)
    with pytest.raises(ValueError, match="Leakage"): d.validate()
