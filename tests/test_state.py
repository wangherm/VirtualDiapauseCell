import numpy as np
import pytest
import torch
from .fixtures import observation_fixture
from vdc.state import (StateConfig, ProgrammeStateModel, fit_state, StatePredictor,
                       masked_weighted_mse, train_scaling)


def cfg():
    return StateConfig(hidden_dim=16, latent_dim=8, layers=1, heads=2, validation_every=5)


def test_model_forward_backward():
    b = observation_fixture()
    m = ProgrammeStateModel(len(b.feature_ids), cfg())
    out = m(torch.tensor(b.values), torch.tensor(b.mask), torch.tensor(b.coverage))
    assert out["programme"].shape == b.values.shape and out["clock"].shape == b.clock.shape
    out["programme"].square().mean().backward()
    assert m.encoder.layers[0].linear1.weight.grad is not None


def test_no_clock_supervision_no_gradient():
    x = torch.tensor([.3, .4], requires_grad=True)
    loss = masked_weighted_mse(x, torch.zeros(2), torch.zeros(2, dtype=torch.bool), torch.ones(2))
    loss.backward(); assert torch.equal(x.grad, torch.zeros(2))


def test_clock_not_clipped_to_zero_one():
    b = observation_fixture(); m = ProgrammeStateModel(len(b.feature_ids), cfg())
    with torch.no_grad(): m.clock_head.weight.zero_(); m.clock_head.bias.fill_(1.4)
    out = m(torch.tensor(b.values), torch.tensor(b.mask), torch.tensor(b.coverage))
    assert torch.all(out["clock"] > 1)


def test_checkpoint_roundtrip(tmp_path):
    b = observation_fixture(); report = fit_state(b, tmp_path / "run", 5, cfg())
    assert report["test_split_evaluated"] is False and report["scientific_success"] is False
    p1 = StatePredictor.load(tmp_path / "run").predict(b)
    p2 = StatePredictor.load(tmp_path / "run").predict(b)
    np.testing.assert_array_equal(p1["programme"], p2["programme"])


def test_resume_matches_uninterrupted_cpu(tmp_path):
    b = observation_fixture()
    fit_state(b, tmp_path / "full", 10, cfg())
    fit_state(b, tmp_path / "resume", 5, cfg())
    fit_state(b, tmp_path / "resume", 10, cfg(), resume=True)
    a = torch.load(tmp_path / "full/last.pt", weights_only=True)
    z = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    for key in a["model"]: torch.testing.assert_close(a["model"][key], z["model"][key], rtol=0, atol=0)


def test_resume_rejects_data_change(tmp_path):
    b = observation_fixture(); fit_state(b, tmp_path / "r", 5, cfg())
    b.values[0, 0] += 1
    with pytest.raises(ValueError, match="changed"): fit_state(b, tmp_path / "r", 10, cfg(), resume=True)


def test_existing_run_not_overwritten(tmp_path):
    b = observation_fixture(); fit_state(b, tmp_path / "r", 5, cfg())
    with pytest.raises(FileExistsError): fit_state(b, tmp_path / "r", 5, cfg())


def test_untrained_clock_not_emitted(tmp_path):
    b = observation_fixture(); b.clock_mask[:] = False; b.clock_reference_id = None
    fit_state(b, tmp_path / "r", 5, cfg())
    assert StatePredictor.load(tmp_path / "r").predict(b)["clock"] is None


def test_scope_or_order_change_requires_calibration(tmp_path):
    b = observation_fixture(); fit_state(b, tmp_path / "r", 5, cfg())
    b.feature_ids = list(reversed(b.feature_ids))
    with pytest.raises(ValueError, match="mismatch"): StatePredictor.load(tmp_path / "r").predict(b)


def test_train_scaling_ignores_validation():
    b = observation_fixture(); idx = b.indices("train"); weights = np.ones(len(idx))
    m1, s1 = train_scaling(b.values[idx], b.mask[idx], weights)
    b.values[b.indices("validation")] += 10000
    m2, s2 = train_scaling(b.values[idx], b.mask[idx], weights)
    np.testing.assert_array_equal(m1, m2); np.testing.assert_array_equal(s1, s2)


def test_semantic_cache_order_dimensions_checked():
    with pytest.raises(ValueError, match="Semantic"):
        ProgrammeStateModel(12, cfg(), torch.zeros(11, 8))


def test_frozen_semantic_buffer():
    m = ProgrammeStateModel(12, cfg(), torch.ones(12, 8))
    assert not m.semantics.requires_grad
    assert m.semantic_projection.weight.requires_grad


def test_explicit_transfer_keeps_local_scaling_and_resume_provenance(tmp_path):
    from vdc.io import read_json
    b=observation_fixture();fit_state(b,tmp_path/'parent',5,cfg())
    local=observation_fixture();local.values+=10
    local.context['species']='second_synthetic_context'
    fit_state(local,tmp_path/'child',5,cfg(),pretrained=tmp_path/'parent')
    m=read_json(tmp_path/'child/run.json');keys=m['transfer']['copied_parameters']
    assert keys and all(not k.startswith(('clock_head.','semantic_projection.')) for k in keys)
    assert 'semantics' not in keys
    assert not np.allclose(StatePredictor.load(tmp_path/'parent').mean,StatePredictor.load(tmp_path/'child').mean)
    fit_state(local,tmp_path/'child',10,cfg(),resume=True)
    assert read_json(tmp_path/'child/run.json')['transfer']==m['transfer']
    changed=observation_fixture();changed.feature_ids=list(reversed(changed.feature_ids))
    with pytest.raises(ValueError,match='ID order'):
        fit_state(changed,tmp_path/'bad',5,cfg(),pretrained=tmp_path/'parent')


def test_locked_test_cannot_run_accidentally(tmp_path):
    from vdc.evaluate import evaluate_state
    b = observation_fixture(); fit_state(b, tmp_path / "r", 5, cfg())
    with pytest.raises(ValueError, match="Locked"):
        evaluate_state(tmp_path / "r", b, tmp_path / "eval", split="locked_test")


def test_explicit_synthetic_evaluation_stays_synthetic(tmp_path):
    from vdc.evaluate import evaluate_state
    b = observation_fixture(); fit_state(b, tmp_path / "r", 5, cfg())
    r = evaluate_state(tmp_path / "r", b, tmp_path / "eval", split="locked_test", allow_locked_test=True)
    assert r["data_kind"] == "synthetic" and r["approval"] == "requires_scientific_review"
    assert len(r["baseline_comparisons"]) == 3
