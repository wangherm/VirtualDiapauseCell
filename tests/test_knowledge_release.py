import copy
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from .fixtures import observation_fixture
from vdc.knowledge import audit_knowledge, retrieve, masked_mean_pool, completion_example
from vdc.release import create_release
from vdc.state import fit_state, StateConfig


def records():
    return [{"record_id": "r1", "study_family": "paper1", "source_ref": "synthetic-test-reference",
             "split": "train", "kind": "object_description", "object_id": "P0", "reviewed": True,
             "text": "dormancy programme description"}]


def test_knowledge_family_leakage():
    r = records(); other = {**r[0], "record_id": "r2", "split": "test"}
    with pytest.raises(ValueError, match="crosses"): audit_knowledge(r+[other])


def test_unreviewed_knowledge_not_gold():
    r = records(); r[0]["reviewed"] = False
    with pytest.raises(ValueError, match="Unreviewed"): audit_knowledge(r)


def test_retrieval_excludes_heldout_before_scoring():
    r = records(); assert retrieve(r, "dormancy", {"paper1"}) == []
    assert retrieve(r, "dormancy", set())[0]["record_id"] == "r1"


def test_pooling_ignores_padding():
    h = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]]])
    y = masked_mean_pool(h, torch.tensor([[1, 1, 0]]))
    torch.testing.assert_close(y, torch.tensor([[2., 3.]]))


class CharacterTokenizer:
    """A contract fixture, not a substitute Qwen tokenizer."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}|" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(x) for x in text]}


def test_completion_only_mask():
    t = CharacterTokenizer(); prompt = [{"role": "user", "content": "question"}]
    r = completion_example(t, prompt, "answer", 100)
    assert r["labels"][0] == -100
    assert [x for x in r["labels"] if x != -100] == [ord(x) for x in "answer|"]


def test_overlong_sft_not_silently_truncated():
    with pytest.raises(ValueError, match="length"):
        completion_example(CharacterTokenizer(), [{"role": "user", "content": "question"}], "answer", 10)


def test_synthetic_checkpoint_cannot_promote(tmp_path):
    b = observation_fixture()
    fit_state(b, tmp_path / "r", 5, StateConfig(hidden_dim=16, latent_dim=8, layers=1, heads=2, validation_every=5))
    (tmp_path / "review.json").write_text('{}')
    with pytest.raises(ValueError, match="Synthetic"):
        create_release(tmp_path / "r", tmp_path / "review.json", tmp_path / "release")


def test_local_service_contract(monkeypatch):
    import vdc.service as svc
    b = observation_fixture(); scope = b.scope
    class FakePredictor:
        def predict(self, bundle):
            bundle.validate()
            if bundle.scope != scope: raise ValueError("scope mismatch")
            return {"programme": bundle.values, "clock": np.ones(len(bundle.rows))*.4,
                    "measured_mask": bundle.mask, "uncertainty_status": "not_calibrated"}
    monkeypatch.setattr(svc, "verify_release", lambda path: {"capabilities": ["clock"], "scope": scope})
    monkeypatch.setattr(svc.StatePredictor, "load", lambda path: FakePredictor())
    client = TestClient(svc.create_app("mock-only-no-released-model"))
    assert client.get('/health').status_code == 200
    payload = {**scope, "values": b.values[:2].tolist(), "mask": b.mask[:2].tolist(), "coverage": b.coverage[:2].tolist()}
    r = client.post('/predict/state', json=payload)
    assert r.status_code == 200 and r.json()["depth"] is None
    assert "programme_prediction" not in r.json()  # Not approved by the mocked release.
    payload["feature_ids"] = list(reversed(payload["feature_ids"]))
    assert client.post('/predict/state', json=payload).status_code == 422
