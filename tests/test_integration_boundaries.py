import copy
import numpy as np
import pytest
from .fixtures import observation_fixture, response_fixture
from .test_state import cfg
from vdc.state import fit_state
from vdc.evaluate import evaluate_state
from vdc.response import ResponseDataset
from vdc.observation import normalise_expression


def test_unavailable_clock_absent_from_saved_predictions(tmp_path):
    b = observation_fixture()
    b.clock_mask[:] = False
    b.clock_reference_id = None
    fit_state(b, tmp_path / 'run', 5, cfg())
    with np.load(tmp_path / 'run/validation_predictions.npz', allow_pickle=False) as a:
        assert 'clock' not in a.files


def test_evaluation_rejects_relabelled_training_sample(tmp_path):
    b = observation_fixture()
    fit_state(b, tmp_path / 'run', 5, cfg())
    q = copy.deepcopy(b)
    for row in q.rows:
        if row['split'] == 'train':
            row['split'] = 'test'
    with pytest.raises(ValueError, match='overlaps'):
        evaluate_state(tmp_path / 'run', q, tmp_path / 'eval')


def test_estimated_counts_are_explicit_not_rounded():
    x = np.array([[1.2, 2.7]])
    with pytest.raises(ValueError, match='integers'):
        normalise_expression(x, np.ones_like(x, bool), 'counts')
    np.testing.assert_allclose(
        normalise_expression(x, np.ones_like(x, bool), 'estimated_counts'),
        np.log1p(x / x.sum() * 10000))


def test_response_corruption_is_detected(tmp_path):
    response_fixture('endpoint').save(tmp_path)
    p = tmp_path / 'response.npz'
    p.write_bytes(p.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum'):
        ResponseDataset.load(tmp_path)


def test_evaluation_rejects_renamed_training_unit(tmp_path):
    b = observation_fixture()
    fit_state(b, tmp_path / 'run', 5, cfg())
    q = copy.deepcopy(b)
    for row in q.rows:
        if row['split'] == 'train':
            row['split'] = 'test'
            row['observation_id'] += '-renamed'
    with pytest.raises(ValueError, match='Leakage'):
        evaluate_state(tmp_path / 'run', q, tmp_path / 'eval')


def test_internal_data_cannot_be_unlocked_by_plain_test_label():
    from vdc.contracts import audit_rows
    row = {'observation_id': 'internal-1', 'biological_unit': 'fish-1',
           'study_family': 'private', 'origin': 'internal', 'link_ids': [], 'split': 'test'}
    with pytest.raises(ValueError, match='locked_test'):
        audit_rows([row], 'unit_holdout')
