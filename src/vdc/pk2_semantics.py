"""Strict PK2 semantic conditions shared by dispatch, training and publication."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .io import object_hash, read_json, sha256
from .pk1_numeric import semantic_cache


def semantic_mode(job):
    mode = job.get('semantics', 'zero')
    aliases = {'base': 'base', 'new_domain': 'domain', 'shuffled_new_domain': 'shuffle',
               'domain': 'domain', 'shuffle': 'shuffle', 'zero': 'zero'}
    if mode not in aliases or (job.get('family') == 'semantic_increment' and mode == 'zero'):
        raise ValueError('Unknown or missing PK2 semantic condition: ' + str(mode))
    return aliases[mode]


def matrix_hash(matrix):
    x = np.asarray(matrix, dtype='<f4', order='C')
    return object_hash({'shape': list(x.shape), 'float32_sha256': hashlib.sha256(x.tobytes()).hexdigest()})


def validate_semantics(job, matrix, provenance, features, dimension):
    mode = semantic_mode(job)
    x = np.asarray(matrix, dtype=np.float32)
    if x.shape != (len(features), dimension) or not np.isfinite(x).all():
        raise ValueError('Semantic shape or finite-value contract failed')
    if mode == 'zero':
        if np.any(x): raise ValueError('Zero condition contains nonzero semantics')
    else:
        expected = 'frozen_base' if mode == 'base' else 'domain_adapter'
        if not provenance or provenance.get('weight_kind') != expected or not provenance.get('arrays_sha256'):
            raise ValueError('Semantic cache provenance missing or wrong model kind')
        if not np.any(x) or not np.all(np.any(x != 0, axis=1)):
            raise ValueError('Nonzero semantic condition contains zero vectors')
        if mode != 'base' and not provenance.get('adapter_hash'):
            raise ValueError('Domain semantics require actual adapter identity')
        if provenance.get('feature_ids') != features:
            raise ValueError('Semantic feature order was not verified')
        if mode == 'shuffle':
            perm = provenance.get('permutation', [])
            if sorted(perm) != list(range(len(features))) or perm == list(range(len(features))):
                raise ValueError('Shuffle must have a nonidentity bijection')
            if provenance.get('unshuffled_matrix_hash') == matrix_hash(x):
                raise ValueError('Shuffle did not change the semantic matrix')
    return {'condition': mode, 'matrix_hash': matrix_hash(x), 'feature_ids': features,
            'shape': list(x.shape), 'nonzero_entries': int(np.count_nonzero(x)),
            'rms': float(np.sqrt(np.mean(x.astype(float)**2))), 'provenance': provenance}


def load_semantics(base, job, features, config):
    mode = semantic_mode(job)
    if mode == 'zero': return None, None
    base = Path(base)
    which = 'base' if mode == 'base' else 'domain'
    matrix, provenance = semantic_cache(base / ('embeddings_' + which), features,
                                       'frozen_base' if mode == 'base' else 'domain_adapter')
    provenance = {**provenance, 'feature_ids': features}
    if mode != 'base':
        selected = read_json(base / 'select_adapter/selection.json')['selected']['job']
        adapter = base / selected / 'trained/adapter'
        actual = object_hash({p.name: sha256(p) for p in sorted(adapter.iterdir()) if p.is_file()})
        if actual != provenance['adapter_hash']:
            raise ValueError('Embedding cache was not computed from the selected adapter')
    if mode == 'shuffle':
        permutation = np.random.default_rng(config['semantic_shuffle_seed']).permutation(len(matrix))
        provenance.update(permutation=permutation.tolist(), ablation='shuffled_domain',
                          unshuffled_matrix_hash=matrix_hash(matrix))
        matrix = matrix[permutation]
    validate_semantics(job, matrix, provenance, features, config['semantic_dim'])
    return matrix, provenance


def check_forward(predictor, bundle, expected):
    """Verify the saved buffer and test its actual influence using identical input/weights."""
    from .pk2_numeric import query
    buffer = predictor.model.semantics
    if matrix_hash(buffer.detach().cpu().numpy()) != matrix_hash(expected):
        raise ValueError('Loaded checkpoint semantic buffer differs from the requested condition')
    original = query(predictor, bundle)
    saved = buffer.detach().clone()
    try:
        with torch.no_grad(): buffer.zero_()
        control = query(predictor, bundle)
    finally:
        with torch.no_grad(): buffer.copy_(saved)
    deltas = {k: float(np.max(np.abs(original[k] - control[k]))) for k in ('programme', 'latent')}
    if not all(np.isfinite(v) for v in deltas.values()) or max(deltas.values()) == 0:
        raise ValueError('Nonzero semantic buffer has no measurable forward effect')
    return {'status': 'passed', 'comparison': 'same_checkpoint_actual_buffer_vs_zero',
            'max_abs_delta': deltas, 'scientific_benefit_claim': False}


def preflight(bundle, config, matrix):
    from .state import ProgrammeStateModel
    idx=bundle.indices('train')[:2]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model=ProgrammeStateModel(len(bundle.feature_ids),config,torch.as_tensor(matrix)).eval()
        x=torch.tensor(bundle.values[idx]);m=torch.tensor(bundle.mask[idx]);c=torch.tensor(bundle.coverage[idx])
        with torch.inference_mode():
            actual=model(x,m,c)['programme'];model.semantics.zero_();control=model(x,m,c)['programme']
        delta=float((actual-control).abs().max())
    if not np.isfinite(delta) or delta==0:raise ValueError('Pretraining semantic forward check failed')
    return {'status':'passed','train_observations':len(idx),'max_abs_delta':delta,
            'meaning':'Real input and cached vectors; untrained numerical head connectivity only'}
