"""Versioned development inputs and knowledge, shared by all research candidates."""
from pathlib import Path
import numpy as np
from .io import read_json, read_jsonl, write_json, sha256, object_hash, save_npz
from .clock_wave_data import load_data
from .clock_wave import inputs, hidden_partition, anchor_axis, NotIdentifiable, identities
from .contracts import audit_rows
from .admission import audit_internal_task, role_manifest
from .observation import normalise_expression


def connected_rows(rows, policy):
    """Close pool, control and cohort links before weighting/evaluation."""
    records = {r['sample_key']: r for r in policy['samples']}
    parents = {}
    def root(k):
        parents.setdefault(k, k)
        if parents[k] != k:
            parents[k] = root(parents[k])
        return parents[k]
    def keys(r):
        p = records.get(r.get('source_sample_key'), {})
        return list(set(r['link_ids']+p.get('link_ids', []) +
                        [r['study_family']+':'+r['biological_unit']] +
                        ([p['cohort_id']] if p.get('cohort_id') else [])))
    for r in rows:
        ks = keys(r)
        for k in ks[1:]:
            parents[root(k)] = root(ks[0])
    groups = {}; result = []
    for r in rows:
        group = root(keys(r)[0])
        if groups.setdefault(group, r['split']) != r['split']:
            raise ValueError('Linked material/cohort crosses splits')
        result.append({**r, 'biological_unit': group})
    return result


def source_fingerprint(data, config):
    files = {}
    for view in config['views']:
        for name in ('data.json', 'counts.npz'):
            p = Path(data)/view/name
            files[view+'/'+name] = sha256(p)
    policy, _ = role_manifest()
    return {'files': files, 'role_hash': object_hash(policy)}


def prepare_view(folder, config, out):
    counts, genes, definitions, rows = load_data(folder)
    if any(r['split'] not in {'train', 'validation'} for r in rows):
        raise ValueError('Only existing authorized development rows are admitted')
    if any(r['origin'] == 'synthetic' for r in rows):
        raise ValueError('Research run requires real observations')
    audit_rows(rows, 'unit_holdout'); audit_internal_task(rows, 'state')
    policy, _ = role_manifest()
    grouped = connected_rows(rows, policy)
    tr = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    va = np.array([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    if len(tr) < 3 or not len(va):
        raise ValueError('Insufficient train/validation rows')
    y, scores, amplitude = inputs(counts, genes, definitions)
    hidden = hidden_partition(genes)
    full = normalise_expression(counts, np.ones_like(counts, bool), 'counts')
    target_ids = sorted([i for i,g in enumerate(genes) if g in hidden and full[tr,i].var()>1e-8],
                        key=lambda i: (-full[tr,i].var(), i))[:config['target_genes']]
    visible = sorted([i for i,g in enumerate(genes) if g not in hidden and y[tr,i].var()>1e-8],
                     key=lambda i: (-y[tr,i].var(), i))[:config['visible_genes']]
    if not target_ids or not visible:
        raise ValueError('No eligible train-variable genes')
    ty = identities(rows); classes = sorted({ty[i] for i in tr})
    onehot = np.array([[float(t == k) for k in classes]+[float(t not in classes)] for t in ty])
    try:
        clock, reference = anchor_axis(y, genes, rows)
    except NotIdentifiable as exc:
        clock = np.full(len(rows), np.nan); reference = {'unavailable': str(exc)}
    available = np.isfinite(clock)[:, None]
    features = {'rank': scores['values'], 'amplitude': amplitude,
                'coverage': scores['coverage'], 'programme_available': scores['mask'].astype(float),
                'genes': y[:, visible], 'identity': onehot,
                'clock': np.where(available, clock[:, None], 0.),
                'clock_available': available.astype(float)}
    write_json(out/'preprocessing.json', {'target_genes': [genes[i] for i in target_ids],
        'visible_genes': [genes[i] for i in visible], 'identity_classes': classes,
        'reference': reference, 'target_scale': 'log1p(full-library counts per 10000)',
        'input_scale': 'target-excluded library normalization and ranks',
        'missing_clock_encoding': '0 plus availability=0; not an estimated clock',
        'feature_selection': 'training rows only', 'heldout_used_for_fit': False})
    save_npz(out/'train.npz', **{k:v[tr] for k,v in features.items()}, target=full[np.ix_(tr,target_ids)])
    save_npz(out/'query.npz', **{k:v[va] for k,v in features.items()})
    save_npz(out/'truth.npz', target=full[np.ix_(va,target_ids)])
    write_json(out/'train_rows.json', [grouped[i] for i in tr])
    write_json(out/'query_rows.json', [grouped[i] for i in va])
    info = {'training_profiles': len(tr), 'validation_profiles': len(va),
            'training_units': len({grouped[i]['biological_unit'] for i in tr}),
            'validation_units': len({grouped[i]['biological_unit'] for i in va}),
            'target_genes': len(target_ids), 'visible_genes': len(visible),
            'features': {k:v.shape[1] for k,v in features.items()},
            'train_clock_available': int(available[tr].sum()),
            'validation_clock_available': int(available[va].sum()),
            'evaluation_scope': 'adaptive_development_not_independent_confirmation'}
    write_json(out/'summary.json', info)
    return info


def knowledge_records(path):
    seen = set(); result = []
    for r in read_jsonl(path):
        if r.get('split') != 'train' or r.get('reviewed') is not True:
            continue
        text = r.get('text', '').strip(); source = r.get('source_ref', '')
        if not text or not source.startswith('https://') or 'drive.google' in source:
            continue
        key = (source, text)
        if key in seen: continue
        seen.add(key)
        result.append({'id': 'E'+str(len(result)+1), 'source': source,
                       'locator': r.get('source_locator'), 'evidence': text,
                       'evidence_type': r.get('evidence_type'),
                       'quality': r.get('label_source', 'curated_source')})
    if not result: raise ValueError('No reviewed training evidence in corpus')
    return result


def retrieve(records, query, limit=20):
    """Local lexical retrieval; no new web results or invented citations."""
    import re
    words = set(re.findall(r'\w+', query.lower()))
    ranked = sorted(records, key=lambda r: (-len(words & set(re.findall(r'\w+', r['evidence'].lower()))), r['id']))
    return ranked[:limit]
