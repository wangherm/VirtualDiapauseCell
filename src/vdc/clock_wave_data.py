"""One cached development-only count extraction, with immutable source roles."""
from pathlib import Path
import copy, os
import numpy as np
from .io import read_json, write_json, save_npz, sha256, object_hash
from .admission import role_manifest, audit_internal_task
from .contracts import ObservationBundle, audit_rows
from .application_data import development_counts
from .pk1_assets import verify_files


def canonical(label, crosswalk):
    label = crosswalk.get('aliases', {}).get(label, label)
    matches = [k for k, labels in crosswalk['groups'].items() if label in labels]
    if len(matches) > 1: raise ValueError('Ambiguous identity crosswalk')
    return matches[0] if matches else 'unknown'


def coarsen(counts, rows, crosswalk):
    groups = {}; omitted = []
    for i, row in enumerate(rows):
        typ = canonical(row['cell_type'], crosswalk)
        if typ == 'unknown': omitted.append(row['observation_id']); continue
        groups.setdefault((row['source_sample_key'], typ), []).append(i)
    values = []; newrows = []
    for (key, typ), idx in sorted(groups.items()):
        r = copy.deepcopy(rows[idx[0]])
        if len({rows[i]['split'] for i in idx}) != 1: raise ValueError('Identity aggregation crosses splits')
        r.update(observation_id=key+':coarse:'+typ, coarse_identity=typ, cell_type=typ,
                 cell_count=sum(rows[i].get('cell_count', 0) for i in idx),
                 member_observation_ids=[rows[i]['observation_id'] for i in idx])
        values.append(counts[idx].sum(0)); newrows.append(r)
    if not values: raise ValueError('No explicit coarse identities mapped')
    return np.array(values), newrows, omitted


def prepare(private, out, crosswalk):
    private, out = Path(private), Path(out)
    os.environ['VDC_ROLE_MANIFEST'] = str(private/'sample_roles.json')
    policy, _ = role_manifest(); lock = read_json(private/'source_lock.json')
    if object_hash(lock) != policy['source_lock_hash']: raise ValueError('Role/source identity mismatch')
    verify_files(private/'input', lock['files'])
    write_json(out/'sample_roles.json', policy)
    audit = {'role_hash':policy['approval']['content_hash'], 'source_lock_hash':object_hash(lock),
             'source_files':lock['files'], 'crosswalk_hash':object_hash(crosswalk), 'views':{},
             'reserved_read_for_modelling':False, 'historical_biotime_used':False}
    # Core file is opened for two aggregations once at preparation, never once per fit.
    for view in ('bulk','core','core_celltypes'):
        prep = private/'prepared'/view; b = ObservationBundle.load(prep)
        audit_internal_task(b.rows, 'state')
        if any(r['split'] not in {'train','validation'} for r in b.rows): raise ValueError('Nondevelopment prepared row')
        contract = read_json(prep/'expression_contract.json'); definitions = read_json(prep/'programmes.json')
        if object_hash(definitions) != contract['context']['programme_definition_id']: raise ValueError('Programme members changed')
        genes = contract['gene_ids']; counts = development_counts(private, policy, view, b, genes)
        if not np.equal(counts, np.rint(counts)).all(): raise ValueError('Fractional inputs cannot be silently cast to counts')
        rows = b.rows
        if view == 'core_celltypes':
            counts, rows, omitted = coarsen(counts, rows, crosswalk); view = 'coarse'
        else: omitted = []
        folder = out/view
        save_npz(folder/'counts.npz', counts=counts.astype(np.int64))
        write_json(folder/'data.json', {'genes':genes, 'definitions':definitions, 'rows':rows,
                    'source_prepared_fingerprint':b.fingerprint, 'counts_sha256':sha256(folder/'counts.npz'),
                    'omitted_unmapped_annotations':omitted,
                    'aggregation':'sum raw counts of eligible source type profiles per pool, then normalize; source min-cell filter retained' if view=='coarse' else 'existing development units'})
        audit['views'][view] = {'profiles':len(rows), 'units':len({r['biological_unit'] for r in rows}),
                               'data_sha256':sha256(folder/'data.json'), 'counts_sha256':sha256(folder/'counts.npz')}
    write_json(out/'audit.json', audit)
    return audit


def load_data(folder):
    folder = Path(folder); m = read_json(folder/'data.json')
    if sha256(folder/'counts.npz') != m['counts_sha256']: raise ValueError('Cached counts changed')
    with np.load(folder/'counts.npz', allow_pickle=False) as z: counts = z['counts'].copy()
    return counts, m['genes'], m['definitions'], copy.deepcopy(m['rows'])


def split_fold(rows, policy, held_unit, path):
    """Close over every material link and cohort, preserving original role manifest."""
    records = {r['sample_key']:r for r in policy['samples'] if r['role']=='development'}
    selected = {r['source_sample_key'] for r in rows if r['biological_unit']==held_unit}
    if not selected: raise ValueError('Unknown fold unit')
    def links(r):
        return set(r['link_ids']+[r['sample_key'],r['biological_unit']]+([r['cohort_id']] if r.get('cohort_id') else []))
    changed = True
    while changed:
        old = set(selected); inherited = set().union(*(links(records[k]) for k in selected))
        selected |= {k for k,r in records.items() if links(r) & inherited}; changed = old != selected
    assignments = {k:('validation' if k in selected else 'train') for k in records}
    f = {'protocol':'PK2_development_resampling', 'parent_approval_hash':policy['approval']['content_hash'],
         'assignments':assignments, 'held_unit':held_unit, 'grouping':'transitive material/cohort closure'}
    f['hash'] = object_hash(f); write_json(path, f)
    os.environ['VDC_DEVELOPMENT_FOLD'] = str(Path(path).resolve())
    result = copy.deepcopy(rows)
    for r in result: r.update(split=assignments[r['source_sample_key']], development_fold_hash=f['hash'])
    audit_rows(result, 'unit_holdout'); audit_internal_task(result, 'state')
    return result
