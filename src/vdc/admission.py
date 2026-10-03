"""Sample-specific internal admission. No default permission is inferred from a filename."""
from pathlib import Path
import os
from .io import read_json, object_hash


def role_manifest(path=None):
    path = path or os.environ.get('VDC_ROLE_MANIFEST')
    if not path:
        raise ValueError('internal development requires VDC_ROLE_MANIFEST; default remains locked_test')
    policy = read_json(Path(path))
    return validate_policy(policy)


def validate_policy(policy):
    if policy.get('protocol') != 'VDC_PK1' or policy.get('status') != 'approved':
        raise ValueError('PK1 sample roles require explicit approval of the saved draft')
    content = {k: v for k, v in policy.items() if k not in {'approval', 'status'}}
    if policy.get('approval', {}).get('content_hash') != object_hash(content):
        raise ValueError('Approved role content changed; review a new draft')
    records = policy.get('samples', [])
    keys = [r['sample_key'] for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate original sample in admission policy')
    if not records:
        raise ValueError('Empty admission policy')
    linked = {}
    for r in records:
        if r['role'] not in {'development', 'reserved_evaluation', 'prediction_only', 'excluded', 'temporal_query'}:
            raise ValueError('Unknown sample role')
        if r['role'] == 'development' and r['split'] not in {'train', 'validation'}:
            raise ValueError('Development policy must name train or validation')
        if r['role'] != 'development' and r['split'] != 'locked_test':
            raise ValueError('Non-development samples must stay locked_test')
        if r['role'] == 'temporal_query':
            if r.get('evaluation_scope') != 'within_cohort_future_only' or not r.get('parent_sample_key'):
                raise ValueError('Temporal query requires a declared parent and limited evaluation scope')
            # Cohort lineage is preserved separately from the harvested pool/library.
            # It cannot establish independent-unit evaluation across time.
            parent = next((s for s in records if s['sample_key'] == r['parent_sample_key']), None)
            if not parent or parent['role'] != 'development' or parent.get('cohort_id') != r.get('cohort_id'):
                raise ValueError('Temporal parent/cohort absent from development')
        for link in r.get('link_ids', []) + [r['sample_key'], r['biological_unit']]:
            previous = linked.setdefault(link, (r['role'], r['split']))
            if previous != (r['role'], r['split']):
                raise ValueError('Original material crosses policy roles/splits: ' + link)
    cohorts = {}
    for r in records:
        if r['role'] == 'development' and r.get('cohort_id'):
            previous = cohorts.setdefault(r['cohort_id'], r['split'])
            if previous != r['split']:
                raise ValueError('Development cohort crosses train/validation')
    return policy, {r['sample_key']: r for r in records}


def validate_internal_row(row, purpose='development'):
    policy, samples = role_manifest()
    key = row.get('source_sample_key')
    record = samples.get(key)
    if not record or row.get('admission_protocol') != policy['protocol']:
        raise ValueError('Internal sample absent from the approved PK1 allowlist')
    if row.get('admission_hash') != policy['approval']['content_hash']:
        raise ValueError('Internal bundle and role policy differ')
    if purpose == 'development':
        if record['role'] != 'development' or row['split'] != record['split']:
            raise ValueError('Reserved/query sample cannot enter development')
    elif purpose not in record.get('allowed_tasks', []):
        raise ValueError('Task is not admitted for this internal sample')
    if row['biological_unit'] != record['biological_unit']:
        raise ValueError('Derived observation changed its original biological unit')
    if not set(record['link_ids'] + [key]).issubset(row['link_ids']):
        raise ValueError('Derived observation dropped original sample/control links')
    return record


def audit_internal_task(rows,task):
    """Check task permission for the development rows actually used by a fitter."""
    for row in rows:
        if row.get('origin')=='internal' and row.get('split') in {'train','validation'}:
            validate_internal_row(row)
            validate_internal_row(row,purpose=task)


def approve_roles(path, reviewer):
    """Explicit local CLI action; never called by a training worker."""
    from .io import write_json
    if not reviewer.strip():
        raise ValueError('Name the reviewer')
    p = Path(path); policy = read_json(p)
    if policy.get('blocking_questions'):
        raise ValueError('Resolve blocking_questions in the concrete role draft first')
    content = {k: v for k, v in policy.items() if k not in {'approval', 'status'}}
    policy.update(status='approved', approval={'reviewer': reviewer,
                  'content_hash': object_hash(content), 'action': 'explicit_local_review'})
    validate_policy(policy)
    write_json(p, policy)
    return policy['approval']['content_hash']
