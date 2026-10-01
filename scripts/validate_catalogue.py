"""Validate the definition snapshot offline. This is not training authorization."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def read(root, name):
    return json.loads((root / name).read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def unique(rows, label):
    ids=[r['id'] for r in rows]
    require(len(ids)==len(set(ids)), f'Duplicate {label} IDs')
    return set(ids)

def validate(root=ROOT):
    root=Path(root)
    catalogue=read(root,'knowledge/modules.json')
    modules=catalogue['modules']
    sources=read(root,'knowledge/sources.json')['sources']
    evidence=read(root,'knowledge/evidence_seed.json')['records']
    datasets=read(root,'knowledge/datasets.json')['datasets']
    mids=unique(modules,'module'); sids=unique(sources,'source')
    unique(evidence,'evidence'); unique(datasets,'dataset')
    subids=set()
    for s in sources:
        require(s['url'].startswith('https://'),f"Missing source URL: {s['id']}")
        require(s['training_eligible'] is False, 'Definition release must not claim training-ready sources')
    for m in modules:
        require(re.fullmatch(r'M\d{2}',m['id']), 'Invalid module ID')
        require(m['functional_question'] and m['observables'] and m['do_not_infer'], 'Incomplete module definition')
        require(set(m['source_ids']) <= sids, f"Unknown source in {m['id']}")
        require(m['training_ready'] is False and m['gene_membership']==[], 'Gene membership is not yet audited')
        for s in m['submodules']:
            require(s['id'].startswith(m['id']+'.') and s['id'] not in subids, 'Duplicate or misparented submodule')
            subids.add(s['id'])
    for e in evidence:
        require(e['source_id'] in sids and set(e['module_ids']) <= mids, 'Broken evidence reference')
        for key in ('context','intervention','evidence_type','measured_outcome','branch','source_locator','paper_family','limits'):
            require(bool(e.get(key)), f"Missing {key} in {e['id']}")
        require(e['training_eligible'] is False, 'Unreviewed seed notes cannot become gold')
        require(e['review_status']=='needs_domain_review' and e['reviewed_by'] is None, 'Unexpected review claim')
        require(e['partition']=='unassigned' and e['internal_derived'] is False, 'Evidence split/internal boundary violated')
    for d in datasets:
        require(d['source_id'] is None or d['source_id'] in sids, 'Unknown dataset source')
        require(d['training_eligible'] is False, 'No dataset has completed admission')
        require(d['downloaded'] is False and d['files']==[], 'This definition snapshot has no expression files')
        require(bool(d['admission_blockers']), 'Missing dataset admission audit')
        if d['role']=='frozen_test':
            require(d['split']=='frozen_test' and d['status']=='locked','Frozen test boundary violated')
        else:
            require(d['split']=='unassigned' and d['status']=='candidate_not_admitted', 'Unreviewed dataset admitted')
    snapshot=(root/catalogue['go_snapshot']).resolve()
    require(snapshot.is_relative_to(root.resolve()), 'Snapshot path escapes repository')
    meta=read(snapshot,'manifest.json')
    raw=(snapshot/'quickgo_response.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==meta['sha256'], 'GO snapshot checksum mismatch')
    terms=json.loads(raw)['results']
    ids={x['id'] for x in terms}
    needed={g for m in modules for g in m['go_anchors']}
    require(ids==needed==set(meta['ids']), 'GO IDs do not match module anchors')
    require(len(terms)==len(ids)==meta['term_count'], 'GO count mismatch')
    require(not any(t.get('isObsolete') for t in terms), 'Obsolete GO term requires review')
    require(meta['gene_membership_downloaded'] is False,'Ontology terms are not gene sets')
    return dict(status='passed',scope='definition_consistency_only',modules=len(modules),submodules=len(subids),
                sources=len(sources),evidence_notes=len(evidence),datasets=len(datasets),go_terms=len(ids),
                expression_datasets_downloaded=0,training_ready=False,model_training_executed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    result=validate(p.parse_args().root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
