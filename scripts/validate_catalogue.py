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
    core=read(root,'knowledge/core_tasks.json')
    require(core['version']=='0.2.0' and core['status']=='definition_only','Unexpected core definition version/status')
    require(catalogue['layer']=='functional_annotation','Functional domains must retain their interpretation role')
    require(catalogue['core_tasks_path']=='knowledge/core_tasks.json','Broken core definition link')
    cids=unique(core['core_tasks'],'core task')
    require(cids=={'DC01','DC02','DC03','DC04'},'Missing core task')
    require(core['organising_axis']=='DC02' and set(core['co_primary_tasks'])=={'DC02','DC03','DC04'},'Clock and waves must remain co-primary')
    for task in core['core_tasks']:
        require(set(task['functional_links']) <= mids,'Unknown functional link')
        require(task['implemented'] is False and bool(task['required_outputs']),'Core algorithms are not yet implemented')
    axes=core['coordinate_registry']
    axis_ids=unique(axes,'coordinate')
    require(axis_ids=={'whole_embryo_biotime','biotime_bulk','biotime_sc','cell_type_specific_biotime','M4_exit_clock'},'Coordinate identities must remain distinct')
    for axis in axes:
        require(axis['axis_artifact_imported'] is False and axis['training_eligible'] is False,'Internal coordinate artifacts are not imported or admitted')
    depth=core['depth_policy']
    require(depth['derivation']=='requires_measured_response' and depth['history_is_depth'] is False and depth['depth_equals_one_minus_clock'] is False,'Depth cannot be derived from duration or inverse clock')
    require(depth['missing_response_output']=='unavailable','Missing depth observations must remain unavailable')
    wave=core['wave_policy']
    require(wave['preserve_unstandardised_amplitude'] is True and wave['enrichment_p_is_amplitude'] is False,'Wave amplitude must be measured, not enrichment significance')
    require(wave['lag_requires_identifiability'] is True and wave['tf_rna_is_tf_activity'] is False,'Wave interpretation boundary violated')
    provenance=core['design_provenance']
    for field in ('extracted_text_published','internal_result_tables_published','internal_artifacts_imported','internal_training_authorized','claim_of_blind_design'):
        require(provenance[field] is False,'Internal design review does not authorize data use or blind-design claims')
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
    return dict(status='passed',scope='definition_consistency_only',core_tasks=len(cids),coordinate_definitions=len(axes),modules=len(modules),submodules=len(subids),
                sources=len(sources),evidence_notes=len(evidence),datasets=len(datasets),go_terms=len(ids),
                expression_datasets_downloaded=0,training_ready=False,model_training_executed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    result=validate(p.parse_args().root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
