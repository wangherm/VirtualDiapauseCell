"""Read-only source checks and explicit reuse of previously executed Alpha assets."""
from pathlib import Path
import shutil
import numpy as np
from .io import read_json,write_json,sha256,object_hash
from .admission import role_manifest
from .contracts import ObservationBundle


def checked_member(root,name):
    root=Path(root).resolve();p=(root/name).resolve()
    if not p.is_relative_to(root):raise ValueError('Artifact path escapes its declared root')
    return p


def verify_files(root,files):
    if not files:raise ValueError('Empty source inventory')
    for name,expected in files.items():
        p=checked_member(root,name)
        if not p.is_file() or sha256(p)!=expected:raise ValueError('Source file missing or changed: '+name)


def private_audit(root,out):
    root=Path(root);policy,_=role_manifest(root/'sample_roles.json')
    lock=read_json(root/'source_lock.json')
    if object_hash(lock)!=policy['source_lock_hash']:raise ValueError('Approved source lock changed')
    verify_files(root/'input',lock['files'])
    views={};ids=None
    for name in ('bulk','core','core_celltypes'):
        b=ObservationBundle.load(root/'prepared'/name)
        if any(r['origin']!='internal' or r['split'] not in {'train','validation'} for r in b.rows):raise ValueError('Private development views required')
        if ids is None:ids=b.feature_ids
        if ids!=b.feature_ids:raise ValueError('Private programme orders differ')
        contract=read_json(root/'prepared'/name/'expression_contract.json')
        if sha256(root/'prepared'/name/'gene_expression.npz')!=contract['expression_sha256']:raise ValueError('Expression changed')
        views[name]={'fingerprint':b.fingerprint,'observations':len(b.rows),
            'independent_units':len(set(r['biological_unit'] for r in b.rows)),'split_counts':{s:len(b.indices(s)) for s in ('train','validation')}}
    report={'views':views,'feature_ids':ids,'role_hash':policy['approval']['content_hash'],
        'reserved_used_for_fitting':False,'source_hash_audit_reads_bytes_without_model_fitting':True}
    write_json(Path(out)/'audit.json',report);return report


def alpha_module(alpha,name):
    alpha=Path(alpha);m=read_json(alpha/'module_status.json')['modules'][name]
    if m['execution_status']!='completed':raise ValueError('Required Alpha source was not completed: '+name)
    folder=checked_member(alpha,m['artifact']);verify_files(folder,m['files'])
    return folder,m


def import_public(alpha,out):
    out=Path(out);sources={}
    for name,relative in [('data_dauer','data/dauer'),('data_ard','data/ard'),('clock_reference','clock_reference')]:
        folder,m=alpha_module(alpha,name)
        shutil.copytree(folder,out/relative)
        sources[name]={'files':m['files'],'source_artifact':m['artifact']}
    for name in ('data/dauer','data/ard','clock_reference/bundle'):
        b=ObservationBundle.load(out/name)
        if any(r['origin']!='public' or r['split'] not in {'train','validation'} for r in b.rows):raise ValueError('Only existing public development may enter PK1')
    write_json(out/'source_reuse.json',{'kind':'verified_existing_public_development','sources':sources,
        'new_public_studies_added':False,'old_holdout_roles_changed':False})


def reuse_knowledge(alpha,out):
    folder,m=alpha_module(alpha,'knowledge');out=Path(out)
    status=read_json(folder/'status.json')
    if status.get('optimizer_steps',0)<1:raise ValueError('No executed domain training in source status')
    adapter=folder/'adapter'
    if not (adapter/'adapter_model.safetensors').is_file():raise ValueError('Actual Alpha adapter weights missing; a report-only archive is insufficient')
    shutil.copytree(adapter,out/'adapter')
    write_json(out/'provenance.json',{'mode':'reuse_existing_trained_alpha_adapter',
        'source_training_status':status,'source_inventory_hash':object_hash(m['files']),
        'adapter_files':{p.name:sha256(p) for p in (out/'adapter').iterdir() if p.is_file()},
        'new_qwen_training_executed':False,'expanded_knowledge_corpus_completed':False,
        'limitation':'Alpha small source-curated pilot; not the planned 20-30 family knowledge expansion'})
