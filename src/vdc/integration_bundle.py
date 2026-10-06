"""Portable INT1 fitted artifacts. Export never fits and never copies query predictions."""
from pathlib import Path
import shutil
from .io import read_json, write_json, sha256, object_hash
from .pk1_assets import verify_files, checked_member

ROUTES={'bulk':'clock_identity_residual','core':'clock_identity','coarse':'clock_identity'}
SPECIES='Nothobranchius furzeri'
CONTEXT='embryonic_diapause_exit'


def implementation():
    names=('integration_bundle.py','integration.py','integration_knowledge.py','integration_report.py','integration_service.py',
           'clock_wave.py','clock_wave_revision.py','identity.py','identity_protocol.py','application.py',
           'application_data.py','observation.py','response.py','knowledge.py')
    return {n:sha256(Path(__file__).with_name(n)) for n in names}


def seal(root,meta):
    import importlib.metadata,platform
    root=Path(root)
    files={p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='bundle.json'}
    versions={'python':platform.python_version()}
    for package in ('numpy','scipy','torch','transformers','peft','anndata','fastapi'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not_installed'
    manifest={**meta,'files':files,'implementation':implementation(),'format':'VDC-INT1-v1','export_environment':versions}
    manifest['bundle_id']=object_hash(manifest);write_json(root/'bundle.json',manifest)
    return manifest


def load_manifest(root):
    root=Path(root);m=read_json(root/'bundle.json')
    if m['bundle_id']!=object_hash({k:v for k,v in m.items() if k!='bundle_id'}):raise ValueError('Bundle identity changed')
    if m['format']!='VDC-INT1-v1' or m['implementation']!=implementation():raise ValueError('Bundle code contract changed; export with matching release')
    verify_files(root,m['files']);return m


def export_model(stage,output,integration=None):
    stage=Path(stage).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('Export to a new directory')
    snap=read_json(stage/'stage_snapshot.json');verify_files(stage,snap['files'])
    if object_hash({k:v for k,v in snap.items() if k!='snapshot_id'})!=snap['snapshot_id']:raise ValueError('Stage snapshot identity mismatch')
    def copy(source,target):
        source=Path(source);dest=checked_member(out,target);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    views={}
    for view,route in ROUTES.items():
        task='numeric_'+view+'_C2';dest='views/'+view+'/baseline'
        for n in ('model.json','arrays.npz','direct.npz','readout.npz'):copy(stage/'tasks'/task/'model'/n,dest+'/model/'+n)
        for n in ('readout.npz','matched_direct.npz'):copy(stage/'tasks'/('residual_'+view)/n,dest+'/'+n)
        for n in ('wave_parameters.npz','shapes.json'):copy(stage/'tasks'/('gene_waves_'+view)/n,dest+'/'+n)
        copy(stage/'tasks'/('support_'+view)/'result.json',dest+'/support.json')
        model=read_json(out/dest/'model/model.json')
        views[view]={'route':route,'species':SPECIES,'context':CONTEXT,'gene_ids':model['genes'],
                     'baseline':dest,'domain':None,'identity_values':model['types'],'programme_ids':[d['id'] for d in model['definitions']]}
    copy(stage/'tasks/identity_original/reference.json','identity/reference.json')
    meta={'views':views,'source_stage_snapshot':snap['snapshot_id'],'science_status':'experimental_development',
          'knowledge':None,'public_models':{},'regulons':{},'full_ready':False}
    if integration:
        source=Path(integration);fitted=read_json(source/'fit_manifest.json');verify_files(source,fitted['files'])
        if fitted['source_stage_snapshot']!=snap['snapshot_id']:raise ValueError('Integration and baseline stage differ')
        for view in ROUTES:
            dest='views/'+view+'/domain';folder=source/'fits'/view/'domain'
            for n in ('model/model.json','model/arrays.npz','model/direct.npz','model/readout.npz','readout.npz','matched_direct.npz','wave_parameters.npz','shapes.json','semantic.json','semantic_graph.npz'):
                copy(folder/n,dest+'/'+n)
            views[view]['domain']=dest
        for prefix in ('semantics','knowledge','adapter','public','regulons'):
            for p in (source/prefix).rglob('*'):
                if p.is_file():copy(p,p.relative_to(source).as_posix())
        copy(source/'comparisons.json','fit_comparisons.json')
        copy(source/'common_support.json','fit_comparisons_common.json')
        meta.update(knowledge=fitted['knowledge'],public_models=fitted['public_models'],regulons=fitted['regulons'],
                    full_ready=True,fit_manifest_hash=sha256(source/'fit_manifest.json'))
    return seal(out,meta)
