"""Source-backed units and executable input-dependency checks for new R2 tasks."""
from pathlib import Path
import numpy as np
from .io import read_json,write_json,save_npz,sha256
from .research_r2_artifacts import metadata_rows,TASK_FILES
from .research_r2_tools import resolve_assets,inside
from .research_r2_runtime import execute,numeric_npz,audit_groups

UNIT_FIELDS={'replication_unit_id','linked_embryo_key','pool_id','embryo_id','culture_id','sample_id','sample_key','observation_id','library_id'}
UNIT_TYPES={'single_embryo','five_embryo_pool','section_from_embryo','whole_embryo','pool','cell','sample','unknown'}


def verify_lineage(prepared,inv,provenance):
    resolved={};datasets={};out=[]
    for split in ('train','query'):
        records=[]
        for row in read_json(Path(prepared)/(split+'_rows.json')):
            evidence=row.get('unit_evidence',{})
            aid=evidence.get('asset');field=evidence.get('unit_field')
            if aid not in provenance or field not in UNIT_FIELDS:
                raise ValueError('Every task row needs unit_evidence {asset, row_field, row_value, unit_field}; use original metadata and an ID field, not material type')
            if aid not in datasets:
                resolve_assets(inv,[aid]);datasets[aid]=metadata_rows(inv['assets'][aid]['path'])
            matches=[r for r in datasets[aid] if str(r.get(evidence.get('row_field')))==str(evidence.get('row_value'))]
            if len(matches)!=1: raise ValueError('Unit evidence must resolve to exactly one original metadata row')
            src=matches[0];value=str(src.get(field,'')).strip()
            if not value or value.lower() in UNIT_TYPES or value in ('None','nan'):
                raise ValueError('Replication unit is absent or is a material type')
            if not row.get('unit_type') or str(row.get('replication_unit_id'))!=value:
                raise ValueError('Separate unit_type from replication_unit_id; ID must match deposited evidence')
            links=list(row.get('link_ids',[]))
            for key in ('linked_embryo_key','shared_control_id','culture_id'):
                v=src.get(key)
                if v is not None and str(v).strip() not in ('','nan','None'):
                    if str(row.get(key,''))!=str(v): raise ValueError('Preserve original '+key+' in task rows')
                    links.append(inv['assets'][aid]['source']+':'+key+':'+str(v))
            records.append({**row,'study_family':inv['assets'][aid]['source'],
                'biological_unit':value,'link_ids':sorted(set(links)),
                'unit_audit':'original_metadata_ID_and_links; independent_replication_not_proven'})
            resolved[aid]=sha256(inv['assets'][aid]['path'])
        out.append(records)
    audit_groups(*out)
    return out,{'status':'source_linked','metadata_sha256':resolved,'independent_replication_proven':False}


def verify_dependency(prepared,contract,inv,provenance,code,out,runtime,repo):
    definition=contract.get('input_dependency',{})
    if definition.get('kind')=='observed_covariates':
        return {'status':'target_isolation_not_claimed','interpretation':'conditional prediction from declared observed features; not a strict hidden-target benchmark'}
    if definition.get('kind')!='target_isolated':
        raise ValueError('Declare input_dependency.kind as observed_covariates or target_isolated; strict isolation requires a source-backed feature probe')
    import pandas as pd
    aid=definition.get('matrix_asset')
    if aid not in provenance: raise ValueError('Dependency probe matrix must be a mounted original asset')
    resolve_assets(inv,[aid])
    # Explicit first implementation: gene-by-sample deposited CSV/TSV. Unsupported formats fail visibly.
    path=Path(inv['assets'][aid]['path'])
    if not path.name.endswith(('.csv','.csv.gz','.tsv','.tsv.gz')):
        raise ValueError('Strict source-backed probe currently supports gene-by-sample CSV/TSV; implement a source adapter for other formats')
    frame=pd.read_csv(path,index_col=0,sep='\t' if '.tsv' in path.name else ',')
    frame.columns=frame.columns.astype(str);frame.index=frame.index.astype(str)
    rows=read_json(Path(prepared)/'train_rows.json')+read_json(Path(prepared)/'query_rows.json')
    columns=[r['matrix_column'] for r in rows]
    if not frame.index.is_unique or not frame.columns.is_unique or len(set(columns))!=len(columns):
        raise ValueError('Ambiguous probe matrix row/column mapping')
    raw=frame.loc[:,columns].to_numpy(dtype=float).T
    if not np.isfinite(raw).all() or (raw<0).any(): raise ValueError('Count-scale dependency probe requires finite nonnegative source values')
    schema=read_json(Path(prepared)/'schema.json')
    targets=definition.get('target_genes',[])
    if set(targets)!=set(schema['targets']) or not targets: raise ValueError('Probe targets must exactly match the sealed target schema')
    ti=frame.index.get_indexer(targets)
    if (ti<0).any(): raise ValueError('Probe targets missing from original matrix')
    if not (Path(code)/'feature_probe.py').is_file(): raise ValueError('Preparation code must provide feature_probe.py')
    expected=np.vstack([numeric_npz(Path(prepared)/'train.npz',('x',))['x'],numeric_npz(Path(prepared)/'query.npz',('x',))['x']])
    train_n=len(numeric_npz(Path(prepared)/'train.npz',('x',))['x'])
    observed=[]
    seed=int(sha256(path)[:8],16);rng=np.random.default_rng(seed)
    for name in ('original','changed_targets','changed_visible'):
        values=raw.copy()
        indices=ti if name=='changed_targets' else np.setdiff1d(np.arange(raw.shape[1]),ti)
        if name!='original': values[:,indices]=values[:,indices]*rng.uniform(.1,7,(len(values),len(indices)))+rng.uniform(1,100,(len(values),len(indices)))
        inp=Path(out)/name/'input.npz'
        save_npz(inp,counts=values)
        meta=Path(out)/'probe_schema.json'
        write_json(meta,{'genes':list(frame.index),'targets':targets,'features':schema['features'],'train_rows':train_n})
        execute(code,{'probe.npz':inp,'probe_schema.json':meta},Path(out)/name/'out',
                ['python','/code/feature_probe.py'],runtime,repo=repo)
        observed.append(numeric_npz(Path(out)/name/'out/features.npz',('x',))['x'])
    if any(x.shape!=expected.shape for x in observed): raise ValueError('Dependency probe feature shape differs from task')
    if not np.allclose(observed[0],expected,rtol=1e-6,atol=1e-8):
        raise ValueError('Probe does not reproduce actual prepared features from the original matrix')
    delta=float(np.max(np.abs(observed[0]-observed[1])))
    if not np.allclose(observed[0],observed[1],rtol=1e-7,atol=1e-9):
        raise ValueError('Changing hidden targets changed visible features; preprocessing violates declared target isolation')
    if np.allclose(observed[0],observed[2],rtol=1e-7,atol=1e-9):
        raise ValueError('Feature probe ignored visible-source perturbation; constant/replayed features are not an executable transform')
    return {'status':'passed_counterfactual_probe','max_hidden_target_delta':delta,'source_sha256':sha256(path),
        'scope':'declared targets and exact feature transform tested; not proof of all scientific assumptions'}


def verified_bundle(prepared,contract,inv,provenance,code,out,runtime,repo):
    prepared=Path(prepared); out=Path(out)
    missing=[n for n in TASK_FILES if not (prepared/n).is_file()]
    if missing: raise ValueError('Preparation is not task_ready; missing '+', '.join(missing))
    rows,units=verify_lineage(prepared,inv,provenance)
    dependency=verify_dependency(prepared,contract,inv,provenance,code,out/'dependency_probe',runtime,repo)
    import shutil
    audited=out/'audited_bundle';audited.mkdir(parents=True,exist_ok=False)
    for name in TASK_FILES: shutil.copyfile(prepared/name,audited/name)
    for split,records in zip(('train','query'),rows):write_json(audited/(split+'_rows.json'),records)
    receipt={'lineage':units,'input_dependency':dependency,'prepared_files':{n:sha256(prepared/n) for n in TASK_FILES}}
    write_json(out/'contract_audit.json',receipt)
    return audited,receipt
