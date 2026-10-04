"""Descriptive stress substitute on archived normalised RNA; never a count-model input."""
from pathlib import Path
import numpy as np
from .io import read_json,write_json,save_npz,sha256,object_hash
from .pk1_data import h5_column
from .observation import score_programmes
from .pk1_assets import verify_files


def prepare_normalised(file,record,policy,definitions,out,config):
    import h5py
    out=Path(out);rows={'stress_train':[],'stress_queries':[]};data={k:[] for k in rows};coverage={k:[] for k in rows};masks={k:[] for k in rows}
    rule=config['stress_substitute'];matching=rule['reference_conditions']
    allowed=[r for r in policy['samples'] if
        (r['source']=='core' and r['role']=='development' and r['split']=='train' and r['condition'] in matching.values()) or
        (r['source']=='sc_reserved' and r['role'] in config['allowed_query_roles'] and r['condition'] in matching)]
    missing=[]
    with h5py.File(file,'r') as f:
        names=h5_column(f['obs/sample']);genes=h5_column(f['raw/var/_index']).tolist();matrix=f['raw/X'];ptr=matrix['indptr'][:]
        if matrix.attrs.get('encoding-type')!='csr_matrix':raise ValueError('Explicit archived normalised CSR raw.X required')
        for r in allowed:
            idx=np.flatnonzero(names==r['sample_id'])
            if not len(idx):missing.append({'sample_key':r['sample_key'],'reason':'absent_from_normalised_archive'});continue
            total=np.zeros(len(genes))
            for i in idx:
                a,b=ptr[i:i+2];v=matrix['data'][a:b]
                if not np.isfinite(v).all() or (v<0).any():raise ValueError('Nonnegative normalised raw.X required; scaled X is prohibited')
                np.add.at(total,matrix['indices'][a:b],v)
            # Mean of archived normalised values, not pseudobulk UMI counts.
            s=score_programmes((total/len(idx))[None,:],np.ones((1,len(genes)),bool),genes,definitions,
                config['min_genes_per_programme'],config['min_programme_coverage'])
            k='stress_train' if r['split']=='train' else 'stress_queries'
            rows[k].append({**r,'cell_count':len(idx)});data[k].append(s['values'][0]);coverage[k].append(s['coverage'][0]);masks[k].append(s['mask'][0])
    files={}
    for k in rows:
        if rows[k]:
            save_npz(out/(k+'.npz'),values=np.asarray(data[k]),mask=np.asarray(masks[k]),coverage=np.asarray(coverage[k]),feature_ids=np.array([p['id'] for p in definitions]))
            files[k+'.npz']=sha256(out/(k+'.npz'))
    return {'rows':rows,'files':files,'unavailable':missing,'source_registry':record,'source_sha256':sha256(file),
        'definition_hash':object_hash(definitions),'aggregation':'within-pool mean of archived normalised raw.X, then within-profile rank programme score',
        'scale':'archived_normalised_expression_not_counts','counts_status':'pending_user_source',
        'claim':'descriptive stress/control difference; composition/batch/preprocessing confounding unresolved',
        'clock':None,'depth':None,'counts_model_used':False}


def fit_reference(supplement,snapshot,policy,config):
    root=Path(supplement);m=read_json(root/'manifest.json');s=m.get('stress_substitute')
    if not s:return {'status':'unavailable','reason':'normalised_stress_supplement_absent','raw_counts':'pending_user_source'}
    verify_files(root,{'stress_train.npz':s['files']['stress_train.npz']})
    original={r['sample_key']:r for r in policy['samples']};rows=s['rows']['stress_train']
    for r in rows:
        if r['sample_key'] not in original or r!= {**original[r['sample_key']],'cell_count':r['cell_count']} or r['role']!='development' or r['split']!='train':raise ValueError('Stress reference must use unchanged original train rows only')
    with np.load(root/'stress_train.npz',allow_pickle=False) as a:
        y=a['values'];mask=a['mask'];features=a['feature_ids'];conditions=sorted(set(r['condition'] for r in rows));means=[];available=[]
        for condition in conditions:
            ix=[i for i,r in enumerate(rows) if r['condition']==condition];n=mask[ix].sum(0)
            means.append(np.where(mask[ix],y[ix],0).sum(0)/np.maximum(n,1));available.append(n>0)
    save_npz(Path(snapshot)/'stress/reference.npz',mean=np.array(means),mask=np.array(available),conditions=np.array(conditions),feature_ids=features)
    meta={'status':'descriptive_reference_fitted','fit_rows':rows,'reference_conditions':config['stress_substitute']['reference_conditions'],
        'scale':s['scale'],'aggregation':s['aggregation'],'counts_model_used':False,'raw_counts':'pending_user_source'}
    write_json(Path(snapshot)/'stress/reference.json',meta);return meta


def apply_stress(run,supplement):
    from .application import verify_freeze,inventory
    run=Path(run);f=verify_freeze(run);out=run/'applications/stress';root=Path(supplement)
    if not f['supplement_identity'] or sha256(root/'manifest.json')!=f['supplement_identity']['manifest_sha256']:raise ValueError('Stress supplement identity changed')
    m=read_json(root/'manifest.json')['stress_substitute'];verify_files(root,{'stress_queries.npz':m['files']['stress_queries.npz']})
    ref=read_json(run/'snapshot/stress/reference.json');policy=read_json(run/'snapshot/sample_roles.json');original={r['sample_key']:r for r in policy['samples']}
    with np.load(run/'snapshot/stress/reference.npz',allow_pickle=False) as a:
        means=a['mean'];rm=a['mask'];conditions=a['conditions'].tolist();features=a['feature_ids']
    reports=[]
    with np.load(root/'stress_queries.npz',allow_pickle=False) as a:
        if a['feature_ids'].tolist()!=features.tolist():raise ValueError('Stress programme order changed')
        for i,r in enumerate(m['rows']['stress_queries']):
            if r!={**original[r['sample_key']],'cell_count':r['cell_count']} or r['role'] not in f['config']['allowed_query_roles'] or r['split']!='locked_test':raise ValueError('Stress role mismatch')
            name=object_hash([r['sample_key'],'normalised_substitute'])[:20];dest=out/name;condition=ref['reference_conditions'][r['condition']]
            if condition not in conditions:raise ValueError('No original train control for '+condition)
            k=conditions.index(condition);mask=a['mask'][i]&rm[k]
            save_npz(dest/'descriptive.npz',observed=a['values'][i],reference=means[k],difference=np.where(mask,a['values'][i]-means[k],np.nan),mask=mask,coverage=a['coverage'][i])
            write_json(dest/'projection.json',{'feature_ids':features.tolist(),'scale':m['scale'],'aggregation':m['aggregation'],'control_condition':condition,'matched_control_pool_count':sum(x['condition']==condition for x in ref['fit_rows'])})
            result={'sample_key':r['sample_key'],'role':r['role'],'status':'queried','result_folder':name,'freeze_id':f['freeze_id'],
                'task':'normalised_stress_descriptive_substitute','raw_counts':'pending_user_source','clock':None,'depth':None,'counts_model_used':False,
                'scope':m['claim'],'files':inventory(dest)}
            # Exclude previous result from its own inventory on exact-run resume.
            result['files'].pop('result.json',None);write_json(dest/'result.json',result);reports.append(result)
    write_json(out/'result.json',{'group':'stress','rows':reports,'raw_counts':'pending_user_source','freeze_id':f['freeze_id']});return reports
