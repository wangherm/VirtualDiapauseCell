"""Real-public-data ENGINE regression, no LLM calls and no biological validation claim."""
import argparse
import json
import os
from pathlib import Path
from vdc.io import read_json,write_json,object_hash,sha256
from vdc.research_r2 import Engine
from vdc.research_r2_devices import configure_devices

ROOT=Path(__file__).resolve().parents[1]
META_CODE='''import json,re
from vdc.research_r2_artifacts import metadata_rows
a=json.load(open('/input/assets.json'));key=next(iter(a))
rows=[]
for r in metadata_rows(a[key]['container_path']):
    m=re.search(r'Library name:\\s*(D(?:1|4|15|30)\\.(?:0|6|24)\\.([123]))(?:\\s|$)',r.get('description',''))
    if m: rows.append(dict(sample_id=r['sample_id'],matrix_column=m[1],replicate=m[2],metadata_asset=key))
assert len(rows)==36, len(rows)
json.dump(rows,open('/out/metadata.json','w'))
'''
TRANSFORM='''import numpy as np
def features(counts,genes,targets,names):
    hidden=np.isin(genes,targets)
    denom=counts[:,~hidden].sum(axis=1)
    idx=[genes.index(g) for g in names]
    return np.log1p(counts[:,idx]/np.maximum(denom[:,None],1)*1e6)
'''
PROBE='''import json,numpy as np
from transform import features
m=json.load(open('/input/probe_schema.json'));c=np.load('/input/probe.npz')['counts']
np.savez_compressed('/out/features.npz',x=features(c,m['genes'],m['targets'],m['features']))
'''
PREPARE='''import json,numpy as np,pandas as pd
from pathlib import Path
from transform import features
a=json.load(open('/input/assets.json'));d=json.load(open('/input/artifacts.json'))
matrix=next(v['container_path'] for v in a.values() if v['container_path'].endswith('.csv.gz'))
rows=json.load(open(next(iter(d.values()))['container_path']))
rows=sorted(rows,key=lambda r:(r['replicate']=='3',r['matrix_column']))
frame=pd.read_csv(matrix,index_col=0);genes=frame.index.astype(str).tolist()
c=frame[[r['matrix_column'] for r in rows]].to_numpy(dtype=float).T
n=sum(r['replicate']!='3' for r in rows)
order=np.argsort(-np.var(np.log1p(c[:n]),axis=0),kind='stable');targets=[genes[i] for i in order[:4]]
names=[genes[i] for i in order[4:68]];x=features(c,genes,targets,names);y=np.log1p(c[:,order[:4]])
out=Path('/out/bundle');out.mkdir()
np.savez_compressed(out/'train.npz',x=x[:n],y=y[:n]);np.savez_compressed(out/'query.npz',x=x[n:]);np.savez_compressed(out/'truth.npz',y=y[n:])
metadata=[]
for r in rows:
    metadata.append(dict(study_family='GSE288723',biological_unit=r['sample_id'],unit_type='deposited_worm_pool_library',
        replication_unit_id=r['sample_id'],origin='real',matrix_column=r['matrix_column'],
        link_ids=['GSE288723:conservative_repblock:'+r['replicate']],
        unit_evidence=dict(asset=r['metadata_asset'],row_field='sample_id',row_value=r['sample_id'],unit_field='sample_id')))
json.dump(metadata[:n],open(out/'train_rows.json','w'));json.dump(metadata[n:],open(out/'query_rows.json','w'))
json.dump(dict(features=names,targets=targets),open(out/'schema.json','w'))
'''
FIT='''import numpy as np
z=np.load('/input/train.npz');x=z['x'];y=z['y'];xm=x.mean(0);ym=y.mean(0);xc=x-xm
coef=np.linalg.solve(xc.T@xc+ALPHA*np.eye(x.shape[1]),xc.T@(y-ym)).T
np.savez_compressed('/out/model.npz',coef=coef,intercept=ym-coef@xm)
'''
PREDICT='''import numpy as np
m=np.load('/model/model.npz');x=np.load('/input/query.npz')['x']
np.savez_compressed('/out/predictions.npz',prediction=x@m['coef'].T+m['intercept'])
'''


def regression(private,out,runtime_root):
    config=read_json(ROOT/'configs/research_r2.json')
    config['runtime'].update(runtime_root=str(runtime_root),memory='4g',cpus=2,timeout_seconds=180)
    config['runtime']=configure_devices(config['runtime'],'cpu')
    engine=Engine(ROOT,private,out,config)
    matches=lambda name:next(k for k,v in engine.inv['assets'].items() if v['source']=='GSE288723' and v['name']==name)
    matrix=matches('GSE288723_raw_counts.csv.gz');meta=matches('GSE288723_family.soft.gz')
    def call(a):
        print('ENGINE_CHECK_START '+a['id'],flush=True)
        engine.state['actions'][a['id']]={'type':a['type'],'signature':object_hash(a),'status':'running'};engine.save()
        result=engine.action(a)
        engine.state['actions'][a['id']].update(status='completed',result=result);engine.save()
        print('ENGINE_CHECK_PASSED '+a['id'],flush=True)
        return result
    call({'id':'real_metadata','type':'prepare','asset_ids':[meta],'files':{'prepare.py':META_CODE},'command':['python','/code/prepare.py']})
    readable=call({'id':'read_real_metadata','type':'inspect_preparation','preparation':'real_metadata','path':'metadata.json'})
    assert 'GSM' in readable['text'] and readable['public']
    call({'id':'real_bundle','type':'prepare','asset_ids':[matrix],
        'artifact_inputs':[{'preparation':'real_metadata','path':'metadata.json'}],
        'files':{'prepare.py':PREPARE,'transform.py':TRANSFORM,'feature_probe.py':PROBE},'command':['python','/code/prepare.py']})
    schema=read_json(out/'actions/real_bundle/output/bundle/schema.json')
    c={'question':'Engineering regression of public metadata to two comparable candidates and export; not autonomous research',
        'mission':'state','metric':'macro_unit_mse','target_definition':'Four genes chosen on training variance only',
        'target_scale':'log1p deposited raw counts','allowed_inputs':'64 visible genes; CPM denominator excludes all four targets',
        'unit_definition':'Deposited GSM libraries linked by conservative replicate-index blocks; longitudinal matching not proven',
        'preprocessing':'Training-only variance selection; visible-only CPM and log1p',
        'split_rationale':'Engineering fixture: replicate blocks 1/2 fit, block 3 query; adaptive, not independent biological validation',
        'source_ids':['GSE288723'],'input_dependency':{'kind':'target_isolated','matrix_asset':matrix,'target_genes':schema['targets']}}
    call({'id':'seal_real_task','type':'register_task','preparation':'real_bundle','subdir':'bundle','task':'engineering_public','contract':c})
    for name,alpha,parent in [('candidate_a',2.,None),('candidate_b',8.,'candidate_a')]:
        call({'id':name,'type':'experiment','task':'engineering_public','parent':parent,
            'files':{'fit.py':FIT.replace('ALPHA',str(alpha)),'predict.py':PREDICT},
            'rule':{'reference':'parent' if parent else 'same_input_ridge','minimum_relative_improvement':0}})
    call({'id':'export_real','type':'export','experiment':'candidate_b','refit':True})
    result={'status':'passed','scope':'real_public_data_engineering_regression','autonomous_research_executed':False,
        'api_calls_executed':0,'source_sha256':{k:engine.inv['assets'][k]['sha256'] for k in (matrix,meta)},
        'steps':['public_metadata_values','derived_readonly_mount','source_units','hidden_target_counterfactual','two_candidates','parent_comparison','refit_export_reload'],
        'comparisons':{k:v['comparison'] for k,v in engine.state['experiments'].items()},
        'code_sha256':{p.name:sha256(p) for p in (ROOT/'src/vdc').glob('research_r2*.py')},
        'runtime_id':engine.config['runtime']['image_id'],'biological_validation':False}
    write_json(out/'ENGINE_CHECK.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-root',type=Path,default=os.environ.get('VDC_PRIVATE_ROOT','/root/autodl-tmp/vdc-private'))
    p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a fresh engineering check directory')
    regression(a.private_root,a.out,a.runtime_root)
