"""Bounded offline assembly using existing CW fits, actual semantic caches and response models."""
from pathlib import Path
import shutil,os,copy
import numpy as np
from .io import read_json,read_jsonl,write_json,write_jsonl,save_npz,sha256,object_hash
from .pk1_assets import verify_files,checked_member
from .integration_bundle import ROUTES,implementation
from .clock_wave_data import load_data
from .clock_wave import ClockWave,evaluate,identities
from .clock_wave_revision import residual_task
from .clock_wave_tasks import gene_waves_task
from .pk2_semantics import load_semantics,matrix_hash


def graph_from(vectors,definitions,genes,mode,seed):
    vectors=np.asarray(vectors,float);permutation=list(range(len(vectors)))
    if vectors.shape[0]!=len(definitions) or not np.isfinite(vectors).all() or not np.all(np.any(vectors!=0,axis=1)):raise ValueError('Invalid actual semantic vectors')
    if mode=='shuffle':
        groups={};rng=np.random.default_rng(seed);genes=set(genes)
        for i,d in enumerate(definitions):
            n=len(d['members']);cover=len(set(d['members'])&genes)/max(n,1)
            groups.setdefault((int(np.log2(max(n,1)))//2,int(cover*2)),[]).append(i)
        for indices in groups.values():
            if len(indices)>1:
                shift=int(rng.integers(1,len(indices)))
                for j,k in zip(indices,indices[shift:]+indices[:shift]):permutation[j]=k
        if permutation==list(range(len(vectors))):raise ValueError('No legal nonidentity semantic permutation')
    vectors=vectors[permutation];norm=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
    graph=np.maximum(norm@norm.T,0);np.fill_diagonal(graph,0);graph/=max(float(graph.sum(1).max()),1.)
    if not np.any(graph):raise ValueError('Degenerate zero semantic graph')
    return graph,{'matrix_hash':matrix_hash(vectors),'graph_hash':object_hash(graph.tolist()),'permutation':permutation,'nonzero_edges':int(np.count_nonzero(graph))}


def fit_view(stage,data,output,view,condition,vectors=None,provenance=None,root=None):
    """One fixed fit; existing numerical implementations are the only consumers."""
    stage,data,out=Path(stage),Path(data),Path(output);root=Path(root or Path(__file__).resolve().parents[2])
    x,g,defs,rows=load_data(data/view);base=ClockWave.load(stage/'tasks'/('numeric_'+view+'_C2')/'model')
    if g!=base.meta['genes'] or defs!=base.meta['definitions']:raise ValueError('Frozen feature contract differs')
    cfg=base.meta['config'];graph=None;meta={'condition':condition,'new_fit':condition!='zero'}
    if condition not in {'zero','base','domain','shuffle'}:raise ValueError('Unknown fixed semantic condition')
    if condition!='zero':
        graph,meta_graph=graph_from(vectors,defs,g,condition,cfg['semantic_shuffle_seed'])
        meta.update(meta_graph,provenance=provenance,consumer='ClockWave.fit(semantic_graph=graph)',strength=cfg['semantic_strength'])
        save_npz(out/'semantic_graph.npz',graph=graph)
    # Work directory uses the canonical task layout, without opening a master H5AD again.
    work=out/'work';shutil.copytree(data/view,work/'data'/view,dirs_exist_ok=True)
    dest=work/'tasks'/('numeric_'+view+'_C2')/'model'
    if condition=='zero':
        shutil.copytree(stage/'tasks'/('numeric_'+view+'_C2')/'model',dest,dirs_exist_ok=True)
        model=ClockWave.load(dest)
    else:
        model=ClockWave.fit(x,g,defs,rows,'C2',cfg,graph);model.save(dest)
        delta=float(np.max(np.abs(model.arrays['coef']-base.arrays['coef'])))
        if delta<=0:raise ValueError('Semantic graph was not consumed by wave coefficients')
        meta.update(coefficient_max_abs_delta=delta,scientific_benefit_claim=False)
    target=out/'model';shutil.copytree(dest,target,dirs_exist_ok=True)
    va=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);rr=[rows[i] for i in va]
    result=evaluate(ClockWave.load(target),x[va],rr,out/'evaluation')
    if condition=='zero':
        for name in ('readout.npz','matched_direct.npz'):shutil.copy2(stage/'tasks'/('residual_'+view)/name,out/name)
        for name in ('wave_parameters.npz','shapes.json'):shutil.copy2(stage/'tasks'/('gene_waves_'+view)/name,out/name)
        residual=read_json(stage/'tasks'/('residual_'+view)/'result.json')
    else:
        residual_task(work,view,out/'residual_fit',cfg)
        for name in ('readout.npz','matched_direct.npz'):shutil.copy2(out/'residual_fit'/name,out/name)
        gene_waves_task(work,view,out/'gene_waves_fit',cfg,root)
        for name in ('wave_parameters.npz','shapes.json'):shutil.copy2(out/'gene_waves_fit'/name,out/name)
        residual=read_json(out/'residual_fit/result.json')
    meta['source_clock_model_hash']=sha256(stage/'tasks'/('numeric_'+view+'_C2')/'model/model.json')
    write_json(out/'semantic.json',meta)
    primary=residual['metrics']['in_reference']['clock_identity_residual']['metric'] if view=='bulk' else result['model_in_reference']
    # Actual reload of both coefficients and dependent readouts happens in the reused consumers.
    answer={'view':view,'condition':condition,'primary':primary,'matched_ridge':residual['metrics']['matched_fit_direct']['in_reference'],
            'numeric':result,'semantic':meta,'selection':'fixed domain candidate; no selection on these scores'}
    write_json(out/'result.json',answer);return answer


def assemble(stage,output):
    stage=Path(stage).resolve();out=Path(output).resolve();root=Path(__file__).resolve().parents[2]
    c=read_json(stage/'config.json');snap=read_json(stage/'stage_snapshot.json');verify_files(stage,snap['files'])
    if snap['status']!='frozen_experimental':raise ValueError('Use completed CW stage')
    verify_files(stage,read_json(stage/'queue_status.json')['tasks']['prepare']['files'])
    os.environ['VDC_ROLE_MANIFEST']=str(Path(c['private_root'])/'sample_roles.json')
    if sha256(os.environ['VDC_ROLE_MANIFEST'])!=c['role_manifest_sha256']:raise ValueError('Frozen sample roles changed')
    source=Path(c['source']);prior=read_json(source/'config.json');queue=read_json(source/'queue_status.json')['tasks']
    source_inputs={name:queue[name]['files'] for name in ('embeddings_base','embeddings_domain','select_adapter','corpus','public_regulon','local_regulon')}
    selected_name=read_json(source/'tasks/select_adapter/selection.json')['selected']['job']
    source_inputs[selected_name]=queue[selected_name]['files']
    signature=object_hash({'stage':snap['snapshot_id'],'stage_config':sha256(stage/'config.json'),
                          'code':implementation(),'build':sha256(__file__),'source':sha256(source/'config.json'),'source_inputs':source_inputs})
    if out.exists() and (out/'assembly.json').exists() and read_json(out/'assembly.json')['signature']!=signature:raise ValueError('Assembly source/config/code changed; use new output')
    out.mkdir(parents=True,exist_ok=True);write_json(out/'assembly.json',{'signature':signature,'source_stage_snapshot':snap['snapshot_id'],'stage_path':str(stage)})
    def source_task(name):
        q=queue[name]
        if q['status'] not in {'completed','evaluated_new','reused_verified'}:raise ValueError('Required source task incomplete: '+name)
        p=source/'tasks'/name;verify_files(p,q['files']);return p
    for n in ('embeddings_base','embeddings_domain','select_adapter','corpus'):source_task(n)
    selected=read_json(source/'tasks/select_adapter/selection.json')['selected']['job'];source_task(selected)
    adapter=source/'tasks'/selected/'trained/adapter';actual=object_hash({p.name:sha256(p) for p in sorted(adapter.iterdir()) if p.is_file()})
    if actual!=read_json(stage/'tasks/qwen_domain/provenance.json')['adapter_hash']:raise ValueError('Selected adapter differs from completed stage')
    if not (adapter/'adapter_model.safetensors').is_file():raise ValueError('Actual adapter weights missing')
    shutil.copytree(adapter,out/'adapter',dirs_exist_ok=True)
    # Only train-family source text enters retrieval. Completions and evaluation questions are discarded.
    records=read_jsonl(source/'tasks/corpus/corpus.jsonl')
    keep=[{k:r[k] for k in ('record_id','study_family','source_ref','split','kind','text','reviewed','evidence_type','source_locator','review_basis') if k in r}
          for r in records if r['split']=='train' and r.get('reviewed')]
    if not keep:raise ValueError('No admitted source text for retrieval')
    write_jsonl(out/'knowledge/records.jsonl',keep)
    from .pk2_knowledge import objects
    for mode in ('base','domain'):
        cache=source/'tasks'/('embeddings_'+mode);meta=read_json(cache/'embeddings.json')
        records=objects(meta['record_ids'],root)
        if object_hash(records)!=meta['corpus_hash']:raise ValueError('Programme embedding text changed')
        shutil.copytree(cache,out/'semantics'/mode,dirs_exist_ok=True)
        write_jsonl(out/'semantics'/mode/'objects.jsonl',records)
    comparisons=[];states=read_json(out/'progress.json') if (out/'progress.json').exists() else {}
    for view in ROUTES:
        _,g,defs,_=load_data(stage/'data'/view);features=[d['id'] for d in defs]
        for mode in ('zero','base','domain','shuffle'):
            job=view+'_'+mode;folder=out/'fits'/view/mode
            if states.get(job,{}).get('status')=='completed':verify_files(folder,states[job]['files']);comparisons.append(read_json(folder/'result.json'));print('REUSED_VERIFIED',job,flush=True);continue
            states[job]={'status':'running'};write_json(out/'progress.json',states);print('START',job,flush=True)
            vectors=provenance=None
            if mode!='zero':
                which='base' if mode=='base' else 'domain'
                with np.load(source/'tasks'/('embeddings_'+which)/'embeddings.npz',allow_pickle=False) as a:dimension=a['vectors'].shape[1]
                vectors,provenance=load_semantics(source/'tasks',{'semantics':which},features,{'semantic_dim':dimension})
            result=fit_view(stage,stage/'data',folder,view,mode,vectors,provenance,root);comparisons.append(result)
            states[job]={'status':'completed','execution_kind':'reused_verified' if mode=='zero' else 'fitted',
                'files':{p.relative_to(folder).as_posix():sha256(p) for p in folder.rglob('*') if p.is_file() and 'work' not in p.relative_to(folder).parts}}
            write_json(out/'progress.json',states);print('COMPLETED',job,flush=True)
    write_json(out/'comparisons.json',comparisons)
    pk1=Path(prior['pk1_run']);public=export_public(pk1,out,root);regulons={}
    for name,label in [('public_regulon','public'),('local_regulon','local')]:
        p=source_task(name);dest=out/'regulons'/label;dest.mkdir(parents=True,exist_ok=True)
        for f in ('members.json','result.json'):shutil.copy2(p/f,dest/f)
        regulons[label]={'path':'regulons/'+label+'/members.json','scope':'Caenorhabditis elegans external unsigned targets' if label=='public' else 'killifish train-only coexpression candidates; not causal TF activity'}
    meta={'source_stage_snapshot':snap['snapshot_id'],'signature':signature,'knowledge':{'revision':prior['model_revision'],'adapter_hash':actual,
          'base_model':'Qwen/Qwen3-4B-Instruct-2507','records_hash':sha256(out/'knowledge/records.jsonl')},'public_models':public,'regulons':regulons}
    meta['files']={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and 'work' not in p.relative_to(out).parts and p.name not in {'fit_manifest.json','progress.json'}}
    write_json(out/'fit_manifest.json',meta);return meta


def export_public(pk1,out,root):
    public={};pk1=Path(pk1);out=Path(out);root=Path(root)
    from .pk1_assets import alpha_module
    from .response import ResponseDataset
    for name in ('endpoint','transition','functional'):
        p,_=alpha_module(pk1,name);target=out/'public'/name;target.mkdir(parents=True,exist_ok=True)
        if name=='functional':
            for f in ('model.npz','result.json'):shutil.copy2(p/f,target/f)
            info=read_json(p/'result.json');scope={'species':'Caenorhabditis elegans','endpoint':info['endpoint'],'protocol':info['protocol'],'input':'time_and_history_group_assay_not_expression'}
            functional=read_json(root/'data/curated/functional_fig1b.json')
            if functional['endpoint']!=info['endpoint'] or functional['protocol']!=info['protocol']:raise ValueError('Functional protocol changed')
            train=[r for r in functional['records'] if r['split']=='train']
            controls={str(h):sum(r['counts']['young_adult'] for r in train if r['hours_after_release']==h)/sum(r['total'] for r in train if r['hours_after_release']==h) for h in sorted({float(r['hours_after_release']) for r in train})}
            write_json(target/'baselines.json',controls)
            public[name]={'path':'public/'+name,'scope':scope}
        else:
            for f in ('response_model.json','response_model.npz'):shutil.copy2(p/f,target/f)
            if name=='transition':
                with np.load(p/'time_baseline.npz',allow_pickle=False) as a:save_npz(target/'time_baseline.npz',**{k:a[k] for k in ('coefficients','mean','scale')})
            d=ResponseDataset.load(p/'dataset');tr=[i for i,r in enumerate(d.rows) if r['split']=='train']
            # Train-only conditional means are exported as parameters, never query predictions.
            groups={}
            for i in tr:
                key=tuple(d.action[i].tolist())+(() if d.elapsed is None else (float(d.elapsed[i]),))
                groups.setdefault(key,[]).append(i)
            controls=[]
            for key,ix in groups.items():
                count=d.target_mask[ix].sum(0)
                mean=np.where(count>0,np.where(d.target_mask[ix],d.target[ix],0.).sum(0)/np.maximum(count,1),0.)
                controls.append({'key':list(key),'mean':mean.tolist(),'available':(count>0).tolist(),'training_units':len({d.rows[i]['biological_unit'] for i in ix})})
            write_json(target/'baselines.json',controls)
            public[name]={'path':'public/'+name,'scope':d.scope,'species':'Caenorhabditis elegans',
                'admitted_actions':np.unique(d.action[tr],axis=0).tolist(),
                'elapsed_range':None if d.elapsed is None else [float(d.elapsed[tr].min()),float(d.elapsed[tr].max())],
                'source':'frozen_PK1_response; newer response experiments remain separate'}
    return public
