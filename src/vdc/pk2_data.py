"""PK2 data adapters using the existing observation and admission contracts."""
from pathlib import Path
import copy,csv,gzip,os,re
from collections import defaultdict
import numpy as np
from .io import read_json,read_jsonl,write_json,write_jsonl,save_npz,sha256,object_hash
from .contracts import ObservationBundle
from .observation import normalise_expression,score_programmes
from .alpha_numeric import gene_data
from .pk1_numeric import KillifishExitReference

class DataUnavailable(RuntimeError):pass

def verify_acquisition(acquired,studies):
    from .pk1_assets import verify_files
    status=read_json(Path(acquired)/'acquisition_status.json')['studies']
    for study in studies:
        entry=status.get(study,{})
        if entry.get('execution_status')!='completed':raise DataUnavailable(study+' acquisition is not completed')
        verify_files(Path(acquired)/study,entry['files'])

def annotation_members(path,identifiers='symbol'):
    genes=defaultdict(set);symbols=defaultdict(set);go=defaultdict(set)
    with Path(path).open(encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            gene=r['Gene stable ID'];term=r['GO term accession'];symbol=r['Gene name']
            if symbol:symbols[symbol].add(gene)
            if term:go[gene].add(term)
    for symbol,ids in symbols.items():
        if len(ids)==1:
            for term in go[next(iter(ids))]:genes[term].add(symbol)
    if identifiers=='symbol':return genes
    stable=defaultdict(set)
    for g,terms in go.items():
        for t in terms:stable[t].add(g)
    return stable

def defs_from_members(members,features,source):
    return [{'id':t,'members':{g:1. for g in sorted(members.get(t,()))},'source_ref':source} for t in features if members.get(t)]

def public_rows(samples,accessions,study):
    lookup={s['accession']:s['fields'] for s in samples};rows=[]
    for gsm in accessions:
        f=lookup[gsm];title=f['Sample_title'][0]
        if study=='GSE202844':rep=int(re.search(r'rep(\d+)$',title)[1]);condition=title.rsplit(', rep',1)[0]
        elif study=='GSE221467':rep=int(re.search(r'_rep(\d+)_',title)[1]);condition=re.sub(r'_rep\d+_TPM$','',title)
        elif study=='GSE124109':rep=int(title.rsplit('_',1)[1]);condition=title.rsplit('_',1)[0]
        else:raise ValueError('Unimplemented public design')
        val=2 if study=='GSE221467' else 3
        rows.append({'observation_id':gsm,'study_family':study,'biological_unit':f'{study}:conservative_rep_block:{rep}','origin':'public',
            'split':'validation' if rep==val else 'train','link_ids':[f'{study}:conservative_rep_block:{rep}'],
            'condition':condition,'reference_eligible':False,'replicate_block':rep,
            'lineage_status':'replicate-index grouping is conservative; common culture ancestry not fully deposited'})
    return rows

def save_public(x,mask,genes,rows,features,members,source,out,scale,extra):
    definitions=defs_from_members(members,features,source)
    scored=score_programmes(x,mask,genes,definitions,min_genes=3,min_coverage=.5)
    values=np.zeros((len(rows),len(features)),np.float32);available=np.zeros_like(values,bool);coverage=np.zeros_like(values)
    for j,fid in enumerate(scored['feature_ids']):
        k=features.index(fid);values[:,k]=scored['values'][:,j];available[:,k]=scored['mask'][:,j];coverage[:,k]=scored['coverage'][:,j]
    context={'species':extra['species'],'system':rows[0]['study_family'],'material':extra['material'],'assay':scale,
        'programme_definition_id':object_hash(definitions),'preprocessing_id':object_hash({'genes':genes,'scale':scale,'scoring':scored['definition']})}
    b=ObservationBundle(values,available,coverage,features,rows,context,'unit_holdout',np.zeros(len(rows)),np.zeros(len(rows),bool),None).validate();b.save(out)
    save_npz(Path(out)/'gene_expression.npz',expression=x,genes=np.array(genes),measured_mask=mask)
    write_json(Path(out)/'expression_contract.json',{'expression_sha256':sha256(Path(out)/'gene_expression.npz'),'scale':scale,**extra})
    write_json(Path(out)/'programmes.json',definitions)
    write_json(Path(out)/'admission.json',{'training_admitted':True,'source':source,'features':features,
        'train':len(b.indices('train')),'validation':len(b.indices('validation')),'mask_coverage':float(available.mean()),
        'clock_supervision':False,'grouping':'conservative replicate blocks; not independent study generalisation',**extra})
    return b

def prepare_public(study,acquired,annotation,features,out):
    verify_acquisition(acquired,[study])
    source=Path(acquired)/study;samples=read_jsonl(source/'samples.jsonl')
    if study=='GSE221467':
        meta=read_json(source/'locus_TPM/locus_expression.json')
        with np.load(source/'locus_TPM/locus_expression.npz',allow_pickle=False) as a:x=a['values'].copy();mask=a['mask'].copy()
        ids=[f['gene_id'].split('.')[0] for f in meta['features']];counts=defaultdict(int)
        for g in ids:counts[g]+=1
        keep=[i for i,g in enumerate(ids) if counts[g]==1]
        excluded=sorted(g for g,n in counts.items() if n>1);genes=[ids[i] for i in keep];x=x[:,keep];mask=mask[:,keep]
        rows=public_rows(samples,meta['sample_accessions'],study);members=annotation_members(annotation,'stable')
        y=normalise_expression(x,mask,'linear_abundance');scale='published_TPM_log1p_no_locus_double_counting'
        extra={'species':'Mus musculus','material':'E14_ESC','excluded_multilocus_gene_ids':excluded,'missing_loci_masked':True}
    else:
        view='counts' if study=='GSE202844' else 'FPKM';meta=read_json(source/view/'expression.json')
        with np.load(source/view/'expression.npz',allow_pickle=False) as a:x=a['values'].copy();mask=a['mask'].copy()
        keep=[i for i,g in enumerate(meta['feature_ids']) if not g.startswith('ERCC-')];genes=[meta['feature_ids'][i] for i in keep];x=x[:,keep];mask=mask[:,keep]
        y=normalise_expression(x,mask,'counts' if view=='counts' else 'linear_abundance');scale='endogenous_log1p_CPM10k' if view=='counts' else 'published_FPKM_log1p'
        rows=public_rows(samples,meta['sample_accessions'],study);members=annotation_members(annotation)
        extra={'species':'Mus musculus' if view=='counts' else 'Rattus norvegicus','material':'ESC' if view=='counts' else 'fibroblasts',
            'excluded_ERCC_features':len(meta['feature_ids'])-len(keep),'published_cohort_normalisation':'Cuffnorm all samples' if view=='FPKM' else 'none added'}
        if view=='counts':save_npz(Path(out)/'raw_counts.npz',counts=x,genes=np.array(genes))
    extra['annotation_sha256']=sha256(annotation)
    return save_public(y,mask,genes,rows,features,members,object_hash({'source':read_json(source/'status.json'),'annotation':sha256(annotation)}),out,scale,extra)

def prepare_dauer_arrays(acquired,raw,gaf,features,out):
    verify_acquisition(acquired,['GSE3169'])
    lock=read_json(Path(__file__).resolve().parents[2]/'configs/pk2_public_sources.json')['studies']['GSE3169']
    entry=next(e for e in lock if e['file']=='GSE3169_family.soft.gz')
    if sha256(Path(raw)/entry['file'])!=entry['sha256']:raise ValueError('Dauer platform/source file hash changed')
    source=Path(acquired)/'GSE3169';platforms={};platform=None;active=False;header=None
    with gzip.open(Path(raw)/'GSE3169_family.soft.gz','rt',encoding='utf-8') as f:
        for line in f:
            line=line.rstrip('\r\n')
            if line.startswith('^PLATFORM = '):platform=line.split(' = ')[1];platforms[platform]={}
            elif line=='!platform_table_begin':active=True;header=None
            elif line=='!platform_table_end':active=False
            elif active:
                row=line.split('\t')
                if header is None:header=row
                else:
                    row+=['']*(len(header)-len(row));r=dict(zip(header,row));platforms[platform][r['ID']]=r.get('ORF','')
    aliases=defaultdict(set);members=defaultdict(set)
    with gzip.open(gaf,'rt',encoding='utf-8') as f:
        for line in f:
            if line.startswith('!'):continue
            r=line.rstrip('\n').split('\t')
            if len(r)!=17 or r[0]!='WB' or r[12]!='taxon:6239':continue
            for alias in [r[1],r[2]]+r[10].split('|'):
                if alias:aliases[alias].add(r[1])
            if 'NOT' not in r[3].split('|') and r[6]!='ND' and 'PMID:41104926' not in r[5].split('|'):members[r[4]].add(r[1])
    series=read_jsonl(source/'dauer_mtc_samples.jsonl');sample_rows=read_jsonl(source/'samples.jsonl');lookup={r['accession']:r for r in sample_rows}
    values=[];genes=set();rows=[];seen=set()
    for item in series:
        gsm=item['sample_accession'];group=item['series_group'];hour=int(item['nominal_time_token']);key=(group,hour)
        if key in seen:raise DataUnavailable('More than one deposited hybridisation per series/hour; review replacement policy')
        seen.add(key);path=source/'dauer_mtc'/gsm;m=read_json(path/'expression.json');aggregated=defaultdict(list)
        with np.load(path/'expression.npz',allow_pickle=False) as a:
            for probe,value,valid in zip(m['probe_ids'],a['values'],a['mask']):
                alias=platforms[item['platform']].get(probe,'');ids=aliases.get(alias,set())
                if valid and len(ids)==1:aggregated[next(iter(ids))].append(float(value))
        v={g:float(np.mean(v)) for g,v in aggregated.items()};values.append(v);genes.update(v)
        rows.append({'observation_id':gsm,'study_family':'GSE3169_Dauer_MTC','biological_unit':group,'origin':'public',
            'split':'validation' if group.endswith(':4') else 'train','link_ids':[group],
            'condition':'Dauer Exit','elapsed_hours_since_release':hour,'reference_eligible':False,
            'time_provenance':'GEO Series_summary explicitly identifies T as hour and -2 as replacement same RNA',
            'shared_reference':'RefB-JW common technical reference; series holdout is not independent reference material'})
    genes=sorted(genes);x=np.zeros((len(rows),len(genes)));mask=np.zeros_like(x,bool)
    for i,values_i in enumerate(values):
        for j,g in enumerate(genes):
            if g in values_i:x[i,j]=values_i[g];mask[i,j]=True
    return save_public(x,mask,genes,rows,features,members,'GO_WB:'+sha256(gaf),out,'published_log2_common_reference_ratio',
        {'species':'Caenorhabditis elegans','material':'starting_dauer_pool_series','probe_aggregation':'mean finite probes mapping unambiguously to one WBGene',
         'raw_family_sha256':sha256(Path(raw)/'GSE3169_family.soft.gz'),'series_count':4,'no_cross_platform_clock_shared':True})

def combine_sources(paths,out,features):
    bundles=[ObservationBundle.load(p) for p in paths];rows=[];xs=[];ms=[];cs=[];contexts={}
    for b in bundles:
        if any(r['origin']!='public' or r['split'] not in {'train','validation'} for r in b.rows):raise ValueError('Only public development enters source sets')
        idx=[b.feature_ids.index(fid) for fid in features]
        xs.append(b.values[:,idx]);ms.append(b.mask[:,idx]);cs.append(b.coverage[:,idx]);rows.extend(copy.deepcopy(b.rows))
        contexts[b.rows[0]['study_family']]=b.scope
    n=len(rows);identity=object_hash({'sources':[b.fingerprint for b in bundles],'features':features})
    context={'species':'explicit_multi_context','system':identity,'material':'study_specific','assay':'study_specific_scales',
        'programme_definition_id':object_hash(features),'preprocessing_id':identity}
    b=ObservationBundle(np.concatenate(xs),np.concatenate(ms),np.concatenate(cs),features,rows,context,'unit_holdout',np.zeros(n),np.zeros(n,bool),None).validate();b.save(out)
    write_json(Path(out)/'source_set.json',{'identity':identity,'contexts':contexts,'fingerprints':[b.fingerprint for b in bundles],
        'normalisation':'per-study fitted on train by fit_state(multi_context=True)','clock_shared':False})
    return b

def local_input(prepared,out,mode='programme',fold_index=None,fraction=None,subset_seed=42,annotation=None):
    from .admission import role_manifest
    prepared=Path(prepared);out=Path(out);b=ObservationBundle.load(prepared);x,genes=gene_data(prepared)
    if fold_index is not None:
        policy,samples=role_manifest();units=sorted({r['biological_unit'] for r in b.rows})
        if not 1<=fold_index<=len(units):raise DataUnavailable('Fold index outside actual development units')
        held=units[fold_index-1];heldkeys={r['source_sample_key'] for r in b.rows if r['biological_unit']==held}
        heldcohorts={samples[k].get('cohort_id') for k in heldkeys}-{None,''}
        assignments={r['source_sample_key']:('validation' if r['biological_unit']==held or samples[r['source_sample_key']].get('cohort_id') in heldcohorts else 'train') for r in b.rows}
        f={'protocol':'PK2_development_resampling','parent_approval_hash':policy['approval']['content_hash'],'assignments':assignments,'held_unit':held}
        f['hash']=object_hash(f);write_json(out/'fold.json',f);os.environ['VDC_DEVELOPMENT_FOLD']=str((out/'fold.json').resolve())
        for r in b.rows:r.update(split=assignments[r['source_sample_key']],development_fold_hash=f['hash'])
    if fraction is not None:
        units=sorted({r['biological_unit'] for r in b.rows if r['split']=='train'});order=np.random.default_rng(subset_seed).permutation(units).tolist()
        selected=set(order[:max(1,int(np.ceil(len(units)*fraction)))]);indices=[i for i,r in enumerate(b.rows) if r['split']=='validation' or r['biological_unit'] in selected]
        b=b.subset(indices);x=x[indices]
        write_json(out/'subset.json',{'fraction':fraction,'subset_seed':subset_seed,'training_units':sorted(selected),'nested_order':order,'validation_unchanged':True})
    try:
        if prepared.name=='core_celltypes' and fold_index is None and fraction is None:
            core=ObservationBundle.load(prepared.parent/'core');cx,cg=gene_data(prepared.parent/'core')
            if cg!=genes:raise ValueError('Whole-pool and type-profile gene orders differ')
            ref=KillifishExitReference().fit(cx,cg,core.rows)
        else:ref=KillifishExitReference().fit(x,genes,b.rows)
    except ValueError as exc:raise DataUnavailable(str(exc)) from exc
    ref.save(out/'reference');b.clock=ref.predict(x,genes);b.clock_mask=np.array(['late' not in r.get('condition','').lower() for r in b.rows]);b.clock_reference_id=ref.reference_id
    if mode=='gene':
        tr=b.indices('train');variance=x[tr].var(0);idx=np.argsort(-variance,kind='stable')[:min(2000,(variance>1e-8).sum())]
        if len(idx)<8:raise DataUnavailable('Too few variable training genes')
        b.values=x[:,idx];b.mask=np.ones_like(b.values,bool);b.coverage=np.ones_like(b.values);b.feature_ids=[genes[i] for i in idx]
    elif mode=='fine':
        members=annotation_members(annotation,'stable');present=set(genes);eligible=[t for t,v in members.items() if 10<=len(v&present)<=200 and len(v&present)/len(v)>=.5]
        features=sorted(eligible)[:96]
        if len(features)<64:raise DataUnavailable('Fewer than 64 supported fine GO terms')
        definitions=defs_from_members(members,features,'Ensembl:'+sha256(annotation));s=score_programmes(x,np.ones_like(x,bool),genes,definitions)
        b.values,b.mask,b.coverage,b.feature_ids=s['values'],s['mask'],s['coverage'],s['feature_ids'];write_json(out/'programmes.json',definitions)
    elif mode!='programme':raise ValueError('Unknown feature mode')
    b.context={**b.context,'programme_definition_id':object_hash({'parent':b.context['programme_definition_id'],'mode':mode,'features':b.feature_ids})};b.save(out/'bundle')
    write_json(out/'input.json',{'mode':mode,'bundle_fingerprint':b.fingerprint,'features':len(b.feature_ids),'fold_refit':fold_index is not None,
        'selected_by':'training_variance_only' if mode=='gene' else 'fixed_annotation_ID_order_and_member_coverage' if mode=='fine' else 'existing_approved_panel','reserved_used':False})
    return b
