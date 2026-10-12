"""Host-owned artifact lineage, readable public metadata and explicit input references."""
import csv
import gzip
import json
from pathlib import Path
from .io import read_json, write_json, sha256, object_hash
from .research_r2_tools import inside, identifier, inspect_assets, resolve_assets

TASK_FILES = ('train.npz','query.npz','truth.npz','train_rows.json','query_rows.json','schema.json')
METADATA_FIELDS = set('sample_id sample_key observation_id library library_id count_column study_family species assay resolution package condition condition_original condition_standardized genotype treatment time_value time_unit time_reference elapsed_hours history_days section replicate replicate_block biological_unit unit_type unit_kind replication_unit_id linked_embryo_key linked_embryo_key_status pool_id embryo_id culture_id shared_control_id batch_id cell_type origin role split status'.split())


def current_roles(inv):
    for source in inv['sources'].values():
        source.setdefault('historical_policy', {k:source[k] for k in ('role','rule','ai_brief') if k in source})
        source['role']='available_for_adaptive_research'
        source['rule']='All uploaded roles authorised for adaptive research; retain actual sample relationships and source scales.'
        old=source.get('ai_brief','')
        if any(t in old.lower() for t in ('archive only','archive-only','do not train','not for training','holdout','heldout','excluded family')):
            source['ai_brief']=source.get('title',source.get('system','Source available for adaptive research.'))
        source['current_access']='authorised_all_roles; not an independent blind test'
    return inv


def metadata_rows(path):
    """Read deposited tables/JSON/SOFT; never execute code or load pickle."""
    path=Path(path); opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',errors='replace') as f:
        if path.name.endswith(('.soft','.soft.gz')):
            rows=[]; row=None
            for line in f:
                if line.startswith('^SAMPLE = '):
                    row={'sample_id':line.split(' = ',1)[1].strip()};rows.append(row)
                elif row is not None and line.startswith('!Sample_') and ' = ' in line:
                    key,value=line.rstrip().split(' = ',1)
                    if key=='!Sample_characteristics_ch1' and ': ' in value:
                        k,v=value.split(': ',1);row[k.replace(' ','_')]=v
                    elif key in ('!Sample_title','!Sample_description','!Sample_source_name_ch1','!Sample_extract_protocol_ch1'):
                        k=key.removeprefix('!Sample_');row[k]=row.get(k,'')+' '+value
            return rows
        if path.suffix=='.json':
            obj=json.load(f)
            if isinstance(obj,dict):
                obj=next((obj[k] for k in ('samples','rows','observations') if isinstance(obj.get(k),list)),obj)
            if not isinstance(obj,list) or any(not isinstance(r,dict) for r in obj):
                raise ValueError('Metadata requires a row list, samples/rows/observations list, CSV, TSV or GEO SOFT')
            return obj
        delimiter='\t' if path.name.endswith(('.tsv','.tsv.gz')) else ','
        return list(csv.DictReader(f,delimiter=delimiter))


def read_metadata(inv,asset,offset=0):
    _,provenance=resolve_assets(inv,[asset])
    rows=metadata_rows(inv['assets'][asset]['path']); offset=max(0,int(offset))
    public=inv['assets'][asset]['public']
    safe=[];truncated=[]
    for r in rows[offset:offset+5]:
        selected={k:v for k,v in r.items() if (public or k in METADATA_FIELDS)
                  and isinstance(v,(str,int,float,bool,type(None)))}
        for key,value in list(selected.items()):
            if isinstance(value,str) and len(value)>800:
                selected[key]=value[:800];truncated.append(key)
        if len(json.dumps(selected))>9000:
            raise ValueError('Metadata row exceeds bounded response; inspect columns and prepare a specific public summary')
        if safe and len(json.dumps(safe+[selected]))>9000:break
        safe.append(selected)
    if rows and not any(safe): raise ValueError('No approved metadata fields; use structural inspection for this file')
    return {'asset':asset,'source_sha256':provenance[asset]['sha256'],'rows':safe,'total_rows':len(rows),
        'next_offset':offset+len(safe) if offset+len(safe)<len(rows) else None,'public':public,
        'truncated_text_fields':sorted(set(truncated)),
        'policy':'public_metadata' if public else 'original_source_approved_metadata_fields_only'}


def artifact_index(run,prep,inv):
    root=Path(run)/'actions'/identifier(prep)
    provenance=read_json(root/'provenance.json')
    # Source membership is checked against the host inventory, not a candidate's public flag.
    public=bool(provenance) and all(k in inv['assets'] and inv['assets'][k]['public'] for k in provenance)
    files={}
    for p in sorted((root/'output').rglob('*')):
        if p.is_symlink(): raise ValueError('Linked preparation output is not allowed')
        if p.is_file():
            rel=p.relative_to(root/'output').as_posix()
            files[rel]={'sha256':sha256(p),'bytes':p.stat().st_size,'public':public,
                'ref':{'preparation':prep,'path':rel}}
    bundles={}
    folders={str(Path(n).parent).replace('\\','/') for n in files}
    for folder in sorted(folders):
        prefix='' if folder=='.' else folder+'/'
        present=[n for n in TASK_FILES if prefix+n in files]
        if present:
            bundles[folder]={'status':'task_ready' if len(present)==len(TASK_FILES) else 'matrix_ready' if any(n.endswith('.npz') for n in present) else 'metadata_ready',
                'missing':[n for n in TASK_FILES if n not in present],
                'registration':{'preparation':prep,'subdir':'' if folder=='.' else folder} if len(present)==len(TASK_FILES) else None,
                'validation':'file_completeness_only; registration validates arrays, lineage and dependencies'}
    state='task_ready' if any(v['status']=='task_ready' for v in bundles.values()) else 'matrix_ready' if any(n.endswith(('.npz','.npy','.h5ad','.mtx')) for n in files) else 'metadata_ready' if files else 'no_artifacts'
    record={'preparation':prep,'status':state,'public':public,'source_ids':sorted({v['source'] for v in provenance.values()}),
        'files':files,'bundles':bundles,'content_signature':object_hash({n:v['sha256'] for n,v in files.items()})}
    write_json(root/'artifact_index.json',record)
    return record


def checked_artifact(run,prep,relative,inv):
    root=Path(run)/'actions'/identifier(prep)
    index=read_json(root/'artifact_index.json') if (root/'artifact_index.json').is_file() else artifact_index(run,prep,inv)
    p=inside(root/'output',relative); item=index['files'].get(relative)
    if item is None or not p.is_file() or sha256(p)!=item['sha256']:
        raise ValueError('Preparation artifact changed or missing; create a new preparation version')
    # Do not accept an edited index's privacy flag in lieu of source lineage.
    provenance=read_json(root/'provenance.json')
    public=bool(provenance) and all(k in inv['assets'] and inv['assets'][k]['public'] for k in provenance)
    return p,{**item,'public':public},provenance


def inspect_preparation(run,inv,a):
    prep=identifier(a['preparation']); root=Path(run)/'actions'/prep
    index=read_json(root/'artifact_index.json') if (root/'artifact_index.json').exists() else artifact_index(run,prep,inv)
    offset=max(0,int(a.get('offset',0)))
    if not a.get('path'):
        names=list(index['files']);return {**{k:index[k] for k in ('preparation','status','public','bundles')},
            'artifacts':{n:index['files'][n] for n in names[offset:offset+5]},'total_files':len(names),'next_offset':offset+5 if offset+5<len(names) else None}
    p,item,_=checked_artifact(run,prep,a['path'],inv)
    if item['public'] and p.suffix in ('.json','.jsonl','.csv','.tsv','.txt','.md'):
        with p.open(encoding='utf-8',errors='replace') as f:
            # Character offset, independent of line/array paging.
            f.read(offset); text=f.read(10001)
        return {**item,'text':text[:10000],'next_offset':offset+10000 if len(text)>10000 else None,
            'content_origin':'public_sources_only','untrusted_source_material':True}
    result=inspect_assets({'assets':{'output':{'path':str(p),'name':p.name,'bytes':p.stat().st_size,'source':prep,'public':False}}},['output'],offset)
    return {**result,'ref':item['ref'],'sha256':item['sha256'],
        'readability':'structure_only; read_metadata on original source for approved private sample relationships'}


def preparation_inputs(run,inv,refs):
    mounts={};provenance={};manifest={}
    for ref in refs:
        p,item,raw=checked_artifact(run,ref['preparation'],ref['path'],inv)
        key='derived_'+object_hash(ref)[:16];name=key+''.join(p.suffixes[-2:])
        mounts[name]=p;provenance.update(raw)
        manifest[key]={**item,'container_path':'/input/'+name,'source_ids':sorted({r['source'] for r in raw.values()})}
    return mounts,provenance,manifest
