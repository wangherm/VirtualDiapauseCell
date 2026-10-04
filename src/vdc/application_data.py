"""Post-freeze count readers. No role mutation and no historical biotime inputs."""
from pathlib import Path
import csv
import numpy as np
from .io import read_json,write_json,save_npz,sha256,object_hash
from .pk1_data import _rows,h5_column
from .observation import normalise_expression,score_programmes
from .contracts import ObservationBundle


def align_counts(counts,genes,expected):
    x=np.asarray(counts,float)
    if x.ndim!=2 or x.shape[1]!=len(genes) or len(set(genes))!=len(genes):raise ValueError('Invalid count matrix or duplicate gene IDs')
    if not np.isfinite(x).all() or (x<0).any() or not np.allclose(x,np.round(x)):raise ValueError('Declared integer counts required')
    if set(genes)!=set(expected):raise ValueError('unsupported_gene_universe_mismatch')
    index={g:i for i,g in enumerate(genes)}
    return x[:,[index[g] for g in expected]]


def non_target_input(counts,genes,definitions,hidden):
    """Target counts cannot affect library size, ranks, or ANY overlapping programme."""
    mask=np.broadcast_to(np.array([g not in hidden for g in genes]),np.shape(counts)).copy()
    expression=normalise_expression(counts,mask,'counts')
    scored=score_programmes(expression,mask,genes,definitions)
    return expression,scored


def projection(counts,genes,entries,policy,prepared,reference,config):
    prepared=Path(prepared);contract=read_json(prepared/'expression_contract.json')
    expected=contract['gene_ids'];x=align_counts(counts,genes,expected)
    definitions=read_json(prepared/'programmes.json')
    y=normalise_expression(x,np.ones_like(x,bool),'counts')
    s=score_programmes(y,np.ones_like(y,bool),expected,definitions,
        config['min_genes_per_programme'],config['min_programme_coverage'])
    template=ObservationBundle.load(reference/'bundle')
    if s['feature_ids']!=template.feature_ids:raise ValueError('Frozen programme order mismatch')
    rows=_rows(policy,entries)
    scopes=[]
    for r,e in zip(rows,entries):
        if e['role'] not in config['allowed_query_roles'] or e['split']!='locked_test':raise ValueError('Non-query role in application')
        if 'locate' not in e['allowed_tasks']:raise ValueError('Sample does not allow location queries')
        r['application_role']=e['role'];r['source_material']=e.get('resolution',e['source'])
        r['application_kind']='current_expression_location_not_future_prediction'
        if 'cell_type' in e:
            r['cell_type']=e['cell_type'];r['cell_count']=e['cell_count']
            r['observation_id']+=':cell_type:'+e['cell_type']
        scopes.append({'observation_id':r['observation_id'],'source_material':r['source_material'],
            'source_count_provenance':e.get('count_provenance','locked_raw_counts'),
            'target_context':template.context,'operation':'explicit_counts_to_frozen_programme_representation',
            'qualification':'same_cohort_later_observation_not_independent_pool_test' if e['role']=='temporal_query' else 'material_or_condition_applicability_stress_test',
            'calibration_claim':False})
    # Context is assigned only by this versioned, audited projection. StatePredictor's scope check remains intact.
    b=ObservationBundle(s['values'],s['mask'],s['coverage'],template.feature_ids,rows,template.context,
        'unit_holdout',np.zeros(len(rows)),np.zeros(len(rows),bool),template.clock_reference_id).validate()
    return b,y,x,expected,definitions,scopes


def bulk_queries(private,policy,source,development=False):
    root=Path(private);roles={'development'} if development else {'reserved_evaluation','prediction_only','temporal_query'}
    entries=[r for r in policy['samples'] if r['source']==source and r['role'] in roles]
    if source=='single_exit':
        genes=read_json(root/'input/single_exit_gene_ids.json');xs=[]
        for e in entries:
            with np.load(root/'input'/('exit_counts_'+e['sample_id']+'.npz'),allow_pickle=False) as a:xs.append(a['counts'])
        return np.array(xs),genes,entries
    with (root/'input/bulk_counts.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);head=next(reader);columns=[head.index(e['sample_id']+'_sorted') for e in entries];genes=[];values=[]
        for row in reader:genes.append(row[0]);values.append([float(row[j]) for j in columns])
    return np.array(values).T,genes,entries


def development_counts(private,policy,view,bundle,genes):
    root=Path(private)
    if view=='bulk':
        matrices=[];keys=[]
        for source in ('bulk','single_exit'):
            x,g,e=bulk_queries(root,policy,source,development=True)
            matrices.append(align_counts(x,g,genes));keys.extend(r['sample_key'] for r in e)
        counts=np.concatenate(matrices);index={k:i for i,k in enumerate(keys)}
        return counts[[index[r['source_sample_key']] for r in bundle.rows]]
    import h5py
    with h5py.File(root/'input/core.h5ad','r') as f:
        names=h5_column(f['obs/sample']);types=h5_column(f['obs/cell_type']);g=h5_column(f['var/_index']).tolist()
        if g!=genes:raise ValueError('Development gene order changed')
        matrix=f['layers/counts'];ptr=matrix['indptr'][:];values=[]
        for r in bundle.rows:
            selected=names==next(e['sample_id'] for e in policy['samples'] if e['sample_key']==r['source_sample_key'])
            if view=='core_celltypes':selected &= types==r['cell_type']
            total=np.zeros(len(genes))
            for i in np.flatnonzero(selected):np.add.at(total,matrix['indices'][ptr[i]:ptr[i+1]],matrix['data'][ptr[i]:ptr[i+1]])
            values.append(total)
    return align_counts(np.array(values),genes,genes)


def sc_queries(supplement,policy,view):
    supplement=Path(supplement);m=read_json(supplement/'manifest.json')
    if m['role_hash']!=policy['approval']['content_hash']:raise ValueError('Supplement role identity differs')
    path=supplement/(view+'.npz')
    if sha256(path)!=m['files'][path.name]:raise ValueError('Supplement counts changed')
    with np.load(path,allow_pickle=False) as a:counts=a['counts'].copy();genes=a['genes'].tolist()
    allowed={r['sample_key']:r for r in policy['samples']};entries=[]
    for item in m['rows'][view]:
        original=allowed[item['sample_key']]
        if original['role'] not in {'reserved_evaluation','prediction_only','temporal_query'}:raise ValueError('Non-query sample in supplement')
        entries.append({**original,**{k:item[k] for k in ('cell_type','cell_count','count_provenance') if k in item}})
    if len(entries)!=len(counts):raise ValueError('Supplement row alignment mismatch')
    return counts,genes,entries,m


def prepare_sc_supplement(archive,roles,out,config,source_cache=None,programmes=None):
    """Aggregate only approved query rows; no predictions or historical clock fields are read.

    The source registry explicitly calls Exit X count-like with incomplete provenance.
    Scaled/normalised stress sources are never silently converted to UMI counts.
    """
    import io,zipfile,h5py
    from .admission import role_manifest
    from .pk1_data import _member,_extract
    policy,_=role_manifest(roles);out=Path(out);cache=Path(source_cache) if source_cache else out/'cache'
    if (out/'manifest.json').exists():raise FileExistsError('Supplement is immutable; choose a new output')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'preparation_protocol.json',{'config':config,'config_hash':object_hash(config),
        'roles_hash':policy['approval']['content_hash'],'operation':'raw query pseudobulk preparation only; no model fitting or inference'})
    with zipfile.ZipFile(archive) as z:
        nested=[n for n in z.namelist() if n.endswith('Killifish_Thesis_Data_Code_Archive_v1.zip')]
        if nested:
            expected=z.read(nested[0]+'.sha256').decode().split()[0]
            inner=_extract(z,nested[0],out/'cache/source.zip',expected)
        else:inner=Path(archive)
    with zipfile.ZipFile(inner) as z:
        registry=list(csv.DictReader(io.StringIO(z.read(_member(z,'metadata/h5ad_source_registry.csv')).decode('utf-8-sig'))))
        rec=next(r for r in registry if r['source_id']=='exit_timecourse')
        file=_extract(z,_member(z,'single_cell/originals/'+rec['source_filename']),cache/'exit.h5ad',rec['sha256'])
        stress_record=next(r for r in registry if r['source_id']=='combined_processed')
        stress_file=_extract(z,_member(z,'single_cell/originals/'+stress_record['source_filename']),cache/'combined.h5ad',stress_record['sha256']) if programmes else None
    requested={r['sample_id']:r for r in policy['samples'] if r['source']=='sc_reserved' and r['role'] in config['allowed_query_roles']}
    results={v:[] for v in ('core','core_celltypes')};rows={v:[] for v in results};missing=[]
    with h5py.File(file,'r') as f:
        # Only explicit source sample identity / archived cell type are admitted metadata.
        sample_key=next((k for k in ('sample','Sample','sample_id') if k in f['obs']),None)
        type_key=next((k for k in ('cell_type','predicted_cell_type','predicted_labels','predicted') if k in f['obs']),None)
        if sample_key is None:raise ValueError('Exit source has no supported explicit sample column; inspect metadata without guessing groups')
        names=h5_column(f['obs'][sample_key]);types=h5_column(f['obs'][type_key]) if type_key else np.full(len(names),'unavailable')
        genes=h5_column(f['var/_index']).tolist();matrix=f['X']
        if not isinstance(matrix,h5py.Group) or matrix.attrs.get('encoding-type')!='csr_matrix':raise ValueError('Explicit CSR Exit X required')
        ptr=matrix['indptr'][:]
        for sample,e in requested.items():
            indices=np.flatnonzero(names==sample)
            if not len(indices):missing.append({'sample_key':e['sample_key'],'reason':'not_in_count_like_exit_source; other archive X/raw matrices have unverified scale'});continue
            whole=np.zeros(len(genes));by_type={};sizes={}
            for i in indices:
                values=matrix['data'][ptr[i]:ptr[i+1]];columns=matrix['indices'][ptr[i]:ptr[i+1]]
                if not np.isfinite(values).all() or (values<0).any() or not np.allclose(values,np.round(values)):raise ValueError('Exit X is not integer count-like')
                np.add.at(whole,columns,values);t=str(types[i]);sizes[t]=sizes.get(t,0)+1
                if t not in by_type:by_type[t]=np.zeros(len(genes))
                np.add.at(by_type[t],columns,values)
            provenance='historical_Exit_X_integer_count_like; UMI_provenance_incomplete; applicability_test'
            results['core'].append(whole);rows['core'].append({'sample_key':e['sample_key'],'count_provenance':provenance,'cell_count':len(indices)})
            if type_key:
                for t,x in sorted(by_type.items()):
                    if sizes[t]>=config['min_cells_per_type']:
                        results['core_celltypes'].append(x);rows['core_celltypes'].append({'sample_key':e['sample_key'],'cell_type':t,'cell_count':sizes[t],'count_provenance':provenance})
            else:missing.append({'sample_key':e['sample_key'],'view':'core_celltypes','reason':'no_explicit_archived_type_column'})
    files={}
    for view in results:
        if results[view]:save_npz(out/(view+'.npz'),counts=np.asarray(results[view],np.int64),genes=np.array(genes));files[view+'.npz']=sha256(out/(view+'.npz'))
    stress=None
    if stress_file:
        from .application_stress import prepare_normalised
        stress=prepare_normalised(stress_file,stress_record,policy,read_json(programmes),out,config);files.update(stress['files'])
    write_json(out/'manifest.json',{'role_hash':policy['approval']['content_hash'],'source_registry':rec,'source_sha256':sha256(file),
        'files':files,'rows':rows,'unavailable':missing,'model_queries_executed':False,'historical_clock_loaded':False,
        'normalised_or_scaled_stress_matrices_not_converted_to_counts':True,'type_annotation':'historical source annotation; exact label match only for conditional waves; no ontology remapping',
        'source_type_column':type_key,'stress_substitute':stress})
    return out
