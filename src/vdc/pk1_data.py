"""Private archive admission and real count-derived views, using existing observation contracts."""
from pathlib import Path
import csv
import hashlib
import io
import json
import zipfile
import re
import numpy as np
from .io import read_json, write_json, sha256, object_hash, save_npz
from .admission import role_manifest
from .contracts import ObservationBundle
from .observation import normalise_expression, score_programmes


def h5_column(obj):
    import h5py
    if isinstance(obj, h5py.Dataset):
        return obj.asstr()[:] if h5py.check_string_dtype(obj.dtype) else obj[:]
    if 'categories' in obj:
        cats = h5_column(obj['categories']); codes = obj['codes'][:]
        return np.array([str(cats[i]) if i >= 0 else '' for i in codes])
    if 'values' in obj and 'mask' in obj:
        values = h5_column(obj['values']).astype(object); values[obj['mask'][:]] = ''
        return values.astype(str)
    raise ValueError('Unsupported H5AD metadata encoding')


def _member(z, suffix):
    found = [n for n in z.namelist() if n.endswith('/' + suffix) or n == suffix]
    if len(found) != 1: raise ValueError('Archive member missing/ambiguous: ' + suffix)
    return found[0]


def _extract(z, member, dest, expected=None):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected and sha256(dest) == expected: return dest
        if not expected: raise FileExistsError('Unverified existing extraction: ' + dest.name)
    tmp = dest.with_name(dest.name + '.partial'); h = hashlib.sha256()
    with z.open(member) as src, tmp.open('wb') as dst:
        while block := src.read(8 * 1024 * 1024): h.update(block); dst.write(block)
    if expected and h.hexdigest() != expected: raise ValueError('Archive member checksum mismatch')
    tmp.replace(dest)
    return dest


def import_archive(archive, private_root, exit_archive=None, excluded_exit_samples=()):
    """Extract only declared sources. Never execute historical notebooks or expose download URLs."""
    private_root = Path(private_root).resolve(); private_root.mkdir(parents=True, exist_ok=True)
    if (private_root/'sample_roles.json').exists():
        raise FileExistsError('Existing reviewed inputs are immutable; use a new private root for reimport')
    archive = Path(archive).resolve()
    with zipfile.ZipFile(archive) as outer:
        nested = [n for n in outer.namelist() if n.endswith('Killifish_Thesis_Data_Code_Archive_v1.zip')]
        if nested:
            expected = outer.read(nested[0] + '.sha256').decode().split()[0]
            inner = _extract(outer, nested[0], private_root/'archive/source.zip', expected)
        else: inner = archive
    with zipfile.ZipFile(inner) as z:
        def table(suffix): return list(csv.DictReader(io.StringIO(z.read(_member(z, suffix)).decode('utf-8-sig'))))
        registry = table('metadata/h5ad_source_registry.csv')
        core = next(r for r in registry if r['source_id'] == 'core_biotime')
        _extract(z, _member(z, 'single_cell/originals/' + core['source_filename']), private_root/'input/core.h5ad', core['sha256'])
        bundles = table('metadata/bulk_bundle_index.csv')
        b = next(r for r in bundles if r['bundle'] == 'bulk_05_shared_analyses')
        bulk_zip = _extract(z, _member(z, 'bulk/bulk_05_shared_analyses.zip'), private_root/'input/bulk_shared.zip', b['sha256'])
        with zipfile.ZipFile(bulk_zip) as bz:
            name = _member(bz, 'data_counts/counts_matrix.csv')
            data = bz.read(name)
            (private_root/'input/bulk_counts.csv').write_bytes(data)
        write_json(private_root/'input/bulk_metadata.json', table('metadata/bulk_sample_metadata.csv'))
        write_json(private_root/'input/sc_metadata.json', table('metadata/single_cell_sample_summary.csv'))
        inventory = table('manifest/source_file_inventory.csv')
        legacy = [r for r in inventory if 'regulon' in r['archive_path'].lower()]
        write_json(private_root/'input/legacy_network_inventory.json', {'status':'legacy_only_fit_units_not_established', 'members':legacy})
    if exit_archive:
        import_exit_counts(exit_archive, private_root/'input', excluded_exit_samples)
    files = {p.name: sha256(p) for p in sorted((private_root/'input').iterdir()) if p.is_file()}
    write_json(private_root/'source_lock.json', {'files':files, 'internal':True, 'remote_byte_identity':'not_assumed',
        'single_embryo_counts':'separate_featurecounts_archive' if exit_archive else 'not_supplied',
        'historical_code_executed':False})
    return draft_roles(private_root)


def read_featurecounts(stream):
    """One verified single-library featureCounts table; filenames do not supply time labels."""
    text = io.TextIOWrapper(stream, encoding='utf-8-sig')
    first = text.readline()
    header = text.readline() if first.startswith('#') else first
    fields = header.rstrip('\r\n').split('\t')
    if fields[:6] != ['Geneid','Chr','Start','End','Strand','Length'] or len(fields) != 7:
        raise ValueError('Expected a single-library featureCounts table')
    genes=[]; counts=[]
    for r in csv.reader(text, delimiter='\t'):
        if len(r) != 7: raise ValueError('Malformed count row')
        v=float(r[-1])
        if not np.isfinite(v) or v < 0 or v != int(v): raise ValueError('Expected nonnegative integer counts')
        genes.append(r[0]); counts.append(v)
    if not genes or len(set(genes)) != len(genes): raise ValueError('Missing or duplicate gene IDs')
    x=np.asarray(counts, dtype=np.int64)
    if x.sum() <= 0: raise ValueError('Empty count library')
    return genes,x


def import_exit_counts(archive, out, excluded_samples=()):
    """Audit all libraries; preserving an excluded library is not permission to fit it."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    samples=[]; audits=[]; gene_order=None
    excluded_samples=set(str(s) for s in excluded_samples)
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if re.search(r'/\d+_counts\.txt$', '/'+n))
        for name in names:
            sid=Path(name).name.removesuffix('_counts.txt')
            if sid in samples: raise ValueError('Ambiguous duplicate Exit library')
            with z.open(name) as f: genes,counts=read_featurecounts(f)
            if gene_order is None: gene_order=genes
            if genes != gene_order: raise ValueError('Exit gene order mismatch; explicit ID alignment required')
            summary_name=name+'.summary'; summary=None
            if summary_name in z.namelist():
                summary=list(csv.DictReader(io.StringIO(z.read(summary_name).decode()),delimiter='\t'))
                assigned=[r for r in summary if r['Status']=='Assigned']
                if len(assigned)!=1 or float(list(assigned[0].values())[1]) != counts.sum():
                    raise ValueError('Assigned reads disagree with count sum')
            audits.append({'sample_id':sid,'genes':len(genes),'total_counts':int(counts.sum()),
                'detected_genes':int((counts>0).sum()),'summary_available':summary is not None,
                'counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),
                'historical_exclusion':sid in excluded_samples,'precise_elapsed_hours':None})
            samples.append(sid)
            save_npz(out/('exit_counts_'+sid+'.npz'), counts=counts)
    if not samples: raise ValueError('No Exit featureCounts files found')
    if excluded_samples-set(samples):raise ValueError('Excluded sample absent from archive')
    write_json(out/'single_exit_gene_ids.json',gene_order)
    write_json(out/'single_exit_audit.json',{'archive_sha256':sha256(archive),'libraries':audits,
        'gene_order_hash':object_hash(gene_order),'sample_number_is_not_time':True})
    return audits


def draft_roles(private_root):
    import h5py
    root=Path(private_root); samples=[]
    with h5py.File(root/'input/core.h5ad','r') as f:
        names=h5_column(f['obs/sample']); conditions=h5_column(f['obs/condition'])
        for name in sorted(set(names)):
            condition=set(conditions[names==name])
            if len(condition)!=1:raise ValueError('Core sample has conflicting conditions')
            cond=next(iter(condition)); key='sc:'+name
            split='validation' if name.endswith('2') or name=='Exit_G2_D1' else 'train'
            if name.startswith('Long_term'):split='validation' if name.endswith('3') else 'train'
            allowed=cond in {'Developing','Early Diapause','Late Diapause','Exit'}
            samples.append({'sample_key':key,'source':'core','sample_id':name,'condition':cond,
                'biological_unit':key,'link_ids':[key], 'role':'development' if allowed else 'reserved_evaluation',
                'split':split if allowed else 'locked_test','allowed_tasks':['state','clock','waves'] if allowed else ['locate'],
                'cohort_id': 'sc_exit:'+name.split('_')[1] if name.startswith('Exit_G') else None,
                'evidence':'source pool labels; G1/G2 temporal lineage confirmed by owner; common batch is not an independent batch test'})
    known={r['sample_id'] for r in samples}
    for r in read_json(root/'input/sc_metadata.json'):
        if r['sample'] in known: continue
        name=r['sample'];key='sc:'+name
        temporal=name.startswith(('Exit_G1_','Exit_G2_'))
        group=name.split('_')[1] if temporal else None
        samples.append({'sample_key':key,'source':'sc_reserved','sample_id':name,'condition':r['condition'],
            'biological_unit':key,'link_ids':[key],'role':'temporal_query' if temporal else 'reserved_evaluation',
            'split':'locked_test','allowed_tasks':['locate','within_cohort_future'] if temporal else ['locate'],
            'cohort_id':'sc_exit:'+group if temporal else None,
            'parent_sample_key':'sc:Exit_'+group+'_D1' if temporal else None,
            'evaluation_scope':'within_cohort_future_only' if temporal else 'reserved_challenge',
            'evidence':'owner confirmed same source cohort sampled across days; no independent-batch claim' if temporal else 'archive metadata'})
    for r in read_json(root/'input/bulk_metadata.json'):
        key='bulk:'+r['sample_id']; cond=r['condition_standardized']
        allowed=r['resolution']=='pooled_exit' or (r['resolution']=='whole_embryo' and cond in {'Developing Day 7','Early Diapause','Late Diapause'})
        split='validation' if (r['resolution']=='pooled_exit' and r['replicate']=='3') or (r['resolution']=='whole_embryo' and r['replicate']=='4') else 'train'
        links=[key]
        if r.get('linked_embryo_key'):links.append('regional:'+r['linked_embryo_key'])
        samples.append({'sample_key':key,'source':'bulk','sample_id':r['sample_id'],'condition':cond,
            'biological_unit':key if not r.get('linked_embryo_key') else 'regional:'+r['linked_embryo_key'], 'link_ids':links,
            'role':'development' if allowed else 'prediction_only' if 'stalled' in cond.lower() else 'reserved_evaluation',
            'split':split if allowed else 'locked_test','allowed_tasks':['state','clock','waves','transition'] if allowed else ['locate'],
            'evidence':r['source'], 'resolution':r['resolution'],
            'elapsed_hours':float(r['time_value']) if r['time_unit']=='hour' and r['time_reference']=='after_exit_induction' else None})
    if (root/'input/single_exit_audit.json').exists():
        libs=read_json(root/'input/single_exit_audit.json')['libraries']
        eligible=sorted([r['sample_id'] for r in libs if not r['historical_exclusion']],
            key=lambda sid:hashlib.sha256(('VDC_PK1_single_exit_roles_v1:'+sid).encode()).hexdigest())
        # Metadata-only deterministic allocation, fixed before any expression-based selection.
        if len(eligible)!=29:raise ValueError('Expected 29 historically admitted Exit libraries; review a new protocol')
        for r in libs:
            sid=r['sample_id'];key='single_exit:'+sid
            role='excluded' if r['historical_exclusion'] else 'development' if sid in eligible[:20] else 'reserved_evaluation'
            split=('validation' if sid in eligible[16:20] else 'train') if role=='development' else 'locked_test'
            samples.append({'sample_key':key,'source':'single_exit','sample_id':sid,'condition':'Exit',
                'biological_unit':key,'link_ids':[key],'role':role,'split':split,
                'allowed_tasks':['state','clock','waves'] if role=='development' else ['locate'] if role!='excluded' else [],
                'elapsed_hours':None,'evidence':'raw featureCounts; owner confirmed historical exclusion' if role=='excluded' else 'raw featureCounts; sample ID is not elapsed time'})
    policy={'protocol':'VDC_PK1','status':'draft','source_lock_hash':object_hash(read_json(root/'source_lock.json')),
        'blocking_questions':[],
        'notes':['Owner confirmed same G1/G2 source cohorts across time; temporal queries are not independent-unit tests',
                 'Owner confirmed historical exclusion is retained',
                 'No legacy biotime used for labels or role selection',
                 'Unknown between-condition lineage is not claimed as proven independence',
                 'Reserved expression must not be used for fitting; future queries require a frozen snapshot'], 'samples':samples}
    path=root/'sample_roles.json'
    if path.exists():raise FileExistsError('Existing role review is preserved; use a new private root for reimport')
    write_json(path,policy)
    return path


def go_definitions(tsv, feature_ids):
    members={fid:{} for fid in feature_ids}
    with Path(tsv).open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['GO term accession'] in members:members[r['GO term accession']][r['Gene stable ID']]=1.
    return [{'id':fid,'members':members[fid],'source_ref':'Ensembl BioMart snapshot SHA256:'+sha256(tsv)} for fid in feature_ids if len(members[fid])>=3]


def _rows(policy, entries, suffix=''):
    result=[]
    for e in entries:
        result.append({'observation_id':e['sample_key']+suffix,'biological_unit':e['biological_unit'],
            'source_sample_key':e['sample_key'],'study_family':'internal_killifish_PK1', 'split':e['split'],
            'origin':'internal','link_ids':e['link_ids'], 'condition':e['condition'],
            'cohort_id':e.get('cohort_id'),'elapsed_hours_since_release':e.get('elapsed_hours'),
            'reference_eligible':'late' not in e['condition'].lower(),
            'admission_protocol':'VDC_PK1','admission_hash':policy['approval']['content_hash']})
    return result


def prepare_killifish(private_root, out, annotation, feature_ids, modality):
    import h5py
    root=Path(private_root); out=Path(out);policy,role_map=role_manifest(root/'sample_roles.json')
    if out.exists():raise FileExistsError('Prepared view is immutable; choose a new output directory')
    dropped=[]
    lock=read_json(root/'source_lock.json')
    if object_hash(lock)!=policy['source_lock_hash']:raise ValueError('Source lock changed after role review')
    for name, expected in lock['files'].items():
        if sha256(root/'input'/name)!=expected:raise ValueError('Private input changed: '+name)
    source='core' if modality=='core_celltypes' else modality
    entries=[r for r in policy['samples'] if r['source']==source and r['role']=='development']
    if not entries:raise ValueError('No admitted development samples')
    if modality=='bulk':
        ids=[r['sample_id'] for r in entries];genes=[];matrix=[]
        with (root/'input/bulk_counts.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f);head=next(reader)
            columns=[head.index(k+'_sorted') for k in ids]
            for row in reader:genes.append(row[0]);matrix.append([float(row[j]) for j in columns])
        x=np.array(matrix).T
        if (root/'input/single_exit_gene_ids.json').exists():
            exit_genes=read_json(root/'input/single_exit_gene_ids.json')
            if set(exit_genes)!=set(genes):raise ValueError('Bulk/Exit gene universe differs; explicit alignment policy required')
            index={g:i for i,g in enumerate(exit_genes)};order=[index[g] for g in genes]
            extra=[r for r in policy['samples'] if r['source']=='single_exit' and r['role']=='development']
            arrays=[]
            for e in extra:
                with np.load(root/'input'/('exit_counts_'+e['sample_id']+'.npz'),allow_pickle=False) as a:
                    arrays.append(a['counts'][order])
            if arrays:x=np.concatenate([x,np.array(arrays)]);entries.extend(extra)
    elif modality in {'core','core_celltypes'}:
        with h5py.File(root/'input/core.h5ad','r') as f:
            names=h5_column(f['obs/sample']);genes=list(h5_column(f['var/_index']))
            selected=np.isin(names,[r['sample_id'] for r in entries])
            counts=f['layers/counts']
            if counts.attrs.get('encoding-type')!='csr_matrix':raise ValueError('Explicit CSR counts required')
            # Load only authorized rows, including a guard against future source expansion.
            ptr=counts['indptr'][:];ind=counts['indices'];data=counts['data'];x=[];expanded=[]
            types=h5_column(f['obs/cell_type']) if modality=='core_celltypes' else np.full(len(names),'all')
            for e in entries:
                which=np.flatnonzero(selected & (names==e['sample_id']))
                summed={str(t):np.zeros(len(genes),dtype=np.float64) for t in set(types[which])}
                cell_counts={str(t):int((types[which]==t).sum()) for t in set(types[which])}
                for i in which:
                    v=data[ptr[i]:ptr[i+1]]
                    if (v<0).any() or not np.allclose(v,np.round(v)):raise ValueError('Core layer is not integer counts')
                    np.add.at(summed[str(types[i])],ind[ptr[i]:ptr[i+1]],v)
                for t in sorted(summed):
                    if modality=='core_celltypes' and cell_counts[t]<20:
                        dropped.append({'sample_key':e['sample_key'],'cell_type':t,'cells':cell_counts[t],
                                        'reason':'fewer_than_20_observed_cells'})
                        continue
                    x.append(summed[t]);expanded.append({**e,'cell_type':t,'cell_count':cell_counts[t]})
            entries=expanded
            x=np.array(x)
    else:raise ValueError('Unknown modality')
    rows=_rows(policy,entries)
    for r,e in zip(rows,entries):
        if 'cell_type' in e:
            r['cell_type']=e['cell_type'];r['cell_count']=e['cell_count']
            if modality=='core_celltypes':r['observation_id']+=':cell_type:'+e['cell_type']
    genes=[str(g) for g in genes]
    if len(set(genes))!=len(genes):raise ValueError('Duplicate gene identifiers')
    definitions=go_definitions(annotation,feature_ids)
    if [p['id'] for p in definitions]!=feature_ids:raise ValueError('Shared programme membership mismatch')
    y=normalise_expression(x,np.ones_like(x,bool),'counts')
    scored=score_programmes(y,np.ones_like(y,bool),genes,definitions)
    if not scored['mask'].all():raise ValueError('Insufficient programme coverage; revise a versioned panel explicitly')
    context={'species':'Nothobranchius furzeri','system':'PK1_development_states',
        'material':'whole_and_five_embryo_pools' if modality=='bulk' else 'pool_by_cell_type_pseudobulk' if modality=='core_celltypes' else 'single_cell_derived_pool_pseudobulk',
        'assay':'bulk_counts' if modality=='bulk' else 'sum_of_documented_UMI_counts',
        'programme_definition_id':object_hash(definitions),'preprocessing_id':object_hash({'scale':'counts','gene_ids':genes}),
        'admission_hash':policy['approval']['content_hash']}
    b=ObservationBundle(scored['values'],scored['mask'],scored['coverage'],feature_ids,rows,context,'unit_holdout',
        np.zeros(len(rows)),np.zeros(len(rows),bool),None).validate()
    b.save(out);save_npz(out/'gene_expression.npz',expression=y.astype('float32'),genes=np.array(genes))
    write_json(out/'programmes.json',definitions)
    write_json(out/'expression_contract.json',{'input_scale':'counts','gene_ids':genes,'context':context,
        'expression_sha256':sha256(out/'gene_expression.npz')})
    write_json(out/'admission.json',{'origin':'internal','role_hash':policy['approval']['content_hash'],
        'samples':len(rows),'independent_units':len(set(r['biological_unit'] for r in rows)),
        'reserved_expression_used_for_model':False,'source_quality_audit_includes_all_libraries':True,
        'single_cell_model':False,'resolution':context['material'],
        'cell_type_state':'pool_by_cell_type_observations' if modality=='core_celltypes' else 'not_claimed_for_whole_pool_model',
        'independent_batch_generalisation':False})
    write_json(out/'observation_coverage.json',{'minimum_cells_per_cell_type_pool':20 if modality=='core_celltypes' else None,
        'dropped_observations':dropped,'cell_type_annotation':'archived core annotation; not a newly independently validated classifier' if modality!='bulk' else None})
    return b
