"""Lossless public acquisition views; these are not admitted training bundles."""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import io
import re
import tarfile
import numpy as np
from .io import read_json,write_json,write_jsonl,save_npz,sha256


def family_metadata(path, table_callback=None):
    samples=[];current=None;columns=None;selected=False;ids=[];values=[]
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for raw in f:
            line=raw.rstrip('\r\n')
            if line.startswith('^'):
                if current is not None:samples.append(current)
                current={'accession':line.split(' = ',1)[1],'fields':{}} if line.startswith('^SAMPLE = ') else None
                columns=None;selected=False
            elif current is not None and line.startswith('!Sample_') and ' = ' in line:
                key,value=line.split(' = ',1);current['fields'].setdefault(key[1:],[]).append(value)
            elif current is not None and line=='!sample_table_begin':
                selected=table_callback is not None and 'Dauer MTC#' in current['fields'].get('Sample_title',[''])[0]
                columns=[];ids=[];values=[]
            elif current is not None and line=='!sample_table_end':
                if selected:table_callback(current,ids,values)
                columns=None;selected=False
            elif selected and columns is not None:
                fields=line.split('\t')
                if not columns:
                    columns=fields
                    if 'ID_REF' not in columns or 'VALUE' not in columns:raise ValueError('SOFT expression table lacks ID_REF/VALUE')
                else:
                    ids.append(fields[columns.index('ID_REF')]);value=fields[columns.index('VALUE')]
                    values.append(float(value) if value not in {'','null','NA','NaN'} else np.nan)
        if current is not None:samples.append(current)
    if not samples or len({s['accession'] for s in samples})!=len(samples):raise ValueError('Invalid family sample list')
    return samples


def read_table(path):
    with gzip.open(path,'rt',encoding='utf-8-sig') as f:
        rows=csv.reader((line for line in f if not line.startswith('#')),delimiter='\t')
        header=next(rows);body=list(rows)
    if len(header)<2 or not body or any(len(r)!=len(header) for r in body):raise ValueError('Ragged/empty table')
    return header,body


def sample_mapping(columns,samples,kind):
    mapping={}
    for s in samples:
        fields=s['fields'];keys=fields.get('Sample_description',[]) if kind=='description' else fields.get('Sample_title',[])
        if kind=='NHDF_set1':
            title=fields['Sample_title'][0]
            match=re.fullmatch(r'NHDF, (.+) \(set1\)',title)
            if not match:raise ValueError('Unrecognised NHDF title')
            key=match[1];keys=[key+'_Exp1' if key=='unsynch_Rep1' else key]
        for key in keys:
            if key in mapping and mapping[key]!=s['accession']:raise ValueError('Ambiguous sample mapping')
            mapping[key]=s['accession']
    missing=[c for c in columns if c not in mapping]
    if missing:raise ValueError('Unresolved expression sample names: '+', '.join(missing))
    return [mapping[c] for c in columns]


def numeric_table(path,out,scale,samples,mapping_kind=None):
    header,rows=read_table(path)
    if {'baseMean','log2FoldChange','padj'}.issubset(header):
        raise ValueError('Differential statistics are not sample expression')
    features=[r[0] for r in rows]
    if len(set(features))!=len(features):raise ValueError('Duplicate feature IDs require explicit reconciliation')
    x=np.array([r[1:] for r in rows],float).T
    if not np.isfinite(x).all():raise ValueError('Non-finite numeric table')
    if scale in {'counts','FPKM'} and (x<0).any():raise ValueError('Negative abundance')
    if scale=='counts' and not np.equal(x,np.floor(x)).all():raise ValueError('Counts are not integers')
    mapped=sample_mapping(header[1:],samples,mapping_kind) if mapping_kind else None
    save_npz(Path(out)/'expression.npz',values=x,mask=np.ones_like(x,bool))
    result={'feature_ids':features,'columns':header[1:],'sample_accessions':mapped,'scale':scale,
        'input_sha256':sha256(path),'shape':list(x.shape),'orientation':'samples_by_features',
        'ERCC_feature_count':sum(g.startswith('ERCC-') for g in features),
        'sample_mapping_status':'explicit_title_alias' if mapping_kind=='NHDF_set1' else 'exact' if mapped else 'not_resolved',
        'training_admitted':False,'split_assignment':'not_assigned',
        'preprocessing':'published_values_preserved; no new normalisation or imputation'}
    write_json(Path(out)/'expression.json',result)
    return {k:v for k,v in result.items() if k not in {'feature_ids','columns','sample_accessions'}}


def tet_loci(path,out,samples):
    matrices=[];all_keys=set();accessions=[];duplicated={};lookup={s['accession'] for s in samples}
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():continue
            match=re.fullmatch(r'(GSM\d+)_.*_TPM\.bed\.gz',Path(member.name).name)
            if not match or match[1] not in lookup:raise ValueError('Unexpected TET archive member')
            # Read in memory, never extract arbitrary tar paths to the filesystem.
            rows=list(csv.reader(io.StringIO(gzip.decompress(tar.extractfile(member).read()).decode('utf-8')),delimiter='\t'))
            if not rows or len(rows[0])!=7 or rows[0][:6]!=['#chr','start','end','gene_name','gene_ID','strand'] or any(len(r)!=7 for r in rows):raise ValueError('Unexpected TPM columns')
            values={};gene_counts=defaultdict(int)
            for r in rows[1:]:
                key=(r[4],r[0],r[1],r[2],r[5]);value=float(r[6])
                if key in values or not np.isfinite(value) or value<0:raise ValueError('Invalid/duplicate TET locus')
                values[key]=value;gene_counts[r[4]]+=1
            all_keys.update(values);matrices.append(values);accessions.append(match[1])
            duplicated[match[1]]={g:n for g,n in gene_counts.items() if n>1}
    if set(accessions)!=lookup or len(accessions)!=len(lookup):raise ValueError('TET archive does not cover sample metadata exactly')
    keys=sorted(all_keys);index={key:i for i,key in enumerate(keys)}
    x=np.zeros((len(matrices),len(keys)));mask=np.zeros_like(x,bool)
    for i,values in enumerate(matrices):
        for key,v in values.items():x[i,index[key]]=v;mask[i,index[key]]=True
    save_npz(Path(out)/'locus_expression.npz',values=x,mask=mask)
    write_json(Path(out)/'locus_expression.json',{'sample_accessions':accessions,'scale':'published_TPM',
        'features':[dict(zip(['gene_id','chromosome','start','end','strand'],k)) for k in keys],
        'duplicate_gene_ids':duplicated,'missing_loci_are_masked_not_measured_zero':True,'training_admitted':False,
        'gene_aggregation':'not_performed; review locus-to-gene policy before programme scoring'})
    return {'samples':len(matrices),'loci':len(keys),'gene_aggregation':'pending','training_admitted':False}


def acquire_study(accession,files,raw,out,download,fetcher):
    raw=Path(raw);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths={e['file']:fetcher(e,raw,download) for e in files}
    primary=[]
    def dauer_table(sample,ids,values):
        title=sample['fields']['Sample_title'][0]
        match=re.search(r'Dauer MTC#(\d+)T(\d+)(-2)?',title)
        if not match or len(ids)!=len(set(ids)):raise ValueError('Ambiguous Dauer table or probe identifiers')
        platform=sample['fields']['Sample_platform_id'][0];x=np.array(values,float)
        mask=np.isfinite(x);folder=out/'dauer_mtc'/sample['accession']
        save_npz(folder/'expression.npz',values=np.where(mask,x,0),mask=mask)
        write_json(folder/'expression.json',{'sample_accession':sample['accession'],'platform':platform,
            'probe_ids':ids,'scale':'log2_channel2_over_common_reference','training_admitted':False})
        primary.append({'sample_accession':sample['accession'],'platform':platform,
            'series_group':f'{accession}:Dauer_MTC:{match[1]}','nominal_time_token':match[2],
            'elapsed_hours':'not_assigned_until_protocol_mapping','replacement_hybridisation':bool(match[3]),
            'biological_unit_grouping':'starting_pool_series','gene_mapping':'pending'})
    samples=family_metadata(paths[accession+'_family.soft.gz'],dauer_table if accession=='GSE3169' else None)
    write_jsonl(out/'samples.jsonl',samples)
    result={'accession':accession,'sample_metadata_count':len(samples),'files_verified':len(paths),
        'training_admitted':False,'role':'public_candidate_development','private_data_used':False,
        'processing_status':'acquired','views':{},'remaining':['grouped_split_and_programme_mapping']}
    if accession=='GSE221467':
        result['views']['locus_TPM']=tet_loci(paths['GSE221467_RAW.tar'],out/'locus_TPM',samples)
        result['remaining'].append('locus_to_gene_aggregation')
    elif accession=='GSE202844':
        for name,scale in [('GSE202844_M3KO_RNAseq_raw_counts_with_ERCCs.txt.gz','counts'),
                           ('GSE202844_M3KO_RNAseq_logcounts_cell_number_normalized.txt.gz','published_cell_normalised_log2')]:
            result['views'][scale]=numeric_table(paths[name],out/scale,scale,samples,'description')
    elif accession=='GSE124109':
        name='GSE124109_genes.processed.fpkm_table.txt.gz'
        result['views']['FPKM']=numeric_table(paths[name],out/'FPKM','FPKM',samples,'title')
        result['remaining'].append('published_all_sample_normalisation_scope')
    elif accession=='GSE104616':
        result['views']['log2_RMA']=numeric_table(paths['GSE104616_NHDF_1_matrix_data.txt.gz'],out/'log2_RMA','published_log2_RMA',samples,'NHDF_set1')
        result['role']='descriptive_comparator';result['remaining']+=['technical_replication','published_joint_RMA_ComBat','probe_to_gene_mapping']
    elif accession=='GSE303716':
        headers={}
        for name,p in paths.items():
            if name.endswith('.txt.gz'):
                header,rows=read_table(p)
                if not {'baseMean','log2FoldChange','padj'}.issubset(header):raise ValueError('SMAD2 attachment format changed; audit before use')
                headers[name]={'columns':header,'rows':len(rows),'type':'differential_statistics_not_sample_matrix'}
        write_json(out/'differential_tables.json',headers)
        result.update(processing_status='acquired_differential_only',role='knowledge_or_published_contrast_only',
            remaining=['obtain_sample_expression_matrix_or_reprocess_SRA','grouped_split_and_programme_mapping'])
    elif accession=='GSE3169':
        write_jsonl(out/'dauer_mtc_samples.jsonl',primary)
        result['views']['Dauer_MTC']={'arrays':len(primary),'starting_pool_series':len({r['series_group'] for r in primary}),
            'platforms':sorted({r['platform'] for r in primary}),'scale':'log2_common_reference_ratio'}
        result['remaining']+=['platform_probe_to_gene_mapping','time_protocol_mapping']
    else:raise ValueError('No parser for this accession')
    write_json(out/'status.json',result);return result
