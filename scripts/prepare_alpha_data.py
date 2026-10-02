"""Task-specific real data admission. Old test columns never enter numeric arrays."""
from pathlib import Path
import csv
import gzip
import re
import numpy as np
from prepare_public_pilot import fetch_locked, parse_metadata, programme_definitions, ROOT
from vdc.contracts import ObservationBundle
from vdc.observation import normalise_expression, score_programmes
from vdc.io import read_json, write_json, object_hash, save_npz, sha256


def save_bundle(expression, genes, rows, definitions, context, scale, out):
    if any(r['split'] not in {'train', 'validation'} for r in rows):
        raise ValueError('Alpha prepares development units only')
    y = normalise_expression(expression, np.ones_like(expression, bool), scale)
    scored = score_programmes(y, np.ones_like(y, bool), genes, definitions)
    train = np.array([r['split'] == 'train' for r in rows])
    keep = scored['mask'][train].all(0)
    accepted = [p for p, k in zip(definitions, keep) if k]
    context = {**context, 'programme_definition_id': object_hash(accepted),
        'preprocessing_id': object_hash({'scale': scale, 'genes': genes, 'method': scored['definition'], 'min_genes': 3, 'min_coverage': .5})}
    b = ObservationBundle(scored['values'][:, keep], scored['mask'][:, keep], scored['coverage'][:, keep],
        [p['id'] for p in accepted], rows, context, 'unit_holdout', np.zeros(len(rows)), np.zeros(len(rows), bool), None).validate()
    b.save(out); write_json(out/'programmes.json', accepted)
    save_npz(out/'gene_expression.npz', expression=y.astype(np.float32), genes=np.array(genes))
    write_json(out/'expression_contract.json', {'gene_ids': genes, 'input_scale': scale, 'context': context,
        'expression_sha256': sha256(out/'gene_expression.npz'), 'normalisation': 'log1p_library_10000' if scale == 'estimated_counts' else 'log1p',
        'old_test_used': False, 'internal_data_used': False})
    return b


def prepare_dauer(out, download):
    lock = read_json(ROOT/'configs/public_pilot_sources.json')
    paths = {k: fetch_locked(v, ROOT/'data/raw/GSE288723', download) for k, v in lock['files'].items()}
    metadata = parse_metadata(paths['metadata'])
    with gzip.open(paths['expression'], 'rt', newline='') as f:
        reader = csv.reader(f); header = next(reader)[1:]
        selected, rows, excluded = [], [], []
        for i, name in enumerate(header):
            m = re.fullmatch(r'D(1|4|15|30)\.(0|6|24)\.([12])', name)
            if not m:
                excluded.append({'library': name, 'reason': 'old_test_or_unadmitted_arm'})
                continue
            day, hour, rep = map(int, m.groups())
            rows.append({**metadata[name], 'split': 'train' if rep == 1 else 'validation',
                'link_ids': [f'GSE288723:conservative_replicate_block:{rep}'], 'replicate_block': rep,
                'history_days': day, 'elapsed_hours_since_release': hour, 'reference_eligible': True,
                'lineage_status': 'replicate_index_is_not_proven_longitudinal_matching'})
            selected.append(i+1)
        genes, matrix = [], []
        for line in reader:
            genes.append(line[0]); matrix.append([float(line[i]) for i in selected])
    if len(rows) != 24: raise ValueError('Expected 12 train and 12 validation dauer pools')
    anchors = {a for m in read_json(ROOT/'knowledge/modules.json')['modules'] for a in m['go_anchors']}
    definitions = programme_definitions(paths['annotations'], anchors)
    b = save_bundle(np.array(matrix).T, genes, rows, definitions,
        {'species': 'Caenorhabditis elegans', 'system': 'daf-2 dauer and exit', 'material': 'whole_worm_pool',
         'assay': 'bulk_RNA_seq_Salmon_estimated_counts'}, 'estimated_counts', out)
    tf = set()
    with gzip.open(paths['annotations'], 'rt', encoding='utf-8') as f:
        for line in f:
            if line.startswith('!'): continue
            c = line.rstrip().split('\t')
            if c[4] == 'GO:0003700' and 'NOT' not in c[3].split('|') and c[12] == 'taxon:6239' and c[6] != 'ND' and 'PMID:41104926' not in c[5]:
                tf.add(c[1])
    write_json(out/'tf_identity.json', {'genes': sorted(tf & set(genes)), 'source_sha256': sha256(paths['annotations']),
        'definition': 'direct GO:0003700 positive annotation; TF RNA abundance, NOT measured TF activity'})
    write_json(out/'admission.json', {'source_manifest': lock, 'excluded': excluded, 'prepared': len(rows),
        'train': 12, 'validation': 12, 'test': 0, 'limitations': ['Within-study conservative blocks; shared culture lineage incompletely deposited',
        'RNA is pooled, not individual tracking; release pairs are matched condition groups only', 'GO direct annotations overlap; programme scores are proxies'],
        'bundle_fingerprint': b.fingerprint})
    return b


def prepare_ard(out, download):
    lock = read_json(ROOT/'configs/alpha_sources.json')['GSE291659']
    paths = {k: fetch_locked(v, ROOT/'data/raw/GSE291659', download) for k, v in lock.items()}
    text = gzip.open(paths['metadata'], 'rt', encoding='utf-8').read()
    metadata = {}
    for chunk in text.split('^SAMPLE = ')[1:]:
        lines = chunk.splitlines(); gsm = lines[0]
        title = next(l.split(' = ', 1)[1] for l in lines if l.startswith('!Sample_title = '))
        genotype = next(l.split('genotype: ', 1)[1] for l in lines if l.startswith('!Sample_characteristics_ch1 = genotype:'))
        rep = int(re.search(r'replicat (\d+)', title).group(1))
        condition = 'REC' if 'refeeding' in title else '48'
        g = 'daf1_hlh30' if 'hlh-30' in genotype and 'daf-1' in genotype else 'hlh30' if 'hlh-30' in genotype else 'daf1' if 'daf-1' in genotype else 'N2'
        name = f'{g}_{condition}_REP{rep}'
        metadata[name] = {'observation_id': gsm, 'library': name, 'genotype': genotype, 'genotype_code': g,
            'condition': condition, 'replicate_block': rep, 'biological_unit': gsm, 'study_family': 'GSE291659',
            'origin': 'public', 'split': 'train' if rep <= 2 else 'validation' if rep == 3 else 'test',
            'link_ids': [f'GSE291659:conservative_replicate_block:{rep}'], 'reference_eligible': True,
            'persistent_genotype_is_context': True, 'new_acute_ko': False}
    with gzip.open(paths['expression'], 'rt', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t'); header = next(reader)
        names = header[2:]
        if len(names) != 30 or not set(names) <= set(metadata): raise ValueError('Audited 30-column matrix changed')
        selected = [i+2 for i, n in enumerate(names) if metadata[n]['split'] != 'test']
        rows = [metadata[n] for n in names if metadata[n]['split'] != 'test']
        genes, matrix = [], []
        for line in reader:
            genes.append(line[0]); matrix.append([float(line[i]) for i in selected])
    if len(genes) != len(set(genes)) or any(not re.fullmatch(r'WBGene\d+', g) for g in genes): raise ValueError('Invalid gene identifiers')
    # Same versioned gene-set catalogue, distinct context/preprocessing and separate model.
    definitions = read_json(out.parent/'dauer/programmes.json')
    b = save_bundle(np.array(matrix).T, genes, rows, definitions,
        {'species': 'Caenorhabditis elegans', 'system': 'adult_reproductive_diapause_N2_hlh30_daf1',
         'material': 'whole_worm_pool', 'assay': 'bulk_RNA_seq_deposited_DESeq2_normalised_counts'}, 'linear_abundance', out)
    write_json(out/'admission.json', {'source_manifest': lock, 'metadata_samples': len(metadata), 'matrix_samples': len(names),
        'missing_expression_libraries': sorted(set(metadata)-set(names)), 'prepared': len(rows), 'test_columns_not_parsed': 7,
        'label_source': 'experimental_metadata', 'genotype_mapping': 'explicit title/genotype to N2,hlh30,daf1,daf1_hlh30',
        'limitations': ['Depositor labels values DESeq2 normalised counts; not raw counts',
        'Constitutive genotype contrast, not acute KO; shared N2 controls remain within split',
        'Replicate index conservative grouping, not proven independent batch or longitudinal matching'],
        'bundle_fingerprint': b.fingerprint})
    return b
