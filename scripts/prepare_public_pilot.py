"""One audited public cohort; no catalogue-wide downloads and no clock fabrication."""
from __future__ import annotations
import argparse
import csv
import gzip
import io
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
from vdc.contracts import ObservationBundle
from vdc.io import read_json, write_json, object_hash, sha256
from vdc.observation import normalise_expression, score_programmes

ROOT = Path(__file__).resolve().parents[1]


def fetch_locked(entry, directory, download):
    path = directory / entry['file']
    if not path.exists():
        if not download:
            raise FileNotFoundError(f'{path}: use --download or copy the verified file')
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(entry['url'], headers={
            'User-Agent': 'VirtualDiapauseCell/0.4', 'Accept': 'application/json'})
        temporary = path.with_suffix(path.suffix + '.partial')
        with urllib.request.urlopen(req, timeout=60) as response, temporary.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if sha256(temporary) != entry['sha256']:
            raise ValueError(f'Source changed: {entry["file"]}; review a new snapshot explicitly')
        temporary.replace(path)
    if sha256(path) != entry['sha256']:
        raise ValueError(f'Checksum mismatch: {path}')
    return path


def parse_metadata(path):
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        text = f.read()
    rows = {}
    for block in text.split('^SAMPLE = ')[1:]:
        gsm = block.splitlines()[0].strip()
        def one(field):
            found = re.findall(r'^!' + field + r' = (.+)$', block, re.M)
            if len(found) != 1:
                raise ValueError(f'{gsm}: expected one {field}')
            return found[0].strip()
        lib = one('Sample_description').removeprefix('Library name: ')
        genotype = re.findall(r'^!Sample_characteristics_ch1 = genotype: (.+)$', block, re.M)
        if len(genotype) != 1 or lib in rows:
            raise ValueError('Missing genotype or duplicate library')
        if 'three biological replicates of 8000 worms' not in block:
            raise ValueError('Expected pooled biological replication protocol changed')
        rows[lib] = {'observation_id': gsm, 'library': lib,
                     'title': one('Sample_title'), 'genotype': genotype[0].strip(),
                     'biological_unit': gsm, 'unit_kind': 'pool_of_8000_worms',
                     'study_family': 'PMID41104926_GSE288723', 'origin': 'public'}
    return rows


def programme_definitions(gaf, anchors):
    members = defaultdict(set)
    evidence = defaultdict(set)
    with gzip.open(gaf, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('!'):
                continue
            c = line.rstrip('\n').split('\t')
            if len(c) != 17:
                raise ValueError('Expected GAF 2.2 17-column record')
            if (c[0] != 'WB' or c[12] != 'taxon:6239' or 'NOT' in c[3].split('|')
                    or c[6] == 'ND' or 'PMID:41104926' in c[5].split('|')):
                continue
            if c[4] in anchors:
                members[c[4]].add(c[1]); evidence[c[4]].add(c[6])
    return [{'id': term, 'members': {gene: 1.0 for gene in sorted(members[term])},
             'source_ref': 'GO_WormBase_snapshot_sha256:' + sha256(gaf),
             'membership_policy': 'direct_positive_GAF_only_no_descendant_propagation',
             'evidence_codes': sorted(evidence[term])}
            for term in sorted(anchors) if len(members[term]) >= 3]


def prepare(raw, output, download=False):
    if output.exists():
        raise FileExistsError('Choose a new prepared-data output; do not overwrite splits')
    lock = read_json(ROOT / 'configs/public_pilot_sources.json')
    paths = {k: fetch_locked(v, raw, download) for k, v in lock['files'].items()}
    metadata = parse_metadata(paths['metadata'])
    with gzip.open(paths['expression'], 'rt', encoding='utf-8', newline='') as f:
        reader = csv.reader(f); header = next(reader)[1:]; data = list(reader)
    if len(header) != len(set(header)) or set(header) != set(metadata):
        raise ValueError('Expression columns and GEO libraries differ')
    genes = [r[0] for r in data]
    if any(not re.fullmatch(r'WBGene\d+', g) for g in genes) or len(set(genes)) != len(genes):
        raise ValueError('Expected unique WormBase stable gene IDs')
    expression = np.array([r[1:] for r in data], dtype=float).T
    if not np.isfinite(expression).all() or (expression < 0).any():
        raise ValueError('Invalid estimated counts')
    selected, rows, excluded = [], [], []
    for i, name in enumerate(header):
        match = re.fullmatch(r'D(1|4|15|30)\.(0|6|24)\.([123])', name)
        if not match:
            excluded.append({**metadata[name], 'reason': 'UV perturbation or L3 reference reserved for separate admission'})
            continue
        days, hours, replicate = map(int, match.groups())
        # Conservative blocks join the same replicate index across all histories/timepoints.
        # They are NOT asserted to be proven longitudinal pairs or independent study batches.
        rows.append({**metadata[name], 'split': {1: 'train', 2: 'validation', 3: 'test'}[replicate],
                     'link_ids': [f'GSE288723:conservative_replicate_block:{replicate}'],
                     'history_days': days, 'elapsed_hours_since_release': hours,
                     'time_origin': 'dauer_exit_induction', 'reference_eligible': True,
                     'replicate_block': replicate, 'lineage_status': 'cross_condition_lineage_not_deposited',
                     'persistent_genotype_is_context': True, 'new_acute_ko': False})
        selected.append(i)
    if len(rows) != 36:
        raise ValueError('Expected 36 untreated dauer/exit libraries')
    x = expression[selected]; measured = np.ones_like(x, dtype=bool)
    y = normalise_expression(x, measured, 'estimated_counts')
    modules = read_json(ROOT / 'knowledge/modules.json')['modules']
    anchors = {term for m in modules for term in m['go_anchors']}
    definitions = programme_definitions(paths['annotations'], anchors)
    scored = score_programmes(y, measured, genes, definitions, min_genes=3, min_coverage=0.5)
    # No expression-dependent feature selection using validation/test.
    train = np.array([r['split'] == 'train' for r in rows])
    keep = scored['mask'][train].all(0)
    accepted = [p for p, use in zip(definitions, keep) if use]
    if len(accepted) < 2:
        raise ValueError('Insufficient mapped programmes; no substitute features fabricated')
    context = {'species': 'Caenorhabditis elegans', 'system': 'daf-2 dauer and exit',
               'material': 'whole_worm_pool', 'assay': 'bulk_RNA_seq_Salmon_estimated_counts',
               'programme_definition_id': object_hash(accepted),
               'preprocessing_id': object_hash({'method': scored['definition'], 'scale': 'estimated_counts',
                                                'genes': genes, 'min_genes': 3, 'min_coverage': 0.5})}
    b = ObservationBundle(scored['values'][:, keep], scored['mask'][:, keep], scored['coverage'][:, keep],
                          [p['id'] for p in accepted], rows, context, 'unit_holdout',
                          np.zeros(len(rows)), np.zeros(len(rows), bool), None).validate()
    b.save(output)
    write_json(output / 'programmes.json', accepted)
    write_json(output / 'excluded_samples.json', excluded)
    report = {'accession': 'GSE288723', 'source_file_manifest': lock,
              'downloaded_expression_samples': len(header), 'genes': len(genes),
              'prepared_samples': len(rows), 'programmes': len(accepted),
              'unavailable_anchors': sorted(anchors - set(b.feature_ids)),
              'splits': {s: len(b.indices(s)) for s in ('train', 'validation', 'test')},
              'split_policy': 'unit_holdout_with_conservative_replicate_blocks',
              'fractional_counts_preserved': True, 'clock_labels': 'unavailable',
              'internal_data_used': False, 'prior_holdouts_reassigned': False,
              'biological_generalisation_validated': False,
              'limitations': ['One study; shared culture/batch lineage across conditions is not fully deposited',
                             'Replicate blocks are conservative grouping, not proven longitudinal matching',
                             'GO direct annotations are incomplete overlapping proxies, not measured pathway activity',
                             'Elapsed hours are metadata, never molecular-clock supervision',
                             'Global per-cell RNA abundance is not recoverable from these relative measurements'],
              'bundle_fingerprint': b.fingerprint}
    write_json(output / 'admission.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=ROOT / 'data/raw/GSE288723')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    import json
    print(json.dumps(prepare(args.raw, args.out, args.download), indent=2))
