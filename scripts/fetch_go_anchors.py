"""Fetch actual GO term metadata; never infer gene membership or training labels."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def fetch(output: Path):
    modules = json.loads((ROOT / 'knowledge/modules.json').read_text(encoding='utf-8'))['modules']
    ids = sorted({term for m in modules for term in m['go_anchors']})
    url = 'https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/' + ','.join(ids)
    req = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'VirtualDiapauseCell/0.1'})
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read()
    data = json.loads(raw)
    terms = data['results']
    if {x['id'] for x in terms} != set(ids):
        raise ValueError('QuickGO did not return every requested ID; no snapshot written')
    if any(x.get('isObsolete') for x in terms):
        raise ValueError('Obsolete anchor found; review dictionary before publishing')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new empty snapshot directory')
    (output / 'quickgo_response.json').write_bytes(raw)
    meta = {'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'url': url,
            'sha256': hashlib.sha256(raw).hexdigest(), 'term_count': len(terms),
            'ids': ids, 'kind': 'ontology_term_metadata_only',
            'gene_membership_downloaded': False,
            'terms': [{'id': t['id'], 'name': t['name'], 'isObsolete': t.get('isObsolete', False)} for t in terms]}
    (output / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'downloaded_verified', 'terms': len(terms), 'output': str(output)}, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    fetch(parser.parse_args().output_dir)
