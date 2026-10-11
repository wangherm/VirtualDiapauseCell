"""Public asset locks and the bounded data context exposed to the research LLM."""
from pathlib import Path, PurePosixPath
import hashlib
import re
from urllib.parse import urlsplit
from .io import read_json, object_hash


def validate_asset_manifest(manifest):
    files = manifest['files']
    seen = set()
    if not files:
        raise ValueError('Empty public asset manifest')
    for e in files:
        if not re.fullmatch(r'(GSE|PRJEB)\d+', e['study']):
            raise ValueError('Invalid public study identifier')
        name = e['file']
        if PurePosixPath(name).name != name or name in {'.', '..'} or '\\' in name or ':' in name:
            raise ValueError('Asset filename must be a basename')
        key = (e['study'], name)
        if key in seen:
            raise ValueError('Duplicate asset destination')
        seen.add(key)
        u = urlsplit(e['url'])
        if (u.scheme != 'https' or u.hostname not in {'ftp.ncbi.nlm.nih.gov', 'ftp.sra.ebi.ac.uk'}
                or u.username or u.password or u.query or u.fragment or u.port):
            raise ValueError('Only pinned official public download URLs are accepted')
        if type(e['bytes']) is not int or e['bytes'] <= 0:
            raise ValueError('Expected positive byte count')
        algorithms = [a for a in ('sha256', 'md5') if a in e]
        if len(algorithms) != 1 or not re.fullmatch(
                r'[0-9a-f]{64}' if algorithms[0] == 'sha256' else r'[0-9a-f]{32}', e[algorithms[0]]):
            raise ValueError('Expected one pinned SHA256 or ENA MD5')
        if algorithms[0] == 'md5' and u.hostname != 'ftp.sra.ebi.ac.uk':
            raise ValueError('MD5 is only accepted for ENA public reads')
    return files


def asset_key(entry):
    return entry['study'] + '/' + entry['file']


def file_digests(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block); md5.update(block)
    return {'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def verified_file(path, entry):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return None
    actual = file_digests(path)
    algorithm = 'sha256' if 'sha256' in entry else 'md5'
    return actual if actual[algorithm] == entry[algorithm] else None


def cache_index(roots, entries):
    """Index only matching basename/size candidates, without following directories' symlinks."""
    import os
    wanted = {(e['file'], e['bytes']) for e in entries}
    result = {}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        for parent, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {'.git', '.venv', '.venv-autodl-vdc', '__pycache__'})
            for name in sorted(names):
                if name not in {k[0] for k in wanted}:
                    continue
                path = Path(parent)/name
                if path.is_symlink():
                    continue
                key = (name, path.stat().st_size)
                if key in wanted:
                    paths = result.setdefault(key, [])
                    if path not in paths:
                        paths.append(path)
    return result


def load_public_context(repo, config, report_path=None):
    """Only public study guidance and aggregate acquisition facts enter the API prompt.

    A completed receipt is rechecked against actual files, not trusted as a flag.
    Nothing here grants sample admission or adds matrices to the numerical assay.
    """
    relative = config.get('public_data_catalogue')
    if not relative:
        return None
    catalogue = read_json(Path(repo)/relative)
    manifest = read_json(Path(repo)/'configs/research_public_assets.json')
    entries = validate_asset_manifest(manifest)
    status = None
    if report_path:
        status = read_json(report_path)
        if status.get('manifest_hash') != object_hash(manifest):
            raise ValueError('Public acquisition manifest differs; audit the selected snapshot')
        if set(status.get('files', {})) != {asset_key(e) for e in entries}:
            raise ValueError('Public acquisition receipt does not cover the full requested file list')
    elif config.get('require_public_assets'):
        raise ValueError('Set VDC_RESEARCH_ASSET_REPORT to the completed public acquisition status.json')
    by_study = {}
    for e in entries:
        record = status['files'][asset_key(e)] if status else {}
        good = record.get('status') == 'verified'
        if good:
            actual = verified_file(record['path'], e)
            if not actual or actual['sha256'] != record.get('observed_sha256'):
                raise ValueError('Acquired public file changed: ' + asset_key(e))
        if config.get('require_public_assets') and not good:
            raise ValueError('Public acquisition is incomplete: ' + asset_key(e))
        summary = by_study.setdefault(e['study'], {'requested_files': 0, 'verified_files': 0})
        summary['requested_files'] += 1
        summary['verified_files'] += int(good)
    # Keep full source notes on disk; send a fixed concise field set, never local paths.
    compact = {'global_rules': catalogue.get('global_rules', []),
               'studies': [{k: row[k] for k in ('accession', 'system', 'role', 'ai_brief') if k in row}
                           for row in catalogue['studies']]}
    return {'manifest_hash': object_hash(manifest), 'catalogue_hash': object_hash(catalogue), 'catalogue': compact,
            'acquisition': by_study, 'public_matrices_used_in_current_assay': False,
            'current_executable_views': config['views'],
            'rule': 'Public assets inform research planning. Only the supplied development views are executable now. '
                    'Never report an experiment on these public matrices as executed. Raw reads need quantification; '
                    'new processed studies need sample/scale/group mapping and versioned admission. '
                    'No change to inherited holdouts or excluded knowledge families.'}
