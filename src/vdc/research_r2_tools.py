"""R2 public research tools and read-only local asset inventory.

Tools return public evidence or structural summaries, never private matrix values.
This is a VDC implementation inspired by the cited papers, not their reproduction.
"""
from pathlib import Path, PurePosixPath
import hashlib
import ipaddress
import json
import os
import re
import socket
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from .io import read_json, write_json, sha256, object_hash
from .research_assets import verified_file, asset_key


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', value):
        raise ValueError('Expected a short alphanumeric identifier')
    return value


def inside(root, relative):
    root = Path(root).resolve()
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or '..' in rel.parts or '\\' in relative or ':' in relative:
        raise ValueError('Invalid relative path')
    p = root.joinpath(*rel.parts)
    if not p.resolve().is_relative_to(root) or p.is_symlink():
        raise ValueError('Path escapes artifact root')
    return p


def tree_hash(root):
    root = Path(root)
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlinks are not research artifacts')
        if p.is_file():
            files[p.relative_to(root).as_posix()] = sha256(p)
    return files


def write_code(root, files):
    if not isinstance(files, dict) or not files or len(files) > 100:
        raise ValueError('Candidate requires 1..100 source files')
    if sum(len(v.encode()) for v in files.values() if isinstance(v, str)) > 2_000_000:
        raise ValueError('Candidate source exceeds 2 MB')
    for name, value in files.items():
        if not isinstance(value, str):
            raise ValueError('Source files must contain text')
        p = inside(root, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value, encoding='utf-8', newline='\n')
    return tree_hash(root)


def inventory(repo, private, receipt=None, extra_roots=()):
    """Do not enforce R1 holdouts here: user authorized all-role R2 research.

    Original role files stay unmodified. Files missing on this server stay missing.
    Historical run trees are not indiscriminately mounted (they include API logs).
    """
    repo, private = Path(repo), Path(private)
    catalogue = read_json(repo/'knowledge/research_public_data.json')
    studies = catalogue['studies']
    sources = {s['accession']: {**s, 'historical_role': s.get('role'),
        'role': 'available_for_adaptive_research', 'disposition': 'not_processed'} for s in studies}
    sources['killifish_private'] = {'role': 'available_for_adaptive_research',
        'disposition': 'not_processed', 'historical_roles_preserved': True,
        'scope': 'All uploaded authorised sources; actual files determine availability'}
    assets = {}
    def add(path, source, public=False, expected=None):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            return
        key = 'asset_' + object_hash(str(path.resolve()))[:16]
        stat = path.stat()
        assets[key] = {'path': str(path.resolve()), 'source': source,
            'name': path.name, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'public': public, 'expected': expected}
    entries = read_json(repo/'configs/research_public_assets.json')['files']
    records = read_json(receipt).get('files', {}) if receipt and Path(receipt).is_file() else {}
    for e in entries:
        r = records.get(asset_key(e), {})
        candidates = [Path(r['path'])] if r.get('path') else []
        candidates += [private/'public_assets'/e['study']/e['file'], repo/'data/raw'/e['study']/e['file']]
        found = next((p for p in candidates if p.is_file()), None)
        if found:
            add(found, e['study'], True, e)
    # Include original archives, extracted files and derived caches, not only R1 views.
    roots = [private/n for n in ('input', 'archive', 'raw', 'sources', 'data', 'prepared', 'archives', 'original', 'annotations')]
    roots += [Path(p) for p in extra_roots]
    for pointer in ('LATEST_STAGE.txt', 'LATEST_PK1.txt', 'LATEST_APPLICATION.txt'):
        p = private/pointer
        if p.is_file():
            parent = Path(p.read_text().strip())
            if parent.is_relative_to(private.resolve()):
                roots += [parent/'data']
    permitted = ('.zip', '.tar', '.gz', '.bz2', '.h5ad', '.h5', '.hdf5', '.npz', '.npy',
                 '.csv', '.tsv', '.txt', '.mtx', '.json', '.jsonl', '.xlsx', '.xls', '.parquet')
    for p in private.glob('*'):
        if p.is_file() and (p.name == 'sample_roles.json' or p.suffix in ('.zip', '.h5ad', '.npz')):
            add(p, 'killifish_private')
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for parent, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {'.git', 'api', 'release', '__pycache__', '.venv'}
                             and not (Path(parent)/d).is_symlink())
            for name in sorted(names):
                p = Path(parent)/name
                if p.suffix.lower() in permitted and not any(t in name.lower() for t in ('secret', 'credential', '.env', 'api_key')):
                    add(p, 'killifish_private')
    for sid, source in sources.items():
        source['asset_ids'] = [k for k, a in assets.items() if a['source'] == sid]
        source['availability'] = 'files_present_not_processed' if source['asset_ids'] else 'missing_on_server'
    return {'schema': 'VDC_R2_ALL_DATA_v1', 'sources': sources, 'assets': assets,
            'all_original_roles_open': True, 'evaluation_scope': 'adaptive_development_or_retrospective',
            'original_roles_modified': False}


def asset_context(inv):
    return {'sources': [{k: s[k] for k in ('accession', 'system', 'ai_brief', 'unit_rule', 'current_access',
            'scale_or_assay', 'availability', 'disposition', 'asset_ids') if k in s} | {'id': sid}
        for sid, s in inv['sources'].items()],
        'assets': {k: {f: a[f] for f in ('source', 'name', 'bytes', 'public')}
                   for k, a in inv['assets'].items()}}


def resolve_assets(inv, ids):
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError('Select explicit distinct asset IDs')
    result = {}
    provenance = {}
    for key in ids:
        a = inv['assets'][key]; p = Path(a['path'])
        if p.is_symlink() or not p.is_file():
            raise ValueError('Asset unavailable')
        if a.get('expected') and not verified_file(p, a['expected']):
            raise ValueError('Public asset differs from pinned checksum: ' + key)
        digest = sha256(p)
        if a.get('sha256') and digest != a['sha256']:
            raise ValueError('Previously used asset changed: ' + key)
        a['sha256'] = digest
        # Keep a useful extension but never a host path in the container manifest.
        suffix = ''.join(p.suffixes[-2:])
        result[key + suffix] = p
        provenance[key] = {'source': a['source'], 'sha256': digest, 'bytes': p.stat().st_size,
                           'container_path': '/input/' + key + suffix}
    return result, provenance


def inspect_assets(inv, ids, offset=0):
    """Only structural metadata is sent to the external research model."""
    import zipfile
    result = {}; offset=max(0,int(offset))
    for key in ids:
        a = inv['assets'][key]; p = Path(a['path'])
        detail = {k: a[k] for k in ('name', 'bytes', 'source', 'public')}
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                detail['archive_members'] = [{'name': i.filename, 'bytes': i.file_size} for i in z.infolist()[offset:offset+100]]
                detail['member_count'] = len(z.infolist())
                if p.suffix=='.npz':
                    import numpy.lib.format as fmt
                    detail['arrays']=[]
                    for entry in z.infolist()[offset:offset+100]:
                        if not entry.filename.endswith('.npy'):continue
                        with z.open(entry) as stream:
                            version=fmt.read_magic(stream)
                            if version==(1,0):shape,order,dtype=fmt.read_array_header_1_0(stream)
                            elif version==(2,0):shape,order,dtype=fmt.read_array_header_2_0(stream)
                            else:continue
                        detail['arrays'].append({'name':entry.filename,'shape':shape,'dtype':str(dtype)})
        elif p.name.endswith(('.tar', '.tar.gz', '.tgz')):
            import tarfile
            with tarfile.open(p) as tar:
                detail['archive_members'] = []
                for i, member in enumerate(tar):
                    if i < offset:continue
                    if i >= offset+100: break
                    detail['archive_members'].append({'name':member.name, 'bytes':member.size})
        elif p.suffix in ('.h5ad','.h5','.hdf5'):
            import h5py
            with h5py.File(p,'r') as f:
                structures=[]
                def visit(name,obj):
                    if isinstance(obj,h5py.Dataset):structures.append({'name':name,'shape':obj.shape,'dtype':str(obj.dtype)})
                f.visititems(visit)
                detail['datasets']=structures[offset:offset+100]
                detail['dataset_count']=len(structures)
        elif not a['public'] and p.suffix in ('.csv','.tsv','.txt'):
            import csv
            with p.open(encoding='utf-8',errors='replace') as stream:
                header=stream.readline(16000)
            delimiter='\t' if '\t' in header else ','
            detail['column_names']=next(csv.reader([header],delimiter=delimiter))[:200]
            detail['data_values_returned']=False
        elif not a['public'] and p.suffix=='.json' and p.stat().st_size<2_000_000:
            obj=read_json(p)
            detail['json_type']=type(obj).__name__
            if isinstance(obj,dict):detail['fields']={k:type(v).__name__ for k,v in list(obj.items())[:100]}
            elif isinstance(obj,list):
                detail['length']=len(obj)
                if obj and isinstance(obj[0],dict):detail['row_fields']=list(obj[0])[:100]
        elif a['public'] and (p.suffix in ('.txt','.tsv','.csv','.json') or p.name.endswith(('.soft.gz','.csv.gz','.tsv.gz'))):
            import gzip
            op = gzip.open if p.suffix=='.gz' else open
            with op(p, 'rt', encoding='utf-8', errors='replace') as f:
                detail['public_header_excerpt'] = f.read(5000)
        result[key] = detail
    return result


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.hidden += 1
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.hidden = max(0, self.hidden-1)
    def handle_data(self, text):
        if not self.hidden: self.parts.append(text)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def public_url(url):
    u = urllib.parse.urlsplit(url)
    if u.scheme != 'https' or not u.hostname or u.username or u.password or u.port not in (None, 443):
        raise ValueError('Only public credential-free HTTPS URLs')
    if u.fragment or len(url) > 3000:
        raise ValueError('Invalid public URL')
    addresses = socket.getaddrinfo(u.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('Non-public network destination')
    return url


def fetch_public(url, out, max_bytes=16_000_000):
    """Credential-free broker; redirects validated, binaries never executed here."""
    import urllib.error
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    original = url
    opener = urllib.request.build_opener(NoRedirect)
    for _ in range(6):
        public_url(url)
        req = urllib.request.Request(url, headers={'User-Agent': 'VDC-Research-R2/1.0', 'Accept-Encoding': 'identity'})
        try:
            with opener.open(req, timeout=45) as response:
                body = response.read(max_bytes+1)
                content_type = response.headers.get('Content-Type', '')
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308): raise
            url = urllib.parse.urljoin(url, exc.headers['Location'])
    else: raise ValueError('Too many public redirects')
    if len(body) > max_bytes: raise ValueError('Public document exceeds bounded fetch limit; request a specific file')
    digest = hashlib.sha256(body).hexdigest()
    path = out/(digest+'.source'); path.write_bytes(body)
    if body.startswith(b'%PDF'):
        import io
        from pypdf import PdfReader
        text = '\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(body)).pages)
    else:
        text = body.decode('utf-8', errors='replace')
        if 'html' in content_type:
            parser = PlainText(); parser.feed(text); text = '\n'.join(parser.parts)
    (out/(digest+'.txt')).write_text(text, encoding='utf-8')
    record = {'id': digest, 'url': original, 'resolved_url': url, 'sha256': digest,
              'bytes': len(body), 'content_type': content_type, 'text_characters': len(text),
              'license': 'not_inferred; inspect source before redistribution'}
    write_json(out/(digest+'.json'), record)
    record.update(literature_kind=literature_kind(original,content_type),
                  related_resources=literature_links(original,body.decode('utf-8',errors='ignore')))
    write_json(out/(digest+'.json'),record)
    return {**record, 'excerpt': text[:10000], 'untrusted_source_material': True}


def literature_kind(url,content_type=''):
    u=urllib.parse.urlsplit(url)
    if u.hostname in ('arxiv.org','www.arxiv.org') and u.path.startswith('/abs/'):
        return 'abstract_page'
    if 'pdf' in content_type or (u.hostname in ('arxiv.org','www.arxiv.org') and u.path.startswith(('/pdf/','/html/'))):
        return 'full_text'
    if u.hostname=='raw.githubusercontent.com' or u.path.endswith(('.py','.r','.R','.jl','.ipynb')):
        return 'source_code_author_association_unverified'
    return 'web_page'


def literature_links(url,text):
    links=[]
    if 'arxiv.org/abs/' in url:
        paper=url.split('/abs/',1)[1].split('?',1)[0]
        links=[{'kind':'full_text_option','url':'https://arxiv.org/html/'+paper},
               {'kind':'full_text_option','url':'https://arxiv.org/pdf/'+paper}]
    for value in sorted(set(re.findall(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',text))):
        links.append({'kind':'linked_repository_not_yet_read','url':value})
    return links[:20]


def search_literature(query, out):
    if not isinstance(query, str) or not 1 <= len(query) <= 500:
        raise ValueError('Use a short public scientific query; no private identifiers')
    # Europe PMC supports public metadata and full-text links without new API keys.
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(
        {'query': query, 'format': 'json', 'pageSize': 8, 'resultType': 'core'})
    doc = fetch_public(url, out)
    data = json.loads((Path(out)/(doc['id']+'.txt')).read_text(encoding='utf-8'))
    hits = [{k: r[k] for k in ('id', 'source', 'pmcid', 'doi', 'title', 'authorString', 'pubYear',
                              'abstractText', 'fullTextUrlList') if k in r}
            for r in data.get('resultList', {}).get('result', [])]
    return {'query': query, 'receipt': doc['id'], 'hits': hits,
            'untrusted_source_material': True}
