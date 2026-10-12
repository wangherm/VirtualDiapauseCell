"""Public-data broker: official discovery, bounded acquisition and explicit provenance."""
import hashlib
import csv
import io
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from .io import read_json, write_json, object_hash, sha256
from .research_r2_tools import fetch_public, public_url, NoRedirect


def official_url(url, source):
    u=urllib.parse.urlsplit(url)
    if u.scheme!='https' or u.username or u.password or u.query or u.fragment or u.port not in (None,443):
        raise ValueError('Data downloads require credential-free official HTTPS file URLs without query strings')
    if u.hostname=='ftp.ncbi.nlm.nih.gov' and re.fullmatch(r'GSE\d+',source):
        if not u.path.startswith('/geo/series/') or '/'+source+'/' not in u.path:
            raise ValueError('GEO path does not match admitted accession')
    elif u.hostname in ('ftp.sra.ebi.ac.uk','ftp.ebi.ac.uk') and re.fullmatch(r'PRJ[EDN][AB]\d+',source):
        if not u.path.startswith(('/vol1/fastq/','/vol1/run/','/pub/databases/ena/')):
            raise ValueError('Not an ENA data file path')
    else:
        raise ValueError('This broker supports official GEO/ENA files; other repositories need a reviewed host adapter')
    return url


def discover(engine,a):
    source=a['source'];reason=a.get('research_reason','')
    if not isinstance(reason,str) or len(reason.strip())<20:
        raise ValueError('Describe the research question and why this missing/new dataset is relevant')
    if re.fullmatch(r'GSE\d+',source):
        bucket=source[:-3]+'nnn' if len(source[3:])>3 else 'GSEnnn'
        urls=[f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={source}&targ=self&form=text&view=brief',
              f'https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/{source}/suppl/']
    elif re.fullmatch(r'PRJ[EDN][AB]\d+',source):
        urls=['https://www.ebi.ac.uk/ena/portal/api/filereport?'+urllib.parse.urlencode({
            'accession':source,'result':'read_run','fields':'study_accession,sample_accession,run_accession,fastq_ftp,fastq_md5,fastq_bytes','format':'tsv'})]
    else: raise ValueError('Discover a GEO GSE or ENA PRJEB/PRJNA/PRJDB accession')
    docs=[fetch_public(url,engine.run/'literature') for url in urls]
    body=(engine.run/'literature'/(docs[0]['id']+'.txt')).read_text(encoding='utf-8')
    if (source.startswith('GSE') and not re.search(r'^\^SERIES\s*=\s*'+source+r'\s*$',body,re.M)) or (not source.startswith('GSE') and not any(r.get('study_accession')==source for r in csv.DictReader(io.StringIO(body),delimiter='\t'))):
        raise ValueError('Official metadata did not confirm this accession; source not admitted')
    engine.inv['sources'].setdefault(source,{'id':source,'title':source,'asset_ids':[],
        'disposition':'not_processed','public':True,'role':'available_for_adaptive_research'})
    record={'source':source,'research_reason':reason,'official_records':[d['id'] for d in docs],
        'status':'discovered_not_downloaded','license':'not_inferred','adaptive_only':True}
    engine.inv['sources'][source]['discovery']=record
    engine.save()
    return {**record,'documents':docs,'next_step':'Read official filenames/size/checksum; download exact file, then inspect and prepare. This is not training.'}


def probe(engine,a):
    source=a['source'];url=official_url(a['url'],source)
    if source not in engine.inv['sources']:raise ValueError('Discover the public accession first')
    public_url(url)
    req=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'VDC-Research-R2/2','Accept-Encoding':'identity'})
    with urllib.request.build_opener(NoRedirect).open(req,timeout=45) as response:
        size=response.headers.get('Content-Length')
        result={'source':source,'url':url,'file':Path(urllib.parse.urlsplit(url).path).name,
            'bytes':int(size) if size else None,'etag':response.headers.get('ETag'),
            'last_modified':response.headers.get('Last-Modified'),'checksum':'not_inferred_from_etag',
            'status':'official_headers_only_not_downloaded'}
    write_json(engine.run/'public_probes'/(object_hash(url)+'.json'),result)
    return result


def acquire(engine,a):
    source=a['source'];url=official_url(a['url'],source)
    if source not in engine.inv['sources']: raise ValueError('Use discover_data before downloading a new source')
    name=a['file']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]{0,240}',name): raise ValueError('Use a safe plain filename')
    if urllib.parse.unquote(Path(urllib.parse.urlsplit(url).path).name)!=name:
        raise ValueError('Filename must match the official file URL')
    # ENA projects and run filenames use different IDs: bind every new ENA URL to its official record.
    discovery=engine.inv['sources'][source].get('discovery')
    if not source.startswith('GSE'):
        if not discovery: raise ValueError('Use discover_data to retrieve this ENA project file report first')
        texts='\n'.join((engine.run/'literature'/(d+'.txt')).read_text(encoding='utf-8') for d in discovery['official_records'])
        matches=[]
        for row in csv.DictReader(io.StringIO(texts),delimiter='\t'):
            if row.get('study_accession')!=source:continue
            for remote,md5,nbytes in zip(row.get('fastq_ftp','').split(';'),row.get('fastq_md5','').split(';'),row.get('fastq_bytes','').split(';')):
                if url.removeprefix('https://')==remote:matches.append((md5,int(nbytes)))
        if len(matches)!=1:raise ValueError('File must resolve once in the official project file report')
        md5,nbytes=matches[0]
        if int(a.get('bytes',0))!=nbytes or a.get('md5',md5).lower()!=md5.lower():
            raise ValueError('Requested size/checksum differs from official project record')
        a={**a,'md5':md5}
    limit=int(engine.config.get('public_download_max_bytes',20_000_000_000))
    reserve=int(engine.config.get('public_download_reserve_bytes',10_000_000_000))
    size=int(a.get('bytes',0))
    if not 0<size<=limit: raise ValueError('Provide the official positive byte size within the per-file download limit')
    root=engine.private/'public_assets'/source;root.mkdir(parents=True,exist_ok=True)
    path=root/name;receipt=root/(name+'.vdc_source.json')
    expected={k:a[k] for k in ('sha256','md5') if a.get(k)}
    for alg,digest in expected.items():
        if not re.fullmatch('[a-fA-F0-9]{'+str(64 if alg=='sha256' else 32)+'}',digest): raise ValueError('Invalid checksum')
    identity={'url':url,'bytes':size,'checksums':expected}
    if path.exists():
        old=read_json(receipt) if receipt.exists() else None
        if old is None:
            from .research_assets import verified_file
            pinned=next((v.get('expected') for v in engine.inv['assets'].values()
                if Path(v['path']).resolve()==path.resolve() and v.get('expected')),None)
            if pinned and pinned.get('url',url)==url and path.stat().st_size==size and verified_file(path,pinned):
                for alg,value in expected.items():
                    with path.open('rb') as stream:
                        if hashlib.file_digest(stream,alg).hexdigest()!=value.lower():
                            raise ValueError('Requested checksum differs from existing verified source')
                old={'identity':identity,'sha256':sha256(path),'headers':{},
                    'verification':'reused_existing_pinned_source','previous_pin':pinned,'license':'not_inferred'}
                write_json(receipt,old)
        if not old or old['identity']!=identity or path.stat().st_size!=size or sha256(path)!=old['sha256']:
            raise ValueError('Existing file lacks matching provenance or changed; do not overwrite; review a new snapshot')
        record=old
    else:
        if shutil.disk_usage(root).free < size+reserve:
            raise ValueError('Insufficient disk space including configured reserve; only this acquisition is blocked')
        public_url(url)
        opener=urllib.request.build_opener(NoRedirect)
        # Official static file paths should not redirect; any redirect needs new explicit discovery.
        req=urllib.request.Request(url,headers={'User-Agent':'VDC-Research-R2/2','Accept-Encoding':'identity'})
        partial=root/(name+'.part');hashes={k:hashlib.new(k) for k in set(expected)|{'sha256'}}
        count=0
        with opener.open(req,timeout=180) as response,partial.open('wb') as stream:
            if response.headers.get('Content-Length') and int(response.headers['Content-Length'])!=size:
                raise ValueError('Official byte size changed; rediscover before retrying')
            while block:=response.read(1024*1024):
                count+=len(block)
                if count>size or shutil.disk_usage(root).free < reserve+len(block):
                    raise ValueError('Download exceeded declared size or disk reserve')
                stream.write(block)
                for h in hashes.values(): h.update(block)
            headers={k:response.headers.get(k) for k in ('ETag','Last-Modified','Content-Length')}
        if count!=size: raise ValueError('Incomplete download; partial retained, next attempt restarts this file')
        if any(hashes[k].hexdigest()!=v.lower() for k,v in expected.items()): raise ValueError('Provider checksum mismatch; partial retained')
        record={'identity':identity,'sha256':hashes['sha256'].hexdigest(),'headers':headers,
            'verification':'provider_checksum_and_size' if expected else 'observed_sha256_and_declared_size; provider_checksum_unavailable',
            'research_reason':a.get('research_reason',discovery.get('research_reason') if discovery else 'catalogue source'),
            'license':'not_inferred','restart_policy':'partial_file_restart; existing_verified_files_reused'}
        partial.replace(path);write_json(receipt,record)
    key='asset_'+object_hash(str(path.resolve()))[:16]
    engine.inv['assets'][key]={'path':str(path.resolve()),'name':name,'source':source,'bytes':size,
        'mtime_ns':path.stat().st_mtime_ns,'public':True,'sha256':record['sha256'],
        'expected':{'sha256':record['sha256'],'bytes':size},'download_provenance':record}
    ids=engine.inv['sources'][source].setdefault('asset_ids',[])
    if key not in ids:ids.append(key)
    engine.save()
    return {'status':'downloaded_not_prepared','asset':key,'source':source,'bytes':size,
        'sha256':record['sha256'],'verification':record['verification'],'trained':False}
