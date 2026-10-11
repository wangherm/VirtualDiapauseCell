"""Transport/contract fixtures only; no live downloads or biological success claims."""
import hashlib
import io
from pathlib import Path
import pytest
from vdc.io import write_json, read_json, object_hash
from vdc.research_assets import validate_asset_manifest, load_public_context, asset_key
from scripts.acquire_research_data import acquire
from scripts import prepare_public_pilot as downloader


def entry(data=b'abcd', name='matrix.gz', study='GSE123'):
    return {'study':study,'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'url':'https://ftp.ncbi.nlm.nih.gov/geo/'+name,'resume':True}


@pytest.mark.parametrize('change', [
    {'file':'../escape'}, {'file':'x\\escape'}, {'file':'C:escape'},
    {'url':'https://drive.google.com/private'}, {'url':'https://ftp.ncbi.nlm.nih.gov/x?token=secret'},
    {'sha256':'bad'}, {'bytes':0},
])
def test_manifest_rejects_unsafe_or_unpinned_entries(change):
    with pytest.raises(ValueError):validate_asset_manifest({'files':[{**entry(),**change}]})


def test_duplicate_manifest_destination_rejected():
    with pytest.raises(ValueError,match='Duplicate'):validate_asset_manifest({'files':[entry(),entry()]})


def test_verified_cache_reused_without_network_and_same_size_corruption_rejected(tmp_path,monkeypatch):
    cache=tmp_path/'cache';cache.mkdir();(cache/'matrix.gz').write_bytes(b'abcd')
    monkeypatch.setattr(downloader.urllib.request,'urlopen',lambda *a,**k:pytest.fail('Network called'))
    manifest={'files':[entry()]}
    s=acquire(manifest,tmp_path/'raw',tmp_path/'run',[cache])
    assert s['status']=='completed' and s['files']['GSE123/matrix.gz']['action']=='reused_cache'
    assert not (tmp_path/'raw/GSE123/matrix.gz').exists()  # No duplicate copy.
    (cache/'matrix.gz').write_bytes(b'xxxx')
    s=acquire(manifest,tmp_path/'raw',tmp_path/'run',[cache])
    assert s['status']=='partial' and s['failed']==1  # Old completed state cannot mask mutation.


def test_independent_missing_file_does_not_cancel_verified_files(tmp_path):
    cache=tmp_path/'cache';cache.mkdir();(cache/'matrix.gz').write_bytes(b'abcd')
    manifest={'files':[entry(),entry(name='missing.gz')]}
    s=acquire(manifest,tmp_path/'raw',tmp_path/'run',[cache])
    assert s['verified']==1 and s['failed']==1 and s['training_admitted'] is False


def test_md5_range_resume_and_sha_receipt(tmp_path,monkeypatch):
    data=b'abcdef';e=entry(data)
    del e['sha256'];e.update(md5=hashlib.md5(data).hexdigest(),url='https://ftp.sra.ebi.ac.uk/vol1/test')
    (tmp_path/'matrix.gz.partial').write_bytes(data[:3])
    class Response(io.BytesIO):
        status=206
        headers={'Content-Range':'bytes 3-5/6'}
    def open_url(req,timeout):
        assert req.headers['Range']=='bytes=3-'
        return Response(data[3:])
    monkeypatch.setattr(downloader.urllib.request,'urlopen',open_url)
    assert downloader.fetch_locked(e,tmp_path,True).read_bytes()==data


def test_context_is_bound_to_receipt_and_rechecks_files_without_sending_paths(tmp_path):
    repo=tmp_path/'repo';cache=tmp_path/'cache';cache.mkdir();(cache/'matrix.gz').write_bytes(b'abcd')
    manifest={'files':[entry()]}
    write_json(repo/'configs/research_public_assets.json',manifest)
    write_json(repo/'knowledge/catalogue.json',{'studies':[{'accession':'GSE123','role':'archive_only'}]})
    config={'views':['bulk'],'public_data_catalogue':'knowledge/catalogue.json','require_public_assets':True}
    acquire(manifest,tmp_path/'raw',tmp_path/'run',[cache])
    report=tmp_path/'run/status.json'
    context=load_public_context(repo,config,report)
    assert context['acquisition']['GSE123']['verified_files']==1
    assert context['public_matrices_used_in_current_assay'] is False
    assert str(tmp_path) not in str(context)
    with pytest.raises(ValueError,match='VDC_RESEARCH_ASSET_REPORT'):load_public_context(repo,config)
    s=read_json(report);s['manifest_hash']='other';write_json(report,s)
    with pytest.raises(ValueError,match='manifest differs'):load_public_context(repo,config,report)
    s['manifest_hash']=object_hash(manifest);write_json(report,s)
    (cache/'matrix.gz').write_bytes(b'xxxx')
    with pytest.raises(ValueError,match='changed'):load_public_context(repo,config,report)


def test_committed_manifest_and_holdout_policy():
    repo=Path(__file__).resolve().parents[1]
    files=validate_asset_manifest(read_json(repo/'configs/research_public_assets.json'))
    assert len(files)==44 and len({e['study'] for e in files})==18
    catalogue=read_json(repo/'knowledge/research_public_data.json')
    held=next(s for s in catalogue['studies'] if s['accession']=='GSE185815')
    assert held['role']=='inherited_holdout_or_excluded_family'


def test_condensed_memory_keeps_all_failures_and_recent_details():
    from vdc.research import research_memory
    rows=[{'round':i,'execution':'failed' if i==0 else 'evaluated','relative_mse':2,
           'hypothesis':'question','metrics':{'bulk':{'mse':2}},'reason':'failure evidence',
           'reflection':{'conclusion':'inconclusive','next_question':'next'}} for i in range(7)]
    memory=research_memory(rows,detailed_last=2)
    assert [r['round'] for r in memory]==list(range(7))
    assert memory[0]['execution']=='failed' and memory[0]['reason']=='failure evidence'
    assert 'metrics' not in memory[0] and memory[-1]['metrics']==rows[-1]['metrics']
    assert rows[0]['metrics']  # No mutation of actual saved experiment history.
