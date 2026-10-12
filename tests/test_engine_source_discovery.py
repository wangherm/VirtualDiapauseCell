"""A data-free release must reuse real cache locations from acquisition/parent records."""
import importlib.util
from pathlib import Path
import pytest
from vdc.io import write_json,sha256

spec=importlib.util.spec_from_file_location('engine_source_smoke',Path(__file__).resolve().parents[1]/'scripts/smoke_research_engine.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def assets(tmp_path):
    repo=tmp_path/'release';private=tmp_path/'private';cache=tmp_path/'older_cache';cache.mkdir()
    entries=[]
    for name in ('GSE288723_raw_counts.csv.gz','GSE288723_family.soft.gz'):
        path=cache/name;path.write_bytes(('unit fixture '+name).encode())
        entries.append({'study':'GSE288723','file':name,'bytes':path.stat().st_size,'sha256':sha256(path)})
    write_json(repo/'configs/research_public_assets.json',{'files':entries})
    return repo,private,cache,entries


def test_release_resolves_acquisition_pointer_without_copying_data(tmp_path):
    repo,private,cache,entries=assets(tmp_path)
    receipt=private/'runs/download/status.json'
    write_json(receipt,{'files':{'GSE288723/'+e['file']:{'status':'verified','path':str(cache/e['file'])} for e in entries}})
    (private/'LATEST_RESEARCH_DATA.txt').write_text(str(receipt.parent)+'\n')
    found=module.locate_probe_sources(repo,private)
    assert len(found)==2 and all(Path(a['path']).parent==cache for _,a in found.values())
    assert not (repo/'data').exists() and not (private/'public_assets').exists()


def test_parent_inventory_recovers_from_stale_receipt(tmp_path):
    repo,private,cache,entries=assets(tmp_path);parent=private/'runs/parent'
    write_json(parent/'inventory.json',{'assets':{str(i):{'source':'GSE288723','name':e['file'],
        'path':str(cache/e['file']),'public':True} for i,e in enumerate(entries)}})
    receipt=private/'stale.json';write_json(receipt,{'files':{'GSE288723/'+e['file']:{'path':'/missing/'+e['file']} for e in entries}})
    assert len(module.locate_probe_sources(repo,private,receipt,parent))==2


def test_named_cache_is_not_accepted_without_matching_checksum(tmp_path):
    repo,private,cache,entries=assets(tmp_path);parent=private/'runs/parent'
    write_json(parent/'inventory.json',{'assets':{str(i):{'source':'GSE288723','name':e['file'],
        'path':str(cache/e['file']),'public':True} for i,e in enumerate(entries)}})
    (cache/entries[0]['file']).write_bytes(b'changed')
    with pytest.raises(FileNotFoundError,match='checksum_mismatches'):
        module.locate_probe_sources(repo,private,parent=parent)


def test_missing_source_names_are_reported_not_stopiteration(tmp_path):
    repo,private,_,_=assets(tmp_path)
    with pytest.raises(FileNotFoundError,match='GSE288723_raw_counts.csv.gz'):
        module.locate_probe_sources(repo,private)
