import importlib.util
from pathlib import Path
import types
import sys
import pytest
from vdc.io import write_json


def module():
    p=Path(__file__).resolve().parents[1]/'scripts/download_pk1_private.py'
    spec=importlib.util.spec_from_file_location('pk1download',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_download_failure_does_not_echo_private_identifiers(tmp_path,monkeypatch,capsys):
    m=module();token='PRIVATE_FIXTURE_IDENTIFIER'
    def fail(**kw):
        print(token);raise RuntimeError(token)
    monkeypatch.setitem(sys.modules,'gdown',types.SimpleNamespace(download_folder=fail))
    p=tmp_path/'config.local.json';write_json(p,{'folder_url':token,'files':[]})
    with pytest.raises(RuntimeError) as exc:m.download(p,tmp_path/'out')
    assert token not in str(exc.value)
    assert token not in capsys.readouterr().out


def test_download_refuses_ambiguous_names_and_path_traversal():
    m=module();r=types.SimpleNamespace(path='archive.zip',id='fixture')
    c={'files':[{'name':'archive.zip','sha256':'0'*64,'bytes':10}]}
    with pytest.raises(ValueError,match='ambiguous'):m.resolve_files(c,[r,r])
    c['files'][0]['name']='../archive.zip'
    with pytest.raises(ValueError,match='basename'):m.resolve_files(c,[r])
