"""Synthetic engineering fixtures; live real-model HTTP verification is separate."""
import pytest
from fastapi.testclient import TestClient
from tests.fixtures import observation_fixture
from vdc.state import fit_state,StateConfig
from vdc.io import write_json,read_json,sha256
from vdc.pk2_service import freeze_service,create_app

@pytest.fixture
def source(tmp_path):
    b=observation_fixture().subset(list(range(30)));run=tmp_path/'run'
    folder=run/'state_bulk_K0_42'
    fit_state(b,folder,2,StateConfig(hidden_dim=16,latent_dim=8,layers=1,heads=2,validation_every=1))
    write_json(folder/'result.json',{'fixture_only':True})
    b.save(run/'clock_bulk/bundle')
    modules={}
    for name in ['state_bulk_K0_42','clock_bulk']:
        modules[name]={'execution_status':'completed','files':{p.relative_to(run/name).as_posix():sha256(p) for p in (run/name).rglob('*') if p.is_file()}}
    write_json(run/'module_status.json',{'modules':modules});write_json(run/'run_manifest.json',{'signature':'synthetic_fixture'})
    return run

def test_synthetic_model_not_deployed_as_research(source,tmp_path):
    with pytest.raises(ValueError,match='Synthetic'):freeze_service(source,None,tmp_path/'snapshot')

def test_queries_identity_reload_and_no_unknown_samples(source,tmp_path):
    target=tmp_path/'snapshot';m=freeze_service(source,None,target,allow_synthetic_fixture=True)
    with TestClient(create_app(target)) as client:
        assert client.get('/health').json()['snapshot_id']==m['snapshot_id']
        rows=client.get('/samples?view=bulk').json();assert len(rows)==30
        payload={'model':'state_bulk_K0_42','observation_ids':[rows[-1]['observation_id']]}
        response=client.post('/analyse',json=payload);assert response.status_code==200
        first=response.json();assert first['mode']=='synthetic_software_test'
        assert first['result']['depth']['status']=='unavailable'
        assert client.post('/analyse',json={**payload,'observation_ids':['synthetic-35']}).status_code==422
        assert client.post('/analyse',json={**payload,'observation_ids':[rows[0]['observation_id']]*2}).status_code==422
    with TestClient(create_app(target)) as client:assert client.post('/analyse',json=payload).json()==first
    with (target/'state_bulk_K0_42/best.pt').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed'):create_app(target)

def test_snapshot_will_not_overwrite(source,tmp_path):
    target=tmp_path/'snapshot';freeze_service(source,None,target,allow_synthetic_fixture=True)
    with pytest.raises(FileExistsError):freeze_service(source,None,target,allow_synthetic_fixture=True)

def test_unknown_implementation_refuses_loading(source,tmp_path,monkeypatch):
    from vdc import pk2_service
    target=tmp_path/'snapshot';freeze_service(source,None,target,allow_synthetic_fixture=True)
    monkeypatch.setattr(pk2_service,'implementation_identity',lambda:{'changed':True})
    with pytest.raises(ValueError,match='implementation changed'):create_app(target)
