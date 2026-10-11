import json
import numpy as np
import pytest
from unittest.mock import patch
from vdc.research_model import GraphModel, baseline_specs, validate_graph
from vdc.research_data import connected_rows, knowledge_records
from vdc.research_api import Client, parse_object, endpoint
from vdc.research import score, freeze_artifacts, verify_artifacts, validate_proposal


def data(n=8):
    rng=np.random.default_rng(14)
    x={k:rng.normal(size=(n,3)) for k in ('rank','amplitude','genes')}
    x.update(identity=np.eye(2)[np.arange(n)%2],clock=np.arange(n)[:,None]/n,clock_available=np.ones((n,1)))
    x.update(coverage=np.ones((n,3)),programme_available=np.ones((n,3)))
    rows=[{'study_family':'fixture','biological_unit':str(i//2)} for i in range(n)]
    return x, rng.normal(size=(n,4)), rows


def test_graph_serialization_and_query_batch_independence(tmp_path):
    x,y,rows=data()
    spec={'nodes':[{'id':'a','op':'concat','inputs':['genes','amplitude']},
                   {'id':'b','op':'standardize','inputs':['a']},
                   {'id':'c','op':'pca','inputs':['b'],'components':2},
                   {'id':'d','op':'interaction','inputs':['c','identity']}], 'output':'d','alpha':2.}
    m=GraphModel(spec).fit(x,y,rows); m.save(tmp_path)
    loaded=GraphModel.load(tmp_path)
    np.testing.assert_allclose(loaded.predict(x),m.predict(x),atol=1e-12)
    np.testing.assert_allclose(loaded.predict({k:v[:1] for k,v in x.items()}),m.predict(x)[:1],atol=1e-12)
    before={k:v.copy() for k,v in loaded.params.items()}
    loaded.predict({k:v*10 for k,v in x.items()})
    for k,v in before.items():np.testing.assert_array_equal(v,loaded.params[k])


@pytest.mark.parametrize('op',['exec','eval','load','__import__'])
def test_graph_no_execution_escape(op):
    with pytest.raises(ValueError):validate_graph({'nodes':[{'id':'a','op':op,'inputs':['genes']}],'output':'a','alpha':1})


def test_nonfinite_predictions_cannot_win():
    _,y,rows=data()
    with pytest.raises(ValueError):score(np.full_like(y,np.nan),y,rows)
    assert score(y,y,rows)['coverage']==1
    assert score(y,y,rows)['macro_unit_mse']==0


def test_linked_cohort_closure():
    rows=[{'study_family':'s','biological_unit':str(i),'link_ids':[str(i)],'source_sample_key':str(i),'split':'train'} for i in range(3)]
    policy={'samples':[{'sample_key':str(i),'cohort_id':'same','link_ids':[]} for i in range(3)]}
    assert len({r['biological_unit'] for r in connected_rows(rows,policy)})==1
    rows[-1]['split']='validation'
    with pytest.raises(ValueError,match='cohort'):connected_rows(rows,policy)


def test_knowledge_deduplicates_and_excludes_evaluation(tmp_path):
    record={'split':'train','reviewed':True,'text':'A fact','source_ref':'https://example.org/paper'}
    f=tmp_path/'e.jsonl';f.write_text('\n'.join(json.dumps(r) for r in [record,record,{**record,'split':'test','text':'test answer'}]))
    assert len(knowledge_records(f))==1


def test_completed_artifact_tampering_detected(tmp_path):
    (tmp_path/'p.txt').write_text('a');freeze_artifacts(tmp_path);verify_artifacts(tmp_path)
    (tmp_path/'p.txt').write_text('b')
    with pytest.raises(ValueError,match='changed'):verify_artifacts(tmp_path)


def test_partial_json_is_not_repaired():
    with pytest.raises(json.JSONDecodeError):parse_object('{"hypothesis":"good",')
    with pytest.raises(json.JSONDecodeError):parse_object('prefix {"a":1} suffix')


def api_env(monkeypatch):
    monkeypatch.setenv('VDC_RESEARCH_API_BASE','https://provider.example/v1')
    monkeypatch.setenv('VDC_RESEARCH_MODEL','test-model')
    monkeypatch.setenv('VDC_RESEARCH_API_KEY','fixture-secret-not-real')


def api_config():
    return {'max_output_tokens':200,'max_api_calls':2,'max_token_budget':10000,'max_request_budget':10000,'api_timeout_seconds':10}


class Response:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,n):
        return json.dumps({'choices':[{'message':{'content':'{"ok":true}'},'finish_reason':'stop'}],
                           'usage':{'total_tokens':15}}).encode()


def test_api_cache_budget_and_no_secret_persistence(tmp_path,monkeypatch):
    api_env(monkeypatch);c=Client(tmp_path,api_config())
    with patch('urllib.request.urlopen',return_value=Response()) as request:
        assert c.call('a','sys','question')['ok']
        assert c.call('a','sys','question')['ok'];assert request.call_count==1
        c.call('b','sys','second')
        with pytest.raises(RuntimeError,match='budget'):c.call('c','sys','third')
    assert all('fixture-secret-not-real' not in p.read_text() for p in (tmp_path/'api').glob('*'))


def test_api_error_requires_explicit_retry(tmp_path,monkeypatch):
    api_env(monkeypatch);c=Client(tmp_path,api_config())
    with patch('urllib.request.urlopen',side_effect=TimeoutError('private response')):
        with pytest.raises(RuntimeError):c.call('a','sys','question')
    with patch('urllib.request.urlopen') as request:
        with pytest.raises(RuntimeError,match='retry'):c.call('a','sys','question')
        request.assert_not_called()
    assert 'private response' not in next((tmp_path/'api').glob('*')).read_text()


def test_endpoint_rejects_embedded_secrets(monkeypatch):
    api_env(monkeypatch);monkeypatch.setenv('VDC_RESEARCH_API_BASE','https://provider.example/v1?token=bad')
    with pytest.raises(ValueError):endpoint()


def test_citations_and_docker_capability_gate():
    p={k:'text' for k in ('question','hypothesis','falsification','expected_observation','change_from_previous')}
    p.update(alternatives=['alternative'],evidence_ids=['E1'],limitations=[],literature_queries=[],
             implementation={'backend':'graph','spec':baseline_specs()['visible_ridge']})
    validate_proposal(p,[{'id':'E1'}],False)
    with pytest.raises(ValueError):validate_proposal(p,[{'id':'E2'}],False)
    p['implementation']={'backend':'python_docker','code':'print(1)'}
    with pytest.raises(ValueError,match='Docker'):validate_proposal(p,[{'id':'E1'}],False)


def test_target_counts_cannot_change_features():
    from vdc.clock_wave import inputs, hidden_partition
    genes=['g'+str(i) for i in range(80)];hidden=hidden_partition(genes)
    counts=np.random.default_rng(9).integers(1,100,(6,80));changed=counts.copy()
    changed[:,[i for i,g in enumerate(genes) if g in hidden]]*=100
    definitions=[{'id':'P','members':{g:1 for g in genes},'source_ref':'fixture'}]
    a=inputs(counts,genes,definitions);b=inputs(changed,genes,definitions)
    np.testing.assert_allclose(a[0],b[0]);np.testing.assert_allclose(a[1]['values'],b[1]['values']);np.testing.assert_allclose(a[2],b[2])


def test_real_numeric_loop_with_simulated_api_and_reflection_resume(tmp_path,monkeypatch):
    """Software integration only: explicitly mocked API, no AI-research claim."""
    from vdc import research as r
    from vdc.io import save_npz, write_json
    repo=tmp_path/'repo';(repo/'scripts').mkdir(parents=True)
    (repo/'scripts/run_research.py').write_text('# fixture')
    corpus=tmp_path/'e.jsonl';corpus.write_text(json.dumps({'split':'train','reviewed':True,'text':'A fact',
                                                         'source_ref':'https://example.org/paper'}))
    monkeypatch.setattr(r,'source_fingerprint',lambda *a:{'fixture':'software_only'})
    def prepare(source,config,out):
        x,y,rows=data();out.mkdir(parents=True)
        save_npz(out/'train.npz',**x,target=y);save_npz(out/'query.npz',**x)
        save_npz(out/'truth.npz',target=y)
        write_json(out/'train_rows.json',rows);write_json(out/'query_rows.json',rows)
        write_json(out/'summary.json',{'software_fixture':True});return {'software_fixture':True}
    monkeypatch.setattr(r,'prepare_view',prepare)
    fits=[]
    def execute(impl,run,out,config,repo):
        fits.append(str(out));write_json(out/'implementation.json',impl)
        return r.evaluate_candidate(impl,run,out,config)
    monkeypatch.setattr(r,'run_isolated',execute)
    class FakeClient:
        fail=True
        contexts=[]
        def __init__(self,*a):pass
        def call(self,key,system,user,validator=None):
            if key.endswith('proposal'):
                self.contexts.append(json.loads(user))
                p={k:'fixture hypothesis' for k in ('question','hypothesis','falsification','expected_observation','change_from_previous')}
                spec=baseline_specs()['visible_ridge'];spec['alpha']=2 if '001' in key else 3
                p.update(alternatives=['alternative'],evidence_ids=['E1'],limitations=['software fixture'],literature_queries=[],
                         implementation={'backend':'graph','spec':spec})
                return p
            if self.fail:
                FakeClient.fail=False;raise RuntimeError('simulated reflection interruption')
            obj={'conclusion':'inconclusive','reasoning':'software test','competing_explanations':[],
                 'next_question':'compare an alternative','limitations':['not actual AI research']}
            if validator:validator(obj)
            return obj
    monkeypatch.setattr(r,'Client',FakeClient)
    config={'views':['fixture'],'allow_python_docker':False,'rounds':2,'evidence_per_call':10,'research_goal':'fixture'}
    run=tmp_path/'run'
    with pytest.raises(RuntimeError,match='reflection'):r.run_research(repo,tmp_path,run,config,corpus)
    assert len(fits)==4
    result=r.run_research(repo,tmp_path,run,config,corpus)
    assert result['completed_experiments']==2
    assert len(fits)==5  # First experiment did not refit after interrupted reflection.
    assert FakeClient.contexts[-1]['previous_experiments'][0]['reflection']['next_question']=='compare an alternative'
    r.run_research(repo,tmp_path,run,config,corpus)
    assert len(fits)==5
