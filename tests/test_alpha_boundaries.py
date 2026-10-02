"""Synthetic fixtures below test boundaries only; they are never registered as biological results."""
from pathlib import Path
import copy
import json
import sys
import numpy as np
import pytest
from .fixtures import observation_fixture
from vdc.state import StateConfig,fit_state,StatePredictor
from vdc.evaluate import evaluate_state
from vdc.clock_reference import ExitReference
from vdc.io import read_json,write_json,object_hash

ROOT=Path(__file__).resolve().parents[1]


def test_validation_forward_never_receives_other_splits(tmp_path,monkeypatch):
    b=observation_fixture();cfg=StateConfig(hidden_dim=16,latent_dim=8,layers=1,heads=2,validation_every=2)
    fit_state(b,tmp_path/'state',2,cfg)
    manifest=read_json(tmp_path/'state/run.json')
    dev=b.subset([i for i,r in enumerate(b.rows) if r['split'] in {'train','validation'}])
    assert manifest['bundle_fingerprint']==dev.fingerprint
    called=[];original=StatePredictor.predict
    def spy(self,query):
        called.extend(r['split'] for r in query.rows)
        assert all(r['split']=='validation' for r in query.rows)
        return original(self,query)
    monkeypatch.setattr(StatePredictor,'predict',spy)
    evaluate_state(tmp_path/'state',b,tmp_path/'evaluation',split='validation')
    assert len(called)==len(b.indices('validation'))


def clock_fixture():
    rows=[]
    for split in ('train','validation'):
        for h in (1,4,15,30):
            for t in (0,6,24):rows.append({'split':split,'history_days':h,'elapsed_hours_since_release':t,'observation_id':f'{split}:{h}:{t}'})
    genes=[f'WBGene{i:06}' for i in range(120)]
    x=np.random.default_rng(42).normal(size=(24,120))
    x+=np.array([r['elapsed_hours_since_release']/24 for r in rows])[:,None]
    return x,genes,rows


def test_reference_never_fits_validation_or_clips(tmp_path):
    x,genes,rows=clock_fixture();a=ExitReference().fit(x,genes,rows);a.save(tmp_path)
    changed=x.copy();changed[12:]*=10000;b=ExitReference().fit(changed,genes,rows)
    np.testing.assert_array_equal(a.indices,b.indices);np.testing.assert_allclose(a.direction,b.direction)
    assert set(a.metadata['axis_genes']).isdisjoint(a.metadata['readout_genes'])
    np.testing.assert_allclose(a.predict(x,genes),ExitReference.load(tmp_path).predict(x,genes))
    # A new sample beyond the reference is reported beyond 1, never forced to an endpoint.
    far=np.tile(a.mean+a.scale*(a.origin+3*a.direction/np.dot(a.direction,a.direction)),(1,1))
    q=x[:1].copy();q[:,a.indices]=far
    assert a.predict(q,genes)[0]==pytest.approx(3)


def test_reference_rejects_modified_artifact(tmp_path):
    x,g,r=clock_fixture();ExitReference().fit(x,g,r).save(tmp_path)
    with (tmp_path/'axis.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):ExitReference.load(tmp_path)


def test_curated_corpus_has_sources_and_no_preserved_families():
    from vdc.io import read_jsonl
    from vdc.knowledge import audit_knowledge
    r=read_jsonl(ROOT/'knowledge/alpha/corpus.jsonl');c=read_json(ROOT/'configs/all_modules_alpha.json')
    report=audit_knowledge(r,set(c['excluded_families']))
    assert report['study_families']==3
    for row in r:
        assert row['source_locator'] and row['review_basis']
        assert row['label_source']=='source_curated_weak_reference'
        answer=json.loads(row['completion']);assert answer['source']==row['study_family']
    assert any(json.loads(row['completion'])['uncertain'] for row in r if row['split']=='train')


def test_functional_group_counts_are_not_expression_labels():
    m=read_json(ROOT/'data/curated/functional_fig1b.json')
    assert len(m['records'])==24
    for r in m['records']:
        assert sum(r['counts'].values())==r['total']
        assert r['replicate_block'] in {1,2}
        assert r['source_cells'].startswith('Fig1B!')
    assert 'expression' not in m['records'][0]


def test_registry_rejects_changed_weights(tmp_path):
    from vdc.experimental import build_registry,create_experimental_app
    from vdc.io import sha256
    folder=tmp_path/'state_base';folder.mkdir();(folder/'best.pt').write_bytes(b'artifact_only_boundary_fixture')
    write_json(tmp_path/'module_status.json',{'modules':{'state_base':{'execution_status':'completed',
        'artifact':'state_base','files':{'best.pt':sha256(folder/'best.pt')}}}})
    build_registry(tmp_path);(folder/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):create_experimental_app(tmp_path)


def test_runner_partial_and_resume_integrity(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_all_modules import summarize,inventory,verify_inventory
    (tmp_path/'file.json').write_text('{}');manifest=inventory(tmp_path)
    assert verify_inventory(tmp_path,manifest)
    (tmp_path/'file.json').write_text('{"changed":true}')
    assert not verify_inventory(tmp_path,manifest)
    report=summarize(tmp_path,{'functional':{'execution_status':'blocked_data','reason':'unpaired'}})
    assert report['overall_status']=='partial'


def test_no_gpu_is_a_blocker_not_success(monkeypatch):
    import torch
    from vdc.knowledge_train import require_gpu
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='GPU'):require_gpu()


def test_adapter_embedding_really_uses_peft(tmp_path,monkeypatch):
    # Dependency stubs test routing, not numerical or biological correctness.
    import types,torch
    from vdc.knowledge import embed_objects
    calls=[]
    class Tokenizer:
        pad_token_id=0
        def __call__(self,*args,**kw):return {'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.tensor([[1,1]])}
    class Model:
        adapted=False
        def to(self,*args):return self
        def eval(self):return self
        def __call__(self,**kw):return types.SimpleNamespace(hidden_states=[torch.ones(1,2,3)*(2 if self.adapted else 1)])
    class Loader:
        @staticmethod
        def from_pretrained(*args,**kw):return Model()
    class Peft:
        @staticmethod
        def from_pretrained(model,path,**kw):calls.append(path);model.adapted=True;return model
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoModelForCausalLM=Loader,AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**kw:Tokenizer())))
    monkeypatch.setitem(sys.modules,'peft',types.SimpleNamespace(PeftModel=Peft))
    adapter=tmp_path/'adapter';adapter.mkdir();(adapter/'adapter_config.json').write_text('{}')
    records=[{'record_id':'x','object_id':'P1','study_family':'fixture','source_ref':'fixture','split':'train','kind':'object_description','reviewed':True,'text':'fixture'}]
    p=embed_objects(records,tmp_path/'vectors','fixture','a'*40,adapter=adapter)
    assert calls==[str(adapter)] and p['weight_kind']=='domain_adapter' and p['adapter_hash']
    with np.load(tmp_path/'vectors/embeddings.npz') as a:np.testing.assert_array_equal(a['vectors'],np.full((1,3),2))
