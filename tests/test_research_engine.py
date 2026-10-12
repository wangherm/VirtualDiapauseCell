import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from vdc.io import read_json,write_json,sha256,save_npz
from vdc.research_r2_artifacts import artifact_index,inspect_preparation,preparation_inputs,read_metadata,current_roles
from vdc.research_r2_contracts import verify_lineage,verify_dependency
from vdc.research_r2_progress import request_fingerprint,progress_signature,observe_progress
from vdc.research_r2_downloads import acquire,discover,official_url
from vdc.research_r2_continue import continue_engine_phase
from tests.test_research_r2 import make_engine


def original(tmp_path,public=True):
    p=tmp_path/'original.json'
    write_json(p,[{'sample_id':'s1','linked_embryo_key':'e1','biological_unit':'section_from_embryo',
                   'condition':'early','raw_counts':[12345],'private_numeric':87654}])
    inv={'sources':{'s':{'asset_ids':['a'],'disposition':'not_processed'}},'assets':{'a':{
        'path':str(p),'name':p.name,'source':'s','public':public,'bytes':p.stat().st_size}}}
    return inv,{'a':{'source':'s','sha256':sha256(p)}}


def test_public_metadata_readable_and_lineage_cannot_be_relabelled(tmp_path):
    inv,prov=original(tmp_path)
    root=tmp_path/'actions/p';write_json(root/'provenance.json',prov)
    write_json(root/'output/metadata.json',{'actual_sample_ids':['GSM123']})
    idx=artifact_index(tmp_path,'p',inv)
    assert idx['status']=='metadata_ready'
    a={'preparation':'p','path':'metadata.json'}
    assert 'GSM123' in inspect_preparation(tmp_path,inv,a)['text']
    inv['assets']['a']['public']=False
    r=inspect_preparation(tmp_path,inv,a)
    assert 'text' not in r and 'GSM123' not in json.dumps(r)
    mounts,ancestors,refs=preparation_inputs(tmp_path,inv,[a])
    assert len(mounts)==1 and set(ancestors)=={'a'} and not next(iter(refs.values()))['public']
    write_json(root/'output/metadata.json',{'changed':True})
    with pytest.raises(ValueError,match='changed'):preparation_inputs(tmp_path,inv,[a])
    with pytest.raises(ValueError):preparation_inputs(tmp_path,inv,[{'preparation':'p','path':'../../state.json'}])


def test_private_metadata_values_are_projected_from_original(tmp_path):
    inv,_=original(tmp_path,False)
    r=read_metadata(inv,'a')
    assert r['rows'][0]['linked_embryo_key']=='e1'
    assert '12345' not in json.dumps(r) and '87654' not in json.dumps(r)
    inv['sources']['s']['ai_brief']='archive-only, do not train'
    inv=current_roles(inv)
    assert 'archive-only' not in inv['sources']['s']['ai_brief']
    assert 'archive-only' in inv['sources']['s']['historical_policy']['ai_brief']


def test_metadata_only_is_not_task_ready(tmp_path):
    inv,prov=original(tmp_path);root=tmp_path/'actions/p'
    write_json(root/'provenance.json',prov);write_json(root/'output/schema.json',{})
    r=artifact_index(tmp_path,'p',inv)
    assert r['status']=='metadata_ready' and 'train.npz' in r['bundles']['.']['missing']
    assert r['bundles']['.']['registration'] is None


def test_original_units_and_links_are_not_material_types(tmp_path):
    inv,prov=original(tmp_path);row={'replication_unit_id':'s1','unit_type':'section',
        'linked_embryo_key':'e1','unit_evidence':{'asset':'a','row_field':'sample_id','row_value':'s1','unit_field':'biological_unit'}}
    write_json(tmp_path/'train_rows.json',[row]);write_json(tmp_path/'query_rows.json',[])
    with pytest.raises(ValueError,match='ID field'):verify_lineage(tmp_path,inv,prov)
    row['unit_evidence']['unit_field']='sample_id';write_json(tmp_path/'train_rows.json',[row])
    rows,audit=verify_lineage(tmp_path,inv,prov)
    assert rows[0][0]['biological_unit']=='s1' and 's:linked_embryo_key:e1' in rows[0][0]['link_ids']
    row.pop('linked_embryo_key');write_json(tmp_path/'train_rows.json',[row])
    with pytest.raises(ValueError,match='Preserve original'):verify_lineage(tmp_path,inv,prov)


def test_counterfactual_catches_all_gene_cpm(tmp_path):
    # Tests host validation with a stand-in executor, not a claimed isolated run.
    p=tmp_path/'counts.csv';p.write_text('gene,s1,s2\na,2,4\nb,10,20\n')
    inv={'assets':{'matrix':{'path':str(p),'source':'s','public':True}}};prov={'matrix':{'source':'s'}}
    write_json(tmp_path/'train_rows.json',[{'matrix_column':'s1'}]);write_json(tmp_path/'query_rows.json',[{'matrix_column':'s2'}])
    write_json(tmp_path/'schema.json',{'features':['b'],'targets':['a']});(tmp_path/'feature_probe.py').write_text('# test fixture')
    raw=np.array([[2.,10.],[4.,20.]])
    def transform(c):return np.log1p(c[:,1:]/c.sum(axis=1,keepdims=True)*1e6)
    x=transform(raw);save_npz(tmp_path/'train.npz',x=x[:1]);save_npz(tmp_path/'query.npz',x=x[1:])
    def execute(code,inputs,out,*args,**kw):
        save_npz(out/'features.npz',x=transform(np.load(inputs['probe.npz'])['counts']))
    contract={'input_dependency':{'kind':'target_isolated','matrix_asset':'matrix','target_genes':['a']}}
    with patch('vdc.research_r2_contracts.execute',side_effect=execute):
        with pytest.raises(ValueError,match='Changing hidden targets'):
            verify_dependency(tmp_path,contract,inv,prov,tmp_path,tmp_path/'probe',{},tmp_path)


def test_content_not_action_id_and_recoverable_stall(tmp_path):
    e=make_engine(tmp_path)
    a={'id':'first','type':'read_vdc','path':'src/vdc/core.py'}
    path=e.repo/'src/vdc/core.py';path.parent.mkdir(exist_ok=True,parents=True);path.write_text('x=1')
    assert request_fingerprint(e,a)==request_fingerprint(e,{**a,'id':'second'})
    first=request_fingerprint(e,a);path.write_text('x=2');assert first!=request_fingerprint(e,a)
    e.config.update(no_progress_warning_decisions=1,no_progress_stop_decisions=2)
    before=progress_signature(e);observe_progress(e,before);assert e.state['engine_progress']['diagnostic_mode']
    observe_progress(e,before);assert e.state['status']=='stalled_no_progress'
    with patch('vdc.research_r2.Client') as c:
        e.run_loop();c.assert_not_called()


@pytest.mark.parametrize('url',['http://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1234/suppl/x.gz',
    'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE9999/suppl/x.gz',
    'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1234/suppl/x.gz?token=secret',
    'https://example.org/x.gz'])
def test_download_rejects_unbound_or_credential_urls(url):
    with pytest.raises(ValueError):official_url(url,'GSE1234')


def test_download_observed_hash_is_not_provider_verified_and_reuses(tmp_path):
    import io
    e=make_engine(tmp_path);e.inv['sources']['GSE1234']={'asset_ids':[]}
    a={'source':'GSE1234','file':'x.gz','url':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1234/suppl/x.gz','bytes':4}
    class Response(io.BytesIO):headers={'Content-Length':'4'}
    with patch('vdc.research_r2_downloads.public_url'),patch('vdc.research_r2_downloads.shutil.disk_usage',return_value=SimpleNamespace(free=100_000_000_000)),patch('vdc.research_r2_downloads.urllib.request.build_opener') as op:
        op.return_value.open.return_value=Response(b'data')
        r=acquire(e,a);assert 'provider_checksum_unavailable' in r['verification'] and not r['trained']
        again=acquire(e,a);assert again['sha256']==r['sha256'] and op.call_count==1
        bad={**a,'bytes':5}
        with pytest.raises(ValueError,match='changed'):acquire(e,bad)


def test_download_disk_full_is_branch_failure(tmp_path):
    e=make_engine(tmp_path);e.inv['sources']['GSE1234']={'asset_ids':[]}
    with patch('vdc.research_r2_downloads.shutil.disk_usage',return_value=SimpleNamespace(free=1)):
        with pytest.raises(ValueError,match='disk space'):
            acquire(e,{'source':'GSE1234','file':'x.gz','url':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1234/suppl/x.gz','bytes':4})


def test_discovery_admits_only_confirmed_official_accession(tmp_path):
    e=make_engine(tmp_path)
    def fetch(url,out):
        out.mkdir(parents=True,exist_ok=True)
        doc='metadata' if 'query/acc' in url else 'files'
        (out/(doc+'.txt')).write_text('^SERIES = GSE1234\n!Series_title = real study\n' if doc=='metadata' else 'files')
        return {'id':doc,'url':url}
    with patch('vdc.research_r2_downloads.fetch_public',side_effect=fetch):
        r=discover(e,{'source':'GSE1234','research_reason':'Test additional measured recovery time courses'})
        assert r['status']=='discovered_not_downloaded' and not e.inv['sources']['GSE1234']['asset_ids']
        with pytest.raises(ValueError,match='did not confirm'):
            discover(e,{'source':'GSE9999','research_reason':'Test additional measured recovery time courses'})
    assert 'GSE9999' not in e.inv['sources']


def test_ena_file_binding_uses_official_checksum(tmp_path):
    import io
    e=make_engine(tmp_path)
    e.inv['sources']['PRJEB123']={'asset_ids':[],'discovery':{'official_records':['ena']}}
    (e.run/'literature').mkdir()
    (e.run/'literature/ena.txt').write_text('study_accession\tfastq_ftp\tfastq_md5\tfastq_bytes\nPRJEB123\tftp.sra.ebi.ac.uk/vol1/fastq/ERR1/x.gz\t8d777f385d3dfec8815d20f7496026dc\t4\n')
    class Response(io.BytesIO):headers={'Content-Length':'4'}
    with patch('vdc.research_r2_downloads.public_url'),patch('vdc.research_r2_downloads.shutil.disk_usage',return_value=SimpleNamespace(free=100_000_000_000)),patch('vdc.research_r2_downloads.urllib.request.build_opener') as op:
        op.return_value.open.return_value=Response(b'data')
        r=acquire(e,{'source':'PRJEB123','file':'x.gz','url':'https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR1/x.gz','bytes':4})
        assert r['verification']=='provider_checksum_and_size'
        with pytest.raises(ValueError,match='official project'):
            acquire(e,{'source':'PRJEB123','file':'wrong.gz','url':'https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR1/wrong.gz','bytes':4})


def test_preparation_references_and_new_ledger_survive_phase_migration(tmp_path):
    from vdc.research_r2_tools import tree_hash
    e=make_engine(tmp_path)
    import shutil
    dst=e.private/'runs/old';dst.parent.mkdir(parents=True);shutil.move(str(e.run),dst);e.run=dst
    inv,prov=original(e.private,True);e.inv=inv
    write_json(dst/'actions/prep/provenance.json',prov)
    write_json(dst/'actions/prep/output/metadata.json',{'sample':'GSM123'})
    e.state.update(status='budget_exhausted_partial',next_decision=1001)
    e.state['actions']['prep']={'type':'prepare','status':'completed','signature':'old','result':{}}
    e.save();before=tree_hash(dst)
    with patch('vdc.research_r2.resolve_runtime',return_value='test-runtime'):
        child=continue_engine_phase(e.repo,e.private,dst,e.private/'runs/new',e.config)
    assert tree_hash(dst)==before
    assert child.state['actions']['prep']['result']['artifacts']['status']=='metadata_ready'
    assert 'GSM123' in child.action({'id':'read','type':'inspect_preparation','preparation':'prep','path':'metadata.json'})['text']
    assert child.budget()['requests']==0


def test_new_phase_preserves_trained_history_and_resets_only_new_ledger(tmp_path):
    from vdc.research_r2_tools import tree_hash
    e=make_engine(tmp_path);parent=e.run
    # make_engine uses a sibling run; move into the authorised private tree.
    import shutil
    dst=e.private/'runs/old';dst.parent.mkdir(parents=True);shutil.move(str(parent),dst);e.run=dst;parent=dst
    e.state.update(status='budget_exhausted_partial',next_decision=1000)
    e.state['experiments']['old']={'metrics':{'macro_unit_mse':1.2}};e.state['tasks']['old']={'signature':'old'}
    write_json(parent/'api/old.json',{'usage':{'total_tokens':20},'budget_charge':30})
    e.save();before=tree_hash(parent)
    with patch('vdc.research_r2.resolve_runtime',return_value='test-runtime'):
        child=continue_engine_phase(e.repo,e.private,parent,e.private/'runs/new',e.config)
    assert tree_hash(parent)==before
    assert child.budget()['requests']==0 and child.state['next_decision']==1
    assert not child.state['experiments'] and read_json(child.run/'history/state.json')['experiments']['old']['metrics']['macro_unit_mse']==1.2
    assert read_json(child.run/'continuation.json')['historical_reported_tokens']==20


def test_abstract_citation_is_not_method_reproduction(tmp_path):
    from vdc.research_r2_tools import literature_kind,literature_links
    e=make_engine(tmp_path);url='https://arxiv.org/abs/2603.08127'
    assert literature_kind(url)=='abstract_page' and len(literature_links(url,''))==2
    write_json(e.run/'literature/doc.json',{'url':url})
    e.state['experiments']['a']={'evidence':['doc'],'method_use':'some idea'};e.state['literature_reads']={'doc':[0]}
    assert not e.method_evidence()
    write_json(e.run/'literature/doc.json',{'url':'https://arxiv.org/pdf/2603.08127'})
    assert e.method_evidence()  # read full-text + declared implementation mapping, not reproduction proof
