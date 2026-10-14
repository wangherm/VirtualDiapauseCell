"""Protocol tests, not claimed autonomous research or biological evaluation."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from vdc.io import write_json, read_json, object_hash
from vdc.research_continuity import size
from vdc.research_bridge import snapshot, submit, process_one, rpc, PROTOCOL
from vdc.research_codex import controller, codex_command, advise, transport, run_codex
from tests.test_research_r2 import make_engine
from tests.test_research_engine import original
from vdc.research_r2_artifacts import artifact_index, checked_artifact


def test_joint_code_metadata_and_all_batch_results_survive_pressure(tmp_path):
    e = make_engine(tmp_path)
    pairs = [('code', 'read_preparation_code'), ('metadata', 'read_metadata'),
             ('schema', 'inspect_preparation'), ('old_action', 'read_action')]
    for key, kind in pairs:
        e.state['actions'][key] = {'type': kind, 'status': 'completed',
            'result': {'text': key+'中文'*20000}}
    e.state['last_batch_actions'] = [k for k, _ in pairs]
    e.state['observations'] = e.state['last_batch_actions'][:]
    c = e.context()
    assert set(c['recent_actions']) == {'code', 'metadata', 'schema', 'old_action'}
    assert all(r['excerpt_not_complete'] for r in c['recent_actions'].values())
    assert size(c) <= c['evidence_policy']['budget_bytes']
    full = read_json(e.run/'working_evidence.json')
    assert full['records']['code']['result']['text'] == 'code'+'中文'*20000
    # Even after an unrelated batch, code and metadata remain together.
    e.state['last_batch_actions'] = ['old_action']
    c = e.context()
    assert {'code', 'metadata'} <= set(c['retained_evidence'])


def test_precise_wrong_path_directory_and_integrity_errors(tmp_path):
    inv, prov = original(tmp_path)
    write_json(tmp_path/'actions/p/provenance.json', prov)
    write_json(tmp_path/'actions/p/output/bundle/schema.json', {'features': ['x']})
    artifact_index(tmp_path, 'p', inv)
    with pytest.raises(ValueError, match='bundle/schema.json'):
        checked_artifact(tmp_path, 'p', 'schema.json', inv)
    with pytest.raises(ValueError, match='directory'):
        checked_artifact(tmp_path, 'p', 'bundle', inv)
    write_json(tmp_path/'actions/p/output/bundle/schema.json', {'tampered': True})
    with pytest.raises(ValueError, match='changed'):
        checked_artifact(tmp_path, 'p', 'bundle/schema.json', inv)


def request_for(e, snap=None):
    s = snap or snapshot(e)
    return {'protocol': PROTOCOL, 'index': s['next_decision'],
            'snapshot_sha256': s['snapshot_sha256'],
            'proposal': {'actions': [{'id': 'inventory_one', 'type': 'inventory'}],
                         'memory': 'Software fixture; no scientific experiment'}}


def test_mailbox_idempotence_hash_binding_and_stale_protection(tmp_path):
    e = make_engine(tmp_path)
    request = request_for(e)
    assert submit(e.run, request)['status'] == 'submitted'
    assert submit(e.run, request)['status'] == 'already_submitted'
    with patch.object(e, 'action', wraps=e.action) as act:
        assert process_one(e)
        assert act.call_count == 1
        assert not process_one(e)
        assert act.call_count == 1
    result = rpc(e.run, {'operation': 'poll', 'index': 1})
    assert result['request_sha256'] == object_hash(request)
    assert result['new_api_calls_on_worker'] == 0
    changed = json.loads(json.dumps(request)); changed['proposal']['memory'] = 'different'
    with pytest.raises(ValueError, match='different content'):
        submit(e.run, changed)
    changed['index'] = 2
    with pytest.raises(ValueError, match='Stale snapshot'):
        submit(e.run, changed)


def test_uncertain_execution_does_not_automatically_retrain(tmp_path):
    e = make_engine(tmp_path); request = request_for(e)
    submit(e.run, request)
    write_json(e.run/'mailbox/000001.running.json', {'started_at': 0})
    with patch.object(e, 'action') as action:
        with pytest.raises(RuntimeError, match='Uncertain'):
            process_one(e)
        action.assert_not_called()


def test_controller_recovers_same_submission_after_network_disconnect(tmp_path):
    e = make_engine(tmp_path); snapshot(e)
    workspace = tmp_path/'local'
    connection = {'transport': 'files', 'run': str(e.run)}
    disconnected = False
    calls = []
    def remote(c, packet):
        nonlocal disconnected
        calls.append(packet['operation'])
        result = rpc(e.run, packet)
        if packet['operation'] == 'submit' and not disconnected:
            process_one(e)
            disconnected = True
            raise ConnectionError('test dropped reply after real tool completed')
        return result
    proposal = {'actions': [{'id': 'once', 'type': 'inventory'}]}
    with patch('vdc.research_codex.transport', side_effect=remote), patch('vdc.research_codex.run_codex', return_value=proposal) as primary:
        with pytest.raises(ConnectionError):
            controller(connection, workspace, turns=1)
        assert read_json(workspace/'controller_state.json')['pending']['index'] == 1
        result = controller(connection, workspace, turns=1)
        assert result['next_turn'] == 2
        assert primary.call_count == 1
        assert e.state['next_decision'] == 2
        assert len(e.state['actions']) == 1


def test_advisor_cannot_submit_actions_and_has_separate_usage(tmp_path, monkeypatch):
    monkeypatch.setenv('VDC_RESEARCH_API_KEY', 'unit-test-value-not-real')
    with patch('vdc.research_codex.Client') as client:
        client.return_value.call.return_value = {'concerns': 'test',
            'recommended_next_experiment': 'compare alternatives', 'evidence_gaps': 'test-only'}
        result = advise(tmp_path, 1, 'Review this proposal', {'progress': {}}, {})
        assert result['worker_actions_executed'] == 0
        assert result['decision_authority'] == 'local_codex'
        assert result['status'] == 'completed'
        assert 'actions' not in result['response']
        advise(tmp_path, 1, 'Review this proposal', {'progress': {}}, {})
        assert client.return_value.call.call_count == 1


def test_codex_resume_uses_explicit_session_without_bypass(tmp_path):
    args = codex_command('codex', tmp_path, tmp_path/'out.txt', 'session-fixture')
    assert args[args.index('resume')+1] == 'session-fixture'
    assert '--last' not in args
    assert not any('bypass' in x or 'danger-full' in x for x in args)
    assert 'sandbox_mode="workspace-write"' in args


def test_ssh_arguments_are_quoted_without_keys_in_payload(tmp_path):
    c = {'transport': 'ssh', 'ssh_target': 'fixture-host', 'python': '/path with space/python',
         'worker_script': '/frozen release/worker.py', 'run': '/private/run'}
    from types import SimpleNamespace
    with patch('vdc.research_codex.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='{"ok":true}')) as runner:
        assert transport(c, {'operation': 'snapshot'})['ok']
        args = runner.call_args.args[0]
        assert 'StrictHostKeyChecking=yes' in args
        assert "'/path with space/python'" in args[-1]
        assert runner.call_args.kwargs.get('shell') is None
    with pytest.raises(ValueError):
        transport({**c, 'ssh_target': '-oProxyCommand=bad'}, {})


def test_external_decision_keeps_no_progress_stop(tmp_path):
    e = make_engine(tmp_path)
    e.config.update(no_progress_warning_decisions=1, no_progress_stop_decisions=2)
    snapshot(e)
    for i in range(2):
        r = request_for(e)
        r['proposal']['actions'][0]['id'] = 'inspect_'+str(i)
        submit(e.run, r); process_one(e)
    assert e.state['status'] == 'stalled_no_progress'
    r = request_for(e)
    with pytest.raises(ValueError, match='stopped'):
        submit(e.run, r)


def test_private_report_includes_frozen_controller_source_not_release_data(tmp_path):
    import importlib.util
    import zipfile
    script = Path(__file__).resolve().parents[1]/'scripts/pack_research_r2.py'
    spec = importlib.util.spec_from_file_location('pack_codex_test', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    run = tmp_path/'run'
    write_json(run/'manifest.json', {'protocol': 'test'})
    write_json(run/'status.json', {'status': 'software_fixture'})
    source = run/'release/src/vdc/research_codex.py'
    source.parent.mkdir(parents=True); source.write_text('# fixture')
    write_json(run/'release/private/data.json', {'must_not_export': True})
    write_json(run/'actions/p/output/data.json', {'must_not_export': True})
    packed = module.pack(run)
    with zipfile.ZipFile(packed) as archive:
        assert 'release/src/vdc/research_codex.py' in archive.namelist()
        assert 'release/private/data.json' not in archive.namelist()
        assert 'actions/p/output/data.json' not in archive.namelist()


def test_codex_scratch_is_separate_from_host_receipts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    write_json(tmp_path/'context.json', {'mission': 'software test only'})
    write_json(tmp_path/'working_evidence.json', {'records': {}})
    (tmp_path/'tool_contract.md').write_text('fixture')
    monkeypatch.setenv('VDC_RESEARCH_API_KEY', 'not-a-real-key')
    def fake_exec(args, **kwargs):
        assert kwargs['cwd'] == tmp_path/'agent'
        assert 'VDC_RESEARCH_API_KEY' not in kwargs['env']
        assert (kwargs['cwd']/'context.json').is_file()
        assert not (kwargs['cwd']/'turns').exists()
        Path(args[args.index('-o')+1]).write_text('{"actions":[{"id":"x","type":"inventory"}]}')
        kwargs['stdout'].write('{"type":"thread.started","thread_id":"test-session"}\n')
        return SimpleNamespace(returncode=0)
    with patch('vdc.research_codex.shutil.which', return_value='codex'), patch('vdc.research_codex.subprocess.run', side_effect=fake_exec):
        assert run_codex(tmp_path, 1)['actions'][0]['id'] == 'x'
    assert read_json(tmp_path/'codex_session.json')['thread_id'] == 'test-session'
    assert read_json(tmp_path/'turns/000001/codex_receipt.json')['status'] == 'completed'
    write_json(tmp_path/'context.json', {'mission': 'changed'})
    with pytest.raises(ValueError, match='changed'):
        run_codex(tmp_path, 1)
