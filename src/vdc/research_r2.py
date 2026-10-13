"""Persistent tool-driven research, reusing VDC's API, numerical and data helpers."""
from pathlib import Path
import json
import os
import shutil
import time
import numpy as np
from .io import read_json, write_json, object_hash, save_npz, sha256
from .research_api import Client
from .research_r2_tools import (identifier, inside, tree_hash, write_code, inventory,
    asset_context, inspect_assets, resolve_assets, search_literature, fetch_public)
from .research_r2_runtime import (resolve_runtime, execute, seal_task, verify_task,
    numeric_npz, run_candidate)
from .research_r2_artifacts import (current_roles, read_metadata, artifact_index,
    inspect_preparation, preparation_inputs)
from .research_r2_progress import READ_TOOLS, request_fingerprint, progress_signature, observe_progress


from .research_protocol import SYSTEM, TOOLS, decision


def public_result(result):
    """Host-produced result fields only; generated logs and arbitrary JSON stay local."""
    return {k: v for k, v in result.items() if k not in ('model_files', 'code_files', 'files', 'path')}


class Engine:
    def __init__(self, repo, private, run, config, receipt=None, roots=()):
        self.repo, self.private, self.run = Path(repo).resolve(), Path(private).resolve(), Path(run).resolve()
        self.config = config; self.run.mkdir(parents=True, exist_ok=True)
        self.config['runtime']['image_id'] = resolve_runtime(config['runtime'])
        fingerprint = {p.relative_to(self.repo).as_posix(): sha256(p)
            for p in (self.repo/'src/vdc').glob('*.py')}
        record = {'protocol': 'VDC_AI4AI_R2', 'config': config, 'code': object_hash(fingerprint),
                  'all_original_roles_open': True, 'independent_confirmation': False,
                  'upstream_implementation': 'VDC tool controller; inspired mechanisms, not upstream algorithm reproduction'}
        m = self.run/'manifest.json'
        if m.exists() and read_json(m) != record:
            raise ValueError('Code/config/image changed: use new run; never reuse old completed actions')
        write_json(m, record)
        ip = self.run/'inventory.json'
        self.inv = current_roles(read_json(ip) if ip.exists() else inventory(self.repo, self.private, receipt, roots))
        write_json(ip, self.inv)
        sp = self.run/'state.json'
        self.state = read_json(sp) if sp.exists() else {'status': 'ready', 'actions': {}, 'tasks': {},
            'experiments': {}, 'exports': {}, 'memories': [], 'next_decision': 1}
        self.prior = None
        pointer=self.private/'LATEST_RESEARCH.txt'
        if pointer.is_file():
            prior=Path(pointer.read_text().strip()).resolve()
            if prior.is_relative_to(self.private) and (prior/'status.json').is_file():
                self.prior=prior

    def save(self):
        write_json(self.run/'state.json', self.state); write_json(self.run/'inventory.json', self.inv)

    def runtime(self,device='cpu'):
        from .research_r2_devices import select_device
        c = dict(self.config['runtime'])
        if (self.run/'wheels').exists(): c['wheels'] = str(self.run/'wheels')
        return select_device(c,device)

    def task_path(self, key):
        return self.run/'tasks'/identifier(key)

    def action(self, a):
        kind, key = a['type'], identifier(a['id']); out = self.run/'actions'/key
        if kind=='read_metadata': return read_metadata(self.inv,a['asset'],a.get('offset',0))
        if kind=='read_history':
            section=a['section']
            if section not in ('tasks','experiments'): raise ValueError('Select historical tasks or experiments')
            p=self.run/'history/state.json'
            history=read_json(p).get(section,{}) if p.exists() else {}
            items=list(history.items());offset=max(0,int(a.get('offset',0)))
            lineage=read_json(self.run/'continuation.json') if (self.run/'continuation.json').exists() else {}
            return {'records':dict(items[offset:offset+3]),'total':len(items),
                'limitations':lineage.get('historical_limitations','Historical results are not new-phase validation; review their original contracts.')}
        if kind=='read_preparation_code':
            path=inside(self.run/'actions'/identifier(a['preparation'])/'code',a['path'])
            if not path.is_file():
                available=[p.relative_to(path.parent).as_posix() for p in path.parent.glob('*.py')]
                raise ValueError('Code file absent; available sibling Python files: '+str(available))
            text=path.read_text(encoding='utf-8');offset=max(0,int(a.get('offset',0)))
            return {'text':text[offset:offset+8000],'characters':len(text),'source_sha256':sha256(path)}
        if kind == 'list_actions':
            rows = [{'id': k, **{f: v.get(f) for f in ('type', 'status', 'error_type')}}
                    for k,v in self.state['actions'].items() if k != key
                    and (not a.get('status') or v['status'] == a['status'])
                    and (not a.get('tool_type') or v['type'] == a['tool_type'])]
            offset = max(0, int(a.get('offset', 0)))
            return {'actions': rows[offset:offset+20], 'total': len(rows),
                    'next_offset': offset+20 if offset+20<len(rows) else None}
        if kind == 'read_action':
            previous = self.state['actions'][identifier(a['action'])]
            text = json.dumps(previous, ensure_ascii=False)
            offset = max(0, int(a.get('offset', 0)))
            return {'action': a['action'], 'text': text[offset:offset+8000],
                    'characters': len(text), 'next_offset': offset+8000 if offset+8000<len(text) else None}
        if kind == 'read_task':
            text = json.dumps(self.state['tasks'][identifier(a['task'])], ensure_ascii=False)
            offset = max(0, int(a.get('offset', 0)))
            return {'task': a['task'], 'text': text[offset:offset+8000],
                    'characters': len(text), 'next_offset': offset+8000 if offset+8000<len(text) else None}
        if kind == 'inspect_preparation':
            prep = identifier(a['preparation'])
            previous = self.state['actions'][prep]
            if previous['type'] != 'prepare' or previous['status'] != 'completed':
                raise ValueError('Select a completed preparation')
            return inspect_preparation(self.run,self.inv,a)
        if kind == 'inventory':
            context = asset_context(self.inv)
            assets = [(k,v) for k,v in context['assets'].items() if not a.get('source') or v['source']==a['source']]
            offset = max(0,int(a.get('offset',0)))
            return {'assets':dict(assets[offset:offset+40]), 'total_assets':len(assets),
                    'next_offset':offset+40 if offset+40<len(assets) else None}
        if kind == 'inspect': return inspect_assets(self.inv, a['asset_ids'],a.get('offset',0))
        if kind == 'read_r1':
            if self.prior is None: return {'status':'not_found_on_server'}
            n=int(a['round'])
            if not 1<=n<=10000: raise ValueError('Invalid round')
            p=self.prior/'rounds'/f'{n:03d}'
            proposal=read_json(p/'proposal.json'); result=read_json(p/'outcome.json')
            return {'implementation':proposal.get('implementation'),'hypothesis':proposal.get('hypothesis'),
                    'metrics':result.get('metrics'),'execution':result.get('execution'),
                    'caution':'Historical reflection is not machine evidence; information content may differ from baselines'}
        if kind in ('discover_data','probe_download','download'):
            from .research_r2_downloads import discover,probe,acquire
            return {'discover_data':discover,'probe_download':probe,'download':acquire}[kind](self,a)
        if kind == 'read_vdc':
            path = inside(self.repo, a['path'])
            section=path.parts[len(self.repo.parts)]
            if not ((section in ('src','scripts') and path.suffix=='.py') or
                    (section=='knowledge' and path.suffix in ('.json','.jsonl','.md','.txt'))):
                raise ValueError('Only existing public VDC code or knowledge is readable')
            text = path.read_text(encoding='utf-8'); offset = max(0, int(a.get('offset', 0)))
            if a['path']=='knowledge/research_public_data.json':
                text=json.dumps({'studies':asset_context(self.inv)['sources'],
                    'current_access':'all_roles_authorised; historical policies are provenance only'},ensure_ascii=False)
            return {'text': text[offset:offset+14000], 'characters': len(text)}
        if kind == 'search': return search_literature(a['query'], self.run/'literature')
        if kind == 'fetch': return fetch_public(a['url'], self.run/'literature')
        if kind == 'read_public':
            doc = identifier(a['document']); path = self.run/'literature'/(doc+'.txt')
            text = path.read_text(encoding='utf-8'); offset = max(0, int(a.get('offset', 0)))
            from .research_r2_tools import literature_kind
            record=read_json(self.run/'literature'/(doc+'.json'))
            kind=record.get('literature_kind',literature_kind(record['url'],record.get('content_type','')))
            self.state.setdefault('literature_reads',{}).setdefault(doc,[]).append(offset)
            return {'document':doc,'text':text[offset:offset+10000],'characters':len(text),
                'next_offset':offset+10000 if offset+10000<len(text) else None,
                'literature_kind':kind,'untrusted':True,'reproduction_claim':False}
        if kind == 'dependencies':
            import re
            packages = a['packages']
            if not isinstance(packages, list) or not 1 <= len(packages) <= 20 or any(
                not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9_.+!-]+', p) for p in packages):
                raise ValueError('Pin public package versions; no URLs or pip options')
            write_code(out/'code', {'request.json': json.dumps(packages)})
            r = execute(out/'code', {}, out/'wheels', ['python', '-m', 'pip', 'download',
                '--only-binary=:all:', '--dest', '/out', *packages], self.runtime(), network=True)
            dest = self.run/'wheels'; dest.mkdir(exist_ok=True)
            for p in (out/'wheels').iterdir():
                target = dest/p.name
                if target.exists() and sha256(target) != sha256(p): raise ValueError('Dependency cache conflict')
                shutil.copyfile(p, target)
            return {'status': 'downloaded', 'packages': packages, 'wheel_count': len(r['files'])}
        if kind == 'prepare':
            mounts, provenance = resolve_assets(self.inv, a['asset_ids']) if a.get('asset_ids') else ({},{})
            derived, ancestors, references=preparation_inputs(self.run,self.inv,a.get('artifact_inputs',[]))
            if not mounts and not derived: raise ValueError('Select original asset_ids or explicit artifact_inputs')
            original_manifest=dict(provenance)
            mounts.update(derived);provenance.update(ancestors)
            out.mkdir(parents=True, exist_ok=True)
            write_json(out/'assets.json', original_manifest); mounts['assets.json'] = out/'assets.json'
            write_json(out/'artifacts.json',references); mounts['artifacts.json']=out/'artifacts.json'
            write_code(out/'code', a['files'])
            write_json(out/'provenance.json', provenance)
            # Persist attempts before executing, even if parsing fails.
            for p in provenance.values():
                s = self.inv['sources'][p['source']]
                s.setdefault('attempts', []).append(key)
            self.save()
            device=a.get('device','cpu')
            r = execute(out/'code', mounts, out/'output', a['command'], self.runtime(device), repo=self.repo)
            artifacts=artifact_index(self.run,key,self.inv)
            artifacts={**artifacts,'file_count':len(artifacts['files']),'files':dict(list(artifacts['files'].items())[:30])}
            return {'status': 'executed', 'output_files': list(r['files']), 'seconds': r['seconds'],
                    'selected_device':device,'gpu_usage_status':r.get('gpu_usage_status'),
                    'source_ids': sorted({p['source'] for p in provenance.values()}),
                    'numeric_training_claim': False,'artifacts':artifacts}
        if kind == 'register_task':
            prep = identifier(a['preparation']); p = self.run/'actions'/prep
            if self.state['actions'].get(prep, {}).get('status') != 'completed':
                raise ValueError('Preparation has not completed')
            sources = {v['source'] for v in read_json(p/'provenance.json').values()}
            path = inside(p/'output', a['subdir']) if a.get('subdir') else p/'output'
            contract={**a['contract'],'preparation_id':prep,'preparation_subdir':a.get('subdir',''),
                      'source_fingerprints':read_json(p/'provenance.json')}
            if self.config.get('engine_contract_version',1)>=2:
                from .research_r2_contracts import verified_bundle
                path,audit=verified_bundle(path,contract,self.inv,contract['source_fingerprints'],p/'code',out,
                    self.runtime(a.get('device','cpu')),self.repo)
                contract['execution_audit']=audit
            c = seal_task(path, self.task_path(a['task']), contract, sources)
            self.state['tasks'][a['task']] = c
            return {k:c[k] for k in ('signature', 'mission', 'fit_units', 'query_units', 'coverage_denominator')}
        if kind == 'experiment':
            task = self.task_path(a['task']); c = verify_task(task)
            references = read_json(task/'baselines.json')
            comparable = {k:v for k,v in self.state['experiments'].items() if v['task_signature'] == c['signature']}
            if a.get('parent'):
                if a['parent'] not in comparable: raise ValueError('Parent missing or not comparable to this task')
                references['parent'] = comparable[a['parent']]['metrics']
            if comparable:
                references['best'] = min(comparable.values(), key=lambda r:r['metrics'][c['metric']])['metrics']
            evidence = a.get('evidence', [])
            for doc in evidence:
                if not (self.run/'literature'/(identifier(doc)+'.json')).is_file():
                    raise ValueError('Method citation was not actually retrieved')
            write_code(out/'code', a['files'])
            device=a.get('device','cpu')
            result = run_candidate(task, out, out/'code', self.runtime(device), self.repo, references,
                                   a.get('rule', {'reference':'same_input_ridge'}))
            execution=[read_json(p) for p in out.glob('*_execution.json')]
            result.update(task=a['task'], parent=a.get('parent'), evidence=evidence, mission=c['mission'],
                          method_use=a.get('method_use','not_described'),
                          selected_device=device,device_reason=a.get('device_reason','not_provided'),
                          execution_seconds=sum(e['seconds'] for e in execution),
                          gpu_usage_status=('allocation_observed' if any(e.get('gpu_memory_observed_bytes') for e in execution)
                              else 'not_observed_or_unavailable') if device=='cuda' else 'not_requested',
                          gpu_memory_observed_bytes=max((e.get('gpu_memory_observed_bytes') or 0 for e in execution),default=0))
            self.state['experiments'][key] = result
            for sid in c['source_ids']:
                self.inv['sources'][sid]['disposition'] = 'numerically_used'
                self.inv['sources'][sid].setdefault('experiments', []).append(key)
            return public_result(result)
        if kind == 'disposition':
            sid, prep = a['source'], identifier(a['preparation']); source = self.inv['sources'][sid]
            if prep not in source.get('attempts', []): raise ValueError('No actual preparation attempt for this source')
            if not isinstance(a.get('reason'), str) or len(a['reason']) < 15:
                raise ValueError('Give a concrete reason and evidence')
            if source['disposition'] != 'numerically_used': source['disposition'] = 'attempted_not_modelled'
            source['reason'] = a['reason']; source['reason_is_agent_interpretation'] = True
            return {'status': source['disposition'], 'source': sid}
        if kind == 'export': return self.export(a)
        if kind == 'finish':
            missing = self.missing()
            if any(missing.values()): return {'status': 'scope_incomplete_continue', 'missing': missing}
            self.state.update(status='completed_declared_scope', final_summary=a['summary'])
            return {'status': 'completed_declared_scope', 'scientific_validation': 'adaptive_only'}
        raise ValueError('Unknown research tool')

    def export(self, a):
        key, exp = identifier(a['id']), identifier(a['experiment'])
        r = self.state['experiments'][exp]; task = self.task_path(r['task']); c = verify_task(task)
        selected_runtime=self.runtime(r.get('selected_device','cpu'))
        out = self.run/'exports'/key
        if out.exists(): raise ValueError('Export ID already exists')
        out.mkdir(parents=True)
        code = self.run/'actions'/exp/'code'
        shutil.copytree(code, out/'code')
        if c.get('preparation_id'):
            preparation=self.run/'actions'/c['preparation_id']
            shutil.copytree(preparation/'code',out/'preparation_code')
            shutil.copyfile(preparation/'provenance.json',out/'preparation_provenance.json')
        shutil.copyfile(task/'schema.json', out/'schema.json')
        if a.get('refit', True):
            tr = numeric_npz(task/'train.npz', ('x', 'y'))
            q = numeric_npz(task/'query.npz', ('x',)); y = numeric_npz(task/'truth.npz', ('y',))['y']
            staging = self.run/'actions'/key/'refit_inputs'
            save_npz(staging/'train.npz', x=np.concatenate([tr['x'],q['x']]), y=np.concatenate([tr['y'],y]))
            write_json(staging/'rows.json', read_json(task/'train_rows.json')+read_json(task/'query_rows.json'))
            execute(code, {'train.npz': staging/'train.npz', 'rows.json': staging/'rows.json',
                          'schema.json': task/'schema.json'}, out/'model',
                    ['python','/code/fit.py'], selected_runtime, repo=self.repo)
        else:
            shutil.copytree(self.run/'actions'/exp/'model', out/'model')
        # Genuine new subprocess inference on exported artifacts, twice; no training labels.
        paths = []
        for n in ('reload1', 'reload2'):
            p = self.run/'actions'/key/n
            execute(out/'code', {'query.npz': task/'query.npz', 'schema.json': out/'schema.json'}, p,
                    ['python','/code/predict.py'], selected_runtime, model=out/'model', repo=self.repo)
            paths.append(numeric_npz(p/'predictions.npz', ('prediction',))['prediction'])
        if paths[0].shape != tuple(c['coverage_denominator']) or not np.allclose(paths[0],paths[1],rtol=1e-5,atol=1e-6):
            raise ValueError('Export reload/shape check failed')
        # Bundle frozen runtime source and wheels: no dependency on original run paths.
        shutil.copytree(self.repo/'src', out/'runtime/src', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if (self.run/'wheels').exists(): shutil.copytree(self.run/'wheels', out/'wheels')
        runtime = {k:v for k,v in selected_runtime.items() if k != 'wheels'}
        report = {'expert': key, 'task': r['task'], 'task_contract': c,
                  'source_experiment': exp, 'refit_all_task_rows': bool(a.get('refit', True)),
                  'refit_scope': 'all compatible rows of this task, not all project sources',
                  'development_metrics_before_refit': r['metrics'],
                  'independent_confirmation': False, 'runtime': runtime,
                  'input_contract': 'preprocessed 2D x in query.npz matching schema.json; raw preprocessing remains preparation code',
                  'files': tree_hash(out), 'reload_verified': True}
        write_json(out/'model_manifest.json', report)
        self.state['exports'][key] = {'experiment': exp, 'task': r['task'], 'reload_verified': True}
        write_json(self.run/'exports/index.json', self.state['exports'])
        return {'status': 'exported', **self.state['exports'][key], 'expert': key,
                'raw_matrix_inference': False}

    def method_evidence(self):
        from .research_r2_tools import literature_kind
        for exp in self.state['experiments'].values():
            if exp.get('method_use','not_described')=='not_described': continue
            for doc in exp.get('evidence',[]):
                p=self.run/'literature'/(identifier(doc)+'.json')
                if not p.exists() or doc not in self.state.get('literature_reads',{}): continue
                record=read_json(p)
                kind=record.get('literature_kind',literature_kind(record['url'],record.get('content_type','')))
                if kind in ('full_text','source_code_author_association_unverified'): return True
        return False

    def missing(self):
        missions = {e['mission'] for e in self.state['experiments'].values()}
        return {'missions': sorted({'state','response','transfer'}-missions),
                'decisions_before_finish': max(0, self.config.get('min_decisions_before_finish',0)-self.state['next_decision']),
                'sources': [k for k,v in self.inv['sources'].items() if v['disposition']=='not_processed'],
                'export_required': not bool(self.state['exports']),
                'method_evidence_required': not self.method_evidence()}

    def context(self):
        actions = list(self.state['actions'].items())
        # Observation order includes reused results, rather than only insertion order.
        observed = self.state.get('observations', [])[-4:]
        recent = [(k, self.state['actions'][k]) for k in observed] if observed else actions[-4:]
        fetched = {v.get('result', {}).get('url') for _,v in actions
                   if v['type']=='fetch' and v['status']=='completed'}
        # No arbitrary stdout or data files in the context, just host-generated results.
        inv = asset_context(self.inv)
        inv['assets'] = dict(list(inv['assets'].items())[:24])
        inv['asset_listing'] = 'Use inventory with source and offset to page remaining files'
        for source in inv['sources']:
            source['asset_count'] = len(source.get('asset_ids', []))
            source['asset_ids'] = source.get('asset_ids', [])[:8]
            for k in ('ai_brief', 'unit_rule', 'scale_or_assay'):
                if k in source: source[k] = source[k][:250]
        result = {'mission': self.config['mission'], 'scope': 'all_authorised_data_adaptive_research',
            'method_seeds_to_fetch':[u for u in self.config.get('method_seeds',[]) if u not in fetched],
            'progress': {'completed_decisions': self.state['next_decision']-1,
                'completed_preparations': sum(v['type']=='prepare' and v['status']=='completed' for _,v in actions),
                'registered_tasks': len(self.state['tasks']), 'experiments': len(self.state['experiments']),
                'exports': len(self.state['exports']),
                'retrieval': 'list_actions(tool_type, offset), read_action(action, offset), read_task(task, offset), inspect_preparation(preparation, path)',
                'recent_preparations': [{'id':k, 'status':v['status']} for k,v in actions if v['type']=='prepare'][-8:],
                'fetched_documents': [{'action':k, 'document':v.get('result',{}).get('id')}
                    for k,v in actions if v['type']=='fetch' and v['status']=='completed'][-12:]},
            'inventory': inv, 'tasks': {k:{f:v[f] for f in ('question','mission','signature','fit_units','query_units')}
                for k,v in list(self.state['tasks'].items())[-12:]},
            'experiments': {k:public_result(v) for k,v in list(self.state['experiments'].items())[-12:]},
            'recent_actions': dict(recent), 'older_actions': {'count':len(actions),
                'retrieval':'list_actions and read_action; recent tasks only, read_task recovers older contracts'},
            'memory': self.state['memories'][-8:],
            'missing_scope': self.missing(), 'runtime': {k:v for k,v in self.runtime().items() if k != 'wheels'},
            'budget': self.budget()}
        result['public_data_acquisition']={'max_file_bytes':self.config.get('public_download_max_bytes',20_000_000_000),
            'disk_reserve_bytes':self.config.get('public_download_reserve_bytes',10_000_000_000),'new_sources_authorised':True}
        result['engine_progress']=self.state.get('engine_progress',{'stagnant_decisions':0,'diagnostic_mode':False})
        if self.prior is not None:
            prior=read_json(self.prior/'status.json')
            rounds=[r for r in prior.get('rounds',[]) if r.get('relative_mse') is not None]
            result['r1_memory']={'completed_experiments':prior.get('completed_experiments'),
                'best':{k:v for k,v in min(rounds,key=lambda r:r['relative_mse']).items()
                        if k in ('round','hypothesis','metrics','relative_mse')} if rounds else None,
                'limits':'Fixed hidden-gene assay; candidate sees more genes than programme baseline; no public matrix training. Round2 parent-comparison verdict was wrong.',
                'access':'read_r1 tool; retain negative results'}
        # Keep complete JSON, trim only repeated detail; full records remain on disk.
        maximum = self.config['max_request_budget'] - self.config['max_output_tokens'] - len((SYSTEM+'\n'+TOOLS).encode()) - 256
        from .research_continuity import continuity_context
        return continuity_context(self, result, maximum)

    def budget(self):
        records = [read_json(p) for p in (self.run/'api').glob('*.json')]
        return {'requests':len(records), 'request_limit':self.config['max_api_calls'],
                'reserved_token_upper_bound':sum(r['budget_charge'] for r in records),
                'token_limit':self.config['max_token_budget'],
                'reported_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in records),
                'unknown_usage_requests':sum(not r.get('usage') for r in records)}

    def run_loop(self, retry=False, resume_after_stall=False):
        if self.state['status']=='stalled_no_progress':
            if not resume_after_stall: return self.state
            self.state['engine_progress']={'stagnant_decisions':0,'diagnostic_mode':False,'operator_resumed':True}
            self.state['status']='ready'
        client = Client(self.run, self.config, retry_errors=retry)
        # No fixed 12-round completion. Budget and actual mission coverage terminate.
        while self.state['status'] != 'completed_declared_scope':
            if (self.run/'STOP').exists():
                self.state['status']='paused_by_operator'; break
            budget = self.budget()
            if budget['requests'] >= budget['request_limit'] or (budget['token_limit'] is not None and budget['reserved_token_upper_bound'] >= budget['token_limit']):
                self.state['status']='budget_exhausted_partial'; break
            index = self.state['next_decision']; folder = self.run/'decisions'/f'{index:04d}'
            before=progress_signature(self)
            cp = folder/'context.json'
            if not cp.exists(): write_json(cp,self.context())
            try:
                print(f'API_START decision_{index:04d} requests={budget["requests"]}/{budget["request_limit"]}', flush=True)
                proposal = client.call(f'decision_{index:04d}', SYSTEM+'\n'+TOOLS,
                    json.dumps(read_json(cp),ensure_ascii=False), validator=decision)
                print(f'API_COMPLETED decision_{index:04d}', flush=True)
            except Exception as exc:
                if str(exc)=='API call/token upper-bound budget exhausted; no further requests submitted':
                    self.state['status']='budget_exhausted_partial'; break
                # A returned but malformed answer is known billed work, not an
                # uncertain transport retry. Start a NEW corrective decision.
                records=sorted((self.run/'api').glob(f'decision_{index:04d}_*.json'))
                r=read_json(records[-1]) if records else {}
                if (r.get('status')=='failed' and r.get('error_type') in ('ValueError','JSONDecodeError')
                        and r.get('finish_reason') in ('stop','length') and isinstance(r.get('content'),str)):
                    self.state['format_failures']=self.state.get('format_failures',0)+1
                    self.state['memories'].append('Previous answer rejected as incomplete/invalid tool JSON. No actions executed. Emit a smaller complete decision; use at most one substantial code action.')
                    self.state['next_decision']+=1
                    self.state['status']='running' if self.state['format_failures']<3 else 'format_failures_partial'
                    self.save();self.report()
                    if self.state['format_failures']>=3:break
                    continue
                self.state.update(status='api_stopped', error_type=type(exc).__name__)
                self.save(); self.report(); raise
            self.execute_decision(proposal, index)
            if self.state['status']=='stalled_no_progress':
                print('STOPPED_NO_PROGRESS: inspect engine_progress in status.json; no new paid request',flush=True)
                break
        self.save(); self.report(); return self.state

    def execute_decision(self, proposal, index):
        """Single execution path shared by legacy controller and external Codex worker."""
        decision(proposal)
        if index != self.state['next_decision']:
            raise ValueError('Stale decision index; reload context before submitting')
        folder = self.run/'decisions'/f'{index:04d}'
        before = progress_signature(self)
        if not (folder/'context.json').exists():
            write_json(folder/'context.json', self.context())
        write_json(folder/'decision.json', proposal)
        from .research_continuity import update_working_state
        update_working_state(self, proposal)
        self.state['last_batch_actions'] = [a['id'] for a in proposal['actions']]
        self.state['format_failures']=0
        self.state['status']='running'; self.save()
        for action in proposal['actions']:
            key=action['id']; fingerprint=object_hash(action)
            previous=self.state['actions'].get(key)
            if previous:
                if previous['signature'] != fingerprint:
                    self.state['memories'].append('Action ID conflict: '+key+'; use a new ID.')
                elif previous['status']=='running':
                    previous.update(status='interrupted',reason='Outcome uncertain; inspect artifacts and choose new action ID, no automatic retraining')
                self.state.setdefault('observations', []).append(key)
                self.save()
                print('REUSED_RESULT '+key+' '+previous['status'], flush=True)
                continue
            entry={'type':action['type'],'signature':fingerprint,'status':'running'}
            self.state['actions'][key]=entry; self.save()
            print('START '+key+' '+action['type'],flush=True)
            try:
                digest=request_fingerprint(self,action) if action['type'] in READ_TOOLS else None
                if digest:
                    self.state.setdefault('request_observations',[]).append(digest)
                    entry['content_request_fingerprint']=digest
                cache=self.state.setdefault('content_cache',{})
                prior_key=cache.get(digest) if digest else None
                if prior_key and self.state['actions'][prior_key]['status']=='completed':
                    result=self.state['actions'][prior_key]['result']
                    entry['reused_from']=prior_key
                    print('REUSED_CONTENT '+key+' from '+prior_key,flush=True)
                else:
                    result=self.action(action)
                if digest:
                    cache[digest]=key
                entry.update(status='completed',result=result)
            except Exception as exc:
                # Do not forward arbitrary exception strings from private parsers.
                import re
                # Engine errors contain structural facts. Never forward parser stdout.
                diagnostic = str(exc).replace(str(self.private), '<private>').replace(str(self.run), '<run>')[:500]
                secret = os.environ.get('VDC_RESEARCH_API_KEY')
                if secret: diagnostic = diagnostic.replace(secret, '[REDACTED]')
                frames = []; error_classes = []
                for log in (self.run/'actions'/key).glob('*.log'):
                    with log.open('rb') as stream:
                        stream.seek(max(0,log.stat().st_size-64000))
                        text = stream.read().decode('utf-8', errors='replace')
                    frames += re.findall(r'File "(/code/[^"\n]+)", line (\d+)',text)[-4:]
                    error_classes += re.findall(r'^([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception)):',text,re.M)[-2:]
                entry.update(status='failed',error_type=type(exc).__name__,diagnostic=diagnostic,
                             candidate_error_frames=frames,candidate_error_classes=error_classes)
                write_json(self.run/'actions'/key/'error.json',
                    {'error_type':type(exc).__name__, 'detail':str(exc)[:1000], 'local_only':True})
            self.state.setdefault('observations', []).append(key)
            self.save(); print(entry['status'].upper()+' '+key,flush=True)
            if self.state['status']=='completed_declared_scope': break
        self.state['memories'].append(str(proposal.get('memory',''))[:2000])
        self.state['next_decision']+=1
        observe_progress(self,before)
        self.save(); self.report()
        self.state['last_batch_actions'] = [key for key in self.state['last_batch_actions'] if key in self.state['actions']]
        self.save()
        return {key: self.state['actions'][key] for key in self.state['last_batch_actions']}

    def report(self):
        # Incomplete scope never becomes success simply because a batch ended.
        summary={'status':self.state['status'], 'decisions':self.state['next_decision']-1,
            'actions':len(self.state['actions']), 'experiments':len(self.state['experiments']),
            'failed_actions':sum(a['status']=='failed' for a in self.state['actions'].values()),
            'comparable_revisions':sum(bool(e.get('parent')) for e in self.state['experiments'].values()),
            'supported_declared_comparisons':sum(e['comparison']['status']=='supported_on_task' for e in self.state['experiments'].values()),
            'exports':len(self.state['exports']), 'missing':self.missing(), 'budget':self.budget(),
            'universal_model_validated':False, 'independent_confirmation':False}
        summary['engine_progress']=self.state.get('engine_progress',{})
        if self.config.get('controller') == 'local_codex_cli_with_optional_deepseek_advisor':
            summary['controller'] = {'primary': 'local_codex_cli', 'advisor': 'deepseek_when_requested',
                'worker_llm_calls': 0, 'local_llm_usage': 'see local controller turns/ and advisor/ ledgers',
                'external_decisions_completed': self.state['next_decision']-1,
                'no_claim_of_primary_api_usage_from_worker_budget': True}
        summary['artifact_states']={k:v.get('result',{}).get('artifacts',{}).get('status','legacy_not_indexed')
            for k,v in self.state['actions'].items() if v['type']=='prepare' and v['status']=='completed'}
        if (self.run/'continuation.json').exists():
            lineage=read_json(self.run/'continuation.json')
            summary['continuation']={k:lineage[k] for k in ('parent_run','inherited_actions','inherited_decisions','inherited_api_requests')}
            summary['continuation'].update({k:v for k,v in lineage.items() if k in ('counter_mode','historical_reported_tokens','historical_unknown_usage_requests','historical_experiments','new_phase_request_limit','new_phase_token_limit')})
            summary['new_actions']=len(self.state['actions'])-lineage['inherited_actions']
            summary['new_decisions']=summary['decisions']-(0 if lineage.get('counter_mode')=='new_phase' else lineage['inherited_decisions'])
        write_json(self.run/'status.json',summary)
        lines=['# VDC AI4AI R2 — universal dormancy research', '',
            '状态：'+summary['status'], '',
            'Universal 是研究目标，不是已经验证的跨物种能力。所有比较属于开发/回顾性研究。', '',
            '|来源|实际用途|', '|---|---|']
        lines += ['|'+k+'|'+v['disposition']+'|' for k,v in self.inv['sources'].items()]
        lines += ['', '|实验|任务|MSE|机器比较结论|', '|---|---|---|---|']
        lines += [f"|{k}|{v['task']}|{v['metrics']['macro_unit_mse']}|{v['comparison']['status']}|"
                  for k,v in self.state['experiments'].items()]
        lines += ['', '缺失范围：`'+json.dumps(summary['missing'],ensure_ascii=False)+'`', '',
            ('exports/ 包含实际导出的模型。' if summary['exports'] else '尚未导出模型。')+'准备与实验记录保留在本 run。',
            '模型输入默认是任务声明的数值 x；未经实现并核验的原始表达转换不冒称一键 raw 推理。']
        (self.run/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
