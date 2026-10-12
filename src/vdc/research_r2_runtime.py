"""Isolated multi-file execution, task sealing and deterministic R2 comparisons."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
import uuid
import numpy as np
from .io import read_json, write_json, object_hash, save_npz
from .metrics import study_unit_weights
from .research import score
from .research_r2_tools import inside, tree_hash, identifier


def docker_image(image):
    r = subprocess.run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                       capture_output=True, text=True, timeout=30)
    value = r.stdout.strip()
    if r.returncode or not value.startswith('sha256:'):
        raise RuntimeError('Isolated Docker runtime/image unavailable; run setup_research_r2.sh. No host-code fallback.')
    return value


def resolve_runtime(config):
    if config.get('backend') == 'linux_landlock_chroot':
        from .research_r2_linux import runtime_info
        return runtime_info(config)['runtime_id']
    return docker_image(config['image'])


def execute(code, inputs, out, command, config, model=None, repo=None, network=False):
    """Network is only for dependency staging WITHOUT private inputs or model mounts."""
    if (not isinstance(command, list) or not command or
            any(not isinstance(v, str) or '\x00' in v for v in command)):
        raise ValueError('Command must be an argv list')
    if network and (inputs or model or repo):
        raise ValueError('Dependency execution cannot mount research data')
    if config.get('backend') == 'linux_landlock_chroot':
        from .research_r2_linux import execute_linux
        return execute_linux(code,inputs,out,command,config,model,repo,network)
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True); out.chmod(0o777)
    code = Path(code).resolve(); tree_hash(code)
    name = 'vdc-r2-' + uuid.uuid4().hex[:16]
    args = ['docker', 'run', '--rm', '--name', name, '--read-only', '--cap-drop=ALL',
        '--security-opt=no-new-privileges', '--user', '65534:65534', '--pids-limit', '256',
        '--memory', config.get('memory', '24g'), '--cpus', str(config.get('cpus', 4)),
        '--network', 'bridge' if network else 'none',
        '--tmpfs', '/tmp:rw,nosuid,size=2g', '--workdir', '/code',
        '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=2', '-e', 'OPENBLAS_NUM_THREADS=2',
        '-e', 'HOME=/tmp', '-v', str(code)+':/code:ro', '-v', str(out)+':/out:rw']
    for name_in, path in inputs.items():
        if '/' in name_in or ':' in name_in or '\\' in name_in or name_in in ('.', '..'):
            raise ValueError('Input mount name must be a basename')
        path = Path(path).resolve()
        args += ['-v', str(path)+':/input/'+name_in+':ro']
    if model:
        tree_hash(model); args += ['-v', str(Path(model).resolve())+':/model:ro']
    if repo:
        public_source=out.parent/(out.name+'_vdc_source')
        for allowed in ('src','scripts','configs','knowledge'):
            selected=Path(repo)/allowed
            if selected.exists():shutil.copytree(selected,public_source/allowed,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        args += ['-v', str(public_source.resolve())+':/vdc:ro', '-e', 'PYTHONPATH=/code:/vdc/src']
    if config.get('wheels'):
        args += ['-v', str(Path(config['wheels']).resolve())+':/wheels:ro']
    if config.get('gpu') and not network:
        args += ['--gpus', 'all']
    args += [config['image_id'], *command]
    log = out.parent/(out.name+'.log')
    start = time.monotonic(); reason = None
    try:
        with log.open('wb') as stream:
            p = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT)
            while p.poll() is None:
                time.sleep(.5)
                if time.monotonic()-start > config.get('timeout_seconds', 1800):
                    reason = 'timeout'; break
                if log.stat().st_size > 20_000_000:
                    reason = 'log_limit'; break
                used = sum(f.stat().st_size for f in out.rglob('*') if f.is_file() and not f.is_symlink())
                if used > config.get('max_output_bytes', 20_000_000_000):
                    reason = 'output_disk_limit'; break
            if reason:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=30)
                p.wait(timeout=30)
            code_rc = p.returncode
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=30)
    record = {'exit_code': code_rc, 'stop_reason': reason, 'seconds': round(time.monotonic()-start, 3),
              'image_id': config['image_id'], 'gpu_requested': bool(config.get('gpu')),
              'network': network, 'files': tree_hash(out)}
    write_json(out.parent/(out.name+'_execution.json'), record)
    if reason or code_rc:
        # Log stays local: candidate stdout may contain private expression.
        raise RuntimeError('Candidate failed: '+str(reason or code_rc)+'; local log retained, not sent to API')
    return record


def numeric_npz(path, required, max_bytes=512_000_000):
    import zipfile
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > max_bytes:
            raise ValueError('Array archive exceeds configured evaluator limit')
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in required}
    for a in arrays.values():
        if a.dtype.kind not in 'fiu' or a.ndim != 2 or not a.size or not np.isfinite(a).all():
            raise ValueError('Expected nonempty finite numeric 2D arrays')
    return arrays


def audit_groups(train_rows, query_rows):
    """Connected units, shared controls and lineage links cannot cross a task split."""
    parents = {}
    def root(k):
        parents.setdefault(k, k)
        if parents[k] != k: parents[k] = root(parents[k])
        return parents[k]
    def links(r):
        s, u = str(r['study_family']), str(r['biological_unit'])
        if not s or not u: raise ValueError('Missing biological units')
        return [s+':unit:'+u]+['link:'+str(k) for k in r.get('link_ids', [])]
    for r in train_rows+query_rows:
        ks = links(r)
        for k in ks[1:]: parents[root(k)] = root(ks[0])
    seen = {}
    result = []
    for name, rows in (('fit', train_rows), ('query', query_rows)):
        group = []
        for r in rows:
            key = root(links(r)[0])
            if seen.setdefault(key, name) != name:
                raise ValueError('Linked biological material crosses task fit/query split')
            if r.get('origin') == 'synthetic':
                raise ValueError('Synthetic sources cannot be admitted as real R2 research')
            group.append({**r, 'biological_unit': key})
        result.append(group)
    return result


def seal_task(prepared, task, contract, source_ids):
    """Prepared files become an immutable comparison; preprocessing remains auditable."""
    required = ('question', 'mission', 'target_definition', 'target_scale', 'allowed_inputs',
                'unit_definition', 'preprocessing', 'split_rationale', 'source_ids')
    if any(not contract.get(k) for k in required):
        raise ValueError('Incomplete task contract: '+', '.join(required))
    if contract['mission'] not in ('state', 'response', 'transfer'):
        raise ValueError('Task mission must be state, response or transfer')
    metric=contract.get('metric','macro_unit_mse')
    if metric not in ('macro_unit_mse','macro_unit_mae'):
        raise ValueError('Numerical leaderboards support macro_unit_mse/mae; other analyses remain analysis artifacts')
    if not set(contract['source_ids']) <= set(source_ids):
        raise ValueError('Task names sources not actually mounted during preparation')
    prepared, task = Path(prepared), Path(task)
    if task.exists(): raise ValueError('Task ID already sealed; choose a new version')
    tr = numeric_npz(prepared/'train.npz', ('x', 'y'))
    query = numeric_npz(prepared/'query.npz', ('x',))
    truth = numeric_npz(prepared/'truth.npz', ('y',))
    if tr['x'].shape[0] != tr['y'].shape[0] or query['x'].shape[0] != truth['y'].shape[0]:
        raise ValueError('Input/target row counts differ')
    if tr['x'].shape[1] != query['x'].shape[1] or tr['y'].shape[1] != truth['y'].shape[1]:
        raise ValueError('Fit/query feature or target schema differs')
    rows = audit_groups(read_json(prepared/'train_rows.json'), read_json(prepared/'query_rows.json'))
    if len(rows[0]) != len(tr['x']) or len(rows[1]) != len(query['x']):
        raise ValueError('Row metadata differs from arrays')
    schema = read_json(prepared/'schema.json')
    if len(schema['features']) != tr['x'].shape[1] or len(schema['targets']) != tr['y'].shape[1]:
        raise ValueError('Schema must name every input feature and target')
    task.mkdir(parents=True)
    # Re-serialize exact approved fields: no hidden query targets in extra NPZ keys.
    save_npz(task/'train.npz', **tr)
    save_npz(task/'query.npz', **query)
    save_npz(task/'truth.npz', **truth)
    write_json(task/'train_rows.json', rows[0]); write_json(task/'query_rows.json', rows[1])
    write_json(task/'schema.json', schema)
    record = {**contract, 'evaluation_scope': 'adaptive_development_or_retrospective',
        'metric': metric, 'coverage_denominator': list(truth['y'].shape),
        'fit_units': len({r['biological_unit'] for r in rows[0]}),
        'query_units': len({r['biological_unit'] for r in rows[1]}),
        'files': tree_hash(task)}
    record['signature'] = object_hash(record)
    write_json(task/'contract.json', record)
    baseline_task(task)
    return record


def verify_task(task):
    task = Path(task); c = read_json(task/'contract.json')
    actual = tree_hash(task)
    if any(actual.get(n) != h for n, h in c['files'].items()):
        raise ValueError('Sealed task inputs/targets changed')
    expected = c['signature']
    if object_hash({k:v for k,v in c.items() if k != 'signature'}) != expected:
        raise ValueError('Sealed comparison contract changed')
    return c


def baseline_task(task):
    tr = numeric_npz(task/'train.npz', ('x', 'y')); q = numeric_npz(task/'query.npz', ('x',))['x']
    y = numeric_npz(task/'truth.npz', ('y',))['y']
    train_rows, rows = read_json(task/'train_rows.json'), read_json(task/'query_rows.json')
    mean = np.average(tr['y'], axis=0, weights=study_unit_weights(train_rows))
    from .pk2_numeric import dual_ridge
    if len(tr['x']) <= 4096:
        model = dual_ridge(tr['x'], tr['y'], train_rows, alpha=1.0)
        save_npz(task/'baseline_model/parameters.npz', **model.parameters)
        ridge = model(q)
    else:
        # Same weighted, standardized objective; iterative solve avoids an n*n matrix.
        from scipy.sparse.linalg import lsqr
        w=study_unit_weights(train_rows); w=w/w.mean()
        mean_x=np.average(tr['x'],axis=0,weights=w)
        scale=np.maximum(np.sqrt(np.average((tr['x']-mean_x)**2,axis=0,weights=w)),.001)
        z=(tr['x']-mean_x)/scale; sw=np.sqrt(w[:,None]); a=z*sw
        coef=np.column_stack([lsqr(a,(tr['y'][:,i]-mean[i])*sw[:,0],damp=1.0,
                                  atol=1e-7,btol=1e-7,iter_lim=2000)[0] for i in range(tr['y'].shape[1])])
        ridge=((q-mean_x)/scale)@coef+mean
        save_npz(task/'baseline_model/parameters.npz', coefficients=coef,mean=mean_x,scale=scale,target_mean=mean)
    predictions = {'train_mean': np.tile(mean, (len(q), 1)), 'same_input_ridge': ridge}
    records = {}
    for name, pred in predictions.items():
        save_npz(task/'baselines'/name/'predictions.npz', prediction=pred, target=y)
        records[name] = score(pred, y, rows)
    write_json(task/'baselines.json', records)


def compare(metrics, references, rule):
    """Never let an LLM substitute a weaker reference for its declared parent."""
    reference = rule.get('reference', 'same_input_ridge')
    if reference not in references:
        return {'status': 'not_evaluable', 'reason': 'Declared comparator unavailable', 'reference': reference}
    metric=rule.get('metric','macro_unit_mse')
    current, old = metrics[metric], references[reference][metric]
    threshold = float(rule.get('minimum_relative_improvement', 0.0))
    if not 0 <= threshold <= 1: raise ValueError('Invalid improvement threshold')
    improvement = (old-current)/old if old > 0 else (0.0 if current == 0 else None)
    passed = improvement is not None and improvement > threshold
    return {'status': 'supported_on_task' if passed else 'not_supported_on_task',
            'reference': reference, 'relative_improvement': improvement,
            'metric':metric,
            'minimum_relative_improvement': threshold, 'independent_confirmation': False}


def run_candidate(task, out, code, config, repo, references, rule):
    task, out = Path(task), Path(out); contract = verify_task(task)
    if 'metric' in rule and rule['metric']!=contract['metric']:
        raise ValueError('Comparison metric differs from sealed task')
    rule={**rule,'metric':contract['metric']}
    write_json(out/'predeclared_comparison.json', {'rule': rule, 'references': references,
               'task_signature': contract['signature']})
    execute(code, {'train.npz': task/'train.npz', 'schema.json': task/'schema.json',
                  'rows.json': task/'train_rows.json'}, out/'model',
            ['python', '/code/fit.py'], config, repo=repo)
    before = tree_hash(out/'model')
    predictions = []
    for n in ('predict1', 'predict2'):
        execute(code, {'query.npz': task/'query.npz', 'schema.json': task/'schema.json'}, out/n,
                ['python', '/code/predict.py'], config, model=out/'model', repo=repo)
        predictions.append(numeric_npz(out/n/'predictions.npz', ('prediction',))['prediction'])
    if tree_hash(out/'model') != before: raise ValueError('Predict modified fitted model')
    if not np.allclose(predictions[0], predictions[1], rtol=1e-5, atol=1e-6):
        raise ValueError('Fresh-process predictions differ after reload')
    verify_task(task)
    y = numeric_npz(task/'truth.npz', ('y',))['y']
    metrics = score(predictions[0], y, read_json(task/'query_rows.json'))
    save_npz(out/'predictions.npz', prediction=predictions[0], target=y)
    result = {'metrics': metrics, 'comparison': compare(metrics, references, rule),
              'task_signature': contract['signature'], 'model_files': before,
              'code_files': tree_hash(code), 'reload_verified': True}
    write_json(out/'result.json', result)
    return result
