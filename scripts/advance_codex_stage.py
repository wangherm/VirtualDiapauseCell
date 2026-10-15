"""Prepare/reopen an existing reviewed worker; never call a research LLM."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vdc.io import read_json, write_json


class StageTransportError(RuntimeError):
    pass


def run_stage_process(argv, label, retry_safe=False, timeout=90, input_text=None):
    """Retry only reads/idempotent staging, never an uncertain phase mutation."""
    attempts=3 if retry_safe else 1
    for attempt in range(1,attempts+1):
        print(f'STAGE_IO {label} attempt={attempt}/{attempts}',flush=True)
        try:
            kwargs={'input':input_text} if input_text is not None else {'stdin':subprocess.DEVNULL}
            result=subprocess.run(argv,text=True,encoding='utf-8',capture_output=True,
                                  timeout=timeout,**kwargs)
        except subprocess.TimeoutExpired:
            failure='SSH command timed out'
        else:
            if result.returncode==0:return result
            if result.returncode!=255:
                raise StageTransportError(f'{label}: remote command failed (exit {result.returncode}); '+result.stderr[-2000:])
            failure='SSH transport failed (exit 255): '+result.stderr[-1000:]
        if attempt<attempts:
            print(f'STAGE_RETRY {label}: safe staging/read only',flush=True)
            time.sleep(attempt*2)
        else:
            suffix=('Safe staging/read failed; phase mutation was not retried.' if retry_safe else
                    'Remote outcome may be uncertain; inspect saved phase receipt before restarting. No automatic retry.')
            raise StageTransportError(f'{label}: {failure}. {suffix}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--brief',type=Path,required=True)
    p.add_argument('--expected-next',type=int,required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    from vdc.research_bridge import lock
    with lock(a.workspace,'.controller.lock'):
        state=read_json(a.workspace/'controller_state.json')
        c=read_json(a.workspace/'connection.local.json')
        if c.get('transport')!='ssh':raise ValueError('This launcher requires the existing SSH transport')
        if state['next_turn']!=a.expected_next or state.get('pending') is not None:
            raise ValueError('Local next turn differs or a decision is pending; do not rewind')
        if (a.workspace/'STOP').exists():raise ValueError('Local STOP is present')
        turn=a.workspace/'turns'/f'{a.expected_next:06d}'
        if any((turn/name).exists() for name in ('codex_receipt.json','request.json','response.json')):
            raise ValueError('Next local decision has evidence; inspect before migration')
        # Command arguments are individually shell quoted. No credential contents read.
        target=c['ssh_target']
        if not target or target.startswith('-') or any(x.isspace() for x in target):
            raise ValueError('Expected one SSH host alias or user@hostname')
        common=['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
                '-o','ServerAliveInterval=5','-o','ServerAliveCountMax=3']
        if c.get('identity_file'):common+=['-i',str(Path(c['identity_file']).expanduser())]
        if c.get('known_hosts_file'):
            known=str(c['known_hosts_file']).replace('\\','/').replace('"','\\"')
            common+=['-o','UserKnownHostsFile="'+known+'"']
        ssh=['ssh',*common,'-p',str(c['ssh_port']),c['ssh_target']]
        def remote(args,label,retry_safe=False,input_text=None):
            return run_stage_process(ssh+[shlex.join(args)],label,retry_safe,input_text=input_text).stdout
        snap=json.loads(remote([c['python'],c['worker_script'],'rpc','--run',c['run']],
            'read_worker_snapshot',True,json.dumps({'operation':'snapshot'})))
        if snap['next_decision']!=a.expected_next or snap['status'] not in ('completed_declared_scope','running'):
            raise ValueError('Remote phase or decision differs')
        staging=c['run']+'/operator_stage_inputs/'+str(a.expected_next)
        remote(['mkdir','-p',staging],'create_staging_directory',True)
        for src,name in ((Path(__file__).with_name('reopen_codex_stage.py'),'reopen_codex_stage.py'),(a.brief,'brief.md')):
            run_stage_process(['scp',*common,'-P',str(c['ssh_port']),str(src),c['ssh_target']+':'+shlex.quote(staging+'/'+name)],'upload_'+name,True)
        private=str(Path(c['run']).parent.parent).replace('\\','/')
        args=[c['python'],staging+'/reopen_codex_stage.py','--run',c['run'],
            '--private-root',private,'--brief',staging+'/brief.md','--expected-next',str(a.expected_next)]
        if a.apply:args+=['--apply']
        report=json.loads(remote(args,'apply_phase' if a.apply else 'validate_phase'))
        report['research_api_calls_executed']=0
        if a.apply:
            # Do not launch a duplicate worker on a repeated initial-stage command.
            raw=remote([c['python'],'-c',
                'import json,pathlib; p=pathlib.Path('+repr(c['run']+'/worker_status.json')+'); print(p.read_text() if p.exists() else "{}")'],'read_worker_heartbeat',True)
            heartbeat=json.loads(raw)
            alive=heartbeat.get('status')=='waiting_or_executing' and time.time()-heartbeat.get('updated_at',0)<30
            if not alive:
                screen='vdc_codex_stage_'+str(a.expected_next)+'_'+str(int(time.time()))
                remote(['screen','-dmS',screen,'bash',c['run']+'/worker.sh',c['python'],c['run'],private],'start_worker')
                report['screen']=screen
            for _ in range(10):
                raw=remote([c['python'],'-c',
                    'import pathlib; print(pathlib.Path('+repr(c['run']+'/worker_status.json')+').read_text())'],'check_worker_ready',True)
                heartbeat=json.loads(raw)
                if heartbeat.get('status')=='waiting_or_executing' and time.time()-heartbeat.get('updated_at',0)<30:break
                time.sleep(0.5)
            else:raise RuntimeError('Phase reopened but worker heartbeat not ready; inspect console, do not rewind')
            report['worker_ready']=True
            report['local_next_turn_unchanged']=state['next_turn']
            report['brief_sha256']=hashlib.sha256(a.brief.read_bytes()).hexdigest()
            write_json(a.workspace/'stage_reviews'/f'before_{a.expected_next:06d}.json',report)
        print(json.dumps(report,indent=2))


if __name__=='__main__':
    try:main()
    except StageTransportError as exc:
        print(json.dumps({'status':'transport_failed','diagnostic':str(exc),'controller_started':False}),file=sys.stderr)
        raise SystemExit(1)
