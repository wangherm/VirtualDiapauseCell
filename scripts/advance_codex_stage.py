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
from vdc.research_codex import transport


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
        snap=transport(c,{'operation':'snapshot'})
        if snap['next_decision']!=a.expected_next or snap['status'] not in ('completed_declared_scope','running'):
            raise ValueError('Remote phase or decision differs')
        # Command arguments are individually shell quoted. No credential contents read.
        common=['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes']
        if c.get('identity_file'):common+=['-i',str(Path(c['identity_file']).expanduser())]
        if c.get('known_hosts_file'):
            known=str(c['known_hosts_file']).replace('\\','/').replace('"','\\"')
            common+=['-o','UserKnownHostsFile="'+known+'"']
        ssh=['ssh',*common,'-p',str(c['ssh_port']),c['ssh_target']]
        def remote(args):
            r=subprocess.run(ssh+[shlex.join(args)],text=True,encoding='utf-8',capture_output=True,timeout=90)
            if r.returncode:raise RuntimeError('Stage command failed: '+r.stderr[-3000:])
            return r.stdout
        staging=c['run']+'/operator_stage_inputs/'+str(a.expected_next)
        remote(['mkdir','-p',staging])
        for src,name in ((Path(__file__).with_name('reopen_codex_stage.py'),'reopen_codex_stage.py'),(a.brief,'brief.md')):
            subprocess.run(['scp',*common,'-P',str(c['ssh_port']),str(src),c['ssh_target']+':'+shlex.quote(staging+'/'+name)],check=True,timeout=90)
        private=str(Path(c['run']).parent.parent).replace('\\','/')
        args=[c['python'],staging+'/reopen_codex_stage.py','--run',c['run'],
            '--private-root',private,'--brief',staging+'/brief.md','--expected-next',str(a.expected_next)]
        if a.apply:args+=['--apply']
        report=json.loads(remote(args))
        report['research_api_calls_executed']=0
        if a.apply:
            # Do not launch a duplicate worker on a repeated initial-stage command.
            raw=remote([c['python'],'-c',
                'import json,pathlib; p=pathlib.Path('+repr(c['run']+'/worker_status.json')+'); print(p.read_text() if p.exists() else "{}")'])
            heartbeat=json.loads(raw)
            alive=heartbeat.get('status')=='waiting_or_executing' and time.time()-heartbeat.get('updated_at',0)<30
            if not alive:
                screen='vdc_codex_stage_'+str(a.expected_next)+'_'+str(int(time.time()))
                remote(['screen','-dmS',screen,'bash',c['run']+'/worker.sh',c['python'],c['run'],private])
                report['screen']=screen
            for _ in range(10):
                raw=remote([c['python'],'-c',
                    'import pathlib; print(pathlib.Path('+repr(c['run']+'/worker_status.json')+').read_text())'])
                heartbeat=json.loads(raw)
                if heartbeat.get('status')=='waiting_or_executing' and time.time()-heartbeat.get('updated_at',0)<30:break
                time.sleep(0.5)
            else:raise RuntimeError('Phase reopened but worker heartbeat not ready; inspect console, do not rewind')
            report['worker_ready']=True
            report['local_next_turn_unchanged']=state['next_turn']
            report['brief_sha256']=hashlib.sha256(a.brief.read_bytes()).hexdigest()
            write_json(a.workspace/'stage_reviews'/f'before_{a.expected_next:06d}.json',report)
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
