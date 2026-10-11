"""Run/resume the first actual API-driven VDC research loop."""
import argparse
import json
import os
from pathlib import Path
import sys
from vdc.io import read_json, write_json
from vdc.research import run_research, evaluate_candidate
from vdc.research_api import endpoint
from run_all_modules import exclusive_run

ROOT=Path(__file__).resolve().parents[1]


def locate_data(private, explicit=None):
    if explicit:
        data=Path(explicit).resolve()
    else:
        pointer=private/'LATEST_STAGE.txt'
        if not pointer.is_file(): raise ValueError('Set VDC_RESEARCH_DATA to the verified CW-stage data directory')
        stage=Path(pointer.read_text(encoding='utf-8').strip())
        if not (stage/'stage_snapshot.json').is_file(): raise ValueError('LATEST_STAGE lacks stage_snapshot.json')
        data=stage/'data'
    for view in ('bulk','core','coarse'):
        if not (data/view/'data.json').is_file() or not (data/view/'counts.npz').is_file():
            raise ValueError('Missing real cached input: '+str(data/view))
    return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['doctor','run','prepare','evaluate'])
    p.add_argument('--private-root',type=Path,default=Path(os.environ.get('VDC_PRIVATE_ROOT','/root/autodl-tmp/vdc-private')))
    p.add_argument('--data',default=os.environ.get('VDC_RESEARCH_DATA'))
    p.add_argument('--run',type=Path); p.add_argument('--out',type=Path)
    p.add_argument('--config',type=Path,default=os.environ.get('VDC_RESEARCH_CONFIG',ROOT/'configs/research_round1.json'))
    p.add_argument('--corpus',type=Path,default=ROOT/'knowledge/pk2/literature.jsonl')
    p.add_argument('--retry-api-errors',action='store_true')
    a=p.parse_args()
    if a.command=='evaluate':
        if a.run is None or a.out is None: p.error('evaluate needs --run and --out')
        evaluate_candidate(read_json(a.out/'implementation.json'),a.run,a.out,read_json(a.run/'config.json'))
        return 0
    data=locate_data(a.private_root,a.data)
    os.environ['VDC_ROLE_MANIFEST']=str((a.private_root/'sample_roles.json').resolve())
    if a.command=='doctor':
        import torch
        from vdc.research_data import source_fingerprint
        from vdc.research_assets import load_public_context
        fingerprint=source_fingerprint(data,read_json(a.config))
        public_context=load_public_context(ROOT,read_json(a.config),os.environ.get('VDC_RESEARCH_ASSET_REPORT'))
        print(json.dumps({'data':str(data),'data_fingerprint':fingerprint,'torch':torch.__version__,
            'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'api_key_set':bool(os.environ.get('VDC_RESEARCH_API_KEY')),'model_set':bool(os.environ.get('VDC_RESEARCH_MODEL')),
            'public_assets':public_context['acquisition'] if public_context else 'not_attached',
            'public_matrices_used_in_current_assay':False,
            'api_calls_executed':0},indent=2));return 0
    if a.run is None: p.error('run/prepare needs explicit --run')
    if a.command=='run': endpoint()  # Fail before fitting if backend credentials are absent.
    a.run=a.run.resolve();a.run.mkdir(parents=True,exist_ok=True)
    with exclusive_run(a.run):
        (a.run/'error.json').unlink(missing_ok=True)
        try:
            result=run_research(ROOT,data,a.run,read_json(a.config),a.corpus,a.retry_api_errors,a.command=='prepare')
        except Exception as exc:
            # API errors are already sanitized. Do not dump environment or request headers.
            write_json(a.run/'error.json',{'type':type(exc).__name__,'message':str(exc)[:700]})
            current=read_json(a.run/'status.json') if (a.run/'status.json').exists() else {}
            current.update(status='interrupted_or_failed',error_type=type(exc).__name__)
            write_json(a.run/'status.json',current)
            print('FAILED:',type(exc).__name__,str(exc));return 1
    print(json.dumps({'run':str(a.run),'status':result['status'],'completed_experiments':result['completed_experiments']},indent=2))
    return 0 if result['status']=='prepared_no_api_calls' or result['completed_experiments'] else 2


if __name__=='__main__':sys.exit(main())
