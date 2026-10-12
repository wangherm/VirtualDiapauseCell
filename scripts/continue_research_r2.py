"""Prepare a new frozen release from a stopped preparation-only R2 run. No API calls."""
import argparse
import json
import os
import shutil
from pathlib import Path
from vdc.io import read_json
from vdc.research_r2_continue import continue_preparation_run, continue_engine_phase
from vdc.research_r2_devices import configure_devices
from run_all_modules import exclusive_run

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-phase',action='store_true')
    p.add_argument('--parent',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--private-root',type=Path,default=os.environ.get('VDC_PRIVATE_ROOT','/root/autodl-tmp/vdc-private'))
    a=p.parse_args()
    config=read_json(ROOT/'configs/research_r2.json')
    config['runtime']['runtime_root']=os.environ.get('VDC_R2_RUNTIME_ROOT',str(a.private_root/'r2_runtime_devices_v2'))
    config['runtime']=configure_devices(config['runtime'],os.environ.get('VDC_R2_DEVICE_POLICY','cpu'))
    # The same lock as the runner prevents copying live preparations.
    with exclusive_run(a.parent):
        engine=(continue_engine_phase if a.engine_phase else continue_preparation_run)(ROOT,a.private_root,a.parent,a.out,config)
    release=a.out/'release';release.mkdir()
    for name in ('src','scripts','configs','knowledge'):
        shutil.copytree(ROOT/name,release/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(ROOT/'configs/research_r2.json',release/'configs/research_r2_selected.json')
    if (ROOT/'RELEASE_COMMIT.txt').is_file():
        shutil.copyfile(ROOT/'RELEASE_COMMIT.txt',release/'RELEASE_COMMIT.txt')
    print(json.dumps({'run':str(a.out),'status':'ready_not_started','budget':engine.budget(),
        'parent_modified':False,'api_calls_executed':0}))


if __name__=='__main__':main()
