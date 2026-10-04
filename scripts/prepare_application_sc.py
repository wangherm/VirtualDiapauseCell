"""Make a small private raw pseudobulk supplement; never train or query a model."""
from pathlib import Path
import argparse
from vdc.application_data import prepare_sc_supplement
from vdc.io import read_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--roles',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-cache',type=Path);p.add_argument('--programmes',type=Path)
    a=p.parse_args();prepare_sc_supplement(a.archive,a.roles,a.out,read_json(Path(__file__).resolve().parents[1]/'configs/application_v1.json'),a.source_cache,a.programmes)
