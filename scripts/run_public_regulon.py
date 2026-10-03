"""Fit a real external regulon branch using existing public Alpha development data."""
from pathlib import Path
import argparse
from prepare_public_pilot import ROOT,fetch_locked
from vdc.io import read_json
from vdc.regulon import public_regulon_task

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--download',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Choose a new regulon output directory')
    network=fetch_locked(read_json(ROOT/'configs/pk1_regulon_source.json'),ROOT/'data/raw/pk1_public',a.download)
    gaf=fetch_locked(read_json(ROOT/'configs/public_pilot_sources.json')['files']['annotations'],ROOT/'data/raw/GSE288723',a.download)
    print(public_regulon_task(a.public_run,network,gaf,a.out))
