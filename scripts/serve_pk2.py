"""Serve a frozen development snapshot on loopback only."""
from pathlib import Path
import argparse,json
from vdc.pk2_service import freeze_service,create_app
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    f=sub.add_parser('freeze');f.add_argument('--pk1-run',type=Path,required=True);f.add_argument('--roles',type=Path,required=True);f.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('serve');s.add_argument('--snapshot',type=Path,required=True);s.add_argument('--port',type=int,default=8765)
    a=p.parse_args()
    if a.cmd=='freeze':
        m=freeze_service(a.pk1_run,a.roles,a.out);print(json.dumps({'snapshot_id':m['snapshot_id'],'models':len(m['models']),'profile':m['profile']}))
    else:
        import torch,uvicorn
        torch.set_num_threads(2)
        uvicorn.run(create_app(a.snapshot),host='127.0.0.1',port=a.port,workers=1)
