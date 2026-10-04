#!/usr/bin/env bash
set -euo pipefail
RELEASE="$(cd "$(dirname "$0")/.." && pwd)"
: "${VDC_PYTHON:?Set VDC_PYTHON to the existing CUDA Python}"
: "${1:?Pass the interrupted run directory}"
"$VDC_PYTHON" - "$1" "$RELEASE" <<'PY'
import json,os,subprocess,sys
from pathlib import Path
run=Path(sys.argv[1]).resolve();release=Path(sys.argv[2]).resolve()
c=json.loads((run/'config.json').read_text())
env=os.environ.copy()
for key,var in {'private_root':'VDC_PRIVATE_ROOT','pk1_run':'VDC_PK1_RUN','acquire_run':'VDC_ACQUIRE_RUN',
                'public_raw':'VDC_PK2_RAW','model_path':'VDC_QWEN_MODEL'}.items():
    env[var]=c[key]
env.update(VDC_PK2_RUN=str(run),VDC_RESUME='1')
if c.get('repair_from'):
    env['VDC_PK2_REPAIR_FROM']=c['repair_from'];env['VDC_PK2_SERVICE_PORT']=str(c['service_port'])
else:env.pop('VDC_PK2_REPAIR_FROM',None)
subprocess.run(['bash',str(release/'scripts/launch_pk2_training.sh')],env=env,check=True)
PY
