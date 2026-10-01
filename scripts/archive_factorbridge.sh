#!/usr/bin/env bash
# Non-destructive local server archive. No remote connection and no deletion.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/archive_factorbridge.sh SOURCE ARCHIVE_ROOT [--execute]' >&2
  exit 2
fi
SOURCE=$(realpath -e -- "$1")
DEST=$(realpath -m -- "$2")
case "$(basename -- "$SOURCE")" in
  FactorBridge|FactorBridge-upload) ;;
  *) echo 'Expected the explicitly named FactorBridge or FactorBridge-upload directory' >&2; exit 2;;
esac
[[ -d "$SOURCE" && -f "$SOURCE/pyproject.toml" && -d "$SOURCE/factorbridge" ]] || { echo 'Not a FactorBridge checkout' >&2; exit 2; }
[[ "$DEST" != "$SOURCE" && "$DEST" != "$SOURCE/"* ]] || { echo 'Archive destination is inside source' >&2; exit 2; }
[[ ${3:-} == '' || ${3:-} == '--execute' ]] || { echo 'Unknown option' >&2; exit 2; }
BYTES=$(du -sb --exclude=.venv --exclude=__pycache__ -- "$SOURCE" | cut -f1)
PARENT="$DEST"
while [[ ! -d "$PARENT" ]]; do PARENT=$(dirname -- "$PARENT"); done
FREE=$(df -PB1 -- "$PARENT" | awk 'NR==2 {print $4}')
printf 'SOURCE=%s\nARCHIVE_ROOT=%s\nESTIMATED_SOURCE_BYTES=%s\nFREE_BYTES=%s\n' "$SOURCE" "$DEST" "$BYTES" "$FREE"
[[ "$FREE" -gt "$((BYTES + 1073741824))" ]] || { echo 'Insufficient room for an uncompressed copy plus margin' >&2; exit 1; }
if [[ ${3:-} != '--execute' ]]; then
  echo 'PLAN ONLY. Stop old jobs, then add --execute to create a copy. Source will remain unchanged.'
  exit 0
fi
# Best-effort detection of workers with cwd under the old checkout.
python - "$SOURCE" <<'PY'
import os, pathlib, sys
root=pathlib.Path(sys.argv[1]); active=[]
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid(): continue
    try:
        cwd=(p/'cwd').resolve(strict=True)
        parts=(p/'cmdline').read_bytes().split(b'\0')
        name=pathlib.Path(os.fsdecode(parts[0])).name if parts else ''
        if cwd.is_relative_to(root) and (name.startswith(('python','wget','curl','tar')) or (name in ('bash','sh') and len([x for x in parts if x])>1)):
            active.append(p.name)
    except (FileNotFoundError, PermissionError, ProcessLookupError): pass
if active: raise SystemExit('Stop/check active source-directory worker PIDs first: '+','.join(active))
PY
mkdir -p -- "$DEST"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BASE="FactorBridge_${STAMP}_$$"
PARTIAL="$DEST/$BASE.tar.partial"
[[ ! -e "$PARTIAL" && ! -e "$DEST/$BASE.tar" ]] || { echo 'Archive already exists' >&2; exit 1; }
tar --exclude='./.venv' --exclude='*/__pycache__' --exclude='*.pyc' -cf "$PARTIAL" -C "$SOURCE" .
tar -tf "$PARTIAL" > "$DEST/$BASE.contents.txt"
mv -- "$PARTIAL" "$DEST/$BASE.tar"
(cd "$DEST" && sha256sum "$BASE.tar" > "$BASE.sha256" && sha256sum -c "$BASE.sha256")
printf 'ARCHIVE_CREATED=%s\nSOURCE_UNCHANGED=%s\n' "$DEST/$BASE.tar" "$SOURCE"
