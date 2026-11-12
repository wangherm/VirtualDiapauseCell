from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'CODE_PROVENANCE.json').read_text(encoding='utf-8'))
for item in manifest:
    p=root/item['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=item['path'])
print('Code hashes and Python syntax verified; no scientific code executed.')
