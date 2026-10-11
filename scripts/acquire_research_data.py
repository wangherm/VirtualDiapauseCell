"""Complete the pinned public asset set; reuse caches and never download private data."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_public_pilot import fetch_locked
from run_all_modules import exclusive_run, utc
from vdc.io import read_json, write_json, object_hash
from vdc.research_assets import validate_asset_manifest, asset_key, verified_file, cache_index

ROOT = Path(__file__).resolve().parents[1]


def acquire(manifest, raw, out, caches, download=False, workers=2):
    entries = validate_asset_manifest(manifest)
    raw, out = Path(raw).resolve(), Path(out).resolve()
    raw.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    signature = object_hash(manifest)
    with exclusive_run(out), exclusive_run(raw):
        if (out/'manifest.json').exists() and read_json(out/'manifest.json') != manifest:
            raise ValueError('Changed file selection: choose a new acquisition run')
        write_json(out/'manifest.json', manifest)
        state = {'schema': 'research-public-acquisition-v1', 'manifest_hash': signature,
                 'status': 'running', 'started': utc(), 'files': {},
                 'private_downloads': 0, 'training_executed': False, 'training_admitted': False}
        for e in entries:
            state['files'][asset_key(e)] = {'status': 'pending'}

        def save():
            state['updated'] = utc()
            state['verified'] = sum(r['status'] == 'verified' for r in state['files'].values())
            state['failed'] = sum(r['status'] == 'failed' for r in state['files'].values())
            state['total'] = len(entries)
            write_json(out/'status.json', state)

        save()
        print('INDEX existing public caches; private matrices are not read or downloaded', flush=True)
        index = cache_index([raw, *caches], entries)

        def task(e):
            key = asset_key(e); target = raw/e['study']/e['file']
            start = time.monotonic()
            if target.exists():
                actual = verified_file(target, e)
                if not actual:
                    raise ValueError('Destination differs from lock; preserve it and select another raw directory')
                path, action = target, 'reused_destination'
            else:
                path = None
                for candidate in index.get((e['file'], e['bytes']), []):
                    actual = verified_file(candidate, e)
                    if actual:
                        path = candidate; break
                if path is not None:
                    action = 'reused_cache'
                else:
                    if not download:
                        raise FileNotFoundError('Missing; download mode required')
                    # Two workers may be in flight; reserve space for the largest other file.
                    reserve = max(x['bytes'] for x in entries) if workers > 1 else 0
                    if shutil.disk_usage(raw).free < e['bytes'] + reserve + 256 * 1024**2:
                        raise OSError('Insufficient free space for bounded concurrent downloads')
                    path = fetch_locked(e, target.parent, True)
                    actual = verified_file(path, e)
                    if not actual:
                        raise ValueError('Downloaded bytes failed verification')
                    action = 'downloaded_or_resumed'
            print('VERIFIED ' + key + ' ' + action, flush=True)
            return {'status': 'verified', 'action': action, 'path': str(path), 'bytes': e['bytes'],
                    'observed_sha256': actual['sha256'], 'seconds': round(time.monotonic()-start, 2),
                    'pinned_algorithm': 'sha256' if 'sha256' in e else 'md5',
                    'training_admitted': False}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(task, e): e for e in entries}
            for future in as_completed(futures):
                e = futures[future]; key = asset_key(e)
                try:
                    state['files'][key] = future.result()
                except Exception as exc:
                    state['files'][key] = {'status': 'failed', 'error_type': type(exc).__name__, 'reason': str(exc)[:500]}
                    print('FAILED ' + key + ': ' + str(exc), flush=True)
                save()
        state['status'] = 'completed' if state['verified'] == len(entries) else 'partial'
        save()
        lines = ['# 公共数据补全报告', '', f"状态：{state['status']}；核验 {state['verified']}/{len(entries)}。",
                 '', '私有数据未下载、未改分组；未调用 LLM、未执行训练。',
                 '文件完整不等于已经进入 AI4AI 数值实验。当前执行入口仍为现有 bulk/core/coarse 开发缓存。',
                 '', '|文件|状态|方式／原因|', '|---|---|---|']
        for key, record in state['files'].items():
            detail = record.get('action', record.get('reason', '')).replace('|', '/').replace('\n', ' ')
            lines.append(f"|{key}|{record['status']}|{detail}|")
        (out/'REPORT_CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'configs/research_public_assets.json')
    p.add_argument('--raw', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', action='append', type=Path, default=[])
    p.add_argument('--download', action='store_true'); p.add_argument('--workers', type=int, choices=[1, 2], default=2)
    a = p.parse_args()
    result = acquire(read_json(a.manifest), a.raw, a.out, a.cache, a.download, a.workers)
    print(json.dumps({k: result[k] for k in ('status', 'verified', 'total', 'failed')}, indent=2))
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    sys.exit(main())
