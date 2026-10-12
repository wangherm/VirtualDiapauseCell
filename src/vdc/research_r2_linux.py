"""AutoDL-compatible isolation without nested Docker; actual probe is mandatory.

Trusted runtime and approved inputs are linked/copied into a disposable chroot.
Landlock makes all files read-only except /out and /tmp; candidate drops root,
cannot create network sockets, see host processes, or read controller credentials.
No mounting, user namespaces, docker daemon or pip replacement of host torch.
"""
from pathlib import Path
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from .io import read_json, write_json, object_hash, sha256
from .research_r2_tools import tree_hash


def link_or_copy(src, dst):
    src, dst=Path(src),Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists(): return
    # Private files with restrictive modes are copied before making a readable input.
    if not src.stat().st_mode & stat.S_IROTH:
        shutil.copyfile(src,dst);dst.chmod(0o444);return
    try: os.link(src,dst)
    except OSError: shutil.copyfile(src,dst)


def copy_runtime(src, root):
    src=Path(src); dest=Path(root)/str(src).lstrip('/')
    if src.is_dir():
        shutil.copytree(src,dest,symlinks=True,copy_function=link_or_copy,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','.cache','.git'))
    elif src.exists():
        if src.is_symlink():
            # /etc/resolv.conf and /bin/sh often point outside the selected paths.
            # Materialize the trusted file, not an unusable host-only symlink.
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src.resolve(),dest);dest.chmod(src.stat().st_mode & 0o777)
        else:link_or_copy(src,dest)


def build_runtime(repo, out):
    if sys.platform!='linux' or os.geteuid()!=0:
        raise RuntimeError('AutoDL isolation requires Linux root with CAP_SYS_CHROOT; no unrestricted fallback')
    repo,out=Path(repo).resolve(),Path(out).resolve()
    if out.exists(): raise ValueError('Runtime output already exists; choose a new runtime snapshot')
    out.mkdir(parents=True)
    subprocess.run(['cc','-O2','-Wall','-Wextra',str(repo/'scripts/r2_jail.c'),'-o',str(out/'jail')],check=True)
    root=out/'root';root.mkdir()
    # Copy only trusted runtime paths, never /root, the project, or the private data root.
    # Preserve interpreter absolute paths and its venv's pyvenv.cfg.
    for prefix in {Path(sys.base_prefix),Path(sys.prefix)}:
        if prefix in (Path('/usr'),Path('/usr/local')):
            # System prefixes also contain kernels/WSL drivers and unrelated apps.
            # Only Python's standard library and interpreter libraries belong here.
            for pattern in ('python*','libpython*'):
                for lib in (prefix/'lib').glob(pattern): copy_runtime(lib,root)
        else:
            for name in ('lib','lib64'):
                if (prefix/name).exists(): copy_runtime(prefix/name,root)
        copy_runtime(prefix/'pyvenv.cfg',root)
        for name in ('python','python3','python3.12'):
            p=prefix/'bin'/name
            if p.exists():
                target=root/str(p).lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists(): shutil.copyfile(p.resolve(),target);target.chmod(0o755)
    for name in ('/lib/x86_64-linux-gnu','/lib64','/usr/lib/x86_64-linux-gnu','/etc/ssl/certs','/usr/share/ca-certificates'):
        if Path(name).exists(): copy_runtime(name,root)
    for name in ('/bin/sh','/bin/bash','/usr/bin/env','/etc/resolv.conf','/etc/nsswitch.conf'):
        if Path(name).exists(): copy_runtime(name,root)
    (root/'usr/bin').mkdir(parents=True,exist_ok=True)
    p=root/'usr/bin/python';p.symlink_to(sys.executable)
    (root/'etc').mkdir(exist_ok=True)
    (root/'etc/passwd').write_text('nobody:x:65534:65534:nobody:/tmp:/bin/sh\n')
    (root/'etc/group').write_text('nogroup:x:65534:\n')
    (root/'etc/hosts').write_text('127.0.0.1 localhost\n')
    import importlib.metadata
    manifest={'backend':'linux_landlock_chroot','python':sys.executable,
        'jail_sha256':sha256(out/'jail'), 'packages':sorted((d.metadata.get('Name',''),d.version)
            for d in importlib.metadata.distributions()),'created_at':time.time(),
        'runtime_files_are_trusted':True,'private_data_in_runtime':False}
    # Snapshot files are read-only to candidates; source changes require a new runtime.
    manifest['runtime_id']=object_hash(manifest)
    write_json(out/'runtime.json',manifest)
    return manifest


def runtime_info(config):
    root=Path(config['runtime_root']).resolve();m=read_json(root/'runtime.json')
    if sha256(root/'jail')!=m['jail_sha256']: raise ValueError('Isolation launcher changed')
    if not (root/'probe_passed.json').is_file() and not config.get('probe_mode'):
        raise RuntimeError('Runtime lacks actual isolation probe; run setup_research_r2.sh')
    if (root/'probe_passed.json').is_file() and read_json(root/'probe_passed.json')['runtime_id']!=m['runtime_id']:
        raise ValueError('Probe belongs to a different runtime')
    return m


def execute_linux(code, inputs, out, command, config, model=None, repo=None, network=False):
    if network and (inputs or model or repo): raise ValueError('Network staging cannot mount data')
    m=runtime_info(config);runtime=Path(config['runtime_root']).resolve()
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    jail=out.parent/(out.name+'_jail_'+uuid.uuid4().hex[:8]);jail.mkdir()
    root=jail/'root'
    shutil.copytree(runtime/'root',root,symlinks=True,copy_function=link_or_copy)
    for p in ('out','tmp','input','dev','code'):(root/p).mkdir(exist_ok=True)
    (root/'out').chmod(0o777);(root/'tmp').chmod(0o777)
    shutil.copytree(code,root/'code',dirs_exist_ok=True,copy_function=link_or_copy)
    for name,path in inputs.items():
        if '/' in name or '\\' in name or ':' in name or name in ('.','..'):raise ValueError('Invalid input basename')
        link_or_copy(path,root/'input'/name)
    for path,name in ((model,'model'),(repo,'vdc'),(config.get('wheels'),'wheels')):
        if path and name=='vdc':
            for allowed in ('src','scripts','configs','knowledge'):
                selected=Path(path)/allowed
                if selected.exists():shutil.copytree(selected,root/name/allowed,symlinks=False,
                    copy_function=link_or_copy,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        elif path:shutil.copytree(path,root/name,symlinks=False,copy_function=link_or_copy)
    # Character devices are recreated, not symlinked out of the chroot.
    devices=['null','zero','random','urandom']
    if config.get('gpu'):
        devices += [p.name for p in Path('/dev').glob('nvidia*') if p.is_char_device()]
    for name in devices:
        s=(Path('/dev')/name).stat()
        os.mknod(root/'dev'/name,stat.S_IFCHR|0o666,s.st_rdev)
    memory=str(config.get('memory','64g')).lower()
    memory_bytes=int(memory[:-1])*1024**3 if memory.endswith('g') else int(memory)
    argv=list(command)
    if argv[0]=='python':argv[0]=m['python']
    elif not argv[0].startswith('/'):argv[0]='/bin/'+argv[0]
    env={'PATH':'/usr/bin:/bin','HOME':'/tmp','TMPDIR':'/tmp','LANG':'C.UTF-8',
         'PYTHONPATH':'/code:/vdc/src','PYTHONDONTWRITEBYTECODE':'1',
         'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    env['R2_CPUS']=str(config.get('cpus',8))
    args=[str(runtime/'jail'),str(root),'1' if network else '0',str(memory_bytes),
          str(config.get('timeout_seconds',7200)),*argv]
    log=out.parent/(out.name+'.log'); start=time.monotonic();reason=None
    try:
        with log.open('wb') as stream:
            p=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                               env=env,start_new_session=True)
            while p.poll() is None:
                time.sleep(.5)
                if time.monotonic()-start>config.get('timeout_seconds',7200): reason='timeout'
                elif log.stat().st_size>20_000_000:reason='log_limit'
                elif sum(f.stat().st_size for f in (root/'out').rglob('*') if f.is_file() and not f.is_symlink())>config.get('max_output_bytes',50_000_000_000):reason='output_disk_limit'
                if reason:
                    os.killpg(p.pid,signal.SIGKILL);p.wait();break
            rc=p.returncode
            # A candidate cannot leave background descendants running after its entrypoint.
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        tree_hash(root/'out')
        for f in (root/'out').iterdir():
            if (out/f.name).exists(): raise ValueError('Output already exists')
            shutil.move(str(f),str(out/f.name))
    finally:
        # Only this newly created disposable jail is removed; source hardlinks remain intact.
        if jail.parent!=out.parent or not jail.name.startswith(out.name+'_jail_'):
            raise ValueError('Unexpected disposable path')
        shutil.rmtree(jail)
    result={'exit_code':rc,'stop_reason':reason,'seconds':round(time.monotonic()-start,3),
            'runtime_id':m['runtime_id'],'backend':'linux_landlock_chroot','network':network,
            'files':tree_hash(out)}
    write_json(out.parent/(out.name+'_execution.json'),result)
    if reason or rc:raise RuntimeError('Isolated candidate failed: '+str(reason or rc)+'; local log retained')
    return result
