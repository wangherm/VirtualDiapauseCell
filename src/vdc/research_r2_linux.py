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
    mode=src.stat().st_mode
    executable=bool(mode & 0o111)
    def copy_readable():
        # copyfile does not preserve mode, notably across system/data disks.
        # Never chmod a hardlink: that would also change the host source.
        shutil.copyfile(src,dst)
        dst.chmod(0o555 if executable else 0o444)
    if not mode & stat.S_IROTH or (executable and not mode & stat.S_IXOTH):
        copy_readable();return
    try: os.link(src,dst)
    except OSError: copy_readable()


def readable_directories(root):
    """Normalize only copied directories, never files or symlink targets."""
    for directory, _, _ in os.walk(root, followlinks=False):
        Path(directory).chmod(0o755)


def memory_limits(config):
    """Resident budget is distinct from library mappings / reserved addresses."""
    memory=str(config.get('memory','64g')).lower()
    resident=int(memory[:-1])*1024**3 if memory.endswith('g') else int(memory)
    if resident<=0:raise ValueError('Resident memory budget must be positive')
    # Even CPU work imports the installed CUDA-enabled Torch libraries.
    # A small RLIMIT_AS can abort import while resident usage is still tiny.
    address_space=0 if config.get('gpu') else max(resident,64*1024**3)
    return resident,address_space


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
            if dest.is_symlink():dest.unlink()
            elif dest.exists():return
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
    # AutoDL may inject the compute driver outside the system library directories.
    driver_paths=[]
    for folder in ('/usr/local/nvidia/lib','/usr/local/nvidia/lib64','/usr/local/cuda/lib64'):
        if Path(folder).is_dir():
            copy_runtime(folder,root);driver_paths.append(folder)
    ldconfig=shutil.which('ldconfig') or '/sbin/ldconfig'
    found=subprocess.run([ldconfig,'-p'],capture_output=True,text=True,check=True)
    for line in found.stdout.splitlines():
        if '=>' not in line:continue
        name,path=line.strip().split('=>',1);path=Path(path.strip())
        if name.startswith(('libcuda.so','libnvidia-','libnvcuvid')) and path.exists():
            copy_runtime(path,root);driver_paths.append(str(path.parent))
    (root/'usr/bin').mkdir(parents=True,exist_ok=True)
    p=root/'usr/bin/python';p.symlink_to(sys.executable)
    (root/'etc').mkdir(exist_ok=True)
    (root/'etc/passwd').write_text('nobody:x:65534:65534:nobody:/tmp:/bin/sh\n')
    (root/'etc/group').write_text('nogroup:x:65534:\n')
    (root/'etc/hosts').write_text('127.0.0.1 localhost\n')
    readable_directories(root)
    import importlib.metadata
    manifest={'backend':'linux_landlock_chroot','python':sys.executable,
        'jail_sha256':sha256(out/'jail'), 'packages':sorted((d.metadata.get('Name',''),d.version)
            for d in importlib.metadata.distributions()),'created_at':time.time(),
        'runtime_files_are_trusted':True,'private_data_in_runtime':False,
        'driver_library_paths':sorted(set(driver_paths)),
        'executor_version':'r2_device_choice_v2'}
    # Snapshot files are read-only to candidates; source changes require a new runtime.
    manifest['runtime_id']=object_hash(manifest)
    write_json(out/'runtime.json',manifest)
    return manifest


def runtime_info(config):
    root=Path(config['runtime_root']).resolve();m=read_json(root/'runtime.json')
    if m.get('executor_version')!='r2_device_choice_v2':
        raise ValueError('Runtime predates executable-permission fix; build a new runtime snapshot')
    if sha256(root/'jail')!=m['jail_sha256']: raise ValueError('Isolation launcher changed')
    if not (root/'probe_passed.json').is_file() and not config.get('probe_mode'):
        raise RuntimeError('Runtime lacks actual isolation probe; run setup_research_r2.sh')
    if (root/'probe_passed.json').is_file() and read_json(root/'probe_passed.json')['runtime_id']!=m['runtime_id']:
        raise ValueError('Probe belongs to a different runtime')
    return m


def execute_linux(code, inputs, out, command, config, model=None, repo=None, network=False):
    if network and (inputs or model or repo): raise ValueError('Network staging cannot mount data')
    trace=bool(config.get('diagnostic_trace'))
    if trace and (not config.get('probe_mode') or inputs or model or repo or network):
        raise ValueError('Syscall tracing is only for an offline probe with no data mounts')
    tracer=shutil.which('strace') if trace else None
    if trace and not tracer:raise RuntimeError('GPU diagnostic requires strace on the host')
    m=runtime_info(config);runtime=Path(config['runtime_root']).resolve()
    if config.get('gpu') and not config.get('probe_mode'):
        probe=read_json(runtime/'gpu_probe_passed.json')
        if probe.get('status')!='passed' or probe.get('runtime_id')!=m['runtime_id'] or not probe.get('reload'):
            raise ValueError('GPU execution requires matching actual CUDA training/reload probe')
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
    readable_directories(root)
    (root/'out').chmod(0o777);(root/'tmp').chmod(0o777)
    # Character devices are recreated, not symlinked out of the chroot.
    devices=['null','zero','random','urandom']
    if config.get('gpu'):
        devices += [p.name for p in Path('/dev').glob('nvidia*') if p.is_char_device()]
        devices += [str(p.relative_to('/dev')) for p in Path('/dev/nvidia-caps').glob('*') if p.is_char_device()]
        # Driver metadata only. Never expose host /proc processes or environments.
        for info in ('/proc/driver/nvidia/version','/proc/driver/nvidia/params'):
            if Path(info).is_file():
                target=root/info.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(Path(info).read_bytes())
    for name in devices:
        s=(Path('/dev')/name).stat()
        (root/'dev'/name).parent.mkdir(parents=True,exist_ok=True)
        os.mknod(root/'dev'/name,stat.S_IFCHR|0o666,s.st_rdev)
        (root/'dev'/name).chmod(0o666)  # mknod is affected by the parent's umask.
    memory_bytes,address_space_bytes=memory_limits(config)
    argv=list(command)
    if argv[0]=='python':argv[0]=m['python']
    elif not argv[0].startswith('/'):argv[0]='/bin/'+argv[0]
    env={'PATH':'/usr/bin:/bin','HOME':'/tmp','TMPDIR':'/tmp','LANG':'C.UTF-8',
         'PYTHONPATH':'/code:/vdc/src','PYTHONDONTWRITEBYTECODE':'1',
         'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    env['R2_CPUS']=str(config.get('cpus',8))
    env['VDC_DEVICE']='cuda' if config.get('gpu') else 'cpu'
    env['CUDA_VISIBLE_DEVICES']='0' if config.get('gpu') else ''
    env['LD_LIBRARY_PATH']=':'.join(m.get('driver_library_paths',[]))
    from .research_r2_devices import process_group_usage,gpu_process_memory
    args=[str(runtime/'jail'),str(root),'1' if network else '0',str(address_space_bytes),
          str(config.get('timeout_seconds',7200)),*argv]
    trace_path=out.parent/(out.name+'_syscalls.log') if trace else None
    if trace:
        args=[tracer,'-f','-qq','-s','160','-e',
              'trace=%file,ioctl,prctl,socket,connect,clone,clone3',
              '-o',str(trace_path),*args]
    log=out.parent/(out.name+'.log'); start=time.monotonic();reason=None
    peak_rss=0;peak_gpu=None;last_gpu=0
    try:
        with log.open('wb') as stream:
            p=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                               env=env,start_new_session=True)
            while p.poll() is None:
                time.sleep(.5)
                pids,rss=process_group_usage(p.pid);peak_rss=max(peak_rss,rss)
                if config.get('gpu') and time.monotonic()-last_gpu>=1:
                    measured=gpu_process_memory(pids);last_gpu=time.monotonic()
                    if measured is not None:peak_gpu=max(peak_gpu or 0,measured)
                if time.monotonic()-start>config.get('timeout_seconds',7200): reason='timeout'
                elif rss>memory_bytes:reason='resident_memory_limit'
                elif log.stat().st_size>20_000_000:reason='log_limit'
                elif trace_path and trace_path.exists() and trace_path.stat().st_size>20_000_000:reason='trace_limit'
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
            'selected_device':env['VDC_DEVICE'],'peak_process_group_rss_bytes':peak_rss,
            'gpu_memory_observed_bytes':peak_gpu,
            'gpu_usage_status':('allocation_observed' if peak_gpu else 'not_observed_or_unavailable') if config.get('gpu') else 'not_requested',
            'memory_limit_mode':'sampled_group_rss' if config.get('gpu') else 'per_process_AS_and_sampled_group_rss',
            'resident_memory_limit_bytes':memory_bytes,
            'address_space_limit_bytes':address_space_bytes or None,
            'memory_policy_version':'separate_address_space_v1',
            'diagnostic_trace':str(trace_path) if trace_path else None,
            'runtime_id':m['runtime_id'],'backend':'linux_landlock_chroot','network':network,
            'files':tree_hash(out)}
    write_json(out.parent/(out.name+'_execution.json'),result)
    if reason or rc:raise RuntimeError('Isolated candidate failed: '+str(reason or rc)+'; inspect local log: '+str(log))
    return result
