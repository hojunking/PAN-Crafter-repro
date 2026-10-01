"""Local-only operational boundaries; never signals another process."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from .common import CAMPAIGN, timestamp, atomic_json, read_json

class SafetyStop(RuntimeError):
    pass

def isolated_paths(project_root, work_root, legacy_roots=(), data_root=None):
    project=Path(project_root).resolve(); work=Path(work_root).resolve()
    if not project.is_dir() or project==project.parent:
        raise SafetyStop('BLOCKED_PATH: project root')
    if project not in work.parents or work==project:
        raise SafetyStop('BLOCKED_PATH: work root must be inside the new project, not its root')
    for raw in legacy_roots:
        old=Path(raw).resolve()
        if project==old or project in old.parents or old in project.parents or work==old or old in work.parents:
            raise SafetyStop('BLOCKED_PATH: project/work overlaps existing experiment')
    if data_root:
        data=Path(data_root).resolve()
        if data==work or data in work.parents or work in data.parents:
            raise SafetyStop('BLOCKED_PATH: data/work overlap')
    return dict(project_root=str(project),work_root=str(work),legacy_roots=[str(Path(x).resolve()) for x in legacy_roots])

def _command(args):
    try:
        run=subprocess.run(args,text=True,capture_output=True,timeout=15,check=False)
        return dict(returncode=run.returncode,stdout=run.stdout.strip(),stderr=run.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(returncode=None,stdout='',stderr=str(exc))

def inventory(work_root, server_identity=None):
    identity=Path(server_identity).resolve() if server_identity else None
    mapped=identity.read_text().strip() if identity and identity.is_file() else None
    parent=Path(work_root).resolve()
    while not parent.exists(): parent=parent.parent
    usage=shutil.disk_usage(parent)
    return dict(at_utc=timestamp(),hostname=platform.node(),server=mapped,
        server_identity_file=str(identity) if identity else None,python=sys.version,executable=sys.executable,
        gpu=_command(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits']),
        gpu_processes=_command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']),
        processes=_command(['ps','-eo','pid,ppid,comm,args']),
        ram=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if Path('/proc/meminfo').exists() else {},
        disk=dict(total=usage.total,used=usage.used,free=usage.free),
        mountinfo=Path('/proc/self/mountinfo').read_text() if Path('/proc/self/mountinfo').exists() else '')

def admit(inv, *, expected_hostname=None, next_block_bytes=0, atomic_reserve_bytes=0, own_pid=None):
    if inv.get('server')!='s2': raise SafetyStop('BLOCKED_SERVER: actual mapping is not s2')
    if expected_hostname and inv.get('hostname')!=expected_hostname:
        raise SafetyStop('BLOCKED_IDENTITY: hostname changed')
    required=max(20*1024**3,2*int(next_block_bytes)+int(atomic_reserve_bytes))
    if inv['disk']['free']<required: raise SafetyStop('PAUSED_DISK: insufficient reserve')
    if inv['gpu'].get('returncode')!=0 or not inv['gpu'].get('stdout'):
        raise SafetyStop('WAIT_RESOURCE: GPU inventory unavailable')
    if inv['gpu_processes'].get('returncode')!=0:
        raise SafetyStop('WAIT_RESOURCE: GPU process inventory unavailable')
    lines=[line for line in inv['gpu_processes']['stdout'].splitlines() if line.strip()]
    if own_pid is not None:
        # Host PID namespace is mandatory in the launcher. Exclude only this
        # process, never a container, user, server or arbitrary process family.
        lines=[line for line in lines if len(line.split(','))<2 or line.split(',')[1].strip()!=str(own_pid)]
    if lines: raise SafetyStop('WAIT_RESOURCE: existing GPU compute process; no process was stopped')
    return dict(status='RESOURCE_AVAILABLE',required_disk_bytes=required)

@contextmanager
def exclusive_lock(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise SafetyStop('WRITER_CONFLICT: controller already running') from exc
        try: yield stream
        finally: fcntl.flock(stream,fcntl.LOCK_UN)

def control(work_root, action=None):
    path=Path(work_root)/'control.json'
    if action is not None:
        if action not in ('RUN','PAUSE','STOP_AFTER_BLOCK','STOP'):
            raise ValueError('Unsupported operator action')
        atomic_json(path,dict(schema='PANDEP_CONTROL_v1',campaign_id=CAMPAIGN,
                            action=action,reason=None,updated_at_utc=timestamp()))
    if not path.exists(): return dict(action='PAUSE')
    value=read_json(path)
    if value.get('campaign_id')!=CAMPAIGN: raise SafetyStop('BLOCKED_IDENTITY: control campaign')
    return value
