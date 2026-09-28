"""Local paths and provenance; no historical campaign activation or mutation."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
from fh12.common import atomic_json,object_sha,read_json,sha256,utcnow

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN_ID='PANDA_REBUTTAL_B01_WV3_S135_20260928_v1'
SERVERS=('s1','s3','s5')


def campaign_dir(root=ROOT): return Path(root)/'work_dir/_panda_rb/20260928/B01'
def server_dir(server,root=ROOT):
    if server not in SERVERS: raise ValueError('B01 only permits s1, s3, s5; s2/s4 are out of scope')
    return campaign_dir(root)/'control'/server
def run_dir(run_id,root=ROOT):
    from panda_rb.plan import case_for
    row=case_for(run_id,root)
    return campaign_dir(root)/'RB01'/row['server']/('R'+str(row['repeat']))/row['case_id']
def read(path,default=None):
    return read_json(path) if Path(path).is_file() else ({} if default is None else default)


@contextmanager
def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if read_json(path)!=value: raise ValueError('Immutable artifact changed: '+str(path))
    else: atomic_json(path,value)
    return value


def append_event(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(dict(at_utc=utcnow(),**value),sort_keys=True,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def source_identity(root=ROOT):
    from fh12.common import source_identity as legacy
    root=Path(root); identity=legacy(root)
    paths=list((root/'panda_rb').glob('*.py'))+list((root/'tools').glob('panda_rb_*.py'))
    for name in ('fh20r1', 'model', 'pa', 'tools/metrics'):
        paths += list((root/name).rglob('*.py'))
    # B01 imports this evaluator, not the unrelated Sheet workers in that package.
    # Local uncommitted worker files must not invalidate a clean peer's e-cache.
    paths += [root/'reporting_extra/__init__.py', root/'reporting_extra/evaluation.py']
    for path in sorted(paths):
        if path.is_file() and not path.name.startswith('test_'):
            identity['files'][str(path.relative_to(root))]=sha256(path)
    identity['content_sha256']=object_sha(identity['files'])
    identity['campaign_id']=CAMPAIGN_ID
    frozen = read(root/'runtime_release.json')
    if frozen:
        verify_files(frozen['files'], root)
        identity['git_release']=frozen['git_origin']
        identity['frozen_release_sha256']=frozen['release_sha256']
    return identity


def verify_files(files,root=ROOT):
    for relative,digest in files.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe artifact path')
        if sha256(Path(root)/path)!=digest: raise ValueError('Artifact bytes changed: '+relative)


def docker_required():
    if not Path('/.dockerenv').is_file():
        raise RuntimeError('Actual B01 experiment must run in the pinned Docker environment')
