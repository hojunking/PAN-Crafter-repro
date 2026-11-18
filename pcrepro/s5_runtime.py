"""Pull-and-start entrypoint for the fixed QB/GF2 study; never stops other jobs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from pcrepro.common import ROOT,atomic_json,read,read_json,sha256,source_identity,locked,utcnow
from pcrepro.deployment import process_start
from pcrepro.finite import folder,require_s5,require_idle

WALD='01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py'


def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.PIPE).strip()


def alive(launcher):
    pid=launcher.get('pid');start=launcher.get('process_start')
    return bool(pid and start and process_start(pid)==start)


def status(root=ROOT):
    out=folder(root);launch=read(out/'launcher.json');study=read(out/'status.json')
    active=study.get('active_run')
    return dict(runtime=read(out/'runtime_release.json'),launcher=launch,launcher_alive=alive(launch),
        study=study,training=read(Path(root)/'work_dir'/active/'meta/training_status.json') if active else {},
        control=read(out/'control.json'))


def resolve_inputs(root):
    """Fast presence/size check. prepare subsequently hashes and scans all tensors."""
    root=Path(root);assets=read_json(root/'pcrepro/s5_assets.json')
    missing=[]
    for name,spec in assets['data_files'].items():
        p=root/name
        if not p.is_file() or p.stat().st_size!=spec['bytes']:missing.append(name)
    if missing:raise FileNotFoundError('Missing/wrong-sized s5 data files:\n'+'\n'.join(missing))
    supplied=os.environ.get('PANCRAFTER_DLPAN')
    candidates=[Path(supplied)] if supplied else [root.parent/'DLPan-Toolbox',root/'external/DLPan-Toolbox',Path('/dlpan')]
    for path in candidates:
        if (path/WALD).is_file():
            if sha256(path/WALD)!=assets['wald_sha256']:
                raise ValueError('DLPan evaluator differs from the reviewed module: '+str(path/WALD))
            return path.resolve()
    raise FileNotFoundError('Set PANCRAFTER_DLPAN to the existing DLPan-Toolbox containing '+WALD)


def _link(source,target):
    if target.is_symlink() and target.resolve()==source.resolve():return
    if target.exists() or target.is_symlink():raise ValueError('Existing runtime asset differs: '+str(target))
    target.symlink_to(source.resolve(),target_is_directory=True)


def snapshot(root=ROOT):
    """Pin a committed worktree once; later parent pulls do not change a run."""
    root=Path(root).resolve();require_s5(root);out=folder(root)
    with locked(out/'deployment.lock'):
        saved=read(out/'runtime_release.json')
        if saved:
            target=Path(saved['path'])
            if (target.parent!=root.parent or not target.name.startswith(root.name+'-runtime-pcpaper-qbgf2-s5-')
                    or git(target,'rev-parse','HEAD')!=saved['commit']):
                raise ValueError('Pinned s5 runtime path/commit changed')
            for name,digest in saved['files'].items():
                if name.startswith('external/'):continue
                if sha256(target/name)!=digest:raise ValueError('Pinned runtime source changed: '+name)
            if str(Path(sys.executable).absolute())!=saved['python']:
                raise ValueError('Use the recorded Python environment: '+saved['python'])
            if sha256(Path(saved['dlpan'])/WALD)!=saved['files']['external/DLPan/wald_utilities.py']:
                raise ValueError('Pinned DLPan evaluator changed')
            return saved
        dlpan=resolve_inputs(root)
        os.environ['PANCRAFTER_DLPAN']=str(dlpan)
        files=source_identity(root)['files']
        names=[n for n in files if not n.startswith('external/')]
        names += [str(p.relative_to(root)) for p in (root/'pcrepro').glob('test_*.py')]
        # A tracked symlink's target must also come from this commit.
        for name in list(names):
            p=root/name
            if p.is_symlink():names.append(str(p.resolve().relative_to(root)))
        tracked=set(git(root,'ls-files','-z','--',*names).split('\0'))
        if set(names)-tracked:raise ValueError('Required execution files are not committed: '+str(sorted(set(names)-tracked)))
        dirty=git(root,'diff','HEAD','--name-only','--',*names)
        if dirty:raise ValueError('Required execution source has local edits:\n'+dirty)
        commit=git(root,'rev-parse','HEAD')
        target=root.parent/(root.name+'-runtime-pcpaper-qbgf2-s5-'+commit[:12])
        if not target.exists():
            subprocess.run(['git','worktree','add','--detach',str(target),commit],cwd=root,check=True)
        if git(target,'rev-parse','HEAD')!=commit:raise ValueError('Runtime checkout belongs to another commit')
        for name,digest in files.items():
            if not name.startswith('external/') and sha256(target/name)!=digest:
                raise ValueError('Committed source differs from reviewed checkout: '+name)
        (root/'work_dir').mkdir(exist_ok=True)
        for name in ('data','work_dir'):_link(root/name,target/name)
        (target/'gspread').mkdir(exist_ok=True)
        marker=target/'gspread/server.txt'
        if marker.exists() and marker.read_text().strip()!='s5':raise ValueError('Foreign runtime server marker')
        marker.write_text('s5\n')
        receipt=dict(path=str(target),origin=str(root),commit=commit,files=files,
            python=str(Path(sys.executable).absolute()),dlpan=str(dlpan),created_at_utc=utcnow())
        atomic_json(out/'runtime_release.json',receipt)
        return receipt


def start(root=ROOT,resume=False):
    root=Path(root).resolve();require_s5(root);out=folder(root)
    with locked(out/'launch.lock'):
        previous=read(out/'launcher.json')
        if alive(previous):return dict(state='ALREADY_RUNNING',pid=previous['pid'],log=previous['log'])
        if read(out/'control.json').get('command')=='STOP_NOW_SAFE' and not resume:
            raise ValueError('Study is paused; use the resume action')
        require_idle();release=snapshot(root);runtime=Path(release['path'])
        env=dict(os.environ,PANCRAFTER_DLPAN=release['dlpan'],PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        if env.get('CUDA_VISIBLE_DEVICES') in ('','-1'):
            raise ValueError('CUDA_VISIBLE_DEVICES hides CUDA; use the s5 training environment')
        command=[release['python'],'-u',str(runtime/'tools/pcrepro_qbgf2.py')]
        log=out/'logs/launcher.log';log.parent.mkdir(parents=True,exist_ok=True)
        preparation=out/'logs/prepare.log'
        print('Preparing fixed QB/GF2 seed2025 study; log: '+str(preparation),flush=True)
        with preparation.open('a') as stream:
            code=subprocess.call(command+['prepare','--seeds','2025'],cwd=runtime,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if code:raise RuntimeError('Preparation failed; see '+str(preparation))
        if resume:atomic_json(out/'control.json',dict(command='CONTINUE',at_utc=utcnow()))
        with log.open('a') as stream:
            child=subprocess.Popen(command+['run'],cwd=runtime,env=env,stdout=stream,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        identity=process_start(child.pid)
        launch=dict(pid=child.pid,process_start=identity,log=str(log),runtime=release['path'],
            commit=release['commit'],started_at_utc=utcnow())
        atomic_json(out/'launcher.json',launch)
        time.sleep(.5)
        if child.poll() is not None and child.returncode:
            raise RuntimeError('Runner exited with '+str(child.returncode)+'; see '+str(log))
        return dict(state='LAUNCHED' if child.poll() is None else 'FINISHED',**launch)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('start','status','pause','resume'));args=p.parse_args(argv)
    if args.action=='status':return status()
    if args.action=='pause':
        require_s5(ROOT)
        if not read(folder()/'runtime_release.json'):raise ValueError('No deployed s5 study')
        value=dict(command='STOP_NOW_SAFE',at_utc=utcnow());atomic_json(folder()/'control.json',value);return value
    return start(resume=args.action=='resume')
