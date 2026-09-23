"""Freeze an explicitly started campaign; later pulls cannot mutate a live run."""
from pathlib import Path
import os
import subprocess

from ablr2.common import camp, locked, read, immutable_json, source_identity,sha256
from ablr2.plan import verify_lane


def verify_release(receipt,root,server,*,extension=False):
    """Audit the original file manifest directly, not with newer imported code."""
    root=Path(root).resolve();target=Path(receipt['path'])
    prefix=root.name+('-runtime-ablr2x-' if extension else '-runtime-ablr2-')+server+'-'
    if target.parent!=root.parent or not target.name.startswith(prefix):
        raise ValueError('Unexpected frozen runtime path')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
    if head!=receipt['git_commit']:raise ValueError('Frozen runtime commit changed')
    for name,digest in receipt['files'].items():
        if name=='external/DLPan/wald_utilities.py':
            if digest=='UNAVAILABLE':continue
            base=Path(os.environ.get('PANCRAFTER_DLPAN',str(root.parent/'DLPan-Toolbox')))
            path=base/'01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py'
        else:
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe frozen source manifest')
            path=target/relative
        if not path.is_file() or sha256(path)!=digest:
            raise ValueError('Frozen runtime changed; restore its exact source before resuming: '+name)
    return target


def frozen_checkout(root,server,*,extension=False):
    verify_lane(server)
    root=Path(root).resolve()
    folder=camp(root,server)
    with locked(folder / 'deployment.lock'):
        pointer=folder/('runtime_release_ablr2x.json' if extension else 'runtime_release.json')
        receipt=read(pointer)
        if receipt:
            target=verify_release(receipt,root,server,extension=extension)
            if extension and source_identity(target)['files']!=receipt['files']:
                raise ValueError('Extended frozen runtime source set changed')
            return target
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        release=source_identity(root)['files']
        files=[name for name in release if not name.startswith('external/')]
        files+=['tools/ablr2_start.sh']
        files+=['ablr2/'+p.name for p in (root/'ablr2').glob('test_*.py')]
        tracked=set(subprocess.check_output(['git','ls-files','-z','--',*files],cwd=root).decode().split('\0'))
        if set(files)-tracked: raise ValueError('Commit the ABLR2 implementation before starting the frozen release')
        changed=subprocess.check_output(['git','diff','HEAD','--name-only','--',*files],cwd=root,text=True).splitlines()
        if changed: raise ValueError('Uncommitted numerical release files: '+', '.join(changed))
        runtime_name='ablr2x' if extension else 'ablr2'
        target=root.parent/f'{root.name}-runtime-{runtime_name}-{server}-{head[:12]}'
        if not target.exists():
            subprocess.run(['git','worktree','add','--detach',str(target),head],cwd=root,check=True)
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
        if actual!=head or source_identity(target)['files']!=release:
            raise ValueError('Frozen checkout differs from committed release')
        (root/'work_dir').mkdir(exist_ok=True)
        for name in ('data','work_dir'):
            source,link=root/name,target/name
            if not source.exists(): continue
            if link.is_symlink() and link.resolve()==source.resolve(): continue
            if link.exists() or link.is_symlink(): raise ValueError('Existing runtime asset link differs: '+str(link))
            os.symlink(source.resolve(),link,target_is_directory=True)
        (target/'gspread').mkdir(exist_ok=True)
        for source in (root/'gspread').glob('*.json'):
            link=target/'gspread'/source.name
            if not link.exists() and not link.is_symlink(): os.symlink(source.resolve(),link)
        immutable_json(pointer,dict(path=str(target),git_commit=head,files=release,
            **(dict(extension=True,original_runtime_release=read(folder/'runtime_release.json')) if extension else {})))
        return target
