"""Pin one committed runtime without stashing unrelated changes or old assets."""
import os
from pathlib import Path
import subprocess
from gfp40.common import ROOT,camp,read,locked,immutable_json,source_identity
from gfp40.plan import verify_lane

def verify_legacy_documents(root):
    """A committed runtime must contain all pinned transitive plan documents."""
    from fh12.source_documents import pinned_document
    from qg40.plan import SOURCE_SHAS as qg
    from g20.plan import SOURCE_SHAS as g20
    from l100.plan import SOURCE_SHAS as l100
    return {name:str(pinned_document(root,name,digest)) for name,digest in {**qg,**g20,**l100}.items()}


def process_start(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,ValueError,IndexError,TypeError):return None


def frozen_checkout(root,server):
    verify_lane(server);root=Path(root).resolve()
    with locked(camp(root,server)/'deployment.lock'):
        receipt=read(camp(root,server)/'runtime_release.json')
        if receipt:
            target=Path(receipt['path'])
            if target.parent!=root.parent or not target.name.startswith(root.name+'-runtime-gfp40-'+server+'-'):
                raise ValueError('Unexpected GFP40 frozen checkout')
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
            if head!=receipt['git_commit'] or source_identity(target)['files']!=receipt['files']:
                raise ValueError('Frozen GFP40 release changed; do not migrate a live run')
            verify_legacy_documents(target)
            return target
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        release=source_identity(root)['files']
        files=[name for name in release if not name.startswith('external/')]
        files+=['tools/gfp40_start.sh']
        files+=['gfp40/'+p.name for p in (root/'gfp40').glob('test_*.py')]
        tracked=set(subprocess.check_output(['git','ls-files','-z','--',*files],cwd=root).decode().split('\0'))
        if set(files)-tracked:raise ValueError('Commit GFP40 sources/tests before starting')
        changed=subprocess.check_output(['git','diff','HEAD','--name-only','--',*files],cwd=root,text=True).splitlines()
        if changed:raise ValueError('Uncommitted release source: '+', '.join(changed))
        target=root.parent/f'{root.name}-runtime-gfp40-{server}-{head[:12]}'
        if not target.exists():subprocess.run(['git','worktree','add','--detach',str(target),head],cwd=root,check=True)
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()!=head
            or source_identity(target)['files']!=release):raise ValueError('Frozen checkout differs')
        verify_legacy_documents(target)
        (root/'work_dir').mkdir(exist_ok=True)
        for name in ('data','work_dir'):
            source,link=root/name,target/name
            if not source.exists():continue
            if link.is_symlink() and link.resolve()==source.resolve():continue
            if link.exists() or link.is_symlink():raise ValueError('Existing runtime asset link differs: '+str(link))
            os.symlink(source.resolve(),link,target_is_directory=True)
        (target/'gspread').mkdir(exist_ok=True)
        for source in (root/'gspread').glob('*.json'):
            link=target/'gspread'/source.name
            if not link.exists() and not link.is_symlink():os.symlink(source.resolve(),link)
        immutable_json(camp(root,server)/'runtime_release.json',dict(path=str(target),git_commit=head,files=release))
        return target
