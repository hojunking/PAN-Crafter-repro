"""A committed, immutable release per local reproduction campaign."""
import os
from pathlib import Path
import subprocess
from pcrepro.common import ROOT,camp,read,locked,immutable_json,source_identity
from pcrepro.plan import verify_server,verify_sources

def process_start(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,ValueError,IndexError,TypeError):return None

def frozen_checkout(root,server):
    root=Path(root).resolve();verify_server(server);verify_sources(root)
    with locked(camp(root,server)/'deployment.lock'):
        receipt=read(camp(root,server)/'runtime_release.json')
        if receipt:
            target=Path(receipt['path'])
            if target.parent!=root.parent or not target.name.startswith(root.name+'-runtime-pcrepro-'+server+'-'):
                raise ValueError('Unexpected PCREPRO release path')
            current=source_identity(target)
            if current['files']!=receipt['files'] or current['git_release']!=receipt['git_commit']:
                raise ValueError('Frozen PCREPRO release changed')
            verify_sources(target);return target
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        files=source_identity(root)['files']
        names=[n for n in files if not n.startswith('external/')]
        names += [str(p.relative_to(root)) for p in (root/'pcrepro').glob('test_*.py')]
        tracked=set(subprocess.check_output(['git','ls-files','-z','--',*names],cwd=root).decode().split('\0'))
        if set(names)-tracked:raise ValueError('Commit PCREPRO sources/tests before deployment')
        dirty=subprocess.check_output(['git','diff','HEAD','--name-only','--',*names],cwd=root,text=True).splitlines()
        if dirty:raise ValueError('Uncommitted execution source: '+', '.join(dirty))
        target=root.parent/f'{root.name}-runtime-pcrepro-{server}-{commit[:12]}'
        if not target.exists():subprocess.run(['git','worktree','add','--detach',str(target),commit],cwd=root,check=True)
        if source_identity(target)['files']!=files:raise ValueError('Frozen source differs')
        for name in ('data','work_dir'):
            source,link=root/name,target/name
            if not source.exists():continue
            if link.is_symlink() and link.resolve()==source.resolve():continue
            if link.exists() or link.is_symlink():raise ValueError('Existing runtime asset differs')
            os.symlink(source.resolve(),link,target_is_directory=True)
        (target/'gspread').mkdir(exist_ok=True)
        for source in (root/'gspread').glob('*.json'):
            link=target/'gspread'/source.name
            if not link.exists() and not link.is_symlink():os.symlink(source.resolve(),link)
        immutable_json(camp(root,server)/'runtime_release.json',dict(path=str(target),origin_root=str(root),git_commit=commit,files=files))
        return target
