"""One committed release per local continuous service; no cross-server lock."""
import os
from pathlib import Path
import subprocess

from g23sens.common import ROOT,camp,read,locked,immutable_json,source_identity,verify_server
from g23sens.plan import verify_sources


def _link_tree(source, target):
    """Link untracked large assets without replacing committed runtime content."""
    if not source.exists(): return
    if not target.exists() and not target.is_symlink():
        os.symlink(source.resolve(),target,target_is_directory=source.is_dir());return
    if target.is_symlink():
        if target.resolve()!=source.resolve(): raise ValueError('Runtime asset link changed: '+str(target))
        return
    if source.is_dir() and target.is_dir():
        for child in source.iterdir(): _link_tree(child,target/child.name)
    elif source.is_file() and target.is_file():
        from g23sens.common import sha256
        if sha256(source)!=sha256(target): raise ValueError('Committed runtime asset differs')
    else: raise ValueError('Runtime asset type differs')


def frozen_checkout(root, server):
    root=Path(root).resolve();verify_server(server);verify_sources(root)
    with locked(camp(root,server)/'deployment.lock'):
        receipt=read(camp(root,server)/'runtime_release.json')
        if receipt:
            target=Path(receipt['path'])
            if target.parent!=root.parent or not target.name.startswith(root.name+'-runtime-g23sens-'+server+'-'):
                raise ValueError('Unexpected G23 release path')
            current=source_identity(target)
            if current['files']!=receipt['files'] or current['git_release']!=receipt['git_commit']:
                raise ValueError('Frozen G23 release changed')
            verify_sources(target);return target
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        files=source_identity(root)['files']
        names=[n for n in files if not n.startswith('external/')]
        names += [str(p.relative_to(root)) for p in (root/'g23sens').glob('test_*.py')]
        tracked=set(subprocess.check_output(['git','ls-files','-z','--',*names],cwd=root).decode().split('\0'))
        if set(names)-tracked:raise ValueError('Commit G23 execution sources/tests first: '+str(sorted(set(names)-tracked)))
        dirty=subprocess.check_output(['git','diff','HEAD','--name-only','--',*names],cwd=root,text=True).splitlines()
        if dirty:raise ValueError('Uncommitted execution source: '+', '.join(dirty))
        target=root.parent/f'{root.name}-runtime-g23sens-{server}-{commit[:12]}'
        if not target.exists():subprocess.run(['git','worktree','add','--detach',str(target),commit],cwd=root,check=True)
        if source_identity(target)['files']!=files:raise ValueError('Frozen source differs')
        for name in ('data','work_dir','assets'):_link_tree(root/name,target/name)
        (target/'gspread').mkdir(exist_ok=True)
        for source in (root/'gspread').glob('*.json'):_link_tree(source,target/'gspread'/source.name)
        immutable_json(camp(root,server)/'runtime_release.json',dict(path=str(target),origin_root=str(root),git_commit=commit,files=files))
        return target
