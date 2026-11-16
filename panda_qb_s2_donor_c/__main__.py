"""Recover QB representatives first, then train the five fixed donor cases."""
import argparse
import json
import subprocess
import sys
from . import plan


def recover(apply):
    from .predecessor import require_finished,roots
    require_finished()
    for name,_ in roots():
        command=[sys.executable,'-m','panda_qb_repfix','--campaign',name]
        if apply:command.append('--apply')
        subprocess.run(command,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('catalog','check','check-sheet','recover','publish','run'))
    parser.add_argument('--server',choices=('s2',),default='s2')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--apply',action='store_true')
    a=parser.parse_args()
    from .release import verify
    proof=verify(environment=a.command in ('check','run'))
    if a.command=='catalog':print(plan.catalog('s2'),end='');return
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!='s2':raise ValueError('Execute on physical s2')
    if a.command in ('recover','check-sheet'):
        recover(a.command=='recover' and a.apply)
        if a.command=='recover':return
    if a.command=='check':
        from . import assets
        from .predecessor import require_finished
        from ta2.evaluation import evaluator_identity
        require_finished();source,registered=assets.sources('s2')
        pins=dict(evaluator=evaluator_identity(),train_inventory=registered.inventory.manifest(),
                  split_manifests={s:source.manifest_sha(s) for s in ('train','val','rr','fr')})
        _,_,_,rows=assets.check_source('s2',registered,pins)
        proof.update(new_stage1_runs=0,stage2_runs=5,donor_steps=[r['step'] for r in rows],teacher_architecture='W128_D123_UBIL',alpha=0)
        print(json.dumps(proof));return
    if a.command=='run':recover(True)
    from .install import install
    install()
    from panda_input3.__main__ import main as shared_main
    sys.argv=['panda_qb_s2_donor_c',a.command,'--server','s2','--device',a.device]
    if a.apply:sys.argv.append('--apply')
    shared_main()


if __name__=='__main__':main()
