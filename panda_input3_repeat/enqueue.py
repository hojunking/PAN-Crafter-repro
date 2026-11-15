"""Append R2 after the current C30 cycle, without stopping an active run."""
import argparse
import os
import subprocess
import sys
from panda_input3 import enqueue as core
from panda_input3_original import enqueue as previous
from panda_unified_campaign import enqueue as base
from . import recipes
from .predecessor import finished
from .release import verify


def name(server): return f'panda-input3-c30-r2-{server}-v1'


def command(server, image, *, gpu=False, detached=False, dlpan=None):
    cmd = core.command(server, image, gpu=gpu, detached=detached, dlpan=dlpan)
    output = recipes.ROOT / 'work_dir/panda_input3_repeat'; output.mkdir(parents=True, exist_ok=True)
    cmd[-4:-4] = ['--mount', f'type=bind,src={output},dst=/workspace/work_dir/panda_input3_repeat']
    cmd[-1] = 'panda_input3_repeat'
    if detached:
        cmd[cmd.index('--name') + 1] = name(server)
        for i, arg in enumerate(cmd):
            if arg.startswith('panda.campaign='): cmd[i] = 'panda.campaign=' + recipes.CAMPAIGN_ID
            if arg.startswith('panda.source='): cmd[i] = 'panda.source=' + verify()['source_sha256']
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', choices=('s1', 's2', 's3'), required=True)
    p.add_argument('--image', default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan', default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--check-only', action='store_true'); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); proof = verify()
    if (recipes.ROOT / 'gspread/server.txt').read_text().strip() != a.server:
        raise ValueError('Execute on matching physical server')
    current = base.inspect(name(a.server))
    if current:
        core.authenticate(current, a.server, recipes.CAMPAIGN_ID, proof['source_sha256'],
            [['python', '-m', 'panda_input3_repeat', 'run', '--server', a.server]])
        if current['State']['Running']:
            print('Already queued/running ' + name(a.server)); return
        if not a.resume and not a.check_only:
            raise ValueError('Inspect stopped R2 queue, then use --resume')
    for action in ('check', 'check-sheet'):
        subprocess.run(command(a.server, a.image, dlpan=a.dlpan)
                       + [action, '--server', a.server], check=True)
    if a.check_only: return
    if not finished(a.server):
        active = base.inspect(previous.name(a.server))
        if active and active['State']['Running']:
            previous.authenticate(active, a.server, new=True)
        else:
            # Resume the existing queue through its own authenticated launcher.
            cmd = [sys.executable, '-m', 'panda_input3_original.enqueue', '--server', a.server,
                   '--image', a.image, '--resume']
            if a.dlpan: cmd += ['--dlpan', a.dlpan]
            subprocess.run(cmd, check=True)
    subprocess.run(command(a.server, a.image, dlpan=a.dlpan)
                   + ['publish', '--server', a.server, '--apply'], check=True)
    if current: subprocess.run(['docker', 'start', name(a.server)], check=True)
    else:
        subprocess.run(command(a.server, a.image, gpu=True, detached=True, dlpan=a.dlpan)
                       + ['run', '--server', a.server], check=True)
    print('Added one R2 cycle: A -> B -> C, each fresh 30K; ' + name(a.server), flush=True)


if __name__ == '__main__': main()
