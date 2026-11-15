"""Install a resumable handoff: finish started originals, replace pending jobs."""
import argparse
import os
import subprocess
from panda_input3 import enqueue as original
from panda_input3_targets import enqueue as targets
from panda_unified_campaign import enqueue as base
from . import recipes
from .release import verify


def name(server): return f'panda-input3-c30-{server}-v1'


def command(server, image, *, gpu=False, detached=False, dlpan=None):
    cmd = original.command(server, image, gpu=gpu, detached=detached, dlpan=dlpan)
    output = recipes.ROOT / 'work_dir/panda_input3_c30'; output.mkdir(parents=True, exist_ok=True)
    cmd[-4:-4] = ['--mount', f'type=bind,src={output},dst=/workspace/work_dir/panda_input3_c30']
    cmd[-1] = 'panda_input3_c30'
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
        raise ValueError('Execute on the matching physical server')
    current = base.inspect(name(a.server))
    if current:
        original.authenticate(current, a.server, recipes.CAMPAIGN_ID, proof['source_sha256'],
            [['python', '-m', 'panda_input3_c30', 'run', '--server', a.server]])
        if current['State']['Running']:
            print('Already running ' + name(a.server)); return
        if not a.resume and not a.check_only:
            raise ValueError('Inspect the stopped C30 runner, then use --resume')
    for container, new in ((original.name(a.server), False), (targets.name(a.server), True)):
        info = base.inspect(container)
        if info: targets.authenticate(info, a.server, new=new)
    for action in ('check', 'check-sheet'):
        subprocess.run(command(a.server, a.image, dlpan=a.dlpan) + [action, '--server', a.server], check=True)
    if a.check_only: return
    # Briefly save/restart the active optimizer solely to install the new queue
    # dispatcher. It retains its original config, RNG, stream and 40/50K endpoint.
    targets.pause(targets.name(a.server), a.server)
    targets.pause(original.name(a.server), a.server)
    subprocess.run(command(a.server, a.image, dlpan=a.dlpan) + ['activate', '--server', a.server], check=True)
    subprocess.run(command(a.server, a.image, dlpan=a.dlpan) + ['publish', '--server', a.server, '--apply'], check=True)
    if current:
        subprocess.run(['docker', 'start', name(a.server)], check=True)
    else:
        subprocess.run(command(a.server, a.image, gpu=True, detached=True, dlpan=a.dlpan)
                       + ['run', '--server', a.server], check=True)
    print('Started ' + name(a.server) + ': finish originals -> pending C30 replacements -> fresh C30 reruns')


if __name__ == '__main__': main()
