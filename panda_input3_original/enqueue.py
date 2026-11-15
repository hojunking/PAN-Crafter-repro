"""Gracefully replace the publisher, then resume the exact active optimizer state."""
import argparse
import json
import os
import subprocess
import time
from directshift48.reporting import atomic_json
from panda_input3 import enqueue as core
from panda_input3_targets import enqueue as targets
from panda_input3_c30 import enqueue as c30, recipes
from panda_input3_c30.release import verify as training_release
from panda_unified_campaign import enqueue as base
from .release import verify


def name(server): return f'panda-input3-original-pan-{server}-v1'


def command(server, image, *, gpu=False, detached=False, dlpan=None):
    cmd = c30.command(server, image, gpu=gpu, detached=detached, dlpan=dlpan)
    cmd[-1] = 'panda_input3_original'
    if detached:
        cmd[cmd.index('--name') + 1] = name(server)
        i = cmd.index('--label')
        cmd[i:i] = ['--label', 'panda.reporting=' + verify()['source_sha256']]
    return cmd


def authenticate(info, server, *, new=False):
    module = 'panda_input3_original' if new else 'panda_input3_c30'
    core.authenticate(info, server, recipes.CAMPAIGN_ID, training_release()['source_sha256'],
                      [['python', '-m', module, 'run', '--server', server]])
    if new and info['Config']['Labels'].get('panda.reporting') != verify()['source_sha256']:
        raise ValueError('Reporting container release differs')


def pause(container, server, *, new=False):
    info = base.inspect(container)
    if not info: return
    authenticate(info, server, new=new)
    if info['State']['Running']:
        subprocess.run(['docker', 'kill', '--signal=SIGTERM', container], check=True)
        deadline = time.monotonic() + 240
        while base.inspect(container)['State']['Running']:
            if time.monotonic() > deadline: raise RuntimeError('Still saving checkpoint; never force-kill')
            time.sleep(2)
    subprocess.run(['docker', 'update', '--restart=no', container], check=True, capture_output=True)
    from panda_donor_read_guard.guard import sha
    for root in (recipes.ORIGINAL_ROOT(server), recipes.campaign_root(server)):
        path = root / 'campaign_state.json'
        if not path.exists(): continue
        state = json.loads(path.read_text())
        for job in state['runs']:
            if job['status'] == 'PAUSED' and job['step']:
                meta = json.loads((root / 'runs' / job['run_id'] / 'latest.json').read_text())
                if meta['step'] != job['step'] or sha(recipes.original.local_path(meta['path'])) != meta['file_sha256']:
                    raise ValueError('Graceful-stop checkpoint/state mismatch')
        atomic_json(root / 'publication/original_pan/cutover_state.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', choices=('s1', 's2', 's3'), required=True)
    p.add_argument('--image', default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan', default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--check-only', action='store_true'); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); verify()
    if (recipes.ROOT / 'gspread/server.txt').read_text().strip() != a.server:
        raise ValueError('Execute on the matching physical server')
    from panda_input3_c30.transition import read_cutover
    read_cutover(a.server)
    current = base.inspect(name(a.server))
    if current:
        authenticate(current, a.server, new=True)
        if current['State']['Running']:
            print('Already running ' + name(a.server)); return
        if not a.resume and not a.check_only:
            raise ValueError('Inspect stopped runner, then use --resume')
    previous = base.inspect(c30.name(a.server))
    if previous: authenticate(previous, a.server)
    for campaign in ('original', 'c30'):
        for action in ('check', 'check-sheet'):
            subprocess.run(command(a.server, a.image, dlpan=a.dlpan)
                           + [action, '--server', a.server, '--campaign', campaign], check=True)
    if a.check_only: return
    pause(c30.name(a.server), a.server)
    targets.pause(targets.name(a.server), a.server)
    targets.pause(core.name(a.server), a.server)
    # Old publishers are now stopped. Re-select both namespaces from retained
    # original-PAN metrics and remove only verified obsolete owned result rows.
    for campaign in ('original', 'c30'):
        subprocess.run(command(a.server, a.image, dlpan=a.dlpan)
            + ['publish', '--server', a.server, '--campaign', campaign, '--apply'], check=True)
    if current: subprocess.run(['docker', 'start', name(a.server)], check=True)
    else:
        subprocess.run(command(a.server, a.image, gpu=True, detached=True, dlpan=a.dlpan)
                       + ['run', '--server', a.server], check=True)
    print('Resumed unchanged INPUT3/C30 training; ORIGINAL PAN is the selection reference: ' + name(a.server))


if __name__ == '__main__': main()
