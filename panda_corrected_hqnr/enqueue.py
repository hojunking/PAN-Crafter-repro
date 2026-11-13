"""Gracefully replace only the same ZEROQ runner; preserve its full training state."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from panda_eq40_zeroq import plan, enqueue as old
from panda_unified_campaign.enqueue import inspect
from .release import verify


def name(server):
    return old.name(server) + '-corrected'


def command(server, image, *, gpu=False, detached=False, dlpan=None):
    cmd = old.command(server, image, gpu=gpu, detached=detached, dlpan=dlpan)
    if detached:
        cmd[cmd.index('--name') + 1] = name(server)
        cmd[-1:-1] = ['--label', 'panda.selection=' + verify()['policy_release']]
    return cmd + ['python', '-m', 'panda_corrected_hqnr']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', choices=('s1', 's2', 's3'), required=True)
    p.add_argument('--image', default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan', default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--check-only', action='store_true'); p.add_argument('--wait', action='store_true')
    a = p.parse_args(); release = verify(); existing = inspect(name(a.server))
    if existing:
        labels = existing['Config'].get('Labels') or {}
        expected = {'panda.source': release['training_release'], 'panda.selection': release['policy_release'],
                    'panda.campaign': plan.CAMPAIGN_ID, 'panda.server': a.server}
        if any(labels.get(k) != v for k, v in expected.items()):
            raise ValueError('Corrected runner identity changed')
        mounts = [m['Source'] for m in existing['Mounts'] if m['Destination'] == '/workspace']
        if len(mounts) != 1 or Path(mounts[0]).resolve() != plan.ROOT.resolve():
            raise ValueError('Wrong checkout')
        if existing['Config']['Cmd'] != ['python', '-m', 'panda_corrected_hqnr', 'run', '--server', a.server]:
            raise ValueError('Unexpected corrected runner command')
        if existing['State']['Running']:
            print('Already running ' + name(a.server)); return
    for action in ('check', 'check-sheet'):
        subprocess.run(command(a.server, a.image, dlpan=a.dlpan) + [action, '--server', a.server], check=True)
    if a.check_only:
        return
    if (plan.ROOT / 'gspread/server.txt').read_text().strip() != a.server:
        raise ValueError('Wrong physical server')
    previous = inspect(old.name(a.server))
    if previous and previous['State']['Running']:
        old.authenticate(previous, a.server, 'panda_eq40_zeroq')
        subprocess.run(['docker', 'kill', '--signal=TERM', old.name(a.server)], check=True)
        deadline = time.monotonic() + 300
        while inspect(old.name(a.server))['State']['Running']:
            if time.monotonic() > deadline:
                raise RuntimeError('Graceful checkpoint pending; no forced stop')
            time.sleep(2)
    old.stop_predecessors(a.server)
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
        if not a.wait:
            raise RuntimeError('Unrelated GPU work; use --wait')
        print('Waiting for unrelated GPU work', flush=True); time.sleep(30)
    if existing:
        subprocess.run(['docker', 'start', name(a.server)], check=True)
    else:
        subprocess.run(command(a.server, a.image, gpu=True, detached=True, dlpan=a.dlpan)
                       + ['run', '--server', a.server], check=True)
    print('Started ' + name(a.server) + '; same checkpoints/optimizer/RNG, corrected-PAN selection')


if __name__ == '__main__':
    main()
