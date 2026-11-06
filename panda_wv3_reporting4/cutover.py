"""Checkpoint-safe reporting-only transition for the existing s1 W128 campaign."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from directshift48.reporting import atomic_json, utc_now
from panda_wv3_main_rgb.enqueue import inspect_container
from panda_wv3_w128 import plan
from panda_wv3_w128.enqueue import command
from panda_wv3_w128.release import verify

OLD = 'panda-wv3-w128-s1-v1'
NEW = 'panda-wv3-w128-s1-4dp-v1'
ARGS = ['python', '-m', 'panda_wv3_reporting4', 'run', '--kind', 'w128']


def overlay_pin():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    old, new = inspect_container(OLD), inspect_container(NEW)
    if args.status:
        print(json.dumps({name: None if info is None else info['State']
                          for name, info in [(OLD, old), (NEW, new)]})); return
    release = verify()
    root = plan.campaign_root('s1')
    path = root / 'reporting4/cutover.json'
    if new is not None:
        receipt = json.loads(path.read_text())
        if new['Config']['Cmd'] != ARGS or receipt['overlay_pin'] != overlay_pin():
            raise ValueError('Unexpected reporting container or changed overlay')
        if old is not None and old['State']['Running']:
            raise ValueError('Both old and new trainers appear active')
        if new['State']['Running']:
            print('Already running', NEW); return
        if not args.resume:
            raise ValueError('Use --resume for the stopped reporting4 container')
        if args.apply:
            subprocess.run(['docker', 'start', NEW], check=True)
        return
    if old is None or old['Config']['Cmd'] != ['python', '-m', 'panda_wv3_w128', 'run']:
        raise ValueError('Original registered W128 trainer not found')
    labels = old['Config'].get('Labels', {})
    if labels.get('panda.source') != release['source_sha256'] or labels.get('panda.campaign') != plan.CAMPAIGN_ID:
        raise ValueError('Original source/campaign identity differs')
    if not args.apply:
        print(json.dumps({'action': 'GRACEFUL_RESUME_WITH_4DP_REPORTER', 'source': release,
                          'old': OLD, 'new': NEW, 'overlay_pin': overlay_pin()})); return
    if old['State']['Running']:
        subprocess.run(['docker', 'kill', '--signal', 'TERM', OLD], check=True)
        deadline = time.monotonic() + 180
        while inspect_container(OLD)['State']['Running']:
            if time.monotonic() > deadline:
                raise RuntimeError('Waiting for graceful checkpoint save; no forced kill or replacement')
            time.sleep(1)
    state = json.loads((root / 'campaign_state.json').read_text())
    if any(r.get('status') == 'RUNNING' for r in state['runs']):
        raise ValueError('Original trainer did not commit its stopped state')
    from panda_wv3_w128.runtime import latest_checkpoint
    stopped = [r for r in state['runs'] if r.get('status') == 'PAUSED']
    if len(stopped) != 1:
        raise ValueError('Expected exactly one stopped run for reporting cutover')
    checkpoint = latest_checkpoint(root / 'runs' / stopped[0]['run_id'])
    if checkpoint is None:
        raise ValueError('No complete resume generation after graceful stop')
    verify()  # Training pins remain exactly as admitted.
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if processes:
        raise ValueError('Another GPU job is active; leave original experiment safely stopped')
    launch = command(old['Image'])
    launch.remove('--rm')
    launch[2:2] = ['--detach', '--name', NEW, '--label', 'panda.campaign=' + plan.CAMPAIGN_ID,
                  '--label', 'panda.server=s1', '--label', 'panda.source=' + release['source_sha256'],
                  '--label', 'panda.reporting=4dp']
    receipt = {'at': utc_now(), 'old_container': OLD, 'new_container': NEW,
               'source_release': release, 'overlay_pin': overlay_pin(),
               'resume_checkpoint': str(checkpoint), 'stopped_run': stopped[0],
               'training_config_changed': False, 'status': 'PREPARED'}
    atomic_json(path, receipt)
    subprocess.run(launch + ARGS, check=True)
    receipt['status'] = 'STARTED'
    atomic_json(path, receipt)
    print(json.dumps(receipt, default=str))


if __name__ == '__main__':
    main()
