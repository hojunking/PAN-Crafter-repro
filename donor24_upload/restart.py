"""Restart only this repo/server's host uploader, never its training process."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from donor24 import plan, publication as P
from sheet_publish_v2.transport import GoogleSheets
from .transport import prepare, TRANSPORT


def publisher_argv(argv, server):
    """Exact Python module commands; never match shell strings or GPU workers."""
    if not argv or not Path(argv[0]).name.startswith('python'):
        return False
    args = list(argv[1:])
    while args and args[0] in ('-u', '-B'):
        args.pop(0)
    if len(args) < 6 or args[0] != '-m' or args[1] not in ('donor24', 'donor24_upload'):
        return False
    if args[2] not in ('publish', 'publish-auto') or '--apply' not in args:
        return False
    try:
        return args.count('--server') == 1 and args[args.index('--server') + 1] == server
    except IndexError:
        return False


def identity(pid):
    proc = Path('/proc', str(pid))
    try:
        if proc.stat().st_uid != os.getuid():
            return None
        argv = [s.decode() for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
        if not argv:
            return None
        # stat field 22; comm can itself contain spaces or parentheses.
        birth = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        return {'pid': pid, 'birth': birth, 'cwd': str((proc / 'cwd').resolve(strict=True)), 'argv': argv}
    except (OSError, UnicodeError, IndexError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, choices=('s1', 's3'))
    parser.add_argument('--credentials', default='gspread/account.json')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    os.chdir(plan.ROOT)
    root = plan.ROOT / 'work_dir/donor24' / plan.CAMPAIGN_ID / args.server / 'WV3'
    if not (root / 'preflight.json').is_file():
        raise RuntimeError('Local donor24 preflight missing; no publisher stopped')
    lock_path = plan.ROOT / 'work_dir/donor24' / ('restart-' + args.server + '.lock')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        value = P.project(root, args.server)
        if value['conflicts']:
            raise ValueError('Evidence/cost conflicts; old publisher left running')
        adapter = GoogleSheets(args.credentials)
        prepare(adapter.snapshot(value['destination']), value, args.server)
        old = []
        for proc in Path('/proc').iterdir():
            if proc.name.isdigit():
                ident = identity(int(proc.name))
                if (ident and ident['cwd'] == str(plan.ROOT) and
                        publisher_argv(ident['argv'], args.server)):
                    old.append(ident)
        if not args.apply:
            print(json.dumps({'action': 'RESTART_DRY_RUN', 'old_pids': [p['pid'] for p in old],
                              'training_untouched': True}))
            return 0
        for ident in old:
            if identity(ident['pid']) != ident:
                raise RuntimeError('Publisher identity changed; refusing signal')
            os.kill(ident['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 15
        while any(identity(p['pid']) == p for p in old):
            if time.monotonic() >= deadline:
                raise RuntimeError('Old uploader did not exit; no SIGKILL or second writer')
            time.sleep(.1)
        log_path = plan.ROOT / 'work_dir/donor24' / ('publisher-bootstrap-' + args.server + '.log')
        command = [sys.executable, '-u', '-m', 'donor24_upload', 'publish-auto',
                   '--server', args.server, '--credentials', str(Path(args.credentials).resolve()), '--apply']
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with log_path.open('ab') as log:
            child = subprocess.Popen(command, cwd=str(plan.ROOT), env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'action': 'PUBLISHER_STARTED', 'old_pids': [p['pid'] for p in old],
                   'pid': child.pid, 'server': args.server, 'at': P.R.utc_now(),
                   'log': str(log_path), 'training_untouched': True, 'readback_verified': False}
        target = root / 'sheet_publish_donor24/restart_receipt.json'
        P.R.atomic_json(target, receipt)
        deadline = time.monotonic() + 55
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError('New uploader exited; inspect ' + str(log_path))
            poll = root / 'sheet_publish_donor24/last_poll.json'
            if poll.exists():
                report = json.loads(poll.read_text())
                if (report.get('publisher_pid') == child.pid and report.get('transport') == TRANSPORT
                        and report.get('readback_verified')):
                    receipt.update(action='PUBLISHER_RESTART_VERIFIED', readback_verified=True,
                                   sheet_rows=report.get('sheet_rows'))
                    P.R.atomic_json(target, receipt)
                    print(json.dumps(receipt))
                    return 0
            time.sleep(1)
        print(json.dumps(receipt))
        print('Uploader running; initial cloud readback still pending. Inspect log/last_poll.', file=sys.stderr)
        return 2
