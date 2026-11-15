"""Finish started 40/50K jobs, then execute fresh 30K jobs in the recorded order."""
import argparse
import json
import signal
import subprocess
import sys
from directshift48.reporting import atomic_json, utc_now
from . import recipes
from .release import verify
from .transition import read_cutover, next_original, original_finished


def finish_original(server, device):
    manifest = read_cutover(server)
    from panda_input3_targets.reporting import install
    install()
    from panda_input3 import runtime
    runtime.next_job = lambda state: next_original(state, manifest)
    if original_finished(server):
        from panda_input3.publication import publish
        publish(recipes.ORIGINAL_ROOT(server), server, apply=True)
        return
    from panda_input3.__main__ import main
    sys.argv = ['panda_input3_targets', 'run', '--server', server, '--device', device]
    main()
    if original_finished(server):
        path = recipes.ORIGINAL_ROOT(server) / 'campaign_state.json'
        state = json.loads(path.read_text())
        state.update(status='ORIGINAL_RUNS_FINISHED_PENDING_REPLACED_BY_C30', updated_at=utc_now())
        atomic_json(path, state)


def supervise(server, device):
    read_cutover(server)
    stopped = [False]; child = [None]

    def forward(signum, frame):
        stopped[0] = True
        if child[0] is not None and child[0].poll() is None:
            child[0].send_signal(signum)

    previous = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for action in ('finish-original', 'train'):
            if stopped[0]: return 143
            if action == 'train' and not original_finished(server): return 143
            child[0] = subprocess.Popen([sys.executable, '-m', 'panda_input3_c30', action,
                                         '--server', server, '--device', device])
            code = child[0].wait()
            if code: return code
        return 143 if stopped[0] else 0
    finally:
        for signum, handler in previous.items(): signal.signal(signum, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('catalog', 'check', 'check-sheet', 'publish',
                                      'activate', 'finish-original', 'train', 'run'))
    p.add_argument('--server', choices=('s1', 's2', 's3'), required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--apply', action='store_true')
    a = p.parse_args(); verify()
    if a.command in ('activate', 'finish-original', 'train', 'run') or a.apply:
        if (recipes.ROOT / 'gspread/server.txt').read_text().strip() != a.server:
            raise ValueError('Run on the matching physical server; no donor transfer')
    if a.command == 'activate':
        from .transition import prepare
        print(json.dumps(prepare(a.server))); return
    if a.command == 'run':
        raise SystemExit(supervise(a.server, a.device))
    if a.command == 'finish-original':
        finish_original(a.server, a.device); return
    if a.command == 'train':
        if not original_finished(a.server):
            raise ValueError('Finish the already-started original run before C30')
    from .install import install
    install()
    from panda_input3.__main__ import main as shared_main
    sys.argv = ['panda_input3_c30', 'run' if a.command == 'train' else a.command,
                '--server', a.server, '--device', a.device] + (['--apply'] if a.apply else [])
    shared_main()


if __name__ == '__main__': main()
