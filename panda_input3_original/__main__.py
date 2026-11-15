"""Run the unchanged INPUT3/C30 queue with original-PAN publication in every worker."""
import argparse
import signal
import subprocess
import sys
from .release import verify
from panda_input3_c30 import recipes, transition


def supervise(server, device):
    transition.read_cutover(server)
    stopped = [False]; child = [None]

    def forward(signum, frame):
        stopped[0] = True
        if child[0] is not None and child[0].poll() is None: child[0].send_signal(signum)

    previous = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for action in ('finish-original', 'train'):
            if stopped[0]: return 143
            if action == 'train' and not transition.original_finished(server): return 143
            child[0] = subprocess.Popen([sys.executable, '-m', 'panda_input3_original', action,
                                        '--server', server, '--device', device])
            code = child[0].wait()
            if code: return code
        return 143 if stopped[0] else 0
    finally:
        for s, handler in previous.items(): signal.signal(s, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('check', 'check-sheet', 'publish', 'run', 'finish-original', 'train'))
    p.add_argument('--server', choices=('s1', 's2', 's3'), required=True)
    p.add_argument('--campaign', choices=('original', 'c30'), default='c30')
    p.add_argument('--device', default='cuda:0'); p.add_argument('--apply', action='store_true')
    a = p.parse_args(); verify()
    if a.command in ('run', 'finish-original', 'train') or a.apply:
        if (recipes.ROOT / 'gspread/server.txt').read_text().strip() != a.server:
            raise ValueError('Run on the matching physical server')
    if a.command == 'run': raise SystemExit(supervise(a.server, a.device))
    if a.command == 'train' and not transition.original_finished(a.server):
        raise ValueError('Finish the started original jobs before C30')
    if a.command != 'finish-original' and (a.command == 'train' or a.campaign == 'c30'):
        from panda_input3_c30.install import install
        install()
    from .reporting import install
    install()
    if a.command == 'finish-original':
        from panda_input3_c30.__main__ import finish_original
        finish_original(a.server, a.device); return
    from panda_input3.__main__ import main as shared_main
    sys.argv = ['panda_input3_original', 'run' if a.command == 'train' else a.command,
                '--server', a.server, '--device', a.device] + (['--apply'] if a.apply else [])
    shared_main()


if __name__ == '__main__': main()
