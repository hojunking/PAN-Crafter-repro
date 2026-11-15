"""Run the second cycle after the current cycle has finished."""
import argparse
import sys
from .release import verify
from . import recipes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('catalog', 'check', 'check-sheet', 'publish', 'run'))
    p.add_argument('--server', choices=('s1', 's2', 's3'), required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--apply', action='store_true')
    a = p.parse_args(); verify()
    if a.command == 'run' or a.apply:
        if (recipes.ROOT / 'gspread/server.txt').read_text().strip() != a.server:
            raise ValueError('Execute on matching physical server; keep the local donor')
    if a.command == 'run':
        from .predecessor import wait
        if not wait(a.server): raise SystemExit(143)
    from .install import install
    install()
    from panda_input3.__main__ import main as shared_main
    sys.argv = ['panda_input3_repeat', a.command, '--server', a.server, '--device', a.device]
    if a.apply: sys.argv.append('--apply')
    shared_main()


if __name__ == '__main__': main()
