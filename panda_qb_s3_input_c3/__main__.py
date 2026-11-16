"""Existing s3 QB Stage1 -> C R1/R2/R3; no Stage1 retraining or weight transfer."""
import argparse
import sys
from . import plan
from .release import verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('catalog','check','check-sheet','publish','run'))
    p.add_argument('--server',choices=('s3',),default='s3')
    p.add_argument('--device',default='cuda:0');p.add_argument('--apply',action='store_true')
    a=p.parse_args();verify()
    if a.command=='catalog':print(plan.catalog('s3'),end='');return
    if a.command=='run' or a.apply:
        if (plan.ROOT/'gspread/server.txt').read_text().strip()!='s3':
            raise ValueError('Run on physical s3 with its own QB donor')
    if a.command=='run':
        from .predecessor import wait
        if not wait():raise SystemExit(143)
    from .install import install
    install()
    from panda_input3.__main__ import main as shared_main
    sys.argv=['panda_qb_s3_input_c3',a.command,'--server','s3','--device',a.device]
    if a.apply:sys.argv.append('--apply')
    shared_main()


if __name__=='__main__':main()
