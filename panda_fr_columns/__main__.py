"""Run an unchanged experiment with the FR67 reporting adapter installed."""
import argparse
import importlib
import json
import runpy
import sys
from .adapter import GoogleSheets
from .publication import install

ALLOWED={'panda_wv3_s2_eq40':('s1',),'panda_sensor_eq40':('s2','s3'),
         'panda_wv3_tab3_rgbn':('s3',)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',choices=tuple(ALLOWED),required=True)
    a,rest=p.parse_known_args()
    s=argparse.ArgumentParser(add_help=False);s.add_argument('--server',required=True)
    parsed,_=s.parse_known_args(rest)
    if parsed.server not in ALLOWED[a.package]:raise ValueError('Package/server mismatch')
    plan=importlib.import_module(a.package+'.plan')
    importlib.import_module(a.package+'.release').verify()
    install()
    if 'run' in rest or ('publish' in rest and '--apply' in rest):
        marker=plan.ROOT/'gspread/server.txt'
        if marker.read_text().strip()!=parsed.server:raise ValueError('Wrong physical server')
        pub=importlib.import_module(a.package+'.publication')
        adapter=GoogleSheets(plan.ROOT/'gspread/account.json')
        print(json.dumps(adapter.ensure_layout(pub.destination(parsed.server),
            plan.campaign_root(parsed.server)/'publication/fr_columns',apply=True)),flush=True)
    sys.argv=[a.package]+rest
    runpy.run_module(a.package,run_name='__main__')


if __name__=='__main__':main()
