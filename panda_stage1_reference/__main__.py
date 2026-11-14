import argparse,json
from .report import publish_source
p=argparse.ArgumentParser();p.add_argument('--server',choices=('s1','s2','s3'),required=True)
p.add_argument('--sensor',choices=('WV3','QB','GF2'),required=True);p.add_argument('--apply',action='store_true')
a=p.parse_args();print(json.dumps(publish_source(a.server,a.sensor,apply=a.apply),ensure_ascii=False))
