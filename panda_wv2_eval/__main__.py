import argparse
import json
from .common import ROOT, OUT, PLAN, verify_release, source_evidence, read, dependency_ready


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['check','check-sheet','project','publish'])
    args=parser.parse_args()
    release=verify_release()
    if args.command=='check':
        from .data import Sources
        from ta2.evaluation import evaluator_identity
        source=Sources();evaluator=evaluator_identity()
        for case in PLAN['cases']:
            source_evidence(case)
            import torch
            saved=torch.load(OUT/'sources'/case['case_id']/'model.pt',map_location='cpu',weights_only=False)
            if saved['pins']['evaluator']!=evaluator:raise ValueError('Frozen source evaluator/environment differs')
        print(json.dumps({'source_release':release,'data_verified':['WV2 RR20','WV2 FR20'],
                          'models':len(PLAN['cases']),'dependency':dependency_ready(read(ROOT/PLAN['dependency']['state_path']))}));return
    from .publication import project,publish
    if args.command=='project':print(json.dumps(project(),ensure_ascii=False))
    else:print(json.dumps(publish(apply=args.command=='publish')))


if __name__=='__main__':main()
