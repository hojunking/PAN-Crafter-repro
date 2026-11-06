import argparse
from contextlib import nullcontext
import json

from .publication import install


def main():
    parser = argparse.ArgumentParser(description='Four-decimal WV3 reporting without changing pinned training code')
    parser.add_argument('command', choices=['project', 'publish', 'run'])
    parser.add_argument('--kind', choices=['rgb', 'w128'], required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    module = install(args.kind)
    if args.command == 'run':
        if args.kind != 'w128':
            parser.error('Only the currently active W128 trainer is supported')
        from panda_wv3_w128.runtime import run
        result = run('s1', device='cuda:0', publish=True)
    else:
        context = nullcontext()
        if args.kind == 'rgb':
            from checkpoint_cleanup.ledger import Ledger
            from checkpoint_cleanup.publication import adapted
            context = adapted('panda_wv3_main_rgb', Ledger())
        with context:
            root = module.plan.campaign_root('s1')
            if args.command == 'publish':
                result = module.publish(root, 's1', apply=args.apply)
            else:
                value = module.project(root, 's1')
                for row in value['rows']:
                    module.verify_owned(row, value, 's1')
                result = {k: value[k] for k in ('rows', 'notes', 'view_hash', 'destination')}
    if args.output:
        from directshift48.reporting import atomic_json
        atomic_json(args.output, result)
        print(json.dumps({'output': args.output, 'status': 'VERIFIED'}))
    else:
        print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
