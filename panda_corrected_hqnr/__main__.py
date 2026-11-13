"""Resume existing ZEROQ runs with corrected-PAN selection and reporting."""
import argparse
import json
from panda_eq40_zeroq import plan
from .policy import install, adopt_runtime_policy, POLICY
from .release import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('check', 'check-sheet', 'publish', 'run'))
    p.add_argument('--server', choices=('s1', 's2', 's3'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    result = verify(environment=a.command in ('check', 'run'))
    install()
    if a.command == 'check':
        from panda_eq40_zeroq.assets import sources
        src, _ = sources(a.server)
        result.update(sensor=src.sensor, policy=POLICY, runs=len(plan.queue(a.server)))
    else:
        from panda_eq40_zeroq.publication import publish, destination
        if a.command == 'run' or a.apply:
            if (plan.ROOT / 'gspread/server.txt').read_text().strip() != a.server:
                raise ValueError('Wrong physical server')
            from panda_fr_columns.adapter import GoogleSheets
            GoogleSheets(plan.ROOT / 'gspread/account.json').ensure_layout(destination(a.server),
                plan.campaign_root(a.server) / 'publication/fr_columns', apply=True)
        if a.command == 'run':
            adopt_runtime_policy(a.server)
            from panda_eq40_zeroq.coordinator import run
            result = run(a.server, device=a.device)
        else:
            result = publish(plan.campaign_root(a.server), a.server,
                             apply=a.command == 'publish' and a.apply)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
