"""Explicit, audited code re-admission only before any registered training run.

This is not a general checkpoint/code-pin override. It preserves the original
30h clock, source replay evidence and failed admission in an audit record.
"""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time

from directshift48.reporting import atomic_json, utc_now
from directshift48.runtime import environment_manifest
from panda_stage2_condq.training import configure_precision
from ta2.evaluation import evaluator_identity
from . import plan
from .runtime import code_pin


def recover(root, *, reason, device='cuda'):
    root=Path(root).resolve(strict=True)
    if not reason.strip(): raise ValueError('Explicit recovery reason required')
    with (root/'.controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((root/'campaign_state.json').read_text())
        admission=json.loads((root/'admission.json').read_text())
        if (state.get('status')!='FAILED' or state.get('runs') or state.get('timing')
                or any((root/'runs').glob('**/*.pt')) or 'M' in state.get('donors',{})):
            raise ValueError('Recovery is only allowed before registered training/pilots complete')
        configure_precision()
        config=admission['config']
        if (config!=json.loads(json.dumps(plan.default_config(config['server'])))
                or admission['evaluator']!=evaluator_identity()
                or admission['environment']!=environment_manifest(device)):
            raise ValueError('Cannot change scientific config/evaluator/environment during prep repair')
        current=code_pin()
        changed={k for k in set(current)|set(admission['code']) if current.get(k)!=admission['code'].get(k)}
        if not changed or any(not (k.startswith('panda30r/') or k.startswith('scripts/run_panda30r')) for k in changed):
            raise ValueError('Only isolated Panda30R launch implementation changes may be re-admitted')
        backup=root/'preparation_recovery'/f'{time.time_ns()}.json'
        atomic_json(backup,{'at':utc_now(),'reason':reason,'old_admission':admission,
            'old_state':deepcopy(state),'changed_code_files':sorted(changed),'new_code':current,
            'original_started_unix_preserved':state['started_unix']})
        admission['code']=current
        atomic_json(root/'admission.json',admission)
        state.update(status='READY_TO_RESUME_PREPARATION',
            elapsed_seconds=max(state.get('elapsed_seconds',0.),time.time()-state['started_unix']))
        state.setdefault('preparation_recovery',[]).append({'at':utc_now(),'reason':reason,'audit_path':str(backup)})
        state.pop('error',None); state.pop('traceback',None)
        atomic_json(root/'campaign_state.json',state)
        return {'status':state['status'],'changed_code_files':sorted(changed),
                'original_started_unix':state['started_unix'],'audit_path':str(backup)}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--reason',required=True)
    args=parser.parse_args()
    print(json.dumps(recover(args.root,reason=args.reason)))
