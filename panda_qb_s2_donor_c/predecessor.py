"""Require completed s2 INPUT3 runs; never restart or cancel a held campaign."""
from contextlib import ExitStack
import fcntl
import json
from . import plan

PREVIOUS=(
    ('original','panda_input3','PANDA_S123_INPUT3_20261115_v1'),
    ('c30','panda_input3_c30','PANDA_S123_INPUT3_C30_20261115_v1'),
    ('repeat','panda_input3_repeat','PANDA_S123_INPUT3_C30_R2_20261116_v1'))


def roots():
    return [(name,plan.ROOT/'work_dir'/package/campaign/'s2/QB') for name,package,campaign in PREVIOUS]


def require_finished():
    with ExitStack() as stack:
        for (name,root),(_,_,campaign) in zip(roots(),PREVIOUS):
            state=json.loads((root/'campaign_state.json').read_text())
            if state.get('campaign_id')!=campaign or state.get('server')!='s2':raise ValueError('Wrong predecessor identity')
            rows=state.get('runs',[])
            allowed={'COMPLETED','REPLACED_BY_C30'} if name=='original' else {'COMPLETED'}
            prefix='INPUT3' if name=='original' else 'INPUT3C30'
            repeat=2 if name=='repeat' else 1
            expected={f'{prefix}_QB_S2_{case}_R{repeat}_S{261018100+repeat}'
                      for case in ('A_RAW','B_RAW_CORRECTED','C_CORRECTED_DIFF')}
            if (len(rows)!=3 or {r['run_id'] for r in rows}!=expected
                    or any(r['status'] not in allowed for r in rows)):
                raise ValueError('Finish the existing s2 INPUT3 queue first: '+name)
            lock=stack.enter_context((root/'runner.lock').open('r'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return dict(status='PREVIOUS_INPUT3_COMPLETED',campaigns=[v[2] for v in PREVIOUS])
