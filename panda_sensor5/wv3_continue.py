"""Authenticate deferred TUNE completion without rewriting HARD36 admission.

HARD36 may have started while TUNE30 had explicit budget skips. Completing those
skipped TUNE runs later changes the predecessor receipt, although no HARD model,
data, donor, or existing TUNE evidence changed. This bridge retains the original
HARD admission and records the authenticated additional predecessor evidence in
an independent receipt. All other original admission checks still run.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

from directshift48.reporting import atomic_json, utc_now
from .transition import ROOT, campaign_root

SCHEMA='SENSOR5_HARD36_DEFERRED_TUNE_CONTINUATION_V1'


def bind_entrypoint(directory):
    from .wv3 import adapter_pin
    proof={'schema':'SENSOR5_HARD36_CONTINUATION_ENTRYPOINT_PIN_V1',
        'bridge_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wv3_adapter_pin':adapter_pin()}
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=directory/'sensor5_continuation_entrypoint.json'
    if path.exists():
        if json.loads(path.read_text())!=proof:
            raise ValueError('HARD36 continuation entrypoint source pin changed')
    else:
        atomic_json(path,proof)
    return proof


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def reconcile_predecessor(old,current):
    """Only added completion of previously explicit skips may alter the proof."""
    if old==current:
        return []
    expected_schema='PANDA_HARD36_COMPLETE_TUNE30_GATE_V1'
    if old.get('schema')!=expected_schema or current.get('schema')!=expected_schema:
        raise ValueError('Unknown HARD predecessor receipt schema')
    allowed={'state_sha256','runs','predecessor_budget_skips'}
    if {k:v for k,v in old.items() if k not in allowed}!={k:v for k,v in current.items() if k not in allowed}:
        raise ValueError('HARD predecessor identity/policy changed')
    if (old.get('successor_skips')!=[] or current.get('successor_skips')!=[]
            or old.get('predecessor_mutated') is not False
            or current.get('predecessor_mutated') is not False
            or current.get('predecessor_budget_skips')!=[]):
        raise ValueError('Deferred TUNE must now have its complete unchanged registry')
    old_runs=old.get('runs',[]); current_runs=current.get('runs',[])
    before={r['run_id']:r for r in old_runs}; after={r['run_id']:r for r in current_runs}
    if len(before)!=len(old_runs) or len(after)!=len(current_runs):
        raise ValueError('Duplicate predecessor run proof')
    if any(after.get(k)!=v for k,v in before.items()):
        raise ValueError('Previously completed TUNE checkpoint/evaluation proof changed')
    extra=set(after)-set(before)
    skips=old.get('predecessor_budget_skips',[])
    if len(set(skips))!=len(skips) or extra!=set(skips) or set(skips)&set(before):
        raise ValueError('Newly completed TUNE runs must exactly match original budget skips')
    return sorted(extra)


def install_bridge(server,*,root=ROOT):
    from panda_hard36 import transition
    original=transition.predecessor_receipt
    directory=campaign_root(root,'panda_hard36',server)
    admission_path=directory/'admission.json'
    if not admission_path.exists():
        return None
    with (directory/'.controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=admission_path.read_bytes(); admission=json.loads(raw)
        old=admission.get('immutable_predecessor_migration')
        state=json.loads((directory/'campaign_state.json').read_text())
        if not isinstance(old,dict) or state.get('migration_receipt')!=old:
            raise ValueError('Saved HARD state does not authenticate its original predecessor admission')
        current=original(root,server)
        added=reconcile_predecessor(old,current)
        if old==current:
            return None
        # This process still invokes the original admission implementation,
        # which compares config, code, source, evaluator, environment and donor.
        proof={'schema':SCHEMA,'server':server,
            'policy':'USER_20261026_SENSOR5_THEN_ALL_REMAINING_WV3',
            'original_admission_sha256':hashlib.sha256(raw).hexdigest(),
            'original_predecessor':old,'current_predecessor':current,
            'additional_completed_tune_runs':added,
            'hard_admission_preserved':True,'all_other_admission_checks_unchanged':True,
            'bridge_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        path=directory/'sensor5_predecessor_continuation.json'
        if path.exists():
            existing=json.loads(path.read_text())
            if {k:v for k,v in existing.items() if k!='created_at'}!=proof:
                raise ValueError('Existing HARD continuation bridge evidence changed')
        else:
            atomic_json(path,dict(proof,created_at=utc_now()))
    def authenticated_predecessor(call_root,call_server):
        if Path(call_root).resolve()!=Path(root).resolve() or call_server!=server:
            raise ValueError('Continuation bridge requested for a different server/root')
        # Revalidate immutable current checkpoint proofs at the actual admission.
        if original(call_root,call_server)!=current:
            raise ValueError('TUNE evidence changed after continuation validation')
        if admission_path.read_bytes()!=raw:
            raise ValueError('Original HARD admission changed before continuation')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=proof['bridge_source_sha256']:
            raise ValueError('HARD continuation bridge source changed before admission')
        return old
    transition.predecessor_receipt=authenticated_predecessor
    return proof


def main(argv=None):
    from . import wv3
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv.count('--package')!=1 or argv.count('--server')!=1:
        raise ValueError('Explicit package/server required for continuation')
    package=argv[argv.index('--package')+1]
    server=argv[argv.index('--server')+1]
    if package!='panda_hard36' or server not in ('s1','s3'):
        raise ValueError('Continuation bridge is registered only for HARD36')
    bind_entrypoint(campaign_root(ROOT,package,server))
    install_bridge(server)
    return wv3.main(argv)


if __name__=='__main__':
    raise SystemExit(main())
