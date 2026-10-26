"""Read-only predecessor gate; never prune, stop, or rewrite TUNE30 work."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from panda_tune30 import plan as previous
from . import plan

PREVIOUS_CAMPAIGN = previous.CAMPAIGN_ID


def _local(path, root):
    value = Path(path)
    if str(value).startswith('/workspace/'):
        value = root / value.relative_to('/workspace')
    elif not value.is_absolute():
        value = root / value
    return value


def _file_sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def predecessor_receipt(root, server):
    root = Path(root).resolve()
    previous.baseline(server)
    directory = root/'work_dir/panda_tune30'/PREVIOUS_CAMPAIGN/server/'WV3'
    path = directory/'campaign_state.json'
    if not path.is_file():
        raise ValueError('WAITING_TUNE30_STATE: no predecessor state; nothing started')
    raw = path.read_bytes()
    state = json.loads(raw)
    if state.get('status') not in ('FINISHED_REGISTERED_QUEUE','FINISHED_WITH_BUDGET_SKIPS'):
        raise ValueError('WAITING_TUNE30_COMPLETE: '+str(state.get('status')))
    expected = [previous.resolve_case(server,c,r) for g in previous.groups(server) for c,r in g['runs']]
    runs = state.get('runs',[])
    if len({r['run_id'] for r in runs}) != len(runs):
        raise ValueError('Duplicate predecessor runs')
    by_id = {r['run_id']:r for r in runs}
    skipped = set()
    if state['status'] == 'FINISHED_WITH_BUDGET_SKIPS':
        for group in previous.groups(server):
            rows = [r for r in state.get('groups',[]) if r.get('group') == group['name']]
            if not rows or rows[-1].get('admitted') is not False:
                continue
            row = rows[-1]
            ids = {previous.resolve_case(server,c,r)['run_id'] for c,r in group['runs']}
            if (row.get('reason') != 'INSUFFICIENT_30H_BUDGET'
                    or set(row.get('pending',[])) != ids or ids & set(by_id)):
                raise ValueError('Invalid predecessor budget skip evidence')
            skipped.update(ids)
        if not skipped:
            raise ValueError('Budget terminal state without explicit skipped groups')
    if set(by_id) | skipped != {c['run_id'] for c in expected}:
        raise ValueError('TUNE30 queue incomplete or amended; explicit review required')
    receipts = []
    for cfg in expected:
        if cfg['run_id'] in skipped:
            continue
        run = by_id[cfg['run_id']]
        if run.get('status')!='COMPLETED' or run.get('step')!=cfg['planned_horizon']:
            raise ValueError('TUNE30 run did not reach its registered horizon')
        attempt = _local(run['directory'],root)
        intended = directory/'runs'/cfg['run_id']/'attempt_01'
        if attempt.resolve()!=intended.resolve():
            raise ValueError('TUNE30 attempt is outside its registered run')
        evidence = attempt/f"evaluation_{cfg['planned_horizon']:07d}.json"
        metadata = attempt/'models'/f"step_{cfg['planned_horizon']:07d}.json"
        checkpoint = metadata.with_suffix('.pt')
        if not all(p.is_file() and not p.is_symlink() for p in (evidence,metadata,checkpoint)):
            raise ValueError('Missing final TUNE30 evaluation/checkpoint evidence')
        ev, sidecar = json.loads(evidence.read_text()), json.loads(metadata.read_text())
        qual, artifact = ev.get('qualification',{}), sidecar.get('artifact',{})
        if (not qual.get('same_state_verified') or not qual.get('metrics_complete')
                or not qual.get('metrics_finite') or not artifact.get('checkpoint_verified')
                or artifact.get('completed_update')!=cfg['planned_horizon']
                or _file_sha(checkpoint)!=artifact.get('file_sha256')
                or artifact.get('model_state_sha256')!=ev.get('state_before',{}).get('model_state_sha256')
                or sidecar.get('config',{}).get('run_id')!=cfg['run_id']):
            raise ValueError('Invalid same-state final TUNE30 evidence')
        receipts.append({'run_id':cfg['run_id'],'step':cfg['planned_horizon'],
                         'checkpoint_sha256':artifact['file_sha256'],
                         'evaluation_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()})
    return {'schema':'PANDA_HARD36_COMPLETE_TUNE30_GATE_V1','server':server,
            'predecessor_campaign':PREVIOUS_CAMPAIGN,'state_sha256':hashlib.sha256(raw).hexdigest(),
            'runs':receipts,'predecessor_budget_skips':sorted(skipped),
            'successor_skips':[], 'predecessor_mutated':False}


def container_gate(server):
    name=f'panda-tune30-hardonly-wv3-{server}-v1'
    result=subprocess.run(['docker','inspect','--format','{{json .State}}',name],capture_output=True,text=True)
    if result.returncode:
        raise ValueError('Cannot verify predecessor container exit: '+name)
    state=json.loads(result.stdout)
    if state.get('Running') or state.get('Status')!='exited' or state.get('ExitCode')!=0:
        raise ValueError('Predecessor container must exit normally; nothing stopped or replaced')
    return {'name':name,'status':'exited','exit_code':0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s1','s3'),required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--check-container',action='store_true')
    args=parser.parse_args(argv)
    try:
        receipt=predecessor_receipt(plan.ROOT,args.server)
        if args.check_container:
            receipt['container']=container_gate(args.server)
        print(json.dumps({'status':'READY','receipt':receipt}))
        return 0
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps({'status':'WAITING_OR_INVALID_PREDECESSOR','reason':str(exc),'mutated':False}))
        return 2 if args.require_complete else 0


if __name__=='__main__':
    raise SystemExit(main())
