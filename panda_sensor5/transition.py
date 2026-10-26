"""Read-only proof that the previously active WV3 run finished before SENSOR5."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            value.update(block)
    return value.hexdigest()


def campaign_root(root,package,server):
    plan=importlib.import_module(package+'.plan')
    dataset=plan.dataset_for_server(server) if package=='panda_sensor5' else 'WV3'
    return Path(root)/'work_dir'/package/plan.CAMPAIGN_ID/server/dataset


def state_of(root,package,server):
    path=campaign_root(root,package,server)/'campaign_state.json'
    return json.loads(path.read_text()) if path.exists() else None


def local(path,root):
    path=Path(path)
    return Path(root)/path.relative_to('/workspace') if str(path).startswith('/workspace/') else path


def completed_run(root,package,server,run_id):
    plan=importlib.import_module(package+'.plan')
    configs=[plan.resolve_case(server,c,r) for g in plan.groups(server) for c,r in g['runs']]
    cfgs=[c for c in configs if c['run_id']==run_id]
    if len(cfgs)!=1:
        raise ValueError('Unregistered or duplicate cutover run')
    cfg=cfgs[0]
    state=state_of(root,package,server)
    runs=[] if state is None else [r for r in state['runs'] if r['run_id']==run_id]
    if len(runs)!=1 or runs[0]['status']!='COMPLETED' or runs[0]['step']!=cfg['planned_horizon']:
        raise ValueError('Current WV3 run has not completed its registered horizon')
    run=runs[0]
    directory=local(run['directory'],root)
    expected=campaign_root(root,package,server)/'runs'/run_id/'attempt_01'
    if directory.resolve()!=expected.resolve():
        raise ValueError('Current WV3 run is outside its registered output')
    horizon=cfg['planned_horizon']
    evidence=directory/f'evaluation_{horizon:07d}.json'
    metadata=directory/'models'/f'step_{horizon:07d}.json'
    checkpoint=metadata.with_suffix('.pt')
    for path in (evidence,metadata,checkpoint):
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing final current-run checkpoint/evaluation')
    ev,meta=json.loads(evidence.read_text()),json.loads(metadata.read_text())
    q,a=ev.get('qualification',{}),meta.get('artifact',{})
    if (not all(q.get(k) for k in ('same_state_verified','metrics_complete','metrics_finite'))
            or not a.get('checkpoint_verified') or a.get('completed_update')!=horizon
            or file_sha(checkpoint)!=a.get('file_sha256')
            or a.get('model_state_sha256')!=ev.get('state_before',{}).get('model_state_sha256')
            or meta.get('config',{}).get('run_id')!=run_id):
        raise ValueError('Invalid final current-run evidence')
    return {'package':package,'run_id':run_id,'step':horizon,
            'checkpoint_sha256':a['file_sha256'],'evaluation_sha256':file_sha(evidence)}


def predecessor_receipt(root,server):
    root=Path(root)
    directory=root/'work_dir/panda_sensor5_launchers'/server
    initial=json.loads((directory/'initial_cutover.json').read_text())
    package=initial.get('package','panda_tune30')
    target=initial.get('target_run_id')
    if target:
        proof=completed_run(root,package,server,target)
    else:
        # A server whose queue already ended has no current run to interrupt.
        state=state_of(root,package,server)
        if state is None or state['status'] not in ('FINISHED_REGISTERED_QUEUE','FINISHED_WITH_BUDGET_SKIPS'):
            raise ValueError('No active cutover target and predecessor is not terminal')
        proof={'package':package,'already_terminal_before_sensor5':True,
               'initial_status':initial['initial_status']}
    return {'schema':'SENSOR5_COMPLETED_CURRENT_WV3_RUN_GATE_V1','server':server,
        'proof':proof,'request':'USER_20261026_DATASET5_THEN_RESUME_WV3',
        'successor_skips':[], 'remaining_wv3_deferred_not_cancelled':True}


def container_gate(server,*,root=ROOT):
    directory=Path(root)/'work_dir/panda_sensor5_launchers'/server
    initial=json.loads((directory/'initial_cutover.json').read_text())
    package=initial.get('package','panda_tune30'); short=package.removeprefix('panda_')
    if initial.get('target_run_id'):
        name=json.loads((directory/f'wv3_{short}_finish-current.json').read_text())['container']
    else:
        name=f'panda-{short}-hardonly-wv3-{server}-v1'
    state=json.loads(subprocess.run(['docker','inspect','--format','{{json .State}}',name],
        capture_output=True,text=True,check=True).stdout)
    if state.get('Running') or state.get('Status')!='exited' or state.get('ExitCode')!=0:
        raise ValueError('Cutover WV3 container must exit normally before new dataset')
    return {'container':name,'status':'exited','exit_code':0}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',choices=('s1','s3'),required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--require-complete',action='store_true')
    p.add_argument('--check-container',action='store_true')
    args=p.parse_args(argv)
    try:
        value=predecessor_receipt(ROOT,args.server)
        if args.check_container:
            value['container']=container_gate(args.server)
        print(json.dumps({'status':'READY','receipt':value}))
        return 0
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as exc:
        print(json.dumps({'status':'WAITING_OR_INVALID_CURRENT_WV3','reason':str(exc),'mutated':False}))
        return 2 if args.require_complete else 0


if __name__=='__main__':
    raise SystemExit(main())
