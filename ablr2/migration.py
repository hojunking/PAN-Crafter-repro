"""Append-only release handover; never rewrite a running frozen worktree.

Only explicit activation writes cooperative control/receipts. Original state,
seed ledger, lease, registration and checkpoints retain their names and bytes.
"""
from pathlib import Path
import hashlib
import os
import tempfile
import subprocess
import sys
import time

from ablr2.common import (ROOT,camp,run_dir,read,read_json,atomic_json,immutable_json,locked,
                         object_sha,sha256,utcnow,source_identity,RuntimePaused,append_event)
from ablr2.plan import verify_lane,EXTENSION_ID

PRESERVE_FILES=('state.json','cases.json','seed_ledger.json','seed_ledger.jsonl','task_ledger.jsonl',
    'all_attempts.jsonl','admission_ledger.jsonl','lease.json','lease_ledger.jsonl','control.json',
    'runtime_release.json','registration.json','preflight.json','dataset_manifest.json','design.json','policy.json')


def _origin(root,server):
    receipt=read(camp(root,server)/'runtime_release.json')
    if receipt:
        from ablr2.deployment import verify_release
        verify_release(receipt,_workspace_root(root,server,receipt),server)
    return receipt


def _workspace_root(root,server,origin):
    root=Path(root).resolve();target=Path(origin['path'])
    marker='-runtime-ablr2-'+server+'-'
    if marker not in target.name:raise ValueError('Invalid original runtime name')
    candidate=target.parent/target.name.split(marker,1)[0]
    if (candidate/'work_dir').resolve()!=(root/'work_dir').resolve():
        raise ValueError('Original runtime belongs to another shared workspace')
    return candidate


def _metadata(root,server):
    folder=camp(root,server);paths=[folder/name for name in PRESERVE_FILES]
    paths+=list((folder/'stages').glob('*.json'))+list((folder/'thresholds').glob('*.json'))
    paths+=list((folder/'runs').glob('*/meta/*.json'))+list((folder/'runs').glob('*/meta/config.resolved.yaml'))
    paths+=list((folder/'runs').glob('*/init_manifest.json'))+list((folder/'runs').glob('*/last/identity.json'))
    return {str(p.relative_to(folder)):sha256(p) for p in sorted(set(paths)) if p.is_file()}


def _live_controller(root,server):
    from ablr2.resources import process_start
    pid=read(camp(root,server)/'runner.pid.json')
    return bool(pid.get('process_start_ticks') and process_start(pid.get('pid'))==pid['process_start_ticks'])


def plan_migration(root,server):
    """Read-only; a report is not authorization and never rewinds a cursor."""
    verify_lane(server);folder=camp(root,server);state=read(folder/'state.json');origin=_origin(root,server)
    active=state.get('active_run');active_row=state.get('runs',{}).get(active,{})
    incomplete_started=sorted(path.parent.parent.name
        for path in (folder/'runs').glob('*/meta/training_start_manifest.json')
        if not state.get('runs',{}).get(path.parent.parent.name,{}).get('complete'))
    return dict(schema='ABLR2X_MIGRATION_PLAN_v1',extension_id=EXTENSION_ID,server=server,
        original_release=origin or None,legacy_present=bool(origin or (folder/'registration.json').exists()),
        original_controller_live=_live_controller(root,server),active_run=active,
        incomplete_started_runs=incomplete_started,
        active_run_complete=bool(active_row.get('complete')),cycle=state.get('cycle'),
        active_stage=state.get('active_stage'),incumbent=state.get('incumbent'),
        state_sha256=object_sha(state) if state else None,metadata_sha256=_metadata(root,server),
        actual_state=state,training_started=False,other_servers_touched=False)


def _copy_snapshot(folder,target,expected):
    """Each byte copy is checked; the final boundary snapshot is made under lock."""
    for name,digest in expected.items():
        source=folder/name;destination=target/name;payload=source.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=digest:raise RuntimePaused('Original metadata changed during snapshot; retry at boundary')
        if destination.exists():
            if sha256(destination)!=digest:raise ValueError('Immutable migration backup differs')
            continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(prefix='.snapshot_',dir=destination.parent)
        try:
            with os.fdopen(fd,'wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
            os.replace(temp,destination)
        finally:
            if os.path.exists(temp):os.unlink(temp)


def request_migration(root,server,*,operator_authorized=False):
    verify_lane(server)
    if operator_authorized is not True:raise PermissionError('Explicit release handover request required')
    folder=camp(root,server)
    with locked(folder/'migration.lock'):
        old=read(folder/'migration/request.json')
        if old:return old
        plan=plan_migration(root,server)
        value=dict(schema='ABLR2X_MIGRATION_REQUEST_v1',extension_id=EXTENSION_ID,server=server,
            requested_at_utc=utcnow(),operator_action=True,original_plan=plan,
            original_lease_retained=True,old_authorization_not_extended=True,
            policy='Finish current admitted run on its original release; preserve all counters and seeds')
        _copy_snapshot(folder,folder/'migration/request_snapshot'/object_sha(plan['metadata_sha256']),plan['metadata_sha256'])
        immutable_json(folder/'migration/request.json',value)
        if plan['legacy_present']:
            # Supported by the original ABLR2 controller; do not send any signal.
            previous=read(folder/'control.json')
            if previous.get('command') not in ('STOP_NOW_SAFE','STOP_AFTER_RUN'):
                atomic_json(folder/'control.json',dict(command='STOP_AFTER_RUN',at_utc=utcnow(),
                    operator_action=True,reason='ABLR2X_RELEASE_HANDOVER',migration_request_sha256=object_sha(value)))
        return value


def finalize_migration(root,server,new_runtime=None):
    """Publish a new execution owner only after original GPU/controller work ends."""
    verify_lane(server);folder=camp(root,server)
    completed=read(folder/'migration/transition.json')
    if completed:
        if completed.get('server')!=server or completed.get('extension_id')!=EXTENSION_ID:
            raise ValueError('Migration receipt belongs to another lane')
        return completed
    plan=plan_migration(root,server)
    if not plan['legacy_present']:
        return dict(status='FRESH_EXTENSION_LANE',server=server,complete=True,legacy_present=False)
    request=read(folder/'migration/request.json')
    if not request:raise RuntimePaused('WAIT_MIGRATION: explicitly request original run-boundary handover first')
    if (request.get('server')!=server or request.get('extension_id')!=EXTENSION_ID
            or request.get('operator_action') is not True
            or request.get('original_plan',{}).get('original_release')!=plan['original_release']):
        raise ValueError('Original release/migration request identity changed')
    new_release=read(folder/'runtime_release_ablr2x.json')
    if not new_release or Path(new_release['path']).resolve()!=Path(new_runtime or root).resolve():
        raise RuntimePaused('WAIT_PINNED_EXTENSION_RELEASE: freeze the committed extension first')
    from ablr2.deployment import verify_release
    verify_release(new_release,_workspace_root(root,server,plan['original_release']),server,extension=True)
    from ablr2.resources import idle_evidence
    evidence=idle_evidence()
    if plan['original_controller_live'] or not evidence['idle']:
        raise RuntimePaused('WAIT_ORIGINAL_RUN_EXIT: no old/new GPU overlap')
    if plan['active_run'] and not plan['active_run_complete']:
        raise RuntimePaused('WAIT_ORIGINAL_RUN_COMPLETION: resume the admitted job under its original release/lease')
    if plan['incomplete_started_runs']:
        raise RuntimePaused('WAIT_ORIGINAL_STARTED_JOBS: every admitted original job needs its original-source '
            'training/evaluation completion before handover: '+', '.join(plan['incomplete_started_runs']))
    with locked(folder/'runner.lock'),locked(folder/'migration.lock'):
        stable=plan_migration(root,server)
        if stable['state_sha256']!=plan['state_sha256']:raise RuntimePaused('Original state changed at release handover')
        _copy_snapshot(folder,folder/'migration/boundary_snapshot',stable['metadata_sha256'])
        origin=stable['original_release'];registration=read(folder/'registration.json')
        routes={}
        for path in (folder/'runs').glob('*/meta/training_start_manifest.json'):
            start=read_json(path)
            if start.get('source_identity')==registration.get('source_identity'):
                routes[path.parent.parent.name]=dict(runtime=origin['path'],source_identity=start['source_identity'])
        value=dict(schema='ABLR2X_RELEASE_TRANSITION_v1',extension_id=EXTENSION_ID,server=server,
            complete=True,status='BOUNDARY_HANDOVER_COMPLETE',verified_at_utc=utcnow(),
            request_sha256=object_sha(request),original_release=origin,
            original_registration=registration,original_metadata_sha256=stable['metadata_sha256'],
            state_sha256=stable['state_sha256'],original_state=stable['actual_state'],
            new_runtime=str(Path(new_runtime or root).resolve()),
            new_release=new_release,
            original_execution_routes=routes,resource_evidence=evidence,
            original_files_rewritten=False,cursor_seed_costs_preserved=True,other_servers_touched=False)
        immutable_json(folder/'migration/transition.json',value)
        # Only the stop generated by this exact handover is discharged. A later
        # operator STOP_NOW_SAFE/STOP_AFTER_RUN remains effective.
        control=read(folder/'control.json')
        if (control.get('command')=='STOP_AFTER_RUN' and control.get('reason')=='ABLR2X_RELEASE_HANDOVER'
                and control.get('migration_request_sha256')==object_sha(request)):
            next_control=dict(command='CONTINUE',at_utc=utcnow(),operator_action=True,
                reason='EXPLICIT_ABLR2X_HANDOVER_COMPLETE',migration_receipt_sha256=object_sha(value))
            atomic_json(folder/'control.json',next_control)
            append_event(folder/'control_ledger.jsonl','HANDOVER_CONTROL_DISCHARGED',**next_control)
        return value


def assert_ready(root,server):
    return finalize_migration(root,server,new_runtime=root)


def start_waiter(root,server,*,foreground=False,upload=True):
    """CPU-only owner waiting at an explicitly requested cooperative boundary."""
    verify_lane(server);folder=camp(root,server)
    if foreground:return dict(exit_code=await_handover(root,server,upload=upload,foreground_controller=True),status='HANDOVER_WAIT_ENDED')
    from ablr2.resources import process_start
    with locked(folder/'migration/waiter_start.lock'):
        previous=read(folder/'migration/waiter.pid.json')
        if previous.get('start_ticks') and process_start(previous.get('pid'))==previous['start_ticks']:
            return dict(status='ALREADY_WAITING_FOR_HANDOVER',pid=previous['pid'],training_started=False)
        command=[sys.executable,'-u',str(Path(root)/'tools/ablr2_runner.py'),'await-handover',
                 '--server',server,'--in-place']+([] if upload else ['--no-upload'])
        log=folder/'migration/waiter.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            child=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(folder/'migration/waiter.pid.json',dict(pid=child.pid,start_ticks=process_start(child.pid)))
        return dict(status='WAITING_FOR_SAFE_HANDOVER',pid=child.pid,training_started=False,
            original_job_not_interrupted=True,automatic_lease_renewal=False,log=str(log))


def await_handover(root,server,*,upload=True,poll_seconds=15,foreground_controller=True):
    """Never run original training ourselves, reset a manual stop, or add a lease."""
    verify_lane(server);folder=camp(root,server)
    from ablr2.common import authorization_context,before_deadline
    with locked(folder/'migration/waiter.lock'):
        while True:
            try:
                authorization_context(root,server)
                control=read(folder/'control.json')
                own_stop=control.get('reason')=='ABLR2X_RELEASE_HANDOVER'
                if control.get('command') not in (None,'CONTINUE') and not own_stop:
                    raise RuntimePaused('OPERATOR_STOPPED: handover waiter will not restart automatically')
                plan=plan_migration(root,server)
                if plan['legacy_present'] and plan['active_run'] and not plan['active_run_complete']:
                    lease=read(folder/'lease.json')
                    if not lease.get('expires_utc') or not before_deadline(lease['expires_utc']):
                        raise RuntimePaused('WAIT_ORIGINAL_FINITE_LEASE: explicit legacy lease renewal required')
                    if not plan['original_controller_live']:
                        from ablr2.resources import idle_evidence
                        if idle_evidence()['idle']:
                            raise RuntimePaused('ORIGINAL_JOB_STOPPED: recover it on original source before handover')
                if (plan['legacy_present'] and plan['incomplete_started_runs']
                        and not plan['original_controller_live']):
                    raise RuntimePaused('ORIGINAL_STARTED_JOBS_INCOMPLETE: original-source recovery required; '
                        'continuous authorization does not admit legacy recovery')
                try:
                    assert_ready(root,server)
                    if server=='s3':
                        from ablr2.handoff import verify_ready
                        verify_ready(root,server)
                except RuntimePaused as exc:
                    atomic_json(folder/'migration/waiter_status.json',dict(status='WAITING_FOR_SAFE_HANDOVER',
                        reason=str(exc),at_utc=utcnow(),training_started=False,automatic_lease_renewal=False))
                    time.sleep(poll_seconds)
                    continue
                from ablr2.controller import start
                result=start(root,server,foreground=foreground_controller,upload=upload)
                atomic_json(folder/'migration/waiter_status.json',dict(status='HANDOVER_DISPATCHED',
                    result=result,at_utc=utcnow(),automatic_lease_renewal=False))
                return int(result.get('exit_code',0))
            except RuntimePaused as exc:
                atomic_json(folder/'migration/waiter_status.json',dict(status='PAUSED_SAFE',reason=str(exc),
                    at_utc=utcnow(),training_started=False,automatic_restart=False))
                return 75


def execution_root(root,case):
    """Already-started legacy work/recovery stays on its authenticated old source."""
    verify_lane(case.server_id,case.sensor)
    transition=read(camp(root,case.server_id)/'migration/transition.json')
    route=transition.get('original_execution_routes',{}).get(case.run_id)
    if not route:return Path(root)
    origin=transition['original_release']
    from ablr2.deployment import verify_release
    # Runtime callers may already be inside the extension worktree.
    origin_root=_workspace_root(root,case.server_id,origin)
    verify_release(origin,origin_root,case.server_id)
    start=read_json(run_dir(case.run_id,root)/'meta/training_start_manifest.json')
    if start.get('source_identity')!=route['source_identity'] or route['runtime']!=origin['path']:
        raise ValueError('Original job execution identity changed during migration')
    return Path(route['runtime'])


def install_source_bridge(root,server,parity_path,*,operator_authorized=False):
    verify_lane(server)
    if operator_authorized is not True:raise PermissionError('Explicit measured parity bridge installation required')
    folder=camp(root,server);origin=read_json(folder/'registration.json')['source_identity'];consumer=source_identity(root)
    from ablr2.runtime_parity import validate_runtime_parity
    parity=read_json(parity_path);validate_runtime_parity(parity,origin,consumer,root)
    if parity.get('original',{}).get('server')!=server or parity.get('consumer',{}).get('server')!=server:
        raise ValueError('Runtime parity was measured on another sensor/server lane')
    digest=object_sha(parity);relative='migration/parity/'+digest+'.json'
    immutable_json(folder/relative,parity)
    bridge=dict(schema='ABLR2X_MEASURED_SOURCE_BRIDGE_v1',extension_id=EXTENSION_ID,server=server,
        origin_source_identity=origin,consumer_source_identity=consumer,parity_path=relative,
        parity_object_sha256=digest,parity_file_sha256=sha256(folder/relative),
        operator_action=True,numerical_equivalence_claim='MEASURED_SOURCE_BOUND_PROTOCOL_ONLY',
        c03_pairing_claim=False,created_at_utc=utcnow())
    old=read(folder/'migration/source_bridge.json')
    if old:
        if any(old.get(k)!=v for k,v in bridge.items() if k!='created_at_utc'):
            raise ValueError('A different source bridge requires a new explicit transition identity')
        return old
    immutable_json(folder/'migration/source_bridge.json',bridge)
    return bridge


def ensure_source_bridge(root,server):
    """Measure parity automatically at the explicitly activated pinned handover."""
    verify_lane(server);folder=camp(root,server)
    registration=read(folder/'registration.json')
    if not registration:return None
    origin=registration['source_identity'];consumer=source_identity(root)
    if origin==consumer:return None
    if (folder/'migration/source_bridge.json').exists():
        return validate_source_bridge(root,server,origin,consumer)
    transition=read(folder/'migration/transition.json')
    if not transition.get('complete'):raise RuntimePaused('WAIT_MEASURED_PARITY: handover must finish first')
    from ablr2.runtime_parity import compare_runtime_releases
    path=folder/'migration/measured_runtime_parity.json'
    compare_runtime_releases(transition['original_release']['path'],root,server,output_path=path)
    return install_source_bridge(root,server,path,operator_authorized=True)


def validate_source_bridge(root,server,origin,consumer):
    verify_lane(server);folder=camp(root,server);bridge=read(folder/'migration/source_bridge.json')
    if (bridge.get('schema')!='ABLR2X_MEASURED_SOURCE_BRIDGE_v1' or bridge.get('server')!=server
            or bridge.get('extension_id')!=EXTENSION_ID or bridge.get('operator_action') is not True
            or bridge.get('origin_source_identity')!=origin or bridge.get('consumer_source_identity')!=consumer):
        raise ValueError('Different release requires an exact measured runtime-parity bridge')
    relative=Path(bridge.get('parity_path',''))
    if relative.is_absolute() or relative.parts[:2]!=('migration','parity') or '..' in relative.parts:
        raise ValueError('Unsafe runtime parity artifact path')
    path=folder/relative;parity=read_json(path)
    if sha256(path)!=bridge['parity_file_sha256'] or object_sha(parity)!=bridge['parity_object_sha256']:
        raise ValueError('Measured runtime parity artifact changed')
    from ablr2.runtime_parity import validate_runtime_parity
    validate_runtime_parity(parity,origin,consumer,root)
    if parity.get('original',{}).get('server')!=server or parity.get('consumer',{}).get('server')!=server:
        raise ValueError('Runtime parity was measured on another sensor/server lane')
    return dict(bridge,receipt_sha256=object_sha(bridge))
