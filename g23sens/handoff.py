"""Cooperative current-run handoff; never signal or rewrite an old queue."""
from pathlib import Path
import subprocess
import time

from g23sens.common import (ROOT,camp,read,atomic_json,immutable_json,object_sha,
                           sha256,utcnow,verify_server,RuntimePaused,locked)


def inventory(root=ROOT):
    from pcrepro.handoff import inventory as inspect
    return inspect(root)


def _local_owners(rows,server):
    """Other declared lanes are protected, not a cross-server scheduling lock."""
    local=[row for row in rows if row.get('server')==server]
    unknown=[row for row in rows if row.get('server') not in ('s1','s2','s3','s4','s5')]
    unsupported=[row for row in local if row.get('script')!='pcrepro_runner.py']+unknown
    if unsupported:
        raise RuntimePaused('Old owner has no verified current-RUN handoff: '+
            str([(r['pid'],r['script'],r.get('server')) for r in unsupported])+
            '. Drain that owner explicitly; no forced stop or whole-cycle fallback.')
    return local


def _valid_request(value,root,server):
    if (value.get('schema')!='G23SENS_HANDOFF_REQUEST_v1' or value.get('server')!=server
            or value.get('shared_work_dir')!=str((Path(root)/'work_dir').resolve())
            or value.get('policy')!='CURRENT_RUN_BOUNDARY_ONLY'
            or value.get('original_state_sha256')!=object_sha(value.get('original_state',{}))
            or value.get('original_active_run')!=value.get('original_state',{}).get('active_run')):
        raise ValueError('Handoff request belongs to another lane/workspace/policy')


def request_boundary(root,server,*,activated=False):
    verify_server(server)
    if not activated:raise PermissionError('Explicit start is required for old-run handoff')
    root=Path(root);folder=camp(root,server)
    with locked(folder/'handoff/request.lock'):
        rows=inventory(root);_local_owners(rows,server)
        path=folder/'handoff/request.json';previous=read(path)
        old=root/'work_dir/_pcrepro'/server
        if previous:
            _valid_request(previous,root,server)
            current_control=read(old/'control.json')
            if (current_control.get('handoff_request_sha256') is not None
                    and current_control['handoff_request_sha256']!=object_sha(previous)):
                raise ValueError('Original control/request identity changed')
            if previous.get('old_new_admission_blocked') and current_control.get('command') not in (
                    'STOP_AFTER_CURRENT_RUN','STOP_NOW_SAFE'):
                raise RuntimePaused('Old admission was reenabled after immutable handoff request')
            return previous
        status=read(old/'status.json');control=read(old/'control.json')
        if status and status.get('server')!=server:
            raise ValueError('Original PCREPRO status belongs to another server')
        receipt=dict(schema='G23SENS_HANDOFF_REQUEST_v1',server=server,
            shared_work_dir=str((root/'work_dir').resolve()),requested_at_utc=utcnow(),
            policy='CURRENT_RUN_BOUNDARY_ONLY',observed_processes=rows,
            protected_other_lane_processes=[r for r in rows if r.get('server') not in (None,server)],
            original_state=status,original_state_sha256=object_sha(status),
            original_active_run=status.get('active_run'),original_control=control,
            original_registration=read(old/'registration.json'),
            old_new_admission_blocked=old.exists(),old_queue_weights_results_preserved=True,signals_sent=False)
        # STOP_NOW_SAFE is stronger and remains byte-identical.
        if old.exists() and control.get('command') not in ('STOP_NOW_SAFE','STOP_AFTER_CURRENT_RUN'):
            if control.get('command') not in (None,'CONTINUE','STOP_AFTER_CYCLE'):
                raise RuntimePaused('Unknown original controller command; no overwrite')
            atomic_json(old/'control.json',dict(command='STOP_AFTER_CURRENT_RUN',
                reason='Explicit G23 SENS current-run handoff',at_utc=utcnow(),
                handoff_request_sha256=object_sha(receipt)))
        immutable_json(path,receipt)
        return receipt


def gpu_rows():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    return [dict(pid=int(parts[0].strip()),gpu_uuid=parts[1].strip())
        for line in result.splitlines() if len(parts:=line.split(','))==2 and parts[0].strip().isdigit()]


def ensure_gpu_idle(gpu_uuid=None):
    active=[r for r in gpu_rows() if gpu_uuid is None or r['gpu_uuid']==gpu_uuid]
    if active:raise RuntimePaused('Selected GPU still owned by another process: '+str(active))


def _completed_run(root,server,run):
    """Read original completion artifacts without importing new trainer policies."""
    from pcrepro.plan import case_for
    from fh12.common import object_sha as original_sha
    case=case_for(run)
    if case.server!=server:raise ValueError('Original active run belongs to a protected lane')
    wd=Path(root)/'work_dir'/run;summary=read(wd/'official/summary.json')
    if (summary.get('complete') is not True or summary.get('run_id')!=run
            or summary.get('case',{}).get('server')!=server or summary.get('actual_updates')!=case.updates
            or set(summary.get('selections',{}))!={'EXACT_50000','RR_VAL_ERGAS_MIN'}):
        raise RuntimePaused('ORIGINAL_RUN_INCOMPLETE: finish the current original run/evaluation before takeover: '+run)
    source=case.source_run_id or run;source_wd=Path(root)/'work_dir'/source
    status=read(source_wd/'meta/training_status.json')
    if status.get('training_complete') is not True or status.get('actual_updates')!=50000:
        raise RuntimePaused('ORIGINAL_50K_INCOMPLETE: '+source)
    evidence={}
    for selection,label in (('EXACT_50000','exact_50000'),('RR_VAL_ERGAS_MIN','best_val')):
        record=summary['selections'][selection];pointer=read(source_wd/'checkpoints'/(label+'.json'))
        directory=pointer.get('directory','')
        if not directory or Path(directory).name!=directory or not directory.startswith(label+'_'):
            raise ValueError('Original checkpoint pointer is absent or unsafe')
        checkpoint=source_wd/'checkpoints'/directory;identity=read(checkpoint/'identity.json')
        if (original_sha(identity)!=pointer.get('identity_sha256') or identity!=record.get('checkpoint_identity')
                or identity.get('model_sha256')!=record.get('checkpoint_sha256')
                or sha256(checkpoint/'model.safetensors')!=identity['model_sha256']
                or selection=='EXACT_50000' and identity.get('update')!=50000):
            raise ValueError('Original completed checkpoint identity changed')
        relative=Path(record.get('evaluation_path',''))
        if relative.is_absolute() or relative.parts[:2]!=('official','evaluations') or '..' in relative.parts:
            raise ValueError('Original evaluation path is unsafe')
        metrics_path=wd/relative;metrics=read(metrics_path);cursor=read(metrics_path.parent/'evaluation_cursor.json')
        if (sha256(metrics_path)!=record.get('evaluation_manifest_sha256') or not cursor.get('complete')
                or metrics.get('metadata',{}).get('checkpoint_sha256')!=record['checkpoint_sha256']
                or any(not metrics.get(k,{}).get('official_complete') or metrics[k]!=record.get(k) for k in ('rr','fr'))):
            raise RuntimePaused('ORIGINAL_EVALUATION_INCOMPLETE: '+run)
        evidence[selection]=dict(checkpoint_sha256=record['checkpoint_sha256'],
            evaluation_manifest_sha256=record['evaluation_manifest_sha256'])
    return dict(run_id=run,status='ORIGINAL_RUN_COMPLETE',summary_sha256=sha256(wd/'official/summary.json'),selections=evidence)


def verify_boundary(root,server,request,gpu_uuid=None):
    """No process absence alone can turn an interrupted run into completion."""
    _valid_request(request,root,server);rows=inventory(root);local=_local_owners(rows,server)
    if local:raise RuntimePaused('WAIT_ORIGINAL_CURRENT_RUN: '+str([r['pid'] for r in local]))
    ensure_gpu_idle(gpu_uuid)
    old=Path(root)/'work_dir/_pcrepro'/server;state=read(old/'status.json')
    if request.get('old_new_admission_blocked'):
        if read(old/'control.json').get('command') not in ('STOP_NOW_SAFE','STOP_AFTER_CURRENT_RUN'):
            raise RuntimePaused('Old admission was reenabled during handoff')
    if state and state.get('server')!=server:raise ValueError('Original state changed its server identity')
    if state.get('active_run'):
        raise RuntimePaused('ORIGINAL_ACTIVE_CURSOR: original owner must finish/clear its run; no cursor rewriting')
    active=request.get('original_active_run');evidence=[]
    if active:evidence.append(_completed_run(root,server,active))
    return dict(schema='G23SENS_HANDOFF_COMPLETE_v1',status='COMPLETE',server=server,
        request_sha256=object_sha(request),at_utc=utcnow(),boundary='CURRENT_RUN',selected_gpu_uuid=gpu_uuid,
        verified_original_runs=evidence,original_final_state_sha256=object_sha(state),
        protected_other_lane_processes=[r for r in rows if r.get('server') not in (None,server)],
        no_signals=True,no_old_artifact_rewrite=True)


def wait_boundary(root,server,gpu_uuid=None,*,activated=False):
    request=request_boundary(root,server,activated=activated);last=None
    while True:
        folder=camp(root,server)
        if (folder/'STOP_NOW_SAFE').exists() or (folder/'STOP_AFTER_RUN').exists():
            raise RuntimePaused('Handoff paused by operator before new admission')
        rows=inventory(root);local=_local_owners(rows,server)
        if not local:
            result=verify_boundary(root,server,request,gpu_uuid)
            previous=read(folder/'handoff/complete.json')
            if previous:
                if previous.get('request_sha256')!=result['request_sha256']:
                    raise ValueError('Completed handoff request changed')
                return previous
            immutable_json(folder/'handoff/complete.json',result);return result
        ids=tuple(r['pid'] for r in local)
        if ids!=last:print('Waiting for old current run and its evaluation:',ids,flush=True);last=ids
        time.sleep(10)
