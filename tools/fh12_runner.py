#!/usr/bin/env python
"""One-command, local-only FH12 scheduling. Nothing starts on import or dry-run.

start: register once, drain current work without terminating it, then preflight,
fresh T, exact50K calibration and finite S queue. Restart never resets the clock.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fh12.plan import (CAMPAIGN_ID, METHOD_REVISION, REGISTRY_REVISION, SOURCE_PLAN,
                      SERVERS, WINDOW_HOURS, CLOSE_HOURS, SETUP_CALIBRATION_HOURS,
                      cases_for, teacher_for, registry_rows, registry_sha256)

HOLD_PROTOCOL = "FH12_FUTURE_ADMISSIONS_HOLD_v1"
TERMINAL_RUNS = {"DONE", "FAILED", "SKIPPED_BUDGET"}


def iso(seconds=None):
    return dt.datetime.fromtimestamp(time.time() if seconds is None else seconds,
                                    dt.timezone.utc).isoformat()


def timestamp(value):
    result = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError("Explicit UTC offset required")
    return result.timestamp()


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else ({} if default is None else default)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def campaign_dir(root, server):
    return Path(root)/'work_dir/_fh12'/server


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def detect_server(root, explicit=None):
    path = Path(root)/'gspread/server.txt'
    raw = path.read_text().strip().lower() if path.is_file() else ''
    match = re.search(r'(?:^|[^a-z0-9])s([1-5])(?:$|[^0-9])', raw)
    local = 's' + match[1] if match else ('s'+raw if raw in '12345' and len(raw)==1 else None)
    if explicit is not None and local is not None and explicit != local:
        raise ValueError(f"Requested {explicit} but local gspread/server.txt identifies {local}")
    server = explicit or local
    if server not in SERVERS:
        raise ValueError("Cannot identify server: use --server s1..s5 once, or existing gspread/server.txt")
    return server


def process_inventory():
    result = subprocess.run(['ps','-eo','pid=,args='], capture_output=True, text=True, check=True)
    controllers = {'mix20h_runner.py','_run_cases.sh','qrecon24_waiter.sh'}
    workers = {'main.py','fh12_train.py','fh12_postrun.py','mix20h_postrun.py',
               'qrecon24_select.py','eval_fr_paperset.py','noa_eval.py','noa_eval_all.py',
               'aligner_scope_audit.py','s1_aligner_analysis.py','fh12_preflight.py','fh12_calibrate.py'}
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        try:
            args = shlex.split(fields[1])
        except ValueError:
            continue
        if '-c' in args[:3]:
            continue
        script = next((Path(s).name for s in args[:3] if Path(s).name in controllers | workers), None)
        if script:
            rows.append(dict(pid=int(fields[0]), script=script,
                             controller=script in controllers))
    return rows


def gpu_processes():
    try:
        value = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
                                '--format=csv,noheader,nounits'], capture_output=True,
                               text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return value.stdout.strip().splitlines() if value.returncode == 0 else None


def hold_owned(doc, server):
    return (doc.get('protocol_id') == HOLD_PROTOCOL and doc.get('campaign_id') == CAMPAIGN_ID
            and doc.get('server') == server)


def check_takeover(root, server, inventory):
    hold = Path(root)/'work_dir/_eval_phase/hold.json'
    if hold.exists() and not hold_owned(read_json(hold), server):
        raise ValueError("Another evaluation owns hold.json; it is preserved, not silently released")
    # M20's loaded code rechecks hold.json at every case boundary. Older generic
    # controllers do not; we cannot promise safe draining of unknown controllers.
    if not (Path(root)/'work_dir/_qrc24_mix20h/plan_manifest.json').exists():
        unsafe = [r for r in inventory if r['script'] == '_run_cases.sh']
        if unsafe:
            raise ValueError("Active legacy _run_cases has no verified hold hook; finish/pause its future queue first. Current training is untouched")


def projection(case, rows, root):
    """End-to-end estimates include evaluation exactly once."""
    base = case.reservation_hours
    samples = [r['end_to_end_hours'] for r in rows.values()
               if r.get('role') == case.role and r.get('tier') == 'CORE'
               and r.get('end_to_end_hours', 0) > 0]
    if samples:
        factor = 1.1 if case.tier == 'RESERVE_D' else 1.2 if case.tier == 'RESERVE_W' else 1.0
        base = max(base, max(samples) * factor)
    measured = read_json(Path(root)/'work_dir'/case.run_id/'meta/runtime_projection.json')
    s = measured.get('training_seconds_per_update'); e = measured.get('evaluator_checkpoint_seconds')
    if s is not None and e is not None and all(math.isfinite(float(x)) and float(x) >= 0 for x in (s,e)):
        base = max(base, (50000*float(s) + 50*float(e))/3600)
    return base


def admission(remaining_h, next_h, unfinished_core_h=0., required_eval_h=0.,
              setup_remaining_h=0., reserve=False):
    """next_h and core_h already include their required evaluation; backlog only extra."""
    margin = 1.15 if reserve else 1.0
    required = CLOSE_HOURS + setup_remaining_h + margin * (next_h + unfinished_core_h + required_eval_h)
    return dict(admitted=remaining_h >= required, remaining_hours=remaining_h,
                required_hours=required, next_hours=next_h, unfinished_core_hours=unfinished_core_h,
                required_eval_hours=required_eval_h, setup_remaining_hours=setup_remaining_h,
                safety_factor=margin)


def prepare(root, server, device='cuda', upload=True, dry_run=False):
    root = Path(root); camp = campaign_dir(root, server)
    from tools.gen_fh12_configs import generate
    changed = generate(root, server, check=True)
    if changed:
        raise ValueError(f"Missing/modified FH12 release configs: {changed}; pull the complete release")
    hashes = {c.run_id: sha(root/f'config/{c.run_id}.yaml') for c in cases_for(server)}
    previous = read_json(camp/'window.json')
    if previous and (previous.get('campaign_id') != CAMPAIGN_ID or previous.get('server_id') != server
                     or previous.get('config_hashes') != hashes
                     or previous.get('source_plan_sha256') != sha(root/SOURCE_PLAN)
                     or previous.get('registry_sha256') != registry_sha256()):
        raise ValueError("Registered window identity/configs changed; cannot reset/reuse this campaign")
    inventory = process_inventory()
    check_takeover(root, server, inventory)
    now = time.time()
    release=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    window = previous or dict(campaign_id=CAMPAIGN_ID, server_id=server,
        method_revision=METHOD_REVISION, registry_revision=REGISTRY_REVISION,
        registry_sha256=registry_sha256(), source_plan_sha256=sha(root/SOURCE_PLAN),
        window_start_utc=iso(now), deadline_utc=iso(now+WINDOW_HOURS*3600),
        config_hashes=hashes, device=device, upload_enabled=upload,
        local_release=release.stdout.strip() if release.returncode==0 else 'unavailable',
        clock_policy='immutable_local_first_explicit_start', no_global_lock=True)
    if dry_run:
        return dict(status='DRY_RUN', window_would_be=window, effective_queue=registry_rows(server),
                    current_processes=inventory, mutates_runtime=False)
    write_json(camp/'window.json', window)
    # Written only by explicit start, used by existing watchdog/shell entrypoints.
    pointer=root/'work_dir/_fh12/local_server.txt'
    temp=pointer.with_suffix(f'.tmp.{os.getpid()}'); temp.write_text(server+'\n'); os.replace(temp,pointer)
    if not (camp/'takeover_manifest.json').exists():
        write_json(camp/'takeover_manifest.json', dict(at_utc=iso(), processes=inventory,
            policy='finish_existing_run_and_postrun; hold_only_future_admissions; no_kill',
            old_pending_runs='not_appended', previous_hold=read_json(root/'work_dir/_eval_phase/hold.json')))
    write_json(root/'work_dir/_eval_phase/hold.json', dict(protocol_id=HOLD_PROTOCOL,
        campaign_id=CAMPAIGN_ID, server=server, at_utc=iso(),
        reason='FH12 owns future admissions; current legacy run is allowed to finish',
        auto_release=False))
    write_json(camp/'effective_queue.json', dict(campaign_id=CAMPAIGN_ID,
        registry_revision=REGISTRY_REVISION, rows=registry_rows(server), config_hashes=hashes,
        runtime_state='REGISTERED_PREFLIGHT_PENDING', no_global_lock=True))
    return dict(status='REGISTERED', window=window)


def load_window(root, server):
    doc = read_json(campaign_dir(root,server)/'window.json')
    if not doc or doc.get('campaign_id') != CAMPAIGN_ID or doc.get('server_id') != server:
        raise ValueError("FH12 window absent/invalid; use fh12_start.sh")
    if doc.get('registry_sha256') != registry_sha256():
        raise ValueError("FH12 registry changed after start")
    for run, expected in doc['config_hashes'].items():
        if sha(Path(root)/'config'/f'{run}.yaml') != expected:
            raise ValueError(f"Registered config changed: {run}")
    return doc


def command(root, args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        return subprocess.run([sys.executable]+args, cwd=root, stdout=stream,
                              stderr=subprocess.STDOUT).returncode


def wait_resources(root, server, window, state, wait):
    camp = campaign_dir(root,server)
    while True:
        if time.time() >= timestamp(window['deadline_utc']):
            state['status']='DEFERRED_BUDGET'; write_json(camp/'status.json',state); return False
        rows = process_inventory()
        check_takeover(root,server,rows)
        if not hold_owned(read_json(Path(root)/'work_dir/_eval_phase/hold.json'),server):
            raise ValueError("FH12 future-admission hold is missing; refusing unsafe concurrent launch")
        busy = [r for r in rows if not r['controller']]
        gpu = gpu_processes() if window['device'] != 'cpu' else []
        if not busy and gpu == []:
            return True
        state.update(status='DRAINING_EXISTING', existing_workers=busy, gpu_processes=gpu)
        write_json(camp/'status.json',state)
        if not wait:
            return False
        time.sleep(15)


def train_complete(root, case):
    folder = Path(root)/'work_dir'/case.run_id/'candidates/50000'
    identity = read_json(folder/'identity.json')
    if not (folder/'model.safetensors').is_file() or int(identity.get('step',identity.get('update',-1))) != 50000:
        return False
    if identity.get('model_sha256') != sha(folder/'model.safetensors'):
        raise ValueError(f'Invalid exact50K checksum: {case.run_id}')
    return True


def pending_evaluation_hours(entries, rows, root):
    """Only already-trained, unfinished reports: not eval included in future train reservations."""
    result=0.
    for case in entries:
        if train_complete(root,case) and rows.get(case.run_id,{}).get('status')!='DONE':
            measured=read_json(Path(root)/'work_dir'/case.run_id/'meta/runtime_projection.json')
            result+=max(.10,float(measured.get('evaluator_checkpoint_seconds',0))*50/3600)
    return result


def run_campaign(root, server, wait=True):
    root = Path(root); camp = campaign_dir(root,server); window = load_window(root,server)
    with contextlib.ExitStack() as stack:
        stack.enter_context(locked(camp/'.runner.lock'))
        # A pre-M20 watchdog may attempt to restart its generic shell queue.
        # Hold its admission lock when no loaded M20 runner owns that lock;
        # M20 instead is safely parked by our case-boundary hold.json above.
        if not (root/'work_dir/_qrc24_mix20h/plan_manifest.json').exists():
            stack.enter_context(locked(root/'work_dir/.cases_chain.lock'))
        state = read_json(camp/'status.json', dict(status='REGISTERED', runs={}))
        state.setdefault('runs',{}); rows = state['runs']
        if state.get('status') in {'DONE','PAUSED_DEADLINE','DEFERRED_BUDGET','FAILED_PREFLIGHT','FAILED_CALIBRATION'}:
            return 0
        if not wait_resources(root,server,window,state,wait):
            return 3
        entries = cases_for(server)
        # Drain time counts toward the immutable 12h window. Do not create a
        # teacher if the still-unfinished mandatory pipeline no longer fits.
        unfinished = [c for c in entries if c.tier == 'CORE' and not train_complete(root,c)
                      and rows.get(c.run_id,{}).get('status') not in TERMINAL_RUNS]
        setup_left = max(0., SETUP_CALIBRATION_HOURS - state.get('setup_hours',0.)) if not state.get('calibration_complete') else 0.
        gate = admission((timestamp(window['deadline_utc'])-time.time())/3600,
                         sum(projection(c,rows,root) for c in unfinished),
                         required_eval_h=pending_evaluation_hours(entries,rows,root),
                         setup_remaining_h=setup_left)
        state['initial_admission']=gate
        if not gate['admitted']:
            state['status']='DEFERRED_BUDGET'; write_json(camp/'status.json',state); return 0
        if not state.get('preflight_complete'):
            start=time.time(); state['status']='PREFLIGHT'; write_json(camp/'status.json',state)
            rc=command(root,['tools/fh12_preflight.py','--server',server,'--device',window['device']],camp/'preflight.log')
            state['setup_hours']=state.get('setup_hours',0.)+(time.time()-start)/3600
            if rc:
                state.update(status='FAILED_PREFLIGHT',preflight_exit_code=rc)
                write_json(camp/'status.json',state); return rc
            state['preflight_complete']=True
            queue=read_json(camp/'effective_queue.json'); queue['runtime_state']='PREFLIGHT_PASS'
            write_json(camp/'effective_queue.json',queue)
        for index,case in enumerate(entries):
            row=rows.setdefault(case.run_id,dict(role=case.role,tier=case.tier,status='PENDING'))
            if row['status']=='FAILED' and case.role=='T':
                state['status']='FAILED_TEACHER'; break
            if row['status'] in TERMINAL_RUNS:
                continue
            if case.role=='S' and not state.get('calibration_complete'):
                if not train_complete(root,teacher_for(server)):
                    state['status']='FAILED_TEACHER'; break
                gate=admission((timestamp(window['deadline_utc'])-time.time())/3600,
                    sum(projection(c,rows,root) for c in entries[index:] if c.tier=='CORE'
                        and not train_complete(root,c) and rows.get(c.run_id,{}).get('status') not in TERMINAL_RUNS),
                    setup_remaining_h=max(0.,SETUP_CALIBRATION_HOURS-state.get('setup_hours',0.)))
                if not gate['admitted']:
                    state.update(status='DEFERRED_BUDGET',calibration_admission=gate); break
                state['status']='CALIBRATION'; write_json(camp/'status.json',state); start=time.time()
                try:
                    teacher=teacher_for(server)
                    rc=command(root,['tools/fh12_calibrate.py','--teacher-run',teacher.run_id,
                        '--server',server,'--device',window['device'],'--deadline-utc',window['deadline_utc']],
                        camp/'calibration.log')
                    if rc==75:
                        raise TimeoutError('Calibration paused at immutable deadline')
                    if rc:
                        raise RuntimeError(f'Calibration child exited {rc}; see calibration.log')
                    ref=camp/'references'/teacher.run_id/'reference_manifest.json'
                    if read_json(ref).get('teacher_run_id')!=teacher.run_id:
                        raise ValueError('Calibration child did not publish the required local Teacher reference')
                    state['reference_manifest']=str(ref); state['calibration_complete']=True
                    if window.get('upload_enabled'):
                        teacher=teacher_for(server)
                        command(root,['tools/fh12_postrun.py',teacher.run_id,'--upload','--upload-only'],
                                root/'work_dir'/teacher.run_id/'fh12_upload_retry.log')
                        teacher_post=read_json(root/'work_dir'/teacher.run_id/'official/postrun_status.json')
                        rows[teacher.run_id].update(postrun=teacher_post,upload_pending=not teacher_post.get('sheet_uploaded',False))
                except TimeoutError as error:
                    state.update(status='PAUSED_DEADLINE',calibration_error=str(error)); break
                except Exception as error:
                    state.update(status='FAILED_CALIBRATION',calibration_error=str(error)); break
                finally:
                    state['setup_hours']=state.get('setup_hours',0.)+(time.time()-start)/3600
                    write_json(camp/'status.json',state)
            completed=train_complete(root,case)
            core_left=sum(projection(c,rows,root) for c in entries[index+1:] if c.tier=='CORE'
                and not train_complete(root,c) and rows.get(c.run_id,{}).get('status') not in TERMINAL_RUNS)
            gate=admission((timestamp(window['deadline_utc'])-time.time())/3600,
                0. if completed else projection(case,rows,root),core_left,
                required_eval_h=pending_evaluation_hours(entries[index:],rows,root),
                setup_remaining_h=max(0.,SETUP_CALIBRATION_HOURS-state.get('setup_hours',0.))
                    if not state.get('calibration_complete') else 0.,reserve=case.tier!='CORE')
            row['admission']=gate
            if not gate['admitted']:
                if case.tier=='CORE':
                    state['status']='DEFERRED_BUDGET'; break
                row['status']='SKIPPED_BUDGET'; write_json(camp/'status.json',state); continue
            if not wait_resources(root,server,window,state,wait):
                return 3
            wd=root/'work_dir'/case.run_id; start=time.time()
            # Saving exact50K weights precedes its metric transaction. A crash
            # in that interval must resume update 50000 just to finish the grid,
            # not strand postrun with incomplete metrics or repeat optimizer work.
            finishing=completed and not read_json(wd/'official/raw_grid.json').get('complete')
            if not completed or finishing:
                resume=(wd/'last/training_state.pt').is_file()
                if (row.get('training_started') or finishing) and not resume:
                    row.update(status='FAILED',reason='interrupted_without_exact_resume; never_fresh_retry')
                    write_json(camp/'status.json',state)
                    if case.role=='T': state['status']='FAILED_TEACHER'; break
                    continue
                row.update(status='TRAINING',training_started=True); state['status']='RUNNING'
                write_json(camp/'status.json',state)
                args=['tools/fh12_train.py','--config',f'config/{case.run_id}.yaml',
                      '--device',window['device'],'--deadline-utc',window['deadline_utc']]
                if resume: args.append('--resume')
                rc=command(root,args,wd/'fh12_train.log'); row['training_exit_code']=rc
                if rc==75:
                    row['status']='PAUSED_DEADLINE'
                    row['actual_updates']=read_json(wd/'meta/training_status.json').get('actual_updates')
                    state['status']='PAUSED_DEADLINE'; break
                if rc or not train_complete(root,case):
                    row.update(status='FAILED',reason='training_exit_or_missing_exact50k');
                    if case.role=='T': state['status']='FAILED_TEACHER'; break
                    write_json(camp/'status.json',state); continue
                row['training_complete']=True
            row['status']='EVALUATING'; write_json(camp/'status.json',state)
            args=['tools/fh12_postrun.py',case.run_id,'--device',window['device'],
                  '--deadline-utc',window['deadline_utc']]
            if window.get('upload_enabled'): args.append('--upload')
            rc=command(root,args,wd/'fh12_postrun.log'); row['evaluation_exit_code']=rc
            post=read_json(wd/'official/postrun_status.json'); row['postrun']=post
            if rc or not post.get('official_complete'):
                row['status']='EVALUATION_PENDING'; state['status']='EVALUATION_PENDING'; break
            row.update(status='DONE',end_to_end_hours=(time.time()-start)/3600,
                       finished_at_utc=iso(),training_complete=True,
                       upload_pending=bool(window.get('upload_enabled') and not post.get('sheet_uploaded')))
            write_json(camp/'status.json',state)
        else:
            state['status']='DONE'
        state['updated_at_utc']=iso(); write_json(camp/'status.json',state)
    return 0


def launch(root,server):
    camp=campaign_dir(root,server)
    try:
        with locked(camp/'.runner.lock'): pass
    except BlockingIOError:
        return dict(status='ALREADY_RUNNING')
    with (camp/'runner.log').open('a') as stream:
        process=subprocess.Popen([sys.executable,str(Path(root)/'tools/fh12_runner.py'),'run',
            '--server',server],cwd=root,stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    return dict(status='RUNNER_SUBMITTED',pid=process.pid,log=str(camp/'runner.log'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','run','status','retry-upload'])
    parser.add_argument('--server',choices=SERVERS)
    parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--no-upload',action='store_true')
    args=parser.parse_args(); server=detect_server(ROOT,args.server); camp=campaign_dir(ROOT,server)
    if args.action=='status':
        print(json.dumps(dict(window=read_json(camp/'window.json'),state=read_json(camp/'status.json'),
                              queue=read_json(camp/'effective_queue.json')),indent=2)); return 0
    if args.action=='start':
        if args.dry_run:
            result=prepare(ROOT,server,args.device,not args.no_upload,True)
        else:
            with locked(camp/'.start.lock'):
                result=prepare(ROOT,server,args.device,not args.no_upload)
                result['launch']=launch(ROOT,server)
        print(json.dumps(result,indent=2)); return 0
    if args.dry_run:
        parser.error('--dry-run is only supported by start; status is always read-only')
    if args.action=='retry-upload':
        state=read_json(camp/'status.json'); codes=[]
        for case in cases_for(server):
            if state.get('runs',{}).get(case.run_id,{}).get('training_complete'):
                codes.append(command(ROOT,['tools/fh12_postrun.py',case.run_id,'--upload','--upload-only'],
                                     ROOT/'work_dir'/case.run_id/'fh12_upload_retry.log'))
                post=read_json(ROOT/'work_dir'/case.run_id/'official/postrun_status.json')
                state['runs'][case.run_id].update(postrun=post,upload_pending=not post.get('sheet_uploaded',False))
        write_json(camp/'status.json',state)
        return max(codes,default=0)
    return run_campaign(ROOT,server)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,BlockingIOError) as error:
        print(f'FH12: {error}',file=sys.stderr); sys.exit(2)
