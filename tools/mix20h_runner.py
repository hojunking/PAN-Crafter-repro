#!/usr/bin/env python
"""Explicit, finite M20 runner. Nothing is activated by importing/generating configs.

prepare --server sN --start-at <shared UTC T0> [--dry-run] [--launch]
start [--server sN] [--dry-run] [--no-upload]  # one command, local first-start 20h
run [--wait] [--upload]   # resume SAME immutable wall-clock window
status / report          # read-only; no GPU or network

An existing case is not killed. Prepare installs an expired *legacy* deadline as
a case-boundary barrier; the dedicated runner waits for the shared chain lock.
The real 20h deadline lives in plan_manifest.json and is never reset by restart.
"""
import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kdv.mix20h_plan import (CAMPAIGN_ID, QUEUE_REVISION, SELECTOR_ID,
                            GRID_STEPS, PREP_RESERVE_HOURS, cases_for, reservation_hours)

CAMP = Path('work_dir/_qrc24_mix20h')
CLOSED = {'DONE', 'BUDGET_CLOSED', 'STOPPED_BUDGET'}
LEGACY_EVAL_PROTOCOL = 'PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918'
LEGACY_FILES = ('work_dir/cases_queue.txt', 'work_dir/cases_queue_handover.txt',
                'work_dir/cases_deadline.txt', 'work_dir/campaign_gates_enabled.txt',
                'work_dir/_qrecon24/queue_active.txt', 'work_dir/_qrecon24/queue_effective.txt',
                'work_dir/_qrecon24/mandatory_runs.txt', 'work_dir/_pakd50/mandatory_runs.txt',
                'work_dir/_pakd50/extra_priority.txt', 'work_dir/_pakd50/reservations.json',
                'work_dir/_qrecon24/recipe_lock.json', 'work_dir/_eval_phase/hold.json')


def timestamp(value):
    t = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('UTC offset required; naive local timestamp is forbidden')
    return t.timestamp()


def iso(seconds=None):
    return dt.datetime.fromtimestamp(time.time() if seconds is None else seconds,
                                    dt.timezone.utc).isoformat()


def read_json(path, default=None):
    if not Path(path).exists():
        return {} if default is None else default
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(value); os.replace(tmp, path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def event(root, name, **fields):
    path = root / CAMP / 'events.jsonl'
    with path.open('a') as f:
        f.write(json.dumps(dict(at_utc=iso(), event=name, **fields), ensure_ascii=False) + '\n')


@contextlib.contextmanager
def locked(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def process_inventory():
    """Only recognized executable/script tokens; never dump shell environments."""
    text = subprocess.run(['ps', '-eo', 'pid=,comm=,args='], capture_output=True,
                          text=True, check=True).stdout
    out = []
    scripts = {'main.py', '_run_cases.sh', 'qrecon24_waiter.sh', 'mix20h_runner.py',
               'qrecon24_select.py', 'eval_fr_paperset.py', 'aligner_scope_audit.py',
               'noa_eval.py', 'noa_eval_all.py', 's1_aligner_analysis.py', 'mix20h_postrun.py'}
    for line in text.splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3 or int(fields[0]) == os.getpid():
            continue
        try:
            argv = shlex.split(fields[2])
        except ValueError:
            continue
        if '-c' in argv[:3]:  # shell command text is NOT a running trainer
            continue
        script = next((a for a in argv[1:3] if Path(a).name in scripts), None)
        if script:
            config = argv[argv.index('--config') + 1] if '--config' in argv else None
            out.append(dict(pid=int(fields[0]), executable=fields[1], script=script,
                            config=config, run_id=Path(config).stem if config else None))
    return out


def gpu_processes():
    try:
        p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True,
                           timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.strip().splitlines() if p.returncode == 0 else None


def active_gpu_workers():
    """Evaluation also owns the GPU while between models/CPU metric phases."""
    waiting = {'_run_cases.sh', 'qrecon24_waiter.sh', 'mix20h_runner.py'}
    return [p for p in process_inventory() if Path(p['script']).name not in waiting]


def legacy_hold_digest(root, server):
    path = root / 'work_dir/_eval_phase/hold.json'
    if not path.exists():
        return None
    held = read_json(path)
    if held.get('protocol_id') != LEGACY_EVAL_PROTOCOL or held.get('server') != server:
        raise ValueError('unrecognized/current hold is not the legacy evaluation hold; preserved')
    return digest(path)


def archive_legacy_hold(root, server, expected_sha):
    """Only after old queues are suppressed. Preserve unfinished evaluation history."""
    path = root / 'work_dir/_eval_phase/hold.json'
    if not path.exists():
        return None
    if not expected_sha or digest(path) != expected_sha:
        receipt = dict(status='new_hold_preserved', observed_at_utc=iso(), expected_sha256=expected_sha,
                       reason='hold changed during preparation; runner waits without deleting it')
        write_json(root / CAMP / 'hold_takeover.json', receipt)
        event(root, 'new_hold_preserved', **receipt)
        return receipt
    if legacy_hold_digest(root, server) != expected_sha:
        raise ValueError('evaluation hold identity changed during migration; preserved')
    archived = root / CAMP / 'released_holds' / f'hold.{expected_sha}.json'
    archived.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, archived)
    receipt = dict(archived_path=str(archived.relative_to(root)), sha256=expected_sha,
                   status='archived',
                   released_at_utc=iso(), reason='explicit one-command M20 start supersedes prior evaluation phase',
                   previous_evaluation_complete=False, unfinished_evaluation_policy='preserved_not_required_for_M20')
    write_json(root / CAMP / 'hold_takeover.json', receipt)
    event(root, 'legacy_hold_archived', **receipt)
    return receipt


def run_inventory(root, run):
    wd = root / 'work_dir' / run
    if not wd.exists():
        return dict(status='unstarted')
    checkpoints = sorted(str(p.relative_to(root)) for p in wd.glob('*/model.safetensors'))
    complete = all((wd / 'results' / f'{split}_best_hqnr.mat').is_file()
                   for split in ('reduced', 'full'))
    return dict(status='completed' if complete else ('started_interrupted' if checkpoints else 'present_unverified'),
                checkpoints=checkpoints, config_sha256=digest(wd / 'meta/config.yaml')
                if (wd / 'meta/config.yaml').is_file() else None)


def inventory(root, server):
    files = {}
    old_runs = set()
    for rel in LEGACY_FILES:
        p = root / rel
        if p.is_file():
            content = p.read_text()
            files[rel] = dict(sha256=digest(p), content=content)
            old_runs.update(x.strip() for x in content.splitlines()
                            if x.strip().startswith('PAKD50_') and ' ' not in x.strip())
    return dict(host=os.uname().nodename, source_server=server, observed_at_utc=iso(),
                processes=process_inventory(), gpu_processes=gpu_processes(), files=files,
                historical_runs={r: run_inventory(root, r) for r in sorted(old_runs)},
                new_runs={c.run_id: run_inventory(root, c.run_id) for c in cases_for(server)})


def estimate_hours(server, end_to_end_hours):
    samples = [float(x) for x in end_to_end_hours if math.isfinite(float(x)) and float(x) > 0]
    if len(samples) < 2:
        return reservation_hours(server)
    recent = sorted(samples[-8:])
    # Few samples: slowest. Later: nearest-rank 90th percentile. No duplicate .10h.
    baseline = max(recent) if len(recent) < 5 else recent[math.ceil(.9 * len(recent)) - 1]
    return baseline * 1.10


def prep_remaining_hours(now, started_at):
    """Reserve preparation only while it has not already elapsed in the window."""
    return max(0.0, PREP_RESERVE_HOURS - max(0.0, (now - started_at) / 3600.0))


def admission(remaining_h, run_h, backlog_h=0.0, close_h=.5, pending_runs=2, prep_h=0.0):
    needed = pending_runs * run_h + backlog_h + close_h + prep_h
    return dict(admitted=remaining_h >= needed, required_hours=needed,
                remaining_hours=remaining_h, pending_runs=pending_runs,
                prep_reserve_hours=prep_h)


def load_plan(root, require_ready=True):
    p = read_json(root / CAMP / 'plan_manifest.json')
    if p.get('campaign_id') != CAMPAIGN_ID or p.get('queue_revision') != QUEUE_REVISION:
        raise ValueError('missing/incompatible M20 plan manifest')
    if abs(timestamp(p['deadline_at_utc']) - timestamp(p['campaign_started_at_utc']) - 20 * 3600) > .001:
        raise ValueError('M20 deadline must remain exactly T0+20h')
    if p['run_ids'] != [c.run_id for c in cases_for(p['server_id'])]:
        raise ValueError('runtime queue differs from explicit registry')
    if require_ready and p.get('activation_state') != 'ready':
        raise ValueError('incomplete migration; repeat prepare with the SAME T0 to recover')
    return p


def validate_configs(root, server):
    import yaml
    from tools.gen_mix20h_configs import build_config
    hashes = {}
    for c in cases_for(server):
        path = root / 'config' / (c.run_id + '.yaml')
        if not path.is_file():
            raise ValueError(f'missing config: {path}')
        actual = yaml.safe_load(path.read_text())
        expected = build_config(c.run_id, root=root)
        if actual != expected:
            raise ValueError(f'config/registry mismatch: {c.run_id}')
        hashes[c.run_id] = digest(path)
    queue = root / 'config/queues' / f'qrc24_mix20h_{server}.txt'
    actual_queue = [x.strip() for x in queue.read_text().splitlines() if x.strip() and not x.startswith('#')]
    if actual_queue != list(hashes):
        raise ValueError('static queue is not generated from the same explicit registry')
    return hashes


def verify_assets(root, server):
    """Read-only CPU integrity preflight, before any operational queue write."""
    import yaml
    from kdv.mix20h_runtime import validate_definition
    cfg = yaml.safe_load((root / 'config' / (cases_for(server)[0].run_id + '.yaml')).read_text())
    validate_definition(cfg)
    k = cfg['kdv']; meta = k['mix20h']; checked = {}
    def local(path):
        p = Path(path)
        return p if p.is_absolute() else root / p
    for role, expected in meta['expected_dataset_hashes'].items():
        wanted = expected.get('sha256') if isinstance(expected, dict) else expected
        path = local(cfg[role]['dataroot'])
        got = digest(path)
        if got != wanted:
            raise ValueError(f'dataset hash mismatch: {role}')
        checked[role] = dict(path=str(path), sha256=got)
    donor = local(k['donor']['source']) / 'model.safetensors'
    if digest(donor) != k['donor']['expected_sha256']:
        raise ValueError('T0 checkpoint hash mismatch')
    checked['teacher'] = dict(path=str(donor), sha256=digest(donor))
    cue = local(k['qrecon']['asset']); cue_info = read_json(cue)
    cue_npz = local(cue_info.get('npz', str(cue.with_suffix('.npz'))))
    cue_sha = digest(cue_npz)
    if cue_sha != meta['expected_cue_sha256'] or cue_sha != cue_info['npz_sha256']:
        raise ValueError('AXIS16 cue bytes mismatch')
    cue_manifest_sha = digest(cue)
    if cue_manifest_sha != meta['expected_cue_manifest_sha256']:
        raise ValueError('AXIS16 cue manifest mismatch')
    checked['cue'] = dict(path=str(cue_npz), sha256=cue_sha, manifest_sha256=cue_manifest_sha)
    # Trainer repeats these checks against actual loaded states and dataloader.
    return checked


def prepare(root, server, start_at, dry_run=True, upload=False, *, takeover_eval_hold=False,
            clock_policy='shared_explicit_20h'):
    """Pure preview unless explicitly applied. Never launch/kill from this function."""
    existing_path = root / CAMP / 'plan_manifest.json'
    existing = load_plan(root, require_ready=False) if existing_path.exists() else None
    t0 = timestamp(start_at) if start_at else (timestamp(existing['campaign_started_at_utc']) if existing else None)
    if existing and (existing['server_id'] != server or timestamp(existing['campaign_started_at_utc']) != t0):
        raise ValueError('restart cannot change server/T0/deadline; preserve old campaign and request a new revision')
    hashes = validate_configs(root, server)
    inv = inventory(root, server)
    collisions = [r for r, v in inv['new_runs'].items() if v['status'] != 'unstarted'] if not existing else []
    held = (root / 'work_dir/_eval_phase/hold.json').exists()
    takeover = bool(takeover_eval_hold and (not existing or existing.get('activation_state') != 'ready'))
    hold_sha = legacy_hold_digest(root, server) if held and takeover else None
    result = dict(server=server, campaign_id=CAMPAIGN_ID, queue_revision=QUEUE_REVISION,
                  actual_yaml_count=len(hashes), base_count=sum(c.tier == 'base' for c in cases_for(server)),
                  reserve_count=sum(c.tier == 'reserve' for c in cases_for(server)),
                  next_two_real_run_ids=list(hashes)[:2], namespace_collisions=collisions,
                  old_eval_hold=held, inventory=inv, selector_auto_id=SELECTOR_ID,
                  status='BLOCKED_LOCAL_COLLISION' if collisions else ('BLOCKED_LOCAL_EVAL_HOLD' if held and not takeover else
                         ('NEEDS_SHARED_START_AT' if t0 is None else 'PREPARED_NOT_ASSET_VERIFIED')),
                  deadline_at_utc=iso(t0 + 20 * 3600) if t0 is not None else None)
    if dry_run:
        return result
    if collisions or (held and not takeover) or t0 is None:
        raise ValueError(result['status'])
    if time.time() >= t0 + 20 * 3600:
        raise ValueError('BUDGET_CLOSED: supplied common window already expired')
    assets = verify_assets(root, server)
    camp = root / CAMP
    camp.mkdir(parents=True, exist_ok=True)
    with locked(camp / '.migration.lock'):
        # Re-read under lock: concurrent applies cannot reset an already prepared window.
        existing = load_plan(root, require_ready=False) if existing_path.exists() else None
        if existing:
            if (existing['config_hashes'] != hashes or existing['server_id'] != server or
                    timestamp(existing['campaign_started_at_utc']) != t0):
                raise ValueError('campaign identity/config/clock changed during preparation')
            if existing.get('activation_state') == 'ready':
                return dict(result, status='ALREADY_PREPARED')
        # Snapshot before replacing any operational pointer; no historical assets deleted.
        if not (camp / 'migration_manifest.json').exists():
            write_json(camp / 'migration_manifest.json', inv)
        preserved = read_json(camp / 'migration_manifest.json')
        for rel, value in preserved['files'].items():
            if not (camp / 'before' / rel).exists():
                write_text(camp / 'before' / rel, value['content'])
        plan = existing or dict(campaign_id=CAMPAIGN_ID, queue_revision=QUEUE_REVISION, server_id=server,
                    campaign_started_at_utc=iso(t0), started_at_utc=iso(t0), deadline_at_utc=iso(t0 + 20 * 3600),
                    local_timezone=str(dt.datetime.now().astimezone().tzinfo), run_ids=list(hashes),
                    config_hashes=hashes, close_reserve_hours=.5, known_evaluation_backlog_hours=0.0,
                    requires_recipe_lock=False, requires_other_server_results=False,
                    prepared_at_utc=iso(), upload_enabled=bool(upload), verified_assets=assets,
                    clock_policy=clock_policy, legacy_hold_takeover_sha256=hold_sha,
                    activation_state='preparing')
        write_json(existing_path, plan)  # authoritative legacy suppression marker
        queue = f'# {QUEUE_REVISION}; explicit finite list, pair admission by mix20h_runner\n' + '\n'.join(hashes) + '\n'
        for p in ('queue_effective.txt', 'mandatory_runs.txt'):
            write_text(camp / p, queue)
        write_text(root / 'work_dir/cases_queue.txt', queue)
        write_text(root / 'work_dir/cases_queue_handover.txt', queue)
        # Old, already-parsed shell loop reads this at each case boundary.
        write_text(root / 'work_dir/cases_deadline.txt', iso(min(time.time(), t0) - 1) + '\n')
        if not (camp / 'pair_reservations.json').exists():
            write_json(camp / 'pair_reservations.json', dict(queue_revision=QUEUE_REVISION, pairs={}))
        if not (camp / 'budget.json').exists():
            write_json(camp / 'budget.json', dict(started_at_utc=iso(t0), deadline_at_utc=plan['deadline_at_utc'],
                                                end_to_end_hours=[], run_events=[]))
        if not (camp / 'status.json').exists():
            write_json(camp / 'status.json', dict(status='PREPARED', runs={}, queue_revision=QUEUE_REVISION))
        # Publish M20 suppression/barrier before releasing the old evaluation phase.
        if held and takeover:
            archive_legacy_hold(root, server, plan.get('legacy_hold_takeover_sha256'))
        plan['activation_state'] = 'ready'
        write_json(existing_path, plan)
        event(root, 'prepared', preserved_running=inv['processes'], old_pending_policy='superseded_pending_M20')
    return dict(result, status='PREPARED')


def launch_background(root, upload=False):
    try:
        with locked(root / CAMP / '.runner.lock'):
            pass
    except BlockingIOError:
        return dict(launch_status='RUNNER_ALREADY_ACTIVE')
    log = root / CAMP / 'runner.log'
    with log.open('a') as stream:
        process = subprocess.Popen([sys.executable, str(root / 'tools/mix20h_runner.py'), 'run', '--wait'] +
                                   (['--upload'] if upload else []), cwd=root, stdout=stream,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    return dict(launch_status='RUNNER_SUBMITTED', runner_pid=process.pid, log=str(log))


def start_campaign(root, server=None, start_at=None, dry_run=False, upload=True):
    """User-facing single action. No manual prepare, clock, hold release or commit gate.

    Default operational policy changed by the user's simplification request:
    each server receives 20h from its first start; --start-at retains shared T0.
    Existing campaigns keep their original deadline and newly asserted holds.
    """
    from tools.gen_pakd50_configs import server_id
    from tools.mix20h_launch_config import ensure_configs
    root = Path(root)
    local = root / 'gspread/server.txt'
    local_server = server_id(local.read_text()) if local.is_file() else None
    if local_server is None and (root / CAMP / 'plan_manifest.json').is_file():
        local_server = load_plan(root, require_ready=False)['server_id']
    server = server_id(server) if server else local_server
    if server not in ('s1', 's2', 's3', 's4', 's5'):
        raise ValueError('cannot identify server: set gspread/server.txt or pass --server sN')
    if local_server and local_server != server:
        raise ValueError(f'local server {local_server} disagrees with requested {server}')

    def execute():
        path = root / CAMP / 'plan_manifest.json'
        existing = load_plan(root, require_ready=False) if path.exists() else None
        if existing and (existing['server_id'] != server or
                         (start_at and timestamp(start_at) != timestamp(existing['campaign_started_at_utc']))):
            raise ValueError('start cannot reset an existing server/T0/deadline')
        start = start_at or (existing['campaign_started_at_utc'] if existing else iso())
        policy = (existing or {}).get('clock_policy', 'shared_explicit_20h' if start_at else 'local_first_start_20h')
        config = ensure_configs(root, server, dry_run=dry_run)
        if dry_run:
            held = (root / 'work_dir/_eval_phase/hold.json').exists()
            takeover = held and (not existing or existing.get('activation_state') != 'ready')
            if takeover:
                legacy_hold_digest(root, server)
            inv = inventory(root, server)
            collisions = [r for r, data in inv['new_runs'].items() if data['status'] != 'unstarted'] if not existing else []
            return dict(status='BLOCKED_LOCAL_COLLISION' if collisions else 'WOULD_START', server=server,
                        dry_run=True, clock_policy=policy, campaign_started_at_utc=start,
                        deadline_at_utc=iso(timestamp(start) + 20*3600), configuration=config,
                        legacy_hold_would_be_archived=bool(takeover), namespace_collisions=collisions,
                        upload_enabled=bool(upload), runtime_not_activated=True)
        if existing and existing.get('activation_state') == 'ready':
            if validate_configs(root, server) != existing['config_hashes']:
                raise ValueError('existing campaign configuration changed')
            result = dict(status='ALREADY_PREPARED', server=server, deadline_at_utc=existing['deadline_at_utc'])
        else:
            result = prepare(root, server, start, dry_run=False, upload=upload,
                             takeover_eval_hold=True, clock_policy=policy)
        plan = load_plan(root)
        result.update(configuration=config, clock_policy=plan.get('clock_policy', policy),
                      campaign_started_at_utc=plan['campaign_started_at_utc'])
        state = read_json(root / CAMP / 'status.json')
        if state.get('status') in CLOSED or time.time() >= timestamp(plan['deadline_at_utc']):
            result.update(launch_status='CAMPAIGN_CLOSED_NO_RESTART',
                          status=state.get('status') if state.get('status') in CLOSED else 'BUDGET_CLOSED')
        else:
            result.update(launch_background(root, upload=upload))
        return result

    if dry_run:
        return execute()
    with locked(root / CAMP / '.start.lock'):
        return execute()


def wait_for_resources(root, plan, state, wait):
    """Automatic resource waiting, including CPU gaps in legacy GPU evaluators."""
    while True:
        if time.time() >= timestamp(plan['deadline_at_utc']):
            state['status'] = 'BUDGET_CLOSED'
            write_json(root / CAMP / 'status.json', state)
            return False
        busy = gpu_processes()
        workers = active_gpu_workers()
        held = (root / 'work_dir/_eval_phase/hold.json').exists()
        before_start = time.time() < timestamp(plan['campaign_started_at_utc'])
        if busy == [] and not workers and not held and not before_start:
            return True
        state.update(status='WAIT_RESOURCE', gpu_processes=busy, active_gpu_workers=workers,
                     eval_hold=held, waiting_for_start_time=before_start)
        write_json(root / CAMP / 'status.json', state)
        if not wait:
            return False
        time.sleep(15)


def checkpoint_for_resume(root, run):
    """Newest *completed update*, never newest filesystem modification time.

Epoch checkpoints use the same 1010-step grid/config eval_epoch relationship
as the trainer (1010 / 5 = 202 updates per epoch). If that relationship cannot
be established, epoch names are ambiguous and are not eligible for resume.
"""
    wd = root / 'work_dir' / run
    epoch_steps = None
    config = root / 'config' / (run + '.yaml')
    if config.is_file():
        import yaml
        cfg = yaml.safe_load(config.read_text()) or {}
        period = cfg.get('eval_epoch')
        if (isinstance(period, int) and not isinstance(period, bool) and period > 0
                and GRID_STEPS[0] % period == 0
                and (cfg.get('kdv') or {}).get('candidate_grid_id') == 'GRID1010_50K_v1'):
            epoch_steps = GRID_STEPS[0] // period
    paths = []
    for pattern in ('checkpoint-*', 'epoch-*', 'candidates/step-*'):
        for p in wd.glob(pattern):
            match = re.fullmatch(r'(checkpoint(?:-budget)?|epoch|step)-(\d+)', p.name)
            if match is None or (match[1] == 'epoch' and epoch_steps is None):
                continue
            step = int(match[2]) * (epoch_steps if match[1] == 'epoch' else 1)
            if not 0 <= step <= GRID_STEPS[-1]:
                continue
            required = ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl',
                        'custom_checkpoint_0.pkl', 'custom_checkpoint_1.pkl', 'custom_checkpoint_2.pkl')
            if p.name.startswith('checkpoint-budget-') and not (p / 'mix20h_resume.json').is_file():
                continue  # save_state may have been interrupted before its completion marker
            if match[1] == 'step':
                try:
                    marker = read_json(p / 'mix20h_evaluation_complete.json')
                    if (marker.get('step') != step or marker.get('run_id') != run or
                            marker.get('checkpoint_sha256') != digest(p / 'model.safetensors')):
                        continue
                except (OSError, ValueError):
                    continue  # candidate save/evaluation was not atomically committed
            if all((p / name).is_file() and (p / name).stat().st_size > 0 for name in required):
                priority = 2 if match[1] == 'step' else int(match[1] == 'checkpoint-budget')
                paths.append((step, priority, p.name, p))
    return max(paths, key=lambda item: item[:3])[-1] if paths else None


def run_command(root, cmd, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        return subprocess.run(cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def postrun_backlog(root, plan, state, upload=False):
    """Finite local close phase. Failed evaluation never calls run.sh again."""
    for run, row in state.get('runs', {}).items():
        if row.get('status') not in {'evaluation_pending', 'upload_pending'}:
            continue
        if row['status'] == 'upload_pending' and not upload:
            continue
        if time.time() >= timestamp(plan['deadline_at_utc']):
            break  # explicit CPU-only upload command remains available after the window
        cmd = [sys.executable, 'tools/mix20h_postrun.py', run, '--device', 'cuda',
               '--deadline-utc', plan['deadline_at_utc']] + (['--upload'] if upload else [])
        start = time.time()
        rc = run_command(root, cmd, root / 'work_dir' / run / 'mix20h_postrun.log')
        post = read_json(root / 'work_dir' / run / 'results/mix20h_postrun_status.json')
        row.update(status='evaluation_pending' if rc else ('finished' if post.get('sheet_uploaded') else 'upload_pending'),
                   postrun=post, evaluation_exit_code=rc)
        event(root, 'postrun_retry', run_id=run, exit_code=rc, hours=(time.time()-start)/3600)
        write_json(root / CAMP / 'status.json', state)
    return state


def run_campaign(root, wait=False, upload=False):
    plan = load_plan(root); server = plan['server_id']; camp = root / CAMP
    upload = bool(upload or plan.get('upload_enabled'))
    prior = read_json(camp / 'status.json')
    closed = prior.get('status') in CLOSED
    retries = any(v.get('status') == 'evaluation_pending' or
                  (upload and v.get('status') == 'upload_pending') for v in prior.get('runs', {}).values())
    if closed and (not retries or time.time() >= timestamp(plan['deadline_at_utc'])):
        return 0
    # A second watchdog/manual instance must not compete for either GPU or writes.
    with locked(camp / '.runner.lock'):
        chain = (root / 'work_dir/.cases_chain.lock').open('a')
        while True:
            if time.time() >= timestamp(plan['deadline_at_utc']):
                state = read_json(camp / 'status.json'); state['status'] = 'BUDGET_CLOSED'
                write_json(camp / 'status.json', state); return 0
            try:
                fcntl.flock(chain, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    return 3
                time.sleep(15)
        with chain:
            resource_state = read_json(camp / 'status.json')
            if not wait_for_resources(root, plan, resource_state, wait):
                return 0 if resource_state['status'] == 'BUDGET_CLOSED' else 3
            if validate_configs(root, server) != plan['config_hashes']:
                raise ValueError('config hashes changed since preparation')
            if closed:
                postrun_backlog(root, plan, prior, upload)
                return 0
            write_text(root / 'work_dir/cases_deadline.txt', plan['deadline_at_utc'] + '\n')
            state = read_json(camp / 'status.json'); state['status'] = 'RUNNING'; state.setdefault('runs', {})
            budget = read_json(camp / 'budget.json'); samples = budget.setdefault('end_to_end_hours', [])
            reservations = read_json(camp / 'pair_reservations.json')
            entries = cases_for(server)
            for i in range(0, len(entries), 2):
                pair = entries[i:i + 2]
                pending = [c for c in pair if state['runs'].get(c.run_id, {}).get('status') not in
                           {'finished', 'failed', 'stopped_budget', 'evaluation_pending', 'upload_pending'}]
                if not pending:
                    continue
                admission_now = time.time()
                remaining = (timestamp(plan['deadline_at_utc']) - admission_now) / 3600
                pending_eval_h = .10 * sum(v.get('status') == 'evaluation_pending' for v in state['runs'].values())
                check = admission(remaining, estimate_hours(server, samples),
                                  float(plan['known_evaluation_backlog_hours']) + pending_eval_h, pending_runs=len(pending),
                                  prep_h=prep_remaining_hours(admission_now, timestamp(plan['campaign_started_at_utc'])))
                if not check['admitted']:
                    state['status'] = 'BUDGET_CLOSED'; state['not_admitted_from_pair'] = pair[0].pair_id
                    event(root, 'pair_not_admitted', pair_id=pair[0].pair_id, **check)
                    break
                reservations['pairs'][pair[0].pair_id] = dict(status='admitted', run_ids=[c.run_id for c in pair],
                                                             admitted_at_utc=iso(), **check)
                write_json(camp / 'pair_reservations.json', reservations)
                for c in pending:
                    # A reserved pair is not permission to start its mate after
                    # an unexpectedly long first run/postrun exhausts the window.
                    # Keep the not-started run absent from the run-state table:
                    # it is neither a failed startup nor a partial training run.
                    if time.time() >= timestamp(plan['deadline_at_utc']):
                        state.update(status='BUDGET_CLOSED', not_admitted_from_pair=c.pair_id,
                                     budget_not_started_run_id=c.run_id)
                        reservations['pairs'][c.pair_id].update(status='deadline_before_pair_complete',
                                                               not_started_run_id=c.run_id, closed_at_utc=iso())
                        write_json(camp / 'pair_reservations.json', reservations)
                        write_json(camp / 'status.json', state)
                        event(root, 'deadline_before_run', run_id=c.run_id, pair_id=c.pair_id,
                              deadline_at_utc=plan['deadline_at_utc'])
                        break
                    # A hold/resource owner may have changed during the previous
                    # training run. Respect it again at the safe case boundary.
                    state['waiting_run_id'] = c.run_id
                    if not wait_for_resources(root, plan, state, wait):
                        return 0 if state['status'] == 'BUDGET_CLOSED' else 3
                    state['status'] = 'RUNNING'
                    wd = root / 'work_dir' / c.run_id
                    previous = state['runs'].get(c.run_id, {})
                    # Do not restart completed training merely because postprocessing was interrupted.
                    trained = previous.get('training_complete', False)
                    runtime = read_json(wd / 'meta/mix20h_run_manifest.json')
                    trained = bool(trained or runtime.get('training_complete'))
                    finishing_required = trained and not all(path.is_file() for path in (
                        wd / 'candidates/step-50000/mix20h_evaluation_complete.json',
                        wd / 'meta/finished_at.txt', wd / 'results/reduced_best_hqnr.mat',
                        wd / 'results/full_best_hqnr.mat'))
                    if runtime.get('status') in {'stopped_budget', 'training_complete_eval_pending_budget'}:
                        state['runs'][c.run_id] = dict(previous, status='stopped_budget')
                        state['status'] = 'STOPPED_BUDGET'; break
                    admission_doc = dict(campaign_id=CAMPAIGN_ID, queue_revision=QUEUE_REVISION,
                                         run_id=c.run_id, server_id=server, pair_id=c.pair_id,
                                         status='admitted', deadline_at_utc=plan['deadline_at_utc'],
                                         config_sha256=plan['config_hashes'][c.run_id], admitted_at_utc=iso())
                    write_json(wd / 'meta/mix20h_admission.json', admission_doc)
                    start = time.time()
                    state['runs'][c.run_id] = dict(previous, status='running', started_at_utc=previous.get('started_at_utc', iso(start)),
                                                   training_complete=trained)
                    write_json(camp / 'status.json', state)
                    resumed = False
                    if not trained or finishing_required:
                        resume = checkpoint_for_resume(root, c.run_id)
                        if trained and resume and not re.fullmatch(r'(?:checkpoint(?:-budget)?|step)-50000', resume.name):
                            resume = None  # completed training may only finish evaluation, never repeat its tail
                        resumed = resume is not None
                        if (previous or finishing_required) and not resume:
                            state['runs'][c.run_id].update(status='evaluation_pending' if trained else 'failed',
                                                          reason='interrupted_without_exact_checkpoint')
                            event(root, 'failed', run_id=c.run_id, reason='no exact resume; never fresh retry'); continue
                        cmd = ['bash', 'tools/run.sh', c.run_id]
                        if resume:
                            cmd += ['--resume', str(resume)]
                        rc = run_command(root, cmd, wd / 'mix20h_runner.log')
                        state['runs'][c.run_id]['train_exit_code'] = rc
                        if rc != 0:
                            stopped_runtime = read_json(wd / 'meta/mix20h_run_manifest.json')
                            state['runs'][c.run_id].update(status='stopped_budget' if rc == 5 else ('evaluation_pending' if trained else 'failed'),
                                                          finished_at_utc=iso(), training_complete=bool(stopped_runtime.get('training_complete')),
                                                          actual_updates=stopped_runtime.get('actual_updates'))
                            hours = (time.time() - start) / 3600
                            state['runs'][c.run_id]['elapsed_hours'] = hours
                            budget['run_events'].append(dict(run_id=c.run_id, elapsed_hours=hours, status=state['runs'][c.run_id]['status']))
                            write_json(camp / 'budget.json', budget)
                            event(root, 'training_exit', run_id=c.run_id, exit_code=rc, elapsed_hours=hours)
                            if rc == 5:
                                state['status'] = 'STOPPED_BUDGET'; break
                            write_json(camp / 'status.json', state); continue
                        state['runs'][c.run_id]['training_complete'] = True
                        write_json(camp / 'status.json', state)
                    cmd = [sys.executable, 'tools/mix20h_postrun.py', c.run_id, '--device', 'cuda',
                           '--deadline-utc', plan['deadline_at_utc']]
                    if upload:
                        cmd += ['--upload']
                    rc = run_command(root, cmd, wd / 'mix20h_postrun.log')
                    post = read_json(wd / 'results/mix20h_postrun_status.json')
                    status = 'evaluation_pending' if rc else ('finished' if post.get('sheet_uploaded') else 'upload_pending')
                    hours = (time.time() - start) / 3600
                    state['runs'][c.run_id].update(status=status, evaluation_exit_code=rc,
                                                  postrun=post, finished_at_utc=iso(), end_to_end_hours=hours)
                    if not trained and not resumed and not previous and rc == 0:
                        samples.append(hours)
                    budget['run_events'].append(dict(run_id=c.run_id, elapsed_hours=hours, status=status))
                    event(root, 'run_finished', run_id=c.run_id, status=status, end_to_end_hours=hours)
                    write_json(camp / 'budget.json', budget); write_json(camp / 'status.json', state)
                if state['status'] in {'STOPPED_BUDGET', 'BUDGET_CLOSED'}:
                    break
            else:
                state['status'] = 'DONE'
            postrun_backlog(root, plan, state, upload)
            state['finished_at_utc'] = iso()
            state['overrun_seconds'] = max(0, time.time() - timestamp(plan['deadline_at_utc']))
            write_json(camp / 'status.json', state)
            event(root, 'closed', status=state['status'], overrun_seconds=state['overrun_seconds'])
    return 0


def report(root):
    plan = load_plan(root); state = read_json(root / CAMP / 'status.json')
    rows = []
    for c in cases_for(plan['server_id']):
        wd = root / 'work_dir' / c.run_id
        target = read_json(wd / 'results' / f'qrecon24_target_selection_{SELECTOR_ID}.json')
        exact = read_json(wd / 'results/exact50k_official_AON.json')
        raw_max = {}
        if (wd / 'checkpoint_metrics.csv').is_file():
            with (wd / 'checkpoint_metrics.csv').open() as stream:
                finite = []
                for record in csv.DictReader(stream):
                    try:
                        value = float(record['raw_original.hqnr'])
                        if math.isfinite(value):
                            finite.append(dict(step=int(float(record['step'])), hqnr=value))
                    except (ValueError, TypeError, KeyError):
                        continue
            raw_max = max(finite, key=lambda item: (item['hqnr'], -item['step'])) if finite else {}
        rows.append(dict(**c.to_dict(), runtime=state.get('runs', {}).get(c.run_id, {'status': 'not_admitted'}),
                         legacy_best=read_json(wd / 'best_hqnr_meta.json'), raw_max=raw_max, target=target, exact50k=exact))
    profiles = {}; pairs = []
    for profile in ('G23', 'B20A03'):
        subset = [r for r in rows if r['profile'] == profile]
        official = [r['target'] for r in subset if r['target'].get('target_status') == 'official'
                    and r['runtime'].get('postrun', {}).get('official_target_complete')]
        errors = [r['target']['ergas'] for r in official]
        profiles[profile] = dict(planned=len(subset), training_complete=sum(bool(r['runtime'].get('training_complete')) for r in subset),
                                 h_pass_runs=len(official), joint_pass_runs=sum(bool(r.get('joint_pass')) for r in official),
                                 target_ergas_values=errors,
                                 target_ergas_min=min(errors) if errors else None,
                                 target_ergas_median=statistics.median(errors) if errors else None)
    for i in range(0, len(rows), 2):
        members = {r['profile']: r for r in rows[i:i+2]}
        a, b = members['G23'], members['B20A03']
        valid = all(r['runtime'].get('postrun', {}).get('official_target_complete') and
                    r['target'].get('target_status') == 'official' for r in (a, b))
        verified = all(read_json(root/'work_dir'/r['run_id']/'meta/mix20h_run_manifest.json').get('pair_status') == 'pair_verified' for r in (a, b))
        exact_valid = all(r['exact50k'].get('official_complete') for r in (a,b))
        completed = all(r['runtime'].get('training_complete') and
                        r['runtime'].get('postrun', {}).get('official_target_complete') and
                        r['runtime'].get('postrun', {}).get('exact50k_complete') for r in (a, b))
        pairs.append(dict(pair_id=a['pair_id'], seed=a['seed'], tier=a['tier'], pair_verified=verified,
                          completed=completed, complete_verified_pair=completed and verified,
                          G23_h_pass=a['target'].get('target_status') == 'official',
                          B20A03_h_pass=b['target'].get('target_status') == 'official',
                          target_comparable=valid and verified,
                          target_E_B_minus_G=(b['target']['target']['ergas']-a['target']['target']['ergas']) if valid and verified else None,
                          exact50k_E_B_minus_G=(b['exact50k']['ergas']-a['exact50k']['ergas']) if exact_valid and verified else None,
                          reason=None if valid and verified else 'incomplete/unverified pair or one profile fails H threshold'))
    completion = {}
    for tier in ('base', 'reserve'):
        subset = [r for r in rows if r['tier'] == tier]
        completion[tier] = dict(planned=len(subset),
                                training_complete=sum(bool(r['runtime'].get('training_complete')) for r in subset),
                                completed_verified_pairs=sum(p['tier'] == tier and p['complete_verified_pair'] for p in pairs))
    for summary in profiles.values():
        summary['completed_verified_seed_pairs'] = sum(p['complete_verified_pair'] for p in pairs)
    return dict(campaign_id=CAMPAIGN_ID, queue_revision=QUEUE_REVISION, server_id=plan['server_id'],
                deadline_at_utc=plan['deadline_at_utc'], status=state.get('status'), runs=rows,
                completion_summary=completion, profile_summary=profiles, paired_comparisons=pairs,
                note='Same server/seed pairs only; mixed beta runs are not one homogeneous seed cohort.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['start', 'prepare', 'run', 'status', 'report'])
    ap.add_argument('--server', choices=['s1', 's2', 's3', 's4', 's5'])
    ap.add_argument('--start-at', help='Optional common timezone-aware T0; start defaults to local first start')
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--launch', action='store_true')
    ap.add_argument('--wait', action='store_true')
    uploads = ap.add_mutually_exclusive_group()
    uploads.add_argument('--upload', action='store_true')
    uploads.add_argument('--no-upload', action='store_true', help='start only: omit automatic Sheet upload')
    ap.add_argument('--check-assets', action='store_true', help='read-only CPU hash checks in prepare preview')
    args = ap.parse_args()
    try:
        if args.command == 'start':
            result = start_campaign(ROOT, args.server, args.start_at, args.dry_run, upload=not args.no_upload)
            if args.check_assets and args.dry_run:
                result['verified_assets'] = verify_assets(ROOT, result['server'])
            print(json.dumps({k: v for k, v in result.items() if k != 'inventory'}, indent=2))
        elif args.command == 'prepare':
            server = args.server or (ROOT / 'gspread/server.txt').read_text().strip()
            from tools.gen_pakd50_configs import server_id
            result = prepare(ROOT, server_id(server), args.start_at, args.dry_run, args.upload)
            if args.check_assets and args.dry_run:
                result['verified_assets'] = verify_assets(ROOT, server_id(server))
            # Avoid dumping stored queue contents and unbounded process output in normal UI.
            print(json.dumps({k: v for k, v in result.items() if k != 'inventory'}, indent=2))
            if args.launch and not args.dry_run:
                print(json.dumps(launch_background(ROOT, args.upload)))
        elif args.command == 'run':
            return run_campaign(ROOT, args.wait, args.upload)
        elif args.command == 'status':
            p = load_plan(ROOT)
            print(json.dumps(dict(plan=p, status=read_json(ROOT / CAMP / 'status.json'),
                                  remaining_hours=(timestamp(p['deadline_at_utc']) - time.time()) / 3600), indent=2))
        else:
            print(json.dumps(report(ROOT), indent=2))
    except BlockingIOError:
        print('WAIT_RESOURCE: another runner/migration holds the shared lock', file=sys.stderr); return 3
    except (ValueError, KeyError, OSError) as e:
        print(f'BLOCKED_LOCAL: {e}', file=sys.stderr); return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
