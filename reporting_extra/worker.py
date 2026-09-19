"""CPU-only, additive reporting service; never controls training or its queue."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from fh12.common import ROOT, atomic_json, read_json, sha256, utcnow

SCHEMA = 'PAN_EXTRA_WORKER_v1'
POLL_SECONDS = 120
RETRY_SECONDS = 900


def state_dir(root, server):
    if server not in {'s1', 's2', 's3', 's4', 's5'}:
        raise ValueError('Expected server s1..s5')
    return Path(root) / 'work_dir/_extra_metrics' / server


def ready_runs(root, server):
    """Only the local server's completed, real (not reuse-link) FH runs."""
    found = []
    for campaign in ('FH12', 'FH20R1'):
        for folder in (Path(root) / 'work_dir').glob(f'{campaign}_{server.upper()}_*'):
            status = folder / 'official/postrun_status.json'
            training = folder / 'meta/training_status.json'
            if not status.is_file() or not training.is_file() or (folder / 'meta/reuse_reference.json').exists():
                continue
            try:
                official, train = read_json(status), read_json(training)
            except (OSError, ValueError):
                continue  # An incomplete/atomic replacement must not admit a live run.
            if (official.get('official_complete') and train.get('training_complete')
                    and train.get('actual_updates') == 50000):
                found.append((status.stat().st_mtime_ns, folder.name))
    return [name for _, name in sorted(found, reverse=True)]


def job_fingerprint(root, run):
    """Cheap retry/skip key; the evaluator performs the full cryptographic audit."""
    root = Path(root)
    folder = root / 'work_dir' / run
    paths = list((folder / 'official').glob('*.json'))
    paths += list((folder / 'supplemental_metrics').glob('*.json'))
    paths += list((root / 'reporting_extra').glob('*.py'))
    paths += [root / 'tools/metrics/jqm.py', folder / 'meta/config.resolved.yaml']
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(sha256(path).encode())
    return digest.hexdigest()


def cpu_environment():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               NUMEXPR_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    return env


def process_run(root, run):
    """Single-process job with a separate lock; failure cannot alter original files."""
    if not re.fullmatch(r'(FH12|FH20R1)_S[1-5]_[A-Za-z0-9_]+', run):
        raise ValueError('Invalid supplemental run ID')
    from tools.fh12_runner import detect_server
    server = detect_server(root)
    if not run.startswith(('FH12_' + server.upper() + '_', 'FH20R1_' + server.upper() + '_')):
        raise ValueError('Refusing a run from a different server')
    folder = Path(root) / 'work_dir' / run / 'supplemental_metrics'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        from reporting_extra.evaluation import process
        from reporting_extra.upload import upload_with_base_recovery
        print(f'[{utcnow()}] supplemental evaluation {run}', flush=True)
        process(run, root=root, device='cpu')
        receipt = upload_with_base_recovery(run, root=root)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        return receipt


def run_job(root, server, run):
    log = state_dir(root, server) / (run + '.log')
    with log.open('a') as stream:
        result = subprocess.run(
            [sys.executable, '-u', str(Path(root) / 'tools/extra_metrics.py'),
             'process', '--server', server, '--run', run],
            cwd=root, env=cpu_environment(), stdout=stream, stderr=subprocess.STDOUT,
        )
    return result.returncode


def tick(root, server, state, job=run_job, now=time.time):
    folder = state_dir(root, server)
    folder.mkdir(parents=True, exist_ok=True)
    jobs = state.setdefault('jobs', {})
    candidates = ready_runs(root, server)
    state.update(schema=SCHEMA, server_id=server, pid=os.getpid(), updated_at_utc=utcnow(),
                 ready_count=len(candidates), device='cpu', threads=2)
    for run in candidates:
        fingerprint = job_fingerprint(root, run)
        prev = jobs.get(run, {})
        if prev.get('fingerprint') == fingerprint:
            if prev.get('status') == 'VERIFIED' or prev.get('retry_after', 0) > now():
                continue
        state['active_run'] = run
        jobs[run] = dict(status='RUNNING', started_at_utc=utcnow(),
                         attempts=prev.get('attempts', 0) + 1)
        atomic_json(folder / 'status.json', state)
        try:
            code = job(root, server, run)
            if code != 0:
                raise RuntimeError(f'Job exit {code}; see {run}.log')
            receipt = read_json(Path(root) / 'work_dir' / run / 'supplemental_metrics/upload_receipt.json')
            report = Path(root) / 'work_dir' / run / 'supplemental_metrics/report.json'
            from fh12.upload import GIDS
            body = read_json(report)
            if (receipt.get('readback_verified') is not True or receipt.get('run_id') != run
                    or receipt.get('report_sha256') != sha256(report)
                    or receipt.get('schema') != 'PAN_SUPPLEMENTAL_UPLOAD_v1'
                    or receipt.get('server_id') != server or receipt.get('gid') != GIDS[server]
                    or body.get('run_id') != run or body.get('server_id') != server
                    or receipt.get('campaign_id') != body.get('campaign_id')):
                raise ValueError('Job did not produce a matching verified upload receipt')
            jobs[run].update(status='VERIFIED', finished_at_utc=utcnow(), row=receipt['row'])
        except Exception as exc:
            jobs[run].update(status='RETRY_PENDING', error=str(exc), retry_after=now() + RETRY_SECONDS)
        jobs[run]['fingerprint'] = job_fingerprint(root, run)
        state.update(active_run=None, updated_at_utc=utcnow())
        atomic_json(folder / 'status.json', state)
    atomic_json(folder / 'status.json', state)
    return state


def watch(root, server, once=False):
    folder = state_dir(root, server)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'ALREADY_RUNNING'}
        os.nice(15)
        path = folder / 'status.json'
        state = read_json(path) if path.is_file() else {}
        while True:
            tick(root, server, state)
            if once:
                return state
            time.sleep(POLL_SECONDS)


def running(root, server):
    folder = state_dir(root, server)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def install_cron(root, server, runner=subprocess.run):
    """Add only our own recoverable restart line; preserve every training cron line."""
    marker = f'# PANCRAFTER-EXTRA-METRICS-{server}'
    current = runner(['crontab', '-l'], capture_output=True, text=True)
    if current.returncode not in (0, 1):
        raise RuntimeError('Cannot safely read existing crontab')
    if current.returncode == 1 and 'no crontab' not in current.stderr.lower():
        raise RuntimeError('Crontab read failed: ' + current.stderr.strip())
    lines = [s for s in current.stdout.splitlines() if not s.rstrip().endswith(marker)]
    command = ' '.join(shlex.quote(str(s)) for s in (
        sys.executable, Path(root) / 'tools/extra_metrics.py', 'start', '--server', server, '--if-enabled'))
    lines.append(f'*/5 * * * * {command} >/dev/null 2>&1 {marker}')
    runner(['crontab', '-'], input='\n'.join(lines) + '\n', text=True, check=True, capture_output=True)


def start(root, server, if_enabled=False, cron=True):
    folder = state_dir(root, server)
    folder.mkdir(parents=True, exist_ok=True)
    enabled = folder / 'enabled.json'
    if if_enabled and not enabled.is_file():
        return {'status': 'NOT_ENABLED'}
    if not enabled.is_file():
        atomic_json(enabled, dict(schema=SCHEMA, server_id=server, enabled_at_utc=utcnow(),
                                 scope='FH12/FH20R1 completed local runs; CPU; additive metrics'))
    result = {'server': server, 'log': str(folder / 'worker.log')}
    if not if_enabled and cron:
        try:
            install_cron(root, server)
            result['restart'] = 'cron_every_5_minutes'
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            result['restart_warning'] = str(exc)
    if running(root, server):
        return dict(result, status='ALREADY_RUNNING')
    with (folder / 'worker.log').open('a') as stream:
        child = subprocess.Popen(
            [sys.executable, '-u', str(Path(root) / 'tools/extra_metrics.py'), 'watch', '--server', server],
            cwd=root, env=cpu_environment(), stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    return dict(result, status='STARTED', pid=child.pid)
