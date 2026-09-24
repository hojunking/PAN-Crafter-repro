"""Cooperative G23-to-MAIN-A cutover; no process signals or foreign-lane writes.

Run cutover on the host so Docker identity and actual work_dir mounts can be
checked. The new GPU container only verifies the resulting local receipt.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import time

from maina_hqnr.common import (ROOT, camp, verify_server, read, read_json, atomic_json,
    immutable_json, object_sha, sha256, utcnow, RuntimePaused, locked)

OLD_CAMPAIGN = 'PANDA_G23_SENS_WV3_S45_20260923_v1'
REASON = 'OPERATOR_MIGRATION_TO_MAINA_HQNR'


def process_record(pid):
    base = Path('/proc') / str(pid)
    try:
        args = [part.decode(errors='replace') for part in (base / 'cmdline').read_bytes().split(b'\0') if part]
        fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), args=args, cwd=str((base / 'cwd').resolve(strict=True)),
                    start_ticks=fields[19], uid=base.stat().st_uid)
    except (OSError, ValueError, IndexError):
        return None


def inventory(root=ROOT):
    shared = (Path(root) / 'work_dir').resolve()
    rows = []
    for path in Path('/proc').glob('[0-9]*'):
        row = process_record(path.name)
        if not row or row['pid'] == os.getpid():
            continue
        for index, arg in enumerate(row['args']):
            name = Path(arg).name
            if name not in ('g23sens_runner.py', '_watchdog.sh', '_run_cases.sh'):
                continue
            script = (Path(row['cwd']) / arg).resolve()
            script_root = script.parent.parent
            if ((script_root / 'work_dir').resolve() != shared
                    and (Path(row['cwd']) / 'work_dir').resolve() != shared):
                continue
            server = None
            if '--server' in row['args']:
                pos = row['args'].index('--server')
                if pos + 1 < len(row['args']):
                    server = row['args'][pos + 1]
            row.update(script=name, script_path=str(script), script_root=str(script_root),
                       server=server, action=row['args'][index + 1] if index + 1 < len(row['args']) else '',
                       shared_work_dir=str(shared))
            rows.append(row)
            break
    return rows


def docker_inventory(server):
    """Inspect only the recognized old lane, without exposing environment secrets."""
    try:
        ids = subprocess.check_output(['docker', 'ps', '-q', '--filter',
            'label=org.pancrafter.g23sens.server=' + server], text=True, timeout=10).split()
    except FileNotFoundError:
        return []
    if not ids:
        return []
    result = []
    for raw in json.loads(subprocess.check_output(['docker', 'inspect', *ids], text=True, timeout=10)):
        result.append(dict(id=raw['Id'], image=raw['Image'], pid=raw['State']['Pid'],
            running=raw['State']['Running'], restart_policy=raw['HostConfig']['RestartPolicy']['Name'],
            command=raw['Config'].get('Cmd'), labels=raw['Config'].get('Labels') or {},
            mounts=[dict(source=m['Source'], destination=m['Destination'], rw=m['RW']) for m in raw['Mounts']]))
    return result


def gpu_rows():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    return [dict(pid=int(parts[0].strip()), gpu_uuid=parts[1].strip())
            for line in text.splitlines() if len(parts := line.split(',')) == 2 and parts[0].strip().isdigit()]


def ensure_gpu_idle(gpu_uuid):
    if not gpu_uuid:
        raise RuntimePaused('A concrete local GPU UUID is required')
    active = [row for row in gpu_rows() if row['gpu_uuid'] == gpu_uuid]
    if active:
        raise RuntimePaused('WAIT_GPU_RELEASE: ' + str(active))


def _state(root, server):
    value = read(Path(root) / 'work_dir/g23sens' / server / 'state.json')
    if value and (value.get('server') != server or value.get('campaign_id') != OLD_CAMPAIGN):
        raise ValueError('Old state belongs to another campaign/server')
    return value


def _runner_support(script):
    tree = ast.parse(Path(script).read_text())
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    return dict(stop='stop' in strings and 'STOP_AFTER_RUN' in strings,
                safe_now='--safe-now' in strings and 'STOP_NOW_SAFE' in strings)


def _outbox_snapshot(old):
    return {str(path.relative_to(old)): sha256(path) for path in sorted((old / 'outbox').rglob('*')) if path.is_file()}


def request_boundary(root, server, gpu_uuid, *, activated=False):
    verify_server(server)
    if not activated:
        raise PermissionError('Explicit local start required for cutover')
    root = Path(root); folder = camp(root, server); old = root / 'work_dir/g23sens' / server
    with locked(folder / 'handoff/request.lock'):
        previous = read(folder / 'handoff/request.json')
        shared = str((root / 'work_dir').resolve())
        if previous:
            if previous['server'] != server or previous['shared_work_dir'] != shared or previous['gpu_uuid'] != gpu_uuid:
                raise ValueError('Cutover receipt belongs to a different owner/GPU')
            if previous['old_present'] and not (old / 'STOP_AFTER_RUN').exists():
                raise RuntimePaused('Old admission was reenabled; no silent override')
            return previous
        rows = inventory(root)
        unknown = [row for row in rows if row['server'] not in ('s1', 's2', 's3', 's4', 's5')]
        if unknown:
            raise RuntimePaused('Unverified watchdog/owner; use its supported stop procedure before cutover: ' + str(unknown))
        local = [row for row in rows if row['server'] == server]
        if any(row['script'] != 'g23sens_runner.py' for row in local):
            raise RuntimePaused('Only the recognized G23 runner is an automatic migration target')
        state = _state(root, server)
        containers = docker_inventory(server)
        if len(containers) > 1:
            raise RuntimePaused('Multiple old G23 container owners; resolve explicitly')
        for container in containers:
            mounts_shared = any(str(Path(m['source']).resolve()) == shared for m in container['mounts'])
            if not mounts_shared or container['restart_policy'] not in ('', 'no'):
                raise RuntimePaused('Old Docker work_dir/restart ownership not safe for automatic handoff')
        scripts = {row['script_path'] for row in local}
        if containers:
            release = containers[0]['labels'].get('org.pancrafter.g23sens.runtime')
            if not release or (Path(release) / 'work_dir').resolve() != Path(shared):
                raise RuntimePaused('Old container runtime does not map to the actual shared work_dir')
            scripts.add(str(Path(release) / 'tools/g23sens_runner.py'))
        capabilities = {script: _runner_support(script) for script in scripts}
        if scripts and not all(value['stop'] for value in capabilities.values()):
            raise RuntimePaused('Active release has no verified stop boundary; no signals will be guessed')
        safe = bool(scripts) and all(value['safe_now'] for value in capabilities.values())
        request = dict(schema='MAINA_G23_CUTOVER_v1', server=server, gpu_uuid=gpu_uuid,
            shared_work_dir=shared, old_campaign=OLD_CAMPAIGN, old_present=old.exists(),
            original_state=state, original_state_sha256=object_sha(state), original_processes=local,
            protected_other_lane_processes=[row for row in rows if row['server'] != server],
            old_containers=containers, old_runtime_release=read(old / 'runtime_release.json'),
            original_outbox=_outbox_snapshot(old), capability=capabilities,
            policy='SAFE_UPDATE_BOUNDARY' if safe else 'CURRENT_RUN_BOUNDARY',
            fallback_reason=None if safe else 'No active verified safe-now implementation; current-run boundary only',
            pause_reason=REASON, requested_at_utc=utcnow(), signals_sent=False)
        if old.exists():
            # Same public markers as the verified runner; no controller/queue rewrites.
            if not (old / 'STOP_AFTER_RUN').exists():
                atomic_json(old / 'STOP_AFTER_RUN', dict(operator_request=True, reason=REASON, at_utc=utcnow()))
            if safe and not (old / 'STOP_NOW_SAFE').exists():
                atomic_json(old / 'STOP_NOW_SAFE', dict(operator_request=True, reason=REASON, at_utc=utcnow()))
        immutable_json(folder / 'handoff/request.json', request)
        return request


def _preserved_run(root, server, old_state):
    run_id = old_state.get('active_run_id')
    if not run_id:
        return dict(status='NO_ACTIVE_RUN')
    if Path(run_id).name != run_id:
        raise ValueError('Unsafe old run path')
    attempt = old_state.get('runs', {}).get(run_id, {}).get('attempt', old_state.get('attempt', 0))
    if not isinstance(attempt, int) or attempt < 0:
        raise ValueError('Invalid old attempt')
    work = Path(root) / 'work_dir/g23sens' / server / 'runs' / run_id / f'attempt{attempt:03d}'
    identity = read(work / 'last/identity.json'); state_path = work / 'last/training_state.pt'
    if not identity.get('full_state') or not state_path.is_file() or sha256(state_path) != identity.get('training_state_sha256'):
        raise RuntimePaused('OLD_STATE_NOT_PRESERVED: full-state checkpoint is missing or unverifiable')
    import torch
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    required = ('model_state', 'optimizer', 'scheduler', 'rng', 'sampler', 'update', 'source_identity', 'bindings_sha256')
    if not state.get('full_state') or any(key not in state for key in required):
        raise RuntimePaused('OLD_STATE_NOT_PRESERVED: optimizer/RNG/sampler/source missing')
    if state['update'] != identity.get('update') or state['scheduler'].get('last_epoch') != state['update']:
        raise RuntimePaused('Old optimizer/scheduler update boundary differs')
    if not any(name.startswith('aligner.') for name in state['model_state']) or not any(name.startswith('backbone.') for name in state['model_state']):
        raise ValueError('Old full state does not contain both A and U')
    from fh12.model import state_hash
    import numpy as np
    import random
    update = state['update']
    if isinstance(update, bool) or not isinstance(update, int) or not 0 <= update <= 50000:
        raise RuntimePaused('Old completed-update count invalid')
    for key in ('source_identity', 'bindings_sha256', 'config_sha256', 'stream_sha256', 'run_id', 'attempt'):
        if key not in identity or state.get(key) != identity[key]:
            raise RuntimePaused('Old full-state identity differs: ' + key)
    if state.get('run_id') != run_id or state.get('attempt') != attempt:
        raise RuntimePaused('Old full-state run/attempt differs')
    if state_hash(state['model_state']) != identity.get('state_hash'):
        raise RuntimePaused('Old full-state A/U tensor identity differs')
    rng = state['rng']; sampler = state['sampler']; optimizer = state['optimizer']
    if not isinstance(rng, dict) or set(rng) != {'python', 'numpy', 'torch', 'cuda'}:
        raise RuntimePaused('Old Python/NumPy/Torch/CUDA RNG state missing')
    try:
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
        torch.Generator(device='cpu').set_state(rng['torch'])
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimePaused('Old RNG state is not restorable') from error
    if (not isinstance(rng['cuda'], (list, tuple)) or not rng['cuda']
            or any(not isinstance(v, torch.Tensor) or v.dtype != torch.uint8 or v.ndim != 1 or v.numel() == 0
                   for v in rng['cuda'])):
        raise RuntimePaused('Old CUDA RNG state is missing or invalid')
    stream_manifest = read(work / 'stream_manifest.json')
    if (not isinstance(sampler, dict) or sampler.get('schema') != 'G23SENS_STREAM_CURSOR_v1'
            or sampler.get('cursor') != update or sampler.get('completed_updates') != update
            or not stream_manifest or object_sha(stream_manifest) != state['stream_sha256']
            or sampler.get('manifest_sha256') != stream_manifest.get('manifest_sha256')
            or sampler.get('metadata_sha256') != stream_manifest.get('metadata_sha256')
            or not isinstance(sampler.get('consumed_prefix_sha256'), str)
            or len(sampler['consumed_prefix_sha256']) != 64):
        raise RuntimePaused('Old sampler cursor/stream identity is incomplete or changed')
    if (not isinstance(optimizer, dict) or set(optimizer) != {'state', 'param_groups'}
            or len(optimizer['param_groups']) != 2
            or {g.get('name') for g in optimizer['param_groups']} != {'U', 'A'}):
        raise RuntimePaused('Old A/U optimizer state is incomplete')
    parameters = [p for group in optimizer['param_groups'] for p in group.get('params', [])]
    if not parameters or len(set(parameters)) != len(parameters):
        raise RuntimePaused('Old optimizer parameter ownership differs')
    if update > 0:
        if set(optimizer['state']) != set(parameters):
            raise RuntimePaused('Old optimizer moments are incomplete')
        for item in optimizer['state'].values():
            if not {'step', 'exp_avg', 'exp_avg_sq'} <= set(item) or float(item['step']) != update:
                raise RuntimePaused('Old optimizer step/moments differ from completed updates')
    for name, expected in (('meta/config.resolved.yaml', state['config_sha256']),
                           ('meta/consumed_bindings.json', state['bindings_sha256'])):
        from maina_hqnr.common import read_config
        path = work / name
        if not path.is_file() or object_sha(read_config(path)) != expected:
            raise RuntimePaused('Old full-state consumed config/assets identity differs: ' + name)
    return dict(status='PRESERVED', run_id=run_id, attempt=attempt, actual_updates=state['update'],
                full_state_identity=identity, state_path=str(state_path),
                candidates={str(path.relative_to(work)): sha256(path) for path in sorted((work / 'candidates').rglob('identity.json'))},
                evaluation_ledger={str(path.relative_to(work)): sha256(path) for path in sorted((work / 'official').rglob('*.json'))},
                training_status=read(work / 'meta/training_status.json'))


def verify_boundary(root, server, request, gpu_uuid):
    root = Path(root); folder = camp(root, verify_server(server)); old = root / 'work_dir/g23sens' / server
    if request['server'] != server or request['gpu_uuid'] != gpu_uuid or request['shared_work_dir'] != str((root / 'work_dir').resolve()):
        raise ValueError('Cutover ownership mismatch')
    if request['old_present'] and not (old / 'STOP_AFTER_RUN').exists():
        raise RuntimePaused('Old admission stop disappeared')
    if any(row['server'] == server for row in inventory(root)) or docker_inventory(server):
        raise RuntimePaused('WAIT_OLD_SAFE_BOUNDARY: old controller/worker still running')
    ensure_gpu_idle(gpu_uuid)
    state = _state(root, server)
    evidence = _preserved_run(root, server, request['original_state'])
    # Admission can race the stop request; verify the final active run as well.
    final_evidence = _preserved_run(root, server, state)
    if request['old_present']:
        for name in ('service.lock', 'runner.lock'):
            with locked(old / name):
                pass
        with locked(root / 'work_dir/g23sens' / ('gpu-' + gpu_uuid + '.lock')):
            pass
    receipt = dict(status='COMPLETE', old_campaign=OLD_CAMPAIGN, server=server,
        shared_work_dir=request['shared_work_dir'], gpu_uuid=gpu_uuid, pause_reason=REASON,
        paused_at_utc=utcnow(), request_sha256=object_sha(request), old_state=state,
        preserved_run=evidence, final_active_run=final_evidence,
        preserved_outbox=_outbox_snapshot(old), gpu_released=True,
        old_processes_exited=True, old_new_admission_blocked=True, signals_sent=False,
        original_results_not_deleted=True, original_state_not_marked_complete=True)
    atomic_json(folder / 'handoff/complete.json', receipt)
    return receipt


def wait_boundary(root, server, gpu_uuid, *, activated=False, poll_seconds=10):
    request = request_boundary(root, server, gpu_uuid, activated=activated)
    while True:
        if (camp(root, server) / 'STOP_NOW_SAFE').exists() or (camp(root, server) / 'STOP_AFTER_RUN').exists():
            raise RuntimePaused('New campaign operator stop retained during cutover')
        try:
            return verify_boundary(root, server, request, gpu_uuid)
        except RuntimePaused as error:
            if not str(error).startswith(('WAIT_OLD_SAFE_BOUNDARY', 'WAIT_GPU_RELEASE')):
                raise
            print(str(error), flush=True)
            time.sleep(min(float(poll_seconds), 30.))


def verify_local_receipt(root, server, gpu_uuid):
    """Container-side recheck without requiring a Docker socket."""
    receipt = read(camp(root, server) / 'handoff/complete.json')
    if (receipt.get('status') != 'COMPLETE' or receipt.get('server') != server
            or receipt.get('gpu_uuid') != gpu_uuid
            or receipt.get('shared_work_dir') != str((Path(root) / 'work_dir').resolve())):
        raise RuntimePaused('Host-side safe cutover has not completed for this workspace/GPU')
    request = read(camp(root, server) / 'handoff/request.json')
    if object_sha(request) != receipt['request_sha256']:
        raise ValueError('Cutover request changed')
    old = Path(root) / 'work_dir/g23sens' / server
    if request['old_present'] and not (old / 'STOP_AFTER_RUN').exists():
        raise RuntimePaused('Old admission reenabled')
    ensure_gpu_idle(gpu_uuid)
    return receipt
