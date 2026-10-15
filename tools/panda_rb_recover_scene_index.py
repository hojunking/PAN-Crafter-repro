#!/usr/bin/env python3
"""Audited serialization-only recovery of the original B01 frozen release.

The original files/checkpoint/cache identities remain intact. Only the native
row dictionary constructor is replaced in memory, and every native/stress
context records the effective evaluator patch. No training math is replaced.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ORIGINAL_EVALUATION_SHA = '51f249d96e6543a4ed3cede1daeb5873cefc3510c74f83c749ee6eedc97b18da'
PATCH_ID = 'PANDA_B01_SCENE_INDEX_SERIALIZATION_v1'
OLD = 'per_scene.append(dict(split=split, scene_index=i, source_h5_row=i, **metrics,'
NEW = 'per_scene.append(dict(metrics, split=split, scene_index=i, source_h5_row=i,'
ROOT = Path(__file__).resolve().parents[1]


def digest(value): return hashlib.sha256(value).hexdigest()


def corrected_function(source, expected_sha=ORIGINAL_EVALUATION_SHA):
    if digest(source) != expected_sha:
        raise ValueError('Recovery only admits the exact audited original evaluator')
    text = source.decode()
    if text.count(OLD) != 1: raise ValueError('Expected exactly one native row constructor')
    fixed = text.replace(OLD, NEW)
    functions = [n for n in ast.parse(fixed).body if isinstance(n, ast.FunctionDef) and n.name == '_native_selection']
    if len(functions) != 1: raise ValueError('Unexpected native evaluator structure')
    node = functions[0]
    segment = ast.get_source_segment(fixed, node)
    identity = dict(patch_id=PATCH_ID, original_evaluation_sha256=digest(source),
                    corrected_evaluation_sha256=digest(fixed.encode()), native_function_sha256=digest(segment.encode()),
                    change='dict(metrics, authoritative scene metadata); no inference, loss, selector, or metric math changes')
    return node, identity


def install_patch(evaluation, root, receipt):
    from panda_rb.common import object_sha, sha256
    root = Path(root).resolve()
    if Path(evaluation.__file__).resolve() != root / 'panda_rb/evaluation.py':
        raise ValueError('Evaluator is not from the frozen source mount')
    node, patch = corrected_function(Path(evaluation.__file__).read_bytes())
    patch['entrypoint_sha256'] = sha256(__file__)
    if patch != receipt['patch']: raise ValueError('Recovery script/function bytes differ from the receipt')
    if getattr(evaluation, '_scene_index_recovery', None):
        raise ValueError('Refusing a second runtime override')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(Path(__file__)) + ':native-row-fix', 'exec'),
         evaluation.__dict__, namespace)
    evaluation._native_selection = namespace['_native_selection']
    original_context = evaluation._context
    def context(*args, **kwargs):
        # All original checkpoint/config/data/source checks run BEFORE tagging.
        result = original_context(*args, **kwargs)
        original = result['context']['evaluator_identity']
        effective = dict(original, base_evaluator_content_sha256=original['content_sha256'],
                         serialization_hotfix=patch)
        effective['content_sha256'] = object_sha(dict(base=original['content_sha256'], serialization_hotfix=patch))
        result['context'] = dict(result['context'], evaluator_identity=effective,
                                 evaluation_runtime_patch=patch)
        return result
    evaluation._context = context
    evaluation._scene_index_recovery = patch
    return patch


def _bootstrap(root):
    root = Path(root).resolve()
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    return root


def run_inside(server, root, receipt_path):
    root = _bootstrap(root)
    from panda_rb.common import docker_required, read_json, verify_files, sha256, server_dir, append_event
    from panda_rb import evaluation
    from panda_rb.controller import run
    docker_required()
    receipt = read_json(receipt_path)
    if receipt['server'] != server or receipt['root'] != str(root): raise ValueError('Recovery server/root mismatch')
    release = read_json(root / 'runtime_release.json')
    if release['release_sha256'] != receipt['release_sha256']:
        raise ValueError('Not the original frozen training release')
    verify_files(release['files'], root)
    for path, expected in receipt['protected_files'].items():
        if sha256(path) != expected: raise ValueError('Preserved checkpoint/cache evidence changed: ' + path)
    patch = install_patch(evaluation, root, receipt)
    append_event(server_dir(server, root) / 'events.jsonl', dict(event='SERIALIZATION_RECOVERY_ACTIVE', patch=patch))
    print(json.dumps(dict(event='SERIALIZATION_RECOVERY_ACTIVE', patch=patch, retrain_completed_models=False)), flush=True)
    return run(server, root=root, binding=receipt['binding'], device='cuda', retry_technical=True)


def launch(server, root=ROOT, dry_run=False):
    root = _bootstrap(root)
    from panda_rb.common import read_json, read, sha256, atomic_json, locked, server_dir, verify_files, utcnow
    from panda_rb.plan import campaign_dir, run_dir, training_runs
    from panda_rb.deployment import DEFAULT_IMAGE, asset_mounts, git_mounts, legacy_processes
    control = server_dir(server, root)
    deployment = read_json(control / 'deployment.json')
    if deployment['server'] != server or deployment['root'] != str(root) or deployment['image_id'] != DEFAULT_IMAGE:
        raise ValueError('Recovery must preserve the original server/root/image')
    release = campaign_dir(root) / 'releases' / deployment['release_sha256']
    manifest = read_json(release / 'runtime_release.json')
    if manifest['release_sha256'] != deployment['release_sha256']: raise ValueError('Release identity mismatch')
    verify_files(manifest['files'], release)
    _node, patch = corrected_function((release / 'panda_rb/evaluation.py').read_bytes())
    patch['entrypoint_sha256'] = sha256(__file__)
    directory = control / 'recoveries' / (PATCH_ID + '_' + patch['entrypoint_sha256'][:16])
    protected = {str(control / 'deployment.json'): sha256(control / 'deployment.json'),
                 deployment['binding']: sha256(deployment['binding'])}
    completed = []
    for case in training_runs(server, root):
        wd = run_dir(case['run_id'], root); selected = wd / 'checkpoints/selection_manifest.json'
        if not selected.exists(): continue
        doc = read_json(selected)
        if doc.get('actual_updates') != 50000 or not doc.get('complete'):
            raise ValueError('Partial checkpoint selection cannot be treated as completed training')
        protected[str(selected)] = sha256(selected)
        for name in ('exact50000', 'val_selected'):
            for artifact in ('identity.json', 'model.safetensors'):
                path = wd / 'checkpoints' / name / artifact
                protected[str(path)] = sha256(path)
        for path in (wd / 'native').rglob('testimg*.npy'): protected[str(path)] = sha256(path)
        completed.append(case['run_id'])
    if not completed: raise ValueError('This recovery requires an existing completed 50K source')
    for path in (campaign_dir(root) / 'common/weights').iterdir():
        if path.is_file(): protected[str(path)] = sha256(path)
    receipt = dict(schema='PANDA_B01_SERIALIZATION_RECOVERY_v1', server=server, root=str(root),
        release_sha256=deployment['release_sha256'], binding=deployment['binding'], image_id=DEFAULT_IMAGE,
        patch=patch, protected_files=protected, completed_training_not_repeated=completed)
    if dry_run: return dict(dry_run=True, receipt=receipt, no_runtime_mutation=True)
    with locked(campaign_dir(root) / 'control/launch.lock'):
        owner = read(root / 'work_dir/_panda_rb/local_owner.json')
        if owner.get('server') != server: raise ValueError('Local B01 owner belongs to a different server')
        active = subprocess.check_output(['docker', 'ps', '--filter', 'label=panda_rb.campaign=B01',
            '--filter', 'label=panda_rb.server=' + server, '--format', '{{.Names}}'], text=True).strip()
        if active: return dict(already_running=active)
        state = read(control / 'status.json')
        if state.get('state') == 'STOP_FOR_REVIEW': return dict(state='STOP_FOR_REVIEW')
        if not directory.exists() and (state.get('state') != 'TECHNICAL_FAILURE'
                or state.get('error') != "dict() got multiple values for keyword argument 'scene_index'"):
            raise ValueError('First recovery must match the audited scene_index failure')
        gpu = deployment['gpu']
        busy = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True).strip()
        if busy or legacy_processes(root): raise RuntimeError('Existing GPU/PAN owner; nothing is terminated')
        image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', DEFAULT_IMAGE], text=True))[0]['Id']
        if image != DEFAULT_IMAGE: raise ValueError('Pinned image is unavailable')
        directory.mkdir(parents=True, exist_ok=True)
        entry = directory / 'entrypoint.py'
        if entry.exists() and sha256(entry) != patch['entrypoint_sha256']: raise ValueError('Recovery entrypoint changed')
        if not entry.exists(): shutil.copyfile(__file__, entry)
        # One immutable receipt per attempt, also preserving the original error.
        import uuid
        attempt = uuid.uuid4().hex
        receipt_path = directory / (attempt + '.json')
        atomic_json(receipt_path, receipt)
        atomic_json(directory / (attempt + '_prior_status.json'), state)
        name = f'panda-rb-b01-{server}-sceneindex-{attempt[:8]}'
        command = ['docker', 'run', '--detach', '--name', name, '--network', 'none', '--pid', 'host',
            '--label', 'panda_rb.campaign=B01', '--label', 'panda_rb.server=' + server,
            '--label', 'panda_rb.recovery=' + PATCH_ID, '--gpus', 'device=' + gpu,
            '--user', f'{os.getuid()}:{os.getgid()}', '--shm-size', '8g',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=2', '-e', 'PYTHONUNBUFFERED=1',
            '-e', 'HF_HOME=/tmp/panda_rb_hf', '-e', 'MPLCONFIGDIR=/tmp/panda_rb_mpl',
            '-e', 'PANDA_RB_IMAGE_ID=' + DEFAULT_IMAGE,
            '-v', f'{release}:{root}:ro', '-v', f'{root / "work_dir"}:{root / "work_dir"}:rw',
            '-v', f'{entry}:/opt/panda_rb_scene_index_recovery.py:ro']
        for path in sorted(set(git_mounts(root)) | set(asset_mounts(root, deployment['binding']))):
            command += ['-v', f'{path}:{path}:ro']
        if os.environ.get('PANCRAFTER_DLPAN'):
            command += ['-e', 'PANCRAFTER_DLPAN=' + str(Path(os.environ['PANCRAFTER_DLPAN']).resolve())]
        command += ['-w', str(root), '--entrypoint', 'python', DEFAULT_IMAGE,
            '/opt/panda_rb_scene_index_recovery.py', '--inside', '--server', server,
            '--root', str(root), '--receipt', str(receipt_path)]
        if legacy_processes(root): raise RuntimeError('PAN queue raced admission; no process is terminated')
        identifier = subprocess.check_output(command, text=True).strip()
        atomic_json(root / 'work_dir/_panda_rb/local_owner.json', dict(owner, container=name, recovery=PATCH_ID))
        atomic_json(directory / (attempt + '_launch.json'), dict(container=name, container_id=identifier, at_utc=utcnow()))
        return dict(started=True, container=name, container_id=identifier, receipt=str(receipt_path),
                    completed_training_not_repeated=completed, original_source_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, choices=['s1', 's3', 's5'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.inside:
        if not args.receipt: parser.error('--receipt is required inside the container')
        result = run_inside(args.server, args.root, args.receipt)
    else: result = launch(args.server, args.root, args.dry_run)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__': main()
