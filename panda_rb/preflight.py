"""Mandatory automatic checks before first B01 admission; no score gates."""
import os
from pathlib import Path
import shutil

from panda_rb.common import ROOT, atomic_json, docker_required, object_sha, read, server_dir, source_identity, utcnow, sha256
from panda_rb.plan import registry, validate_plan, weights_dir, training_runs, run_dir, campaign_dir


def storage_admission(server, root=ROOT):
    """Subtract this cohort's already retained files; recheck even on resume."""
    from panda_rb.training import required_storage_bytes
    estimate = required_storage_bytes(runs=8)
    remaining = 0; retained = 0; seen = set()
    for case in training_runs(server, root):
        allocated = 0
        for path in run_dir(case['run_id'], root).rglob('*'):
            if not path.is_file(): continue
            stat = path.stat(); inode = (stat.st_dev, stat.st_ino)
            if inode in seen: continue
            seen.add(inode); allocated += stat.st_size
        retained += allocated
        remaining += max(0, estimate['estimated_per_run_bytes'] - allocated)
    needed = remaining + 5 * 1024**3
    target = server_dir(server, root); existing = target
    while not existing.exists(): existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < needed:
        raise OSError(f'Insufficient B01 storage: free={free}, conservative remaining={needed}; no checkpoint deletion is authorized')
    return dict(**estimate, estimated_remaining_bytes=needed, already_retained_bytes=retained, available_bytes=free)


def prepare(server, binding_path, root=ROOT, device='cuda'):
    import torch
    from panda_rb.bindings import prepare_binding, validate_binding
    from panda_rb.weights import prepare_weights, load_weights
    from panda_rb.training import training_smoke
    from panda_rb.stress import warp_convention_test
    from panda_rb.evaluation import evaluator_identity
    from panda_rb.reference_probe import reference_probe

    root = Path(root); target = server_dir(server, root)
    docker_required()
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Actual campaign requires a working CUDA Docker runtime')
    if not os.environ.get('PANDA_RB_IMAGE_ID'):
        raise RuntimeError('Pinned Docker image identity was not supplied by the launch wrapper')
    # Explicit baseline FP32 policy, never inherit TF32 flags from another campaign.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    plan = validate_plan(root)
    capacity = storage_admission(server, root)
    evaluator = evaluator_identity(root)  # Fail a missing/wrong Wald mount before GPU caches or training.
    if not Path(binding_path).exists():
        if server != 's1':
            raise FileNotFoundError('Import the common s1 F1 binding/weights package; local F3/F5 substitution is forbidden')
        prepare_binding(binding_path, root=root)
    _cfg, binding, _data, _q = validate_binding(binding_path)
    all_seeds = [seed for block in registry(root)['independent_student_seeds'].values() for seed in block]
    absent = [seed for seed in all_seeds if not (weights_dir(root) / f'seed_{seed}.json').is_file()]
    if absent:
        if server != 's1':
            raise FileNotFoundError('Import the common frozen all-six-seed weights; do not recompute e/ranks on another GPU')
        prepare_weights(binding_path, weights_dir(root), device=device, seeds=all_seeds, root=root)
    maps = {}; e_caches = set(); current_source = source_identity(root)
    for seed in all_seeds:
        arrays, manifest = load_weights(weights_dir(root), seed, binding=binding)
        maps[str(seed)] = manifest['cache_sha256']
        if manifest['cache_identity']['source_identity']['content_sha256'] != current_source['content_sha256']:
            raise ValueError('Common six-seed mapping source differs from this numerical release')
        e_caches.add(manifest['train_e_cache_sha256'])
        if arrays['q'].shape != (9714, 4):
            raise ValueError('Full WV3 train x four-view controls are required')
    if len(e_caches) != 1:
        raise ValueError('All six permutations must derive from the same frozen F1 train-error cache')
    identity = dict(protocol=plan['protocol_identity'], binding_common_sha256=binding['common_sha256'],
                    weight_files=maps, source_identity=source_identity(root),
                    image_id=os.environ.get('PANDA_RB_IMAGE_ID'), evaluator=evaluator,
                    device_name=torch.cuda.get_device_name(device))
    report_path = target / 'preflight.json'; old = read(report_path)
    if old and old.get('identity') != identity:
        raise ValueError('Frozen preflight source/runtime/binding changed; not an authorized technical resume')
    if old.get('passed'):
        reference_probe(binding_path, root=root, device=device)
        return old
    target.mkdir(parents=True, exist_ok=True)
    seed = registry(root)['independent_student_seeds'][server][0]
    smoke = training_smoke(binding_path, weights_dir(root), seed=seed, device=device, root=root)
    if not smoke.get('actual_batch48', smoke.get('actual_batch48_tested', False)):
        # This key is an explicit production assertion, not a CPU pass promoted to CUDA.
        raise ValueError('Real batch48 CUDA parity smoke was not performed')
    probe = reference_probe(binding_path, root=root, device=device)
    result = dict(passed=True, identity=identity, plan=plan, smoke=smoke,
                  storage=capacity, reference_probe_sha256=sha256(campaign_dir(root) / 'common/reference_probe/report.json'),
                  reference_probe_patches=probe['n_reference_patches'], reference_probe_student_repeats=0,
                  warp_convention=warp_convention_test(device), checked_at_utc=utcnow(),
                  no_training_updates_admitted=True, no_score_based_admission=True)
    atomic_json(report_path, result)
    return result
