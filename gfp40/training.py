"""Isolated P40 Student training: parent U+A FT, paired streams and exact resume.

No old campaign trainer is invoked or patched. Native model/metric primitives
are shared, but FT initialization, schedules and provenance are explicit here.
"""
from __future__ import annotations

import datetime as dt
import math
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from fh12.training import atomic_torch, restore_rng, save_resume_bundle, tensor_cpu_tree
from g20.data import build_dataset
from g20.evaluation import FRMetrics, evaluate_model
from g20.model import build_model, state_hash
from gfp40.common import (ROOT, CAMPAIGN_ID, apply_runtime_policy, atomic_json,
    before_deadline, camp, check_deadline, immutable_json, object_sha, read, read_config,
    read_json, resolved_path, sha256, source_identity, utcnow)
from gfp40.diagnostics import (diagnostic_ids, low_active_mass, parameter_snapshot,
    parameter_update_norm, preserve_runtime, rng_state, run_diagnostics)
from gfp40.losses import routed_student_backward, student_losses
from gfp40.plan import diagnostic_steps, fullstate_steps, grid_steps, validate_config
from gfp40.stream import PairedBatchStream
from qg40.exposure import exposure_report, record_batch, validate_counts


def cosine_factor(completed_updates, warmup=100, total=20000):
    if (isinstance(completed_updates, bool) or int(completed_updates) != completed_updates
            or not 0 <= completed_updates <= total or total not in (20000, 60000, 100000) or warmup != 100):
        raise ValueError('P40 requires warmup100 and its exact20K/60K/100K horizon')
    if completed_updates < warmup:
        return float(completed_updates) / warmup
    return max(0., .5 * (1. + math.cos(math.pi * (completed_updates - warmup) / (total - warmup))))


def make_scheduler(optimizer, warmup=100, total=20000):
    if [group.get('name') for group in optimizer.param_groups] != ['U', 'A']:
        raise ValueError('P40 scheduler requires independent explicit U/A groups')
    cosine_factor(0, warmup, total)
    return torch.optim.lr_scheduler.LambdaLR(optimizer,
        [lambda t: cosine_factor(t, warmup, total), lambda t: cosine_factor(t, warmup, total)])


def model_hashes(model):
    return dict(full=state_hash(model.state_dict()), U=state_hash(model.backbone.state_dict()),
                A=state_hash(model.aligner.state_dict()))


def initialize_student(case, teacher, parent_model=None):
    """FT uses the loaded parent itself, never cloning Teacher A over parent A."""
    teacher.requires_grad_(False).eval()
    if case.is_ft:
        if parent_model is None:
            raise ValueError('FT requires the exact verified parent U and A')
        original = model_hashes(parent_model)
        model = parent_model.requires_grad_(True)
        init = dict(mode='PARENT_STUDENT_U_AND_A_WEIGHTS', parent_preserved=True,
                    teacher_aligner_recloned=False, model_seed_used_for_initialization=False,
                    hashes=original)
    else:
        if parent_model is not None or not case.model_seed:
            raise ValueError('Fresh Student must not inherit any parent U')
        model, init = build_model(case.input_layout, case.width, case.depth, case.model_seed,
            role='S', teacher_aligner_state=teacher.aligner.state_dict(), num_bands=4)
        init.update(mode='FRESH_U_TEACHER_A_CLONE', hashes=model_hashes(model))
    if set(map(id, model.parameters())).intersection(map(id, teacher.parameters())):
        raise ValueError('Teacher and Student share parameters')
    if not case.is_ft and state_hash(model.aligner.state_dict()) != state_hash(teacher.aligner.state_dict()):
        raise ValueError('Fresh Student A differs from its independent frozen Teacher clone')
    return model, dict(init, optimizer_reset_U=True, optimizer_reset_A=True, scheduler_reset=True)


def make_optimizer(model, cfg):
    optimizer = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=cfg['learning_rate'], name='U'),
        dict(params=model.aligner.parameters(), lr=cfg['gfp40']['a_peak_lr'], name='A')],
        betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    if optimizer.state:
        raise ValueError('New FT/fresh optimizers must have zero moments/counters')
    return optimizer


def runtime_context(cfg, root=ROOT, deadline_arg=None):
    from gfp40.policy import CampaignWindow
    window = read_json(resolved_path(cfg['gfp40']['window_path'], root))
    local = CampaignWindow.from_dict(window)
    start = local.t0_utc
    if local.server != cfg['gfp40']['server_id']:
        raise ValueError('P40 requires this server\'s own immutable t0+40h window')
    if deadline_arg and deadline_arg != window['deadline_utc']:
        raise ValueError('P40 deadline cannot be overridden/restarted')
    if dt.datetime.now(dt.timezone.utc) < start:
        raise ValueError('P40 window has not started')
    return window, (start + dt.timedelta(hours=37)).isoformat(), window['deadline_utc']


def val_diverged(parent_val, records):
    """The same parent-relative safety rule applies to every FT arm."""
    if parent_val is None:
        return False
    if not math.isfinite(parent_val) or parent_val <= 0:
        raise ValueError('Finite positive parent validation anchor required')
    ordered = sorted(records, key=lambda row: row['update'])
    if len(ordered) < 2:
        return False
    return all(float(row['val_ergas']) >= 1.10 * parent_val for row in ordered[-2:])


def select_validation(records):
    if not records:
        return None
    if not all(math.isfinite(row['val_ergas']) for row in records):
        raise FloatingPointError('Nonfinite validation candidate')
    return min(records, key=lambda row: (row['val_ergas'], row['update']))


def pause_reason(signaled, optimizer_deadline, control_path):
    if read(control_path).get('command') == 'STOP_NOW_SAFE':
        return 'PAUSED_CONTROL'
    if signaled:
        return 'PAUSED_SIGNAL'
    return None if before_deadline(optimizer_deadline) else 'PARTIAL_TIME_LIMIT'


def verify_batch_metadata(stream, meta, profile):
    """Prove consumed loader IDs/views/q-gamma are the registered draws."""
    if meta.shape != (stream.batch_size, 7) or meta.dtype != torch.int64:
        raise ValueError('P40 source/view/gamma metadata shape/dtype differs')
    expected = torch.tensor(stream.batch_at(stream.cursor), dtype=torch.int64)
    actual = meta[:, [0, 1, 5, 6]].cpu()
    if (not torch.equal(actual, expected) or not bool((meta[:, 2:4] == 1).all())
            or not torch.equal(meta[:, 4].cpu(), expected[:, 2] if profile == 'MIX'
                               else torch.full((stream.batch_size,), stream.native_gamma_id, dtype=torch.int64))):
        raise ValueError('P40 source/view/gamma draw or native-input identity differs')
    # Family-local gamma indices are not the pairing key across distributions.
    return actual[:, [0, 1, 3]].tolist()


def verify_paired_initialization(case, hashes, prefix, root=ROOT):
    from gfp40.plan import CASES
    checked = []
    for peer in CASES:
        if (peer.run_id == case.run_id or peer.server != case.server
                or peer.stream_seed != case.stream_seed or peer.parent_id != case.parent_id
                or peer.model_seed != case.model_seed or peer.is_ft != case.is_ft):
            continue
        folder = Path(root) / 'work_dir' / peer.run_id
        previous = read(folder / 'init_manifest.json')
        if not previous:
            continue
        old_prefix = read(folder / 'diagnostics/paired_stream_first256.json')
        if (previous['hashes'] != hashes or old_prefix['sample_view_uniform_sha256']
                != prefix['sample_view_uniform_sha256']):
            raise ValueError('Matched P40 arms differ at t=0 U/A/full tensors or sample/view/gamma prefix')
        checked.append(peer.case_id)
    return checked


def _resolved_profile_assets(cfg, assets, native_train, root, device=None):
    """Only scales and cache change for mixcal; frozen Teacher stays identical."""
    from gfp40.data import GFP40Dataset
    field, native = cfg['gfp40'], assets['reference']
    reference, q, mixed = dict(native), assets['q'], None
    if field['profile'] in ('CTRL', 'MIX'):
        from gfp40.calibration import load_family_reference
        mixed, q = load_family_reference(resolved_path(field['mixed_calibration_manifest'], root),
                                        native_reference=native, dataset=native_train, device=device)
        if (mixed['family'] != field['family']
                or sha256(resolved_path(field['augmentation_manifest'], root))
                != mixed['augmentation_manifest_sha256']):
            raise ValueError('Bound family/calibration/augmentation identity differs')
        for key in ('tau_R', 'q_ref', 'q_cache_sha256', 'q_cache_path', 'q_shape'):
            reference[key] = mixed[key]
        reference['mixed_calibration_sha256'] = object_sha(mixed)
        reference['native_tau_R'], reference['native_q_ref'] = native['tau_R'], native['q_ref']
    augmentation = (resolved_path(field['augmentation_manifest'], root)
                    if field['augmentation_manifest'] else None)
    family = field['family'] if field['profile'] != 'NATIVE0' else 'G025'
    dataset = GFP40Dataset(native_train, augmentation_manifest=augmentation,
                          arm=field['profile'], family=family)
    return dataset, reference, q, mixed


def train_run(config_path, device='cuda', resume=False, deadline_arg=None, root=ROOT):
    from gfp40.assets import load_training_assets
    from gfp40.controller import authorize_train
    root, config_path = Path(root).resolve(), Path(config_path)
    cfg = read_config(config_path)
    case = validate_config(cfg, require_bound=True)
    field, profile, total = cfg['gfp40'], cfg['gfp40']['profile'], case.updates
    policy = apply_runtime_policy(root)
    if field['runtime_policy_sha256'] != policy['sha256']:
        raise ValueError('Bound P40 runtime policy differs')
    authorize_train(root, case.server_id, config_path)
    window, optimizer_deadline, final_deadline = runtime_context(cfg, root, deadline_arg)
    control_path = camp(root, case.server_id) / 'control.json'
    check_deadline(optimizer_deadline)
    dev = torch.device(device)
    if dev.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Registered P40 training requires CUDA; no CPU fallback')
    random.seed(case.seed); np.random.seed(case.seed); torch.manual_seed(case.seed)
    torch.cuda.manual_seed_all(case.seed)
    wd = resolved_path(cfg['work_dir'], root)
    data = read_json(resolved_path(field['dataset_manifest'], root))
    if (data.get('schema') != 'G20_DATA_v1' or data.get('sensor') != 'GF2'
            or data.get('num_bands') != 4 or data.get('max_pixel') != 1023):
        raise ValueError('P40 requires unchanged GF2/C4/DN1023 data')
    datasets = {split: build_dataset(data, split, root=root) for split in ('train', 'val', 'rr', 'fr')}
    assets = load_training_assets(cfg, root=root, device=str(dev), deadline=optimizer_deadline)
    if assets.get('native_online_parity') is not None:
        atomic_json(wd / 'meta/native_reference_online_parity.json', assets['native_online_parity'])
    teacher = assets['teacher'].requires_grad_(False).eval()
    train_dataset, reference, q, mixed = _resolved_profile_assets(cfg, assets, datasets['train'], root, str(dev))
    q = torch.as_tensor(q, dtype=torch.float32, device=dev)
    expected_q_shape = (datasets['train'].base_count, 4) + ((len(train_dataset.gammas),) if mixed else ())
    if tuple(q.shape) != expected_q_shape or not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError('P40 full-training sample/view/gamma q cache differs')
    model, init = initialize_student(case, teacher, assets.get('parent_model'))
    model.to(dev).train()
    teacher_hash = state_hash(teacher.state_dict())
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg['num_warmup'], total)
    stream = PairedBatchStream(datasets['train'].base_count, cfg['batch_size'], case.stream_seed,
                              family=field['family'] if mixed else 'G025')
    prefix = stream.initial_prefix()
    verify_paired_initialization(case, init['hashes'], prefix, root)
    identity_context = dict(config_sha256=object_sha(cfg), data_sha256=object_sha(data),
        source_identity=source_identity(root), reference_sha256=object_sha(reference),
        parent_sha256=object_sha(assets['parent']) if assets.get('parent') else None,
        mixed_calibration_sha256=object_sha(mixed) if mixed else None,
        paired_prefix_sha256=object_sha(prefix))
    ids = diagnostic_ids(datasets['train'].base_count)
    immutable_json(wd / 'meta/training_reference.json', reference)
    immutable_json(wd / 'diagnostics/base_ids.json', ids)
    immutable_json(wd / 'diagnostics/paired_stream_first256.json', prefix)
    records = read(wd / 'official/raw_grid.json').get('records', [])
    val_records = [dict(update=row['update'], checkpoint_identity=row['checkpoint_identity'],
                        val_ergas=row['val_ergas']) for row in records]
    counts = torch.zeros((datasets['train'].base_count, 4), dtype=torch.int64)
    gamma_counts = {name: torch.zeros(len(train_dataset.gammas), dtype=torch.int64)
                    for name in ('drawn', 'effective')}
    update, train_seconds, eval_seconds, io_seconds, diagnostic_seconds = 0, 0., 0., 0., 0.
    previous_peak = 0
    consumed_prefix = []
    if resume:
        last = read_json(wd / 'last/identity.json')
        if last.get('training_state_sha256') != sha256(wd / 'last/training_state.pt'):
            raise ValueError('P40 resume checksum mismatch')
        state = torch.load(wd / 'last/training_state.pt', map_location='cpu', weights_only=False)
        if (not state.get('full_state') or state.get('precision') != 'fp32'
                or any(state.get(k) != v or last.get(k) != v for k, v in identity_context.items())
                or state_hash(state['model_state']) != last.get('state_hash')
                or read_config(wd / 'meta/config.resolved.yaml') != cfg):
            raise ValueError('P40 exact-resume model/config/source/assets differ')
        start = read_json(wd / 'meta/training_start_manifest.json')
        if any(start.get(k) != v for k, v in identity_context.items()):
            raise ValueError('P40 resume start provenance differs')
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        stream.load_state_dict(state['sampler'])
        consumed_prefix = state.get('consumed_prefix', [])
        update = int(state['update'])
        if not 0 <= update <= total or scheduler.last_epoch != update or stream.completed_updates != update:
            raise ValueError('P40 resume optimizer/scheduler/sampler update mismatch')
        if (len(consumed_prefix) != min(update * cfg['batch_size'], 256)
                or consumed_prefix != prefix['sample_view_uniform'][:len(consumed_prefix)]):
            raise ValueError('P40 actual consumed prefix differs on resume')
        counts = validate_counts(state.get('exposure_counts'), datasets['train'].base_count, update, cfg['batch_size'])
        stored_gamma = state.get('gamma_counts', {})
        for name in gamma_counts:
            value = stored_gamma.get(name)
            if (not torch.is_tensor(value) or value.dtype != torch.int64 or value.shape != gamma_counts[name].shape
                    or bool((value < 0).any()) or int(value.sum()) != update * cfg['batch_size']):
                raise ValueError('Resume drawn/effective gamma exposure counts differ')
            gamma_counts[name] = value.clone()
        train_seconds, eval_seconds = state['training_seconds'], state['evaluation_seconds']
        io_seconds, diagnostic_seconds = state['io_seconds'], state['diagnostic_seconds']
        previous_peak = state.get('peak_memory_bytes') or 0
        for collection in (records, val_records):
            if len({r['update'] for r in collection}) != len(collection):
                raise ValueError('Duplicate candidate evaluation on resume')
            for record in collection:
                if record['update'] > update or record['update'] not in grid_steps(total):
                    raise ValueError('Saved candidate grid ahead of resumed optimizer')
                if any(record.get('checkpoint_identity', {}).get(k) != v for k, v in identity_context.items()):
                    raise ValueError('Saved candidate reference/source differs')
        restore_rng(state['rng'])
    elif records or val_records or (wd / 'meta/training_start_manifest.json').exists() or (wd / 'last').exists():
        raise ValueError('Run already started; only exact in-run resume is allowed')
    else:
        immutable_json(wd / 'meta/config.resolved.yaml', cfg)
        immutable_json(wd / 'init_manifest.json', dict(init, **identity_context))
        immutable_json(wd / 'meta/training_start_manifest.json', dict(campaign_id=CAMPAIGN_ID,
            case_id=case.case_id, run_id=case.run_id, horizon_updates=total,
            parent_step=case.parent_step or 0, init_mode='FT' if case.is_ft else 'FRESH',
            window=window, optimizer_deadline_utc=optimizer_deadline, started_at_utc=utcnow(),
            precision='fp32', device=str(dev), runtime_policy=policy,
            optimizer_reset_U=True, optimizer_reset_A=True, paired_stream=prefix,
            coefficients_profile=profile, **identity_context))
    pause = [False]
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in handlers:
        signal.signal(sig, lambda *_: pause.__setitem__(0, True))
    grid, fullsteps, diagnostics = grid_steps(total), fullstate_steps(total), diagnostic_steps(total)
    engine = None
    parent_val = read(wd / 'official/parent_anchor.json').get('val_ergas')

    def peak():
        return max(previous_peak, torch.cuda.max_memory_allocated(dev))

    def full_state():
        return dict(full_state=True, update=update, local_updates=update,
            lifetime_updates=(case.parent_step or 0) + update, model_state=tensor_cpu_tree(model.state_dict()),
            optimizer=tensor_cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
            precision='fp32', scaler={}, rng=rng_state(), sampler=stream.state_dict(),
            exposure_counts=counts.clone(), consumed_prefix=consumed_prefix, training_seconds=train_seconds,
            gamma_counts={k: v.clone() for k, v in gamma_counts.items()},
            evaluation_seconds=eval_seconds, io_seconds=io_seconds, diagnostic_seconds=diagnostic_seconds,
            peak_memory_bytes=peak(), **identity_context)

    def save_resume():
        nonlocal io_seconds
        started = time.monotonic()
        save_resume_bundle(wd, full_state(), dict(update=update, full_state=True,
            state_hash=state_hash(model.state_dict()), **identity_context))
        io_seconds += time.monotonic() - started

    def save_candidate(folder, include_state):
        nonlocal io_seconds
        started = time.monotonic()
        folder.parent.mkdir(parents=True, exist_ok=True)
        expected = dict(update=update, local_step=update, parent_step=case.parent_step or 0,
            lifetime_step=(case.parent_step or 0) + update, role='S', sensor='GF2', num_bands=4,
            input_layout='PLH', full_state=bool(include_state), state_hash=state_hash(model.state_dict()),
            **identity_context)
        if folder.exists():
            value = read_json(folder / 'identity.json')
            if (any(value.get(k) != v for k, v in expected.items())
                    or value.get('model_sha256') != sha256(folder / 'model.safetensors')
                    or value.get('training_state_sha256') != sha256(folder / 'training_state.pt')):
                raise ValueError('Immutable P40 candidate changed')
            return value
        pending = Path(tempfile.mkdtemp(prefix=f'.{update}-', dir=folder.parent))
        try:
            save_file({k: t.detach().cpu().contiguous() for k, t in model.state_dict().items()}, str(pending / 'model.safetensors'))
            value = dict(expected, model_sha256=sha256(pending / 'model.safetensors'), saved_at_utc=utcnow())
            state = full_state() if include_state else dict(full_state=False, update=update)
            state['model_sha256'] = value['model_sha256']
            atomic_torch(pending / 'training_state.pt', state)
            value['training_state_sha256'] = sha256(pending / 'training_state.pt')
            atomic_json(pending / 'identity.json', value)
            os.replace(pending, folder)
            return value
        finally:
            if pending.exists(): shutil.rmtree(pending)
            io_seconds += time.monotonic() - started

    def write_grid():
        records.sort(key=lambda row: row['update']); val_records.sort(key=lambda row: row['update'])
        pending = [step for step in grid if step not in {r['update'] for r in records}]
        context = dict(campaign_id=CAMPAIGN_ID, case_id=case.case_id, run_id=case.run_id,
            sensor='GF2', horizon_updates=total, expected_steps=list(grid), **identity_context)
        atomic_json(wd / 'official/raw_grid.json', dict(context, records=records, n_evaluated=len(records), complete=not pending))
        atomic_json(wd / 'official/validation_grid.json', dict(context, records=val_records, complete=len(val_records) == len(grid)))
        atomic_json(wd / 'official/evaluation_debt.json', dict(context, pending_steps=pending, n_pending=len(pending)))
        selected = select_validation(val_records)
        if selected:
            atomic_json(wd / 'best_val_ergas_meta.json', dict(step=selected['update'],
                candidate_path=str(wd / 'candidates' / str(selected['update'])),
                checkpoint_sha256=selected['checkpoint_identity']['model_sha256'],
                val_ergas=selected['val_ergas'], selector='VAL_ERGAS_MIN_STEP_MIN', partial_grid=bool(pending)))
        if records:
            raw = min(records, key=lambda r: (-r['fr']['hqnr'], -r['rr']['scc'], r['rr']['ergas'], r['update']))
            atomic_json(wd / 'raw_hqnr_aux_meta.json', dict(step=raw['update'],
                checkpoint_sha256=raw['checkpoint_identity']['model_sha256'],
                hqnr=raw['fr']['hqnr'], scc=raw['rr']['scc'], ergas=raw['rr']['ergas'],
                selector='HQNR_MAX_SCC_MAX_ERGAS_MIN_STEP_MIN', auxiliary_only=True,
                test_aware=True, candidate_budget=len(grid), partial_grid=bool(pending)))

    def status(name):
        endpoint = read(wd / 'candidates' / str(total) / 'identity.json')
        atomic_json(wd / 'meta/training_status.json', dict(status=name, actual_updates=update,
            local_updates=update, parent_step=case.parent_step or 0,
            lifetime_updates=(case.parent_step or 0) + update, horizon_updates=total,
            training_complete=update == total, n_evaluated=len(records),
            pending_steps=[s for s in grid if s not in {r['update'] for r in records}],
            training_seconds=train_seconds, evaluation_seconds=eval_seconds,
            diagnostic_seconds=diagnostic_seconds, io_seconds=io_seconds, peak_memory_bytes=peak(),
            sample_exposure=exposure_report(counts, cfg['batch_size'], stream.epoch, stream.cursor),
            gamma_exposure={key: {str(g): int(value[i]) for i, g in enumerate(train_dataset.gammas)}
                            for key, value in gamma_counts.items()},
            optimizer_deadline_utc=optimizer_deadline, deadline_utc=final_deadline,
            training_completed_at_utc=endpoint.get('saved_at_utc'), updated_at_utc=utcnow()))

    def diagnose():
        nonlocal diagnostic_seconds
        path = wd / 'diagnostics' / f'probe_{update}.json'
        if path.exists():
            previous = read_json(path)
            if previous.get('model_state_hash') != state_hash(model.state_dict()):
                raise ValueError('Diagnostic checkpoint changed on resume')
            return
        started = time.monotonic()
        baseline = read(wd / 'diagnostics/probe_0.json').get('c_student')
        value = run_diagnostics(model, teacher, train_dataset, q, reference, profile, update,
            ids, str(dev), parent_delta=baseline, heavy=update in (0, 5000, 20000, 40000, total), deadline=final_deadline)
        value.update(model_state_hash=state_hash(model.state_dict()), **identity_context)
        immutable_json(path, value)
        diagnostic_seconds += time.monotonic() - started
        atomic_json(wd / 'diagnostics/soft_active_mass.json', low_active_mass(
            [read_json(p) for p in (wd / 'diagnostics').glob('probe_*.json')]))

    def boundary():
        nonlocal engine, eval_seconds, parent_val
        save_resume()
        if update in fullsteps and update not in grid:
            save_candidate(wd / 'restart_fullstates' / str(update), True)
        # Publish exact endpoint before potentially deferred/expired diagnostics.
        candidate = save_candidate(wd / 'candidates' / str(update), update in fullsteps) if update in grid else None
        # Separate cosine60K curve checkpoints are never RAW/VAL candidates.
        curve = None
        if total == 60000 and update in (20000, 40000):
            curve = save_candidate(wd / 'diagnostics/curve_checkpoints' / str(update), True)
            curve_path = wd / 'diagnostics' / f'curve_{update}.json'
            if not curve_path.exists():
                started = time.monotonic()
                with preserve_runtime(model, teacher):
                    engine = engine or FRMetrics(datasets['fr'])
                    curve_metrics = evaluate_model(model, datasets, dev, engine, final_deadline,
                                                   include_q=True, with_val=True)
                immutable_json(curve_path, dict(update=update, checkpoint_identity=curve,
                    candidate_eligible=False, optimizer_horizon=60000, **curve_metrics))
                eval_seconds += time.monotonic() - started
        if update in diagnostics:
            diagnose()
        if update == 0 and case.is_ft and parent_val is None:
            started = time.monotonic()
            with preserve_runtime(model, teacher):
                engine = engine or FRMetrics(datasets['fr'])
                anchor = evaluate_model(model, datasets, dev, engine, final_deadline, include_q=True, with_val=True)
            anchor.update(model_state_hash=state_hash(model.state_dict()), parent_step=case.parent_step,
                          candidate_eligible=False, **identity_context)
            immutable_json(wd / 'official/parent_anchor.json', anchor)
            parent_val = anchor['val_ergas']
            eval_seconds += time.monotonic() - started
            save_resume()
        if update not in grid:
            return
        if not any(row['update'] == update for row in records):
            check_deadline(final_deadline)
            started = time.monotonic()
            with preserve_runtime(model, teacher):
                engine = engine or FRMetrics(datasets['fr'])
                result = evaluate_model(model, datasets, dev, engine, final_deadline, include_q=True, with_val=True)
            record = dict(update=update, local_step=update, lifetime_step=(case.parent_step or 0) + update,
                          checkpoint_identity=candidate, **result)
            records.append(record)
            val_records.append(dict(update=update, checkpoint_identity=candidate, val_ergas=result['val_ergas']))
            atomic_json(wd / 'official' / f'candidate_{update}.json', record)
            eval_seconds += time.monotonic() - started
            print(f'[GFP40 {case.case_id} {update}/{total}] HQNR={result["fr"]["hqnr"]:.8f} '
                  f'SCC={result["rr"]["scc"]:.8f} ERGAS={result["rr"]["ergas"]:.8f} '
                  f'valERGAS={result["val_ergas"]:.8f}', flush=True)
        if state_hash(teacher.state_dict()) != teacher_hash:
            raise ValueError('P40 frozen Teacher mutated')
        write_grid(); save_resume()

    try:
        status('RUNNING'); write_grid(); boundary()
        if case.is_ft and update and parent_val is None:
            raise ValueError('FT cannot resume without its immutable native parent anchor')
        if case.is_ft:
            from gfp40.anchor import verify_and_save_parent_anchor
            anchor = read_json(wd / 'official/parent_anchor.json')
            if any(anchor.get(key) != value for key, value in identity_context.items()):
                raise ValueError('Parent anchor source/data/reference identity changed')
            verify_and_save_parent_anchor(case, assets['parent'], anchor,
                wd / 'official/parent_anchor_comparison.json', root)
        while update < total:
            if val_diverged(parent_val, val_records):
                save_resume(); status('ABORTED_VAL_DIVERGENCE'); return 76
            reason = pause_reason(pause[0], optimizer_deadline, control_path)
            if reason:
                save_resume(); status(reason); return 75
            if stream.cursor == stream.count:
                stream.new_epoch()
            loader = DataLoader(train_dataset, batch_sampler=stream.remaining_batches(),
                num_workers=cfg['num_worker'], pin_memory=True,
                generator=torch.Generator().manual_seed(stream.worker_seed))
            for gt, _lms, ms, lp, pan, meta in loader:
                reason = pause_reason(pause[0], optimizer_deadline, control_path)
                if reason:
                    save_resume(); status(reason); return 75
                started = time.monotonic()
                actual_triples = verify_batch_metadata(stream, meta, profile)
                model.train()
                gt, ms, lp, pan = [x.to(dev, non_blocking=True) for x in (gt, ms, lp, pan)]
                meta_dev = meta.to(dev)
                optimizer.zero_grad(set_to_none=True)
                out = model(pan, ms, lp)
                with torch.no_grad():
                    target = teacher(pan, ms, lp)
                if not all(bool(torch.isfinite(x).all()) for x in (out['y'], out['delta'], target['y'], target['delta'])):
                    raise FloatingPointError('Nonfinite Student/Teacher output/correction')
                q_values = q[meta_dev[:, 0], meta_dev[:, 1]]
                if q.ndim == 3:
                    q_values = q_values.gather(1, meta_dev[:, 4:5]).squeeze(1)
                weights = reference['q_ref'] / (reference['q_ref'] + q_values)
                losses = student_losses(out, target, gt, reference['tau_R'], weights,
                                        profile, completed_updates=update)
                if not bool(torch.isfinite(losses['L_U']) & torch.isfinite(losses['L_A'])):
                    raise FloatingPointError('Nonfinite P40 fitting loss')
                routed_student_backward(model, losses)
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError('Missing/nonfinite routed gradients')
                probe_update = (update + 1 in diagnostics or (update + 1) % cfg['log_iter'] == 0)
                before = parameter_snapshot(model) if probe_update else None
                # The forward/backward can itself cross 37h; never optimizer.step afterwards.
                reason = pause_reason(pause[0], optimizer_deadline, control_path)
                if reason:
                    save_resume(); status(reason); return 75
                optimizer.step(); scheduler.step(); update += 1; stream.advance()
                consumed_prefix.extend(actual_triples[:max(0, 256 - len(consumed_prefix))])
                if len(consumed_prefix) == 256:
                    if consumed_prefix != prefix['sample_view_uniform']:
                        raise ValueError('Actual first256 sample/view/gamma pairing differs')
                    actual_path = wd / 'diagnostics/consumed_first256.json'
                    if not actual_path.exists():
                        immutable_json(actual_path, dict(expected_prefix_sha256=object_sha(prefix),
                            sample_view_uniform=consumed_prefix, count=256, actual_loader_verified=True))
                record_batch(counts, meta[:, :4])
                for name, column in (('effective', 4), ('drawn', 5)):
                    gamma_counts[name] += torch.bincount(meta[:, column], minlength=len(train_dataset.gammas))
                train_seconds += time.monotonic() - started
                if before is not None:
                    atomic_json(wd / 'diagnostics' / f'update_norm_{update}.json', dict(local_step=update,
                        update_norm=parameter_update_norm(model, before), coefficients=losses['coefficients'],
                        learning_rates_next={g['name']: g['lr'] for g in optimizer.param_groups},
                        routing='U<-hard+soft+qedge; A<-qhard', teacher_frozen=True))
                if update in grid or update in fullsteps or update in diagnostics:
                    boundary(); status('RUNNING')
                    if val_diverged(parent_val, val_records):
                        save_resume(); status('ABORTED_VAL_DIVERGENCE'); return 76
                if update % cfg['log_iter'] == 0:
                    print(f'[GFP40 {case.case_id}/{profile}] local={update}/{total} '
                          f'lifetime={(case.parent_step or 0) + update}', flush=True)
                if update == total:
                    break
        save_resume(); write_grid()
        status('OFFICIAL_EVAL_COMPLETE' if len(records) == len(grid) else 'TRAIN_COMPLETE_EVAL_PENDING')
        return 0
    except TimeoutError:
        save_resume(); write_grid()
        status('PARTIAL_TIME_LIMIT' if update < total else 'TRAIN_COMPLETE_EVAL_PENDING')
        return 75
    except FloatingPointError:
        status('FAILED_NUMERICAL')
        raise
    except Exception:
        status('BLOCKED_INTEGRITY')
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def recover_exact_endpoint(root, wd):
    """CPU-only reconciliation after endpoint publication, never new training."""
    wd, root = Path(wd), Path(root)
    cfg = read_config(wd / 'meta/config.resolved.yaml')
    case = validate_config(cfg, require_bound=True)
    if read(wd / 'meta/training_status.json').get('training_complete'):
        return False
    folder = wd / 'candidates' / str(case.updates)
    if not folder.is_dir():
        return False
    identity = read_json(folder / 'identity.json')
    start = read_json(wd / 'meta/training_start_manifest.json')
    if (identity.get('update') != case.updates or not identity.get('full_state')
            or identity.get('config_sha256') != object_sha(cfg)
            or identity.get('source_identity') != source_identity(root)
            or identity.get('model_sha256') != sha256(folder / 'model.safetensors')
            or identity.get('training_state_sha256') != sha256(folder / 'training_state.pt')):
        raise ValueError('P40 endpoint recovery identity differs')
    state = torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=False)
    for key in ('config_sha256', 'source_identity', 'data_sha256', 'reference_sha256',
                'parent_sha256', 'mixed_calibration_sha256', 'paired_prefix_sha256'):
        if state.get(key) != identity.get(key) or start.get(key) != identity.get(key):
            raise ValueError('P40 endpoint provenance changed')
    if (not state.get('full_state') or state.get('update') != case.updates
            or state.get('scheduler', {}).get('last_epoch') != case.updates
            or state_hash(state['model_state']) != identity.get('state_hash')
            or state_hash(load_file(str(folder / 'model.safetensors'))) != identity.get('state_hash')):
        raise ValueError('P40 endpoint fullstate/model/scheduler differs')
    data = read_json(resolved_path(cfg['gfp40']['dataset_manifest'], root))
    if object_sha(data) != identity['data_sha256']:
        raise ValueError('P40 endpoint dataset identity changed')
    stream = PairedBatchStream(data['splits']['train']['count'], cfg['batch_size'], case.stream_seed,
                              family=cfg['gfp40']['family'] if cfg['gfp40']['profile'] != 'NATIVE0' else 'G025')
    stream.load_state_dict(state['sampler'])
    if stream.completed_updates != case.updates:
        raise ValueError('P40 endpoint sampler differs')
    validate_counts(state['exposure_counts'], data['splits']['train']['count'], case.updates, cfg['batch_size'])
    records = read(wd / 'official/raw_grid.json').get('records', [])
    status = dict(status='TRAIN_COMPLETE_EVAL_PENDING', training_complete=True,
        actual_updates=case.updates, local_updates=case.updates, parent_step=case.parent_step or 0,
        lifetime_updates=(case.parent_step or 0) + case.updates, horizon_updates=case.updates,
        pending_steps=[s for s in grid_steps(case.updates) if s not in {r['update'] for r in records}],
        n_evaluated=len(records), recovered_from_exact_final=True,
        training_completed_at_utc=identity.get('saved_at_utc'), updated_at_utc=utcnow(),
        **{k: state[k] for k in ('training_seconds', 'evaluation_seconds', 'io_seconds', 'diagnostic_seconds')})
    atomic_json(wd / 'meta/training_status.json', status)
    return status
