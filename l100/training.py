"""Fresh LOCAL-T 50K/100K training with exact resume and endpoint-first Teachers."""
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
import yaml
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader

from fh12.training import BatchStream, atomic_torch, restore_rng, save_resume_bundle, tensor_cpu_tree
from g20.data import build_dataset
from g20.evaluation import FRMetrics, evaluate_model, validation_ergas
from g20.model import build_model, state_hash
from l100.common import (ROOT, CAMPAIGN_ID, apply_runtime_policy, atomic_json, before_deadline,
    camp, check_deadline, immutable_json, object_sha, read, read_json, resolved_path, sha256, source_identity, utcnow)
from l100.losses import routed_student_backward, student_losses, teacher_loss
from l100.plan import (PROFILES, diagnostic_steps, fullstate_steps, grid_steps,
                       validate_config as _validate_config)
from qg40.exposure import record_batch, exposure_report, validate_counts


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def cosine_factor(completed_updates, warmup=100, total=50000):
    if not 0 <= completed_updates <= total or total not in (50000, 100000) or warmup != 100:
        raise ValueError('LOCAL-T scheduler requires its exact fresh horizon and warmup100')
    if completed_updates < warmup:
        return float(completed_updates) / warmup
    return max(0., .5 * (1. + math.cos(math.pi * (completed_updates - warmup) / (total - warmup))))


def aligner_factor(completed_updates, profile):
    if profile not in PROFILES and profile != 'C100':
        raise ValueError('Unregistered LOCAL-T schedule profile')
    if profile != 'ARW':
        return 1.
    if completed_updates < 5000:
        return .1
    return .1 + .9 * (completed_updates - 5000) / 5000 if completed_updates < 10000 else 1.


def make_scheduler(optimizer, profile, warmup=100, total=50000):
    if [group.get('name') for group in optimizer.param_groups] != ['U', 'A']:
        raise ValueError('Scheduler requires explicit U/A groups')
    aligner_factor(0, profile)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [
        lambda t: cosine_factor(t, warmup, total),
        lambda t: cosine_factor(t, warmup, total) * aligner_factor(t, profile)])


def validate_config(cfg):
    return _validate_config(cfg, require_bound=True)


def runtime_context(cfg, root, deadline_arg=None, *, resume=False):
    field = cfg['l100']
    window = read_json(resolved_path(field['window_path'], root))
    if window.get('campaign_id') != CAMPAIGN_ID:
        raise ValueError('LOCAL-T requires its own immutable campaign window')
    start = dt.datetime.fromisoformat(window['t0_utc'].replace('Z', '+00:00'))
    end = dt.datetime.fromisoformat(window['deadline_utc'].replace('Z', '+00:00'))
    if start.tzinfo is None or end.tzinfo is None or end - start != dt.timedelta(hours=20):
        raise ValueError('LOCAL-T must use the common t0+20h deadline')
    if deadline_arg and deadline_arg != window['deadline_utc']:
        raise ValueError('LOCAL-T deadline cannot be overridden or restarted')
    now = dt.datetime.now(dt.timezone.utc)
    if now < start:
        raise ValueError('LOCAL-T common window has not started')
    if not resume and now >= start + dt.timedelta(hours=16):
        receipt = read(camp(root, field['server_id']) / 'admissions' / (field['block_id'] + '.json'))
        from l100.policy import _validate_admitted, CampaignWindow
        from l100.plan import block_for
        # Policy owns the whole-block reservation; a new case is not a new clock.
        _validate_admitted(receipt, block_for(field['block_id']), CampaignWindow.from_dict(window))
        if field['run_id'] not in receipt.get('run_ids', []):
            raise TimeoutError('No new LOCAL-T block after 16h')
    return window, (start + dt.timedelta(hours=18)).isoformat()


def _clone_check(student, teacher, dataset, device):
    if set(map(id, student.parameters())) & set(map(id, teacher.parameters())):
        raise ValueError('Student and Teacher parameters cannot be shared')
    if state_hash(student.aligner.state_dict()) != state_hash(teacher.aligner.state_dict()):
        raise ValueError('Student A is not the exact independent local Teacher clone')
    saved = rng_state()
    modes = [(module, module.training) for net in (student, teacher) for module in net.modules()]
    try:
        student.eval(); teacher.eval()
        row = dataset.base(0)
        pan, ms, lp = [row[i].unsqueeze(0).to(device) for i in (4, 2, 3)]
        with torch.no_grad():
            c_s, c_t = student(pan, ms, lp)['delta'], teacher(pan, ms, lp)['delta']
        if not bool(torch.isfinite(c_s).all()) or not torch.equal(c_s, c_t):
            raise ValueError('Initial local Teacher/Student native correction differs')
        return dict(passed=True, independent_parameters=True, c_max_abs_error=0.,
                    aligner_state_sha256=state_hash(student.aligner.state_dict()))
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(saved)


def diagnostic_ids(base_count):
    """Common train-only P1 IDs; local generators never consume training RNG."""
    from qg40.calibration import select_calibration_indices
    calibration = select_calibration_indices(base_count)
    selected = np.random.default_rng(1234).choice(base_count, 128, replace=False).tolist()
    gradient = selected[:24]
    overlap = sorted(set(selected).intersection(calibration.tolist()))
    payload = dict(schema='L100_DIAGNOSTIC_IDS_v1', base_count=base_count, seed=1234,
        train128=selected, gradient24=gradient, calibration3072_sha256=object_sha(calibration.tolist()),
        train128_calibration_overlap=overlap, gradient24_calibration_overlap=sorted(set(gradient).intersection(overlap)),
        gradient_subset_of_train128=True, scope='original train base IDs only; ROT4 unchanged')
    return dict(payload, ids_sha256=object_sha(payload))


def recover_exact_endpoint(root, wd):
    """Controller-facing exact-endpoint crash reconciliation without training."""
    wd = Path(wd)
    cfg = yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text())
    return recover_final_status(wd, cfg, root)


def recover_final_status(wd, cfg, root=ROOT):
    """CPU-only reconciliation of an immutable exact endpoint, even after 18h."""
    root, wd = Path(root).resolve(), Path(wd)
    case = validate_config(cfg)
    previous = read(wd / 'meta/training_status.json')
    if previous.get('training_complete') is True:
        if previous.get('actual_updates') != case.updates:
            raise ValueError('Completed status is not the registered exact endpoint')
        return False
    folder = wd / 'candidates' / str(case.updates)
    if not folder.exists():
        return False
    if wd.resolve() != resolved_path(cfg['work_dir'], root).resolve() or wd.name != case.run_id:
        raise ValueError('Endpoint recovery run/config path mismatch')
    if yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text()) != cfg:
        raise ValueError('Endpoint recovery resolved config changed')
    apply_runtime_policy(root)
    start = read_json(wd / 'meta/training_start_manifest.json')
    data = read_json(resolved_path(cfg['l100']['dataset_manifest'], root))
    expected = dict(config_sha256=object_sha(cfg), data_sha256=object_sha(data),
                    source_identity=source_identity(root), reference_sha256=start.get('reference_sha256'))
    from l100.plan import valid_sha
    if (start.get('run_id') != case.run_id or start.get('campaign_id') != CAMPAIGN_ID
            or start.get('horizon_updates') != case.updates
            or any(start.get(k) != v for k, v in expected.items())
            or (case.role == 'T' and expected['reference_sha256'] is not None)
            or (case.role == 'S' and not valid_sha(expected['reference_sha256']))):
        raise ValueError('Recovery start/source/data/reference identity differs')
    identity = read_json(folder / 'identity.json')
    if (identity.get('update') != case.updates or identity.get('full_state') is not True
            or identity.get('role') != case.role or identity.get('sensor') != 'GF2'
            or identity.get('input_layout') != case.input_layout or identity.get('num_bands') != 4
            or identity.get('model_sha256') != sha256(folder / 'model.safetensors')
            or identity.get('training_state_sha256') != sha256(folder / 'training_state.pt')
            or any(identity.get(k) != v for k, v in expected.items())):
        raise ValueError('Recovery endpoint/checksum/identity differs')
    state = torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=False)
    if (state.get('full_state') is not True or state.get('update') != case.updates
            or state.get('precision') != 'fp32' or state.get('scheduler', {}).get('last_epoch') != case.updates
            or state.get('model_sha256') != identity['model_sha256']
            or state_hash(state['model_state']) != identity.get('state_hash')
            or state_hash(load_file(str(folder / 'model.safetensors'), device='cpu')) != identity.get('state_hash')
            or any(state.get(k) != v for k, v in expected.items())):
        raise ValueError('Recovery endpoint full-state/model/scheduler differs')
    stream = BatchStream(data['splits']['train']['count'], cfg['batch_size'], case.seed)
    if state_hash({'order': stream.order, 'rotations': stream.rotations}) != start.get('sampler_hash'):
        raise ValueError('Recovery initial data/view prefix differs')
    stream.load_state_dict(state['sampler'])
    if stream.epoch * stream.count + stream.cursor != case.updates:
        raise ValueError('Recovery sampler update differs')
    counts = validate_counts(state.get('exposure_counts'), len(state['exposure_counts']), case.updates, cfg['batch_size'])
    if len(counts) != data['splits']['train']['count']:
        raise ValueError('Recovery exposure population differs')
    grid = read(wd / 'official/raw_grid.json')
    if grid and (any(grid.get(k) != v for k, v in expected.items()) or grid.get('run_id') != case.run_id
                 or grid.get('horizon_updates') != case.updates or grid.get('campaign_id') != CAMPAIGN_ID):
        raise ValueError('Recovery evaluation grid identity differs')
    steps = [r['update'] for r in grid.get('records', [])]
    if len(set(steps)) != len(steps) or any(s not in grid_steps(case.updates) for s in steps):
        raise ValueError('Recovery evaluation grid steps differ')
    completed = identity.get('saved_at_utc')
    if completed:
        when = dt.datetime.fromisoformat(completed.replace('Z', '+00:00'))
        if when.tzinfo is None:
            raise ValueError('Endpoint save timestamp must include a timezone')
    result = dict(status='TRAIN_COMPLETE_EVAL_PENDING', training_complete=True, actual_updates=case.updates,
        horizon_updates=case.updates, n_evaluated=len(steps),
        pending_steps=[s for s in grid_steps(case.updates) if s not in steps],
        training_seconds=state['training_seconds'], evaluation_seconds=state['evaluation_seconds'],
        diagnostic_seconds=state.get('diagnostic_seconds', 0.), io_seconds=state.get('io_seconds', 0.),
        peak_memory_bytes=state.get('peak_memory_bytes'), peak_memory_scope='training+scheduled_eval',
        sample_exposure=exposure_report(counts, cfg['batch_size'], stream.epoch, stream.cursor),
        recovered_from_exact_final=True, training_completed_at_utc=completed,
        recovery_candidate_sha256=identity['model_sha256'], updated_at_utc=utcnow())
    atomic_json(wd / 'meta/training_status.json', result)
    return result


def train_run(config_path, device='cuda', resume=False, deadline_arg=None, root=ROOT):
    from l100.references import load_reference
    root = Path(root).resolve()
    cfg = yaml.safe_load(Path(config_path).read_text())
    case = validate_config(cfg)
    policy = apply_runtime_policy(root)
    if cfg['l100']['runtime_policy_sha256'] != policy['sha256']:
        raise ValueError('Bound numerical runtime-policy SHA differs')
    from l100.controller import authorize_train
    authorize_train(root, case.server_id, Path(config_path))
    window, deadline = runtime_context(cfg, root, deadline_arg, resume=resume)
    check_deadline(deadline)
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    random.seed(case.seed); np.random.seed(case.seed); torch.manual_seed(case.seed)
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(case.seed)
    field, total = cfg['l100'], case.updates
    grid, fullsteps, diagnostics = grid_steps(total), fullstate_steps(case.role, total), diagnostic_steps(case.role, total)
    wd = resolved_path(cfg['work_dir'], root)
    release = source_identity(root)
    data = read_json(resolved_path(field['dataset_manifest'], root))
    if (data.get('schema') != 'G20_DATA_v1' or data.get('sensor') != 'GF2'
            or data.get('num_bands') != 4 or data.get('max_pixel') != 1023):
        raise ValueError('LOCAL-T requires the verified unchanged GF2/C4 data')
    splits = ('train', 'val') if case.role == 'T' else ('train', 'val', 'rr', 'fr')
    datasets = {split: build_dataset(data, split, root=root) for split in splits}
    if len(datasets['train']) != datasets['train'].base_count:
        raise ValueError('Training requires original base IDs and explicit ROT4 draws')
    diagnostic_population = diagnostic_ids(datasets['train'].base_count)
    teacher = reference = q = None
    if case.role == 'S':
        teacher, reference, q = load_reference(case.reference_id, case.server_id, root, str(dev),
            manifest_path=resolved_path(field['reference_manifest'], root) if field['reference_manifest'] else None,
            dataset_manifest=data, deadline=deadline)
        for key, refkey in (('teacher_sha256', 'teacher_checkpoint_sha256'), ('tau_R', 'tau_R'),
                            ('q_ref', 'q_ref'), ('q_cache_sha256', 'q_cache_sha256')):
            if field[key] != reference[refkey]:
                raise ValueError('Resolved local Student reference binding differs: ' + key)
        for key, refkey in (('teacher_checkpoint', 'teacher_checkpoint'), ('q_cache', 'q_cache_path')):
            if resolved_path(field[key], root).resolve() != resolved_path(reference[refkey], root).resolve():
                raise ValueError('Resolved local Student artifact path differs: ' + key)
        if reference['teacher_run_id'] != case.teacher_run_id or reference['teacher_update'] != case.teacher_updates:
            raise ValueError('Student local Teacher identity/horizon differs')
        q = torch.as_tensor(q, device=dev, dtype=torch.float32)
        if tuple(q.shape) != (len(datasets['train']), 4) or not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
            raise ValueError('Invalid full-train four-view q cache')
    model, init = build_model(case.input_layout, case.width, case.depth, case.seed, role=case.role,
        teacher_aligner_state=teacher.aligner.state_dict() if teacher is not None else None, num_bands=4)
    model.to(dev).train()
    if teacher is not None:
        init['clone_P0'] = _clone_check(model, teacher, datasets['train'], dev)
    teacher_hash = state_hash(teacher.state_dict()) if teacher is not None else None
    identity_context = dict(config_sha256=object_sha(cfg), data_sha256=object_sha(data),
        source_identity=release, reference_sha256=object_sha(reference) if reference is not None else None)
    optimizer = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=cfg['learning_rate'], name='U'),
        dict(params=model.aligner.parameters(), lr=field['aligner_lr'], name='A')],
        betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    scheduler = make_scheduler(optimizer, case.profile, cfg['num_warmup'], total)
    stream = BatchStream(len(datasets['train']), cfg['batch_size'], case.seed)
    corruption = torch.Generator(device='cpu').manual_seed(field['corruption_seed'])
    counts = torch.zeros((len(datasets['train']), 4), dtype=torch.int64)
    update, train_seconds, eval_seconds, io_seconds = 0, 0., 0., 0.
    previous_peak = 0
    records = read(wd / 'official/raw_grid.json').get('records', [])
    val_records = read(wd / 'official/validation_grid.json').get('records', [])
    if resume:
        state_path = wd / 'last/training_state.pt'
        last_identity = read_json(wd / 'last/identity.json')
        if last_identity.get('training_state_sha256') != sha256(state_path):
            raise ValueError('Full-state resume checksum differs')
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if (state.get('full_state') is not True or state.get('precision') != 'fp32'
                or any(state.get(k) != v or last_identity.get(k) != v for k, v in identity_context.items())
                or state_hash(state['model_state']) != last_identity.get('state_hash')):
            raise ValueError('Resume model/source/data/reference identity mismatch')
        if yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text()) != cfg:
            raise ValueError('Resume configuration changed')
        start_manifest = read_json(wd / 'meta/training_start_manifest.json')
        if (start_manifest.get('run_id') != case.run_id or start_manifest.get('horizon_updates') != total
                or start_manifest.get('campaign_id') != CAMPAIGN_ID
                or any(start_manifest.get(k) != v for k, v in identity_context.items())):
            raise ValueError('Resume training-start provenance differs')
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        stream.load_state_dict(state['sampler']); corruption.set_state(state['corruption_rng'])
        update = int(state['update'])
        if not 0 <= update <= total or scheduler.last_epoch != update or stream.epoch * stream.count + stream.cursor != update:
            raise ValueError('Resume horizon/scheduler/sampler mismatch')
        counts = validate_counts(state.get('exposure_counts'), len(datasets['train']), update, cfg['batch_size'])
        train_seconds, eval_seconds, io_seconds = state['training_seconds'], state['evaluation_seconds'], state['io_seconds']
        previous_peak = state.get('peak_memory_bytes') or 0
        for record in records + val_records:
            if record['update'] > update or record['update'] not in grid:
                raise ValueError('Saved evaluation grid is ahead of the resumed optimizer')
            if any(record.get('checkpoint_identity', {}).get(k) != v for k, v in identity_context.items()):
                raise ValueError('Saved evaluation record provenance differs')
        for name in ('raw_grid.json', 'validation_grid.json'):
            stored = read(wd / 'official' / name)
            if stored and (stored.get('run_id') != case.run_id or stored.get('horizon_updates') != total
                           or any(stored.get(k) != v for k, v in identity_context.items())):
                raise ValueError('Saved evaluation grid provenance differs')
        restore_rng(state['rng'])
    elif records or val_records or (wd / 'meta/training_start_manifest.json').exists() or (wd / 'last').exists():
        raise ValueError('Run already exists; exact resume required')
    else:
        (wd / 'meta').mkdir(parents=True, exist_ok=True)
        (wd / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        atomic_json(wd / 'init_manifest.json', dict(init, **identity_context))
        atomic_json(wd / 'meta/training_start_manifest.json', dict(campaign_id=CAMPAIGN_ID,
            run_id=case.run_id, horizon_updates=total, **identity_context, window=window, device=str(dev),
            precision='fp32', runtime_policy=policy, started_at_utc=utcnow(),
            sampler_hash=state_hash({'order': stream.order, 'rotations': stream.rotations}),
            rng_roles=dict(data_order=case.seed + 300000, augmentation=case.seed + 400000,
                           corruption=field['corruption_seed'], workers=case.seed + 500000)))
    immutable_json(wd / 'diagnostics/base_ids.json', diagnostic_population)
    pause = [False]
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in handlers:
        signal.signal(sig, lambda *_: pause.__setitem__(0, True))
    engine = None

    def peak():
        return max(previous_peak, torch.cuda.max_memory_allocated(dev)) if dev.type == 'cuda' else None

    def full_state():
        return dict(full_state=True, update=update, model_state=tensor_cpu_tree(model.state_dict()),
            optimizer=tensor_cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
            scaler={}, precision='fp32', rng=rng_state(), sampler=stream.state_dict(),
            corruption_rng=corruption.get_state(), exposure_counts=counts.clone(), **identity_context,
            training_seconds=train_seconds, evaluation_seconds=eval_seconds, io_seconds=io_seconds,
            diagnostic_seconds=0., peak_memory_bytes=peak())

    def save_resume():
        nonlocal io_seconds
        started = time.monotonic()
        save_resume_bundle(wd, full_state(), dict(update=update, full_state=True,
            state_hash=state_hash(model.state_dict()), **identity_context))
        io_seconds += time.monotonic() - started

    def save_candidate(folder, include_state):
        folder.parent.mkdir(parents=True, exist_ok=True)
        digest = state_hash(model.state_dict())
        expected = dict(update=update, role=case.role, sensor='GF2', num_bands=4, input_layout=case.input_layout,
                        full_state=bool(include_state), state_hash=digest, **identity_context)
        if folder.exists():
            previous = read_json(folder / 'identity.json')
            if (any(previous.get(k) != v for k, v in expected.items())
                    or previous.get('model_sha256') != sha256(folder / 'model.safetensors')
                    or previous.get('training_state_sha256') != sha256(folder / 'training_state.pt')):
                raise ValueError('Immutable candidate differs; refusing overwrite')
            return previous
        pending = Path(tempfile.mkdtemp(prefix=f'.{update}-', dir=folder.parent))
        try:
            save_file({k: t.detach().cpu().contiguous() for k, t in model.state_dict().items()}, str(pending / 'model.safetensors'))
            result = dict(expected, model_sha256=sha256(pending / 'model.safetensors'), saved_at_utc=utcnow())
            state = full_state() if include_state else dict(full_state=False, update=update)
            state['model_sha256'] = result['model_sha256']
            atomic_torch(pending / 'training_state.pt', state)
            result['training_state_sha256'] = sha256(pending / 'training_state.pt')
            atomic_json(pending / 'identity.json', result)
            os.replace(pending, folder)
            return result
        finally:
            if pending.exists():
                shutil.rmtree(pending)

    def write_grid():
        records.sort(key=lambda row: row['update'])
        pending = [step for step in grid if step not in {r['update'] for r in records}]
        context = dict(campaign_id=CAMPAIGN_ID, run_id=case.run_id, sensor='GF2', horizon_updates=total,
                       expected_steps=list(grid), **identity_context)
        atomic_json(wd / 'official/raw_grid.json', dict(context, records=records, n_evaluated=len(records), complete=not pending))
        atomic_json(wd / 'official/validation_grid.json', dict(context, records=val_records,
            complete={r['update'] for r in val_records} == set(grid)))
        atomic_json(wd / 'official/evaluation_debt.json', dict(context, pending_steps=pending,
            n_pending=len(pending), official_complete=not pending, teacher_transition_requires_full_grid=False))
        if records:
            best = min(records, key=lambda r: (-r['fr']['hqnr'], -r['rr']['scc'], r['rr']['ergas'], r['update']))
            atomic_json(wd / 'best_hqnr_meta.json', dict(step=best['update'],
                candidate_path=str(wd / 'candidates' / str(best['update'])),
                checkpoint_sha256=best['checkpoint_identity']['model_sha256'],
                hqnr=best['fr']['hqnr'], scc=best['rr']['scc'], ergas=best['rr']['ergas'],
                selector='HQNR_MAX_SCC_MAX_ERGAS_MIN_STEP_MIN', test_aware=True, partial_grid=len(records) < 50))

    def status(name):
        previous = read(wd / 'meta/training_status.json')
        completed = previous.get('training_completed_at_utc')
        if update == total and not completed:
            completed = read(wd / 'candidates' / str(total) / 'identity.json').get('saved_at_utc')
        atomic_json(wd / 'meta/training_status.json', dict(status=name, actual_updates=update,
            horizon_updates=total, training_complete=update == total, n_evaluated=len(records),
            pending_steps=[s for s in grid if s not in {r['update'] for r in records}],
            training_seconds=train_seconds, evaluation_seconds=eval_seconds, diagnostic_seconds=0., io_seconds=io_seconds,
            peak_memory_bytes=peak(), peak_memory_scope='training+scheduled_eval',
            sample_exposure=exposure_report(counts, cfg['batch_size'], stream.epoch, stream.cursor),
            deadline_utc=deadline, training_completed_at_utc=completed, updated_at_utc=utcnow()))

    def boundary():
        nonlocal engine, eval_seconds
        save_resume()
        if update in fullsteps:
            save_candidate(wd / 'restart_fullstates' / str(update), True)
        if update in diagnostics:
            atomic_json(wd / 'diagnostics' / f'required_{update}.json', dict(update=update,
                status='P1_UNAVAILABLE', priority='P1', base_training_blocked=False,
                reason='Extended estimator/gradient/border analysis deferred; immutable diagnostic checkpoint retained',
                checkpoint=str(wd / 'restart_fullstates' / str(update)), **identity_context))
        if update not in grid:
            return
        candidate = save_candidate(wd / 'candidates' / str(update), update in fullsteps)
        existing = records if case.role == 'S' else val_records
        if any(row['update'] == update for row in existing):
            return
        check_deadline(deadline)
        saved = rng_state()
        started = time.monotonic()
        try:
            if case.role == 'T':
                val = validation_ergas(model, datasets['val'], dev, deadline)
                val_records.append(dict(update=update, checkpoint_identity=candidate, val_ergas=val))
                print(f'[L100 Teacher {update}/{total}] HQNR=NOT_EVALUATED SCC=NOT_EVALUATED ERGAS=NOT_EVALUATED valERGAS={val:.8f}', flush=True)
            else:
                engine = engine or FRMetrics(datasets['fr'])
                result = evaluate_model(model, datasets, dev, engine, deadline, include_q=True, with_val=True)
                record = dict(update=update, checkpoint_identity=candidate, **result)
                records.append(record)
                val_records.append(dict(update=update, checkpoint_identity=candidate, val_ergas=result['val_ergas']))
                atomic_json(wd / 'official' / f'candidate_{update}.json', record)
                print(f'[L100 Student {update}/{total}] HQNR={result["fr"]["hqnr"]:.8f} SCC={result["rr"]["scc"]:.8f} ERGAS={result["rr"]["ergas"]:.8f} valERGAS={result["val_ergas"]:.8f}', flush=True)
        finally:
            restore_rng(saved)
            eval_seconds += time.monotonic() - started
        if teacher is not None and state_hash(teacher.state_dict()) != teacher_hash:
            raise ValueError('Frozen local Teacher mutated')
        write_grid(); save_resume()

    try:
        status('RUNNING'); write_grid(); boundary()
        while update < total:
            if pause[0] or not before_deadline(deadline):
                save_resume(); status('PAUSED_SIGNAL' if pause[0] else 'PARTIAL_TIME_LIMIT')
                return 75
            if stream.cursor == stream.count:
                stream.new_epoch()
            loader = DataLoader(datasets['train'], batch_sampler=stream.remaining_batches(),
                num_workers=cfg['num_worker'], pin_memory=dev.type == 'cuda',
                generator=torch.Generator().manual_seed(case.seed + 500000 + stream.epoch))
            for gt, _lms, ms, lp, pan, meta in loader:
                if pause[0] or not before_deadline(deadline):
                    save_resume(); status('PAUSED_SIGNAL' if pause[0] else 'PARTIAL_TIME_LIMIT')
                    return 75
                started = time.monotonic()
                model.train()
                gt, ms, lp, pan = [x.to(dev, non_blocking=True) for x in (gt, ms, lp, pan)]
                optimizer.zero_grad(set_to_none=True)
                out = model(pan, ms, lp)
                if not bool(torch.isfinite(out['y']).all()) or not bool(torch.isfinite(out['delta']).all()):
                    raise FloatingPointError('Nonfinite native model output/correction')
                if teacher is None:
                    losses = teacher_loss(model, out, gt, pan, ms, update, corruption, consistency_weight=1e-4)
                    if not bool(torch.isfinite(losses['total'])):
                        raise FloatingPointError('Nonfinite Teacher loss')
                    losses['total'].backward()
                else:
                    with torch.no_grad():
                        teacher_out = teacher(pan, ms, lp)
                    weights = (reference['q_ref'] / (reference['q_ref'] + q[meta[:, 0].to(dev), meta[:, 1].to(dev)])).detach()
                    losses = student_losses(out, teacher_out, gt, reference['tau_R'], weights, profile=case.profile)
                    if not bool(torch.isfinite(losses['L_U']) & torch.isfinite(losses['L_A'])):
                        raise FloatingPointError('Nonfinite Student loss')
                    routed_student_backward(model, losses)
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError('Missing/nonfinite routed gradients')
                optimizer.step(); scheduler.step(); update += 1; stream.cursor += 1
                record_batch(counts, meta)
                train_seconds += time.monotonic() - started
                if update in grid or update in fullsteps:
                    boundary(); status('RUNNING')
                if update % cfg['log_iter'] == 0:
                    print(f'[L100 {case.role}/{case.profile}] update={update}/{total}', flush=True)
                if update == total:
                    break
        save_resume(); write_grid()
        status('OFFICIAL_EVAL_COMPLETE' if len(records) == 50 else 'TRAIN_COMPLETE_EVAL_PENDING')
        return 0
    except TimeoutError:
        save_resume(); write_grid()
        status('PARTIAL_TIME_LIMIT' if update < total else 'TRAIN_COMPLETE_EVAL_PENDING')
        return 75
    except Exception:
        status('BLOCKED_INTEGRITY')
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
