"""TA2-only update engine and durable checkpoint/evaluation transactions.

The author registry is immutable. Evaluation failures resume evaluation, not a
new seed or a new training run. No old campaign is imported or stopped here.
"""
import contextlib
import copy
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

from .common import atomic_json, append_json, digest, file_sha, immutable_json, load_json, now
from .data import NativeStream, transform_shift
from .evaluation import EvaluationLedger, evaluate_checkpoint, hqnr_grid
from .losses import compute_objective
from .model import FIXED_POLICIES, TeacherModel, state_hash


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def save_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('Cannot resume CUDA RNG on CPU')
        torch.cuda.set_rng_state_all(value['cuda'])


@contextlib.contextmanager
def preserve_runtime(model):
    state = rng_state(); modes = {m: m.training for m in model.modules()}
    gradients = {p: None if p.grad is None else p.grad.detach().clone() for p in model.parameters()}
    try:
        yield
    finally:
        for module, mode in modes.items():
            module.training = mode
        for param, grad in gradients.items():
            param.grad = grad
        restore_rng(state)


def configure_runtime(seed, device):
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: no silent CPU fallback')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # CUDA grid_sample backward has no deterministic implementation in the
    # pinned runtime. Record this limitation, do not promise bitwise CUDA replay.
    torch.use_deterministic_algorithms(False)


def cosine_factor(completed, total, warmup=100):
    """Legacy FH12/diffusers multiplier, BEFORE update completed+1."""
    if completed < warmup:
        return float(completed) / max(1, warmup)
    return max(0., .5 * (1 + math.cos(math.pi * (completed-warmup) / max(1, total-warmup))))


def objective_inputs(model, bundle, cfg, ids, rots, device):
    inputs = {}
    if model.policy in FIXED_POLICIES:
        shift = bundle.case_shifts(cfg, 'train')[ids].to(device)
    elif model.policy == 'GLOBAL_LEARNED':
        shift = model.global_shift[None].expand(len(ids), -1)
    else:
        shift = None
    if shift is not None:
        inputs['fixed_shift'] = shift if rots is None else transform_shift(shift, rots)
    if cfg['struct_descriptor'] == 'PSEUDO_HUBER':
        pseudo = bundle.case_shifts('S07', 'train')[ids].to(device)
        inputs['pseudo_shift'] = pseudo if rots is None else transform_shift(pseudo, rots)
    if cfg['band_weights'] == 'CAL_FROZEN':
        inputs['band_weights'] = bundle.band_weights()['weights']
    return inputs


def predict_native(model, bundle, cfg, pan, ms, lms, split, index):
    # B06 GT cache is train-only; case_shifts explicitly switches to LMS here.
    shift = (bundle.case_shifts(cfg, split)[index:index+1].to(pan)
             if model.policy in FIXED_POLICIES else None)
    return model(pan, ms, lms, shift=shift)


class UpdateEngine:
    """A single full batch48 update, with exact sample-mean micro accumulation."""
    def __init__(self, model, bundle, cfg, device='cuda'):
        self.model, self.bundle, self.cfg, self.device = model, bundle, dict(cfg), device
        self.completed = 0
        self.stream = NativeStream(len(bundle.datasets['train']), int(cfg['seed']) + 300000)
        self.epsilon_rng = torch.Generator().manual_seed(int(cfg['seed']) + 500000)
        self.geometry_rng = torch.Generator().manual_seed(int(cfg['seed']) + 600000)
        groups = [dict(params=list(model.u_parameters()), lr=cfg['lr_u'], role='U', base_lr=cfg['lr_u'])]
        a_params = list(model.a_parameters())
        if a_params:
            groups.append(dict(params=a_params, lr=cfg['lr_a'], role='A', base_lr=cfg['lr_a']))
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(cfg['betas']), eps=cfg['eps'],
                                          weight_decay=cfg['weight_decay'])
        self.native_stream_sha256 = digest(self.stream.state_dict())

    def state_dict(self):
        return cpu_tree(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            completed_step=self.completed, stream=self.stream.state_dict(), epsilon_rng=self.epsilon_rng.get_state(),
            geometry_rng=self.geometry_rng.get_state(), rng=rng_state(), config_sha256=digest(self.cfg)))

    def load_state_dict(self, state):
        if state['config_sha256'] != digest(self.cfg):
            raise ValueError('Resume cannot change registered config/microbatch')
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.stream.load_state_dict(state['stream']); self.epsilon_rng.set_state(state['epsilon_rng'])
        self.geometry_rng.set_state(state['geometry_rng']); self.completed = int(state['completed_step'])
        if not 0 <= self.completed <= self.cfg['total_updates']:
            raise ValueError('Invalid completed update counter')
        restore_rng(state['rng'])

    def update(self):
        if self.completed >= self.cfg['total_updates']:
            raise ValueError('Cannot exceed preregistered optimizer budget')
        started = time.monotonic(); cfg = self.cfg; self.model.train()
        ids, rots = self.stream.next(cfg['batch_size'])
        # Always draw complete logical streams, including loss-disabled cases.
        draws = torch.rand(cfg['batch_size'], 2, generator=self.epsilon_rng)
        radius, angle = 2 * draws[:, 0].sqrt(), 2 * math.pi * draws[:, 1]
        epsilon = torch.stack((radius * angle.sin(), radius * angle.cos()), dim=1).to(self.device)
        choice = int(torch.randint(0, 12, (), generator=self.geometry_rng))
        aux = (48, 56, 64)[choice % 3] if cfg['auxiliary'] == 'CROP' else (1, 2, 4, 8)[choice % 4]
        self.last_attempt=dict(attempted_step=self.completed+1,sample_ids=ids,rotations=rots,
                               epsilon=epsilon.detach().cpu().tolist(),auxiliary_choice=aux)
        multiplier = cosine_factor(self.completed, cfg['total_updates'], cfg['warmup_updates'])
        for group in self.optimizer.param_groups:
            group['lr'] = group['base_lr'] * multiplier
        self.optimizer.zero_grad(set_to_none=True)
        keys = ('total', 'rec', 'epsilon', 'struct', 'scale', 'weighted_epsilon', 'weighted_struct', 'weighted_scale')
        numbers = dict.fromkeys(keys, 0.)
        effective = 0; structure = []; corrections=[]
        for start in range(0, len(ids), cfg['micro_batch']):
            end = start + cfg['micro_batch']; subset = ids[start:end]; rotations = rots[start:end]
            batch = self.bundle.batch('train', subset, rotations, device=self.device)
            result = compute_objective(self.model, batch, cfg, self.completed+1, epsilon=epsilon[start:end],
                auxiliary_choice=aux, **objective_inputs(self.model, self.bundle, cfg, subset, rotations, self.device))
            weight = len(subset) / len(ids)
            (result['total'] * weight).backward()
            for key in keys:
                numbers[key] += float(result[key].detach()) * weight
            effective += result['diagnostics']['epsilon_effective_count']
            corrections.append(result['output']['correction'].detach().cpu())
            item = result['diagnostics'].get('structure', {})
            if 'rho' in item:
                valid = item['valid']
                structure.append(dict(sample_weight=weight, raw_rho=float(item['rho'][valid].mean()) if valid.any() else None,
                    rho2=float(item['rho2'][valid].mean()) if valid.any() else None,
                    valid_band_count=float(item['valid_band_count']), low_texture_fraction=float(item['low_texture_fraction'])))
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise FloatingPointError('NONFINITE_GRADIENT: optimizer not advanced')
        norms = {g['role']: float(sum(p.grad.detach().double().square().sum() for p in g['params'] if p.grad is not None).sqrt())
                 for g in self.optimizer.param_groups}
        # FP32 parameter snapshots also establish actual U/A update magnitudes.
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError('NONFINITE_PARAMETER: last durable checkpoint remains authoritative')
        for state in self.optimizer.state.values():
            if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):
                raise FloatingPointError('NONFINITE_ADAM_MOMENT')
        updates = {}
        for role, prefix in (('U', 'backbone.'), ('A', 'aligner.'), ('global_shift', 'global_shift')):
            terms = [(p.detach().double()-before[n].double()).square().sum()
                     for n, p in self.model.named_parameters() if n.startswith(prefix)]
            updates[role] = float(torch.stack(terms).sum().sqrt()) if terms else None
        a_updates = {}
        for group in ('whole','stem','body','fc1','fc2','bias'):
            terms=[]
            for name,param in self.model.named_parameters():
                if not (name.startswith('aligner.') or name=='global_shift'):
                    continue
                category='fc2' if 'fc2' in name else 'fc1' if 'fc1' in name else 'stem' if 'stem' in name else 'body'
                if group=='whole' or group==category or (group=='bias' and name.endswith('bias')):
                    terms.append((param.detach().double()-before[name].double()).square().sum())
            a_updates[group]=float(torch.stack(terms).sum().sqrt()) if terms else None
        self.completed += 1
        c=torch.cat(corrections)
        return dict(numbers, completed_step=self.completed, elapsed_seconds=time.monotonic()-started,
                    sample_ids=ids, rotations=rots, auxiliary_choice=aux, epsilon_effective_count=effective,
                    grad_norm=norms, actual_parameter_update=updates, actual_a_group_update=a_updates, structural_statistics=structure,
                    correction_mean=c.mean(0).tolist(),correction_mean_norm=float(c.norm(dim=1).mean()),
                    learning_rates={g['role']:g['lr'] for g in self.optimizer.param_groups})


class StopAtBoundary:
    def __init__(self):
        self.requested = False; self.previous = {}

    def __enter__(self):
        def handler(signum, frame):
            self.requested = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.previous[sig] = signal.signal(sig, handler)
        return self

    def __exit__(self, *args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


def disk_guard(directory, minimum_gib=5):
    if shutil.disk_usage(directory).free < minimum_gib * 2**30:
        raise OSError('TA2_DISK_LOW: preserve existing checkpoints, no automatic deletion')


def run_training(bundle, cfg, run_dir, identity, device='cuda', *, event_hook=None, finalize_hook=None,status_hook=None):
    """Return PAUSED or TRAINING/SELECTION/DIAGNOSTICS_COMPLETE, never fake success."""
    identity=dict(identity,run_id=cfg['run_id'],config_sha256=digest(cfg))
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    immutable_json(run_dir/'identity.json', identity); immutable_json(run_dir/'config.json', cfg)
    configure_runtime(cfg['seed'], device)
    model = TeacherModel(cfg['bands'], cfg['seed'], cfg).to(device)
    engine = UpdateEngine(model, bundle, cfg, device)
    initial = dict(model.initial_manifest(), native_stream_sha256=engine.native_stream_sha256)
    immutable_json(run_dir/'initialization.json', initial)
    latest = run_dir/'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Resume source/data/environment identity differs')
        engine.load_state_dict(saved['state'])
    else:
        save_torch(latest, dict(identity=identity, state=engine.state_dict()))
    ledger = EvaluationLedger(run_dir/'selection', identity, cfg['total_updates'])
    events = set(hqnr_grid(cfg['total_updates']))
    if cfg['total_updates'] == 100000:
        events.add(50000)
    event_done = set()

    def early_trace(step):
        target=run_dir/'early_gradients'/f'step_{step:06d}.json'
        if not target.exists():
            from .probes import lightweight_gradient
            with preserve_runtime(model):
                value=lightweight_gradient(model,bundle,cfg,step,device)
            immutable_json(target,value)
    if engine.completed in (0,1,100):
        early_trace(engine.completed)

    def checkpoint(step):
        target = run_dir/'checkpoints'/f'step_{step:06d}.pt'
        if not target.exists():
            disk_guard(run_dir)
            save_torch(target, dict(model=cpu_tree(model.state_dict()), completed_step=step, identity=identity,
                                    model_state_sha256=state_hash(model)))
        return target

    def evaluate_event(step):
        target = checkpoint(step)
        stored = torch.load(target, map_location=device, weights_only=False)
        if stored['identity'] != identity or stored['completed_step'] != step:
            raise ValueError('Candidate checkpoint identity differs')
        current = cpu_tree(model.state_dict())
        try:
            model.load_state_dict(stored['model'], strict=True)
            with preserve_runtime(model):
                report = evaluate_checkpoint(model, bundle.datasets, run_dir/'evaluation'/f'step_{step:06d}',
                    sensor=cfg['dataset'], checkpoint_path=target, completed_step=step, context=identity,
                    device=device, predict=lambda m,p,ms,l,s,i: predict_native(m,bundle,cfg,p,ms,l,s,i))
                ledger.register(report)
                if event_hook:
                    event_hook(model, bundle, cfg, run_dir, target, step, report, device)
            event_done.add(step)
        finally:
            model.load_state_dict(current, strict=True)

    # A kill during evaluator leaves the preceding committed update intact.
    # Historical missing scenes/candidates recover from preserved weights only.
    for step in sorted(events):
        if step <= engine.completed:
            if not (run_dir/'checkpoints'/f'step_{step:06d}.pt').exists():
                if step != engine.completed:
                    raise ValueError('Lost historical candidate weights; cannot fabricate HQNR50')
            evaluate_event(step)
    with StopAtBoundary() as stop:
        while engine.completed < cfg['total_updates']:
            if stop.requested:
                save_torch(latest, dict(identity=identity, state=engine.state_dict()))
                atomic_json(run_dir/'status.json', dict(status='PAUSED', completed_step=engine.completed, updated=now()))
                return dict(status='PAUSED', completed_step=engine.completed)
            if engine.completed % 100 == 0:
                disk_guard(run_dir)
            try:
                log = engine.update()
            except Exception as exc:
                atomic_json(run_dir/'failed_update.json',dict(error_type=type(exc).__name__,error=str(exc),
                    last_completed_in_memory=engine.completed,last_attempt=getattr(engine,'last_attempt',None),
                    durable_resume_file='latest.pt',failed_state_is_not_a_checkpoint=True,updated=now()))
                raise
            if engine.completed in (1,100):
                early_trace(engine.completed)
            if engine.completed <= 2 or engine.completed % 20 == 0 or engine.completed in events:
                append_json(run_dir/'training.jsonl', log)
                print('[TA2] ' + cfg['run_id'] + ' step=' + str(engine.completed) +
                      ' rec=%.8f eps=%.8f struct=%.8f' % (log['rec'], log['epsilon'], log['struct']), flush=True)
            if status_hook and (engine.completed==1 or engine.completed%1000==0 or engine.completed in events):
                status_hook(cfg,run_dir,engine.completed,'RUNNING')
            if engine.completed % 100 == 0 or engine.completed in events:
                save_torch(latest, dict(identity=identity, state=engine.state_dict()))
                atomic_json(run_dir/'status.json', dict(status='TRAINING', completed_step=engine.completed,
                                                       updated=now(), pid=os.getpid(), latest_log=log))
            if engine.completed in events:
                evaluate_event(engine.completed)
        selected = ledger.status()
        if selected['status'] != 'HQNR_SELECTION_COMPLETE':
            raise RuntimeError('Training completed but selection remains pending')
        if stop.requested:
            atomic_json(run_dir/'status.json',dict(status='PAUSED',completed_step=engine.completed,updated=now()))
            return dict(status='PAUSED',completed_step=engine.completed)
        final_result = (finalize_hook(model, bundle, cfg, run_dir, selected, device)
                        if finalize_hook else dict(status='DIAGNOSTICS_PENDING'))
        status = 'COMPLETE' if final_result.get('status') == 'COMPLETE' else 'DIAGNOSTICS_PENDING'
        result = dict(status=status, completed_step=engine.completed, selection=selected,
                      diagnostics=final_result, updated=now())
        atomic_json(run_dir/'status.json', result)
        return result
