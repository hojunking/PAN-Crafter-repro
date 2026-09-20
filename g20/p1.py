"""Bounded analysis receipts, never additional optimization or a BASE gate."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import logging

from g20.common import atomic_json, read, sha256


def _incomplete_receipt(path, context, error, **extra):
    """A diagnostics-only write failure cannot invalidate a saved checkpoint."""
    detail = f'{type(error).__name__}: {error}'
    logging.getLogger(__name__).warning('G20 P1 incomplete at %s: %s', path, detail)
    try:
        atomic_json(path, dict(context, status='INCOMPLETE', priority='P1',
            base_training_blocked=False, branch_admissible=False, error=detail, **extra))
    except Exception as receipt_error:
        logging.getLogger(__name__).warning(
            'G20 P1 receipt unavailable at %s; BASE may continue, no evidence published: %s: %s',
            path, type(receipt_error).__name__, receipt_error)


def bounded_deadline(shared, seconds=60.):
    stop = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    if shared:
        end = datetime.fromisoformat(str(shared).replace('Z', '+00:00'))
        if end.tzinfo is None:
            raise ValueError('P1 deadline needs timezone')
        stop = min(stop, end)
    return stop.isoformat()


def record_training_diagnostics(wd, model, teacher, datasets, *, update, device,
                                profile, reference, q_cache, consistency_weight,
                                identity, deadline=None):
    """Failed P1 leaves evidence unavailable; corrupted saved evidence stays fatal.

    A failed attempt is not replayed on ordinary full-state resume. Saved normal
    checkpoints can be analysed separately without adding optimizer updates.
    """
    from fh12.calibration import _write_npz_immutable
    from g20.diagnostics import capture_diagnostics
    from g20.model import state_hash
    from g20.training import rng_state, restore_rng
    start = time.monotonic()
    wd = Path(wd)
    digest = state_hash(model.state_dict())
    context = dict(identity, update=update, model_state_hash=digest)
    path = wd / 'diagnostics' / f'update_{update}.json'
    arrays_path = path.with_suffix('.npz')
    required = path.with_name(f'required_{update}.json')
    old = read(path)
    if old:
        if (any(old.get(k) != v for k, v in context.items())
                or old.get('arrays_sha256') != sha256(arrays_path)):
            raise ValueError('Saved P1 checkpoint/arrays provenance changed')
    saved_rng = rng_state()
    modules = [(m, m.training) for net in (model, teacher) if net is not None for m in net.modules()]
    params = [(p, p.requires_grad, None if p.grad is None else p.grad.detach().clone())
              for net in (model, teacher) if net is not None for p in net.parameters()]
    teacher_digest = state_hash(teacher.state_dict()) if teacher is not None else None
    try:
        if not old and not read(required).get('attempt_complete'):
            try:
                atomic_json(required, dict(context, status='TO_MEASURE', branch_admissible=False))
                measured, arrays = capture_diagnostics(model, teacher, datasets['train'],
                    device=device, profile=profile, tau_R=reference.get('tau_R'),
                    q_cache=q_cache, q_ref=reference.get('q_ref'),
                    consistency_weight=consistency_weight,
                    deadline_utc=bounded_deadline(deadline))
                _write_npz_immutable(arrays_path, arrays)
                atomic_json(path, dict(measured, **context, arrays_path=str(arrays_path),
                                       arrays_sha256=sha256(arrays_path)))
                atomic_json(required, dict(context, status='MEASURED', attempt_complete=True,
                    distribution_samples=128, gradient_samples=24, evidence=str(path), branch_admissible=False))
            except Exception as error:
                _incomplete_receipt(required, context, error, attempt_complete=True)
        size_path = wd / 'diagnostics' / f'alignment_sizes_{update}.json'
        previous = read(size_path)
        if previous and any(previous.get(k) != v for k, v in context.items()):
            raise ValueError('Saved size-analysis checkpoint provenance changed')
        if not previous:
            try:
                from qg40.alignment_diagnostics import capture_alignment_sizes
                report = capture_alignment_sizes(model, datasets, device=device,
                    deadline_utc=bounded_deadline(deadline, 30.))
                atomic_json(size_path, dict(report, **context, priority='P1', base_training_blocked=False))
            except Exception as error:
                _incomplete_receipt(size_path, context, error)
    finally:
        for module, mode in modules:
            module.training = mode
        for parameter, requires_grad, gradient in params:
            parameter.requires_grad_(requires_grad)
            parameter.grad = gradient
        restore_rng(saved_rng)
        if (state_hash(model.state_dict()) != digest or (teacher is not None
                and state_hash(teacher.state_dict()) != teacher_digest)):
            raise ValueError('P1 analysis mutated model weights; original full-state must be preserved')
    return time.monotonic() - start
