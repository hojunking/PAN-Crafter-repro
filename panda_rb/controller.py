"""Finite B01 admissions. No historical queues, remote writes or extra seeds."""
from pathlib import Path
import traceback

from panda_rb.common import ROOT, append_event, atomic_json, locked, read, server_dir, utcnow
from panda_rb.plan import MODES, binding_path, run_dir, schedule, training_runs, validate_plan, weights_dir


def status(server, root=ROOT):
    validate_plan(root)
    runs = []
    for case in training_runs(server, root):
        wd = run_dir(case['run_id'], root)
        train = read(wd / 'meta/training_status.json')
        runs.append(dict(case, training=train.get('status', 'NOT_STARTED'),
                         update=train.get('actual_updates', train.get('update', 0)),
                         native=(wd / 'native/metrics.json').is_file(),
                         curves={mode: (wd / 'stress' / mode / 'completion.json').is_file() for mode in MODES},
                         failure=read(wd / 'meta/controller_failure.json')))
    return dict(server=server, controller=read(server_dir(server, root) / 'status.json'),
                runs=runs, note='File presence is status only; the runner verifies sealed artifacts before skipping.')


def run(server, root=ROOT, binding=None, device='cuda', retry_technical=False):
    """Run the preregistered finite local queue; technical errors require explicit retry."""
    from panda_rb.common import docker_required
    from panda_rb.preflight import prepare
    from panda_rb.training import train_run
    from panda_rb.evaluation import evaluate_native, evaluate_stress, validate_artifacts
    from panda_rb.reporting import summarize

    root = Path(root).resolve(); control = server_dir(server, root)
    docker_required()
    validate_plan(root)
    binding = Path(binding or binding_path(root))
    with locked(control / 'controller.lock'):
        old = read(control / 'status.json')
        if old.get('state') == 'TECHNICAL_FAILURE' and not retry_technical:
            raise RuntimeError('Technical failure: inspect evidence and rerun with --retry-technical; seed is unchanged')
        prepare(server, binding, root=root, device=device)
        atomic_json(control / 'status.json', dict(state='RUNNING', server=server, at_utc=utcnow()))
        for action in schedule(server, root):
            if action['action'] == 'STOP_FOR_REVIEW':
                report = summarize(root=root, servers=[server], verify=True)
                atomic_json(control / 'summary.json', report)
                atomic_json(control / 'status.json', dict(state='STOP_FOR_REVIEW', server=server,
                    completed_students=report['completed_students'], completed_curves=report['completed_curves'],
                    failed_students=report['failed_students'], expected_students=8, expected_curves=16,
                    next_experiments_admitted=False, at_utc=utcnow()))
                return report
            run_id = action['run_id']; wd = run_dir(run_id, root)
            failure_path = wd / 'meta/controller_failure.json'
            failure = read(failure_path)
            if failure.get('kind') == 'NUMERICAL_FAILURE':
                append_event(control / 'events.jsonl', dict(event='SKIP_FAILED_SAME_SEED', action=action))
                continue
            append_event(control / 'events.jsonl', dict(event='BEGIN', action=action))
            try:
                if action['action'] == 'TRAIN_NATIVE_EVAL':
                    selected = wd / 'checkpoints/selection_manifest.json'
                    if not selected.exists():
                        resume = (wd / 'checkpoints/last').exists()
                        returned = train_run(run_id, binding, weights_dir(root), root=root, device=device, resume=resume)
                        if returned == 75:
                            paused = dict(state='PAUSED_SIGNAL', server=server, run_id=run_id, at_utc=utcnow())
                            atomic_json(control / 'status.json', paused)
                            return paused
                    report = evaluate_native(run_id, binding, root=root, device=device)
                elif action['action'] == 'STRESS_CURVE':
                    # Evaluation itself rechecks exact50000 and the full 49 x 20 grid.
                    report = evaluate_stress(run_id, binding, action['mode'], root=root, device=device)
                else:
                    raise ValueError('Unknown preregistered action')
                validate_artifacts(report, wd)
            except FloatingPointError as exc:
                # Keep the registered seed/failure in the denominator. No replacement seed.
                atomic_json(failure_path, dict(kind='NUMERICAL_FAILURE', run_id=run_id,
                    action=action, error=str(exc), traceback=traceback.format_exc(), at_utc=utcnow()))
                append_event(control / 'events.jsonl', dict(event='NUMERICAL_FAILURE', action=action, error=str(exc)))
                continue
            except Exception as exc:
                atomic_json(failure_path, dict(kind='TECHNICAL_FAILURE', run_id=run_id,
                    action=action, error=str(exc), traceback=traceback.format_exc(), at_utc=utcnow()))
                atomic_json(control / 'status.json', dict(state='TECHNICAL_FAILURE', server=server,
                    action=action, error=str(exc), at_utc=utcnow()))
                raise
            if failure.get('kind') == 'TECHNICAL_FAILURE':
                atomic_json(failure_path, dict(kind='RESOLVED', previous=failure, at_utc=utcnow()))
            append_event(control / 'events.jsonl', dict(event='COMPLETE', action=action,
                                                       report_sha256=report['payload_sha256']))
        raise AssertionError('The B01 schedule must end with STOP_FOR_REVIEW')
