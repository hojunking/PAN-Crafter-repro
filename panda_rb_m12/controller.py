"""Finite local queue, separate diagnostic debt, no Sheet writes or old queues."""
from pathlib import Path
import traceback
from panda_rb_m12.common import ROOT, atomic_json, append_event, locked, read, server_dir, utcnow
from panda_rb_m12.plan import MODES, STEP1, binding_path, weights_dir, run_dir, schedule, training_runs, validate_plan


def status(server, root=ROOT):
    validate_plan(root)
    rows = []
    for case in training_runs(server, root):
        wd = run_dir(case['run_id'], root)
        rows.append(dict(case, training=read(wd / 'meta/training_status.json'),
                         native=(wd / 'native/metrics.json').exists(),
                         curves={m: (wd / 'stress' / m / 'completion.json').exists() for m in MODES}
                         if case['case_id'] in STEP1 else {},
                         failure=read(wd / 'meta/controller_failure.json')))
    return dict(server=server, controller=read(server_dir(server, root) / 'status.json'), runs=rows,
                note='Status existence is not completion verification; run/report verify seals')


def run(server, root=ROOT, binding=None, device='cuda', retry_technical=False, activate=False):
    if not activate:
        raise PermissionError('Training requires explicit --activate')
    from panda_rb_m12.common import docker_required
    from panda_rb_m12.preflight import prepare
    from panda_rb_m12.training import train_run
    from panda_rb_m12.evaluation import evaluate_native, evaluate_stress
    from panda_rb_m12.reporting import summarize
    docker_required(); validate_plan(root)
    root = Path(root).resolve(); control = server_dir(server, root)
    binding = Path(binding or binding_path(root))
    with locked(control / 'controller.lock'):
        previous = read(control / 'status.json')
        if previous.get('state') == 'STOP_FOR_REVIEW':
            return previous
        if previous.get('state') == 'TECHNICAL_FAILURE' and not retry_technical:
            raise RuntimeError('Inspect failure then retry same seed with --retry-technical')
        try:
            prepare(server, binding, root, device)
        except Exception as exc:
            state = 'WAITING_FOR_GPU_OWNER' if str(exc).startswith('WAITING_FOR_GPU_OWNER:') else 'TECHNICAL_FAILURE'
            failed = dict(state=state, server=server, stage='PREFLIGHT', error=str(exc), at_utc=utcnow())
            atomic_json(control / 'status.json', failed)
            if state == 'WAITING_FOR_GPU_OWNER':
                return failed
            raise
        atomic_json(control / 'status.json', dict(state='RUNNING', server=server, at_utc=utcnow()))
        debts = read(control / 'diagnostic_debt.json', default={'tasks': {}})['tasks']
        def technical_failure(exc, run_id=None, path=None):
            record = dict(kind='TECHNICAL_FAILURE', run_id=run_id, error=str(exc),
                          traceback=traceback.format_exc(), at_utc=utcnow())
            if path is not None:
                atomic_json(path, record)
            atomic_json(control / 'status.json', dict(state='TECHNICAL_FAILURE', server=server,
                                                      run_id=run_id, error=str(exc), at_utc=utcnow()))
        for action in schedule(server, root):
            kind = action['action']; run_id = action['run_id']
            if kind in ('VERIFY_ASSETS_AND_RECOVER_OLD_EVIDENCE', 'DIAGNOSE_F1_INITIAL_AND_EXISTING_FINAL_ALIGNER'):
                continue  # Already verified by mandatory automatic preflight.
            if kind == 'SAVE_PHASE_REPORT_NO_SCORE_GATE':
                try:
                    report = summarize(root=root, servers=[server], verify=True)
                except Exception as exc:
                    technical_failure(exc)
                    raise
                atomic_json(control / ('phase_report_' + action['phase'] + '.json'), report)
                continue  # No score predicate and no cross-server barrier.
            if kind == 'FINAL_REPORT_AND_STOP_FOR_REVIEW':
                try:
                    report = summarize(root=root, servers=[server], verify=True)
                except Exception as exc:
                    technical_failure(exc)
                    raise
                atomic_json(control / 'summary.json', report)
                finished = dict(state='STOP_FOR_REVIEW', server=server, at_utc=utcnow(),
                    expected_students=40, expected_native=80, expected_curves=96,
                    completed_students=report['completed_students'], completed_native=report['completed_native'],
                    completed_curves=report['completed_curves'], diagnostic_debt=len(debts),
                    sheets_uploaded=False, further_training_admitted=False)
                atomic_json(control / 'status.json', finished)
                return finished
            wd = run_dir(run_id, root); failure_path = wd / 'meta/controller_failure.json'
            if read(failure_path).get('kind') == 'NUMERICAL_FAILURE':
                continue  # Failed seed remains in denominator, never replaced.
            if kind == 'QUEUE_VERIFIED_UPLOAD':
                # Local evidence spool only; no network or implicit writer activation.
                from panda_rb_m12.evaluation import validate_artifacts
                native = read(wd / 'native/metrics.json')
                if native:
                    try:
                        validate_artifacts(native, wd)
                    except Exception as exc:
                        technical_failure(exc, run_id, failure_path)
                        raise
                    atomic_json(control / 'upload_spool' / (run_id + '.json'), dict(
                        run_id=run_id, native_payload_sha256=native['payload_sha256'], state='PENDING_EXPLICIT_S1_WRITER'))
                continue
            append_event(control / 'events.jsonl', dict(event='BEGIN', action=action))
            # Training errors cannot be disguised as diagnostic/upload debts.
            if kind == 'TRAIN_FRESH_50K_AND_NATIVE' and not (wd / 'checkpoints/selection_manifest.json').exists():
                try:
                    value = train_run(run_id, binding, weights_dir(root), device=device,
                                      resume=(wd / 'checkpoints/last').exists(), root=root)
                    if value == 75:
                        paused = dict(state='PAUSED_SIGNAL', server=server, run_id=run_id, at_utc=utcnow())
                        atomic_json(control / 'status.json', paused)
                        return paused
                    if value != 0:
                        raise RuntimeError('Trainer did not report successful completion: ' + repr(value))
                    failed = read(failure_path)
                    if failed.get('kind') == 'TECHNICAL_FAILURE':
                        atomic_json(failure_path, dict(kind='RESOLVED',
                            run_id=run_id, previous_failure=failed, resolved_at_utc=utcnow()))
                except FloatingPointError as exc:
                    atomic_json(failure_path, dict(kind='NUMERICAL_FAILURE', run_id=run_id, error=str(exc),
                                                   traceback=traceback.format_exc(), at_utc=utcnow()))
                    continue
                except Exception as exc:
                    technical_failure(exc, run_id, failure_path)
                    raise
            key = run_id + '|' + ('NATIVE' if kind == 'TRAIN_FRESH_50K_AND_NATIVE' else action['mode'])
            try:
                if kind == 'TRAIN_FRESH_50K_AND_NATIVE':
                    report = evaluate_native(run_id, binding, root=root, device=device)
                elif kind == 'STRESS_INFERENCE':
                    report = evaluate_stress(run_id, binding, action['mode'], root=root, device=device)
                else:
                    raise ValueError('Unknown schedule action: ' + kind)
                debts.pop(key, None)
                failed = read(failure_path)
                if failed.get('kind') == 'TECHNICAL_FAILURE':
                    atomic_json(failure_path, dict(kind='RESOLVED', run_id=run_id,
                        previous_failure=failed, resolved_at_utc=utcnow()))
                append_event(control / 'events.jsonl', dict(event='COMPLETE', action=action,
                                                           payload_sha256=report['payload_sha256']))
            except (ValueError, AssertionError) as exc:
                # Checkpoint/source/protocol/schema drift is not a slow optional
                # diagnostic. Stop this cohort instead of concealing corruption.
                technical_failure(exc, run_id, failure_path)
                raise
            except Exception as exc:
                debts[key] = dict(action=action, error=str(exc), traceback=traceback.format_exc(), at_utc=utcnow())
                append_event(control / 'events.jsonl', dict(event='DIAGNOSTIC_PENDING', action=action, error=str(exc)))
            atomic_json(control / 'diagnostic_debt.json', dict(tasks=debts))
        raise AssertionError('Finite queue stop missing')


def retry_diagnostics(server, root=ROOT, binding=None, device='cuda', activate=False):
    if not activate:
        raise PermissionError('Diagnostic inference requires explicit --activate; never retrains')
    from panda_rb_m12.evaluation import evaluate_native, evaluate_stress
    from panda_rb_m12.common import docker_required
    docker_required(); validate_plan(root)
    from panda_rb_m12.preflight import prepare
    control = server_dir(server, root); binding = binding or binding_path(root)
    with locked(control / 'controller.lock'):
        prepare(server, binding, root, device)
        debts = read(control / 'diagnostic_debt.json', {'tasks': {}})['tasks']
        for key, item in list(debts.items()):
            row = item['action']
            try:
                if row['action'] == 'TRAIN_FRESH_50K_AND_NATIVE':
                    evaluate_native(row['run_id'], binding, root=root, device=device)
                else:
                    evaluate_stress(row['run_id'], binding, row['mode'], root=root, device=device)
                del debts[key]
            except (ValueError, AssertionError) as exc:
                atomic_json(control / 'status.json', dict(state='TECHNICAL_FAILURE', server=server,
                    run_id=row['run_id'], error=str(exc), at_utc=utcnow()))
                raise
            except Exception as exc:
                debts[key] = dict(item, error=str(exc), at_utc=utcnow())
            atomic_json(control / 'diagnostic_debt.json', dict(tasks=debts))
        previous = read(control / 'status.json')
        if previous.get('state') == 'STOP_FOR_REVIEW':
            from panda_rb_m12.reporting import summarize
            report = summarize(root=root, servers=[server], verify=True)
            atomic_json(control / 'summary.json', report)
            atomic_json(control / 'status.json', dict(previous, diagnostic_debt=len(debts),
                completed_students=report['completed_students'], completed_native=report['completed_native'],
                completed_curves=report['completed_curves'], diagnostics_retried_at_utc=utcnow()))
        return dict(pending_diagnostics=len(debts), training_updates=0, sheets_uploaded=False)
