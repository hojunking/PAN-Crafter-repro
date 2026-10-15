"""Bounded, restart-visible donor -> Stage 2 orchestration.

Existing experiment packages are read-only dependencies. Interrupted attempts
are never overwritten or silently spliced into another seed's trajectory.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import time
import traceback

import h5py
import numpy as np
import torch

from directshift48.training import build_direct_stage1, DirectStage1Trainer
from directshift48.runtime import code_pin as parent_code_pin, environment_manifest
from directshift48.reporting import atomic_json, append_jsonl, utc_now
from edgeres48.stream import RRStream, set_learning_rates
from panda_method.edge_data import EdgeDataSources
from panda_method.reconstruction import ReconstructionSpec, build_reconstructor
from ta2.evaluation import digest, evaluator_identity, file_sha
from . import plan
from .method import DirectStage2, DirectCalibrationScales, calibrate_training
from .training import StudentTrainer
from .reporting import emit_evaluation, emit_status

ROOT = Path(__file__).resolve().parents[1]


class BudgetExpired(RuntimeError):
    pass


def code_pin():
    return {**parent_code_pin(), **{str(p.relative_to(ROOT)): file_sha(p)
            for p in sorted((ROOT / 'donor30').glob('*.py'))}}


def load_local_payload(path, expected_sha=None):
    path = Path(path).resolve(strict=True)
    if expected_sha is not None and file_sha(path) != expected_sha:
        raise ValueError('Local checkpoint file changed: ' + str(path))
    # Only campaign-owned checkpoints are accepted by the controller; these
    # include optimizer/RNG state and are not third-party pickle downloads.
    return torch.load(path, map_location='cpu', weights_only=False)


def build_trainer(cfg, sources, *, device='cpu', donor_stage=None, calibration=None):
    """Strict fresh construction; no old campaign run-ID validator involved."""
    b, seeds = cfg['baseline_resolved'], cfg['stream_seeds']
    common = dict(gradient_log_interval=b['gradient_log_interval'],
                  effective_batch_size=b['effective_batch_RR'], microbatch_size=b['microbatch_RR'])
    if cfg['phase'] == 'donor':
        stage = build_direct_stage1(sources,
            teacher_spec=ReconstructionSpec(**b['teacher_architecture'], input_layout='P0', base_reference='BICUBIC_MS'),
            teacher_seed=seeds['T_init'], aligner_seed=seeds['A_init'], epsilon_seed=seeds['rr_epsilon'],
            aligner_view_margin=b['aligner_architecture']['view_margin'], roi_margin=b['absolute_roi_margin'],
            boundary_policy=b['boundary_policy'], lambda_abs=cfg['lambda_abs'], lambda_eps=cfg['lambda_epsilon']).to(device)
        groups = [{'params': stage.aligner.parameters(), 'lr': cfg['lr_A'], 'role': 'A'},
                  {'params': stage.teacher.parameters(), 'lr': cfg['lr_U'], 'role': 'T'}]
        trainer_type = DirectStage1Trainer
    elif cfg['phase'] == 'stage2':
        if donor_stage is None:
            payload = load_local_payload(cfg['donor_checkpoint'], cfg['donor_checkpoint_sha256'])
            donor = build_trainer(payload['config'], sources)
            donor.load_state_dict(payload['trainer'])
            from .evaluation import model_state_hash
            if model_state_hash(donor.stage) != cfg['donor_state_hash']:
                raise ValueError('Selected A/T state differs from checkpoint lineage')
            donor_stage = donor.stage
        if calibration is None:
            path = Path(cfg['calibration_path'])
            if file_sha(path) != cfg['calibration_sha256']:
                raise ValueError('Calibration file changed')
            calibration = DirectCalibrationScales.from_dict(json.loads(path.read_text()))
        student = build_reconstructor(ReconstructionSpec(**cfg['student_architecture'],
            input_layout='P0', base_reference='BICUBIC_MS'), seed=seeds['S_init'], role='S')
        stage = DirectStage2(donor_stage, student, mode=cfg['stage2_mode'],
                            calibration=calibration, lambda_spatial=cfg['lambda_edge']).to(device)
        groups = [{'params': stage.trainable_parameters(), 'lr': cfg['lr_U'], 'role': 'T'}]
        trainer_type = StudentTrainer
    else:
        raise ValueError('Unknown training phase')
    optimizer = torch.optim.AdamW(groups, betas=tuple(b['optimizer_betas']),
                                  eps=b['optimizer_eps'], weight_decay=cfg['weight_decay'])
    trainer = trainer_type(stage, optimizer, **common)
    trainer.config = deepcopy(cfg)
    return trainer


def same_predictions(left, right):
    """The saved donor is re-run, not merely selected from metric text."""
    for split in ('rr', 'fr'):
        a, b = left[split]['per_scene'], right[split]['per_scene']
        if len(a) != 20 or len(b) != 20 or a != b:
            raise ValueError('Reloaded donor predictions/metrics differ in ' + split)
    if left['state_before'] != right['state_before']:
        raise ValueError('Reloaded donor identity changed')


class Campaign:
    def __init__(self, config, *, output='work_dir/donor30', device='cuda'):
        plan.validate_config(config)
        self.config, self.device = deepcopy(config), device
        self.root = Path(output).resolve() / plan.CAMPAIGN_ID / config['server'] / config['dataset']
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / 'campaign_state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'started_unix': time.time(), 'created_at': utc_now(), 'runs': [], 'observations': [], 'timing': {}}
        self.code, self.evaluator_pin = code_pin(), evaluator_identity()
        self.sources, self.stop_requested = None, False
        # Combine persisted wall-clock with monotonic time: clock rollback cannot
        # extend an active process. Downtime counts against the same 30h deadline.
        self.initial_elapsed = max(0., self.state.get('elapsed_seconds', 0.),
                                   time.time() - self.state['started_unix'])
        self.monotonic_start = time.monotonic()

    def persist(self):
        self.state['elapsed_seconds'] = self.elapsed()
        atomic_json(self.state_path, self.state)

    def elapsed(self):
        return max(time.time() - self.state['started_unix'],
                   self.initial_elapsed + time.monotonic() - self.monotonic_start)

    @property
    def deadline(self):
        return time.time() + max(0., self.config['budget']['total_seconds'] - self.elapsed())

    def check_time(self, *, closeout=True):
        reserve = self.config['budget']['closeout_seconds'] if closeout else 0
        if self.stop_requested or self.elapsed() >= self.config['budget']['total_seconds'] - reserve:
            raise BudgetExpired('STOP_REQUESTED' if self.stop_requested else '30H_BUDGET_CLOSEOUT')

    def verify_code(self):
        if self.code != code_pin():
            raise RuntimeError('Executable source changed; refusing mixed-code continuation')

    def sync(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)

    def release(self):
        gc.collect()
        if str(self.device).startswith('cuda'):
            torch.cuda.empty_cache()

    def admission(self, catalog_path):
        from .evaluation import source_manifest
        self.check_time()
        b = self.config['baseline_resolved']
        if file_sha(catalog_path) != b['source_registry_sha256']:
            raise ValueError('Source catalog differs from registered SHA256')
        self.sources = EdgeDataSources(catalog_path, root=ROOT, sensor=self.config['dataset'])
        for split in ('train', 'val', 'rr', 'fr'):
            self.sources.raw(split, [0])
            self.check_time()
        environment = environment_manifest(self.device)
        pin = {'config': self.config, 'code': self.code, 'evaluator': self.evaluator_pin,
               'sources': source_manifest(self.sources), 'environment': environment}
        path = self.root / 'admission.json'
        if path.exists() and json.loads(path.read_text()) != pin:
            raise ValueError('Admission pin changed; existing campaign cannot be mixed')
        atomic_json(path, pin)
        atomic_json(self.root / 'environment.json', environment)
        self.persist()

    def calibrate(self, trainer):
        with h5py.File(self.sources._verify('train'), 'r') as source:
            count = len(source['pan'])
        ids = [f'{self.sources.sensor}:train:{i}' for i in range(count)]
        micro = self.config['baseline_resolved']['microbatch_RR']
        def batches():
            for first in range(0, count, micro):
                self.check_time()
                yield self.sources.batch('train', list(range(first, min(count, first + micro))),
                    device=self.device, augmentation=(False, False, 0))
        started = time.monotonic()
        scales = calibrate_training(trainer.stage, batches(), expected_sample_ids=ids)
        self.sync()
        return scales, time.monotonic() - started

    def cpu_trainer_to_gpu(self, trainer):
        from .cost import ensure_cost
        self.check_time()
        ensure_cost(trainer.config, self.sources, retained_trainer=trainer,
                    device=self.device, deadline_unix=self.deadline - self.config['budget']['closeout_seconds'])
        trainer.stage.to(self.device)
        return trainer

    def evaluate(self, trainer):
        from .evaluation import official_evaluate
        return official_evaluate(trainer, self.sources, evaluator_pin=self.evaluator_pin,
            device=self.device, deadline_unix=self.deadline,
            on_scene=lambda *args, **kwargs: self.check_time(closeout=False))

    def trace(self, trainer, stream, directory=None):
        started = time.monotonic()
        lr = set_learning_rates(trainer, trainer.config)
        if trainer.config['phase'] == 'stage2':
            lr = {'S': lr['T'], 'A': 0., 'T_frozen': 0.}
        result = trainer.step_accumulated(stream.next_update())
        self.sync()
        seconds = time.monotonic() - started
        if directory:
            append_jsonl(directory / 'train.jsonl', {'step': trainer.updates,
                'seconds_including_IO': seconds, 'lr': lr, **result})
        if directory and (trainer.updates == 1 or trainer.updates % 100 == 0):
            print(f"[{trainer.config['run_id']}] step={trainer.updates} loss={result['loss']:.7g} update_s={seconds:.3f}", flush=True)
        return seconds

    def preflight(self):
        """Disposable full-method timings on THIS server, never donor candidates."""
        if self.state.get('preflight_complete'):
            return
        from .evaluation import model_state_hash
        cfg = plan.resolve_donor_case(self.config)
        cfg['disposable_preflight'] = True
        trainer = self.cpu_trainer_to_gpu(build_trainer(cfg, self.sources))
        stream = RRStream(self.sources, cfg, device=self.device)
        timings = []
        for _ in range(120):
            self.check_time()
            timings.append(self.trace(trainer, stream))
        started = time.monotonic()
        evaluation = self.evaluate(trainer)
        eval_seconds = time.monotonic() - started
        scales, calibration_seconds = self.calibrate(trainer)
        # Valid synthetic lineage for configuration only. Never saved/selected.
        fake = {'run_id': cfg['run_id'], 'completed_update': trainer.updates,
            'HQNR': 0., 'SCC': 0., 'ERGAS': 1., 'checkpoint_path': 'DISPOSABLE_PREFLIGHT',
            'state_hash': model_state_hash(trainer.stage), 'checkpoint_verified': True,
            'metrics_complete': True, 'metrics_finite': True, 'same_state_verified': True,
            'server': cfg['server'], 'dataset': cfg['dataset']}
        provisional = plan.select_donor([fake], cfg['hqnr_min'], allow_fallback=True)
        student_cfg = plan.resolve_stage2_case(self.config, 'eq', donor=provisional)
        student_cfg['disposable_preflight'] = True
        donor_stage = trainer.stage.cpu()
        del stream, trainer
        self.release()
        student = build_trainer(student_cfg, self.sources, donor_stage=donor_stage, calibration=scales)
        del donor_stage
        student = self.cpu_trainer_to_gpu(student)
        stream = RRStream(self.sources, student_cfg, device=self.device)
        student_timings = []
        for _ in range(120):
            self.check_time()
            student_timings.append(self.trace(student, stream))
        started = time.monotonic()
        self.evaluate(student)
        eval_seconds = max(eval_seconds, time.monotonic() - started)
        self.state['timing'] = {
            'donor_update_seconds': float(np.quantile(timings[20:], .9)),
            'stage2_update_seconds': float(np.quantile(student_timings[20:], .9)),
            'evaluation_seconds': eval_seconds, 'calibration_seconds': calibration_seconds,
            'checkpoint_seconds_estimate': 10., 'basis': 'SAME_SERVER_DISPOSABLE_120_UPDATES_FULL_TRAIN_CALIBRATION'}
        self.state['preflight_complete'] = True
        atomic_json(self.root / 'preflight.json', {'timing': self.state['timing'],
            'eligible_for_selection': False, 'disposable_weights_discarded': True,
            'note': 'No preflight observation enters checkpoint or sheet selection.'})
        del stream, student
        self.release()
        self.persist()

    def estimate(self, phase):
        t = self.state['timing']
        return plan.estimated_run_seconds(t[phase + '_update_seconds'], t['evaluation_seconds'],
            saving_seconds=len(plan.evaluation_steps()) * t['checkpoint_seconds_estimate'])

    def latest_outcomes(self, phase=None):
        latest = {}
        for run in sorted(self.state['runs'], key=lambda r: r['attempt']):
            if phase is None or run['phase'] == phase:
                latest[run['run_id']] = run['status']
        return latest

    def admitted(self, cfg, *, reserve_seconds=0.):
        decision = plan.admission_decision(self.elapsed(), self.estimate(cfg['phase']) + reserve_seconds,
                                           budget=self.config['budget'])
        append_jsonl(self.root / 'decisions.jsonl', {'at': utc_now(), 'run_id': cfg['run_id'], **decision})
        return decision['admitted']

    def _attempt(self, cfg):
        previous = [r for r in self.state['runs'] if r['run_id'] == cfg['run_id']]
        if any(r['status'] == 'COMPLETED' for r in previous):
            return None
        cfg = deepcopy(cfg)
        cfg['attempt'] = 1 + max((r['attempt'] for r in previous), default=0)
        base = self.root / 'runs' / cfg['run_id']
        while (base / ('attempt_%02d' % cfg['attempt'])).exists():
            cfg['attempt'] += 1
        # A process death does not reset the campaign clock or reuse/overwrite
        # partially evaluated files. New attempt is explicitly fresh.
        for run in previous:
            if run['status'] == 'RUNNING':
                run['status'] = 'INTERRUPTED_PROCESS_EXIT'
                atomic_json(Path(run['directory']) / 'run_state.json', run)
                old_cfg = json.loads((Path(run['directory']) / 'config.json').read_text())
                emit_status(self.root, old_cfg, run['status'], step=run.get('step', 0))
        return cfg

    def run_case(self, cfg, *, reserve_seconds=0.):
        from .artifacts import save_candidate, save_resume
        cfg = self._attempt(cfg)
        if cfg is None:
            return
        plan.validate_case(cfg, config=self.config)
        directory = self.root / 'runs' / cfg['run_id'] / ('attempt_%02d' % cfg['attempt'])
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / 'config.json', cfg)
        status = {'run_id': cfg['run_id'], 'attempt': cfg['attempt'], 'directory': str(directory),
                  'phase': cfg['phase'], 'step': 0, 'status': 'RUNNING', 'started_at': utc_now()}
        self.state['runs'].append(status)
        def record_status():
            atomic_json(directory / 'run_state.json', status)
            self.persist()
        if not self.admitted(cfg, reserve_seconds=reserve_seconds):
            status['status'] = 'SKIPPED_BUDGET'
            emit_status(self.root, cfg, status['status'])
            record_status()
            return
        emit_status(self.root, cfg, 'RUNNING')
        record_status()
        trainer = stream = None
        try:
            trainer = self.cpu_trainer_to_gpu(build_trainer(cfg, self.sources))
            stream = RRStream(self.sources, cfg, device=self.device)
            timings = []
            for _ in range(cfg['planned_horizon']):
                self.check_time()
                if trainer.updates % 100 == 0:
                    self.verify_code()
                timings.append(self.trace(trainer, stream, directory))
                status['step'] = trainer.updates
                if trainer.updates % 100 == 0:
                    record_status()
                if trainer.updates not in cfg['evaluation_steps']:
                    continue
                started = time.monotonic()
                evaluation = self.evaluate(trainer)
                self.state['timing']['evaluation_seconds'] = max(self.state['timing']['evaluation_seconds'], time.monotonic() - started)
                eval_path = directory / ('evaluation_%07d.json' % trainer.updates)
                if eval_path.exists():
                    raise RuntimeError('Refusing to overwrite existing evaluation')
                atomic_json(eval_path, evaluation)
                started = time.monotonic()
                saved = save_candidate(trainer, stream, evaluation, directory, cfg,
                    reload_factory=lambda resolved: build_trainer(resolved, self.sources))
                self.state['timing']['checkpoint_seconds_estimate'] = max(
                    self.state['timing']['checkpoint_seconds_estimate'], time.monotonic() - started)
                artifact = saved['artifact'] if saved else None
                if saved:
                    observation = {**saved['selection_record'], 'phase': cfg['phase'],
                        'server': cfg['server'], 'dataset': cfg['dataset'], 'attempt': cfg['attempt'],
                        'evaluation_path': str(eval_path), 'file_sha256': artifact['file_sha256']}
                    self.state['observations'].append(observation)
                emit_evaluation(self.root, cfg, evaluation, artifact=artifact)
                print(f"[{cfg['case']}/{cfg['replica']}] step={trainer.updates} "
                      f"HQNR={evaluation['fr']['metrics']['HQNR']:.9f} "
                      f"SCC={evaluation['rr']['metrics']['SCC']:.9f} ERGAS={evaluation['rr']['metrics']['ERGAS']:.9f} saved={bool(saved)}", flush=True)
                self.state['timing'][cfg['phase'] + '_update_seconds'] = float(np.quantile(timings[-1000:], .9))
                record_status()
            status['status'] = 'COMPLETED'
        except BudgetExpired as exc:
            status.update(status='STOPPED_BUDGET', reason=str(exc))
        except Exception as exc:
            status.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
            print(status['traceback'], flush=True)
        finally:
            if trainer is not None and stream is not None and not trainer.poisoned:
                try:
                    save_resume(trainer, stream, directory, cfg)
                except Exception as exc:
                    status['resume_save_error'] = repr(exc)
            status['ended_at'] = utc_now()
            record_status()
            emit_status(self.root, cfg, status['status'], step=status['step'], extra=status)
            del stream, trainer
            self.release()

    def prepare_selected_donor(self):
        from .evaluation import model_state_hash
        observations = [r for r in self.state['observations'] if r['phase'] == 'donor']
        donor = plan.select_donor(observations, self.config['donor']['hqnr_min'],
                                 self.config['donor']['allow_exploratory_fallback'])
        if donor is None:
            raise RuntimeError('No verified saved local donor. Stage 2 cannot fabricate an A/T state.')
        payload = load_local_payload(donor['checkpoint_path'], donor['file_sha256'])
        trainer = build_trainer(payload['config'], self.sources, device=self.device)
        trainer.load_state_dict(payload['trainer'])
        if model_state_hash(trainer.stage) != donor['state_hash']:
            raise ValueError('Selected saved donor state changed')
        original = json.loads(Path(donor['evaluation_path']).read_text())
        same_predictions(original, self.evaluate(trainer))
        scales, seconds = self.calibrate(trainer)
        self.state['timing']['calibration_seconds'] = seconds
        path = self.root / ('calibration_' + donor['state_hash'] + '.json')
        if path.exists() and json.loads(path.read_text()) != scales.to_dict():
            raise ValueError('Same donor produced inconsistent calibration')
        atomic_json(path, scales.to_dict())
        donor.update(reloaded_full40_verified=True, calibration_path=str(path), calibration_sha256=file_sha(path))
        atomic_json(self.root / 'selected_donor.json', donor)
        self.state['selected_donor'] = donor
        self.persist()
        del trainer
        self.release()
        return donor

    def run(self, *, catalog_path=ROOT / 'ablr2/sensor_sources.json'):
        lock = (self.root / '.controller.lock').open('a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError('This donor30 campaign already has an active controller')
        previous_handlers = {}
        def stop(signum, frame):
            self.stop_requested = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, stop)
        try:
            self.persist()
            self.admission(catalog_path)
            from .recovery import recover_campaign
            recovered = recover_campaign(self.root, self.config['server'])
            known = {(r['run_id'], r.get('attempt', 1), r['completed_update'], r['state_hash'])
                     for r in self.state['observations']}
            for observation in recovered['observations']:
                key = (observation['run_id'], observation.get('attempt', 1),
                       observation['completed_update'], observation['state_hash'])
                if key not in known:
                    self.state['observations'].append(observation)
                    known.add(key)
            self.persist()
            self.preflight()
            # Reserve both independent donors and at least the three primary
            # matched Stage2 controls before consuming a full donor trajectory.
            reserve_core = 3 * self.estimate('stage2') + self.state['timing']['calibration_seconds']
            for index, replica in enumerate(('R1', 'R2')):
                self.check_time()
                cfg = plan.resolve_donor_case(self.config, replica)
                reserve = reserve_core + (self.estimate('donor') if index == 0 else 0.)
                self.run_case(cfg, reserve_seconds=reserve)
            completed_donors = {r['run_id'] for r in self.state['runs']
                                if r['phase'] == 'donor' and r['status'] == 'COMPLETED'}
            if len(completed_donors) != 2:
                raise RuntimeError('Both independent donor runs must complete before Stage 2; '
                                   'HQNR fallback is authorized, failed/incomplete training is not a completed donor trial.')
            self.check_time()
            donor = self.prepare_selected_donor()
            for cfg in plan.stage2_cases(self.config, donor):
                self.check_time()
                cfg.update(calibration_path=donor['calibration_path'], calibration_sha256=donor['calibration_sha256'],
                           donor_checkpoint_sha256=donor['file_sha256'])
                if cfg.get('optional') and any(status != 'COMPLETED'
                                               for status in self.latest_outcomes('stage2').values()):
                    emit_status(self.root, cfg, 'SKIPPED_REQUIRED_CASE_INCOMPLETE')
                    continue
                self.run_case(cfg)
            selected = plan.select_stage2([r for r in self.state['observations'] if r['phase'] == 'stage2'],
                                          self.config['donor']['hqnr_min'])
            atomic_json(self.root / 'selected_student.json', selected)
            self.state['status'] = ('FINISHED_QUEUE' if all(status == 'COMPLETED' for status in self.latest_outcomes().values())
                                    else 'FINISHED_WITH_FAILURES_OR_BUDGET_SKIPS')
        except BudgetExpired as exc:
            self.state.update(status='STOPPED_BUDGET', reason=str(exc))
        except Exception as exc:
            self.state.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            students = [r for r in self.state['observations'] if r['phase'] == 'stage2']
            if students:
                atomic_json(self.root / 'selected_student.json', plan.select_stage2(
                    students, self.config['donor']['hqnr_min']))
            self.state['elapsed_seconds'] = self.elapsed()
            self.persist()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        return self.state
