"""Additive Stage 1 controller; old live campaign code is never patched."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from donor30.runtime import (Campaign as BaseCampaign, BudgetExpired, build_trainer,
    code_pin as parent_code_pin, load_local_payload, same_predictions)
from donor30.artifacts import read_selection
from edgeres48.stream import RRStream
from ta2.evaluation import evaluator_identity, file_sha
from . import plan
from .reporting import atomic_json, append_jsonl, utc_now, emit_status, emit_evaluation

ROOT = plan.ROOT


def code_pin():
    files = [*ROOT.glob('donor24/*.py'), *ROOT.glob('donor24/*.txt'),
             *ROOT.glob('config/donor24/*'), plan.BASELINE_PATH]
    return {**parent_code_pin(), **{str(p.relative_to(ROOT)): file_sha(p) for p in files}}


class Campaign(BaseCampaign):
    def __init__(self, config, *, output='work_dir/donor24', device='cuda'):
        plan.validate_config(config)
        self.config, self.device = deepcopy(config), device
        self.root = Path(output).resolve()/plan.CAMPAIGN_ID/config['server']/'WV3'
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root/'campaign_state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'started_unix': time.time(), 'created_at': utc_now(), 'runs': [], 'observations': [], 'timing': {}}
        self.code, self.evaluator_pin = code_pin(), evaluator_identity()
        self.sources, self.stop_requested = None, False
        self.initial_elapsed = max(0., self.state.get('elapsed_seconds', 0.), time.time()-self.state['started_unix'])
        self.monotonic_start = time.monotonic()

    def verify_code(self):
        if self.code != code_pin():
            raise RuntimeError('Executable/config source changed during donor24')

    def check_time(self, *, closeout=True):
        reserve = self.config['budget']['closeout_seconds'] if closeout else 0
        if self.stop_requested or self.elapsed() >= self.config['budget']['total_seconds']-reserve:
            raise BudgetExpired('STOP_REQUESTED' if self.stop_requested else '24H_BUDGET_CLOSEOUT')

    def evaluate(self, trainer):
        from .evaluation import official_evaluate
        return official_evaluate(trainer, self.sources, evaluator_pin=self.evaluator_pin,
            device=self.device, deadline_unix=self.deadline,
            on_scene=lambda *a, **kw: self.check_time(closeout=False))

    def probes(self, trainer):
        from directshift48.diagnostics import run_probes
        spec = deepcopy(trainer.config['baseline_resolved']['probes'])
        # Dedicated geometric probes; training already logs isolated gradients.
        spec.pop('gradient_rows', None)
        return run_probes(trainer, self.sources, spec, device=self.device,
            evaluator_pin=self.evaluator_pin, deadline_unix=self.deadline-3600)

    def preflight(self):
        if self.state.get('preflight_complete'):
            return
        cfg = plan.resolve_case(self.config, 'A0')
        cfg['disposable_preflight'] = True
        trainer = self.cpu_trainer_to_gpu(build_trainer(cfg, self.sources))
        stream = RRStream(self.sources, cfg, device=self.device)
        timings = []
        for _ in range(120):
            self.check_time()
            timings.append(self.trace(trainer, stream))
        evaluation = self.evaluate(trainer)
        probe = self.probes(trainer)
        if not all(evaluation['qualification'].get(k) is True for k in
                   ('metrics_complete','metrics_finite','same_state_verified')) or probe['status'] != 'COMPLETE':
            raise RuntimeError('Disposable WV3 full40/probe preflight incomplete')
        self.state['timing'] = {'donor_update_seconds': float(np.quantile(timings[20:], .9)),
            'evaluation_seconds': evaluation['elapsed_seconds'], 'checkpoint_seconds_estimate': 10.,
            'probe_seconds': max(10., probe['elapsed_seconds']),
            'basis': 'THIS_SERVER_WV3_120_UPDATES_FULL40_AND_PROBES'}
        self.state['preflight_complete'] = True
        atomic_json(self.root/'preflight.json', {'timing': self.state['timing'],
            'eligible_for_selection': False, 'disposable_weights_discarded': True,
            'dataset': 'WV3', 'server': self.config['server'],
            'evaluation': evaluation, 'geometry_probe': probe})
        del stream, trainer
        self.release()
        self.persist()

    def estimate(self, phase='donor'):
        if phase != 'donor':
            raise ValueError('This search never runs Stage 2')
        t = self.state['timing']
        return (50000*t['donor_update_seconds'] + len(plan.evaluation_steps()) *
            (t['evaluation_seconds']+t['checkpoint_seconds_estimate']) + 4*t['probe_seconds'])

    def recover_candidates(self):
        # Recover checkpoint pointers even if a process died after saving but
        # before appending campaign_state. Failed/incomplete runs remain excluded.
        records = {}
        for run in self.state['runs']:
            directory = self.root/'runs'/run['run_id']/('attempt_%02d' % run['attempt'])
            cfg = json.loads((directory/'config.json').read_text())
            for r in read_selection(directory)['saved']:
                records[(r['run_id'], r['attempt'], r['completed_update'])] = {
                    **r, 'phase': 'donor', 'server': cfg['server'], 'dataset': 'WV3',
                    'case': cfg['case'], 'replica': cfg['replica'],
                    'evaluation_path': str(directory/('evaluation_%07d.json' % r['completed_update']))}
        for r in self.state['observations']:
            records[(r['run_id'], r['attempt'], r['completed_update'])] = r
        self.state['observations'] = list(records.values())

    def prepare_local_candidate(self):
        from .evaluation import model_state_hash
        chosen = plan.choose_completed(self.state['observations'], self.state['runs'])
        if chosen is None:
            raise RuntimeError('No completed reloadable donor trial; no candidate fabricated')
        path = self.root/'runs'/chosen['run_id']/('attempt_%02d' % chosen['attempt'])/'models'/('step_%07d.pt' % chosen['completed_update'])
        payload = load_local_payload(path, chosen.get('file_sha256'))
        trainer = build_trainer(payload['config'], self.sources, device=self.device)
        trainer.load_state_dict(payload['trainer'])
        if model_state_hash(trainer.stage) != chosen['state_hash']:
            raise ValueError('Saved donor model identity changed')
        original = json.loads(Path(chosen['evaluation_path']).read_text())
        same_predictions(original, self.evaluate(trainer))
        chosen.update(checkpoint_path=str(path), file_sha256=file_sha(path),
            reloaded_full40_verified=True, shared_donor_selected=False,
            next_step='COMPARE_SERVER_NOMINEES_THEN_BIND_ONE_SHARED_A_T_SHA')
        atomic_json(self.root/'local_donor_candidate.json', chosen)
        self.state['local_donor_candidate'] = chosen
        del trainer
        self.release()

    def run(self, *, catalog_path=ROOT/'ablr2/sensor_sources.json'):
        with (self.root/'.controller.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = {}
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, lambda *a: setattr(self, 'stop_requested', True))
            try:
                self.persist()
                self.admission(catalog_path)
                self.recover_candidates()
                self.preflight()
                for case in self.config['search_order']:
                    self.check_time()
                    # A full R2 is reserved before EVERY fresh search admission.
                    self.run_case(plan.resolve_case(self.config, case), reserve_seconds=self.estimate())
                winner = plan.choose_completed(self.state['observations'], self.state['runs'], replica='R1')
                if winner is None:
                    raise RuntimeError('No completed R1; confirmation cannot be invented')
                atomic_json(self.root/'confirmation_recipe.json', winner)
                self.run_case(plan.resolve_case(self.config, winner['case'], 'R2'))
                self.check_time()
                self.prepare_local_candidate()
                outcomes = self.latest_outcomes()
                self.state['all_registered_runs_completed'] = all(v == 'COMPLETED' for v in outcomes.values())
                self.state['confirmation_completed'] = outcomes.get(
                    plan.resolve_case(self.config, winner['case'], 'R2')['run_id']) == 'COMPLETED'
                self.state['status'] = 'FINISHED_LOCAL_SEARCH_SHARED_DONOR_PENDING'
            except BudgetExpired as exc:
                self.state.update(status='STOPPED_BUDGET', reason=str(exc))
            except Exception as exc:
                self.state.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
                raise
            finally:
                self.persist()
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            return self.state

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
        from donor30.artifacts import save_candidate, save_resume
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
                        'case': cfg['case'], 'replica': cfg['replica'],
                        'evaluation_path': str(eval_path), 'file_sha256': artifact['file_sha256']}
                    self.state['observations'].append(observation)
                emit_evaluation(self.root, cfg, evaluation, artifact=artifact)
                print(f"[{cfg['case']}/{cfg['replica']}] step={trainer.updates} "
                      f"HQNR={evaluation['fr']['metrics']['HQNR']:.9f} "
                      f"SCC={evaluation['rr']['metrics']['SCC']:.9f} ERGAS={evaluation['rr']['metrics']['ERGAS']:.9f} saved={bool(saved)}", flush=True)
                self.state['timing'][cfg['phase'] + '_update_seconds'] = float(np.quantile(timings[-1000:], .9))
                if trainer.updates in (8000, 14000, 20000, 50000):
                    probe = self.probes(trainer)
                    atomic_json(directory / ('diagnostics_%07d.json' % trainer.updates), probe)
                    self.state['timing']['probe_seconds'] = max(self.state['timing']['probe_seconds'], probe['elapsed_seconds'])
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
