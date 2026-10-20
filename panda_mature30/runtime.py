"""Durable local30h mature-donor controller. No cross-server dependencies."""
from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
import time
import traceback

import numpy as np
import torch

from directshift48.reporting import atomic_json, append_jsonl, utc_now
from directshift48.runtime import environment_manifest
from panda_method.edge_data import EdgeDataSources
from panda_stage2_condq.sources import RegisteredRRTrain
from panda_stage2_condq.training import configure_precision
from panda30r.runtime import recover_trace
from ta2.evaluation import evaluator_identity, file_sha
from . import plan
from .contract import file_identity


class BudgetExpired(RuntimeError):
    """A durable campaign deadline or safe-stop request, never silent completion."""


def plain(value):
    return json.loads(json.dumps(value))


def code_pin():
    from panda40.runtime import code_pin as shared
    files = list((plan.ROOT / 'panda_mature30').glob('*.py'))
    files += list((plan.ROOT / 'panda_band30').glob('*.py'))
    files += list((plan.ROOT / 'panda_warp30').glob('*.py'))
    files += list((plan.ROOT / 'panda30r').glob('*.py'))
    files += list((plan.ROOT / 'scripts').glob('run_panda_mature30*.sh'))
    files += [plan.ROOT / 'align/__init__.py', plan.ROOT / 'align/resample.py']
    files += [plan.ROOT / 'ta2/registration.py']
    files += [plan.PLAN_PATH]
    files += [plan.ROOT / 'panda_midpoint/resampling.py', plan.ROOT / 'panda_midpoint/contract.py']
    return {**shared(), **{str(p.relative_to(plan.ROOT)): file_sha(p) for p in sorted(files)}}


def improvement(evaluation, selection):
    row = {'HQNR':evaluation['fr']['metrics']['HQNR'],
           'ERGAS':evaluation['rr']['metrics']['ERGAS'],
           'SCC':evaluation['rr']['metrics']['SCC'],
           'completed_update':evaluation['state_before']['completed_update']}
    plan.rank(row)
    best, best_h = selection.get('best_selected'), selection.get('best_hqnr')
    reasons=[]
    if best is None or plan.rank(row)>plan.rank(best):
        reasons.append('BEST_HQNR_THEN_ERGAS')
    if best_h is None or (row['HQNR'],-row['ERGAS'],-row['completed_update']) > (
            best_h['HQNR'],-best_h['ERGAS'],-best_h['completed_update']):
        reasons.append('BEST_HQNR_BACKUP')
    return row,reasons



class Campaign:
    def __init__(self, server, *, output='work_dir/panda_mature30', device='cuda',
                 clock=time.time):
        self.config = plain(plan.default_config(server))
        self.server, self.device = server, device
        self.root = Path(output).resolve() / plan.CAMPAIGN_ID / server / 'WV3'
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / 'campaign_state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'started_unix': clock(), 'created_at': utc_now(), 'runs': [], 'groups': [],
            'timing': {}, 'donors': {}}
        self.clock, self.closeout = clock, False
        if (type(self.state.get('started_unix')) not in (int, float)
                or not np.isfinite(self.state['started_unix'])
                or self.state['started_unix'] > clock()):
            raise ValueError('Invalid durable campaign start time')
        elapsed = self.state.get('elapsed_seconds', 0.)
        if type(elapsed) not in (int, float) or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid durable elapsed time')
        self._opened_monotonic = time.monotonic()
        self._elapsed_at_open = max(self.state.get('elapsed_seconds', 0.), clock()-self.state['started_unix'])
        self.stop_requested = False
        self.code, self.evaluator_pin = code_pin(), evaluator_identity()
        self.sources = self.registered = None

    def persist(self):
        self.state['elapsed_seconds'] = self.elapsed_seconds()
        atomic_json(self.state_path, self.state)

    def elapsed_seconds(self):
        return max(self.clock()-self.state['started_unix'],
                   self._elapsed_at_open + time.monotonic()-self._opened_monotonic,
                   self.state.get('elapsed_seconds', 0.))

    def check(self, **_):
        if self.stop_requested:
            raise BudgetExpired('STOP_REQUESTED')
        if shutil.disk_usage(self.root).free < 2 * 1024**3:
            raise BudgetExpired('DISK_CLOSEOUT_2GiB_RESERVE')
        elapsed = self.elapsed_seconds()
        limit = self.config['budget']['total_seconds']
        if not self.closeout:
            limit -= self.config['budget']['closeout_seconds']
        if elapsed >= limit:
            raise BudgetExpired('DURABLE_30H_DEADLINE')

    def verify_code(self):
        if self.code != code_pin():
            raise RuntimeError('Campaign source changed; refusing mixed-code continuation')

    @property
    def deadline_unix(self):
        return self.state['started_unix'] + self.config['budget']['total_seconds']

    def sync(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)

    def release(self):
        gc.collect()
        if str(self.device).startswith('cuda'):
            torch.cuda.empty_cache()

    def admission(self, catalog):
        from .evaluation import source_manifest
        self.check()
        configure_precision()
        if file_sha(catalog) != self.config['baseline_resolved']['source_registry_sha256']:
            raise ValueError('Registered WV3 data catalog SHA changed')
        self.sources = EdgeDataSources(catalog, root=plan.ROOT, sensor='WV3')
        for split in ('train', 'val', 'rr', 'fr'):
            self.sources.raw(split, [0])
        self.registered = RegisteredRRTrain(self.sources)
        pin = plain({'config': self.config, 'code': self.code, 'evaluator': self.evaluator_pin,
            'sources': source_manifest(self.sources), 'environment': environment_manifest(self.device)})
        path = self.root / 'admission.json'
        if path.exists() and json.loads(path.read_text()) != pin:
            raise ValueError('Admission/data/code/environment changed; use a new campaign root')
        atomic_json(path, pin)
        self.state['status'] = 'RUNNING'
        self.persist()


    def estimate_seconds(self, cfg):
        """Conservative local observation estimate; prior is explicitly labeled."""
        key = self.timing_key(cfg)
        timing = self.state['timing'].get(key, {})
        prior = self.config['time_prior_hours']['stage1' if cfg['phase']=='donor' else 'stage2'] * 3600
        estimate = prior
        if timing.get('updates_observed', 0) >= 100 and timing.get('eval_seconds', 0) > 0:
            estimate = (cfg['planned_horizon'] * timing['update_p90']
                        + len(cfg['evaluation_steps']) * timing['eval_seconds']
                        + len(cfg['evaluation_steps']) * timing.get('save_seconds', 0))
        if cfg['phase'] == 'stage2':
            estimate += self.config['budget']['calibration_reserve_seconds']
        return estimate * self.config['budget']['safety_factor']

    def admit_group(self, group):
        pending = [plan.resolve_case(self.server, c, r) for c,r in group['runs']]
        pending = [cfg for cfg in pending if not self.completed(cfg)]
        measured = all(self.state['timing'].get(self.timing_key(cfg),{}).get('updates_observed',0)>=100
                       and self.state['timing'].get(self.timing_key(cfg),{}).get('eval_seconds',0)>0
                       for cfg in pending)
        required = sum(self.estimate_seconds(cfg) for cfg in pending)
        remaining = (self.config['budget']['total_seconds']
                     - self.config['budget']['closeout_seconds'] - self.elapsed_seconds())
        receipt = {'group':group['name'], 'optional':group['optional'], 'at':utc_now(),
                   'required_seconds':required, 'remaining_seconds':remaining,
                   'measured_operator_timing':measured,
                   'admitted':required <= remaining and (measured or not group['optional']),
                   'pending':[c['run_id'] for c in pending]}
        self.state['groups'].append(receipt)
        self.persist()
        return receipt['admitted']

    def group_started(self, group):
        ids = {plan.resolve_case(self.server,c,r)['run_id'] for c,r in group['runs']}
        return any(r['run_id'] in ids for r in self.state['runs'])

    def register_donor(self, case):
        from .selection import select_checkpoints, validate_receipt
        from .donor import load_donor
        from .reporting import emit_evaluation
        self.check()
        cfg = plan.resolve_case(self.server, case)
        if self.server == 's1':
            directory = plan.ROOT / plan.LEGACY_SOURCE
        else:
            finished = self.completed(cfg)
            if finished is None:
                raise ValueError('Complete local Stage1 required before RRE selection')
            directory = Path(finished['directory'])
        path = self.root / 'donors' / 'rre_selection.json'
        if path.exists():
            receipt = validate_receipt(json.loads(path.read_text()))
        else:
            checkpoints = sorted((directory/'models').glob('step_*.pt'))
            def progress(info):
                self.check()
                print(f"[MATURE30 SELECT] {info}", flush=True)
            receipt = select_checkpoints(checkpoints, self.sources, server=self.server,
                output=path, device=self.device, on_progress=progress)
        selected = receipt['selected']
        donor = load_donor(selected['checkpoint_path'], selected['file_sha256'],
            self.registered, server=self.server, selector_receipt=receipt, device='cpu')
        donor.assert_frozen()
        row = {'label':case, 'source_config':selected['source_config'],
               'checkpoint_path':selected['checkpoint_path'],
               'checkpoint_sha256':selected['file_sha256'],
               'evaluation_path':selected['source_evaluation_path'],
               'selected_step':selected['completed_update'],
               'binding':deepcopy(donor.binding), 'source_server':self.server,
               'selector_receipt_path':str(path), 'selector_sha256':receipt['receipt_sha256'],
               'complete_source_horizon':50000, 'exploratory':True}
        if case in self.state['donors'] and self.state['donors'][case] != plain(row):
            raise ValueError('Immutable donor changed')
        pointer = self.root/'donors'/f'{case}_best_val_rre.json'
        if pointer.exists() and json.loads(pointer.read_text()) != plain(row):
            raise ValueError('Immutable RRE donor pointer changed')
        atomic_json(pointer, row)
        self.state['donors'][case] = row
        self.persist()
        evaluation = json.loads(Path(selected['source_evaluation_path']).read_text())
        artifact = json.loads(Path(selected['checkpoint_path']).with_suffix('.json').read_text())['artifact']
        emit_evaluation(self.root, cfg, evaluation, artifact=artifact,
                        timestamp=evaluation['recorded_at'], selector_receipt=receipt)
        del donor
        self.release()

    def load_donor(self, cfg, *, device='cpu'):
        from .donor import load_donor
        from .calibration import bind_donor
        from .spec import MatureSpec
        selected = self.state['donors'][cfg['donor_version']]
        receipt = json.loads(Path(selected['selector_receipt_path']).read_text())
        return bind_donor(load_donor(selected['checkpoint_path'], selected['checkpoint_sha256'],
            self.registered, server=self.server, selector_receipt=receipt, device=device),
            MatureSpec(**cfg['stage2_spec']))

    def calibration(self, cfg):
        from .calibration import calibrate_geometry, load_stats_bank
        from .spec import MatureSpec, geometry_manifest
        self.check()
        spec = MatureSpec(**cfg['stage2_spec'])
        donor = self.load_donor(cfg)
        artifacts = {}
        try:
            for geometry in spec.geometry_keys:
                self.check()
                key = plan.canonical_sha256({'geometry':geometry_manifest(geometry),
                    'binding':donor.binding})[:24]
                path = self.root/'calibration'/f"{cfg['donor_version']}_{key}.npz"
                receipt_path = path.with_suffix('.json')
                if path.exists():
                    if not receipt_path.is_file():
                        raise ValueError('Calibration has no complete receipt; inspect, do not overwrite')
                    saved = json.loads(receipt_path.read_text())
                    if saved.get('file_sha256') != file_sha(path):
                        raise ValueError('Immutable calibration file SHA changed')
                else:
                    if receipt_path.exists():
                        raise ValueError('Calibration receipt without file')
                    donor.aligner.to(self.device); donor.teacher.to(self.device)
                    def progress(info):
                        self.check()
                        if info['seen'] % 600 == 0 or info['seen'] == info['expected']:
                            print(f"[MATURE30 CALIBRATION {geometry}] {info}", flush=True)
                    saved = calibrate_geometry(donor, self.registered, spec=spec,
                        geometry_key=geometry, output=path, k=5, batch_size=12,
                        memory_budget_bytes=3*1024**3, device=self.device, on_progress=progress)
                    atomic_json(receipt_path, saved)
                artifacts[geometry] = {'path':str(path), 'file_sha256':saved['file_sha256']}
            stats = load_stats_bank(donor, spec, artifacts, memory_budget_bytes=3*1024**3)
            return artifacts, stats
        finally:
            del donor
            self.release()

    def build(self, cfg):
        if cfg['phase']=='donor':
            from .stage1 import build_stage1, Stage1Stream
            trainer = build_stage1(cfg, self.sources, device='cpu')
        else:
            from .engine import build_fresh_stage2
            artifacts, stats_bank = self.calibration(cfg)
            donor = self.load_donor(cfg)
            selected = self.state['donors'][cfg['donor_version']]
            cfg.update(donor_checkpoint=selected['checkpoint_path'],
                donor_checkpoint_sha256=selected['checkpoint_sha256'],
                donor_binding=deepcopy(donor.binding), donor_exploratory=True,
                parent_run=selected['source_config']['run_id'],
                selector_receipt_sha256=selected['selector_sha256'],
                calibration_artifacts=artifacts)
            trainer = build_fresh_stage2(cfg, donor, stats_bank, evaluator_pin=self.evaluator_pin)
        # No fabricated cost: native-network cost is optional and left blank until
        # separately measured; timing admission includes all actual evaluations.
        trainer.stage.to(self.device)
        if cfg['phase']=='donor':
            stream = Stage1Stream(self.sources, cfg, device=self.device)
        else:
            from panda_stage2_condq.stream import TrainStream, StreamSpec
            trainer.stage.move_calibration(self.device, memory_budget_bytes=512*1024**2)
            stream = TrainStream(self.registered,
                StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],
                           **cfg['augmentation']),
                microbatch_size=trainer.training.microbatch_size,
                effective_batch_size=trainer.training.effective_batch_size, device=self.device)
        trainer.config = deepcopy(cfg)
        self.state['timing'].setdefault(self.timing_key(cfg),
            {'update_p90':0., 'eval_seconds':0., 'save_seconds':0., 'updates_observed':0})
        return trainer, stream

    def evaluate(self, trainer):
        from .evaluation import official_evaluate
        return official_evaluate(trainer, self.sources, evaluator_pin=self.evaluator_pin,
                                 device=self.device, on_scene=lambda *a, **kw: self.check())

    def trace(self, trainer, stream):
        from .engine import advance
        tick = time.monotonic()
        result = advance(trainer, stream)
        self.sync()
        return result, time.monotonic() - tick

    @staticmethod
    def timing_key(cfg):
        # Full-strength PAN measurements conservatively cover .5/.25/0 variants;
        # changing only displacement magnitude does not add an operator.
        return ':'.join(str(cfg[k]) for k in ('engine_kind','upsampler','frame','aligner_outputs'))

    def completed(self, cfg):
        return next((r for r in self.state['runs'] if r['run_id'] == cfg['run_id']
                     and r['status'] == 'COMPLETED'), None)

    def pending_skip(self, group):
        path = self.root/'pending_control.json'
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        allowed = {g['name'] for g in self.config['groups']}
        if (set(value) != {'request_id','reason','skip_groups'}
                or not value['request_id'] or not value['reason']
                or not isinstance(value['skip_groups'], list)
                or not set(value['skip_groups']) <= allowed):
            raise ValueError('Only explicit pending-group skips are permitted')
        skip = group['name'] in value['skip_groups']
        if skip and self.group_started(group):
            raise ValueError('Cannot skip started group; request an explicit checkpoint-safe stop')
        if skip:
            self.state.setdefault('pending_amendments', []).append(
                {'at':utc_now(), 'group':group['name'], **value})
            self.persist()
        return skip

    def _save(self,trainer,stream,path,cfg,evaluation=None):
        if plain(trainer.config)!=plain(cfg) or getattr(trainer,'poisoned',False):
            raise ValueError('Only matching nonpoisoned complete trainer can be checkpointed')
        path=Path(path)
        from .engine import save_checkpoint
        tick=time.monotonic()
        saved=save_checkpoint(path,trainer,stream,code_pin=self.code,
            evaluator_pin=self.evaluator_pin,evaluation=evaluation)
        timing=self.state['timing'].setdefault(self.timing_key(cfg),{})
        timing['save_seconds']=max(timing.get('save_seconds',0.),time.monotonic()-tick)
        from .evaluation import model_state_hash
        saved.update(checkpoint_path=str(path.resolve()),file_sha256=file_sha(path),
            state_hash=model_state_hash(trainer.stage),weights_saved=True,
            model_state_sha256=model_state_hash(trainer.stage),checkpoint_verified=True,
            identity=deepcopy(evaluation['state_before']) if evaluation else {},
            completed_update=trainer.updates,campaign_resume_complete=True)
        atomic_json(path.with_suffix('.json'),{'artifact':saved,'config':cfg,'code_pin':self.code,
            'evaluator_pin':self.evaluator_pin,'evaluation':evaluation})
        return saved

    def _restore(self,artifact,trainer,stream,cfg):
        path=Path(artifact['checkpoint_path'])
        from .engine import load_checkpoint
        return load_checkpoint(path,artifact['file_sha256'],trainer,stream,code_pin=self.code,evaluator_pin=self.evaluator_pin)

    def resume_artifact(self,directory,cfg):
        """Recover an atomically completed checkpoint even before pointer write.

        Only this run's owned models directory is scanned. Its highest complete
        file must pass the full config/code/evaluator/state loader; a corrupt
        latest artifact is never silently skipped for an earlier one.
        """
        directory=Path(directory); models=directory/'models'
        candidates=[]
        for path in models.glob('*.pt'):
            match=re.fullmatch(r'(?:step|stop)_(\d{7})(?:_\d+)?\.pt',path.name)
            if match:
                if path.is_symlink(): raise ValueError('Linked resume checkpoint prohibited')
                candidates.append((int(match.group(1)),path.stat().st_mtime_ns,path))
        latest=models/'latest_resume.json'
        if not candidates:
            if latest.exists(): raise ValueError('Resume pointer without checkpoint')
            return None
        step,_,path=max(candidates)
        if not 0<step<=cfg['planned_horizon']: raise ValueError('Resume step outside registered horizon')
        sidecar=path.with_suffix('.json')
        if sidecar.exists():
            value=json.loads(sidecar.read_text())
            if (plain(value['config'])!=plain(cfg) or value['code_pin']!=self.code
                    or value['evaluator_pin']!=self.evaluator_pin):
                raise ValueError('Resume sidecar provenance changed')
            artifact=value['artifact']
            if file_sha(path)!=artifact['file_sha256'] or artifact['completed_update']!=step:
                raise ValueError('Resume sidecar file/update mismatch')
            if Path(artifact['checkpoint_path']).resolve()!=path.resolve():
                raise ValueError('Resume sidecar points outside its own file')
            return artifact
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if (plain(payload.get('config'))!=plain(cfg) or payload.get('pins')!=
                {'code':self.code,'evaluator':self.evaluator_pin}
                or payload.get('trainer',{}).get('updates')!=step):
            raise ValueError('Orphan complete checkpoint provenance changed')
        return {'checkpoint_path':str(path.resolve()),'file_sha256':file_sha(path),
                'completed_update':step,'recovered_before_sidecar':True}

    def recover_evaluated_checkpoint(self,path,trainer,cfg):
        """Finish evidence publication after atomic checkpoint / metadata crash."""
        from .evaluation import model_state_hash
        from panda40.stage1 import _check_evaluation
        path=Path(path)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        sha=file_sha(path); state=model_state_hash(trainer.stage)
        evaluation=payload.get('evaluation')
        if (plain(payload.get('config'))!=plain(cfg) or payload.get('pins')!=
                {'code':self.code,'evaluator':self.evaluator_pin}
                or payload.get('model_state_sha256',payload.get('state_hash'))!=state
                or not evaluation or not evaluation.get('recorded_at')):
            raise ValueError('Incomplete checkpoint lacks its original same-state official evaluation')
        ident={key:cfg[key] for key in ('run_id','attempt','seed','dataset')}
        ident.update(completed_update=trainer.updates,model_state_sha256=state)
        _check_evaluation(evaluation,ident)
        sidecar=path.with_suffix('.json')
        if sidecar.exists():
            value=json.loads(sidecar.read_text()); artifact=value['artifact']
            if value.get('evaluation')!=evaluation or artifact['file_sha256']!=sha:
                raise ValueError('Recovered sidecar/evaluation/file differs')
        else:
            artifact={'checkpoint_path':str(path.resolve()),'file_sha256':sha,
                'state_hash':state,'model_state_sha256':state,'completed_update':trainer.updates,
                'weights_saved':True,'checkpoint_verified':True,'identity':evaluation['state_before'],
                'campaign_resume_complete':True,'resume_only':False,
                'verification':'RECOVERED_COMPLETE_PINNED_STATE_AND_EMBEDDED_EVALUATION'}
            atomic_json(sidecar,{'artifact':artifact,'config':cfg,'code_pin':self.code,
                'evaluator_pin':self.evaluator_pin,'evaluation':evaluation})
        return artifact,evaluation

    def run_case(self,base):
        from .reporting import emit_status,emit_evaluation
        if self.completed(base): return
        requested=plain(base)
        previous=[r for r in self.state['runs'] if r['run_id']==base['run_id']]
        if len(previous)>1:
            raise ValueError('One resumable attempt per registered run required')
        if previous:
            status=previous[0]; directory=Path(status['directory'])
            if status.get('poisoned'):
                raise ValueError('Partial optimizer failure requires explicit investigation, not silent restart')
            if json.loads((directory/'requested_config.json').read_text())!=requested:
                raise ValueError('Run request changed on resume')
        else:
            directory=self.root/'runs'/base['run_id']/'attempt_01'
            directory.mkdir(parents=True,exist_ok=False)
            atomic_json(directory/'requested_config.json',requested)
            status={'run_id':base['run_id'],'case':base['case'],'replica':base['replica'],
                'attempt':1,'directory':str(directory),'phase':base['phase'],'step':0,'started_at':utc_now()}
            self.state['runs'].append(status)
        status['status']='RUNNING'; cfg=deepcopy(requested); trainer=stream=None
        selection_path=directory/'models'/'selection.json'
        selection=json.loads(selection_path.read_text()) if selection_path.exists() else {
            'best_selected':None,'best_hqnr':None,'saved':[],
            'policy':deepcopy(cfg['checkpoint_selection']),
            'purpose':('RETAIN_ALL_AND_MONITOR_HQNR; ACTUAL_BEST_IN_DONORS_RRE_RECEIPT'
                       if cfg['phase']=='donor' else 'STAGE2_BEST_HQNR_THEN_ERGAS')}
        latest=directory/'models'/'latest_resume.json'
        def persist():
            atomic_json(directory/'run_state.json',status); self.persist()
        persist()
        try:
            trainer,stream=self.build(cfg)
            config_path=directory/'config.json'
            if config_path.exists() and json.loads(config_path.read_text())!=plain(cfg):
                raise ValueError('Resolved donor/calibration/parent config differs on resume')
            atomic_json(config_path,cfg)
            saved=self.resume_artifact(directory,cfg)
            if saved is not None:
                trainer,stream=self._restore(saved,trainer,stream,cfg)
                status['resumed_from_step']=trainer.updates
            status['step']=trainer.updates
            recover_trace(directory,trainer.updates)
            emit_status(self.root,cfg,'RUNNING',step=trainer.updates)
            times=[]
            # A stopped checkpoint at an evaluation boundary can be republished
            # on resume without advancing or repeating its optimizer update.
            while True:
                if trainer.updates in cfg['evaluation_steps']:
                    step=trainer.updates; evidence_path=directory/f'evaluation_{step:07d}.json'
                    model_path=directory/'models'/f'step_{step:07d}.pt'
                    if evidence_path.exists() and model_path.with_suffix('.json').exists():
                        evaluation=json.loads(evidence_path.read_text())
                        artifact=json.loads(model_path.with_suffix('.json').read_text())['artifact']
                        from .evaluation import model_state_hash
                        if (artifact['model_state_sha256']!=model_state_hash(trainer.stage)
                                or file_sha(model_path)!=artifact['file_sha256']):
                            raise ValueError('Resumed official evaluated state differs from retained weights')
                    else:
                        if model_path.exists():
                            artifact,evaluation=self.recover_evaluated_checkpoint(model_path,trainer,cfg)
                        else:
                            evaluation=self.evaluate(trainer)
                            evaluation['recorded_at']=utc_now()
                            artifact=self._save(trainer,stream,model_path,cfg,evaluation)
                        atomic_json(evidence_path,evaluation)
                    row,reasons=improvement(evaluation,selection)
                    if cfg['phase'] == 'donor':
                        # This local inventory must never label an HQNR peak as
                        # the Stage1 best. The actual choice is made after H50
                        # by register_donor() from all retained candidates.
                        reasons = [r for r in reasons if r == 'BEST_HQNR_BACKUP']
                    row.update(run_id=cfg['run_id'],attempt=cfg['attempt'],seed=cfg['seed'],
                        checkpoint_path=str(model_path),file_sha256=artifact['file_sha256'],
                        checkpoint_verified=True,metrics_complete=True,metrics_finite=True,
                        same_state_verified=True,state_hash=artifact['model_state_sha256'],
                        evaluation_path=str(evidence_path),reasons=reasons)
                    if not any(r['completed_update']==step for r in selection['saved']):
                        selection['saved'].append(row)
                    if 'BEST_HQNR_THEN_ERGAS' in reasons: selection['best_selected']=row
                    if 'BEST_HQNR_BACKUP' in reasons: selection['best_hqnr']=row
                    atomic_json(selection_path,selection); atomic_json(latest,artifact)
                    if cfg['phase'] == 'stage2':
                        emit_evaluation(self.root,cfg,evaluation,artifact=artifact,timestamp=evaluation['recorded_at'])
                    print(f"[{cfg['case']}/{cfg['replica']}] local={step} lifetime={cfg['start_update']+step} "
                        f"HQNR={row['HQNR']:.9f} SCC={row['SCC']:.9f} ERGAS={row['ERGAS']:.9f}",flush=True)
                    if times:
                        t=self.state['timing'][self.timing_key(cfg)]
                        t['update_p90']=max(t['update_p90'],float(np.quantile(times[-1000:],.9)))
                        t['updates_observed']=max(t['updates_observed'],trainer.updates)
                        t['eval_seconds']=max(t['eval_seconds'],evaluation['elapsed_seconds'])
                    persist()
                if trainer.updates>=cfg['planned_horizon']: break
                self.check()
                if trainer.updates%100==0: self.verify_code()
                if shutil.disk_usage(self.root).free<2*1024**3:
                    raise BudgetExpired('DISK_CLOSEOUT_2GiB_RESERVE')
                trace,seconds=self.trace(trainer,stream); times.append(seconds)
                status['step']=trainer.updates
                append_jsonl(directory/'train.jsonl',{'at':utc_now(),'step':trainer.updates,
                    'local_update':trainer.updates,'lifetime_update':cfg['start_update']+trainer.updates,
                    'seconds_including_IO':seconds,**({'trace':trace} if trainer.updates%100==0 else {})})
            status.update(status='COMPLETED',completed_at=utc_now(),selection=selection)
            emit_status(self.root,cfg,'COMPLETED',step=trainer.updates)
        except BaseException as exc:
            status.update(status='STOPPED' if isinstance(exc,BudgetExpired) else 'FAILED',error=repr(exc),
                poisoned=bool(trainer is not None and getattr(trainer,'poisoned',False)))
            if trainer is not None and trainer.updates and not getattr(trainer,'poisoned',False):
                try:
                    stop=directory/'models'/f'stop_{trainer.updates:07d}_{time.time_ns()}.pt'
                    saved=self._save(trainer,stream,stop,cfg)
                    atomic_json(latest,saved); status['step']=trainer.updates
                except Exception as save_error:
                    status['stop_checkpoint_error']=repr(save_error)
            try: emit_status(self.root,cfg,status['status'],step=status['step'],extra={'error':repr(exc)})
            except Exception as report_error: status['report_error']=repr(report_error)
            raise
        finally:
            persist(); del trainer,stream; self.release()


    def run(self, *, catalog=plan.ROOT / 'ablr2/sensor_sources.json'):
        with (self.root / '.controller.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old = {s: signal.signal(s, lambda *a: setattr(self, 'stop_requested', True))
                   for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                self.persist()
                self.admission(catalog)
                if self.server == 's1':
                    self.register_donor(plan.DONORS[0])
                for group in self.config['groups']:
                    if self.pending_skip(group):
                        continue
                    needs_donor = any(case not in plan.DONORS for case, _ in group['runs'])
                    if needs_donor and plan.DONORS[0] not in self.state['donors']:
                        self.state['groups'].append({
                            'group': group['name'], 'admitted': False,
                            'reason': 'MISSING_COMPLETED_LOCAL_RRE_SELECTED_DONOR', 'at': utc_now()})
                        self.persist()
                        continue
                    # A previously admitted group resumes within the same
                    # deadline; do not charge its full50K a second time.
                    if not self.group_started(group) and not self.admit_group(group):
                        continue
                    for case, replica in group['runs']:
                        self.check()
                        cfg = plan.resolve_case(self.server, case, replica)
                        self.run_case(cfg)
                        if cfg['phase'] == 'donor':
                            self.register_donor(case)

                skipped = [g for g in self.state['groups'] if not g['admitted']]
                self.state['status'] = ('FINISHED_WITH_BUDGET_SKIPS' if skipped else
                    'FINISHED_WITH_EXPLICIT_SKIPS' if self.state.get('pending_amendments') else
                    'FINISHED_REGISTERED_QUEUE')
            except BudgetExpired as exc:
                self.state.update(status='STOPPED_SAFE', reason=str(exc))
            except Exception as exc:
                self.state.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
                raise
            finally:
                self.persist()
                for sig, handler in old.items():
                    signal.signal(sig, handler)
            return self.state

