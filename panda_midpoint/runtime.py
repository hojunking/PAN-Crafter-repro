"""Isolated s3 midpoint queue: existing local donor, no inherited wall deadline.

This module never starts a donor, changes another campaign's queue, kills a
process, or acquires an inter-server lock. A nonblocking local controller lock
only prevents duplicate ownership of this new output tree. GPU admission is
handled by the launcher.
"""
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


class BudgetExpired(RuntimeError):
    """Compatibility name for safe stop; there is no inherited hour budget."""


def plain(value):
    return json.loads(json.dumps(value))


def code_pin():
    from panda40.runtime import code_pin as shared
    files = list((plan.ROOT / 'panda_midpoint').glob('*.py'))
    files += list((plan.ROOT / 'panda30r').glob('*.py'))
    files += list((plan.ROOT / 'scripts').glob('run_panda_midpoint*.sh'))
    files += [plan.ROOT / 'align/__init__.py', plan.ROOT / 'align/resample.py']
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
        reasons.append('HQNR_GATE_ERGAS_SELECTOR')
    if best_h is None or (row['HQNR'],-row['ERGAS'],-row['completed_update']) > (
            best_h['HQNR'],-best_h['ERGAS'],-best_h['completed_update']):
        reasons.append('BEST_HQNR_BACKUP')
    return row,reasons




def discover_mature_donor(server, *, root=plan.ROOT):
    """Locate only the registered same-server M FINAL50K; missing means stop.

    No scalar ranking, donor fallback, training, cross-server import or network
    operation occurs. Metadata/file/tensor provenance is validated separately.
    """
    from panda30r.plan import CAMPAIGN_ID as source_campaign
    if server != 's3':
        raise ValueError('Only local s3 mature donor is authorized')
    base = Path(root).resolve() / 'work_dir/panda30r' / source_campaign / server / 'WV3'
    receipt = base / 'donors/M.json'
    if not receipt.is_file():
        raise FileNotFoundError('Existing local s3 mature M donor is required; no automatic training fallback: ' + str(receipt))
    row = json.loads(receipt.read_text())
    body = {k: v for k, v in row.items() if k != 'binding_sha256'}
    if (row.get('binding_sha256') != plan.canonical_sha256(body)
            or row.get('source_server') != server or row.get('mature') is not True
            or row.get('target_campaign_id') != source_campaign
            or row.get('version') != 'M' or row.get('completed_update') != 50000):
        raise ValueError('Existing M donor receipt identity/hash is invalid')
    cfg = row['source_config']
    if (cfg.get('server') != server or cfg.get('campaign_id') != source_campaign
            or cfg.get('engine_kind') != 'mature_teacher' or cfg.get('case') != 'MATURE'
            or cfg.get('phase') != 'donor' or cfg.get('start_update', 0) + cfg.get('planned_horizon', 0) != 50000):
        raise ValueError('Source is not same-server original frozen-A mature Teacher')
    run = cfg['run_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run) or type(cfg.get('attempt')) is not int or cfg['attempt'] < 1:
        raise ValueError('Invalid local donor run/attempt')
    directory = base / 'runs' / run / f"attempt_{cfg['attempt']:02d}"
    checkpoint = directory / 'models' / Path(row['checkpoint_path']).name
    evaluation = directory / f"evaluation_{cfg['planned_horizon']:07d}.json"
    replay = base / 'donors/M_replay_full40.json'
    if (not checkpoint.is_file() or file_sha(checkpoint) != row.get('checkpoint_sha256')
            or not replay.is_file() or file_sha(replay) != row.get('replay_sha256')):
        raise ValueError('Existing M checkpoint or original full40 replay SHA changed')
    state = json.loads((directory / 'run_state.json').read_text())
    if (state.get('status') != 'COMPLETED' or state.get('step') != cfg['planned_horizon']
            or state.get('run_id') != run):
        raise ValueError('Existing M donor has not completed its original horizon')
    proof = json.loads(evaluation.read_text())
    from panda40.stage1 import _check_evaluation
    expected = {key: cfg[key] for key in ('run_id', 'attempt', 'seed', 'dataset')}
    expected['completed_update'] = cfg['planned_horizon']
    _check_evaluation(proof, expected)
    if proof.get('metric_function_injection') is not False or proof.get('full_method_eligible') is not True:
        raise ValueError('Real official source-donor evaluation required')
    for split in ('rr', 'fr'):
        ids = [r['scene_id'] for r in proof[split]['per_scene']]
        if len(ids) != 20 or set(ids) != {f'WV3:{split}:{i}' for i in range(20)}:
            raise ValueError('Original M donor full20 scene inventory changed')
    return {**row, 'checkpoint_path': str(checkpoint.resolve()), 'evaluation_path': str(evaluation.resolve()),
            'source_receipt': str(receipt.resolve()), 'source_receipt_sha256': file_sha(receipt),
            'official_source_state_sha256': proof['state_before']['model_state_sha256']}


class Campaign:
    def __init__(self, server, *, output='work_dir/panda_midpoint', device='cuda',
                 replicas=plan.DEFAULT_REPLICAS):
        self.config = plain(plan.default_config(server, replicas))
        self.server, self.device = server, device
        self.root = Path(output).resolve() / plan.CAMPAIGN_ID / server / 'WV3'
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / 'campaign_state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'started_unix': time.time(), 'created_at': utc_now(), 'runs': [], 'groups': [],
            'timing': {}, 'donors': {}}
        self.stop_requested = False
        self.code, self.evaluator_pin = code_pin(), evaluator_identity()
        self.sources = self.registered = None

    def persist(self):
        self.state['elapsed_seconds'] = max(self.state.get('elapsed_seconds', 0.),
                                            time.time() - self.state['started_unix'])
        atomic_json(self.state_path, self.state)

    def check(self, **_):
        if self.stop_requested:
            raise BudgetExpired('STOP_REQUESTED')
        if shutil.disk_usage(self.root).free < 2 * 1024**3:
            raise BudgetExpired('DISK_CLOSEOUT_2GiB_RESERVE')

    def verify_code(self):
        if self.code != code_pin():
            raise RuntimeError('Campaign source changed; refusing mixed-code continuation')

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

    def prepare_donors(self):
        from .engine import load_mature_donor
        from panda_method.calibration import canonical_sha256, state_dict_sha256
        candidate = discover_mature_donor(self.server)
        donor = load_mature_donor(candidate['checkpoint_path'], candidate['checkpoint_sha256'],
                                 self.registered, device='cpu')
        try:
            actual = canonical_sha256({'A': state_dict_sha256(donor.aligner),
                                       'T': state_dict_sha256(donor.teacher)})
            if actual != candidate['official_source_state_sha256']:
                raise ValueError('Imported M tensors differ from source full40 evaluation')
            candidate['original_donor_binding'] = deepcopy(donor.binding)
        finally:
            del donor
            self.release()
        path = self.root / 'donors/M.json'
        if path.exists() and json.loads(path.read_text()) != plain(candidate):
            raise ValueError('Pinned local mature donor changed')
        atomic_json(path, candidate)
        self.state['donors']['M'] = candidate
        self.persist()

    def load_donor(self, label='M', *, include_teacher=True, device='cpu'):
        from .engine import load_mature_donor
        from .calibration import bind_donor
        if label != 'M':
            raise ValueError('Only the existing mature M donor is registered')
        row = self.state['donors'][label]
        return bind_donor(load_mature_donor(row['checkpoint_path'], row['checkpoint_sha256'],
            self.registered, include_teacher=include_teacher, device=device))

    def calibration(self, label='M'):
        from .calibration import calibrate
        from panda_stage2_condq.calibration import CalibrationStats
        self.check()
        path = self.root / 'calibration' / f'{label}_MIDPOINT_MASK2.npz'
        receipt_path = path.with_suffix('.json')
        prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        donor = self.load_donor(label)
        try:
            if path.exists():
                sha = file_sha(path)
                if prior is not None and sha != prior.get('file_sha256'):
                    raise ValueError('Immutable midpoint calibration SHA changed')
                stats = CalibrationStats.load(path, expected_binding=donor.binding,
                    expected_file_sha256=sha, expected_mask_policy=plan.LOSS_MASK_POLICY,
                    memory_budget_bytes=3 * 1024**3)
                if stats.calibration_kind != 'exact_condq' or stats.metadata['g_grouping']['k_requested'] != 5:
                    raise ValueError('Exact train-only midpoint K5 calibration required')
                saved = {**(prior or {}), 'path': str(path), 'sha256': sha, 'file_sha256': sha,
                    'artifact_sha256': stats.artifact_sha256, 'donor_binding': donor.binding,
                    'mask_policy': plan.LOSS_MASK_POLICY, 'tau_e': stats.tau_e,
                    'sample_count': stats.metadata['sample_count']}
                if prior and any(prior.get(k) != saved[k] for k in
                                 ('artifact_sha256', 'donor_binding', 'mask_policy')):
                    raise ValueError('Calibration receipt/binding mismatch')
                del stats
            else:
                if prior:
                    raise ValueError('Calibration receipt exists without complete file')
                donor.aligner.to(self.device)
                donor.teacher.to(self.device)
                def progress(info):
                    self.check()
                    if info['seen'] % 600 == 0 or info['seen'] == info['expected']:
                        print(f'[MIDPOINT CALIBRATION] {info}', flush=True)
                saved = calibrate(donor, self.registered, output=path, k=5, batch_size=12,
                    memory_budget_bytes=3 * 1024**3, include_condq=True,
                    mask_policy=plan.LOSS_MASK_POLICY, device=self.device, on_progress=progress)
                saved['sha256'] = saved['file_sha256']
            atomic_json(receipt_path, saved)
            return path, saved
        finally:
            del donor
            self.release()

    def build(self, cfg):
        from .engine import build_fresh_stage2
        from .cost import ensure_cost
        from panda_stage2_condq.calibration import CalibrationStats
        from panda_stage2_condq.stream import TrainStream, StreamSpec
        from donor30.artifacts import capture_rng, restore_rng
        calpath, receipt = self.calibration()
        donor = self.load_donor()
        stats = CalibrationStats.load(calpath, expected_binding=donor.binding,
            expected_file_sha256=receipt['sha256'], expected_mask_policy=plan.LOSS_MASK_POLICY)
        selected = self.state['donors']['M']
        cfg.update(donor_checkpoint=selected['checkpoint_path'],
            donor_checkpoint_sha256=selected['checkpoint_sha256'], donor_binding=deepcopy(donor.binding),
            donor_exploratory=selected['exploratory'], parent_run=selected['source_config']['run_id'],
            calibration_path=str(calpath), calibration_sha256=receipt['sha256'])
        trainer = build_fresh_stage2(cfg, donor, stats, evaluator_pin=self.evaluator_pin)
        rng = capture_rng()
        try:
            ensure_cost(cfg, self.sources, retained_trainer=trainer, device=self.device,
                        deadline_unix=None, cost_root=self.root)
        finally:
            restore_rng(rng)
        trainer.stage.to(self.device)
        trainer.stage.move_calibration(self.device, memory_budget_bytes=512 * 1024**2)
        stream_spec = StreamSpec(cfg['stream_seeds']['rr_data'], cfg['stream_seeds']['rr_augmentation'],
                                 **cfg['augmentation'])
        stream = TrainStream(self.registered, stream_spec,
            microbatch_size=trainer.training.microbatch_size,
            effective_batch_size=trainer.training.effective_batch_size, device=self.device)
        trainer.config = deepcopy(cfg)
        self.state['timing'].setdefault(self.timing_key(cfg), {'update_p90': 0., 'eval_seconds': 0.})
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
        return 'stage2_midpoint'

    def completed(self, cfg):
        return next((r for r in self.state['runs'] if r['run_id'] == cfg['run_id']
                     and r['status'] == 'COMPLETED'), None)

    def _save(self,trainer,stream,path,cfg,evaluation=None):
        if plain(trainer.config)!=plain(cfg) or getattr(trainer,'poisoned',False):
            raise ValueError('Only matching nonpoisoned complete trainer can be checkpointed')
        path=Path(path)
        from .engine import save_checkpoint
        saved=save_checkpoint(path,trainer,stream,code_pin=self.code,
            evaluator_pin=self.evaluator_pin,evaluation=evaluation)
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
        load_checkpoint(path,artifact['file_sha256'],trainer,stream,code_pin=self.code,evaluator_pin=self.evaluator_pin)
        return trainer,stream

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
            'best_selected':None,'best_hqnr':None,'saved':[],'policy':plan.POLICY.manifest()}
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
                    row.update(run_id=cfg['run_id'],attempt=cfg['attempt'],seed=cfg['seed'],
                        checkpoint_path=str(model_path),file_sha256=artifact['file_sha256'],
                        checkpoint_verified=True,metrics_complete=True,metrics_finite=True,
                        same_state_verified=True,state_hash=artifact['model_state_sha256'],
                        evaluation_path=str(evidence_path),reasons=reasons)
                    if not any(r['completed_update']==step for r in selection['saved']):
                        selection['saved'].append(row)
                    if 'HQNR_GATE_ERGAS_SELECTOR' in reasons: selection['best_selected']=row
                    if 'BEST_HQNR_BACKUP' in reasons: selection['best_hqnr']=row
                    atomic_json(selection_path,selection); atomic_json(latest,artifact)
                    emit_evaluation(self.root,cfg,evaluation,artifact=artifact,timestamp=evaluation['recorded_at'])
                    print(f"[{cfg['case']}/{cfg['replica']}] local={step} lifetime={cfg['start_update']+step} "
                        f"HQNR={row['HQNR']:.9f} SCC={row['SCC']:.9f} ERGAS={row['ERGAS']:.9f}",flush=True)
                    if times:
                        t=self.state['timing'][self.timing_key(cfg)]
                        t['update_p90']=max(t['update_p90'],float(np.quantile(times[-1000:],.9)))
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
                self.prepare_donors()
                for group in self.config['groups']:
                    for case, replica in group['runs']:
                        self.check()
                        cfg = plan.resolve_case(self.server, case, replica, replicas=self.config['replicas'])
                        self.run_case(cfg)
                self.state['status'] = 'FINISHED_REGISTERED_QUEUE'
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
