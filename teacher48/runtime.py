"""Independent per-server campaign; metadata first, no failed-model recovery files."""
from __future__ import annotations

from dataclasses import asdict
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from collections import deque

import numpy as np
import torch

from panda_method.geometry import SceneAwarePANAligner
from panda_method.losses import AbsoluteGradientLoss, SpatialLossSpec
from panda_method.reconstruction import ReconstructionSpec, build_reconstructor
from panda_method.stages import Stage1, TrainingSources
from ta2.evaluation import atomic_json, digest, evaluator_identity, file_sha
from .data import Sources, TrainingStream
from .evaluation import official_evaluation, retain_qualified, state_hash
from .plan import (CAMPAIGN_ID, PARENT_CAMPAIGN_ID, METHOD_CONTRACT, METHOD_FILE_NAME,
                   METHOD_SHA256, HEAD_CONDITIONING, LANES, LOSS_ORDER, OPTIMIZER_ORDER, OPTIMIZER_CASES,
                   admit_new_run, estimate_remaining_seconds, gap_rank, pareto_front,
                   representative, rr_reserve, resolve_case, should_prune, validate_baseline, validate_method_config)
from .reporting import LocalOutbox, build_record, validate_record
from .training import Trainer, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]
METHOD_FILE = ROOT/'research_log'/METHOD_FILE_NAME
METHOD_SHA = METHOD_SHA256


def append_jsonl(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())


def training_config(config):
    validate_method_config(config)
    b = config['baseline_resolved']
    return TrainingConfig.from_mapping(dict(b, seed=config['seed'], horizon=config['planned_horizon'],
        lambda_abs=config['lambda_abs'], lambda_eps=config['lambda_epsilon'],
        lr_A=config['lr_A'], lr_U=config['lr_U'], weight_decay=config['weight_decay'],
        warmup_updates=b['warmup_updates0'], min_lr_A=b['min_lr'], min_lr_U=b['min_lr'],
        scheduler=b['scheduler_family']))


def build_trainer(config, sources, device):
    validate_method_config(config)
    b, seeds = config['baseline_resolved'], config['stream_seeds']
    architecture = b['teacher_architecture']
    if set(architecture) != {'bands', 'width', 'depth'} or architecture['bands'] != sources.bands:
        raise ValueError('Pinned existing U-Net architecture requires exactly bands/width/depth')
    if b['aligner_architecture']['name'] != 'SceneAwarePANAligner':
        raise ValueError('Legacy aligner cannot replace MethodPlan-v1.3')
    if SceneAwarePANAligner.HEAD_CONDITIONING != HEAD_CONDITIONING:
        raise ValueError('Aligner head semantics differ from the pinned method')
    spec = ReconstructionSpec(**architecture, input_layout=b['teacher_input_layout'], base_reference=b['teacher_base'])
    if sources.manifest['uses_lpan'] != spec.uses_lpan:
        raise ValueError('LPAN data and input-layout contract differ')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seeds['A_init'])
        aligner = SceneAwarePANAligner(sources.bands, view_margin=b['aligner_architecture']['view_margin'])
    teacher = build_reconstructor(spec, seed=seeds['T_init'], role='T')
    spatial = dict(b['gradient_kernel_roi_normalization'])
    spatial['rgb_indices'] = tuple(spatial['rgb_indices'])
    if spatial['sensor'] != sources.dataset or spatial['radiometry'] != 'normalized_minus1_plus1':
        raise ValueError('Spatial-loss sensor/units do not match actual input normalization')
    stage = Stage1(aligner, teacher, AbsoluteGradientLoss(SpatialLossSpec(**spatial)),
        sources=TrainingSources(sources.group_hashes['rr_train'], sources.group_hashes['fr_train'], sources.mapping_sha256),
        epsilon_seed=seeds['rr_epsilon'], boundary_policy=b['boundary_policy'],
        calibration_protocol='STAGE1_ONLY_NO_CALIBRATION').to(device)
    return Trainer(stage, training_config(config))


def streams(config, sources, device):
    b, seeds = config['baseline_resolved'], config['stream_seeds']
    return {s: TrainingStream(sources, s+'_train', seed=seeds[s+'_data'],
        augmentation_seed=seeds[s+'_augmentation'], microbatch=b['microbatch_'+s.upper()],
        accumulation=b['accumulation_'+s.upper()], augmentation=b['augmentation'], device=device)
        for s in ('rr', 'fr')}


def validate_runtime_baseline(b, server):
    validate_baseline(b)
    if file_sha(METHOD_FILE) != METHOD_SHA:
        raise ValueError('Method specification changed since campaign plan')
    if b['precision'] != 'fp32' or b['RR_train_HW'] != [64, 64]:
        raise ValueError('Pin the supported FP32, PAN64 RR training recipe')
    for key in ('boundary_policy', 'qual_export', 'probes', 'artifact_reserve_seconds','probe_reserve_seconds','scheduler_indexing'):
        if key not in b:
            raise ValueError('Runtime baseline must pin '+key)
    if b['boundary_policy'] not in ('report', 'require_spatial_support'):
        raise ValueError('Explicit fixed-ROI boundary policy required')
    export = b['qual_export']
    bands, peak = {'WV3': (8,2047.), 'QB': (4,2047.), 'GF2': (4,1023.)}[LANES[server]['dataset']]
    if (export['display_range_DN'] != [0, peak] or export['roi'] != 'FULL_SCENE'
            or len(set(export['rgb_indices'])) != 3 or any(not 0 <= i < bands for i in export['rgb_indices'])
            or export['edge_gain'] <= 0 or export['error_gain'] <= 0):
        raise ValueError('Pin full-scene RGB display and nonadaptive edge/error gains')
    if any(not isinstance(b[k],(int,float)) or not np.isfinite(b[k]) or b[k]<=0
           for k in ('artifact_reserve_seconds','probe_reserve_seconds')):
        raise ValueError('Positive, justified artifact/export time allowance required')
    if b['scheduler_indexing'] != 'ONE_BASED_WARMUP_LINEAR_PEAK_THEN_COSINE_EXACT_ENDPOINT_MIN':
        raise ValueError('Explicitly acknowledge the pinned update-index LR convention')
    training_config(resolve_case(server, baseline=b))


def source_code_pin():
    paths = sorted([*ROOT.glob('teacher48/*.py'), *ROOT.glob('panda_method/*.py'),
                    ROOT/'pa/warp.py', ROOT/'pa/offset.py', ROOT/'model/pancrafter_paper.py',
                    ROOT/'fh12/model.py'])
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


class Campaign:
    def __init__(self, output_root, server, baseline, *, device='cuda'):
        if server not in LANES:
            raise ValueError('Only s1/s2/s3 are in this campaign')
        self.server, self.dataset, self.baseline, self.device = server, LANES[server]['dataset'], baseline, device
        self.root = Path(output_root)/CAMPAIGN_ID/server/self.dataset
        self.root.mkdir(parents=True, exist_ok=True)
        self.outbox = LocalOutbox(self.root)
        self.runs, self.observations, self.times = [], [], []
        self.progress,self.artifacts=[],[]
        self.costs={}
        self.update_latencies=deque(maxlen=4096)
        self.evaluator_pin, self.sources = None, None
        self.code_pin = source_code_pin()
        self.provenance = {'config_sha256': digest(baseline), 'dataset_manifest_sha256': None,
                           'evaluator_revision': baseline.get('evaluator_revision')}
        self.provenance['code_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        existing=list(self.outbox.outbox.glob('T48-*.json'))
        self._sequence=len(existing)
        for path in existing:
            record=json.loads(path.read_text())
            validate_record(record)
            selector=record['identity']['selector']
            if selector.startswith('STATUS_') and selector.rsplit('_',1)[-1].isdigit():
                self._sequence=max(self._sequence,int(selector.rsplit('_',1)[-1]))

    def emit(self, config, status, *, step=0, selector=None, scope='STATUS', metrics=None, sha=None, extra=None):
        if config:
            validate_method_config(config)
        self._sequence += 1
        case = config.get('case', 'ADMISSION')
        selector = selector or f'STATUS_{status}_{self._sequence:06d}'
        fields = {'Status': status, 'Repeat': int(config['replica'][1:]) if config.get('replica') else None, 'Seed': config.get('seed'),
            'Attempt': config.get('attempt'), 'Updates': step, 'Lifetime_updates': step,
            'Selected_step': step, 'U_peak_lr': config.get('lr_U'), 'A_peak_lr': config.get('lr_A'),
            'Model / inputs': json.dumps({'architecture': self.baseline.get('teacher_architecture'),
                'layout': self.baseline.get('teacher_input_layout'), 'base': self.baseline.get('teacher_base')}),
            'JQM_variant': 'SRF-substitute NNLS-normalized MTF41 phase2; not SIPSA-equivalent' if metrics else None,
            'Sort_case': int(config['loss_case'][1:]) if config.get('loss_case') else None}
        costs=self.costs.get((config.get('run_id'),config.get('attempt')))
        if costs:
            fields.update({'Params(M)':costs['parameters']/1e6,'Train(h)':costs['train_seconds']/3600,
                           'Eval(h)':costs['eval_seconds']/3600,'Wall(h)':(time.monotonic()-costs['start'])/3600,
                           'Cost_scope':'A+T params; observed train/data/eval wall; inference/FLOPs unmeasured'})
        record = build_record(server=self.server, dataset=self.dataset,
            run_id=config.get('run_id', f'{CAMPAIGN_ID}_{self.server}_ADMISSION'), case_id=case,
            selector=selector, eval_scope=scope, eval_protocol_id=digest(self.evaluator_pin or {'status_only':True}),
            checkpoint_sha=sha, metrics=metrics, fields=fields,
            extra={'lambda_abs': config.get('lambda_abs'), 'lambda_eps_RR': config.get('lambda_eps_RR'),
                   'lambda_eps_FR': config.get('lambda_eps_FR'), 'retention_policy': 'STRICT_TARGET_ONLY',
                   **(extra or {})}, provenance={**self.provenance, 'config_sha256': digest(config)})
        self.outbox.enqueue(record)
        if scope in ('STATUS','ARTIFACT'):
            self.progress.append({'run_id':record['identity']['run'],'case':case,'status':status,'step':step,
                                  'attempt':config.get('attempt'),'record_id':record['record_id'],'extra':extra})
        return record['record_id']

    def admission(self, sources_path):
        try:
            validate_runtime_baseline(self.baseline, self.server)
            self.evaluator_pin = evaluator_identity()
            if self.baseline['evaluator_revision'] != digest(self.evaluator_pin):
                raise ValueError('baseline evaluator_revision must equal the actual evaluator identity SHA256')
            self.sources = Sources(sources_path, dataset=self.dataset, baseline=self.baseline)
            self.provenance['dataset_manifest_sha256'] = self.sources.sha256
            trainer = build_trainer(resolve_case(self.server, baseline=self.baseline), self.sources, self.device)
            from .diagnostics import validate_probe_spec
            validate_probe_spec(self.baseline['probes'], self.sources)
            stream = streams(resolve_case(self.server, baseline=self.baseline), self.sources, self.device)
            # Real source/ROI/layout validation, no optimizer update or weights saved.
            with trainer.evaluation_context(gradients=True):
                for branch in ('rr', 'fr'):
                    trainer.objective.evaluate(stream[branch]()[0], branch)
            report = {'status': 'ADMITTED', 'source_manifest_sha256': self.sources.sha256,
                'prior_mapping_sha256': self.sources.mapping_sha256, 'evaluator': self.evaluator_pin,
                'code_files': self.code_pin, **METHOD_CONTRACT,
                'parent_campaign_id': PARENT_CAMPAIGN_ID,
                'baseline_sha256': digest(self.baseline), 'test_aware': True}
            atomic_json(self.root/'baseline_resolved.json', self.baseline, immutable=True)
            atomic_json(self.root/'admission/report.json', report, immutable=True)
            self.emit({}, 'ADMITTED', extra=report)
            return report
        except Exception as exc:
            report = {'status': 'BLOCKED_DATA_PROTOCOL', 'error': f'{type(exc).__name__}: {exc}'}
            self.emit({}, 'BLOCKED_DATA_PROTOCOL', extra=report)
            append_jsonl(self.root/'admission/failures.jsonl', report)
            raise

    def elapsed(self):
        return time.time()-self.started_at

    def _estimate(self, horizon):
        from .plan import evaluation_steps
        return estimate_remaining_seconds(remaining_updates=horizon, update_p90_seconds=self.timing['update_p90_seconds'],
            remaining_evaluations=len(evaluation_steps(horizon)), rr_fr_eval_seconds=self.timing['eval_seconds'],
            artifact_seconds=self.timing.get('artifact_seconds',self.baseline['artifact_reserve_seconds'])*len(evaluation_steps(horizon))
                + self.timing.get('probe_seconds',self.baseline['probe_reserve_seconds'])*(3+(horizon>=100000)+(horizon>=150000)))

    def _evaluate(self, trainer, config, *, selector, root):
        def on_scene(row):
            append_jsonl(root/'evaluation_progress.jsonl',row)
            self.emit(config,'SCENE_MEASURED',step=trainer.updates,selector=selector,
                scope=row['branch'].upper()+'_SCENE:'+row['scene_id'],metrics=row['metrics'],
                sha=row['identity']['model_state_sha256'],extra={k:v for k,v in row.items() if k != 'metrics'})
        try:
            evaluation, raw = official_evaluation(trainer, config, self.sources, device=self.device,
                evaluator_revision=self.baseline['evaluator_revision'], evaluator_pin=self.evaluator_pin,
                deadline_unix=getattr(self,'started_at',time.time())+48*3600,
                on_scene=on_scene)
        except Exception as exc:
            failure={'selector':selector,'completed_update':trainer.updates,'metrics_complete':False,
                     'error':f'{type(exc).__name__}: {exc}','model_state_sha256':state_hash(trainer.stage)}
            append_jsonl(root/'evaluations.jsonl',failure)
            self.emit(config,'EVALUATION_FAILED',step=trainer.updates,selector=selector,scope='EVALUATION_FAILURE',
                sha=failure['model_state_sha256'],extra=failure)
            raise
        # Journal precedes every artifact write and every network operation.
        append_jsonl(root/'evaluations.jsonl', evaluation)
        for branch in ('rr', 'fr'):
            for row in evaluation[branch]['per_scene']:
                append_jsonl(root/'evaluations_per_scene.jsonl', dict(branch=branch,
                    identity=evaluation['state_after'], **row))
        ident = evaluation['state_after']; gate = evaluation['qualification']
        key=(config['run_id'],config['attempt'])
        if key in self.costs:
            self.costs[key]['eval_seconds']+=evaluation['elapsed_seconds']
        metrics = {**evaluation['rr']['metrics'], **evaluation['fr']['metrics']}
        self.emit(config, 'TARGET_PASS' if gate['joint_pass'] else 'TARGET_MISS', step=trainer.updates,
            selector=selector, scope='RR_FR_COMPLETE20', metrics=metrics, sha=ident['model_state_sha256'],
            extra={**gate, 'same_state_verified': gate['same_state_verified'], 'evaluation_identity': ident,
                   'counts': {'RR':len(evaluation['rr']['per_scene']), 'FR':len(evaluation['fr']['per_scene'])},
                   'evaluation_seconds': evaluation['elapsed_seconds'], 'configuration': config})
        flat = {**metrics, **gate, **ident, 'schedule':config['schedule'], 'loss_case':config['loss_case'],
                'optimizer_case':config['optimizer_case'], 'replica':config['replica'], 'seed':config['seed']}
        return evaluation, raw, flat

    def timing_probe(self):
        config = resolve_case(self.server, baseline=self.baseline)
        # Disposable separate probe seed; this is not a candidate or a repetition.
        from .plan import stream_seeds
        config.update(seed=261013900, stream_seeds=stream_seeds(261013900),
                      run_id=config['run_id']+'_TIMING_ONLY', case='TIMING_ONLY')
        trainer = build_trainer(config, self.sources, self.device)
        data = streams(config, self.sources, self.device)
        latencies = []
        for _ in range(300):
            if self.elapsed() >= 48*3600:
                raise TimeoutError('Campaign deadline reached during disposable timing')
            started = time.monotonic()
            trace = trainer.step(data['rr'](), data['fr']())
            append_jsonl(self.root/'admission/timing_train_metrics.jsonl',trace)
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            latencies.append(time.monotonic()-started)
        evaluation, raw = official_evaluation(trainer, config, self.sources, device=self.device,
            evaluator_revision=self.baseline['evaluator_revision'], evaluator_pin=self.evaluator_pin,
            deadline_unix=self.started_at+48*3600,
            on_scene=lambda row:self.emit(config,'TIMING_ONLY_NOT_CANDIDATE',step=300,selector='TIMING_ONLY',
                scope='PREFLIGHT_'+row['branch'].upper()+'_SCENE:'+row['scene_id'],metrics=row['metrics'],
                sha=row['identity']['model_state_sha256'],extra={'not_candidate':True,
                    **{k:v for k,v in row.items() if k != 'metrics'}}))
        del raw
        # This numeric preflight result never authorizes retaining probe weights.
        append_jsonl(self.root/'admission/timing_evaluation.jsonl', evaluation)
        self.emit(config, 'TIMING_ONLY_NOT_CANDIDATE', step=300, selector='TIMING_ONLY', scope='PREFLIGHT_RR_FR',
            metrics={**evaluation['rr']['metrics'], **evaluation['fr']['metrics']},
            sha=evaluation['state_after']['model_state_sha256'], extra={'same_state_verified':True,
                'not_candidate':True, 'elapsed_seconds':evaluation['elapsed_seconds']})
        self.timing = {'update_p90_seconds':float(np.quantile(latencies[20:], .9)),
                       'update_median_seconds':float(np.median(latencies[20:])),
                       'eval_seconds':evaluation['elapsed_seconds'], 'disposable_updates':300,
                       'artifact_seconds':self.baseline['artifact_reserve_seconds'],
                       'probe_seconds':self.baseline['probe_reserve_seconds']}
        atomic_json(self.root/'admission/timing.json', self.timing)
        del trainer, data

    def run_case(self, config):
        validate_method_config(config)
        root = self.root/'runs'/config['run_id']/f"attempt_{config['attempt']}"
        if root.exists():
            raise ValueError('Attempt directory exists: no exact resume of unqualified weights; use a new attempt')
        atomic_json(root/'config_resolved.json', config, immutable=True)
        self.emit(config, 'STARTED')
        history = dict(config, evaluations=[], completed_update=0)
        data = trainer = None
        status, reason, last_eval = 'FAILED', None, None
        try:
            if source_code_pin() != self.code_pin:
                raise ValueError('Code changed after campaign admission')
            trainer, data = build_trainer(config, self.sources, self.device), streams(config, self.sources, self.device)
            self.costs[(config['run_id'],config['attempt'])]={'parameters':sum(p.numel() for p in trainer.stage.aligner.parameters())+sum(p.numel() for p in trainer.stage.teacher.parameters()),
                'train_seconds':0.,'eval_seconds':0.,'start':time.monotonic()}
            atomic_json(root/'initialization_hashes.json', trainer.manifest(), immutable=True)
            last_period = time.monotonic(); period = []
            while trainer.updates < config['planned_horizon']:
                # Reserve measured full evaluation + artifact allowance before hard stop.
                reserve = 1.2*(self.timing['eval_seconds']+self.timing['artifact_seconds']+self.timing['probe_seconds'])
                if self.elapsed()+reserve >= 48*3600:
                    status, reason = 'TIMEBOX_LAST', '48H_DEADLINE_RESERVE'
                    break
                tick = time.monotonic(); rr, fr = data['rr'](), data['fr']()
                trace = trainer.step(rr, fr, data_wait_seconds=time.monotonic()-tick)
                self.update_latencies.append(time.monotonic()-tick)
                self.costs[(config['run_id'],config['attempt'])]['train_seconds']+=time.monotonic()-tick
                if len(self.update_latencies)>=20:
                    self.timing['update_p90_seconds']=float(np.quantile(self.update_latencies,.9))
                append_jsonl(root/'train_metrics.jsonl', trace)
                history['completed_update'] = trainer.updates
                period.append(trace['loss'])
                if time.monotonic()-last_period >= 3600:
                    self.emit(config, 'RUNNING', step=trainer.updates, scope='TRAIN_INTERVAL',
                        extra={'loss_mean':float(np.mean(period)), 'n_updates':len(period), 'latest':trace})
                    period=[]; last_period=time.monotonic()
                if trainer.updates in config['evaluation_steps']:
                    selector = 'EXACT_FINAL' if trainer.updates == config['planned_horizon'] else 'CADENCE'
                    last_eval, raw, flat = self._evaluate(trainer, config, selector=selector, root=root)
                    history['evaluations'].append(flat); self.observations.append(flat)
                    self.timing['eval_seconds'] = max(self.timing['eval_seconds'],last_eval['elapsed_seconds'])
                    if flat['joint_pass']:
                        self._retain(trainer, config, data, last_eval, raw)
                    del raw
                    decision = should_prune(history, self.runs)
                    if decision['prune']:
                        status, reason = 'PRUNED', decision
                        break
                if trainer.updates in (10000,25000,50000,100000,150000):
                    from .diagnostics import run_probes
                    probe_started=time.monotonic()
                    diagnostic = run_probes(trainer,self.sources,self.baseline,self.device)
                    self.timing['probe_seconds']=max(self.timing['probe_seconds'],time.monotonic()-probe_started)
                    append_jsonl(root/'diagnostics_numeric.jsonl', diagnostic)
                    # Split nested probe groups so BI remains below the cell limit.
                    for name,value in numeric_chunks(diagnostic):
                        self.emit(config,'MEASURED',step=trainer.updates,selector='FIXED_PROBE',
                            scope='DIAGNOSTIC:'+name,sha=state_hash(trainer.stage),extra={name:value})
            else:
                status = 'COMPLETE'
            if status == 'TIMEBOX_LAST' and trainer.updates and (last_eval is None or last_eval['state_after']['completed_update'] != trainer.updates):
                last_eval, raw, flat = self._evaluate(trainer, config, selector='TIMEBOX_LAST', root=root)
                history['evaluations'].append(flat); self.observations.append(flat)
                if flat['joint_pass']:
                    self._retain(trainer, config, data, last_eval, raw)
                del raw
        except Exception as exc:
            status, reason = 'FAILED', f'{type(exc).__name__}: {exc}'
        history.update(status=status, reason=reason)
        if history['evaluations']:
            best_numeric=min(history['evaluations'],key=lambda r:(-r['HQNR'],r['ERGAS'],r['completed_update']))
            numeric_keys=('HQNR','D_s','D_lambda','JQM','ERGAS','SCC','SAM','PSNR','SSIM','Q8','Q4','RMSE','CC')
            self.emit(config,status,step=best_numeric['completed_update'],selector='HQNR_MAX_OBSERVED_NUMERIC',
                scope='RR_FR_COMPLETE20',metrics={k:best_numeric[k] for k in numeric_keys if k in best_numeric},
                sha=best_numeric['model_state_sha256'],extra={'same_state_verified':True,
                    'not_an_extra_seed':True,'weights_retained_only_if_joint_pass':best_numeric['joint_pass']})
        self.runs.append(history)
        atomic_json(root/'final_status.json', history, immutable=True)
        self.emit(config,status,step=history['completed_update'],extra={'reason':reason})
        self.summarize()
        del trainer, data
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        return history

    def _retain(self, trainer, config, data, evaluation, raw):
        started=time.monotonic()
        try:
            artifact = retain_qualified(self.root/'qualified',trainer,config,self.sources,evaluation,raw,
                stream_states={k:v.state() for k,v in data.items()},
                provenance={**self.provenance,'evaluator':self.evaluator_pin,'code_files':self.code_pin})
            self.artifacts.append({'run_id':config['run_id'],'attempt':config['attempt'],
                'completed_update':trainer.updates,'model_state_sha256':evaluation['state_after']['model_state_sha256'],
                'status':artifact['status'],'path':artifact['artifact_path'],'checkpoint_file_sha256':artifact['checkpoint_file_sha256']})
            self.emit(config,artifact['status'],step=trainer.updates,selector='TARGET_QUALIFIED',
                scope='ARTIFACT',sha=evaluation['state_after']['model_state_sha256'],
                metrics={**evaluation['rr']['metrics'],**evaluation['fr']['metrics']},
                extra={'path':artifact['artifact_path'],'file_sha256':artifact['checkpoint_file_sha256'],
                       'export_error':artifact.get('export_error'),'same_state_verified':True})
        except Exception as exc:
            self.artifacts.append({'run_id':config['run_id'],'attempt':config['attempt'],
                'completed_update':trainer.updates,'model_state_sha256':evaluation['state_after']['model_state_sha256'],
                'status':'TARGET_PASS_ARTIFACT_FAILED','error':str(exc)})
            self.emit(config,'TARGET_PASS_ARTIFACT_FAILED',step=trainer.updates,scope='ARTIFACT',
                sha=evaluation['state_after']['model_state_sha256'],extra={'error':str(exc)})
        finally:
            self.timing['artifact_seconds']=max(self.timing['artifact_seconds'],time.monotonic()-started)

    def summarize(self):
        best = representative(self.observations)
        best_artifacts=[r for r in self.artifacts if best and r['model_state_sha256']==best['model_state_sha256']]
        summary = {'campaign':CAMPAIGN_ID,'server':self.server,'dataset':self.dataset,'test_aware':True,
            'status':'TARGET_QUALIFIED' if best else 'NO_TARGET_QUALIFIED_TEACHER',
            'representative_numeric':best,'representative_artifact_records':best_artifacts,
            'representative_available':any(r['status']=='QUAL_COMPLETE' for r in best_artifacts),
            'qualified_min_ergas_reserve':rr_reserve(self.observations),
            'runs':self.runs,'all_evaluations':self.observations,'case_progress':self.progress,
            'artifacts':self.artifacts,'repeat_statistics':repeat_statistics(self.runs),
            'numeric_pareto':pareto_front(self.observations) if self.observations else [],
            'pending_uploads':len(self.outbox.pending())}
        atomic_json(self.root/'summaries/current.json',summary)
        return summary

    def launch(self, sources_path, *, resume_campaign=False, on_locked=None):
        """No cross-server selection locks; local duplicate launch fails immediately."""
        with (self.root/'campaign.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            clock = self.root/'campaign.json'
            if clock.exists():
                if not resume_campaign:
                    raise ValueError('Campaign already started; use --resume-campaign to keep its original48h clock')
                saved=json.loads(clock.read_text())
                if (saved['campaign']!=CAMPAIGN_ID or saved['server']!=self.server or saved['dataset']!=self.dataset
                        or saved.get('method_contract') != METHOD_CONTRACT
                        or saved.get('parent_campaign_id') != PARENT_CAMPAIGN_ID
                        or saved['deadline_unix'] != saved['started_unix']+48*3600):
                    raise ValueError('Campaign clock identity changed')
                self.started_at=saved['started_unix']
                if not 0<=self.elapsed()<48*3600:
                    raise ValueError('Original48h budget expired; cannot silently start another48h')
                self._recover_numeric_history()
            else:
                if resume_campaign:
                    raise ValueError('No existing campaign clock to resume')
                self.started_at=time.time()
                atomic_json(clock,{'campaign':CAMPAIGN_ID,'server':self.server,'dataset':self.dataset,
                    'method_contract':METHOD_CONTRACT,'parent_campaign_id':PARENT_CAMPAIGN_ID,
                    'started_unix':self.started_at,'deadline_unix':self.started_at+48*3600,
                    'started_at':dt.datetime.now(dt.timezone.utc).isoformat()},immutable=True)
            if on_locked:
                on_locked()
            self.admission(sources_path)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
            self.timing_probe()
            queue=[resolve_case(self.server,baseline=self.baseline,loss_case=f) for f in LOSS_ORDER[self.server]]
            for config in queue:
                self.emit(config,'PLANNED')
            for config in queue:
                if self.elapsed() >= 22*3600:
                    self.emit(config,'NOT_STARTED_BUDGET',extra={'phase':'F_PHASE_CLOSED'})
                    continue
                if self._admit_case(config, phase_end=22*3600):
                    self.run_case(config)
            anchors=self._anchors()
            for anchor in anchors:
                order=list(OPTIMIZER_ORDER[self.server])+[t for t in OPTIMIZER_CASES if t not in OPTIMIZER_ORDER[self.server]]
                for t in order:
                    config=resolve_case(self.server,baseline=self.baseline,loss_case=anchor['loss_case'],optimizer_case=t)
                    if not config['runnable']:
                        self.emit(config,'SKIP_DUPLICATE_CONFIG'); continue
                    if self._admit_case(config,phase_end=36*3600):
                        self.run_case(config)
            # Fresh long-schedule candidates; never extend H50 weights.
            for anchor in self._anchors():
                for schedule in ('H100','H150'):
                    config=resolve_case(self.server,baseline=self.baseline,loss_case=anchor['loss_case'],
                        optimizer_case=anchor['optimizer_case'],schedule=schedule)
                    if self._admit_case(config,phase_end=36*3600):
                        self.run_case(config)
            for anchor in self._anchors():
                for replica in ('R2','R3'):
                    config=resolve_case(self.server,baseline=self.baseline,loss_case=anchor['loss_case'],
                        optimizer_case=anchor['optimizer_case'],schedule=anchor['schedule'],replica=replica)
                    if self._admit_case(config,phase_end=48*3600):
                        self.run_case(config)
            summary=self.summarize()
            for name,value in numeric_chunks({'repeat_statistics':summary['repeat_statistics'],
                'qualified_representative':summary['representative_numeric'],
                'qualified_rr_reserve':summary['qualified_min_ergas_reserve'],
                'representative_available':summary['representative_available'],
                'artifacts':summary['artifacts'],'numeric_pareto':summary['numeric_pareto']},path='FINAL_SUMMARY'):
                self.emit({},'FINAL_SUMMARY',scope=name,extra={'summary':value})
            self.emit({},'CAMPAIGN_COMPLETE',extra={'status':summary['status'],'pending_uploads':summary['pending_uploads']})
            return self.summarize()

    def _admit_case(self,config,*,phase_end):
        previous=[r for r in self.runs if r['run_id']==config['run_id']]
        if any(r['status'] in ('COMPLETE','PRUNED','TIMEBOX_LAST') for r in previous):
            return False
        if previous:
            config['attempt']=max(r['attempt'] for r in previous)+1
            self.emit(config,'FRESH_RETRY',extra={'exact_resume':False,'original_clock_preserved':True})
        full=self._estimate(config['planned_horizon'])
        # 25k uses actual cadence count up to that point, not unsupported H25.
        minimum=25000*self.timing['update_p90_seconds']+14*(self.timing['eval_seconds']+self.timing['artifact_seconds'])+2*self.timing['probe_seconds']
        decision=admit_new_run(elapsed_seconds=self.elapsed(),estimated_to_25k_seconds=minimum,estimated_full_seconds=full)
        protected_f00=config['loss_case']=='F00' and config['optimizer_case'] is None and config['replica']=='R1'
        if not protected_f00 and self.elapsed()+1.2*full > phase_end:
            decision={'admitted':False,'reason':'PHASE_BUDGET_ENDPOINT'}
        if not decision['admitted']:
            self.emit(config,'NOT_STARTED_BUDGET',extra=decision)
        return decision['admitted']

    def _recover_numeric_history(self):
        """Recover bookkeeping only; unqualified weights do not exist on disk."""
        self.runs=[]; self.observations=[]; self.progress=[]
        for path in sorted(self.outbox.outbox.glob('T48-*.json')):
            rec=json.loads(path.read_text())
            validate_record(rec)
            if rec['identity']['eval_scope'] not in ('STATUS','ARTIFACT'):
                continue
            fields=rec['source']['fields']
            self.progress.append({'run_id':rec['identity']['run'],'case':fields['Case_ID'],
                'status':fields['Status'],'step':fields['Selected_step'],'attempt':fields.get('Attempt'),
                'record_id':rec['record_id'],'extra':rec['source']['extra']})
        for path in sorted((self.root/'runs').glob('*/attempt_*/config_resolved.json')):
            validate_method_config(json.loads(path.read_text()))
            final=path.parent/'final_status.json'
            if final.exists():
                run=json.loads(final.read_text())
                validate_method_config(run)
            else:
                run=json.loads(path.read_text())
                run.update(status='INTERRUPTED',reason='PROCESS_ENDED_WITHOUT_FINAL_STATUS',completed_update=0,evaluations=[])
                evaluations=path.parent/'evaluations.jsonl'
                if evaluations.exists():
                    for line in evaluations.read_text().splitlines():
                        rec=json.loads(line)
                        if 'qualification' not in rec:
                            continue
                        if any(rec.get('state_after',{}).get(k) != v for k,v in METHOD_CONTRACT.items()):
                            raise ValueError('Recovered evaluation method identity differs from v1.3')
                        flat={**rec['rr']['metrics'],**rec['fr']['metrics'],**rec['qualification'],**rec['state_after'],
                              'schedule':run['schedule'],'loss_case':run['loss_case'],'optimizer_case':run['optimizer_case'],
                              'replica':run['replica'],'seed':run['seed']}
                        run['evaluations'].append(flat)
                        run['completed_update']=max(run['completed_update'],flat['completed_update'])
                atomic_json(final,run,immutable=True)
            for observation in run['evaluations']:
                if (any(observation.get(k) != v for k,v in METHOD_CONTRACT.items())
                        or any(observation.get(k) != run[k] for k in ('run_id','attempt','seed'))):
                    raise ValueError('Recovered final-status evaluation has incompatible method/run identity')
            self.runs.append(run); self.observations.extend(run['evaluations'])
        self.artifacts=[]
        for path in sorted((self.root/'qualified').glob('*/*/*/*/model_manifest.json')):
            manifest=json.loads(path.read_text()); ident=manifest['identity']
            validate_method_config(manifest['config'])
            if any(ident.get(k) != v for k,v in METHOD_CONTRACT.items()):
                raise ValueError('Recovered qualified checkpoint method identity differs from v1.3')
            checkpoint=path.parent/'teacher_aligner.pt'
            valid=checkpoint.is_file() and file_sha(checkpoint)==manifest['file_sha256']
            self.artifacts.append({'run_id':ident['run_id'],'attempt':ident['attempt'],
                'completed_update':ident['completed_update'],'model_state_sha256':ident['model_state_sha256'],
                'path':str(path.parent),'checkpoint_file_sha256':manifest['file_sha256'],
                'status':manifest['status'] if valid else 'TARGET_PASS_ARTIFACT_FAILED'})

    def _anchors(self):
        candidates=[]
        for run in self.runs:
            # Technical failures and incomplete <25k do not nominate recipes.
            if run['status'] not in ('COMPLETE','PRUNED','TIMEBOX_LAST') or run['completed_update']<25000:
                continue
            records=run['evaluations']
            chosen=representative(records) or (min(records,key=lambda r:gap_rank(r,server=self.server)) if records else None)
            if chosen:
                candidates.append(chosen)
        first=representative(candidates) or (min(candidates,key=lambda r:gap_rank(r,server=self.server)) if candidates else None)
        if first is None:
            return []
        # Second anchor only for a distinct nondominated quality tradeoff.
        others=[r for r in pareto_front(candidates) if (r['loss_case'],r['optimizer_case'])!=(first['loss_case'],first['optimizer_case'])
                and ((r['HQNR']>first['HQNR'] and r['ERGAS']>first['ERGAS']) or (r['HQNR']<first['HQNR'] and r['ERGAS']<first['ERGAS']))]
        return [first]+([min(others,key=lambda r:gap_rank(r,server=self.server))] if others else [])


def numeric_chunks(value,path='probe',limit=18000):
    """Lossless recursive numeric scopes, below fixed64 BI cell-size limits."""
    if len(json.dumps(value,ensure_ascii=False,allow_nan=False))<=limit:
        yield path,value
    elif isinstance(value,dict):
        for k,v in value.items():
            yield from numeric_chunks(v,path+'/'+str(k),limit)
    elif isinstance(value,list):
        for i,v in enumerate(value):
            yield from numeric_chunks(v,path+'/'+str(i),limit)
    else:
        raise ValueError('Unsplit scalar exceeds the BI cell limit')


def repeat_statistics(runs):
    """Use one full-horizon record per actual seed, not repeated selectors."""
    groups={}
    for run in runs:
        key=(run['loss_case'],run['optimizer_case'],run['schedule'])
        groups.setdefault(key,{}).setdefault(run['seed'],[]).append(run)
    result=[]
    for key,seeds in groups.items():
        rows=[]; failures=[]
        for seed,attempts in seeds.items():
            completed=[r for r in attempts if r['status']=='COMPLETE' and r['completed_update']==r['planned_horizon']]
            if not completed:
                failures.append({'seed':seed,'statuses':[r['status'] for r in attempts]});continue
            run=sorted(completed,key=lambda r:r['attempt'])[0]
            # Numeric HQNR-max observation includes target misses; no survivor filter.
            chosen=min(run['evaluations'],key=lambda r:(-r['HQNR'],r['ERGAS'],r['completed_update']))
            rows.append(chosen)
        result.append({'loss_case':key[0],'optimizer_case':key[1],'schedule':key[2],
            'attempted_independent_seeds':len(seeds),'complete_seeds':len(rows),'failed_or_incomplete_seeds':failures,
            'target_hit_count':sum(any(e['joint_pass'] for e in run['evaluations']) for ss in seeds.values()
                                   for run in sorted([r for r in ss if r['status']=='COMPLETE'],key=lambda r:r['attempt'])[:1]),
            'selector':'HQNR_MAX_OBSERVED_NUMERIC_ALL_COMPLETE_SEEDS',
            'metrics':{k:{'mean':float(np.mean([r[k] for r in rows])) if rows else None,
                          'sample_std':float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else None}
                       for k in ('HQNR','ERGAS','SCC')}})
    return result
