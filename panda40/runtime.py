"""Bounded two-lane orchestration without mutating any old campaign module."""
from copy import deepcopy
from dataclasses import asdict
import fcntl
import gc
import json
from pathlib import Path
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
from ta2.evaluation import evaluator_identity, file_sha
from . import plan


class BudgetExpired(RuntimeError):
    pass


def code_pin():
    from s2dshift48.runtime import code_pin as parent
    return {**parent(), **{str(p.relative_to(plan.ROOT)):file_sha(p)
        for p in sorted((plan.ROOT/'panda40').glob('*.py'))},
        **{str(p.relative_to(plan.ROOT)):file_sha(p)
           for p in sorted((plan.ROOT/'scripts').glob('run_panda40*.sh'))}}


def improvement(evaluation, selection):
    h = evaluation['fr']['metrics']['HQNR']
    e = evaluation['rr']['metrics']['ERGAS']
    step = evaluation['state_before']['completed_update']
    row = {'HQNR':h,'ERGAS':e,'SCC':evaluation['rr']['metrics']['SCC'],
           'completed_update':step}
    plan.rank(row)
    best = selection.get('best_selected')
    best_h = selection.get('best_hqnr')
    reasons = []
    if best is None or plan.rank(row) > plan.rank(best):
        reasons.append('HQNR_GATE_ERGAS_SELECTOR')
    if best_h is None or (h,-e,-step) > (best_h['HQNR'],-best_h['ERGAS'],-best_h['completed_update']):
        reasons.append('BEST_HQNR_BACKUP')
    return row, reasons


class Campaign:
    def __init__(self, server, *, output='work_dir/panda40', device='cuda'):
        self.config = plan.default_config(server)
        self.device = device
        self.root = Path(output).resolve()/plan.CAMPAIGN_ID/server/'WV3'
        self.root.mkdir(parents=True,exist_ok=True)
        self.state_path = self.root/'campaign_state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'started_unix':time.time(),'created_at':utc_now(),'runs':[], 'groups':[], 'timing':{}}
        self.initial_elapsed = max(self.state.get('elapsed_seconds',0.),time.time()-self.state['started_unix'])
        self.tick = time.monotonic()
        self.code, self.evaluator_pin = code_pin(), evaluator_identity()
        self.stop_requested = False
        self.sources = self.registered = None

    def elapsed(self):
        value = max(time.time()-self.state['started_unix'],
            self.initial_elapsed+time.monotonic()-self.tick,
            self.state.get('elapsed_seconds',0.),getattr(self,'_elapsed_highwater',0.))
        self._elapsed_highwater = value
        return value

    @property
    def deadline(self):
        return time.time()+max(0.,self.config['budget']['total_seconds']-self.elapsed())

    def persist(self):
        self.state['elapsed_seconds'] = self.elapsed()
        atomic_json(self.state_path,self.state)

    def check(self, *, reserve=True):
        remaining = self.config['budget']['total_seconds']-self.elapsed()
        if self.stop_requested or remaining <= (self.config['budget']['closeout_seconds'] if reserve else 0):
            raise BudgetExpired('STOP_REQUESTED' if self.stop_requested else '40H_BUDGET_CLOSEOUT')

    def verify_code(self):
        if self.code != code_pin():
            raise RuntimeError('Implementation changed during campaign; refusing mixed-code training')

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
            raise ValueError('Registered data catalog SHA changed')
        self.sources = EdgeDataSources(catalog,root=plan.ROOT,sensor='WV3')
        for split in ('train','val','rr','fr'):
            self.sources.raw(split,[0])
        self.registered = RegisteredRRTrain(self.sources)
        pin = {'config':self.config,'code':self.code,'evaluator':self.evaluator_pin,
               'sources':source_manifest(self.sources),'environment':environment_manifest(self.device)}
        path = self.root/'admission.json'
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(pin)):
            raise ValueError('Admission/data/code/environment changed; do not reuse this campaign root')
        atomic_json(path,pin)
        self.state['status'] = 'RUNNING'
        self.persist()

    def evaluate(self, trainer):
        from .evaluation import official_evaluate
        return official_evaluate(trainer,self.sources,evaluator_pin=self.evaluator_pin,
            device=self.device,deadline_unix=self.deadline,
            on_scene=lambda *a,**kw:self.check(reserve=False))

    def bind_donor(self):
        from .donor import local_candidates, verify_file, bind
        from donor30.runtime import build_trainer, load_local_payload
        from donor24.evaluation import official_evaluate as replay_old
        path = self.root/'fixed_donor.json'
        if path.exists():
            row = json.loads(path.read_text())
            body = {k:v for k,v in row.items() if k!='binding_sha256'}
            if (row.get('binding_sha256')!=plan.canonical_sha256(body)
                    or row.get('target_campaign_id')!=plan.CAMPAIGN_ID
                    or row.get('source_server')!=self.config['server']
                    or file_sha(self.root/'donor_replay_full40.json')!=row.get('donor_replay_sha256')):
                raise ValueError('Fixed donor/replay evidence changed')
            verify_file(row)
            if row.get('selection_policy') != plan.POLICY.manifest() or not row.get('reloaded_full40_verified'):
                raise ValueError('Fixed donor contract changed')
            self.state['donor'] = row
            return
        rows,rejected = local_candidates(plan.ROOT/'work_dir',self.config['server'])
        atomic_json(self.root/'donor_inventory.json',{'candidates':rows,'rejected':rejected})
        chosen = plan.choose(rows)
        if chosen is None:
            # One fallback, charged against the same 40h clock; never swap the
            # donor after any Stage 2 case begins.
            cfg = plan.resolve_case(self.config['server'],'D_A1')
            self.preflight_cases([cfg])
            remaining = self.config['budget']['total_seconds']-self.elapsed()-self.config['budget']['closeout_seconds']
            if self.estimate(cfg)*self.config['budget']['safety_factor'] > remaining:
                raise BudgetExpired('MISSING_DONOR_FALLBACK_H50_DOES_NOT_FIT')
            self.run_case(cfg)
            completed = [r for r in self.state['runs']
                if r['run_id']==cfg['run_id'] and r['status']=='COMPLETED']
            directory = Path(completed[-1]['directory'])
            cfg = json.loads((directory/'config.json').read_text())
            chosen = json.loads((directory/'models'/'selection.json').read_text())['best_selected']
            chosen.update(source='PANDA40_FRESH_MISSING_DONOR_FALLBACK',source_config=cfg,
                          source_server=self.config['server'])
            from .stage1 import load_stage1_checkpoint
            trainer,stream,_ = load_stage1_checkpoint(chosen['checkpoint_path'],verify_file(chosen),cfg,
                self.sources,device=self.device,code_pin=self.code,evaluator_pin=self.evaluator_pin,
                restore_global_rng=False)
            del stream
            replay = self.evaluate(trainer)
        else:
            payload = load_local_payload(chosen['checkpoint_path'],verify_file(chosen))
            trainer = build_trainer(payload['config'],self.sources,device=self.device)
            trainer.load_state_dict(payload['trainer'])
            del payload
            replay = replay_old(trainer,self.sources,evaluator_pin=self.evaluator_pin,
                device=self.device,deadline_unix=self.deadline,
                on_scene=lambda *a,**kw:self.check())
        original = json.loads(Path(chosen['evaluation_path']).read_text())
        for split in ('rr','fr'):
            if original[split]['per_scene'] != replay[split]['per_scene']:
                raise ValueError('Frozen donor full40 reload prediction/metric mismatch: '+split)
        if original['state_before']['model_state_sha256'] != replay['state_before']['model_state_sha256']:
            raise ValueError('Frozen donor A/T identity mismatch')
        atomic_json(self.root/'donor_replay_full40.json',replay)
        row = bind(chosen,replay_verified=True)
        row['donor_replay_sha256']=file_sha(self.root/'donor_replay_full40.json')
        row['target_campaign_id']=plan.CAMPAIGN_ID
        row['binding_sha256']=plan.canonical_sha256(row)
        atomic_json(path,row)
        self.state['donor'] = row
        del trainer
        self.release()
        self.persist()

    def calibration(self, mask):
        from panda_stage2_condq.donor import load_donor
        from panda_stage2_condq.runtime import calibrate
        from panda_stage2_condq.calibration import CalibrationStats
        if mask not in (plan.LOSS_MASK_POLICY, plan.MASK_POLICY):
            raise ValueError('Only registered MASK2/full-image calibration policies are permitted')
        label = 'MASK2' if mask == plan.LOSS_MASK_POLICY else 'FULL'
        path = self.root/'calibration'/f'{label}.npz'
        receipt = path.with_suffix('.json')
        selected = self.state['donor']
        self.check()
        prior = None
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            sha = prior.get('file_sha256') or prior.get('sha256')
            if (sha is None or file_sha(path) != sha
                    or (prior.get('sha256') is not None and prior['sha256'] != sha)):
                raise ValueError('Immutable calibration file changed')
        if path.exists():
            # Recover only an integrity-checked complete immutable artifact.
            donor = load_donor(selected['checkpoint_path'],selected['checkpoint_sha256'],self.registered)
            sha = file_sha(path)
            stats = CalibrationStats.load(path,expected_binding=donor.binding,
                expected_file_sha256=sha,expected_mask_policy=mask,memory_budget_bytes=3*1024**3)
            if (stats.calibration_kind != 'exact_condq'
                    or stats.metadata['g_grouping']['k_requested'] != 5):
                raise ValueError('Campaign calibration must be complete exact conditional-q K5')
            if prior is not None and (prior.get('donor_binding') != donor.binding
                    or prior.get('mask_policy') != mask
                    or prior.get('artifact_sha256') != stats.artifact_sha256):
                raise ValueError('Calibration receipt donor/policy/content identity mismatch')
            saved = {**(prior or {}),'path':str(path),'sha256':sha,'file_sha256':sha,
                'artifact_sha256':stats.artifact_sha256,'calibration_schema':stats.schema,
                'kind':stats.calibration_kind,'memory_bytes':stats.memory_bytes,
                'recovered_complete_artifact':prior is None,'donor_binding':donor.binding,
                'mask_policy':mask,'tau_e':stats.tau_e,'sample_count':stats.metadata['sample_count'],
                'n_e':stats.metadata['n_e'],'n_q':stats.metadata['n_q']}
            del stats,donor
        else:
            if prior is not None:
                raise ValueError('Calibration receipt exists without its immutable artifact')
            remaining = self.config['budget']['total_seconds']-self.elapsed()-self.config['budget']['closeout_seconds']
            estimate = self.calibration_overhead({'phase':'stage2',
                'stage2_spec':{'variant':'panda_condq'},'loss_mask_policy':mask})
            if remaining < estimate:
                raise BudgetExpired('Not enough budget for the missing complete '+label+' calibration')
            donor = load_donor(selected['checkpoint_path'],selected['checkpoint_sha256'],
                               self.registered,device=self.device)
            def progress(info):
                self.check()
                if info['seen'] % 600 == 0 or info['seen'] == info['expected']:
                    print(f'[CALIBRATION {label}] {info}',flush=True)
            saved = calibrate(donor,self.registered,output=path,k=5,batch_size=12,
                memory_budget_bytes=3*1024**3,include_condq=True,mask_policy=mask,
                device=self.device,on_progress=progress)
            # calibrate() returns file_sha256, not sha256. Keep one verified
            # value under the campaign alias used by config/checkpoint records.
            saved['sha256'] = saved['file_sha256']
            del donor
        atomic_json(receipt,saved)
        if 'elapsed_seconds' in saved:
            self.state.setdefault('calibration_timing',{})[label] = saved['elapsed_seconds']
        self.release()
        return path,saved

    def calibration_overhead(self,cfg):
        """One-time calibration allowance; add once per admitted group, not seed.

        The FULL control cannot reuse MASK2 tau_e. Its first build is charged
        against observed exact MASK2 calibration time (600s before observation).
        This estimate is not a promise: progress callbacks enforce the deadline.
        """
        if cfg['phase'] != 'stage2' or cfg['stage2_spec']['variant'] == 'gt_only':
            return 0.
        mask = cfg['loss_mask_policy']
        if mask not in (plan.LOSS_MASK_POLICY, plan.MASK_POLICY):
            raise ValueError('Unregistered calibration policy')
        label = 'MASK2' if mask == plan.LOSS_MASK_POLICY else 'FULL'
        if (self.root/'calibration'/f'{label}.npz').exists():
            return 0.
        timings = self.state.get('calibration_timing',{})
        return max([600., *[float(v) for v in timings.values()]])

    def build(self,cfg):
        if cfg['phase'] == 'donor':
            from .stage1 import build_stage1,Stage1Stream
            trainer = build_stage1(cfg,self.sources)
            stream_factory = lambda:Stage1Stream(self.sources,cfg,device=self.device)
        else:
            from panda_stage2_condq.donor import load_donor
            from panda_stage2_condq.spec import Stage2Spec
            from panda_stage2_condq.method import ConditionalQStage2
            from panda_stage2_condq.training import StudentTrainer,TrainingSpec
            from panda_stage2_condq.stream import TrainStream,StreamSpec
            from panda_stage2_condq.calibration import CalibrationStats
            selected = self.state['donor']
            if cfg['stage2_spec'].get('mask_policy') != cfg['loss_mask_policy']:
                raise ValueError('Stage 2 declared loss/calibration mask policy mismatch')
            uses_teacher = cfg['stage2_spec']['variant'] != 'gt_only'
            stats = None
            calpath = receipt = None
            if uses_teacher:
                calpath,receipt = self.calibration(cfg['loss_mask_policy'])
            donor = load_donor(selected['checkpoint_path'],selected['checkpoint_sha256'],
                              self.registered,include_teacher=uses_teacher)
            if uses_teacher:
                stats = CalibrationStats.load(calpath,expected_binding=donor.binding,
                    expected_file_sha256=receipt['sha256'],expected_mask_policy=cfg['loss_mask_policy'])
            cfg.update(donor_checkpoint=selected['checkpoint_path'],
                donor_checkpoint_sha256=selected['checkpoint_sha256'],donor_binding=donor.binding,
                donor_exploratory=selected['exploratory'],
                calibration_path=None if calpath is None else str(calpath),
                calibration_sha256=None if receipt is None else receipt['sha256'])
            stage = ConditionalQStage2(Stage2Spec(**cfg['stage2_spec']),donor.aligner,donor.teacher,
                                      donor_binding=donor.binding,stats=stats)
            training = TrainingSpec(full_audit_interval=100)
            cfg['training_spec'] = asdict(training)
            trainer = StudentTrainer(stage,training,evaluation_pin=self.evaluator_pin,hqnr_target=.959)
            trainer.config = deepcopy(cfg)
            stream_spec = StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'])
            stream_factory = lambda:TrainStream(self.registered,stream_spec,microbatch_size=12,
                                                effective_batch_size=48,device=self.device)
        from .cost import ensure_cost
        ensure_cost(cfg,self.sources,retained_trainer=trainer,device=self.device,
                    deadline_unix=self.deadline-self.config['budget']['closeout_seconds'],cost_root=self.root)
        trainer.stage.to(self.device)
        if cfg['phase'] == 'stage2' and trainer.stage.stats is not None:
            trainer.stage.move_calibration(self.device,memory_budget_bytes=512*1024**2)
        return trainer,stream_factory()

    def trace(self,trainer,stream):
        start = time.monotonic()
        if trainer.config['phase'] == 'donor':
            from .stage1 import advance_stage1
            report = advance_stage1(trainer,stream)
        else:
            before_stream, before_update = stream.state_dict(), trainer.updates
            try:
                report = trainer.step_accumulated(stream.next_update())
            except Exception:
                if not trainer.poisoned and trainer.updates == before_update:
                    stream.load_state_dict(before_stream)
                raise
        self.sync()
        return report,time.monotonic()-start

    @staticmethod
    def timing_key(cfg):
        return cfg['aligner_kind'] if cfg['phase'] == 'donor' else ('stage2_gt' if cfg['case']=='G' else 'stage2_q')

    def preflight_cases(self,cases):
        for base in cases:
            key = self.timing_key(base)
            if key in self.state['timing']:
                continue
            cfg = deepcopy(base)
            cfg['disposable_preflight'] = True
            trainer,stream = self.build(cfg)
            try:
                times = []
                for _ in range(100):
                    self.check()
                    _,seconds = self.trace(trainer,stream)
                    times.append(seconds)
                result = self.evaluate(trainer)
                if not all(result['qualification'][k] is True for k in
                           ('metrics_complete','metrics_finite','same_state_verified')):
                    raise ValueError('Preflight failed full40 metric/state checks')
                # Catch checkpoint/outbox integration problems in the disposable
                # pilot, before an actual H50 run. Nothing is enqueued/published.
                from .reporting import make_record, validate_record
                started = time.monotonic()
                with tempfile.TemporaryDirectory(prefix='.preflight-checkpoint-', dir=self.root) as temp:
                    artifact = self._save(trainer,stream,Path(temp)/'disposable.pt',cfg,result)
                    validate_record(make_record(cfg,result,artifact,root=self.root))
                checkpoint_seconds = time.monotonic()-started
                self.state['timing'][key] = {'update_p90':float(np.quantile(times[10:],.9)),
                    'eval_seconds':result['elapsed_seconds'],
                    'checkpoint_seconds':max(10.,checkpoint_seconds)}
                atomic_json(self.root/f'preflight_{key}.json',{'timing':self.state['timing'][key],
                    'evaluation':result,'eligible_for_selection':False,'weights_discarded':True,
                    'checkpoint_roundtrip_verified':True,'local_outbox_envelope_validated':True,
                    'published_to_outbox':False})
            finally:
                del trainer,stream
                self.release()
            self.persist()

    def estimate(self,cfg):
        timing = self.state['timing'][self.timing_key(cfg)]
        return (plan.HORIZON*timing['update_p90']+
                len(cfg['evaluation_steps'])*(timing['eval_seconds']+timing['checkpoint_seconds']))

    def completed(self,cfg):
        return any(r['run_id']==cfg['run_id'] and r['status']=='COMPLETED' for r in self.state['runs'])

    def _save(self,trainer,stream,path,cfg,evaluation=None):
        from donor30.artifacts import _same
        from .evaluation import model_state_hash
        from .stage1 import _check_evaluation
        path = Path(path)
        if not _same(cfg,trainer.config) or getattr(trainer,'poisoned',False):
            raise ValueError('Only a complete trainer with this exact case config can be saved')
        before = model_state_hash(trainer.stage)
        identity = {key:cfg[key] for key in ('run_id','attempt','seed','dataset')}
        identity.update(completed_update=trainer.updates,model_state_sha256=before)
        _check_evaluation(evaluation,identity)
        if cfg['phase'] == 'donor':
            from .stage1 import save_stage1_checkpoint
            artifact = save_stage1_checkpoint(trainer,stream,path,cfg,evaluation,
                code_pin=self.code,evaluator_pin=self.evaluator_pin)
            # The complete resume payload uses the minimal same-pair identity;
            # publication additionally binds the verified official source and
            # evaluator fingerprints. Never invent them from the run config.
            if evaluation is not None:
                artifact['identity'] = deepcopy(evaluation['state_before'])
            return artifact
        from panda_stage2_condq.checkpoint import save_checkpoint
        if path.with_suffix('.json').exists():
            raise FileExistsError('Refusing to overwrite existing Student checkpoint evidence')
        trainer.assert_contract(full_audit=True)
        before_stream = stream.state_dict()
        if (before_stream['epoch']*stream.count+before_stream['cursor'] !=
                trainer.updates*trainer.training.effective_batch_size):
            raise ValueError('Student optimizer and no-drop sampler are not at the same update boundary')
        expected = trainer.state_dict(stream_state=before_stream)
        artifact = save_checkpoint(path,trainer,stream_state=before_stream)
        payload = torch.load(path,map_location='cpu',weights_only=False)
        if (not _same(payload['trainer'],expected)
                or not _same(payload['trainer'],trainer.state_dict(stream_state=stream.state_dict()))
                or model_state_hash(trainer.stage) != before):
            raise ValueError('Student model/optimizer/scheduler/RNG/sampler checkpoint roundtrip differs')
        trainer.assert_contract(full_audit=True)
        artifact.update(file_sha256=artifact['sha256'],checkpoint_path=artifact['path'],
            checkpoint_verified=True,state_hash=before,weights_saved=True,
            model_state_sha256=before,identity=deepcopy(evaluation['state_before']) if evaluation else identity,
            completed_update=trainer.updates,predictions_reverified=False,
            verification='FULL_TRAINER_STREAM_RNG_FILE_ROUNDTRIP_AND_FROZEN_BINDING_AUDIT',
            resume_only=evaluation is None,campaign_resume_complete=True)
        atomic_json(path.with_suffix('.json'),{'artifact':artifact,'config':cfg,'code_pin':self.code,
            'evaluator_pin':self.evaluator_pin,'evaluation':evaluation,
            'selection_policy':plan.POLICY.manifest(),'resume':'EXPLICIT_FULL_STATE_AVAILABLE'})
        return artifact

    def run_case(self,base):
        from .reporting import emit_status,emit_evaluation
        if self.completed(base):
            return
        cfg = deepcopy(base)
        previous = [r for r in self.state['runs'] if r['run_id']==cfg['run_id']]
        for old in previous:
            if old['status']=='RUNNING':
                old['status']='INTERRUPTED_PROCESS_EXIT'
                atomic_json(Path(old['directory'])/'run_state.json',old)
        cfg['attempt'] = max((r['attempt'] for r in previous),default=0)+1
        directory = self.root/'runs'/cfg['run_id']/f"attempt_{cfg['attempt']:02d}"
        while directory.exists():
            cfg['attempt'] += 1
            directory = directory.parent/f"attempt_{cfg['attempt']:02d}"
        directory.mkdir(parents=True)
        status = {'run_id':cfg['run_id'],'case':cfg['case'],'replica':cfg['replica'],
            'attempt':cfg['attempt'],'directory':str(directory),'phase':cfg['phase'],
            'status':'RUNNING','step':0,'started_at':utc_now()}
        self.state['runs'].append(status)
        trainer=stream=None
        selection = {'best_selected':None,'best_hqnr':None,'saved':[], 'policy':plan.POLICY.manifest()}
        def persist():
            atomic_json(directory/'run_state.json',status)
            self.persist()
        # A power/process loss during model construction, calibration or cost
        # profiling must leave an attributable attempt, not an orphan directory.
        atomic_json(directory/'requested_config.json',cfg)
        persist()
        try:
            self.check()
            trainer,stream = self.build(cfg)
            atomic_json(directory/'config.json',cfg)
            emit_status(self.root,cfg,'RUNNING')
            persist()
            times=[]
            while trainer.updates < cfg['planned_horizon']:
                self.check()
                if trainer.updates % 100 == 0:
                    self.verify_code()
                report,seconds = self.trace(trainer,stream)
                times.append(seconds)
                status['step'] = trainer.updates
                append_jsonl(directory/'train.jsonl',{'step':trainer.updates,'seconds_including_IO':seconds,**report})
                if trainer.updates==1 or trainer.updates%100==0:
                    print(f"[{cfg['case']}/{cfg['replica']}] step={trainer.updates} loss={report['loss']:.7g} update_s={seconds:.3f}",flush=True)
                    persist()
                if trainer.updates not in cfg['evaluation_steps']:
                    continue
                evaluation = self.evaluate(trainer)
                eval_path = directory/f'evaluation_{trainer.updates:07d}.json'
                if eval_path.exists():
                    raise FileExistsError('Refusing to overwrite official evaluation: '+str(eval_path))
                atomic_json(eval_path,evaluation)
                row,reasons = improvement(evaluation,selection)
                artifact=None
                if reasons:
                    checkpoint_started = time.monotonic()
                    artifact = self._save(trainer,stream,directory/'models'/f'step_{trainer.updates:07d}.pt',cfg,evaluation)
                    timing = self.state['timing'][self.timing_key(cfg)]
                    timing['checkpoint_seconds'] = max(timing['checkpoint_seconds'],
                                                       time.monotonic()-checkpoint_started)
                    checkpoint = artifact.get('checkpoint_path') or artifact.get('path')
                    sha = artifact.get('file_sha256') or artifact.get('sha256')
                    row.update(run_id=cfg['run_id'],attempt=cfg['attempt'],seed=cfg['seed'],
                        checkpoint_path=checkpoint,file_sha256=sha,checkpoint_verified=True,
                        metrics_complete=True,metrics_finite=True,same_state_verified=True,
                        state_hash=evaluation['state_before']['model_state_sha256'],
                        evaluation_path=str(eval_path),reasons=reasons)
                    selection['saved'].append(row)
                    if 'HQNR_GATE_ERGAS_SELECTOR' in reasons:
                        selection['best_selected']=row
                    if 'BEST_HQNR_BACKUP' in reasons:
                        selection['best_hqnr']=row
                    atomic_json(directory/'models'/'selection.json',selection)
                emit_evaluation(self.root,cfg,evaluation,artifact=artifact)
                print(f"[{cfg['case']}/{cfg['replica']}] step={trainer.updates} HQNR={row['HQNR']:.9f} SCC={row['SCC']:.9f} ERGAS={row['ERGAS']:.9f} saved={bool(artifact)}",flush=True)
                key = self.timing_key(cfg)
                timing = self.state['timing'][key]
                timing['update_p90']=float(np.quantile(times[-1000:],.9))
                timing['eval_seconds']=max(timing['eval_seconds'],evaluation['elapsed_seconds'])
                persist()
            if not (directory/'models'/f'step_{trainer.updates:07d}.pt').exists():
                self._save(trainer,stream,directory/'models'/f'final_{trainer.updates:07d}.pt',cfg)
            status.update(status='COMPLETED',completed_at=utc_now(),selection=selection)
            emit_status(self.root,cfg,'COMPLETED',step=trainer.updates)
        except BaseException as exc:
            status.update(status='STOPPED' if isinstance(exc,BudgetExpired) else 'FAILED',error=repr(exc))
            if trainer is not None and trainer.updates and not getattr(trainer,'poisoned',False):
                status['step'] = trainer.updates
                try:
                    status['stop_checkpoint'] = self._save(trainer,stream,
                        directory/'models'/f'stopped_{trainer.updates:07d}.pt',cfg)
                except Exception as save_error:
                    status['stop_checkpoint_error']=repr(save_error)
            try:
                emit_status(self.root,cfg,status['status'],step=status['step'],extra={'error':repr(exc)})
            except Exception as reporting_error:
                # Preserve the scientific/training failure instead of replacing
                # it with a secondary status/outbox error.
                status['failure_reporting_error']=repr(reporting_error)
            raise
        finally:
            persist()
            del trainer,stream
            self.release()

    def run(self, *, catalog=plan.ROOT/'ablr2/sensor_sources.json'):
        with (self.root/'.controller.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            old={sig:signal.signal(sig,lambda *a:setattr(self,'stop_requested',True))
                 for sig in (signal.SIGTERM,signal.SIGINT)}
            try:
                self.persist()
                self.admission(catalog)
                if self.config['server']=='s3':
                    self.bind_donor()
                    pilots = [plan.resolve_case('s3',c) for c in ('G','Q')]
                else:
                    pilots = [plan.resolve_case('s1',c) for c in ('D_A1','S_A1')]
                self.preflight_cases(pilots)
                atomic_json(self.root/'preflight.json',{'complete':True,'timing':self.state['timing']})
                for group in plan.groups(self.config['server']):
                    self.check()
                    cases = [plan.resolve_case(self.config['server'],c,r) for c,r in group['runs']]
                    pending = [cfg for cfg in cases if not self.completed(cfg)]
                    calibration_seconds = max((self.calibration_overhead(cfg) for cfg in pending),default=0.)
                    needed = (sum(self.estimate(cfg) for cfg in pending)+calibration_seconds)*self.config['budget']['safety_factor']
                    remaining = self.config['budget']['total_seconds']-self.elapsed()-self.config['budget']['closeout_seconds']
                    admitted = needed <= remaining
                    record = {'name':group['name'],'estimated_seconds':needed,'calibration_seconds':calibration_seconds,'remaining_seconds':remaining,
                              'status':'ADMITTED' if admitted else 'SKIPPED_BUDGET','at':utc_now()}
                    self.state['groups'].append(record)
                    self.persist()
                    if not admitted:
                        # No cherry-picking single seeds from an unadmitted pair.
                        continue
                    for cfg in pending:
                        self.run_case(cfg)
                    record['status']='COMPLETED'
                    self.persist()
                self.state['status']='FINISHED_REGISTERED_QUEUE'
            except BudgetExpired as exc:
                self.state.update(status='STOPPED_BUDGET',reason=str(exc))
            except Exception as exc:
                self.state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
                raise
            finally:
                self.persist()
                for sig,handler in old.items():
                    signal.signal(sig,handler)
            return self.state
