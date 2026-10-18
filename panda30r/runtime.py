"""Local-lineage donor/schedule controller with durable wall-time accounting.

All scientific changes live in this package. Existing running campaign code,
data, original-reference metrics and sheet selection rules are not mutated.
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
from ta2.evaluation import evaluator_identity, file_sha
from . import plan


class BudgetExpired(RuntimeError):
    pass


def plain(value):
    return json.loads(json.dumps(value))


def code_pin():
    from panda40.runtime import code_pin as shared
    return {**shared(), **{str(p.relative_to(plan.ROOT)): file_sha(p)
        for p in sorted((plan.ROOT/'panda30r').glob('*.py'))},
        **{str(p.relative_to(plan.ROOT)): file_sha(p)
        for p in sorted((plan.ROOT/'scripts').glob('run_panda30r*.sh'))}}


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


def recover_trace(directory, step):
    """Preserve discarded work, and keep a contiguous selected-lineage timing log.

    A hard exit may leave updates after the last full checkpoint. Never append
    repeated step numbers to the cost source; preserve the original trace in an
    audit backup before atomically retaining its verified checkpoint prefix.
    """
    path=Path(directory)/'train.jsonl'
    if not path.exists():
        if step: raise ValueError('Resume checkpoint lacks its measured training prefix')
        return
    prefix=[]
    with path.open('rb') as stream:
        for expected in range(1,step+1):
            line=stream.readline()
            row=json.loads(line)
            if not line.endswith(b'\n') or row.get('step')!=expected or 'seconds_including_IO' not in row:
                raise ValueError('Resume timing prefix is incomplete/noncontiguous')
            prefix.append(line)
        remainder=stream.read()
    if remainder:
        backup=path.with_name(f'train_before_resume_{time.time_ns()}.jsonl')
        # Backup survives all failures; no training evidence is deleted.
        os.link(path,backup)
        fd,temp=tempfile.mkstemp(prefix='.resume-trace-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as output:
                output.writelines(prefix); output.flush(); os.fsync(output.fileno())
            os.replace(temp,path)
        finally:
            if os.path.exists(temp): os.unlink(temp)


class Campaign:
    def __init__(self, server, *, output='work_dir/panda30r', device='cuda'):
        self.config=plain(plan.default_config(server)); self.server=server; self.device=device
        self.root=Path(output).resolve()/plan.CAMPAIGN_ID/server/'WV3'
        self.root.mkdir(parents=True,exist_ok=True)
        self.state_path=self.root/'campaign_state.json'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'started_unix':time.time(),'created_at':utc_now(),'runs':[],'groups':[],
            'timing':{},'donors':{}}
        self.initial_elapsed=max(self.state.get('elapsed_seconds',0.),time.time()-self.state['started_unix'])
        self.tick=time.monotonic(); self.stop_requested=False
        self.code,self.evaluator_pin=code_pin(),evaluator_identity()
        self.sources=self.registered=None

    def elapsed(self):
        value=max(time.time()-self.state['started_unix'],
            self.initial_elapsed+time.monotonic()-self.tick,
            self.state.get('elapsed_seconds',0.),getattr(self,'_elapsed_highwater',0.))
        self._elapsed_highwater=value
        return value

    @property
    def deadline(self):
        return time.time()+max(0.,self.config['budget']['total_seconds']-self.elapsed())

    def remaining_for_work(self):
        return max(0.,self.config['budget']['total_seconds']-self.elapsed()-self.config['budget']['closeout_seconds'])

    def persist(self):
        self.state['elapsed_seconds']=self.elapsed()
        atomic_json(self.state_path,self.state)

    def check(self, *, reserve=True):
        remaining=self.config['budget']['total_seconds']-self.elapsed()
        if self.stop_requested or remaining <= (self.config['budget']['closeout_seconds'] if reserve else 0):
            raise BudgetExpired('STOP_REQUESTED' if self.stop_requested else '30H_WALL_BUDGET_CLOSEOUT')

    def verify_code(self):
        if self.code!=code_pin():
            raise RuntimeError('Campaign source changed; refusing mixed-code resume/training')

    def sync(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)

    def release(self):
        gc.collect()
        if str(self.device).startswith('cuda'):
            torch.cuda.empty_cache()

    def admission(self,catalog):
        from .evaluation import source_manifest
        self.check(); configure_precision()
        if file_sha(catalog)!=self.config['baseline_resolved']['source_registry_sha256']:
            raise ValueError('Registered WV3 data catalog SHA changed')
        self.sources=EdgeDataSources(catalog,root=plan.ROOT,sensor='WV3')
        for split in ('train','val','rr','fr'):
            self.sources.raw(split,[0])
        self.registered=RegisteredRRTrain(self.sources)
        pin=plain({'config':self.config,'code':self.code,'evaluator':self.evaluator_pin,
            'sources':source_manifest(self.sources),'environment':environment_manifest(self.device)})
        path=self.root/'admission.json'
        if path.exists() and json.loads(path.read_text())!=pin:
            raise ValueError('Admission/data/code/environment changed; do not reuse this campaign root')
        atomic_json(path,pin); self.state['status']='RUNNING'; self.persist()

    def evaluate(self,trainer):
        from .evaluation import official_evaluate
        return official_evaluate(trainer,self.sources,evaluator_pin=self.evaluator_pin,
            device=self.device,deadline_unix=self.deadline,
            on_scene=lambda *a,**kw:self.check(reserve=False))

    def load_donor(self,label,*,include_teacher=True,device='cpu'):
        from .source_donor import load_donor
        from .engine import load_mature_donor
        row=self.state['donors'][label]
        importer=load_mature_donor if row.get('mature') else load_donor
        return importer(row['checkpoint_path'],row['checkpoint_sha256'],self.registered,
                        include_teacher=include_teacher,device=device)

    def bind_import(self,label,candidate,*,mature=False):
        """Replay all 40 official scenes before admitting any A/T version.

        Historical imported numbers are evidence, not silently reused metrics.
        Replay must agree within tight numerical tolerance, with exact tensor
        identity. Test scenes are never used by a training/calibration iterator.
        """
        from .engine import build_donor_replay
        path=self.root/'donors'/f'{label}.json'
        ckpt=Path(candidate['checkpoint_path']).resolve(strict=True)
        sha=candidate.get('checkpoint_sha256') or candidate.get('file_sha256')
        if not sha or file_sha(ckpt)!=sha:
            raise ValueError('Imported donor file SHA changed')
        replay_path=self.root/'donors'/f'{label}_replay_full40.json'
        if path.exists():
            row=json.loads(path.read_text()); body={k:v for k,v in row.items() if k!='binding_sha256'}
            if (row.get('binding_sha256')!=plan.canonical_sha256(body)
                    or row['checkpoint_sha256']!=sha or row['source_server']!=self.server
                    or row.get('mature')!=mature or file_sha(replay_path)!=row['replay_sha256']):
                raise ValueError('Pinned donor/replay evidence changed')
            self.state['donors'][label]=row; return row
        cfg=plan.resolve_case(self.server,'FRESH_A1')
        cfg.update(engine_kind='donor_replay',case=f'IMPORT_{label}',start_update=candidate['completed_update'],
            run_id=f'P30R_WV3_{self.server.upper()}_IMPORT_{label}',planned_horizon=1,
            initialization='READ_ONLY_IMPORT_REPLAY',parent_run=candidate['source_config']['run_id'])
        trainer=build_donor_replay(cfg,self.registered,ckpt,sha,device=self.device,
            evaluator_pin=self.evaluator_pin,mature=mature)
        try:
            replay=self.evaluate(trainer)
            original=json.loads(Path(candidate['evaluation_path']).read_text())
            if replay['state_before']['model_state_sha256']!=original['state_before']['model_state_sha256']:
                raise ValueError('Imported checkpoint tensors differ from official source state')
            residual={}
            for split in ('rr','fr'):
                a={r['scene_id']:r['metrics'] for r in original[split]['per_scene']}
                b={r['scene_id']:r['metrics'] for r in replay[split]['per_scene']}
                if a.keys()!=b.keys() or len(a)!=20:
                    raise ValueError('Imported full40 official scene inventory changed')
                errors=[]
                for scene,metrics in a.items():
                    if metrics.keys()!=b[scene].keys():
                        raise ValueError('Imported official metric inventory changed')
                    for key,x in metrics.items():
                        y=b[scene][key]; errors.append(abs(x-y))
                        if not np.isclose(x,y,rtol=1e-6,atol=1e-7):
                            raise ValueError(f'Imported replay differs: {split}/{scene}/{key}: {x} -> {y}')
                residual[split]=max(errors,default=0.)
            atomic_json(replay_path,replay)
            row={**deepcopy(candidate),'checkpoint_path':str(ckpt),'checkpoint_sha256':sha,
                'source_server':self.server,'target_campaign_id':plan.CAMPAIGN_ID,'version':label,
                'mature':mature,'reloaded_full40_verified':True,
                'replay_sha256':file_sha(replay_path),'replay_max_abs_residual':residual,
                'HQNR':replay['fr']['metrics']['HQNR'],'ERGAS':replay['rr']['metrics']['ERGAS'],
                'exploratory':replay['fr']['metrics']['HQNR']<.959,
                'binding_scope':'FIXED_LOCAL_LINEAGE_NOT_CROSS_SERVER_SELECTOR'}
            row['binding_sha256']=plan.canonical_sha256(row)
            atomic_json(path,row); self.state['donors'][label]=row; self.persist()
            return row
        finally:
            del trainer; self.release()

    def prepare_donors(self):
        from .donors import discover_pair
        inventory=self.root/'donor_inventory.json'
        if inventory.exists():
            pair=json.loads(inventory.read_text())['pair']
        else:
            pair=discover_pair(self.server,root=plan.ROOT)
            if pair is None:
                cfg=plan.resolve_case(self.server,'FRESH_A1')
                self.preflight_cases([cfg])
                # Reserve core Student prior plus T maturation before a fallback.
                core=[plan.resolve_case(self.server,c,r) for g in plan.groups(self.server)
                      if not g['optional'] for c,r in g['runs']]
                needed=(self.estimate(cfg)+sum(self.estimate(c) for c in core)+7200+1800)*1.2
                if needed>self.remaining_for_work():
                    raise BudgetExpired('FRESH_LINEAGE_PLUS_MATURATION_AND_CORE_DO_NOT_FIT')
                self.run_case(cfg)
                run=self.completed(cfg); directory=Path(run['directory'])
                selected=json.loads((directory/'models'/'selection.json').read_text())['best_selected']
                early=self.candidate(directory,selected['completed_update'])
                late=self.candidate(directory,50000)
                pair={'early':early,'late':late,'origin':'ONE_FRESH_LOCAL_A1'}
            atomic_json(inventory,{'pair':pair,'created_at':utc_now()})
        self.bind_import('E',pair['early'])
        if self.server=='s1':
            self.bind_import('L',pair['late'])
        early=self.state['donors']['E']
        if early['completed_update']>=50000:
            raise ValueError('Registered early/late experiment requires an early source checkpoint <50K')
        cfg=plan.resolve_case(self.server,'MATURE',source_update=early['completed_update'])
        self.preflight_cases([cfg])
        if not self.completed(cfg):
            core=[plan.resolve_case(self.server,c,r) for g in plan.groups(self.server)
                  if not g['optional'] for c,r in g['runs']]
            needed=(self.estimate(cfg)+sum(self.estimate(c) for c in core)+1800)*1.2
            if needed>self.remaining_for_work():
                raise BudgetExpired('MATURATION_AND_CORE_SEED_PAIRS_DO_NOT_FIT')
        self.run_case(cfg)
        directory=Path(self.completed(cfg)['directory'])
        self.bind_import('M',self.candidate(directory,cfg['planned_horizon'],lifetime=50000),mature=True)

    def candidate(self,directory,step,*,lifetime=None):
        directory=Path(directory); cfg=json.loads((directory/'config.json').read_text())
        artifact=json.loads((directory/'models'/f'step_{step:07d}.json').read_text())['artifact']
        evidence=directory/f'evaluation_{step:07d}.json'
        return {'checkpoint_path':str(directory/'models'/f'step_{step:07d}.pt'),
            'checkpoint_sha256':artifact['file_sha256'],'file_sha256':artifact['file_sha256'],
            'source_config':cfg,'source_server':self.server,'completed_update':step if lifetime is None else lifetime,
            'evaluation_path':str(evidence),'source':'THIS_CAMPAIGN_COMPLETE_STATE'}

    def calibration(self,label):
        from panda_stage2_condq.runtime import calibrate
        from panda_stage2_condq.calibration import CalibrationStats
        path=self.root/'calibration'/f'{label}_MASK2.npz'; receipt_path=path.with_suffix('.json')
        self.check(); prior=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        donor=self.load_donor(label)
        try:
            if path.exists():
                sha=file_sha(path)
                if prior is not None and sha!=prior.get('file_sha256'):
                    raise ValueError('Immutable calibration file SHA changed')
                stats=CalibrationStats.load(path,expected_binding=donor.binding,
                    expected_file_sha256=sha,expected_mask_policy=plan.LOSS_MASK_POLICY,
                    memory_budget_bytes=3*1024**3)
                if stats.calibration_kind!='exact_condq' or stats.metadata['g_grouping']['k_requested']!=5:
                    raise ValueError('Exact train-only conditional-q K5 required')
                saved={**(prior or {}),'path':str(path),'sha256':sha,'file_sha256':sha,
                    'artifact_sha256':stats.artifact_sha256,'donor_binding':donor.binding,
                    'mask_policy':plan.LOSS_MASK_POLICY,'tau_e':stats.tau_e,
                    'sample_count':stats.metadata['sample_count']}
                if prior and any(prior.get(k)!=saved[k] for k in
                                 ('artifact_sha256','donor_binding','mask_policy')):
                    raise ValueError('Calibration receipt/binding differs')
                del stats
            else:
                if prior:
                    raise ValueError('Calibration receipt exists without complete file')
                if self.remaining_for_work()<self.calibration_overhead(label):
                    raise BudgetExpired('Missing full-train calibration does not fit')
                donor.aligner.to(self.device); donor.teacher.to(self.device)
                def progress(info):
                    self.check()
                    if info['seen']%600==0 or info['seen']==info['expected']:
                        print(f'[CALIBRATION {label}] {info}',flush=True)
                saved=calibrate(donor,self.registered,output=path,k=5,batch_size=12,
                    memory_budget_bytes=3*1024**3,include_condq=True,mask_policy=plan.LOSS_MASK_POLICY,
                    device=self.device,on_progress=progress)
                saved['sha256']=saved['file_sha256']
            atomic_json(receipt_path,saved)
            self.state.setdefault('calibration_timing',{})[label]=saved.get('elapsed_seconds',600.)
            return path,saved
        finally:
            del donor; self.release()

    def calibration_overhead(self,label):
        if (self.root/'calibration'/f'{label}_MASK2.npz').exists():
            return 0.
        return max([600.,*[float(x) for x in self.state.get('calibration_timing',{}).values()]])

    def parent(self,cfg):
        parent=plan.resolve_case(self.server,cfg['parent_case'],cfg['parent_replica'])
        status=self.completed(parent)
        if status is None:
            raise ValueError('Tail requires its same-seed completed M_Q parent')
        row=self.candidate(Path(status['directory']),cfg['parent_update'])
        if file_sha(row['checkpoint_path'])!=row['file_sha256']:
            raise ValueError('Immutable retained fork checkpoint changed')
        cfg.update(parent_run=parent['run_id'],parent_checkpoint=row['checkpoint_path'],
            parent_checkpoint_sha256=row['file_sha256'],parent_binding={
                'run_id':parent['run_id'],'seed':parent['seed'],'step':cfg['parent_update'],
                'file_sha256':row['file_sha256']})
        return row

    def build(self,cfg):
        from . import engine
        kind=cfg['engine_kind']; stream=None
        if kind=='fresh_donor':
            from panda40.stage1 import build_stage1,Stage1Stream
            trainer=build_stage1(cfg,self.sources)
            factory=lambda:Stage1Stream(self.sources,cfg,device=self.device)
        elif kind=='mature_teacher':
            donor=self.state['donors']['E']
            cfg.update(parent_run=donor['source_config']['run_id'],
                donor_checkpoint=donor['checkpoint_path'],donor_checkpoint_sha256=donor['checkpoint_sha256'],
                parent_binding={'file_sha256':donor['checkpoint_sha256'],'step':cfg['start_update']})
            trainer,stream=engine.build_mature_trainer(cfg,self.registered,donor['checkpoint_path'],
                donor['checkpoint_sha256'],evaluator_pin=self.evaluator_pin)
        elif kind.startswith('tail_'):
            row=self.parent(cfg)
            trainer,stream=engine.build_tail_trainer(cfg,self.registered,row['checkpoint_path'],
                row['file_sha256'],evaluator_pin=self.evaluator_pin)
        elif kind=='stage2_condq':
            from panda_stage2_condq.calibration import CalibrationStats
            from panda_stage2_condq.stream import TrainStream,StreamSpec
            label=cfg['donor_version']; uses_teacher=cfg['stage2_spec']['variant']!='gt_only'
            calpath=receipt=stats=None
            if uses_teacher:
                calpath,receipt=self.calibration(label)
            donor=self.load_donor(label,include_teacher=uses_teacher)
            if uses_teacher:
                stats=CalibrationStats.load(calpath,expected_binding=donor.binding,
                    expected_file_sha256=receipt['sha256'],expected_mask_policy=plan.LOSS_MASK_POLICY)
            selected=self.state['donors'][label]
            cfg.update(donor_checkpoint=selected['checkpoint_path'],
                donor_checkpoint_sha256=selected['checkpoint_sha256'],donor_binding=donor.binding,
                donor_exploratory=selected['exploratory'],parent_run=selected['source_config']['run_id'],
                calibration_path=None if calpath is None else str(calpath),
                calibration_sha256=None if receipt is None else receipt['sha256'])
            trainer=engine.build_fresh_stage2(cfg,donor,stats,evaluator_pin=self.evaluator_pin)
            spec=StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation'])
            factory=lambda:TrainStream(self.registered,spec,microbatch_size=12,effective_batch_size=48,device=self.device)
        else:
            raise ValueError('Unsupported registered engine')
        from .cost import ensure_cost
        # The disposable profiler must not perturb a continuation's RNG.
        from donor30.artifacts import capture_rng,restore_rng
        rng=capture_rng()
        try:
            ensure_cost(cfg,self.sources,retained_trainer=trainer,device=self.device,
                deadline_unix=self.deadline-self.config['budget']['closeout_seconds'],cost_root=self.root)
        finally:
            restore_rng(rng)
        trainer.stage.to(self.device)
        # Native continuation imported optimizer moments on CPU before profiling.
        for state in trainer.optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor) and key!='step':
                    state[key]=value.to(self.device)
        if kind=='stage2_condq' and trainer.stage.stats is not None:
            trainer.stage.move_calibration(self.device,memory_budget_bytes=512*1024**2)
        if stream is None:
            stream=factory()
        else:
            stream.device=self.device
        trainer.config=deepcopy(cfg)
        return trainer,stream

    def trace(self,trainer,stream):
        start=time.monotonic()
        if trainer.config['engine_kind']=='fresh_donor':
            from panda40.stage1 import advance_stage1
            result=advance_stage1(trainer,stream)
        else:
            from .engine import advance
            result=advance(trainer,stream)
        self.sync()
        return result,time.monotonic()-start

    @staticmethod
    def timing_key(cfg):
        if cfg['engine_kind']=='stage2_condq':
            return 'stage2_gt' if cfg['stage2_spec']['variant']=='gt_only' else 'stage2_q'
        return cfg['engine_kind']

    def preflight_cases(self,cases):
        for base in cases:
            key=self.timing_key(base)
            if key in self.state['timing']:
                continue
            cfg=plain(base); cfg['disposable_preflight']=True
            trainer,stream=self.build(cfg)
            try:
                times=[]
                for _ in range(min(100,cfg['planned_horizon'])):
                    self.check(); _,seconds=self.trace(trainer,stream); times.append(seconds)
                result=self.evaluate(trainer)
                from .reporting import make_record,validate_record
                started=time.monotonic()
                with tempfile.TemporaryDirectory(prefix='.preflight-',dir=self.root) as temp:
                    saved=self._save(trainer,stream,Path(temp)/'disposable.pt',cfg,result)
                    validate_record(make_record(cfg,result,saved,root=self.root))
                    size=Path(saved['checkpoint_path']).stat().st_size
                    # Real load, not just serialization, before any production run.
                    if cfg['engine_kind']!='fresh_donor':
                        from .engine import load_checkpoint
                        load_checkpoint(saved['checkpoint_path'],saved['file_sha256'],trainer,stream,
                            code_pin=self.code,evaluator_pin=self.evaluator_pin)
                self.state['timing'][key]={'update_p90':float(np.quantile(times[10:] or times,.9)),
                    'eval_seconds':result['elapsed_seconds'],
                    'checkpoint_seconds':max(10.,time.monotonic()-started),'checkpoint_bytes':size}
                atomic_json(self.root/f'preflight_{key}.json',{'timing':self.state['timing'][key],
                    'evaluation':result,'published_to_outbox':False,'weights_discarded':True,
                    'eligible_for_selection':False,'checkpoint_roundtrip_verified':True})
                self.persist()
            finally:
                del trainer,stream; self.release()

    def estimate(self,cfg):
        if self.completed(cfg): return 0.
        completed=0
        status=next((r for r in self.state['runs'] if r['run_id']==cfg['run_id']),None)
        if status:
            latest=Path(status['directory'])/'models'/'latest_resume.json'
            if latest.exists():
                completed=json.loads(latest.read_text())['completed_update']
                if type(completed) is not int or not 0<=completed<=cfg['planned_horizon']:
                    raise ValueError('Invalid durable resume progress')
        remaining=cfg['planned_horizon']-completed
        timing=self.state['timing'].get(self.timing_key(cfg))
        if timing is None:
            # Historical measured local totals are priors, never claimed as this GPU's speed.
            key=self.timing_key(cfg)
            seconds=1.6*3600 if key in ('stage2_gt','tail_gt') else 2.1*3600
            if key=='fresh_donor': seconds=2.6*3600
            if key=='mature_teacher': seconds=2.1*3600
            if key.startswith('tail_fc2'): seconds=3.6*3600
            return seconds*remaining/50000
        return (remaining*timing['update_p90']+
                sum(s>=completed for s in cfg['evaluation_steps'])*(timing['eval_seconds']+timing['checkpoint_seconds']))

    def completed(self,cfg):
        return next((r for r in self.state['runs'] if r['run_id']==cfg['run_id'] and r['status']=='COMPLETED'),None)

    def _save(self,trainer,stream,path,cfg,evaluation=None):
        if plain(trainer.config)!=plain(cfg) or getattr(trainer,'poisoned',False):
            raise ValueError('Only matching nonpoisoned complete trainer can be checkpointed')
        path=Path(path)
        if cfg['engine_kind']=='fresh_donor':
            from panda40.stage1 import save_stage1_checkpoint
            saved=save_stage1_checkpoint(trainer,stream,path,cfg,evaluation,
                code_pin=self.code,evaluator_pin=self.evaluator_pin)
        else:
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
        if cfg['engine_kind']=='fresh_donor':
            from panda40.stage1 import load_stage1_checkpoint
            return load_stage1_checkpoint(path,artifact['file_sha256'],cfg,self.sources,
                device=self.device,code_pin=self.code,evaluator_pin=self.evaluator_pin,
                restore_global_rng=True)[:2]
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

    def admit(self,groups,*,name):
        cases=[plan.resolve_case(self.server,c,r) for g in groups for c,r in g['runs']]
        pending=[c for c in cases if not self.completed(c)]
        if not pending:
            row={'name':name,'estimated_seconds':0.,'calibration_seconds':0.,'estimated_disk_bytes':0.,
                'remaining_seconds':self.remaining_for_work(),'pending_runs':0,'status':'ADMITTED','at':utc_now()}
            self.state['groups'].append(row); self.persist(); return row
        labels={c['donor_version'] for c in pending if c['engine_kind']=='stage2_condq'
                and c['stage2_spec']['variant']!='gt_only'}
        calibration=sum(self.calibration_overhead(label) for label in labels)
        seconds=(sum(self.estimate(c) for c in pending)+calibration)*self.config['budget']['safety_factor']
        # Checkpoint every official evaluation, not just winning scalars.
        sizes=[t.get('checkpoint_bytes',64*1024**2) for t in self.state['timing'].values()]
        disk=sum(len(c['evaluation_steps']) for c in pending)*max(sizes or [64*1024**2])*1.2+5*1024**3
        okay=seconds<=self.remaining_for_work() and disk<=shutil.disk_usage(self.root).free
        row={'name':name,'estimated_seconds':seconds,'calibration_seconds':calibration,
            'estimated_disk_bytes':disk,'remaining_seconds':self.remaining_for_work(),
            'pending_runs':len(pending),'status':'ADMITTED' if okay else 'SKIPPED_BUDGET_OR_DISK','at':utc_now()}
        self.state['groups'].append(row); self.persist(); return row

    def run(self,*,catalog=plan.ROOT/'ablr2/sensor_sources.json'):
        with (self.root/'.controller.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            old={s:signal.signal(s,lambda *a:setattr(self,'stop_requested',True)) for s in (signal.SIGINT,signal.SIGTERM)}
            try:
                self.persist(); self.admission(catalog); self.prepare_donors()
                groups=plan.groups(self.server)
                # Independent common G/Q pilots; calibration/profiling are in wall budget.
                pilots=[plan.resolve_case(self.server,'M_G'),plan.resolve_case(self.server,'M_Q')]
                self.preflight_cases(pilots)
                core=[g for g in groups if not g['optional']]
                reserved=self.admit(core,name='ALL_CORE_PAIRED_RUNS')
                if reserved['status']!='ADMITTED':
                    raise BudgetExpired('MEASURED_CORE_SEED_PAIRS_DO_NOT_FIT')
                skipped_blocks=set()
                for index,group in enumerate(groups):
                    self.check()
                    cases=[plan.resolve_case(self.server,c,r) for c,r in group['runs']]
                    if all(self.completed(c) for c in cases): continue
                    block=group.get('budget_block',group['name'])
                    if block in skipped_blocks: continue
                    block_groups=[g for g in groups[index:] if g.get('budget_block',g['name'])==block]
                    if group['optional']:
                        first=self.admit(block_groups,name=block+'_BEFORE_PILOT')
                        if first['status']!='ADMITTED':
                            skipped_blocks.add(block); continue
                    self.preflight_cases(cases)
                    # Reserve remaining core, not merely the next pair, after updated timing.
                    remaining_core=[g for g in groups[index:] if not g['optional']]
                    admitted=self.admit(remaining_core or block_groups,name=group['name'])
                    if admitted['status']!='ADMITTED':
                        if not group['optional']: raise BudgetExpired('CORE_NO_LONGER_FITS:'+group['name'])
                        skipped_blocks.add(block); continue
                    for cfg in cases: self.run_case(cfg)
                    admitted['status']='COMPLETED'; self.persist()
                reserved['status']='COMPLETED'; self.state['status']='FINISHED_REGISTERED_QUEUE'
                self.state['comparison_completeness']={g['name']:{
                    'expected_replicas':[r for _,r in g['runs']],
                    'completed_replicas':[r for c,r in g['runs'] if self.completed(plan.resolve_case(self.server,c,r))],
                    'complete':all(self.completed(plan.resolve_case(self.server,c,r)) for c,r in g['runs'])}
                    for g in groups}
            except BudgetExpired as exc:
                self.state.update(status='STOPPED_BUDGET',reason=str(exc))
            except Exception as exc:
                self.state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc()); raise
            finally:
                self.persist()
                for sig,handler in old.items(): signal.signal(sig,handler)
            return self.state
