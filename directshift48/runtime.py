"""Prior-free DirectShift48 controller; isolated from live EdgeRes campaigns."""
from __future__ import annotations
from collections import deque
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from panda_method.edge_data import EdgeDataSources
from panda_method.reconstruction import ReconstructionSpec
from .training import build_direct_stage1, DirectStage1Trainer
from ta2.evaluation import digest, evaluator_identity, file_sha, JQM_VARIANT
from . import plan
from .evaluation import (official_evaluation, retain_qualified, expected_ids,
                         source_manifest, model_state_hash)
from .reporting import LocalOutbox, build_records_split, atomic_json, append_jsonl, utc_now
from edgeres48.stream import RRStream, set_learning_rates

ROOT = Path(__file__).resolve().parents[1]


def numeric_leaves(value, prefix=''):
    """Bounded scalar paths for complete nested summaries in fixed64 notes."""
    if isinstance(value,dict):
        if not value:
            yield prefix, {}
        for key,item in value.items():
            yield from numeric_leaves(item,prefix+'/'+str(key))
    elif isinstance(value,list):
        if not value:
            yield prefix, []
        for index,item in enumerate(value):
            yield from numeric_leaves(item,prefix+'/'+str(index))
    else:
        yield prefix,value


def build_trainer(config, sources, device):
    plan.validate_method_config(config)
    b, seeds = config['baseline_resolved'], config['stream_seeds']
    stage = build_direct_stage1(sources,
        teacher_spec=ReconstructionSpec(**b['teacher_architecture'], input_layout='P0', base_reference='BICUBIC_MS'),
        aligner_view_margin=b['aligner_architecture']['view_margin'], roi_margin=b['absolute_roi_margin'],
        teacher_seed=seeds['T_init'], aligner_seed=seeds['A_init'], epsilon_seed=seeds['rr_epsilon'],
        boundary_policy=b['boundary_policy'], lambda_abs=config['lambda_abs'], lambda_eps=config['lambda_epsilon']).to(device)
    optimizer = torch.optim.AdamW([
        {'params': stage.aligner.parameters(), 'lr': config['lr_A'], 'role': 'A'},
        {'params': stage.teacher.parameters(), 'lr': config['lr_U'], 'role': 'T'}],
        betas=tuple(b['optimizer_betas']), eps=b['optimizer_eps'], weight_decay=config['weight_decay'])
    trainer = DirectStage1Trainer(stage, optimizer, gradient_log_interval=b['gradient_log_interval'],
        effective_batch_size=b['effective_batch_RR'], microbatch_size=b['microbatch_RR'])
    trainer.config = config
    return trainer


def code_pin():
    paths = [*ROOT.glob('directshift48/*.py'), *ROOT.glob('panda_method/*.py'),
             *ROOT.glob('sheet_publish_v2/*.py'), ROOT/'edgeres48/stream.py',
             ROOT/'pa/warp.py', ROOT/'pa/offset.py', ROOT/'pa/aligner.py',
             ROOT/'model/pancrafter_paper.py', ROOT/'model/pancrafter.py',
             ROOT/'fh12/model.py', ROOT/'ta2/evaluation.py']
    return {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(paths)}


def environment_manifest(device):
    """Record explicit runtime facts only; never dump environment credentials."""
    cuda = str(device).startswith('cuda') and torch.cuda.is_available()
    return {'python':sys.version, 'platform':platform.platform(), 'torch':str(torch.__version__),
        'numpy':np.__version__, 'cuda_build':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(), 'device':str(device),
        'device_name':torch.cuda.get_device_name(device) if cuda else platform.processor(),
        'torch_cpu_threads':torch.get_num_threads(), 'torch_interop_threads':torch.get_num_interop_threads(),
        'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'), 'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS'),
        'MKL_THREADING_LAYER':os.environ.get('MKL_THREADING_LAYER'),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'training_policy':{'amp':False,'tf32':False,'cudnn_benchmark':False},
        'scope':'Environment record, not a bitwise cross-device determinism guarantee'}


def probe_spec(config, *, clean=False):
    # The baseline, not the runtime, owns fixed performance-independent row IDs.
    # DirectShift has one mandatory clean support policy, not legacy domains.
    return deepcopy(config['baseline_resolved']['probes'])


class Campaign:
    def __init__(self, output, server, baseline, *, device='cuda'):
        plan.validate_baseline(baseline)
        self.server, self.dataset = server, plan.LANES[server]['dataset']
        if baseline['dataset'] != self.dataset:
            raise ValueError('Baseline sensor/server mismatch')
        self.baseline, self.device = deepcopy(baseline), device
        self.root = Path(output).resolve()/plan.CAMPAIGN_ID/server/self.dataset
        self.root.mkdir(parents=True, exist_ok=True)
        self.outbox = LocalOutbox(self.root)
        self.sources = None
        self.evaluator_pin = evaluator_identity()
        self.code = code_pin()
        self.state_path = self.root/'campaign_state.json'
        self.state = {'runs': [], 'observations': [], 'timing': {}, 'created_at': utc_now()}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        self.sequence = len(list(self.outbox.outbox.glob('DS3-*.json')))
        self.latencies = deque(maxlen=2048)
        self.state.setdefault('decisions', {})
        self.provenance = {'code_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
            'config_sha256': digest(baseline), 'evaluator_revision': digest(self.evaluator_pin),
            'dataset_manifest_sha256': None}

    def persist(self):
        atomic_json(self.state_path, self.state)

    def verify_code(self):
        if code_pin() != self.code:
            raise RuntimeError('Campaign executable code changed after admission; no mixed-code continuation')

    def emit(self, config, status, *, step=0, scope='STATUS', selector=None, metrics=None,
             state_sha=None, checkpoint_sha=None, extra=None, fields=None):
        self.sequence += 1
        records = build_records_split(server=self.server, dataset=self.dataset, run_id=config['run_id'],
            case_id=config['case'], selector=selector or f'{status}_{self.sequence:08d}',
            eval_scope=scope, eval_protocol_id=digest(self.evaluator_pin),
            checkpoint_sha=checkpoint_sha, model_state_sha=state_sha, metrics=metrics,
            fields={'Status': status, 'Repeat': int(config['replica'][1:]), 'Seed': config['seed'],
                'Attempt': config['attempt'], 'Updates': step, 'Lifetime_updates': step, 'Selected_step': step,
                'U_peak_lr': config['lr_U'], 'A_peak_lr': config['lr_A'],
                'Sort_case': int(config['loss_case'][2:]), 'JQM_variant': JQM_VARIANT if metrics else None,
                'Model / inputs': json.dumps({'teacher': self.baseline['teacher_architecture'],
                    'input': 'P0/BICUBIC_MS', 'alignment_reference': 'PROVIDED_STORED_LMS'}), **(fields or {})},
            extra={'configuration': config, **(extra or {})}, provenance=self.provenance)
        for record in records:
            self.outbox.enqueue(record)
        return records[0]['record_id']

    def admission(self, catalog_path):
        plan_path = ROOT/plan.PLAN_RELATIVE_PATH
        method_path = ROOT/'research_log'/plan.METHOD_FILE_NAME
        if file_sha(plan_path) != plan.PLAN_SHA256 or file_sha(method_path) != plan.METHOD_SHA256:
            raise ValueError('Method/experiment document changed after configuration pin')
        if file_sha(catalog_path) != self.baseline['source_registry_sha256']:
            raise ValueError('Source registry differs from the baseline SHA256 pin')
        overrides = None
        override_path = self.baseline.get('lms_override_path')
        if override_path:
            path = ROOT/override_path
            if file_sha(path) != self.baseline.get('lms_override_sha256'):
                raise ValueError('LMS override differs from the baseline SHA256 pin')
            overrides = json.loads(path.read_text())['splits']
        elif self.dataset == 'QB':
            raise ValueError('QB requires explicitly pinned user-selected original distributed LMS')
        self.sources = EdgeDataSources(catalog_path, root=ROOT, sensor=self.dataset, lms_overrides=overrides)
        # Force full source hash/shape checks before starting the independent clock.
        for split in ('train', 'val', 'rr', 'fr'):
            self.sources.raw(split, [0])
        manifest = source_manifest(self.sources)
        self.provenance['dataset_manifest_sha256'] = digest(manifest)
        environment = environment_manifest(self.device)
        self.provenance['environment_sha256'] = digest(environment)
        pin = {'baseline': self.baseline, 'sources': manifest, 'evaluator': self.evaluator_pin,
               'code': self.code, 'environment':environment}
        old = self.root/'admission.json'
        if old.exists() and json.loads(old.read_text()) != pin:
            raise ValueError('Existing campaign admission changed; use a new output directory, never mix protocols')
        self.verify_code()
        source_files = {**self.code, **self.evaluator_pin.get('files',{}),
                        str(plan_path.relative_to(ROOT)):plan.PLAN_SHA256,
                        str(method_path.relative_to(ROOT)):plan.METHOD_SHA256}
        snapshot = self.root/'method_source'
        for relative, sha in source_files.items():
            source, target = ROOT/relative, snapshot/relative
            if file_sha(source) != sha:
                raise ValueError('Code/doc changed during source snapshot: '+relative)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if file_sha(target) != sha:
                    raise ValueError('Existing immutable source snapshot changed: '+relative)
            else:
                shutil.copyfile(source,target)
            if file_sha(target) != sha:
                raise ValueError('Copied source snapshot failed SHA256 verification: '+relative)
        atomic_json(snapshot/'manifest.json',{'files':source_files,'sha256':digest(source_files)})
        atomic_json(old, pin)
        atomic_json(self.root/'environment.json', environment)
        atomic_json(self.root/'catalog.json', plan.catalog(self.server))
        return pin

    def elapsed(self):
        return time.time()-self.state['started_unix']

    @property
    def deadline(self):
        return self.state['started_unix']+48*3600

    def sync(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)

    def trace(self, config, trainer, stream, directory):
        if trainer.updates % 100 == 0:
            self.verify_code()
        started = time.monotonic()
        lr = set_learning_rates(trainer, config)
        batches = stream.next_update()
        row = trainer.step_accumulated(batches)
        self.sync()
        seconds = time.monotonic()-started
        self.latencies.append(seconds)
        append_jsonl(directory/'train.jsonl', {'step': trainer.updates, 'lr': lr,
            'seconds_including_IO': seconds, **row})
        if trainer.updates == 1 or trainer.updates % 100 == 0 or row.get('term_gradient_norm') is not None:
            summary = {'step': trainer.updates, 'lr': lr, 'update_seconds_p90': float(np.quantile(self.latencies,.9)),
                'loss': row.get('loss'), 'gradient_norm': row.get('gradient_norm'),
                'term_gradient_norm': row.get('term_gradient_norm'),
                'term_gradient_cosine': row.get('term_gradient_cosine'),
                'optimizer_update_norm': row.get('optimizer_update_norm'),
                'gradient_parameter_state_update':trainer.updates-1,
                'attached_model_hash_parameter_state_update':trainer.updates,
                'gradient_and_hash_note':'Gradients are evaluated on pre-update parameters; attached model SHA identifies post-update state.',
                **{k:row.get(k) for k in ('rec','abs_rr','weighted_abs','eps_rr','weighted_eps_rr')}}
            append_jsonl(directory/'train_summary.jsonl', summary)
            self.emit(config, 'TRAIN_TRACE', step=trainer.updates, scope='TRACE_100_UPDATE_SUMMARY',
                state_sha=model_state_hash(trainer.stage), extra={'numeric_observations': summary,
                'weights_saved': False, 'diagnostics_only': True, 'eligible_for_selection': False})
            print(f"[{config['case']}/{config['replica']}] step={trainer.updates} loss={row.get('loss')} "
                  f"p90={summary['update_seconds_p90']:.3f}s", flush=True)
        return row

    def diagnostics(self, trainer, config, directory, *, clean=False):
        self.verify_code()
        from .diagnostics import run_probes
        start = time.monotonic()
        result = run_probes(trainer, self.sources, probe_spec(config, clean=clean),
            device=self.device, evaluator_pin=self.evaluator_pin, deadline_unix=self.deadline)
        atomic_json(directory/f'diagnostics_{trainer.updates:07d}.json', result)
        # Inline numeric leaves, not inaccessible JSON links or overloaded metric columns.
        # Batch bounded groups retain exact scalars and domain names in fixed64 notes.
        def leaves(value, prefix=''):
            if isinstance(value, dict):
                for k,v in value.items(): yield from leaves(v, prefix+'/'+str(k))
            elif isinstance(value, list):
                for i,v in enumerate(value): yield from leaves(v, prefix+'/'+str(i))
            else: yield (prefix,value)
        all_values = list(leaves(result))
        for offset in range(0,len(all_values),60):
            self.emit(config, 'DIAGNOSTIC', step=trainer.updates, scope='DIAGNOSTIC_LEAN',
                state_sha=model_state_hash(trainer.stage),
                extra={'numeric_observations': dict(all_values[offset:offset+60]), 'weights_saved': False,
                       'diagnostics_only': True, 'eligible_for_selection': False})
        if result.get('status') != 'COMPLETE':
            raise RuntimeError('Required diagnostic incomplete; see exact local diagnostic report')
        duration = time.monotonic()-start
        self.state['timing']['probe_seconds'] = max(self.state['timing'].get('probe_seconds', 0), duration)
        return duration

    def evaluate(self, trainer, config, stream, directory, *, eligible=True):
        self.verify_code()
        record, outputs = official_evaluation(trainer, config, self.sources,
            device=self.device, evaluator_pin=self.evaluator_pin, deadline_unix=self.deadline)
        gate = record['qualification']
        if not eligible:
            gate = {**gate, 'joint_pass': False, 'full_method_eligible': False,
                    'selection': 'DISPOSABLE_PREFLIGHT_NOT_ELIGIBLE'}
            record['qualification'] = gate
            record['full_method_eligible'] = False
        atomic_json(directory/f'evaluation_{trainer.updates:07d}.json', record)
        identity = record['state_before']; sha = identity['model_state_sha256']
        merged = {**record['rr']['metrics'], **record['fr']['metrics']}
        rid = self.emit(config, 'QUALIFIED' if gate['joint_pass'] else 'OBSERVED', step=trainer.updates,
            scope='EVAL_SUMMARY' if eligible else 'PREFLIGHT_EVAL', metrics=merged, state_sha=sha,
            extra={**gate, 'same_state_verified': gate['same_state_verified'], 'identity': identity,
                   'weights_saved': False, 'checkpoint_path': None, 'source_evaluation': str(directory/f'evaluation_{trainer.updates:07d}.json')})
        for split in ('rr','fr'):
            for scene in record[split]['per_scene']:
                self.emit(config, 'OBSERVED', step=trainer.updates, scope=f"{split.upper()}_SCENE:{scene['scene_id']}",
                    metrics=scene['metrics'], state_sha=sha, extra={'identity': identity, 'summary_record_id': rid,
                        'weights_saved': False, 'checkpoint_path': None})
        if gate['joint_pass']:
            artifact_started = time.monotonic()
            try:
                saved = retain_qualified(directory/'qual', trainer, config, self.sources, record, outputs,
                    stream_states={'rr': stream.state_dict()}, provenance={**self.provenance,
                        'method_source_snapshot':str(self.root/'method_source'),
                        'method_source_manifest_sha256':file_sha(self.root/'method_source/manifest.json')},
                    deadline_unix=self.deadline)
            except Exception as exc:
                self.emit(config,'TARGET_PASS_ARTIFACT_FAILED',step=trainer.updates,scope='QUAL_ARTIFACT',
                    state_sha=sha,extra={'joint_pass':True,'error':f'{type(exc).__name__}: {exc}',
                    'snapshot_may_have_been_saved':True,'artifact_verified':False})
                raise
            self.emit(config, 'QUAL_ARTIFACT', step=trainer.updates, scope='QUAL_ARTIFACT', state_sha=sha,
                checkpoint_sha=saved.get('checkpoint_file_sha256'), extra={'artifact': saved, 'joint_pass': True})
            self.state['timing']['artifact_seconds'] = max(self.state['timing'].get('artifact_seconds',0),
                time.monotonic()-artifact_started)
            self.state['timing']['artifact_timing_measured'] = True
            if config['run_id'] not in self.state.setdefault('cost_runs',[]):
                try:
                    self.benchmark(trainer,config,directory)
                    self.state['cost_runs'].append(config['run_id'])
                except Exception as exc:
                    self.emit(config,'COST_FAILED',step=trainer.updates,extra={'error':f'{type(exc).__name__}: {exc}'})
        observation = {**gate, 'run_id': config['run_id'], 'attempt': config['attempt'],
            'completed_update': trainer.updates, 'record_id': rid, 'model_state_sha256': sha,
            'evaluation_seconds': record.get('wall_seconds', record.get('elapsed_seconds'))}
        print(f"[{config['case']}/{config['replica']}] step={trainer.updates} "
              f"HQNR={merged['HQNR']:.6f} SCC={merged['SCC']:.6f} ERGAS={merged['ERGAS']:.6f} "
              f"qualified={gate['joint_pass']}", flush=True)
        self.state['timing']['rr_fr_eval_seconds'] = max(self.state['timing'].get('rr_fr_eval_seconds', 0),
            record.get('elapsed_seconds', record.get('wall_seconds', 0)))
        return observation

    def benchmark(self, trainer, config, directory):
        from .diagnostics import benchmark_inference
        self.verify_code()
        result = benchmark_inference(trainer,self.sources,device=self.device,warmup=3,repeats=10,
                                     rows={'rr':0,'fr':0},deadline_unix=self.deadline)
        atomic_json(directory/f'cost_{trainer.updates:07d}.json',result)
        if result.get('status') != 'COMPLETE':
            raise RuntimeError('Native inference benchmark incomplete; no successful cost claim')
        for split, values in result['observations'].items():
            ms = values['full_pipeline']['median_ms']
            self.emit(config,'COST',step=trainer.updates,scope='COST_'+split.upper(),
                state_sha=model_state_hash(trainer.stage),fields={'Infer(ms)':ms,
                    'Params(M)':result['trainable_params']/1e6,
                    'Cost_scope':result['Infer_ms_scope']+f'; {split.upper()} batch1 FP32 device={self.device}'},
                extra={'numeric_observations':values,'metadata':{k:v for k,v in result.items() if k!='observations'},
                       'weights_saved':False,'diagnostics_only':True,'eligible_for_selection':False})
        return result

    def preflight(self):
        config = plan.resolve_case(self.server, baseline=self.baseline)
        directory = self.root/'preflight'; directory.mkdir(exist_ok=True)
        from .smoke import verify_method_contracts
        numerical = verify_method_contracts()
        atomic_json(directory/'numerical_contracts.json', numerical)
        if numerical['status'] != 'COMPLETE':
            raise RuntimeError('DirectShift numerical contracts failed; no official training started')
        config = {**config, 'disposable_preflight': True}
        atomic_json(directory/'config.json', config)
        trainer = build_trainer(config, self.sources, self.device)
        stream = RRStream(self.sources, config, device=self.device)
        self.emit(config, 'PREFLIGHT_STARTED', extra={'disposable_weights': True})
        for _ in range(300):
            if self.elapsed() > 2*3600:
                raise TimeoutError('Disposable timing exceeded 2h; no official run started')
            self.trace(config, trainer, stream, directory)
        start = time.monotonic()
        self.evaluate(trainer, config, stream, directory, eligible=False)
        eval_seconds = time.monotonic()-start
        probe_seconds = self.diagnostics(trainer, config, directory, clean=True)
        diagnostic_path=directory/f'diagnostics_{trainer.updates:07d}.json'
        if not diagnostic_path.exists() or json.loads(diagnostic_path.read_text()).get('status')!='COMPLETE':
            raise RuntimeError('Disposable preflight diagnostics incomplete; no official run admitted')
        self.benchmark(trainer,config,directory)
        timing = {'update_p90_seconds': float(np.quantile(self.latencies,.9)),
            'rr_fr_eval_seconds': eval_seconds, 'probe_seconds': probe_seconds,
            'artifact_seconds': self.baseline.get('artifact_reserve_seconds',300.),
            'effective_updates':300, 'disposable':True, 'artifact_timing_measured':False,
            'preflight_complete':True}
        self.state['timing'] = timing; self.persist()
        self.emit(config, 'PREFLIGHT_PASSED', extra=timing)
        del trainer, stream
        if str(self.device).startswith('cuda'): torch.cuda.empty_cache()

    def estimate(self, config, *, updates=None):
        t = self.state['timing']; n = updates or config['planned_horizon']
        cadence = sum(i<=n for i in config['evaluation_steps'])
        return plan.estimate_remaining_seconds(remaining_updates=n,
            update_p90_seconds=max(t['update_p90_seconds'],float(np.quantile(self.latencies,.9)) if self.latencies else 0),
            remaining_evaluations=cadence, rr_fr_eval_seconds=t['rr_fr_eval_seconds'],
            artifact_seconds=cadence*t['artifact_seconds'], probe_seconds=4*t['probe_seconds'])

    def next_attempt(self, config):
        attempts = [r['attempt'] for r in self.state['runs'] if r['run_id']==config['run_id']]
        return max(attempts, default=0)+1

    def run_case(self, config, *, clean=False):
        self.verify_code()
        plan.validate_method_config(config)
        if not config['runnable']:
            raise ValueError('Never execute NOT_TRIGGERED or duplicate configurations')
        config = {**config, 'attempt': self.next_attempt(config)}
        directory = self.root/'runs'/config['run_id']/f"attempt_{config['attempt']:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory/'config.json',config)
        row = {**config, 'status':'STARTED', 'evaluations':[], 'completed_update':0}
        self.state['runs'].append(row); self.persist()
        self.emit(config, 'STARTED', extra={'configuration':config})
        trainer = stream = None
        last_eval = last_probe = -1
        try:
            trainer=build_trainer(config,self.sources,self.device)
            stream=RRStream(self.sources,config,device=self.device)
            self.diagnostics(trainer,config,directory,clean=clean); last_probe=0
            for _ in range(config['planned_horizon']):
                # Keep enough wall time for final full evaluation/diagnostic and closeout.
                reserve=7200+1.2*(self.state['timing']['rr_fr_eval_seconds']+self.state['timing']['probe_seconds'])
                if time.time()+reserve >= self.deadline:
                    row['status']='TIMEBOX_STOP'; break
                self.trace(config,trainer,stream,directory)
                row['completed_update']=trainer.updates
                if trainer.updates in config['evaluation_steps']:
                    observation=self.evaluate(trainer,config,stream,directory)
                    last_eval=trainer.updates
                    row['evaluations'].append(observation); self.state['observations'].append(observation); self.persist()
                    pruning=plan.should_prune(row,self.state['runs'][:-1],
                        stagnation_window=self.baseline['pruning']['stagnation_window'],
                        tolerance=self.baseline['pruning']['tolerance'])
                    if pruning['prune']:
                        row.update(status='PRUNED_DOMINATED',pruning=pruning); break
                if trainer.updates in (10000,25000,config['planned_horizon']):
                    self.diagnostics(trainer,config,directory,clean=clean); last_probe=trainer.updates
            else:
                row['status']='COMPLETED'
            if trainer.updates>0 and last_eval!=trainer.updates and time.time()+self.state['timing']['rr_fr_eval_seconds']<self.deadline:
                observation=self.evaluate(trainer,config,stream,directory)
                row['evaluations'].append(observation); self.state['observations'].append(observation)
            if last_probe!=trainer.updates and time.time()+self.state['timing']['probe_seconds']<self.deadline:
                self.diagnostics(trainer,config,directory,clean=clean)
        except Exception as exc:
            row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            atomic_json(directory/'failure.json', {'error':row['error'],'traceback':row['traceback'],
                'weights_saved':False,'checkpoint_path':None,'completed_update':row['completed_update']})
            print(row['traceback'],flush=True)
        finally:
            row['finished_at']=utc_now(); self.persist()
            self.emit(config,row['status'],step=row['completed_update'],extra={
                'error':row.get('error'),'weights_saved_on_failure':False,'pruning':row.get('pruning')})
            del trainer,stream
            if str(self.device).startswith('cuda'): torch.cuda.empty_cache()
        return row

    def retry_pending_artifacts(self):
        """Only already-qualified snapshots can be loaded for closeout repair."""
        from .evaluation import retry_qualified
        for path in sorted(self.root.glob('runs/**/artifact_manifest.json')):
            manifest = json.loads(path.read_text())
            if manifest.get('status') != 'QUAL_PENDING':
                continue
            if time.time()+self.state['timing'].get('artifact_seconds',300)>=self.deadline:
                break
            config = manifest['config']
            try:
                result = retry_qualified(path.parent,self.sources,
                    build_trainer=lambda c:build_trainer(c,self.sources,self.device),device=self.device,
                    deadline_unix=self.deadline)
                self.emit(config,'QUAL_ARTIFACT_RETRY',step=manifest['identity']['completed_update'],
                    scope='QUAL_ARTIFACT',state_sha=result['model_state_sha256'],
                    checkpoint_sha=result['file_sha256'],extra={'artifact':result,'joint_pass':True})
            except Exception as exc:
                self.emit(config,'TARGET_PASS_ARTIFACT_PENDING',step=manifest['identity']['completed_update'],
                    extra={'path':str(path),'error':f'{type(exc).__name__}: {exc}'})

    def completed_recipe(self, config):
        return any(r['run_id']==config['run_id'] and r['status'] in ('COMPLETED','PRUNED_DOMINATED','TIMEBOX_STOP','FAILED') for r in self.state['runs'])

    def admit(self, config, *, phase_end, remaining=2):
        selected=self.select_recipe()
        decision=plan.admit_new_run(elapsed_seconds=self.elapsed(),
            estimated_to_25k_seconds=self.estimate(config,updates=25000),estimated_full_seconds=self.estimate(config),
            selected_recipe_seconds=self.estimate(selected) if selected else None,
            remaining_confirmations=remaining, phase_end_hours=phase_end)
        self.emit(config,decision['reason'],extra=decision)
        return decision['admitted']

    def select_recipe(self):
        # No target-gap fallback: all valid official snapshots use HQNR first.
        candidates = []
        for run in self.state['runs']:
            if (run['replica'] != 'R1' or run['completed_update'] < 25000
                    or run['status'] in ('FAILED', 'ABORTED_PROCESS_RESTART_FRESH_ATTEMPT')):
                continue
            chosen = plan.select_candidate(run['evaluations'])
            if chosen is not None:
                candidates.append((run, chosen))
        return min(candidates, key=lambda pair: plan.representative_rank(pair[1]))[0] if candidates else None

    def execute(self, catalog_path):
        with (self.root/'.controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.admission(catalog_path)
            if self.state.get('finished_at'):
                raise ValueError('Campaign already closed; replay publisher or choose a new output root')
            if str(self.device).startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; no CPU training fallback')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            if 'started_unix' not in self.state:
                self.state['started_unix'] = time.time()
            for run in self.state['runs']:
                if run['status'] == 'STARTED':
                    run['status'] = 'ABORTED_PROCESS_RESTART_FRESH_ATTEMPT'
            self.persist()
            if not self.state['timing'].get('preflight_complete'):
                try:
                    self.preflight()
                except Exception as exc:
                    atomic_json(self.root/'preflight_failure.json', {'error':f'{type(exc).__name__}: {exc}',
                        'traceback':traceback.format_exc(), 'weights_saved':False, 'at':utc_now()})
                    raise
            for case in (() if 'optimizer_anchor' in self.state or 'confirmation_recipe' in self.state
                         else plan.LOSS_ORDER[self.server]):
                config = plan.resolve_case(self.server, baseline=self.baseline, loss_case=case)
                if self.completed_recipe(config):
                    continue
                if not self.admit(config, phase_end=22):
                    break
                row = self.run_case(config, clean=True)
                # A failing protected baseline is evidence of a shared defect, not
                # authorization to burn through the same defect across the menu.
                if case == 'DF00' and row['status'] == 'FAILED':
                    self.closeout()
                    raise RuntimeError('Protected DF00 failed; numeric failure preserved, inspect before a fresh campaign')
            # Fix the optimizer anchor before testing optimizer variants.
            if 'optimizer_anchor' not in self.state:
                self.state['optimizer_anchor'] = self.select_recipe()
                self.persist()
            anchor = self.state['optimizer_anchor']
            if anchor and 'confirmation_recipe' not in self.state:
                atomic_json(self.root/'optimizer_anchor.json', anchor)
                for option in plan.OPTIMIZER_ORDER[self.server]:
                    config = plan.resolve_case(self.server, baseline=self.baseline,
                        loss_case=anchor['loss_case'], optimizer_case=option)
                    if self.completed_recipe(config):
                        continue
                    if not self.admit(config, phase_end=36):
                        break
                    self.run_case(config, clean=True)
            # Persist a frozen recipe so process restart cannot switch between R2/R3.
            if 'confirmation_recipe' not in self.state:
                selected = self.select_recipe()
                self.state['confirmation_recipe'] = None if selected is None else {
                    k: selected[k] for k in ('loss_case', 'optimizer_case', 'schedule')}
                self.persist()
            recipe = self.state['confirmation_recipe']
            atomic_json(self.root/'selected_recipe.json', recipe)
            if recipe:
                for index, replica in enumerate(('R2', 'R3')):
                    config = plan.resolve_case(self.server, baseline=self.baseline, replica=replica, **recipe)
                    if self.completed_recipe(config):
                        continue
                    if not self.admit(config, phase_end=46, remaining=1-index):
                        break
                    self.run_case(config, clean=True)
            return self.closeout()

    def closeout(self):
        self.retry_pending_artifacts()
        observations=self.state['observations']
        valid=[r for r in observations if all(r.get(k) is True for k in ('metrics_complete','metrics_finite','same_state_verified','full_method_eligible'))]
        selected = self.state.get('confirmation_recipe') or self.select_recipe()
        seed_rows=[]
        for replica, seed in zip(('R1','R2','R3'),plan.LANES[self.server]['seeds']):
            runs=[] if selected is None else [r for r in self.state['runs'] if r['seed']==seed
                and all(r[k]==selected[k] for k in ('loss_case','optimizer_case','schedule'))]
            finished=[r for r in runs if r['status']=='COMPLETED'
                      and r['completed_update']==r['planned_horizon']]
            chosen=max(finished or runs,key=lambda r:r['attempt']) if runs else None
            eligible=[] if chosen is None else [o for o in chosen['evaluations'] if all(o.get(k) is True
                for k in ('metrics_complete','metrics_finite','same_state_verified'))]
            qualified=[o for o in eligible if o['joint_pass']]
            # One selector across every seed, regardless of target success.
            # Qualification/success is independent from this descriptive statistic.
            point=min(eligible,key=plan.representative_rank) if eligible else None
            seed_rows.append({'replica':replica,'seed':seed,'attempt':chosen['attempt'] if chosen else None,
                'status':chosen['status'] if chosen else 'NOT_STARTED_BUDGET',
                'completed_horizon':bool(finished),'completed_update':chosen['completed_update'] if chosen else 0,
                'point':point,'success':bool(qualified),'attempts':len(runs)})
        complete=[r for r in seed_rows if r['completed_horizon'] and r['point'] is not None]
        moments={}
        for metric in ('HQNR','ERGAS'):
            values=[r['point'][metric] for r in complete]
            moments[metric]={'n_completed':len(values),'mean':float(np.mean(values)) if values else None,
                'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
                'sd_reason':None if len(values)>1 else 'Need at least two independent completed seeds'}
        artifact_inventory=[]
        for path in sorted(self.root.glob('runs/**/artifact_manifest.json')):
            manifest=json.loads(path.read_text())
            artifact_inventory.append({'path':str(path),'status':manifest.get('status'),
                'model_state_sha256':manifest.get('model_state_sha256'),
                'weights_saved':manifest.get('weights_saved'),'export_error':manifest.get('export_error')})
        loss_inventory=[]
        for case in plan.LOSS_CASES:
            runs=[r for r in self.state['runs'] if r['loss_case']==case and r['optimizer_case'] is None
                  and r['replica']=='R1' and r['schedule']=='H50']
            loss_inventory.append({'case':case,'status':runs[-1]['status'] if runs else
                'NOT_STARTED_BUDGET','run_ids':[r['run_id'] for r in runs]})
        optimizer_inventory = []
        anchor = self.state.get('optimizer_anchor')
        for option in plan.OPTIMIZER_ORDER[self.server]:
            runs = [r for r in self.state['runs'] if r['optimizer_case'] == option and r['replica'] == 'R1']
            optimizer_inventory.append({'loss_case':anchor['loss_case'] if anchor else None,
                'optimizer_case':option, 'status':runs[-1]['status'] if runs else
                'NOT_STARTED_BUDGET' if anchor else 'NO_ELIGIBLE_ANCHOR',
                'run_ids':[r['run_id'] for r in runs]})
        result={'campaign_id':plan.CAMPAIGN_ID,'server':self.server,'dataset':self.dataset,
            'representative':plan.representative(valid),'rr_reserve':plan.rr_reserve(valid),
            'pareto':plan.pareto_front(valid), 'elapsed_hours':self.elapsed()/3600,
            'completed_runs':sum(r['status']=='COMPLETED' for r in self.state['runs']),
            'confirmation_n':len({(r['loss_case'],r['optimizer_case'],r['schedule'],r['replica']) for r in self.state['runs']
                if r['status']=='COMPLETED' and r['replica'] in ('R2','R3')}),
            'selection_rule':'HQNR_DESC_THEN_ERGAS_ASC', 'test_aware':True,
            'local_raw_event_count':len(list(self.outbox.outbox.glob('DS3-*.json'))),
            'selected_recipe':None if selected is None else {k:selected[k] for k in ('loss_case','optimizer_case','schedule')},
            'selected_recipe_seeds':seed_rows,'completed_seed_statistics':moments,
            'success_rate':{'n_success':sum(r['success'] for r in complete),'n_completed':len(complete),
                'rate':sum(r['success'] for r in complete)/len(complete) if complete else None,
                'n_registered':3,'partial_or_failed_excluded_from_completed_denominator':True},
            'statistic_point_selection':'BEST_OBSERVED_HQNR_THEN_ERGAS; same selector and same-state pair for every completed seed; success separately means any joint-pass snapshot',
            'artifact_inventory':artifact_inventory,'qual_pending':sum(r['status']=='QUAL_PENDING' for r in artifact_inventory),
            'loss_inventory':loss_inventory,'optimizer_inventory':optimizer_inventory,
            'scope_note':'Registered menu with explicit non-execution reasons; not exhaustive completion of all combinations'}
        atomic_json(self.root/'summary.json',result)
        config=plan.resolve_case(self.server,baseline=self.baseline)
        # Numeric payload is splittable; all Pareto points/seed values stay inline
        # in bounded rows as well as the exact complete local summary.json.
        self.emit(config,'CAMPAIGN_CLOSED',scope='CAMPAIGN_AGGREGATE',
            extra={'numeric_observations':dict(numeric_leaves(result)),
                   'weights_saved':False,'aggregate_no_single_model':True})
        self.state['finished_at']=utc_now();self.persist()
        atomic_json(self.root/'publisher_close.json',{'training_process_finished':True})
        return result
