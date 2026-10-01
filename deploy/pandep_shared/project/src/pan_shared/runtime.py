"""Campaign integration. No legacy imports, implicit GPU admission or sheet routing."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time

from .common import CAMPAIGN, atomic_json, append_jsonl, canonical_sha, read_json, timestamp
from .preflight import check_identity
from .safety import SafetyStop, admit, control, inventory


def owned(work, *parts):
    root = Path(work).resolve()
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root):
        raise SafetyStop('BLOCKED_PATH: campaign artifact escapes work root')
    return path


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise SafetyStop('BLOCKED_IDENTITY: immutable artifact differs: '+str(path))
    else:
        atomic_json(path, value)
    return value


def sync_sheet(work_root, credentials, sheet_id):
    from .sheets import GoogleSheetAdapter, SheetsUploader, SHEET_ID
    if sheet_id != SHEET_ID:
        raise SafetyStop('BLOCKED_SHEET_SCHEMA: only existing deployment tab is authorized')
    work = Path(work_root).resolve()
    adapter = GoogleSheetAdapter(credentials, sheet_id=sheet_id)
    return SheetsUploader(adapter, owned(work, 'outbox')).sync()


class CampaignRuntime:
    def __init__(self, work_root, credentials=None, device='cuda:0'):
        import torch
        self.root = Path(work_root).resolve()
        self.manifest = check_identity(self.root)
        self.registry = read_json(self.root/'cases.json')
        self.dataset_manifest = read_json(self.root/'dataset_manifest.json')
        self.subsets = read_json(self.root/'subset_manifest.json')
        self.credentials, self.device = credentials, torch.device(device)
        if self.device.type != 'cuda':
            raise SafetyStop('BLOCKED_ENVIRONMENT: formal s2 campaign requires admitted CUDA')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        self.datasets = {}

    def dataset(self, sensor, split):
        from .data import SensorDataset
        key = (sensor, split)
        if key not in self.datasets:
            self.datasets[key] = SensorDataset(self.dataset_manifest, sensor, split)
        return self.datasets[key]

    def admission(self, task=None):
        from .verify import check_gates
        check_identity(self.root)
        receipt = check_gates(self.root)
        if str(self.device)!=receipt['q00']['device']:
            raise SafetyStop('BLOCKED_GATES: requested CUDA device differs from validated Q00 device')
        inv = inventory(self.root, self.manifest['server_identity_file'])
        # Device UUID/driver/name are immutable; free memory changes legitimately.
        signature = [[field.strip() for i, field in enumerate(line.split(',')) if i != 3]
                     for line in inv['gpu']['stdout'].splitlines() if line.strip()]
        expected = receipt.get('gpu_identity')
        if expected is not None and signature != expected:
            raise SafetyStop('BLOCKED_IDENTITY: GPU UUID/hardware/driver changed')
        # Conservative forward reserve for retained models, resume copies and logs.
        # Recalibrate upward from all observed block artifact growth; never prune.
        estimates = read_json(self.root/'resource_estimates.json') if (self.root/'resource_estimates.json').exists() else {}
        budget = max(2*1024**3, estimates.get('max_block_bytes', 0))
        try:
            result = admit(inv, expected_hostname=self.manifest['hostname'],
                           own_pid=os.getpid(), next_block_bytes=budget,
                           atomic_reserve_bytes=1024**3)
        except SafetyStop as error:
            if str(error).startswith('WAIT_RESOURCE:'):
                result = {'status': 'WAIT_RESOURCE', 'reason': str(error)}
            else:
                raise
        atomic_json(owned(self.root, 'telemetry/latest_admission.json'),
                    {'at': timestamp(), 'task': task, 'inventory': inv, 'decision': result})
        return result

    def validate(self, trainer, kind):
        from torch.utils.data import Subset
        from .evaluate import validation
        if kind == 'probe':
            result = {}
            for split in ('train', 'val'):
                datasets = {s: Subset(self.dataset(s, split), self.subsets['sensors'][s][split+'_probe_ids'])
                            for s in trainer.sensors}
                result[split] = validation(trainer.model, datasets, self.device, batch_size=trainer.microbatch)
            return {'kind': 'fixed128_unaugmented_train_and_val', 'results': result,
                    'subset_sha256': self.subsets['subset_sha256'], 'selector_candidate': False}
        if kind != 'validation':
            raise ValueError('Unknown validation event')
        datasets = {s: self.dataset(s, 'val') for s in trainer.sensors}
        result = validation(trainer.model, datasets, self.device, batch_size=trainer.microbatch)
        for sensor in trainer.sensors:
            if result['sensors'][sensor]['n_samples'] != self.dataset_manifest['sensors'][sensor]['splits']['val']['count']:
                raise ValueError('Partial full-validation cannot select BEST')
        return result

    def benchmark(self, trainer, checkpoint, selector_scope):
        from .evaluate import PROTOCOL, benchmark_checkpoint, evaluator_sha, profile_model
        from .export import export_candidate
        from .sheets import SheetsUploader, SHEET_ID, result_id
        started = time.monotonic()
        datasets = {s: {split: self.dataset(s, split) for split in ('rr', 'fr')} for s in trainer.sensors}
        reports = benchmark_checkpoint(trainer.model, checkpoint['model_path'], datasets,
                    self.device, owned(self.root, 'evaluation_cache'), identity=trainer.identity)
        if set(reports)!=set(trainer.sensors):
            raise SafetyStop('BLOCKED_EVALUATOR: partial sensor benchmark result')
        for sensor, report in reports.items():
            if (report['identity']['checkpoint_sha256']!=checkpoint['checkpoint_sha256']
                    or report['identity']['evaluator_sha256']!=evaluator_sha()
                    or report['protocol']!=PROTOCOL
                    or report['rr'].get('n_scenes')!=20 or report['fr'].get('n_scenes')!=20):
                raise SafetyStop('BLOCKED_EVALUATOR: benchmark identity/protocol/coverage mismatch')
        outputs = {}
        outbox = SheetsUploader(None, owned(self.root, 'outbox'))
        for sensor, report in reports.items():
            profile_key = canonical_sha({'source': self.manifest['source_sha256'],
                'architecture': trainer.identity['architecture_sha256'], 'sensor': sensor,
                'environment': self.manifest['environment_sha256'], 'evaluator': evaluator_sha()})
            profile_path = owned(self.root, 'profiles', profile_key+'.json')
            if profile_path.exists():
                profile = read_json(profile_path)
            else:
                profile = profile_model(trainer.model, sensor, self.device)
                freeze(profile_path, profile)
            elapsed = time.monotonic()-started
            costs = trainer.state['times']
            observation = dict(trainer.config, server='s2', sensor=sensor,
                train_sensors=list(trainer.sensors), completed_step=trainer.step,
                selected_step=checkpoint['completed_step'],
                checkpoint_sha256=report['identity']['checkpoint_sha256'],
                evaluator_sha256=report['identity']['evaluator_sha256'],
                selector_scope=selector_scope, protocol_id=PROTOCOL['id'],
                rr=report['rr'], fr=report['fr'], recorded_at=timestamp(),
                cost={**profile, 'train_hours': costs['training']/3600,
                      'eval_hours': (costs['validation']+costs['benchmark']+elapsed)/3600,
                      'wall_hours': (sum(costs.values())+elapsed)/3600,
                      'scope': 'cumulative active run, not sensor-additive; training excludes data_wait; wall excludes queue; '+profile['scope']},
                notes={'identity': trainer.identity, 'selected_checkpoint': checkpoint,
                    'selected_exposures': checkpoint['exposures'],
                    'current_exposures': trainer.sampler.exposures(),
                    'candidate_count': trainer.state['candidate_count'],
                    'selected_through_global_step': trainer.step,
                    'protocol': report['protocol'], 'profile': profile,
                    'subset_sha256': self.subsets['subset_sha256'],
                    'comparison': 'matched global150K*k(shared) vs50K*k(single); seed paired; per-sensor sample exposure authoritative',
                    'data_wait_seconds': costs['data_wait'], 'checkpoint_seconds': costs['checkpoint'],
                    'teacher': 'NONE', 'deployment_status': 'candidate_only'})
            key = result_id(observation)
            path = owned(self.root, 'observations', key+'.json')
            # A retry must replay the ORIGINAL timestamp/cost/notes as well as metrics.
            if path.exists():
                previous = read_json(path)
                if (result_id(previous) != key or previous.get('payload_sha256') !=
                        canonical_sha({k:v for k,v in previous.items() if k!='payload_sha256'})):
                    raise SafetyStop('BLOCKED_IDENTITY: observation key/seal mismatch')
                stable = ('run_id','case_id','config_sha256','repeat','seed','attempt','sensor',
                          'selected_step','completed_step','checkpoint_sha256','evaluator_sha256',
                          'selector_scope','protocol_id','rr','fr')
                if any(previous[k]!=observation[k] for k in stable) or previous['notes']['identity']!=trainer.identity:
                    raise SafetyStop('BLOCKED_IDENTITY: observation metrics/provenance differ on retry')
                observation=previous
            else:
                observation['payload_sha256']=canonical_sha(observation)
                atomic_json(path, observation)
            try:
                outbox.enqueue(observation)
            except Exception as error:
                append_jsonl(owned(self.root, 'sheet_errors.jsonl'),
                    {'at': timestamp(), 'result_id': key, 'error': str(error), 'source_observation': str(path)})
            outputs[sensor] = {'result_id': key, 'observation': str(path),
                'HQNR': report['fr']['hqnr'], 'SCC': report['rr']['scc'], 'ERGAS': report['rr']['ergas']}
            print(json.dumps({'event': 'BENCHMARK', 'run_id': trainer.config['run_id'],
                'selector': selector_scope, 'sensor': sensor, **outputs[sensor]}), flush=True)
        run = trainer.config
        if run['case_id']=='C00' and run['repeat']==1 and trainer.step==150000:
            if selector_scope in ('EXACT','BEST_JOINT_VAL_TO_B0003'):
                outputs['candidate_bundle'] = str(export_candidate(trainer.model, checkpoint, run, self.root,
                    selector_scope, device=self.device, provenance={
                        'source_sha256': self.manifest['source_sha256'], 'config_sha256': run['config_sha256'],
                        'data_sha256': self.manifest['dataset_sha256'], 'subset_sha256': self.manifest['subset_sha256'],
                        'protocol_sha256': evaluator_sha()}))
        self.try_sync()
        return outputs

    def try_sync(self):
        if not self.credentials:
            return {'status': 'OUTBOX_ONLY'}
        from .sheets import SHEET_ID
        try:
            result = {'status': 'SYNC_COMPLETE', 'results': sync_sheet(self.root, self.credentials, SHEET_ID)}
        except Exception as error:
            result = {'status': 'PENDING_SHEET', 'error': str(error)}
        append_jsonl(owned(self.root, 'sheet_sync.jsonl'), {'at': timestamp(), **result})
        return result

    def runner(self, run):
        from .evaluate import evaluator_sha, profile_model
        from .model import build_model
        from .train import Trainer
        root = owned(self.root, 'runs', run['run_id'])
        model = build_model(run)
        architecture = model.architecture_manifest()
        identity = {k: run[k] for k in ('campaign_id','run_id','case_id','repeat','seed','attempt','config_sha256')}
        identity.update(server='s2', source_sha256=self.manifest['source_sha256'],
            environment_sha256=self.manifest['environment_sha256'], data_sha256=self.manifest['dataset_sha256'],
            subset_sha256=self.manifest['subset_sha256'], protocol_sha256=evaluator_sha(),
            architecture_sha256=canonical_sha(architecture), selector_policy_sha256=canonical_sha({
                'primary':self.registry['primary_selector'],
                'secondary_shared':self.registry['secondary_selector_shared'],
                'secondary_single':self.registry['secondary_selector_single'],
                'ties':'earlier completed step', 'candidate_steps':'every10K and block25K/50K, once per step; no step0'}))
        freeze(root/'config.resolved.json', run)
        freeze(root/'architecture_manifest.json', architecture)
        freeze(root/'run_manifest.json', identity)
        # Measure before AdamW/gradient allocation: process peak memory must not
        # accidentally include training optimizer state in an inference column.
        model=model.to(self.device).float()
        for sensor in run['sensors']:
            key=canonical_sha({'source':self.manifest['source_sha256'],
                'architecture':identity['architecture_sha256'],'sensor':sensor,
                'environment':self.manifest['environment_sha256'],'evaluator':evaluator_sha()})
            path=owned(self.root,'profiles',key+'.json')
            if not path.exists():
                freeze(path,profile_model(model,sensor,self.device))
        datasets = {s: self.dataset(s, 'train') for s in run['sensors']}
        ids = {s: self.subsets['sensors'][s]['half_ids'] if run['train_fraction']==.5 else list(range(len(datasets[s])))
               for s in run['sensors']}
        trainer = Trainer(model, datasets, run, identity, root, sample_ids=ids,
            evaluate=self.validate, benchmark=self.benchmark, device=self.device)
        original_run = trainer.run_until
        runtime = self
        def measured_run(target_step, control_check=None):
            checked = -1
            initial_bytes = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            def checked_control():
                nonlocal checked
                action = control_check() if control_check else 'RUN'
                if action in ('STOP','PAUSE'):
                    return action
                if trainer.step % 100 == 0 and trainer.step != checked:
                    checked = trainer.step
                    inv = inventory(runtime.root, runtime.manifest['server_identity_file'])
                    estimate_path=runtime.root/'resource_estimates.json'
                    estimate=read_json(estimate_path) if estimate_path.exists() else {}
                    decision = admit(inv, expected_hostname=runtime.manifest['hostname'], own_pid=os.getpid(),
                                     next_block_bytes=max(2*1024**3,estimate.get('max_block_bytes',0)),
                                     atomic_reserve_bytes=1024**3)
                    append_jsonl(owned(runtime.root,'telemetry/resources.jsonl'),
                        {'at': timestamp(), 'run_id':run['run_id'], 'step':trainer.step,
                         'disk':inv['disk'], 'gpu':inv['gpu'], 'decision':decision})
                return action
            result = original_run(target_step, checked_control)
            growth = max(0, sum(p.stat().st_size for p in root.rglob('*') if p.is_file())-initial_bytes)
            path = owned(runtime.root,'resource_estimates.json')
            old = read_json(path) if path.exists() else {}
            atomic_json(path, {'max_block_bytes':max(growth,old.get('max_block_bytes',0)), 'updated_at':timestamp()})
            return result
        trainer.run_until = measured_run
        return trainer

    def stage_report(self, stage, state):
        from .reporting import write_stage_report
        write_stage_report(self.root, stage, self.registry)
        self.try_sync()

