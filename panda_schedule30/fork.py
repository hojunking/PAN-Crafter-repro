"""Explicit audited 25K transfer; ordinary resume never accepts changed contracts.

Source weights, Adam moments/counters, stream and RNG are restored using the
unchanged original loader. Only then is the same state transferred into the
registered new schedule. The original checkpoint and sidecar are never edited.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import torch

from donor30.artifacts import _same, restore_rng, capture_rng
from panda_method.calibration import state_dict_sha256
from ta2.evaluation import file_sha
from . import plan
from .contract import CASES


def local_path(value):
    path=Path(value)
    if str(path).startswith('/workspace/'):
        return plan.ROOT/str(path)[len('/workspace/'):]
    return path if path.is_absolute() else plan.ROOT/path


def _transfer_validated(source, target, stream, payload):
    """Copy complete validated state without importing the old method contract."""
    source.assert_contract()
    target.assert_contract()
    inherited=target.training.inherited_updates
    if source.updates != inherited or target.updates != 0 or not inherited:
        raise ValueError('Transfer requires a fresh target and the exact declared source update')
    if (source._optimizer_names != target._optimizer_names
            or source.training.effective_batch_size != target.training.effective_batch_size
            or source.training.microbatch_size != target.training.microbatch_size
            or source.stage.student_initial_state_sha256 != target.stage.student_initial_state_sha256):
        raise ValueError('Source/target Student initialization, optimizer order or batch contract differs')
    for field in ('alpha','beta','gamma','delta','policy_id','mask_policy'):
        if getattr(source.stage.spec,field) != getattr(target.stage.spec,field):
            raise ValueError('Continuation may change only the registered LR schedule, not '+field)
    if source.stage._donor_binding != target.stage._donor_binding:
        raise ValueError('Continuation must retain exactly the same frozen A/T donor')
    source_stats={k:s.artifact_sha256 for k,s in source.stage.stats_bank.items()}
    target_stats={k:s.artifact_sha256 for k,s in target.stage.stats_bank.items()}
    if source_stats != target_stats:
        raise ValueError('Continuation must retain the exact source calibration bank')
    state=source.state_dict(stream_state=stream.state_dict())
    try:
        target.stage.student.load_state_dict(source.stage.student.state_dict(),strict=True)
        target.optimizer.load_state_dict(deepcopy(state['optimizer']))
        target.stage.set_completed_update(inherited)
        target.updates=inherited
        target.assert_contract()
        if (not _same(target.optimizer.state_dict(),source.optimizer.state_dict())
                or state_dict_sha256(target.stage.student) != state_dict_sha256(source.stage.student)
                or not _same(stream.state_dict(),payload['stream'])):
            raise ValueError('Complete state differs after explicit schedule transfer')
        if payload['rng'].get('torch_cuda') is not None and not torch.cuda.is_initialized():
            # set_rng_state_all can be lazy on a CPU-only model validation.
            # Materialize the existing device inventory before proving equality.
            torch.cuda.init()
        restore_rng(payload['rng'])
        if not _same(capture_rng(),payload['rng']):
            raise ValueError('Complete CPU/CUDA/Python/NumPy RNG differs after transfer')
    except Exception:
        target.poisoned=True
        raise
    return {'student_state_sha256':state_dict_sha256(target.stage.student),
            'optimizer_preserved':True,'optimizer_step_preserved':inherited,
            'rng_preserved':True,'data_stream_preserved':True,
            'calibration_preserved':source_stats,
            'source_last_lr':target.optimizer.param_groups[0]['lr'],
            'next_lr':target.training.rate(inherited+1)}


def initialize_from_source(trainer, stream, donor, stats_bank, *, source_checkpoint,
                           source_sha, evaluator_pin):
    from panda_scale30 import plan as old_plan
    from panda_scale30.runtime import code_pin as source_code_pin
    from panda_scale30.engine import build_fresh_stage2, load_checkpoint
    from panda_scale30.calibration import load_stats_bank
    from panda_scale30.spec import MatureSpec as SourceSpec
    from panda40.stage1 import _check_evaluation
    from panda30r.engine import _check_stream_progress

    cfg=trainer.config
    registered=CASES.get(cfg.get('case'))
    if not registered or registered['inherited_updates'] != 25000:
        raise ValueError('Registered source25K continuation required')
    path=Path(source_checkpoint).resolve(strict=True)
    if path != plan.source_checkpoint_path(cfg).resolve(strict=True) or path.is_symlink():
        raise ValueError('Fork source must be the exact registered same-server 25K artifact')
    if file_sha(path) != source_sha:
        raise ValueError('Fork source checkpoint SHA changed')
    sidecar=json.loads(path.with_suffix('.json').read_text())
    payload=torch.load(path,map_location='cpu',weights_only=False)
    expected=old_plan.resolve_case(cfg['server'],registered['source_case'],cfg['replica'])
    original=payload.get('config',{})
    if (any(original.get(k)!=json.loads(json.dumps(v)) for k,v in expected.items())
            or original != sidecar.get('config')
            or original.get('seed') != cfg['seed']
            or original.get('stream_seeds') != cfg['stream_seeds']
            or original.get('augmentation') != cfg['augmentation']):
        raise ValueError('Source configuration is not the registered same-server/seed baseline')
    pins={'code':source_code_pin(),'evaluator':evaluator_pin}
    artifact=sidecar.get('artifact',{})
    if (payload.get('pins')!=pins or sidecar.get('code_pin')!=pins['code']
            or sidecar.get('evaluator_pin')!=evaluator_pin
            or artifact.get('file_sha256')!=source_sha
            or artifact.get('completed_update')!=25000
            or payload.get('trainer',{}).get('updates')!=25000
            or json.loads(json.dumps(payload.get('evaluation')))!=sidecar.get('evaluation')
            or payload.get('model_state_sha256')!=artifact.get('model_state_sha256')):
        raise ValueError('Original source code/evaluator/step/artifact provenance does not match')
    identity={k:original[k] for k in ('run_id','attempt','seed','dataset')}
    identity.update(completed_update=25000,model_state_sha256=payload['model_state_sha256'])
    _check_evaluation(payload['evaluation'],identity)
    # Paths may have been recorded inside /workspace. Normalize file access
    # only; the serialized source config remains byte-for-byte unchanged.
    artifacts=deepcopy(original['calibration_artifacts'])
    for item in artifacts.values():
        item['path']=str(local_path(item['path']))
    # The original loader writes frozen modules while proving checkpoint state.
    # Use independent copies so target donor tensor versions/devices are untouched.
    source_donor=replace(donor,aligner=deepcopy(donor.aligner).cpu(),
                         teacher=deepcopy(donor.teacher).cpu())
    original_bank=load_stats_bank(source_donor,SourceSpec(**original['stage2_spec']),artifacts,
                                 memory_budget_bytes=3*1024**3)
    source=build_fresh_stage2(original,source_donor,original_bank,evaluator_pin=evaluator_pin,device='cpu')
    # Unchanged legacy loader proves full source state, original optimizer,
    # original scheduler, sampler progress, frozen donor and model hash.
    load_checkpoint(path,source_sha,source,stream,code_pin=pins['code'],evaluator_pin=evaluator_pin)
    _check_stream_progress(stream,25000,trainer.training.effective_batch_size)
    details=_transfer_validated(source,trainer,stream,payload)
    return {'schema':'PANDA_SCHEDULE30_EXPLICIT_SOURCE25K_TRANSFER_V1',
            'source_checkpoint':str(path),'source_file_sha256':source_sha,
            'source_run_id':original['run_id'],'source_campaign_id':original['campaign_id'],
            'source_server':original['server'],'source_seed':original['seed'],
            'source_completed_update':25000,'target_absolute_horizon':35000,
            'new_updates':10000,'source_model_state_sha256':payload['model_state_sha256'],
            'source_code_pin_sha256':plan.canonical_sha256(pins['code']),
            'source_evaluator_pin_sha256':plan.canonical_sha256(evaluator_pin),
            'source_sidecar':str(path.with_suffix('.json')),
            'source_optimizer_and_rng_validated_by_unchanged_loader':True,**details}
