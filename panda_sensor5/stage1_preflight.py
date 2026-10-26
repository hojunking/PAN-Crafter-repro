"""Disposable real-source Stage1 learning and exact-state restoration gate."""
from copy import deepcopy
import gc
import time

import torch

from donor30.artifacts import _same
from donor30.evaluation import model_state_hash
from panda_method.calibration import state_dict_sha256
from .stage1 import build_stage1, Stage1Stream, advance_stage1, save_stage1, load_stage1


def validate_stage1(campaign, cfg, attempt):
    """Preserve sources/settings, use a new disposable A/T and never publish it."""
    campaign.check()
    cfg = deepcopy(cfg)
    cfg['disposable_preflight'] = True
    device = getattr(campaign, 'device', 'cuda')
    trainer = build_stage1(cfg, campaign.sources, device=device)
    stream = Stage1Stream(campaign.sources, cfg, device=device)
    started = time.monotonic()
    initial = {role:state_dict_sha256(getattr(trainer.stage, role)) for role in ('aligner','teacher')}
    first = advance_stage1(trainer, stream)
    after = {role:state_dict_sha256(getattr(trainer.stage, role)) for role in ('aligner','teacher')}
    if any(initial[role] == after[role] for role in initial):
        raise ValueError('Stage1 preflight must update both Aligner and Teacher')
    saved_state, saved_stream = deepcopy(trainer.state_dict()), deepcopy(stream.state_dict())
    weights_one = {n:p.detach().cpu().clone() for n,p in trainer.stage.named_parameters()}
    path = attempt / 'stage1_resume.pt'
    campaign.check()
    artifact = save_stage1(path, trainer, stream, code_pin=campaign.code,
                          evaluator_pin=campaign.evaluator_pin)
    second = advance_stage1(trainer, stream)
    expected_weights = {n:p.detach().cpu().clone() for n,p in trainer.stage.named_parameters()}
    expected_stream = deepcopy(stream.state_dict())
    expected_hash = model_state_hash(trainer.stage)
    del trainer, stream
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    campaign.check()
    trainer, stream, _ = load_stage1(path, artifact['file_sha256'], cfg, campaign.sources,
        device=device, code_pin=campaign.code, evaluator_pin=campaign.evaluator_pin)
    if not _same(saved_state, trainer.state_dict()) or not _same(saved_stream, stream.state_dict()):
        raise ValueError('Stage1 model/optimizer/RNG/stream restore must be exact')
    replay = advance_stage1(trainer, stream)
    differences = [p.detach().cpu().double()-expected_weights[n].double()
                   for n,p in trainer.stage.named_parameters()]
    max_abs = max(float(x.abs().max()) for x in differences)
    norm = sum(float(x.square().sum()) for x in differences)**.5
    update_norm = sum(float((expected_weights[n].double()-p.double()).square().sum())
                      for n,p in weights_one.items())**.5
    relative = norm/max(update_norm,1e-30)
    if max_abs > 1e-8 or relative > 1e-4 or not _same(expected_stream, stream.state_dict()):
        raise ValueError(f'Stage1 replay exceeds FP32 roundoff bounds: {max_abs}, {relative}')
    if abs(second['loss']-replay['loss']) > 1e-8:
        raise ValueError('Stage1 replay loss mismatch')
    shapes = {}
    trainer.stage.eval()
    with torch.no_grad():
        for split in ('rr','fr'):
            campaign.check()
            batch = campaign.sources.batch(split,[0],device=device)
            output = trainer.stage.native_prediction(batch)
            if output['y'].shape != batch.native_lms.shape or not bool(torch.isfinite(output['y']).all()):
                raise ValueError('Stage1 native four-band prediction mismatch')
            shapes[split] = list(output['y'].shape)
    result = {'case':cfg['case'], 'dataset':cfg['dataset'], 'status':'PASS', 'phase':'donor',
        'updates':trainer.updates, 'exact_resume_state':True,
        'update_replay_bitwise':model_state_hash(trainer.stage)==expected_hash,
        'update_replay_max_abs':max_abs, 'update_replay_relative_to_update_l2':relative,
        'native_shapes':shapes, 'alignment_target':cfg['alignment_target'],
        'loss':second['loss'], 'disposable_resume_checkpoint':str(path),
        'seconds':time.monotonic()-started, 'production_donor_reused':False}
    del trainer, stream, saved_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
