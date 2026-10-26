"""Real-data CUDA update/resume checks before entering the immutable queue.

Uses the same full-training calibration later consumed by the campaign. No
preflight Student metrics are published or reused by production Students.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from directshift48.reporting import atomic_json, utc_now
from donor30.artifacts import _same
from . import plan
from .engine import advance, save_checkpoint, load_checkpoint
from .evaluation import model_state_hash
from .runtime import Campaign
from .spec import geometry_manifest
from ta2.evaluation import file_sha


def validate_hard_only_step(result):
    """Validate executed losses, not only the requested configuration."""
    if (result.get('student_views') != 1 or result.get('consistency_lambda') != 0.
            or result.get('soft_weighted') != 0.
            or result.get('loss') != result.get('hard_weighted')
            or any(name in result for name in ('consistency_raw', 'consistency_weighted'))):
        raise ValueError('Preflight executed an unregistered extra loss or Student view')


def attempt_directory(directory):
    """Retry disposable validation without overwriting a previous checkpoint."""
    path = Path(directory) / f'attempt_{time.time_ns()}'
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_case(campaign, cfg, attempt):
    """Disposable exact-source update/resume check; never touches parent artifacts."""
    campaign.check()
    case = cfg['case']
    cfg['disposable_preflight'] = True
    trainer, stream = campaign.build(cfg)
    initial_updates = trainer.updates
    if initial_updates != cfg.get('inherited_updates',0):
        raise ValueError('Hard36 preflight must begin at its exact registered source state')
    student_parameters = sum(p.numel() for p in trainer.stage.student.parameters())
    if (trainer.stage.spec.sampling_variant != cfg['sampling_variant']
            or trainer.stage.spec.input_variant != cfg['input_variant']
            or student_parameters != cfg['student_sampling']['student_parameters']):
        raise ValueError('Student sampling variant or parameter count differs from the registered architecture')
    before = model_state_hash(trainer.stage)
    tick = time.monotonic()
    campaign.check()
    first = advance(trainer, stream)
    saved_state = trainer.state_dict(stream_state=stream.state_dict())
    weights_at_one = {name:p.detach().cpu().clone()
                      for name,p in trainer.stage.student.named_parameters()}
    path = attempt / (case + '_resume.pt')
    campaign.check()
    save_checkpoint(path, trainer, stream, code_pin=campaign.code,
                    evaluator_pin=campaign.evaluator_pin)
    campaign.check()
    second = advance(trainer, stream)
    expected = model_state_hash(trainer.stage)
    weights_at_two = {name:p.detach().cpu().clone()
                      for name,p in trainer.stage.student.named_parameters()}
    expected_stream = stream.state_dict()
    if expected == before or first['gradient']['student'] <= 0:
        raise ValueError('Preflight did not update Student weights')
    campaign.check()
    trainer, stream = load_checkpoint(path, file_sha(path), trainer, stream,
                                     code_pin=campaign.code, evaluator_pin=campaign.evaluator_pin)
    if not _same(saved_state, trainer.state_dict(stream_state=stream.state_dict())):
        raise ValueError('Checkpoint model/optimizer/scheduler/RNG/stream restore is not exact')
    campaign.check()
    resumed = advance(trainer, stream)
    for result_at_update in (first, second, resumed):
        validate_hard_only_step(result_at_update)
    # The unchanged production CUDA policy permits nondeterministic backward
    # reductions. Exact state restore above is mandatory; replay rounding
    # is bounded separately, never used to excuse state/stream mismatch.
    differences = [p.detach().cpu().double() - weights_at_two[name].double()
                   for name,p in trainer.stage.student.named_parameters()]
    max_abs = max(float(x.abs().max()) for x in differences)
    diff_norm = sum(float(x.square().sum()) for x in differences) ** .5
    update_norm = sum(float((weights_at_two[name].double()-value.double()).square().sum())
                      for name,value in weights_at_one.items()) ** .5
    relative_update_error = diff_norm / max(update_norm, 1e-30)
    if max_abs > 1e-8 or relative_update_error > 1e-4:
        raise ValueError(f'CUDA replay exceeds roundoff bounds: {max_abs}, {relative_update_error}')
    if (resumed['update'] != initial_updates + 2 or abs(second['loss'] - resumed['loss']) > 1e-8
            or not _same(expected_stream, stream.state_dict())):
        raise ValueError('Restored update/loss/stream mismatch')
    # Real RR/FR inference; no GT/Teacher/calibration used for inference.
    trainer.stage.eval()
    inference_shapes, applied_correction_l2 = {}, {}
    scale = geometry_manifest(trainer.stage.spec.geometry_at_completed(trainer.updates))['scale']
    if scale != 1. or cfg.get('student_views', 1) != 1:
        raise ValueError('Preflight requires full Aligner scale 1 and one Student view')
    for split in ('rr','fr'):
        campaign.check()
        from .warp import warp_pan
        batch = campaign.sources.batch(split, [0], device='cuda')
        output = trainer.stage.native_prediction(batch)
        if (not bool(torch.isfinite(output['y']).all())
                or not torch.equal(output['pan_correction'], output['u'] * scale)
                or not torch.equal(output['pan_aligned'], warp_pan(batch.pan, output['u'] * scale))
                or not bool(output['pan_correction'].any())
                or bool(output['ms_correction'].any()) or bool(output['output_correction'].any())):
            raise ValueError('Native inference did not apply exactly the registered positive PAN scale')
        student_pan = output['student_pan_input']
        if (student_pan.shape[1] != cfg['student_input']['pan_channels']
                or not torch.equal(student_pan[:, :1], output['pan_aligned'])
                or (cfg['input_variant'] == 'CORRECTED_DIFF' and
                    not torch.equal(student_pan[:, 1:2], output['pan_aligned'] - batch.pan))):
            raise ValueError('Native Student does not receive the registered corrected PAN/difference input')
        inference_shapes[split] = list(output['y'].shape)
        applied_correction_l2[split] = float(torch.linalg.vector_norm(output['pan_correction'], dim=1).mean())
        del batch, output
    row = {'case':case, 'status':'PASS', 'updates':trainer.updates,
           'initial_updates':initial_updates,
           'sampling_variant':cfg['sampling_variant'],
           'input_variant':cfg['input_variant'],
           'student_warp':cfg['student_warp'],
           'student_input':cfg['student_input'],
           'full_train_support_validated_in_calibration':True,
           'student_parameter_count':student_parameters,
           'exact_resume_state':True, 'loss':second['loss'],
           'update_replay_bitwise':model_state_hash(trainer.stage)==expected,
           'update_replay_max_abs':max_abs,
           'update_replay_relative_to_update_l2':relative_update_error,
           'scale': scale, 'alpha':trainer.stage.spec.alpha, 'gamma':trainer.stage.spec.gamma,
           'student_lr':trainer.training.lr, 'beta':trainer.stage.spec.beta,
           'student_views':1, 'rotation_consistency_weight':0.,
           'native_shapes':inference_shapes, 'applied_correction_l2':applied_correction_l2,
           'disposable_resume_checkpoint':str(path),
           'seconds':time.monotonic()-tick, 'production_student_reused':False}
    print(json.dumps(row), flush=True)
    del trainer, stream
    gc.collect(); torch.cuda.empty_cache()
    return row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1','s3'), required=True)
    args = parser.parse_args()
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu('cuda')
    campaign = Campaign(args.server)
    registered = {}
    for group in plan.groups(args.server):
        for case, replica in group['runs']:
            registered.setdefault(case, replica)
    cases = list(registered)
    passed_path = campaign.root / 'preflight' / 'passed.json'
    if passed_path.exists():
        passed = json.loads(passed_path.read_text())
        if (passed.get('code_pin') != campaign.code or
                passed.get('evaluation_pin') != campaign.evaluator_pin or
                [row.get('case') for row in passed.get('rows', [])] != cases or
                any(row.get('status') not in ('PASS','DEFERRED_UNTIL_REGISTERED_PARENT') for row in passed['rows'])):
            raise ValueError('Existing preflight belongs to a different implementation')
        print('Exact implementation preflight already passed; production Students remain fresh.', flush=True)
        return 0
    campaign.admission(plan.ROOT / 'ablr2/sensor_sources.json')
    campaign.register_donor(plan.DONORS[0])
    directory = campaign.root / 'preflight'
    directory.mkdir(exist_ok=True)
    attempt = attempt_directory(directory)
    result = []
    for case in cases:
        cfg = plan.resolve_case(args.server,case,registered[case])
        if cfg.get('source_package') == 'panda_hard36' and not plan.source_checkpoint_path(cfg).is_file():
            result.append({'case':case,'status':'DEFERRED_UNTIL_REGISTERED_PARENT',
                           'source_case':cfg['source_case'], 'production_validation_required':True})
            continue
        result.append(validate_case(campaign,cfg,attempt))
    campaign.verify_code()
    atomic_json(passed_path, {'at':utc_now(), 'rows':result,
                'code_pin':campaign.code, 'evaluation_pin':campaign.evaluator_pin})
    campaign.state['status'] = 'PREFLIGHT_PASSED_READY_FOR_REGISTERED_TRAINING'
    campaign.persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
