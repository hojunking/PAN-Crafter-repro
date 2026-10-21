"""Real-data CUDA update/resume checks before entering the immutable queue.

Uses the same full-training calibration later consumed by the campaign. No
preflight Student metrics are published; every production Student starts fresh.
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
from ta2.evaluation import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1','s3'), required=True)
    args = parser.parse_args()
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu('cuda')
    campaign = Campaign(args.server)
    passed_path = campaign.root / 'preflight' / 'passed.json'
    if passed_path.exists():
        passed = json.loads(passed_path.read_text())
        if (passed.get('code_pin') != campaign.code or
                passed.get('evaluation_pin') != campaign.evaluator_pin or
                [row.get('case') for row in passed.get('rows', [])] != ['ZERO','PAIR0','ROTCONS'] or
                any(row.get('status') != 'PASS' for row in passed['rows'])):
            raise ValueError('Existing preflight belongs to a different implementation')
        print('Exact implementation preflight already passed; production Students remain fresh.', flush=True)
        return 0
    campaign.admission(plan.ROOT / 'ablr2/sensor_sources.json')
    campaign.register_donor(plan.DONORS[0])
    directory = campaign.root / 'preflight'
    directory.mkdir(exist_ok=True)
    result = []
    for case in ('ZERO','PAIR0','ROTCONS'):
        cfg = plan.resolve_case(args.server, case)
        cfg['disposable_preflight'] = True
        trainer, stream = campaign.build(cfg)
        before = model_state_hash(trainer.stage)
        tick = time.monotonic()
        first = advance(trainer, stream)
        saved_state = trainer.state_dict(stream_state=stream.state_dict())
        weights_at_one = {name:p.detach().cpu().clone()
                          for name,p in trainer.stage.student.named_parameters()}
        path = directory / (case + '_resume.pt')
        save_checkpoint(path, trainer, stream, code_pin=campaign.code,
                        evaluator_pin=campaign.evaluator_pin)
        second = advance(trainer, stream)
        expected = model_state_hash(trainer.stage)
        weights_at_two = {name:p.detach().cpu().clone()
                          for name,p in trainer.stage.student.named_parameters()}
        expected_stream = stream.state_dict()
        if expected == before or first['gradient']['student'] <= 0:
            raise ValueError('Preflight did not update Student weights')
        trainer, stream = load_checkpoint(path, file_sha(path), trainer, stream,
                                         code_pin=campaign.code, evaluator_pin=campaign.evaluator_pin)
        if not _same(saved_state, trainer.state_dict(stream_state=stream.state_dict())):
            raise ValueError('Checkpoint model/optimizer/scheduler/RNG/stream restore is not exact')
        resumed = advance(trainer, stream)
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
        if (resumed['update'] != 2 or abs(second['loss'] - resumed['loss']) > 1e-8
                or not _same(expected_stream, stream.state_dict())):
            raise ValueError('Restored update/loss/stream mismatch')
        # Real RR/FR inference; no GT/Teacher/calibration used for inference.
        trainer.stage.eval()
        inference_shapes = {}
        for split in ('rr','fr'):
            batch = campaign.sources.batch(split, [0], device='cuda')
            output = trainer.stage.native_prediction(batch)
            if not bool(torch.isfinite(output['y']).all()) or bool(output['pan_correction'].any()):
                raise ValueError('Invalid ZERO native inference')
            inference_shapes[split] = list(output['y'].shape)
            del batch, output
        row = {'case':case, 'status':'PASS', 'updates':trainer.updates,
               'exact_resume_state':True, 'loss':second['loss'],
               'update_replay_bitwise':model_state_hash(trainer.stage)==expected,
               'update_replay_max_abs':max_abs,
               'update_replay_relative_to_update_l2':relative_update_error,
               'consistency_raw':second['consistency_raw'],
               'consistency_weighted':second['consistency_weighted'],
               'student_views':second['student_views'], 'native_shapes':inference_shapes,
               'seconds':time.monotonic()-tick, 'production_student_reused':False}
        result.append(row)
        print(json.dumps(row), flush=True)
        del trainer, stream
        gc.collect(); torch.cuda.empty_cache()
    campaign.verify_code()
    atomic_json(passed_path, {'at':utc_now(), 'rows':result,
                'code_pin':campaign.code, 'evaluation_pin':campaign.evaluator_pin})
    campaign.state['status'] = 'PREFLIGHT_PASSED_READY_FOR_FRESH_TRAINING'
    campaign.persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
