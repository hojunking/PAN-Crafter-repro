"""Reproducible component/smoke receipt; explicitly NOT campaign launch approval."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import unittest

from .common import atomic_json, file_sha, load_json


def verify(args):
    import numpy as np
    import torch
    import h5py
    from .brovey import brovey8, config_metadata
    from .data import RRDataset

    if args.output.exists():
        raise ValueError('Use a new verification receipt path; old receipts are retained')
    names = ['tools.test_panpv_input_audit', 'panpv.test_data', 'panpv.test_training',
             'panpv.test_evaluation', 'panpv.test_brovey']
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    record = dict(schema='PANPV_COMPONENT_VERIFICATION_v1',
                  at_utc=datetime.now(timezone.utc).isoformat(),
                  tests=dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                             skipped=len(result.skipped), successful=result.wasSuccessful()),
                  brovey=config_metadata(), actual_input_smokes=[], model_smoke=None,
                  formal_training_started=False, campaign_admission=False,
                  fr_evaluator_validated=False, donor_strict_loaded=False,
                  private_metrics_published=False)
    if not result.wasSuccessful() or result.skipped:
        atomic_json(args.output, record)
        raise RuntimeError('All component tests must pass without skipped GDAL tests')
    prepared = load_json(args.preparation)
    for scene in prepared['scenes']:
        split = load_json(scene['split_candidate'])
        lp = load_json(scene['lpan_manifest'])
        dataset = RRDataset(split['source'], split, 'train', lp,
                            training_crop=64, allow_provisional=True)
        item = dataset[0]
        # Only a train-side input smoke, never an audit score or model fitting.
        pan = ((item['pan'][0].numpy() + 1) * 1023.5).astype(np.float32)
        ms = ((item['ms'].numpy() + 1) * 1023.5).astype(np.float32)
        pred = brovey8(pan, ms)
        record['actual_input_smokes'].append(dict(
            scene_id=scene['scene_id'], raw_index=item['raw_index'],
            virtual_train_patches=len(dataset), output_shape=list(pred.shape),
            all_finite=bool(np.isfinite(pred).all()), mask_pixels=int(item['mask'].sum()),
            provisional=dataset.provisional, target_values_used=False,
            scope='TRAIN_INPUT_ONLY_BROVEY_INTERFACE_SMOKE_NOT_A_COMPARISON_RESULT'))
    if args.smoke_gpu:
        from deploy.pandep_shared.project.src.pan_shared.model import SingleSensorPLHUNet
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        model = SingleSensorPLHUNet('WV3', 104, (1, 2, 2), seed=261002).cuda()
        rng = torch.Generator().manual_seed(261002)
        pan = (torch.rand((2, 1, 64, 64), generator=rng) * 2 - 1).cuda()
        ms = (torch.rand((2, 8, 16, 16), generator=rng) * 2 - 1).cuda()
        low = (torch.rand((2, 1, 16, 16), generator=rng) * 2 - 1).cuda()
        expected = torch.nn.functional.interpolate(ms, scale_factor=4, mode='bicubic', align_corners=False)
        output = model('WV3', pan, ms, low)
        error = float((output - expected).abs().max().item())
        if error != 0:
            raise AssertionError('SC step0 must equal internal MS-up exactly')
        before = model.heads['WV3'].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
        loss = output.abs().mean()
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError('Nonfinite/disconnected SC gradient')
        optimizer.step()
        if torch.equal(before, model.heads['WV3'].weight):
            raise AssertionError('SC head did not update')
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise AssertionError('Nonfinite SC parameters')
        record['model_smoke'] = dict(
            hardware=torch.cuda.get_device_name(), architecture=model.architecture_manifest(),
            precision='FP32', amp=False, tf32=False, sc0_ms_up_max_abs=error,
            one_disposable_synthetic_update=True, model_saved=False,
            formal_microbatch_chosen=False, donor_parity_checked=False)
    repo = Path(__file__).resolve().parents[1]
    sources = sorted((repo / 'panpv').glob('*.py')) + [repo / 'panpv/Dockerfile', repo / 'tools/panpv_input_audit.py']
    record['source_sha256'] = {str(path.relative_to(repo)): file_sha(path) for path in sources}
    record['remaining'] = [
        'Deliver exact donor weights/config/preprocessing provenance from s2',
        'Provide RR creation/coordinate/filter-support evidence before freezing the spatial split',
        'Validate real-data model context, choose fixed production microbatch, and integrate FR tile metrics',
        'Complete end-to-end case queue, private reporting/outbox, and log export before formal launch',
    ]
    atomic_json(args.output, record)
    print(f'Component verification saved: {args.output}; formal training NOT started', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-gpu', action='store_true')
    verify(parser.parse_args())


if __name__ == '__main__':
    main()
