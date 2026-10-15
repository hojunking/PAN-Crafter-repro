"""Collect v3 intake evidence and run regression tests; never auto-launch training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import unittest

from .common import atomic_json, file_sha, load_json


def verify(args):
    output = args.work_root / 'preparation_status.json'
    if output.exists():
        raise ValueError('Preserve existing preparation receipt; use a new work root/revision')
    preflight = args.work_root / 'preflight'
    paths = {key: preflight / name for key, name in dict(
        package='package_intake.json', arrays='data_verification.json',
        numerical='numerical_parity.json', context='context_smoke.json').items()}
    records = {key: load_json(path) for key, path in paths.items()}
    if records['package']['state'] != 'TRANSFER_VERIFIED_NOT_TRAINING_ADMISSION':
        raise ValueError('Package verification incomplete')
    if records['arrays']['state'] != 'DATA_ARRAYS_AND_FROZEN_SPLITS_VERIFIED_NOT_CAMPAIGN_ADMISSION':
        raise ValueError('Full-array verification incomplete')
    if not records['numerical']['passed'] or records['context']['state'] != 'PASS':
        raise ValueError('Numerical/context smoke did not pass')
    root = Path(records['package']['input_root']).resolve(strict=True)
    mapping = load_json(root / 'canonical_mapping.json')
    expected_ids = {scene['scene_id'] for scene in mapping['scenes']}
    if len(expected_ids) != 5:
        raise ValueError('Expected five distinct scene identities')
    for key in ('arrays', 'numerical', 'context'):
        if {scene['scene_id'] for scene in records[key]['scenes']} != expected_ids:
            raise ValueError('Evidence covers different scene identities: ' + key)
    for key in ('arrays', 'context'):
        if records[key]['package_receipt_sha256'] != file_sha(paths['package']):
            raise ValueError('Evidence is bound to a different intake revision')
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'tools.test_panpv_input_audit', 'panpv.test_data', 'panpv.test_training',
        'panpv.test_evaluation', 'panpv.test_brovey', 'panpv.test_v3_package',
        'panpv.test_v3_data', 'panpv.test_v3_parity', 'panpv.test_v3_preflight'])
    tests = unittest.TextTestRunner(verbosity=1).run(suite)
    test_record = dict(run=tests.testsRun, failures=len(tests.failures), errors=len(tests.errors),
                       skipped=len(tests.skipped), successful=tests.wasSuccessful())
    atomic_json(preflight / 'regression_tests.json', test_record)
    if not tests.wasSuccessful() or tests.skipped:
        raise RuntimeError('All tests must pass without skipped runtime dependencies')
    scenes = records['arrays']['scenes']
    total = {split: sum(scene['counts'][split] for scene in scenes) for split in ('train', 'val', 'audit')}
    native_bytes = sum(s['shape_ms'][0] * s['shape_ms'][1] * 16 * 8 * 4 for s in scenes)
    repo = Path(__file__).resolve().parents[1]
    result = dict(
        schema='PANPV_V3_PREPARATION_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        state='NEW_DATA_VERIFIED_FORMAL_CAMPAIGN_NOT_STARTED',
        input_root=str(root), input_archive_sha256=records['package']['archive_sha256'],
        old_inputs_and_artifacts_preserved=True, frozen_provider_split_reused=True,
        counts=total, scenes=[{k: s[k] for k in ('scene_id', 'category', 'sha256', 'counts',
                                                'patch_index_sha256', 'blocks_sha256')} for s in scenes],
        rr_scope='FILTERED_WITHIN_SCENE_SPATIAL_HOLDOUT',
        rr_generation='FROZEN_GAUSSIAN_APPROXIMATION_NOT_OFFICIAL_BENCHMARK_PARITY',
        fr_scope='PREVIOUSLY_APPROVED_TILE_SCOPE; NEW_V3_TILE_COORDINATES_PENDING',
        original_vendor_tiff_geometry_independently_verified=False,
        source_array_scan='ALL_PIXELS; shape/type/finite/range/binary-mask checks',
        degradation_parity='480_SPARSE_WINDOWS; NOT_EXHAUSTIVE_ALL_OUTPUT_RECOMPUTATION',
        model_context_parity='TRAIN_SIDE_NONPRETRAINED_DIAGNOSTIC_ONLY; NOT_DONOR_PARITY',
        storage=dict(free_bytes=shutil.disk_usage(args.work_root).free,
                     native_float32_all5_one_method_bytes=native_bytes,
                     native_float32_all5_four_methods_bytes=4*native_bytes,
                     full_scene_output_scope_restored=False),
        evidence={key: dict(path=str(path.resolve()), sha256=file_sha(path)) for key, path in paths.items()},
        tests=test_record,
        source_sha256={str(p.relative_to(repo)): file_sha(p) for p in sorted((repo / 'panpv').glob('*.py'))},
        formal_training_started=False, donor_strict_loaded=False, sheet_writes=0,
        remaining=[
            'Deliver exact donor model.safetensors SHA256 6ad7cff8fc60542ab687a405f7b55430926942f008a65345de99fa69fa8b3846 and config',
            'Strict-load donor, verify ZS/FT0 numerical equality and freeze production microbatch',
            'Freeze v3 FR tile list/common masks and final RR/FR evaluation integration',
            'Complete campaign queue and dedicated practical-validation reporting integration',
        ])
    atomic_json(output, result)
    print(result['state'], total, 'tests=', tests.testsRun, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    verify(parser.parse_args())


if __name__ == '__main__':
    main()
