"""Historical controls cannot be substituted, relabeled or silently recovered."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ta2.evaluation import evaluator_identity, file_sha
from panda_wv3_main_rgb import plan as old_plan, evaluation as old_eval
from panda_wv3_ablation_rgb import plan as s3_plan, evaluation as s3_eval, selection
from panda_wv3_hyper import plan, reuse


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class HistoricalFixture:
    def __init__(self, root, server='s1'):
        self.root = root
        self.server = server
        self.module = old_plan if server == 's1' else s3_plan
        self.evaluation = old_eval if server == 's1' else s3_eval
        self.cfg = plan.resolve(server, 'H2_REF' if server == 's1' else 'H1_REF', 1)
        self.old = self.module.resolve(server, 'M0_OURS' if server == 's1' else 'S1_A5_ALL', 1)
        self.source = root / 'source'
        self.directory = self.source / 'runs' / self.old['run_id']
        self.target = root / 'new'
        self.pins = {'source_release': 'old-release', 'evaluator': evaluator_identity(),
            'train_inventory': {'sample_ids': ['training-only']},
            'split_manifests': {'train': 'train', 'val': 'val', 'rr': 'rr', 'fr': 'fr'}}
        self.donor = {'receipt_sha256': 'correct-donor'}
        self.calibration = {'artifact_sha256': 'correct-calibration'}
        write(self.directory / 'config.json', self.old)
        write(self.source / 'campaign_state.json', {'campaign_id': self.module.CAMPAIGN_ID,
            'server': server, 'runs': [{'run_id': self.old['run_id'], 'status': 'COMPLETED',
                'step': self.cfg['horizon']}]})
        write(self.source / 'admission.json', self.pins)
        write(root / 'old_config/release.json', {'source_sha256': 'old-release'})
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / 'train.jsonl').write_text(''.join(
            json.dumps({'step': i, 'seconds': .01}) + '\n'
            for i in range(1, self.cfg['horizon'] + 1)))
        if server == 's3':
            inventory = {'selector_protocol_sha256': plan.digest(selection.PROTOCOL),
                'sample_count': 1080, 'eligible': [True] * 1080, 'eligible_count': 1080,
                'sample_ids': [f'WV3:val:{i}' for i in range(1080)],
                'excluded_gt_low_texture_count': 0}
            inventory['inventory_sha256'] = plan.digest(inventory)
            write(self.source / 'validation_inventory.json', inventory)
            rows = [{'sample_id': f'WV3:val:{i}', 'eligible': True,
                'rre_px': .2, 'valid': True} for i in range(1080)]
            self.validation = {**selection.aggregate_rows(rows, inventory),
                'per_sample': rows, 'protocol': selection.PROTOCOL, 'Val_ERGAS': 2.1,
                'epsilon': {'EPE50': .1, 'EPE90': .3, 'gain_mean': .9,
                    'gain_median': .9, 'count': 4320}}
        for step in self.cfg['evaluation_steps']:
            checkpoint = self.directory / 'models' / f'step_{step:07d}.pt'
            meta = {'step': step, 'config_sha256': self.old['config_sha256'],
                'model_sha256': 'historical-model', 'path': str(checkpoint),
                'file_sha256': 'pruned-weight-historical-sha'}
            record = {'schema': ('WV3_MAIN_RGB_EVAL_V1' if server == 's1'
                else 'WV3_RGB_ABLATION_EVAL_V1'), 'run_id': self.old['run_id'],
                'config_sha256': self.old['config_sha256'], 'step': step,
                'model_sha256': 'historical-model', 'stage': self.cfg['stage'],
                'eval_protocol_id': plan.digest(self.pins['evaluator']),
                'same_state_verified': True, 'reference_warp': False, 'inference_tta': False,
                'checkpoint': meta, 'donor_selection': self.donor,
                'calibration_sha256': self.calibration['artifact_sha256']}
            for split in ('rr', 'fr'):
                values = ({'CC': .9, 'ERGAS': 2., 'PSNR': 38., 'Q8': .9,
                    'RMSE': 20., 'SAM': 2.7, 'SCC': .98, 'SSIM': .99} if split == 'rr'
                    else {'HQNR': .96, 'D_s': .03, 'D_lambda': .01, 'JQM': .98})
                rows = [{'scene_id': f'WV3:{split}:{i}', 'metrics': deepcopy(values),
                    'rre_rgb': {'registration': {'valid': True, 'shift': [0., .25]},
                        'rre_px': .25}} for i in range(20)]
                record[split] = {'per_scene': rows, 'metrics': {
                    k: float(np.mean([r['metrics'][k] for r in rows])) for k in values}}
            record['test_rre_rgb'] = {s: old_eval.rre_summary(record[s]['per_scene'])
                for s in ('rr', 'fr')}
            if server == 's3':
                record.pop('donor_selection')
                record.pop('calibration_sha256')
                record['validation'] = {k: v for k, v in self.validation.items()
                    if k not in ('per_sample', 'protocol')}
                write(self.directory / f'validation_{step:07d}.json', self.validation)
            write(self.directory / f'evaluation_{step:07d}.json', record)

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(plan, 'ROOT', self.root))
        stack.enter_context(patch.object(self.module, 'CONFIG', self.root / 'old_config'))
        stack.enter_context(patch.object(self.module, 'campaign_root', return_value=self.source))
        stack.enter_context(patch.object(reuse.assets, 'source_evidence', return_value=(
            {}, self.donor, self.pins, self.calibration)))
        return stack


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = HistoricalFixture(Path(self.temp.name))
        self.stack = self.fixture.patches()
        self.addCleanup(self.stack.close)
        reuse._HASH_CACHE.clear()
        reuse._TRACE_CACHE.clear()

    def test_pruned_weights_allow_numeric_only_original_identity(self):
        f = self.fixture
        receipt = reuse.try_reuse(f.cfg, f.target, pins=f.pins)
        self.assertEqual(receipt['weight_status'], ['METRICS_ONLY_WEIGHT_PRUNED'])
        self.assertFalse(receipt['new_inference_performed'])
        self.assertEqual(receipt['fixed_steps'], [29000, 30000])
        self.assertEqual(receipt, reuse.verify_reuse(f.cfg, f.target, pins=f.pins))
        records = reuse.load_reused_records(f.cfg, f.target)
        self.assertEqual(len(records), 30)
        self.assertEqual(records[-1]['run_id'], f.old['run_id'])
        self.assertNotEqual(records[-1]['run_id'], f.cfg['run_id'])
        self.assertEqual(records[-1]['reuse']['weight_status'], 'METRICS_ONLY_WEIGHT_PRUNED')

    def test_wrong_donor_or_calibration_rejected_without_receipt(self):
        f = self.fixture
        p = f.directory / 'evaluation_0029000.json'
        record = json.loads(p.read_text())
        record['calibration_sha256'] = 'another-repeat'
        write(p, record)
        self.assertIsNone(reuse.try_reuse(f.cfg, f.target, pins=f.pins))
        self.assertFalse((f.target / 'reuse_receipt.json').exists())
        self.assertIn('donor/calibration', (f.target / 'reuse_rejected.json').read_text())

    def test_missing_fixed_endpoint_falls_back_to_fresh(self):
        f = self.fixture
        (f.directory / 'evaluation_0029000.json').unlink()
        self.assertIsNone(reuse.try_reuse(f.cfg, f.target, pins=f.pins))
        self.assertEqual(json.loads((f.target / 'reuse_rejected.json').read_text())['action'],
            'FRESH_SAME_RECIPE_AND_SEED')

    def test_changed_admitted_data_cannot_reuse(self):
        f = self.fixture
        pins = deepcopy(f.pins)
        pins['split_manifests']['rr'] = 'different-test-population'
        self.assertIsNone(reuse.try_reuse(f.cfg, f.target, pins=pins))

    def test_retained_weight_hash_is_checked_and_postaccept_mutation_fails(self):
        f = self.fixture
        evpath = f.directory / 'evaluation_0029000.json'
        record = json.loads(evpath.read_text())
        checkpoint = Path(record['checkpoint']['path'])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'synthetic-original-weight')
        record['checkpoint']['file_sha256'] = file_sha(checkpoint)
        write(checkpoint.with_suffix('.json'), record['checkpoint'])
        write(evpath, record)
        receipt = reuse.try_reuse(f.cfg, f.target, pins=f.pins)
        self.assertEqual(next(r for r in receipt['records'] if r['step'] == 29000)['weight_status'],
            'VERIFIED_RETAINED_WEIGHT')
        checkpoint.write_bytes(b'synthetic-changed--weight')
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA'):
            reuse.try_reuse(f.cfg, f.target, pins=f.pins)

    def test_wrong_seed_scientific_projection_cannot_match(self):
        f = self.fixture
        altered = deepcopy(f.old)
        altered['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'science differs: seed'):
            reuse._compare_science(f.cfg, altered)
        self.assertIsNone(reuse.source_spec(plan.resolve('s1', 'H2_A100', 1)))

    def test_s3_stage1_fixed50k_and_rgb_validation_are_required(self):
        self.stack.close()
        with tempfile.TemporaryDirectory() as directory:
            f = HistoricalFixture(Path(directory), 's3')
            with f.patches():
                receipt = reuse.try_reuse(f.cfg, f.target, pins=f.pins)
                self.assertEqual(receipt['fixed_steps'], [50000])
                self.assertEqual(len(receipt['records']), 50)
                self.assertIn('validation_sha256', receipt['records'][-1])
                p = f.directory / 'validation_0050000.json'
                value = json.loads(p.read_text())
                value['per_sample'][0]['rre_px'] = .9
                # This changes population evidence even if its median stays .2.
                write(p, value)
                with self.assertRaisesRegex(ValueError, 'evidence changed'):
                    reuse.verify_reuse(f.cfg, f.target, pins=f.pins)


if __name__ == '__main__':
    unittest.main()
