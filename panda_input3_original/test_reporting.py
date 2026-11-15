from copy import deepcopy
import json
import unittest
from directshift48.reporting import INDEX
from panda_input3 import plan, publication as pub
from panda_input3_targets.test_reporting import record
from panda_input3_targets import reporting as prior
from panda_fr_columns import publication as fr
from . import reporting
from .policy import passes, hqnr_passes, choose, POLICY


def original_record(cfg, step=1000, **changes):
    r = record(cfg, step, **changes)
    r['fr']['metrics']['HQNR'] = changes.get('original_h', plan.targets(cfg['sensor'])['HQNR_min'])
    return r


class OriginalPanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): reporting.install()

    def test_original_only_gates_for_all_three_sensors(self):
        for server in ('s1', 's2', 's3'):
            cfg = plan.queue(server)[0]; t = plan.targets(cfg['sensor'])
            r = original_record(cfg, h=.8)
            self.assertTrue(passes(r, cfg))  # corrected HQNR can fail
            r['fr_corrected_pan']['metrics']['HQNR'] = .999
            r['fr']['metrics']['HQNR'] = t['HQNR_min'] - 1e-8
            self.assertFalse(passes(r, cfg)); self.assertEqual(choose([r], 'stage2', cfg), [])
            for val in (float('nan'), float('inf')):
                r['fr']['metrics']['HQNR'] = val
                self.assertFalse(hqnr_passes(r, cfg))
            self.assertFalse(passes(original_record(cfg, e=t['ERGAS_max'] + 1e-8), cfg))
        cfg = plan.queue('s3')[0]
        self.assertFalse(passes(original_record(cfg, p=45.07999999), cfg))

    def test_representative_exception_endpoint_and_no_cross_step_merge(self):
        cfg = plan.queue('s3')[0]
        a = original_record(cfg, 1000, e=.50, p=45.07, h=.8)
        b = original_record(cfg, 2000, e=.51, p=45.1)
        c = original_record(cfg, cfg['horizon'], original_h=.9, e=.49, p=46.)
        selected = choose([a, b, c], 'stage2', cfg)
        self.assertEqual([(r['step'], slot) for r, _, slot in selected], [(1000, 'ROW0'), (2000, 'STEP_0002000')])
        row, raw = pub.make_row(cfg, 'RUNNING', a, selected[0][1], 'ROW0_ORIGINAL')
        self.assertIn('PSNR 미달', row[INDEX['Selection@Step']])
        self.assertEqual(json.loads(raw)['publication']['selection_reference'], 'ORIGINAL_PAN')

    def test_v3_migration_preserves_verified_history_and_foreign_rows(self):
        cfg = plan.queue('s1')[0]
        r = original_record(cfg, original_h=.9)  # old corrected gate passes; new original gate fails
        with reporting.legacy_context():
            oldrow, oldnote = prior.make_row(cfg, 'COMPLETED', r, 'PASS_ALL_STEPS', 'STEP_0001000_ORIGINAL')
        status, note = pub.make_row(cfg, 'COMPLETED', None, 'STATUS', 'ROW0_ORIGINAL')
        value = dict(configs={cfg['run_id']: cfg}, proof={(cfg['run_id'], 1000): r}, rows=[status], notes=[note])
        pub.verify_owned(oldrow, value, 's1')
        foreign = deepcopy(oldrow); foreign[INDEX['Campaign_ID']] = 'UNRELATED_CAMPAIGN'
        snapshot = dict(rows=[foreign, oldrow], notes=[oldnote, oldnote])
        self.assertEqual(prior.retired_rows(snapshot, value, 's1'), [7])
        tampered = deepcopy(oldrow); tampered[INDEX['HQNR↑']] = .99
        with self.assertRaises(ValueError): pub.verify_owned(tampered, value, 's1')

    def test_new_rows_keep_dual_fr_same_checkpoint_and_immutable_config(self):
        cfg = plan.queue('s1')[0]; original_cfg = deepcopy(cfg)
        r = original_record(cfg, h=.90)
        row, note = pub.make_row(cfg, 'COMPLETED', r, 'HQNR_QUALIFIED_MIN_ERGAS', 'ROW0_ORIGINAL')
        value = dict(policy=POLICY, configs={cfg['run_id']: cfg},
                     proof={(cfg['run_id'], 1000): r}, rows=[row], notes=[note])
        pub.verify_owned(row, value, 's1')
        self.assertEqual(cfg, original_cfg)
        self.assertEqual(row[INDEX['HQNR↑']], .9585)
        self.assertIn('[기존 PAN 선정]', row[INDEX['Experiment']])
        cells = fr.extras(value)[row[INDEX['Result_ID']]]
        self.assertEqual(cells[0]['userEnteredValue']['numberValue'], .90)
        self.assertEqual(json.loads(cells[0]['note'])['official_selection_reference'], 'ORIGINAL_PAN')
        self.assertEqual(json.loads(note)['configuration'], original_cfg)
        tampered = deepcopy(row); tampered[INDEX['Selection@Step']] = '보정 PAN · 조건 충족'
        with self.assertRaises(ValueError): pub.verify_owned(tampered, value, 's1')

    def test_full_precision_wv3_boundary_and_original_tie_break(self):
        cfg = plan.queue('s1')[0]
        r = original_record(cfg, e=2.0400477356896936)
        self.assertTrue(hqnr_passes(r, cfg)); self.assertFalse(passes(r, cfg))
        self.assertEqual(len(choose([r], 'stage2', cfg)), 1)
        a = original_record(cfg, 1000, original_h=.959, h=.999)
        b = original_record(cfg, 2000, original_h=.960, h=.8)
        self.assertEqual(choose([a, b], 'stage2', cfg)[0][0]['step'], 2000)


if __name__ == '__main__': unittest.main()
