from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_qb_rre import plan, runner, failures, reporting
from panda_qb_rre.best_publication import project
from panda_qb40.tests.test_runner import FakeCampaign
from panda_sensor_refine.orchestration import sha


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PublicationFailureTests(unittest.TestCase):
    def test_same_targets_and_disjoint_sheet_namespace(self):
        for server in ('s1', 's3'):
            with tempfile.TemporaryDirectory() as tmp:
                value = project(tmp, server)
                self.assertFalse(value['conflicts'])
                self.assertEqual(value['publication_thresholds']['hqnr_min'], .9315)
                self.assertEqual(value['publication_thresholds']['ergas_max'], 3.5)
                self.assertTrue(value['publication_thresholds']['publish_all_joint_passes'])
                self.assertTrue(value['publication_thresholds']['deduplicate_representative_and_pass'])
                self.assertEqual(value['destination']['sheet_title'], 'QB-ablations')
        self.assertEqual(reporting.RECORD_PREFIX, 'PQRRE-')

    def test_saved_support_failure_holds_all_three_students(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = plan.ROOT / 'config/bglobal_rr/qb_original_lms.json'
            target = root / 'config/bglobal_rr/qb_original_lms.json'
            target.parent.mkdir(parents=True); target.write_bytes(source.read_bytes())
            folder = root / plan.G.OUTPUT / plan.CAMPAIGN_ID / 's1/QB'
            cfg = plan.resolve_case('s1', 'QB5_S1_A400_E030')
            directory = folder / 'runs' / cfg['run_id'] / 'attempt_01'
            row = dict(run_id=cfg['run_id'], case=cfg['case'], replica='R1', phase='donor',
                       directory=str(directory), status='FAILED', step=12000, poisoned=False, error=failures.SUPPORT_ERROR)
            for name, value in [('config', cfg), ('requested_config', cfg), ('run_state', row)]:
                dump(directory / (name + '.json'), value)
            (directory / 'train.jsonl').write_text('{"step":12000}\n')
            checkpoint = directory / 'models/stop_0012000.pt'
            checkpoint.parent.mkdir(); checkpoint.write_bytes(b'test checkpoint, never a model')
            dump(directory / 'models/latest_resume.json', dict(checkpoint_path=str(checkpoint),
                 file_sha256=sha(checkpoint), completed_update=12000, campaign_resume_complete=True, checkpoint_verified=True))
            camp = FakeCampaign(folder, 0); camp.config = plan.default_config('s1')
            camp.state.update(runs=[deepcopy(row)], started_unix=1000)
            with patch.object(plan, 'ROOT', root), patch('panda_sensor_focus.early_stop.run_case') as execute:
                result = runner.run(camp, catalog='fixture')
            execute.assert_not_called()
            self.assertEqual(result['status'], failures.STATUS)
            self.assertEqual(len(result['blocked_runs']), 3)
            self.assertEqual(result['runs'][0], row)
            self.assertFalse(result['all_registered_completed'])
            with patch.object(plan, 'ROOT', root):
                checkpoint.write_bytes(b'tampered')
                with self.assertRaises(ValueError):
                    failures.verify(camp, row)

    def test_unknown_failure_is_not_treated_as_a_skipped_branch(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('panda_sensor_focus.early_stop.finish_stopped', return_value=False), \
                patch('panda_sensor_focus.early_stop.run_case', side_effect=ValueError('bad source hash')):
            camp = FakeCampaign(tmp, 0); camp.config = plan.default_config('s1')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                runner.run(camp, catalog='fixture')
            self.assertEqual(camp.state['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
