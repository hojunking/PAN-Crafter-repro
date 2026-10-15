"""v1.3 semantic isolation and same-forward diagnostic tests; all offline."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from panda_method.test_stages import batch
from . import plan, reporting
from .__main__ import catalog
from .evaluation import official_evaluation
from .runtime import Campaign, METHOD_FILE, METHOD_SHA
from .test_evaluation import SyntheticSources, evaluation_config, metrics
from .test_plan import baseline, evaluation, qualify
from .test_reporting import record
from .test_training import model, config, take
from .training import Trainer
from ta2.evaluation import atomic_json, file_sha


class RevisionTests(unittest.TestCase):
    def test_document_pin_catalogs_and_destination_do_not_follow_url_fragment(self):
        self.assertEqual(file_sha(METHOD_FILE), METHOD_SHA)
        self.assertEqual(METHOD_SHA, plan.METHOD_SHA256)
        for server in plan.LANES:
            generated = json.loads(json.dumps(catalog(server)))
            path = Path(__file__).resolve().parents[1]/'config/teacher48'/f'{server}_catalog.json'
            self.assertEqual(json.loads(path.read_text()), generated)
            self.assertEqual(generated['parent_campaign_id'], plan.PARENT_CAMPAIGN_ID)
            self.assertEqual(generated['method_revision'], 'MethodPlan-v1.3')
            dest = reporting.destination(server, plan.LANES[server]['dataset'])
            self.assertEqual(dest['spreadsheet_id'], '1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE')
            self.assertTrue(dest['sheet_title'].endswith('-ablations'))
            self.assertNotEqual(dest['sheet_id'], 261013013)  # supplied URL's WV2-main tab
        with self.assertRaises(reporting.ReportingError):
            reporting.destination('s1', 'WV2')

    def test_old_or_missing_baseline_contract_never_implicitly_upgrades(self):
        for changes in ({'method_revision': 'MethodPlan-v1.2'},
                        {'head_conditioning': 'unit_scene_direction'},
                        {'method_sha256': '7a4408e3294db467e3f503cfe6cf2b5d3a8a035267923b2530f825b7a3bd6f2e'}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'v1.3'):
                plan.resolve_case('s1', baseline={**baseline(), **changes})
        old = baseline()
        old.pop('head_conditioning')
        with self.assertRaisesRegex(ValueError, 'head_conditioning'):
            plan.resolve_case('s1', baseline=old)

    def test_new_namespace_and_fresh_only_config(self):
        case = plan.resolve_case('s1', baseline=baseline())
        self.assertTrue(case['run_id'].startswith('S1M13_48H_'))
        plan.validate_method_config(case)
        for key, bad in (('run_id', case['run_id'].replace('M13', 'M12')),
                         ('campaign_id', plan.PARENT_CAMPAIGN_ID),
                         ('initialization', 'RESUME'), ('donor_checkpoint', 'old-head.pt')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan.validate_method_config({**case, key: bad})

    def test_equal_old_identities_still_fail_the_joint_target_gate(self):
        record = evaluation()
        self.assertTrue(qualify(record)['joint_pass'])
        record['method_revision'] = 'MethodPlan-v1.2'
        for key in ('state_before', 'rr_state', 'fr_state', 'state_after'):
            record[key]['method_revision'] = 'MethodPlan-v1.2'
        self.assertFalse(qualify(record)['same_state_verified'])
        self.assertFalse(qualify(record)['joint_pass'])

    def test_old_inference_config_rejected_even_when_tensor_shapes_match(self):
        trainer = Trainer(model(), config())
        old = evaluation_config()
        old['method_revision'] = 'MethodPlan-v1.2'
        with self.assertRaisesRegex(ValueError, 'v1.3'):
            official_evaluation(trainer, old, SyntheticSources(), device='cpu',
                                evaluator_revision='TEST', evaluator_pin={}, metric_functions=metrics())

    def test_old_outbox_not_relabelled_even_with_new_campaign_id(self):
        with self.assertRaisesRegex(reporting.ReportingError, 'v1.2'):
            record(run_id='S1M12_48H_OLD_RUN')
        new = record()
        note = json.loads(new['row'][reporting.INDEX['Original description / notes']])
        note['method_revision'] = 'MethodPlan-v1.2'
        new['row'][reporting.INDEX['Original description / notes']] = reporting.canonical_json(note)
        new['payload_sha256'] = reporting._sha(new['row'])
        with self.assertRaisesRegex(reporting.ReportingError, 'semantics'):
            reporting.validate_record(new)

    def test_copied_old_clock_and_configs_do_not_resume_as_v13(self):
        with tempfile.TemporaryDirectory() as root:
            campaign = Campaign(root, 's2', baseline(), device='cpu')
            clock = {'campaign': plan.CAMPAIGN_ID, 'server': 's2', 'dataset': 'QB',
                     'started_unix': 1., 'deadline_unix': 1.+48*3600}
            atomic_json(campaign.root/'campaign.json', clock)
            with self.assertRaisesRegex(ValueError, 'identity'):
                campaign.launch('never-used', resume_campaign=True)
            old = plan.resolve_case('s2', baseline=baseline())
            old['method_revision'] = 'MethodPlan-v1.2'
            target = campaign.root/'runs'/old['run_id']/'attempt_1/config_resolved.json'
            atomic_json(target, old)
            with self.assertRaisesRegex(ValueError, 'v1.3'):
                campaign._recover_numeric_history()
            self.assertFalse((target.parent/'final_status.json').exists())

    def test_same_forward_fc1_and_raw_prior_stats(self):
        trainer = Trainer(model(), config())
        for branch in ('rr', 'fr'):
            observation = batch(branch.upper())
            expected = []
            handle = trainer.stage.aligner.fc1.register_forward_hook(
                lambda module, args, result: expected.append(result.detach().clone()))
            try:
                result = trainer.objective.evaluate(observation, branch)
            finally:
                handle.remove()
            self.assertEqual(len(expected), 2)  # native + epsilon, no diagnostic extra forward
            numeric = result['numeric']
            self.assertAlmostEqual(float(numeric['fc1_preactivation_min']), float(expected[0].min()))
            self.assertAlmostEqual(float(numeric['fc1_preactivation_max']), float(expected[0].max()))
            self.assertAlmostEqual(float(numeric['fc1_preactivation_rms']), float(expected[0].square().mean().sqrt()))
            self.assertEqual(float(numeric['prior_dy_min']), float(observation.prior_dy_dx[:, 0].min()))
            self.assertEqual(float(numeric['prior_dx_max']), float(observation.prior_dy_dx[:, 1].max()))

    def test_recovered_final_status_cannot_smuggle_old_evaluations(self):
        with tempfile.TemporaryDirectory() as root:
            campaign = Campaign(root, 's2', baseline(), device='cpu')
            case = plan.resolve_case('s2', baseline=baseline())
            path = campaign.root/'runs'/case['run_id']/'attempt_1'
            atomic_json(path/'config_resolved.json', case)
            observation = {**plan.METHOD_CONTRACT, 'method_revision': 'MethodPlan-v1.2',
                           'run_id': case['run_id'], 'seed': case['seed'], 'attempt': 1}
            final = {**case, 'status': 'COMPLETE', 'evaluations': [observation]}
            atomic_json(path/'final_status.json', final)
            with self.assertRaisesRegex(ValueError, 'final-status evaluation'):
                campaign._recover_numeric_history()
            self.assertFalse(campaign.runs)

    def test_numeric_row_cannot_relabel_old_evaluation_identity(self):
        with self.assertRaisesRegex(reporting.ReportingError, 'obsolete method'):
            record(extra={'same_state_verified': True, 'evaluation_identity': {
                **plan.METHOD_CONTRACT, 'method_revision': 'MethodPlan-v1.2'}})

    def test_microbatch_range_logs_are_extrema_not_averages(self):
        from dataclasses import replace
        trainer = Trainer(model(), config(microbatch_RR=1, microbatch_FR=1))
        observations = {branch: replace(batch(branch.upper()),
                         prior_dy_dx=torch.tensor([[.25, -.5], [1.5, .1]])) for branch in ('rr','fr')}
        trace = trainer.step([take(observations['rr'], [i]) for i in (0,1)],
                             [take(observations['fr'], [i]) for i in (0,1)])
        for branch in ('rr','fr'):
            self.assertEqual(trace[branch+'_prior_dy_min'], .25)
            self.assertEqual(trace[branch+'_prior_dy_max'], 1.5)
            self.assertEqual(trace[branch+'_prior_dx_min'], -.5)
            self.assertEqual(trace[branch+'_prior_dy_mean'], .875)
            self.assertLessEqual(trace[branch+'_fc1_preactivation_min'], trace[branch+'_fc1_preactivation_max'])

    def test_old_snapshot_manifest_rejected_before_weights_change(self):
        trainer = Trainer(model(), config())
        snapshot = trainer.snapshot_state()
        snapshot['manifest']['method_revision'] = 'MethodPlan-v1.2'
        with self.assertRaisesRegex(ValueError, 'manifest|contract'):
            trainer.restore_state(snapshot)


if __name__ == '__main__':
    unittest.main()
