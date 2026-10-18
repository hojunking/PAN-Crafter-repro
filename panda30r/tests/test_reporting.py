from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from donor24 import plan as old_plan
from donor30.test_reporting import evaluation as old_evaluation
from panda30r import reporting as R, publication as P


def config(server='s4', phase='donor'):
    cfg = old_plan.resolve_case(old_plan.default_config('s1'), 'A0')
    cfg['server'] = server
    cfg['baseline_resolved']['server'] = server
    cfg.update(campaign_id=R.CAMPAIGN_ID, method_revision=R.METHOD_REVISION,
               run_id='P30R_'+cfg['run_id'], phase=phase, aligner_kind='direct',
               engine_kind='mature_teacher' if phase=='donor' else 'stage2_condq', start_update=0)
    if phase == 'stage2':
        cfg['student_architecture'] = {'bands': 8, 'width': 112, 'depth': [1, 2, 3]}
        cfg['loss_mask_policy'] = 'DETACHED_BICUBIC4_SUPPORT_PER_SAMPLE_MEAN_V1'
    return cfg


def evaluation(cfg, step=1000, h=.959, e=2.2):
    ev = old_evaluation(cfg, step, h, e)
    ev.update(server=cfg['server'], dataset=cfg['dataset'], phase=cfg['phase'])
    for key in ('state_before', 'rr_state', 'fr_state', 'state_after'):
        ev[key].update({k: cfg[k] for k in ('campaign_id', 'method_revision', 'server', 'dataset', 'phase')})
        ev[key]['alignment_protocol_digest'] = 'a'*64
        ev[key]['engine_kind'] = cfg['engine_kind']
        ev[key]['lifetime_update'] = cfg.get('start_update',0)+step
    return ev


def persist(root, cfg, step=1000, h=.959, e=2.2):
    ev = evaluation(cfg, step, h, e)
    path = Path(root)/'runs'/cfg['run_id']/'attempt_01'
    path.mkdir(parents=True, exist_ok=True)
    R.atomic_json(path/'config.json', cfg)
    R.atomic_json(path/f'evaluation_{step:07d}.json', ev)
    return R.emit_evaluation(root, cfg, ev, timestamp='2026-10-16T00:00:00+00:00')


class ReportingTests(unittest.TestCase):
    def test_actual_registry_all_servers_and_cases_serialize_without_schema_drift(self):
        from panda30r import plan
        for server in plan.SERVERS:
            configs=[plan.resolve_case(server,case,replica)
                     for group in plan.groups(server) for case,replica in group['runs']]
            configs += [plan.resolve_case(server,'FRESH_A1'),
                        plan.resolve_case(server,'MATURE',source_update=7000)]
            for cfg in configs:
                with self.subTest(server=server,case=cfg['case'],replica=cfg['replica']):
                    record=R.make_record(cfg,evaluation(cfg,step=1000))
                    R.validate_record(record)
                    self.assertEqual(record['source']['fields']['Lifetime_updates'],cfg['start_update']+1000)
                    self.assertEqual(record['destination']['sheet_title'],'WV3-ablations')

    def test_local_lifetime_and_parent_lineage_do_not_conflate(self):
        cfg = config('s3', 'stage2')
        cfg.update(engine_kind='tail_fc2_eps', start_update=40000, planned_horizon=10000,
                   parent_run='P30R_parent_Q_ALOW', lr_A=1e-6)
        rec = R.make_record(cfg, evaluation(cfg, step=10000))
        fields = rec['source']['fields']
        self.assertEqual(fields['Updates'], 10000)
        self.assertEqual(fields['Selected_step'], 10000)
        self.assertEqual(fields['Lifetime_updates'], 50000)
        self.assertEqual(fields['Parent_run'], 'P30R_parent_Q_ALOW')
        self.assertEqual(fields['A_peak_lr'], 1e-6)
        self.assertEqual(fields['Role'], 'Student+ScheduledAligner')
        R.validate_record(rec)
        ev = evaluation(cfg, step=10000)
        for key in ('state_before','rr_state','fr_state','state_after'):
            ev[key]['lifetime_update'] = 10000
        with self.assertRaisesRegex(ValueError, 'lineage'):
            R.make_record(cfg, ev)

    def test_mature_teacher_retains_inherited_updates(self):
        cfg = config('s1', 'donor')
        cfg.update(start_update=7000, planned_horizon=43000, lr_A=0.)
        rec = R.make_record(cfg, evaluation(cfg, step=43000))
        self.assertEqual(rec['source']['fields']['Lifetime_updates'],50000)
        self.assertEqual(rec['source']['fields']['Role'],'Teacher+FrozenAligner')

    def test_imported_replay_cannot_be_misreported_as_new_trained_run(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=config('s3','donor')
            cfg.update(engine_kind='donor_replay',start_update=7000)
            # Even forged eligible legacy-shaped fixtures cannot bypass replay exclusion.
            persist(root,cfg)
            value=P.project_campaign(root,'s3')
            self.assertFalse(value['rows'])
            self.assertIn('IMPORTED_REPLAY_IS_NOT_NEW_RUN_RESULT',
                          [item['reason'] for item in value['conflicts']])

    def test_stage1_and_stage2_roundtrip_shared_destination(self):
        for server, phase in ((s,p) for s in ('s1','s3','s4') for p in ('donor','stage2')):
            cfg = config(server, phase)
            rec = R.make_record(cfg, evaluation(cfg))
            R.validate_record(rec)
            self.assertEqual(rec['destination']['sheet_title'], 'WV3-ablations')
            self.assertIn(cfg['loss_mask_policy'], rec['row'][R.INDEX['Approach']])
            self.assertEqual(rec['row'][R.INDEX['Experiment']], cfg['run_display'])

    def test_checkpoint_gate_does_not_replace_display_gate(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config('s4', 'stage2')
            persist(root, cfg, 1000, .9591, 2.1)
            persist(root, cfg, 2000, .961, 2.4)
            value = P.project_campaign(root, 's4')
            self.assertFalse(value['conflicts'])
            self.assertEqual(value['selected'][0]['step'], 2000)
            persist(root, cfg, 4000, .960, 2.045)
            persist(root, cfg, 5000, .961, 2.04)
            value = P.project_campaign(root, 's4')
            self.assertFalse(value['conflicts'])
            self.assertEqual([v['step'] for v in value['selected']], [4000, 5000])

    def test_synthetic_partial_tampered_or_wrong_identity_cannot_publish(self):
        for kind in ('synthetic', 'partial', 'changed_config', 'wrong_campaign'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                cfg = config(); persist(root, cfg)
                path = next(Path(root).glob('runs/*/attempt_01/evaluation_*.json'))
                value = json.loads(path.read_text())
                if kind == 'synthetic': value['metric_function_injection'] = True
                if kind == 'partial': value['fr']['per_scene'].pop()
                if kind == 'wrong_campaign': value['state_after']['campaign_id'] = 'D24'
                if kind == 'changed_config':
                    changed = deepcopy(cfg); changed['lambda_abs'] = 999
                    R.atomic_json(path.parent/'config.json', changed)
                R.atomic_json(path, value)
                self.assertTrue(P.project_campaign(root, 's4')['conflicts'])

    def test_mixed_campaign_record_rejected(self):
        cfg = config(); ev = evaluation(cfg)
        for key in ('state_before', 'rr_state', 'fr_state', 'state_after'):
            ev[key]['campaign_id'] = 'D24'
        with self.assertRaisesRegex(ValueError, 'identity/config'):
            R.make_record(cfg, ev)

    def test_cost_absence_blocks_only_publication(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config(); persist(root, cfg)
            self.assertTrue(P.project(root, 's4')['conflicts'])
            self.assertFalse(P.project(root, 's4', require_cost=False)['conflicts'])
            self.assertEqual(len(list((Path(root)/'outbox').glob('*.json'))), 1)

    def test_immutable_outbox_and_ledger_duplicate_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config(); record = persist(root, cfg)
            R.LocalOutbox(root).enqueue(record)
            self.assertEqual(len((Path(root)/'records.jsonl').read_text().splitlines()), 1)
            tamper = deepcopy(record); tamper['row'][0] = 'oops'
            with self.assertRaises(ValueError): R.LocalOutbox(root).enqueue(tamper)


if __name__ == '__main__':
    unittest.main()
