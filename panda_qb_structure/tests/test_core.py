"""Synthetic CPU contract checks; no GPU, experiment metrics, or Sheet writes."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from panda_qb_structure import registry as G


class StructureTests(unittest.TestCase):
    def child(self, code):
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_registry_manifest_and_factorial(self):
        self.assertEqual({s: len(v) for s, v in G.RUNS.items()}, {'s1': 10, 's3': 8})
        self.assertEqual(sum(G.cases()[c]['total_updates'] for runs in G.RUNS.values() for c, _ in runs), 540000)
        for case, settings in G.cases().items():
            base = G.cases()['QB4_CONTROL']
            self.assertEqual({k for k in base if base[k] != settings[k]}, set(G.CHANGES[case]['axes']))
        path = Path(__file__).resolve().parents[1] / 'experiment_manifest.json'
        self.assertEqual(json.loads(path.read_text()), json.loads(json.dumps(G.manifest())))
        self.assertFalse(any(rep in ('R1', 'R2') and case == 'QB4_WD000' for case, rep in G.RUNS['s3']))

    def test_local_lanes_reporting_costs_and_source_registration(self):
        for server in ('s1', 's3'):
            self.child(f"server={server!r}\n" + '''
from panda_qb_structure.adapter import install
from panda_qb_structure import registry as G
from panda_qb_structure import donor as D
from pathlib import Path
from copy import deepcopy
if server=='s3':
 source=D.metadata(server);source.update(selected_step=39000,checkpoint_sha256='c'*64,receipt_sha256='d'*64)
 original=D.metadata;D.metadata=lambda s,root=D.ROOT:deepcopy(source) if s=='s3' else original(s,root)
p=install(server)
assert install(server) is p
from panda_sensor_focus import reporting as R,publication
from panda_sensor_focus.runtime import code_pin
from panda_sensor_focus.cost.profiling import registered_profiles,build_deployment
from panda_sensor_focus.cost import signature
from donor30.plan import canonical_sha256,stream_seeds
assert R.destination(server)['sheet_title']=='QB-ablations'
assert publication.LANES[server]==('QB','QB-ablations',261001011,.9315,3.5)
assert p.default_config(server)['stage1_runs']==0
assert p.default_config(server)['fixed_donor']['seed']==G.SOURCES[server]['seed']
assert Path('panda_qb_structure/cases_'+server+'.csv').read_text()==p.catalog_csv(server)
assert 'panda_qb_structure/spec.py' in code_pin()
for case,rep in G.RUNS[server]:
 cfg=p.resolve_case(server,case,rep);R._config(cfg)
 assert cfg['server']==cfg['baseline_resolved']['server']==cfg['donor_origin_server']==server
 assert cfg['dataset']=='QB' and cfg['joint_target']==G.TARGET and cfg['hqnr_min']==.9315
 assert cfg['config_digest']==canonical_sha256(p.default_config(server))
 assert cfg['baseline_resolved']['qual_export']['display_range_DN']==[0,2047]
 assert all(cfg['stream_seeds'][k]==v for k,v in stream_seeds(G.SEEDS[rep]).items())
 assert cfg['stage2_spec']['sampling_variant']==cfg['sampling_variant']
 assert cfg['stage2_spec']['input_variant']==cfg['input_variant']
 assert cfg['student_input']['channels']==(6 if cfg['input_variant']=='CORRECTED_DIFF' else 5)
 assert 'AdamW wd=' in R.approach_description(cfg)
 assert ('6ch' if cfg['input_variant']=='CORRECTED_DIFF' else '5ch') in R.approach_description(cfg)
 bad=deepcopy(cfg);bad['training_spec']['weight_decay']=.123
 try:R._config(bad)
 except ValueError:pass
 else:raise AssertionError('WD mismatch accepted')
profiles=registered_profiles(server)
assert len(profiles)==4
import torch
torch.set_num_threads(1)
for cfg,step in profiles:
 signature(cfg,step)
 model=build_deployment(cfg,None,step)
 assert sum(t.numel() for t in model.inference.student.parameters())==cfg['student_sampling']['student_parameters']
from panda_sensor_focus.best_publication import project
import tempfile
with tempfile.TemporaryDirectory() as temp:
 projection=project(temp,server)
 assert not projection['conflicts'] and not projection['rows']
 assert projection['publication_thresholds']['hqnr_min']==.9315
 assert projection['publication_thresholds']['ergas_max']==3.5
from panda_sensor_focus.orchestration import predecessor_receipt
assert predecessor_receipt(Path('/missing'),server)['predecessor_queue_gate'] is False
try:install('s3' if server=='s1' else 's1')
except RuntimeError:pass
else:raise AssertionError('mixed local source origins accepted')
''')

    def test_initial_corrected_and_difference_outputs_match(self):
        self.child('''
from panda_qb_structure.adapter import install
p=install('s1')
import torch
from panda_sensor_focus.reconstruction import build_reconstructor
from panda_sensor_focus.spec import MatureSpec
from panda_sensor_focus.base_reconstruction import reconstruct_with_base
torch.set_num_threads(1);torch.manual_seed(7)
pan=torch.rand(1,1,32,32);original=torch.rand_like(pan);base=torch.rand(1,4,32,32);ms=torch.rand(1,4,8,8)
for sampling in ('BASE','UBIL'):
 models=[]
 for variant in ('CORRECTED','CORRECTED_DIFF'):
  spec=MatureSpec('QB','panda_condq',10,.25,0.,.5,1e-6,sampling_variant=sampling,input_variant=variant)
  model=build_reconstructor(spec.reconstruction_spec,seed=10,sampling_variant=sampling,input_variant=variant)
  x=pan if variant=='CORRECTED' else torch.cat((pan,pan-original),1)
  models.append((model,reconstruct_with_base(model,x,base,ms)['y']))
 assert torch.equal(models[0][1],models[1][1])
 left,right=models[0][0],models[1][0]
 assert torch.count_nonzero(right.backbone.input.weight[:,1:2])==0
 assert torch.equal(left.backbone.input.weight[:,:1],right.backbone.input.weight[:,:1])
 assert torch.equal(left.backbone.input.weight[:,1:],right.backbone.input.weight[:,2:])
''')

    def test_control_one_update_matches_authentic_prior_engine(self):
        for server in ('s1', 's3'):
            self.child(f"server={server!r}\n" + '''
from copy import deepcopy
import torch,tempfile,importlib
from panda_qb_structure import donor as D,registry as G
if server=='s3':
 source=D.metadata(server);source.update(selected_step=39000,checkpoint_sha256='c'*64,receipt_sha256='d'*64)
 old=D.metadata;D.metadata=lambda s,root=D.ROOT:deepcopy(source) if s=='s3' else old(s,root)
from panda_qb_structure.adapter import install
p=install(server)
from panda_s1loss9 import plan as prior,engine as old_engine,spec as old_spec,calibration as old_cal
from panda_s1loss9.tests.test_loss_contract import paired_donor,exact_one_image_bank
from panda_sensor_focus.tests.test_core import sources,bank,PINS
from panda_sensor_focus import engine as new_engine,spec as new_spec,calibration as new_cal
from panda_stage2_condq.stream import TrainStream,StreamSpec
from donor30.artifacts import _same
torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as temp:
 reg=sources(temp,'QB',count=1)
 fresh=p.resolve_case(server,'QB4_CONTROL','R3')
 # The authentic preceding engine is evaluated with the same fresh R3 streams.
 # Only synthetic test configs change seed; historical files are never edited.
 previous=prior.resolve_case(server,'QB_A200_E030' if server=='s1' else 'QB3_CONTROL','R1')
 previous.update(seed=fresh['seed'],stream_seeds=deepcopy(fresh['stream_seeds']))
 previous['stage2_spec']['seed']=fresh['stage2_spec']['seed']
 assert previous['training_spec']==fresh['training_spec']
 assert previous['stage2_spec']==fresh['stage2_spec']
 assert previous['augmentation']==fresh['augmentation']
 for cfg in (previous,fresh):cfg['training_spec'].update(effective_batch_size=1,microbatch_size=1)
 source_cfg=prior.resolve_case(server,G.DONOR_CASE)
 d=paired_donor(source_cfg,reg);registered_source=D.metadata(server)
 d.binding.update(source_campaign_id=registered_source['campaign_id'],
  donor_checkpoint_id=registered_source['checkpoint_sha256'],
  donor_checkpoint_sha256=registered_source['checkpoint_sha256'],donor_step=registered_source['selected_step'])
 trainers=[];streams=[]
 for cfg,S,C,E,collector in ((previous,old_spec,old_cal,old_engine,exact_one_image_bank),
                            (fresh,new_spec,new_cal,new_engine,bank)):
  spec=S.MatureSpec(**cfg['stage2_spec']);bound=C.bind_donor(d,spec)
  stats=collector(bound,spec,reg);cfg['donor_binding']=deepcopy(bound.binding)
  trainer=E.build_fresh_stage2(cfg,bound,stats,evaluator_pin=PINS['evaluator_pin'])
  trainers.append(trainer)
  stream=TrainStream(reg,StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation']),microbatch_size=1,effective_batch_size=1)
  streams.append(stream)
 left,right=trainers
 assert all(torch.equal(a,b) for a,b in zip(left.stage.student.parameters(),right.stage.student.parameters()))
 a=old_engine.advance(left,streams[0]);b=new_engine.advance(right,streams[1])
 for key in ('loss','gt_raw','hard_weighted','soft_weighted','w_h_mean','w_h_max'):
  assert a[key]==b[key],(server,key,a[key],b[key])
 assert all(torch.equal(a,b) for a,b in zip(left.stage.student.parameters(),right.stage.student.parameters()))
 assert _same(left.optimizer.state_dict(),right.optimizer.state_dict())
 assert _same(streams[0].state_dict(),streams[1].state_dict())
''')

    def test_actual_updates_and_exact_resume_all_graphs(self):
        for server in ('s1', 's3'):
            self.child(f"server={server!r}\n" + '''
from copy import deepcopy
from panda_qb_structure import donor as D,registry as G
if server=='s3':
 source=D.metadata(server);source.update(selected_step=39000,checkpoint_sha256='c'*64,receipt_sha256='d'*64)
 old=D.metadata;D.metadata=lambda s,root=D.ROOT:deepcopy(source) if s=='s3' else old(s,root)
from panda_qb_structure.adapter import install
p=install(server)
import torch,tempfile
from pathlib import Path
from panda_sensor_focus.tests.test_core import sources,bank,PINS
from panda_sensor_focus.calibration import bind_donor
from panda_sensor_focus.spec import MatureSpec
from panda_sensor_focus.engine import build_fresh_stage2,advance,save_checkpoint,load_checkpoint
from panda_sensor_focus.evaluation import configured_pipeline
from panda_stage2_condq.stream import TrainStream,StreamSpec
from panda_stage2_condq.donor import LoadedDonor
from panda_method.calibration import state_dict_sha256
from panda_s1loss9 import plan as sp,stage1
from donor30.artifacts import _same
torch.set_num_threads(1)
cases=['QB4_DIFFBASE','QB4_P0UBIL','QB4_P0BASE','QB4_WD000'] if server=='s1' else ['QB4_P0BASE','QB4_WD000']
for case in cases:
 with tempfile.TemporaryDirectory() as temp:
  reg=sources(temp,'QB',count=1);rep='R3' if server=='s3' and case=='QB4_WD000' else 'R1'
  cfg=p.resolve_case(server,case,rep);cfg['training_spec'].update(effective_batch_size=1,microbatch_size=1)
  source_cfg=sp.resolve_case(server,G.DONOR_CASE)
  source_train=stage1.build_stage1(source_cfg,reg.sources)
  a,t=source_train.stage.aligner,source_train.stage.teacher
  for m in (a,t):m.requires_grad_(False).eval()
  dsource=D.metadata(server)
  binding={**reg.inventory.manifest(),'source_split':'RR_train_only',
   'aligner_state_sha256':state_dict_sha256(a),'teacher_state_sha256':state_dict_sha256(t),
   'donor_checkpoint_id':dsource['checkpoint_sha256'],'donor_checkpoint_sha256':dsource['checkpoint_sha256'],
   'donor_run_id':source_cfg['run_id'],'donor_step':dsource['selected_step'],
   'reconstruction_spec':{'bands':4,'width':112,'depth':[1,2,3],'input_layout':'P0','base_reference':'BICUBIC_MS'},
   'scharr':{'kernel':'absolute_xy_/32','padding':'reflect1','rgb_indices':[2,1,0],'roi_margin':8},
   'warp':'lanczos4_reflect_FP32_current_pan_px','band30_teacher':stage1.teacher_binding(source_cfg),
   'mature30_teacher':stage1.teacher_binding(source_cfg),'band30_reconstruction_spec':t.spec.manifest(),
   'source_campaign_id':dsource['campaign_id']}
  donor=LoadedDonor(a,t,binding,{'synthetic_test_only':True})
  spec=MatureSpec(**cfg['stage2_spec']);donor=bind_donor(donor,spec);stats=bank(donor,spec,reg)
  cfg['donor_binding']=deepcopy(donor.binding)
  trainer=build_fresh_stage2(cfg,donor,stats,evaluator_pin=PINS['evaluator_pin'])
  configured_pipeline(trainer,reg.sources)
  seeds=StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation'])
  stream=TrainStream(reg,seeds,microbatch_size=1,effective_batch_size=1)
  before=state_dict_sha256(trainer.stage.student)
  result=advance(trainer,stream)
  assert result['loss']==result['hard_weighted'] and result['soft_weighted']==0
  assert state_dict_sha256(trainer.stage.student)!=before
  assert trainer.optimizer.param_groups[0]['weight_decay']==cfg['weight_decay']
  path=Path(temp)/'resume.pt';saved=save_checkpoint(path,trainer,stream,**PINS)
  expected=advance(trainer,stream);state=deepcopy(trainer.state_dict())
  fresh=build_fresh_stage2(cfg,donor,stats,evaluator_pin=PINS['evaluator_pin'])
  restream=TrainStream(reg,seeds,microbatch_size=1,effective_batch_size=1)
  fresh,restream=load_checkpoint(path,saved['file_sha256'],fresh,restream,**PINS)
  assert _same(advance(fresh,restream),expected) and _same(fresh.state_dict(),state)
  assert state_dict_sha256(a)==binding['aligner_state_sha256'] and state_dict_sha256(t)==binding['teacher_state_sha256']
''')

    def test_donor_paths_preserve_local_origin_and_dynamic_selection(self):
        from copy import deepcopy
        from panda_qb_structure.donor import verify_provenance, receipt_path
        for server in ('s1', 's3'):
            source = G.SOURCES[server]; step = 44000 if server == 's1' else 39000
            cfg = {'server': server, 'dataset': 'QB', 'campaign_id': source['campaign_id'],
                   'case': G.DONOR_CASE, 'seed': source['seed'], 'phase': 'donor',
                   'planned_horizon': 50000, 'lambda_abs': 2., 'lambda_epsilon': .03, 'run_id': 'local'}
            path = receipt_path(server, '/fixture').parents[2] / 'runs/local/attempt_01/models' / f'step_{step:07d}.pt'
            receipt = {'server': server, 'campaign_id': source['campaign_id'],
                       'selected': {'source_config': cfg, 'completed_update': step, 'checkpoint_path': str(path)}}
            self.assertEqual(verify_provenance(receipt, server, '/fixture')['completed_update'], step)
            bad = deepcopy(receipt); bad['selected']['source_config']['server'] = 's3' if server == 's1' else 's1'
            with self.assertRaises(ValueError): verify_provenance(bad, server, '/fixture')
            bad = deepcopy(receipt); bad['selected']['checkpoint_path'] = '/fixture/foreign/step.pt'
            with self.assertRaises(ValueError): verify_provenance(bad, server, '/fixture')

    def test_numeric_calibration_audit_ignores_only_provenance_tags(self):
        import tempfile
        import numpy as np
        from copy import deepcopy
        from panda_qb_structure.calibration_audit import compare_archives
        metadata = {'tau_e': .012, 'n_e': 4, 'n_q': 4, 'sample_count': 1,
                    'binding': {'sensor': 'QB', 'teacher_state_sha256': 'a' * 64},
                    'schema': 'authentic_old_schema'}
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder) / 'old.npz', Path(folder) / 'new.npz'
            def save(path, meta, q):
                np.savez(path, metadata_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
                         g_cutpoints_f64=np.array([.2], dtype=np.float64), q_0000=np.asarray(q, np.float32))
            save(old, metadata, [.1, .2])
            changed = deepcopy(metadata); changed['schema'] = 'new_campaign_schema'
            save(new, changed, [.1, .2])
            self.assertTrue(compare_archives(new, old)['numeric_tables_exact'])
            save(new, changed, [.1, .25])
            self.assertFalse(compare_archives(new, old)['numeric_tables_exact'])
            changed['binding']['teacher_state_sha256'] = 'b' * 64
            save(new, changed, [.1, .2])
            self.assertEqual(compare_archives(new, old)['physical_binding_differences'], ['teacher_state_sha256'])

    def test_optional_historical_read_failures_do_not_hide_current_corruption(self):
        import tempfile
        import numpy as np
        from unittest.mock import patch
        from panda_qb_structure import calibration_audit as A
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); current = root / 'current.npz'; reference = root / 'reference.npz'
            meta = {'tau_e': .012, 'binding': {'sensor': 'QB'}}
            np.savez(current, metadata_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
                     g_cutpoints_f64=np.array([.2], np.float64), q_0000=np.array([.1], np.float32))
            reference.write_bytes(b'not a valid numpy archive')
            configs = []
            for replica in ('R1', 'R2'):
                control = G.controls('s1', replica)
                run = f"TEST_{control['case']}_NOSOFT_{replica}_S{control['seed']}"
                cfg_path = root / 'work_dir' / control['package'] / control['campaign_id'] / 's1/QB/runs' / run / 'attempt_01/config.json'
                cfg_path.parent.mkdir(parents=True)
                cfg = {**control, 'calibration_artifacts': {'s100': {
                    'path': str(reference), 'file_sha256': A.file_sha(reference)}}}
                cfg_path.write_text('invalid json' if replica == 'R1' else json.dumps(cfg))
                configs.append(cfg_path)
            artifacts = {'s100': {'path': str(current), 'file_sha256': A.file_sha(current)}}
            proof = A.audit(root, root / 'campaign', 's1', artifacts)
            self.assertFalse(proof['historical_control_comparison_valid'])
            rows = json.loads(Path(proof['path']).read_text())['comparisons']
            self.assertEqual([r['status'] for r in rows],
                             ['REFERENCE_CONFIG_READ_FAILED', 'REFERENCE_ARCHIVE_READ_FAILED'])
            self.assertTrue(all(r.get('reason') for r in rows))
            original_read = Path.read_text
            def denied(path, *args, **kwargs):
                if path == configs[1]: raise OSError('fixture reference config read failure')
                return original_read(path, *args, **kwargs)
            with patch.object(Path, 'read_text', denied):
                failed = A.historical_configs(root, 's1')
                self.assertEqual(failed[1]['status'], 'REFERENCE_CONFIG_READ_FAILED')
            original_sha = A.file_sha
            def denied_sha(path):
                if Path(path) == reference: raise OSError('fixture reference archive read failure')
                return original_sha(path)
            with patch.object(A, 'file_sha', denied_sha):
                proof = A.audit(root, root / 'campaign', 's1', artifacts)
                rows = json.loads(Path(proof['path']).read_text())['comparisons']
                self.assertEqual(rows[1]['status'], 'REFERENCE_FILE_READ_FAILED')
            current.write_bytes(b'corrupt current training archive')
            # Even a matching caller hash cannot convert malformed current data
            # into an optional reference failure.
            artifacts['s100']['file_sha256'] = A.file_sha(current)
            with self.assertRaises(ValueError): A.audit(root, root / 'campaign', 's1', artifacts)


if __name__ == '__main__': unittest.main()
