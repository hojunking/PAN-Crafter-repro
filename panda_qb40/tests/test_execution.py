"""Real FP32 Student updates at registered batch sizes and complete resume."""
import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
import torch
from donor30.artifacts import _same
from panda_method.calibration import state_dict_sha256,canonical_sha256
from panda_sensor5.tests.test_core import sources
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.stream import TrainStream,StreamSpec
from panda_qb40 import plan
from panda_qb40.stage1 import build_stage1,teacher_binding
from panda_qb40.calibration import bind_donor,binding_for_geometry
from panda_qb40.calibration_core import CalibrationBuilder
from panda_qb40.spec import MatureSpec
from panda_qb40.engine import build_fresh_stage2,advance,save_checkpoint,load_checkpoint
from panda_qb40.evaluation import configured_pipeline
PINS={'code_pin':{'synthetic_cpu':'a'*64},'evaluator_pin':{'synthetic_cpu':'b'*64}}

def donor(cfg,reg):
 t=build_stage1(cfg,reg.sources);a,t=t.stage.aligner,t.stage.teacher
 for m in (a,t):m.requires_grad_(False).eval()
 binding={**reg.inventory.manifest(),'source_split':'RR_train_only',
  'aligner_state_sha256':state_dict_sha256(a),'teacher_state_sha256':state_dict_sha256(t),
  'donor_checkpoint_id':'c'*64,'donor_run_id':cfg['run_id'],'donor_step':1000,
  'reconstruction_spec':dict(bands=4,width=112,depth=[1,2,3],input_layout='P0',base_reference='BICUBIC_MS'),
  'scharr':dict(kernel='absolute_xy_/32',padding='reflect1',rgb_indices=[2,1,0],roi_margin=8),
  'warp':'lanczos4_reflect_FP32_current_pan_px','band30_teacher':teacher_binding(cfg),
  'mature30_teacher':teacher_binding(cfg),'band30_reconstruction_spec':t.spec.manifest(),
  'source_campaign_id':cfg['campaign_id']}
 return LoadedDonor(a,t,binding,{'synthetic_only':True})

def bank(d,spec,reg):
 ids=list(reg.sample_ids);v=torch.linspace(.1,.9,64*64).reshape(1,1,64,64).repeat(len(ids),1,1,1)
 c=CalibrationBuilder(ids,binding_for_geometry(d.binding,'s100'),k=5,memory_budget_bytes=64*1024**2,
  include_condq=True,mask_policy=spec.mask_policy)
 c.add(ids,v,v.flip(-1),v.flip(-2),sensor='QB',loss_mask=torch.ones_like(v,dtype=torch.bool))
 return {'s100':c.finalize()}

class ExecutionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_registered_b96_actual_adam_stream_and_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   reg=sources(tmp,'QB',count=12)
   cfg=plan.resolve_case('s1','QB5_B96C15LR100')
   # Authentic historical model class and source binding; test data only.
   from panda_s1loss9 import plan as old
   from panda_s1loss9.tests.test_loss_contract import paired_donor
   source=old.resolve_case('s1','S1_A200_E030');d=paired_donor(source,reg)
   d.binding['source_campaign_id']=source['campaign_id']
   spec=MatureSpec(**cfg['stage2_spec']);d=bind_donor(d,spec);stats=bank(d,spec,reg)
   cfg['donor_binding']=deepcopy(d.binding)
   trainer=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
   configured_pipeline(trainer,reg.sources)
   seeds=StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation'])
   stream=TrainStream(reg,seeds,microbatch_size=12,effective_batch_size=96)
   initial=state_dict_sha256(trainer.stage.student)
   first=advance(trainer,stream)
   self.assertEqual(stream.epoch,8)
   self.assertEqual(first['loss'],first['hard_weighted']);self.assertEqual(first['soft_weighted'],0.)
   self.assertNotEqual(initial,state_dict_sha256(trainer.stage.student))
   saved=save_checkpoint(Path(tmp)/'resume.pt',trainer,stream,**PINS)
   second=advance(trainer,stream);state=deepcopy(trainer.state_dict())
   fresh=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
   restream=TrainStream(reg,seeds,microbatch_size=12,effective_batch_size=96)
   fresh,restream=load_checkpoint(Path(tmp)/'resume.pt',saved['file_sha256'],fresh,restream,**PINS)
   self.assertTrue(_same(advance(fresh,restream),second))
   self.assertTrue(_same(fresh.state_dict(),state));self.assertTrue(_same(restream.state_dict(),stream.state_dict()))
   self.assertEqual(state_dict_sha256(d.aligner),d.binding['aligner_state_sha256'])
   self.assertEqual(state_dict_sha256(d.teacher),d.binding['teacher_state_sha256'])

 def test_new_branch_donor_and_w128_native_forward(self):
  from panda_qb40.cost.profiling import build_deployment
  for server,case in [('s1','QB5_S1_A200_E100'),('s3','QB5_S1_A400_E100')]:
   with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
    reg=sources(tmp,'QB',count=12);cfg=plan.resolve_case(server,case)
    trainer=build_stage1(cfg,reg.sources)
    rr=reg.batch([0]);out=trainer.stage(rr)
    self.assertAlmostEqual(float(out['loss']),float(out['rec']+cfg['lambda_abs']*out['abs_rr']+cfg['lambda_epsilon']*out['eps_rr']),places=7)
    out['loss'].backward()
    self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in trainer.stage.aligner.parameters()))
    self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in trainer.stage.teacher.parameters()))
    model=build_deployment(plan.resolve_case('s3','QB5_W128H050'),None,1000)
    with torch.no_grad():y=model.native_prediction(rr)['y']
    self.assertEqual(y.shape,rr.gt.shape)
    self.assertEqual(model.inference.student.spec.width,128)

 def test_stage1_full_optimizer_rng_save_restore(self):
  from panda_qb40.stage1 import Stage1Stream,advance_stage1,save_stage1,load_stage1
  with tempfile.TemporaryDirectory() as tmp:
   reg=sources(tmp,'QB',count=12);cfg=plan.resolve_case('s1','QB5_S1_A200_E100')
   trainer=build_stage1(cfg,reg.sources);stream=Stage1Stream(reg.sources,cfg,device="cpu")
   advance_stage1(trainer,stream)
   artifact=save_stage1(Path(tmp)/'stage1.pt',trainer,stream,**PINS)
   expected=advance_stage1(trainer,stream);state=deepcopy(trainer.state_dict())
   fresh,restream,_=load_stage1(Path(tmp)/'stage1.pt',artifact['file_sha256'],cfg,reg.sources,**PINS)
   actual=advance_stage1(fresh,restream)
   self.assertTrue(_same(expected,actual));self.assertTrue(_same(state,fresh.state_dict()))
   self.assertTrue(_same(stream.state_dict(),restream.state_dict()))

if __name__=='__main__':unittest.main()
