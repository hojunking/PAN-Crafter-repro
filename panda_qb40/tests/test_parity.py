"""Check the unchanged loss/forward against the authentic preceding engine."""
import unittest,tempfile
from copy import deepcopy
import torch
from panda_sensor5.tests.test_core import sources
from panda_s1loss9.tests.test_loss_contract import paired_donor,exact_one_image_bank
from panda_s1loss9 import plan as old_plan,engine as old_engine,spec as old_spec,calibration as old_cal
from panda_qb40 import plan,engine,spec,calibration
from panda_qb40.tests.test_execution import bank,PINS

class ParityTest(unittest.TestCase):
 def test_same_weights_forward_hard_loss_and_gradient(self):
  torch.set_num_threads(1)
  with tempfile.TemporaryDirectory() as tmp:
   reg=sources(tmp,'QB',count=1)
   source=old_plan.resolve_case('s1','S1_A200_E030');d=paired_donor(source,reg)
   d.binding['source_campaign_id']=source['campaign_id']
   old=old_plan.resolve_case('s1','QB_A200_E030');new=plan.resolve_case('s1','QB5_C60LR050')
   trainers=[]
   for cfg,S,C,E,collect in [(old,old_spec,old_cal,old_engine,exact_one_image_bank),(new,spec,calibration,engine,bank)]:
    model=S.MatureSpec(**cfg['stage2_spec']);bound=C.bind_donor(d,model)
    cfg['donor_binding']=deepcopy(bound.binding)
    trainers.append(E.build_fresh_stage2(cfg,bound,collect(bound,model,reg),evaluator_pin=PINS['evaluator_pin']))
   outputs=[]
   for t in trainers:
    result=t.stage(reg.batch([0]));result['loss'].backward();outputs.append(result)
   self.assertTrue(torch.equal(outputs[0]['loss'],outputs[1]['loss']))
   for (na,a),(nb,b) in zip(trainers[0].stage.student.named_parameters(),trainers[1].stage.student.named_parameters()):
    self.assertEqual(na,nb);self.assertTrue(torch.equal(a,b))
    self.assertTrue(torch.equal(a.grad,b.grad))

if __name__=='__main__':unittest.main()
