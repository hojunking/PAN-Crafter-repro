import unittest,tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from checkpoint_cleanup import ledger as L
from checkpoint_cleanup.prepare import classify

class RetirementTests(unittest.TestCase):
 def test_scope_and_inclusive_cutoffs(self):
  r={'family':'panda_wv3_ablation','phase':'stage2','status':'COMPLETED','joint_pass_count':0}
  self.assertEqual(classify(r,{'horizon':30000}),('ABLATION_EARLY_STEPS',20000))
  self.assertEqual(classify(r,{'horizon':50000}),('ABLATION_EARLY_STEPS',30000))
  for key,value in [('phase','stage1'),('phase','donor'),('status','RUNNING'),('status','STOPPED')]:
   x={**r,key:value};self.assertIsNone(classify(x,{'horizon':30000}))
 def test_success_and_active_campaign_are_not_failed_cases(self):
  r={'family':'panda_wv3_main_rgb','phase':'stage2','status':'COMPLETED','joint_pass_count':1}
  self.assertIsNone(classify(r,{}));r['joint_pass_count']=0;self.assertIsNotNone(classify(r,{}))
  r['family']='panda_wv3_w128';self.assertIsNone(classify(r,{}))
 def test_tombstone_needs_committed_event_and_unchanged_evidence(self):
  with tempfile.TemporaryDirectory() as t,patch.object(L,'ROOT',Path(t)):
   root=Path(t);folder=root/'ledger';folder.mkdir();model=root/'work_dir/campaign/models/step_0001000.pt';model.parent.mkdir(parents=True)
   proof=root/'proof.json';proof.write_text('{}');h='a'*64
   entry={'path':str(model.relative_to(root)),'file_sha256':h,'evidence':[{'path':'proof.json','sha256':L.sha(proof)}],'reason':'TEST','verified_at':'test'}
   m={'schema':L.SCHEMA,'delete':[entry]};m['manifest_sha256']=L.digest(m);L.write(folder/'manifest.json',m)
   with self.assertRaises(ValueError):L.Ledger(folder).evidence(model,h)
   L.event(folder,{'event':'DELETED','path':entry['path'],'file_sha256':h,'manifest_sha256':m['manifest_sha256']})
   self.assertFalse(L.Ledger(folder).evidence(model,h)['weights_currently_available'])
   proof.write_text('changed')
   with self.assertRaises(ValueError):L.Ledger(folder).evidence(model,h)
 def test_symlinks_and_path_escape_rejected(self):
  with tempfile.TemporaryDirectory() as t,patch.object(L,'ROOT',Path(t)):
   root=Path(t);outside=root/'outside.pt';outside.write_bytes(b'x');work=root/'work_dir';work.mkdir();p=work/'link.pt';p.symlink_to(outside)
   with self.assertRaises(ValueError):L.safe_file(p)
   with self.assertRaises(ValueError):L.safe_file(outside)
   with self.assertRaises(ValueError):L.safe_file(work/'../outside.pt')
if __name__=='__main__':unittest.main()
