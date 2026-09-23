"""Design-only tests. No GPU, external service, or production runner invocation."""
import copy
import unittest
import build_registry as b

class RegistryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.doc=b.build(); cls.rows=cls.doc['all']; cls.byid={r['run_id']:r for r in cls.rows}
 def test_01_inventory(self):
  self.assertEqual(len(self.rows),300)
  self.assertEqual(len(self.byid),300)
 def test_02_legacy_preservation(self):
  for old in self.doc['legacy']:
   now=self.byid[old['run_id']]
   self.assertTrue(all(now[k]==v for k,v in old.items()))
 def test_03_append_scope(self):
  for srv in ('s1','s2'):
   new=[r for r in self.rows if r['server']==srv and r['admission_action']=='APPEND_IF_ABSENT']
   self.assertEqual(len(new),5)
   self.assertEqual({r['case_id'] for r in new},{'C17'})
 def test_04_gf2_full_five(self):
  gf=[r for r in self.rows if r['server']=='s3']
  self.assertEqual(len(gf),100)
  self.assertEqual(sum(r['role']=='T' for r in gf),10)
  self.assertEqual(sum(r['role']=='S' for r in gf),90)
  self.assertEqual({r['max_dn'] for r in gf},{'1023'})
  self.assertEqual({r['rr_q'] for r in gf},{'Q4'})
 def test_05_shared_architecture(self):
  for r in self.rows:
   self.assertEqual((r['width'],r['depth']),('112','1,2,3') if r['role']=='T' else ('104','1,2,2'))
   self.assertEqual(int(r['stored_input_channels']),int(r['bands'])+(1 if r['role']=='T' else 3))
 def test_06_c17_no_teacher_inference(self):
  c=next(r for r in self.doc['catalog'] if r['case_id']=='C17')
  self.assertEqual(c['teacher'],'TZERO')
  self.assertEqual(c['aligner'],'CLONE_TRAINABLE')
  for key in ('teacher_predictions_used','teacher_q_used'):
   self.assertEqual(c[key],'False')
  for key in ('alpha','beta','lambda_edge'):
   self.assertEqual(float(c[key]),0.)
  self.assertEqual(c['q_aligner'],'CONST_HALF')
 def test_07_c17_anchor_equivalence(self):
  for z in [r for r in self.rows if r['case_id']=='C17']:
   a=self.byid[z['anchor_run_id']]
   self.assertEqual(a['case_id'],'C03')
   for k in ('student_seed','teacher_seed','seed_group','width','depth','u_peak_lr','a_peak_lr','num_updates','requires_calibration'):
    self.assertEqual(a[k],z[k])
   self.assertNotEqual(a['teacher_run_id'],z['teacher_run_id'])
 def test_08_local_reference_bindings(self):
  for r in self.rows:
   if r['role']=='S' and r['requires_teacher']=='True':
    t=self.byid[r['teacher_run_id']]
    self.assertEqual(t['server'],r['server'])
    self.assertEqual(t['sensor'],r['sensor'])
    self.assertEqual(t['teacher_seed'],r['teacher_seed'])
    self.assertEqual(t['reference_id'],r['reference_id'])
 def test_09_no_hidden_reference(self):
  for r in self.rows:
   if r['case_id'] in ('C00','C01','C02','C10'):
    self.assertEqual(r['requires_teacher'],'False')
    self.assertFalse(r['reference_id'])
    self.assertFalse(r['teacher_run_id'])
 def test_10_order_unchanged(self):
  for srv in ('s1','s2'):
   old=[r['run_id'] for r in sorted([x for x in self.doc['legacy'] if x['server']==srv],key=lambda r:(r['sweep_id'],float(r['queue_rank'])))]
   new=[r['run_id'] for r in self.rows if r['server']==srv and r['case_id']!='C17']
   self.assertEqual(old,new)
  b.validate(self.doc)
 def test_11_single_line_labels(self):
  for r in self.rows:
   self.assertNotIn('\n',r['display_label'])
   self.assertNotIn('\r',r['display_label'])
   self.assertNotRegex(r['display_label'],r'\b(?:SS|TS)\d+')
 def test_12_graph_scope(self):
  g={x['relation_id']:x for x in self.doc['graph']}
  self.assertEqual(len(g),19)
  self.assertEqual((g['A17']['parent'],g['A17']['child']),('C17','C03'))
  self.assertEqual(g['A17_SCRATCH']['queue_enabled'],'False')
 def test_13_until_stop_and_protected_lanes(self):
  r=self.doc['registry']; l=r['loop']
  self.assertEqual(set(r['lanes']),{'s1','s2','s3'})
  self.assertEqual(set(r['protected_servers']),{'s4','s5'})
  self.assertIsNone(l['max_campaign_cycles'])
  self.assertIsNone(l['lease_hours'])
  self.assertEqual(l['service_mode'],'UNTIL_OPERATOR_STOP')
  self.assertFalse(l['shared_lock'])
  self.assertFalse(l['cross_server_barrier'])
  self.assertTrue(l['retain_local_duplicate_guard'])
 def test_14_method_and_training_budget(self):
  for r in self.rows:
   self.assertEqual(r['num_updates'],'50000')
   self.assertEqual(r['cosine_total'],'50000')
   self.assertEqual(r['panmix_enabled'],'False')
   self.assertEqual(r['runtime_bound'],'False')
   self.assertEqual(r['evidence_role'],'DESIGN_ONLY_NOT_LAUNCHED')
 def test_15_deterministic_rebuild(self):
  self.assertEqual(self.doc,b.build())
 def test_16_reject_corrupted_gf2_scale(self):
  bad=copy.deepcopy(self.doc)
  row=next(x for x in bad['all'] if x['server']=='s3')
  row['max_dn']='2047'
  with self.assertRaises(AssertionError): b.validate(bad)
 def test_17_reject_c17_u_seed_change(self):
  bad=copy.deepcopy(self.doc)
  row=next(x for x in bad['all'] if x['case_id']=='C17')
  row['student_seed']='999'
  with self.assertRaises(AssertionError): b.validate(bad)
 def test_18_reject_legacy_mutation(self):
  bad=copy.deepcopy(self.doc)
  row=next(x for x in bad['all'] if x['server']=='s1' and x['case_id']=='C00')
  row['u_peak_lr']='0.0002'
  with self.assertRaises(AssertionError): b.validate(bad)

if __name__=='__main__': unittest.main()
