"""CPU real-tensor parity, frozen A/T and complete Student resume; synthetic data only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import torch
from donor30.artifacts import _same
from panda_method.calibration import state_dict_sha256
from panda_sensor5.tests.test_core import sources
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.stream import TrainStream,StreamSpec
from panda_qb_s2 import plan as original,stage1
from panda_qb_s2 import spec as original_spec,calibration as original_calibration
from panda_qb_s2.method import MatureStage2 as OriginalStage2
from panda_qb_s2_frozen import plan,registry as G,calibration,spec,engine
from panda_qb_s2_frozen.calibration_core import CalibrationBuilder
from panda_qb_s2_frozen.method import MatureStage2
from panda_qb_s2_frozen.evaluation import configured_pipeline

PINS={'code_pin':{'synthetic_cpu':'a'*64},'evaluator_pin':{'synthetic_cpu':'b'*64}}

def synthetic_donor(reg,alias='D4'):
    pin=G.donor(alias);cfg=original.resolve_case('s2',pin['source_case'])
    module=stage1.build_stage1(cfg,reg.sources).stage
    a,t=module.aligner,module.teacher
    for m in (a,t):m.requires_grad_(False).eval()
    binding={**reg.inventory.manifest(),'source_split':'RR_train_only',
        'aligner_state_sha256':state_dict_sha256(a),'teacher_state_sha256':state_dict_sha256(t),
        'donor_checkpoint_id':pin['checkpoint_sha256'],'donor_checkpoint_sha256':pin['checkpoint_sha256'],
        'donor_run_id':cfg['run_id'],'donor_step':48000,
        'mature30_selector_sha256':pin['selector_receipt_sha256'],
        'reconstruction_spec':dict(bands=4,width=112,depth=[1,2,3],input_layout='P0',base_reference='BICUBIC_MS'),
        'scharr':dict(kernel='absolute_xy_/32',padding='reflect1',rgb_indices=[2,1,0],roi_margin=8),
        'warp':'lanczos4_reflect_FP32_current_pan_px','band30_teacher':stage1.teacher_binding(cfg),
        'mature30_teacher':stage1.teacher_binding(cfg),'band30_reconstruction_spec':t.spec.manifest(),
        'source_campaign_id':cfg['campaign_id']}
    return LoadedDonor(a,t,binding,{'synthetic_only':True})

def bank(d,model,reg):
    ids=list(reg.sample_ids)
    v=torch.linspace(.1,.9,64*64).reshape(1,1,64,64).repeat(len(ids),1,1,1)
    builder=CalibrationBuilder(ids,calibration.binding_for_geometry(d.binding,'s100'),k=5,
        memory_budget_bytes=64*1024**2,include_condq=True,mask_policy=model.mask_policy)
    builder.add(ids,v,v.flip(-1),v.flip(-2),sensor='QB',loss_mask=torch.ones_like(v,dtype=torch.bool))
    return {'s100':builder.finalize()}

class ScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_archived_baseline_r1_forward_loss_gradient_and_calibration_exact_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg=sources(tmp,'QB',count=1)
            for alias in ('D4','D2'):
                d=synthetic_donor(reg,alias)
                oldcfg=original.resolve_case('s2',G.donor(alias)['source_case'].replace('_S1_','_S2_'))
                ns=spec.MatureSpec(**oldcfg['stage2_spec']);os=original_spec.MatureSpec(**oldcfg['stage2_spec'])
                nd=calibration.bind_donor(d,ns);od=original_calibration.bind_donor(d,os)
                self.assertEqual(nd.binding,od.binding)
                stats=bank(nd,ns,reg)
                a=OriginalStage2(os,od.aligner,od.teacher,donor_binding=od.binding,stats_bank=stats)
                b=MatureStage2(ns,nd.aligner,nd.teacher,donor_binding=nd.binding,stats_bank=stats)
                x,y=a(reg.batch([0])),b(reg.batch([0]))
                self.assertTrue(torch.equal(x['loss'],y['loss']))
                x['loss'].backward();y['loss'].backward()
                for ap,bp in zip(a.student.parameters(),b.student.parameters()):
                    self.assertTrue(torch.equal(ap,bp))
                    self.assertTrue(torch.equal(ap.grad,bp.grad))
                path=Path(tmp)/(alias+'.npz');receipt=stats['s100'].save(path)
                loaded=calibration.load_stats_bank(nd,ns,{'s100':{'path':str(path),'file_sha256':receipt['file_sha256']}},memory_budget_bytes=64*1024**2)
                self.assertEqual(loaded['s100'].metadata,stats['s100'].metadata)

    def test_every_recipe_actual_graph_loss_and_frozen_modules(self):
        from panda_qb_s2_frozen.cost.profiling import build_deployment
        with tempfile.TemporaryDirectory() as tmp:
            reg=sources(tmp,'QB',count=1);donors={a:synthetic_donor(reg,a) for a in ('D4','D2')}
            for c in plan.CASES:
                r='R2' if c in ('Q0_BASE','Q6_D2_BASE') else 'R1'
                cfg=plan.resolve_case('s2',c,r);s=spec.MatureSpec(**cfg['stage2_spec'])
                d=calibration.bind_donor(donors[G.job('s2',c,r)['donor_alias']],s)
                cfg['donor_binding']=deepcopy(d.binding)
                trainer=engine.build_fresh_stage2(cfg,d,bank(d,s,reg),evaluator_pin=PINS['evaluator_pin'])
                configured_pipeline(trainer,reg.sources)
                out=trainer.stage(reg.batch([0]));out['loss'].backward()
                self.assertTrue(torch.isfinite(out['loss']))
                self.assertEqual(sum(p.numel() for p in trainer.stage.student.parameters()),cfg['student_sampling']['student_parameters'])
                for module in (trainer.stage.aligner,trainer.stage.teacher):
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
                deployment=build_deployment(cfg,reg.sources,1000)
                with torch.no_grad():y=deployment.native_prediction(reg.batch([0]))['y']
                self.assertEqual(y.shape,reg.batch([0]).gt.shape)

    def test_w128_actual_adam_rng_stream_complete_resume_and_frozen_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg=sources(tmp,'QB',count=12);cfg=plan.resolve_case('s2','Q3_W128H050')
            s=spec.MatureSpec(**cfg['stage2_spec']);d=calibration.bind_donor(synthetic_donor(reg),s)
            cfg['donor_binding']=deepcopy(d.binding);stats=bank(d,s,reg)
            trainer=engine.build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
            ss=StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation'])
            stream=TrainStream(reg,ss,microbatch_size=12,effective_batch_size=48)
            engine.advance(trainer,stream)
            saved=engine.save_checkpoint(Path(tmp)/'resume.pt',trainer,stream,**PINS)
            expected=engine.advance(trainer,stream);state=deepcopy(trainer.state_dict())
            fresh=engine.build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
            fs=TrainStream(reg,ss,microbatch_size=12,effective_batch_size=48)
            fresh,fs=engine.load_checkpoint(Path(tmp)/'resume.pt',saved['file_sha256'],fresh,fs,**PINS)
            self.assertTrue(_same(engine.advance(fresh,fs),expected))
            self.assertTrue(_same(fresh.state_dict(),state))
            self.assertTrue(_same(fs.state_dict(),stream.state_dict()))
            self.assertEqual(state_dict_sha256(d.aligner),d.binding['aligner_state_sha256'])
            self.assertEqual(state_dict_sha256(d.teacher),d.binding['teacher_state_sha256'])
