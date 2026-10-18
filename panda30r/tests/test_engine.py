"""Actual Direct/W112 CPU numerical and restart checks for isolated engines."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from directshift48.method import DirectPANAligner, epsilon_loss
from donor30.artifacts import _same
from donor30.masking import masked_mean
from pa.warp import warp_pan
from panda_method.calibration import state_dict_sha256
from panda_method.reconstruction import build_reconstructor
from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY
from panda_stage2_condq.training import TrainingSpec
from panda_stage2_condq.tests.test_method import batch
from panda30r.engine import (NativeAdaptation, NativeTrainer, _import_moments,
    _validate_config, advance, save_checkpoint, load_checkpoint,
    build_mature_trainer, build_tail_trainer, load_mature_donor)

torch.set_num_threads(1)


def config(kind="tail_fc2_eps"):
    mature = kind == "mature_teacher"
    training = TrainingSpec(lr=1e-4 if mature else 1e-5,
        total_updates=50000 if mature else 10000,warmup_updates=100 if mature else 0,
        effective_batch_size=1,microbatch_size=1,full_audit_interval=100)
    return {"run_id":"TEST_"+kind,"campaign_id":"PANDA30R_TEST","server":"s1",
        "attempt":1,"seed":261018101,"dataset":"WV3","engine_kind":kind,
        "start_update":7000 if mature else 40000,
        "planned_horizon":43000 if mature else 10000,
        "training_spec":asdict(training),"epsilon_seed":12345}


def fixture(kind="tail_fc2_eps", *, update=0):
    cfg = config(kind)
    spec = Stage2Spec("WV3","gt_only",cfg["seed"],0.,0.,0.,1e-6,mask_policy=MASK2_POLICY)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(333)
        aligner = DirectPANAligner(8)
        with torch.no_grad():
            aligner.fc2.weight.normal_(std=.001)
            aligner.fc2.bias.copy_(torch.tensor([.5,-.25]))
        reconstruction = build_reconstructor(spec.reconstruction_spec,seed=42,role="T" if kind=="mature_teacher" else "S")
    binding = {"sensor":"WV3","source_manifest_sha256":"a"*64,"sample_count":1,
        "aligner_state_sha256":state_dict_sha256(aligner),
        "teacher_state_sha256":state_dict_sha256(reconstruction)}
    stage = NativeAdaptation(aligner,reconstruction,cfg,donor_binding=binding,parent_sha="c"*64)
    trainer = NativeTrainer(stage,cfg,evaluator_pin={"test":"explicit"})
    # Synthetic completed-parent moments. They test exact preservation, not
    # claims that a real model was trained for 40K/7K within this unit fixture.
    source = {"optimizer":deepcopy(trainer.optimizer.state_dict())}
    source["optimizer"]["param_groups"] = source["optimizer"]["param_groups"][:1]
    expected = [("teacher." if kind=="mature_teacher" else "")+name
                for name,_ in reconstruction.named_parameters()]
    source["optimizer_parameter_names"] = [expected]
    ids = source["optimizer"]["param_groups"][0]["params"]
    for pid,parameter in zip(ids,reconstruction.parameters()):
        source["optimizer"]["state"][pid] = {"step":torch.tensor(float(cfg["start_update"])),
            "exp_avg":torch.zeros_like(parameter),"exp_avg_sq":torch.zeros_like(parameter)}
    _import_moments(trainer,source,0,expected)
    if update:
        trainer.updates = update
        stage.set_progress(update)
        stage.forward_calls = update
        for group in trainer.optimizer.param_groups:
            group["lr"] = trainer._head_lr(update) if group["role"]=="A" else trainer._rec_lr(update)
            steps = min(update,5000) if group["role"]=="A" else cfg["start_update"]+update
            for parameter in group["params"]:
                trainer.optimizer.state[parameter] = {"step":torch.tensor(float(steps)),
                    "exp_avg":torch.zeros_like(parameter),"exp_avg_sq":torch.zeros_like(parameter)}
        trainer.assert_contract()
    return trainer


class MemoryStream:
    """Small no-I/O deterministic stream to isolate full checkpoint machinery."""
    count = 1
    def __init__(self,lifetime):
        self.epoch,self.cursor=lifetime,0
    def state_dict(self):
        return {"epoch":self.epoch,"cursor":self.cursor,"fixture":True}
    def load_state_dict(self,state):
        if state.get("fixture") is not True:
            raise ValueError("different stream")
        self.epoch,self.cursor=state["epoch"],state["cursor"]
    def next_update(self):
        self.epoch+=1
        return [batch()]


class EngineTests(unittest.TestCase):
    def test_original_architecture_and_no_teacher_calibration_in_tail(self):
        trainer=fixture()
        self.assertIsNone(trainer.stage.teacher)
        self.assertFalse(hasattr(trainer.stage,"stats"))
        self.assertEqual(trainer.stage.student.spec.width,112)
        self.assertEqual(trainer.stage.student.spec.depth,(1,2,3))
        self.assertEqual([n for n,p in trainer.stage.aligner.named_parameters() if p.requires_grad],
                         ["fc2.weight","fc2.bias"])

    def test_tail_gradients_and_native_onewarp_plus_epsilon_inputwarp(self):
        trainer=fixture()
        stage=trainer.stage
        body={n:p.detach().clone() for n,p in stage.aligner.named_parameters() if not n.startswith("fc2.")}
        head=stage.aligner.fc2.weight.detach().clone()
        with patch("panda30r.engine.warp_pan",wraps=warp_pan) as warp:
            result=trainer.step_accumulated([batch()])
            self.assertEqual(warp.call_count,2) # native reconstruction and independent synthetic PAN only
        self.assertEqual(result["update"],1)
        self.assertGreater(result["gradient"]["S"],0)
        self.assertGreater(result["gradient"]["A"],0)
        self.assertFalse(torch.equal(head,stage.aligner.fc2.weight))
        for name,parameter in stage.aligner.named_parameters():
            if name in body:
                self.assertTrue(torch.equal(body[name],parameter))
                self.assertIsNone(parameter.grad)
        trainer.assert_contract()

    def test_mature_rec_only_frozen_A_and_original_schedule(self):
        trainer=fixture("mature_teacher")
        stage=trainer.stage
        original=state_dict_sha256(stage.aligner)
        epsilon=stage._epsilon_rng.get_state().clone()
        with patch("panda30r.engine.warp_pan",wraps=warp_pan) as warp:
            result=trainer.step_accumulated([batch()])
            self.assertEqual(warp.call_count,1)
        self.assertEqual(original,state_dict_sha256(stage.aligner))
        self.assertTrue(torch.equal(epsilon,stage._epsilon_rng.get_state()))
        self.assertEqual(result["eps_rr"],0.)
        self.assertEqual(result["abs_rr"],0.)
        self.assertEqual(result["learning_rates"]["T"],trainer.training.rate(7001))
        self.assertTrue(all(float(s["step"])==7001 for s in trainer.optimizer.state.values()))

    def test_mask_detached_recomputed_and_loss_matches_existing_reduction(self):
        trainer=fixture("tail_gt")
        out=trainer.stage(batch())
        self.assertFalse(out["loss_mask"].requires_grad)
        self.assertEqual(out["loss_mask"].dtype,torch.bool)
        self.assertLess(float(out["loss_mask"].float().mean()),1.)
        expected=masked_mean((out["native"]["y"]-batch().gt).abs(),out["loss_mask"])
        torch.testing.assert_close(out["loss"],expected,rtol=0,atol=0)

    def test_inference_never_mutates_trainable_head_and_drops_GT(self):
        trainer=fixture()
        stage=trainer.stage
        before=[p.requires_grad for p in stage.parameters()]
        observation=batch(split="test")
        for call in (stage.native_prediction,stage.inference_model()):
            out=call(replace(observation,gt=torch.tensor(float("nan"))))
            self.assertFalse(out["y"].requires_grad)
            self.assertEqual(before,[p.requires_grad for p in stage.parameters()])
        trainer.assert_contract()

    def test_fixed_ROI8_violation_rolls_back_epsilon_and_stream(self):
        trainer=fixture()
        with torch.no_grad():
            trainer.stage.aligner.fc2.bias.fill_(10.)
        stream=MemoryStream(40000)
        before=trainer.stage._epsilon_rng.get_state().clone()
        with self.assertRaisesRegex(ValueError,"SUPPORT_VIOLATION"):
            advance(trainer,stream)
        self.assertEqual(stream.epoch,40000)
        self.assertEqual(trainer.updates,0)
        self.assertEqual(trainer.stage.forward_calls,0)
        self.assertFalse(trainer.poisoned)
        self.assertTrue(torch.equal(before,trainer.stage._epsilon_rng.get_state()))

    def test_native_anchor_is_detached_in_epsilon_objective(self):
        native=torch.tensor([[.3,-.2]],requires_grad=True)
        shifted=torch.tensor([[.2,-.1]],requires_grad=True)
        loss=epsilon_loss(shifted,native,torch.tensor([[1.,2.]]))
        loss.backward()
        self.assertIsNone(native.grad)
        self.assertIsNotNone(shifted.grad)

    def test_weak_absolute_uses_existing_scharr_and_explicit_coefficients(self):
        trainer=fixture("tail_fc2_abs")
        observation=batch()
        out=trainer.stage(observation)
        expected_abs=trainer.stage.spatial.absolute_loss(out["native"]["pan_aligned"],observation.gt,roi_margin=8)
        torch.testing.assert_close(out["abs_rr"],expected_abs,rtol=0,atol=0)
        torch.testing.assert_close(out["loss"],out["rec"]+.2*out["abs_rr"]+.003*out["eps_rr"],rtol=0,atol=0)
        cfg=config("tail_fc2_abs")
        cfg.update(start_update=20000,planned_horizon=30000)
        cfg["training_spec"]["total_updates"]=30000
        _validate_config(cfg)

    def test_exact_5000_boundary_freezes_head_and_does_not_decay_it_later(self):
        trainer=fixture(update=4999)
        trainer.step_accumulated([batch()])
        self.assertFalse(trainer.stage.head_active)
        self.assertTrue(all(not p.requires_grad for p in trainer.stage.aligner.parameters()))
        saved=state_dict_sha256(trainer.stage.aligner)
        rng=trainer.stage._epsilon_rng.get_state().clone()
        out=trainer.step_accumulated([batch()])
        self.assertEqual(saved,state_dict_sha256(trainer.stage.aligner))
        self.assertTrue(torch.equal(rng,trainer.stage._epsilon_rng.get_state()))
        self.assertEqual(out["eps_rr"],0.)
        group=trainer.optimizer.param_groups[1]
        self.assertEqual(group["lr"],0.)
        self.assertTrue(all(float(trainer.optimizer.state[p]["step"])==5000 for p in group["params"]))

    def test_complete_resume_identical_next_update_and_cross_config_rejected(self):
        trainer=fixture()
        stream=MemoryStream(40000)
        advance(trainer,stream)
        pins={"code_pin":{"engine.py":"d"*64},"evaluator_pin":{"test":"explicit"}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"resume.pt"
            artifact=save_checkpoint(path,trainer,stream,**pins)
            other=fixture()
            other_stream=MemoryStream(40000)
            load_checkpoint(path,artifact["file_sha256"],other,other_stream,**pins)
            advance(trainer,stream)
            advance(other,other_stream)
            self.assertTrue(_same(trainer.state_dict(),other.state_dict()))
            self.assertEqual(stream.state_dict(),other_stream.state_dict())
            with self.assertRaises(FileExistsError):
                save_checkpoint(path,trainer,stream,**pins)
            other.config["run_id"]="OTHER_RUN"
            with self.assertRaisesRegex(ValueError,"contract mismatch"):
                load_checkpoint(path,artifact["file_sha256"],other,other_stream,**pins)

    def test_mutation_source_and_schedule_guards(self):
        trainer=fixture()
        with torch.no_grad():
            trainer.stage.aligner.fc1.bias.add_(1.)
        with self.assertRaisesRegex(ValueError,"Frozen Aligner"):
            trainer.assert_contract()
        cfg=config()
        cfg["training_spec"]["lr"]=1e-4
        with self.assertRaisesRegex(ValueError,"1e-5"):
            _validate_config(cfg)
        trainer=fixture()
        bad=replace(batch(),lineage=replace(batch().lineage,split="test"))
        with self.assertRaises(ValueError):
            trainer.step_accumulated([bad])
        self.assertEqual(trainer.updates,0)

    def test_real_h5_legacy_teacher_import_preserves_moments_sampler_and_pair(self):
        from donor24 import plan
        from donor30.runtime import build_trainer
        from donor30.artifacts import SCHEMA,capture_rng
        from panda40.stage1 import Stage1Stream
        from donor24.evaluation import model_state_hash
        from edgeres48.stream import learning_rates
        from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
        from ta2.evaluation import file_sha
        with tempfile.TemporaryDirectory() as directory:
            registered=synthetic_sources(directory)
            old_cfg=plan.resolve_case(plan.default_config("s1"),"A0")
            old_cfg["run_id"]="SYNTHETIC_LEGACY_IMPORT"
            old_cfg["baseline_resolved"].update(effective_batch_RR=1,microbatch_RR=1,accumulation_RR=1)
            old=build_trainer(old_cfg,registered.sources)
            old.updates=7
            old.stage.forward_calls=7
            for group in old.optimizer.param_groups:
                group["lr"]=learning_rates(old_cfg,7)[group["role"]]
                for parameter in group["params"]:
                    old.optimizer.state[parameter]={"step":torch.tensor(7.),
                        "exp_avg":torch.zeros_like(parameter),"exp_avg_sq":torch.zeros_like(parameter)}
            stream=Stage1Stream(registered.sources,old_cfg,device="cpu")
            for _ in range(7):
                stream.next_update()
            joint=model_state_hash(old.stage)
            payload={"schema":SCHEMA,"config":old_cfg,"trainer":old.state_dict(),
                "stream":stream.state_dict(),"rng":capture_rng(),"state_hash":joint,
                "identity":{"model_state_sha256":joint,"completed_update":7,
                            "run_id":old_cfg["run_id"],"dataset":"WV3"}}
            path=Path(directory)/"legacy.pt"
            torch.save(payload,path)
            cfg=config("mature_teacher")
            cfg.update(start_update=7,planned_horizon=49993)
            trainer,new_stream=build_mature_trainer(cfg,registered,path,file_sha(path))
            self.assertEqual(state_dict_sha256(old.stage.aligner),state_dict_sha256(trainer.stage.aligner))
            self.assertEqual(state_dict_sha256(old.stage.teacher),state_dict_sha256(trainer.stage.teacher))
            self.assertEqual(new_stream.epoch,stream.epoch)
            self.assertEqual(new_stream.cursor,stream.cursor)
            self.assertTrue(_same(new_stream.rng.bit_generator.state,stream.rng.bit_generator.state))
            self.assertTrue(all(float(s["step"])==7 for s in trainer.optimizer.state.values()))
            before=state_dict_sha256(trainer.stage.aligner)
            advance(trainer,new_stream)
            self.assertEqual(trainer.updates,1)
            self.assertEqual(before,state_dict_sha256(trainer.stage.aligner))

    def test_standard_parent_import_three_tails_share_S_moments_and_sampler(self):
        from panda_stage2_condq.tests.test_method import fixture as old_fixture
        from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
        from panda_stage2_condq.method import ConditionalQStage2
        from panda_stage2_condq.training import StudentTrainer
        from panda_stage2_condq.stream import TrainStream,StreamSpec
        from panda_stage2_condq.checkpoint import save_checkpoint as save_parent
        from ta2.evaluation import file_sha
        with tempfile.TemporaryDirectory() as directory:
            registered=synthetic_sources(directory)
            temporary=old_fixture("gt_only")
            binding=temporary.donor_binding
            binding.update({k:v for k,v in registered.inventory.manifest().items() if k in
                ("sensor","source_manifest_sha256","source_h5_sha256","catalog_sha256","sample_count","sample_ids_sha256")})
            cfg=config("tail_gt")
            spec=Stage2Spec("WV3","gt_only",cfg["seed"],0.,0.,0.,1e-6,mask_policy=MASK2_POLICY)
            stage=ConditionalQStage2(spec,temporary.aligner,donor_binding=binding)
            parent=StudentTrainer(stage,TrainingSpec(total_updates=50000,effective_batch_size=1,microbatch_size=1))
            parent.updates=40000
            parent.optimizer.param_groups[0]["lr"]=parent.training.rate(40000)
            for p in parent.stage.student.parameters():
                parent.optimizer.state[p]={"step":torch.tensor(40000.),
                    "exp_avg":torch.zeros_like(p),"exp_avg_sq":torch.zeros_like(p)}
            stream=TrainStream(registered,StreamSpec(123,456),microbatch_size=1,effective_batch_size=1)
            stream.epoch,stream.cursor=divmod(40000,stream.count)
            path=Path(directory)/"parent.pt"
            save_parent(path,parent,stream_state=stream.state_dict())
            expected=state_dict_sha256(parent.stage.student)
            for kind in ("tail_gt","tail_fc2_eps","tail_fc2_abs"):
                trainer,actual=build_tail_trainer(config(kind),registered,path,file_sha(path))
                self.assertEqual(expected,state_dict_sha256(trainer.stage.student))
                self.assertTrue(_same(actual.state_dict(),stream.state_dict()))
                self.assertTrue(all(float(trainer.optimizer.state[p]["step"])==40000
                    for p in trainer.stage.student.parameters()))
                self.assertIsNone(trainer.stage.teacher)
                trainer.assert_contract()
            changed=config("tail_gt")
            changed["seed"]+=1
            with self.assertRaisesRegex(ValueError,"seed"):
                build_tail_trainer(changed,registered,path,file_sha(path))


if __name__=="__main__":
    unittest.main()
