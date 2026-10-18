"""CPU-only real Campaign.build integration; profiler and cloud stay disabled."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from donor24 import plan as donor_plan
from donor24.evaluation import model_state_hash
from donor30.artifacts import SCHEMA,capture_rng,_same
from donor30.runtime import build_trainer
from edgeres48.stream import learning_rates
from panda40.stage1 import Stage1Stream
from panda_method.calibration import state_dict_sha256
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from ta2.evaluation import file_sha
from panda30r import plan
from panda30r.runtime import Campaign
from panda30r.engine import save_checkpoint,load_mature_donor,build_fresh_stage2


def source_checkpoint(root,registered):
    cfg=donor_plan.resolve_case(donor_plan.default_config("s1"),"A0")
    cfg["run_id"]="SYNTHETIC_RUNTIME_SOURCE"
    cfg["baseline_resolved"].update(effective_batch_RR=1,microbatch_RR=1,accumulation_RR=1)
    trainer=build_trainer(cfg,registered.sources)
    trainer.updates=7; trainer.stage.forward_calls=7
    for group in trainer.optimizer.param_groups:
        group["lr"]=learning_rates(cfg,7)[group["role"]]
        for p in group["params"]:
            trainer.optimizer.state[p]={"step":torch.tensor(7.),"exp_avg":torch.zeros_like(p),
                                         "exp_avg_sq":torch.zeros_like(p)}
    stream=Stage1Stream(registered.sources,cfg,device="cpu")
    for _ in range(7): stream.next_update()
    joint=model_state_hash(trainer.stage)
    payload={"schema":SCHEMA,"config":cfg,"trainer":trainer.state_dict(),"stream":stream.state_dict(),
        "rng":capture_rng(),"state_hash":joint,"identity":{"model_state_sha256":joint,
            "completed_update":7,"run_id":cfg["run_id"],"dataset":"WV3"}}
    path=Path(root)/"source.pt"; torch.save(payload,path)
    return {"checkpoint_path":str(path),"checkpoint_sha256":file_sha(path),
            "source_config":cfg,"completed_update":7,"exploratory":True,"mature":False},payload


class RuntimeEngineTests(unittest.TestCase):
    def test_real_mature_build_contract_survives_profiler_and_config_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            registered=synthetic_sources(directory)
            donor,payload=source_checkpoint(directory,registered)
            with patch("panda30r.runtime.code_pin",return_value={"test":"d"*64}), \
                 patch("panda30r.runtime.evaluator_identity",return_value={"test":"explicit"}):
                campaign=Campaign("s1",output=Path(directory)/"campaign",device="cpu")
            campaign.registered=registered; campaign.sources=registered.sources
            campaign.state["donors"]["E"]=donor
            cfg=plan.resolve_case("s1","MATURE",source_update=7)
            cfg["training_spec"].update(effective_batch_size=1,microbatch_size=1)
            def disposable_profile(*args,**kwargs):
                torch.manual_seed(987654)  # cost construction must not perturb continuation RNG
                return {}
            with patch("panda30r.cost.ensure_cost",side_effect=disposable_profile):
                trainer,stream=campaign.build(cfg)
            trainer.assert_contract()
            self.assertEqual(trainer.config,cfg)
            self.assertTrue(torch.equal(torch.random.get_rng_state(),payload["rng"]["torch_cpu"]))
            self.assertEqual(stream.cursor,payload["stream"]["cursor"])
            self.assertEqual(trainer.optimizer.param_groups[0]["lr"],learning_rates(donor["source_config"],7)["T"])
            self.assertTrue(all(float(s["step"])==7 for s in trainer.optimizer.state.values()))
            self.assertEqual(trainer.config["parent_binding"]["file_sha256"],donor["checkpoint_sha256"])

    def test_completed_mature_adapter_can_start_unchanged_fresh_gt_student(self):
        from panda30r.engine import build_mature_trainer
        from panda30r.tests.test_engine import config
        with tempfile.TemporaryDirectory() as directory:
            registered=synthetic_sources(directory)
            donor,_=source_checkpoint(directory,registered)
            cfg=config("mature_teacher"); cfg.update(start_update=7,planned_horizon=49993)
            trainer,stream=build_mature_trainer(cfg,registered,donor["checkpoint_path"],
                donor["checkpoint_sha256"],evaluator_pin={"test":"explicit"})
            # Synthetic terminal counters test serialization/admission only.
            # This is not evidence that any scientific 50K training ran.
            trainer.updates=49993; trainer.stage.set_progress(49993); trainer.stage.forward_calls=49993
            trainer.optimizer.param_groups[0]["lr"]=0.
            for item in trainer.optimizer.state.values(): item["step"].fill_(50000.)
            stream.epoch,stream.cursor=divmod(50000,stream.count)
            artifact=save_checkpoint(Path(directory)/"mature.pt",trainer,stream,
                code_pin={"engine":"d"*64},evaluator_pin={"test":"explicit"})
            with patch("panda30r.engine.build_reconstructor",side_effect=AssertionError("GT-only loaded T")):
                loaded=load_mature_donor(artifact["checkpoint_path"],artifact["file_sha256"],
                    registered,include_teacher=False)
            self.assertIsNone(loaded.teacher)
            self.assertEqual(loaded.binding["donor_step"],50000)
            self.assertEqual(loaded.binding["aligner_state_sha256"],trainer.stage.donor_binding["aligner_state_sha256"])
            self.assertEqual(loaded.binding["teacher_state_sha256"],state_dict_sha256(trainer.stage.teacher))
            target=plan.resolve_case("s3","M_G")
            target["training_spec"].update(effective_batch_size=1,microbatch_size=1)
            student=build_fresh_stage2(target,loaded,None,evaluator_pin={"test":"explicit"})
            student.assert_contract()
            self.assertNotEqual(state_dict_sha256(student.stage.student),loaded.binding["teacher_state_sha256"])


if __name__=="__main__":
    unittest.main()
