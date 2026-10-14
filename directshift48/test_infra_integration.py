"""Exercise actual runtime event production through immutable Sheets projection.

The expensive image evaluator is stubbed with an exact RR20/FR20 numeric result;
Campaign.evaluate/emit, reporting, durable evidence and projection are real.
No experiment, GPU workload, network write or prior training state is changed.
"""
from copy import deepcopy
import json
import hashlib
import subprocess
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import plan, publication, reporting, runtime
from .test_reporting import arguments
from .test_publication import persist_legacy
from sheet_publish_v2.test_selection import record as legacy_record


def harness(root, server="s1"):
    args = arguments(server=server, h=.95 if server == "s1" else .96,
                     e=2.1 if server == "s1" else .53)
    cfg = args["extra"]["configuration"]
    campaign = runtime.Campaign.__new__(runtime.Campaign)
    campaign.root = Path(root)
    campaign.server, campaign.dataset = server, cfg["dataset"]
    campaign.baseline, campaign.device = cfg["baseline_resolved"], "cpu"
    campaign.evaluator_pin = {"offline_integration_fixture": "numeric evaluator stub"}
    protocol = runtime.digest(campaign.evaluator_pin)
    campaign.provenance = {**args["provenance"], "evaluator_revision": protocol}
    campaign.sources = SimpleNamespace(sensor=cfg["dataset"])
    campaign.sequence = 0
    campaign.outbox = reporting.LocalOutbox(campaign.root)
    campaign.state = {"started_unix": time.time(), "timing": {}}
    campaign.verify_code = lambda: None
    directory = campaign.root / "runs" / cfg["run_id"] / "attempt_01"
    directory.mkdir(parents=True)
    reporting.atomic_json(directory / "config.json", cfg)
    ident = {**args["extra"]["identity"], "evaluator_revision": protocol}
    result = {"record_type": "EVAL_SUMMARY", "full_method_eligible": True,
              "measurement_domain": "OFFICIAL_FULL20_NATIVE", "metric_function_injection": False,
              "test_aware": True, "eval_protocol_id": protocol, "elapsed_seconds": 12.5,
              **{key: ident for key in ("state_before", "rr_state", "fr_state", "state_after")}}
    for split in ("rr", "fr"):
        metrics = {key: value for key, value in args["metrics"].items()
                   if (key in ("HQNR", "D_s", "D_lambda", "JQM")) == (split == "fr")}
        result[split] = {"metrics": metrics, "per_scene": [
            {"scene_id": f"{cfg['dataset']}:{split}:{i}", "sample_id": f"{cfg['dataset']}:{split}:{i}",
             "evaluation_image_id": f"{cfg['dataset']}:{split}:{i}", "metrics": metrics,
             "correction_dy_dx": [0., 0.], "shift_unit": "CURRENT_PAN_PIXEL"} for i in range(20)]}
    result["qualification"] = plan.qualify_evaluation(result, server=server,
        expected_rr_ids=[f"{cfg['dataset']}:rr:{i}" for i in range(20)],
        expected_fr_ids=[f"{cfg['dataset']}:fr:{i}" for i in range(20)])
    trainer = SimpleNamespace(updates=1000)
    return campaign, cfg, directory, trainer, result


class RuntimePublicationIntegration(unittest.TestCase):
    def test_real_evaluate_emit_local_records_to_qualified_view_s1_s3(self):
        for server in ("s1", "s3"):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as directory:
                campaign, cfg, run, trainer, result = harness(Path(directory) / cfg_name(server), server)
                with patch.object(runtime, "official_evaluation", return_value=(result, {"rr": [], "fr": []})):
                    observation = campaign.evaluate(trainer, cfg, None, run)
                self.assertFalse(observation["joint_pass"])
                self.assertEqual(len(list(campaign.outbox.outbox.glob("DS3-*.json"))), 41)
                projected = publication.project_campaign(campaign.root, server)
                self.assertFalse(projected["conflicts"], projected["conflicts"])
                self.assertEqual(projected["raw_count"], 41)
                self.assertEqual(projected["summary_count"], 1)
                self.assertEqual(len(projected["rows"]), 1)
                self.assertEqual(projected["rows"][0][publication.INDEX["Selector"]], "BEST_HQNR_FALLBACK")
                self.assertEqual(projected["selected"][0]["hqnr"], result["fr"]["metrics"]["HQNR"])
                self.assertEqual(projected["selected"][0]["ergas"], result["rr"]["metrics"]["ERGAS"])
                self.assertEqual(projected["rows"][0][publication.INDEX["Checkpoint_SHA"]], "")

    def test_real_producer_combines_old_campaign_without_metrics_or_identity_relabel(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign, cfg, run, trainer, result = harness(Path(directory) / "WV3")
            with patch.object(runtime, "official_evaluation", return_value=(result, {})):
                campaign.evaluate(trainer, cfg, None, run)
            old = Path(directory) / "old" / "WV3"
            legacy = legacy_record(h=.946, e=2.33)
            persist_legacy(old, legacy)
            combined = publication.combined_projection(campaign.root, "s1", [old])
            self.assertFalse(combined["conflicts"])
            self.assertEqual(len(combined["rows"]), 2)
            self.assertEqual(combined["selected"][0]["source_result_id"], legacy["record_id"])
            self.assertEqual(combined["rows"][0][publication.INDEX["HQNR↑"]], .946)
            self.assertEqual(combined["rows"][1][publication.INDEX["HQNR↑"]], .95)
            self.assertNotEqual(combined["rows"][0][publication.INDEX["Campaign_ID"]],
                                combined["rows"][1][publication.INDEX["Campaign_ID"]])

    def test_container_absolute_source_evaluation_resolves_by_owned_run_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign, cfg, run, trainer, result = harness(Path(directory) / "WV3")
            original_emit = campaign.emit
            def emit_in_container(*args, **kwargs):
                extra = deepcopy(kwargs.get("extra") or {})
                if "source_evaluation" in extra:
                    relative = Path(extra["source_evaluation"]).relative_to(campaign.root)
                    extra["source_evaluation"] = str(Path("/workspace/work_dir/directshift48") /
                        plan.CAMPAIGN_ID / "s1/WV3" / relative)
                kwargs["extra"] = extra
                return original_emit(*args, **kwargs)
            with patch.object(campaign, "emit", side_effect=emit_in_container), \
                    patch.object(runtime, "official_evaluation", return_value=(result, {})):
                campaign.evaluate(trainer, cfg, None, run)
            value = publication.project_campaign(campaign.root, "s1")
            self.assertFalse(value["conflicts"], value["conflicts"])
            self.assertEqual(len(value["rows"]), 1)
            note = json.loads(value["rows"][0][publication.INDEX["Original description / notes"]])
            self.assertTrue(note["publication"]["evaluation_evidence"]["path"].startswith(str(campaign.root)))
            self.assertTrue(note["extra_parameters"]["source_evaluation"].startswith("/workspace/"))

    def test_disposable_preflight_never_enters_official_view(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign, cfg, run, trainer, result = harness(Path(directory) / "WV3")
            cfg = {**cfg, "disposable_preflight": True}
            reporting.atomic_json(run / "config.json", cfg)
            with patch.object(runtime, "official_evaluation", return_value=(result, {})):
                campaign.evaluate(trainer, cfg, None, run, eligible=False)
            projected = publication.project_campaign(campaign.root, "s1")
            self.assertFalse(projected["conflicts"])
            self.assertEqual(projected["rows"], [])
            self.assertEqual(projected["summary_count"], 0)

    def test_runtime_qualified_artifact_uses_real_file_sha_not_model_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign, cfg, run, trainer, result = harness(Path(directory) / "WV3")
            result["fr"]["metrics"]["HQNR"] = .961
            result["rr"]["metrics"]["ERGAS"] = 2.03
            result["qualification"] = plan.qualify_evaluation(result, server="s1",
                expected_rr_ids=[f"WV3:rr:{i}" for i in range(20)],
                expected_fr_ids=[f"WV3:fr:{i}" for i in range(20)])
            self.assertTrue(result["qualification"]["joint_pass"])
            reporting.atomic_json(campaign.root / "method_source/manifest.json", {})
            campaign.state["cost_runs"] = [cfg["run_id"]]
            checkpoint = run / "qual" / "teacher_aligner.pt"
            checkpoint.parent.mkdir()
            # This file is an explicit artifact-transport fixture, not a model.
            checkpoint.write_bytes(b"offline artifact fixture")
            sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            artifact = {"weights_saved": True, "file_sha256": sha, "checkpoint_file_sha256": sha,
                        "model_state_sha256": result["state_after"]["model_state_sha256"],
                        "identity": result["state_after"], "checkpoint_path": str(checkpoint),
                        "status": "QUAL_PENDING", "qualitative_files": {}}
            with patch.object(runtime, "official_evaluation", return_value=(result, {})), \
                    patch.object(runtime, "retain_qualified", return_value=artifact):
                campaign.evaluate(trainer, cfg, SimpleNamespace(state_dict=lambda: {}), run)
            projected = publication.project_campaign(campaign.root, "s1")
            self.assertFalse(projected["conflicts"], projected["conflicts"])
            row = projected["rows"][0]
            self.assertEqual(row[publication.INDEX["Selector"]], "PASS_ALL_STEPS")
            self.assertEqual(row[publication.INDEX["Checkpoint_SHA"]], sha)
            self.assertNotEqual(sha, result["state_after"]["model_state_sha256"])
            note = json.loads(row[publication.INDEX["Original description / notes"]])
            self.assertTrue(note["publication"]["artifact"]["weights_saved"])
            self.assertEqual(note["publication"]["artifact"]["qualitative_files"], {})

    def test_cli_publish_passes_history_and_has_no_implicit_apply(self):
        from .__main__ import main
        with tempfile.TemporaryDirectory() as directory, patch.object(publication, "publish", return_value=0) as call:
            self.assertEqual(main(["publish", "--server", "s3", "--output", directory,
                                   "--history-root", "/history/lane"]), 0)
            positional, keywords = call.call_args
            self.assertEqual(positional[1], "s3")
            self.assertEqual(positional[0], Path(directory) / plan.CAMPAIGN_ID / "s3/GF2")
            self.assertFalse(keywords["apply"])
            self.assertFalse(keywords["watch"])
            self.assertEqual(keywords["history_roots"], ["/history/lane"])

    def test_cpu_cli_guard_never_queries_gpu(self):
        from . import __main__ as cli
        with patch.object(cli.subprocess, "check_output", side_effect=AssertionError("No GPU query")):
            cli.require_idle_gpu("cpu")

    def test_busy_cuda_direct_cli_rejects_before_campaign_constructor(self):
        from . import __main__ as cli
        with patch.object(cli.subprocess, "check_output", return_value="1234, active-training, 4000 MiB\n"), \
                patch.object(runtime, "Campaign", side_effect=AssertionError("No campaign launch")):
            with self.assertRaisesRegex(RuntimeError, "Active GPU"):
                cli.main(["run", "--server", "s1"])

    def test_cuda_query_failure_is_fail_closed_and_empty_result_is_idle(self):
        from . import __main__ as cli
        for error in (FileNotFoundError("nvidia-smi"), subprocess.CalledProcessError(1, "nvidia-smi"),
                      subprocess.TimeoutExpired("nvidia-smi", 20)):
            with self.subTest(error=type(error).__name__), \
                    patch.object(cli.subprocess, "check_output", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
                    cli.require_idle_gpu("cuda:0")
        with patch.object(cli.subprocess, "check_output", return_value="\n") as query:
            cli.require_idle_gpu("cuda")
            self.assertEqual(query.call_args.kwargs["timeout"], 20)


def cfg_name(server):
    return plan.LANES[server]["dataset"]


if __name__ == "__main__":
    unittest.main()
