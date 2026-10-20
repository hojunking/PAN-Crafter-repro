"""Strict same-checkpoint selection and row ownership; entirely offline."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_mature30 import reporting as R, publication as P
from .test_reporting import config, evaluation, persist, reused_receipt


class PublicationTests(unittest.TestCase):
    def test_fallback_not_all_steps_and_final_is_additional(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=1000, h=.955, e=2.2)
            persist(root, step=2000, h=.959, e=2.1)
            persist(root, step=50000, h=.95, e=2.04)
            view = P.project_campaign(root, "s1")
            self.assertFalse(view["conflicts"], view["conflicts"])
            self.assertEqual([s["step"] for s in view["selected"]], [2000,50000])
            self.assertEqual([r[R.INDEX["Selector"]] for r in view["rows"]],
                             ["BEST_HQNR_FALLBACK","FINAL50K"])
            self.assertEqual(view["raw_count"], 3)

    def test_passed_rows_and_final_deduplicate(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=1000, h=.965, e=2.2)
            persist(root, step=2000, h=.960, e=2.045)
            persist(root, step=50000, h=.961, e=2.04)
            view = P.project_campaign(root,"s1")
            self.assertFalse(view["conflicts"],view["conflicts"])
            self.assertEqual([s["step"] for s in view["selected"]],[2000,50000])
            self.assertTrue(all(r[R.INDEX["Selector"]]=="PASS_ALL_STEPS" for r in view["rows"]))
            self.assertIn("FINAL50K",view["rows"][-1][R.INDEX["Selection@Step"]])

    def test_best_at_final_is_one_row(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root,step=1000,h=.94)
            persist(root,step=50000,h=.95)
            view=P.project_campaign(root,"s1")
            self.assertEqual(len(view["rows"]),1)
            self.assertIn("FINAL50K",view["rows"][0][R.INDEX["Selection@Step"]])

    def test_stage1_selected_source_publishes_despite_low_hqnr(self):
        with tempfile.TemporaryDirectory() as root:
            receipt,ev=reused_receipt(Path(root)/"old")
            R.emit_evaluation(root,config(phase="donor"),ev,selector_receipt=receipt)
            view=P.project_campaign(root,"s1")
            self.assertFalse(view["conflicts"],view["conflicts"])
            self.assertEqual(len(view["rows"]),1)
            row=view["rows"][0]
            self.assertEqual(row[R.INDEX["Selector"]],"BEST_VAL_RRE")
            self.assertEqual(row[R.INDEX["HQNR↑"]],.9207)
            self.assertEqual(row[R.INDEX["ERGAS↓"]],2.0401)
            self.assertTrue(row[R.INDEX["Checkpoint_SHA"]])

    def test_native_s3_stage1_is_not_mislabeled_as_reused_training(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=config("s3",phase="donor")
            receipt,ev=reused_receipt(root,server="s3",source_config=cfg)
            selected=receipt["selected"]
            artifact={"weights_saved":True,"file_sha256":selected["file_sha256"],
                "model_state_sha256":selected["source_state"]["model_state_sha256"],
                "identity":selected["source_state"],
                "checkpoint_path":"/workspace/old-container-prefix/models/source.pt"}
            R.emit_evaluation(root,cfg,ev,selector_receipt=receipt,artifact=artifact)
            view=P.project_campaign(root,"s3")
            self.assertFalse(view["conflicts"],view["conflicts"])
            row=view["rows"][0]
            self.assertEqual(row[R.INDEX["Updates"]],50000)
            self.assertEqual(row[R.INDEX["Role"]],"Teacher+Aligner")
            self.assertEqual(row[R.INDEX["Selector"]],"BEST_VAL_RRE")

    def test_docker_source_receipt_evaluation_and_checkpoint_resolve_on_host(self):
        with tempfile.TemporaryDirectory(prefix="mature30-publication-test-") as base, \
                patch.object(P, "ROOT", Path(base)):
            root=Path(base)/"work_dir"/"campaign"
            root.mkdir(parents=True)
            receipt,ev=reused_receipt(root/"old")
            def docker(path):
                return "/workspace/" + str(Path(path).relative_to(P.ROOT))
            receipt["source_root"]=docker(receipt["source_root"])
            for key in ("source_root","source_evaluation_path","checkpoint_path"):
                receipt["selected"][key]=docker(receipt["selected"][key])
            receipt["receipt_sha256"]=R.scientific_digest(
                {k:v for k,v in receipt.items() if k!="receipt_sha256"})
            cfg=config(phase="donor")
            record=R.make_record(cfg,ev,root=root,selector_receipt=receipt)
            extra=record["source"]["extra"]
            compact=extra["selector_receipt"]
            compact["source_receipt_path"]=docker(compact["source_receipt_path"])
            compact["receipt_sha256"]=R.scientific_digest(
                {k:v for k,v in compact.items() if k!="receipt_sha256"})
            record=R.build_record(cfg,scope="EVAL_SUMMARY",selector="BEST_VAL_RRE",
                step=50000,metrics=record["source"]["metrics"],identity=ev["state_before"],
                extra=extra,artifact=extra["artifact"],timestamp=record["created_at"])
            R.LocalOutbox(root).enqueue(record)
            view=P.project_campaign(root,"s1")
            self.assertFalse(view["conflicts"],view["conflicts"])
            self.assertEqual(len(view["rows"]),1)
            evidence=json.loads(view["rows"][0][R.INDEX["Original description / notes"]])
            self.assertTrue(evidence["publication"]["artifact"]["checkpoint_path"].startswith(str(root)))
            # Scientific evidence remains the original Docker path, not host-rewritten.
            self.assertTrue(evidence["extra_parameters"]["selector_receipt"]["selected"]
                            ["checkpoint_path"].startswith("/workspace/"))

    def test_bounded_resolver_rejects_foreign_and_traversal_paths(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,"OUTSIDE"):
                P._resolve_evidence_path("/etc/passwd",root)
            with self.assertRaisesRegex(ValueError,"UNSAFE"):
                P._resolve_evidence_path("/workspace/../etc/passwd",root)

    def test_stage2_docker_checkpoint_and_evaluation_paths_resolve_on_host(self):
        with tempfile.TemporaryDirectory(prefix="mature30-publication-test-") as base, \
                patch.object(P, "ROOT", Path(base)):
            root=Path(base)/"work_dir"/"campaign"
            root.mkdir(parents=True)
            cfg=config(policy="HALF"); ev=evaluation(cfg)
            directory=root/"runs"/cfg["run_id"]/"attempt_01"
            directory.mkdir(parents=True)
            R.atomic_json(directory/"config.json",cfg)
            R.atomic_json(directory/"evaluation_0001000.json",ev)
            checkpoint=directory/"models"/"step_0001000.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"stage2 checkpoint fixture")
            docker_root="/workspace/"+str(root.relative_to(P.ROOT))
            artifact={"weights_saved":True,"file_sha256":P._file_sha(checkpoint),
                "model_state_sha256":ev["state_before"]["model_state_sha256"],
                "identity":ev["state_before"],
                "checkpoint_path":docker_root+"/"+str(checkpoint.relative_to(root))}
            record=R.make_record(cfg,ev,root=docker_root,artifact=artifact)
            R.LocalOutbox(root).enqueue(record)
            view=P.project_campaign(root,"s1")
            self.assertFalse(view["conflicts"],view["conflicts"])
            self.assertEqual(view["rows"][0][R.INDEX["Checkpoint_SHA"]],artifact["file_sha256"])

    def test_partial_synthetic_or_applied_geometry_tampering_rejected(self):
        for kind in ("partial","synthetic","shift","phase","references"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                cfg=config(policy="DOWN")
                ev=evaluation(cfg,20000)
                if kind=="partial": ev["rr"]["per_scene"].pop()
                if kind=="synthetic": ev["metric_function_injection"]=True
                if kind=="shift": ev["rr"]["per_scene"][0]["pan_applied_dy_dx"]=[1.,0.]
                if kind=="phase": ev["mature30_deployment"]["geometry_key"]="s050"
                if kind=="references": ev["reference_warp"]=True
                persist(root,cfg,20000,ev=ev)
                self.assertTrue(P.project_campaign(root,"s1")["conflicts"])

    def test_schedule_allows_phase_changes_but_no_mixed_rr_fr_state(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=config(policy="DOWN")
            persist(root,cfg,10000)
            persist(root,cfg,20000)
            persist(root,cfg,50000)
            view=P.project_campaign(root,"s1")
            self.assertFalse(view["conflicts"],view["conflicts"])

    def test_row_updates_only_own_campaign_and_server(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root,step=1000,h=.95)
            old=P.project_campaign(root,"s1")
            foreign=[""]*64
            foreign[R.INDEX["Campaign_ID"]]="FOREIGN"
            snapshot={"rows":[foreign,*old["rows"]],"notes":["foreign",*old["notes"]]}
            persist(root,step=2000,h=.951)
            new=P.project_campaign(root,"s1")
            requests=P.row_requests(snapshot,new,"s1")
            self.assertEqual(len(requests),1)
            self.assertEqual(requests[0]["updateCells"]["range"]["startRowIndex"],6)
            bad=deepcopy(snapshot)
            bad["rows"][1][R.INDEX["HQNR↑"]]=.999
            with self.assertRaisesRegex(ValueError,"metrics"):
                P.row_requests(bad,new,"s1")

    def test_missing_cost_does_not_invent_or_hide_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            view=P.project(root,"s1")
            self.assertFalse(view["conflicts"],view["conflicts"])
            self.assertEqual(len(view["rows"]),1)
            self.assertTrue(P.project(root,"s1",require_cost=True)["conflicts"])


if __name__ == "__main__":
    unittest.main()
