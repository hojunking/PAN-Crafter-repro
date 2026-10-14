"""Offline immutable reporting tests. No training, GPU or Google access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import reporting as R
from . import plan


def config(server="s1"):
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "config/directshift48" / (server + ".baseline.json")).read_text())
    return plan.resolve_case(server, baseline=baseline)


def arguments(server="s1", step=1000, h=.95, e=2.1, **changes):
    cfg = config(server)
    state = plan.canonical_sha256([cfg["run_id"], step])
    ident = {**plan.METHOD_CONTRACT, "run_id": cfg["run_id"], "attempt": 1,
             "seed": cfg["seed"], "completed_update": step,
             "model_state_sha256": state, "alignment_protocol_digest": cfg["alignment_protocol_digest"],
             "inference_policy_digest": "c" * 64, "evaluator_revision": "d" * 64}
    metrics = {"HQNR": h, "ERGAS": e, "D_s": .02, "D_lambda": .02, "JQM": .98,
               "SCC": .99, "SAM": 2., "PSNR": 40., "SSIM": .98,
               "Q8" if server == "s1" else "Q4": .95, "RMSE": 20., "CC": .99}
    extra = {"configuration": cfg, "identity": ident, "same_state_verified": True,
             "metrics_complete": True, "metrics_finite": True, "full_method_eligible": True,
             "HQNR": h, "ERGAS": e, "joint_pass": False, "weights_saved": False,
             "source_evaluation": "/workspace/lane/runs/%s/attempt_01/evaluation_%07d.json" % (cfg["run_id"], step),
             "scene_completeness": {s: {"expected": 20, "completed": 20, "missing": [],
                                       "duplicates": [], "unexpected": []} for s in ("rr", "fr")}}
    result = dict(server=server, dataset=cfg["dataset"], run_id=cfg["run_id"], case_id=cfg["case"],
        selector="OBSERVED_%07d" % step, eval_scope="EVAL_SUMMARY", eval_protocol_id="d" * 64,
        checkpoint_sha=None, model_state_sha=state, metrics=metrics,
        fields={"Attempt": 1, "Selected_step": step, "Repeat": 1, "Seed": cfg["seed"],
                "Status": "OBSERVED", "U_peak_lr": cfg["lr_U"], "A_peak_lr": cfg["lr_A"]},
        extra=extra, provenance={"config_sha256": plan.canonical_sha256(cfg["baseline_resolved"]),
                "code_commit": "c" * 40, "dataset_manifest_sha256": "f" * 64,
                "environment_sha256": "e" * 64,
                "evaluator_revision": "d" * 64}, timestamp="2026-10-14T01:00:00+00:00")
    result.update(changes)
    return result


def record(**changes):
    return R.build_record(**arguments(**changes))


class ReportingTests(unittest.TestCase):
    def test_admitted_environment_provenance_survives_status_and_evaluation(self):
        for status_only in (False, True):
            args = arguments()
            if status_only:
                args.update(metrics={}, model_state_sha=None, eval_scope="STATUS",
                    selector="PREFLIGHT_STARTED", extra={"configuration":config(), "disposable_weights":True})
            value = R.build_record(**args)
            R.validate_record(value)
            note = json.loads(value["row"][R.INDEX["Original description / notes"]])
            self.assertEqual(note["environment_sha256"], "e"*64)
            self.assertEqual(value["source"]["provenance"]["environment_sha256"], "e"*64)

    def test_invalid_environment_provenance_rejected(self):
        args = arguments()
        args["provenance"]["environment_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(R.ReportingError, 'environment_sha256'):
            R.build_record(**args)

    def test_fixed64_direct_namespace_full_precision(self):
        item = record(h=.959123456789123)
        R.validate_record(item)
        self.assertEqual(len(item["row"]), 64)
        self.assertEqual(item["row"][R.INDEX["HQNR↑"]], .959123456789123)
        self.assertEqual(item["row"][R.INDEX["Checkpoint_SHA"]], "")
        self.assertTrue(item["record_id"].startswith("DS3-"))
        self.assertEqual(item["identity"]["namespace"], R.RECORD_NAMESPACE)

    def test_only_s1_s3(self):
        R.validate_record(record(server="s3", h=.97, e=.5))
        for server, sensor in (("s2", "QB"), ("s1", "GF2"), ("s3", "WV3")):
            with self.assertRaises(R.ReportingError):
                R.destination(server, sensor)

    def test_reject_legacy_fields_identity_and_config(self):
        for field in ("b_s", "b_global", "b_i", "edge_match", "edge_protocol_digest"):
            args = arguments()
            args["extra"][field] = "obsolete"
            with self.subTest(field=field), self.assertRaises(R.ReportingError):
                R.build_record(**args)
        args = arguments()
        args["extra"]["identity"]["edge_protocol_digest"] = "a" * 64
        with self.assertRaises(R.ReportingError):
            R.build_record(**args)
        args = arguments()
        args["extra"]["configuration"]["epsilon_radius_rr"] = 2.
        with self.assertRaises(ValueError):
            R.build_record(**args)

    def test_no_fake_saved_file(self):
        with self.assertRaises(R.ReportingError):
            record(checkpoint_sha="f" * 64)
        args = arguments()
        args["extra"]["weights_saved"] = True
        with self.assertRaises(R.ReportingError):
            R.build_record(**args)

    def test_mixed_state_forbidden(self):
        args = arguments()
        args["extra"]["same_state_verified"] = False
        with self.assertRaises(R.ReportingError):
            R.build_record(**args)

    def test_tampered_source_is_rejected(self):
        item = record()
        item["source"]["metrics"]["HQNR"] = .999
        with self.assertRaises(R.ReportingError):
            R.validate_record(item)

    def test_numeric_payload_split_without_loss(self):
        args = arguments()
        args.update(metrics={}, eval_scope="DIAGNOSTIC_LEAN")
        values = {str(i): "x" * 300 for i in range(80)}
        args["extra"]["numeric_observations"] = values
        rows = R.build_records_split(**args)
        self.assertGreater(len(rows), 1)
        rebuilt = {}
        for item in rows:
            R.validate_record(item)
            self.assertLessEqual(len(item["row"][R.INDEX["Original description / notes"]]), 20000)
            rebuilt.update(item["source"]["extra"]["numeric_observations"])
        self.assertEqual(rebuilt, values)

    def test_outbox_duplicate_and_recovery_no_cloud_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            outbox = R.LocalOutbox(directory)
            item = record()
            outbox.enqueue(item)
            outbox.enqueue(item)
            self.assertEqual(len((Path(directory) / "records.jsonl").read_text().splitlines()), 1)
            (outbox.outbox / (item["record_id"] + ".json")).unlink()
            self.assertEqual(outbox.recover(), 1)
            self.assertEqual(outbox.pending(), [item])
            self.assertFalse(hasattr(outbox, "publish_once"))
            self.assertFalse(hasattr(R, "GSpreadTransport"))

    def test_id_conflict_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            outbox = R.LocalOutbox(directory)
            first = record()
            outbox.enqueue(first)
            with self.assertRaises(R.RecordConflict):
                outbox.enqueue(record(h=.951))
            self.assertEqual(json.loads(next(outbox.outbox.glob("DS3-*.json")).read_text()), first)

    def test_bad_journal_escape_and_partial_ledger_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            outbox = R.LocalOutbox(directory)
            with self.assertRaises(R.ReportingError):
                outbox.enqueue(record(), journal="../escape.jsonl")
            (Path(directory) / "records.jsonl").write_bytes(b'{"broken":')
            with self.assertRaises(R.ReportingError):
                outbox.recover()


if __name__ == "__main__":
    unittest.main()
