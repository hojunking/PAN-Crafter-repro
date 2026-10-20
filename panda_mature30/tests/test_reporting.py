"""CPU/offline publication fixtures; no Google authentication or training."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from panda_band30.tests.test_reporting import config as old_config, evaluation as old_evaluation
from panda_mature30 import reporting as R


def config(server="s1", phase="stage2", policy="FULL"):
    from panda_mature30.spec import policy_manifest
    cfg = old_config("G8_S1_U23" if phase == "donor" else "G8_PAN_U23")
    cfg.update(server=server, campaign_id=R.CAMPAIGN_ID, method_revision=R.METHOD_REVISION,
               run_id="PM30_WV3_" + server.upper() + "_" + phase.upper() + "_" + policy,
               run_display="Mature " + phase + " " + policy, case=policy,
               phase=phase, policy_id=policy, engine_kind="mature_" + phase)
    cfg.setdefault("stage2_spec", {})["beta"] = 0.
    cfg["geometry"] = policy_manifest(policy)
    return cfg


def evaluation(cfg, step=1000, h=.95, e=2.1):
    ev = old_evaluation(cfg, step, h, e)
    policy = cfg["policy_id"]
    key = {"FULL":"s100", "HALF":"s050", "LOW":"s025", "CAP085":"cap085",
           "DOWN":"s050" if step <= 10000 else "s0375" if step <= 30000 else "s025"}[policy]
    scale = {"s100":1., "s050":.5, "s0375":.375, "s025":.25}.get(key)
    cap = .85 if key == "cap085" else None
    geometry = {"schema":"PANDA_MATURE30_APPLIED_GEOMETRY_V1", "geometry_key":key,
        "operator":"RADIAL_CAP" if cap else "SCALE", "scale":scale, "cap_px":cap,
        "upsampler":"interp23", "frame":"pan", "output_coordinates":"MS",
        "ms_shift_factor":0., "output_shift_factor":0., "shift_unit":"CURRENT_PAN_PIXEL_NO_X4",
        "evaluation_references":"ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20"}
    ev["mature30_deployment"] = {"schema":"PANDA_MATURE30_DEPLOYMENT_V1",
        "policy_id":policy, "completed_update":step, "geometry_key":key, "geometry":geometry}
    return ev


def persist(root, cfg=None, step=1000, h=.95, e=2.1, ev=None):
    cfg = cfg or config()
    ev = ev or evaluation(cfg, step, h, e)
    folder = Path(root) / "runs" / cfg["run_id"] / "attempt_01"
    folder.mkdir(parents=True, exist_ok=True)
    R.atomic_json(folder / "config.json", cfg)
    R.atomic_json(folder / f"evaluation_{step:07d}.json", ev)
    return R.emit_evaluation(root, cfg, ev, timestamp="2026-10-20T00:00:00+00:00")


def reused_receipt(root, server="s1", source_config=None):
    source = deepcopy(source_config) if source_config is not None else old_config("G8_S1_U23")
    source["server"] = server
    ev = old_evaluation(source, 50000, .9207, 2.0401)
    folder = Path(root) / "runs" / source["run_id"] / "attempt_01"
    folder.mkdir(parents=True, exist_ok=True)
    R.atomic_json(folder / "config.json", source)
    evpath = folder / "evaluation_0050000.json"
    R.atomic_json(evpath, ev)
    checkpoint = folder / "checkpoint_0050000.pt"
    checkpoint.write_bytes(b"test checkpoint fixture")
    import hashlib
    selected = {"source_config":source, "source_root":str(Path(root).resolve()),
        "source_state":ev["state_before"], "source_run_id":source["run_id"],
        "source_evaluation_path":str(evpath.resolve()),
        "checkpoint_path":str(checkpoint.resolve()),
        "file_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "completed_update":50000,
        "validation":{"RRE50":.1, "RRE90":.3, "Val_ERGAS":2.0,
                      "per_sample":[{"sample":i, "residual":.1} for i in range(1080)]}}
    receipt = {"schema":"PANDA_MATURE30_RRE_SELECTION_V1", "selector":"BEST_VAL_RRE",
        "campaign_id":R.CAMPAIGN_ID, "server":server, "selected":selected,
        "candidates":[selected], "source_root":str(Path(root).resolve())}
    receipt["receipt_sha256"] = R.scientific_digest(receipt)
    return receipt, ev


class ReportingTests(unittest.TestCase):
    def test_soft_enabled_cannot_be_mislabelled_as_hard_only(self):
        cfg = config()
        cfg["stage2_spec"]["beta"] = .05
        with self.assertRaisesRegex(ValueError, "Hard-only"):
            R.make_record(cfg, evaluation(cfg))

    def test_every_metric_and_raw_precision_preserved_both_servers(self):
        for server in ("s1", "s3"):
            cfg = config(server)
            ev = evaluation(cfg, h=.951234567891234)
            rec = R.make_record(cfg, ev)
            R.validate_record(rec)
            self.assertEqual(len(rec["row"]), 64)
            for name in ("RMSE↓", "CC↑", "JQM↑", "Q4/Q8↑"):
                self.assertIsInstance(rec["row"][R.INDEX[name]], (int,float))
            self.assertEqual(rec["source"]["metrics"]["HQNR"], .951234567891234)
            self.assertEqual(rec["destination"]["sheet_title"], "WV3-ablations")

    def test_mixed_state_and_wrong_lane_rejected(self):
        cfg = config(); ev = evaluation(cfg)
        ev["fr_state"]["model_state_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Mixed"):
            R.make_record(cfg, ev)
        cfg["server"] = "s4"
        with self.assertRaises(ValueError):
            R.make_record(cfg, evaluation(cfg))

    def test_reuse_preserves_source_identity_and_zero_new_training(self):
        with tempfile.TemporaryDirectory() as root:
            receipt, ev = reused_receipt(Path(root) / "old")
            cfg = config(phase="donor")
            rec = R.emit_evaluation(root, cfg, ev, selector_receipt=receipt)
            R.validate_record(rec)
            extra = rec["source"]["extra"]
            self.assertEqual(extra["identity"], ev["state_before"])
            self.assertEqual(rec["row"][R.INDEX["Updates"]], 0)
            self.assertEqual(rec["row"][R.INDEX["Lifetime_updates"]], 50000)
            self.assertEqual(rec["row"][R.INDEX["Parent_run"]], ev["state_before"]["run_id"])
            compact = extra["selector_receipt"]
            self.assertNotIn("per_sample", compact["selected"]["validation"])
            self.assertEqual(R.scientific_digest(json.loads(
                Path(compact["source_receipt_path"]).read_text())), R.scientific_digest(receipt))
            self.assertNotEqual(compact["receipt_sha256"], receipt["receipt_sha256"])

    def test_status_and_immutable_local_outbox(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config()
            first = persist(root, cfg)
            R.LocalOutbox(root).enqueue(first)
            self.assertEqual(len((Path(root)/"records.jsonl").read_text().splitlines()), 1)
            status = R.emit_status(root, cfg, "COMPLETED", step=50000)
            R.validate_record(status)
            tampered = deepcopy(first); tampered["row"][0] = "changed"
            with self.assertRaises(ValueError):
                R.LocalOutbox(root).enqueue(tampered)

    def test_cpu_publisher_import_has_no_training_or_cloud_dependency(self):
        code = """
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'h5py', 'google'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Reject())
import panda_mature30.publication, panda_mature30.transport
"""
        result = subprocess.run([sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
