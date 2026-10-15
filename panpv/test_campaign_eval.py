"""Synthetic frozen-protocol and independent native metric checks."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

from panpv.common import canonical_sha, file_sha
from panpv.campaign_eval import (FR_HALO, FRTileAccumulator, _context, _metric_mask,
                                _timed_predict, evaluate_scene, freeze_protocol, fr_tile_metrics)
from panpv.test_v3_data import write_scene
from panpv.v3_data import SceneV3


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.child = torch.nn.Dropout()

    def forward(self, sensor, pan, ms, lpan):
        # Deliberately consume RNG to test the diagnostic isolation contract.
        torch.rand(1)
        return torch.nn.functional.interpolate(ms, scale_factor=4, mode="bicubic", align_corners=False)*self.scale


class CampaignEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/"inputs").mkdir()
        self.path = write_scene(self.root/"inputs"/"scene.h5")
        self.tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.tf32
        self.temp.cleanup()

    def scene(self):
        return SceneV3(self.path, expected_sha256=file_sha(self.path))

    def test_freeze_uses_all_provider_audit_not_training_or_resplit(self):
        with self.scene() as scene:
            p = freeze_protocol(scene)
            self.assertEqual(p, freeze_protocol(scene))
            self.assertEqual(len(p["audit"]), 1)
            e = p["audit"][0]
            self.assertEqual((e["raw_index"], e["row"], e["col"], e["block_id"]), (2, 128, 1152, 2))
            self.assertTrue(e["diagnostic_fr"])
            self.assertEqual(e["rr_origin"], [136, 1164])
            self.assertEqual(e["fr_origin"], [549, 4649])
            self.assertFalse(p["full_scene"])
            self.assertFalse(p["rr_benchmark_parity"])
            self.assertEqual(e["rr_valid32_windows"], 1)
            self.assertEqual(e["fr_valid32_windows"], 49)

    def test_protocol_requires_external_byte_binding(self):
        with SceneV3(self.path) as scene, self.assertRaisesRegex(ValueError, "Externally SHA"):
            freeze_protocol(scene)

    def test_freeze_before_inference_checks_all_context_not_only_center(self):
        # Native FR centre starts PAN(512,4608); invalid pixel is in its halo.
        with h5py.File(self.path, "r+") as f:
            f["native/pan_valid"][0, 0, 390, 4500] = 0
        with self.scene() as scene, self.assertRaisesRegex(ValueError, "Invalid native context"):
            freeze_protocol(scene)

    def test_corrupt_protocol_and_changed_audit_rejected(self):
        with self.scene() as scene:
            p = freeze_protocol(scene)
            p["audit"][0]["row"] += 64
            with self.assertRaisesRegex(ValueError, "Corrupt frozen protocol"):
                evaluate_scene(scene, p, method="SC_SCENE", model=TinyModel(), device="cpu", output_dir=self.root/"out")
            p["protocol_sha256"] = canonical_sha({k:v for k,v in p.items() if k != "protocol_sha256"})
            with self.assertRaisesRegex(ValueError, "coordinates differ"):
                evaluate_scene(scene, p, method="SC_SCENE", model=TinyModel(), device="cpu", output_dir=self.root/"out")

    def test_frozen_file_not_silently_replaced(self):
        with self.scene() as scene:
            path = self.root/"protocol.json"
            freeze_protocol(scene, path)
            before = file_sha(path)
            freeze_protocol(scene, path)
            self.assertEqual(before, file_sha(path))

    def test_diagnostics_only_json_and_preserve_model_rng(self):
        model = TinyModel().train()
        model.child.eval()
        before = torch.get_rng_state().clone()
        with self.scene() as scene:
            p = freeze_protocol(scene)
            result = evaluate_scene(scene, p, method="SC_SCENE", model=model, device="cpu", output_dir=self.root/"out")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["output_files"], [])
        self.assertEqual(list((self.root/"out").glob("*.h5*")), [])
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(result["rr"]["samples"], 1)
        self.assertEqual(result["fr"]["tiles"], 1)
        self.assertIn("HQNR", result)
        self.assertIn("SCC", result)
        self.assertIn("ERGAS", result)
        self.assertIsNone(result["fr"]["metrics"]["jqm"])
        self.assertFalse(result["official_benchmark"])
        self.assertGreaterEqual(result["infer_seconds"], 0)
        self.assertTrue(np.isfinite(result["infer_seconds"]))
        self.assertEqual(result["infer_seconds"], result["inference"]["rr_seconds"]+result["inference"]["fr_seconds"])
        self.assertEqual(result["inference"]["rr_calls"], 1)
        self.assertEqual(result["inference"]["fr_calls"], 1)
        self.assertEqual(result["parameters"], 1)
        self.assertEqual(result["params_m"], .000001)
        self.assertTrue(result["hardware"].startswith("CPU:"))
        self.assertNotIn("end_to_end_seconds", result)

    def test_final_raw_outputs_masks_and_complete_cache_identity(self):
        def overrange(sample, *_):
            return np.full((8, *sample["pan"].shape[-2:]), 3000, np.float32)
        with self.scene() as scene:
            p = freeze_protocol(scene)
            with mock.patch("panpv.campaign_eval._predict", side_effect=overrange), \
                 mock.patch("panpv.brovey.config_metadata", return_value={"synthetic": True}):
                args = dict(method="BROVEY8", final=True, output_dir=self.root/"out")
                result = evaluate_scene(scene, p, **args)
                cached = evaluate_scene(scene, p, **args)
            self.assertEqual(result, cached)
            self.assertEqual(len(result["output_files"]), 1)
            with h5py.File(self.root/"out"/"predictions.h5", "r") as f:
                self.assertTrue(f.attrs["complete"])
                self.assertFalse(f.attrs["clipped"])
                self.assertEqual(f["fr/prediction"].shape, (1, 8, 256, 256))
                self.assertEqual(f["rr/prediction"].dtype, np.dtype("float32"))
                self.assertEqual(float(f["fr/prediction"][0].max()), 3000)
                self.assertEqual(int(f["rr/metric_valid32"][:].sum()), 1024)
                self.assertEqual(int(f["fr/metric_valid32"][:].sum()), 49*1024)
                self.assertEqual(int(f["fr/valid"][:].sum()), 256*256)
            self.assertEqual(result["rr"]["prediction_out_of_range_elements"], 8*64*64)
            self.assertEqual(result["parameters"], 0)
            self.assertEqual(result["params_m"], 0)
            self.assertTrue(result["hardware"].startswith("CPU:"))
            with mock.patch("panpv.brovey.config_metadata", return_value={"synthetic": False}), \
                 self.assertRaisesRegex(ValueError, "mismatched/corrupt"):
                evaluate_scene(scene, p, **args)

    def test_changed_weights_cannot_reuse_diagnostic(self):
        model = TinyModel()
        with self.scene() as scene:
            p = freeze_protocol(scene)
            args = dict(method="FT_WV3_SCENE", model=model, device="cpu", output_dir=self.root/"out")
            evaluate_scene(scene, p, **args)
            with torch.no_grad():
                model.scale.add_(1)
            with self.assertRaisesRegex(ValueError, "mismatched/corrupt"):
                evaluate_scene(scene, p, **args)

    def test_partial_preserved_not_reused_as_complete(self):
        out = self.root/"out"
        out.mkdir()
        partial = out/"predictions.h5.partial"
        with h5py.File(partial, "w") as f:
            f.attrs["interrupted_test"] = True
        with self.scene() as scene:
            evaluate_scene(scene, freeze_protocol(scene), method="SC_SCENE", model=TinyModel(),
                           device="cpu", output_dir=out)
        self.assertTrue((out/"interrupted_000"/"predictions.h5.partial").exists())
        self.assertFalse((out/"predictions.h5").exists())

    def test_output_cannot_pollute_input_directory(self):
        with self.scene() as scene, self.assertRaisesRegex(ValueError, "source-data"):
            evaluate_scene(scene, freeze_protocol(scene), method="SC_SCENE", model=TinyModel(),
                           device="cpu", output_dir=self.path.parent/"outputs")

    def test_metric_mask_anchors_to_source_not_patch(self):
        self.assertEqual(int(_metric_mask(64, (8, 12)).sum()), 32*32)
        self.assertEqual(int(_metric_mask(64, (0, 0)).sum()), 64*64)

    def test_fr_matches_direct_primitive_reference_and_has_real_context(self):
        from deploy.pandep_shared.project.src.pan_shared.metrics.eval_fr import mtf_filter, imresize_matlab, _blockproc_uqi
        from deploy.pandep_shared.project.src.pan_shared.metrics.interp23 import interp23tap
        from deploy.pandep_shared.project.src.pan_shared.metrics.q2n import q2n
        rng = np.random.default_rng(53)
        sample = dict(pan=rng.uniform(100, 1500, (1, 512, 512)).astype("float32"),
                      ms=rng.uniform(100, 1500, (8, 128, 128)).astype("float32"),
                      masks={"pan":np.ones((1, 512, 512), bool)})
        pred = rng.uniform(-300, 2600, (8, 512, 512)).astype("float32")
        measured = fr_tile_metrics(pred, sample, (0, 0))
        fused = np.clip(pred.astype("float64"), 0, 2047).transpose(1, 2, 0)
        p = sample["pan"][0].astype("float64")
        lms = interp23tap(sample["ms"].transpose(1, 2, 0).astype("float64"), 4)
        low = interp23tap(imresize_matlab(p, .25)[..., None], 4)[..., 0]
        s = np.s_[128:384, 128:384]
        dl = 1-float(q2n(lms[s], mtf_filter(fused, "wv2", 4)[s], 32, 32)[0])
        ds = np.abs([_blockproc_uqi(fused[s][..., b], p[s], 32)-_blockproc_uqi(lms[s][..., b], low[s], 32)
                     for b in range(8)]).mean()
        self.assertAlmostEqual(measured["metrics"]["hqnr"], (1-dl)*(1-ds), places=12)
        self.assertEqual(measured["valid_windows"], 64)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            fr_tile_metrics(pred, sample, (0, 0), halo=80)

    def test_fr_duplicate_and_invalid_context_rejected(self):
        with self.scene() as scene:
            p = freeze_protocol(scene)
            e = p["audit"][0]
            sample = _context(scene, e, "native")
        pred = np.full((8, 512, 512), 1024, np.float32)
        acc = FRTileAccumulator()
        acc.add(pred, sample, e["fr_origin"], sample_id=e["raw_index"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            acc.add(pred, sample, e["fr_origin"], sample_id=e["raw_index"])
        sample["masks"]["pan"][0, 1, 1] = False
        with self.assertRaisesRegex(ValueError, "fully valid"):
            fr_tile_metrics(pred, sample, e["fr_origin"])

    def test_native_metric128_context_matches_larger160_context(self):
        rng = np.random.default_rng(37)
        full = dict(pan=rng.uniform(300, 1500, (1, 576, 576)).astype("float32"),
                    ms=rng.uniform(300, 1500, (8, 144, 144)).astype("float32"),
                    masks={"pan": np.ones((1, 576, 576), bool)})
        pred = rng.uniform(300, 1500, (8, 576, 576)).astype("float32")
        small = dict(pan=full["pan"][:, 32:-32, 32:-32], ms=full["ms"][:, 8:-8, 8:-8],
                     masks={"pan": np.ones((1, 512, 512), bool)})
        a = fr_tile_metrics(pred, full, (5, 9), halo=160)
        b = fr_tile_metrics(pred[:, 32:-32, 32:-32], small, (5, 9), halo=128)
        self.assertEqual(a["valid_windows"], b["valid_windows"])
        for key in a["metrics"]:
            self.assertAlmostEqual(a["metrics"][key], b["metrics"][key], places=10, msg=key)

    def test_predict_timing_synchronizes_cuda_and_excludes_prior_work(self):
        calls = []
        def sync(*_):
            calls.append("sync")
        def clock():
            calls.append("clock")
            return 2.0 if calls.count("clock") == 1 else 2.75
        def prediction(*_):
            calls.append("predict")
            return np.zeros((8, 64, 64), np.float32)
        with mock.patch("torch.cuda.synchronize", side_effect=sync), \
             mock.patch("panpv.campaign_eval.time.perf_counter", side_effect=clock), \
             mock.patch("panpv.campaign_eval._predict", side_effect=prediction):
            result, seconds = _timed_predict({}, TinyModel(), "cuda:0")
        self.assertEqual(calls, ["sync", "clock", "predict", "sync", "clock"])
        self.assertEqual(seconds, .75)
        self.assertEqual(result.shape, (8, 64, 64))

    def test_predict_timing_rejects_nan_and_negative_measurements(self):
        for ending in (float("nan"), .5):
            with mock.patch("panpv.campaign_eval._predict", return_value=np.zeros((8, 64, 64))), \
                 mock.patch("panpv.campaign_eval.time.perf_counter", side_effect=[1.0, ending]), \
                 self.assertRaisesRegex(RuntimeError, "Invalid measured"):
                _timed_predict({}, None, "cpu")


if __name__ == "__main__":
    unittest.main()
