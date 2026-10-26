"""RGB/Lanczos calibration must use its actual support and reject old donors."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from panda_sensor5.base_reconstruction import BandReconstructionSpec
from panda_sensor5.calibration import base_protocol, validate_base_binding
from panda_sensor5.calibration_core import (
    CalibrationBuilder, CalibrationStats, MASK2_ERROR_REDUCTION,
    WARP_CONTRACT, sample_ids_sha256, validate_binding,
)
from panda_sensor5.contract import CAMPAIGN_ID
from panda_sensor5.warp import warp_support_mask
from panda_stage2_condq.spec import MASK2_POLICY


def binding(sensor="QB"):
    return {
        "sensor": sensor, "source_split": "RR_train_only", "sample_count": 1,
        "sample_ids_sha256": sample_ids_sha256([sensor + ":train:0"]),
        "source_manifest_sha256": "a" * 64,
        "aligner_state_sha256": "b" * 64, "teacher_state_sha256": "c" * 64,
        "donor_checkpoint_id": "d" * 64,
        "reconstruction_spec": {"bands": 4, "width": 112, "depth": [1, 2, 3],
                                "input_layout": "P0", "base_reference": "BICUBIC_MS"},
        "scharr": {"kernel": "absolute_xy_/32", "padding": "reflect1",
                   "rgb_indices": [2, 1, 0], "roi_margin": 8},
        "warp": WARP_CONTRACT,
        "band30_teacher": {"campaign_id": CAMPAIGN_ID, "upsampler": "interp23",
                           "aligner_outputs": 2, "alignment_target": "SCHARR_OF_RGB_MEAN_GT",
                           "pan_channels": 1},
        "band30_reconstruction_spec": BandReconstructionSpec(
            4, 112, (1, 2, 3), "P0", "BICUBIC_MS", 1).manifest(),
    }


def population():
    mask = warp_support_mask(64, 64, torch.tensor([[1.25, -.5]], dtype=torch.float32))
    e = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64) / 1000
    # Invalid boundary values must not contaminate tau_e.
    e = torch.where(mask, e, torch.full_like(e, 1.e6))
    g = torch.arange(64, dtype=torch.float32)[None, None, :, None].expand(1, 1, 64, 64) / 64
    q = (torch.arange(64, dtype=torch.float32)[None, None, None, :] % 7).expand(1, 1, 64, 64) / 7
    return e, g, q, mask


def build_stats(sensor="QB"):
    b = binding(sensor)
    collector = CalibrationBuilder([sensor + ":train:0"], b, k=5,
        memory_budget_bytes=8 * 1024**2, include_condq=True, mask_policy=MASK2_POLICY)
    e, g, q, mask = population()
    collector.add([sensor + ":train:0"], e, g, q, sensor=sensor, loss_mask=mask)
    return collector.finalize(), b, (e, g, q, mask)


class CalibrationContractTests(unittest.TestCase):
    def test_current_sensor_binding_and_truthful_operator_contract(self):
        for sensor in ("QB", "GF2"):
            b = binding(sensor)
            self.assertEqual(validate_binding(b), json.loads(json.dumps(b)))
            bound = validate_base_binding(b)
            self.assertEqual(bound["mature30_base"], base_protocol())
            self.assertEqual(base_protocol()["teacher_warp"], "STAGE1_LANCZOS4_REFLECT_FULL_U")
            self.assertEqual(base_protocol()["actual_support_operator"],
                             "TEACHER_STUDENT_LANCZOS4_INTERSECTION")

    def test_foreign_or_mean4_or_bicubic_donor_rejected(self):
        variants = []
        b = binding(); b["warp"] = "bicubic_border_FP32_align_corners_false_current_pan_px"
        variants.append(b)
        b = binding(); b["band30_teacher"]["alignment_target"] = "SCHARR_OF_MEAN4_GT"
        variants.append(b)
        b = binding(); b["band30_teacher"]["campaign_id"] = "PANDA_S13_SENSOR5_STAGE12_20261026_v1"
        variants.append(b)
        for b in variants:
            with self.subTest(binding=b), self.assertRaises(ValueError):
                validate_base_binding(b)

    def test_actual_lanczos_mask_controls_tau_and_exact_ecdf_roundtrip(self):
        stats, b, (e, g, q, mask) = build_stats()
        count = int(mask.sum())
        self.assertLess(count, 64 * 64)
        self.assertTrue(bool(mask[..., 7:-7, 7:-7].all()))
        self.assertEqual(stats.metadata["n_e"], count)
        self.assertEqual(stats.metadata["valid_e_counts_by_sample"], [["QB:train:0", count]])
        self.assertEqual(stats.metadata["n_q"], 48 * 48)
        self.assertEqual(stats.metadata["teacher_error_reduction"], MASK2_ERROR_REDUCTION)
        self.assertEqual(stats.tau_e, float(np.median(e[mask].numpy().astype(np.float64))))
        ranks = stats.lookup_exact(g[..., 8:-8, 8:-8], q[..., 8:-8, 8:-8])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.npz"
            receipt = stats.save(path)
            loaded = CalibrationStats.load(path, expected_binding=b,
                expected_file_sha256=receipt["file_sha256"], expected_mask_policy=MASK2_POLICY,
                memory_budget_bytes=8 * 1024**2)
            loaded.assert_integrity(full_hash=True)
            self.assertEqual(loaded.artifact_sha256, stats.artifact_sha256)
            self.assertTrue(torch.equal(ranks,
                loaded.lookup_exact(g[..., 8:-8, 8:-8], q[..., 8:-8, 8:-8])))

    def test_old_bicubic_artifact_cannot_be_loaded_as_lanczos(self):
        from panda_stage2_condq.calibration import CalibrationBuilder as OldBuilder
        old = deepcopy(binding())
        old["warp"] = "bicubic_border_FP32_align_corners_false_current_pan_px"
        old["band30_teacher"].update(campaign_id="PANDA_S13_SENSOR5_STAGE12_20261026_v1",
                                      alignment_target="SCHARR_OF_MEAN4_GT")
        collector = OldBuilder(["QB:train:0"], old, k=5,
            memory_budget_bytes=8 * 1024**2, include_condq=True, mask_policy=MASK2_POLICY)
        e, g, q, mask = population()
        collector.add(["QB:train:0"], e, g, q, sensor="QB", loss_mask=mask)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.npz"
            collector.finalize().save(path)
            with self.assertRaisesRegex(ValueError, "operator binding mismatch"):
                CalibrationStats.load(path, expected_binding=binding(),
                                      expected_mask_policy=MASK2_POLICY)


if __name__ == "__main__":
    unittest.main()
