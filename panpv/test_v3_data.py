"""Synthetic CPU checks; no private samples or supplied code are executed."""
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest

import h5py
import numpy as np

from panpv.common import canonical_sha, file_sha
from panpv.v3_data import BANDS, LP_PROFILE, RR_PROFILE, SCHEMA, RRPatchV3, SceneV3, _array_sha


def write_scene(path):
    h, w = 512, 1536
    blocks = np.array([[0, 512*i, 512, 512, i] for i in range(3)], dtype="int32")
    index = np.array([[128, 128+512*i, i, i] for i in range(3)], dtype="int32")
    contract = dict(scene_id="SYNTHETIC", scene_shape_hw=[h, w], buffer=128, patch_size=64)
    provenance = dict(contract, block_assignment_sha256=canonical_sha(dict(contract=contract, blocks=blocks.tolist())),
                      patch_index_sha256=_array_sha(index))
    with h5py.File(path, "w") as f:
        f.attrs.update(schema=SCHEMA, complete=True, scene_id="SYNTHETIC", ratio=4,
                       axis_order="NCHW", band_order=json.dumps(BANDS),
                       value_encoding="delivery_DN_unscaled", nominal_max=2047,
                       source_ms_col=12, source_ms_row=8)
        metadata = {
            "normalization": dict(nominal_max=2047, already_normalized=False,
                                  model_expression="2*x/2047-1", clipping="none"),
            "geometry": dict(accepted=True, crop_ms_xywh=[12, 8, w, h], integer_pan_offset_xy=[-7, 5]),
            "rr_profile": dict(id=RR_PROFILE, ratio=4),
            "lp_profile": dict(id=LP_PROFILE, ratio=4),
            "split_provenance": provenance,
            "build_receipt": dict(split=dict(counts=dict(train=1, validation=1, audit=1),
                                             patch_index_sha256=_array_sha(index), provenance=provenance)),
        }
        for key, value in metadata.items():
            f.create_dataset("metadata/"+key+"_json", data=json.dumps(value))
        arrays = {"native/ms": (8, h, w, "uint16"), "native/pan": (1, h*4, w*4, "uint16"),
                  "native/lpan": (1, h, w, "float32"), "rr/pan": (1, h, w, "float32"),
                  "rr/ms": (8, h//4, w//4, "float32"), "rr/lpan": (1, h//4, w//4, "float32")}
        for name, (bands, hh, ww, dtype) in arrays.items():
            f.create_dataset(name, (1, bands, hh, ww), dtype=dtype, fillvalue=1024,
                             chunks=(1, 1, 64, 64), compression="gzip")
            f.create_dataset(name+"_valid", (1, 1, hh, ww), dtype="uint8", fillvalue=1,
                             chunks=(1, 1, 64, 64), compression="gzip")
        f.create_dataset("native/common_valid_ms", (1, 1, h, w), dtype="uint8", fillvalue=1,
                         chunks=(1, 1, 64, 64), compression="gzip")
        f["rr/target"] = f["native/ms"]
        f["rr/target_valid"] = f["native/ms_valid"]
        f.create_dataset("splits/blocks", data=blocks)
        f.create_dataset("splits/candidate_patch_index", data=index)
        f.create_dataset("splits/candidate_keep", data=np.ones(3, dtype="uint8"))
        d = f.create_dataset("splits/patch_index", data=index)
        d.attrs.update(target_size=64, dependency_buffer=128, columns="row,col,split_code,block_id")
    return path


class V3DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = write_scene(Path(self.temp.name)/"scene.h5")

    def tearDown(self):
        self.temp.cleanup()

    def test_supplied_split_counts_and_no_false_campaign_gate(self):
        with SceneV3(self.path, expected_sha256=file_sha(self.path)) as scene:
            summary = scene.summary()
            self.assertEqual(summary["counts"], dict(train=1, val=1, audit=1))
            self.assertFalse(summary["campaign_gates_verified"])
            self.assertFalse(summary["rr_redegraded"])
            self.assertEqual(len(summary["patch_index_sha256"]), 64)
            ids = [RRPatchV3(scene, split)[0]["raw_index"] for split in ("train", "val", "audit")]
            self.assertEqual(ids, [0, 1, 2])

    def test_rr_shapes_normalization_exactly_once_and_no_clipping(self):
        with h5py.File(self.path, "r+") as f:
            f["rr/pan"][0, 0, 128, 128] = 0
            f["rr/pan"][0, 0, 128, 129] = 3000
        with SceneV3(self.path, expected_sha256=file_sha(self.path)) as scene:
            a, b = RRPatchV3(scene)[0], RRPatchV3(scene, normalize=False)[0]
            self.assertEqual(tuple(a["pan"].shape), (1, 64, 64))
            self.assertEqual(tuple(a["ms"].shape), (8, 16, 16))
            self.assertEqual(tuple(a["lpan"].shape), (1, 16, 16))
            self.assertEqual(tuple(a["gt"].shape), (8, 64, 64))
            self.assertEqual(a["pan"][0, 0, 0].item(), -1)
            self.assertTrue(a["mask"][0, 0, 0].item())  # finite DN0 is valid
            self.assertGreater(a["pan"][0, 0, 1].item(), 1)
            np.testing.assert_array_equal(a["pan"], b["pan"].numpy()*np.float32(2/2047)-np.float32(1))
            self.assertFalse(a["provisional"])

    def test_target_pan_joint_support(self):
        with h5py.File(self.path, "r+") as f:
            f["rr/pan_valid"][0, 0, 128, 128] = 0
        with SceneV3(self.path) as scene:
            mask = RRPatchV3(scene)[0]["mask"]
            self.assertFalse(mask[0, 0, 0])
            self.assertEqual(int(mask.sum()), 64*64-1)

    def test_native_coordinates_and_no_resampling(self):
        with SceneV3(self.path) as scene:
            a = scene.read_region("native", 4, 8, 12, 16)
            self.assertEqual(a["pan"].shape, (1, 48, 64))
            self.assertEqual(a["ms"].shape, (8, 12, 16))
            self.assertEqual(a["coordinates"]["source_ms_row_col"], [12, 20])
            self.assertEqual(a["coordinates"]["source_pan_row_col"], [53, 73])
            self.assertFalse(a["normalized"])

    def test_sha_mismatch_and_diagnostic_reader(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            SceneV3(self.path, expected_sha256="0"*64)
        with SceneV3(self.path, verify_bytes=False) as scene:
            self.assertIsNone(scene.sha256)
            self.assertTrue(RRPatchV3(scene)[0]["provisional"])
        with self.assertRaisesRegex(ValueError, "requires byte"):
            SceneV3(self.path, expected_sha256="0"*64, verify_bytes=False)

    def test_local_fingerprint_without_external_binding_is_provisional(self):
        with SceneV3(self.path) as scene:
            self.assertIsNotNone(scene.sha256)
            self.assertTrue(RRPatchV3(scene).provisional)
            self.assertTrue(RRPatchV3(scene)[0]["provisional"])

    def test_raw_dn_for_baselines_is_not_training_ready(self):
        with SceneV3(self.path, expected_sha256=file_sha(self.path)) as scene:
            self.assertFalse(RRPatchV3(scene).provisional)
            raw = RRPatchV3(scene, normalize=False)
            self.assertTrue(raw.provisional)
            self.assertTrue(raw[0]["provisional"])
            self.assertEqual(float(raw[0]["pan"].mean()), 1024)

    def test_direct_symlink_scene_is_rejected_before_resolution(self):
        linked = self.path.with_name("linked.h5")
        linked.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "symlink scene"):
            SceneV3(linked)

    def test_source_mutation_rejected_even_after_lazy_open(self):
        with SceneV3(self.path) as scene:
            scene.read_region("rr", 128, 128, 64, 64)
            old = self.path.stat()
            os.utime(self.path, ns=(old.st_atime_ns, old.st_mtime_ns+1000000))
            with self.assertRaisesRegex(ValueError, "changed since"):
                RRPatchV3(scene)[0]

    def test_pickle_reopens_a_readonly_process_local_handle(self):
        with SceneV3(self.path) as scene:
            original = RRPatchV3(scene)[0]
            copied = pickle.loads(pickle.dumps(scene))
            try:
                self.assertIsNone(copied._file)
                np.testing.assert_array_equal(RRPatchV3(copied)[0]["gt"], original["gt"])
                self.assertEqual(copied._file.mode, "r")
            finally:
                copied.close()

    def test_duplicate_patch_rejected(self):
        with h5py.File(self.path, "r+") as f:
            f["splits/patch_index"][1, :2] = f["splits/patch_index"][0, :2]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            SceneV3(self.path)

    def test_overlapping_blocks_rejected(self):
        with h5py.File(self.path, "r+") as f:
            f["splits/blocks"][1, 1] = 448
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            SceneV3(self.path)

    def test_buffer_crossing_rejected(self):
        with h5py.File(self.path, "r+") as f:
            f["splits/patch_index"][0, 0] = 64
        with self.assertRaisesRegex(ValueError, "dependency buffer"):
            SceneV3(self.path)

    def test_non_global_phase_rejected(self):
        with h5py.File(self.path, "r+") as f:
            f["splits/patch_index"][0, 0] = 132
        with self.assertRaisesRegex(ValueError, "phase"):
            SceneV3(self.path)

    def test_low_resolution_invalid_mask_rejected(self):
        with h5py.File(self.path, "r+") as f:
            f["rr/lpan_valid"][0, 0, 32, 32] = 0
        with SceneV3(self.path) as scene, self.assertRaisesRegex(ValueError, "low-resolution"):
            RRPatchV3(scene)[0]

    def test_dependency_support_invalid_rejected(self):
        with h5py.File(self.path, "r+") as f:
            f["native/common_valid_ms"][0, 0, 0, 0] = 0
        with SceneV3(self.path) as scene, self.assertRaisesRegex(ValueError, "dependency support"):
            RRPatchV3(scene)[0]

    def test_nonfinite_and_nonbinary_mask_rejected(self):
        for dataset, value in (("rr/pan", np.nan), ("rr/pan_valid", 2)):
            with self.subTest(dataset=dataset):
                write_scene(self.path)
                with h5py.File(self.path, "r+") as f:
                    f[dataset][0, 0, 128, 128] = value
                with SceneV3(self.path) as scene, self.assertRaisesRegex(ValueError, "Nonfinite"):
                    RRPatchV3(scene)[0]

    def test_external_and_soft_links_rejected(self):
        for link in (h5py.ExternalLink("untrusted.h5", "data"), h5py.SoftLink("/native/ms")):
            with self.subTest(link=type(link)):
                write_scene(self.path)
                with h5py.File(self.path, "r+") as f:
                    f["extra_link"] = link
                with self.assertRaisesRegex(ValueError, "external/soft"):
                    SceneV3(self.path)

    def test_virtual_dataset_rejected(self):
        with h5py.File(self.path, "r+") as f:
            layout = h5py.VirtualLayout((1,), dtype="float32")
            layout[:] = h5py.VirtualSource("untrusted.h5", "data", shape=(1,))[:]
            f.create_virtual_dataset("extra_vds", layout)
        with self.assertRaisesRegex(ValueError, "external/virtual"):
            SceneV3(self.path)

    def test_copied_target_instead_of_hardlink_rejected(self):
        with h5py.File(self.path, "r+") as f:
            del f["rr/target"]
            f.create_dataset("rr/target", shape=f["native/ms"].shape, dtype="uint16", fillvalue=1024)
        with self.assertRaisesRegex(ValueError, "hard-link"):
            SceneV3(self.path)

    def test_crop_mismatch_and_wrong_normalization_rejected(self):
        for key, value in (("source_ms_row", 12), ("nominal_max", 65535), ("complete", False)):
            with self.subTest(key=key):
                write_scene(self.path)
                with h5py.File(self.path, "r+") as f:
                    f.attrs[key] = value
                with self.assertRaises(ValueError):
                    SceneV3(self.path)

    def test_read_bounds_and_index_types(self):
        with SceneV3(self.path) as scene:
            for args in (("rr", 1, 0, 64, 64), ("native", -1, 0, 64, 64),
                         ("native", 500, 0, 64, 64), ("rr", 0, 0, True, 64)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    scene.read_region(*args)
            for index in (-1, True, 0.0):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    RRPatchV3(scene)[index]
            with self.assertRaises(IndexError):
                RRPatchV3(scene)[1]

    def test_candidate_filter_mismatch_rejected(self):
        with h5py.File(self.path, "r+") as f:
            f["splits/candidate_keep"][0] = 0
        with self.assertRaisesRegex(ValueError, "candidate/filter"):
            SceneV3(self.path)

    def test_candidate_provenance_hash_rejected(self):
        with h5py.File(self.path, "r+") as f:
            f["splits/candidate_patch_index"][0, 0] += 64
            f["splits/patch_index"][0, 0] += 64
        with self.assertRaisesRegex(ValueError, "provenance SHA"):
            SceneV3(self.path)

    def test_block_provenance_hash_rejected(self):
        with h5py.File(self.path, "r+") as f:
            f["splits/blocks"][0, 2] = 448
        with self.assertRaisesRegex(ValueError, "block assignment provenance"):
            SceneV3(self.path)


if __name__ == "__main__":
    unittest.main()
