"""Synthetic-only regression tests for the read-only PANPV HDF5 auditor."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np


SPEC = importlib.util.spec_from_file_location("panpv_input_audit", Path(__file__).with_name("panpv_input_audit.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class InputAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "input"
        self.root.mkdir()

    def write(self, name="sample.h5", **arrays):
        path = self.root / name
        with h5py.File(path, "w") as handle:
            handle.attrs["product"] = "UNCONFIRMED_TEST_FIXTURE"
            for key, value in arrays.items():
                handle.create_dataset(key, data=value)
        return path

    @staticmethod
    def dataset(report, path):
        return next(row for row in report["datasets"] if row["path"] == path)

    def test_full_statistics_finite_zero_threshold(self):
        path = self.write(data=np.array([[0, 1, 2047, 2048, -1, np.nan, np.inf, -np.inf]], dtype=np.float32))
        result = audit.audit_h5(path, chunk_bytes=1024)
        full = self.dataset(result, "/data")["numeric"]["full"]
        self.assertEqual(full["elements"], 8)
        self.assertEqual(full["finite"], 5)
        self.assertEqual(full["nonfinite"], 3)
        self.assertEqual(full["zero"], 1)
        self.assertEqual(full["greater_than_2047"], 1)
        self.assertEqual((full["min"], full["max"]), (-1, 2048))
        json.dumps(result, allow_nan=False)

    def test_dataset_bytes_are_not_changed(self):
        path = self.write(pan=np.arange(256, dtype=np.uint16).reshape(1, 1, 16, 16))
        before = (audit.file_sha(path), path.stat().st_mtime_ns)
        result = audit.audit_h5(path)
        self.assertEqual(before, (audit.file_sha(path), path.stat().st_mtime_ns))
        self.assertEqual(before[0], result["sha256"])
        self.assertEqual(result["open_mode"], "r")

    def test_samples_are_fixed_uniform_and_limited(self):
        values = np.arange(300 * 8 * 4 * 4, dtype=np.uint16).reshape(300, 8, 4, 4)
        path = self.write(gt=values)
        result = self.dataset(audit.audit_h5(path), "/gt")["numeric"]
        sampled = result["sampled"]
        ids = np.linspace(0, 299, 128, dtype=np.int64).tolist()
        self.assertEqual(sampled["axis0_indices"], ids)
        self.assertTrue(sampled["sampled_only"])
        self.assertEqual(len(sampled["per_band_sampled"]), 8)
        self.assertEqual(sampled["all_values"]["quantiles"]["0.5"], float(np.median(values[ids])))
        self.assertEqual(result["full"]["elements"], values.size)

    def test_meta_duplicate_keys_keep_scene_context(self):
        meta = np.array([[0, 0, 0, 2], [0, 64, 0, 2], [0, 64, 0, 3],
                         [1, 0, 0, 1], [1, 128, 64, 1]], dtype=np.int64)
        path = self.write(meta=meta)
        result = self.dataset(audit.audit_h5(path), "/meta")["meta_audit"]
        self.assertEqual(result["distinct_scene_xy"], 4)
        self.assertEqual(result["duplicate_scene_xy_extra_rows"], 1)
        self.assertEqual(result["scenes"]["unique_count"], 2)
        self.assertEqual(result["categories"]["unique_count"], 3)
        self.assertEqual(result["grid_by_scene"][0]["x"]["step_gcd"], 64)
        self.assertFalse(result["spatial_overlap_verified"])

    def test_meta_noninteger_and_nonfinite_are_not_coerced(self):
        path = self.write(meta=np.array([[0, 1.5, 0, 1], [0, np.nan, 0, 1], [0, 64, 0, 1]]))
        result = self.dataset(audit.audit_h5(path), "/meta")["meta_audit"]
        self.assertEqual(result["invalid_nonfinite_or_noninteger_rows"], 2)
        self.assertEqual(result["status"], "INVALID_META_ROWS")

    def test_rr_is_candidate_not_native_gt_claim(self):
        path = self.write(pan=np.zeros((2, 1, 64, 64)), ms=np.zeros((2, 8, 16, 16)),
                          gt=np.zeros((2, 8, 64, 64)))
        result = audit.audit_h5(path)
        self.assertEqual(result["geometry_candidates"][0]["classification_candidate"], "RR_TUPLE")
        self.assertFalse(result["native_fr_verified"])
        self.assertFalse(result["split_frozen"])

    def test_grouped_native_pair_still_requires_semantics(self):
        path = self.write(**{"fr/pan": np.zeros((2, 1, 64, 64)), "fr/ms": np.zeros((2, 8, 16, 16))})
        result = audit.audit_h5(path)
        self.assertEqual(result["native_fr_presence"], "CANDIDATE_REQUIRES_METADATA_VERIFICATION")
        self.assertFalse(result["native_fr_verified"])

    def test_soft_and_external_links_are_rejected(self):
        for name, link in [("soft", h5py.SoftLink("/data")),
                           ("external", h5py.ExternalLink("absent.h5", "/data"))]:
            with self.subTest(name=name):
                path = self.write(name + ".h5", data=np.ones(4))
                with h5py.File(path, "a") as handle:
                    handle["bad"] = link
                with self.assertRaisesRegex(audit.UnsafeHDF5, "BLOCKED_HDF5_LINK"):
                    audit.audit_h5(path)

    def test_virtual_dataset_is_rejected_without_reading_source(self):
        path = self.root / "vds.h5"
        layout = h5py.VirtualLayout(shape=(3,), dtype=np.float32)
        layout[:] = h5py.VirtualSource("not-present.h5", "source", shape=(3,))
        with h5py.File(path, "w", libver="latest") as handle:
            handle.create_virtual_dataset("vds", layout)
        with self.assertRaisesRegex(audit.UnsafeHDF5, "BLOCKED_EXTERNAL_DATASET"):
            audit.audit_h5(path)

    def test_external_storage_rejected_without_reading(self):
        path = self.root / "external-data.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("external", shape=(3,), dtype=np.float32,
                                  external=[("unavailable.bin", 0, h5py.h5f.UNLIMITED)])
        with self.assertRaisesRegex(audit.UnsafeHDF5, "BLOCKED_EXTERNAL_DATASET"):
            audit.audit_h5(path)

    def test_hardlink_cycle_and_alias_safe(self):
        path = self.write(data=np.ones(4))
        with h5py.File(path, "a") as handle:
            handle["alias"] = handle["data"]
            group = handle.create_group("group")
            group["self"] = group
        result = audit.audit_h5(path)
        self.assertEqual(len(result["datasets"]), 1)
        self.assertEqual(len(result["hard_link_aliases"]), 2)

    def test_empty_and_scalar_datasets(self):
        path = self.write(empty=np.zeros((0, 4)), scalar=np.asarray(42), text=np.asarray(b"encoding unknown"))
        result = audit.audit_h5(path)
        self.assertEqual(self.dataset(result, "/empty")["numeric"]["full"]["elements"], 0)
        self.assertEqual(self.dataset(result, "/scalar")["numeric"]["full"]["min"], 42)
        self.assertEqual(self.dataset(result, "/text")["numeric"]["status"], "NOT_REAL_NUMERIC")

    def test_file_symlink_rejected(self):
        path = self.write(data=np.ones(4))
        link = self.root / "symlink.h5"
        link.symlink_to(path)
        with self.assertRaises(audit.UnsafeHDF5):
            audit.audit_h5(link)
        report = audit.audit_root(self.root)
        self.assertEqual(report["blocked_count"], 1)

    def test_cli_new_output_and_no_overwrite(self):
        self.write(data=np.ones(4))
        output = Path(self.tmp.name) / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main([str(self.root), "--output", str(output)]), 0)
        payload = output.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            audit.main([str(self.root), "--output", str(output)])
        self.assertEqual(output.read_bytes(), payload)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            audit.main([str(self.root), "--output", str(self.root / "bad.json")])

    def test_stream_chunks_cover_every_element_once(self):
        for shape in [(), (0, 3), (5,), (9, 5, 7), (5, 8, 31, 17)]:
            with self.subTest(shape=shape):
                counts = np.zeros(shape, dtype=np.int32)
                for selection in audit._slices(shape, 49):
                    self.assertLessEqual(counts[selection].size, 49)
                    counts[selection] += 1
                self.assertTrue(np.all(counts == 1))


if __name__ == "__main__":
    unittest.main()
