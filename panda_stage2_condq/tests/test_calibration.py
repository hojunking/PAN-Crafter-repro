"""Synthetic reference checks; no production data, GPU jobs, or old artifacts."""
from pathlib import Path
import json
import hashlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from panda_stage2_condq.calibration import (
    CalibrationBuilder, CalibrationStats, WARP_CONTRACT,
    sample_ids_sha256, validate_binding,
)


def binding(ids=("a", "b"), sensor="WV3"):
    return {
        "sensor": sensor, "source_split": "RR_train_only",
        "sample_ids_sha256": sample_ids_sha256(ids), "source_manifest_sha256": "a" * 64,
        "aligner_state_sha256": "b" * 64, "teacher_state_sha256": "c" * 64,
        "donor_checkpoint_id": "d" * 64,
        "reconstruction_spec": {"bands": 8 if sensor == "WV3" else 4, "width": 112,
                                "depth": [1, 2, 3], "input_layout": "P0", "base_reference": "BICUBIC_MS"},
        "scharr": {"kernel": "absolute_xy_/32", "padding": "reflect1",
                   "rgb_indices": [4, 2, 1] if sensor == "WV3" else [2, 1, 0], "roi_margin": 8},
        "warp": WARP_CONTRACT,
    }


def maps(values, *, shape=(18, 18), exterior=999.0):
    result = torch.full((1, 1, *shape), exterior, dtype=torch.float32)
    result[..., 8:-8, 8:-8] = torch.tensor(values, dtype=torch.float32).reshape(1, 1, shape[0]-16, shape[1]-16)
    return result


def stats_from_roi(g_values, q_values, *, k=2, shape=(18, 18)):
    ids = ["one"]
    b = CalibrationBuilder(ids, binding(ids), k=k, memory_budget_bytes=8 * 1024 * 1024)
    b.add(ids, torch.ones(1, 1, *shape), maps(g_values, shape=shape), maps(q_values, shape=shape), sensor="WV3")
    return b.finalize()


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_full_image_even_global_median_not_roi_or_batch_median(self):
        ids = ["a", "b", "c"]
        b = CalibrationBuilder(ids, binding(ids), k=1, memory_budget_bytes=8 * 1024 * 1024)
        # 3 uneven batches: true global median 2, not mean(batch medians)=3.
        for name, value in zip(ids, [0.0, 2.0, 7.0]):
            e = torch.full((1, 1, 18, 18), value)
            e[..., 8:-8, 8:-8] = 1000.0
            b.add([name], e, maps([0, 1, 2, 3]), maps([0, 1, 2, 3]), sensor="WV3")
        result = b.finalize()
        self.assertEqual(result.tau_e, 2.0)
        self.assertEqual(result.metadata["n_e"], 3 * 18 * 18)
        self.assertEqual(result.metadata["n_q"], 3 * 2 * 2)
        self.assertEqual(result.metadata["bin_counts"], [12])

    def test_midpoint_even_uses_two_centers_fp64(self):
        ids = ["a", "b"]
        b = CalibrationBuilder(ids, binding(ids), include_condq=False, memory_budget_bytes=1024 * 1024)
        low = np.float32(1.0)
        high = np.nextafter(low, np.float32(2.0))
        e = torch.stack([torch.full((1, 18, 18), float(low)), torch.full((1, 18, 18), float(high))])
        b.add(ids, e, sensor="WV3")
        result = b.finalize()
        self.assertEqual(result.tau_e, (float(low) + float(high)) / 2)
        self.assertNotEqual(result.tau_e, float(np.float32((float(low) + float(high)) / 2)))

    def test_exact_ecdf_right_ties_and_extremes(self):
        result = stats_from_roi([0, 0, 0], [.1, .1, .3], k=1, shape=(17, 19))
        query = torch.tensor([.1, .2, .3, 0., 1.], dtype=torch.float32).reshape(1, 1, 1, 5)
        ranks = result.lookup_exact(torch.zeros_like(query), query)
        torch.testing.assert_close(ranks.flatten(), torch.tensor([2/3, 2/3, 1., 0., 1.]))
        self.assertEqual(result.metadata["q_ties_by_bin"][0]["tied_sample_count"], 2)

    def test_zero_q_can_have_rank_one(self):
        result = stats_from_roi([1, 1, 1, 1], [0, 0, 0, 0], k=5)
        query = torch.zeros(1, 1, 1, 1)
        self.assertEqual(float(result.lookup_exact(query, query)), 1.0)
        self.assertEqual(result.metadata["q_ties_by_bin"][0]["zero_count"], 4)

    def test_g_cutpoint_ties_go_to_upper_group(self):
        result = stats_from_roi([0, 1, 1, 2], [.1, .2, .3, .4], k=2)
        self.assertEqual(result.bin_counts, (1, 3))
        rank = result.lookup_exact(torch.tensor([[[[1.0]]]]), torch.tensor([[[[.2]]]]))
        self.assertAlmostEqual(float(rank), 1/3, places=7)
        counts = result.group_counts(torch.tensor([0., 1., 1., 2., 3.]).reshape(1, 1, 1, 5))
        self.assertEqual(counts.tolist(), [1, 4])
        self.assertEqual(result.bin_counts, (1, 3))

    def test_quantiles_fp64_and_lookup_never_rounds_cutpoints_to_fp32(self):
        high = float(np.nextafter(np.float32(1), np.float32(2)))
        result = stats_from_roi([1, 1, high, high], [.1, .1, .9, .9], k=2)
        self.assertEqual(result.metadata["g_grouping"]["raw_quantile_cutpoints"], [(1.0 + high) / 2])
        g = torch.tensor([1., high]).reshape(1, 1, 1, 2)
        q = torch.full_like(g, .2)
        torch.testing.assert_close(result.lookup_exact(g, q).flatten(), torch.tensor([1., 0.]))

    def test_duplicate_boundaries_and_empty_bins_merge_deterministically(self):
        result = stats_from_roi([0, 0, 0, 1], [0, 1, 2, 3], k=8)
        self.assertGreater(result.metadata["g_grouping"]["duplicate_boundaries_removed"], 0)
        self.assertTrue(result.metadata["g_grouping"]["empty_bin_merges"])
        self.assertEqual(sum(result.bin_counts), 4)
        self.assertTrue(all(n > 0 for n in result.bin_counts))
        again = stats_from_roi([0, 0, 0, 1], [0, 1, 2, 3], k=8)
        self.assertEqual(result.artifact_sha256, again.artifact_sha256)

    def test_constant_g_collapses_to_one_nonempty_group(self):
        result = stats_from_roi([3, 3, 3, 3], [1, 2, 3, 4], k=5)
        self.assertEqual(result.bin_counts, (4,))
        self.assertEqual(result.metadata["g_grouping"]["k_effective"], 1)

    def test_training_global_not_batch_dependent(self):
        result = stats_from_roi([0, 1, 2, 3], [.1, .2, .7, .8], k=2)
        g = torch.tensor([[[[2.0]]]])
        q = torch.tensor([[[[.7]]]])
        scalar = result.lookup_exact(g, q)
        batch = result.lookup_exact(g.expand(5, 1, 8, 9), q.expand(5, 1, 8, 9))
        self.assertTrue(torch.equal(batch, scalar.expand_as(batch)))

    def test_inventory_order_and_batch_partition_do_not_change_artifact(self):
        ids = ["a", "b"]
        one = CalibrationBuilder(ids, binding(ids), k=2, memory_budget_bytes=1024*1024)
        two = CalibrationBuilder(list(reversed(ids)), binding(ids), k=2, memory_budget_bytes=1024*1024)
        e = torch.stack([torch.ones(1, 18, 18), torch.ones(1, 18, 18)*2])
        g = torch.cat([maps([0, 1, 2, 3]), maps([4, 5, 6, 7])])
        q = g.clone()
        one.add(ids, e, g, q, sensor="WV3")
        two.add(["b"], e[1:], g[1:], q[1:], sensor="WV3")
        two.add(["a"], e[:1], g[:1], q[:1], sensor="WV3")
        self.assertEqual(one.finalize().artifact_sha256, two.finalize().artifact_sha256)

    def test_missing_ids_fail_and_builder_cannot_resume(self):
        b = CalibrationBuilder(["a", "b"], binding(), include_condq=False, memory_budget_bytes=1024*1024)
        b.add(["a"], torch.ones(1, 1, 18, 18), sensor="WV3")
        with self.assertRaisesRegex(ValueError, "missing"):
            b.finalize()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            b.add(["b"], torch.ones(1, 1, 18, 18), sensor="WV3")

    def test_duplicate_unknown_wrong_split_sensor_augmented_support_fail(self):
        cases = [({"split": "val"}, ["a"]), ({"split": "test"}, ["a"]),
                 ({"resolution": "FR"}, ["a"]), ({"sensor": "GF2"}, ["a"]),
                 ({"unaugmented": False}, ["a"]), ({"support_ok": False}, ["a"]),
                 ({}, ["foreign"]), ({}, ["a", "a"])]
        for kwargs, ids in cases:
            with self.subTest(kwargs=kwargs, ids=ids):
                b = CalibrationBuilder(["a", "b"], binding(), include_condq=False, memory_budget_bytes=1024*1024)
                with self.assertRaises(ValueError):
                    b.add(ids, torch.ones(len(ids), 1, 18, 18), **{"sensor": "WV3", **kwargs})
                with self.assertRaises(RuntimeError):
                    b.finalize()
        b = CalibrationBuilder(["a", "b"], binding(), include_condq=False, memory_budget_bytes=1024*1024)
        b.add(["a"], torch.ones(1, 1, 18, 18), sensor="WV3")
        with self.assertRaises(ValueError):
            b.add(["a"], torch.ones(1, 1, 18, 18), sensor="WV3")

    def test_rejects_bad_binding_and_inventory(self):
        for field, value in [("sensor", "GF2"), ("source_split", "RR_test"),
                             ("source_manifest_sha256", "missing"), ("warp", "old")]:
            bad = binding()
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_binding(bad)
        with self.assertRaises(ValueError):
            CalibrationBuilder(["a", "changed"], binding(), memory_budget_bytes=1024*1024)
        with self.assertRaises(ValueError):
            sample_ids_sha256(["a", "a"])
        self.assertEqual(sample_ids_sha256(["한글", "ascii"]),
                         hashlib.sha256(json.dumps(sorted(["한글", "ascii"]), sort_keys=True,
                                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest())

    def test_budget_fails_before_population_allocation_without_fallback(self):
        b = CalibrationBuilder(["a", "b"], binding(), memory_budget_bytes=1)
        with mock.patch("panda_stage2_condq.calibration.np.empty") as allocate:
            with self.assertRaisesRegex(MemoryError, "no approximation"):
                b.add(["a"], torch.ones(1, 1, 18, 18), maps([0, 1, 2, 3]), maps([0, 1, 2, 3]), sensor="WV3")
            allocate.assert_not_called()

    def test_tau_only_skips_q_collection_and_lookup(self):
        b = CalibrationBuilder(["a"], binding(["a"]), include_condq=False, memory_budget_bytes=1024*1024)
        b.add(["a"], torch.ones(1, 1, 18, 18), sensor="WV3")
        result = b.finalize()
        self.assertEqual(result.metadata["n_q"], 0)
        self.assertEqual(result.bin_counts, ())
        with self.assertRaises(ValueError):
            result.lookup_exact(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))

    def test_map_dtype_shape_gradient_and_nonfinite_guards(self):
        bad_maps = [torch.ones(1, 1, 18, 18).double(), torch.ones(1, 2, 18, 18),
                    torch.ones(1, 1, 18, 18, requires_grad=True), torch.full((1, 1, 18, 18), float("nan")),
                    torch.full((1, 1, 18, 18), -1.0)]
        for e in bad_maps:
            b = CalibrationBuilder(["a"], binding(["a"]), include_condq=False, memory_budget_bytes=1024*1024)
            with self.subTest(shape=e.shape, dtype=e.dtype), self.assertRaises(ValueError):
                b.add(["a"], e, sensor="WV3")

    def test_lookup_shape_dtype_finite_and_detach(self):
        result = stats_from_roi([0, 1, 2, 3], [0, 1, 2, 3])
        g = torch.ones(1, 1, 1, 1, requires_grad=True)
        q = torch.ones_like(g, requires_grad=True)
        self.assertFalse(result.lookup_exact(g, q).requires_grad)
        for bad in [q.double(), q.flatten(), torch.full_like(q, float("nan")), -q]:
            with self.assertRaises(ValueError):
                result.lookup_exact(g, bad)
        self.assertIs(result.to("cpu"), result)
        with self.assertRaises(MemoryError):
            result.to("cpu", memory_budget_bytes=1)
        with self.assertRaisesRegex(ValueError, "explicit memory_budget"):
            result.to("cuda")

    def test_binding_requires_exact_donor_data_operator_match(self):
        result = stats_from_roi([0, 1, 2, 3], [0, 1, 2, 3])
        for key in ["teacher_state_sha256", "aligner_state_sha256", "source_manifest_sha256", "sample_ids_sha256"]:
            bad = result.binding
            bad[key] = "f" * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "binding mismatch"):
                result.validate_binding(bad)
        bad = result.binding
        bad["unexpected_extra"] = "new provenance"
        with self.assertRaises(ValueError):
            result.validate_binding(bad)

    def test_atomic_roundtrip_hash_and_immutable_no_overwrite(self):
        result = stats_from_roi([0, 1, 2, 3], [0, .1, .1, .3])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"stats.npz"
            receipt = result.save(path)
            loaded = CalibrationStats.load(path, expected_binding=result.binding, expected_file_sha256=receipt["file_sha256"])
            self.assertEqual(loaded.artifact_sha256, result.artifact_sha256)
            self.assertEqual(loaded.metadata, result.metadata)
            self.assertEqual(loaded.bin_counts, result.bin_counts)
            with self.assertRaises(FileExistsError):
                result.save(path)
            with self.assertRaises(ValueError):
                CalibrationStats.load(path, expected_binding=result.binding, expected_file_sha256="0"*64)
            with self.assertRaises(MemoryError):
                CalibrationStats.load(path, expected_binding=result.binding, memory_budget_bytes=1)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_atomic_failure_never_publishes_partial_artifact(self):
        result = stats_from_roi([0, 1, 2, 3], [0, 1, 2, 3])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"stats.npz"
            with mock.patch("panda_stage2_condq.calibration.np.savez", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    result.save(path)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_tampered_archive_is_rejected(self):
        result = stats_from_roi([0, 1, 2, 3], [0, 1, 2, 3])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"stats.npz"
            result.save(path)
            with np.load(path, allow_pickle=False) as archive:
                contents = {key: archive[key] for key in archive.files}
            contents["q_0000"][0] += .01
            tampered = Path(tmp)/"tampered.npz"
            np.savez(tampered, **contents)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                CalibrationStats.load(tampered, expected_binding=result.binding)

    def test_archive_declared_population_cannot_bypass_load_memory_budget(self):
        result = stats_from_roi([0, 1, 2, 3], [0, 1, 2, 3])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"stats.npz"
            result.save(path)
            with np.load(path, allow_pickle=False) as archive:
                contents = {key: archive[key] for key in archive.files}
            metadata = json.loads(contents["metadata_json"].tobytes())
            metadata["bin_counts"][0] = 1
            contents["metadata_json"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
            bad = Path(tmp)/"bad.npz"
            np.savez(bad, **contents)
            with self.assertRaisesRegex(ValueError, "archive size/dtype"):
                CalibrationStats.load(bad, expected_binding=result.binding, memory_budget_bytes=1024*1024)

    def test_metadata_copies_cannot_mutate_stats(self):
        result = stats_from_roi([0, 1, 2, 3], [0, 1, 2, 3])
        value = result.metadata
        value["tau_e"] = -100
        value["binding"]["sensor"] = "GF2"
        self.assertEqual(result.tau_e, 1)
        self.assertEqual(result.binding["sensor"], "WV3")

    def test_table_mutation_is_rejected_before_lookup(self):
        for name in ["_cuts", "_q"]:
            with self.subTest(name=name):
                result = stats_from_roi([0, 1, 2, 3], [0, 1, 2, 3])
                table = result._cuts if name == "_cuts" else result._q[0]
                table.add_(0.1)
                with self.assertRaisesRegex(ValueError, "immutable calibration tensor"):
                    result.lookup_exact(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1))

    def test_full_audit_detects_data_alias_mutation(self):
        result = stats_from_roi([0, 1, 2, 3], [0, 1, 2, 3])
        result._q[0].data.add_(0.1)
        with self.assertRaisesRegex(ValueError, "content SHA256"):
            result.assert_integrity(full_hash=True)

    def test_gf2_and_qb_binding_rgb_bands(self):
        for sensor in ["GF2", "QB"]:
            b = CalibrationBuilder(["a"], binding(["a"], sensor), include_condq=False, memory_budget_bytes=1024*1024)
            b.add(["a"], torch.ones(1, 1, 18, 18), sensor=sensor)
            self.assertEqual(b.finalize().binding["reconstruction_spec"]["bands"], 4)


if __name__ == "__main__":
    unittest.main()
