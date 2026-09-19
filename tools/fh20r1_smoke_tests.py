#!/usr/bin/env python3
"""CPU/tmp-only N2 smoke integration; no CUDA calls or production data writes."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fh12.model import build_model, state_hash
from fh12.common import atomic_json, sha256
from fh20r1 import smoke as S
from fh20r1.plan import case_for, N2_TEACHER_RUN


class SmokeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory(prefix="fh20r1-smoke-")
        self.root = Path(self.temp.name)
        self.case = replace(case_for(N2_TEACHER_RUN), width=8, depth=(1, 1, 1))
        model, init = build_model("PL", 8, [1, 1, 1], 71001)
        self.donor = {k: v.clone() for k, v in model.aligner.state_dict().items()}
        self.donor["fc2.weight"].fill_(.01); self.donor["fc2.bias"].fill_(.02)
        path = self.root / "init.json"; atomic_json(path, init)
        self.bridge = dict(alias="F2", server="s2", status="PASS", complete=True,
                           resolved_artifacts=dict(init_manifest=str(path)),
                           resolved_artifact_sha256=dict(init_manifest=sha256(path)))

    def tearDown(self): self.temp.cleanup()

    def invoke(self):
        with patch.object(S, "case_for", return_value=self.case), \
             patch.object(S, "load_donor_aligner", return_value=(self.donor, dict(status="VERIFIED"))), \
             patch.object(S.torch.cuda, "empty_cache", side_effect=AssertionError("CPU must not call CUDA")):
            return S.n2_teacher_smoke("s2", self.bridge, "cpu", self.root)

    def test_even_odd_adamw_and_donor_head_preserved_without_mutating_donor_or_rng(self):
        before = state_hash(self.donor); torch.manual_seed(231); rng = torch.get_rng_state().clone()
        report = self.invoke()
        self.assertTrue(report["passed"]); self.assertTrue(report["fresh_F2_U_verified"])
        self.assertEqual(report["actual_batch48_smoke_status"], "NOT_RUN_CPU_DIAGNOSTIC")
        self.assertEqual(report["batch_size"], 2); self.assertEqual(report["patch_size"], 32)
        self.assertEqual([r["offset_active"] for r in report["updates"]], [False, True])
        self.assertEqual([r["update_index"] for r in report["updates"]], [0, 1])
        self.assertEqual(state_hash(self.donor), before); self.assertEqual(report["donor_aligner_hash"], before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(report["credited_new_seconds"], 0)

    def test_unavailable_donor_is_structured_skip_without_build(self):
        with patch.object(S, "load_donor_aligner", return_value=(None, dict(status="BLOCKED_DONOR"))), \
             patch.object(S, "build_n2_teacher", side_effect=AssertionError("must not initialize random A")):
            report = S.n2_teacher_smoke("s2", {}, "cuda", self.root)
        self.assertEqual(report["status"], "SKIPPED_BLOCKED_DONOR"); self.assertFalse(report["passed"])

    def test_other_servers_skip_without_donor_access(self):
        with patch.object(S, "load_donor_aligner", side_effect=AssertionError("local s2 only")):
            self.assertEqual(S.n2_teacher_smoke("s3", {}, "cuda", self.root)["status"], "NOT_APPLICABLE")

    def test_changed_f2_initialization_manifest_fails(self):
        self.bridge["resolved_artifact_sha256"]["init_manifest"] = "wrong"
        with self.assertRaisesRegex(ValueError, "manifest changed"): self.invoke()

    def test_cuda_unavailable_is_not_reported_as_cpu_success(self):
        with patch.object(S, "load_donor_aligner", return_value=(self.donor, dict(status="VERIFIED"))), \
             patch.object(S.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                S.n2_teacher_smoke("s2", self.bridge, "cuda", self.root)


if __name__ == "__main__": unittest.main(verbosity=2)
