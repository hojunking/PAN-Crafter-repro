"""Quantiles, independent fixed eligibility, penalty, and rank semantics."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from panda_method.calibration import canonical_sha256
from pa.warp import warp_pan
from ta2.registration import register_band
from panda_mature30.selection import (
    PROTOCOL, aggregate_rows, target_eligibility, validate_inventory,
    geometry_rank, rank, validate_receipt, SCHEMA, SELECTOR,
)
from panda_mature30.contract import CAMPAIGN_ID


def inventory():
    result = {"sample_count": 1080, "sample_ids": [f"WV3:val:{i}" for i in range(1080)],
              "eligible": [True] * 1080, "eligible_count": 1080,
              "excluded_gt_low_texture_count": 0,
              "selector_protocol_sha256": canonical_sha256(PROTOCOL)}
    result["inventory_sha256"] = canonical_sha256(result)
    return result


class SelectionTests(unittest.TestCase):
    def rows(self):
        return [{"sample_id": f"WV3:val:{i}", "eligible": True, "valid": True,
                 "rre_px": i / 1080., "failure_reason": None} for i in range(1080)]

    def test_quantile_definitions_and_failed_registration_penalty(self):
        rows, pop = self.rows(), inventory()
        out = aggregate_rows(rows, pop)
        self.assertAlmostEqual(out["RRE50"], 539.5 / 1080)
        self.assertAlmostEqual(out["RRE90"], 971.1 / 1080)
        for row in rows[540:]:
            row.update(valid=False, rre_px=12., failure_reason="GRID_BOUNDARY")
        out = aggregate_rows(rows, pop)
        self.assertEqual(out["RRE90"], 12.)
        self.assertEqual(out["failure_count"], 540)
        rows[-1]["rre_px"] = 0
        with self.assertRaisesRegex(ValueError, "penalty"):
            aggregate_rows(rows, pop)

    def test_model_cannot_drop_or_relabel_difficult_sample(self):
        rows, pop = self.rows(), inventory()
        with self.assertRaises(ValueError):
            aggregate_rows(rows[:-1], pop)
        rows[-1]["eligible"] = False
        with self.assertRaises(ValueError):
            aggregate_rows(rows, pop)
        bad = deepcopy(pop)
        bad["eligible"][0] = False
        with self.assertRaises(ValueError):
            validate_inventory(bad)

    def test_gt_only_texture_and_known_sampling_shift(self):
        self.assertFalse(target_eligibility(np.full((64, 64), .3, np.float32))[0])
        generator = torch.Generator().manual_seed(140)
        image = torch.rand(1, 1, 64, 64, generator=generator)
        shift = torch.tensor([[.35, -.6]])
        target = warp_pan(image, shift)
        self.assertTrue(target_eligibility(target[0, 0].numpy())[0])
        record = register_band(image[0, 0].numpy(), target[0, 0].numpy(),
                               sigma_p=.8, sigma_r=.8)
        self.assertTrue(record["valid"])
        np.testing.assert_allclose(record["shift"], shift[0].numpy(), atol=.16)
        registered = warp_pan(image, shift)
        residual = register_band(registered[0, 0].numpy(), target[0, 0].numpy(),
                                 sigma_p=.8, sigma_r=.8)
        self.assertLess(np.linalg.norm(residual["shift"]), .02)

    def test_rank_uses_geometry_before_reconstruction_not_hqnr(self):
        a = {"validation": {"RRE50": .1, "RRE90": .3, "Val_ERGAS": 3.},
             "completed_update": 30000, "HQNR": .80}
        b = {"validation": {"RRE50": .2, "RRE90": .2, "Val_ERGAS": 2.},
             "completed_update": 1000, "HQNR": .99}
        self.assertLess(rank(a), rank(b))
        b["validation"].update(RRE50=.1, RRE90=.3)
        self.assertLess(rank(b), rank(a))
        self.assertEqual(geometry_rank(a), geometry_rank(b))

    def test_full_receipt_recomputes_scores_not_just_hashes(self):
        pop, rows = inventory(), self.rows()
        score = aggregate_rows(rows, pop)
        score.update(per_sample=rows, Val_ERGAS=2.)
        candidate = {"validation": score, "completed_update": 50000,
                     "source_run_id": "source", "source_campaign_id": CAMPAIGN_ID,
                     "source_config": {"server": "s1", "campaign_id": CAMPAIGN_ID}}
        receipt = {"schema": SCHEMA, "selector": SELECTOR,
                   "campaign_id": CAMPAIGN_ID, "server": "s1",
                   "selector_protocol": deepcopy(PROTOCOL), "validation_inventory": pop,
                   "source_history": {"source_run_id": "source"},
                   "candidates": [candidate], "selected": candidate}
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        validate_receipt(receipt)
        receipt["selected"]["validation"]["RRE50"] = .01
        receipt["receipt_sha256"] = canonical_sha256(
            {k: v for k, v in receipt.items() if k != "receipt_sha256"})
        with self.assertRaisesRegex(ValueError, "aggregate"):
            validate_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
