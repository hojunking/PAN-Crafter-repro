from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from directshift48.test_reporting import record
from sheet_publish_v2.test_selection import evidence
from .timing import timing_for_config


def prepare(root, *, steps=3):
    item = record(step=steps)
    config = deepcopy(item["source"]["extra"]["configuration"])
    config["evaluation_steps"] = [steps]
    item["source"]["extra"]["configuration"] = config
    evaluation = evidence(root, item)
    value = json.loads(evaluation.read_text()); value["elapsed_seconds"] = 7.2
    evaluation.write_text(json.dumps(value))
    path = evaluation.parent / "train.jsonl"
    path.write_text("".join(json.dumps({"step": i, "update": i, "seconds_including_IO": .1 * i}) + "\n" for i in range(1, steps + 1)))
    return config, path, evaluation


class TimingTests(unittest.TestCase):
    def run_timing(self, config, root, step=3):
        return timing_for_config(config, step, root, [], Path(root) / "cost")

    def test_exact_sum_and_eval_no_fake_wall(self):
        with tempfile.TemporaryDirectory() as root:
            config, path, _ = prepare(root)
            before = path.read_bytes()
            result = self.run_timing(config, root)
            self.assertAlmostEqual(result["sheet_fields"]["Train(h)"], .6 / 3600)
            self.assertEqual(result["sheet_fields"]["Eval(h)"], .002)
            self.assertNotIn("Wall(h)", result["sheet_fields"])
            self.assertEqual(path.read_bytes(), before)
            cached = self.run_timing(config, root)
            self.assertEqual(cached, result)

    def test_append_uses_prefix_and_partial_future_tail_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            config, path, _ = prepare(root)
            first = self.run_timing(config, root)
            with path.open("a") as stream:
                stream.write(json.dumps({"step": 4, "seconds_including_IO": .4}) + "\n" + '{"step":5')
            result = self.run_timing(config, root, 4)
            self.assertAlmostEqual(result["sheet_fields"]["Train(h)"], 1 / 3600)
            self.assertEqual(self.run_timing(config, root), first)
            partial = self.run_timing(config, root, 5)
            self.assertNotIn("Train(h)", partial["sheet_fields"])
            self.assertIn("PARTIAL_TAIL", partial["omitted"]["Train(h)"])

    def test_same_size_rewrite_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as root:
            config, path, _ = prepare(root)
            self.run_timing(config, root)
            original = path.read_text(); path.write_text(original.replace('0.2', '0.8'))
            result = self.run_timing(config, root)
            self.assertAlmostEqual(result["sheet_fields"]["Train(h)"], 1.2 / 3600)

    def test_bad_trace_values_or_gaps_never_estimated(self):
        cases = [{"step": 2, "seconds_including_IO": .1}, {"step": 1, "seconds_including_IO": -1},
                 {"step": 1, "seconds_including_IO": float("nan")}, {"step": 1, "seconds_including_IO": True}]
        for row in cases:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as root:
                config, path, _ = prepare(root); path.write_text(json.dumps(row) + "\n")
                result = self.run_timing(config, root)
                self.assertNotIn("Train(h)", result["sheet_fields"])
                self.assertIn("INVALID", result["omitted"]["Train(h)"])

    def test_missing_earlier_eval_omits_eval_not_training(self):
        with tempfile.TemporaryDirectory() as root:
            config, path, evaluation = prepare(root)
            config["evaluation_steps"] = [1, 3]
            (evaluation.parent / "config.json").write_text(json.dumps(config))
            result = self.run_timing(config, root)
            self.assertIn("Train(h)", result["sheet_fields"])
            self.assertNotIn("Eval(h)", result["sheet_fields"])
            self.assertIn("STEPS_MISSING", result["omitted"]["Eval(h)"])

    def test_corrupt_cache_rebuilds_and_config_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as root:
            config, _, evaluation = prepare(root)
            result = self.run_timing(config, root)
            cache = next((Path(root) / "cost/timing_cache").glob("*.json"))
            cache.write_text('{"broken":')
            self.assertEqual(self.run_timing(config, root), result)
            changed = {**config, "seed": 99}
            with self.assertRaisesRegex(ValueError, "CONFIG_FILE_MISMATCH"):
                self.run_timing(changed, root)


if __name__ == "__main__":
    unittest.main()
