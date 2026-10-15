"""Serialization recovery never edits original training provenance."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from panda_rb import evaluation
from panda_rb.common import sha256, object_sha
from tools import panda_rb_recover_scene_index as recovery


def original_bytes():
    # Permanent source fix is precisely the audited inverse one-line change.
    text = Path(evaluation.__file__).read_text()
    if recovery.NEW in text: text = text.replace(recovery.NEW, recovery.OLD)
    return text.encode()


class RecoveryTests(unittest.TestCase):
    def test_exact_patch_matches_permanent_function(self):
        node, identity = recovery.corrected_function(original_bytes())
        current = next(n for n in ast.parse(Path(evaluation.__file__).read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == '_native_selection')
        self.assertEqual(ast.dump(node), ast.dump(current))
        self.assertEqual(identity['original_evaluation_sha256'], recovery.ORIGINAL_EVALUATION_SHA)
        self.assertNotEqual(identity['original_evaluation_sha256'], identity['corrected_evaluation_sha256'])

    def test_unreviewed_code_change_rejected(self):
        for source in (original_bytes() + b'\n', original_bytes().replace(b'batch_size=1', b'batch_size=2')):
            with self.assertRaisesRegex(ValueError, 'exact audited'): recovery.corrected_function(source)

    def test_patch_tags_effective_evaluator_without_altering_base_checks_or_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / 'panda_rb/evaluation.py'
            path.parent.mkdir(); path.write_bytes(original_bytes())
            before = sha256(path)
            _node, descriptor = recovery.corrected_function(path.read_bytes())
            descriptor['entrypoint_sha256'] = sha256(recovery.__file__)
            calls = []
            def original_context(*args, **kwargs):
                calls.append((args, kwargs))
                return {'context': {'source_identity': {'content_sha256': 'original-training'},
                                    'evaluator_identity': {'content_sha256': 'base-eval', 'files': {'x': 'y'}}}}
            fake = SimpleNamespace(__file__=str(path), _context=original_context,
                                   evaluator_identity=lambda: {'content_sha256': 'base-eval'})
            recovery.install_patch(fake, root, {'patch': descriptor})
            result = fake._context('run', root=root)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['context']['source_identity']['content_sha256'], 'original-training')
            self.assertEqual(fake.evaluator_identity()['content_sha256'], 'base-eval')
            self.assertNotEqual(result['context']['evaluator_identity']['content_sha256'], 'base-eval')
            self.assertEqual(result['context']['evaluation_runtime_patch'], descriptor)
            self.assertEqual(result['context']['evaluator_identity']['serialization_hotfix'], descriptor)
            self.assertEqual(sha256(path), before)
            with self.assertRaisesRegex(ValueError, 'second runtime'):
                recovery.install_patch(fake, root, {'patch': descriptor})

    def test_receipt_cannot_substitute_another_function(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / 'panda_rb/evaluation.py'
            path.parent.mkdir(); path.write_bytes(original_bytes())
            fake = SimpleNamespace(__file__=str(path))
            with self.assertRaisesRegex(ValueError, 'receipt'):
                recovery.install_patch(fake, root, {'patch': {}})


if __name__ == '__main__': unittest.main()
