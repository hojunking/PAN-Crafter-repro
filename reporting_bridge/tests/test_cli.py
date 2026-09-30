"""Synthetic CLI safety fixtures: never contacts a live spreadsheet."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rb_bridge_cli', ROOT / 'tools/rb_b01_sheets.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class CLITests(unittest.TestCase):
    def test_import_has_no_training_or_network_initialization(self):
        code = ('import runpy,sys; runpy.run_path("tools/rb_b01_sheets.py",run_name="import_test"); '
                'assert not any(x in sys.modules for x in '
                '["torch","gspread","panda_rb.training","panda_rb.evaluation"])')
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)

    def test_protected_output_directories(self):
        for relative in ('.git/inject.json', 'work_dir/_panda_rb/new.json',
                         'data/example.json', 'datasets/abc.json'):
            with self.assertRaises(ValueError):
                cli.safe_output(ROOT / relative)

    def test_alternate_source_root_cannot_be_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with self.assertRaises(ValueError):
                cli.safe_output(source / 'work_dir/_panda_rb/old.json', source)
            self.assertEqual(cli.safe_output(source / 'work_dir/_rb_sheet_upload/new.json', source),
                             source / 'work_dir/_rb_sheet_upload/new.json')

    def test_prepared_integrity_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {'schema': 'RB_B01_PREPARED_v1', 'native_records': []}
            payload['payload_sha256'] = cli.digest(payload)
            path = Path(directory) / 'prepared.json'
            cli.write(path, payload)
            self.assertEqual(payload, cli.prepared_at(Path(directory)))
            payload['native_records'].append({'synthetic': True})
            cli.write(path, payload)
            with self.assertRaises(ValueError):
                cli.prepared_at(path)

    def test_readonly_commands_reject_apply(self):
        for command in ('collect', 'prepare', 'inspect', 'plan', 'verify'):
            with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
                cli.main([command, '--apply'])
            self.assertEqual(caught.exception.code, 2)

    def test_json_keeps_numeric_precision_boolean_and_literal_text(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {'number': 0.9506087266505915, 'bool': False,
                       'text': '=not_a_formula', 'zero': 0, 'null': None}
            path = Path(directory) / 'fixture.json'
            cli.write(path, payload)
            self.assertEqual(json.loads(path.read_text()), payload)


if __name__ == '__main__':
    unittest.main()
