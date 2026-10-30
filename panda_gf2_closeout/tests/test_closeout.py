import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_gf2_closeout import verification as V
from panda_gf2_closeout import __main__ as C


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, step=47000, psnr=45.084138095671655):
    directory = root / V.CAMPAIGN_PATH / 'runs' / V.RUN / 'attempt_01'
    path = directory / 'models' / f'step_{step:07d}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'fixture immutable model ' + str(step).encode())
    digest = V.sha(path)
    cfg = dict(run_id=V.RUN, campaign_id=V.CAMPAIGN, server='s3', dataset='GF2', attempt=1,
               seed=261018102, case='GF2P_C60LR200', replica='R2', calibration_artifacts={})
    state = dict(cfg, completed_update=step, model_state_sha256='modelstate')
    ev = dict(state_before=state, state_after=state, rr_state=state, fr_state=state,
              rr={'metrics': {'ERGAS': .5148830158510865, 'PSNR': psnr}},
              fr={'metrics': {'HQNR': .9646694883162039}},
              qualification=dict(same_state_verified=True, metrics_complete=True, metrics_finite=True))
    artifact = dict(identity=state, completed_update=step, model_state_sha256='modelstate',
                    checkpoint_path=str(path), file_sha256=digest, weights_saved=True,
                    checkpoint_verified=True, campaign_resume_complete=True)
    write(directory / 'config.json', cfg)
    write(path.with_suffix('.json'), dict(config=cfg, evaluation=ev, artifact=artifact))
    write(directory / f'evaluation_{step:07d}.json', ev)
    return directory, path, digest


class CloseoutTests(unittest.TestCase):
    def test_winner_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory, path, digest = fixture(root)
            with patch.object(V, 'WINNER_SHA', digest):
                self.assertEqual(V.verify_winner(root)['step'], 47000)
                path.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'SHA'):
                    V.verify_winner(root)

    def test_he_representative_is_not_three_target_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory, path, digest = fixture(root, psnr=45.075620)
            with patch.object(V, 'WINNER_SHA', digest):
                with self.assertRaisesRegex(ValueError, 'three-metric'):
                    V.verify_winner(root)

    def test_distinct_eval_model_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory, path, digest = fixture(root)
            ep = directory / 'evaluation_0047000.json'; ev = V.read(ep)
            ev['fr_state']['completed_update'] = 43000
            write(ep, ev)
            with patch.object(V, 'WINNER_SHA', digest):
                with self.assertRaisesRegex(ValueError, 'same-state'):
                    V.verify_winner(root)

    def test_archive_all_passes_leaves_originals_and_rep(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, p47, digest = fixture(root)
            _, p43, _ = fixture(root, step=43000, psnr=45.075620)
            _, p48, _ = fixture(root, step=48000, psnr=45.09)
            with patch.object(C, 'WINNER_SHA', digest):
                files, goals = C.protect_all_goals(root, [])
            self.assertEqual({m['step'] for m in goals}, {47000, 48000})
            self.assertTrue(p43.is_file())
            self.assertTrue(p47.is_file())
            self.assertTrue(p48.is_file())
            self.assertTrue(all(V.sha(root / r['archive_path']) == r['sha256'] for r in files))
            again = C.archive(root, [p47], files)
            self.assertEqual(files, again)

    def test_archive_refuses_changed_existing_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, path, _ = fixture(root)
            files = C.archive(root, [path]); path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'archived evidence changed'):
                C.archive(root, [path], files)

    def test_s1_never_signals_or_inspects_docker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'gspread').mkdir(); (root / 'gspread/server.txt').write_text('s1')
            with patch.object(C, 'inspect') as inspect, patch.object(C.os, 'kill') as kill:
                with self.assertRaisesRegex(ValueError, 's3 only'):
                    C.plan(root)
                inspect.assert_not_called(); kill.assert_not_called()

    def test_pid_reuse_never_signals(self):
        before = dict(pid=123, generation='100', argv=['python'], cwd='/repo')
        with patch.object(C, 'process_identity', return_value={**before, 'generation': '200'}), \
                patch.object(C.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'generation'):
                C.stop_processes([before])
            kill.assert_not_called()

    def test_foreign_trainer_rejected(self):
        receipt = dict(container='panda-gf2-psnr4508-s3-v1', container_id='id', image='image',
                       args=['python', '-m', 'panda_gf2_psnr', 'run'])
        value = dict(Id='foreign', Image='image', Config={'Cmd': receipt['args']}, Mounts=[], State={})
        with self.assertRaisesRegex(ValueError, 'identity'):
            V.authenticate_container(receipt, value, Path('/repo'))

    def test_apply_archives_and_stops_controller_before_term_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); events = []
            (root / 'gspread').mkdir(); (root / 'gspread/server.txt').write_text('s3')
            (root / V.LAUNCHER).mkdir(parents=True)
            admission = root / V.CAMPAIGN_PATH / 'admission.json'; write(admission, {'code': {}})
            intent = dict(closeout_code={}, predecessor={}, admission_sha256=V.sha(admission),
                          controller_processes=[{'pid': 123}], trainer={'container': 'exact-gf2'})
            with patch.object(C, 'plan', return_value=intent), \
                    patch.object(C, 'source_pin', return_value={}), \
                    patch.object(C, 'protect_winner', side_effect=lambda *a: events.append('archive') or []), \
                    patch.object(C, 'protect_all_goals', return_value=([], [])), \
                    patch.object(C, 'stop_processes', side_effect=lambda *a: events.append('controller_term')), \
                    patch.object(C, 'matching_processes', return_value=[]), \
                    patch.object(C, 'inspect', return_value={'Id': 'exact-id'}), \
                    patch.object(C, 'authenticate_container', return_value={'Running': True}), \
                    patch.object(C.subprocess, 'run', side_effect=lambda args, **kw: events.append(args)):
                result = C.apply_once(root)
            self.assertEqual(result['status'], 'WAIT_SAFE_TRAINER_EXIT')
            self.assertEqual(events, ['archive', 'controller_term', ['docker', 'kill', '--signal=TERM', 'exact-id']])
            self.assertFalse((root / V.CONTROL / 'retirement.json').exists())
            self.assertEqual(V.read(admission), {'code': {}})

    def test_retirement_gate_without_docker_and_archive_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, path, digest = fixture(root)
            files = C.archive(root, [path]); control = root / V.CONTROL
            write(control / 'intent.json', {'reviewed': True})
            write(root / V.CAMPAIGN_PATH / 'campaign_state.json', {'status': 'STOPPED_SAFE'})
            state_sha = V.sha(root / V.CAMPAIGN_PATH / 'campaign_state.json')
            write(control / 'sheet_projection.json', {'rows': []})
            sheet = dict(readback_verified=True, winner_sha256=digest, campaign_state_sha256=state_sha,
                         projection_sha256=V.sha(control / 'sheet_projection.json'))
            write(control / 'final_sheet_readback.json', sheet)
            receipt = dict(schema=V.SCHEMA, server='s3', campaign_id=V.CAMPAIGN, status='RETIRED_TARGET_MET',
                winner={'sha256': digest, 'step': 47000},
                docker_state=dict(Running=False, Status='exited', OOMKilled=False, ExitCode=1),
                archive=files, qualified_models=[{'path': str(path.relative_to(root)), 'sha256': digest, 'step': 47000}],
                intent_sha256=V.sha(control / 'intent.json'), campaign_state_sha256=state_sha,
                sheet_receipt_sha256=V.sha(control / 'final_sheet_readback.json'))
            write(control / 'retirement.json', receipt)
            with patch.object(V, 'WINNER_SHA', digest), patch.object(V, 'inspect') as inspect:
                self.assertEqual(V.verify_retirement(root, check_runtime=False), receipt)
                inspect.assert_not_called()
                (root / files[0]['archive_path']).write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'archive changed'):
                    V.verify_retirement(root, check_runtime=False)


if __name__ == '__main__':
    unittest.main()
