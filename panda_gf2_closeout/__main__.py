"""s3-only: preserve the verified 47K winner, safely stop GF2, publish retirement evidence."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from directshift48.reporting import atomic_json, utc_now
from .verification import (CAMPAIGN, CAMPAIGN_PATH, CONTROL, LAUNCHER, RUN, STEP,
                           WINNER_SHA, SCHEMA, authenticate_container, inspect,
                           owned, read, sha, verify_retirement, verify_winner)

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    p = Path('/proc') / str(pid)
    try:
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': pid, 'argv': (p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                'cwd': str((p / 'cwd').resolve()), 'generation': stat[19]}
    except FileNotFoundError:
        return None


def matching_processes(root, module, suffix):
    matches = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if p.stat().st_uid != os.getuid():
                continue
            value = process_identity(int(p.name))
            if value is None or '-m' not in value['argv']:
                continue
            argv = value['argv']; index = argv.index('-m')
            if argv[index + 1:index + 2] != [module]:
                continue
            if argv[index + 2:] != suffix or value['cwd'] != str(Path(root).resolve()):
                raise ValueError('Foreign or unexpected GF2 control process')
            matches.append(value)
        except (FileNotFoundError, ProcessLookupError, UnicodeError):
            continue
    if len(matches) > 1:
        raise ValueError('Multiple GF2 control processes')
    return matches


def stop_processes(values):
    for before in values:
        current = process_identity(before['pid'])
        if current is None:
            continue
        if current != before:
            raise ValueError('PID generation/command changed; no signal sent')
        os.kill(before['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        remaining = [v for v in values if process_identity(v['pid']) == v]
        if not remaining:
            return
        time.sleep(.2)
    raise ValueError('GF2 controller did not exit; trainer untouched, no SIGKILL')


def source_pin(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / 'panda_gf2_closeout').glob('*.py'))}


def verify_predecessors(root):
    """GF2 followup was admitted only after older GF2/WV3 queues completed."""
    from panda_gf2_psnr.prerequisites import evidence
    proof = evidence(root)
    modules = {'panda_sensor_focus.queue', 'panda_sensor_refine.queue',
               'panda_s1loss9.queue', 'panda_sensor5.queue', 'panda_hard36.queue'}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if p.stat().st_uid != os.getuid():
                continue
            value = process_identity(int(p.name))
            if value is None or value['cwd'] != str(root.resolve()):
                continue
            argv = value['argv']
            server = argv[argv.index('--server') + 1] if '--server' in argv else None
            module = argv[argv.index('-m') + 1] if '-m' in argv else None
            recovery = any(Path(a).name == 's1loss9_recovery.py' for a in argv) and 'watch' in argv
            if server == 's3' and (module in modules or recovery):
                raise ValueError('An older s3 queue watcher is still active; inspect its terminal state first')
        except (FileNotFoundError, ProcessLookupError, UnicodeError):
            continue
    return dict(proof, older_queue_watchers_absent=True)


def plan(root):
    if (root / 'gspread/server.txt').read_text().strip() != 's3':
        raise ValueError('GF2 closeout is s3 only; s1 training is never signalled')
    predecessor = verify_predecessors(root)
    winner = verify_winner(root)
    receipt = read(root / LAUNCHER / 'launch.json')
    state = authenticate_container(receipt, inspect(receipt['container']), root)
    admission = read(root / CAMPAIGN_PATH / 'admission.json')
    # Check the admitted source bytes, without installing its mutable registry adapter.
    for name, expected in admission['code'].items():
        if sha(owned(root, name)) != expected:
            raise ValueError('Admitted GF2 source changed: ' + name)
    controllers = matching_processes(root, 'panda_gf2_psnr.enqueue', ['--wait'])
    if controllers:
        launcher = read(root / LAUNCHER / 'launcher_source_pin.json')
        if launcher.get('pid') != controllers[0]['pid'] or launcher.get('code') != admission['code']:
            raise ValueError('GF2 controller receipt/source differs')
    return {'schema': 'GF2_CLOSEOUT_INTENT_V1', 'at': utc_now(), 'server': 's3',
            'campaign_id': CAMPAIGN, 'policy': 'USER_20261030_GF2_TARGET_MET_SWITCH_S3_TO_QB',
            'winner': winner, 'trainer': receipt, 'controller_processes': controllers,
            'predecessor': predecessor,
            'docker_state_at_plan': state, 'closeout_code': source_pin(root),
            'admission_sha256': sha(root / CAMPAIGN_PATH / 'admission.json')}


def archive(root, sources, existing=()):
    rows = {r['source_path']: r for r in existing}
    for source in sources:
        source = owned(root, source)
        rel = source.relative_to(root)
        target = root / CONTROL / 'archive' / rel
        digest = sha(source)
        if str(rel) in rows:
            row = rows[str(rel)]
            if row['sha256'] != digest or sha(root / row['archive_path']) != digest:
                raise ValueError('Previously archived evidence changed: ' + str(rel))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != digest:
                raise ValueError('Archive collision')
        else:
            temporary = target.with_name(target.name + '.copying')
            if temporary.exists():
                temporary.unlink()  # Only this utility's own incomplete copy, never training files.
            shutil.copyfile(source, temporary)
            if sha(temporary) != digest or sha(source) != digest:
                raise ValueError('Archive copy/source changed')
            os.replace(temporary, target)
        rows[str(rel)] = {'source_path': str(rel), 'archive_path': str(target.relative_to(root)),
                          'sha256': digest, 'bytes': target.stat().st_size}
    return list(rows.values())


def protect_winner(root, intent):
    w = intent['winner']; path = root / w['path']; directory = path.parent.parent
    cfg = w['config']; campaign = root / CAMPAIGN_PATH; state = read(campaign / 'campaign_state.json')
    selected = state['donors'][cfg['donor_version']]
    donor = owned(root, cfg['donor_checkpoint'])
    if sha(donor) != cfg['donor_checkpoint_sha256']:
        raise ValueError('Frozen donor file changed')
    sources = [path, path.with_suffix('.json'), root / w['evaluation_path'], directory / 'config.json',
               directory / 'requested_config.json', donor, donor.with_suffix('.json'),
               owned(root, selected['selector_receipt_path']), campaign / 'admission.json']
    for item in cfg['calibration_artifacts'].values():
        p = owned(root, item['path'])
        if sha(p) != item['file_sha256']:
            raise ValueError('Calibration file changed')
        sources.append(p)
    # Keep donor source evaluation/config if present, plus campaign donor pointer files.
    donor_dir = donor.parent.parent
    sources += [donor_dir / 'config.json', donor_dir / f"evaluation_{cfg['donor_binding']['donor_step']:07d}.json"]
    sources += sorted((campaign / 'donors').glob('*.json'))
    inventory_path = root / CONTROL / 'archive_manifest.json'
    old = read(inventory_path)['files'] if inventory_path.exists() else []
    files = archive(root, sources, old)
    atomic_json(inventory_path, {'at': utc_now(), 'winner_sha256': WINNER_SHA, 'files': files})
    return files


def protect_all_goals(root, files):
    """Preserve every locally observed three-target pass, including nonrepresentatives."""
    from panda_gf2_psnr.registry import RUNS
    import math
    models, sources = [], []
    for ep in sorted((root / CAMPAIGN_PATH / 'runs').glob('*/attempt_01/evaluation_*.json')):
        ev = read(ep)
        metrics = {'HQNR': ev['fr']['metrics']['HQNR'], 'ERGAS': ev['rr']['metrics']['ERGAS'],
                   'PSNR': ev['rr']['metrics']['PSNR']}
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values())
                or metrics['HQNR'] < .9645 or metrics['ERGAS'] > .522 or metrics['PSNR'] < 45.08):
            continue
        cfg = read(ep.parent / 'config.json'); before = ev['state_before']; step = before['completed_update']
        path = ep.parent / 'models' / f'step_{step:07d}.pt'
        meta = read(path.with_suffix('.json')); a = meta['artifact']
        if (cfg['campaign_id'] != CAMPAIGN or cfg['server'] != 's3' or cfg['dataset'] != 'GF2'
                or (cfg['case'], cfg['replica']) not in RUNS or cfg['run_id'] != ep.parent.parent.name
                or any(before.get(k) != cfg[k] for k in ('campaign_id', 'server', 'dataset', 'run_id', 'attempt', 'seed'))
                or ep.name != f'evaluation_{step:07d}.json' or meta['config'] != cfg or meta['evaluation'] != ev
                or before != ev['state_after'] or before != ev['rr_state'] or before != ev['fr_state']
                or a.get('identity') != before or a.get('model_state_sha256') != before['model_state_sha256']
                or a.get('completed_update') != step or owned(root, a['checkpoint_path']) != path
                or any(a.get(k) is not True for k in ('weights_saved', 'checkpoint_verified', 'campaign_resume_complete'))
                or any(ev.get('qualification', {}).get(k) is not True for k in
                       ('same_state_verified', 'metrics_complete', 'metrics_finite'))
                or sha(path) != a['file_sha256']):
            raise ValueError('Target-qualified GF2 checkpoint is missing or unverified')
        models.append({'run_id': cfg['run_id'], 'step': step, 'path': str(path.relative_to(root)),
                       'sha256': a['file_sha256'], 'metrics': metrics})
        sources += [path, path.with_suffix('.json'), ep, ep.parent / 'config.json']
        # Other winning widths/weights still have their own exact calibration binding.
        for item in cfg['calibration_artifacts'].values():
            p = owned(root, item['path'])
            if sha(p) != item['file_sha256']:
                raise ValueError('Additional winner calibration changed')
            sources.append(p)
    if not any(m['sha256'] == WINNER_SHA for m in models):
        raise ValueError('Known 47K winner disappeared from all-pass inventory')
    return archive(root, sources, files), models


def stopped_inventory(root):
    from panda_gf2_psnr.registry import RUNS, cases
    from panda_sensor_refine.orchestration import authenticate_stop, final_evidence
    import torch
    campaign = root / CAMPAIGN_PATH; state = read(campaign / 'campaign_state.json')
    if state.get('status') not in ('STOPPED_SAFE', 'FINISHED_REGISTERED_QUEUE'):
        raise ValueError('GF2 campaign is not safely stopped/completed')
    seen = set(); rows = []; paths = []
    for row in state['runs']:
        pair = (row['case'], row['replica'])
        if pair not in RUNS or pair in seen:
            raise ValueError('Unknown/duplicate GF2 run')
        seen.add(pair); directory = campaign / 'runs' / row['run_id'] / 'attempt_01'
        cfg = read(directory / 'config.json')
        if (cfg['campaign_id'] != CAMPAIGN or cfg['server'] != 's3' or cfg['dataset'] != 'GF2'
                or cfg['run_id'] != row['run_id'] or (cfg['case'], cfg['replica']) != pair):
            raise ValueError('Stopped run identity differs')
        if row['status'] == 'COMPLETED':
            if row['step'] != cases()[row['case']]['total_updates']:
                raise ValueError('Completed run lacks full horizon')
            proof = final_evidence(campaign, cfg)
            rows.append({'run_id': row['run_id'], 'case': pair[0], 'replica': pair[1],
                         'status': 'COMPLETED', 'step': row['step'], 'final_evidence': proof})
        elif row['status'] == 'STOPPED':
            proof = authenticate_stop(row, cfg, root)
            checkpoint = root / proof['checkpoint']
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            trainer = payload['trainer']
            if (trainer['updates'] != row['step'] or payload['config'] != cfg
                    or not trainer.get('optimizer', {}).get('state')
                    or not trainer.get('scheduler') or not payload.get('stream') or not payload.get('rng')):
                raise ValueError('Stopped checkpoint lacks optimizer/scheduler/data/RNG')
            paths += [checkpoint, checkpoint.with_suffix('.json'), directory / 'models/latest_resume.json',
                      directory / 'run_state.json', directory / 'config.json']
            rows.append({'run_id': row['run_id'], 'case': pair[0], 'replica': pair[1],
                         'status': 'STOPPED_SAFE_BY_USER_AFTER_TARGET', 'step': row['step'],
                         'resume': proof})
        else:
            raise ValueError('Failed/running GF2 run requires investigation, not retirement')
    rows += [{'case': c, 'replica': r, 'status': 'EXCLUDED_UNSTARTED_BY_USER_AFTER_TARGET', 'step': 0}
             for c, r in RUNS if (c, r) not in seen]
    return state, rows, paths


def apply_once(root):
    control = root / CONTROL; control.mkdir(parents=True, exist_ok=True)
    with (control / 'closeout.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (control / 'retirement.json').exists():
            return verify_retirement(root)
        intent_path = control / 'intent.json'
        intent = read(intent_path) if intent_path.exists() else plan(root)
        if intent['closeout_code'] != source_pin(root):
            raise ValueError('Closeout source changed since intent; inspect migration')
        if (root / 'gspread/server.txt').read_text().strip() != 's3':
            raise ValueError('s3 only')
        if verify_predecessors(root) != intent['predecessor']:
            raise ValueError('An older GF2/WV3 queue changed after the closeout intent')
        admission_path = root / CAMPAIGN_PATH / 'admission.json'
        if sha(admission_path) != intent['admission_sha256']:
            raise ValueError('GF2 admission changed after closeout intent')
        for name, expected in read(admission_path)['code'].items():
            if sha(owned(root, name)) != expected:
                raise ValueError('Admitted GF2 source changed before safe closeout')
        if not intent_path.exists():
            atomic_json(intent_path, intent)
        files = protect_winner(root, intent)  # Archive before stopping any process.
        files, qualified = protect_all_goals(root, files)
        atomic_json(control / 'archive_manifest.json', {'at': utc_now(), 'winner_sha256': WINNER_SHA, 'files': files})
        held = control / 'controller_held.json'
        if not held.exists():
            stop_processes(intent['controller_processes'])
            atomic_json(held, {'at': utc_now(), 'processes': intent['controller_processes'],
                               'intent_sha256': sha(intent_path)})
        if matching_processes(root, 'panda_gf2_psnr.enqueue', ['--wait']):
            raise ValueError('GF2 controller reappeared; trainer untouched')
        with (root / LAUNCHER / 'controller.lock').open('a+') as controller_lock:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            current = inspect(intent['trainer']['container'])
            dstate = authenticate_container(intent['trainer'], current, root)
            if dstate['Running']:
                signal_path = control / 'trainer_term.json'
                history = read(signal_path) if signal_path.exists() else {'attempts': []}
                history['attempts'].append({'at': utc_now(), 'container_id': current['Id'],
                                           'signal': 'SIGTERM', 'intent_sha256': sha(intent_path)})
                atomic_json(signal_path, history)
                # Re-sending TERM is safe with the trainer's flag handler; this also
                # recovers a crash between durable intent and delivery. Never KILL.
                subprocess.run(['docker', 'kill', '--signal=TERM', current['Id']], check=True)
                return {'status': 'WAIT_SAFE_TRAINER_EXIT', 'retry_seconds': 15}
            if dstate.get('Status') != 'exited' or dstate.get('ExitCode') not in (0, 1):
                raise ValueError('GF2 trainer exited abnormally')
            state, rows, stopped_paths = stopped_inventory(root)
            files, qualified = protect_all_goals(root, files)  # Include passes produced during safe exit.
            expected_exit = 0 if state['status'] == 'FINISHED_REGISTERED_QUEUE' else 1
            if dstate.get('ExitCode') != expected_exit:
                raise ValueError('Docker exit and GF2 terminal state disagree')
            publisher_args = ['publish', '--apply', '--watch']
            publishers = matching_processes(root, 'panda_gf2_psnr', publisher_args)
            atomic_json(control / 'publisher_hold_intent.json', {'at': utc_now(), 'processes': publishers})
            stop_processes(publishers)
            state_sha = sha(root / CAMPAIGN_PATH / 'campaign_state.json')
            pubdir = root / CAMPAIGN_PATH / 'sheet_publish_panda_sensor_focus'
            sheet_path = control / 'final_sheet_readback.json'
            if not sheet_path.exists():
                subprocess.run([sys.executable, '-m', 'panda_gf2_psnr', 'publish', '--apply'],
                               cwd=root, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), check=True)
                report = read(pubdir / 'last_poll.json'); projection = read(pubdir / 'projection.json')
                from panda_sensor_focus.reporting import INDEX
                published = [r[INDEX['Checkpoint_SHA']] for r in projection['rows']]
                if (report.get('readback_verified') is not True or projection.get('conflicts')
                        or any(published.count(m['sha256']) != 1 for m in qualified)
                        or sha(root / CAMPAIGN_PATH / 'campaign_state.json') != state_sha):
                    raise ValueError('Final stable Sheet readback or target pass row missing')
                atomic_json(control / 'sheet_projection.json', projection)
                final_sheet = {'at': utc_now(), 'readback_verified': True, 'winner_sha256': WINNER_SHA,
                               'campaign_state_sha256': state_sha, 'report': report,
                               'projection_sha256': sha(control / 'sheet_projection.json')}
                atomic_json(sheet_path, final_sheet)
            else:
                final_sheet = read(sheet_path)
                if (final_sheet.get('readback_verified') is not True
                        or final_sheet.get('campaign_state_sha256') != state_sha
                        or final_sheet.get('projection_sha256') != sha(control / 'sheet_projection.json')):
                    raise ValueError('Existing final Sheet evidence changed')
            files = archive(root, stopped_paths + [root / CAMPAIGN_PATH / 'campaign_state.json',
                            sheet_path, control / 'sheet_projection.json'], files)
            atomic_json(control / 'archive_manifest.json', {'at': utc_now(), 'winner_sha256': WINNER_SHA, 'files': files})
            receipt = {'schema': SCHEMA, 'at': utc_now(), 'server': 's3', 'campaign_id': CAMPAIGN,
                       'status': 'RETIRED_TARGET_MET', 'original_campaign_status': state['status'],
                       'winner': {k: v for k, v in intent['winner'].items() if k != 'config'},
                       'trainer': intent['trainer'], 'docker_state': dstate, 'runs': rows,
                       'qualified_models': qualified,
                       'campaign_state_sha256': state_sha, 'archive': files,
                       'intent_sha256': sha(intent_path),
                       'sheet_receipt_sha256': sha(control / 'final_sheet_readback.json'),
                       'remaining_runs_are_excluded_not_completed': True}
            atomic_json(control / 'retirement.json', receipt)
            return verify_retirement(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'apply', 'verify'))
    parser.add_argument('--wait', action='store_true', help='Repeat bounded safe-stop polls; never SIGKILL')
    args = parser.parse_args()
    if args.command == 'plan':
        print(json.dumps(plan(ROOT), indent=2)); return 0
    if args.command == 'verify':
        print(json.dumps(verify_retirement(ROOT), indent=2)); return 0
    while True:
        result = apply_once(ROOT)
        print(json.dumps(result), flush=True)
        if result.get('status') != 'WAIT_SAFE_TRAINER_EXIT':
            return 0
        if not args.wait:
            return 75
        time.sleep(15)


if __name__ == '__main__':
    raise SystemExit(main())
