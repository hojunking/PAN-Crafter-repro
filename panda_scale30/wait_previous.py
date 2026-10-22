"""Start the new campaign only after the existing ZERO30 controller succeeds.

This watcher never opens a Campaign, starts its budget, signals another process,
or creates a new training root while waiting. Only its own receipt is written.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

PREVIOUS_CAMPAIGN = 'PANDA_S13_ZERO_ROTCONS_HARDONLY_30H_20261021_v1'
SUCCESSFUL = frozenset(('FINISHED_REGISTERED_QUEUE', 'FINISHED_WITH_BUDGET_SKIPS',
                        'FINISHED_WITH_EXPLICIT_SKIPS'))


def prior_decision(container, campaign):
    """Fail closed on abnormal exit, incomplete state, or nonterminal run rows."""
    if container.get('Running'):
        return 'WAITING_PREVIOUS'
    if container.get('Status') != 'exited' or container.get('ExitCode') != 0:
        raise RuntimeError('Previous controller did not exit normally; it was not changed')
    if campaign.get('status') not in SUCCESSFUL:
        raise RuntimeError('Previous campaign has no successful terminal receipt; it was not changed')
    if any(row.get('status') != 'COMPLETED' for row in campaign.get('runs', [])):
        raise RuntimeError('Previous campaign retains an incomplete run; it was not changed')
    return 'READY_TO_CHECK_GPU'


def source_pin(root):
    # Same production manifest as the actual runtime. No CUDA is initialized.
    from .runtime import code_pin
    return code_pin()


def check_pin(root, expected):
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Source changed while queued; refusing automatic launch: ' + name)
    if source_pin(root) != expected:
        raise RuntimeError('Source inventory changed while queued; refusing automatic launch')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, choices=('s1', 's3'))
    parser.add_argument('--poll-seconds', type=float, default=30.)
    args = parser.parse_args(argv)
    if not 5 <= args.poll_seconds <= 60:
        raise ValueError('Poll interval must be between 5 and 60 seconds')
    root = Path(__file__).resolve().parents[1]
    if (root / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Local server marker mismatch')
    directory = root / 'work_dir/panda_scale30_launchers' / args.server
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / 'receipt.json'
    previous_path = (root / 'work_dir/panda_zero30' / PREVIOUS_CAMPAIGN /
                     args.server / 'WV3/campaign_state.json')
    previous_name = f'panda-zero30-hardonly-wv3-{args.server}-v1'
    with (directory / '.waiter.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected_pin = source_pin(root)
        receipt = {'server': args.server, 'pid': os.getpid(),
                   'previous_container': previous_name, 'previous_campaign_state': str(previous_path),
                   'budget_started_while_waiting': False, 'source_pin': expected_pin,
                   'registered_at': datetime.now(timezone.utc).isoformat()}

        def record(status, **detail):
            receipt.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **detail)
            atomic_json(receipt_path, receipt)
            print(json.dumps({k: v for k, v in receipt.items() if k != 'source_pin'}), flush=True)

        try:
            while True:
                raw = subprocess.check_output(['docker', 'inspect', previous_name], text=True, timeout=30)
                container = json.loads(raw)[0]['State']
                campaign = json.loads(previous_path.read_text())
                decision = prior_decision(container, campaign)
                if decision == 'WAITING_PREVIOUS':
                    record(decision, previous_status=campaign.get('status'),
                           previous_runs=[{key: row.get(key) for key in ('case','replica','step','status')}
                                          for row in campaign.get('runs', [])])
                    time.sleep(args.poll_seconds)
                    continue
                processes = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                    text=True, stderr=subprocess.STDOUT, timeout=20)
                if processes.strip():
                    record('WAITING_GPU_IDLE', active_compute=processes.strip())
                    time.sleep(args.poll_seconds)
                    continue
                check_pin(root, expected_pin)
                record('STARTING_COST_AND_PREFLIGHT', previous_status=campaign['status'])
                subprocess.run(['bash', 'scripts/start_panda_scale30_costed.sh', args.server],
                               cwd=root, check=True)
                name = f'panda-scale30-hardonly-wv3-{args.server}-v1'
                started = json.loads(subprocess.check_output(['docker','inspect',name], text=True, timeout=30))[0]
                if not started['State'].get('Running'):
                    raise RuntimeError('New controller did not remain running; inspect its log')
                record('LAUNCHED', new_container=name)
                return 0
        except BaseException as exc:
            record('FAILED_NO_PREVIOUS_WORK_CHANGED', error=repr(exc))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
