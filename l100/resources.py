"""Read-only local resource checks; never terminate legacy experiments."""
import os
from pathlib import Path
import subprocess

from qg40.resources import inventory, _model_footprint
from l100.plan import fullstate_steps, grid_steps


def legacy_processes():
    result = []
    markers = ('g20_runner.py', 'qg40_runner.py', 'fh20r1_runner.py', 'fh12_runner.py',
               'train.py', 'main.py --config')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid = int(path.parent.name)
            argv = [value.decode(errors='replace') for value in path.read_bytes().split(b'\0') if value]
            command = ' '.join(argv)
            own_worker = any(Path(arg).name == 'l100_runner.py' and pos + 1 < len(argv)
                and argv[pos + 1] in ('train', 'calibrate', 'preflight', 'endpoint', 'postrun')
                for pos, arg in enumerate(argv))
            if pid != os.getpid() and (own_worker or any(marker in command for marker in markers)):
                # Ignore this implementation's own tests and diagnostic searches.
                if ' -c ' not in command and not command.startswith(('rg ', 'grep ', 'bash -lc ')):
                    result.append(dict(pid=pid, command=command[:512]))
        except (OSError, ValueError):
            continue
    return result


def idle_evidence():
    legacy = legacy_processes()
    try:
        value = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True, timeout=5)
        gpu_pids = [int(line.strip()) for line in value.splitlines() if line.strip().isdigit()]
    except (OSError, subprocess.SubprocessError):
        gpu_pids = None
    return dict(idle=not legacy and gpu_pids == [], legacy=legacy, gpu_pids=gpu_pids)


def assess_block(root, cases):
    measurement = inventory(root)
    total = 0
    for case in cases:
        model = _model_footprint(case.role, case.width, case.depth)
        weight, state = model['state_bytes'], model['state_bytes'] + 2 * model['parameter_bytes']
        # Include candidates, diagnostic copies/fullstate, atomic last duplication,
        # endpoint reference/cache duplication and diagnostic/metadata headroom.
        total += len(grid_steps(case.updates)) * weight
        # Full-state candidates and restart diagnostics are distinct copies.
        total += len(set(grid_steps(case.updates)) & set(fullstate_steps(case.role, case.updates))) * state
        total += len(fullstate_steps(case.role, case.updates)) * (weight + state)
        total += 2 * state + 2 * weight + 256 * 1024 ** 2
    required = int(total * 1.25) + 1024 ** 3
    disk = measurement['disk']
    gpu = measurement['gpu'].get('selected_device')
    reasons = []
    if disk.get('status') != 'MEASURED' or disk.get('free_bytes', 0) < required:
        reasons.append('INSUFFICIENT_OR_UNKNOWN_DISK')
    if gpu is None or '5090' not in gpu['name']:
        reasons.append('EXPECTED_5090_UNAVAILABLE')
    return dict(allowed=not reasons, reasons=reasons, required_disk_bytes=required,
        measurement=measurement, training_vram_requirement='UNMEASURED_UNTIL_REAL_TRAIN',
        nominal_vram_inference=False)
