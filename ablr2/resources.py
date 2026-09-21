"""Read-only resource checks; never stop other campaigns or infer GPU parity."""
from pathlib import Path
import os
import subprocess

from qg40.resources import inventory


def process_start(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,ValueError,IndexError,TypeError): return None


def idle_evidence():
    others = []
    markers = ('g20_runner.py', 'qg40_runner.py', 'l100_runner.py', 'fh20r1_runner.py',
               'fh20r1_train.py', 'fh12_runner.py', 'train.py', 'main.py --config')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid = int(path.parent.name)
            argv = [s.decode(errors='replace') for s in path.read_bytes().split(b'\0') if s]
            cmd = ' '.join(argv)
            own_worker = any(Path(s).name == 'ablr2_runner.py' and i+1 < len(argv)
                and argv[i+1] in ('train','calibrate','postrun','preflight') for i,s in enumerate(argv))
            if pid != os.getpid() and (own_worker or any(m in cmd for m in markers)):
                if ' -c ' not in cmd and not cmd.startswith(('rg ', 'grep ', 'bash -lc ')):
                    others.append(dict(pid=pid, command=cmd[:512]))
        except (OSError, ValueError): pass
    try:
        output = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True, timeout=5)
        pids = [int(s.strip()) for s in output.splitlines() if s.strip().isdigit()]
    except (OSError, subprocess.SubprocessError): pids = None
    return dict(idle=not others and pids == [], other_processes=others, gpu_pids=pids)


def assess_case(root, case):
    from ablr2.model import build_model
    from pa.aligner import PANGlobalAligner
    from ablr2.plan import GRID_STEPS, diagnostic_steps
    clone = case.role == 'S' and case.component['aligner'].startswith('CLONE_')
    model, _ = build_model(bands=case.num_bands, seed=case.seed, role=case.role,
        component=case.component, teacher_aligner_state=PANGlobalAligner(ms_bands=case.num_bands).state_dict() if clone else None)
    weights = sum(t.numel()*t.element_size() for t in model.state_dict().values())
    states = weights + 2*sum(p.numel()*p.element_size() for p in model.parameters())
    # No credit for deletion/compression. Retain all50 weights, diagnostic/full states,
    # paired RR/FR exports at selected checkpoints, atomic last, cache and metadata.
    estimate = (len(GRID_STEPS)*weights + (2*len(diagnostic_steps(case.role))+5)*states
        + 3*20*case.num_bands*(256**2+512**2)*4 + 512*1024**2)
    required = int(estimate*1.25) + 1024**3
    measured = inventory(root)
    reasons = []
    if measured['disk'].get('free_bytes',0) < required: reasons.append('INSUFFICIENT_DISK')
    if measured['gpu'].get('selected_device') is None: reasons.append('NO_VISIBLE_GPU')
    return dict(allowed=not reasons, reasons=reasons, required_disk_bytes=required,
                measurement=measured, gpu_profile_vram='UNMEASURED_UNTIL_REAL_TRAIN',
                prior_campaigns_stopped=False, automatic_pruning=False)
