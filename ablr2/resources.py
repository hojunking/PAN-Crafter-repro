"""Read-only resource checks; never stop other campaigns or infer GPU parity."""
from pathlib import Path
import os
import subprocess

from qg40.resources import inventory


def process_start(pid):
    try:
        fields=Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (OSError,ValueError,IndexError,TypeError): return None


def idle_evidence():
    others = []
    markers = ('g20_runner.py', 'qg40_runner.py', 'l100_runner.py', 'fh20r1_runner.py',
               'fh20r1_train.py', 'fh12_runner.py', 'gfb20_runner.py','gfp40_runner.py',
               'pcrepro_runner.py','train.py', 'main.py --config')
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


def projected_case_write(case):
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
    return int(estimate)


def assess_block(root,cases):
    """Before admitting a finite block; no pruning/credit for eventual deletion."""
    cases=tuple(cases)
    if not cases:raise ValueError('Cannot estimate an empty next block')
    from ablr2.plan import verify_lane
    for case in cases:verify_lane(case.server_id,case.sensor)
    if len({c.server_id for c in cases})!=1:raise ValueError('Resource admission must be local to one lane')
    # Structural duplicates share one CPU parameter-count measurement.
    measured_sizes={};writes=[]
    from ablr2.common import object_sha
    for case in cases:
        key=object_sha(dict(role=case.role,bands=case.num_bands,width=case.width,depth=list(case.depth),
                            component=case.component))
        if key not in measured_sizes:measured_sizes[key]=projected_case_write(case)
        writes.append(measured_sizes[key])
    estimate=sum(writes)
    required=max(100*1024**3,2*estimate,int(estimate*1.25)+1024**3)
    measured = inventory(root)
    reasons = []
    if measured['disk'].get('free_bytes',0) < required: reasons.append('INSUFFICIENT_DISK')
    if measured['gpu'].get('selected_device') is None: reasons.append('NO_VISIBLE_GPU')
    return dict(allowed=not reasons, reasons=reasons, required_disk_bytes=required,
                projected_next_block_write_bytes=estimate,case_count=len(cases),
                disk_policy='max(100GiB,2*next_block_projected_write,prior_stricter_requirement)',
                measurement=measured, gpu_profile_vram='UNMEASURED_UNTIL_REAL_TRAIN',
                prior_campaigns_stopped=False, automatic_pruning=False)


def assess_case(root,case):
    return assess_block(root,(case,))
