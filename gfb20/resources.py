"""Never kill old campaigns or infer admission from nominal GPU memory."""
import os
from pathlib import Path
import subprocess
from qg40.resources import inventory,_model_footprint
from gfb20.plan import grid_steps,diagnostic_steps


def idle_evidence():
    processes=[]
    markers=('g20_runner.py','qg40_runner.py','l100_runner.py','ablr2_runner.py','fh12_runner.py',
             'fh20r1_runner.py','train.py','main.py --config')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid=int(path.parent.name)
            argv=[v.decode(errors='replace') for v in path.read_bytes().split(b'\0') if v]
            command=' '.join(argv)
            own=any(Path(v).name=='gfb20_runner.py' and i+1<len(argv) and argv[i+1] in
                    ('preflight','train','mixed','postrun') for i,v in enumerate(argv))
            if pid!=os.getpid() and (own or any(m in command for m in markers)):
                if ' -c ' not in command and not command.startswith(('rg ','grep ','bash -lc ')):
                    processes.append(dict(pid=pid,command=command[:512]))
        except (OSError,ValueError):pass
    try:
        text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=5)
        pids=[int(s) for s in text.splitlines() if s.strip().isdigit()]
    except (OSError,subprocess.SubprocessError):pids=None
    return dict(idle=not processes and pids==[],other_processes=processes,gpu_pids=pids)


def assess_block(root,cases,*,needs_mixed=False):
    footprint=_model_footprint('S',104,(1,2,2))
    weights=footprint['state_bytes'];state=weights+2*footprint['parameter_bytes']
    estimate=sum(len(grid_steps(c.updates))*weights+(len(diagnostic_steps(c.updates))+4)*(weights+state)
        +3*20*4*(256**2+512**2)*4+512*1024**2 for c in cases)
    if needs_mixed: estimate+=19809*3*(64**2+16**2)*4+4*3072*64**2*4+1024**3
    required=int(estimate*1.25)+1024**3
    measured=inventory(root);gpu=measured['gpu'].get('selected_device')
    reasons=[]
    if measured['disk'].get('free_bytes',0)<required:reasons.append('INSUFFICIENT_DISK')
    if gpu is None or '5090' not in gpu.get('name',''):reasons.append('EXPECTED_5090_UNAVAILABLE')
    return dict(allowed=not reasons,reasons=reasons,required_disk_bytes=required,measurement=measured,
                automatic_pruning=False,prior_campaigns_stopped=False)
