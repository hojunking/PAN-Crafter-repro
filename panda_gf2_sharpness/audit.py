"""Inventory only: distinguish a missing weight from an invalid existing weight."""
from panda_sensor_visuals.common import read,sha,checkpoint_evidence,write
from . import plan as P


def inspect(root=P.ROOT):
    rows=[]
    for case in P.cases():
        directory=root/case['original_directory'];row=dict(case=case,checkpoints=[])
        if not (directory/'config.json').exists():
            row.update(status='MISSING_SOURCE_CONFIG',replay_required=False)
            rows.append(row);continue
        cfg=read(directory/'config.json');P.validate_case(cfg,case)
        row['source_config_sha256']=sha(directory/'config.json')
        for step in P.INSPECT:
            ck=directory/'models'/f'step_{step:07d}.pt';ev=directory/f'evaluation_{step:07d}.json'
            value=dict(step=step,checkpoint=str(ck),evaluation_present=ev.exists())
            if ck.exists():
                _,_,_,side=checkpoint_evidence(directory,step)
                value.update(status='AVAILABLE_VERIFIED',sha256=side['artifact']['file_sha256'])
            else:
                value['status']='WEIGHT_MISSING' if ev.exists() else 'NO_ORIGINAL_EVALUATION'
            row['checkpoints'].append(value)
        missing=[r['step'] for r in row['checkpoints'] if r['step'] in P.EARLY and r['status']=='WEIGHT_MISSING']
        incomplete=[r['step'] for r in row['checkpoints'] if r['step'] in P.EARLY and not r['evaluation_present']]
        row.update(status='READY' if not incomplete else 'MISSING_EARLY_EVALUATION',
                   missing_early=missing,replay_required=bool(missing) and not incomplete)
        rows.append(row)
    return dict(plan=P.manifest(),rows=rows,all_sources_ready=all(r['status']=='READY' for r in rows))


def save(root=P.ROOT):
    value=inspect(root);write(root/P.OUTPUT/'inventory.json',value);return value
