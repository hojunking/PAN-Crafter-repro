"""Read-only official-trajectory audit. No inference, training or queue changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_log/assets/2026-10-28_qb-hqnr-volatility'
SOURCES={}
def read(path):
    raw=path.read_bytes();SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def run(directory):
    cfg=read(directory/'config.json')
    evs=[read(p) for p in sorted(directory.glob('evaluation_*.json'))]
    evs=[e for e in evs if 1000<=e['state_before']['completed_update']<=30000]
    if not evs or cfg['phase']!='stage2':return None
    ids=[s['scene_id'] for s in evs[0]['fr']['per_scene']]
    for e in evs:
        assert all(e['qualification'][k] for k in ('same_state_verified','metrics_complete','metrics_finite'))
        assert len(e['fr']['per_scene'])==20 and [s['scene_id'] for s in e['fr']['per_scene']]==ids
    points=[];transitions=[]
    for e in evs:
        points.append(dict(step=e['state_before']['completed_update'],**e['fr']['metrics'],ERGAS=e['rr']['metrics']['ERGAS']))
    shifts=np.array([[s['correction_dy_dx'] for s in e['fr']['per_scene']] for e in evs])
    for i in range(1,len(evs)):
        if points[i]['step']-points[i-1]['step']!=1000:continue
        a,b=[np.array([[s['metrics'][k] for k in ('HQNR','D_s','D_lambda')] for s in e['fr']['per_scene']]) for e in evs[i-1:i+1]]
        delta=b-a;ds=-(1-(a[:,2]+b[:,2])/2)*delta[:,1];dl=-(1-(a[:,1]+b[:,1])/2)*delta[:,2]
        assert np.max(np.abs(delta[:,0]-ds-dl))<1e-12
        dh=float(delta[:,0].mean())
        transitions.append(dict(start_step=points[i-1]['step'],end_step=points[i]['step'],delta_HQNR=dh,
            delta_ERGAS=points[i]['ERGAS']-points[i-1]['ERGAS'],Ds_contribution=float(ds.mean()),Dlambda_contribution=float(dl.mean()),
            same_direction_scenes=int(np.sum(delta[:,0]*dh>0)),scene_delta_HQNR=delta[:,0].tolist(),
            Ds_scene_contributions=ds.tolist(),Dlambda_scene_contributions=dl.tolist()))
    stats={}
    for lo,hi in ((5000,15000),(15000,30000),(5000,30000)):
        subset=[t for t in transitions if lo<=t['start_step'] and t['end_step']<=hi]
        if subset:
            h=np.array([t['delta_HQNR'] for t in subset]);sd=sum(abs(t['Ds_contribution']) for t in subset);ld=sum(abs(t['Dlambda_contribution']) for t in subset)
            stats[f'{lo}-{hi}']=dict(n=len(h),mean_absolute_change=float(np.abs(h).mean()),std_change=float(h.std()),max_absolute_change=float(np.abs(h).max()),Ds_absolute_contribution_fraction=sd/(sd+ld),mean_same_direction_scenes=float(np.mean([t['same_direction_scenes'] for t in subset])))
    return dict(case=cfg['case'],replica=cfg['replica'],dataset=cfg['dataset'],run_id=cfg['run_id'],
        cfg={k:cfg.get(k) for k in ('lr_S','training_spec','planned_horizon','stage1_loss_parameters','shift_scale','stage2_spec','donor_version','seed')},
        points=points,transitions=transitions,stats=stats,
        frozen_aligner_hash_count=len({e['aligner_state_sha256'] for e in evs}),
        max_change_of_applied_shift_px=float(np.max(np.abs(shifts-shifts[0]))),
        evaluator_revision_count=len({e['state_before']['evaluator_revision'] for e in evs}),
        inference_policy_digest_count=len({e['state_before']['inference_policy_digest'] for e in evs}))

qbroot=ROOT/'work_dir/panda_s1loss9/PANDA_S13_S1LOSS9_RGB_L4_STAGE12_20261026_v1/s1/QB'
state=read(qbroot/'campaign_state.json')
qb=[run(qbroot/'runs'/r['run_id']/'attempt_01') for r in state['runs'] if r['phase']=='stage2']
wv=[]
for package in ('panda_tune30','panda_follow30'):
    for path in (ROOT/'work_dir'/package).glob('*/s1/WV3/campaign_state.json'):
        for r in read(path)['runs']:
            if r['status']=='COMPLETED' and r['case'] in ('S100DIFFUBILC30','S100DIFFUBILC25'):
                v=run(path.parent/'runs'/r['run_id']/'attempt_01');v['package']=package;wv.append(v)
prior=read(ROOT/'results_log/assets/2026-10-27_qb-loss9-internal-probe.json')
probes={name:dict(official=r['official'], modes={k:v['metrics'] for k,v in r['modes'].items()},
                 stem=r['stem'], deployed_reproduction_delta=r['deployed_reproduction_delta']) for name,r in prior['runs'].items()}
census=read(ROOT/'results_log/assets/2026-10-28_dataset-misalignment/summary.json')
geometry={sensor:{split:{ref:dict(total_rows=r['total_rows'],gradient_valid=r['gradient_valid'],quality_consensus=r['quality_consensus'])
    for ref,r in data.items() if ref in ('LMS_RGB','GT_RGB')} for split,data in census['summary'][sensor].items() if split in ('train','rr','fr')}
    for sensor in ('QB','WV3')}
report=dict(at=datetime.now(timezone.utc).isoformat(),scope='READ_ONLY_OFFICIAL_SAME_STATE_TRAJECTORIES',
    caveats=['WV3 comparison is contextual: different dataset, Stage1 reference and possible method revision; not a matched causal experiment.',
             'Finite evaluation records verify complete fixed scene sets; this audit does not rerun inference.',
             'Ds decomposition attributes metric variation, not uniquely geometric motion: UQI also responds to contrast and detail gain.'],
    qb=qb,wv3_context=wv,prior_input_ablation=probes,registration_census=geometry,source_sha256=SOURCES)
OUT.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,8))
for r in qb:
    p=r['points'];x=[v['step']/1000 for v in p];label=r['case'].replace('QB_','')
    for ax,k in zip(axes.flat,('HQNR','D_s','D_lambda','ERGAS')):ax.plot(x,[v[k] for v in p],label=label,linewidth=1.5);ax.set_ylabel(k);ax.set_xlabel('Student updates (K)');ax.grid(alpha=.2)
axes[0,0].axhline(.9,color='gray',linestyle='--',linewidth=1);axes[0,0].axhline(.9315,color='black',linestyle=':',linewidth=1)
axes[0,0].legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(OUT.with_suffix('.png'),dpi=180);fig.savefig(OUT.with_suffix('.pdf'));plt.close(fig)
print(json.dumps({'output':str(OUT), 'QB_runs':len(qb), 'WV3_context_runs':len(wv)}))
