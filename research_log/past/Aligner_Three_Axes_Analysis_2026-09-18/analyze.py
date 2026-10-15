"""Reaggregate existing local evidence only. No inference, training, or Sheet writes."""
import csv, datetime, hashlib, json, re, subprocess, sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path('/home/knuvi/Desktop/song/PAN-Crafter')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.qrecon24_version_audit import audit_run
MANIFEST = {}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()

def read(p, kind='json'):
    p = Path(p); data = p.read_bytes(); rel = str(p.relative_to(ROOT))
    MANIFEST[rel] = hashlib.sha256(data).hexdigest()
    dest = OUT/'inputs'/rel; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
    if kind == 'json': return json.loads(data)
    if kind == 'yaml': return yaml.safe_load(data)
    if kind == 'csv': return list(csv.DictReader(data.decode().splitlines()))
    return data.decode()

def write_csv(name, rows):
    with (OUT/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

rows=[]; validation=[]
for p in sorted((ROOT/'work_dir/_eval_phase/records').glob('*.json')):
    j=read(p); run=j['run']; wd=ROOT/'work_dir'/run
    cfg=read(wd/'meta/config.yaml','yaml'); kd=cfg['kdv']; profile=kd['qrc24']['profile'];seed=int(cfg['seed'])
    a=j['modes']['A_ON'];n=j['modes']['A_BYPASS_RAW'];ar,af=a['rr'],a['fr'];nr,nf=n['rr'],n['fr']
    dh=np.array(nf['per_scene_hqnr'])-np.array(af['per_scene_hqnr'])
    row=dict(run=run,profile=profile,seed=seed,step=j['checkpoint_step'],rA=kd['qrc24']['rA'],a_weight=kd['qrecon']['a_weight'],e_weight=kd['qrecon']['e_weight'],
             on_ergas=ar['ergas'],noa_ergas=nr['ergas'],delta_ergas=nr['ergas']-ar['ergas'],relative_ergas_pct=100*(nr['ergas']/ar['ergas']-1),
             on_hqnr=af['hqnr_raw'],noa_hqnr=nf['hqnr_raw'],delta_hqnr=nf['hqnr_raw']-af['hqnr_raw'],
             delta_dlambda=nf['d_lambda']-af['d_lambda'],delta_ds=nf['d_s']-af['d_s'],fr_scenes_noa_better=int((dh>0).sum()),fr_scenes_on_better=int((dh<0).sum()),
             on_joint=af['hqnr_raw']>=.9585 and ar['ergas']<2.040,noa_joint=nf['hqnr_raw']>=.9585 and nr['ergas']<2.040,checkpoint_sha256=j['checkpoint_sha256'])
    rows.append(row)
    ids={};check={}
    for mode in ('A_ON','A_BYPASS_RAW'):
        d=wd/'results/aligner_eval_v2'/j['checkpoint_sha256'][:16]/mode
        ids[mode]=read(d/'identity.json')
        for split in ('rr','fr'):
            metric=read(d/f'{split}_metrics.json')
            check[f'{mode}_{split}_n20']=metric.get('n')==20
            check[f'{mode}_{split}_record_matches_file']=metric==j['modes'][mode][split]
        check[f'{mode}_config_hash_matches']=ids[mode]['config_sha256']==sha(wd/'meta/config.yaml')
        check[f'{mode}_ckpt_hash_matches_record']=ids[mode]['checkpoint_sha256']==j['checkpoint_sha256']
        check[f'{mode}_calls_valid']=all(ids[mode].get(f'calls_{s}')==dict(aligner_calls=(20 if mode=='A_ON' else 0),warp_calls=(20 if mode=='A_ON' else 0)) for s in ('rr','fr'))
    common=('checkpoint_sha256','config_sha256','fr_h5_sha256','evaluator','device','source_selector')
    check['on_noa_common_identity_matches']=all(ids['A_ON'].get(k)==ids['A_BYPASS_RAW'].get(k) for k in common)
    ck=wd/j['source_selector']/'model.safetensors';check['current_checkpoint_matches']=ck.exists() and sha(ck)==j['checkpoint_sha256']
    va=audit_run(run,write=False);check['factor1_training_manifest']=va['verdict']=='single_definition_factor1'
    check['scene_hqnr_mean_matches']=all(abs(np.mean(v['fr']['per_scene_hqnr'])-v['fr']['hqnr_raw'])<1e-12 for v in (a,n))
    validation.append(dict(run=run,checks=check,passed=all(check.values()),limitations=['RR dataset hash not in original identity','precision/environment not fully recorded','no remote/source-server certification']))

write_csv('noa_pairs.csv',rows)
agg=dict(n_pairs=len(rows),n_ergas_worse=sum(r['delta_ergas']>0 for r in rows),n_hqnr_worse=sum(r['delta_hqnr']<0 for r in rows),
         on_joint=sum(r['on_joint'] for r in rows),noa_joint=sum(r['noa_joint'] for r in rows),on_h_pass=sum(r['on_hqnr']>=.9585 for r in rows),noa_h_pass=sum(r['noa_hqnr']>=.9585 for r in rows),
         unique_seeds=sorted({r['seed'] for r in rows}),all_partial_provenance_checks_passed=all(r['passed'] for r in validation))
for k in ('delta_ergas','delta_hqnr','relative_ergas_pct','delta_dlambda','delta_ds'):
    a=np.array([r[k] for r in rows]);agg[k]=dict(mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()))
agg['frozen_vs_trainable']={}
for name,g in [('frozen',[r for r in rows if r['profile']=='A_FREEZE']),('trainable',[r for r in rows if r['profile']!='A_FREEZE'])]:
    agg['frozen_vs_trainable'][name]=dict(n=len(g),mean_delta_ergas=float(np.mean([r['delta_ergas'] for r in g])),mean_delta_hqnr=float(np.mean([r['delta_hqnr'] for r in g])))
by={(r['profile'],r['seed']):r for r in rows};contrasts=[]
for left,right in [('G22','A_UNIF'),('G22','A_SHUF'),('G23','A03_UNIF'),('G23','A03_SHUF'),('G22','A_FREEZE'),('G23','A_FREEZE'),('G23','G22')]:
    for seed in sorted({r['seed'] for r in rows}):
        if (left,seed) not in by or (right,seed) not in by:continue
        a,b=by[(left,seed)],by[(right,seed)]
        contrasts.append(dict(left=left,right=right,seed=seed,left_step=a['step'],right_step=b['step'],delta_on_ergas=a['on_ergas']-b['on_ergas'],delta_on_hqnr=a['on_hqnr']-b['on_hqnr']))
write_csv('training_contrasts_selected.csv',contrasts)

sizes=[];scene_contrasts=[];size_paths=list((ROOT/'work_dir').glob('*/results/aligner_eval_v2/*/analysis/size_probe_*.json'))
for p in size_paths:
    j=read(p);rr=j['rows']
    for scene in sorted({r['scene'] for r in rr}):
        for size in sorted({r['size'] for r in rr}):
            group=[r for r in rr if r['scene']==scene and r['size']==size]; q=np.array([r['q_size'] for r in group]);dy=np.array([r['dy'] for r in group]);dx=np.array([r['dx'] for r in group])
            sizes.append(dict(run=j['run'],split=j['split'],scene=scene,size=size,n_crop=len(group),q_mean=float(q.mean()),q_median=float(np.median(q)),q_p90=float(np.percentile(q,90)),dy_median=float(np.median(dy)),dx_median=float(np.median(dx)),within_scene_dy_std=float(dy.std(ddof=1)) if len(dy)>1 else None,within_scene_dx_std=float(dx.std(ddof=1)) if len(dx)>1 else None))
        g=[r for r in sizes if r['run']==j['run'] and r['scene']==scene];c=next(r for r in g if r['size']==64);f=max(g,key=lambda r:r['size'])
        scene_contrasts.append(dict(scene=scene,full_minus_crop64_dy=f['dy_median']-c['dy_median'],full_minus_crop64_dx=f['dx_median']-c['dx_median'],full_minus_crop64_q_mean=f['q_mean']-c['q_mean']))
write_csv('qsize_by_scene.csv',sizes);write_csv('full_vs_crop_by_scene.csv',scene_contrasts)

constants=[]
for p in sorted((ROOT/'work_dir').glob('*/palsv18/best_hqnr_constant_control.json')):
    j=read(p);constants.append(dict(run=j['run'],seed=int(re.search(r'_S(\d+)_',j['run']).group(1)),n_scenes=j['n_scenes'],learned_minus_zero=j['learned_minus_zero'],learned_minus_const_mean=j['learned_minus_const_mean'],learned_minus_shuffle=j['learned_minus_shuffle']))
write_csv('constant_controls.csv',constants)
for p in (ROOT/'work_dir').glob('*/palsv18/best_hqnr_rr_sweep.json'):read(p)
official=read(ROOT/'work_dir/_palsv18_campaign/official_best_raw_results.csv','csv')
selection=[]
for row in official:
    if row['case'] not in ('CTRLP0','L000','L1E4','L3E5','L3E4'):continue
    d=read(ROOT/'work_dir'/row['run']/'checkpoint_metrics.csv','csv');d=sorted(d,key=lambda v:int(v['step']));maxstep=int(d[-1]['step'])
    h=[float(v['raw_original.hqnr']) for v in d]
    selection.append(dict(case=row['case'],seed=int(row['seed']),best=max(h),last=h[-1],plateau=float(np.mean([float(v['raw_original.hqnr']) for v in d if int(v['step'])>=maxstep*.8]))))
write_csv('historical_selection_summary.csv',selection)
bs={(r['case'],r['seed']):r for r in selection};dsel=[]
for case in ('L000','L3E5','L1E4','L3E4'):
    for stat in ('best','last','plateau'):
        vals=[bs[(case,s)][stat]-bs[('CTRLP0',s)][stat] for s in (1234,2025,7777)]
        dsel.append(dict(case=case,stat=stat,mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)),positive_seeds=sum(v>0 for v in vals)))
write_csv('historical_selection_deltas.csv',dsel)

# Standalone scientific plots: no training, no scene pooling as independent evidence.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
labels=[r['profile']+' / '+str(r['seed']) for r in rows];ys=np.arange(len(rows));colors=['#d76a33' if r['profile']=='A_FREEZE' else '#366e9a' for r in rows]
for ax,key,title in zip(axs,('delta_ergas','delta_hqnr'),('RR ERGAS: NOA - ON (positive = worse)','FR HQNR: NOA - ON (negative = worse)')):
    vals=[r[key] for r in rows];ax.hlines(ys,0,vals,color=colors,alpha=.65);ax.scatter(vals,ys,c=colors,s=35);ax.axvline(0,color='black',lw=.8);ax.set_yticks(ys,labels);ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2)
fig.suptitle('13 local s1 checkpoint pairs; selected checkpoints; 2 seeds\nOrange: frozen aligner. Descriptive comparisons, not 13 independent replications.',fontsize=11)
fig.savefig(OUT/'noa_pairs.png',dpi=180);fig.savefig(OUT/'noa_pairs.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
for sc in sorted({r['scene'] for r in sizes}):
    g=sorted([r for r in sizes if r['scene']==sc],key=lambda r:r['size']);ax.plot([r['size'] for r in g],[r['q_mean'] for r in g],'-o',label=f'RR scene {sc}')
ax.set_xticks([64,128,256],['64 crops (16/scene)','128 crops (4/scene)','256 full (1/scene)']);ax.set_ylabel('Mean q_size within scene (HR pixels)');ax.set_title('G23 / seed 1234: RR 3-scene pilot only');ax.grid(alpha=.2);ax.legend();fig.savefig(OUT/'qsize_scene_means.png',dpi=180);fig.savefig(OUT/'qsize_scene_means.pdf');plt.close(fig)

(OUT/'summary.json').write_text(json.dumps(dict(noa=agg,training_contrasts=contrasts,qsize_by_scene=sizes,qsize_full_contrasts=scene_contrasts,selection=dsel,constants=constants),indent=2,ensure_ascii=False))
(OUT/'noa_provenance_checks.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False))
(OUT/'input_manifest.json').write_text(json.dumps(dict(captured_at=datetime.datetime.now().isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sha256=MANIFEST),indent=2))
print(json.dumps(dict(noa=agg,qsize_scene=sizes,full_contrasts=scene_contrasts,selection=dsel),indent=2))
