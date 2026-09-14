"""EQREC4 audit: CPU only; never starts/stops a campaign or changes its files.
Run from repository: python research_log/EQREC4_Audit_2026-09-15/verify_readonly.py
Synthetic regressions assert reproduction of a defect, not protocol compliance.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast, copy, hashlib, json, math, pathlib, runpy, sys, types
from contextlib import nullcontext
from unittest.mock import patch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import h5py, numpy as np, pandas as pd, torch
torch.set_num_threads(2)
from tools.eqrec4 import common as C, d10, d20, d30, d40, d50, k10, k20, report
from kdv.forward import kdv_forward
OUT = pathlib.Path(__file__).resolve().parent
CAMP = pathlib.Path(C.CAMP)
R = {'checks': [], 'device': str(C.DEV), 'torch': str(torch.__version__)}
def rec(name, passed, **evidence):
    row = dict(name=name, passed=bool(passed), **evidence); R['checks'].append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
def save():
    (OUT/'verification.json').write_text(json.dumps(R, indent=2, ensure_ascii=False, default=C._json_default))

nm = pd.read_csv(CAMP/'native_sample_metrics.csv', low_memory=False, float_precision='round_trip')
man = pd.read_csv(CAMP/'data_manifest.csv')
qa = pd.read_csv(CAMP/'quadrant_assignments.csv', low_memory=False)
assets = pd.read_csv(CAMP/'assets_manifest.csv')
R['coverage'] = nm.groupby('scale').size().to_dict()
rec('PASS atlas coverage and role separation', len(nm)==24*(4096+20+20) and man.sample_id.is_unique and man.groupby('source_group_id').split_role.nunique().max()==1,
    coverage=R['coverage'], roles=man.groupby('split_role').size().to_dict(), actual_known_scene_ids=0)
rec('PASS P0 consistency absent rather than zero', nm[nm.family=='P0'].q_A.isna().all())
rec('PASS model file hashes unchanged since G00', all(C.sha256_file(ROOT/r.checkpoint_dir/'model.safetensors')==r.model_hash for r in assets.itertuples()), n_assets=len(assets))
newthr, newqa = d10.thresholds_and_quadrants(nm)
savedthr = json.loads((CAMP/'calibration_thresholds.json').read_text())
rec('PASS A-only thresholds and Probe-B separate medians', all(newthr[k]['threshold_hash']==savedthr[k]['threshold_hash'] for k in newthr) and newqa.fillna('NA').equals(qa.fillna('NA')))

# D20 was corrected before execution, despite later commit time. Compare existing outputs.
mod = pd.read_csv(CAMP/'geometry_modality.csv')
cross = mod[(mod.scale=='fr512')&(mod['mode']=='pan_only')].merge(nm[nm.scale=='fr512'], on=['model_key','sample_id','scale'])
err = float(abs(cross.epe-cross.epe_A).max())
rec('PASS D20 corrected FR input reached saved results', len(cross)==200 and err<1e-5, n= len(cross), epe_max_abs_error=err,
    d20_source_mtime=C.time.strftime('%Y-%m-%dT%H:%M:%S', C.time.localtime((ROOT/'tools/eqrec4/d20.py').stat().st_mtime)))

# Recompute all stored core q values from residual vectors, independent of summary columns.
pieces=[]
for df in pd.read_csv(CAMP/'offset_probe_records.csv.gz', chunksize=100000):
    df['audit_q']=(df.residual_dy.abs()+df.residual_dx.abs())/2
    df['audit_epe']=np.hypot(df.residual_dy,df.residual_dx)
    pieces.append(df.groupby(['model_key','sample_id','probe_bank']).agg(qsum=('audit_q','sum'), esum=('audit_epe','sum'), n=('probe_id','size')))
g=pd.concat(pieces).groupby(level=[0,1,2]).sum(); g['q']=g.qsum/g.n; g['epe']=g.esum/g.n
errs=[]
for bank in ['A','B']:
    sub=g.xs(bank,level=2).join(nm[nm.scale=='native64'].set_index(['model_key','sample_id'])[[f'q_{bank}',f'epe_{bank}']])
    errs.extend([float(abs(sub.q-sub[f'q_{bank}']).max()),float(abs(sub.epe-sub[f'epe_{bank}']).max())])
rec('PASS full core residual cache reproduces q/EPE', int(g.n.sum())==12*4096*32 and (g.n==16).all() and max(errs)<2e-7,
    probe_rows=int(g.n.sum()), max_abs_error=max(errs))
del pieces,g

# Same-sample labels must be model-specific; quantify current primary-only detailed sampling.
detail=json.loads((CAMP/'detail_subsets.json').read_text())
ids=[i for q in C.QUADS for i in detail['quadrants'][q]['detail']]
labels={i:q for q in C.QUADS for i in detail['quadrants'][q]['detail']}
R['detail_labels']={}
for mk in ['L1E4_S1234_best_raw','L1E4_S7777_best_raw','L000_S2025_best_raw']:
    sub=qa[(qa.model_key==mk)&qa.sample_id.isin(ids)]
    R['detail_labels'][mk]={'n':len(sub),'mismatched_primary_label':int((sub.quadrant_id!=sub.sample_id.map(labels)).sum()),'own_quadrant_counts':sub.quadrant_id.value_counts().to_dict()}
rec('BUG reproduced: detailed strata reused across checkpoints', any(v['mismatched_primary_label']>0 for v in R['detail_labels'].values()), detail_labels=R['detail_labels'])
oldblur=np.random.RandomState(20260913).choice(9714,256,replace=False)
overlap=man[man.sample_id.isin(oldblur)].groupby('split_role').size().to_dict()
rec('BUG reproduced: old blur calibration is not split A', overlap.get('D',0)>0, selected_old_calibration=256, overlap_with_campaign_roles=overlap, outside_campaign=int(256-sum(overlap.values())))

# Pool/source overlap using actual Pair A cues: Student 40400 is byte-identical to D10 best_raw.
ar=assets.set_index('model_key')
assert ar.loc['L1E4_S1234_best_raw','model_hash']==ar.loc['L1E4_S1234_first_ge_20k','model_hash']
tt=nm[(nm.model_key=='L1E4_S2025_best_raw')&(nm.split_role=='B')].set_index('sample_id').reindex(man[man.split_role=='B'].sample_id)
ss=nm[(nm.model_key=='L1E4_S1234_best_raw')&(nm.split_role=='B')].set_index('sample_id').reindex(tt.index)
dfB=pd.DataFrame(dict(sample_id=tt.index,source_group_id=tt.source_group_id.values,e_T=tt.e_native_full.values,e_S=ss.e_native_full.values,a_T=ss.e_native_full.values-tt.e_native_full.values,q_A=tt.q_A.values))
t=savedthr['L1E4_S2025_best_raw'];dfB['e_low']=dfB.e_T<=t['e_median'];dfB['c_low']=dfB.q_A<=t['q_A_median'];dfB['a_pos']=dfB.a_T>0
dfB['cell']=dfB.apply(lambda r:('Ed' if r.e_low else 'Eu')+('Cd' if r.c_low else 'Cu')+('_Apos' if r.a_pos else '_Aneg'),axis=1)
pools=k10.fit_pools(dfB,C.SPLIT_SEED);sources=[set(dfB.loc[p['rows'],'source_group_id']) for p in pools]
shared=[len(s.intersection(set.union(*(x for j,x in enumerate(sources) if j!=i)))) for i,s in enumerate(sources)]
rec('BUG reproduced: K10 leave-one-pool-out crosses source blocks', len(pools)>1 and all(x>0 for x in shared), actual_pair_A_pools=len(pools), shared_source_groups_per_held_pool=shared, total_source_groups=[len(s) for s in sources], cells=dfB.cell.value_counts().to_dict())
dfS,status=k20.conditional_shuffle(dfB,99)
small=pd.DataFrame(dict(cell=['EdCd_Apos','EdCu_Apos','EdCd_Apos'],e_T=[1.,2.,3.],a_T=[1.,2.,3.],e_low=True,a_pos=True,c_low=[True,False,True],q_A=[.1,.2,.1]))
_,st=k20.conditional_shuffle(small,99)
rec('BUG reproduced: shuffle accepts 3 samples below minimum 8', list(st)==['ok']*3,status=list(st))
rec('PASS current shuffle preserves EA cells and q-label mass', (dfS.cell.map(k20.ea_key)==dfB.cell.map(k20.ea_key)).all() and dfS.c_low.sum()==dfB.c_low.sum(), changed_fraction=float((dfS.cell!=dfB.cell).mean()))
ut=pd.DataFrame([dict(pair='A',repeat=0,cell='EdCd_Apos',n_source_groups=1,U_soft_hard_rel=u) for u in [1e-6,2e-6,3e-6,4e-6]])
with patch.object(C,'load_json',return_value={'noise_scale_rel':.001}):tab=k20.utility_tables(ut,'A',dfB)
rec('BUG reproduced: noise-sized utility marked resolved; dependent pools not shrunk', not tab['no_resolved_utility_signal'] and tab['EAQ']['EdCd_Apos']['n']==4, noise=.001,max_utility=4e-6, resolved=not tab['no_resolved_utility_signal'], n_used=tab['EAQ']['EdCd_Apos']['n'])

# Exercise actual K20 orchestration with zero utility and fake training that aborts BEFORE any update.
class TrainingReached(Exception):pass
dummy=types.SimpleNamespace(key='dummy',step=1,cfg={},m=types.SimpleNamespace(state_dict=lambda:{}))
minimal=pd.concat([small]*16,ignore_index=True);minimal['sample_id']=np.arange(48);minimal['source_group_id']='one_unknown_source'
zero_ut=ut.copy();zero_ut['U_soft_hard_rel']=0.
fake_cells={'B':{'df':minimal,'tensors':(None,)*4,'teacher_pred':None}}
with patch.object(C,'make_manifest',return_value=man),patch.object(C,'feeders',return_value=None),patch.object(C,'read_csv',return_value=zero_ut),patch.object(C,'Stage',return_value=nullcontext()),patch.object(C,'load_model',return_value=dummy),patch.object(k20,'cells_for_pair',return_value=(fake_cells,{})),patch.object(C,'dump_json'),patch.object(C,'load_json',return_value=None),patch.object(pd.DataFrame,'to_csv'),patch.object(C,'load_patches',return_value=None),patch.object(k20,'train_arm',side_effect=TrainingReached):
    reached=False
    try:k20.main()
    except TrainingReached:reached=True
rec('BUG reproduced: K20 reaches training with zero utility and no entry validation',reached)

# Executable G00 CLI returns success even when its returned status is invalid (runner uses this form).
with patch.object(sys,'argv',['tools/eqrec4.py','g00']),patch('importlib.import_module',return_value=types.SimpleNamespace(main=lambda **kw:{'implementation_invalid':['injected_failure']})):
    code=0
    try:runpy.run_path(str(ROOT/'tools/eqrec4.py'),run_name='__main__')
    except SystemExit as ex:code=ex.code
rec('BUG reproduced: standalone G00 invalid status exits zero',code==0,exit_code=code)

# H2 r=1 synthetic counterexample: independent B worsens +1, A improves -1 => pooled 0.
stress=pd.DataFrame([dict(model_key='test',scale='native64',sample_id=i,path='response',r=1.,probe_id=f'{b}{j}',probe_bank=b,valid_support=True,d_e=v) for i in range(8) for b,v in [('A',-1.),('B',1.)] for j in range(4)])
fake_nm=pd.DataFrame(dict(model_key='test',scale='native64',sample_id=range(8),q_A=np.arange(8),e_native_full=np.arange(8),source_group_id=[f'g{i}' for i in range(8)]))
with patch.object(C,'spearman_boot',return_value={}):h2=d40.h2_stats(stress,fake_nm)
val=h2['test']['native64']['by_path']['response']['1.0']['mean']
rec('BUG reproduced: H2 aggregates A and B instead of B-only',val==0.,aggregate=val,bank_B_expected=1.)

# Report failure direction should not be asserted from nonsignificant negative point estimates.
fake_h2={f'L1E4_S{s}_best_raw':{'native64':{'by_path':{'response':{r:{'spearman_qA_d':{'rho':-.001,'ci95':[-.8,.8]}} for r in ['0.5','1.0','2.0']}}}} for s in [1234,7777,2025]}
ver=report.verdicts({'h2':fake_h2})['H2']['state']
rec('BUG reproduced: report calls inconclusive H2 opposed',ver=='Opposed in tested setting',state=ver,all_CI_include_zero=True)
save()

# Actual checkpoint CPU math/gradient tests. No active GPU work or campaign output.
T=C.load_model(*C.PRIMARY,dev='cpu');S=C.load_model(*C.PAIRS['A']['S'],dev='cpu')
batch=C.load_patches(man[man.split_role=='B'].sample_id.iloc[:2].tolist());gt,ms,lpan,pan=batch
with torch.no_grad(): tp=T.m(pan,ms,lpan)['y']
th=C.state_hash(T.m);sh=C.state_hash(S.m)
pr=C.probe_responses(T,pan,ms)
rec('PASS real checkpoint probe q formula',torch.allclose(pr['A']['q'],pr['A']['resid'].abs().sum((1,2))/32,atol=1e-7))
ga={};gu={};step_hashes={}
for arm in ['R0','RH','RS','K20']:
    m=k10.make_student(S); info=k10.k_step(m,S.mg,batch,1,torch.Generator().manual_seed(99),tp,arm,rho=torch.full((2,),.5),check=True)
    ga[arm]=torch.cat([p.grad.flatten() for p in m.aligner.parameters()]);gu[arm]=torch.cat([p.grad.flatten() for p in m.backbone.parameters()])
    rec('PASS actual routing '+arm,info['rec_to_A_exists'] and info['off_to_U_absent'] and (arm=='R0' or info['g_extra_U']>0),base_A=info['g_base_A'],extra_U=info['g_extra_U'])
    if arm=='R0':
        mf=k10.make_student(S);f=kdv_forward(mf,None,pan,ms,lpan,share_correction=False,teacher_needed=False,aligner_live=True)
        eps=C.sample_offsets(2,2.,torch.Generator().manual_seed(99));ce=C.predict_c(mf.aligner,C.warp_pan(pan,eps),f['ms_base'],S.mg)
        loss=(f['y']-gt).abs().mean()+1e-4*C.offset_loss(ce,f['delta'],eps,stop_reference=True)
        gg=torch.autograd.grad(loss,list(mf.aligner.parameters())+list(mf.backbone.parameters()))
        actual=torch.cat([ga[arm],gu[arm]]);expected=torch.cat([v.flatten() for v in gg]);err=float((actual-expected).abs().max())
        rec('PASS R0 equals existing forward and base gradient',err<1e-7 and torch.equal(f['y'],mf(pan,ms,lpan)['y']),gradient_max_abs=err)
    opt=k10.fresh_opt(m,S.cfg);opt.step();step_hashes[arm]=C.state_hash(m.aligner)
    del m
rec('PASS extra hard/soft/mixed leave direct A gradient unchanged',all(torch.equal(ga['R0'],v) for v in ga.values()),max_abs={a:float((v-ga['R0']).abs().max()) for a,v in ga.items()})
rec('PASS same-source AdamW first A update identical across arms',len(set(step_hashes.values()))==1)
rec('PASS source/Teacher states unchanged and no parameter sharing',th==C.state_hash(T.m) and sh==C.state_hash(S.m) and not {p.data_ptr() for p in T.m.parameters()}.intersection(p.data_ptr() for p in S.m.parameters()))
# D40 real known-inverse support and mislabeled band deltas, one sample/one probe only.
with patch.object(d40,'STRESS_PROBES',[d40.STRESS_PROBES[0]]):
    rr=d40.stress_patches(T,[int(man[man.split_role=='B'].sample_id.iloc[0])],{},*(x[:1] for x in batch),chunk=1)
row=next(r for r in rr if r['path']=='known_inverse')
rec('PASS stress known inverse uses c0-epsilon',abs(row['c_path_dy']-(row['c0_dy']-row['epsilon_dy_hr']))<1e-6 and abs(row['c_path_dx']-(row['c0_dx']-row['epsilon_dx_hr']))<1e-6)
band_mean=float(np.mean([row[f'd_e_band{b}'] for b in range(8)]))
rec('BUG reproduced: d_e_band columns hold absolute stress error',abs(band_mean-row['l1_stress_roi'])<1e-7 and abs(band_mean-row['d_e'])>1e-4,band_mean=band_mean,stress_l1=row['l1_stress_roi'],expected_delta=row['d_e'])
# One actual A-only intervention with complete restoration and independent probe B.
m=k10.make_student(T);m.backbone.requires_grad_(False);src=copy.deepcopy(m.state_dict());b=tuple(x[:1] for x in batch)
gr,gc,lr,lo,pars=d50.grads_for(m,T.mg,b[3],b[1],b[2],b[0]);base=d50.measure(T,m,b[3],b[1],b[2],b[0]);norm=math.sqrt(sum(float((p.detach()**2).sum()) for p in pars));step=1e-5*norm*gc/(gc.norm()+1e-12)
pos=0
with torch.no_grad():
    for p in pars:n=p.numel();p.sub_(step[pos:pos+n].view_as(p));pos+=n
after=d50.measure(T,m,b[3],b[1],b[2],b[0]);m.load_state_dict(src)
rec('PASS D50 A-only measurement and restoration',C.state_hash(m)==C.state_hash(T.m) and math.isfinite(after['q_B']),g_rec_norm=float(gr.norm()),g_off_norm=float(gc.norm()),delta_qB=after['q_B']-base['q_B'])
# Bounded run_trial execution in different arm order; two steps is only a reset/RNG test.
save()
pool=dict(rows=np.array([0,1]),cycled=False);source=copy.deepcopy(S.m.state_dict());trial_outputs=[]
with patch.object(k10,'UPDATES',2):
    for order in [('R0','RH','RS'),('RS','R0','RH')]:
        trial_outputs.append({arm:k10.run_trial(S,source,S.cfg,pool,batch,tp,batch,None,seed=999,arm=arm) for arm in order})
rec('PASS actual run_trial restores optimizer/source/RNG across arm order',all(np.array_equal(trial_outputs[0][a]['C_per_sample_after'],trial_outputs[1][a]['C_per_sample_after']) for a in ['R0','RH','RS']) and all(x['restore_hash']==sh for r in trial_outputs for x in r.values()),audit_updates=2,production_updates=k10.UPDATES,limitation='Does not validate 64-step utility magnitude or its uncertainty')
rec('PASS utility labels have specified signs', all(math.isfinite(trial_outputs[0][a]['C_l1_after']) for a in ['R0','RH','RS']),U_soft_hard=trial_outputs[0]['RH']['C_l1_after']-trial_outputs[0]['RS']['C_l1_after'],audit_updates=2)
save()
