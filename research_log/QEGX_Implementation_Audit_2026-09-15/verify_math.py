"""Independent QEGX math on real W104/T0 models, real H5 patches, fixed cue."""
import ast,copy,json,os,sys,tempfile,pathlib,types,random,time
from unittest.mock import patch
from audit_common import *
os.environ.update(ENV);os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch,numpy as np,yaml,h5py
torch.set_num_threads(2)
from tools import gen_pakd50_configs as G,qedge9_cue as Q
from kdv import edge_gate as EG
from kdv.edge_gate import EdgeGate
from kdv.registry import resolve
from kdv.teacher_assets import state_hash
from kdv.losses_rec import GTAnchoredReconstructionKD
from pa.losses import output_edge_loss_per_sample
from pa.model import PAModel
from train_kdv import KDVTrainer
from tools.smoke_cases import build,randomize_zero_params
R={'checks':[],'device':'cpu'}
def record(name,ok,kind='conformance',**kw):
    R['checks'].append(dict(name=name,passed=bool(ok),kind=kind,**kw));(OUT/'math_results.json').write_text(json.dumps(R,indent=2));print(name,ok,json.dumps(kw),flush=True)
man,z=EG.read_asset(G.QEDGE9_CUE_ASSET);arr={k:z[k].copy() for k in z.files}
record('original_cue_identity',man['asset_id']=='526dbd718af54c4a0c3eb5fd81778caa' and man['theta_q']==.3276133416220546 and man['npz_sha256']=='b83cc5d1a25e8766d1813c9d19d956efc24e0f152553e53c4894fa18dcdf563e')
T0,tman=Q.load_T0();hT=state_hash(T0)
record('T0_full_sha_margin_A',tman['file_sha256']=='16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32' and state_hash(T0.aligner)=='493aca181f93ae68' and T0.aligner_margin==4)
cfg=yaml.safe_load((ROOT/'config/PAKD50_QER50_W104_D121_WV3_T0_S1234_FRESH50_v1.yaml').read_text());cal=G.calibration();base=np.array([13,79,111,987]);contract,h5,h5p=Q.contract_from(cfg)
with h5py.File(h5) as f,h5py.File(h5p) as fp:rows=Q.read_rows(f,fp,base)
q,c,e=Q.compute_views(T0,4,rows,contract,2047.)
ix=(base[:,None]*4+np.arange(4)).reshape(-1)
record('actual_online_16_views',float(abs(q-arr['q'][ix]).max())<1e-4 and float(abs(e-arr['e_roi32'][ix]).max())<1e-4 and np.array_equal(q<man['theta_q'],arr['gate_low_q'][ix]),max_dq=float(abs(q-arr['q'][ix]).max()),max_de=float(abs(e-arr['e_roi32'][ix]).max()))
v={k:EG.apply_view(EG.to_feeder_tensor(a,2047.),True,True,0) for k,a in rows.items()};gt,ms,lp,pan=v['gt'],v['ms'],v['lpan'],v['pan'];meta=torch.tensor([[int(i),0,1,1] for i in base])
ns=dict(G=G,calQ=cal,KDVTrainer=KDVTrainer,resolve=resolve,types=types,T0=T0,GTAnchoredReconstructionKD=GTAnchoredReconstructionKD,torch=torch)
node=next(n for n in ast.parse((ROOT/'tools/pakd50_unit_tests.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_stubX');exec(compile(ast.Module(body=[node],type_ignores=[]),'fixture_constructor_only','exec'),ns);make=ns['_stubX']
torch.manual_seed(812);u=build(cfg);randomize_zero_params(u);S=PAModel(u,copy.deepcopy(T0.aligner).requires_grad_(True),aligner_margin=4,sampler=True)
record('student_actual_W104D121',round(sum(p.numel() for p in S.backbone.parameters())/1e6,4)==1.9036,backbone_params=sum(p.numel() for p in S.backbone.parameters()),note='Plan precision is 1.9036 M; initial audit hard-coded an incorrect exact count (1903584). The measured count is 1903624, consistent with the plan.')
gates={m:EdgeGate.load(dict(mode=m,asset=G.QEDGE9_CUE_ASSET,perm_seed=51515),teacher=T0) for m in ('low_q','shuffle')}
def manual_terms(case,tr,info):
    y,yt=info['y'],info['y_t'];es=(y-gt).abs().mean(1);et=(yt-gt).abs().mean(1).detach();d=(et/(et+cal['tau_R'])).detach();a=((es-et).clamp_min(0)/(es+1e-6)).detach();beta=.05 if case=='QE50_B005' else (0 if case in ('QX50','XJ') else .1)
    lh=((1+d)*es).mean();lk=(beta*(1-d)*a*(y-yt.detach()).abs().mean(1)).mean();Ei=output_edge_loss_per_sample(y.float(),gt.float())
    gate=tr.edge_gate or tr.edge_route
    weights=(gate.gate_for(meta,'cpu') if gate and gate.needs_meta else torch.full((4,),gate.c_E) if gate else torch.ones(4))
    eg=cal['lambda_E']*(weights*Ei).mean();eall=cal['lambda_E']*Ei.mean();lo=1e-4*info['loss_off']
    ua=lh+lk+(eall if tr.edge_route else eg);aa=lh+lk+eg+lo
    return ua,aa,weights,lh,lk
for case,mode in [('QE50','low_q'),('QX50','low_q'),('QE50_B005','low_q'),('QEC3','const'),('QES','shuffle'),('QER50','low_q'),('QERS','shuffle')]:
    eg=gates[mode] if mode!='const' else EdgeGate.synthetic('const',c_E=.37)
    tr=make(case,copy.deepcopy(S).train(),eg,server='s3' if case in ('QX50','QE50_B005','QEC3') else 's4',seed=2026 if case in ('QX50','QE50_B005','QEC3') else 1234)
    total,info=tr._step(gt,ms,lp,pan,1,meta=meta);wu,wa,w,lh,lk=manual_terms(case,tr,info);up=list(tr.M.backbone.parameters());ap=list(tr.M.aligner.parameters())
    gu=torch.autograd.grad(wu,up,retain_graph=True);ga=torch.autograd.grad(wa,ap,retain_graph=True)
    total.backward(retain_graph=bool(tr.edge_route));n=tr._apply_edge_route(info,tr.M) if tr.edge_route else 0
    du=max(float((p.grad-g).abs().max()) for p,g in zip(up,gu));da=max(float((p.grad-g).abs().max()) for p,g in zip(ap,ga));ref=max(float(g.abs().max()) for g in ga)
    record(case+'_actual_gradient',du<2e-6 and da<2e-6 and all(p.grad is None for p in T0.parameters()) and state_hash(T0)==hT,A_error=da,U_error=du,A_reference_max=ref,gate=w.tolist(),A_adjusted=n,soft=float(lk))

# Multiple optimizer steps across the LF boundary; a live AdamW moment must stop too.
tr=make('LFQE50',copy.deepcopy(S).train(),gates['low_q']);paramsA=list(tr.M.aligner.parameters());opt=torch.optim.AdamW([{'params':tr.M.backbone.parameters()},{'params':paramsA,'lr':1e-5}],lr=1e-4,weight_decay=.01)
steps=[]
for step in (24999,25000,25001):
    tr.M.aligner.requires_grad_(tr.aligner_active(step));before=state_hash(tr.M.aligner);total,info=tr._step(gt,ms,lp,pan,step,meta=meta);total.backward()
    if not tr.aligner_active(step):
        for p in paramsA:p.grad=None
    oldmoment=copy.deepcopy(opt.state_dict()['state']);opt.step();opt.zero_grad();after=state_hash(tr.M.aligner)
    steps.append(dict(step=step,active=tr.aligner_active(step),A_changed=before!=after,offset=float(info['loss_off']),U_edge=float(info['_edge_w_t'])))
record('LF_real_updates_across_boundary',steps[0]['A_changed'] and all(not x['A_changed'] and x['offset']==0 and x['U_edge']>0 for x in steps[1:]),steps=steps)

# A scaled backward must subtract the identically scaled correction.
tr=make('QER50',copy.deepcopy(S).train(),gates['low_q']);tr.accelerator.gradient_accumulation_steps=2;tr.accelerator.scaler=types.SimpleNamespace(get_scale=lambda:128.)
tot,info=tr._step(gt,ms,lp,pan,1,meta=meta);wu,wa,_,_,_=manual_terms('QER50',tr,info);up=list(tr.M.backbone.parameters());ap=list(tr.M.aligner.parameters());wantA=torch.autograd.grad(wa*64,ap,retain_graph=True);wantU=torch.autograd.grad(wu*64,up,retain_graph=True);(tot*64).backward(retain_graph=True);tr._apply_edge_route(info,tr.M)
record('route_scale_and_accumulation_algebra',max(float((p.grad-g).abs().max()) for p,g in zip(ap,wantA))<2e-5 and max(float((p.grad-g).abs().max()) for p,g in zip(up,wantU))<2e-5,note='CPU scaling algebra, scale=128/accum=2; this is not CUDA AMP execution.')

# Still missing: fixed campaign asset ID / median semantics, despite internal integrity improvements.
with tempfile.TemporaryDirectory(prefix='qegx-cue-contract-') as td:
    p=pathlib.Path(td);m=copy.deepcopy(man);a={k:v.copy() for k,v in arr.items()};m['theta_q']=.4;a['gate_low_q']=(a['q']<.4).astype(np.int8);st,_=EG.strata_of(a['e_roi32'],a['rot'],m['e_roi32']['decile_edges']);a['gate_shuffle']=EG.shuffle_gate(a['gate_low_q'],st,51515)[0]
    np.savez_compressed(p/'cue.npz',**a);m['npz']=str(p/'cue.npz');m['npz_sha256']=EG.sha256_file(p/'cue.npz');m['asset_id']=EG.asset_id(m);(p/'cue.json').write_text(json.dumps(m))
    loaded=EdgeGate.load(dict(mode='low_q',asset=str(p/'cue.json')),teacher=T0)
    record('nonmedian_replacement_asset_accepted',loaded.theta_q==.4 and loaded.manifest['asset_id']!=man['asset_id'],'defect_reproduction',actual_calibration_median=float(np.median(a['q'][a['calib_mask'].astype(bool)])),accepted_theta=loaded.theta_q,accepted_asset_id=m['asset_id'],note='Original asset untouched. All dependent label/shuffle/hash fields recomputed; no expected plan asset ID is pinned in config.')
    try:EdgeGate.load(dict(mode='low_q',asset=str(p/'cue.json')),teacher=T0,previous={'asset_id':man['asset_id']});rejected=False
    except ValueError:rejected=True
    record('previous_asset_id_mismatch_rejected',rejected)
    ce={'c_E':.58,'cue_npz_sha256':man['npz_sha256'],'cue_asset_id':man['asset_id'],'pilot_run':G.QEGX_PILOT_BY_SERVER['s3'],'pilot_tag':'last','pilot_step':50000,'pilot_file_sha256':'f'*64,'pilot_sha256_16':'0123456789abcdef','views':'calibration'};(p/'ce.json').write_text(json.dumps(ce))
    cf=dict(mode='const',asset=G.QEDGE9_CUE_ASSET,c_E_file=str(p/'ce.json'),pilot_run=ce['pilot_run'],pilot_tag='last',pilot_step=50000)
    loaded=EdgeGate.load(cf,teacher=T0);record('unverified_pilot_hash_accepted',loaded.c_E==.58,'defect_reproduction',note='Correct run/tag/step but arbitrary nonempty checkpoint hashes; current loader does not compare actual local pilot bytes.')
R['summary']={kind:dict(total=sum(x['kind']==kind for x in R['checks']),passed=sum(x['kind']==kind and x['passed'] for x in R['checks'])) for kind in ('conformance','defect_reproduction')};R['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');(OUT/'math_results.json').write_text(json.dumps(R,indent=2));print(R['summary'],flush=True)
