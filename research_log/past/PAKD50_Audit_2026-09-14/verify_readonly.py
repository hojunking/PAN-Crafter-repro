"""Read-only CPU audit. Does not construct a training runner or write campaign state.
Usage: python /tmp/pakd50_audit_20260914.py /absolute/repo /tmp/audit.json
"""
import ast, copy, csv, hashlib, io, json, os, pathlib, runpy, subprocess, sys, tempfile, types
from unittest.mock import patch
ROOT = pathlib.Path(sys.argv[1]).resolve()
OUT = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import h5py, numpy as np, torch, yaml
torch.set_num_threads(2)
from tools import gen_pakd50_configs as G
from main import import_class
from kdv.teacher_assets import load_run_model, freeze, state_hash, sha256_file
from kdv.registry import resolve
from kdv.losses_rec import GTAnchoredReconstructionKD
from kdv.forward import kdv_forward
from pa.losses import output_edge_loss
from pa.offset import offset_loss, predict_c, sample_offsets
from pa.warp import warp_pan
from train_kdv import KDVTrainer

result = {'environment': {'torch':str(torch.__version__), 'device':'cpu', 'threads':2}, 'checks':[]}
def record(name, passed, **details):
    result['checks'].append(dict(name=name, passed=bool(passed), **details))
    print(name, 'PASS' if passed else 'FAIL', json.dumps(details), flush=True)

cfg = yaml.safe_load((ROOT/'assets/pakd50/T0_run/meta/config.yaml').read_text())
Model = import_class(cfg['model'])
T, _ = load_run_model('assets/pakd50/T0_run', 'best_hqnr', Model)
freeze(T)
th = state_hash(T)
run = ROOT/'work_dir'/G.pilot_run()
cfgs = yaml.safe_load((run/'meta/config.yaml').read_text())
student_template = copy.deepcopy(T)
student_template.backbone.load_state_dict(torch.load(ROOT/'work_dir/_kdv_init_w112_d123/init_unet_seed1234.pt', map_location='cpu', weights_only=True))
# Nonzero head is needed to exercise aligner gradients; use a stable, already evaluated Student state.
from safetensors.torch import load_file
student_template.load_state_dict(load_file(str(run/'candidates/step-4040/model.safetensors')))
with h5py.File(cfgs['train_feeder_args']['dataroot']) as f:
    gt, ms, pan = [torch.from_numpy(f[k][:2].astype('float32'))/2047*2-1 for k in ('gt','ms','pan')]
with h5py.File(cfgs['train_feeder_args']['dataroot'].replace('.h5','_pan.h5')) as f:
    lpan = torch.from_numpy(f['lpan'][:2].astype('float32'))/2047*2-1
inputs = (gt,ms,lpan,pan)
input_copies = [x.clone() for x in inputs]
cal = {'tau_R':G.calibration()['tau_R'], 'lambda_E':0.3} # test coefficient only, never published as calibration

def make(case):
    tr = object.__new__(KDVTrainer)
    tr.k = G.kdv_block(case,1234,'s1',cal=cal); tr.spec = resolve(tr.k)
    sp = tr.spec; tr.model = copy.deepcopy(student_template).train().requires_grad_(True)
    # A-FR must begin with exactly T0 A for legitimate shared correction.
    if not sp['aligner_trainable']:
        tr.model.aligner.load_state_dict(T.aligner.state_dict()); freeze(tr.model.aligner)
    tr.accelerator = types.SimpleNamespace(unwrap_model=lambda m:m)
    tr.teacher=T; tr.aligner_trainable=sp['aligner_trainable']; tr.aligner_view_margin=4
    tr.share_correction=not tr.aligner_trainable
    tr.protocol=sp['protocol']; tr.radius_hr=sp['radius_hr']; tr.diag_every=1000
    tr.rec_crit=GTAnchoredReconstructionKD(cal['tau_R'],alpha=1,kd_weight=0.1 if case.endswith('Q') else 0,mode=sp['rec_mode'])
    tr.tri=sp['tri']; tr.stat_extra=[]; tr.lam_V=cal['lambda_E'] if sp['stat_enabled'] else 0
    tr.stat_ramp=0; tr.lam_edge=0; tr.lam_geo=0; tr.ramp=5000; tr.lam_gkd=0
    tr.gen=torch.Generator().manual_seed(3234); tr.corr_seed=3234
    tr._ema={}; tr._rr_val_last=float('nan'); tr.args=types.SimpleNamespace(num_iter=50000)
    return tr

for case in ('J0','JR','JQ','XJ','F0','FR','FQ','XF'):
    tr=make(case); be=G.CASES[case][1]
    total,info=tr._step(*inputs,1)
    ys=info['y']; l0=(ys-gt).abs().mean(); ld=ys.sum()*0; lk=ys.sum()*0
    if be!='N0':
        yt=info['y_t']; et=(yt-gt).abs().mean(1); es=(ys-gt).abs().mean(1)
        d=(et/(et+cal['tau_R'])).detach()
        a=((es-et).clamp_min(0)/(es+1e-6)).clamp(max=1).detach()
        ld=(d*es).mean()
        if be=='Q12': lk=(0.1*(1-d)*a*(ys-yt).abs().mean(1)).mean()
    le=cal['lambda_E']*output_edge_loss(ys.float(),gt.float()) if be in ('Q12','X02') else ys.sum()*0
    manual=l0+ld+lk+le
    if tr.aligner_trainable:
        eps=sample_offsets(2,2,torch.Generator().manual_seed(3234))
        mb=torch.nn.functional.interpolate(ms,scale_factor=4,mode='bicubic')
        ce=predict_c(tr.M.aligner,warp_pan(pan,eps),mb,4)
        manual=manual+1e-4*offset_loss(ce,info['delta'],eps,True)
    pars=[p for p in tr.M.parameters() if p.requires_grad]
    ga=torch.autograd.grad(total,pars,retain_graph=True,allow_unused=True)
    gb=torch.autograd.grad(manual,pars,retain_graph=True,allow_unused=True)
    err=max(float((a-b).abs().max()) for a,b in zip(ga,gb) if a is not None and b is not None)
    missing=any((a is None)!=(b is None) for a,b in zip(ga,gb))
    record(case+'_actual_step_vs_manual',torch.allclose(total,manual,atol=1e-7) and err<2e-6 and not missing,loss=float(total),scalar_error=float(abs(total-manual)),gradient_max_abs_error=err)
    y_eval=tr.M(pan,ms,lpan)['y']; record(case+'_forward_equals_inference',torch.equal(ys,y_eval),max_abs_error=float((ys-y_eval).abs().max()))
    if tr.aligner_trainable:
        gu=torch.autograd.grad(info['loss_off'],list(tr.M.backbone.parameters()),retain_graph=True,allow_unused=True)
        gc=torch.autograd.grad(info['loss_off'],info['delta'],retain_graph=True,allow_unused=True)[0]
        record(case+'_offset_ownership',all(g is None for g in gu) and gc is None)
    else:
        h=state_hash(tr.M.aligner)
        opt=torch.optim.AdamW(pars,lr=1e-4,weight_decay=.01); total.backward(); opt.step()
        record(case+'_frozen_adamw',h==state_hash(tr.M.aligner) and all(p.grad is None for p in tr.M.aligner.parameters()))
    del tr,total,info,ga,gb,manual
record('teacher_unchanged_and_inputs_unchanged',state_hash(T)==th and all(torch.equal(x,y) for x,y in zip(inputs,input_copies)))

tr=make('JQ'); tr._fixed_batch=inputs
logs=[]; tr._jsonl=lambda name,data:logs.append((name,data))
t_rng=torch.get_rng_state().clone(); e_rng=tr.gen.get_state().clone(); weights=state_hash(tr.M)
tr._diagnose_fixed(1,tr.M)
record('fixed_diagnostic_rng_weights_grads_unchanged',torch.equal(t_rng,torch.get_rng_state()) and torch.equal(e_rng,tr.gen.get_state()) and weights==state_hash(tr.M) and all(p.grad is None for p in tr.M.parameters()))
result['fixed_diagnostic_keys']=sorted(logs[-1][1])

# Replay existing launch unit tests, but prevent their live campaign_gate subprocess from mutating a campaign.
real_run=subprocess.run
def sandboxed_subprocess(cmd,*args,**kw):
    if isinstance(cmd,list) and any(str(c).endswith('campaign_gate.py') for c in cmd):
        kw['env']={**kw.get('env',os.environ),'PANCRAFTER_CAMPAIGN_GATES':'audit_disabled'}
    return real_run(cmd,*args,**kw)
buf=io.StringIO()
import contextlib
with patch('subprocess.run',side_effect=sandboxed_subprocess),contextlib.redirect_stdout(buf):
    try: runpy.run_path(str(ROOT/'tools/pakd50_unit_tests.py'),run_name='__main__')
    except SystemExit as e: code=e.code
record('existing_pakd50_gate_with_live_gate_disabled',code==0,exit_code=code)
result['existing_gate_stdout']=buf.getvalue()

# Fresh-server dependency: run the actual K05 calibration check statements against a missing local calibration.
tree=ast.parse((ROOT/'tools/pakd50_unit_tests.py').read_text())
nodes=[]
for node in tree.body:
    if 86 <= node.lineno <= 87: nodes.append(node)
captured=[]
scope={'os':os,'ROOT':str(ROOT),'G':G,'json':json,'yaml':yaml,'SEED':777,'check':lambda *a:captured.append(a)}
actual_exists=os.path.exists
with patch('os.path.exists',side_effect=lambda p:False if str(p)==str(ROOT/G.CAL_PATH) else actual_exists(p)):
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'K05-extracted','exec'),scope)
result['fresh_server_calibration_check']=repr(captured)
record('reproduce_fresh_server_K05_failure_before_bootstrap',len(captured)==1 and not captured[0][1],check=repr(captured))
from tools import campaign_gate as gate
with patch.dict(os.environ, {'PANCRAFTER_CAMPAIGN_GATES':''}):
    result['empty_gate_env_actual_enabled_gates']=gate.enabled_gates()
record('reproduce_empty_gate_env_reads_live_pakd50_token', result['empty_gate_env_actual_enabled_gates']==['pakd50'])

# Exact stage-2 copy expression, with old local tau-only and new git calibration, in /tmp.
with tempfile.TemporaryDirectory() as td:
    p=pathlib.Path(td); (p/'work_dir/_pakd50').mkdir(parents=True); (p/'assets/pakd50').mkdir(parents=True)
    (p/'work_dir/_pakd50/calibration_resolved.json').write_text(json.dumps({'tau_R':cal['tau_R']}))
    (p/'assets/pakd50/calibration_resolved.json').write_text(json.dumps(cal))
    line=next(x for x in (ROOT/'tools/pakd50_prepare.sh').read_text().splitlines() if x.startswith('[ -f "$CAMP/calibration_resolved.json" ]'))
    subprocess.run(['bash','-c','CAMP=work_dir/_pakd50\n'+line],cwd=p,check=True)
    remains=json.loads((p/'work_dir/_pakd50/calibration_resolved.json').read_text())
    record('reproduce_stale_local_calibration_after_stage2_copy', 'lambda_E' not in remains,remaining_keys=list(remains))

# Mid-epoch resume using the same DataLoader contract: RNG alone does not retain sampler cursor.
torch.manual_seed(1234); ds=torch.utils.data.TensorDataset(torch.arange(96))
loader=torch.utils.data.DataLoader(ds,batch_size=8,shuffle=True,num_workers=0,drop_last=True)
it=iter(loader); next(it); next(it); saved=torch.get_rng_state().clone(); continuous=next(it)[0].tolist()
torch.set_rng_state(saved); resumed=next(iter(loader))[0].tolist()
record('reproduce_resume_batch_order_divergence',continuous!=resumed,continuous_next=continuous,resumed_next=resumed)

# Current metrics: mean per-scene products, correct 20-image support, exact selector replay.
from pa.selector import BestSelector
rows=list(csv.DictReader((run/'checkpoint_metrics.csv').open())); scenes=list(csv.DictReader((run/'scene_metrics.csv').open()))
sel=BestSelector('raw'); errors=[]
for r in rows:
    rr=[s for s in scenes if s['step']==r['step'] and s['view']=='raw_original']
    if len(rr)!=20: continue
    h=sum((1-float(s['d_lambda']))*(1-float(s['d_s'])) for s in rr)/20
    errors.append(abs(h-float(r['raw_original.hqnr'])))
    sel.update(int(r['step']),int(r['epoch']),float(r['raw_original.hqnr']),float(r['raw_original.fscc']),True,'unused')
record('stored_FR20_scene_product_means',len(errors)==len(rows) and max(errors)<1e-12,checkpoints=len(rows),max_abs_error=max(errors),steps=[int(r['step']) for r in rows])
result['selector_replay']=dict(selected_step=sel.best['step'],hqnr=sel.best['hqnr'],max_hqnr=sel.max_hqnr)
result['runtime_at_snapshot']=json.loads((run/'memory_and_throughput.json').read_text())
result['test_coefficient_note']='lambda_E=0.3 is a synthetic test value; actual campaign lambda_E remains uncalibrated.'
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('Wrote',OUT)
