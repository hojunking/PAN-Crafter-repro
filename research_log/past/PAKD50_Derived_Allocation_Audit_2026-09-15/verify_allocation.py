"""Bounded CPU and /tmp-only operational audit; never launch a campaign.
Usage: <pancrafter-python> -B verify_allocation.py REPO OUTPUT_DIR
The existing unit suite is executed in an isolated checkout because K08 writes
mandatory_runs.txt temporarily. Shell fixtures replace training/upload with mocks.
"""
import contextlib, copy, hashlib, io, json, os, pathlib, shutil, subprocess, sys, tempfile, time, types
from unittest.mock import patch
ROOT = pathlib.Path(sys.argv[1]).resolve()
OUT = pathlib.Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PANCRAFTER_CAMPAIGN_GATES='')
import h5py, torch, yaml
torch.set_num_threads(2)
from tools import gen_pakd50_configs as G, campaign_gate as gate
from main import import_class
from train_kdv import KDVTrainer
from kdv.teacher_assets import freeze, load_run_model, state_hash
from kdv.registry import resolve
from kdv.losses_rec import GTAnchoredReconstructionKD
from pa.losses import output_edge_loss
from pa.offset import offset_loss, predict_c, sample_offsets
from pa.warp import warp_pan

R = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
     'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'device': 'cpu', 'checks': []}
def check(name, ok, kind='conformance', **details):
    R['checks'].append(dict(name=name, passed=bool(ok), kind=kind, **details))
    (OUT/'verification.json').write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(('PASS ' if ok else 'FAIL ') + name + ' ' + json.dumps(details, ensure_ascii=False), flush=True)

def fixture(p, server='s4'):
    # Source/assets/data are read only symlinks; all potential operational writes
    # (tools, config, gspread, work_dir) have their own real directories.
    for src in ROOT.iterdir():
        if src.name in ('.git', '.agents', '.codex', 'research_log', 'results_log', 'outputs', '__pycache__'):
            continue
        if src.name in ('work_dir', 'gspread'):
            (p/src.name).mkdir()
        elif src.name in ('tools', 'config'):
            shutil.copytree(src, p/src.name, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            (p/src.name).symlink_to(src, target_is_directory=src.is_dir())
    (p/'gspread/server.txt').write_text(server)
    (p/'work_dir/_pakd50').mkdir()
    (p/'work_dir/_pakd50/ledger.json').write_text(json.dumps(G.campaign_clock()))
    (p/'work_dir/campaign_gates_enabled.txt').write_text('pakd50\n')
    (p/'work_dir/_pakd50_budget').mkdir()
    (p/'work_dir/_pakd50_budget/ledger.json').write_text(json.dumps({'entries':{}, 'total_gpu_hours':50}))

with tempfile.TemporaryDirectory(prefix='pakd50-derived-unit-') as td:
    p = pathlib.Path(td); fixture(p)
    r = subprocess.run([sys.executable, '-B', str(p/'tools/pakd50_unit_tests.py')], cwd=p, capture_output=True, text=True, timeout=240)
    (OUT/'existing_unit_suite.log').write_text(r.stdout+r.stderr)
    check('existing_unit_suite_in_isolated_checkout', r.returncode == 0, exit_code=r.returncode,
          ok_assertions=sum(l.startswith('  OK ') for l in r.stdout.splitlines()))

cal = json.loads((ROOT/G.ASSET_CAL_PATH).read_text())
expected = {'s2':['JR','XJ','J_R3_NOEDGE','J_N0_EDGE'], 's4':['F0','RC0','RCQ','JR','XJ','J_R3_NOEDGE'], 's5':['PQ','F0','LF0','LFQ','RC0','RCQ']}
config_rows=[]
with tempfile.TemporaryDirectory(prefix='pakd50-derived-config-') as td:
    for srv, cases in expected.items():
        check('explicit_order_'+srv, G.priority_for(srv)==cases and G.mandatory_for(srv)==cases)
        for rn in G.generate(srv, cases, td):
            want = yaml.safe_load((pathlib.Path(td)/(rn+'.yaml')).read_text())
            have = yaml.safe_load((ROOT/'config'/(rn+'.yaml')).read_text())
            k=have['kdv']; sp=resolve(k)
            ok=have==want and have['num_iter']==50000 and have['batch_size']==48 and have['seed']==G.SERVER_SEED[srv]
            # YAML 1e-4 is a string; main.py's argparse float converter resolves it.
            from main import get_parser
            parser=get_parser(); parser.set_defaults(**have); args=parser.parse_args([])
            ok=ok and args.learning_rate==1e-4 and have['lr_scheduler']=='cosine' and have['eval_epoch']==5
            ok=ok and have['model_args']['hidden_size']==112 and have['model_args']['depth']==[1,2,3]
            if k['rec']['case']!='N0':
                ok=ok and k['rec']['tau']==cal['tau_R'] and k['rec']['alpha']==1
            if k['stat']['enabled']: ok=ok and k['stat']['outer_weight']==cal['lambda_E']
            ok=ok and k['budget']['training_deadline']=='2026-09-16T11:22:31'
            config_rows.append(dict(server=srv, run=rn, init_pin=k.get('expect_init'), backend=k['rec']['case'], protocol=sp['protocol']))
            check('config_'+rn, ok)
R['configs']=config_rows
totals={s:sum(G.reservation_for(s,c)['reservation_h'] for c in cs) for s,cs in expected.items()}
check('reservation_arithmetic_core_and_confirmation', all(abs(totals[s]-v)<1e-7 for s,v in {'s2':10.9186666667,'s4':9.69,'s5':10.427}.items()),
      core=totals, slot_total=sum(totals.values())+12.88, barrier_elapsed=max(totals.values())+6.44)

# Real WV3 patches and real learned U weights (to activate nonzero gradients),
# with exactly the released T0 A. This is a loss-contract test, not a training run.
tcfg=yaml.safe_load((ROOT/'assets/pakd50/T0_run/meta/config.yaml').read_text())
Model=import_class(tcfg['model']); T,_=load_run_model('assets/pakd50/T0_run','best_hqnr',Model); freeze(T)
th=state_hash(T); template=copy.deepcopy(T)
from safetensors.torch import load_file
sd=load_file(str(ROOT/'work_dir'/G.pilot_run()/'candidates/step-4040/model.safetensors'))
template.backbone.load_state_dict({k.removeprefix('backbone.'):v for k,v in sd.items() if k.startswith('backbone.')})
dcfg=yaml.safe_load((ROOT/'config'/(G.run_name('JQ',1234)+'.yaml')).read_text())
data=dcfg['train_feeder_args']['dataroot']
with h5py.File(data) as f:
    gt,ms,pan=[torch.from_numpy(f[k][:2].astype('float32'))/2047*2-1 for k in ('gt','ms','pan')]
with h5py.File(data.replace('.h5','_pan.h5')) as f: lpan=torch.from_numpy(f['lpan'][:2].astype('float32'))/2047*2-1
inputs=(gt,ms,lpan,pan); copies=[x.clone() for x in inputs]
def make(case):
    tr=object.__new__(KDVTrainer); tr.k=G.kdv_block(case,2026,'s5',cal=cal); sp=tr.spec=resolve(tr.k)
    tr.model=copy.deepcopy(template).train().requires_grad_(True)
    if not sp['aligner_trainable']: freeze(tr.model.aligner)
    tr.accelerator=types.SimpleNamespace(unwrap_model=lambda m:m,gradient_accumulation_steps=1,scaler=None,is_main_process=True)
    tr.teacher=T; tr.aligner_trainable=sp['aligner_trainable']; tr.aligner_view_margin=4; tr.share_correction=not sp['aligner_trainable']
    tr.protocol=sp['protocol']; tr.radius_hr=sp['radius_hr']; tr.diag_every=1000
    tr.rec_crit=GTAnchoredReconstructionKD(cal['tau_R'],alpha=1,kd_weight=float(tr.k['rec'].get('kd_weight',0)),eps=1e-6,mode=sp['rec_mode']) if sp['rec_case']!='N0' else None
    tr.tri=sp['tri']; tr.stat_extra=[]; tr.lam_V=cal['lambda_E'] if sp['stat_enabled'] else 0
    tr.stat_ramp=0; tr.lam_edge=0; tr.lam_geo=0; tr.ramp=5000; tr.lam_gkd=0
    tr.gen=torch.Generator().manual_seed(4026); tr.corr_seed=4026; tr._ema={}; tr._rr_val_last=float('nan'); tr.args=types.SimpleNamespace(num_iter=50000)
    tr.freeze_until=sp['aligner_freeze_until']; tr.freeze_from=sp['aligner_freeze_from']; tr.route_A=tuple(sp['route_A']); tr._routed=any(q!=1 for q in tr.route_A); tr._sched_last=None
    return tr
def grads(loss, pars):
    return [g if g is not None else torch.zeros_like(p) for g,p in zip(torch.autograd.grad(loss,pars,retain_graph=True,allow_unused=True),pars)]
for case in ('JR','XJ','J_R3_NOEDGE','J_N0_EDGE','F0','RC0','RCQ','PQ','LF0','LFQ'):
    tr=make(case); rng0=tr.gen.get_state().clone()
    if not tr.spec['needs_teacher']: tr.teacher=None # strongest test of loss independence from Teacher
    total,info=tr._step(*inputs,1); y=info['y']; zero=y.sum()*0; l0=(y-gt).abs().mean(); ld=zero; lk=zero
    if info['y_t'] is not None:
        yt=info['y_t']; et=(yt-gt).abs().mean(1); es=(y-gt).abs().mean(1)
        d=(et/(et+cal['tau_R'])).detach(); a=((es-et).clamp_min(0)/(es+1e-6)).clamp(max=1).detach()
        ld=(d*es).mean()
        if tr.spec['rec_case']=='R3': lk=(.1*(1-d)*a*(y-yt).abs().mean(1)).mean()
    le=cal['lambda_E']*output_edge_loss(y.float(),gt.float()) if tr.spec['stat_enabled'] else zero
    lo=zero
    if tr.protocol=='I-AEQ':
        eps=sample_offsets(2,2,torch.Generator().manual_seed(4026))
        mb=torch.nn.functional.interpolate(ms,scale_factor=4,mode='bicubic')
        lo=1e-4*offset_loss(predict_c(tr.M.aligner,warp_pan(pan,eps),mb,4),info['delta'],eps,True)
    manual=l0+ld+lk+le+lo; ap=[p for p in tr.M.aligner.parameters() if p.requires_grad]; bp=list(tr.M.backbone.parameters())
    wantA=grads(l0+lo if case=='PQ' else manual,ap) if ap else []; wantU=grads(manual,bp)
    if ap: rawA=sum(float(x.abs().sum()) for x in grads(l0,ap))
    else: rawA=0.
    softA=sum(float(x.abs().sum()) for x in grads(lk,ap)) if ap else 0
    edgeA=sum(float(x.abs().sum()) for x in grads(le,ap)) if ap else 0
    total.backward(retain_graph=tr._routed)
    if tr._routed: tr._apply_routing(info,tr.M)
    errA=max([float((p.grad-g).abs().max()) for p,g in zip(ap,wantA)] or [0])
    errU=max(float((p.grad-g).abs().max()) for p,g in zip(bp,wantU))
    check(case+'_manual_loss_and_A_U_gradients', abs(float(total-manual))<1e-7 and errA<2e-6 and errU<2e-6,
          scalar_error=abs(float(total-manual)), A_max_abs_error=errA,U_max_abs_error=errU,soft=float(lk),edge=float(le),rec_A_L1=rawA,soft_A_L1=softA,edge_A_L1=edgeA)
    check(case+'_native_student_forward', torch.equal(info['pan_view'],pan) and torch.equal(y,tr.M(pan,ms,lpan)['y']) and not info['corrupt'])
    if case.startswith('RC'):
        check(case+'_no_offset_and_live_reconstruction_A', torch.equal(rng0,tr.gen.get_state()) and float(info['loss_off'])==0 and rawA>0 and info['delta'].requires_grad)
    if case=='J_N0_EDGE': check('J_N0_EDGE_no_teacher_object_needed',info['y_t'] is None and float(le)>0)
    if case=='F0':
        ah=state_hash(tr.M.aligner); opt=torch.optim.AdamW(tr.M.parameters(),lr=1e-4,weight_decay=.01); opt.step()
        check('F0_frozen_A_after_AdamW',state_hash(tr.M.aligner)==ah and all(p.grad is None for p in tr.M.aligner.parameters()))
    del tr,total,manual,info,wantA,wantU

from diffusers.optimization import get_scheduler
for case in ('LF0','LFQ'):
    tr=make(case); opt=torch.optim.AdamW([dict(params=tr.M.backbone.parameters(),lr=1e-4),dict(params=tr.M.aligner.parameters(),lr=1e-5)],weight_decay=.01)
    sch=get_scheduler('cosine',optimizer=opt,num_warmup_steps=0,num_training_steps=50000)
    for step in (24999,25000,25001):
        sch.last_epoch=step; fac=sch.lr_lambdas[0](step)
        for pg,base in zip(opt.param_groups,sch.base_lrs): pg['lr']=base*fac
        tr.M.aligner.requires_grad_(tr.aligner_active(step)); a0=state_hash(tr.M.aligner); u0=state_hash(tr.M.backbone)
        opt.zero_grad(set_to_none=True); total,info=tr._step(*inputs,step); total.backward()
        if not tr.aligner_active(step):
            for p in tr.M.aligner.parameters(): p.grad=None
        opt.step(); sch.step()
        a1=state_hash(tr.M.aligner); u1=state_hash(tr.M.backbone)
        # AdamW A moments are created by the live 24999 update and must stay at 1.
        moments=[float(opt.state[p]['step']) for p in tr.M.aligner.parameters() if p in opt.state]
        check(case+'_actual_AdamW_boundary_'+str(step), (a0!=a1 if step==24999 else a0==a1) and u0!=u1 and bool(moments) and set(moments)=={1.0}
              and (info.get('eq_exercise')==1 if step==24999 else float(info['loss_off'])==0), active=tr.aligner_active(step),A_changed=a0!=a1,U_changed=u0!=u1,A_moment_steps=sorted(set(moments)),U_lr=opt.param_groups[0]['lr'],cosine_global_step=sch.last_epoch)
    del tr,total,info,opt,sch
check('teacher_and_input_data_unchanged',state_hash(T)==th and all(torch.equal(a,b) for a,b in zip(inputs,copies)))

# Full reallocation shell in a fake server, never an active workspace. Unit gate
# is stubbed here (the unmodified suite was run separately above); config generation
# and validation and all --dry-run/--confirm mutations are the actual source.
with tempfile.TemporaryDirectory(prefix='pakd50-derived-operations-') as td:
    p=pathlib.Path(td); fixture(p)
    (p/'tools/pakd50_unit_tests.py').write_text("print('ALL OK (separately verified; shell fixture)')\n")
    (p/'tools/pakd50_requeue.sh').write_text('#!/bin/bash\necho FORBIDDEN_REQUEUE >&2\nexit 93\n')
    env={**os.environ,'PYTHON':sys.executable}
    gen=subprocess.run([sys.executable,'-B','tools/gen_pakd50_configs.py','--server','s4','--cases',','.join(expected['s4'])],cwd=p,env=env,capture_output=True,text=True)
    check('shell_fixture_config_generation',gen.returncode==0)
    def invoke(args,label):
        r=subprocess.run(['bash','tools/pakd50_reallocate.sh',*args],cwd=p,env=env,capture_output=True,text=True,timeout=90)
        (OUT/(label+'.log')).write_text(r.stdout+r.stderr)
        check(label+'_execution',r.returncode==0,exit_code=r.returncode)
        return r
    invoke(['--dry-run'],'dry_run')
    paths=[G.CAL_PATH,G.MANDATORY_FILE,G.RESERVATION_FILE]
    check('BUG_dry_run_creates_live_control_files', all((p/x).exists() for x in paths),kind='defect_reproduction',created=paths)
    invoke(['--confirm','RCQ','--dry-run'],'confirm_RCQ_dry_run')
    extra=(p/G.EXTRA_PRIORITY_FILE).read_text().splitlines(); mandatory=(p/G.MANDATORY_FILE).read_text().splitlines()
    conf=[G.run_name(c,3407) for c in ('RCQ','RC0','F0')]
    check('BUG_confirmation_mandatory_overwritten',all(c in extra for c in conf) and all(c not in mandatory for c in conf),kind='defect_reproduction',extra=extra,mandatory=mandatory)
    invoke(['--confirm','LFQ','--dry-run'],'confirm_LFQ_dry_run')
    extra=(p/G.EXTRA_PRIORITY_FILE).read_text().splitlines()
    check('BUG_repeated_confirmation_exceeds_three_and_changes_WIN', len(extra)==5,kind='defect_reproduction',extra=extra)
    # Dispatch observation: use actual _upload.sh, but every python invocation is
    # a local logging mock; no evaluator, GPU or external upload runs.
    bindir=p/'bin'; bindir.mkdir(); fake=bindir/'python'
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$AUDIT_COMMANDS"\nexit 0\n'); fake.chmod(0o755)
    rn=G.run_name('RCQ',1234); (p/'work_dir'/rn/'results').mkdir(parents=True)
    r=subprocess.run(['bash','tools/_upload.sh',rn],cwd=p,capture_output=True,text=True,timeout=10,env={**env,'CONDA_DEFAULT_ENV':'pancrafter','PATH':str(bindir)+':'+env['PATH'],'AUDIT_COMMANDS':str(p/'commands.txt')})
    commands=(p/'commands.txt').read_text(); (OUT/'upload_mock_commands.log').write_text(commands)
    check('BUG_basic_upload_also_dispatches_pa_diag', 'tools/pa_diag.py' in commands,kind='defect_reproduction',commands=commands.splitlines())

with tempfile.TemporaryDirectory(prefix='pakd50-derived-gate-') as td:
    p=pathlib.Path(td); fixture(p,'s5')
    with patch.object(G,'ROOT',str(p)),patch.object(gate,'ROOT',str(p)):
        # All local core results exist, but no remote barrier/WIN state is present.
        for c in expected['s5']:
            q=p/'work_dir'/G.run_name(c,2026)/'results'; q.mkdir(parents=True)
            for f in ('reduced_best_hqnr.mat','full_best_hqnr.mat'): (q/f).touch()
        pq=G.run_name('PQ',2026)
        check('BUG_completion_accepts_two_empty_files_without_exit_or_exact50K',gate.complete(pq),kind='defect_reproduction')
        extras=[G.run_name(c,9091) for c in ('RCQ','RC0','F0')]
        (p/G.EXTRA_PRIORITY_FILE).write_text('\n'.join(extras)+'\n')
        for tag in extras: (p/'config'/(tag+'.yaml')).write_text('fixture: true\n')
        buf=io.StringIO(); err=io.StringIO(); gate.EMITTED.clear()
        with patch.object(G,'hours_to_deadline',return_value=7.0),patch.object(gate.subprocess,'run',return_value=types.SimpleNamespace(stdout='')),contextlib.redirect_stdout(buf),contextlib.redirect_stderr(err): gate.gate_pakd50()
        emitted=buf.getvalue().splitlines(); (OUT/'gate_confirmation_fixture.log').write_text(buf.getvalue()+err.getvalue())
        check('BUG_gate_admits_confirmation_without_global_core_barrier', emitted==extras,kind='defect_reproduction',emitted=emitted,remaining_h=7.0,example_other_server_core_remaining_h=2.0,required_finish_h=2+6.44)
        rows=G.plan_rows('s5'); check('BUG_plan_table_omits_registered_confirmations',all(r['seed']==2026 for r in rows),kind='defect_reproduction',plan_rows=len(rows),registered_confirmations=3)
        # Old optional tail still enters the new 16-run scope.
        todo,_=G.schedule(True,lambda c:False,40,lambda c:G.reservation_for('s5',c)['reservation_h'],priority=expected['s5'],extra=['DPQ'])
        check('BUG_old_extra_priority_survives_new_scope',todo==expected['s5']+['DPQ'],kind='defect_reproduction',todo=todo)
        todo,dropped=G.schedule(True,lambda c:False,1.8,lambda c:G.reservation_for('s5',c)['reservation_h'],priority=expected['s5'])
        check('BUG_local_admission_skips_higher_priority_to_run_cheaper_F0',todo==['F0'] and 'PQ' in dropped,kind='defect_reproduction',todo=todo,dropped=dropped)

# Minimal real bash handoff reproduction: terminate only the parent runner we
# create; let its training child finish, then inspect whether upload still ran.
with tempfile.TemporaryDirectory(prefix='pakd50-derived-handoff-') as td:
    p=pathlib.Path(td)
    (p/'child.sh').write_text('touch child_started\nwhile [ ! -f release_child ]; do sleep 0.05; done\ntouch child_finished\n')
    (p/'runner.sh').write_text('bash child.sh\ntouch upload_finished\n')
    runner=subprocess.Popen(['bash','runner.sh'],cwd=p,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        until=time.monotonic()+5
        while not (p/'child_started').exists() and time.monotonic()<until: time.sleep(.02)
        assert (p/'child_started').exists()
        runner.terminate(); runner.wait(timeout=3); (p/'release_child').touch()
        until=time.monotonic()+3
        while not (p/'child_finished').exists() and time.monotonic()<until: time.sleep(.02)
        check('BUG_killing_runner_preserves_training_but_loses_following_upload', (p/'child_finished').exists() and not (p/'upload_finished').exists(),kind='defect_reproduction',runner_exit_code=runner.returncode)
    finally:
        (p/'release_child').touch()
        if runner.poll() is None: runner.terminate(); runner.wait(timeout=3)

check('GAP_missing_init_pins_for_s2_s5_confirmation',all(G.init_hash_for(s) is None for s in (777,2026,3407,9091)),kind='readiness_gap',pins={s:G.init_hash_for(s) for s in (1234,777,2026,3407,9091)})
R['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
R['summary']={kind:dict(total=sum(c['kind']==kind for c in R['checks']),passed=sum(c['kind']==kind and c['passed'] for c in R['checks'])) for kind in sorted({c['kind'] for c in R['checks']})}
(OUT/'verification.json').write_text(json.dumps(R,ensure_ascii=False,indent=2))
print(json.dumps(R['summary'],ensure_ascii=False),flush=True)
sys.exit(0 if all(c['passed'] for c in R['checks']) else 1)
