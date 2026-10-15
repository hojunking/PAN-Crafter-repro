"""QEDGE9: 384 actual cached views, exact-path replacement, pilot guard and
real switch handoff on exclusively audit-owned mock parent/child processes."""
import ast,contextlib,copy,io,json,os,pathlib,shutil,subprocess,sys,tempfile,time,types
from unittest.mock import patch
ROOT=pathlib.Path(sys.argv[1]).resolve();OUT=pathlib.Path(sys.argv[2]).resolve();os.chdir(ROOT);sys.path.insert(0,str(ROOT));os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PANCRAFTER_CAMPAIGN_GATES='')
import numpy as np,torch,yaml
torch.set_num_threads(2)
from tools import gen_pakd50_configs as G,qedge9_cue as Q
from kdv import edge_gate as EG
from kdv.edge_gate import EdgeGate
from pa.model import PAModel
from tools.smoke_cases import build
R={'checks':[],'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
def record(name,ok,kind='conformance',**kw):
    R['checks'].append(dict(name=name,passed=bool(ok),kind=kind,**kw));print(name,ok,json.dumps(kw),flush=True)
    (OUT/'extra_verification.json').write_text(json.dumps(R,indent=2))
buf=io.StringIO();cpudir=OUT/'online_cpu';cpudir.mkdir(exist_ok=True)
with patch.object(Q,'CAMP',str(cpudir)),contextlib.redirect_stdout(buf):
    try:Q.verify(types.SimpleNamespace(asset=G.QEDGE9_CUE_ASSET,config=Q.DEFAULT_CONFIG,n=96,seed=0,tol=1e-4))
    except SystemExit as e:rc=e.code
(OUT/'online_cpu.log').write_text(buf.getvalue());vv=json.loads((cpudir/('verify_'+Q.server()+'.json')).read_text());record('actual_online_384_views',rc==0 and vv['max_abs_de']<1e-4 and vv['label_flips']==0,verification=vv,tolerance_note='q CLI tolerance is 1e-4; audit additionally requires e < 1e-4 and zero label flips. Initial audit-only e < 1e-5 check was too strict for CPU/GPU comparison; initial results retained separately.')
man,z=EG.read_asset(G.QEDGE9_CUE_ASSET);arrays={k:z[k].copy() for k in z.files};T,tm=Q.load_T0()
with tempfile.TemporaryDirectory(prefix='qedge9-same-path-') as td:
    p=pathlib.Path(td);jp=p/'cue.json';npz=p/'cue.npz';mm=copy.deepcopy(man);mm['npz']=str(npz)
    def write():
        np.savez_compressed(npz,**arrays);mm['npz_sha256']=EG.sha256_file(npz);jp.write_text(json.dumps(mm))
    cfg=dict(mode='low_q',asset=str(jp));write();e0=EdgeGate.load(cfg,teacher=T)
    arrays['gate_low_q']=1-arrays['gate_low_q'];write();e1=EdgeGate.load(cfg,teacher=T)
    record('same_asset_path_and_config_reload_accepts_opposite_gate',torch.equal(e0.table,1-e1.table),'defect_reproduction',asset_path_unchanged=True,old_hash=e0.manifest['npz_sha256'],new_hash=e1.manifest['npz_sha256'])

with tempfile.TemporaryDirectory(prefix='qedge9-pilot-guard-') as td:
    p=pathlib.Path(td);c=yaml.safe_load((ROOT/Q.DEFAULT_CONFIG).read_text());c['model_args']['depth']=[1,2,3]
    pm=PAModel(build(c),copy.deepcopy(T.aligner),aligner_margin=4,sampler=True).eval();pm.requires_grad_(False)
    wrong=G.run_name('QE50',777,arch='W104_D123');pman=dict(width=104,depth=[1,2,3],tensors_sha256_16='diagnostic_fixture',file_sha256='diagnostic_fixture',tag_meta=dict(step=1010))
    small=copy.deepcopy(man);small['dataset']['n_train']=2;small['calibration']['index_sha256_16']=Q.calibration_ids(2)[1]
    a=types.SimpleNamespace(asset=G.QEDGE9_CUE_ASSET,config=Q.DEFAULT_CONFIG,run=wrong,tag='best_hqnr',all_views=False,chunk=2);log=io.StringIO()
    with patch.object(Q,'CAMP',str(p)),patch.object(G,'QEDGE9_CE_FILE',str(p/'ce.json')):
        with patch.object(EG,'read_asset',return_value=(small,{'gate_low_q':np.array([0,1]*4)})),patch.object(Q,'load_run_model',return_value=(pm,pman)),contextlib.redirect_stdout(log):
            Q.pilot(a)
    ce=json.loads((p/'ce.json').read_text());record('pilot_accepts_wrong_case_seed_depth_non50K_tag',ce['pilot_run']==wrong and ce['pilot_step']==1010 and ce['pilot_depth']==[1,2,3],'defect_reproduction',pilot=ce,note='guard fixture: actual W104D123 model, 2 real base patches x 4 views; loader and cue population restricted to isolate validation')
    (OUT/'pilot_guard.log').write_text(log.getvalue())

# Reuse only fixture constructor, not the other test module's top-level code.
tree=ast.parse((OUT/'verify_qedge9.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixture');ns=dict(ROOT=ROOT,G=G,shutil=shutil,json=json)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'fixture_only','exec'),ns);fixture=ns['fixture']
with tempfile.TemporaryDirectory(prefix='qedge9-owned-handoff-') as td:
    p=pathlib.Path(td);fixture(p,'s5');(p/'tools/pakd50_unit_tests.py').write_text("print('ALL OK (separately checked)')\n");(p/'tools/qedge9_cue.py').write_text("print('cue verified separately')\n")
    stale=G.run_name('LFQ',9091);(p/G.EXTRA_PRIORITY_FILE).write_text(stale+'\n')
    (p/'child.sh').write_text('#!/bin/bash\nfor i in {1..600}; do [ -f release_child ] && break; sleep 0.05; done\ntouch child_completed\n')
    (p/'parent.sh').write_text('#!/bin/bash\nbash child.sh\ntouch parent_postprocessing\n')
    parent=subprocess.Popen(['bash','parent.sh'],cwd=p)
    bindir=p/'mockbin';bindir.mkdir()
    # ps reports only the parent created above, never any real training process.
    (bindir/'ps').write_text('#!'+sys.executable+'\nimport pathlib,sys\np=pathlib.Path("/proc/'+str(parent.pid)+'/stat")\nif p.exists() and p.read_text().split()[2] != "Z":\n print("'+str(parent.pid)+' bash ./tools/_run_cases.sh" if "pid,args" in sys.argv else "bash ./tools/_run_cases.sh")\n')
    (bindir/'crontab').write_text('#!/bin/bash\nif [ "${1:-}" = -l ]; then exit 0; else cat >/dev/null; fi\n')
    for f in bindir.iterdir():f.chmod(0o755)
    (p/'tools/campaign_start.sh').write_text('#!/bin/bash\ncd "$(dirname "$0")/.."\ntouch release_child\nexit 0\n');(p/'tools/campaign_start.sh').chmod(0o755)
    (p/'tools/_watchdog.sh').write_text('#!/bin/bash\nexit 0\n');(p/'tools/_watchdog.sh').chmod(0o755)
    env={**os.environ,'PYTHON':sys.executable,'PATH':str(bindir)+os.pathsep+os.environ['PATH']}
    try:
        r=subprocess.run(['bash','tools/qedge9_switch.sh'],cwd=p,env=env,capture_output=True,text=True,timeout=60);(OUT/'owned_handoff.log').write_text(r.stdout+r.stderr)
        parent.wait(timeout=5)
        for _ in range(100):
            if (p/'child_completed').exists():break
            time.sleep(.02)
        record('switch_kills_owned_parent_loses_postprocessing',(p/'child_completed').exists() and not(p/'parent_postprocessing').exists() and parent.returncode==-15,'defect_reproduction',switch_rc=r.returncode,parent_rc=parent.returncode,child_completed=(p/'child_completed').exists(),postprocessing=(p/'parent_postprocessing').exists(),note='all process ids came from one audit-owned parent; ps/crontab/start/watchdog replaced with isolated stubs')
        with patch.object(G,'ROOT',str(p)):
            extras=G.extra_priority();todo,_=G.schedule(True,lambda _:False,100,lambda _:1,priority=G.priority_for('s5'),extra=extras,exempt=lambda it:G.branch_for('s5',it)=='QEDGE9',blocked=lambda _:False)
        record('switch_preserves_unplanned_old_extra',stale in todo and len(todo)==7,'defect_reproduction',extra=extras,scheduled_count=len(todo))
    finally:
        (p/'release_child').touch()
        if parent.poll() is None:parent.terminate();parent.wait(timeout=5)
R['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');R['summary']={k:dict(total=sum(x['kind']==k for x in R['checks']),passed=sum(x['kind']==k and x['passed'] for x in R['checks'])) for k in ('conformance','defect_reproduction')}
(OUT/'extra_verification.json').write_text(json.dumps(R,indent=2));print(R['summary'],flush=True)
sys.exit(0 if all(x['passed'] for x in R['checks']) else 1)
