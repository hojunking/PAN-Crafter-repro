"""Read-only config audit; negative operational fixtures use only /tmp."""
import csv,io,json,os,pathlib,re,subprocess,sys,tempfile,importlib.util,contextlib
from unittest.mock import patch
from audit_common import *
os.environ.update(ENV);sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import yaml,torch
from tools import gen_pakd50_configs as G
from kdv.resume import EpochState,begin_epoch
R=[]
def rec(name,ok,kind='conformance',**kw):
 R.append(dict(name=name,passed=bool(ok),kind=kind,**kw));(OUT/'ops_results.json').write_text(json.dumps(R,indent=2,ensure_ascii=False));print(name,ok,kw,flush=True)
# Parse exact ordered tables from plan rather than trusting generator constants.
plan=(ROOT/'research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md').read_text();allruns=[];dangling=[]
with tempfile.TemporaryDirectory(prefix='qrc24-gen-') as td:
 for si in range(1,6):
  srv=f's{si}';section=plan.split(f'### 5.{si} ')[1].split('\n##')[0];rows=[]
  for line in section.splitlines():
   v=[s.strip() for s in line.split('|')[1:-1]]
   if len(v)==8 and v[0].isdigit():rows.append(v)
  want=[G.qrc24_run_name(srv,v[1],int(v[2])) for v in rows];queue=[l for l in (ROOT/f'config/queues/qrecon24_{srv}.txt').read_text().splitlines() if l and not l.startswith('#')];G.generate(srv,want,td,projected=None);bad=[]
  for run,v in zip(want,rows):
   src=ROOT/'config'/f'{run}.yaml';cfg=yaml.safe_load(src.read_text());k=cfg['kdv'];a,b=map(float,v[5].split('/'));q=k['qrc24'];pairs=((q['lambda_E'],float(v[3])),(q['rA'],float(v[4])),(q['alpha'],a),(q['beta'],b),(cfg['learning_rate'],float(v[6])))
   if any(abs(x-y)>1e-12 for x,y in pairs) or src.read_bytes()!=(pathlib.Path(td)/src.name).read_bytes():bad.append(run)
   if not (cfg['num_iter']==50000 and cfg['batch_size']==48 and cfg['num_warmup']==100 and cfg['weight_decay']==.01 and k['exact_resume'] and k['select']['retain_all_candidates'] and k['qrecon']['q_ref']==.3276133416220546):bad.append(run+' fixed')
   for control in k.get('control_runs',{}).values():
    if control not in G.priority_for(srv):dangling.append(dict(run=run,control=control))
  reserve=sum(G.reservation_for(srv,r)['reservation_h'] for r in want)
  rec(srv+'_table_config_queue',want==queue==G.priority_for(srv) and not bad and abs(reserve-(1.2*len(want)*G.QRC24_REFERENCE_H[srv]+len(want)*10/60))<1e-10,runs=len(want),mismatches=bad,reservation_h=reserve);allruns+=want
rec('all_68_unique',len(allruns)==68==len(set(allruns)))
rec('nonexistent_G22_controls',len(dangling)==26,'defect_reproduction',affected=len(dangling),examples=dangling[:2]+dangling[-2:])
# Existing exact-resume helper rejects a standard epoch-end save.
es=EpochState();es.set(torch.get_rng_state(),4848,202)
try:begin_epoch([None]*202,es,5050,True);msg=None
except RuntimeError as ex:msg=str(ex)
rec('epoch25_resume_rejected',msg is not None,'defect_reproduction',error=msg,save_epoch=25,batches_per_epoch=202)
# Invoke the actual --extend script. No mocked launch is needed: branch exits before any launcher.
with tempfile.TemporaryDirectory(prefix='qrc24-ops-') as td:
 p=pathlib.Path(td);fixture(p,'s1');base=G.priority_for('s1');extras=G.qrc24_extension_items('s1');mand=p/G.QRC24_MANDATORY_FILE;mand.parent.mkdir(parents=True);mand.write_text('\n'.join(base)+'\n')
 for r in base:
  d=p/'work_dir'/r/'results';d.mkdir(parents=True);(d/'reduced_best_hqnr.mat').touch();(d/'full_best_hqnr.mat').touch()
 lp=p/G.QRC24_LEDGER;lp.parent.mkdir(parents=True);lp.write_text(json.dumps(dict(entries={r:dict(kind='run',status='FINISHED_EXPORT',hours=1.5,hours_total=1.5) for r in base})))
 out=subprocess.run(['bash','tools/qrecon24_switch.sh','--extend'],cwd=p,env=ENV,text=True,capture_output=True,timeout=60);(OUT/'extend.log').write_text(out.stdout+out.stderr)
 after=(p/G.EXTRA_PRIORITY_FILE).read_text();missing=[r for r in extras if r not in mand.read_text()]
 # Execute the actual waiter status() Python body, then the actual waiter should exit DONE without launch.
 out2=subprocess.run(['bash','tools/qrecon24_waiter.sh'],cwd=p,env=ENV,text=True,capture_output=True,timeout=20);(OUT/'waiter_after_extend.log').write_text(out2.stdout+out2.stderr)
 st=json.loads((p/'work_dir/_qrecon24/status.json').read_text())
 rec('extend_not_connected_to_waiter',out.returncode==0 and all(r in after for r in extras) and len(missing)==3 and out2.returncode==0 and st['status']=='DONE','defect_reproduction',extend_rc=out.returncode,mandatory_len=len(mand.read_text().splitlines()),new_configs=sum((p/'config'/f'{r}.yaml').exists() for r in extras),waiter_status=st['status'])
 # With 23.4 training hours + exports reaching 24.6 the script incorrectly stops extending.
 lp.write_text(json.dumps(dict(entries={r:dict(kind='run',status='FINISHED_EXPORT',hours=1.95,hours_total=2.05) for r in base})));(p/G.EXTRA_PRIORITY_FILE).unlink()
 out=subprocess.run(['bash','tools/qrecon24_switch.sh','--extend'],cwd=p,env=ENV,text=True,capture_output=True,timeout=60)
 rec('export_hours_counted_as_training',not (p/G.EXTRA_PRIORITY_FILE).exists() and '확장 불필요' in out.stdout+out.stderr,'defect_reproduction',training_h=23.4,including_exports_h=24.6,exit_code=out.returncode,output=(out.stdout+out.stderr).strip())
 # Run exact embedded mutation block of switch step 5, isolating unrelated launch/gate work.
 xp=p/G.EXTRA_PRIORITY_FILE;old='PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v2';xp.write_text(old+'\n'+'\n'.join(extras)+'\n');script=(p/'tools/qrecon24_switch.sh').read_text();block=script.split('echo "[qrecon24] ⑤')[1].split("<<'PYEOF'")[1].split('\n',1)[1].split('\nPYEOF')[0]
 out=subprocess.run([PY,'-B','-','s1'],input=block,cwd=p,env=ENV,text=True,capture_output=True,timeout=60);(OUT/'switch_extra_migration.log').write_text(out.stdout+out.stderr)
 remain=[l for l in xp.read_text().splitlines() if l and not l.startswith('#')]
 rec('switch_drops_existing_qrc_extensions',out.returncode==0 and remain==[],'defect_reproduction',before=extras,after=remain,archived_files=[f.name for f in xp.parent.glob('extra_priority.pre_qrecon24*')])
 # Verified-complete accepts empty artifacts + duplicate candidate rows + a nonexact manifest.
 run=base[0];d=p/'work_dir'/run;(d/'last').mkdir();(d/'last/model.safetensors').touch();(d/'last_meta.json').write_text('{"step":50000}');(d/'kdv_config_resolved.json').write_text('{"resumed_nonexact":true}');(d/'checkpoint_metrics.csv').write_text('step\n'+'50000\n'*45)
 cmd='import json;from tools import gen_pakd50_configs as G;print(json.dumps(G.verified_complete('+repr(run)+')))';out=subprocess.run([PY,'-B','-c',cmd],cwd=p,env=ENV,text=True,capture_output=True,timeout=60);ver=json.loads(out.stdout)
 rec('incomplete_run_verified',ver['ok'],'defect_reproduction',verification=ver)
# Actual selector main() with controlled metric providers, not GPU inference.
spec=importlib.util.spec_from_file_location('qrc_select',ROOT/'tools/qrecon24_select.py');sel=importlib.util.module_from_spec(spec);spec.loader.exec_module(sel)
with tempfile.TemporaryDirectory(prefix='qrc24-selector-') as td:
 p=pathlib.Path(td);sel.ROOT=str(p);wd=p/'work_dir'/'FAKE';wd.mkdir(parents=True);csvp=wd/'checkpoint_metrics.csv'
 def rows(rs):
  with csvp.open('w') as f:
   w=csv.DictWriter(f,fieldnames=['step','raw_original.hqnr','rr_scc','rr_ergas','rr_sam']);w.writeheader();w.writerows(rs)
 def run_sel(provider=None):
  with patch.object(sys,'argv',['qrecon24_select.py','FAKE','--official']),contextlib.redirect_stdout(io.StringIO()):
   with patch.object(sel,'official_rr',provider):sel.main()
  return json.loads((wd/'results/qrecon24_target_selection.json').read_text())
 def provider(tag,wd,step,dev):return dict(scc=.989,ergas=2.,psnr=38.,sam=2.7,q8=.93,ssim=.98,mat='fixture.mat'),'official'
 rows([{'step':1010,'raw_original.hqnr':.9584999,'rr_scc':.999,'rr_ergas':1.5,'rr_sam':2.}, {'step':2020,'raw_original.hqnr':.9585,'rr_scc':.9,'rr_ergas':3.,'rr_sam':4.}]);o=run_sel(provider)
 rec('strict_threshold_official_rr',o['n_eligible']==1 and o['target']['step']==2020 and o['official'] and o['target']['numeric_pass_all6'])
 rows([{'step':1010,'raw_original.hqnr':.959,'rr_scc':.999,'rr_ergas':1.5,'rr_sam':2.}, {'step':2020,'raw_original.hqnr':.959,'rr_scc':.9,'rr_ergas':3.,'rr_sam':4.}])
 def partial(tag,wd,step,dev):return (None,'checkpoint missing') if step==1010 else provider(tag,wd,step,dev)
 o=run_sel(partial)
 rec('official_mode_mixes_proxy_candidate',o['target']['step']==1010 and o['target_feasible'] and not o['official'] and not o['target'].get('official_rr'),'defect_reproduction',selected_step=o['target']['step'],selected_rr_scc=o['target']['scc'],official=o['official'],notes=o['notes'])
 rec('selector_provenance_missing',not any('sha' in k for k in o) and not any('sha' in k for k in o['target']) and 'display_tie' not in str(o),'defect_reproduction',top_level_fields=sorted(o))
print('SUMMARY',len(R),sum(r['passed'] for r in R))
