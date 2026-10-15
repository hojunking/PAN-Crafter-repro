"""QEGX plan/config and operational edge cases. All mutations are temporary."""
import ast,copy,contextlib,io,json,os,pathlib,random,subprocess,sys,tempfile,time,types
from unittest.mock import patch
from audit_common import *
os.environ.update(ENV);os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch,yaml,numpy as np
torch.set_num_threads(2)
from tools import gen_pakd50_configs as G, campaign_gate as C
from kdv.resume import EpochState,begin_epoch
from kdv import edge_gate as EG
R=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checks=[])
def record(name,ok,kind='conformance',**kw):
    R['checks'].append(dict(name=name,passed=bool(ok),kind=kind,**kw));(OUT/'ops_results.json').write_text(json.dumps(R,indent=2));print(name,ok,json.dumps(kw),flush=True)
def defect(name,ok,**kw):record(name,ok,'defect_reproduction',**kw)
expected={'s3':[(c,2026,'v2') for c in ('J0','JQ','QE50','XJ','QX50','J_R3_NOEDGE','QEC3','QES','J_QB005','QE50_B005')]+[(c,4321,'v2') for c in ('J0','JQ','XJ','QE50','QX50')],
          's4':[(c,1234,'v1') for c in ('QE50','LF0','LFQ','LFQE50','LFX','JE0','QER50','QERS')]+[(c,3407,'v1') for c in ('J0','JQ','QE50','JE0','QER50','QERS')]}
cfgs={}
for srv,entries in expected.items():
    names=[G.run_name(c,s,v,arch='W104_D121') for c,s,v in entries]
    queue=[l for l in (ROOT/f'config/queues/qegx_{srv}.txt').read_text().splitlines() if l and not l.startswith('#')]
    record('queue_'+srv,queue==names and [x for x in G.priority_for(srv) if G.branch_for(srv,x)=='QEGX']==names,n=len(names))
    with tempfile.TemporaryDirectory(prefix='qegx-config-audit-') as td:
        G.generate(srv,names,td)
        for name,(case,seed,ver) in zip(names,entries):
            c=yaml.safe_load((ROOT/'config'/(name+'.yaml')).read_text());k=c['kdv'];g=yaml.safe_load((pathlib.Path(td)/(name+'.yaml')).read_text());cfgs[name]=c
            beta=0.05 if case in ('J_QB005','QE50_B005') else (0 if case in ('J0','LF0','XJ','QX50','LFX') else .1)
            rec=k['rec'];b=k['budget'];rt=k.get('edge_route') or {};eg=k.get('edge_gate') or {}
            record('config_'+name,c==g and c['seed']==seed and c['num_iter']==50000 and c['batch_size']==48 and c['num_warmup']==100 and c['model_args']['hidden_size']==104 and c['model_args']['depth']==[1,2,1] and float(c['learning_rate'])==1e-4 and k['aligner_lr']==1e-5 and float(c['weight_decay'])==.01 and float(rec.get('kd_weight',0))==beta and k['campaign_id']==G.QEGX_CAMPAIGN_ID and k['exact_resume'] and b['required'] and b['ledger']==G.QEGX_LEDGER and 'training_deadline' not in b and b['time_policy']['mode']=='no_hard_limit' and k['candidate_grid_id']=='GRID1010_50K_v1' and k['teacher']['expected_sha256']=='16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32' and k['teacher']['bridge'] and (rec.get('tau',.012463942170143127)==.012463942170143127) and (not k['stat']['enabled'] or k['stat']['outer_weight']==.09075170336956798))
    total=sum(G.reservation_for(srv,n)['reservation_h'] for n in names)
    record('reservation_'+srv,abs(total-({'s3':25.204,'s4':26.2803333333333}[srv]))<1e-8,hours=total)
record('server_alias',G.server_id('s3(5090)')=='s3')

# Actual epoch checkpoint boundary: 9714 // 48 = 202; epoch-25 => step 5050.
loader=torch.utils.data.DataLoader(torch.arange(9714),batch_size=48,shuffle=True,num_workers=4,drop_last=True)
es=EpochState();es.set(torch.get_rng_state(),4848,len(loader))
try:begin_epoch(loader,es,5050,True);err=None
except RuntimeError as e:err=str(e)
defect('epoch_25_checkpoint_cannot_resume',err is not None,epoch_start=4848,checkpoint_step=5050,batches=len(loader),error=err)
# Mid-epoch remains supported for normal four-worker input ordering.
torch.manual_seed(2026);es=EpochState();it,_,_=begin_epoch(loader,es,0,False)
for _ in range(7):next(it)
sd=copy.deepcopy(es.state_dict());post=torch.get_rng_state();wanted=[next(it).tolist() for _ in range(3)];del it
torch.set_rng_state(post);restored=EpochState();restored.load_state_dict(sd);it,skip,info=begin_epoch(loader,restored,7,True);got=[next(it).tolist() for _ in range(3)];del it
record('mid_epoch_resume_4workers',wanted==got and skip==7)

# Completion guard: duplicate CSV rows and absent identity are accepted.
with tempfile.TemporaryDirectory(prefix='qegx-complete-') as td:
    p=pathlib.Path(td);rd=p/'work_dir'/'PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v1';(rd/'results').mkdir(parents=True);(rd/'last').mkdir()
    for x in ('reduced_best_hqnr.mat','full_best_hqnr.mat'):(rd/'results'/x).touch()
    (rd/'last/model.safetensors').touch();(rd/'last_meta.json').write_text('{"step":50000}')
    (rd/'checkpoint_metrics.csv').write_text('step\n'+'50000\n'*45)
    (rd/'kdv_config_resolved.json').write_text('{"resumed_nonexact":true,"exact_resume":false,"evaluator_hash":"WRONG_FIXTURE"}')
    with patch.object(G,'ROOT',str(p)):
        v=G.verified_complete(rd.name)
    defect('verified_complete_accepts_missing_identity_and_duplicate_grid',v['ok'],verification=v)

# Actual switch in isolated repository: no live runner/start/cron/weights involved.
for srv in ('s3','s4'):
  with tempfile.TemporaryDirectory(prefix='qegx-switch-'+srv+'-') as td:
    p=pathlib.Path(td);fixture(p,srv)
    (p/'tools/pakd50_unit_tests.py').write_text("print('ALL OK: unit suite audited separately')\n")
    (p/'tools/qedge9_cue.py').write_text("print('verify/pilot stub: core asset tested separately')\n")
    before={str(x.relative_to(p)):x.read_bytes() for x in (p/'work_dir').rglob('*') if x.is_file()}
    dry=subprocess.run(['bash','tools/qegx_switch.sh','--dry-run'],cwd=p,env={**ENV,'PYTHON':PY},capture_output=True,text=True,timeout=90)
    (OUT/f'switch_{srv}_dry.log').write_text(dry.stdout+dry.stderr)
    controls=[G.MANDATORY_FILE,G.QEGX_MANDATORY_FILE,G.RESERVATION_FILE,G.EXTRA_PRIORITY_FILE,'work_dir/cases_deadline.txt','work_dir/cases_queue.txt']
    record('switch_dry_'+srv,dry.returncode==0 and all((p/x).read_bytes()==before[x] if x in before else not (p/x).exists() for x in controls),rc=dry.returncode,note='Diagnostic logs/directories may be written; operational control files unchanged.')
    if srv=='s4':
        old=G.run_name('JQ',3407);(p/G.EXTRA_PRIORITY_FILE).write_text(old+'\n')
        # Legacy extra is marked complete to avoid irrelevant old config byte differences.
        rr=p/'work_dir'/old/'results';rr.mkdir(parents=True)
        for f in ('reduced_best_hqnr.mat','full_best_hqnr.mat'):(rr/f).touch()
        (p/'tools/campaign_start.sh').write_text('#!/bin/bash\ntouch work_dir/audit_start_called\nexit 0\n')
        (p/'tools/_watchdog.sh').write_text('#!/bin/bash\nexit 0\n')
        (p/'tools/qegx_waiter.sh').write_text('#!/bin/bash\nexit 0\n')
        for f in ('campaign_start.sh','_watchdog.sh','qegx_waiter.sh'):(p/'tools'/f).chmod(0o755)
        z=subprocess.run(['bash','tools/qegx_switch.sh','--refresh-controls'],cwd=p,env={**ENV,'PYTHON':PY},capture_output=True,text=True,timeout=90)
        (OUT/'switch_s4_refresh.log').write_text(z.stdout+z.stderr)
        with patch.object(G,'ROOT',str(p)):extras=G.extra_priority()
        mandatory=(p/G.QEGX_MANDATORY_FILE).read_text() if (p/G.QEGX_MANDATORY_FILE).exists() else ''
        archived=list((p/'work_dir/_pakd50').glob('extra_priority.pre_qegx*'))
        defect('refresh_controls_lost_when_archiving_old_extra',z.returncode==0 and not extras and all(x not in mandatory for x in G.QEGX_REFRESH_CONTROLS['s4']) and any(all(x in f.read_text() for x in G.QEGX_REFRESH_CONTROLS['s4']) for f in archived),rc=z.returncode,extra=extras,new_mandatory_count=len([l for l in mandatory.splitlines() if l and not l.startswith('#')]),refresh_tags=G.QEGX_REFRESH_CONTROLS['s4'])

# Run the exact run_case shell function with a missing c_E3 and a failing trainer stub.
src=(ROOT/'tools/_run_cases.sh').read_text();fn=src[src.index('run_case(){'):src.index('\ni=0\n')]
with tempfile.TemporaryDirectory(prefix='qegx-qec-direct-') as td:
    p=pathlib.Path(td);(p/'tools').mkdir();(p/'work_dir').mkdir()
    (p/'tools/smoke_cases.py').write_text("print('smoke pass')\n")
    (p/'tools/run.sh').write_text('#!/bin/bash\necho invoked >> trainer_calls.txt\nexit 1\n');(p/'tools/run.sh').chmod(0o755)
    sh='''#!/bin/bash
set -u
LEDGER=work_dir/cases_failed.txt
FAILED=""; DEADLINE_SKIPPED=""; MISSING_CFG=""
ledgered(){ return 1; }; complete(){ return 1; }; past_deadline(){ return 1; }; latest_ckpt(){ return 0; }
'''+fn+'\nrun_case PAKD50_QEC3_W104_D121_WV3_T0_S2026_FRESH50_v2 audit\n'
    (p/'exercise.sh').write_text(sh)
    z=subprocess.run(['bash','exercise.sh'],cwd=p,env={**ENV,'PATH':str(pathlib.Path(PY).parent)+os.pathsep+ENV['PATH']},capture_output=True,text=True,timeout=20)
    (OUT/'missing_ce_runner.log').write_text(z.stdout+z.stderr)
    calls=(p/'trainer_calls.txt').read_text().count('invoked') if (p/'trainer_calls.txt').exists() else 0
    with patch.object(C,'ROOT',str(p)):term=C.terminal('PAKD50_QEC3_W104_D121_WV3_T0_S2026_FRESH50_v2')
    defect('direct_queue_missing_ce_is_failed_terminal',calls==2 and term,trainer_attempts=calls,terminal=term,real_ce_exists=False,note='Actual run_case; smoke/run stubs isolate admission. Actual const loader raises FileNotFoundError when c_E3 is absent.')

# Current new ledger is wired into reservations (previous QEDGE9 issue repaired).
with tempfile.TemporaryDirectory(prefix='qegx-ledger-') as td:
    p=pathlib.Path(td);(p/'work_dir/_qegx_budget').mkdir(parents=True);r=G.run_name('QER50',3407,arch='W104_D121')
    (p/G.QEGX_LEDGER).write_text(json.dumps({'entries':{r:dict(kind='run',status='FINISHED',hours_total=2.75)}}))
    with patch.object(G,'ROOT',str(p)):m=G.measured_hours_all('s4');v=G.reservation_for('s4',r,m)
    record('new_ledger_measurement_used',m.get('QER50@W104_D121')==2.75 and abs(v['reference_train_h']-2.75)<1e-8)
R['summary']={kind:dict(total=sum(x['kind']==kind for x in R['checks']),passed=sum(x['kind']==kind and x['passed'] for x in R['checks'])) for kind in ('conformance','defect_reproduction')};R['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');(OUT/'ops_results.json').write_text(json.dumps(R,indent=2));print(R['summary'],flush=True)
