"""Build and hash-verify exact file lists from the read-only s1 audit."""
from pathlib import Path
from collections import Counter
import json
from .ledger import ROOT,DEFAULT,SCHEMA,read,write,sha,rel,local,now,digest,safe_file,fingerprint
AUDIT=ROOT/'results_log/assets/2026-11-06-weight-retention-audit/audit.json'
FAMILIES={'panda_wv3_main_rgb','panda_sensor_focus','panda_qb_refit','panda_qb_structure','panda_qb40','panda_qb_rre','panda_s1loss9','panda_tune30','panda_wv3_ablation'}
TERMINAL={'COMPLETED','EARLY_STOPPED_HQNR'}
def classify(r,cfg):
 if r['status'] not in TERMINAL or r['phase']!='stage2':return None
 if r['family']=='panda_wv3_ablation':
  h=cfg['horizon']
  if h not in (30000,50000):raise ValueError('Unspecified ablation horizon')
  return ('ABLATION_EARLY_STEPS',20000 if h==30000 else 30000)
 if r['family'] in FAMILIES and r['joint_pass_count']==0:return ('COMPLETED_MAIN_OR_QB_NO_JOINT_TARGET',None)
 return None

def prepare(folder=DEFAULT):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'manifest.json').exists():raise ValueError('Immutable manifest already exists; use verify/apply')
 audit=read(AUDIT);protected={};seen=set();rows=[];campaigns={};run_proofs=[]
 # Explicit paper sources, not every candidate listed in copied run state files.
 for p in (ROOT/'results_log/paper_figs').rglob('*.json'):
  if p.name not in {'manifest.json','provenance.json','panels.json','meta.json','ours_test_rre.json'}:continue
  def visit(x):
   if isinstance(x,dict):
    for k,v in x.items():
     if isinstance(v,str) and v.endswith('.pt') and ('checkpoint' in k or k=='original_absolute_path'):
      try:protected[rel(v)]='EXPLICIT_PAPER_OR_FIGURE_SOURCE'
      except ValueError:pass
     else:visit(v)
   elif isinstance(x,list):
    for v in x:visit(v)
  visit(read(p))
 selected=[]
 for r in audit['recent_runs']:
  if r['family'] not in FAMILIES:continue
  directory=local(r['directory']);cfgpath=directory/'config.json'
  cfg=read(cfgpath);rule=classify(r,cfg)
  if rule is None:continue
  campaign=directory
  while campaign.name!='runs':campaign=campaign.parent
  campaign=campaign.parent;statepath=campaign/'campaign_state.json';state=read(statepath)
  live=next(x for x in state['runs'] if x['run_id']==r['run_id'])
  if live['status']!=r['status']:raise ValueError('Run status changed since audit: '+r['run_id'])
  if r['phase']=='stage2' and cfg.get('phase',cfg.get('stage')) not in ('stage2',):raise ValueError('Stage1 cannot be target-pruned')
  actualpaths=sorted(directory.glob('evaluation_*.json'))
  if len(actualpaths)!=r['evaluations']:raise ValueError('Evaluation set changed: '+r['run_id'])
  evs={}
  for old in r['evaluation_checkpoints']:
   p=directory/f"evaluation_{old['step']:07d}.json"
   if sha(p)!=old['record_sha256']:raise ValueError('Evaluation changed: '+str(p))
   e=read(p);h=e['fr']['metrics']['HQNR'];er=e['rr']['metrics']['ERGAS']
   ds=r['dataset'];target=audit['targets_current_user_definition'][ds]
   if h!=old['HQNR'] or er!=old['ERGAS']:raise ValueError('Audit metric mismatch')
   if rule[0]=='COMPLETED_MAIN_OR_QB_NO_JOINT_TARGET' and h>=target['HQNR'] and er<=target['ERGAS']:
    raise ValueError('Cannot retire a successful run as failed')
   evs[old['step']]=(p,old)
  proofs=[{'path':rel(p),'sha256':sha(p)} for p in [cfgpath,statepath]]
  run_proofs.append({'run_id':r['run_id'],'status':r['status'],'rule':rule[0],'cutoff_inclusive':rule[1],'proofs':proofs,
   'evaluations':r['evaluations'],'joint_pass_count':r['joint_pass_count'],'directory':r['directory']})
  campaigns[r['family']]={'root':rel(campaign),'server':'s1'}
  for p in sorted(list((directory/'models').glob('*.pt'))+list((directory/'resume').glob('*.pt'))):
   key=rel(p)
   if key in seen:raise ValueError('Duplicate checkpoint path')
   seen.add(key);safe_file(p);meta=read(p.with_suffix('.json'));a=meta.get('artifact',meta)
   step=a.get('step',a.get('completed_update'));expected=a.get('file_sha256')
   if not isinstance(step,int) or not expected:raise ValueError('Checkpoint lacks identity: '+key)
   if rule[1] is not None and step>rule[1]:continue
   if key in protected:continue
   ev=evs.get(step)
   evidence=proofs[:1]+[{'path':rel(p.with_suffix('.json')),'sha256':sha(p.with_suffix('.json'))}]
   if ev:evidence.append({'path':rel(ev[0]),'sha256':sha(ev[0])})
   rows.append({'path':key,'family':r['family'],'run_id':r['run_id'],'step':step,'reason':rule[0],
    'file_sha256':expected,'fingerprint':fingerprint(p),'evidence':evidence,
    'metrics':None if ev is None else {k:ev[1][k] for k in ('HQNR','ERGAS','joint_pass')},'verified_at':None})
 # Structural exclusions protect all Stage1 donors, all successful runs and current W128.
 if any('/panda_wv3_w128/' in x['path'] or x['run_id'].startswith('WMRGB_WV3_S1_S1_') for x in rows):raise ValueError('Protected active/donor path')
 write(folder/'preview.json',{'created_at':now(),'runs':run_proofs,'candidate_count':len(rows),'bytes':sum(x['fingerprint']['size'] for x in rows),'by_family':dict(Counter(x['family'] for x in rows)),'protected_paper_paths':protected,'mode':'NOT_APPLIED'})
 print('PREVIEW',len(rows),'files',sum(x['fingerprint']['size'] for x in rows)/2**30,'GiB',flush=True)
 # Hash all bytes before any deletion; retained sidecar/evaluation hashes bind evidence.
 for i,r in enumerate(rows,1):
  p=local(r['path'])
  if fingerprint(p)!=r['fingerprint'] or sha(p)!=r['file_sha256'] or fingerprint(p)!=r['fingerprint']:raise ValueError('Weight identity mismatch: '+r['path'])
  r['verified_at']=now()
  if i%100==0:print('HASH_VERIFIED',i,'/',len(rows),flush=True)
 for proof in run_proofs:
  for p in proof['proofs']:
   if sha(local(p['path']))!=p['sha256']:raise ValueError('Run changed during preparation')
 value={'schema':SCHEMA,'created_at':now(),'authorization':'User 2026-11-06: remove all failed main/QB case weights; ablation30K remove <=20K; 50K remove <=30K after donor review.',
  'thresholds':audit['targets_current_user_definition'],'scope':'Audited s1 completed or explicitly early-stopped Stage2 runs only; all Stage1, successful runs, active runs and paper-designated assets protected.',
  'source_audit_sha256':sha(AUDIT),'campaigns':campaigns,'runs':run_proofs,'protected_paper_paths':protected,
  'delete':rows,'deleted_metadata':False,'ablation_50k_local_runs':0,'remote_files_touched':False}
 value['manifest_sha256']=digest(value);write(folder/'manifest.json',value)
 print('MANIFEST_READY',value['manifest_sha256'],flush=True)
 return value
