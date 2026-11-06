"""Retirement-aware wrappers for existing publishers; scientific sources stay pinned."""
from contextlib import contextmanager
from importlib import import_module
from copy import deepcopy
from pathlib import Path
import json
from .ledger import ROOT,DEFAULT,Ledger,local,rel,read,write,now,digest

def modules(family):
 if family=='panda_wv3_main_rgb':
  return import_module(family+'.publication'),import_module('panda_wv3_main_reporting.publication'),'wv3'
 if family in ('panda_wv3_ablation','panda_wv3_ablation_rgb'):
  m=import_module(family+'.publication');return m,m,'wv3'
 if family=='panda_qb_refit':import_module(family+'.adapter').install();base='panda_sensor_focus'
 elif family=='panda_qb_structure':import_module(family+'.adapter').install('s1');base='panda_sensor_focus'
 else:base=family
 return import_module(base+'.publication'),import_module(base+'.best_publication'),'legacy'

@contextmanager
def adapted(family,ledger):
 artifact_module,pub,kind=modules(family)
 name='artifact' if kind=='wv3' else '_artifact';original=getattr(artifact_module,name)
 def retired(meta,path,run=None,step=None):
  entry=ledger.entries.get(rel(path))
  if entry is None or not (ledger.preview or not path.exists()):return None
  if run is not None and (entry['run_id']!=run or entry['step']!=step):raise ValueError('Retirement/run mismatch')
  return ledger.evidence(path,meta['file_sha256'])
 if kind=='wv3':
  def replacement(meta):
   p=local(meta['path']);proof=retired(meta,p)
   if proof is None:return original(meta)
   if meta['step']!=ledger.entries[rel(p)]['step']:raise ValueError('Retirement step mismatch')
   return None
 else:
  def replacement(candidate,root):
   extra=candidate['record']['source']['extra'];cfg=extra['configuration']
   if cfg['phase']=='donor':return original(candidate,root)
   a=extra.get('artifact',{});raw=a.get('checkpoint_path')
   if not raw:return original(candidate,root)
   p=local(raw);proof=retired(a,p,cfg['run_id'],candidate['step'])
   if proof is None:return original(candidate,root)
   return {**a,'checkpoint_path':str(p),'verification':'FILE_SHA256_VERIFIED_BEFORE_USER_RETIREMENT',
       'weights_currently_available':proof['weights_currently_available'],'retirement':proof}
 setattr(artifact_module,name,replacement)
 original_project=pub.project
 def project(*args,**kwargs):
  value=original_project(*args,**kwargs);index=pub.INDEX if hasattr(pub,'INDEX') else pub.original.INDEX
  entries={(e['run_id'],e['step']):e for e in ledger.entries.values() if Path(e['path']).name==f"step_{e['step']:07d}.pt"}
  for i,row in enumerate(value['rows']):
   entry=entries.get((row[index['Run_ID']],row[index['Selected_step']]))
   if entry is None or not (ledger.preview or not local(entry['path']).exists()):continue
   proof=ledger.evidence(entry['path'],entry['file_sha256'])
   note=json.loads(row[index['Original description / notes']]);note['checkpoint_retention']=proof
   row[index['Original description / notes']]=json.dumps(note,sort_keys=True,separators=(',',':'),ensure_ascii=False)
   note2=json.loads(value['notes'][i]);note2['checkpoint_retention']=proof
   value['notes'][i]=json.dumps(note2,sort_keys=True,separators=(',',':'),ensure_ascii=False)
  # Existing row identity, metrics, selector, precision and cost fields stay exact.
  value['view_hash']=digest({'rows':value['rows'],'notes':value['notes']})
  return value
 pub.project=project
 try:yield pub,kind
 finally:pub.project=original_project;setattr(artifact_module,name,original)

def project_or_publish(family,folder=DEFAULT,*,preview=False,apply=False):
 ledger=Ledger(folder,preview=preview);campaign=ledger.manifest['campaigns'][family];root=local(campaign['root']);server=campaign['server']
 if preview and apply:raise ValueError('Cannot publish a simulated retirement')
 with adapted(family,ledger) as (pub,kind):
  # Compute a complete view, including source-record validation and measured costs.
  v=pub.project(root,server,**({'require_cost':True} if kind=='legacy' else {}))
  if v['conflicts']:raise ValueError('Publication conflicts: '+str(v['conflicts'][:3]))
  index=pub.INDEX if hasattr(pub,'INDEX') else pub.original.INDEX
  for row in v['rows']:
   if kind=='wv3':pub.verify_owned(row,v,server)
   else:pub.verify_owned(row,v)
  # Compare every existing value except the deliberately enriched notes column.
  numeric=[[cell for i,cell in enumerate(row) if i!=index['Original description / notes']] for row in v['rows']]
  result={'status':'VERIFIED','at':now(),'family':family,'preview':preview,'rows':len(v['rows']),
   'metrics_identity_sha256':digest(numeric),'manifest_sha256':ledger.manifest['manifest_sha256'],'retirement_evidence_checked':len(ledger.verified),'conflicts':[]}
  name=('pre' if preview else 'post')+'publication_'+family
  write(Path(folder)/(name+'.json'),result)
  write(Path(folder)/(name+'_projection.json'),{k:v[k] for k in ('destination','rows','notes','policy','view_hash')})
  if apply:
   if kind=='wv3':result['cloud']=pub.publish(root,server,apply=True)
   else:
    main=import_module(pub.__name__+'.__main__')
    # __main__ imports project by value; install after wrapping it above.
    if main.project is not pub.project:main.project=pub.project
    code=main.publish(root,server,apply=True,watch=False)
    if code:raise ValueError('Retirement-aware publication failed')
    result['cloud']='PUBLISHED_AND_READBACK_VERIFIED'
  return result
