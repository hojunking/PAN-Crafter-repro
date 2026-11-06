"""Apply only hash-verified, user-authorized exact weight paths."""
from pathlib import Path
import shutil
from .ledger import *
def apply(folder=DEFAULT):
 ledger=Ledger(folder);m=ledger.manifest;folder=Path(folder)
 # Publication replay must be tested with authenticated tombstones first.
 ready=read(folder/'prepublication_verified.json')
 if ready['manifest_sha256']!=m['manifest_sha256'] or ready['status']!='VERIFIED':raise ValueError('Publication preflight missing')
 for r in m['runs']:
  for p in r['proofs']:
   if sha(local(p['path']))!=p['sha256']:raise ValueError('Run/config state changed; review again')
 free=shutil.disk_usage(ROOT).free;count=0;size=0
 for r in m['delete']:
  p=safe_file(r['path']);key=r['path']
  if key in ledger.deleted:
   if p.exists():raise ValueError('Retired path unexpectedly recreated: '+key)
   continue
  if not p.exists():raise ValueError('Uncommitted missing weight: '+key)
  if fingerprint(p)!=r['fingerprint'] or sha(p)!=r['file_sha256'] or fingerprint(p)!=r['fingerprint']:raise ValueError('Changed weight before unlink: '+key)
  for proof in r['evidence']:
   if sha(local(proof['path']))!=proof['sha256']:raise ValueError('Changed retained evidence')
  common=dict(path=key,file_sha256=r['file_sha256'],manifest_sha256=m['manifest_sha256'])
  event(folder,dict(event='DELETE_INTENT',**common))
  p.unlink()
  event(folder,dict(event='DELETED',**common));count+=1;size+=r['fingerprint']['size']
  if count%100==0:print('DELETED',count,'files',flush=True)
 result={'status':'APPLIED','at':now(),'manifest_sha256':m['manifest_sha256'],'files_deleted_this_invocation':count,
  'logical_bytes_deleted_this_invocation':size,'available_bytes_before':free,'available_bytes_after':shutil.disk_usage(ROOT).free,
  'metadata_deleted':False,'remote_files_touched':False}
 write(folder/'applied.json',result)
 for family,c in m['campaigns'].items():
  write(local(c['root'])/'CHECKPOINT_RETIREMENT.json',{'manifest':rel(folder/'manifest.json'),'manifest_sha256':m['manifest_sha256'],
   'at':now(),'family':family,'original_training_status_unchanged':True,'resume_deleted_runs_forbidden':True,
   'republication_command':f'python -m checkpoint_cleanup publish --family {family} --apply',
   'meaning':'Metrics remain original; removed weights are unavailable. Use retirement-aware publisher, not old weight-required entrypoint.'})
 return result
