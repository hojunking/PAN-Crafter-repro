import argparse,json
from pathlib import Path
from .ledger import DEFAULT,read,write,now

def main():
 p=argparse.ArgumentParser(description='Explicit s1 checkpoint retirement and evidence-preserving republication')
 p.add_argument('action',choices=['prepare','project','publish','confirm-prepublication','apply','verify'])
 p.add_argument('--folder',type=Path,default=DEFAULT);p.add_argument('--family');p.add_argument('--preview',action='store_true');p.add_argument('--apply',action='store_true');a=p.parse_args()
 if a.action=='prepare':
  from .prepare import prepare
  m=prepare(a.folder);result={'files':len(m['delete']),'manifest_sha256':m['manifest_sha256']}
 elif a.action in ('project','publish'):
  if not a.family:p.error('--family required')
  from .publication import project_or_publish
  result=project_or_publish(a.family,a.folder,preview=a.preview,apply=a.apply and a.action=='publish')
 elif a.action=='confirm-prepublication':
  m=read(a.folder/'manifest.json')
  proofs=[read(a.folder/('prepublication_'+f+'.json')) for f in m['campaigns']]
  if any(r['status']!='VERIFIED' or r['manifest_sha256']!=m['manifest_sha256'] for r in proofs):raise ValueError('Publication preflight incomplete')
  result={'status':'VERIFIED','at':now(),'manifest_sha256':m['manifest_sha256'],'families':list(m['campaigns'])};write(a.folder/'prepublication_verified.json',result)
 elif a.action=='apply':
  if not a.apply:p.error('Actual removal requires --apply')
  from .apply import apply
  result=apply(a.folder)
 else:
  from .ledger import Ledger,local,sha
  l=Ledger(a.folder);m=l.manifest
  for r in m['delete']:
   if local(r['path']).exists():raise ValueError('Requested deletion not complete: '+r['path'])
   l.evidence(r['path'],r['file_sha256'])
  posts=[]
  for family in m['campaigns']:
   pre=read(a.folder/('prepublication_'+family+'.json'));post=read(a.folder/('postpublication_'+family+'.json'))
   if pre['metrics_identity_sha256']!=post['metrics_identity_sha256'] or post['manifest_sha256']!=m['manifest_sha256']:raise ValueError('Publication metrics or identities changed')
   posts.append(family)
  result={'status':'VERIFIED','at':now(),'removed_files':len(m['delete']),'bytes':sum(r['fingerprint']['size'] for r in m['delete']),
   'retained_evidence_verified':len(l.verified),'republication_metrics_unchanged':posts,'manifest_sha256':m['manifest_sha256']};write(a.folder/'verified.json',result)
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
