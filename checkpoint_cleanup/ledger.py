"""Hash-bound retirement evidence; missing weights are never loadable models."""
from pathlib import Path
import hashlib,json,os,stat as st,datetime
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'work_dir/_operator_storage/20261106_target_cleanup_v1'
SCHEMA='PANDA_USER_WEIGHT_RETIREMENT_V1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def local(p):
 p=Path(p)
 if str(p).startswith('/workspace/'):p=ROOT/p.relative_to('/workspace')
 return p if p.is_absolute() else ROOT/p
def rel(p):return str(local(p).relative_to(ROOT))
def safe_file(p):
 p=local(p)
 if p.suffix!='.pt' or ROOT/'work_dir' not in p.parents or '..' in p.parts:raise ValueError('Unsafe checkpoint path: '+str(p))
 if any(x.is_symlink() for x in [p,*p.parents]):raise ValueError('Symlink checkpoint path: '+str(p))
 if p.exists() and not st.S_ISREG(p.stat().st_mode):raise ValueError('Not regular file: '+str(p))
 return p
def fingerprint(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,links=s.st_nlink)
def event(folder,v):
 with (Path(folder)/'events.jsonl').open('a') as f:json.dump(dict(at=now(),**v),f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
class Ledger:
 def __init__(self,folder=DEFAULT,*,preview=False):
  self.folder=Path(folder);self.manifest=read(self.folder/'manifest.json')
  expected=self.manifest['manifest_sha256'];plain={k:v for k,v in self.manifest.items() if k!='manifest_sha256'}
  if self.manifest['schema']!=SCHEMA or digest(plain)!=expected:raise ValueError('Retirement manifest changed')
  self.entries={r['path']:r for r in self.manifest['delete']};self.preview=preview;self.verified=set();self.deleted=set()
  events=self.folder/'events.jsonl'
  if events.exists():
   for line in events.read_text().splitlines():
    e=json.loads(line)
    if e.get('event')=='DELETED' and e.get('manifest_sha256')==expected:
     r=self.entries.get(e['path'])
     if not r or r['file_sha256']!=e['file_sha256']:raise ValueError('Retirement journal identity mismatch')
     self.deleted.add(e['path'])
 def evidence(self,path,expected_sha):
  path=safe_file(path);key=rel(path);r=self.entries.get(key)
  if r is None or r['file_sha256']!=expected_sha:raise ValueError('Missing checkpoint is not an authenticated retirement: '+key)
  if not self.preview and key not in self.deleted:raise ValueError('Checkpoint deletion not committed: '+key)
  if key not in self.verified:
   for proof in r['evidence']:
    if sha(local(proof['path']))!=proof['sha256']:raise ValueError('Retained evidence changed: '+proof['path'])
   self.verified.add(key)
  return {'status':'PLANNED_RETIREMENT' if self.preview else 'REMOVED_BY_USER_POLICY','weights_currently_available':path.exists(),
   'original_file_sha256':expected_sha,'reason':r['reason'],'manifest_sha256':self.manifest['manifest_sha256'],
   'manifest':rel(self.folder/'manifest.json'),'prior_weight_sha256_verified_at':r['verified_at'],
   'metrics_and_original_records_preserved':True,'retraining_or_loading_from_tombstone_supported':False}
