"""Warm access metadata and retain content/replacement checks around legacy loads."""
from datetime import datetime,timezone
from functools import wraps
import hashlib
import importlib
import json
from pathlib import Path

POLICY='DONOR_READ_STABLE_IDENTITY_AND_SHA256_V1'
FIELDS=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')

def fingerprint(path):
    s=path.stat();return tuple(getattr(s,k) for k in FIELDS)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def protect(loader,audit_path=None):
    @wraps(loader)
    def load(cfg,receipt,registered,root,pins,*,device):
        selected=receipt['selected'];path=Path(root)/selected['checkpoint_path']
        expected=selected['file_sha256'];before=fingerprint(path);atime=path.stat().st_atime_ns
        # This read refreshes relatime before the legacy loader snapshots stat.
        # Hash and identity are checked before invoking any deserializer.
        if sha(path)!=expected or fingerprint(path)!=before:
            raise ValueError('Donor content/replacement changed before guarded load')
        retried=False
        try:
            donor=loader(cfg,receipt,registered,root,pins,device=device)
        except ValueError as exc:
            if str(exc)!='Donor checkpoint changed while loading':raise
            if fingerprint(path)!=before or sha(path)!=expected or path.stat().st_atime_ns==atime:
                raise
            # Only a demonstrated access-time-only transition is retryable.
            retried=True
            donor=loader(cfg,receipt,registered,root,pins,device=device)
        if fingerprint(path)!=before or sha(path)!=expected or fingerprint(path)!=before:
            raise ValueError('Donor content/replacement changed during guarded load')
        if audit_path is not None:
            record={'policy':POLICY,'at':datetime.now(timezone.utc).isoformat(),
                'checkpoint':str(path),'sha256':expected,'file_identity':dict(zip(FIELDS,before)),
                'atime_changed':path.stat().st_atime_ns!=atime,'legacy_retry':retried,
                'source_loader':loader.__module__+'.'+loader.__name__,
                'model_loss_optimizer_schedule_unchanged':True}
            target=Path(audit_path);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
        return donor
    return load

def install(package,server):
    if package not in ('panda_wv3_s2abl5','panda_sensor_main50'):
        raise ValueError('Unregistered campaign')
    plan=importlib.import_module(package+'.plan')
    root=plan.campaign_root(server);audit=root/'donor_read_guard/loads.jsonl'
    # Resolve imported aliases before replacing any module attribute.
    modules=[importlib.import_module(name) for name in
        ('panda_unified_campaign.assets','panda_sensor_eq40.assets',package+'.assets')]
    wrapped={}
    for module in modules:
        original=getattr(module,'_load_selected',None)
        if original is not None:
            if original not in wrapped:wrapped[original]=protect(original,audit)
            module._load_selected=wrapped[original]
    return root
