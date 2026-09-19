"""Qualified interval union, never wall-clock/deadline accounting or parent+child sums."""
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

CAMPAIGN_ID='WV3_FH20R1_20260919_v1'
KINDS=('train','eval','calibration','diagnostic')


def epoch(value):
    if isinstance(value,(int,float)):
        result=float(value)
    else:
        stamp=dt.datetime.fromisoformat(str(value).replace('Z','+00:00'))
        if stamp.tzinfo is None: raise ValueError('Timing requires timezone-aware UTC timestamps')
        result=stamp.timestamp()
    if not math.isfinite(result): raise ValueError('Nonfinite timing')
    return result


def iso(value=None):
    return dt.datetime.fromtimestamp(time.time() if value is None else epoch(value),dt.timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''): h.update(part)
    return h.hexdigest()


def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else ({} if default is None else default)


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(temp,path)


def camp(root,server): return Path(root)/'work_dir/_fh20r1'/server


def union_seconds(intervals):
    merged=[]
    for start,end in sorted((epoch(a),epoch(b)) for a,b in intervals):
        if end<start: raise ValueError('Negative interval')
        if merged and start<=merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
        else: merged.append([start,end])
    return sum(b-a for a,b in merged)


def intervals(root,server):
    path=camp(root,server)/'active_intervals.jsonl'
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record_interval(root,server,*,interval_id,kind,start_utc,end_utc,run_id=None,evidence=None,**metadata):
    budget=read(camp(root,server)/'campaign_budget.json')
    if budget.get('campaign_id')!=CAMPAIGN_ID or not budget.get('actual_start_authorized'):
        raise ValueError('Cannot credit work before explicit FH20R1 start')
    if kind not in KINDS+('waiting',): raise ValueError('Nonqualified ledger category')
    start,end=epoch(start_utc),epoch(end_utc)
    if start<epoch(budget['started_at_utc']) or end<start or end>time.time()+5:
        raise ValueError('Prior-campaign/future/negative interval is not creditable')
    if not interval_id or not evidence: raise ValueError('Timing requires stable ID and committed evidence')
    row=dict(campaign_id=CAMPAIGN_ID,server_id=server,id=str(interval_id),kind=kind,
             start_utc=iso(start),end_utc=iso(end),seconds=end-start,run_id=run_id,evidence=evidence,**metadata)
    directory=camp(root,server);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.ledger.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        prior=next((r for r in intervals(root,server) if r['id']==row['id']),None)
        if prior:
            # A later fullstate may bind the same retained interval to newer
            # evidence. Its execution geometry/identity must remain unchanged.
            keys=('campaign_id','server_id','kind','start_utc','end_utc','run_id')
            if any(prior[k]!=row[k] for k in keys): raise ValueError('Timing ID collision/mutation')
            return False
        with (directory/'active_intervals.jsonl').open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return True


def ingest_timing(root,server,run):
    """Credit only segments bound to the exact latest valid full-state snapshot."""
    root=Path(root); wd=root/'work_dir'/run; path=wd/'meta/timing_committed.json'
    if not path.exists(): return 0
    doc=read(path)
    if doc.get('campaign_id')!=CAMPAIGN_ID or doc.get('run_id')!=run or not doc.get('valid_fullstate'):
        raise ValueError('Timing manifest is not a committed FH20R1 run')
    relative=Path(doc['full_state_identity_path'])
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('Invalid timing evidence path')
    identity_path=wd/relative; identity=read(identity_path)
    state_path=identity_path.parent/'training_state.pt'
    if (identity.get('training_state_sha256')!=doc.get('training_state_sha256') or
            digest(state_path)!=doc['training_state_sha256'] or
            identity.get('update')!=doc.get('committed_update') or
            identity.get('config_sha256')!=doc.get('config_sha256')):
        raise ValueError('Timing is not bound to valid saved fullstate bytes/update/config')
    evidence_sha=digest(path); receipt=camp(root,server)/'timing_evidence'/f'{evidence_sha}.json'
    if not receipt.exists(): write(receipt,doc)
    added=0
    for segment in doc['segments']:
        if segment['kind'] not in KINDS: raise ValueError('Unqualified training segment')
        start,end=epoch(segment['start_utc']),epoch(segment['end_utc'])
        if abs((end-start)-float(segment['seconds']))>max(.01,(end-start)*1e-5):
            raise ValueError('Segment seconds disagree with timestamps')
        if int(segment.get('update_to',0))>int(doc['committed_update']):
            raise ValueError('Unsaved optimizer progress cannot be credited')
        added+=record_interval(root,server,interval_id=f'{run}:{segment["id"]}',kind=segment['kind'],
            start_utc=start,end_utc=end,run_id=run,evidence=str(receipt.relative_to(root)),
            update_from=segment.get('update_from'),update_to=segment.get('update_to'))
    return added


def report(root,server,now=None):
    budget=read(camp(root,server)/'campaign_budget.json'); rows=intervals(root,server)
    effective=union_seconds((r['start_utc'],r['end_utc']) for r in rows if r['kind'] in KINDS)
    result=dict(campaign_id=CAMPAIGN_ID,server_id=server,effective_seconds=effective,
                effective_hours=effective/3600,minimum_effective_hours=20.,interval_count=len(rows),
                wall_elapsed_seconds=max(0.,(time.time() if now is None else epoch(now))-epoch(budget['started_at_utc'])))
    for kind in KINDS+('waiting',):
        result[kind+'_seconds']=union_seconds((r['start_utc'],r['end_utc']) for r in rows if r['kind']==kind)
    result['category_totals_may_overlap']=True
    result['counting_policy']='union_of_qualified_committed_intervals; not_sum_of_category_totals'
    return result
