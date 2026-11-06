"""Audit existing GF2 weights first; replay only missing early outputs, then compare."""
import argparse
import csv
from pathlib import Path
import signal
import subprocess
import sys
import time
from panda_sensor_visuals.common import ROOT,read,write,sha
from . import plan as P


def summarize():
    rows=[]
    for path in sorted((ROOT/P.OUTPUT/'exports').glob('*/step_*/manifest.json')):
        manifest=read(path);folder=path.parent;step=manifest['step']
        ev=read(folder/'provenance'/f'evaluation_{step:07d}.json')
        visual=read(folder/'FR_sharpness.json')['mean']
        rows.append(dict(run_id=manifest['run_id'],step=step,HQNR=ev['fr']['metrics']['HQNR'],
            ERGAS=ev['rr']['metrics']['ERGAS'],PSNR=ev['rr']['metrics']['PSNR'],**visual,
            output=str(folder),checkpoint_sha256=manifest['checkpoint_sha256']))
    directory=ROOT/P.OUTPUT
    if not rows:return []
    fields=list(rows[0]);rows.sort(key=lambda r:(-r['edge_pm2'],r['flat_grain']))
    with (directory/'ranking.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write(directory/'ranking.json',dict(protocol='51_sharpness_qb_gf2.py core metrics; raw-PAN 1/99% rendering',
        rows=rows,reference=read(ROOT/'config/panda_gf2_sharpness/reference_metrics.json'),
        warning='Read sharpness and grain together. Reference methods are supplied metrics, not rerun predictions. All saved checkpoints, including early failures, are diagnostic candidates.'))
    return rows


def run(*,replay_missing=True):
    if (ROOT/'gspread/server.txt').read_text().strip()!='s3':raise ValueError('This worker belongs to s3')
    from .release import verify
    verify(environment=True)
    from .audit import save
    inventory=save();output=ROOT/P.OUTPUT
    if not inventory['all_sources_ready']:raise ValueError('Original GF2 source configs/evaluations missing; see inventory.json')
    stopped=False;child=None
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    def invoke(args):
        nonlocal child
        if stopped:raise InterruptedError('Worker stopped')
        child=subprocess.Popen([sys.executable,'-u','-m',*args])
        rc=child.wait()
        if stopped or rc==130:raise InterruptedError('Child saved/stopped')
        if rc:raise RuntimeError('Child command failed with code '+str(rc)+': '+' '.join(args))
        child=None
    try:
        for entry in inventory['rows']:
            case=entry['case'];run_id=case['run_id'];original=ROOT/case['original_directory']
            write(output/'worker_status.json',dict(status='RUNNING',run_id=run_id))
            replay=output/'replays'/run_id/'attempt_01'
            if replay_missing and entry['replay_required']:
                invoke(['panda_gf2_sharpness.replay','--run-id',run_id])
            steps=sorted(set(P.INSPECT)|({500,3000} if (replay/'complete.json').exists() else set()))
            for step in steps:
                name=f'step_{step:07d}.pt'
                source=original if (original/'models'/name).exists() else replay
                if not (source/'models'/name).exists():continue
                destination=output/'exports'/run_id/f'step_{step:07d}'
                if destination.exists() and (not (destination/'manifest.json').exists()
                        or not (destination/'files_sha256.json').exists()
                        or (destination/'INCOMPLETE.json').exists()):
                    destination.rename(destination.with_name(destination.name+f'.interrupted_{time.time_ns()}'))
                invoke(['panda_sensor_visuals.export','--run-dir',str(source),'--step',str(step),
                        '--output',str(destination),'--splits','fr','--no-references'])
                summarize()
        rows=summarize()
        missing=[dict(run_id=e['case']['run_id'],steps=e['missing_early']) for e in inventory['rows'] if e.get('missing_early') and not replay_missing]
        write(output/'worker_status.json',dict(status='COMPLETED_WITH_MISSING_EARLY' if missing else 'COMPLETED',
            evaluated=len(rows),missing=missing,stage1_training=False,original_runs_unchanged=True))
    except InterruptedError:
        write(output/'worker_status.json',dict(status='STOPPED_RESUMABLE'));return 130
    except BaseException as exc:
        write(output/'worker_status.json',dict(status='FAILED',error=repr(exc)));raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['catalog','audit','run','summary'])
    p.add_argument('--no-replay',action='store_true');args=p.parse_args()
    if args.command=='catalog':
        import json;print(json.dumps(P.manifest(),indent=2));return 0
    from .release import verify
    print(verify(environment=False))
    if args.command=='audit':
        from .audit import save
        value=save()
        for r in value['rows']:print(r['case']['run_id'],r['status'],'missing early:',r.get('missing_early'))
        return 0 if value['all_sources_ready'] else 2
    if args.command=='summary':print(summarize());return 0
    return run(replay_missing=not args.no_replay)


if __name__=='__main__':raise SystemExit(main())
