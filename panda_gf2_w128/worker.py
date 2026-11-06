"""One durable container owns the trainer and its CPU-only Sheet publisher."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from directshift48.reporting import atomic_json, utc_now
from . import registry as G


def run():
    if (G.ROOT/'gspread/server.txt').read_text().strip()!='s3':
        raise ValueError('This worker must be launched on the s3 server')
    root=G.ROOT/G.OUTPUT/G.CAMPAIGN_ID/'s3/GF2'
    root.mkdir(parents=True,exist_ok=True)
    def status(stage,**extra):
        atomic_json(root/'worker_status.json',dict(stage=stage,at=utc_now(),server='s3',**extra))
    stopped=False
    trainer=None; publisher=None
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
        if trainer is not None and trainer.poll() is None: trainer.send_signal(signal.SIGTERM)
    old={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        from .release import verify
        verify(check_environment=True)
        if stopped:
            status('STOPPED_BEFORE_TRAINING');return 1
        # A resumed worker must obtain a new final readback, including when the
        # old publisher finished while the previous controller was stopped.
        upload=root/'sheet_publish_panda_sensor_focus/terminal_upload.json'
        if upload.exists():
            upload.rename(upload.with_name(f'terminal_upload_previous_{time.time_ns()}.json'))
        with (root/'publisher.log').open('ab') as log:
            publisher=subprocess.Popen([sys.executable,'-u','-m','panda_gf2_w128','publish',
                '--server','s3','--apply','--watch'],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        trainer=subprocess.Popen([sys.executable,'-u','-m','panda_gf2_w128','run','--server','s3'])
        status('RUNNING',trainer_pid=trainer.pid,publisher_pid=publisher.pid)
        while trainer.poll() is None:
            if publisher.poll() is not None:
                upload=root/'sheet_publish_panda_sensor_focus/terminal_upload.json'
                verified=(publisher.returncode==0 and upload.exists()
                          and json.loads(upload.read_text()).get('readback_verified'))
                if not verified:
                    trainer.send_signal(signal.SIGTERM)
                    raise RuntimeError('Sheet publisher exited while training; trainer asked to save and stop')
            time.sleep(5)
        if stopped:
            status('STOPPED',trainer_returncode=trainer.returncode);return 1
        if trainer.returncode:
            status('FAILED_TRAINING',trainer_returncode=trainer.returncode);return trainer.returncode
        from .prerequisites import terminal
        finished=terminal(G.ROOT,'s3')
        status('WAIT_SHEET_READBACK',campaign_status=finished['campaign_status'])
        while not stopped:
            upload=root/'sheet_publish_panda_sensor_focus/terminal_upload.json'
            if upload.exists():
                record=json.loads(upload.read_text())
                if record.get('readback_verified') and record.get('campaign_status')==finished['campaign_status']:
                    status('COMPLETED',sheet_verified=True,**finished);return 0
            if publisher.poll() is not None:
                raise RuntimeError('Sheet publisher exited without verified terminal readback')
            time.sleep(5)
        status('STOPPED_WAITING_FOR_SHEET');return 1
    except Exception as exc:
        status('FAILED',error=repr(exc));raise
    finally:
        if trainer is not None and trainer.poll() is None:
            trainer.send_signal(signal.SIGTERM)
            # Let the trainer write the complete optimizer/RNG/stream state.
            trainer.wait()
        if publisher is not None and publisher.poll() is None:
            publisher.terminate()
            try: publisher.wait(timeout=30)
            except subprocess.TimeoutExpired: publisher.kill();publisher.wait()
        for sig,handler in old.items():signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s3',),required=True)
    parser.parse_args()
    raise SystemExit(run())
