"""Reconstruct deleted early Students in a separate directory using original A/T and schedule.

No Stage1 updates. Original config, calibration, seeds, source files and recorded
results remain unchanged. A replay must match original raw RR20/FR20 hashes
at every available original evaluation; otherwise stop with mismatch evidence.
"""
import argparse
from copy import deepcopy
import json
import re
from pathlib import Path
import signal
import time
from panda_sensor_visuals.common import ROOT,read,write,sha,local,source_namespace
from . import plan as P


def matching_predictions(actual,original):
    for split in ('rr','fr'):
        a,b=actual[split]['per_scene'],original[split]['per_scene']
        if len(a)!=20 or len(b)!=20:return False
        if any(x['scene_id']!=y['scene_id'] or x['raw_prediction_sha256']!=y['raw_prediction_sha256'] for x,y in zip(a,b)):
            return False
    return True


def resume_point(folder):
    """Recover an atomically saved checkpoint even if the latest pointer was interrupted."""
    candidates=[]
    for path in (folder/'models').glob('*.pt'):
        match=re.fullmatch(r'(?:step|stop)_(\d{7})(?:_\d+)?\.pt',path.name)
        if match:candidates.append((int(match[1]),path))
    if not candidates:return None
    step,path=max(candidates,key=lambda pair:(pair[0],pair[1].name))
    side=path.with_suffix('.json')
    digest=sha(path)
    if side.exists() and read(side)['artifact']['file_sha256']!=digest:
        raise ValueError('Replay checkpoint changed since saving: '+str(path))
    return dict(checkpoint_path=str(path),file_sha256=digest,completed_update=step)


def run(case,*,device='cuda'):
    if (ROOT/'gspread/server.txt').read_text().strip()!='s3':
        raise ValueError('Replay runs on s3 with its own Stage1 donor only')
    original=ROOT/case['original_directory'];cfg=read(original/'config.json');P.validate_case(cfg,case)
    source_namespace(cfg)
    from .release import verify
    release=verify(environment=True)
    import torch
    from panda_stage2_condq.training import configure_precision
    from panda_method.edge_data import EdgeDataSources
    from panda_stage2_condq.sources import RegisteredRRTrain
    from panda_stage2_condq.stream import TrainStream,StreamSpec
    from panda_sensor_focus import engine
    from panda_sensor_focus.donor import validate_source,load_donor
    from panda_sensor_focus.calibration import bind_donor,load_stats_bank,HOST_CALIBRATION_BUDGET_BYTES
    from panda_sensor_focus.spec import MatureSpec
    from panda_sensor_focus.evaluation import official_evaluate,model_state_hash
    from panda_method.calibration import state_dict_sha256
    from ta2.evaluation import evaluator_identity
    from directshift48.reporting import append_jsonl
    configure_precision();torch.set_num_threads(4)
    folder=ROOT/P.OUTPUT/'replays'/case['run_id']/'attempt_01';folder.mkdir(parents=True,exist_ok=True)
    source_identity=dict(config_sha256=sha(original/'config.json'),source_run=str(original),
        original_horizon=60000,execution_stop=P.STOP_UPDATE,donor_sha256=P.DONOR_SHA,
        release_sha256=release['source_sha256'],purpose='RECONSTRUCT_EARLY_OUTPUTS_NOT_NEW_SCIENTIFIC_RUN')
    if (folder/'replay_identity.json').exists() and read(folder/'replay_identity.json')!=source_identity:
        raise ValueError('Replay identity changed; preserve old replay and inspect')
    write(folder/'replay_identity.json',source_identity)
    if (folder/'complete.json').exists():return folder
    sources=EdgeDataSources(ROOT/'ablr2/sensor_sources.json',root=ROOT,sensor='GF2')
    registered=RegisteredRRTrain(sources)
    receipt=validate_source('s3',ROOT)
    donor=load_donor(local(cfg['donor_checkpoint']),P.DONOR_SHA,registered,server='s3',selector_receipt=receipt)
    spec=MatureSpec(**cfg['stage2_spec']);donor=bind_donor(donor,spec)
    if json.loads(json.dumps(donor.binding))!=cfg['donor_binding']:
        raise ValueError('Original frozen donor binding changed')
    artifacts={k:{'path':str(local(v['path'])),'file_sha256':v['file_sha256']} for k,v in cfg['calibration_artifacts'].items()}
    stats=load_stats_bank(donor,spec,artifacts,memory_budget_bytes=HOST_CALIBRATION_BUDGET_BYTES)
    evaluator=evaluator_identity();code=read(ROOT/'config/panda_gf2_sharpness/release.json')['files']
    trainer=engine.build_fresh_stage2(deepcopy(cfg),donor,stats,evaluator_pin=evaluator)
    trainer.stage.to(device);trainer.stage.move_calibration(device,memory_budget_bytes=512*1024**2)
    stream=TrainStream(registered,StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation']),
        microbatch_size=12,effective_batch_size=48,device=device)
    write(folder/'config.json',cfg)
    before={k:state_dict_sha256(getattr(trainer.stage,k)) for k in ('aligner','teacher')}
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}

    def save(path,evaluation=None):
        artifact=engine.save_checkpoint(path,trainer,stream,code_pin=code,evaluator_pin=evaluator,evaluation=evaluation)
        artifact.update(identity=evaluation['state_before'] if evaluation else {},state_hash=model_state_hash(trainer.stage),
                        campaign_resume_complete=True)
        write(path.with_suffix('.json'),dict(artifact=artifact,config=cfg,code_pin=code,evaluator_pin=evaluator,evaluation=evaluation))
        write(folder/'latest.json',artifact)
        return artifact

    try:
        saved=resume_point(folder)
        if saved:
            trainer,stream=engine.load_checkpoint(local(saved['checkpoint_path']),saved['file_sha256'],trainer,stream,
                code_pin=code,evaluator_pin=evaluator)
            if trainer.updates!=saved['completed_update']:raise ValueError('Resume filename/update mismatch')
            write(folder/'latest.json',saved)
            from panda_sensor_focus.runtime import recover_trace
            recover_trace(folder,trainer.updates)
        while True:
            step=trainer.updates
            if step in P.REPLAY_SAVES:
                ep=folder/f'evaluation_{step:07d}.json';ck=folder/'models'/f'step_{step:07d}.pt'
                if ck.exists():
                    payload=torch.load(ck,map_location='cpu',weights_only=False);evaluation=payload['evaluation']
                    if evaluation is None or payload['model_state_sha256']!=model_state_hash(trainer.stage):
                        raise ValueError('Replay checkpoint/evaluated state mismatch')
                    if not ck.with_suffix('.json').exists():
                        artifact=dict(checkpoint_path=str(ck),file_sha256=sha(ck),completed_update=step,
                            model_state_sha256=payload['model_state_sha256'],state_hash=payload['model_state_sha256'],
                            identity=evaluation['state_before'],checkpoint_verified=True,weights_saved=True)
                        write(ck.with_suffix('.json'),dict(artifact=artifact,config=cfg,code_pin=code,evaluator_pin=evaluator,evaluation=evaluation))
                else:
                    evaluation=official_evaluate(trainer,sources,evaluator_pin=evaluator,device=device)
                    save(ck,evaluation)
                original_eval=original/f'evaluation_{step:07d}.json'
                if original_eval.exists() and not matching_predictions(evaluation,read(original_eval)):
                    write(folder/f'mismatch_{step:07d}.json',dict(actual=evaluation,original_evaluation=str(original_eval),
                        status='NOT_AN_EXACT_REPLAY_STOPPED'))
                    raise ValueError('Replay differs from original predictions; mismatch evidence preserved')
                write(ep,evaluation)
                write(folder/f'replay_check_{step:07d}.json',dict(step=step,
                    original_evaluation_present=(original/f'evaluation_{step:07d}.json').exists(),
                    matches_original=matching_predictions(evaluation,read(original/f'evaluation_{step:07d}.json'))
                        if (original/f'evaluation_{step:07d}.json').exists() else None,
                    status='EARLY_DIAGNOSTIC_ONLY' if step==500 else 'EXACT_ORIGINAL_OUTPUT_REPRODUCED'))
                print(case['run_id'],step,'saved; schedule horizon remains60K',flush=True)
            if step>=P.STOP_UPDATE:
                assert all(state_dict_sha256(getattr(trainer.stage,k))==v for k,v in before.items())
                write(folder/'complete.json',dict(status='COMPLETED_5K_REPLAY',step=step,stage1_unchanged=True,identity=source_identity))
                return folder
            if stopped:
                save(folder/'models'/f'stop_{step:07d}_{time.time_ns()}.pt')
                write(folder/'status.json',dict(status='STOPPED_RESUMABLE',step=step));return None
            if step%100==0:verify(environment=False)
            tick=time.monotonic();engine.advance(trainer,stream)
            if device.startswith('cuda'):torch.cuda.synchronize()
            append_jsonl(folder/'train.jsonl',dict(step=trainer.updates,local_update=trainer.updates,
                seconds_including_IO=time.monotonic()-tick))
    except BaseException as exc:
        write(folder/'status.json',dict(status='FAILED',step=trainer.updates,error=repr(exc)))
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();case=next(c for c in P.cases() if c['run_id']==args.run_id)
    result=run(case)
    if result is None:raise SystemExit(130)
    print(result)


if __name__=='__main__':main()
