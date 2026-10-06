"""One independent queue per assigned server, without legacy handoff gates."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

import torch

from .common import ROOT, atomic_json, digest, file_sha, immutable_json, load_json, now, source_manifest
from .data import DatasetBundle
from .evaluation import evaluate_checkpoint, evaluator_identity
from .model import TeacherModel, state_hash
from .losses import StructSupportViolation
from .plan import LANES, Plan, resolved_config
from .training import (UpdateEngine, configure_runtime, predict_native, preserve_runtime,
                       run_training, save_torch)


@contextlib.contextmanager
def lane_lock(lane):
    lane.mkdir(parents=True, exist_ok=True)
    with (lane/'controller.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another TA2 controller owns this exact server lane')
        yield


def runtime_identity(device):
    configure_runtime(261006000, device)
    gpu = None
    if torch.device(device).type == 'cuda':
        index = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(index)
        gpu = dict(name=props.name, capability=list(torch.cuda.get_device_capability(index)),
                   total_memory=props.total_memory, uuid=os.environ.get('TA2_GPU_UUID',str(getattr(props, 'uuid', 'unavailable'))))
        # Matrix/convolution availability alone does not certify grid backward.
        from pa.warp import warp_pan
        p = torch.randn(1, 1, 64, 64, device=device)
        delta = torch.tensor([[.3, -.4]], device=device, requires_grad=True)
        warp_pan(p, delta).square().mean().backward(); torch.cuda.synchronize()
        if not torch.isfinite(delta.grad).all():
            raise ValueError('Actual CUDA warp-backward preflight failed')
    return dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                image_id=os.environ.get('TA2_IMAGE_ID','HOST_ENVIRONMENT'),host=os.environ.get('TA2_HOSTNAME',platform.node()),
                cudnn=torch.backends.cudnn.version(), gpu=gpu, precision='FP32', tf32=False,
                cuda_grid_backward_bitwise_deterministic=False, cpu_threads=torch.get_num_threads(),
                evaluator=evaluator_identity(ROOT))


def inspect_plan(plan, server, lane):
    rows = plan.queue(server)
    # 50 FP32 U/A weights, latest full optimizer state and two raw selected/final
    # RR20+FR20 dumps. Diagnostics/logs may add appreciably to this lower bound.
    bands = LANES[server][1]
    model_bytes = (2764218 if bands == 8 else 2755574) * 4
    per_run = 50 * model_bytes + 4 * model_bytes + 2 * 20 * bands * (256**2 + 512**2) * 4
    existing = lane if lane.exists() else lane.parent if lane.parent.exists() else ROOT
    free = shutil.disk_usage(existing).free
    return dict(status='PLAN_VALIDATED', server=server, dataset=LANES[server][0], cases=52, runs=len(rows),
        total_updates=sum(r['updates'] for r in rows), plan_sha256=plan.sha256,
        first_core_runs=[r['run_id'] for r in rows[:48]], work_directory=str(lane),
        disk_free_gib=free/2**30, estimated_full_lane_minimum_gib=156*per_run/2**30,
        disk_warning=None if free>156*per_run else 'FULL_LANE_STORAGE_EXCEEDS_CURRENT_FREE_SPACE; no deletion or migration performed',
        automatic_other_server_wait=False, official_fr_masking=False, official_rr_count=20)


def preflight(plan, server, lane, device):
    bundle = DatasetBundle(ROOT, LANES[server][0], lane/'data')
    identity = dict(plan_sha256=plan.sha256, source=digest(source_manifest()),
        data=digest(bundle.manifest['identity']), runtime=runtime_identity(device), server=server,
        campaign_id=plan.queue(server)[0]['campaign_id'])
    if torch.device(device).type=='cuda':
        immutable_json(lane/'identity.json', identity)
    atomic_json(lane/'preflight.json', dict(inspect_plan(plan,server,lane), identity=identity,
                data_status='PASS', actual_cuda_warp_checked=torch.device(device).type=='cuda', checked=now()))
    return bundle, identity


class _SmokeFixedBundle:
    """Lazy exact B03 estimates for smoke IDs only; never a production cache."""
    def __init__(self,bundle):
        self.bundle=bundle; self.datasets=bundle.datasets; self.cache={}
    def batch(self,*args,**kwargs):
        return self.bundle.batch(*args,**kwargs)
    def case_shifts(self,cfg,split):
        outer=self
        class Indexed:
            def __getitem__(self,ids):
                from .registration import estimate_shift
                if isinstance(ids,slice):
                    ids=range(*ids.indices(len(outer.datasets[split])))
                rows=[]
                for i in ids:
                    key=split,int(i)
                    if key not in outer.cache:
                        arrays=outer.datasets[split].arrays
                        outer.cache[key]=estimate_shift(arrays['pan'][i],arrays['lms'][i],max_dn=outer.bundle.max_dn,
                                                        split=split,target='NATIVE_LMS')['shift']
                    rows.append(outer.cache[key])
                return torch.tensor(rows,dtype=torch.float32)
        return Indexed()


def smoke(plan, server, lane, bundle, identity, device, updates=3,micro_batch=12):
    if updates < 3:
        raise ValueError('At least3 smoke updates needed: first legacy warmup LR is zero')
    smoke_identity=dict(identity,micro_batch=micro_batch,updates=updates)
    target = lane/'smoke'/digest(smoke_identity)
    target.mkdir(parents=True, exist_ok=True)
    receipt = target/'receipt.json'
    if receipt.exists():
        previous = load_json(receipt)
        if previous['identity'] == smoke_identity and previous['updates_per_case'] == updates:
            return previous
        raise ValueError('Smoke identity differs')
    cases = ['TA2-B00','TA2-B03','TA2-M05','TA2-M07','TA2-U03']
    rows = []; first_u = None; stream_id = None
    for case in cases:
        row = next(r for r in plan.queue(server) if r['case_id']==case and r['replica']==1)
        cfg = resolved_config(row,micro_batch)
        configure_runtime(cfg['seed'],device)
        model = TeacherModel(cfg['bands'],cfg['seed'],cfg).to(device)
        active_bundle=_SmokeFixedBundle(bundle) if case=='TA2-B03' else bundle
        engine = UpdateEngine(model,active_bundle,cfg,device)
        if first_u is None:
            first_u = state_hash(model.backbone); stream_id = engine.native_stream_sha256
        if first_u != state_hash(model.backbone) or stream_id != engine.native_stream_sha256:
            raise ValueError('Smoke detected unpaired initialization/native stream')
        if torch.device(device).type=='cuda':
            torch.cuda.reset_peak_memory_stats()
        logs = [engine.update() for _ in range(updates)]
        path = target/(case+'.pt'); saved=engine.state_dict(); save_torch(path,saved)
        restored = TeacherModel(cfg['bands'],cfg['seed'],cfg).to(device)
        resumed = UpdateEngine(restored,active_bundle,cfg,device)
        resumed.load_state_dict(torch.load(path,map_location='cpu',weights_only=False))
        if state_hash(restored)!=state_hash(model) or resumed.stream.state_dict()!=engine.stream.state_dict():
            raise ValueError('Smoke checkpoint reload is not identical')
        # Fr/RR model execution has no GT argument (the metric uses GT separately).
        shapes=[]
        with torch.no_grad():
            for split in ('rr','fr'):
                sample=bundle.batch(split,[0],device=device)
                result=predict_native(model,active_bundle,cfg,sample['pan'],sample['ms'],sample['lms'],split,0)
                if not torch.isfinite(result['prediction']).all():
                    raise ValueError('Nonfinite native inference smoke')
                shapes.append(list(result['prediction'].shape))
        metric_smoke=None
        if case=='TA2-M07':
            weights=target/(case+'_weights.pt')
            save_torch(weights,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                    completed_step=updates,identity=identity))
            evaluated=evaluate_checkpoint(model,bundle.datasets,target/'metric_smoke',sensor=cfg['dataset'],
                checkpoint_path=weights,completed_step=updates,context=dict(identity,smoke_only=True),device=device)
            metric_smoke=dict(hqnr=evaluated['fr']['hqnr'],scc=evaluated['rr']['scc'],ergas=evaluated['rr']['ergas'],
                              rr_count=evaluated['rr']['n_scenes'],fr_count=evaluated['fr']['n_scenes'],
                              eligible_for_formal_selection=False)
        rows.append(dict(case_id=case,logs=logs,initial_u_sha256=first_u,native_stream_sha256=stream_id,
            checkpoint_sha256=file_sha(path), inference_shapes=shapes, metric_smoke=metric_smoke,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.device(device).type=='cuda' else None))
        del engine,model,restored,resumed,saved
        if torch.device(device).type=='cuda':
            torch.cuda.empty_cache()
    # Actual fixed estimator path, deliberately on one smoke observation, not
    # a substituted synthetic cache for any formal training input.
    from .registration import estimate_shift
    sample=bundle.get('train',0)
    fixed=estimate_shift(((sample['pan']+1)/2).numpy(),((sample['lms']+1)/2).numpy(),max_dn=1.)
    result=dict(status='PASS',smoke_only=True,formal_optimizer_updates=0,identity=smoke_identity,
                updates_per_case=updates,cases=rows,fixed_estimator=fixed,checked=now())
    immutable_json(receipt,result)
    return result


def event_hooks(lane, identity, publish=False):
    from .reporting import Outbox, summary_records
    outbox=Outbox(lane/'outbox')
    from .worker import Publisher
    publisher=Publisher(lane,identity['server'],outbox,publish).start()

    def context(cfg,step,checkpoint,selector):
        return dict(cfg,completed_step=step,checkpoint_sha256=file_sha(checkpoint),
                    source_revision=identity['source'],attempt=1,selector=selector,status='MEASURED',
                    config_sha256=digest(cfg),data_sha256=identity['data'])

    def publish_pending():
        publisher.notify()

    def event(model,bundle,cfg,run_dir,checkpoint,step,report,device):
        from .probes import measure_gradient_protocol
        scale=cfg['total_updates']//50000
        if step in (1010*scale,25250*scale,cfg['total_updates']):
            target=run_dir/'gradients'/f'step_{step:06d}.json'
            if not target.exists():
                with preserve_runtime(model):
                    gradients=measure_gradient_protocol(model,bundle,cfg,device,step)
                logs=[json.loads(line) for line in (run_dir/'training.jsonl').read_text().splitlines()]
                actual=next((r for r in reversed(logs) if r['completed_step']==step),None)
                if actual is not None:
                    for group in gradients.get('summary',[]):
                        group['actual_parameter_update_norm']=actual.get('actual_a_group_update',{}).get(group['group'])
                        group['update_na_reason']=None if group['actual_parameter_update_norm'] is not None else 'no_parameters_in_group'
                    gradients['actual_update_source']='measured_optimizer_parameter_deltas_at_same_completed_update'
                immutable_json(target,gradients)
            from .reporting import gradient_records
            outbox.enqueue_many(gradient_records(context(cfg,step,checkpoint,'FIXED_PROGRESS'),load_json(target)))
            from .progress import progress_protocol,progress_records
            progress_path=run_dir/'progress'/f'step_{step:06d}.json'
            if not progress_path.exists():
                with preserve_runtime(model):
                    measured=progress_protocol(model,bundle,cfg,step,device)
                immutable_json(progress_path,measured)
            logs=[json.loads(line) for line in (run_dir/'training.jsonl').read_text().splitlines()]
            actual=next((r for r in reversed(logs) if r['completed_step']==step),{})
            outbox.enqueue_many(progress_records(context(cfg,step,checkpoint,'FIXED_PROGRESS'),actual,load_json(progress_path)))
        outbox.enqueue_many(summary_records(context(cfg,step,checkpoint,'FIXED_PROGRESS'),report,'FIXED_PROGRESS'))
        publish_pending()

    def finalize(model,bundle,cfg,run_dir,selected,device):
        from .probes import run_protocol
        from .reporting import response_records
        receipts={}
        gradients=[load_json(p) for p in sorted((run_dir/'gradients').glob('step_*.json'))]
        for selector,step in (('HQNR_MAX50',selected['completed_step']),('EXACT_FINAL',cfg['total_updates'])):
            checkpoint=run_dir/'checkpoints'/f'step_{step:06d}.pt'
            state=torch.load(checkpoint,map_location=device,weights_only=False)
            model.load_state_dict(state['model'],strict=True)
            with preserve_runtime(model):
                report=evaluate_checkpoint(model,bundle.datasets,run_dir/'exports'/selector,
                    sensor=cfg['dataset'],checkpoint_path=checkpoint,completed_step=step,
                    context=load_json(run_dir/'identity.json'),device=device,export_raw=True,
                    predict=lambda m,p,ms,l,s,i:predict_native(m,bundle,cfg,p,ms,l,s,i))
                probe=run_protocol(model,bundle,cfg,run_dir/'diagnostics'/selector,checkpoint,step,device,gradients)
            ctx=context(cfg,step,checkpoint,selector)
            outbox.enqueue_many(summary_records(ctx,report,selector,selection=selected))
            outbox.enqueue_many(response_records(ctx,probe))
            receipts[selector]=dict(status=probe['status'],checkpoint_sha256=file_sha(checkpoint))
            publish_pending()
        return dict(status='COMPLETE' if all(r['status']=='COMPLETE' for r in receipts.values()) else 'PARTIAL',
                    selectors=receipts)
    def status_hook(cfg,run_dir,step,status,error=None):
        from .reporting import status_record
        ctx=dict(cfg,source_revision=identity['source'],attempt=1)
        outbox.enqueue(status_record(ctx,status,step,error))
        publisher.notify()
    return event,finalize,status_hook,outbox,publisher


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('inspect','preflight','smoke','run','status','upload'))
    parser.add_argument('--server',choices=tuple(LANES),required=True)
    parser.add_argument('--work-root',type=Path,default=ROOT/'work_dir/aligner_teacher_lms_v2')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--micro-batch',type=int,default=12)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--smoke-updates',type=int,default=3)
    parser.add_argument('--run-id')
    parser.add_argument('--publish',action='store_true',help='Explicitly enable append-only live analysis upload')
    args=parser.parse_args(argv)
    if args.command=='run' and torch.device(args.device).type!='cuda':
        raise ValueError('Formal registered training requires CUDA; CPU is validation/smoke only')
    if args.command=='run' and not (ROOT/'SOURCE_MANIFEST.json').is_file():
        raise ValueError('Formal run requires frozen source; use bash ta2/start.sh SERVER run')
    torch.set_num_threads(args.cpu_threads)
    plan=Plan(); lane=args.work_root.resolve()/args.server
    # Same persistent bind mount for publisher thread, upload CLI and snapshots.
    os.environ.setdefault('TA2_GOOGLE_QUOTA_DIR',str(lane/'google_quota'))
    if args.command=='inspect':
        print(json.dumps(inspect_plan(plan,args.server,lane),indent=2)); return
    if args.command=='status':
        result={p.parent.name:load_json(p) for p in sorted((lane/'runs').glob('*/status.json'))}
        print(json.dumps(result,indent=2)); return
    with lane_lock(lane):
        if args.command=='upload':
            from .reporting import Outbox
            from .publication import connect_router
            outbox=Outbox(lane/'outbox'); router=connect_router(lane,args.server)
            while True:
                state=outbox.publish(adapter=router,max_records=100)
                print(json.dumps(state,indent=2),flush=True)
                if state.get('errors') or not state.get('pending_records') or not state.get('attempted_this_batch'):
                    return
        bundle,identity=preflight(plan,args.server,lane,args.device)
        if args.command=='preflight':
            print(json.dumps(load_json(lane/'preflight.json'),indent=2)); return
        smoke_receipt=smoke(plan,args.server,lane,bundle,identity,args.device,args.smoke_updates,args.micro_batch)
        if args.command=='smoke':
            print(json.dumps(dict(status=smoke_receipt['status'],smoke_only=True,
                formal_optimizer_updates=smoke_receipt['formal_optimizer_updates'],
                cases=[row['case_id'] for row in smoke_receipt['cases']],
                updates_per_case=smoke_receipt['updates_per_case'],micro_batch=args.micro_batch,
                metric_smoke={row['case_id']:row['metric_smoke'] for row in smoke_receipt['cases'] if row.get('metric_smoke')},
                receipt_path=str(lane/'smoke'/digest(smoke_receipt['identity'])/'receipt.json')),indent=2)); return
        queue=plan.queue(args.server)
        if args.run_id:
            queue=[row for row in queue if row['run_id']==args.run_id]
            if not queue:
                raise ValueError('Run ID is not registered on this server')
        event,finalize,status_hook,outbox,publisher=event_hooks(lane,identity,args.publish)
        from .reporting import asset_records,publication_capacity_estimate
        asset_context=dict(plan.queue(args.server)[0],case_id='TA2-ASSET',source_revision=identity['source'],
                           completed_step=0,status='PASS',attempt=1)
        outbox.enqueue_many(asset_records(asset_context,bundle.manifest,smoke_receipt))
        from .publication import POLICY,USER_APPROVED_PUBLIC_PARENT,PREALLOCATED_SCHEMA
        capacity=publication_capacity_estimate()
        capacity['unpartitioned_central_only_exceeds_cell_limit'] = capacity['added_cells_lower_estimate'] > capacity['workbook_cell_contract_limit']
        registry_path=ROOT/'ta2/detail_books.json'
        destinations=load_json(registry_path) if registry_path.is_file() else {}
        if destinations and (destinations.get('schema')!=PREALLOCATED_SCHEMA or destinations.get('campaign')!=identity['campaign_id']):
            raise ValueError('Approved detail registry campaign/schema differs')
        capacity.update(status='APPROVED_PREALLOCATED_PARTITION_POLICY',policy=POLICY,
            approved_parent_id=USER_APPROVED_PUBLIC_PARENT,approved_existing_public_parent_inheritance=True,
            expected_user_owned_workbooks=78,workbooks_per_server=26,buckets_per_server=13,parts_per_bucket=2,
            registered_workbooks=len(destinations.get('entries',{})),
            registry_status='REGISTERED_REQUIRES_LIVE_ACCESS_READBACK' if destinations else 'REGISTRY_NOT_INSTALLED_LOCAL_DETAILS_PENDING',
            registry_sha256=file_sha(registry_path) if registry_path.is_file() else None,
            remote_permissions_headers_verified_by_this_estimate=False,
            service_account_my_drive_creation_required=False,
            unregistered_part_policy='LOCAL_PENDING_UNTIL_EXPLICIT_USER_OWNED_PROVISIONING')
        atomic_json(lane/'publication_capacity.json',capacity)
        for row in queue:
            run_dir=lane/'runs'/row['run_id']
            cfg=resolved_config(row,args.micro_batch)
            status=load_json(run_dir/'status.json') if (run_dir/'status.json').exists() else {}
            if status and (load_json(run_dir/'config.json')!=cfg or
                    load_json(run_dir/'identity.json')!=dict(identity,run_id=cfg['run_id'],config_sha256=digest(cfg))):
                raise ValueError('Recorded attempt config/source/runtime differs; no silent skip or retry')
            if status.get('status') in ('COMPLETE','NUMERICAL_FAILURE'):
                if status['status']=='COMPLETE':
                    from .evaluation import EvaluationLedger
                    actual=EvaluationLedger(run_dir/'selection',load_json(run_dir/'identity.json'),cfg['total_updates']).status()
                    if actual.get('status')!='HQNR_SELECTION_COMPLETE':
                        raise ValueError('COMPLETE marker has incomplete HQNR evidence')
                    for selector in ('HQNR_MAX50','EXACT_FINAL'):
                        if not (run_dir/'exports'/selector/'report.json').is_file():
                            raise ValueError('COMPLETE marker has missing native export evidence')
                        if not list((run_dir/'diagnostics'/selector).rglob('report.json')):
                            raise ValueError('COMPLETE marker has missing diagnostic evidence')
                continue
            try:
                result=run_training(bundle,cfg,run_dir,identity,args.device,event_hook=event,finalize_hook=finalize,status_hook=status_hook)
                status_hook(cfg,run_dir,result['completed_step'],result['status'])
                from .analysis import refresh_analysis
                refresh_analysis(lane,plan,args.server,outbox)
                if result['status']!='COMPLETE':
                    publisher.close()
                    return
            except (FloatingPointError,StructSupportViolation) as exc:
                failure=load_json(run_dir/'failed_update.json') if (run_dir/'failed_update.json').exists() else {}
                atomic_json(run_dir/'status.json',dict(status='NUMERICAL_FAILURE',error=str(exc),updated=now(),
                                                      seed=cfg['seed'],completed_step=failure.get('last_completed_in_memory',0),
                                                      retry_is_new_independent_seed=False,failure_evidence=failure))
                status_hook(cfg,run_dir,load_json(run_dir/'status.json').get('completed_step',0),'NUMERICAL_FAILURE',str(exc))
                from .analysis import refresh_analysis
                refresh_analysis(lane,plan,args.server,outbox)
            except Exception as exc:
                atomic_json(lane/'blocked.json',dict(run_id=row['run_id'],status='PENDING_RECOVERY',error=str(exc),
                            traceback=traceback.format_exc(),updated=now(),training_restart_required=False))
                publisher.close()
                raise
        statuses={r['run_id']:load_json(lane/'runs'/r['run_id']/'status.json').get('status') for r in queue}
        atomic_json(lane/'queue_status.json',dict(status='REGISTERED_BLOCK_COMPLETE' if
            len(queue)==156 and all(s=='COMPLETE' for s in statuses.values()) else 'REGISTERED_BLOCK_INCOMPLETE',
            runs=statuses,updated=now()))
        publisher.close()
        if args.publish and (publisher.thread is None or not publisher.thread.is_alive()):
            # All registered updates are finished. Remaining transport can drain
            # without holding a live model graph; any API error remains pending.
            from .publication import connect_router
            router=connect_router(lane,args.server)
            while True:
                state=outbox.publish(adapter=router,max_records=100)
                atomic_json(lane/'upload_status.json',state)
                if state.get('errors') or not state.get('pending_records') or not state.get('attempted_this_batch'):
                    break


if __name__=='__main__':
    main()
