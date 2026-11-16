"""Native CUDA check of three frozen donors, input C and exact saved boundary."""
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import tempfile
import torch
from directshift48.reporting import atomic_json,utc_now
from donor30.artifacts import _same,_cpu
from panda_stage2_condq.training import configure_precision
from ta2.evaluation import evaluator_identity
from panda_input3 import training
from panda_input3.model import Stage2
from panda_input3.preflight import dual_fr_probe,differences,numerically_same
from panda_method.calibration import state_dict_sha256
from . import plan,assets
from .release import verify


def run(server='s2',device='cuda:0'):
    plan.sensor_for(server);configure_precision();release=verify(environment=True)
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Native CUDA preflight required on s2')
    root=plan.campaign_root(server);folder=root/'preflight';folder.mkdir(parents=True,exist_ok=True)
    source,registered=assets.sources(server)
    pins=dict(source_release=release['source_sha256'],evaluator=evaluator_identity(),
        train_inventory=registered.inventory.manifest(),
        split_manifests={s:source.manifest_sha(s) for s in ('train','val','rr','fr')})
    cached=folder/'verified.json'
    if cached.exists():
        old=json.loads(cached.read_text())
        if old.get('status')=='VERIFIED' and old.get('pins')==pins:return old
    donors={};initial=[]
    for cfg in [c for c in plan.queue(server) if c['repeat']==1]:
        donor,stats=assets.prepare_reference(cfg,registered,root,pins,device)
        net=Stage2(cfg,donor,stats).to(device);batch=registered.batch([0],device=device)
        with torch.no_grad():
            out=net.native_prediction(batch);terms=net(batch)
            if not all(bool(torch.isfinite(value).all()) for value in terms.values()):
                raise ValueError('Nonfinite loss in fixed donor preflight')
            if out['x_in'].shape!=(1,6,64,64):raise ValueError('Expected six-channel QB C input')
            if not torch.equal(out['x_in'][:,0:1],out['pan_aligned']):raise ValueError('Wrong corrected PAN input')
            if not torch.equal(out['x_in'][:,1:2],out['pan_aligned']-batch.pan):raise ValueError('Wrong difference channel')
            initial.append(state_dict_sha256(net.student))
        donors[str(cfg['donor_step'])]=dict(checkpoint_sha256=donor.binding['donor_checkpoint_sha256'],
            calibration_sha256=stats.artifact_sha256,dual_fr=dual_fr_probe(net,source,pins['evaluator'],device),
            first_loss=float(terms['loss']))
        del net,donor,stats,batch;gc.collect();torch.cuda.empty_cache()
    if len(set(initial))!=1:raise ValueError('R1 initial Student weights changed between donors')
    cfg=plan.queue(server)[0]
    donor,stats=assets.prepare_reference(cfg,registered,root,pins,device)
    net=Stage2(cfg,donor,stats).to(device)
    tr=training.Trainer(cfg,net);stream=training.stream_for(cfg,registered,device)
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        path=Path(tmp)/'resume.pt';first=tr.step(stream.next_update())
        training.save(path,tr,stream,pins);resume_bytes=path.stat().st_size
        evaluated=Path(tmp)/'evaluated.pt';training.save(evaluated,tr,stream,pins,evaluated=True)
        evaluated_bytes=evaluated.stat().st_size
        boundary=dict(model=_cpu(net.state_dict()),optimizer=_cpu(tr.optimizer.state_dict()),
                      stream=deepcopy(stream.state_dict()),rng=training.rng_state())
        second=tr.step(stream.next_update());expected=_cpu(net.state_dict())
        expected_opt=_cpu(tr.optimizer.state_dict());expected_stream=deepcopy(stream.state_dict())
        training.load(path,tr,stream,pins)
        restored=dict(model=_cpu(net.state_dict()),optimizer=_cpu(tr.optimizer.state_dict()),
                      stream=deepcopy(stream.state_dict()),rng=training.rng_state())
        exact={k:_same(boundary[k],restored[k]) for k in boundary}
        if not all(exact.values()):raise ValueError('Saved model/optimizer/RNG/stream boundary differs')
        replay=tr.step(stream.next_update())
        diagnostic=dict(boundary_exact=exact,model_differences=differences(expected,net.state_dict()),
            optimizer_differences=differences(expected_opt,tr.optimizer.state_dict()),
            replay_tolerance=dict(rtol=1e-6,atol=1e-8))
        atomic_json(folder/'resume_diagnostic.json',diagnostic)
        if (second.keys()!=replay.keys() or any(not math.isclose(second[k],replay[k],rel_tol=1e-6,abs_tol=1e-8) for k in second)
            or not _same(expected_stream,stream.state_dict()) or not numerically_same(expected,net.state_dict())
            or not numerically_same(expected_opt,tr.optimizer.state_dict())):
            raise ValueError('Replay exceeds the existing UBIL CUDA tolerance')
        tr.audit()
    result=dict(status='VERIFIED',at=utc_now(),pins=pins,donors=donors,disposable_only=True,
        production_updates=0,checkpoint_boundary_exact=exact,first_update=first,
        same_r1_student_initialization=True,checkpoint_bytes=dict(evaluated=evaluated_bytes,complete_resume=resume_bytes))
    atomic_json(cached,result)
    return result
