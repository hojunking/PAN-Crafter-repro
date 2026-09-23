"""Additive G23 OAT trainer: paired fresh50K, no legacy budget/test selector.

The loss expressions and constructor order mirror QRECON24. Runtime ownership,
continuous admission and result publication belong to the separate controller.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import tempfile
import time

import numpy as np
import torch
from safetensors.torch import save_file

from fh12.training import atomic_torch, tensor_cpu_tree, save_resume_bundle
from kdv.forward import kdv_forward
from kdv.losses_rec import GTAnchoredReconstructionKD
from kdv.qrecon import QWeight, q_weight, _sha_full
from kdv.teacher_assets import load_run_model, freeze, assert_param_disjoint
from model.pancrafter_paper import PANCrafterPaper
from pa.losses import output_edge_loss_per_sample
from g23sens.model import (state_hash, initial_snapshot, build_model,
                           validate_architecture)
from g23sens.diagnostics import (rng_state, restore_rng, isolated_probe,
                                fixed_probe, distribution, DIAGNOSTIC_STEPS)


GRID_STEPS = tuple(range(1010, 50000, 1010)) + (50000,)


def make_criterion(cfg):
    rec = cfg['kdv']['rec']
    if rec.get('case') != 'R3' or float(rec.get('eps', 0)) != 1e-6:
        raise ValueError('Only legacy REC-R3/eps1e-6 is admitted')
    # tau itself remains the raw calibration value. Apply scale precisely once.
    used = float(rec['tau']) * float(rec.get('tau_scale', 1.0))
    criterion = GTAnchoredReconstructionKD(used, alpha=float(rec['alpha']),
        kd_weight=float(rec['kd_weight']), eps=float(rec['eps']), mode='adaptive')
    expected = cfg['g23sens']['case']['resolved_values']['tau_R_used']
    if float(criterion.tau) != float(expected):
        raise ValueError('Criterion tau is not raw tau times the registered scale once')
    return criterion


def make_optimizer(model, cfg):
    if (cfg['optimizer'] != 'AdamW' or float(cfg['learning_rate']) != 1e-4
            or float(cfg['weight_decay']) != .01):
        raise ValueError('G23 SENS fixes AdamW U LR1e-4 and weight decay .01')
    a_lr = float(cfg['kdv']['aligner_lr'])
    expected = float(cfg['g23sens']['case']['resolved_values']['A_peak_lr'])
    if a_lr != expected or a_lr not in (1e-6, 3e-6, 6e-6):
        raise ValueError('A optimizer LR differs from this sensitivity arm')
    groups = [dict(params=list(model.backbone.parameters()), lr=1e-4, name='U'),
              dict(params=list(model.aligner.parameters()), lr=a_lr, name='A')]
    if any(not p.requires_grad for group in groups for p in group['params']):
        raise ValueError('G23 SENS does not freeze Student U or A')
    return torch.optim.AdamW(groups, lr=1e-4, weight_decay=.01)


def make_scheduler(optimizer, cfg):
    from diffusers.optimization import get_scheduler
    if (cfg['lr_scheduler'] != 'cosine' or cfg['num_warmup'] != 100
            or cfg['num_iter'] != 50000):
        raise ValueError('G23 SENS is exactly warmup100/cosine50K')
    return get_scheduler('cosine', optimizer=optimizer,
                         num_warmup_steps=100, num_training_steps=50000)


def forward_objectives(model, teacher, criterion, qweight, gt, ms, lp, pan, meta, lambda_edge):
    """Same native forward/R3/Scharr reduction as train_kdv._step(qrecon)."""
    out = kdv_forward(model, teacher, pan, ms, lp, share_correction=False,
                      teacher_needed=True, aligner_live=True)
    with torch.autocast(device_type=pan.device.type, enabled=False):
        y, t, target = out['y'].float(), out['y_t'].float().detach(), gt.float().detach()
        result = criterion(y, t, target, return_maps=True)
        e_gt = (y-target).abs().mean(1, keepdim=True)
        e_st = (y-t).abs().mean(1, keepdim=True)
        hard = (result.maps['hard_weight']*e_gt).mean((1, 2, 3))
        soft = (result.maps['soft_weight']*e_st).mean((1, 2, 3))
        edge = output_edge_loss_per_sample(y, target)
        wa, we = qweight.weights(meta, pan.device)
        if not torch.equal(wa, we):
            raise ValueError('All sensitivity arms use the same raw-q weights for A and edge')
        weighted_edge = (we*edge).mean()
        lu = (hard+soft+float(lambda_edge)*we*edge).mean()
        la = (wa*hard).mean()
    return out, dict(L_U=lu, L_A=la, H=hard, K=soft, E=edge,
                     weighted_E=weighted_edge, w=wa, maps=result.maps,
                     plain_H=e_gt.mean((1, 2, 3)), plain_K=e_st.mean((1, 2, 3)))


def routed_gradients(model, terms, *, retain_graph=False):
    """Never total.backward(): A receives ONLY the hard objective's gradient."""
    u = [p for p in model.backbone.parameters() if p.requires_grad]
    a = [p for p in model.aligner.parameters() if p.requires_grad]
    if not u or not a or set(map(id,u)) & set(map(id,a)):
        raise ValueError('Disjoint trainable U and A parameter sets are required')
    gu = torch.autograd.grad(terms['L_U'], u, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(terms['L_A'], a, retain_graph=retain_graph, allow_unused=True)
    return gu, ga


def routed_backward(model, terms):
    gu, ga = routed_gradients(model, terms)
    for params, grads in ((model.backbone.parameters(),gu),(model.aligner.parameters(),ga)):
        for param, grad in zip(params, grads):
            param.grad = grad
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('Nonfinite/missing routed gradients')


def qweight_from_raw(raw_q, qref, metadata=None):
    q = np.asarray(torch.as_tensor(raw_q).cpu(), dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError('The frozen cue must cover every original sample and all four rotations')
    weights = q_weight(q, qref)
    result = QWeight.synthetic(torch.from_numpy(weights.astype(np.float32)), qref=qref)
    result.stats = dict(q_raw_sha256=_sha_full(q), w_sha256=_sha_full(weights),
                       q_raw=distribution(q), w=distribution(weights), n_views=int(q.size))
    result.manifest = dict(metadata or {})
    return result


def verify_restored_state(state, context, stream, scheduler):
    if (state.get('full_state') is not True or state.get('precision') != 'fp32'
            or any(state.get(k) != v for k, v in context.items())):
        raise ValueError('Resume source/config/bindings/init/stream identity mismatch')
    update = state['update']
    if not isinstance(update, int) or not 0 <= update <= 50000 or scheduler.last_epoch != update:
        raise ValueError('Resume scheduler/update mismatch')
    stream_update = getattr(stream, 'update', getattr(stream, 'cursor', None))
    if stream_update != update:
        raise ValueError('Resume sampler cursor differs from optimizer updates')
    return update


def _cycle_initialization(cfg, teacher, root):
    from g23sens.common import (cycle_dir, locked, immutable_json, read_json,
                                object_sha, sha256)
    case = cfg['g23sens']['case']
    directory = cycle_dir(root, case['server'], case['cycle'])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'initial_snapshot.pt'
    with locked(directory/'initialization.lock'):
        if path.exists():
            identity = read_json(directory/'initial_snapshot.json')
            if identity['file_sha256'] != sha256(path):
                raise ValueError('The common cycle initialization file changed')
            snapshot = torch.load(path, map_location='cpu', weights_only=False)
            if snapshot['hashes']['A'] != state_hash(teacher.aligner.state_dict()):
                raise ValueError('Common cycle A is not fixed T0')
        else:
            snapshot = initial_snapshot(cfg, teacher)
            atomic_torch(path, snapshot)
            immutable_json(directory/'initial_snapshot.json',dict(
                seed=case['seed'], cycle=case['cycle'], server=case['server'],
                policy=snapshot['policy'], hashes=snapshot['hashes'],
                model_args_sha256=object_sha(cfg['model_args']),file_sha256=sha256(path)))
    # build_model verifies the fresh U hash against the real seeded constructor.
    return snapshot


def train_run(config_path, root=None, device='cuda', resume=False):
    from g23sens.common import (ROOT, read, read_json, read_config, atomic_json,
        immutable_json, object_sha, sha256, utcnow, source_identity,
        apply_runtime_policy, cycle_dir, check_runtime, RuntimePaused)
    from g23sens.plan import validate_config
    from g23sens.assets import load_training_assets, validate_bindings
    from g23sens.data import NativeDataset, BatchStream
    from g23sens.evaluation import evaluate_validation
    from g23sens.controller import authorize_train
    root = Path(root or ROOT).resolve()
    cfg = read_config(config_path)
    validate_config(cfg,root=root,require_bound=True)
    case = cfg['g23sens']['case']
    authorize_train(root, case, config_path)
    validate_architecture(cfg)
    runtime = apply_runtime_policy(root)
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no implicit CPU training fallback')
    if dev.type != 'cuda':
        raise RuntimeError('Production sensitivity runs require CUDA; use numerical unit tests for CPU')
    wd = Path(cfg['work_dir'])
    if not wd.is_absolute(): wd = root/wd
    wd.mkdir(parents=True,exist_ok=True)
    binding_path=Path(cfg['g23sens']['bindings_path'])
    bindings = read_json(binding_path if binding_path.is_absolute() else root/binding_path)
    validate_bindings(bindings, root, case['server'], rehash=True)
    if object_sha(bindings) != cfg['g23sens']['binding_sha256']:
        raise ValueError('Resolved asset binding hash differs')
    assets = load_training_assets(bindings, root)
    teacher, teacher_manifest = load_run_model(assets['teacher_run_path'], 'best_hqnr',
        PANCrafterPaper, bindings['teacher']['checkpoint_sha256'])
    freeze(teacher)
    if (teacher_manifest['tag_meta'].get('step') != 24240
            or teacher.aligner_margin != 4):
        raise ValueError('Only fixed T0 step24240 margin4 is admitted')
    snapshot = _cycle_initialization(cfg, teacher, root)
    model = build_model(cfg, snapshot).to(dev).train()
    teacher.to(dev).eval()
    teacher_hash = state_hash(teacher.state_dict())
    if set(map(id,model.parameters())) & set(map(id,teacher.parameters())):
        raise ValueError('Teacher and Student share parameter objects')
    datasets = {split:NativeDataset(bindings,split,root=root) for split in ('train','val')}
    stream = BatchStream(datasets['train'], batch_size=48, seed=case['seed'],
        initial_torch_rng_state=snapshot['post_constructor_torch_rng'],workers=4,total_updates=50000)
    stream_manifest = stream.manifest() if callable(stream.manifest) else stream.manifest
    stream_sha = object_sha(stream_manifest)
    immutable_json(cycle_dir(root,case['server'],case['cycle'])/'stream_manifest.json',stream_manifest)
    immutable_json(wd/'stream_manifest.json',stream_manifest)
    qweight = qweight_from_raw(assets['raw_q'],cfg['kdv']['qrecon']['q_ref'],bindings['cue'])
    qweight.stats['q_table_sha256'] = qweight.stats['q_raw_sha256']
    qweight.stats['q_raw_sha256'] = bindings['cue']['raw_q_sha256']
    criterion = make_criterion(cfg).to(dev)
    lambda_edge = float(cfg['kdv']['stat']['outer_weight'])
    optimizer = make_optimizer(model,cfg)
    assert_param_disjoint(optimizer,teacher)
    scheduler = make_scheduler(optimizer,cfg)
    resolved = dict(alpha=criterion.alpha,beta=criterion.kd_weight,lambda_E=lambda_edge,
        tau_R_raw=float(cfg['kdv']['rec']['tau']),tau_R_scale=float(cfg['kdv']['rec'].get('tau_scale',1)),
        tau_R_actual=float(criterion.tau),q_ref=qweight.qref,
        optimizer_groups=[dict(name=g['name'],peak_lr=float(g['initial_lr']),
            weight_decay=float(g['weight_decay']),betas=list(g['betas']),eps=g['eps'])
            for g in optimizer.param_groups],gradient_routing='U<-L_U; A<-weighted_H_only',
        precision='fp32',offset_weight=0.,geometry_weight=0.,synthetic_shift_radius=0.)
    context = dict(run_id=case['run_id'],attempt=cfg['g23sens'].get('attempt',0),
        config_sha256=object_sha(cfg),case_spec_sha256=case['case_spec_sha256'],
        bindings_sha256=object_sha(bindings),source_identity=source_identity(root),
        init_U_sha256=snapshot['hashes']['U'],init_A_sha256=snapshot['hashes']['A'],
        stream_sha256=stream_sha,q_raw_sha256=bindings['cue']['raw_q_sha256'],
        q_weight_sha256=qweight.stats['w_sha256'],teacher_sha256=bindings['teacher']['checkpoint_sha256'])
    immutable_json(wd/'meta/config.resolved.yaml',cfg)
    immutable_json(wd/'meta/consumed_bindings.json',bindings)
    immutable_json(wd/'meta/resolved_numerics.json',resolved)
    immutable_json(wd/'init_manifest.json',dict(hashes=snapshot['hashes'],policy=snapshot['policy'],**context))
    immutable_json(wd/'meta/qweights.json',qweight.summary())
    records = read(wd/'val_records.json').get('records',[])
    update = 0
    training_seconds = evaluation_seconds = io_seconds = diagnostic_seconds = 0.
    if resume:
        last = read_json(wd/'last/identity.json')
        state_path = wd/'last/training_state.pt'
        if (last.get('training_state_sha256') != sha256(state_path)
                or any(last.get(k) != v for k,v in context.items())):
            raise ValueError('Resume checkpoint identity/checksum differs')
        state = torch.load(state_path,map_location='cpu',weights_only=False)
        if state_hash(state['model_state']) != last['state_hash']:
            raise ValueError('Resume tensor state checksum differs')
        model.load_state_dict(state['model_state'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        stream.load_state_dict(state['sampler'])
        update = verify_restored_state(state,context,stream,scheduler)
        if any(row['update']>update or row['update'] not in GRID_STEPS for row in records):
            raise ValueError('Validation grid is ahead of restored training state')
        for name in ('training_seconds','evaluation_seconds','io_seconds','diagnostic_seconds'):
            if not math.isfinite(float(state[name])) or state[name]<0:
                raise ValueError('Invalid resume elapsed-time ledger')
        training_seconds = state['training_seconds'];evaluation_seconds=state['evaluation_seconds']
        io_seconds=state['io_seconds'];diagnostic_seconds=state['diagnostic_seconds']
        restore_rng(state['rng'])
    else:
        if (wd/'last').exists() or records or (wd/'meta/training_start_manifest.json').exists():
            raise ValueError('Existing run requires exact resume, never silent fresh overwrite')
        # Reproduce the post-U global state; explicit data stream remains isolated.
        import random
        random.seed(2025);np.random.seed(case['seed']);torch.manual_seed(case['seed'])
        torch.cuda.manual_seed_all(case['seed'])
        torch.set_rng_state(snapshot['post_constructor_torch_rng'])
        immutable_json(wd/'meta/training_start_manifest.json',dict(**context,
            started_at_utc=utcnow(),runtime_policy=runtime,device=str(dev),
            horizon_updates=50000,precision='fp32',initialization_policy=snapshot['policy']))
    probe = stream.fixed_probe()
    pause=[False]
    # An optimizer update is a transaction only once all queued CUDA work has
    # succeeded and the sampler/update cursor has been committed. Never replace
    # the last valid disk checkpoint with a partially mutated optimizer/model.
    safe_update_boundary=True
    handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in handlers: signal.signal(sig,lambda *_:pause.__setitem__(0,True))

    def fullstate():
        return dict(full_state=True,precision='fp32',update=update,
            model_state=tensor_cpu_tree(model.state_dict()),optimizer=tensor_cpu_tree(optimizer.state_dict()),
            scheduler=scheduler.state_dict(),criterion_state=tensor_cpu_tree(criterion.state_dict()),
            rng=rng_state(),sampler=stream.state_dict(),training_seconds=training_seconds,
            evaluation_seconds=evaluation_seconds,io_seconds=io_seconds,diagnostic_seconds=diagnostic_seconds,
            **context)

    def save_resume():
        nonlocal io_seconds
        started=time.monotonic()
        save_resume_bundle(wd,fullstate(),dict(update=update,full_state=True,
            state_hash=state_hash(model.state_dict()),**context))
        io_seconds+=time.monotonic()-started

    def candidate():
        folder=wd/'candidates'/str(update)
        expected=dict(update=update,full_state=update==50000,precision='fp32',
                      state_hash=state_hash(model.state_dict()),**context)
        if folder.exists():
            identity=read_json(folder/'identity.json')
            if (any(identity.get(k)!=v for k,v in expected.items())
                    or identity.get('model_sha256')!=sha256(folder/'model.safetensors')
                    or (update==50000 and identity.get('training_state_sha256')!=sha256(folder/'training_state.pt'))):
                raise ValueError('Existing immutable candidate differs')
            return identity
        folder.parent.mkdir(parents=True,exist_ok=True)
        pending=Path(tempfile.mkdtemp(prefix='.%d-'%update,dir=folder.parent))
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},
                  str(pending/'model.safetensors'))
        identity=dict(expected,model_sha256=sha256(pending/'model.safetensors'),saved_at_utc=utcnow())
        if update==50000:
            state=fullstate();state['model_sha256']=identity['model_sha256']
            atomic_torch(pending/'training_state.pt',state)
            identity['training_state_sha256']=sha256(pending/'training_state.pt')
        atomic_json(pending/'identity.json',identity)
        os.replace(pending,folder)
        return identity

    def status(name,reason=None):
        selected=min(records,key=lambda r:(r['val_ergas'],r['update']))['update'] if records else None
        complete=(name=='TRAIN_COMPLETE_EVAL_PENDING' and update==50000
            and [row['update'] for row in records]==list(GRID_STEPS)
            and all(math.isfinite(row['val_ergas']) for row in records))
        atomic_json(wd/'meta/training_status.json',dict(status=name,actual_updates=update,
            horizon_updates=50000,training_complete=complete,training_seconds=training_seconds,
            evaluation_seconds=evaluation_seconds,io_seconds=io_seconds,diagnostic_seconds=diagnostic_seconds,
            peak_memory_bytes=torch.cuda.max_memory_allocated(dev),selected_val_step=selected,
            primary_step=50000 if update==50000 else None,n_val_evaluated=len(records),
            completed_at=read(wd/'candidates/50000/identity.json').get('saved_at_utc'),
            safe_update_boundary=safe_update_boundary,
            last_valid_checkpoint_update=read(wd/'last/identity.json').get('update'),
            reason=reason,updated_at_utc=utcnow(),**context))

    def boundary():
        nonlocal evaluation_seconds,diagnostic_seconds,io_seconds,records
        if update in DIAGNOSTIC_STEPS:
            path=wd/'diagnostics'/('%05d.json'%update)
            if not path.exists():
                started=time.monotonic()
                report=fixed_probe(model,teacher,criterion,qweight,probe,optimizer,
                    step=update,lambda_edge=lambda_edge,initial_aligner_state=snapshot['A'])
                immutable_json(path,dict(report,**context))
                diagnostic_seconds+=time.monotonic()-started
        if update not in GRID_STEPS:
            save_resume();return
        save_resume()
        started=time.monotonic();identity=candidate();io_seconds+=time.monotonic()-started
        if not any(row['update']==update for row in records):
            started=time.monotonic()
            with isolated_probe(model,teacher):
                metric=evaluate_validation(model,datasets['val'],str(dev),
                    stopcheck=lambda:pause[0])
            value=float(metric['ergas'])
            if not math.isfinite(value):raise FloatingPointError('Nonfinite validation ERGAS')
            records.append(dict(update=update,val_ergas=value,metrics=metric,checkpoint_identity=identity))
            selected=min(records,key=lambda r:(r['val_ergas'],r['update']))['update']
            atomic_json(wd/'val_records.json',dict(records=records,selected_step=selected,
                primary_selection='EXACT_50000',secondary_selection='RR_VAL_ERGAS_MIN_THEN_LOWER_STEP',
                grid=list(GRID_STEPS),**context))
            evaluation_seconds+=time.monotonic()-started
            print('[G23SENS %s %d/50000] HQNR=N/A(val-only) SCC=%.6f ERGAS=%.6f (RR-val)'%
                  (case['case_id'],update,float(metric['scc']),value),flush=True)
        if state_hash(teacher.state_dict())!=teacher_hash:
            raise ValueError('Frozen T0 mutated')
        save_resume()

    try:
        status('RUNNING');boundary()
        while update<50000:
            if pause[0]:raise RuntimePaused('Signal requested a safe update boundary')
            check_runtime(case,root,update)
            batch=stream.next_batch()
            started=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
            gt,ms,lp,pan,meta=[batch[k].to(dev,non_blocking=True) for k in ('gt','ms','lpan','pan','meta')]
            out,terms=forward_objectives(model,teacher,criterion,qweight,gt,ms,lp,pan,meta,lambda_edge)
            if not all(bool(torch.isfinite(x).all()) for x in (out['y'],out['delta'],terms['L_U'],terms['L_A'])):
                raise FloatingPointError('Nonfinite native Student output/correction/objective')
            routed_backward(model,terms)
            safe_update_boundary=False
            optimizer.step();scheduler.step()
            torch.cuda.synchronize(dev)
            stream.advance();update+=1
            safe_update_boundary=True
            training_seconds+=time.monotonic()-started
            if update in GRID_STEPS or update in DIAGNOSTIC_STEPS or update%1000==0:
                boundary();status('RUNNING')
            if update%100==0:
                print('[G23SENS %s] update=%d/50000 L_U=%.8f L_A=%.8f U_lr=%.9g A_lr=%.9g'%
                    (case['case_id'],update,float(terms['L_U']),float(terms['L_A']),
                     optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr']),flush=True)
        if len(records)!=len(GRID_STEPS):raise ValueError('Required validation grid is incomplete')
        save_resume();status('TRAIN_COMPLETE_EVAL_PENDING');return 0
    except (RuntimePaused,InterruptedError) as error:
        if safe_update_boundary:save_resume()
        status('PAUSED_SAFE',str(error));return 75
    except FloatingPointError as error:
        status('DIVERGED',str(error));raise
    except Exception as error:
        # Mid-update OOM / asynchronous CUDA failures may already have mutated
        # some parameters or moments. The old on-disk full state is the only
        # trusted recovery point in that case; do not serialize the live model.
        if safe_update_boundary:save_resume()
        status('TECHNICAL_FAILURE',type(error).__name__+': '+str(error));raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        stream.close()
        for dataset in datasets.values():dataset.close()


def smoke_validate(config_path, root=None, device='cuda'):
    """Actual-asset CUDA smoke without a production run, cycle snapshot or cursor.

    Serialize/restore the full state after update one and repeat update two with
    the same batch. Tolerances acknowledge CUDA backward roundoff; all resumed
    identity, optimizer, RNG and sampler states must restore exactly.
    """
    import copy
    import hashlib
    import io
    from g23sens.common import ROOT,read_config,read_json,object_sha,apply_runtime_policy
    from g23sens.assets import load_training_assets,validate_bindings
    from g23sens.data import NativeDataset,BatchStream
    from g23sens.plan import validate_config
    root=Path(root or ROOT).resolve();cfg=read_config(config_path)
    validate_config(cfg,root=root,require_bound=True)
    if cfg['g23sens']['case']['case_id']!='BASE':
        raise ValueError('Admission smoke must use this cycle BASE definition')
    dev=torch.device(device)
    if dev.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('GPU smoke requires actual CUDA; CPU tests are not GPU evidence')
    apply_runtime_policy(root)
    # This smoke has its own seed even though the numerical BASE coefficients
    # are validated against the real case above. It never publishes a CaseSpec.
    production_seed=int(cfg['seed'])
    cfg=copy.deepcopy(cfg)
    cfg['seed']=(production_seed+2**30)%(2**32)
    path=Path(cfg['g23sens']['bindings_path'])
    bindings=read_json(path if path.is_absolute() else root/path)
    validate_bindings(bindings,root,cfg['g23sens']['case']['server'],rehash=True)
    if object_sha(bindings)!=cfg['g23sens']['binding_sha256']:
        raise ValueError('Smoke asset binding differs')
    assets=load_training_assets(bindings,root)
    teacher,_=load_run_model(assets['teacher_run_path'],'best_hqnr',PANCrafterPaper,
                            bindings['teacher']['checkpoint_sha256'])
    freeze(teacher);teacher.to(dev)
    snapshot=initial_snapshot(cfg,teacher)
    model=build_model(cfg,snapshot).to(dev)
    dataset=NativeDataset(bindings,'train',root)
    stream=BatchStream(dataset,48,cfg['seed'],snapshot['post_constructor_torch_rng'],
                       workers=4,total_updates=3)
    weights=qweight_from_raw(assets['raw_q'],cfg['kdv']['qrecon']['q_ref'])
    criterion=make_criterion(cfg).to(dev);opt=make_optimizer(model,cfg);sch=make_scheduler(opt,cfg)
    teacher_hash=state_hash(teacher.state_dict())

    def one_step():
        model.train();opt.zero_grad(set_to_none=True);batch=stream.next_batch()
        # Batch48 is retained even in smoke, catching actual-memory/shape failures.
        inputs=[batch[k].to(dev) for k in ('gt','ms','lpan','pan','meta')]
        _,terms=forward_objectives(model,teacher,criterion,weights,*inputs,cfg['kdv']['stat']['outer_weight'])
        if not all(bool(torch.isfinite(terms[k])) for k in ('L_U','L_A')):
            raise FloatingPointError('Smoke nonfinite objective')
        routed_backward(model,terms);opt.step();sch.step();stream.advance();torch.cuda.synchronize(dev)

    try:
        diagnostics=fixed_probe(model,teacher,criterion,weights,stream.fixed_probe(2),opt,
            step=0,lambda_edge=cfg['kdv']['stat']['outer_weight'],initial_aligner_state=snapshot['A'])
        one_step()
        packed=io.BytesIO()
        torch.save(dict(model=tensor_cpu_tree(model.state_dict()),optimizer=tensor_cpu_tree(opt.state_dict()),
            scheduler=sch.state_dict(),rng=rng_state(),sampler=stream.state_dict()),packed)
        checkpoint_sha=hashlib.sha256(packed.getvalue()).hexdigest()
        one_step();expected=tensor_cpu_tree(copy.deepcopy(model.state_dict()))
        expected_stream=stream.state_dict();expected_rng=rng_state()
        packed.seek(0);restored=torch.load(packed,map_location='cpu',weights_only=False)
        model.load_state_dict(restored['model'],strict=True)
        opt.load_state_dict(restored['optimizer']);sch.load_state_dict(restored['scheduler'])
        stream.load_state_dict(restored['sampler']);restore_rng(restored['rng'])
        one_step()
        current_rng=rng_state()
        rng_equal=(current_rng['python']==expected_rng['python']
            and current_rng['numpy'][0]==expected_rng['numpy'][0]
            and np.array_equal(current_rng['numpy'][1],expected_rng['numpy'][1])
            and current_rng['numpy'][2:]==expected_rng['numpy'][2:]
            and torch.equal(current_rng['torch'],expected_rng['torch'])
            and all(torch.equal(a,b) for a,b in zip(current_rng['cuda'],expected_rng['cuda'])))
        maximum=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in model.state_dict().items())
        equivalent=all(torch.allclose(v.detach().cpu(),expected[k],rtol=1e-6,atol=1e-7)
                       for k,v in model.state_dict().items())
        if not equivalent or not rng_equal or stream.state_dict()!=expected_stream or sch.last_epoch!=2:
            raise ValueError('GPU smoke exact-resume state/cursor/numerical comparison failed')
        if state_hash(teacher.state_dict())!=teacher_hash:
            raise ValueError('GPU smoke mutated fixed T0')
        return dict(passed=True,device=str(dev),batch_size=48,updates=2,
            smoke_seed=cfg['seed'],production_seed=production_seed,resume_rng_equal=rng_equal,
            resumed_from_update=1,resume_max_abs_error=maximum,resume_rtol=1e-6,resume_atol=1e-7,
            serialized_fullstate_sha256=checkpoint_sha,initialization=snapshot['hashes'],
            stream_sha256=stream.manifest['manifest_sha256'],teacher_state_unchanged=True,
            diagnostics=diagnostics,production_state_written=False,
            note='Actual GPU/asset forward, split gradients, batch48 and fullstate roundtrip; not a reported50K run')
    finally:
        stream.close();dataset.close()
