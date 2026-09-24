"""Fixed fifty-candidate HQNR selection; recovery never retrains a Student."""
from pathlib import Path
from datetime import datetime
import math
import time

from maina_hqnr.common import (ROOT, atomic_json, immutable_json, object_sha,
    read_json, read_config, sha256, source_identity, utcnow)
from maina_hqnr.evaluation import (PROTOCOL, evaluate_candidate, evaluate_rr,
    evaluator_identity, population, preserve_rng, validate_candidate, validate_report)

CAMPAIGN_ID = 'PANDA_MAINA_WV3_S45_HQNR_20260925_v1'
GRID = tuple(range(1010, 50000, 1010)) + (50000,)
SELECTIONS = ('HQNR_MAX50', 'EXACT_50000')
SELECTION_POLICY = dict(primary_selection='HQNR_MAX50', selection_split='FR20', test_aware=True,
    independent_test=False, expected_candidate_count=50, hqnr_reference='RAW_ORIGINAL_PAN',
    checkpoint_tie_breaker='LOWER_COMPLETED_UPDATES', ergas_used_for_selection=False,
    rule_id='MAX_RAW_ORIGINAL_FR20_MEAN_HQNR_THEN_LOWER_STEP_V1')


def select_candidates(records, actual_updates):
    """Pure acceptance gate, intentionally with no ERGAS/threshold arguments."""
    if type(actual_updates) is not int or actual_updates != 50000:
        raise ValueError('HQNR selection cannot complete before actual update50000')
    if (len(records) != 50 or any(type(r.get('update')) is not int for r in records)
            or sorted(r['update'] for r in records) != list(GRID)):
        raise ValueError('Exactly fifty distinct registered candidate steps are required')
    contexts = []
    for row in records:
        validate_candidate(row)
        identity = row.get('checkpoint_identity', {})
        if identity.get('update') != row['update'] or identity.get('model_sha256') != row['checkpoint_sha256']:
            raise ValueError('Evaluated A/U candidate SHA/update differs')
        contexts.append({key: row['metadata'][key] for key in ('protocol_sha256', 'data_sha256', 'population', 'evaluator')})
    if any(context != contexts[0] for context in contexts[1:]):
        raise ValueError('Candidate population/evaluator/data protocol changed within a run')
    best = min(records, key=lambda item: (-item['fr']['hqnr'], item['update']))
    exact = next(r for r in records if r['update'] == 50000)
    return {'HQNR_MAX50': best, 'EXACT_50000': exact}


def _config(run_dir_or_config, root=ROOT):
    path = Path(run_dir_or_config)
    if path.is_dir():
        path = path/'meta/config.resolved.yaml' if (path/'meta/config.resolved.yaml').exists() else path/'config.json'
    cfg = read_config(path)
    field = cfg['maina_hqnr']; case = field['case']
    from maina_hqnr.plan import validate_config
    validate_config(cfg, root=root, require_bound=True)
    if case['server'] not in ('s4', 's5') or case.get('campaign_id', CAMPAIGN_ID) != CAMPAIGN_ID:
        raise ValueError('Only the new s4/s5 MAIN-A campaign is eligible')
    wd = Path(cfg['work_dir'])
    if not wd.is_absolute():
        wd = Path(root)/wd
    if field['source_identity'] != source_identity(root):
        raise ValueError('Evaluation source/runtime differs from the frozen training release')
    binding_path = Path(field['bindings_path'])
    bindings = read_json(binding_path if binding_path.is_absolute() else Path(root)/binding_path)
    if object_sha(bindings) != field['binding_sha256']:
        raise ValueError('Frozen F1/data binding changed')
    return cfg, case, bindings, wd


def _candidate_identity(cfg, wd, step):
    from maina_hqnr.plan import TEACHER_SHA, DATA_SHA
    field = cfg['maina_hqnr']; path = wd/'candidates'/str(step)
    identity = read_json(path/'identity.json')
    expected = dict(update=step, config_sha256=object_sha(cfg), source_identity=field['source_identity'],
                    bindings_sha256=field['binding_sha256'], run_id=field['case']['run_id'],
                    attempt=field['attempt'], teacher_sha256=TEACHER_SHA, data_sha256=DATA_SHA,
                    candidate_grid_sha256=object_sha(list(GRID)))
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Candidate {step} config/source/binding/update differs')
    if identity.get('model_sha256') != sha256(path/'model.safetensors'):
        raise ValueError(f'Candidate {step} A/U weights SHA mismatch')
    for key in ('init_U_sha256', 'init_A_sha256', 'stream_sha256', 'teacher_sha256'):
        if not isinstance(identity.get(key), str) or len(identity[key]) != 64:
            raise ValueError('Missing actual checkpoint '+key)
    return identity


def _training_evidence(cfg, wd):
    status = read_json(wd/'meta/training_status.json')
    if (status.get('actual_updates') != 50000 or status.get('training_complete') is not True
            or status.get('status') not in ('TRAIN_COMPLETE_EVAL_PENDING', 'COMPLETE')):
        raise ValueError('Full completed50K training evidence is required')
    folders = sorted(int(path.name) for path in (wd/'candidates').iterdir() if path.is_dir() and path.name.isdigit())
    if folders != list(GRID):
        raise ValueError('Candidate retention must contain the exact registered fifty steps')
    identities = {step: _candidate_identity(cfg, wd, step) for step in GRID}
    anchor = identities[50000]
    keys = ('config_sha256', 'source_identity', 'bindings_sha256', 'init_U_sha256',
            'init_A_sha256', 'stream_sha256', 'teacher_sha256', 'run_id', 'attempt',
            'data_sha256', 'reference_sha256', 'candidate_grid_sha256')
    if any(any(identity.get(k) != anchor[k] for k in keys) for identity in identities.values()):
        raise ValueError('Fifty A/U candidates do not belong to one initialized run/stream')
    # The full-state receipt is checked before accepting any evaluated result.
    state_path = wd/'candidates/50000/training_state.pt'
    if anchor.get('full_state') is not True or anchor.get('training_state_sha256') != sha256(state_path):
        raise ValueError('Exact50000 full-state artifact is missing or changed')
    import torch
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    if (state.get('update') != 50000 or not state.get('optimizer', {}).get('state')
            or not state.get('rng') or not state.get('sampler')
            or state.get('scheduler', {}).get('last_epoch') != 50000):
        raise ValueError('Exact50000 optimizer/scheduler/RNG/sampler evidence is incomplete')
    if any(float(value['step']) != 50000 for value in state['optimizer']['state'].values()):
        raise ValueError('Optimizer did not complete50000 actual updates')
    from fh12.model import state_hash
    from fh12.training import BatchStream
    if (state.get('full_state') is not True or state.get('precision') != 'fp32'
            or state_hash(state.get('model_state', {})) != anchor.get('state_hash')
            or any(state.get(key) != anchor[key] for key in keys)):
        raise ValueError('Full-state A/U tensor/config/provenance mismatch')
    from safetensors.torch import load_file
    if state_hash(load_file(str(wd/'candidates/50000/model.safetensors'))) != anchor['state_hash']:
        raise ValueError('Full-state A/U tensors differ from the inference safetensors')
    rng = state['rng']
    if set(rng) != {'python', 'numpy', 'torch', 'cuda'} or not isinstance(rng['cuda'], list) or not rng['cuda']:
        raise ValueError('Full production Python/NumPy/Torch/CUDA RNG states are required')
    import random
    import numpy as np
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator().set_state(rng['torch'])
    # Pinned Torch2.4 CUDA Philox states are exactly16 bytes (seed+offset), not
    # CPU MT19937's much longer byte state. Verified from the original F1 file.
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() != 16
           for value in rng['cuda']):
        raise ValueError('Malformed CUDA RNG state')
    from fh20r1.common import load_checkpoint_model
    from maina_hqnr.training import make_optimizer, verify_optimizer_updates
    with preserve_rng():
        model, _ = load_checkpoint_model(cfg, wd/'candidates/50000', device='cpu')
        optimizer = make_optimizer(model, cfg)
        optimizer.load_state_dict(state['optimizer'])
        verify_optimizer_updates(optimizer, 50000)
        for parameter, value in optimizer.state.items():
            for key in ('exp_avg', 'exp_avg_sq'):
                if (not isinstance(value.get(key), torch.Tensor) or value[key].shape != parameter.shape
                        or not torch.isfinite(value[key]).all()):
                    raise ValueError('Optimizer moment coverage/shape/finiteness differs from A/U')
        del model, optimizer
    sampler = state['sampler']
    stream = BatchStream(sampler['n'], sampler['batch_size'], cfg['seed'])
    stream.load_state_dict(sampler)
    if stream.n != 9714 or stream.batch_size != 48 or stream.epoch*stream.count+stream.cursor != 50000:
        raise ValueError('Sampler cursor does not prove50000 actual optimizer updates')
    replay = BatchStream(sampler['n'], sampler['batch_size'], cfg['seed'])
    while replay.epoch < stream.epoch:
        replay.new_epoch()
    replay.cursor = stream.cursor
    replayed = replay.state_dict()
    if any(not torch.equal(replayed[key], sampler[key]) for key in ('order', 'rotations', 'data_rng', 'aug_rng')):
        raise ValueError('Final sampler does not match the registered original seed stream')
    stream_doc = read_json(wd/'stream_manifest.json')
    if (object_sha(stream_doc) != anchor['stream_sha256'] or stream_doc.get('total_updates') != 50000
            or stream_doc.get('seed') != cfg['seed'] or stream_doc.get('n') != sampler['n']):
        raise ValueError('Registered paired stream identity changed')
    for name in ('meta/training_start_manifest.json', 'init_manifest.json'):
        evidence = read_json(wd/name)
        if not evidence or any(evidence.get(key) != anchor[key] for key in keys):
            raise ValueError('Actual training start/initialization evidence differs')
    if any(status.get(key) != anchor[key] for key in keys):
        raise ValueError('Training status is not bound to the completed A/U run')
    original_ledger = read_json(wd/'candidate_ledger.json')
    expected_entries = [dict(update=step, model_sha256=identities[step]['model_sha256'],
                            state_hash=identities[step]['state_hash']) for step in GRID]
    if (not original_ledger or original_ledger.get('expected_steps') != list(GRID)
            or original_ledger.get('records') != expected_entries):
        raise ValueError('Training candidate ledger does not certify all fifty A/U checkpoints')
    return status, identities


def _ledger(wd, identities, *, current_context=None):
    path = wd/'official/candidate_ledger.json'
    ledger = read_json(path) if path.exists() else dict(schema='MAINA_HQNR_CANDIDATES_v1',
        candidate_grid=list(GRID), candidate_grid_sha256=object_sha(list(GRID)), records=[])
    if (ledger.get('schema') != 'MAINA_HQNR_CANDIDATES_v1' or ledger.get('candidate_grid') != list(GRID)
            or ledger.get('candidate_grid_sha256') != object_sha(list(GRID))):
        raise ValueError('Candidate grid/protocol differs')
    records = []; seen = set()
    for item in ledger['records']:
        step = item['update']
        if step in seen or step not in GRID:
            raise ValueError('Duplicate/foreign candidate ledger step')
        seen.add(step)
        report_path = wd/'official/candidates'/str(step)/'fr.json'
        if item['evaluation_sha256'] != sha256(report_path):
            raise ValueError('Saved candidate evaluation bytes changed')
        record = validate_candidate(read_json(report_path))
        if record['checkpoint_identity'] != identities[step] or record['update'] != step:
            raise ValueError('Saved candidate no longer matches A/U checkpoint')
        if current_context and any(record['metadata'][k] != v for k, v in current_context.items()):
            raise ValueError('Resumed evaluator/native scene population changed')
        records.append(record)
    # Prefix progression ensures each selected maximum got JQM on its original pass.
    if sorted(seen) != list(GRID[:len(seen)]):
        raise ValueError('Candidate ledger is not the registered ascending prefix')
    return ledger, records


def _provenance(cfg, identity, evaluator):
    f = cfg['maina_hqnr']
    return dict(runtime_content_sha256=f['source_identity']['content_sha256'],
        runtime_config_sha256=object_sha(cfg), source_sha=f['source_identity']['content_sha256'],
        runtime_commit=f['source_identity'].get('git_release', ''),
        binding_sha256=f['binding_sha256'], teacher_checkpoint_sha256=identity['teacher_sha256'],
        initial_U_sha256=identity['init_U_sha256'], initial_A_sha256=identity['init_A_sha256'],
        stream_sha256=identity['stream_sha256'], data_sha256=identity.get('data_sha256', ''),
        evaluator_sha256=evaluator['sha256'])


def _bound_population(pop, bindings, split, root=ROOT):
    """Certify saved full-scene hashes against the bound, unchanged native files."""
    item = bindings['dataset_manifest']['splits'][split]
    if (pop.get('split') != split or pop.get('count') != 20 or item.get('count') != 20
            or pop.get('raw_h5_sha256') != item['sha256'] or pop.get('lpan_sha256') != item['lpan_sha256']
            or len(pop.get('scenes', [])) != 20 or pop.get('scene_manifest_sha256') != object_sha(pop['scenes'])):
        raise ValueError('Saved native population does not match the fixed F1/main data binding')
    for key, expected in (('dataroot', item['sha256']), ('lpan_path', item['lpan_sha256'])):
        path = Path(item[key]); path = path if path.is_absolute() else Path(root)/path
        if sha256(path) != expected:
            raise ValueError('Bound native evaluation input bytes changed')
    for index, scene in enumerate(pop['scenes']):
        if (scene.get('scene_index') != index or scene.get('scene_id') != f'{item["sha256"]}:{index}'
                or scene.get('native_scene_sha256') != object_sha(scene.get('input_hashes'))):
            raise ValueError('Full native scene identity/hash evidence differs')
    return pop


def _completion(summary, wd):
    return dict(schema='MAINA_HQNR_COMPLETE_v1', campaign_id=CAMPAIGN_ID,
                run_id=summary['run_id'], attempt=summary['attempt'], actual_updates=50000,
                evaluated_candidates=50, status='COMPLETE', policy=SELECTION_POLICY,
                summary_sha256=object_sha(summary), summary_file_sha256=sha256(wd/'official/summary.json'),
                candidate_ledger_sha256=sha256(wd/'official/candidate_ledger.json'))


def run_postrun(run_dir_or_config, root=ROOT, device='cuda', stopcheck=None):
    from fh12.evaluation import FRMetrics
    from fh20r1.common import load_checkpoint_model
    from maina_hqnr.assets import validate_bindings, load_evaluation_datasets
    from maina_hqnr.common import apply_runtime_policy
    with preserve_rng():
        apply_runtime_policy(root)
        import torch
        if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('Production MAIN-A postrun requires CUDA; CPU is acceptance-only, never a published result')
        cfg, case, bindings, wd = _config(run_dir_or_config, root)
        field = cfg['maina_hqnr']
        if (wd/'official/summary.json').exists():
            summary = verify_summary_for_upload(run_dir_or_config, root, require_completion=False)
            immutable_json(wd/'official/COMPLETE.json', _completion(summary, wd))
            return summary
        validate_bindings(bindings, root=root, server=case['server'], rehash=True)
        status, identities = _training_evidence(cfg, wd)
        datasets = load_evaluation_datasets(bindings, root=root)
        pops = {key: population(value, key) for key, value in datasets.items()}
        engine = FRMetrics(datasets['fr']); evaluator = evaluator_identity(engine.wald, root)
        context = dict(population=pops['fr'], evaluator=evaluator,
                       data_sha256=identities[50000].get('data_sha256'))
        ledger, records = _ledger(wd, identities, current_context=context)
        try:
            for step in GRID[len(records):]:
                if stopcheck and stopcheck():
                    raise InterruptedError('Safe postrun candidate-boundary pause')
                model, identity = load_checkpoint_model(cfg, wd/'candidates'/str(step), device=device,
                                                        expected_source=field['source_identity'])
                best = max((r['fr']['hqnr'] for r in records), default=-math.inf)
                folder = wd/'official/candidates'/str(step)
                record = evaluate_candidate(model, datasets['fr'], device, identity['model_sha256'], step, folder,
                    engine=engine, pop=pops['fr'], evaluator=evaluator, best_hqnr=best,
                    data_sha256=context['data_sha256'], stopcheck=stopcheck)
                del model
                record['checkpoint_identity'] = identity
                atomic_json(folder/'fr.json', record)
                records.append(record)
                ledger['records'].append(dict(update=step, checkpoint_sha256=identity['model_sha256'],
                                               evaluation_sha256=sha256(folder/'fr.json')))
                atomic_json(wd/'official/candidate_ledger.json', ledger)
                print(f'MAIN-A FR {len(records)}/50 step={step} HQNR={record["fr"]["hqnr"]:.9f}', flush=True)
            selected = select_candidates(records, status['actual_updates'])
            reports = {}; evaluated = {}
            for label in SELECTIONS:
                if stopcheck and stopcheck():
                    raise InterruptedError('Safe postrun selected-RR boundary pause')
                candidate = selected[label]; digest = candidate['checkpoint_sha256']; step = candidate['update']
                validate_report(candidate['fr'], 'fr', digest, require_jqm=True)
                folder = wd/'official/evaluations'/digest
                if digest not in evaluated:
                    result_path = folder/'metrics.json'
                    if result_path.exists():
                        result = read_json(result_path)
                        if result.get('fr') != candidate['fr'] or result.get('checkpoint_sha256') != digest:
                            raise ValueError('Cached selected evaluation belongs to another A/U candidate')
                        validate_report(result['rr'], 'rr', digest)
                        if result.get('rr_population') != pops['rr'] or result.get('evaluator') != evaluator:
                            raise ValueError('Cached selected RR population/evaluator changed')
                    else:
                        model, _ = load_checkpoint_model(cfg, wd/'candidates'/str(step), device=device,
                                                         expected_source=field['source_identity'])
                        result = evaluate_rr(model, datasets['rr'], device, digest, pop=pops['rr'], stopcheck=stopcheck)
                        del model
                        result = dict(rr=result['rr'], fr=candidate['fr'], checkpoint_sha256=digest,
                                      evaluator=evaluator, rr_population=result['population'], seconds=result['seconds'])
                        atomic_json(result_path, result)
                    evaluated[digest] = (label, result)
                first_label, result = evaluated[digest]
                reports[label] = dict(update=step, checkpoint_sha256=digest, checkpoint_identity=identities[step],
                    alias_of=first_label if first_label != label else None, rr=result['rr'], fr=result['fr'],
                    evaluation_path=str((folder/'metrics.json').relative_to(wd)),
                    evaluation_manifest_sha256=sha256(folder/'metrics.json'))
            start_path = wd/'meta/training_start_manifest.json'
            start = read_json(start_path) if start_path.exists() else {}
            evaluation_seconds = sum(r['seconds'] for r in records) + sum(v[1]['seconds'] for v in evaluated.values())
            training_seconds = status['training_seconds']
            completed_at = utcnow()
            started_at = start.get('started_at_utc', start.get('started_at', ''))
            wall_seconds = (datetime.fromisoformat(completed_at.replace('Z', '+00:00'))-
                            datetime.fromisoformat(started_at.replace('Z', '+00:00'))).total_seconds()
            summary = dict(campaign_id=CAMPAIGN_ID, run_id=case['run_id'], case=case, attempt=field['attempt'],
                complete=True, status='COMPLETE', actual_updates=50000, selection_policy=SELECTION_POLICY,
                candidate_grid_sha256=object_sha(list(GRID)), expected_candidates=50, evaluated_candidates=50,
                candidate_ledger_sha256=sha256(wd/'official/candidate_ledger.json'),
                training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                evaluation_timing_scope='completed candidate FR/JQM and selected RR compute; interrupted work included only in wall_seconds',
                wall_seconds=wall_seconds, wall_timing_scope='UTC actual training-start to postrun completion; includes safe pauses',
                training_validation_seconds=status.get('evaluation_seconds'), checkpoint_io_seconds=status.get('io_seconds'),
                started_at_utc=started_at, completed_at_utc=completed_at, source_identity=field['source_identity'],
                paper_identity_status='PAPERSET_IDENTITY_UNVERIFIED',
                provenance=_provenance(cfg, identities[50000], evaluator), selections=reports)
            from maina_hqnr.upload import validate_summary
            validate_summary(summary, case)
            immutable_json(wd/'official/summary.json', summary)
            immutable_json(wd/'official/COMPLETE.json', _completion(summary, wd))
            atomic_json(wd/'official/postrun_status.json', dict(status='COMPLETE', actual_updates=50000,
                evaluated_candidates=50, training_rerun=False, updated_at_utc=utcnow()))
            primary = reports['HQNR_MAX50']
            print(f'MAIN-A HQNR_MAX50@{primary["update"]} HQNR={primary["fr"]["hqnr"]:.6f} '
                  f'SCC={primary["rr"]["scc"]:.6f} ERGAS={primary["rr"]["ergas"]:.6f}', flush=True)
            return verify_summary_for_upload(run_dir_or_config, root)
        except InterruptedError:
            atomic_json(wd/'official/postrun_status.json', dict(status='PAUSED_SAFE', actual_updates=50000,
                evaluated_candidates=len(records), comparable=False, training_rerun=False, updated_at_utc=utcnow()))
            raise
        except Exception as exc:
            atomic_json(wd/'official/postrun_status.json', dict(status='PENDING_EVAL_NOT_COMPARABLE', actual_updates=50000,
                evaluated_candidates=len(records), comparable=False, training_rerun=False,
                reason=f'{type(exc).__name__}: {exc}', updated_at_utc=utcnow()))
            raise


def verify_summary_for_upload(run_dir_or_config, root=ROOT, *, require_completion=True):
    """Read-only certification, including all50 candidate A/U and metric hashes."""
    from maina_hqnr.upload import validate_summary
    from maina_hqnr.common import apply_runtime_policy
    apply_runtime_policy(root)
    cfg, case, bindings, wd = _config(run_dir_or_config, root)
    status, identities = _training_evidence(cfg, wd)
    ledger, records = _ledger(wd, identities)
    selected = select_candidates(records, status['actual_updates'])
    summary = validate_summary(read_json(wd/'official/summary.json'), case)
    if (summary['candidate_ledger_sha256'] != sha256(wd/'official/candidate_ledger.json')
            or summary['attempt'] != cfg['maina_hqnr']['attempt']):
        raise ValueError('Summary candidate ledger/attempt changed')
    current_evaluator = evaluator_identity(root=root)
    if any(record['metadata']['evaluator'] != current_evaluator for record in records):
        raise ValueError('Saved evaluator identity differs from the frozen current runtime')
    if summary['provenance'] != _provenance(cfg, identities[50000], current_evaluator):
        raise ValueError('Summary training provenance changed')
    _bound_population(records[0]['metadata']['population'], bindings, 'fr', root)
    if any(record['metadata']['data_sha256'] != identities[50000]['data_sha256'] for record in records):
        raise ValueError('Candidate native data identity differs from completed training')
    for label in SELECTIONS:
        row = summary['selections'][label]; candidate = selected[label]
        path = wd/'official/evaluations'/candidate['checkpoint_sha256']/'metrics.json'
        if (row['checkpoint_identity'] != identities[candidate['update']]
                or row['update'] != candidate['update'] or row['fr'] != candidate['fr']
                or row['evaluation_path'] != str(path.relative_to(wd))
                or row['evaluation_manifest_sha256'] != sha256(path)):
            raise ValueError('Selected full-precision HQNR/RR/FR/AU evidence changed')
        measured = read_json(path)
        if (measured['rr'] != row['rr'] or measured['fr'] != row['fr']
                or measured.get('evaluator') != current_evaluator
                or measured.get('checkpoint_sha256') != candidate['checkpoint_sha256']):
            raise ValueError('Uploaded values differ from saved same-checkpoint metrics')
        pop = _bound_population(measured['rr_population'], bindings, 'rr', root)
        if row['rr'].get('scene_manifest_sha256') != pop['scene_manifest_sha256']:
            raise ValueError('Selected RR scene manifest differs from its bound data')
        for metric, scene in zip(row['rr']['per_scene'], pop['scenes']):
            if any(metric[key] != scene[key] for key in ('scene_index', 'scene_id', 'native_scene_sha256')):
                raise ValueError('Selected RR scene correspondence changed')
    if require_completion or (wd/'official/COMPLETE.json').exists():
        if read_json(wd/'official/COMPLETE.json') != _completion(summary, wd):
            raise ValueError('Missing/changed COMPLETE receipt')
    return summary


execute = run_postrun
