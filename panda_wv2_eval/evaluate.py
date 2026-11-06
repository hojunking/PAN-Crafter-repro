"""Complete native WV2 metrics and lossless predictions, with resumable scenes."""
import gc
import hashlib
import os
import time
import numpy as np
import torch
from .common import ROOT, OUT, PLAN, digest, read, sha, write, now, verify_release, verify_dependency_completion


def validate_report(report, case):
    if (report['case_id'] != case['case_id'] or report['source_checkpoint_sha256'] != case['checkpoint_sha256']
            or report['target_dataset'] != 'WV2' or report['training_updates'] != 0
            or not report['weights_unchanged'] or report['data'] != PLAN['data']
            or report['source_release'] != verify_release()):
        raise ValueError('WV2 result identity differs')
    if sha(OUT/'runs'/case['case_id']/'cost.json') != report['cost_file_sha256']:
        raise ValueError('WV2 cost evidence changed')
    for split in ('rr', 'fr'):
        rows = report[split]['per_scene']
        if [r['scene_id'] for r in rows] != [f'WV2:{split}:{i}' for i in range(20)]:
            raise ValueError('Incomplete or reordered WV2 result')
        for key, mean in report[split]['metrics'].items():
            values = [r['metrics'][key] for r in rows]
            if not np.isfinite(values).all() or float(np.mean(values)) != mean:
                raise ValueError('WV2 aggregate differs from scene evidence')
        for row in rows:
            path = OUT / 'runs' / case['case_id'] / row['prediction_npy']
            if sha(path) != row['npy_sha256']:
                raise ValueError('Exported WV2 prediction changed')
    return report


def evaluate(case):
    from panda_stage2_condq.training import configure_precision
    from ta2.evaluation import evaluator_identity, rr_scene, fr_scene, _save_array, JQM_VARIANT
    from directshift48.evaluation import metric_values
    from donor30.cost import _gpu_pids, _stage2_labels
    from tools.metrics.eval_fr import load_dlpan
    from experiment_cost.measurement import profile_stage
    from panda_tune30.cost.profiling import account_student_lanczos4
    from .data import Sources
    from .model import Deployment
    dependency = verify_dependency_completion()
    release = verify_release()
    if _gpu_pids() - {os.getpid()}:
        raise RuntimeError('WV2 waits for an idle GPU after the WV3 queue')
    directory = OUT / 'runs' / case['case_id']
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'evaluation.json'
    if report_path.exists(): return validate_report(read(report_path), case)
    configure_precision()
    source = Sources()
    model = Deployment(case).to('cuda:0').eval()
    identity = evaluator_identity()
    if identity != model.source_evaluator:
        raise ValueError('Source metric code/environment differs from frozen WV3 evaluation')
    before = model.weight_hash()
    # Native cost includes interp23 and input checks; never copy a WV3 cost value.
    cost_path = directory / 'cost.json'
    if cost_path.exists():
        cost = read(cost_path)
        if (cost['source_checkpoint_sha256'] != case['checkpoint_sha256'] or cost['source_release'] != release
                or cost['cost_sha256'] != digest({k:v for k,v in cost.items() if k!='cost_sha256'})):
            raise ValueError('Cached cost belongs to another deployment')
    else:
        cost = profile_stage(model, source, device='cuda:0', warmup=10, repeats=30, rows={'rr':0,'fr':0},
            deadline_unix=time.time()+600, gpu_isolation_assertion={'profiler_host_pid':os.getpid(),
                'other_compute_pids':[], 'mode':'AFTER_WV3_COMPLETED', 'server':'s1'})
        _stage2_labels(cost)
        account_student_lanczos4(cost)
        cost.update(weight_basis='ACTUAL_FROZEN_WV3_CHECKPOINT',
                    weight_dependence='Actual selected checkpoint measured on native WV2 scene0',
                    memory_definition='Per-process peak allocated MiB including A+S, inputs and workspace; no training Teacher/optimizer')
        for split, obs in cost['observations'].items():
            obs['sheet_fields']['FLOPs(G)'] = obs['accounting']['arithmetic_flops_covered_estimate']/1e9
            obs['sheet_fields']['Cost_scope'] = (
                f'WV2 {split.upper()} native batch1 FP32; A+L4+DIFF+interp23+W{case["width"]} D123 UBIL; '
                'synchronized wall median 10 warmup/30 repeats, scene0; includes preprocessing/checks; '
                'excludes IO/H2D/metrics/Teacher; Mem(MB)=peak allocated MiB; FLOPs=2*MAC PARTIAL estimate; '
                'not the prepared-input CUDA-Graph protocol used by WV3-main')
        cost.update(source_checkpoint_sha256=case['checkpoint_sha256'], source_release=release)
        cost['cost_sha256'] = digest(cost)
        write(cost_path, cost)
    if _gpu_pids() - {os.getpid()}: raise RuntimeError('GPU isolation lost during profiling')
    wald = load_dlpan(os.environ.get('PANCRAFTER_DLPAN', str(ROOT.parent / 'DLPan-Toolbox')))
    report = {'schema':'WV2_ZERO_SHOT_EVALUATION_V1', 'case_id':case['case_id'],
        'source_checkpoint_sha256':case['checkpoint_sha256'], 'source_run_id':case['source_run_id'],
        'source_step':case['source_step'], 'training_dataset':'WV3', 'target_dataset':'WV2',
        'training_updates':0, 'adaptation':False, 'inference_tta':False, 'reference_warp':False,
        'source_release':release, 'data':PLAN['data'], 'evaluator':identity, 'JQM_variant':JQM_VARIANT,
        'FR_MTF_sensor':'WV2', 'FR_PAN_GNyq_JQM':.11, 'RR_crop':'[20:-21,20:-21]',
        'dependency':dependency, 'selection':'fixed before WV2 evaluation; all three reported', 'started_at':now()}
    with torch.no_grad():
        for split in ('rr', 'fr'):
            rows = []
            for i in range(20):
                receipt = directory / split / f'scene_{i:02d}.json'
                if receipt.exists():
                    row = read(receipt)
                    if (row['source_release'] != release or row['source_checkpoint_sha256'] != case['checkpoint_sha256']
                            or row['data_sha256'] != PLAN['data'][split]['sha256']
                            or row['scene_id'] != f'WV2:{split}:{i}'
                            or sha(directory / row['prediction_npy']) != row['npy_sha256']):
                        raise ValueError('Cached WV2 scene differs from fixed evaluation identity')
                else:
                    tick = time.monotonic()
                    batch = source.batch(split, [i], device='cuda:0')
                    assert batch.gt is None and batch.lineage.sensor == 'WV2'
                    prediction = model.native_prediction(batch)
                    raw = ((prediction['y'][0].float()+1)*(source.peak/2)).cpu().numpy()
                    arrays = {k:v[0] for k,v in source.raw(split, [i]).items()}
                    scores = (rr_scene(raw, arrays['gt'], 'WV2') if split=='rr' else
                              fr_scene(raw, arrays['pan'], arrays['lms'], arrays['ms'], 'WV2', wald))
                    metrics = metric_values(scores, split, 8)
                    relative = f'{split}/prediction/scene_{i:02d}.npy'
                    npy_sha = _save_array(directory / relative, raw)
                    row = {'scene_id':f'WV2:{split}:{i}', 'metrics':metrics,
                        'source_checkpoint_sha256':case['checkpoint_sha256'], 'source_release':release,
                        'data_sha256':PLAN['data'][split]['sha256'], 'prediction_npy':relative,
                        'npy_sha256':npy_sha, 'raw_prediction_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                        'correction_dy_dx':prediction['u'][0].cpu().tolist(),
                        'elapsed_seconds':time.monotonic()-tick}
                    write(receipt, row)
                    del batch, prediction
                rows.append(row)
                print(case['case_id'], split, i+1, '/20', flush=True)
            report[split] = {'per_scene':rows, 'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}}
    if model.weight_hash() != before or evaluator_identity() != identity:
        raise ValueError('Frozen weights/evaluator changed during WV2 evaluation')
    report.update(weights_unchanged=True, model_sha256=before, finished_at=now(),
                  elapsed_seconds=sum(r['elapsed_seconds'] for s in ('rr','fr') for r in report[s]['per_scene']),
                  cost=cost['observations']['rr']['sheet_fields'], cost_file_sha256=sha(cost_path))
    validate_report(report, case)
    write(report_path, report)
    del model
    gc.collect();torch.cuda.empty_cache()
    return report
