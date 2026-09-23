"""Strict exact50K/validation-min postrun and immutable completion evidence."""
from pathlib import Path
import math
import time

from g23sens.common import (ROOT,read,read_json,read_config,object_sha,sha256,
    source_identity,apply_runtime_policy,run_dir,immutable_json,utcnow)
from g23sens.evaluation import (evaluate_checkpoint,validate_metrics,PROTOCOL,
    evaluator_identity)

SELECTIONS=('EXACT_50000','RR_VAL_ERGAS_MIN')
GRID=tuple(range(1010,50000,1010))+(50000,)
CAMPAIGN_ID='PANDA_G23_SENS_WV3_S45_20260923_v1'


def _config(config_path,root):
    from g23sens.plan import validate_config
    cfg=read_config(config_path);case=validate_config(cfg,root=root,require_bound=True);field=cfg['g23sens']
    apply_runtime_policy(root)
    if field['source_identity']!=source_identity(root):raise ValueError('Evaluation runtime/code differs from training')
    bindings=read_json(field['bindings_path'])
    if object_sha(bindings)!=field['binding_sha256']:raise ValueError('Locked asset binding changed')
    wd=run_dir(case,root,attempt=field['attempt'])
    if Path(cfg['work_dir']).resolve()!=wd.resolve():raise ValueError('Attempt work directory differs')
    return cfg,case,bindings,wd


def selected_evidence(cfg,wd):
    from g23sens.model import load_model
    field=cfg['g23sens'];case=field['case'];status=read_json(wd/'meta/training_status.json')
    if (status.get('training_complete') is not True or status.get('actual_updates')!=50000
            or status.get('status') not in ('TRAIN_COMPLETE_EVAL_PENDING','COMPLETE')):
        raise ValueError('Incomplete/failed training cannot produce a normal completion receipt')
    validation=read_json(wd/'val_records.json')
    if validation is None:raise ValueError('Missing completed native validation grid')
    rows=validation['records']
    if [row['update'] for row in rows]!=list(GRID) or any(not math.isfinite(row['val_ergas']) for row in rows):
        raise ValueError('Validation selection requires the entire fixed native grid')
    best=min(rows,key=lambda r:(r['val_ergas'],r['update']))
    if validation.get('selected_step')!=best['update']:raise ValueError('Secondary is not earliest val-ERGAS minimum')
    expected=dict(config_sha256=object_sha(cfg),source_identity=field['source_identity'],
        bindings_sha256=field['binding_sha256'],case_spec_sha256=case['case_spec_sha256'],
        teacher_sha256=case['fixed']['teacher_checkpoint_sha256'])
    selections={}
    for label,row in zip(SELECTIONS,(rows[-1],best)):
        path=wd/'candidates'/str(row['update']);model,identity=load_model(cfg,path,device='cpu');del model
        if any(identity.get(k)!=v for k,v in expected.items()) or identity['update']!=row['update']:
            raise ValueError('Selection checkpoint provenance/update differs')
        if identity!=row['checkpoint_identity']:raise ValueError('Candidate differs from its recorded validation checkpoint')
        for key in ('init_U_sha256','init_A_sha256','stream_sha256','q_raw_sha256','q_weight_sha256'):
            value=identity.get(key)
            if not isinstance(value,str) or len(value)!=64:raise ValueError('Missing actual '+key)
        if label=='EXACT_50000':
            if identity.get('full_state') is not True or identity.get('precision')!='fp32':raise ValueError('Exact50K must preserve full state')
            statepath=path/'training_state.pt'
            if identity.get('training_state_sha256')!=sha256(statepath):raise ValueError('Exact50K training state bytes changed')
            import torch
            state=torch.load(statepath,map_location='cpu',weights_only=False)
            from g23sens.model import state_hash
            if (state.get('update')!=50000 or state.get('full_state') is not True
                    or state.get('precision')!='fp32'
                    or state.get('scheduler',{}).get('last_epoch')!=50000
                    or not state.get('optimizer',{}).get('state') or not state.get('rng')
                    or state.get('sampler',{}).get('cursor')!=50000
                    or state.get('sampler',{}).get('completed_updates')!=50000
                    or state_hash(state.get('model_state',{}))!=identity.get('state_hash')
                    or any(float(v['step'])!=50000 for v in state['optimizer']['state'].values())
                    or any(state.get(k)!=v for k,v in expected.items())):
                raise ValueError('Exact50K optimizer/scheduler/RNG/sampler evidence missing')
        selections[label]=dict(path=path,identity=identity,val_ergas=row['val_ergas'])
    first,second=[selections[k]['identity'] for k in SELECTIONS]
    for key in ('init_U_sha256','init_A_sha256','stream_sha256','q_raw_sha256','q_weight_sha256'):
        if first[key]!=second[key]:raise ValueError('Selections belong to different training runs')
    return selections


def verify_training_provenance(cfg,wd,selected,bindings):
    identity=selected['EXACT_50000']['identity'];field=cfg['g23sens'];case=field['case']
    start=read_json(wd/'meta/training_start_manifest.json');init=read_json(wd/'init_manifest.json')
    context=('config_sha256','source_identity','bindings_sha256','case_spec_sha256','init_U_sha256',
        'init_A_sha256','stream_sha256','q_raw_sha256','q_weight_sha256','teacher_sha256')
    if any(start.get(k)!=identity[k] or init.get(k)!=identity[k] for k in context):
        raise ValueError('Actual training start/initialization checkpoint context differs')
    stream=read_json(wd/'stream_manifest.json')
    if (start.get('run_id')!=case['run_id'] or start.get('attempt')!=field['attempt']
            or init.get('hashes',{}).get('U')!=identity['init_U_sha256']
            or init.get('hashes',{}).get('A')!=identity['init_A_sha256']
            or object_sha(stream)!=identity['stream_sha256']):
        raise ValueError('Actual U/A/full paired stream evidence changed')
    import torch
    state=torch.load(selected['EXACT_50000']['path']/'training_state.pt',map_location='cpu',weights_only=False)
    if (state['sampler'].get('consumed_prefix_sha256')!=stream['metadata_sha256']
            or state['sampler'].get('metadata_sha256')!=stream['metadata_sha256']
            or stream.get('total_updates')!=50000):raise ValueError('Not all registered sample/augmentation updates were consumed')
    q=read_json(wd/'meta/qweights.json');numbers=read_json(wd/'meta/resolved_numerics.json')
    if (q.get('stats',{}).get('q_raw_sha256')!=identity['q_raw_sha256']
            or q.get('stats',{}).get('w_sha256')!=identity['q_weight_sha256']
            or bindings['cue']['raw_q_sha256']!=identity['q_raw_sha256']
            or q.get('qref')!=case['resolved_values']['q_ref']):raise ValueError('Frozen q/effective weight provenance differs')
    expected=dict(alpha=case['parameters']['alpha'],beta=case['parameters']['beta'],lambda_E=case['parameters']['lambda_E'],
        tau_R_raw=case['fixed']['tau_R_base'],tau_R_scale=case['parameters']['tau_R_scale'],
        tau_R_actual=case['resolved_values']['tau_R_used'],q_ref=case['resolved_values']['q_ref'],
        gradient_routing='U<-L_U; A<-weighted_H_only',precision='fp32',offset_weight=0.,geometry_weight=0.,synthetic_shift_radius=0.)
    if any(numbers.get(k)!=v for k,v in expected.items()):raise ValueError('Actual loss/gate/gradient recipe differs')
    groups=numbers.get('optimizer_groups',[])
    if len(groups)!=2 or [(g['name'],g['peak_lr'],g['weight_decay']) for g in groups]!=[
            ('U',1e-4,.01),('A',case['resolved_values']['A_peak_lr'],.01)]:raise ValueError('Actual optimizer groups differ')
    return start


def _provenance(cfg,identity):
    field=cfg['g23sens'];source=field['source_identity']
    return dict(case_spec_sha256=field['case']['case_spec_sha256'],runtime_config_sha256=object_sha(cfg),
        runtime_commit=source.get('git_release',source.get('commit','')),runtime_content_sha256=source['content_sha256'],
        binding_sha256=field['binding_sha256'],teacher_checkpoint_sha256=identity['teacher_sha256'],
        initial_U_sha256=identity['init_U_sha256'],initial_A_sha256=identity['init_A_sha256'],
        stream_sha256=identity['stream_sha256'],q_raw_sha256=identity['q_raw_sha256'],q_weight_sha256=identity['q_weight_sha256'])


def completion_document(summary,summary_file_sha256):
    """Deterministic document; publication is only after full artifact verification."""
    from g23sens.plan import verify_receipt
    primary=summary['selections']['EXACT_50000'];secondary=summary['selections']['RR_VAL_ERGAS_MIN']
    receipt=dict(schema='G23SENS_VERIFIED_COMPLETION_v1',campaign_id=summary['campaign_id'],
        run_id=summary['run_id'],attempt=summary['attempt'],case_spec_sha256=summary['case']['case_spec_sha256'],
        training_status='COMPLETE_EXACT_50000',actual_updates=summary['actual_updates'],evaluation_status='COMPLETE',
        primary_selection='EXACT_50000',protocol_verified=True,protocol_sha256=object_sha(PROTOCOL),
        checkpoint_sha256=primary['checkpoint_sha256'],rr_checkpoint_sha256=primary['checkpoint_sha256'],
        fr_checkpoint_sha256=primary['checkpoint_sha256'],
        metrics=dict(ergas=primary['rr']['ergas'],hqnr=primary['fr']['hqnr'],d_lambda=primary['fr']['d_lambda'],d_s=primary['fr']['d_s']),
        secondary=dict(selection='RR_VAL_ERGAS_MIN',selected_step=secondary['update'],
            checkpoint_sha256=secondary['checkpoint_sha256'],rr_checkpoint_sha256=secondary['checkpoint_sha256'],
            fr_checkpoint_sha256=secondary['checkpoint_sha256'],alias_of=secondary.get('alias_of')),
        summary_sha256=object_sha(summary),summary_file_sha256=summary_file_sha256,
        provenance=summary['provenance'],paper_identity_status=summary['paper_identity_status'],
        paper_identity_is_not_implied_by_protocol_verification=True,completed_at_utc=summary['completed_at_utc'])
    verify_receipt(summary['case'],receipt)
    return receipt


def _publish_completion(summary,wd):
    document=completion_document(summary,sha256(wd/'official/summary.json'))
    immutable_json(wd/'official/COMPLETE.json',document)
    return document


def run_postrun(config_path,root=ROOT,device='cuda',stopcheck=None):
    from g23sens.assets import validate_bindings
    from g23sens.data import load_evaluation_datasets
    from g23sens.model import load_model
    from g23sens.upload import validate_summary
    cfg,case,bindings,wd=_config(config_path,root);field=cfg['g23sens']
    if (wd/'official/summary.json').exists():
        summary=verify_summary_for_upload(case['run_id'],root,attempt=field['attempt'],require_completion=False)
        _publish_completion(summary,wd)
        return verify_summary_for_upload(case['run_id'],root,attempt=field['attempt'])
    validate_bindings(bindings,root,case['server'],rehash=True,stopcheck=stopcheck)
    selected=selected_evidence(cfg,wd);verify_training_provenance(cfg,wd,selected,bindings)
    datasets=load_evaluation_datasets(bindings,root)
    reports={};evaluated={}
    for label in SELECTIONS:
        if stopcheck and stopcheck():raise InterruptedError('Safe postrun pause')
        evidence=selected[label];identity=evidence['identity'];digest=identity['model_sha256']
        if digest not in evaluated:
            model,_=load_model(cfg,evidence['path'],device=device);folder=wd/'official/evaluations'/digest
            metrics=evaluate_checkpoint(model,datasets,device,digest,folder,stopcheck);del model
            evaluated[digest]=dict(metrics=metrics,path=folder/'metrics.json',selection=label)
        result=evaluated[digest];metrics=result['metrics']
        reports[label]=dict(update=identity['update'],checkpoint_sha256=digest,checkpoint_identity=identity,
            val_ergas=evidence['val_ergas'],alias_of=result['selection'] if result['selection']!=label else None,
            evaluation_path=str(result['path'].relative_to(wd)),evaluation_manifest_sha256=sha256(result['path']),
            rr=metrics['rr'],fr=metrics['fr'])
    status=read_json(wd/'meta/training_status.json');start=read_json(wd/'meta/training_start_manifest.json')
    summary=dict(campaign_id=CAMPAIGN_ID,run_id=case['run_id'],case=case,attempt=field['attempt'],complete=True,
        status='COMPLETE',actual_updates=50000,training_seconds=status['training_seconds'],
        test_seconds=sum(r['metrics']['seconds'] for r in evaluated.values()),
        started_at_utc=start.get('started_at_utc',start.get('started_at','')),
        completed_at_utc=utcnow(),source_identity=field['source_identity'],
        paper_identity_status='VERIFIED' if all(v['metrics']['metadata']['paper_comparable'] for v in evaluated.values()) else 'PAPERSET_IDENTITY_UNVERIFIED',
        provenance=_provenance(cfg,selected['EXACT_50000']['identity']),selections=reports,
        fixed_teacher_training_cost_recharged=False)
    validate_summary(summary,case);immutable_json(wd/'official/summary.json',summary)
    primary=reports['EXACT_50000']
    print(f"G23 EXACT_50000 HQNR={primary['fr']['hqnr']:.6f} SCC={primary['rr']['scc']:.6f} ERGAS={primary['rr']['ergas']:.6f}",flush=True)
    verified=verify_summary_for_upload(case['run_id'],root,attempt=field['attempt'],require_completion=False)
    _publish_completion(verified,wd)
    return verify_summary_for_upload(case['run_id'],root,attempt=field['attempt'])


def verify_summary_for_upload(run,root=ROOT,attempt=0,require_completion=True):
    """Read-only certification; never train, infer, repair metrics, or access Sheets."""
    from g23sens.upload import validate_summary
    from tools.metrics.eval_fr import load_dlpan
    import os
    wd=run_dir(run,root,attempt=attempt)
    cfg,case,bindings,wd=_config(wd/'meta/config.resolved.yaml',root)
    summary=read_json(wd/'official/summary.json');validate_summary(summary,case)
    if not summary.get('complete') or summary['attempt']!=attempt:raise ValueError('Incomplete/foreign attempt summary')
    selected=selected_evidence(cfg,wd);verify_training_provenance(cfg,wd,selected,bindings)
    if summary['provenance']!=_provenance(cfg,selected['EXACT_50000']['identity']):raise ValueError('Summary training provenance changed')
    wald=load_dlpan(os.environ.get('PANCRAFTER_DLPAN',str(Path(root).parent/'DLPan-Toolbox')))
    evaluator=evaluator_identity(wald);verified=True
    for label in SELECTIONS:
        record=summary['selections'][label];identity=selected[label]['identity'];digest=identity['model_sha256']
        path=wd/'official/evaluations'/digest/'metrics.json'
        if (record['checkpoint_sha256']!=digest or record['checkpoint_identity']!=identity
                or record['update']!=identity['update'] or record['val_ergas']!=selected[label]['val_ergas']
                or record['evaluation_path']!=str(path.relative_to(wd))
                or record['evaluation_manifest_sha256']!=sha256(path)):
            raise ValueError('Checkpoint/selection/evaluation bytes changed')
        result=validate_metrics(read_json(path));meta=result['metadata'];cursor=read_json(path.parent/'evaluation_cursor.json')
        if (meta['checkpoint_sha256']!=digest or meta['data_manifest_sha256']!=cfg['g23sens']['binding_sha256'] or meta['evaluator']!=evaluator
                or not cursor.get('complete') or any(meta.get(k)!=v for k,v in cursor['context'].items())):
            raise ValueError('Evaluation cursor/checkpoint/runtime changed')
        for split in ('rr','fr'):
            report=result[split];population=bindings['evaluation'][split+'_manifest']
            if (record[split]!=report or report['n_scenes']!=population['count']
                    or report['scene_ids']!=population['scene_ids'] or report['per_scene']!=cursor['rows'][split]
                    or report['identity_validation']!=population['identity_validation']):
                raise ValueError('Full native population/identity evidence changed')
            expected=bool(population['identity_validation'].get('verified') and getattr(wald,'__file__',None))
            if report['paper_comparable']!=expected:raise ValueError('Paper equivalence overstated')
            verified=verified and expected
    if summary['paper_identity_status']!=('VERIFIED' if verified else 'PAPERSET_IDENTITY_UNVERIFIED'):
        raise ValueError('Summary paper-equivalence claim differs')
    complete_path=wd/'official/COMPLETE.json'
    if require_completion or complete_path.exists():
        if read_json(complete_path)!=completion_document(summary,sha256(wd/'official/summary.json')):
            raise ValueError('Missing or changed verified completion receipt')
    return summary


execute=run_postrun
