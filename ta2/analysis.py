"""Recoverable local three-seed/paired aggregation; outbox only, no network.

Each aggregate version is keyed by its own input evidence, so another case
finishing neither overwrites it nor creates redundant scientific rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ta2.aggregate import paired_difference, scene_paired_interval, seed_aggregate
from ta2.common import atomic_json, digest, file_sha, immutable_json, load_json
from ta2.evaluation import check_seal
from ta2.plan import LANES

SELECTORS=('HQNR_MAX50','EXACT_FINAL')
QUALITY=('hqnr','d_s','d_lambda','jqm','ergas','scc','sam','psnr','ssim','q4','q8','rmse','cc')
MANDATORY_PAIRS=(
    ('B03','B00'),('B04','B00'),('B07','B00'),('B05','B03'),
    ('M05','M01'),('M07','M03'),('M07','M05'),('M07','M06'),('M07','B03'),
    ('S07','M07'),('S06','M07'),('U00','B00'),('U01','B03'),('U02','M05'),('U03','M07'),
    ('X01','M05'),('X00','M07'),('X02','M07'),('X03','X02'),
    ('D00','B00'),('D01','B03'),('D02','M05'),('D03','M07'))


def _diagnostic_path(run,selector,step):
    # Both direct caller layout and controller's selector namespace supported;
    # exact identity is validated after resolving the one existing receipt.
    candidates=[run/'diagnostics'/selector/'diagnostics'/f'step_{step:06d}'/'report.json',
                run/'diagnostics'/f'step_{step:06d}'/'report.json']
    found=[p for p in candidates if p.exists()]
    if len(found)!=1:raise ValueError('Missing or ambiguous selected diagnostic receipt')
    return found[0]


def _values(evaluation,probe):
    values={k:float(evaluation[s][k]) for s in ('rr','fr') for k in QUALITY if k in evaluation[s]}
    for row in probe.get('response_summary',[]):
        if (row['probe']=='AXIS16' and row['radius'] is None and row['mode']=='REESTIMATE_CURRENT_INPUT'
                and row['measurement_domain']=='NATIVE_PATCH' and row['support_scope']=='normal'):
            values['gain_'+row['split']]=row['gain']
            values['offset_mae_'+row['split']]=row['component_mae']
            values['offset_epe_'+row['split']]=row['epe']
    for row in probe.get('geometry_summary',[]):
        if row['after_epe_mean'] is not None:
            values[f"proxy_{row['split']}_{row['target']}_{row['descriptor']}"]=row['after_epe_mean']
    for split,row in probe.get('correction_summary',{}).items():
        values['mean_norm_c_'+split]=row['mean_norm'];values['norm_mean_c_'+split]=row['norm_mean']
    for row in probe.get('cost',[]):
        values['pipeline_ms_'+row['split']]=row['pipeline_ms']
        values['registration_ms_'+row['split']]=row['registration_ms']
    if any(not np.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite completed experiment scalar')
    return values


def collect_runs(lane,queue):
    """Only actual completed experiment reports enter native metrics."""
    lane=Path(lane);records=[];evidence={};evaluations={}
    for row in queue:
        run=lane/'runs'/row['run_id'];statuspath=run/'status.json'
        if not statuspath.exists():continue
        status=load_json(statuspath);state=status.get('status')
        if state not in ('COMPLETE','FAILED','NUMERICAL_FAILURE','BLOCKED'):continue
        ident=load_json(run/'identity.json') if (run/'identity.json').exists() else load_json(lane/'identity.json')
        initial=load_json(run/'initialization.json') if (run/'initialization.json').exists() else {}
        source=ident['source']
        if not isinstance(source,str):source=digest(source)
        base=dict(seed=row['seed'],replica=row['replica'],dataset=row['dataset'],server=row['server'],
                  case_id=row['case_id'],run_id=row['run_id'],source_revision=source,
                  initial_u_sha256=initial.get('U_sha256'),initial_a_sha256=initial.get('A_sha256'),
                  native_stream_sha256=initial.get('native_stream_sha256'),record_type='NATIVE',
                  attempt=1,status=state)
        evidence[row['run_id']]=dict(status_identity_sha256=digest({k:status.get(k) for k in ('status','error','seed')}),
                                     initialization=initial,source_revision=source)
        if state!='COMPLETE':
            records.extend(dict(base,selector=s) for s in SELECTORS);continue
        if any(base[k] is None for k in ('initial_u_sha256','native_stream_sha256')):
            raise ValueError('Completed experiment lacks paired initialization/stream evidence')
        selection=load_json(run/'selection'/'selection.json')
        if (selection.get('status')!='HQNR_SELECTION_COMPLETE' or selection.get('candidates_complete')!=50
                or selection.get('candidates_expected')!=50 or selection.get('test_aware') is not True):
            raise ValueError('Completed experiment lacks50/50 HQNR selector proof')
        for selector in SELECTORS:
            path=run/'exports'/selector/'report.json'
            evaluation=check_seal(load_json(path));step=evaluation['completed_step']
            if (not evaluation.get('complete') or not evaluation.get('rr_fr_same_checkpoint')
                    or evaluation.get('record_type')!='NATIVE' or evaluation.get('gt_in_inference') is not False or evaluation['context']!=ident
                    or evaluation['rr'].get('n_scenes')!=20 or evaluation['fr'].get('n_scenes')!=20
                    or evaluation['fr'].get('masking') is not False
                    or evaluation['fr'].get('reference')!='original_PAN_and_native_LMS' or evaluation['fr'].get('support')!='full512'):
                raise ValueError('Completed native report has wrong data/source/GT/masking/coverage')
            if selector=='HQNR_MAX50' and (step!=selection['completed_step'] or evaluation['checkpoint_sha256']!=selection['checkpoint_sha256']):
                raise ValueError('HQNR selected output was taken from a different checkpoint')
            if selector=='EXACT_FINAL' and (step!=row['updates'] or evaluation['checkpoint_sha256']!=selection['exact_final_sha256']):
                raise ValueError('Exact output was taken from a nonfinal checkpoint')
            probe_path=_diagnostic_path(run,selector,step);probe=check_seal(load_json(probe_path))
            if probe['status']!='COMPLETE' or probe['checkpoint_sha256']!=evaluation['checkpoint_sha256']:
                raise ValueError('Completed run lacks complete same-checkpoint diagnostic evidence')
            records.append(dict(base,selector=selector,completed_step=step,checkpoint_sha256=evaluation['checkpoint_sha256'],
                                **_values(evaluation,probe)))
            evidence[row['run_id']][selector]=dict(evaluation_sha256=file_sha(path),diagnostics_sha256=file_sha(probe_path))
            evaluations[(row['case_id'],row['seed'],selector)]=evaluation
    return records,evidence,evaluations


def _scope_context(row,selector,input_sha,kind,control=None):
    context=dict(row,selector=selector,seed=None,replica=None,attempt=1,status='MEASURED',
                 run_id=f'TA2_{kind}_{row["case_id"]}_{control or "SELF"}_{selector}_{input_sha[:20]}',
                 source_revision='aggregate_input_sha256:'+input_sha,
                 checkpoint_sha256='',control_id=control or '')
    return context


def _metric_records(records,metric):
    # Numeric proxy N/A is excluded for that metric, without fabricating a
    # failure or counting a successful run as a measured proxy observation.
    return [r if r['status']!='COMPLETE' or metric in r else dict(r,status='NOT_APPLICABLE') for r in records]


def refresh_analysis(lane,plan,server,outbox):
    from ta2.reporting import aggregate_records,make_record
    lane=Path(lane);queue=plan.queue(server);seeds=LANES[server][3]
    records,evidence,evaluations=collect_runs(lane,queue)
    templates={r['case_id']:r for r in queue}
    cases=sorted(templates);by_group={}
    for row in records:by_group.setdefault((row['case_id'],row['selector']),[]).append(row)
    outputs=[];summary_files=[]
    folder=lane/'analysis';folder.mkdir(parents=True,exist_ok=True)
    implementation={name:file_sha(Path(__file__).with_name(name)) for name in ('analysis.py','aggregate.py','reporting.py')}

    def save_group(name,input_evidence,build):
        sha=digest(dict(evidence=input_evidence,implementation=implementation));path=folder/'artifacts'/f'{name}_{sha}.json'
        if path.exists():payload=load_json(path)
        else:
            payload=build(sha);immutable_json(path,payload)
        if payload['input_sha256']!=sha:raise ValueError('Aggregate input identity differs')
        outputs.extend(payload['sheet_records']);summary_files.append(str(path.relative_to(lane)))

    for case in cases:
        for selector in SELECTORS:
            group=by_group.get((case,selector),[])
            if not group:continue
            metric_names=sorted({k for r in group if r['status']=='COMPLETE' for k in r
                                 if (k in QUALITY or k.startswith(('gain_','offset_','proxy_','mean_norm_','norm_mean_','pipeline_ms_','registration_ms_')))
                                 and isinstance(r[k],(int,float))}) or ['hqnr']
            input_evidence={r['run_id']:evidence[r['run_id']] for r in group}
            def build_seed(sha,group=group,case=case,selector=selector,metric_names=metric_names):
                aggregates=[];sheet=[];context=_scope_context(templates[case],selector,sha,'SEED')
                for metric in metric_names:
                    agg=seed_aggregate(_metric_records(group,metric),metric=metric,expected_seeds=seeds)
                    aggregates.append(agg);sheet.extend(aggregate_records(context,agg))
                return dict(input_sha256=sha,aggregates=aggregates,sheet_records=sheet)
            save_group(f'seed_{case}_{selector}',input_evidence,build_seed)

    pairs={('TA2-'+t,'TA2-'+c) for t,c in MANDATORY_PAIRS}
    pairs|={(case,template['control_id']) for case,template in templates.items() if template.get('control_id')}
    for treatment,control in sorted(pairs):
        if treatment not in templates or control not in templates:continue
        for selector in SELECTORS:
            t=by_group.get((treatment,selector),[]);c=by_group.get((control,selector),[])
            if not t and not c:continue
            metric_names=sorted({k for r in t+c if r['status']=='COMPLETE' for k in r
                                 if (k in QUALITY or k.startswith(('gain_','offset_','proxy_','mean_norm_','norm_mean_','pipeline_ms_','registration_ms_')))
                                 and isinstance(r[k],(int,float))}) or ['hqnr']
            input_evidence={r['run_id']:evidence[r['run_id']] for r in t+c}
            def build_pair(sha,t=t,c=c,treatment=treatment,control=control,selector=selector,metric_names=metric_names):
                aggregates=[];sheet=[];cis=[];context=_scope_context(templates[treatment],selector,sha,'PAIR',control)
                for metric in metric_names:
                    agg=paired_difference(_metric_records(t,metric),_metric_records(c,metric),metric=metric,expected_seeds=seeds)
                    aggregates.append(agg);sheet.extend(aggregate_records(context,agg))
                    if metric not in QUALITY:continue
                    split='fr' if metric in ('hqnr','d_s','d_lambda','jqm') else 'rr'
                    for seed in seeds:
                        key_t,key_c=(treatment,seed,selector),(control,seed,selector)
                        if key_t not in evaluations or key_c not in evaluations:continue
                        et,ec=evaluations[key_t],evaluations[key_c]
                        if metric not in et[split] or metric not in ec[split]:continue
                        ci=scene_paired_interval(et[split]['per_scene'],ec[split]['per_scene'],metric=metric)
                        cis.append(dict(ci,seed=seed,metric=metric,selector=selector,treatment=treatment,control=control))
                        fields=dict(Metric_name=metric,Estimate=ci['mean'],CI_low=ci['ci95'][0] if ci['ci95'] else None,
                                    CI_high=ci['ci95'][1] if ci['ci95'] else None,N_scenes=ci['n_scenes'],N_train_seeds=1,
                                    Train_seed=seed,Split_probe=split,Paired_scope='scene_cluster_within_one_seed',
                                    Reason='Scene CI separate from ddof1 seed STD; geographic independence unverified')
                        sheet.append(make_record(context,'PAIRED',f'{metric}/scene_CI/{seed}',fields))
                        for stat in ('worst','best'):
                            item=ci[stat];sheet.append(make_record(context,'PAIRED',f'{metric}/{stat}/{seed}',
                                dict(fields,Metric_name=metric+'/'+stat,Estimate=item['difference'],Scene_ID=str(item['scene_index']))))
                        sheet.append(make_record(context,'PAIRED',f'{metric}/scene_median/{seed}',dict(fields,Metric_name=metric+'/median',Estimate=ci['median'])))
                return dict(input_sha256=sha,aggregates=aggregates,scene_cluster_intervals=cis,sheet_records=sheet)
            save_group(f'pair_{treatment}_{control}_{selector}',input_evidence,build_pair)

    # Preregistered epsilon x structure interaction in native-LMS factorial.
    for selector in SELECTORS:
        cells={c:by_group.get(('TA2-'+c,selector),[]) for c in ('M04','M05','M06','M07')}
        input_evidence={r['run_id']:evidence[r['run_id']] for group in cells.values() for r in group}
        if not input_evidence:continue
        def build_interaction(sha,selector=selector,cells=cells):
            aggregates=[];sheet=[];context=_scope_context(templates['TA2-M07'],selector,sha,'INTERACTION','TA2-M06_M05_M04')
            for metric in QUALITY:
                left=paired_difference(cells['M07'],cells['M06'],metric=metric,expected_seeds=seeds)
                right=paired_difference(cells['M05'],cells['M04'],metric=metric,expected_seeds=seeds)
                # Matching within each subtraction is insufficient: the two
                # factorial arms must share the same seed/U/stream provenance.
                paired_difference(cells['M07'],cells['M05'],metric=metric,expected_seeds=seeds)
                rows=[]
                for a,b in zip(left['rows'],right['rows']):
                    okay=a['status']==b['status']=='COMPLETE'
                    rows.append(dict(seed=a['seed'],replica=a.get('replica'),status='COMPLETE' if okay else 'PAIR_INCOMPLETE',
                                     difference=a['difference']-b['difference'] if okay else None,
                                     treatment='TA2-M07',control='TA2-M06_M05_M04'))
                numbers=[r['difference'] for r in rows if r['status']=='COMPLETE']
                if not numbers:continue
                agg=dict(record_type='PAIRED',metric='epsilon_struct_interaction/'+metric,rows=rows,n_pairs=len(numbers),n_expected=3,
                         mean=float(np.mean(numbers)),std=float(np.std(numbers,ddof=1)) if len(numbers)>1 else None,
                         std_ddof=1,status='COMPLETE' if len(numbers)==3 else 'INCOMPLETE',sampling_unit='paired_training_seed_factorial',
                         formula='(M07-M06)-(M05-M04); descriptive_only_three_seeds')
                aggregates.append(agg);sheet.extend(aggregate_records(context,agg))
            return dict(input_sha256=sha,aggregates=aggregates,sheet_records=sheet)
        save_group('interaction_'+selector,input_evidence,build_interaction)
    # An empty initial lane has no fabricated success/failure rows.
    outbox.enqueue_many(outputs)
    result=dict(status='LOCAL_ANALYSIS_REFRESHED',run_evidence_count=len(evidence),record_count=len(outputs),
                artifact_files=summary_files,input_sha256=digest(evidence),network_writes=False)
    atomic_json(folder/'latest.json',result)
    return result
