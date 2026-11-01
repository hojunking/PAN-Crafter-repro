"""Read authenticated local QB outputs; CPU only, no inference or experiment writes."""
import hashlib
import json
import math
import statistics as st
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
PACKAGES = ('panda_s1loss9', 'panda_sensor_refine', 'panda_sensor_focus', 'panda_qb_refit', 'panda_qb_structure')
METRICS = ('HQNR', 'ERGAS', 'PSNR', 'D_s', 'D_lambda')
TARGET = {'HQNR': .9315, 'ERGAS': 3.5}
read = lambda p: json.loads(p.read_text())
mean = lambda values: st.mean(values) if values else None


def correlation(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    mx, my = st.mean(x), st.mean(y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y)) / math.sqrt(
        sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))


def metric_row(e):
    return {'step': e['state_before']['completed_update'],
            **{m:e['fr']['metrics'][m] for m in ('HQNR','D_s','D_lambda')},
            **{m:e['rr']['metrics'][m] for m in ('ERGAS','PSNR')}}


def representative(rows):
    qualified = [r for r in rows if r['HQNR'] >= TARGET['HQNR']]
    return min(qualified, key=lambda r:(r['ERGAS'],-r['HQNR'],r['step'])) if qualified else max(
        rows, key=lambda r:(r['HQNR'],-r['ERGAS'],-r['step']))


def parse_time(value):
    return datetime.fromisoformat(value) if value else None


def evidence_check(e):
    state = e['state_before']
    assert all(e[k] == state for k in ('state_after','rr_state','fr_state'))
    assert all(e['qualification'][k] for k in ('same_state_verified','metrics_complete','metrics_finite'))
    assert e['metric_function_injection'] is False and e['full_method_eligible'] is True
    for split in ('rr','fr'):
        rows=e[split]['per_scene']
        assert len(rows)==20 and {r['scene_id'] for r in rows}=={f'QB:{split}:{i}' for i in range(20)}
        for m in (('ERGAS','PSNR') if split=='rr' else ('HQNR','D_s','D_lambda')):
            assert abs(st.mean(r['metrics'][m] for r in rows)-e[split]['metrics'][m])<1e-10


def paired(current, control):
    a,b=metric_row(current),metric_row(control)
    result={'current_step':a['step'],'control_step':b['step'],
            'aggregate_delta':{m:a[m]-b[m] for m in METRICS},
            'same_aligner_hash':current['aligner_state_sha256']==control['aligner_state_sha256'],
            'per_scene':{},'scene_values':{},'scene_summary':{}}
    for split in ('rr','fr'):
        aa={r['scene_id']:r for r in current[split]['per_scene']}
        bb={r['scene_id']:r for r in control[split]['per_scene']}
        assert aa.keys()==bb.keys()
        ms=('ERGAS','PSNR') if split=='rr' else ('HQNR','D_s','D_lambda')
        delta={key:{m:aa[key]['metrics'][m]-bb[key]['metrics'][m] for m in ms} for key in aa}
        result['per_scene'][split]=delta
        result['scene_values'][split]={key:{'current':{m:aa[key]['metrics'][m] for m in ms},
            'control':{m:bb[key]['metrics'][m] for m in ms},
            'shift_norm_px':aa[key]['correction_norms_px']} for key in aa}
        result['scene_summary'][split]={m:{'mean_delta':st.mean(v[m] for v in delta.values()),
            'median_delta':st.median(v[m] for v in delta.values()),
            'better_count':sum(v[m]<0 if m in ('ERGAS','D_s','D_lambda') else v[m]>0 for v in delta.values()),
            'worse_count':sum(v[m]>0 if m in ('ERGAS','D_s','D_lambda') else v[m]<0 for v in delta.values()),
            'mean_delta_without_scene0':st.mean(v[m] for k,v in delta.items() if k!=f'QB:{split}:0')}
            for m in ms}
        result[f'max_{split}_shift_delta']=max(abs(x-y) for k in aa for x,y in zip(
            aa[k]['correction_dy_dx'],bb[k]['correction_dy_dx']))
        if split=='fr':
            norms=[aa[k]['correction_norms_px'] for k in aa]
            result['delta_Ds_vs_shift_norm_correlation']=correlation(norms,[delta[k]['D_s'] for k in aa])
            spatial=[];spectral=[];interaction=[]
            for key in aa:
                ds,dl=delta[key]['D_s'],delta[key]['D_lambda']; old=bb[key]['metrics']
                spatial.append(-ds*(1-old['D_lambda'])); spectral.append(-dl*(1-old['D_s']))
                interaction.append(ds*dl)
            result['HQNR_delta_exact_scene_decomposition']={
                'spatial_at_old_Dlambda':st.mean(spatial),'spectral_at_old_Ds':st.mean(spectral),
                'interaction':st.mean(interaction),'sum':st.mean(spatial)+st.mean(spectral)+st.mean(interaction)}
    return result


def main():
    report={'captured_utc':datetime.now(timezone.utc).isoformat(),'target':TARGET,
            'runs':[],'pairwise':[],'factorial_10K':[],
            'limitations':['Local s1 only; s3 is reviewed separately from Sheet.',
                'Current campaign status is a read-only point-in-time snapshot; running steps can advance.',
                'Exploratory test-aware selection; representative minima and fixed-step comparisons are distinct.',
                'No inference, GPU access, checkpoint changes, or queue edits.']}
    evidence={};runs={}
    for package in PACKAGES:
        for statepath in sorted((ROOT/'work_dir'/package).glob('*/s1/QB/campaign_state.json')):
            state=read(statepath)
            for run in state['runs']:
                folder=statepath.parent/'runs'/run['run_id']/'attempt_01'
                if run['case'].startswith('S1_') or not (folder/'config.json').exists():continue
                cfg=read(folder/'config.json');evs=[read(p) for p in sorted(folder.glob('evaluation_*.json'))]
                if not evs:continue
                for ev in evs:evidence_check(ev)
                rows=[metric_row(e) for e in evs];key=(run['case'],cfg['replica'])
                evidence[key]={e['state_before']['completed_update']:e for e in evs}
                spec=cfg['stage2_spec'];training=cfg['training_spec']
                item={'package':package,'case':key[0],'replica':key[1],'run_id':run['run_id'],
                    'status':run['status'],'recorded_step':run.get('step'),'latest':rows[-1],
                    'source':str(folder.relative_to(ROOT)),'rows':rows,'representative':representative(rows),
                    'unconditional_min_ERGAS':min(rows,key=lambda r:r['ERGAS']),
                    'joint_passes':[r for r in rows if r['HQNR']>=.9315 and r['ERGAS']<=3.5],
                    'hqnr_pass_count':sum(r['HQNR']>=.9315 for r in rows),
                    'settings':{**{k:spec[k] for k in ('alpha','gamma','student_width','input_variant','sampling_variant')},
                        'lr':training['lr'],'weight_decay':training['weight_decay'],'updates':cfg['planned_horizon'],
                        'batch':training['effective_batch_size'],'donor':cfg['donor_checkpoint_sha256']},
                    'calibration_historical_comparison':cfg.get('calibration_historical_comparison'),
                    'started_at':run.get('started_at'),'ended_at':run.get('completed_at',run.get('early_stopped_at')),
                    'mean_25_30K':{m:mean([r[m] for r in rows if 25000<=r['step']<=30000]) for m in METRICS}}
                if item['ended_at']:
                    item['wall_hours']=(parse_time(item['ended_at'])-parse_time(item['started_at'])).total_seconds()/3600
                if package=='panda_qb_structure' or key in [('QB_A200_E030','R1'),('QB_BASE','R2')]:
                    traces=[];elapsed=0.
                    with (folder/'train.jsonl').open() as f:
                        for line in f:
                            if not line.endswith('\n'):break
                            value=json.loads(line);elapsed+=value['seconds_including_IO']
                            if 'trace' in value:traces.append((value['step'],value['trace']))
                    item['observed_update_hours']=elapsed/3600
                    item['training_windows']={}
                    for start,end in [(1000,5000),(5000,10000),(15000,20000),(25000,30000)]:
                        selected=[t for s,t in traces if start<=s<=end]
                        item['training_windows'][f'{start}_{end}']={k:mean([t[k] for t in selected if k in t])
                            for k in ('gt_raw','hard_weighted','loss','d_e_mean','r_q_roi_mean','w_h_mean','w_h_p95','warp_support_fraction')}
                latest=evs[-1]
                item['shift_stats']={split:{'mean':st.mean(p['correction_norms_px'] for p in latest[split]['per_scene']),
                    'median':st.median(p['correction_norms_px'] for p in latest[split]['per_scene']),
                    'max':max(p['correction_norms_px'] for p in latest[split]['per_scene'])} for split in ('rr','fr')}
                report['runs'].append(item);runs[key]=item
    baselines={'R1':('QB_A200_E030','R1'),'R2':('QB_BASE','R2'),'R3':('QB4_CONTROL','R3')}
    for key,item in runs.items():
        if item['package']!='panda_qb_structure' or item['case']=='QB4_CONTROL':continue
        control=baselines[item['replica']]
        pair={'case':item['case'],'replica':item['replica'],'control':list(control),
            'representative':paired(evidence[key][item['representative']['step']],
                evidence[control][runs[control]['representative']['step']]),
            'same_step':[],'delta_trajectory':[]}
        common=sorted(evidence[key].keys()&evidence[control].keys())
        for step in common:
            a,b=metric_row(evidence[key][step]),metric_row(evidence[control][step])
            pair['delta_trajectory'].append({'step':step,**{m:a[m]-b[m] for m in METRICS}})
        for step in sorted({1000,5000,10000,20000,25000,30000,common[-1]} & set(common)):
            pair['same_step'].append(paired(evidence[key][step],evidence[control][step]))
        pair['mean_same_step_delta_25_30K']={m:mean([r[m] for r in pair['delta_trajectory'] if 25000<=r['step']<=30000]) for m in METRICS}
        report['pairwise'].append(pair)
    for rep in ('R1','R2'):
        d_u=baselines[rep]; d_b=('QB4_DIFFBASE',rep);p_u=('QB4_P0UBIL',rep);p_b=('QB4_P0BASE',rep)
        cells={name:metric_row(evidence[key][10000]) for name,key in [('DIFF_UBIL',d_u),('DIFF_BASE',d_b),('P0_UBIL',p_u),('P0_BASE',p_b)]}
        report['factorial_10K'].append({'replica':rep,'cells':cells,
            'remove_DIFF_UBIL':paired(evidence[p_u][10000],evidence[d_u][10000]),
            'remove_DIFF_BASE':paired(evidence[p_b][10000],evidence[d_b][10000]),
            'BASE_minus_UBIL_when_P0':paired(evidence[p_b][10000],evidence[p_u][10000]),
            'BASE_minus_UBIL_when_DIFF':paired(evidence[d_b][10000],evidence[d_u][10000]),
            'difference_of_differences':{m:(cells['P0_BASE'][m]-cells['P0_UBIL'][m])-
                (cells['DIFF_BASE'][m]-cells['DIFF_UBIL'][m]) for m in METRICS}})
    new=[r for r in report['runs'] if r['package']=='panda_qb_structure']
    report['counts']={'all_scoped_runs':len(report['runs']),'all_official_evaluations':sum(len(r['rows']) for r in report['runs']),
        'all_joint_passes':sum(len(r['joint_passes']) for r in report['runs']),
        'new_runs':len(new),'new_status_counts':dict(Counter(r['status'] for r in new)),
        'new_official_evaluations':sum(len(r['rows']) for r in new),'new_joint_passes':sum(len(r['joint_passes']) for r in new)}
    report['best_H_qualified']=min(({'case':r['case'],'replica':r['replica'],**v} for r in report['runs']
        for v in r['rows'] if v['HQNR']>=.9315),key=lambda r:r['ERGAS'])
    report['new_best_H_qualified']=min(({'case':r['case'],'replica':r['replica'],**v} for r in new
        for v in r['rows'] if v['HQNR']>=.9315),key=lambda r:r['ERGAS'])
    report['calibration_proofs']=[read(p) for p in (ROOT/'work_dir/panda_qb_structure').glob('*/s1/QB/calibration/numeric_parity_*.json')]
    report['estimated_completed_C30_wall_hours']={
        'mean':mean([r['wall_hours'] for r in new if r['status']=='COMPLETED']),
        'range':[min(r['wall_hours'] for r in new if r['status']=='COMPLETED'),
                 max(r['wall_hours'] for r in new if r['status']=='COMPLETED')]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'local_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'captured':report['captured_utc'],'counts':report['counts'],
        'best':report['best_H_qualified'],'new_best':report['new_best_H_qualified'],
        'new_representatives':[{k:r[k] for k in ('case','replica','status','recorded_step','representative','mean_25_30K')} for r in new],
        'runtime':report['estimated_completed_C30_wall_hours']},indent=2))


if __name__=='__main__':main()
