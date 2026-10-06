"""Read-only static checks for the supplied plan. This is NOT a training runner."""
from pathlib import Path
import csv, json, math, collections, sys

ROOT = Path(__file__).resolve().parent

def main() -> int:
    spec = json.loads((ROOT / 'experiment_spec.json').read_text(encoding='utf-8'))
    with (ROOT / 'TeacherOnly_Run_Registry_468.csv').open(encoding='utf-8-sig', newline='') as f:
        runs = list(csv.DictReader(f))
    cases = spec['cases']; byid = {c['case_id']: c for c in cases}
    checks = {}
    checks['52_unique_conditions'] = len(cases) == len(byid) == 52
    checks['468_unique_training_ids'] = len(runs) == len({r['run_id'] for r in runs}) == 468
    mapping = {'s1':'WV3', 's3':'QB', 's5':'GF2'}
    checks['only_authorized_lanes'] = all(mapping.get(r['server']) == r['dataset'] for r in runs)
    groups = collections.defaultdict(list)
    for r in runs:
        groups[(r['dataset'], r['case_id'])].append(r)
    checks['three_repeats_each_dataset_case'] = len(groups) == 156 and all(sorted(int(r['replica']) for r in g) == [1,2,3] for g in groups.values())
    checks['common_seed_set_within_dataset'] = all(len({tuple(sorted(int(r['seed']) for r in g)) for (d,c),g in groups.items() if d==ds})==1 for ds in mapping.values())
    checks['teacher_only_fresh_no_student_kd_q_panmix'] = all(c['training_role']=='TEACHER_ONLY' and c['init']=='FRESH' and not any(c[k] for k in ('student_training','kd','q_weighting','panmix')) for c in cases)
    checks['control_ids_exist'] = all(not c['control_id'] or c['control_id'] in byid for c in cases)
    checks['updates_432x50k_36x100k'] = collections.Counter(int(r['updates']) for r in runs)=={50000:432,100000:36}
    checks['25_2M_updates'] = sum(int(r['updates']) for r in runs)==25200000
    checks['numeric_resolved_epsilon'] = all(math.isfinite(float(r['epsilon_lambda'])) and float(r['epsilon_lambda']) >= 0 for r in runs)
    checks['historical_qb_lambda_kept'] = all(float(r['epsilon_lambda']) == (.0003 if r['dataset']=='QB' else .0001) for r in runs if r['case_id'] in ('TA2-H00','TA2-H01'))
    checks['exactly_50_candidates_each_budget'] = all(len(list(range(1010*(n//50000),n,1010*(n//50000)))+[n])==50 for n in (50000,100000))
    checks['50k_not_added_to_100k_selection_grid'] = 50000 not in (list(range(2020,100000,2020))+[100000])
    checks['eight_factorial_cells'] = len({(byid[f'TA2-M{i:02d}']['a_reference'],byid[f'TA2-M{i:02d}']['epsilon_lambda']>0,byid[f'TA2-M{i:02d}']['struct_lambda']>0) for i in range(8)})==8
    grid = [c for c in cases if c['group'].startswith('04')] + [byid['TA2-M07']]
    checks['nine_grid_points_one_shared_center'] = len(grid)==9 and len({(c['epsilon_lambda'],c['struct_lambda']) for c in grid})==9
    cols = json.loads((ROOT / 'analysis_columns_v2.json').read_text())
    checks['152_unique_columns'] = len(cols)==len(set(cols))==152 and cols[70]=='Attempt'
    checks['no_wallclock_or_success_repetition'] = spec['wall_clock_limit'] is None and spec['repeat_until_success'] is False
    for server,ds in mapping.items():
        rr = [r for r in runs if r['server']==server]
        checks[f'{server}_156_sequential'] = len(rr)==156 and sorted(int(r['queue_seq']) for r in rr)==list(range(1,157))
    report = {'scope':'REGISTRY_STATIC_ONLY; NO CUDA/LOSS/REMOTE TRAINING VERIFICATION','passed':all(checks.values()),'checks':checks}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
