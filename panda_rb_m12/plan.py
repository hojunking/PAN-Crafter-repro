"""Strict executable contract for the supplied M12 registry, not the B01 registry."""
from __future__ import annotations
from collections import Counter
import copy
import csv
import json
import math
from pathlib import Path

from fh12.common import object_sha, sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = Path('research_log/PANDA_REBUTTAL_M12_S135_2026-10-01')
CAMPAIGN = CAMPAIGN_ID = 'PANDA_REBUTTAL_B02_M12_WV3_S135_20261001_v1'
SERVERS = ('s1', 's3', 's5')
STEP1 = ('QFULL', 'QMEAN', 'QSHUF', 'QESUR', 'QEDGE', 'QALIGN')
STEP2 = ('H0', 'HSPMEAN', 'NOADV', 'ADVMEAN')
CASES = STEP1 + STEP2
MODES = ('A_ON', 'A_ZERO_INFERENCE_ONLY', 'A_NATIVE_FIXED', 'KNOWN_SHIFT_INVERSE')
SELECTORS = ('EXACT_50000', 'RR_VAL_ERGAS_MIN')
TEACHER_RUN = 'FH12_S1_T_P0_W112_D123_WV3_S71001_FRESH50_v1'
SOURCE_RUN = 'FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1'
VAL_GRID = tuple(range(1010, 50000, 1010)) + (50000,)
DIAGNOSTIC_STEPS = (0, 1000, 10000, 25000, 50000)
SEEDS = {s: [261001000 + int(s[1:])*100 + r for r in range(1, 5)] for s in SERVERS}


def campaign_dir(root=ROOT):
    return Path(root) / 'work_dir/_panda_rb/20261001/B02_M12'


def binding_path(root=ROOT):
    return campaign_dir(root) / 'common/bindings.json'


def weights_dir(root=ROOT):
    return campaign_dir(root) / 'common/weights'


def registry(root=ROOT):
    return json.loads((Path(root) / PLAN_DIR / 'planning/experiment_registry.json').read_text())


def training_runs(server=None, root=ROOT):
    if server is not None and server not in SERVERS:
        raise ValueError('Only s1/s3/s5 are registered')
    return [r for r in registry(root)['training_runs'] if server is None or r['server'] == server]


def case_for(run_id, root=ROOT):
    rows = [r for r in training_runs(root=root) if r['run_id'] == run_id]
    if len(rows) != 1:
        raise ValueError('Unknown or duplicate M12 run: ' + run_id)
    return rows[0]


def run_dir(run_id, root=ROOT):
    row = case_for(run_id, root)
    return campaign_dir(root) / 'train' / row['server'] / f"R{row['repeat']}" / row['case_id']


def schedule(server, root=ROOT):
    if server not in SERVERS:
        raise ValueError('Unregistered server')
    with (Path(root) / PLAN_DIR / 'planning/server_schedule.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['server'] == server]
    for row in rows:
        row['position'] = int(row['position'])
        if row['repeat']:
            row['repeat'] = int(row['repeat'])
    return rows


def shift_grid(root=ROOT):
    return json.loads((Path(root) / PLAN_DIR / 'planning/shift_grid.json').read_text())


def shifts(root=ROOT):
    return shift_grid(root)['shifts']


def protocol_identity(root=ROOT):
    base = Path(root) / PLAN_DIR
    paths = [base / 'EXPERIMENT_CASES_KR.md'] + sorted((base / 'planning').glob('*'))
    files = {str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()}
    return dict(campaign_id=CAMPAIGN, files=files, content_sha256=object_sha(files))


def validate_plan(root=ROOT):
    reg = registry(root)
    def require(ok, message):
        if not ok:
            raise ValueError('M12 plan: ' + message)
    require(reg['campaign_id'] == CAMPAIGN and reg['servers'] == list(SERVERS), 'campaign/server drift')
    require(reg['seed_table'] == SEEDS, 'seed drift')
    fixed_training = dict(updates=50000, batch_size=48, warmup=100, lr_U=1e-4, lr_A=3e-6,
                          optimizer='AdamW', betas=[.9, .999], eps=1e-8, weight_decay=.01, precision='FP32')
    require(all(reg['training'].get(k) == v for k, v in fixed_training.items()), 'training recipe drift')
    require(reg['training']['validation_grid'] == list(VAL_GRID), 'validation grid drift')
    require(reg['training']['diagnostic_steps'] == list(DIAGNOSTIC_STEPS), 'diagnostic grid drift')
    rows = reg['training_runs']
    require(len(rows) == len({r['run_id'] for r in rows}) == 120, 'exactly 120 unique runs')
    require({r['case_id'] for r in reg['templates']} == set(CASES), 'case set drift')
    for template in reg['templates']:
        case = template['case_id']
        base = case if case in STEP1[:4] else 'QFULL'
        base = 'TRAIN_MEAN' if base == 'QMEAN' else base
        expected = dict(edge_weight='TRAIN_MEAN' if case == 'QALIGN' else base,
                        aligner_weight='TRAIN_MEAN' if case == 'QEDGE' else base,
                        hard_mode={'H0': 'PLAIN', 'HSPMEAN': 'SPATIAL_MEAN'}.get(case, 'ADAPTIVE'),
                        advantage_mode={'NOADV': 'ONE', 'ADVMEAN': 'TRUST_WEIGHTED_SPATIAL_MEAN'}.get(case, 'ADAPTIVE'),
                        alpha=0. if case == 'H0' else 1., beta=.1, lambda_edge=.002)
        require(all(template.get(k) == v for k, v in expected.items()), 'intervention template drift')
    require(len(reg['native_tasks']) == 240 and len(reg['stress_tasks']) == 288, 'evaluation counts')
    balances = {1: Counter(), 2: Counter()}
    for server in SERVERS:
        order = schedule(server, root)
        require([r['position'] for r in order] == list(range(1, len(order)+1)), 'schedule ordering')
        require(order[-1]['action'] == 'FINAL_REPORT_AND_STOP_FOR_REVIEW', 'finite stop missing')
        train_order = [r for r in order if r['action'] == 'TRAIN_FRESH_50K_AND_NATIVE']
        require(len(train_order) == 40 and len({r['run_id'] for r in train_order}) == 40, '40 admissions per server')
        require([r['phase'] for r in train_order] == ['STEP1_QROUTE']*24 + ['STEP2_FITTING']*16, 'local phase order')
        expected_actions = Counter((r['source_run_id'], r['mode']) for r in reg['stress_tasks'] if r['server'] == server)
        require(Counter((r['run_id'], r['mode']) for r in order if r['action'] == 'STRESS_INFERENCE') == expected_actions,
                'scheduled stress task mismatch')
        require(all(r['action'] in ('VERIFY_ASSETS_AND_RECOVER_OLD_EVIDENCE', 'DIAGNOSE_F1_INITIAL_AND_EXISTING_FINAL_ALIGNER',
                    'TRAIN_FRESH_50K_AND_NATIVE', 'STRESS_INFERENCE', 'QUEUE_VERIFIED_UPLOAD',
                    'SAVE_PHASE_REPORT_NO_SCORE_GATE', 'FINAL_REPORT_AND_STOP_FOR_REVIEW') for r in order), 'unknown scheduled action')
        for repeat, seed in enumerate(SEEDS[server], 1):
            block = [r for r in rows if r['server'] == server and r['repeat'] == repeat]
            require(len(block) == 10 and {r['case_id'] for r in block} == set(CASES), 'paired block')
            for row in block:
                phase = 'STEP1_QROUTE' if row['case_id'] in STEP1 else 'STEP2_FITTING'
                expected_id = f"RBM12_WV3_{server.upper()}_R{repeat}_SS{seed}_{row['case_id']}_F50K_v1"
                require(row['run_id'] == expected_id and row['seed'] == seed and row['updates'] == 50000
                        and row['dataset'] == 'WV3' and row['phase'] == phase and not row['old_B01_reuse']
                        and row['teacher_run_id'] == TEACHER_RUN and row['teacher_updates'] == 50000
                        and row['primary_selection'] == SELECTORS[0] and row['secondary_selection'] == SELECTORS[1]
                        and row['initial_U'] == 'FRESH_SHARED_WITHIN_BLOCK'
                        and row['initial_A'] == 'FROZEN_F1_CLONE_THEN_TRAINABLE', 'run identity/recipe')
                require(row['work_dir'] == str(run_dir(expected_id, root).relative_to(root)), 'output namespace')
                native = [t for t in reg['native_tasks'] if t['source_run_id'] == expected_id]
                require(len(native) == 2 and {t['selection'] for t in native} == set(SELECTORS), 'native selectors')
                stress = [t for t in reg['stress_tasks'] if t['source_run_id'] == expected_id]
                require(len(stress) == (4 if row['case_id'] in STEP1 else 0), 'STEP1-only stress')
                if stress:
                    require({t['mode'] for t in stress} == set(MODES) and all(
                        t['shifts'] == 49 and t['rr_scenes'] == 20 and t['optimizer_updates'] == 0
                        and t['selection'] == SELECTORS[0] and t['primary_roi'] == '48:-48_FIXED160'
                        and t['auxiliary_roi'] == '32:-32_FIXED192' for t in stress), 'stress protocol')
            for index, phase in enumerate(('STEP1_QROUTE', 'STEP2_FITTING'), 1):
                selected = [r for r in train_order if r['repeat'] == repeat and r['phase'] == phase]
                for pos, row in enumerate(selected):
                    balances[index][row['case_id'], pos] += 1
    require(len(balances[1]) == 36 and set(balances[1].values()) == {2}, 'STEP1 counterbalance')
    require(len(balances[2]) == 16 and set(balances[2].values()) == {3}, 'STEP2 counterbalance')
    grid = shifts(root)
    expected = [(0., 0.)] + [(r*math.sin(math.radians(a)), r*math.cos(math.radians(a)))
                            for r in (.25, .5, 1., 2., 3., 4.) for a in range(0, 360, 45)]
    require(len(grid) == 49 and all(p['id'] == f'D{i:03}' and all(
        math.isclose(p[k], xy[j], abs_tol=1e-12) for j, k in enumerate(('dy', 'dx')))
        for i, (p, xy) in enumerate(zip(grid, expected))), 'shift grid')
    return dict(status='PASS', training_runs=120, native_observations=240, stress_curves=288,
                independent_students_per_case=12, independent_teachers=1,
                servers=list(SERVERS), protocol_identity=protocol_identity(root))


def build_config(run_id, root=ROOT):
    import yaml
    row = case_for(run_id, root)
    cfg = yaml.safe_load((Path(root) / 'config' / (SOURCE_RUN + '.yaml')).read_text())
    old = cfg.pop('fh20r1')
    fixed = dict(num_iter=50000, num_warmup=100, batch_size=48, num_bands=8, max_pixel=2047.,
                 mixed_precision='no', learning_rate=1e-4, optimizer='AdamW', weight_decay=.01,
                 betas=[.9, .999], eps=1e-8, lr_scheduler='cosine')
    model = dict(hidden_size=104, depth=[1, 2, 2], out_channels=8, attn_locations=[],
                 mode_modulation=False, norm='ln', dropout=0.)
    if (any(cfg.get(k) != v for k, v in fixed.items()) or cfg['model_args'] != model
            or old['teacher_alias'] != 'F1' or old['teacher_run_id'] != TEACHER_RUN
            or old['input_layout'] != 'PLH' or old['profile'] != 'BASE'):
        raise ValueError('Original FH20R1 anchor recipe changed')
    template = next(t for t in registry(root)['templates'] if t['case_id'] == row['case_id'])
    cfg.update(seed=row['seed'], trainer='panda_rb_m12', work_dir=str(run_dir(run_id, root)))
    cfg['panda_rb_m12'] = {**copy.deepcopy(row), **template, 'campaign_id': CAMPAIGN,
        'role': 'S', 'profile': 'BASE', 'input_layout': 'PLH', 'teacher_alias': 'F1',
        'teacher_ref_step': 50000, 'aligner_lr': 3e-6, 'lambda_E': .002, 'view_margin_hr': 4,
        'init_policy': old['init_policy'], 'candidate_grid': list(VAL_GRID),
        'diagnostic_steps': list(DIAGNOSTIC_STEPS), 'bindings_path': str(binding_path(root)),
        'weights_dir': str(weights_dir(root)), 'protocol_identity': protocol_identity(root),
        'source_config': 'config/' + SOURCE_RUN + '.yaml', 'source_numeric_method': old['method_revision']}
    return cfg
