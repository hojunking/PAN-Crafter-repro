"""Isolated registration of independent local Stage1 and eight Student cases."""
from copy import deepcopy
from pathlib import Path
import csv
import io
import sys
from . import registry as G

_installed = False


def own_files(root):
    root = Path(root); folder = root / 'panda_qb_deepen'
    return sorted([p for p in folder.rglob('*.py') if 'tests' not in p.parts] +
        [folder / 'experiment_manifest.json', folder / 'cases_s3.csv',
         root / 'panda_sensor_focus/early_stop.py', root / 'panda_sensor_refine/orchestration.py'])


def install():
    global _installed
    if _installed:
        from panda_s1loss9 import plan
        return plan
    unsafe = [name for name in sys.modules if name.startswith('panda_s1loss9.')
              and name != 'panda_s1loss9.contract']
    if unsafe:
        raise RuntimeError('Dedicated own-Stage1 process required; S1LOSS9 modules already imported')
    from panda_s1loss9 import contract as C
    C.LANES = {'s3': deepcopy(C.LANES['s1'])}
    C.CASES, C.DONORS = G.cases(), (G.DONOR_CASE,)
    C.CAMPAIGN_ID = C.NEXT_CAMPAIGN_ID = C.CONTROL_CAMPAIGN = G.CAMPAIGN_ID
    C.METHOD_REVISION = G.METHOD_REVISION
    original_pair_validation = C.validate_loss_pair_config

    def validate_pair(cfg):
        if cfg.get('server') != 's3' or cfg.get('dataset') != 'QB':
            raise ValueError('This campaign trains local s3/QB only')
        checked = deepcopy(cfg)
        if cfg.get('phase') == 'stage2':
            if (cfg.get('case'), cfg.get('replica')) not in G.RUNS:
                raise ValueError('Unregistered QB Student')
            # Only a validation copy uses the inherited paired-case name.
            # Actual source/config identities retain the registered Student case.
            checked['case'] = 'QB_A200_E030'
        elif cfg.get('case') != G.DONOR_CASE or cfg.get('seed') != G.STAGE1_SEED:
            raise ValueError('Independent s3 Stage1 seed/case differs')
        return original_pair_validation(checked)

    C.validate_loss_pair_config = validate_pair
    from panda_s1loss9 import plan as P
    P.SOURCE_SEED = G.STAGE1_SEED
    P.SEEDS = {'R1': 261018101, 'R2': 261018102}
    P.groups = lambda server: [dict(name=f'{i:02d}_{case.lower()}_{rep.lower()}', case=case,
        optional=False, runs=[(case, rep)], admission='ALL_REGISTERED_RUNS_NO_TIME_CAP')
        for i, (case, rep) in enumerate(((G.DONOR_CASE, 'R1'),) + G.RUNS)]
    old_default, old_resolve, old_csv = P.default_config, P.resolve_case, P.catalog_csv

    def default_config(server):
        if server != 's3': raise ValueError('Only s3/QB is registered')
        cfg = old_default(server)
        cfg.update(stage1_runs=1, stage2_runs=8, stage1_seed=G.STAGE1_SEED,
            plan_revision='QB_DEEPEN_OWN_STAGE1_20261030',
            donor_origin_server='s3', execution_server='s3', allow_cross_server_donor=False,
            stage1_loss_grid={'A200_E030': {'lambda_abs': 2., 'lambda_epsilon': .03}},
            after_campaign='STOP_NO_AUTOMATIC_WV3_RESUME', before_campaign='NONE_INDEPENDENT_LOCAL_STAGE1',
            priority_policy='OWN_STAGE1_THEN_PAIRED_STUDENT_ABLATIONS',
            baseline_control='OWN_LOCAL_STAGE1_WITHIN_S3_STUDENT_CONTROL',
            comparisons=deepcopy(G.CHANGES), historical_controls=deepcopy(G.CONTROLS))
        return cfg

    def resolve_case(server, case, replica='R1', *, attempt=1):
        if server != 's3': raise ValueError('Only s3/QB is registered')
        cfg = old_resolve(server, case, replica, attempt=attempt)
        cfg.update(run_display=f'QB_DEEPEN {case} {replica}', donor_origin_server='s3',
                   execution_server='s3', joint_target=deepcopy(G.TARGET))
        if cfg['phase'] == 'stage2':
            wd = G.cases()[case]['weight_decay']
            cfg['training_spec']['weight_decay'] = wd
            cfg.update(weight_decay=wd,
                question='Independently trained local Stage1; ' + G.CHANGES[case]['purpose'],
                comparison_control={'campaign_id': G.CAMPAIGN_ID, 'server': 's3', 'case': 'QB3_CONTROL',
                    'replica': replica, 'seed': P.SEEDS[replica], 'same_local_selected_donor': True,
                    'fresh_student_no_weights_reused': True, 'changes': deepcopy(G.CHANGES[case])},
                historical_comparison={**deepcopy(G.CONTROLS[replica]),
                    'same_exact_donor_checkpoint': False,
                    'interpretation': 'INDEPENDENT_STAGE1_AND_SERVER_CONFOUNDED_NOT_PURE_STUDENT_ABLATION'})
        else:
            cfg['question'] = 'Independent s3 QB Stage1; full50K then validation RRE50/RRE90/ERGAS selection'
        return cfg

    P.default_config, P.resolve_case = default_config, resolve_case

    def donor_receipt_path(server, case):
        if server != 's3' or case != G.DONOR_CASE: raise ValueError('Local donor required')
        return f'{G.OUTPUT}/{G.CAMPAIGN_ID}/s3/QB/donors/{G.DONOR_CASE}/rre_selection.json'

    P.donor_receipt_path = donor_receipt_path

    def catalog_csv(server):
        rows = list(csv.reader(io.StringIO(old_csv(server)))); rows[0].append('weight_decay')
        for row in rows[1:]: row.append(str(.01 if row[3] == 'donor' else G.cases()[row[4]]['weight_decay']))
        result = io.StringIO(); csv.writer(result, lineterminator='\n').writerows(rows)
        return result.getvalue()

    P.catalog_csv = catalog_csv
    from panda_s1loss9 import runtime, orchestration, reporting, engine, publication
    original_pin, original_build = runtime.code_pin, engine.build_fresh_stage2
    reporting.DATASETS = {'s3': 'QB'}
    publication.LANES = {'s3': ('QB', 'QB-ablations', 261001011, .9315, 3.5)}
    reporting.RECORD_PREFIX = 'PQD3-'
    reporting.RECORD_NAMESPACE = 'PAN_FIXED64_V1:2026-10-30:PANDA_QB_DEEPEN_OWN_STAGE1_S3'
    old_report, old_description = reporting._config, reporting.approach_description

    def report_config(cfg):
        old_report(cfg)
        if cfg['phase'] == 'stage2':
            wd = G.cases()[cfg['case']]['weight_decay']
            if cfg['weight_decay'] != wd or cfg['training_spec']['weight_decay'] != wd:
                raise reporting.ReportingError('Registered AdamW decay differs')
        return cfg

    reporting._config = report_config
    reporting.approach_description = lambda cfg: (old_description(cfg) +
        f" | AdamW wd={cfg['weight_decay']:g} | independently trained s3 Stage1 seed={G.STAGE1_SEED}")

    def build(cfg, *args, **kwargs):
        report_config(cfg)
        return original_build(cfg, *args, **kwargs)

    def code_pin():
        from ta2.evaluation import file_sha
        return {**original_pin(), **{str(p.relative_to(P.ROOT)): file_sha(p) for p in own_files(P.ROOT)}}

    def predecessor(root, server):
        from .prerequisites import independent_start
        if server != 's3': raise ValueError('This queue is registered on s3')
        return independent_start()

    old_terminal = publication._terminal_state

    def terminal_state(root):
        from .prerequisites import read
        path = Path(root) / 'campaign_state.json'
        if path.exists():
            state = read(path)
            if state.get('status') == 'FINISHED_WITH_EXPLICIT_EARLY_STOPS': return state
        return old_terminal(root)

    from .runner import run
    runtime.Campaign.run = run
    runtime.code_pin, orchestration.predecessor_receipt = code_pin, predecessor
    engine.build_fresh_stage2 = build
    publication._terminal_state = terminal_state
    _installed = True
    return P
