"""Install the new registry in one dedicated process; old source files are untouched."""
from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import io
from pathlib import Path
import sys
from . import registry as G

_installed = None


def own_files(root):
    folder = Path(root) / 'panda_qb_structure'
    return sorted([p for p in folder.rglob('*.py') if 'tests' not in p.parts] +
                  [folder / 'experiment_manifest.json', folder / 'cases_s1.csv', folder / 'cases_s3.csv'])


def install(server):
    global _installed
    if server not in G.RUNS:
        raise ValueError('Registered local s1 or s3 required')
    if _installed is not None:
        if _installed != server:
            raise RuntimeError('Use separate processes for different local Stage1 origins')
        from panda_sensor_focus import plan
        return plan
    unsafe = [n for n in sys.modules if n.startswith('panda_sensor_focus.') and n != 'panda_sensor_focus.contract']
    if unsafe:
        raise RuntimeError('Dedicated QB structure process required; focus already imported')
    # Activate the authentic source checkpoint implementation before it is
    # imported by the Student. This changes no source tensors or stored config.
    if server == 's3':
        from panda_qb_deepen.adapter import install as install_source
        install_source()
    from . import donor as D
    from panda_sensor_focus import contract as C
    C.LANES = {s: deepcopy(C.LANES['s1']) for s in G.RUNS}
    C.CASES, C.REGISTERED_RUNS = G.cases(), deepcopy(G.RUNS)
    C.CAMPAIGN_ID = C.NEXT_CAMPAIGN_ID = C.CONTROL_CAMPAIGN = G.CAMPAIGN_ID
    C.METHOD_REVISION = G.METHOD_REVISION
    C.SOURCE_DONORS = {s: D.metadata(s) for s in G.RUNS}
    C.ALL_DONORS = (G.DONOR_CASE,)
    C.LOSS_PAIRS = ((G.DONOR_CASE, 2., .03),)

    def validate_pair(cfg):
        if (cfg.get('server') != server or cfg.get('dataset') != 'QB' or cfg.get('phase') != 'stage2'
                or (cfg.get('case'), cfg.get('replica')) not in G.RUNS[server]
                or cfg.get('donor_version') != G.DONOR_CASE
                or cfg.get('stage1_loss_parameters') != {'lambda_abs': 2., 'lambda_epsilon': .03}
                or cfg.get('epsilon_radius_rr') != 3. or cfg.get('epsilon_distribution') != 'UNIFORM_AREA_DISK'
                or any(cfg.get(k) != 0. for k in ('lambda_abs', 'lambda_epsilon', 'lambda_eps_RR'))):
            raise ValueError('Registered local-donor QB Stage2 configuration required')
        return {'lambda_abs': 2., 'lambda_epsilon': .03}

    def validate_teacher(binding):
        source = D.metadata(server); teacher = binding.get('band30_teacher', {})
        required = {'campaign_id': source['campaign_id'], 'upsampler': 'interp23', 'aligner_outputs': 2,
                    'pan_channels': 1, 'alignment_target': 'SCHARR_OF_RGB_MEAN_GT',
                    'lambda_abs': 2., 'lambda_epsilon': .03}
        if (binding.get('sensor') != 'QB' or teacher != binding.get('mature30_teacher')
                or binding.get('source_campaign_id') != source['campaign_id']
                or binding.get('donor_step') != source['selected_step']
                or binding.get('donor_checkpoint_sha256') != source['checkpoint_sha256']
                or any(teacher.get(k) != v for k, v in required.items())):
            raise ValueError('Frozen A/T must preserve their actual local Stage1 identity')
        return teacher

    def validate_donor(cfg, binding):
        expected = validate_pair(cfg); validate_teacher(binding)
        return expected

    C.validate_loss_pair_config, C.validate_source_teacher = validate_pair, validate_teacher
    C.validate_donor_loss_pair = validate_donor
    from panda_sensor_focus import spec as S
    from .spec import MatureSpec
    S.MatureSpec = MatureSpec
    from panda_sensor_focus import plan as P
    from donor30.plan import stream_seeds, canonical_sha256, LOSS_MASK_POLICY
    from panda_sensor_focus.engine import ScheduleTrainingSpec
    P.SEEDS = deepcopy(G.SEEDS)
    original_default = P.default_config

    def baseline(lane):
        if lane != server: raise ValueError('Use the local lane in this dedicated process')
        value = P.previous.baseline('s1', 3.)
        value['server'] = lane
        value['protocol_decisions']['sensor_focus'] = 'Local completed Stage1; paired input/upsampling and AdamW decay; existing Hard loss'
        return value

    P.baseline = baseline

    def default_config(lane):
        if lane != server: raise ValueError('Use the local lane in this dedicated process')
        value = original_default(lane)
        value.update(seeds=deepcopy(G.SEEDS), joint_target=deepcopy(G.TARGET),
            donor_policy='EACH_SERVER_OWN_COMPLETED_VALIDATION_SELECTED_STAGE1',
            fixed_donor=D.metadata(lane), stage1_runs=0, stage2_runs=len(G.RUNS[lane]),
            allow_cross_server_donor=False, donor_origin_server=lane,
            after_campaign='RESUME_UNCHANGED_WV3' if lane == 's1' else 'STOP_COMPLETED_QB_STRUCTURE',
            comparisons=deepcopy(G.CHANGES), time_cap_seconds=None)
        return value

    def resolve_case(lane, case, replica='R1', *, attempt=1):
        if (lane != server or (case, replica) not in G.RUNS[lane]
                or type(attempt) is not int or attempt < 1):
            raise ValueError('Unregistered QB structure run')
        # The original s1 QB template supplies the unchanged data protocol,
        # optimizer/batch/augmentation, never a source Student state.
        cfg = P.previous.resolve_case('s1', 'QB_BASE', 'R2', attempt=attempt)
        p = deepcopy(C.CASES[case]); seed = G.SEEDS[replica]; seeds = stream_seeds(seed)
        seeds['rr_band_augmentation'] = 1 + int.from_bytes(hashlib.sha256(
            f'donor30:{seed}:rr_band_augmentation'.encode()).digest()[:8], 'big') % (2**63 - 1)
        spec = MatureSpec('QB', 'panda_condq', seeds['S_init'], p['alpha'], 0., p['gamma'], 1e-6,
            mask_policy=LOSS_MASK_POLICY, policy_id='S100', total_updates=30000,
            sampling_variant=p['sampling_variant'], input_variant=p['input_variant'], student_width=112)
        training = ScheduleTrainingSpec(lr=p['lr_S'], weight_decay=p['weight_decay'], total_updates=30000,
            warmup_updates=100, min_lr=0., full_audit_interval=100, inherited_updates=0,
            schedule_kind='WARMUP_COSINE')
        b = baseline(lane)
        control = G.controls(lane, replica)
        cfg.update(campaign_id=G.CAMPAIGN_ID, method_revision=G.METHOD_REVISION,
            server=lane, dataset='QB', case=case, loss_case=case, replica=replica, seed=seed,
            run_id=f'PSFC_QB_{lane.upper()}_{case}_NOSOFT_{replica}_S{seed}',
            run_display=f'QB_STRUCTURE {case} {replica}', stream_seeds=seeds,
            baseline_resolved=b, baseline_alignment_protocol_digest=P.previous.alignment_protocol_digest(b),
            config_digest=canonical_sha256(default_config(lane)), stage2_spec=asdict(spec),
            geometry=S.policy_manifest(spec), training_spec=asdict(training),
            sampling_variant=p['sampling_variant'], input_variant=p['input_variant'],
            student_sampling=C.sampling_manifest(p['sampling_variant'], p['input_variant'], 112),
            student_input=C.input_manifest(p['input_variant'], 112), student_warp=deepcopy(C.WARP_MANIFEST),
            donor_version=G.DONOR_CASE, loss_pair=G.DONOR_CASE, stage1_retrained=False,
            fixed_donor_source=D.metadata(lane), donor_origin_server=lane, execution_server=lane,
            lr_S=p['lr_S'], lr_U=p['lr_S'], weight_decay=p['weight_decay'], tuning_parameters=p,
            joint_target=deepcopy(G.TARGET), question=G.CHANGES[case]['purpose'],
            comparison_control={**control, 'same_exact_donor_checkpoint': True,
                'fresh_student_no_weights_reused': True, 'student_seed_differs_from_control': False,
                'changes': deepcopy(G.CHANGES[case])},
            cross_server_interpretation='INDEPENDENT_STAGE1_AND_HARDWARE; WITHIN_SERVER_PAIRED_ABLATION')
        validate_pair(cfg)
        return cfg

    def catalog_csv(lane):
        if lane != server: raise ValueError('Dedicated local lane required')
        out = io.StringIO(); writer = csv.writer(out, lineterminator='\n')
        writer.writerow(['order', 'server', 'dataset', 'case', 'replica', 'seed', 'updates', 'input',
            'sampling', 'width', 'lr', 'weight_decay', 'alpha', 'beta', 'gamma', 'scale',
            'Stage1_source', 'HQNR_min', 'ERGAS_max'])
        for index, (case, rep) in enumerate(G.RUNS[lane], 1):
            p = C.CASES[case]
            writer.writerow([index, lane, 'QB', case, rep, G.SEEDS[rep], p['total_updates'],
                p['input_variant'], p['sampling_variant'], 112, p['lr_S'], p['weight_decay'],
                p['alpha'], 0, p['gamma'], 1, G.SOURCES[lane]['campaign_id'], .9315, 3.5])
        return out.getvalue()

    P.default_config, P.resolve_case, P.catalog_csv = default_config, resolve_case, catalog_csv
    from panda_sensor_focus import runtime, orchestration, donor, reporting, publication, engine
    donor.receipt_path, donor.validate_source, donor.load_donor = D.receipt_path, D.validate_source, D.load_donor
    reporting.DATASETS = {s: 'QB' for s in G.RUNS}
    publication.LANES = {s: ('QB', 'QB-ablations', 261001011, .9315, 3.5) for s in G.RUNS}
    reporting.RECORD_PREFIX = 'PQST-'
    reporting.RECORD_NAMESPACE = 'PAN_FIXED64_V1:2026-10-31:PANDA_QB_STRUCTURE_LOCALSTAGE1'
    old_report, old_description = reporting._config, reporting.approach_description

    def report_config(cfg):
        old_report(cfg)
        expected = G.cases()[cfg['case']]['weight_decay']
        if cfg.get('weight_decay') != expected or cfg['training_spec'].get('weight_decay') != expected:
            raise reporting.ReportingError('Registered AdamW weight decay differs')
        if cfg.get('fixed_donor_source') != D.metadata(server) or cfg.get('donor_origin_server') != server:
            raise reporting.ReportingError('Local donor metadata differs')
        return cfg

    def description(cfg):
        # Rendering an actual training row requires a selected donor. Cross-server
        # catalogs deliberately do not invent a checkpoint step or hash.
        value = old_description(cfg)
        return value + f" | AdamW wd={cfg['weight_decay']:g} | own {server} Stage1 seed={G.SOURCES[server]['seed']}"

    reporting._config, reporting.approach_description = report_config, description
    original_build = engine.build_fresh_stage2

    def build(cfg, *args, **kwargs):
        report_config(cfg)
        return original_build(cfg, *args, **kwargs)

    engine.build_fresh_stage2 = build
    original_calibration = runtime.Campaign.calibration

    def calibration(campaign, cfg):
        artifacts, stats = original_calibration(campaign, cfg)
        from .calibration_audit import audit
        proof = audit(P.ROOT, campaign.root, server, artifacts)
        cfg['calibration_historical_comparison'] = proof
        if not proof['historical_control_comparison_valid']:
            print('[QB_STRUCTURE CALIBRATION] Historical control comparison NOT VERIFIED: ' + proof['path'], flush=True)
        return artifacts, stats

    runtime.Campaign.calibration = calibration
    original_pin = runtime.code_pin

    def code_pin():
        from ta2.evaluation import file_sha
        files = own_files(P.ROOT)
        if server == 's3':
            from panda_qb_deepen.adapter import own_files as source_files
            files += source_files(P.ROOT)
        return {**original_pin(), **{str(p.relative_to(P.ROOT)): file_sha(p) for p in files}}

    def predecessor(root, lane):
        from .prerequisites import independent_start
        if lane != server: raise ValueError('Wrong local lane')
        return independent_start(lane)

    runtime.code_pin, orchestration.predecessor_receipt = code_pin, predecessor
    _installed = server
    return P
