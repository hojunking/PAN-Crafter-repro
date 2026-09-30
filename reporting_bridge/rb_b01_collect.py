"""Read-only, GPU-free B01 evidence collection outside the numerical source set.

No experiment module is imported. The original frozen artifact validator is
extracted alone after its file hash is verified. Models, trainers and metric
implementations are never instantiated, and measured metrics are never rerun.
"""
from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

CAMPAIGN = 'PANDA_REBUTTAL_B01_WV3_S135_20260928_v1'
PACKAGE_SCHEMA = 'RB_B01_EVIDENCE_PACKAGE_v1'
SERVERS = ('s1', 's3', 's5')
CASES = ('QFULL', 'QMEAN', 'QSHUF', 'QESUR')
MODES = ('A_ON', 'A_ZERO_INFERENCE_ONLY')
SELECTORS = ('EXACT_50000', 'RR_VAL_ERGAS_MIN')
PLAN_DIR = Path('research_log/PANDA_REBUTTAL_STAGED_S135_2026-09-28')
CAMPAIGN_DIR = Path('work_dir/_panda_rb/20260928/B01')
VAL_GRID = list(range(1010, 50000, 1010)) + [50000]
METRICS = ('ergas', 'psnr', 'sam', 'edge_error_dn')
POINT_METRICS = (*METRICS, 'relative_response_l1_sum', 'coverage_all_pan_paths')


class EvidenceError(ValueError):
    pass


def object_sha(value):
    """Original FH12/B01 payload convention, NOT byte SHA or release SHA."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(
        EvidenceError('Nonfinite JSON token: ' + x)))


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def safe_path(base, relative):
    base, relative = Path(base).resolve(), Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact path')
    result = base / relative
    require(result.resolve().is_relative_to(base), 'Artifact symlink escapes its evidence root')
    return result


def verify_files(files, base):
    require(isinstance(files, dict) and bool(files), 'Empty artifact file-hash map')
    for name, expected in files.items():
        require(file_sha(safe_path(base, name)) == expected, 'Artifact bytes changed: ' + name)


def validate_seal(report, directory):
    body = {k: v for k, v in report.items() if k != 'payload_sha256'}
    require(report.get('complete') is True and report.get('payload_sha256') == object_sha(body),
            'Incomplete or modified sealed evaluation report')
    verify_files(report.get('file_hashes'), directory)


def registry(root):
    value = read_json(Path(root) / PLAN_DIR / 'planning/experiment_registry.json')
    rows = value['training_runs']
    require(len(rows) == 24 and len({r['run_id'] for r in rows}) == 24, 'Registry must contain 24 unique Students')
    require(tuple(value['servers']) == SERVERS, 'Unregistered server cohort')
    for server in SERVERS:
        for repeat in (1, 2):
            seed = 9281000 + int(server[1:]) * 100 + repeat
            block = [r for r in rows if r['server'] == server and r['repeat'] == repeat]
            require(len(block) == 4 and {r['case_id'] for r in block} == set(CASES), 'Wrong paired case block')
            for row in block:
                wanted = f"RB01_WV3_{server.upper()}_R{repeat}_SS{seed}_{row['case_id']}_F50K_v1"
                require(row['run_id'] == wanted and row['seed'] == seed and row['dataset'] == 'WV3'
                        and row['updates'] == 50000 and row['primary_selection'] == SELECTORS[0]
                        and row['secondary_selection'] == SELECTORS[1], 'Registry seed/run/budget/selection changed')
    return rows


def validate_grid(grid):
    points = grid['shifts']
    require(len(points) == 49 and len({x['id'] for x in points}) == 49, 'Stress grid must have 49 unique points')
    require(grid['coordinate_order'] == ['dy', 'dx'], 'Stress coordinate convention changed')
    expected = [(0., None)] + [(r, a) for r in (.25, .5, 1., 2., 3., 4.) for a in range(0, 360, 45)]
    for i, (p, (radius, angle)) in enumerate(zip(points, expected)):
        require(p['id'] == f'D{i:03d}' and p['radius_hr'] == radius, 'Stress grid order or radius changed')
        actual_angle = p.get('angle_degrees', p.get('angle_deg'))
        require(actual_angle == angle, 'Stress angle changed')
        if 'angle_deg' in p and 'angle_degrees' in p:
            require(p['angle_deg'] == p['angle_degrees'], 'Conflicting angle aliases')
        dy = 0. if angle is None else radius * math.sin(math.radians(angle))
        dx = 0. if angle is None else radius * math.cos(math.radians(angle))
        require(math.isclose(p['dy'], dy, abs_tol=1e-12) and math.isclose(p['dx'], dx, abs_tol=1e-12),
                'Stress dy/dx changed')
    return points


def selected_checkpoints(manifest):
    require(manifest.get('complete') is True and manifest.get('actual_updates') == 50000
            and manifest.get('test_FR_sweep') is False, 'Not a completed validation-only fresh50K selection')
    records = manifest['validation_records']
    require([r['update'] for r in records] == VAL_GRID, 'Incomplete validation grid')
    require(all(isinstance(r['val_ergas'], (int, float)) and math.isfinite(r['val_ergas']) for r in records),
            'Nonfinite validation score')
    winner = min(records, key=lambda r: (r['val_ergas'], r['update']))
    primary, secondary = manifest['primary'], manifest['secondary']
    for item, selector, update, directory in ((primary, SELECTORS[0], 50000, 'exact50000'),
                                              (secondary, SELECTORS[1], winner['update'], 'val_selected')):
        require(item['selection_id'] == selector and item['update'] == update and item['directory'] == directory,
                'Selection differs from registered endpoint/validation rule')
        require(isinstance(item['checkpoint_sha256'], str) and len(item['checkpoint_sha256']) == 64,
                'Invalid checkpoint SHA')
    require((secondary.get('alias_of') == SELECTORS[0]) ==
            (primary['checkpoint_sha256'] == secondary['checkpoint_sha256']), 'Checkpoint alias mismatch')
    return primary, secondary


def frozen_validator(root, deployment):
    release = safe_path(Path(root) / CAMPAIGN_DIR / 'releases', deployment['release_sha256'])
    manifest = read_json(release / 'runtime_release.json')
    digest = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    require(digest == manifest['release_sha256'] == deployment['release_sha256'], 'Frozen release manifest changed')
    verify_files(manifest['files'], release)
    source = (release / 'panda_rb/evaluation.py').read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_artifacts']
    require(len(nodes) == 1 and not nodes[0].decorator_list, 'Frozen artifact validator not found')
    # This is only the already hash-checked read-only original validator; no
    # imports or other top-level source code (trainer/model) are executed.
    scope = {'Path': Path, 'object_sha': object_sha, 'sha256': file_sha}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(release / 'panda_rb/evaluation.py'), 'exec'), scope)
    receipt = dict(release_sha256=digest, frozen_manifest_file_sha256=file_sha(release / 'runtime_release.json'),
                   validator_file_sha256=file_sha(release / 'panda_rb/evaluation.py'),
                   validator_function_sha256=hashlib.sha256(ast.get_source_segment(source, nodes[0]).encode()).hexdigest(),
                   validator='frozen panda_rb.evaluation.validate_artifacts; isolated AST; no module import',
                   image_id=deployment.get('image_id'))
    return release, scope['validate_artifacts'], receipt


def _runtime_patch_receipt(root, case, release, patch):
    require(patch['patch_id'] == 'PANDA_B01_SCENE_INDEX_SERIALIZATION_v1',
            'Unknown runtime patch requires explicit bridge review')
    original = (release / 'panda_rb/evaluation.py').read_text()
    old = 'per_scene.append(dict(split=split, scene_index=i, source_h5_row=i, **metrics,'
    new = 'per_scene.append(dict(metrics, split=split, scene_index=i, source_h5_row=i,'
    require(original.count(old) == 1, 'Recovery original constructor changed')
    corrected = original.replace(old, new)
    require(hashlib.sha256(corrected.encode()).hexdigest() == patch['corrected_evaluation_sha256'],
            'Recovery corrected evaluator SHA mismatch')
    functions = [n for n in ast.parse(corrected).body if isinstance(n, ast.FunctionDef) and n.name == '_native_selection']
    require(len(functions) == 1 and hashlib.sha256(ast.get_source_segment(corrected, functions[0]).encode()).hexdigest()
            == patch['native_function_sha256'], 'Recovery function SHA mismatch')
    parent = Path(root) / CAMPAIGN_DIR / 'control' / case['server'] / 'recoveries'
    matches = []
    for path in sorted(parent.glob('*/*.json')):
        item = read_json(path)
        if item.get('patch') == patch and item.get('server') == case['server'] and item.get('release_sha256') == release.name:
            entrypoint = path.parent / 'entrypoint.py'
            require(file_sha(entrypoint) == patch['entrypoint_sha256'], 'Recovery entrypoint bytes changed')
            matches.append(dict(receipt_path=str(path.relative_to(root)), receipt_file_sha256=file_sha(path),
                                entrypoint_file_sha256=file_sha(entrypoint)))
    require(bool(matches), 'Missing original runtime recovery receipt/entrypoint')
    return matches


def _context_assets(root, case, wd, context, release):
    import yaml
    config_path = wd / 'meta/config.resolved.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    spec = cfg['panda_rb']
    require(all(spec.get(k) == v for k, v in case.items()), 'Resolved config case/seed/registry mismatch')
    require(spec['campaign_id'] == CAMPAIGN and cfg['seed'] == case['seed'], 'Wrong resolved campaign/seed')
    require(context['case'] == case and context['run_id'] == case['run_id'], 'Native context run mismatch')
    require(cfg['num_iter'] == 50000 and cfg['num_bands'] == 8 and spec['role'] == 'S'
            and spec['input_layout'] == 'PLH', 'Wrong Student/budget/input contract')
    ma = cfg['model_args']
    require(ma['hidden_size'] == 104 and ma['depth'] == [1, 2, 2] and ma['attn_locations'] == []
            and ma['mode_modulation'] is False and ma['norm'] == 'ln', 'B01 architecture mismatch')
    require(object_sha(cfg) == context['config_sha256'], 'Resolved config object SHA mismatch')
    for name, digest in spec['protocol_identity'].items():
        require(file_sha(safe_path(Path(root) / PLAN_DIR, name)) == digest, 'Preregistered protocol bytes changed')
    source = context['source_identity']
    require(source['campaign_id'] == CAMPAIGN and source['content_sha256'] == object_sha(source['files']),
            'Numerical source identity invalid')
    require(source['frozen_release_sha256'] == release.name, 'Wrong source release')
    verify_files(source['files'], release)
    evaluator = context['evaluator_identity']
    verify_files(evaluator['files'], release)
    patch = context.get('evaluation_runtime_patch')
    patch_receipts = []
    base_sha = object_sha(dict(files=evaluator['files'], external_wald_sha256=evaluator['external_wald_sha256']))
    if patch:
        require(evaluator.get('serialization_hotfix') == patch, 'Missing evaluator runtime patch identity')
        require(evaluator['base_evaluator_content_sha256'] == base_sha, 'Base evaluator identity changed')
        require(evaluator['content_sha256'] == object_sha(dict(base=base_sha,
                                                               serialization_hotfix=patch)),
                'Patched evaluator identity changed')
        require(patch['original_evaluation_sha256'] == file_sha(release / 'panda_rb/evaluation.py'),
                'Runtime patch original source changed')
        patch_receipts = _runtime_patch_receipt(root, case, release, patch)
    else:
        require(evaluator['content_sha256'] == base_sha, 'Evaluator content SHA mismatch')
    binding_path = Path(spec['bindings_path'])
    require(binding_path.resolve().is_relative_to((Path(root) / CAMPAIGN_DIR).resolve()), 'Binding outside campaign')
    binding = read_json(binding_path)
    require(object_sha(binding) == context['binding_sha256'], 'Binding object SHA mismatch')
    common = binding['common']
    require(object_sha(common) == binding['common_sha256'] == context['binding_common_sha256'], 'Common F1 SHA mismatch')
    for key in ('q_ref', 'tau_R', 'teacher_checkpoint_sha256'):
        require(binding[key] == common[key], 'Conflicting binding alias: ' + key)
    require(common['teacher_update'] == 50000 and common['teacher_run_id'] == spec['teacher_run_id']
            and binding['server_local_teacher_substitution'] is False, 'Wrong fixed F1 Teacher')
    # Read bytes only: even training_state.pt is hashed, never deserialized.
    for key, digest in common['artifacts_sha256'].items():
        require(file_sha(binding['paths'][key]) == digest, 'F1 artifact bytes changed: ' + key)
    data = binding['dataset_manifest']
    require(object_sha(data) == context['data_sha256'], 'Dataset manifest object SHA mismatch')
    data_content = {s: {k: v for k, v in info.items() if k in
                    ('sha256', 'lpan_sha256', 'sample_order_sha256', 'count', 'shapes')}
                    for s, info in data['splits'].items()}
    require(data_content == context['data_content_identity'], 'Dataset content identity mismatch')
    for split in ('rr', 'fr'):
        require(data_content[split]['count'] == 20, 'Native RR/FR must each have 20 scenes')
    weights_path = Path(spec['weights_dir']) / f"seed_{case['seed']}.json"
    require(weights_path.resolve().is_relative_to((Path(root) / CAMPAIGN_DIR).resolve()), 'Weights outside campaign')
    weights = read_json(weights_path)
    require(weights['repeat_seed'] == case['seed'] and weights['binding_common_sha256'] == binding['common_sha256'],
            'Wrong seed/F1 q map')
    require(file_sha(weights_path.with_suffix('.npz')) == weights['cache_sha256'], 'q map bytes changed')
    require(weights['cache_identity']['source_identity'] == source, 'q map numerical source mismatch')
    manifest_path = wd / 'checkpoints/selection_manifest.json'
    manifest = read_json(manifest_path)
    require(file_sha(manifest_path) == context['selection_manifest_sha256'], 'Selection manifest byte SHA mismatch')
    selections = selected_checkpoints(manifest)
    expected = dict(config_sha256=object_sha(cfg), source_identity=source, binding_sha256=object_sha(binding),
                    weights_sha256=object_sha(weights), data_sha256=object_sha(data))
    for key, value in expected.items():
        require(manifest[key] == value, 'Selection provenance mismatch: ' + key)
    for item in selections:
        folder = safe_path(wd / 'checkpoints', item['directory'])
        identity = read_json(folder / 'identity.json')
        for key, value in expected.items():
            require(identity[key] == value, 'Checkpoint provenance mismatch: ' + key)
        require(identity['run_id'] == case['run_id'] and identity['update'] == item['update'], 'Checkpoint run/update mismatch')
        require(file_sha(folder / 'model.safetensors') == identity['model_sha256'] == item['checkpoint_sha256'],
                'Actual selected checkpoint bytes differ')
    status = read_json(wd / 'meta/training_status.json')
    require(status['run_id'] == case['run_id'] and status['actual_updates'] == 50000
            and status['training_complete'] is True and status['status'] == 'TRAINING_COMPLETE', 'Training not complete')
    for key, value in expected.items():
        require(status[key] == value, 'Training status provenance mismatch: ' + key)
    provenance = dict(config_file_sha256=file_sha(config_path), config_sha256=object_sha(cfg),
        selection_manifest_sha256=file_sha(manifest_path), binding_common_sha256=binding['common_sha256'],
        binding_file_sha256=file_sha(binding_path), data_content_identity_sha256=object_sha(data_content),
        numerical_source_sha256=source['content_sha256'], numerical_source_release=release.name,
        evaluator_sha256=object_sha(evaluator), evaluator_content_sha256=evaluator['content_sha256'],
        teacher_sha256=binding['teacher_checkpoint_sha256'], weights_file_sha256=file_sha(weights_path),
        weights_object_sha256=object_sha(weights), weights_npz_sha256=weights['cache_sha256'],
        raw_map_identity={k: weights['arrays_sha256'][k] for k in ('q', 'e_bar')},
        q_weight_map_sha256=weights['arrays_sha256'][case['case_id']], evaluation_runtime_patch=patch,
        runtime_patch_receipts=patch_receipts,
        runtime_policy={k: source.get(k) for k in ('torch', 'cuda', 'cudnn', 'numpy', 'scipy', 'skimage', 'tf32_matmul', 'tf32_cudnn')},
        source_git_release=source['git_release'], rr_scene_count=20, fr_scene_count=20,
        native_coordinate_frame='native_MS', test_aware_note='Endpoint or validation-only selection; not a claim of no historical test-set development')
    return cfg, binding, status, manifest, provenance


def validate_native(report, wd, manifest):
    require(report.get('schema') == 'PANDA_RB01_NATIVE_v1', 'Wrong native schema')
    require(set(report['selections']) == set(SELECTORS) and report['n_independent_students'] == 1, 'Wrong native selections/Student count')
    require(report.get('n_unique_checkpoints') == 1 + int(manifest['secondary'].get('alias_of') != SELECTORS[0]),
            'Wrong native unique checkpoint count')
    for item in selected_checkpoints(manifest):
        selected = report['selections'][item['selection_id']]
        for key in ('selection_id', 'update', 'checkpoint_sha256'):
            require(selected[key] == item[key], 'Native selected checkpoint mismatch')
        require(selected.get('alias_of') == item.get('alias_of'), 'Native alias mismatch')
        require(selected['n_scenes'] == {'rr': 20, 'fr': 20} and selected['coordinate_frame'] == 'native_MS'
                and selected['eval_mode'] == 'A_ON' and selected['fr_masking'] is False, 'Wrong native coordinate frame/masking')
        require(selected['rr']['crop'] == '20:-21' and selected['rr']['n_scenes'] == 20
                and selected['fr']['support'] == 'full512' and selected['fr']['n_scenes'] == 20, 'Wrong native measurement protocol')
        for split, size in (('rr', 256), ('fr', 512)):
            raw = selected['raw'][split]
            require(len(raw) == 20 and [r['scene_index'] for r in raw] == list(range(20)), 'Wrong native scene order/count')
            for record in raw:
                require(record['h5_row_index'] == record['scene_index'] and record['shape'] == [8, size, size]
                        and record['dtype'] == 'float32', 'Invalid raw metadata')
                require(report['file_hashes'].get(record['path']) == record['file_sha256'], 'Unsealed raw prediction reference')
        csv_relative = ('native/per_scene.csv' if item['selection_id'] == SELECTORS[0] or selected.get('alias_of')
                        else 'native/rr_val_selected/per_scene.csv')
        require(csv_relative in report['file_hashes'], 'Unsealed native scene table')
        with (wd / csv_relative).open(newline='') as stream:
            scene_rows = list(csv.DictReader(stream))
        require(len(scene_rows) == 40 and {(r['split'], r['scene_index']) for r in scene_rows} ==
                {(s, str(i)) for s in ('rr', 'fr') for i in range(20)}, 'Native CSV lost/duplicated scenes')
        for row in scene_rows:
            require(row['source_h5_row'] == row['scene_index'], 'Native source H5 row mismatch')
            require(row['prediction_digest'] == selected['raw'][row['split']][int(row['scene_index'])]['prediction_digest'],
                    'Native CSV prediction identity mismatch')
        for section, split, keys in (('rr', 'rr', ('ergas', 'sam', 'scc', 'psnr', 'ssim', 'q8')),
                ('fr', 'fr', ('hqnr', 'd_lambda', 'd_s')), ('supplemental_rr', 'rr', ('rmse', 'cc')),
                ('supplemental_fr', 'fr', ('jqm',))):
            for key in keys:
                if key not in selected.get(section, {}):
                    continue
                if section.startswith('supplemental_') and selected[section][key] is None:
                    continue  # Explicit unmeasured optional metrics stay blank.
                values = [_numeric(r.get(key)) for r in scene_rows if r['split'] == split]
                require(None not in values, 'Native CSV contains a missing reported metric')
                _close(selected[section][key], statistics.mean(values), 'Native CSV mean mismatch: ' + key)
        if selected.get('alias_of'):
            primary = report['selections'][SELECTORS[0]]
            require({k: v for k, v in selected.items() if k not in ('selection_id', 'update', 'alias_of')} ==
                    {k: v for k, v in primary.items() if k not in ('selection_id', 'update', 'alias_of')},
                    'Same-checkpoint alias metric mismatch')


def _numeric(value):
    if value in ('', None):
        return None
    result = float(value)
    require(math.isfinite(result), 'Nonfinite CSV metric must be explicit null, not NaN')
    return result


def _close(actual, expected, message):
    require((actual is None and expected is None) or
            (actual is not None and expected is not None and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)), message)


def validate_stress(report, wd, native, mode, grid):
    require(report.get('schema') == 'PANDA_RB02_CURVE_v1', 'Wrong stress schema')
    identity = report['identity']
    require(identity['context'] == native['context'] and identity['mode'] == mode
            and identity['source_selection'] == 'EXACT_50000' and identity['update'] == 50000
            and identity['checkpoint_sha256'] == native['selections']['EXACT_50000']['checkpoint_sha256'],
            'Stress is not this exact native50K checkpoint/mode/context')
    require(identity['grid_sha256'] == object_sha(grid), 'Stress grid SHA mismatch')
    require(report['n_scenes'] == 20 and report['n_shifts'] == 49 and report['n_observations'] == 980
            and report['n_independent_students'] == 1 and len(report['curve']) == 49, 'Incomplete stress grid')
    prefix = f'stress/{mode}'
    require(f'{prefix}/per_scene.csv' in report['file_hashes'] and f'{prefix}/curve_summary.json' in report['file_hashes'],
            'Unsealed stress scene/summary table')
    with (wd / prefix / 'per_scene.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 980, 'Stress CSV does not contain all 980 observations')
    expected_pairs = {(p['id'], str(i)) for p in grid['shifts'] for i in range(20)}
    require({(r['shift_id'], r['scene_index']) for r in rows} == expected_pairs, 'Stress CSV lost/duplicated registered scenes')
    require(sum(r['status'] != 'ok' for r in rows) == report['n_numerical_failures'], 'Numerical failure count mismatch')
    require(sum(r.get('invalid_roi_sampling') == 'True' for r in rows) == report['n_invalid_geometry'], 'Invalid geometry count mismatch')
    require(read_json(wd / prefix / 'curve_summary.json')['points'] == report['curve'], 'Curve summary differs from completion')
    for point, shift in zip(report['curve'], grid['shifts']):
        require(all(point.get(k) == v for k, v in shift.items()), 'Curve point differs from registered shift')
        group = [r for r in rows if r['shift_id'] == shift['id']]
        require(point['n_scenes'] == 20 and point['n_failures'] == sum(r['status'] != 'ok' for r in group), 'Point scene/failure count mismatch')
        # These are arithmetic checks of the already measured table, NOT image
        # metric recomputation. Null observations never become zeros.
        for key in POINT_METRICS:
            values = [_numeric(r.get(key)) for r in group]
            mean = statistics.mean(values) if None not in values else None
            std = statistics.stdev(values) if None not in values else None
            _close(point.get(key), mean, 'Curve CSV mean mismatch: ' + key)
            _close(point.get(key + '_scene_std'), std, 'Curve CSV SD mismatch: ' + key)
        for key in METRICS:
            base = report['curve'][0][key]
            expected = None if point[key] is None or base is None else point[key] - base
            _close(point.get('delta_from_zero_' + key), expected, 'Curve delta-from-zero mismatch')


def _status(case):
    return dict(case, campaign_id=CAMPAIGN, expected_updates=50000, expected_native_selectors=2,
        expected_curves=2, expected_points_per_curve=49, actual_updates=None, training_status='FILE_NOT_FOUND',
        native_status='FILE_NOT_FOUND', native_selectors=0,
        curve_status={mode: dict(status='FILE_NOT_FOUND', shifts=0, numerical_failures=None,
                                invalid_geometry=None) for mode in MODES},
        verification_status='NOT_COLLECTED', upload_status='UPLOAD_PENDING', errors=[],
        last_checked_at_utc=datetime.now(timezone.utc).isoformat())


def collect_local(root, server, *, frozen_report_path=None):
    """Verify one server's local artifacts, returning data; never write originals.

    Missing remote servers remain explicit status slots. Optional original
    report is independently cross-checked; the original frozen artifact reader
    is always used in addition to the bridge's stronger context checks.
    """
    root = Path(root).resolve()
    require(server in SERVERS, 'Only original s1/s3/s5 local collection is admitted')
    cases = registry(root)
    grid = read_json(root / PLAN_DIR / 'planning/shift_grid_v1.json')
    validate_grid(grid)
    base = root / CAMPAIGN_DIR
    deployment = read_json(base / 'control' / server / 'deployment.json')
    require(deployment['server'] == server, 'Deployment belongs to another server')
    release, original_validate, verifier = frozen_validator(root, deployment)
    native_entries, curve_entries, statuses, errors = [], [], [], []
    original_report = read_json(frozen_report_path) if frozen_report_path else None
    original_report_audit = {}
    if original_report is not None:
        require(original_report['schema'] == 'PANDA_RB_B01_LOCAL_REPORT_v1'
                and original_report['servers'] == [server], 'Wrong original server report')
        original_report_audit['original_report_file_sha256'] = file_sha(frozen_report_path)
    for case in cases:
        status = _status(case)
        statuses.append(status)
        if case['server'] != server:
            status['verification_status'] = 'REMOTE_EVIDENCE_NOT_SUPPLIED'
            continue
        wd = base / 'RB01' / server / f"R{case['repeat']}" / case['case_id']
        training_path = wd / 'meta/training_status.json'
        if training_path.is_file():
            try:
                training = read_json(training_path)
                status.update(actual_updates=training.get('actual_updates'), training_status=training.get('status'))
            except (OSError, ValueError) as exc:
                status['errors'].append('training status: ' + str(exc))
        failure_path = wd / 'meta/controller_failure.json'
        if failure_path.is_file():
            try:
                status['controller_failure'] = read_json(failure_path)
            except (OSError, ValueError) as exc:
                failure = dict(run_id=case['run_id'], stage='controller_failure',
                               error_type=type(exc).__name__, error=str(exc))
                errors.append(failure); status['errors'].append(failure)
        path = wd / 'native/metrics.json'
        if not path.is_file():
            status['verification_status'] = 'SOURCE_MISSING'
            for mode in MODES:
                if (wd / 'stress' / mode / 'completion.json').is_file():
                    status['curve_status'][mode]['status'] = 'PRESENT_BUT_NATIVE_NOT_VERIFIED'
            continue
        try:
            report = read_json(path)
            validate_seal(report, wd)
            original_validate(report, wd)
            cfg, binding, training, selection, provenance = _context_assets(root, case, wd, report['context'], release)
            validate_native(report, wd, selection)
            if original_report is not None:
                rows = [r for r in original_report['rows'] if r['run_id'] == case['run_id']]
                require(len(rows) == 2 and {r['selection_id'] for r in rows} == set(SELECTORS), 'Original report lost this run selections')
                for row in rows:
                    chosen = report['selections'][row['selection_id']]
                    require(row['checkpoint_sha256'] == chosen['checkpoint_sha256'] and row.get('alias_of') == chosen.get('alias_of'),
                            'Original report checkpoint/alias differs')
                    metrics = {k: v for section in ('rr', 'fr', 'supplemental_rr', 'supplemental_fr')
                               for k, v in chosen.get(section, {}).items() if k in row['metrics']}
                    require(metrics == row['metrics'], 'Original report metrics differ')
            provenance.update(native_payload_sha256=report['payload_sha256'], native_file_sha256=file_sha(path),
                              source_file_relative_path=str(path.relative_to(root)), source_completed_at_utc=report['completed_at_utc'],
                              runtime_device=report['device'])
            receipt = dict(verifier, case_id=case['case_id'], run_id=case['run_id'], status='EVIDENCE_VERIFIED',
                           native_file_sha256=file_sha(path), selection_manifest_sha256=provenance['selection_manifest_sha256'])
            provenance['verification_receipt_sha256'] = object_sha(receipt)
            entry = dict(case=case, report=report, config=cfg, binding=binding, training_status=training,
                         provenance=provenance, verification=receipt)
            native_entries.append(entry)
            status.update(native_status='NATIVE_EVAL_COMPLETE', native_selectors=2, verification_status='EVIDENCE_VERIFIED',
                          source_hash=report['payload_sha256'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failure = dict(run_id=case['run_id'], stage='native', error_type=type(exc).__name__, error=str(exc))
            errors.append(failure); status['errors'].append(failure)
            status.update(native_status='EVIDENCE_INVALID', verification_status='EVIDENCE_INVALID')
            for mode in MODES:
                if (wd / 'stress' / mode / 'completion.json').is_file():
                    status['curve_status'][mode]['status'] = 'PRESENT_BUT_NATIVE_NOT_VERIFIED'
            continue
        for mode in MODES:
            curve_path = wd / 'stress' / mode / 'completion.json'
            if not curve_path.is_file():
                continue
            try:
                curve = read_json(curve_path)
                validate_seal(curve, wd)
                original_validate(curve, wd)
                validate_stress(curve, wd, report, mode, grid)
                cp = dict(provenance, evidence_payload_sha256=curve['payload_sha256'],
                          evidence_file_sha256=file_sha(curve_path), source_file_relative_path=str(curve_path.relative_to(root)),
                          source_completed_at_utc=curve['completed_at_utc'], runtime_device=curve['device'])
                cr = dict(verifier, run_id=case['run_id'], mode=mode, status='EVIDENCE_VERIFIED',
                          evidence_file_sha256=file_sha(curve_path), native_file_sha256=provenance['native_file_sha256'])
                cp['verification_receipt_sha256'] = object_sha(cr)
                curve_entries.append(dict(case=case, report=curve, provenance=cp, verification=cr))
                outcome = 'COLLECTED_WITH_FAILURES' if curve['n_numerical_failures'] or curve['n_invalid_geometry'] else 'CURVE_COMPLETE'
                status['curve_status'][mode] = dict(status=outcome, shifts=49, numerical_failures=curve['n_numerical_failures'],
                                                  invalid_geometry=curve['n_invalid_geometry'])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failure = dict(run_id=case['run_id'], stage=mode, error_type=type(exc).__name__, error=str(exc))
                errors.append(failure); status['errors'].append(failure)
                status['curve_status'][mode]['status'] = 'EVIDENCE_INVALID'
    result = dict(schema=PACKAGE_SCHEMA, campaign_id=CAMPAIGN, source_server=server,
                  created_at_utc=datetime.now(timezone.utc).isoformat(), registry=cases, native=native_entries,
                  curves=curve_entries, status=statuses, errors=errors, verification_receipt=verifier,
                  original_report_audit=original_report_audit,
                  measured=dict(students=len(native_entries), native_observations=2 * len(native_entries),
                                curves=len(curve_entries), points=49 * len(curve_entries)),
                  original_artifacts_modified=False, model_or_metric_execution=False)
    result['package_payload_sha256'] = object_sha(result)
    return result


def validate_package(package):
    require(package.get('schema') == PACKAGE_SCHEMA and package.get('campaign_id') == CAMPAIGN,
            'Wrong evidence package schema/campaign')
    require(package.get('package_payload_sha256') == object_sha({k: v for k, v in package.items() if k != 'package_payload_sha256'}),
            'Modified portable evidence package')
    require(package['source_server'] in SERVERS and len(package['registry']) == 24 and len(package['status']) == 24,
            'Package lost registry or source server')
    for entry in [*package['native'], *package['curves']]:
        require(entry['case']['server'] == package['source_server'] and entry['case'] in package['registry'],
                'Package contains unregistered/other-server evidence')
        require(entry['verification']['status'] == 'EVIDENCE_VERIFIED' and
                entry['provenance']['verification_receipt_sha256'] == object_sha(entry['verification']),
                'Package verification receipt changed')
        report = entry['report']
        require(report['payload_sha256'] == object_sha({k: v for k, v in report.items() if k != 'payload_sha256'}),
                'Portable report payload changed')
    return package


def load_packages(path):
    """Load explicit small collect JSON packages, never scan raw imagery/weights."""
    path = Path(path)
    candidates = [path] if path.is_file() else sorted(path.glob('*.json'))
    packages = []
    for candidate in candidates:
        value = read_json(candidate)
        if value.get('schema') == PACKAGE_SCHEMA:
            packages.append(validate_package(value))
    require(bool(packages), 'No verified B01 evidence package found; use collect on each original server first')
    by_server = {}
    for package in packages:
        server = package['source_server']
        if server in by_server:
            require(package['package_payload_sha256'] == by_server[server]['package_payload_sha256'],
                    'Multiple different packages for one source server; select one explicit verified snapshot')
        by_server[server] = package
    return list(by_server.values())
