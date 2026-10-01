"""Read-only M12 evidence collection, Student-block statistics, and sealed return ZIPs.

Importing/reporting/packaging never starts a trainer, evaluator, or Sheet writer.
"""
from __future__ import annotations

import json
import csv
import io
import os
from pathlib import Path
import tempfile
import zipfile

from panda_rb_m12.common import ROOT, atomic_json, object_sha, read, sha256, utcnow
from panda_rb_m12.plan import campaign_dir, run_dir, training_runs, validate_plan
from panda_rb_m12.analysis import analyze, radius_summary, paired_modes
from reporting_bridge.rb_m12_contract import (
    CAMPAIGN, CASES, MODES, SELECTIONS, SERVERS, NATIVE_PROTOCOL, STRESS_PROTOCOL,
    cohort_id, number, sha, validate_case,
)


def verify_artifacts(report, directory):
    directory = Path(directory).resolve()
    if report.get('complete') is not True:
        raise ValueError('Incomplete evidence is a pending item, never a completed observation')
    payload = {k: v for k, v in report.items() if k != 'payload_sha256'}
    if object_sha(payload) != report.get('payload_sha256'):
        raise ValueError('Evidence payload seal mismatch')
    files = report.get('file_hashes', {})
    if not files:
        raise ValueError('Measured evidence has no original artifact hashes')
    for relative, expected in files.items():
        relative = Path(relative)
        path = directory / relative
        if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(directory):
            raise ValueError('Evidence artifact escapes its owning M12 run')
        if sha256(path) != expected:
            raise ValueError('Evidence bytes changed: ' + str(relative))
    return report


def _provenance(context):
    source = context['source_identity']
    runtime_keys = ('torch', 'cuda', 'cudnn', 'numpy', 'scipy', 'skimage', 'tf32_matmul', 'tf32_cudnn')
    result = dict(numerical_source_sha256=source['content_sha256'],
                  evaluator_sha256=object_sha(context['evaluator_identity']),
                  binding_common_sha256=context['binding_common_sha256'],
                  data_content_identity_sha256=object_sha(context['data_content_identity']),
                  runtime_policy_sha256=object_sha({k: source.get(k) for k in runtime_keys}),
                  teacher_sha256=context.get('teacher_sha256'),
                  raw_map_identity_sha256=context.get('raw_map_identity_sha256'),
                  initialization_sha256=context.get('initialization_sha256'),
                  initial_tensor_hashes=context.get('initial_tensor_hashes'),
                  consumed_stream_sha256=context.get('consumed_stream_sha256'),
                  source_config_sha256=context['config_sha256'])
    cohort_id(result)
    return result


def _metrics(selection):
    result = {}
    for section in ('rr', 'fr', 'supplemental_rr', 'supplemental_fr'):
        for key, value in selection.get(section, {}).items():
            aliases = {'ds': 'd_s', 'dlambda': 'd_lambda', 'q2n': 'q8'}
            key = aliases.get(key, key)
            if key not in ('hqnr', 'd_s', 'd_lambda', 'jqm', 'ergas', 'scc', 'sam', 'psnr', 'ssim', 'q8', 'rmse', 'cc'):
                continue
            value = number(value)
            if key in result and result[key] != value:
                raise ValueError('Metric aliases disagree: ' + key)
            result[key] = value
    for key in ('hqnr', 'd_s', 'd_lambda', 'ergas', 'scc', 'sam', 'psnr', 'ssim', 'q8'):
        number(result.get(key))
    return result


def native_records(case, report, *, config_fields=None, costs=None):
    validate_case(case)
    if report.get('schema') != 'PANDA_M12_NATIVE_v1' or report.get('complete') is not True:
        raise ValueError('Only complete M12 native schema is accepted')
    if report['context']['case'] != case or set(report.get('selections', {})) != set(SELECTIONS):
        raise ValueError('Native run/selection registry identity mismatch')
    provenance = _provenance(report['context'])
    rows = []
    for selector in SELECTIONS:
        selected = report['selections'][selector]
        update = selected['update']
        if selected.get('selection_id') != selector or type(update) is not int:
            raise ValueError('Selector identity mismatch')
        if selector == SELECTIONS[0] and update != 50000:
            raise ValueError('M12 primary must be exact50000')
        if update not in (*range(1010, 50000, 1010), 50000):
            raise ValueError('Selection outside preregistered validation grid')
        rows.append(dict(case, selection_id=selector, update=update,
                         checkpoint_sha256=sha(selected['checkpoint_sha256']),
                         alias_of=selected.get('alias_of'), metrics=_metrics(selected),
                         protocol=NATIVE_PROTOCOL, provenance=provenance,
                         payload_sha256=sha(report['payload_sha256']),
                         config_fields=config_fields or {}, costs=costs or {},
                         jqm_variant=selected.get('jqm_variant', report.get('jqm_variant')),
                         completed_at_utc=report['completed_at_utc']))
    primary, secondary = rows
    same = primary['checkpoint_sha256'] == secondary['checkpoint_sha256']
    if same:
        if secondary['alias_of'] != SELECTIONS[0] or secondary['metrics'] != primary['metrics']:
            raise ValueError('Same checkpoint must preserve exact metric alias; actual selected steps remain distinct')
    elif secondary['alias_of'] is not None:
        raise ValueError('Alias cannot point to a different checkpoint')
    if primary['alias_of'] is not None:
        raise ValueError('Primary must not be a secondary alias')
    return rows


def summarize(root=ROOT, servers=SERVERS, verify=True, plan=None):
    root = Path(root).resolve()
    validate_plan(root)
    if not servers or len(set(servers)) != len(servers) or not set(servers) <= set(SERVERS):
        raise ValueError('Only unique s1/s3/s5 M12 cohorts')
    rows = []; curves = []; missing = []; failures = []; statuses = []; points = []; all_cohorts = set()
    for case in training_runs(root=root):
        if case['server'] not in servers:
            continue
        wd = run_dir(case['run_id'], root)
        state = read(wd / 'meta/training_status.json')
        failure = read(wd / 'meta/controller_failure.json')
        if failure and failure.get('kind') != 'RESOLVED':
            failures.append(dict(case, failure=failure))
        native = read(wd / 'native/metrics.json')
        runrows = []; runcurves = []
        if native:
            if verify:
                verify_artifacts(native, wd)
            # Resolved training hyperparameters are evidence, not inferred tuning.
            import yaml
            config_path = wd / 'meta/config.resolved.yaml'
            cfg = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
            if verify and object_sha(cfg) != native['context']['config_sha256']:
                raise ValueError('Report resolved configuration hash mismatch')
            policy = cfg.get('panda_rb_m12', {})
            binding = read(wd / 'bindings.json')
            if verify and object_sha(binding) != native['context']['binding_sha256']:
                raise ValueError('Report per-run frozen binding hash mismatch')
            config_fields = dict(q_ref=binding.get('q_ref'), tau_R=binding.get('tau_R'),
                                 alpha=policy.get('alpha'), beta=policy.get('beta'), lambda_E=policy.get('lambda_E'),
                                 U_peak_lr=cfg.get('learning_rate'), A_peak_lr=policy.get('aligner_lr'),
                                 teacher_reference='F1 exact50K / ' + str(native['context'].get('teacher_sha256', ''))[:12])
            training_seconds = state.get('timings', {}).get('train')
            costs = dict(train_hours=None if training_seconds is None else training_seconds/3600,
                         eval_hours=native.get('elapsed_seconds', 0)/3600)
            runrows = native_records(case, native, config_fields=config_fields, costs=costs)
            rows.extend(runrows)
            all_cohorts.add(cohort_id(runrows[0]['provenance']))
        else:
            missing.append(dict(case, missing='native'))
        if case['case_id'] in CASES[:6]:
            for mode in MODES:
                artifact = wd / 'stress' / mode / 'completion.json'
                curve = read(artifact)
                if not curve:
                    missing.append(dict(case, missing=mode))
                    continue
                if verify:
                    verify_artifacts(curve, wd)
                identity = curve['identity']
                if (curve.get('schema') != 'PANDA_M12_CURVE_v1' or identity['context']['case'] != case
                        or identity['mode'] != mode or curve.get('n_observations') != 980
                        or len(curve.get('curve', [])) != 49 or identity.get('roi_primary') != 'fixed160'):
                    raise ValueError('M12 stress identity/grid/ROI mismatch')
                # Native inference can be a retry debt while independently verified stress is complete.
                # Validate the frozen training selection rather than requiring native inference first.
                selection_path = wd / 'checkpoints/selection_manifest.json'
                selected = read(selection_path)
                if (sha256(selection_path) != identity['context']['selection_manifest_sha256']
                        or selected.get('complete') is not True or selected.get('actual_updates') != 50000
                        or selected.get('primary', {}).get('selection_id') != SELECTIONS[0]
                        or selected['primary'].get('update') != 50000
                        or selected['primary'].get('checkpoint_sha256') != identity['checkpoint_sha256']):
                    raise ValueError('Stress does not match the frozen exact50000 training selection')
                if runrows and identity['checkpoint_sha256'] != runrows[0]['checkpoint_sha256']:
                    raise ValueError('Stress must use the same measured exact50000 checkpoint')
                provenance = _provenance(identity['context'])
                all_cohorts.add(cohort_id(provenance))
                if runrows and cohort_id(provenance) != cohort_id(runrows[0]['provenance']):
                    raise ValueError('Native/stress evidence cohort mismatch')
                entry = dict(case, mode=mode, points=curve['curve'], auxiliary_points=curve.get('auxiliary_curve', []),
                             checkpoint_sha256=identity['checkpoint_sha256'],
                             n_numerical_failures=curve['n_numerical_failures'],
                             n_invalid_geometry=curve['n_invalid_geometry'],
                             n_auxiliary_failures=curve.get('n_auxiliary_failures', 0))
                curves.append(entry); runcurves.append(entry)
                for i, point in enumerate(curve['curve']):
                    if point.get('id') != f'D{i:03d}' or point.get('n_expected') != 20 or point.get('n_scenes') != 20:
                        raise ValueError('Stress must preserve every scheduled shift and scene')
                    points.append(dict(case, **{k: v for k, v in point.items() if k not in case},
                                       shift_id=point['id'], mode=mode, protocol=STRESS_PROTOCOL,
                                       checkpoint_sha256=identity['checkpoint_sha256'],
                                       grid_sha256=identity['grid_sha256'], payload_sha256=curve['payload_sha256'],
                                       source_artifact=str(artifact.relative_to(root)), provenance=provenance))
        clean_count = sum(all(p.get('clean') is True and p.get('n_valid') == 20 for p in c['points']) for c in runcurves)
        statuses.append(dict(case, actual_updates=state.get('actual_updates', state.get('update', 0)),
                             training_status=state.get('status', 'PENDING'), native_selectors=len(runrows),
                             stress_curves=len(runcurves), clean_curves=clean_count,
                             flagged_curves=len(runcurves)-clean_count, errors=[failure] if failure else [], checked_at_utc=utcnow()))
    if len(all_cohorts) > 1:
        raise ValueError('Native/stress evidence spans different fixed-F1/numerical/data cohorts')
    summary = analyze(rows, servers)
    primary = [r for r in rows if r['selection_id'] == SELECTIONS[0]]
    clean_curves = sum(all(p.get('clean') is True and p.get('n_valid') == 20 for p in c['points']) for c in curves)
    auxiliary_flagged = sum(c['n_auxiliary_failures'] > 0 for c in curves)
    return dict(schema='PANDA_M12_REPORT_v1', campaign_id=CAMPAIGN, servers=list(servers),
                created_at_utc=utcnow(), expected_students=40*len(servers), expected_native=80*len(servers),
                expected_curves=96*len(servers), expected_primary_points=4704*len(servers),
                completed_students=len(primary), completed_native=len(rows), completed_curves=len(curves),
                clean_curves=clean_curves, flagged_curves=len(curves)-clean_curves,
                clean_curve_scope='primary fixed160; auxiliary fixed192 flags reported separately',
                auxiliary_flagged_curves=auxiliary_flagged,
                failed_students=len({f['run_id'] for f in failures}), pending_students=40*len(servers)-len(primary),
                complete=len(primary) == 40*len(servers) and len(curves) == 96*len(servers) and not failures,
                outcome='PROCESSED_WITH_FLAGS' if len(curves) > clean_curves or auxiliary_flagged else 'NO_OBSERVED_STRESS_FLAGS',
                rows=rows, curves=curves, point_records=points, status_records=statuses,
                missing=missing, failures=failures, radius_summary=radius_summary(curves, servers),
                paired_mode_radii=paired_modes(curves, servers),
                historical_cohorts_pooled=False, historical_tables_modified=False, sheets_uploaded=False, **summary)


def package(root, servers, output, plan=None):
    """Read and verify existing evidence only; never generate missing inference."""
    root = Path(root).resolve(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Evidence ZIP is immutable; choose a new output filename')
    report = summarize(root, servers, verify=True, plan=plan)
    if not report['completed_students']:
        raise ValueError('No measured M12 Student; no plan-only completion package')
    files = {}
    for case in training_runs(root=root):
        if case['server'] not in servers:
            continue
        wd = run_dir(case['run_id'], root).resolve()
        if output.is_relative_to(wd):
            raise ValueError('Return package must be outside measured run directories')
        for top in ('meta', 'native', 'stress', 'diagnostics'):
            for path in sorted((wd / top).rglob('*')):
                if not path.is_file():
                    continue
                if not path.resolve().is_relative_to(wd):
                    raise ValueError('Package symlink escapes its measured run')
                files[str(path.relative_to(root))] = sha256(path)
        for name in ('selection_manifest.json', 'exact50000/identity.json', 'val_selected/identity.json'):
            path = wd / 'checkpoints' / name
            if path.is_file():
                if not path.resolve().is_relative_to(wd):
                    raise ValueError('Checkpoint identity alias escapes its measured run')
                files[str(path.relative_to(root))] = sha256(path)
        for name in ('init_manifest.json', 'stream_manifest.json', 'bindings.json', 'case.json'):
            path = wd / name
            if path.is_file():
                if not path.resolve().is_relative_to(wd):
                    raise ValueError('Run provenance escapes its measured run')
                files[str(path.relative_to(root))] = sha256(path)
    for server in servers:
        for folder in (campaign_dir(root) / 'common/response' / server,
                       campaign_dir(root) / 'control' / server):
            for path in sorted(folder.rglob('*')):
                if not path.is_file() or path.suffix not in ('.json', '.jsonl', '.csv', '.md', '.txt', '.npy', '.npz'):
                    continue
                if not path.resolve().is_relative_to(folder.resolve()):
                    raise ValueError('Server diagnostic provenance escapes its owning directory')
                files[str(path.relative_to(root))] = sha256(path)
    manifest = dict(schema='PANDA_M12_RETURN_v1', campaign_id=CAMPAIGN, files=files, report=report,
                    note='Measured evidence and checkpoint identities; no credentials, raw training data, model weights, or resume states')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False))
        for name, expected in files.items():
            if sha256(root / name) != expected:
                raise ValueError('Evidence changed before packaging')
            archive.write(root / name, name)
            if sha256(root / name) != expected:
                raise ValueError('Evidence changed during packaging; incomplete archive is not verified')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Evidence ZIP failed CRC verification')
    receipt = dict(schema='PANDA_M12_PACKAGE_RECEIPT_v1', path=str(output), sha256=sha256(output),
                   bytes=output.stat().st_size, manifest_sha256=object_sha(manifest),
                   measured_students=report['completed_students'], complete=report['complete'])
    atomic_json(output.with_suffix(output.suffix + '.sha256.json'), receipt)
    return receipt


def export_tables(report, output_dir):
    """Materialize the explicitly requested review CSV/MD; no automatic human Sheet edits."""
    if report.get('schema') != 'PANDA_M12_REPORT_v1' or report.get('campaign_id') != CAMPAIGN:
        raise ValueError('Only an already-collected M12 report can be rendered')
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'report.json', report)

    def write_text(name, value):
        destination = output / name
        fd, temporary = tempfile.mkstemp(prefix='.' + name, dir=output)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
                stream.write(value); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()

    def write_csv(name, rows, fields):
        stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows); write_text(name, stream.getvalue())

    native_fields = ['run_id', 'server', 'repeat', 'seed', 'case_id', 'selection_id', 'checkpoint_sha256', 'alias_of',
                     'hqnr', 'd_s', 'd_lambda', 'ergas', 'sam', 'psnr', 'scc', 'ssim', 'q8']
    native = []
    for row in report['rows']:
        native.append({key: row['metrics'].get(key) if key in row['metrics'] else row.get(key) for key in native_fields})
    write_csv('native_students.csv', native, native_fields)
    contrast_rows = []; paired_rows = []
    primary = {(r['contrast'], r['metric']): r for r in report['primary_endpoint_tests']}
    for contrast in (*report['contrasts'], *report['paired_differences']):
        for metric, value in contrast['metrics'].items():
            interval = value.get('ci95') or [None, None]; boot = value.get('bootstrap_ci95') or [None, None]
            test = primary.get((contrast['contrast'], metric), {}) if contrast['selection_id'] == 'EXACT_50000' else {}
            contrast_rows.append(dict(selection=contrast['selection_id'], contrast=contrast['contrast'], metric=metric,
                n=value['n'], expected_n=value['expected_n'], mean_delta=value['mean'], sample_sd=value['sample_std'],
                ci95_low=interval[0], ci95_high=interval[1], bootstrap_low=boot[0], bootstrap_high=boot[1],
                improved=value['improved'], tied=value['tied'], worsened=value['worsened'],
                p_two_sided=test.get('p_two_sided'), p_holm_primary8=test.get('p_holm_primary8')))
            for seed in value['per_seed']:
                paired_rows.append(dict(selection=contrast['selection_id'], contrast=contrast['contrast'], metric=metric, **seed))
    write_csv('contrasts.csv', contrast_rows, ['selection', 'contrast', 'metric', 'n', 'expected_n', 'mean_delta', 'sample_sd',
              'ci95_low', 'ci95_high', 'bootstrap_low', 'bootstrap_high', 'improved', 'tied', 'worsened', 'p_two_sided', 'p_holm_primary8'])
    write_csv('paired_students.csv', paired_rows, ['selection', 'contrast', 'metric', 'server', 'repeat', 'seed', 'delta'])
    # Two ROI protocols stay separate even in downloadable review files.
    for name, field in (('stress_fixed160.csv', 'points'), ('stress_fixed192_auxiliary.csv', 'auxiliary_points')):
        curve_rows = []
        for curve in report['curves']:
            for point in curve[field]:
                curve_rows.append(dict(run_id=curve['run_id'], case_id=curve['case_id'], server=curve['server'],
                                       repeat=curve['repeat'], seed=curve['seed'], mode=curve['mode'], **point))
        keys = list(dict.fromkeys(k for row in curve_rows for k in row)) or ['run_id', 'mode', 'id']
        write_csv(name, curve_rows, keys)
    text = ['# M12 review — fixed F1, new Student-seed cohort', '',
            f"Native-complete Students: {report['completed_students']}/{report['expected_students']}; "
            f"native selections: {report['completed_native']}/{report['expected_native']}; "
            f"processed curves: {report['completed_curves']}/{report['expected_curves']}.", '',
            'EXACT_50000 is primary. RR_VAL_ERGAS_MIN is secondary; identical-weight aliases are not new Students.', '',
            'Mean ± sample SD uses Student seeds (ddof=1). Paired t CIs assume independent approximately normal block differences. '
            'Sensitivity: server-stratified paired bootstrap, 10,000 draws, seed20261001. '
            'Server and seed are confounded. A CI containing zero does not establish equivalence.', '']
    wanted = ('hqnr', 'd_s', 'd_lambda', 'ergas', 'sam', 'psnr')
    selected = {r['case_id']: r for r in report['native_summary'] if r['selection_id'] == 'EXACT_50000' and r['server'] == 'ALL'}
    for label, cases in [('Q1 — q cue', CASES[:4]), ('Q2 — 2×2 routing', ('QMEAN', 'QEDGE', 'QALIGN', 'QFULL')),
                         ('H — hard fitting', ('QFULL', 'H0', 'HSPMEAN')), ('K — advantage', ('QFULL', 'NOADV', 'ADVMEAN'))]:
        text += ['## ' + label, '', '| Case | ' + ' | '.join(wanted) + ' |', '| --- | ' + ' | '.join(['---']*len(wanted)) + ' |']
        for case in cases:
            cells = []
            for metric in wanted:
                value = selected.get(case, {}).get('metrics', {}).get(metric, {})
                cells.append('pending' if not value.get('n') else f"{value['mean']:.6g} ± {value['sample_std']:.3g} (n={value['n']})" if value.get('sample_std') is not None else f"{value['mean']:.6g} (n=1; SD unavailable)")
            text.append('| ' + case + ' | ' + ' | '.join(cells) + ' |')
        text.append('')
    text += ['Full factorial coefficients, paired per-seed deltas, CI/seed wins/ties/losses and Holm-adjusted eight primary tests are in contrasts.csv and paired_students.csv.', '',
             'Historical B01/ABLR2 are not pooled. No paper/manual summary Sheet has been edited. Missing or failed runs are not replaced with new seeds.', '']
    write_text('review_tables.md', '\n'.join(text))
    return dict(output_dir=str(output), files={path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()})
