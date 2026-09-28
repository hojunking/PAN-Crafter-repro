"""Outcome-independent local summaries. Statistical unit = registered Student seed."""
import math
from pathlib import Path
import statistics
import zipfile

from panda_rb.common import ROOT, atomic_json, object_sha, read, sha256, utcnow
from panda_rb.plan import CASES, MODES, SERVERS, campaign_dir, run_dir, training_runs, validate_plan

LOWER = {'ergas', 'sam', 'rmse', 'd_lambda', 'd_s', 'dlambda', 'ds'}


def stats(values):
    values = list(values)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Nonfinite summary observation; never silently drop it')
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None)


def metrics(selection):
    names = {'rr': ['psnr', 'ssim', 'sam', 'ergas', 'scc', 'q8', 'q2n'],
             'fr': ['hqnr', 'd_lambda', 'd_s', 'dlambda', 'ds'],
             'supplemental_rr': ['rmse', 'cc'], 'supplemental_fr': ['jqm']}
    return {key: float(selection[section][key]) for section, keys in names.items()
            for key in keys if key in selection.get(section, {})}


def summarize(root=ROOT, servers=SERVERS, verify=True):
    from panda_rb.evaluation import validate_artifacts
    validate_plan(root)
    if not servers or len(set(servers)) != len(servers) or not set(servers) <= set(SERVERS):
        raise ValueError('Summary admits only unique preregistered s1/s3/s5')
    rows = []; curves = []; missing = []; failures = []
    identities = set(); common_bindings = set(); data_identities = set(); runtime_policies = set(); evaluators = set()
    for case in training_runs(root=root):
        if case['server'] not in servers: continue
        wd = run_dir(case['run_id'], root)
        fail = read(wd / 'meta/controller_failure.json')
        if fail and fail.get('kind') != 'RESOLVED': failures.append(dict(case, failure=fail))
        report = read(wd / 'native/metrics.json')
        if report:
            if verify: validate_artifacts(report, wd)
            if report['context']['case'] != case: raise ValueError('Native result belongs to a different registered seed/case')
            # Paths differ by server; shared numerical source bytes may not.
            identities.add(report['context']['source_identity']['content_sha256'])
            common_bindings.add(report['context']['binding_common_sha256'])
            data_identities.add(object_sha(report['context']['data_content_identity']))
            evaluators.add(object_sha(report['context'].get('evaluator_identity', {})))
            source = report['context']['source_identity']
            runtime_policies.add(object_sha({k: source.get(k) for k in
                ('torch', 'cuda', 'cudnn', 'numpy', 'scipy', 'skimage', 'tf32_matmul', 'tf32_cudnn')}))
            for selection_id, selected in report['selections'].items():
                rows.append(dict(case, selection_id=selection_id, checkpoint_sha256=selected['checkpoint_sha256'],
                                 alias_of=selected.get('alias_of'), metrics=metrics(selected)))
        else: missing.append(dict(case, missing='native'))
        for mode in MODES:
            curve = read(wd / 'stress' / mode / 'completion.json')
            if not curve:
                missing.append(dict(case, missing=mode)); continue
            if verify: validate_artifacts(curve, wd)
            if (curve['identity']['context']['case'] != case or curve['identity']['mode'] != mode
                    or curve.get('n_observations') != 980 or len(curve['curve']) != 49):
                raise ValueError('Stress result lost the preregistered seed/mode/grid')
            if report and curve['identity']['checkpoint_sha256'] != report['selections']['EXACT_50000']['checkpoint_sha256']:
                raise ValueError('Stress modes must use the identical native exact50000 checkpoint')
            curves.append(dict(case, mode=mode, points=curve['curve'],
                               n_numerical_failures=curve['n_numerical_failures'],
                               n_invalid_geometry=curve['n_invalid_geometry']))
    if any(len(values) > 1 for values in (identities, common_bindings, data_identities, runtime_policies, evaluators)):
        raise ValueError('Cannot pool different numerical source/runtime/common F1/data releases')
    native_summary = []; paired = []; curve_summary = []
    for selection_id in ('EXACT_50000', 'RR_VAL_ERGAS_MIN'):
        selected = [r for r in rows if r['selection_id'] == selection_id]
        for case_id in CASES:
            subset = [r for r in selected if r['case_id'] == case_id]
            keys = sorted({k for r in subset for k in r['metrics']})
            for server in ['ALL', *servers]:
                group = subset if server == 'ALL' else [r for r in subset if r['server'] == server]
                native_summary.append(dict(selection_id=selection_id, case_id=case_id, server=server,
                    expected_n=len(servers)*2 if server == 'ALL' else 2,
                    metrics={key: stats(r['metrics'][key] for r in group if key in r['metrics']) for key in keys}))
            if case_id == 'QFULL': continue
            baseline = {(r['server'], r['repeat'], r['seed']): r for r in selected if r['case_id'] == 'QFULL'}
            differences = []
            for row in subset:
                base = baseline.get((row['server'], row['repeat'], row['seed']))
                if base:
                    differences.append(dict(server=row['server'], repeat=row['repeat'], seed=row['seed'],
                        delta={k: row['metrics'][k] - base['metrics'][k]
                               for k in row['metrics'].keys() & base['metrics'].keys()}))
            for server in ['ALL', *servers]:
                group = differences if server == 'ALL' else [d for d in differences if d['server'] == server]
                paired.append(dict(selection_id=selection_id, case_id=case_id, reference='QFULL', server=server,
                    difference='case minus QFULL, within same server/repeat/seed',
                    expected_n=len(servers)*2 if server == 'ALL' else 2, per_seed=group,
                    metrics={k: dict(stats(d['delta'][k] for d in group if k in d['delta']),
                             improved=sum((d['delta'][k] < 0 if k in LOWER else d['delta'][k] > 0)
                                          for d in group if k in d['delta']),
                             ties=sum(d['delta'][k] == 0 for d in group if k in d['delta'])) for k in keys}))
    for case_id in CASES:
        for mode in MODES:
            group = [c for c in curves if c['case_id'] == case_id and c['mode'] == mode]
            if not group: continue
            points = []
            for i in range(49):
                observations = [c['points'][i] for c in group]
                if len({p['id'] for p in observations}) != 1: raise ValueError('Curve grid ordering differs')
                keys = [k for k, v in observations[0].items() if isinstance(v, (float, int))
                        and not k.endswith('_scene_std') and k not in ['n_scenes', 'n_failures', 'radius_hr', 'angle_deg', 'angle_degrees', 'dy', 'dx']]
                points.append(dict(id=observations[0]['id'], metrics={k: dict(
                    stats(p[k] for p in observations if p.get(k) is not None),
                    failed_students=sum(p.get(k) is None for p in observations)) for k in keys}))
            curve_summary.append(dict(case_id=case_id, mode=mode, n_students=len(group),
                                      expected_n=len(servers)*2, points=points))
    stress_paired, radius_summary = stress_contrasts(curves, servers)
    primary = [r for r in rows if r['selection_id'] == 'EXACT_50000']
    return dict(schema='PANDA_RB_B01_LOCAL_REPORT_v1', servers=list(servers), created_at_utc=utcnow(),
        expected_students=8*len(servers), expected_curves=16*len(servers), completed_students=len(primary),
        completed_curves=len(curves), failed_students=len({x['run_id'] for x in failures}),
        complete=len(primary) == 8*len(servers) and len(curves) == 16*len(servers) and not failures,
        statistical_unit='Student seed; never scene, direction, mode, or selection alias',
        server_effect_identifiable=False, server_note='Different seed cohorts confound server main effects',
        rows=rows, native_summary=native_summary, paired_differences=paired, curve_summary=curve_summary,
        paired_mode_curves=stress_paired, radius_summary=radius_summary,
        missing=missing, failures=failures, historical_tables_modified=False, sheets_uploaded=False)


def stress_contrasts(curves, servers):
    """Average directions within a Student first; do not inflate n by 8 or 20."""
    keys = ('ergas', 'psnr', 'sam', 'edge_error_dn', 'relative_response_l1_sum', 'coverage_all_pan_paths')
    paired = []; radii = []; blocks = {}
    for curve in curves:
        block = (curve['case_id'], curve['server'], curve['repeat'], curve['seed'])
        blocks.setdefault(block, {})[curve['mode']] = curve
    differences = []
    for (case, server, repeat, seed), modes in blocks.items():
        if set(modes) != set(MODES): continue
        points = []
        for on, zero in zip(modes['A_ON']['points'], modes['A_ZERO_INFERENCE_ONLY']['points']):
            if on['id'] != zero['id']: raise ValueError('Paired mode grids differ')
            points.append(dict(id=on['id'], radius_hr=on['radius_hr'], **{
                k: None if on.get(k) is None or zero.get(k) is None else on[k]-zero[k] for k in keys}))
        differences.append(dict(case_id=case, server=server, repeat=repeat, seed=seed,
                                 mode='A_ON_MINUS_A_ZERO', points=points))
    for case in CASES:
        for server in ['ALL', *servers]:
            expected = len(servers)*2 if server == 'ALL' else 2
            group = [c for c in differences if c['case_id'] == case and (server == 'ALL' or c['server'] == server)]
            if group:
                points = [dict(id=group[0]['points'][i]['id'], metrics={k: dict(
                    stats(c['points'][i][k] for c in group if c['points'][i][k] is not None),
                    failed_students=sum(c['points'][i][k] is None for c in group)) for k in keys}) for i in range(49)]
                paired.append(dict(case_id=case, server=server, expected_n=expected, n_students=len(group),
                    mode='A_ON_MINUS_A_ZERO', points=points,
                    interpretation='Inference correction dependence, not independently trained no-align'))
            for mode in (*MODES, 'A_ON_MINUS_A_ZERO'):
                group = [c for c in [*curves, *differences] if c['case_id'] == case and c['mode'] == mode
                         and (server == 'ALL' or c['server'] == server)]
                if not group: continue
                for radius in (0., .25, .5, 1., 2., 3., 4.):
                    values = {k: [] for k in keys}; failed = {k: 0 for k in keys}
                    for student in group:
                        points = [p for p in student['points'] if p['radius_hr'] == radius]
                        if len(points) != (1 if radius == 0 else 8): raise ValueError('Missing preregistered radius directions')
                        for k in keys:
                            if any(p.get(k) is None for p in points): failed[k] += 1
                            else: values[k].append(statistics.mean(p[k] for p in points))
                    radii.append(dict(case_id=case, server=server, mode=mode, radius_hr=radius,
                        expected_n=expected, metrics={k: dict(stats(values[k]), failed_students=failed[k]) for k in keys}))
    return paired, radii


def package(root, servers, output):
    """A new ZIP of measured evidence only, never a fake plan-only completion."""
    root = Path(root).resolve(); output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Refusing to replace an evidence ZIP')
    report = summarize(root, servers, verify=True)
    if not report['completed_students']: raise ValueError('No measured Student; cannot create a return package')
    files = {}
    for row in training_runs(root=root):
        if row['server'] not in servers: continue
        wd = run_dir(row['run_id'], root)
        if output.is_relative_to(wd): raise ValueError('Write result ZIP outside the measured run directories')
        for path in sorted(wd.rglob('*')):
            if not path.is_file(): continue
            rel = path.relative_to(wd)
            if rel.parts[0] == 'checkpoints' and not (len(rel.parts) > 1 and rel.parts[1] in
                    ['exact50000', 'val_selected', 'selection_manifest.json']): continue
            files[str(path.relative_to(root))] = sha256(path)
        # rglob deliberately does not traverse directory symlinks. Include the
        # val-selected target through its named alias, without all 50 candidates.
        for selection in ('exact50000', 'val_selected'):
            for name in ('model.safetensors', 'identity.json'):
                path = wd / 'checkpoints' / selection / name
                if path.is_file(): files[str(path.relative_to(root))] = sha256(path)
    common = campaign_dir(root) / 'common'
    for directory in ('weights', 'reference_probe'):
        for path in sorted((common / directory).rglob('*')):
            if path.is_file(): files[str(path.relative_to(root))] = sha256(path)
    for server in servers:
        for path in sorted((campaign_dir(root) / 'control' / server).rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl'):
                files[str(path.relative_to(root))] = sha256(path)
        deployment = read(campaign_dir(root) / 'control' / server / 'deployment.json')
        if deployment:
            release = campaign_dir(root) / 'releases' / deployment['release_sha256']
            for path in sorted(release.rglob('*')):
                if path.is_file(): files[str(path.relative_to(root))] = sha256(path)
    manifest = dict(schema='PANDA_RB_RETURN_v1', files=files, report=report,
                    note='Resume/validation candidates retained locally, not pruned; historical assets excluded')
    import json
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, indent=2, allow_nan=False))
        for name, digest in files.items():
            if sha256(root / name) != digest: raise ValueError('Artifact changed during package creation')
            archive.write(root / name, name)
    result = dict(path=str(output), sha256=sha256(output), bytes=output.stat().st_size,
                  measured_students=report['completed_students'], measured_curves=report['completed_curves'],
                  complete=report['complete'], manifest_sha256=object_sha(manifest))
    atomic_json(output.with_suffix(output.suffix + '.sha256.json'), result)
    return result
