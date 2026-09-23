"""Evidence-bound balanced panels, kept separate from adaptive rechecks.

This module does not choose seeds, replace observations or tune thresholds.
Policy calculations consume full-precision, same-checkpoint records only.
"""
import csv
import io
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev

from fh12.common import ROOT, read_json, object_sha, sha256
from ablr2.common import immutable_json
from ablr2.plan import (MAIN_CASES, LEGACY_MAIN_CASES, COMPARISON_GRAPH,
                       LEGACY_COMPARISON_GRAPH, CAMPAIGN_ID, GRAPH, campaign_id)
from ablr2.postrun import (find_config, _case, _workdir, validate_grid, verify_fullstate,
                           select_records, selection_report, SELECTIONS)
from ablr2 import policy


def _observed_source(row):
    value=row.get('analysis_source_identity',row['source_identity'])
    if value!=row['source_identity'] and not row.get('source_bridge_sha256'):
        raise ValueError('PAIRING_NOT_VERIFIED: changed numerical source has no verified bridge')
    return value


def _metrics(report):
    rr, fr = report['rr'], report['fr']
    values = dict(HQNR=fr['hqnr'], ERGAS=rr['ergas'], E_val=report['val_ergas'],
                  D_lambda=fr['d_lambda'], D_s=fr['d_s'], SAM=rr['sam'], SCC=rr['scc'],
                  PSNR=rr['psnr'], SSIM=rr['ssim'], RMSE=rr['rmse'], CC=rr['cc'],
                  **{key.upper(): rr[key] for key in ('q4', 'q8') if key in rr})
    if fr.get('jqm') is not None:
        values['JQM'] = fr['jqm']
    return values


def case_report(case, root=ROOT, _seen=None):
    """Validate an observed run; partial/mismatched evidence never becomes zero."""
    from ablr2.common import run_dir
    seen = set() if _seen is None else set(_seen)
    if case.run_id in seen:
        raise ValueError('Cyclic logical reuse is not an observation')
    seen.add(case.run_id)
    cfg = find_config(case.run_id, root)
    if _case(cfg) != case:
        raise ValueError('Registered case and actual resolved observation differ')
    wd = _workdir(cfg, root)
    reuse_path = wd / 'meta/reuse.json'
    if reuse_path.is_file():
        from ablr2.common import execution_identity
        from ablr2.plan import case_for
        reuse = read_json(reuse_path)
        if case.phase != 'FIT_ROUND' or reuse.get('schema') != 'ABLR2_REUSE_v1' or reuse.get('logical_case') != asdict(case):
            raise ValueError('Only exact preregistered fitting computations may be reused')
        physical_case = case_for(reuse['physical_run_id'], root=root)
        if physical_case.sensor != case.sensor or physical_case.server_id != case.server_id or physical_case.case_id != case.case_id:
            raise ValueError('Logical reuse crossed sensor/server/component identity')
        physical_cfg = find_config(physical_case.run_id, root)
        logical_signature = execution_identity(cfg, root)
        physical_signature = execution_identity(physical_cfg, root)
        if (reuse.get('logical_execution_signature') != logical_signature
                or reuse.get('physical_execution_signature') != physical_signature
                or logical_signature != physical_signature):
            raise ValueError('Logical reuse is not the exact same numerical computation')
        grid_path = run_dir(physical_case.run_id, root) / 'official/raw_grid.json'
        if reuse.get('physical_raw_grid_sha256') != sha256(grid_path):
            raise ValueError('Physical official-grid proof changed after logical reuse registration')
        physical = case_report(physical_case, root, _seen=seen)
        if reuse.get('source_identity') != physical['source_identity'] or reuse.get('data_sha256') != physical['data_sha256']:
            raise ValueError('Logical reuse source/data evidence differs from the observation')
        # The logical task identity is new; checkpoint/config/metric provenance
        # and the independent-observation identity remain the original run.
        return dict(physical, run_id=case.run_id, phase=case.phase, wave=case.wave, sweep=case.sweep,
                    recipe_id=case.recipe_id, recipe_revision=case.recipe_revision,
                    teacher_seed=case.teacher_seed, student_seed=case.student_seed,
                    reference_id=case.reference_id, physical_run_id=physical_case.run_id,
                    independent_observation_run_id=physical.get('independent_observation_run_id', physical_case.run_id),
                    logical_config_sha256=object_sha(cfg), reused=True, reuse_receipt=reuse,
                    incremental_compute_seconds=0, new_independent_observation=False)
    grid = read_json(wd / 'official/raw_grid.json')
    training = read_json(wd / 'meta/training_status.json')
    status = read_json(wd / 'official/postrun_status.json')
    if (training.get('training_complete') is not True or training.get('actual_updates') != 50000
            or status.get('official_complete') is not True or status.get('n_evaluated') != 50
            or grid.get('complete') is not True):
        raise ValueError('PILOT_INCOMPLETE: unfinished training/evaluation is not an observation')
    validate_grid(case.run_id, cfg, grid, root)
    verify_fullstate(cfg, root)
    selected, reports = select_records(grid['records'], case.sensor), {}
    for key, label in SELECTIONS:
        report = read_json(wd / 'official' / f'{key}.json')
        if report != selection_report(cfg, grid, key, selected[key]):
            raise ValueError('Observation uses a changed checkpoint rule or checkpoint')
        reports[label] = report
    reference = None
    if grid.get('reference_sha256') is not None:
        reference = read_json(wd / 'meta/consumed_reference.json')
        if (object_sha(reference) != grid['reference_sha256'] or reference.get('reference_id') != case.reference_id
                or reference.get('sensor') != case.sensor or reference.get('owner_server') != case.server_id
                or reference.get('teacher_seed') != case.teacher_seed):
            raise ValueError('Observed Student consumed a different local Teacher reference')
    elif case.requires_teacher:
        raise ValueError('Teacher-dependent Student cannot have null reference provenance')
    initial = read_json(wd / 'init_manifest.json')
    started = read_json(wd / 'meta/training_start_manifest.json')
    if (any(initial.get(k) != grid.get(k) or started.get(k) != grid.get(k)
            for k in ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256'))
            or not started.get('sampler_hash') or not initial.get('hashes', {}).get('U')):
        raise ValueError('Initial weights / common sample stream proof missing or changed')
    from ablr2.common import assert_compatible_source, source_identity
    consumer=source_identity(root)
    bridge=assert_compatible_source(grid['source_identity'],consumer,root,case.server_id)
    return dict(run_id=case.run_id, case_id=case.case_id, sensor=case.sensor, server=case.server_id,
                phase=case.phase, wave=case.wave, sweep=case.sweep, recipe_id=case.recipe_id,
                recipe_revision=case.recipe_revision, teacher_seed=case.teacher_seed, student_seed=case.student_seed,
                reference_id=case.reference_id, reference_sha256=grid['reference_sha256'],
                source_identity=grid['source_identity'], data_sha256=grid['data_sha256'],
                analysis_source_identity=consumer,source_bridge_sha256=bridge.get('receipt_sha256') if bridge else None,
                teacher_run_id=reference.get('teacher_run_id') if reference else None,
                teacher_kind=case.teacher_kind,
                config_sha256=object_sha(cfg), VAL=_metrics(reports['RR_VAL_SELECTED']),
                init_U_sha256=initial['hashes']['U'], init_A_sha256=initial['hashes']['A'],
                sampler_sha256=started['sampler_hash'], rng_roles=started['rng_roles'],
                EXACT50K=_metrics(reports['EXACT50K']), RAW_MAX=_metrics(reports['RAW_MAX']),
                val_step=reports['RR_VAL_SELECTED']['step'], exact_step=50000,
                val_checkpoint_sha256=reports['RR_VAL_SELECTED']['checkpoint_sha256'],
                costs={k: training.get(k) for k in ('training_seconds', 'evaluation_seconds', 'io_seconds', 'retry_seconds')},
                development_mode='TEST_AWARE_DEV', independent_test=False,
                independent_observation_run_id=case.run_id, reused=False, new_independent_observation=True)


def assemble_wave(cases, observations, thresholds=None, a17_anchors=None):
    """Pure panel assembly; observations must come from case_report at runtime."""
    cases = tuple(cases)
    case_ids={c.case_id for c in cases if c.role=='S'}
    if case_ids not in (set(LEGACY_MAIN_CASES),set(MAIN_CASES)):
        raise ValueError('PILOT_INCOMPLETE: expected complete legacy17 or extended18 coverage')
    component_cases=MAIN_CASES if case_ids==set(MAIN_CASES) else LEGACY_MAIN_CASES
    extended=len(component_cases)==18
    expected=5*(2+len(component_cases))
    if len(cases) != expected or len({c.run_id for c in cases}) != expected:
        raise ValueError('PILOT_INCOMPLETE: full wave requires five matched complete sweeps')
    contexts = {(c.sensor, c.server_id, c.phase, c.wave, c.recipe_id, c.recipe_revision) for c in cases}
    if len(contexts) != 1:
        raise ValueError('Full panels cannot mix sensor/server/phase/wave/recipe revisions')
    sensor, server, phase, wave, recipe, revision = next(iter(contexts))
    if phase not in ('BOOT5', 'REFRESH5', 'VERIFY5'):
        raise ValueError('Targeted rechecks/fitting cannot enter balanced complete-panel means')
    if set(observations) != {c.run_id for c in cases}:
        raise ValueError('PILOT_INCOMPLETE: every preregistered observation is required')
    if len({object_sha(_observed_source(r)) for r in observations.values()}) != 1 or len({r['data_sha256'] for r in observations.values()}) != 1:
        raise ValueError('Balanced panel cannot mix numerical/source/data identities')
    sweeps = sorted({c.sweep for c in cases})
    if len(sweeps) != 5:
        raise ValueError('PILOT_INCOMPLETE: five complete sweeps are required')
    if a17_anchors is not None and (not extended or set(a17_anchors)!=set(sweeps)):
        raise ValueError('A17 requires one verified anchor per extended sweep')
    panels, exact_panels, rows = [], [], []
    for sweep in sweeps:
        group = [c for c in cases if c.sweep == sweep]
        students = [c for c in group if c.role == 'S']
        teachers = [c for c in group if c.role == 'T']
        if (len(students) != len(component_cases) or {c.case_id for c in students} != set(component_cases)
                or len(teachers) != 2 or {c.case_id for c in teachers} != {'TPLUS', 'TZERO'}
                or len({c.student_seed for c in students}) != 1 or len({c.teacher_seed for c in teachers}) != 1
                or len({c.paired_init_group for c in students}) != 1):
            raise ValueError('Sweep is missing a component or matched seed/initialization')
        by_ref = {c.reference_id: c for c in teachers}
        core_students=[c for c in students if c.case_id!='C17'] if a17_anchors else students
        for members in (core_students, teachers):
            proofs = [observations[c.run_id] for c in members]
            for key in ('init_U_sha256', 'sampler_sha256', 'rng_roles'):
                if len({object_sha(r[key]) for r in proofs}) != 1:
                    raise ValueError('Matched initialization/sample-view stream proof differs: ' + key)
        if len({observations[c.run_id]['init_A_sha256'] for c in teachers}) != 1:
            raise ValueError('TPLUS/TZERO did not begin from matched A weights')
        if a17_anchors:
            anchor=a17_anchors[sweep]
            c17=observations[next(c.run_id for c in students if c.case_id=='C17')]
            if (anchor.get('case_id')!='C03' or anchor.get('sensor')!=sensor
                    or anchor.get('data_sha256')!=c17['data_sha256']
                    or _observed_source(anchor)!=_observed_source(c17)):
                raise ValueError('PAIRING_NOT_VERIFIED: A17 anchor identity differs')
            if any(anchor[key]!=c17[key] for key in ('init_U_sha256','sampler_sha256','rng_roles')):
                raise ValueError('PAIRING_NOT_VERIFIED: C17 and anchor initial U/native stream differ')
        panel, exact = {}, {}
        for case in students:
            if case.requires_teacher and (case.reference_id not in by_ref or case.teacher_seed != by_ref[case.reference_id].teacher_seed):
                raise ValueError('Student reference is outside its matched Teacher sweep')
            observation = observations[case.run_id]
            if any(observation[k] != value for k, value in (('run_id', case.run_id), ('sensor', sensor), ('case_id', case.case_id))):
                raise ValueError('Observation identity differs from the registered panel')
            panel[case.case_id] = observation['VAL']
            exact[case.case_id] = observation['EXACT50K']
            rows.append(observation)
        panels.append(panel)
        exact_panels.append(exact)
    if thresholds is None:
        if phase != 'BOOT5' or recipe != 'R00':
            raise ValueError('Thresholds may only be initialized from complete R00 BOOT5')
        thresholds = policy.calibrate_thresholds(sensor, panels)
    if thresholds.get('sensor') != sensor:
        raise ValueError('Another sensor threshold revision cannot be borrowed')
    relations = {}
    for relation in (COMPARISON_GRAPH if extended else LEGACY_COMPARISON_GRAPH):
        parent, child = relation['parent'], relation['child']
        pairs = [dict(parent=p[parent], child=p[child]) for p in panels]
        exact_pairs = [dict(parent=p[parent], child=p[child]) for p in exact_panels]
        if relation['relation_id']=='A17' and a17_anchors:
            pairs=[dict(parent=p['C17'],child=a17_anchors[sweep]['VAL']) for p,sweep in zip(panels,sweeps)]
            exact_pairs=[dict(parent=p['C17'],child=a17_anchors[sweep]['EXACT50K']) for p,sweep in zip(exact_panels,sweeps)]
        relations[relation['relation_id']] = policy.classify_relation(pairs, thresholds, exact_pairs)
        if relation['relation_id']=='A17_SCRATCH' and a17_anchors:
            for sweep in sweeps:
                points={r['case_id']:r for r in rows if r.get('sweep',sweep)==sweep and r['case_id'] in ('C02','C17')}
                # Runtime rows always carry sweep; synthetic callers should too
                # when requesting explicit A17 repair anchors.
                if set(points)!= {'C02','C17'} or any(points['C02'][k]!=points['C17'][k] for k in ('init_U_sha256','sampler_sha256','rng_roles')):
                    relations[relation['relation_id']]=dict(classification='INCOMPLETE',reason='PAIRING_NOT_VERIFIED',valid=False,n=0,flags=[])
                    break
        if not policy.queue_enabled(relation):
            relations[relation['relation_id']].update(diagnostic_only=True,queue_enabled=False)
    stats = {}
    for case_id in component_cases:
        values = [panel[case_id] for panel in panels]
        common_keys = set.intersection(*(set(v) for v in values))
        stats[case_id] = {k: dict(mean=mean(v[k] for v in values), sample_sd=stdev(v[k] for v in values),
                                  median=median(v[k] for v in values), per_sweep=[v[k] for v in values]) for k in sorted(common_keys)}
    return dict(schema='ABLR2_BALANCED_PANEL_v2' if extended else 'ABLR2_BALANCED_PANEL_v1', campaign_id=campaign_id(sensor), sensor=sensor, server=server,
                phase=phase, wave=wave, recipe_id=recipe, recipe_revision=revision,
                complete=True, sweep_count=5, teacher_runs=10, student_runs=5*len(component_cases), panelrows=rows,
                coverage='EXTENDED18' if extended else 'LEGACY_CORE17',legacy_core17_complete=True,
                extended18_complete=extended,component_cases=list(component_cases),
                c17_status='COMPLETE' if extended else 'PENDING_NOT_IN_LEGACY_PANEL',
                a17_status='FIVE_VERIFIED_PAIRS' if extended else 'WAITING_FIVE_VERIFIED_PAIRS',
                a17_anchor_rows=list(a17_anchors.values()) if a17_anchors else [],repair_anchors_counted_as_components=False,
                thresholds=thresholds, relations=relations, case_statistics=stats,
                flow=policy.flow_dashboard(relations), source_identity=_observed_source(next(iter(observations.values()))),
                data_sha256=next(iter(observations.values()))['data_sha256'],
                repeats_definition='Teacher5 x Student1; scenes are not independent training seeds',
                development_mode='TEST_AWARE_DEV', independent_test=False,
                statistical_significance_claim=False, all_observations_retained=True)


def analyze_wave(cases, thresholds=None, root=ROOT):
    from ablr2.common import camp
    cases = tuple(cases)
    observations = {case.run_id: case_report(case, root) for case in cases}
    anchors={}
    for c17 in (c for c in cases if c.case_id=='C17'):
        from ablr2.extension import paired_anchor_for
        from ablr2.references import validate_teacher_pair
        anchor_case=paired_anchor_for(c17,root)
        anchor=observations.get(anchor_case.run_id) or case_report(anchor_case,root)
        zero=observations[c17.run_id]
        proof=validate_teacher_pair(anchor['teacher_run_id'],zero['teacher_run_id'],root=root)
        if not proof.get('passed'):raise ValueError('PAIRING_NOT_VERIFIED: Teacher pair validation failed')
        anchors[c17.sweep]=dict(anchor,teacher_pair_proof=proof)
    report = assemble_wave(cases, observations, thresholds, a17_anchors=anchors or None)
    destination = camp(root, report['server']) / 'reports' / report['recipe_revision'] / report['phase'] / report['wave']
    if report['extended18_complete']:destination=destination/'extended18'
    if thresholds is None:
        immutable_json(camp(root, report['server']) / 'thresholds_v1.json', report['thresholds'])
    path = destination / 'complete_panel_metrics.json'
    if not report['extended18_complete'] and path.is_file() and read_json(path)!=report:
        # Preserve original pre-extension reports byte-for-byte. A new explicit
        # parity bridge gives a new verification-derived view, not permission
        # to replace a historical legacy95 observation/report.
        source=report['source_identity'].get('content_sha256','')
        if len(source)!=64 or any(c not in '0123456789abcdef' for c in source):source=object_sha(report['source_identity'])
        destination=destination/'verified_extension'/source
        path=destination/'complete_panel_metrics.json'
    immutable_json(path, report)
    write_panel_csv(destination / 'complete_panel_metrics.csv', report)
    from ablr2.plots import render_wave
    render_wave(report, destination / 'figs')
    result = dict(report, report_path=str(path))
    return result


def refresh_extended_reports(root,server,state):
    """Supplement already finished legacy95 panels after their C17 debts clear.

    No controller cursor, decision, old report, threshold, or run status changes.
    The caller decides whether an A17 result belongs to its current recipe.
    """
    from ablr2.common import camp,read,atomic_json,source_identity
    from ablr2.extension import stage_cases
    from ablr2.plan import verify_lane
    verify_lane(server)
    if state.get('server')!=server:raise ValueError('Cross-lane supplemental analysis state')
    folder=camp(root,server);thresholds=read(folder/'thresholds_v1.json');results={}
    for stage in state.get('stages',[]):
        if stage.get('kind') not in ('BOOT5','REFRESH5','VERIFY5') or not stage.get('complete'):continue
        name=stage['stage_id']
        if Path(name).name!=name or name in ('.','..'):raise ValueError('Unsafe supplemental stage identity')
        status_path=folder/'reports/extended18_status'/(name+'.json')
        try:
            cases=tuple(stage_cases(root,server,stage,state=state))
            done=state.get('runs',{})
            core=all(done.get(c.run_id,{}).get('complete') for c in cases if c.case_id!='C17')
            c17=[c for c in cases if c.case_id=='C17']
            complete=len(cases)==100 and len(c17)==5 and all(done.get(c.run_id,{}).get('complete') for c in cases)
            status=dict(stage_id=name,legacy_core17_complete=core,extended18_complete=False,
                status='WAIT_C17_COVERAGE',completed_c17=sum(bool(done.get(c.run_id,{}).get('complete')) for c in c17),
                controller_state_mutated=False,legacy_report_replaced=False)
            if complete and thresholds:
                identity=dict(cases_sha256=object_sha([asdict(c) for c in cases]),
                    thresholds_sha256=object_sha(thresholds),source_identity=source_identity(root))
                previous=read(status_path)
                if previous.get('status')=='COMPLETE' and previous.get('identity')==identity:
                    path=Path(previous['report_path'])
                    if not path.is_file() or sha256(path)!=previous['report_sha256']:
                        raise ValueError('Supplemental report changed after publication')
                    report=read_json(path)
                else:
                    report=analyze_wave(cases,thresholds=thresholds,root=root)
                    path=Path(report['report_path'])
                if report.get('extended18_complete') is not True or report.get('student_runs')!=90:
                    raise ValueError('Supplemental analysis is not actual complete18 coverage')
                status.update(status='COMPLETE',extended18_complete=True,report_path=str(path),
                    report_sha256=sha256(path),identity=identity)
                atomic_json(status_path,status);results[name]=dict(status,report=report)
                continue
            if complete and not thresholds:status['status']='WAIT_FROZEN_SENSOR_THRESHOLDS'
            atomic_json(status_path,status);results[name]=status
        except (ValueError,OSError,KeyError,TypeError) as exc:
            status=dict(stage_id=name,status='PAIRING_NOT_VERIFIED',extended18_complete=False,
                reason=f'{type(exc).__name__}: {exc}',controller_state_mutated=False,legacy_report_replaced=False)
            atomic_json(status_path,status);results[name]=status
    return results


def write_panel_csv(path, report):
    """Same-checkpoint individual points; display never rounds persisted metrics."""
    rows = []
    for observation in report['panelrows']:
        for selection in ('VAL', 'EXACT50K'):
            row = {k: observation[k] for k in ('run_id', 'case_id', 'sensor', 'server', 'phase', 'wave',
                'sweep', 'recipe_id', 'recipe_revision', 'teacher_seed', 'student_seed', 'reference_id')}
            row.update(selection=selection, **observation[selection])
            row.update(source_sha256=observation['source_identity'].get('content_sha256'),
                       data_sha256=observation['data_sha256'], test_aware=True, independent_test=False)
            rows.append(row)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = payload.encode('utf-8')
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError('An immutable full-panel CSV cannot replace an earlier observation')
        return path
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)  # Publish a complete file without replacing a racing writer.
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise ValueError('An immutable full-panel CSV cannot replace an earlier observation')
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def cumulative_balanced_reports(reports):
    """All completed DEV sweeps of one exact recipe; never add recheck rows.

    VERIFY5 remains separately reported confirmation, not fitting evidence in a
    DEV cumulative panel. Every included sweep contributes one point per case.
    """
    reports = tuple(reports)
    if not reports or any(not r.get('complete') or r.get('phase') not in ('BOOT5', 'REFRESH5') for r in reports):
        raise ValueError('Only completed balanced DEV panels may enter cumulative results')
    identity = {(r['sensor'], r['server'], r['recipe_id'], object_sha(r['source_identity']), r['data_sha256'],
                 tuple(r.get('component_cases',LEGACY_MAIN_CASES))) for r in reports}
    if len(identity) != 1:
        raise ValueError('Cumulative results require one exact recipe/source/data/sensor identity')
    observations = [row for report in reports for row in report['panelrows']]
    if len({row['run_id'] for row in observations}) != len(observations):
        raise ValueError('A resumed/reused run is not another independent observation')
    expected_n = 5 * len(reports)
    stats = {}
    component_cases=reports[0].get('component_cases',LEGACY_MAIN_CASES)
    for case_id in component_cases:
        rows = [row for row in observations if row['case_id'] == case_id]
        if len(rows) != expected_n:
            raise ValueError('Cumulative panel is unbalanced across components')
        keys = set.intersection(*(set(row['VAL']) for row in rows))
        stats[case_id] = {key: dict(mean=mean(row['VAL'][key] for row in rows),
            sample_sd=stdev(row['VAL'][key] for row in rows), median=median(row['VAL'][key] for row in rows),
            values=[row['VAL'][key] for row in rows]) for key in sorted(keys)}
    return dict(schema='ABLR2_CUMULATIVE_BALANCED_DEV_v1', sensor=reports[0]['sensor'], recipe_id=reports[0]['recipe_id'],
                sweeps_per_case=expected_n, completed_panels=len(reports), case_statistics=stats,
                source_identity=reports[0]['source_identity'], data_sha256=reports[0]['data_sha256'],
                all_run_ids=[row['run_id'] for row in observations], rechecks_included=False,
                verification_included=False, independent_test=False, test_aware=True)


def pair_panels(cases, relation_id, root=ROOT):
    """Five targeted pairs with mandatory FULL anchor; never a full-panel mean."""
    cases = tuple(cases)
    if relation_id not in GRAPH or not policy.queue_enabled(GRAPH[relation_id]) or any(c.phase != 'RECHECK5' or c.role != 'S' for c in cases):
        raise ValueError('Registered RECHECK5 Student cases and a graph relation required')
    contexts = {(c.sensor, c.server_id, c.wave, c.recipe_id, c.recipe_revision) for c in cases}
    sweeps = sorted({c.sweep for c in cases})
    if len(contexts) != 1 or len(sweeps) != 5:
        raise ValueError('Targeted panel requires five same-recipe, same-lane pairs')
    relation = GRAPH[relation_id]
    required = {relation['parent'], relation['child'], 'C07'}
    pairs, exact_pairs, anchors, provenance = [], [], [], []
    for sweep in sweeps:
        group = [c for c in cases if c.sweep == sweep]
        if len(group) != len(required) or {c.case_id for c in group} != required or len({c.student_seed for c in group}) != 1:
            raise ValueError('Missing matched recheck side/anchor; no selective cancellation')
        rows = {c.case_id: case_report(c, root) for c in group}
        if relation_id=='A17':
            from ablr2.extension import paired_anchor_for
            c17=next(c for c in group if c.case_id=='C17')
            anchor=paired_anchor_for(c17,root)
            if anchor.run_id!=next(c.run_id for c in group if c.case_id=='C03'):
                rows['C03']=case_report(anchor,root)
        if len({r['data_sha256'] for r in rows.values()}) != 1 or len({object_sha(_observed_source(r)) for r in rows.values()}) != 1:
            raise ValueError('Recheck sides have different numerical/data provenance')
        for key in ('init_U_sha256', 'sampler_sha256', 'rng_roles'):
            if len({object_sha(row[key]) for row in rows.values()}) != 1:
                raise ValueError('Recheck sides do not share actual initialization/sample-view streams: ' + key)
        if relation_id in ('A17','D11'):
            from ablr2.references import validate_teacher_pair
            zero=rows['C17' if relation_id=='A17' else 'C11']
            positive=rows['C03' if relation_id=='A17' else 'C07']
            if not positive.get('teacher_run_id') or not zero.get('teacher_run_id'):
                raise ValueError('PAIRING_NOT_VERIFIED: missing matched Teacher endpoint identity')
            proof=validate_teacher_pair(positive['teacher_run_id'],zero['teacher_run_id'],root=root)
            if not proof.get('passed'):raise ValueError('PAIRING_NOT_VERIFIED: Teacher pair proof failed')
            for row in rows.values():row['teacher_pair_proof']=proof
        pairs.append(dict(parent=rows[relation['parent']]['VAL'], child=rows[relation['child']]['VAL']))
        exact_pairs.append(dict(parent=rows[relation['parent']]['EXACT50K'], child=rows[relation['child']]['EXACT50K']))
        anchors.append(rows['C07'])
        provenance.extend(rows.values())
    if (len({row['data_sha256'] for row in provenance}) != 1
            or len({object_sha(_observed_source(row)) for row in provenance}) != 1):
        raise ValueError('A targeted recheck batch cannot mix source/data revisions across sweeps')
    return dict(pairs=pairs, exact_pairs=exact_pairs, anchors=anchors, relation_id=relation_id,
                scope='TARGETED_RECHECK_NOT_FULL_PANEL', independent_teacher_retraining=False)
