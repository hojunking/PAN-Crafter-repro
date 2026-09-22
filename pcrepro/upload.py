"""Dedicated PC-Repro rows: exact raw values, durable spool, verified upsert.

Importing this module neither reads credentials nor accesses a spreadsheet.
API writes are possible only through an explicitly activated upload operation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from pcrepro.common import (ROOT, atomic_json, camp, object_sha, read, read_json,
                            run_dir, sha256, utcnow)
from pcrepro.plan import CAMPAIGN_ID, RECIPE_ID, case_for

SCHEMA = 'PCREPRO_SHEET_v1'
SHEET_TABS = {s: 'PC-Repro-' + s for s in ('s3', 's4', 's5')}
SELECTIONS = ('EXACT_50000', 'RR_VAL_ERGAS_MIN')
RR_METRICS = ('ergas', 'sam', 'psnr', 'ssim', 'scc', 'q4', 'q8', 'rmse', 'cc')
FR_METRICS = ('hqnr', 'd_lambda', 'd_s', 'jqm')
COST_KEYS = ('training_seconds', 'validation_seconds', 'test_seconds', 'profile_seconds',
             'preprocessing_seconds', 'io_seconds', 'wall_seconds')
BASE_COLUMNS = ('Run', 'schema', 'campaign_id', 'run_id', 'recipe_id', 'dataset', 'train_dataset',
    'stage', 'server', 'cycle', 'seed', 'seed_cohort', 'Date', 'started_at_utc', 'started_at_kst',
    'completed_at_utc', 'completed_at_kst', 'actual_updates', 'status', 'reason', 'eval_complete',
    'paper_identity_status',
    'source_commit', 'source_content_sha256', 'data_sha256', 'config_sha256',
    'checkpoint_sha256', 'selection', 'source_run_id', 'source_checkpoint_sha256',
    'summary_sha256', 'recipe_manifest_sha256', 'evaluation_manifest_sha256',
    'params_total', 'params_trainable', 'params_m', 'profile_details', 'actual_gpu_runtime',
    *COST_KEYS, 'source_wv3_training_seconds', 'new_training_cost_recharged',
    'source_lineage_cost_not_recharged', 'selections_alias_same_checkpoint', 'readback_status', 'notes')
SELECTION_COLUMNS = ('selection', 'update', 'checkpoint_sha256', 'source_checkpoint_sha256',
    'evaluation_manifest_sha256', 'alias_of', 'val_ergas', 'rr_n_scenes', 'fr_n_scenes',
    'rr_paper_identity_status', 'fr_paper_identity_status',
    *('rr_' + k for k in RR_METRICS), *('fr_' + k for k in FR_METRICS),
    'jqm_variant', 'jqm_status', 'jqm_reason', 'signed_ds_signed_mean',
    'signed_ds_positive_fraction', 'signed_ds_reconstruction_max_abs_error')
HEADER = BASE_COLUMNS + tuple(label + ' ' + key for label in SELECTIONS for key in SELECTION_COLUMNS)


def _finite(value, name, *, optional=False, nonnegative=False):
    if optional and value in (None, ''): return ''
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Nonfinite/non-numeric ' + name)
    if nonnegative and value < 0: raise ValueError('Negative ' + name)
    return value


def _stamp(value):
    if not value: return '', ''
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None: raise ValueError('Actual timestamp requires explicit timezone')
    return parsed.astimezone(timezone.utc).isoformat(), parsed.astimezone(ZoneInfo('Asia/Seoul')).isoformat()


def _sha(value, label, *, optional=False):
    if optional and value in (None, ''): return ''
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid ' + label + ' SHA256')
    return value


def _checkpoint(record):
    # canonical spelling; checkpoint_sha remains accepted for the documented
    # JSON contract used during controller/trainer integration.
    a, b = record.get('checkpoint_sha256'), record.get('checkpoint_sha')
    if a and b and a != b: raise ValueError('Two checkpoint SHA fields disagree')
    return a or b or record.get('checkpoint_identity', {}).get('model_sha256')


def _costs(summary):
    source = dict(summary.get('costs', {}))
    aliases = {'training_seconds': 'train_seconds', 'test_seconds': 'evaluation_seconds'}
    result = {}
    for key in COST_KEYS:
        value = source.get(key, source.get(aliases.get(key, ''), summary.get(key, '')))
        result[key] = _finite(value, key, optional=True, nonnegative=True)
    return result


def validate_summary(summary, case):
    if (summary.get('campaign_id') != CAMPAIGN_ID or summary.get('recipe_id') != RECIPE_ID
            or summary.get('run_id') != case.run_id):
        raise ValueError('Summary belongs to another campaign/run/recipe')
    if summary.get('case') and summary['case'] != case.to_dict():
        raise ValueError('Summary case differs from deterministic cycle registry')
    actual = summary.get('actual_updates', 0)
    if isinstance(actual, bool) or not isinstance(actual, int) or not 0 <= actual <= 50000:
        raise ValueError('Invalid actual update count')
    zero_shot = case.dataset == 'WV2'
    costs = _costs(summary)
    if zero_shot:
        if actual != 0 or summary.get('source_run_id', case.source_run_id) != case.source_run_id:
            raise ValueError('WV2 must use same-server/cycle WV3 and zero optimizer updates')
        if costs['training_seconds'] not in ('', 0):
            raise ValueError('WV2 source training cost must not be charged again')
    if not summary.get('complete'):
        return summary
    if actual != (0 if zero_shot else 50000):
        raise ValueError('Complete fresh training requires exactly50000 updates')
    data_sha = summary.get('data_sha256', summary.get('data_sha'))
    _sha(data_sha, 'data')
    selections = summary.get('selections', {})
    if set(selections) != set(SELECTIONS):
        raise ValueError('Exactly primary and validation-min selections are required')
    for label, record in selections.items():
        if not isinstance(record.get('update'), int) or record['update'] not in range(1000, 50001, 1000):
            raise ValueError('Selection is outside the fixed validation1000 grid')
        if label == 'EXACT_50000' and record['update'] != 50000:
            raise ValueError('Primary selection is not exact50000')
        checkpoint = _sha(_checkpoint(record), label + ' checkpoint')
        if record.get('data_sha256', data_sha) != data_sha:
            raise ValueError('Selection metric data identity differs')
        if zero_shot and record.get('source_checkpoint_sha256', checkpoint) != checkpoint:
            raise ValueError('WV2 source checkpoint was modified/substituted')
        rr, fr = record['rr'], record['fr']
        qkey = 'q8' if case.dataset in ('WV3', 'WV2') else 'q4'
        other = 'q4' if qkey == 'q8' else 'q8'
        if rr.get(other) not in (None, ''):
            raise ValueError('Wrong-band Q metric must not be presented as applicable')
        for key in ('ergas', 'sam', 'psnr', 'ssim', 'scc', qkey, 'rmse', 'cc'):
            _finite(rr.get(key), label + '.rr.' + key)
        for key in ('hqnr', 'd_lambda', 'd_s'):
            _finite(fr.get(key), label + '.fr.' + key)
        _finite(fr.get('jqm'), label + '.fr.jqm', optional=True)
        if fr.get('jqm') is not None and fr.get('jqm') != '':
            variant = fr.get('jqm_variant', '')
            if not isinstance(variant, str) or 'srf' not in variant.lower() or 'substitut' not in variant.lower():
                raise ValueError('JQM must explicitly identify its SRF-substitute variant')
        if any(isinstance(value.get('n_scenes'),bool) or not isinstance(value.get('n_scenes'),int)
               or value['n_scenes']<=0 for value in (rr,fr)):
            raise ValueError('Complete evaluation requires actual RR/FR manifest scene counts')
    first, second = [selections[label] for label in SELECTIONS]
    if _checkpoint(first) == _checkpoint(second) and any(first[k] != second[k] for k in ('rr', 'fr')):
        raise ValueError('Same checkpoint aliases have different evaluation metrics')
    return summary


def flatten_summary(summary, case, root=ROOT):
    """One row per run; secondary columns never become a separate experiment."""
    validate_summary(summary, case)
    values = {key: '' for key in HEADER}
    started = summary.get('started_at_utc', summary.get('start_time', ''))
    completed = summary.get('completed_at_utc', summary.get('end_time', ''))
    su, sk = _stamp(started); eu, ek = _stamp(completed)
    if su and eu and datetime.fromisoformat(eu) < datetime.fromisoformat(su):
        raise ValueError('Completion precedes actual start')
    source = summary.get('source_identity', {})
    profile = summary.get('profile', {})
    bands = 8 if case.dataset in ('WV3', 'WV2') else 4
    zero_shot = case.dataset == 'WV2'
    values.update(Run=f'PAN-Crafter | C128 D224 LN | CM3A3 k3 | MARs lambda1 | PAN+MS in{bands+1} | 50K',
        schema=SCHEMA, campaign_id=CAMPAIGN_ID, run_id=case.run_id, recipe_id=RECIPE_ID,
        dataset=case.dataset, train_dataset=case.train_dataset, stage=case.stage, server=case.server,
        cycle=case.cycle, seed=case.seed,
        seed_cohort='PAPER_SEED2025_SERVER_REPEAT' if case.cycle == 0 else 'PREDECLARED_NEW_SEED_REPEAT',
        Date=(ek or sk)[:10], started_at_utc=su, started_at_kst=sk,
        completed_at_utc=eu, completed_at_kst=ek,
        actual_updates=summary.get('actual_updates', 0), status=summary.get('status', 'UNKNOWN'),
        reason=summary.get('reason', ''), eval_complete=bool(summary.get('complete')),
        paper_identity_status=summary.get('paper_identity_status', ''),
        source_commit=source.get('git_release', source.get('commit', summary.get('source_commit', ''))),
        source_content_sha256=source.get('content_sha256', ''),
        data_sha256=summary.get('data_sha256', summary.get('data_sha', '')),
        config_sha256=summary.get('config_sha256', ''),
        source_run_id=case.source_run_id or '',
        summary_sha256=object_sha(summary), recipe_manifest_sha256=summary.get('recipe_manifest_sha256', ''),
        evaluation_manifest_sha256=summary.get('evaluation_manifest_sha256', ''),
        params_total=profile.get('params_total', profile.get('total_params', '')),
        params_trainable=profile.get('params_trainable', profile.get('trainable_params', '')),
        params_m=profile.get('params_m', ''), profile_details=profile,
        actual_gpu_runtime=summary.get('runtime', profile.get('runtime', '')),
        source_wv3_training_seconds=summary.get('source_wv3_training_seconds', ''),
        new_training_cost_recharged=False if zero_shot else '',
        source_lineage_cost_not_recharged=zero_shot,
        readback_status='READBACK_PENDING',
        notes='Primary EXACT_50000; secondary RR_VAL_ERGAS_MIN; one experiment per run_id. '
              'Native full-manifest RR/FR; paper identity and actual scene counts recorded separately. '
              'No masking or HQNR(V64). JQM is auxiliary SRF-substitute. '
              'Seed2025 server repeats are not three independent seeds. WV2 uses same-cycle WV3, no training.')
    values.update(_costs(summary))
    if zero_shot: values['training_seconds'] = 0
    if summary.get('complete'):
        selections = summary['selections']
        values['selection'] = 'EXACT_50000'
        values['checkpoint_sha256'] = _checkpoint(selections['EXACT_50000'])
        values['source_checkpoint_sha256'] = values['checkpoint_sha256'] if zero_shot else ''
        values['selections_alias_same_checkpoint'] = _checkpoint(selections[SELECTIONS[0]]) == _checkpoint(selections[SELECTIONS[1]])
        for label, record in selections.items():
            prefix = label + ' '
            values[prefix + 'selection'] = label
            values[prefix + 'update'] = record['update']
            values[prefix + 'checkpoint_sha256'] = _checkpoint(record)
            values[prefix + 'source_checkpoint_sha256'] = _checkpoint(record) if zero_shot else ''
            values[prefix + 'evaluation_manifest_sha256'] = record.get('evaluation_manifest_sha256', summary.get('evaluation_manifest_sha256', ''))
            values[prefix + 'alias_of'] = record.get('alias_of', '')
            values[prefix + 'val_ergas'] = record.get('val_ergas', '')
            for domain, keys in (('rr', RR_METRICS), ('fr', FR_METRICS)):
                values[prefix + domain + '_n_scenes'] = record[domain]['n_scenes']
                values[prefix + domain + '_paper_identity_status'] = record[domain].get(
                    'paper_identity_status',summary.get('paper_identity_status',''))
                for key in keys: values[prefix + domain + '_' + key] = record[domain].get(key, '')
            for key in ('jqm_variant', 'jqm_status', 'jqm_reason'):
                values[prefix + key] = record['fr'].get(key, '')
            for key in ('signed_mean', 'positive_fraction', 'reconstruction_max_abs_error'):
                values[prefix + 'signed_ds_' + key] = record['fr'].get('signed_ds', {}).get(key, '')
    # Complete=false deliberately suppresses all metric columns, including any
    # partial records left in a stopped evaluator's cursor file.
    return {key: json.dumps(val, sort_keys=True, ensure_ascii=False, allow_nan=False)
            if isinstance(val, (dict, list, tuple)) else '' if val is None else val
            for key, val in values.items()}


def status_summary(run, root=ROOT):
    case = case_for(run); wd = run_dir(case, root)
    summary = read(wd / 'official/summary.json')
    if summary: return summary
    training = read(wd / 'meta/training_status.json')
    terminal = read(wd / 'meta/status.json')
    status = dict(training, **terminal)
    status['costs'] = dict(training.get('costs', {}), **terminal.get('costs', {}))
    state = read(camp(root, case.server) / 'status.json').get('runs', {}).get(case.run_id, {})
    start = read(wd / 'meta/training_start_manifest.json')
    if not training and not terminal: raise ValueError('Persisted per-run status required; no invented failure/NOT_STARTED row')
    from pcrepro.common import read_config
    from pcrepro.plan import validate_config
    cfg = read_config(wd/'meta/config.resolved.yaml') if (wd/'meta/config.resolved.yaml').is_file() else None
    if cfg is not None and validate_config(cfg).run_id != case.run_id:
        raise ValueError('Failure row config identity differs')
    for key, expected in dict(campaign_id=CAMPAIGN_ID, recipe_id=RECIPE_ID, run_id=case.run_id).items():
        for document in (training, terminal):
            if document and document.get(key, expected if cfg is not None else None) != expected:
                raise ValueError('Failure status identity differs: '+key)
    bound = cfg.get('pcrepro', {}) if cfg else {}
    source = status.get('source_identity', start.get('source_identity', bound.get('source_identity', {})))
    if not source or not source.get('content_sha256'):
        raise ValueError('Failure status must retain its actual execution source identity')
    if source.get('files') is not None and object_sha(source['files']) != source['content_sha256']:
        raise ValueError('Failure status source content digest differs')
    if bound.get('source_identity') and source != bound['source_identity']:
        raise ValueError('Failure status/config execution source differs')
    return dict(campaign_id=CAMPAIGN_ID, recipe_id=RECIPE_ID, run_id=case.run_id,
        case=case.to_dict(), complete=False, status=status.get('status', state.get('status', 'NOT_STARTED')),
        reason=status.get('reason', state.get('reason', '')),
        actual_updates=status.get('actual_updates', state.get('actual_updates', 0)),
        started_at_utc=start.get('started_at_utc', status.get('started_at_utc', '')),
        completed_at_utc=status.get('completed_at_utc', ''),
        source_identity=source,
        config_sha256=object_sha(cfg) if cfg else '',
        data_sha256=status.get('data_sha256', bound.get('data_sha', '')), costs=status.get('costs', {}),
        source_run_id=case.source_run_id or '')


def row_values(run, root=ROOT):
    case = case_for(run); summary = status_summary(run, root)
    if summary.get('complete'):
        # The evaluator owns checkpoint byte/selection-grid/data certification;
        # upload must never silently bypass it or launch an evaluation itself.
        from pcrepro.postrun import verify_summary_for_upload
        summary = verify_summary_for_upload(case.run_id, root)
    return flatten_summary(summary, case, root)


def _journal(root, server, event):
    path = camp(root, server) / 'reporting/events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


def _verify_envelope(envelope, case):
    if (envelope.get('schema') != 'PCREPRO_UPLOAD_SPOOL_v1'
            or envelope.get('campaign_id') != CAMPAIGN_ID or envelope.get('run_id') != case.run_id
            or envelope.get('server') != case.server
            or envelope.get('payload',{}).get('run_id') != case.run_id
            or envelope.get('payload',{}).get('server') != case.server
            or envelope.get('payload_sha256') != object_sha(envelope.get('payload'))):
        raise ValueError('Upload spool identity/hash mismatch')
    if envelope.get('status') == 'READBACK_VERIFIED':
        receipt = envelope.get('receipt', {})
        if (receipt.get('readback_verified') is not True
                or receipt.get('payload_sha256') != envelope['payload_sha256']
                or receipt.get('run_id') != case.run_id
                or receipt.get('worksheet') != SHEET_TABS[case.server]):
            raise ValueError('Verified spool lacks its matching raw-value readback receipt')


def spool_run(run, root=ROOT):
    case = case_for(run); payload = row_values(case.run_id, root)
    path = camp(root, case.server) / 'upload_spool' / (case.run_id + '.json')
    digest = object_sha(payload); old = read(path)
    if old: _verify_envelope(old, case)
    if old.get('payload_sha256') == digest: return old
    value = dict(schema='PCREPRO_UPLOAD_SPOOL_v1', campaign_id=CAMPAIGN_ID, server=case.server,
        run_id=case.run_id, payload=payload, payload_sha256=digest, status='PENDING', attempts=0,
        first_spooled_at_utc=old.get('first_spooled_at_utc', utcnow()), updated_at_utc=utcnow())
    atomic_json(path, value)
    _journal(root, case.server, dict(event='SPOOLED', run_id=case.run_id,
        payload_sha256=digest, at_utc=utcnow()))
    return value


def _a1(row, col):
    text = ''
    while col: col, rem = divmod(col-1, 26); text = chr(65+rem) + text
    return text + str(row)


def _same(actual, expected):
    if expected == '': return actual in (None, '')
    if isinstance(expected, bool): return str(actual).lower() == str(expected).lower()
    if isinstance(expected, (int, float)):
        try: return math.isfinite(float(actual)) and abs(float(actual)-expected) <= 1e-12*max(1., abs(expected))
        except (ValueError, TypeError): return False
    return str(actual) == str(expected)


def apply_upsert(worksheet, values):
    if (worksheet.title != SHEET_TABS.get(values.get('server'))
            or values.get('campaign_id') != CAMPAIGN_ID or values.get('schema') != SCHEMA
            or tuple(values) != HEADER):
        raise ValueError('Only the exact-schema dedicated PC-Repro server tab may be written')
    current = worksheet.row_values(1)
    if current and current != list(HEADER):
        raise ValueError('Existing tab schema differs; no header overwrite or migration')
    if worksheet.col_count < len(HEADER): worksheet.add_cols(len(HEADER)-worksheet.col_count)
    if not current: worksheet.update(range_name='A1', values=[list(HEADER)], value_input_option='RAW')
    rows = worksheet.get_all_values(value_render_option='UNFORMATTED_VALUE')
    rid = HEADER.index('run_id'); cid = HEADER.index('campaign_id')
    matches = [i for i,row in enumerate(rows[1:], 2) if len(row)>rid and row[rid] == values['run_id']]
    if len(matches)>1: raise ValueError('Duplicate run_id rows; cannot choose an arbitrary result')
    number = matches[0] if matches else max(2, len(rows)+1)
    if matches and (len(rows[number-1]) <= cid or rows[number-1][cid] != CAMPAIGN_ID):
        raise ValueError('Existing run row belongs to another campaign')
    if worksheet.row_count < number: worksheet.add_rows(number-worksheet.row_count)
    worksheet.update(range_name=_a1(number, 1), values=[[values[key] for key in HEADER]], value_input_option='RAW')
    decimal = {label+' '+domain+'_'+key for label in SELECTIONS
        for domain,keys in (('rr',RR_METRICS),('fr',FR_METRICS)) for key in keys}
    decimal |= {label+' val_ergas' for label in SELECTIONS}
    decimal |= {key for key in HEADER if 'signed_ds_' in key or key.endswith('_seconds') or key == 'params_m'}
    worksheet.batch_format([dict(range=_a1(number,i), format={'numberFormat':{'type':'NUMBER','pattern':'0.0000'}})
                            for i,key in enumerate(HEADER,1) if key in decimal])
    actual = worksheet.row_values(number, value_render_option='UNFORMATTED_VALUE')
    if any(not _same(actual[i] if i<len(actual) else '', values[key]) for i,key in enumerate(HEADER)):
        raise ValueError('Unformatted raw-value readback mismatch')
    status_col = HEADER.index('readback_status')+1
    worksheet.update(range_name=_a1(number,status_col), values=[['READBACK_VERIFIED']], value_input_option='RAW')
    if worksheet.row_values(number,value_render_option='UNFORMATTED_VALUE')[status_col-1] != 'READBACK_VERIFIED':
        raise ValueError('Upload status readback mismatch')
    return dict(worksheet=worksheet.title, gid=int(worksheet.id), row=number,
                readback_verified=True, payload_sha256=object_sha(values))


def _worksheet(root, server, columns):
    # Import and credential access happen only after activated=True and durable
    # spool publication in upload_run, never during construction/test/reporting.
    import gspread
    from fh12.upload import legacy_constants
    constants = legacy_constants()
    client = gspread.service_account(filename=str(Path(root)/'gspread'/Path(constants.CRED).name))
    client.http_client.set_timeout(20)
    book = client.open(constants.SHEET)
    try: return book.worksheet(SHEET_TABS[server])
    except gspread.WorksheetNotFound:
        return book.add_worksheet(title=SHEET_TABS[server], rows=100, cols=columns)


def _deliver(case, envelope, root, worksheet=None):
    path = camp(root, case.server)/'upload_spool'/(case.run_id+'.json')
    _verify_envelope(envelope, case)
    envelope = dict(envelope, attempts=envelope.get('attempts',0)+1, updated_at_utc=utcnow())
    atomic_json(path, envelope)  # persist attempt before credential/API access
    try:
        ws = worksheet if worksheet is not None else _worksheet(root,case.server,len(HEADER))
        from reporting_extra.sensor_backfill import write_locks
        with write_locks(root): receipt = apply_upsert(ws,envelope['payload'])
        receipt.update(campaign_id=CAMPAIGN_ID,run_id=case.run_id,uploaded_at_utc=utcnow())
        atomic_json(run_dir(case,root)/'official/upload_receipt.json',receipt)
        envelope.update(status='READBACK_VERIFIED',receipt=receipt)
    except Exception as exc:
        envelope.update(status='PENDING',last_error=f'{type(exc).__name__}: {exc}')
        receipt=dict(run_id=case.run_id,status='UPLOAD_PENDING',reason=envelope['last_error'],readback_verified=False)
    atomic_json(path,envelope)
    atomic_json(run_dir(case,root)/'official/upload_status.json',receipt)
    _journal(root,case.server,dict(event=envelope['status'],run_id=case.run_id,
        payload_sha256=envelope['payload_sha256'],attempt=envelope['attempts'],at_utc=utcnow()))
    return receipt


def upload_run(run, root=ROOT, *, activated=False, worksheet=None):
    if not activated: raise PermissionError('Explicit PCREPRO activation required for live Sheet writes')
    case=case_for(run); envelope=spool_run(case.run_id,root)
    if envelope.get('status')=='READBACK_VERIFIED': return envelope['receipt']
    return _deliver(case,envelope,root,worksheet)


def retry_pending(root, server, *, limit=8, activated=False, worksheet=None):
    if not activated: raise PermissionError('Explicit PCREPRO activation required for upload retries')
    if server not in SHEET_TABS or isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=100:
        raise ValueError('Valid local server and bounded retry batch1..100 required')
    candidates=[];result={}
    for path in (camp(root,server)/'upload_spool').glob('*.json'):
        try:
            value=read_json(path);case=case_for(value['run_id'])
            if case.server!=server or path.stem!=case.run_id:raise ValueError('Foreign server/run upload spool')
            _verify_envelope(value,case)
        except (ValueError,KeyError,OSError) as exc:
            result[path.stem]=dict(status='UPLOAD_BLOCKED_INTEGRITY',readback_verified=False,
                reason=f'{type(exc).__name__}: {exc}');continue
        if value.get('status')!='READBACK_VERIFIED': candidates.append((value.get('updated_at_utc',''),path,value))
    for _,path,envelope in sorted(candidates,key=lambda v:(v[0],str(v[1])))[:limit]:
        try:
            case=case_for(envelope['run_id'])
            if case.server!=server or path.stem!=case.run_id: raise ValueError('Foreign server/run upload spool')
            # A newer completed report supersedes an older failure row, without
            # rerunning training/evaluation. No permanent attempt ceiling.
            current=spool_run(case.run_id,root)
            result[case.run_id]=_deliver(case,current,root,worksheet)
        except (ValueError,KeyError,OSError,RuntimeError) as exc:
            result[path.stem]=dict(status='UPLOAD_BLOCKED_INTEGRITY',readback_verified=False,
                reason=f'{type(exc).__name__}: {exc}')
    atomic_json(camp(root,server)/'reporting/upload_retry_status.json',
                dict(at_utc=utcnow(),maximum_api_attempts=limit,results=result))
    return result


retry_uploads = retry_pending
