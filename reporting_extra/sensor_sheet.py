"""Read-only, standard-library Sheet presentation helpers for sensor campaigns.

No selectors, metrics or stored numeric values are rounded or recomputed here.
Dates describe training, never an upload or a date embedded in a run name.
"""
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo


DATE_TIMEZONE = 'Asia/Seoul'
TRAIN_TIME_SCOPE = 'optimizer-only; evaluation/I/O excluded'
_NOTE_MARKER = '[Run summary: '
_SIGNED_DS = {'signed Ds signed_mean', 'signed Ds abs_mean',
              'signed Ds positive_fraction', 'signed Ds reconstruction_max_abs_error'}


def _case_value(case, key):
    return case.get(key) if isinstance(case, Mapping) else getattr(case, key, None)


def _object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _timestamp(value, *, source, utc_field=False):
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise ValueError(f'{source} must be an ISO-8601 timestamp')
    try:
        instant = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(f'Invalid {source} timestamp') from exc
    if instant.tzinfo is None:
        if not utc_field:
            raise ValueError(f'{source} needs an explicit timezone')
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


def _legacy_timestamp(meta, filename):
    path = meta / filename
    return _timestamp(path.read_text().strip(), source=filename) if path.is_file() else None


def _int_value(value, name, *, positive=False):
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer')
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must be an integer') from exc
    if str(number) != str(value) or number < (1 if positive else 0):
        raise ValueError(f'{name} must be a {"positive" if positive else "nonnegative"} integer')
    return number


def _depth_value(value):
    if value is None or value == '':
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('depth must be a nonempty list/tuple of stage depths')
    depths = tuple(_int_value(item, 'depth') for item in value)
    if any(item is None for item in depths):
        raise ValueError('depth cannot contain a missing stage depth')
    return depths


def _agree(case_value, config_value, name, normalize=lambda value: value):
    left, right = normalize(case_value), normalize(config_value)
    if left is not None and right is not None and left != right:
        raise ValueError(f'Case/config {name} mismatch')
    return left if left is not None else right


def _hours(seconds):
    if seconds is None or seconds == '':
        return ''
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError('training_seconds must be a nonnegative finite number')
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('training_seconds must be a nonnegative finite number')
    return seconds / 3600


def metadata_values(wd, cfg, case, training, postrun, selection):
    """Return display metadata from actual, identity-checked run artifacts.

    Completed training uses status.updated_at_utc, except checkpoint-recovery
    statuses whose update time is a recovery event. Legacy timestamps are an
    explicit-offset fallback. A missing end keeps Wall(h) empty, even when an
    official postrun completion is available. Date prefers the training end,
    otherwise the actual start, converted to Asia/Seoul.
    """
    wd = Path(wd)
    meta = wd / 'meta'
    run_id = _case_value(case, 'run_id')
    if run_id and run_id != wd.name:
        raise ValueError('Case and work-directory run identity mismatch')
    run_id = run_id or wd.name
    if cfg.get('work_dir') and Path(cfg['work_dir']).name != run_id:
        raise ValueError('Config and work-directory run identity mismatch')
    campaign = cfg.get('trainer', '')
    if not isinstance(campaign, str) or (campaign and not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', campaign)):
        raise ValueError('Invalid campaign trainer name')
    field = cfg.get(campaign, {})
    if not isinstance(field, Mapping):
        raise ValueError('Campaign config must be a mapping')
    if field.get('run_id') and field['run_id'] != run_id:
        raise ValueError('Campaign config run identity mismatch')
    start = None
    manifest_path = meta / 'training_start_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, Mapping) or manifest.get('run_id') != run_id:
            raise ValueError('Training-start manifest run identity mismatch')
        if manifest.get('config_sha256') != _object_sha(cfg):
            raise ValueError('Training-start manifest config identity mismatch')
        start = _timestamp(manifest.get('started_at_utc'), source='started_at_utc', utc_field=True)
    if start is None:
        start = _legacy_timestamp(meta, 'started_at.txt')
    finish = None
    if training.get('training_complete') is True and not training.get('recovered_from_exact50k'):
        finish = _timestamp(training.get('updated_at_utc'), source='updated_at_utc', utc_field=True)
    if finish is None:
        finish = _legacy_timestamp(meta, 'finished_at.txt')
    if start is not None and finish is not None and finish < start:
        raise ValueError('Training completion precedes the actual start')
    official_finish = _timestamp(postrun.get('completed_at_utc'), source='postrun completed_at_utc', utc_field=True)
    seed = _agree(_case_value(case, 'seed'), cfg.get('seed'), 'seed',
                  lambda value: _int_value(value, 'seed'))
    model_args = cfg.get('model_args', {})
    if not isinstance(model_args, Mapping):
        raise ValueError('Model config must be a mapping')
    width = _agree(_case_value(case, 'width'), model_args.get('hidden_size'), 'width',
                   lambda value: _int_value(value, 'width', positive=True))
    depth = _agree(_case_value(case, 'depth'), model_args.get('depth'), 'depth', _depth_value)
    layout = _agree(_case_value(case, 'input_layout'), field.get('input_layout'), 'input_layout',
                    lambda value: None if value in (None, '') else value)
    if layout is not None and not isinstance(layout, str):
        raise ValueError('Input layout must be a string')
    if not isinstance(selection, str):
        raise ValueError('Selection must be the uploader\'s explicit selection label')
    # Concatenation is conventional for one-digit stages; delimit larger depths
    # to avoid claiming D111 means either (1, 1, 1) or (11, 1).
    depths = (''.join(map(str, depth)) if all(d < 10 for d in depth)
              else '-'.join(map(str, depth))) if depth is not None else ''
    display_date = finish or start
    values = {'Date': display_date.astimezone(ZoneInfo(DATE_TIMEZONE)).date().isoformat() if display_date else '',
              'Seed': seed if seed is not None else '',
              'Model': f'W{width} D{depths}' if width is not None and depth is not None else '',
              'Input': layout or '', 'Selection': selection,
              'Train(h)': _hours(training.get('training_seconds')),
              'Wall(h)': (finish - start).total_seconds() / 3600 if finish and start else '',
              'Train time scope': TRAIN_TIME_SCOPE}
    if campaign:
        prefix = campaign.upper()
        values.update({prefix + ' started UTC': start.isoformat() if start else '',
                       prefix + ' completed UTC': finish.isoformat() if finish else '',
                       prefix + ' official completed UTC': official_finish.isoformat() if official_finish else '',
                       prefix + ' Date timezone': DATE_TIMEZONE})
    return values


def augment_notes(values, existing):
    """Prepend one short owned summary, retaining the complete original note."""
    if existing is None:
        existing = ''
    if not isinstance(existing, str):
        raise ValueError('Notes must be text')
    components = []
    for label, name in (('Model', ''), ('Seed', 'seed='), ('Input', 'input='), ('Selection', 'selection=')):
        value = values.get(label)
        if value is not None and value != '':
            components.append(name + str(value))
    if not components:
        return existing
    prefix = _NOTE_MARKER + '; '.join(components) + '] '
    if existing.startswith(prefix):
        return existing
    return prefix + existing


def merge_notes(incoming, existing):
    """Preserve pre-existing Sheet prose when refreshing an automatic note.

    Substring checks keep repeated uploads idempotent. Nonempty text is retained
    byte-for-byte, including exception receipts and user line breaks. Structural
    controls (including formula cells) remain the uploader's responsibility.
    """
    incoming = '' if incoming is None else incoming
    existing = '' if existing is None else existing
    if not isinstance(incoming, str) or not isinstance(existing, str):
        raise ValueError('Notes must be text')
    if not incoming.strip():
        return existing if existing.strip() else ''
    if not existing.strip() or existing in incoming:
        return incoming
    if incoming in existing:
        return existing
    return incoming + '\n[Previous Sheet note] ' + existing


def _a1(row, column):
    if isinstance(row, bool) or isinstance(column, bool) or not isinstance(row, int) or not isinstance(column, int) or row < 1 or column < 1:
        raise ValueError('Sheet row/columns must be positive integers')
    letters = ''
    while column:
        column, remainder = divmod(column - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters + str(row)


def display_formats(labels, row, metric_names, prefixes):
    """Plan gspread batch_format entries; never change a stored numeric value."""
    metric_names = set(metric_names)
    prefixes = tuple(dict.fromkeys(prefixes))
    cost_patterns = {'Params(M)': '0.0000', 'FLOPs(G)': '0.0',
                     'Infer(ms)': '0.00', 'Mem(MB)': '0.0',
                     'Train(h)': '0.00', 'Wall(h)': '0.00'}
    entries = {}
    for label, column in labels.items():
        suffixes = [label[len(prefix) + 1:] for prefix in prefixes if label.startswith(prefix + ' ')]
        if label in metric_names or any(suffix in metric_names or suffix in _SIGNED_DS for suffix in suffixes):
            number_format = {'type': 'NUMBER', 'pattern': '0.0000'}
        elif label in cost_patterns:
            number_format = {'type': 'NUMBER', 'pattern': cost_patterns[label]}
        elif label.lower() in ('seed', 'step') or label.lower().endswith((' seed', ' step')):
            number_format = {'type': 'NUMBER', 'pattern': '0'}
        elif label == 'Date' or label.endswith((' started UTC', ' completed UTC', 'Date timezone')):
            number_format = {'type': 'TEXT', 'pattern': '@'}
        else:
            continue
        cell = _a1(row, column)
        previous = entries.get(cell)
        entry = {'range': cell, 'format': {'numberFormat': number_format}}
        if previous is not None and previous != entry:
            raise ValueError('Conflicting number formats for the same Sheet cell')
        entries[cell] = entry
    return list(entries.values())
