"""Observed continuation timing without charging the inherited 25K prefix."""
import hashlib
import math
from pathlib import Path
import json

from experiment_cost.timing import timing_for_config as original_timing


def timing_for_config(config, selected_step, root, history_roots=None, cost_root=None):
    result = original_timing(config, selected_step, root, history_roots, cost_root)
    inherited = config.get('inherited_updates', 0)
    if not inherited:
        return result
    path = (Path(root) / 'runs' / config['run_id'] /
            ('attempt_%02d' % config['attempt']) / 'train.jsonl')
    try:
        if not inherited < selected_step <= config['planned_horizon']:
            raise ValueError('Selected continuation step outside its new-update interval')
        elapsed, count, chain = [], inherited, hashlib.sha256(b'').hexdigest()
        with path.open('rb') as stream:
            for line in stream:
                if count >= selected_step:
                    break
                row = json.loads(line)
                expected = count + 1
                seconds = row['seconds_including_IO']
                if (not line.endswith(b'\n') or type(row.get('step')) is not int
                        or row['step'] != expected or row.get('update', expected) != expected
                        or type(seconds) not in (int, float)
                        or not math.isfinite(seconds) or seconds < 0):
                    raise ValueError('Noncontiguous or invalid continuation trace')
                elapsed.append(seconds)
                chain = hashlib.sha256((chain + hashlib.sha256(line).hexdigest()).encode()).hexdigest()
                count = expected
        if count != selected_step:
            raise ValueError('Continuation trace has not reached selected step')
        result['sheet_fields']['Train(h)'] = math.fsum(elapsed) / 3600
        result['omitted'].pop('Train(h)', None)
        result['proof']['Train(h)'] = {
            'path': str(path), 'selected_step': selected_step,
            'first_new_update': inherited + 1, 'inherited_updates_excluded': inherited,
            'contiguous_updates': selected_step - inherited,
            'selected_prefix_row_hash_chain': chain,
            'scope': 'sum observed new-update IO and optimizer durations; inherited training excluded',
        }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        result['sheet_fields'].pop('Train(h)', None)
        result['proof'].pop('Train(h)', None)
        result['omitted']['Train(h)'] = str(exc)
    return result
