#!/usr/bin/env python3
"""Deterministic PANDA G23 sensitivity cases. This tool does not launch training.

The iterator is lazy and has no campaign time/run limit. A production runner
should consume ONE case, finish/verify it, then consume the next case. Do not
redirect an unlimited stream to disk. Runtime and provenance checks are defined
in IMPLEMENTATION_HANDOFF_KR.md.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

CAMPAIGN = 'PANDA_G23_SENS_WV3_S45_20260923_v1'
SCHEMA = 'PANDA_SENS_CASE_v1'
REFERENCE_COMMIT = '52184d5578a2a10721e435f8fbf4a759497768e5'
TEACHER_SHA256 = '16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32'
SOURCE_CONFIG = 'config/PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S1234_FRESH50_v2.yaml'
Q0 = 0.3276133416220546
TAU0 = 0.012463942170143127
BASE = {
    'alpha': 1.0, 'beta': 0.1, 'lambda_E': 0.002,
    'q_ref_scale': 1.0, 'tau_R_scale': 1.0, 'r_A': 0.03,
}
# Exactly one semantic independent variable differs per variant. A_lr, q_ref,
# and tau_R_used are derived and do not constitute additional experiment axes.
GROUPS = {
    's4': [
        [('AL05', 'alpha', 0.5), ('AL15', 'alpha', 1.5)],
        [('BE005', 'beta', 0.05), ('BE020', 'beta', 0.2)],
        [('ED0006', 'lambda_E', 0.0006), ('ED006', 'lambda_E', 0.006)],
    ],
    's5': [
        [('QR05', 'q_ref_scale', 0.5), ('QR20', 'q_ref_scale', 2.0)],
        [('TR05', 'tau_R_scale', 0.5), ('TR20', 'tau_R_scale', 2.0)],
        [('RA01', 'r_A', 0.01), ('RA06', 'r_A', 0.06)],
    ],
}
FIXED = {
    'dataset': 'WV3', 'train_updates': 50000, 'batch_size': 48,
    'optimizer': 'AdamW', 'weight_decay': 0.01, 'scheduler': 'cosine',
    'warmup_updates': 100, 'U_peak_lr': 0.0001,
    'student_width': 104, 'student_depth': [1, 2, 1],
    'student_input': 'P0: aligned PAN + bicubic-upsampled MS (9 channels)',
    'student_attention_locations': [], 'student_mode_modulation': False,
    'teacher_id': 'T0', 'teacher_checkpoint_step': 24240,
    'teacher_checkpoint_sha256': TEACHER_SHA256,
    'teacher_retraining': False, 'U_init': 'FRESH_SAME_CYCLE_SEED',
    'A_init': 'INDEPENDENT_TRAINABLE_COPY_OF_FROZEN_T0_ALIGNER',
    'input_protocol': 'I-NATIVE-TRANSFER', 'student_offset_weight': 0.0,
    'student_synthetic_shift_radius': 0.0,
    'q_asset': 'assets/qedge9/cue_T0_AXIS16_v1.json',
    'q_source': 'frozen T0 raw q indexed by train sample and actual rotation',
    'q_numerator_factor': 1.0, 'q_weight_mode_A': 'q', 'q_weight_mode_E': 'q',
    'q_weight_mean_renormalization': False, 'uniform_control_weight': 0.5,
    'tau_R_base': TAU0, 'q_ref_base': Q0, 'rec_eps': 1e-6,
    'edge_kind': 'EDGE-H / GT signed Scharr / existing implementation',
    'edge_window_config': 5, 'edge_ramp_updates': 0,
    'U_objective': 'mean(H_i + K_i + lambda_E*w_i*E_i)',
    'A_objective': 'mean(w_i*H_i)',
    'gradient_routing': 'U<-L_U; A<-L_A only; NOT total.backward()',
    'primary_selection': 'EXACT_50000',
    'secondary_selection': 'RR_VAL_ERGAS_MIN_THEN_LOWER_STEP',
    'raw_max_selection_used_for_sensitivity': False,
    'teacher_data_and_q_asset_fixed_across_cycles': True,
    'legacy_feeder_contract': 'crop=False; hflip=True; vflip=True; rot=True; return_meta=True; preserve actual legacy semantics',
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def check_cycle(cycle: int) -> None:
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0:
        raise ValueError('cycle must be a non-negative integer')
    if cycle and 2_000_000 + cycle - 1 >= 2**32:
        raise ValueError('32-bit seed space exhausted; do not silently wrap or reuse seeds')


def seed_for(cycle: int) -> int:
    check_cycle(cycle)
    return 1234 if cycle == 0 else 2_000_000 + cycle - 1


def order_for(server: str, cycle: int) -> list[str]:
    check_cycle(cycle)
    if server not in GROUPS:
        raise ValueError('server must be s4 or s5')
    groups = GROUPS[server]
    shift = cycle % len(groups)
    groups = groups[shift:] + groups[:shift]
    # Three group rotations x alternating low/high order = six-cycle schedule.
    if cycle % 2:
        groups = [list(reversed(g)) for g in groups]
    return ['BASE'] + [item[0] for group in groups for item in group]


def run_id(server: str, cycle: int, case_id: str) -> str:
    return f'SENS_G23_WV3_{server}_C{cycle:06d}_{case_id}_S{seed_for(cycle)}_F50K_v1'


def _variant(server: str, case_id: str) -> tuple[str | None, float | None]:
    if server not in GROUPS:
        raise ValueError('server must be s4 or s5')
    if case_id == 'BASE':
        return None, None
    for group in GROUPS[server]:
        for code, axis, value in group:
            if code == case_id:
                return axis, value
    raise ValueError(f'{case_id!r} is not assigned to {server}')


def config_overrides(p: dict[str, float], seed: int) -> dict[str, Any]:
    """Known computational keys; NOT a full main.py config.

    Crucially, tau is the base value and tau_scale is applied ONCE by train_kdv.
    No new q_ref_scale key is inserted into the strict qrecon registry.
    """
    return {
        'seed': seed, 'num_iter': 50000, 'batch_size': 48,
        'learning_rate': FIXED['U_peak_lr'],
        'kdv.aligner_lr': FIXED['U_peak_lr'] * p['r_A'],
        'kdv.rec.alpha': p['alpha'], 'kdv.rec.kd_weight': p['beta'],
        'kdv.rec.tau': TAU0, 'kdv.rec.tau_scale': p['tau_R_scale'],
        'kdv.stat.outer_weight': p['lambda_E'],
        'kdv.qrecon.q_ref': Q0 * p['q_ref_scale'],
        'kdv.qrecon.a_weight': 'q', 'kdv.qrecon.e_weight': 'q',
        'kdv.qrecon.uniform_weight': 0.5,
    }


def make_case(server: str, cycle: int, case_id: str) -> dict[str, Any]:
    seed = seed_for(cycle)
    axis, value = _variant(server, case_id)
    p = dict(BASE)
    if axis:
        p[axis] = float(value)
    changed = [key for key in BASE if p[key] != BASE[key]]
    if changed != ([] if case_id == 'BASE' else [axis]):
        raise AssertionError('not one-factor-at-a-time')
    qref = Q0 * p['q_ref_scale']
    tau = TAU0 * p['tau_R_scale']
    alr = FIXED['U_peak_lr'] * p['r_A']
    parameter_signature = digest({'fixed': FIXED, 'parameters': p})
    payload = {
        'schema': SCHEMA, 'campaign_id': CAMPAIGN,
        'reference_code_commit': REFERENCE_COMMIT,
        'source_base_config': SOURCE_CONFIG,
        'artifact_type': 'CASE_SPEC_NOT_STANDALONE_TRAINING_CONFIG',
        'server': server, 'dataset': 'WV3', 'cycle': cycle,
        'seed': seed, 'seed_cohort': 'ANCHOR_1234' if cycle == 0 else 'NEW_PAIRED_STUDENT_SEED',
        'case_id': case_id, 'run_id': run_id(server, cycle, case_id),
        'position_in_cycle': order_for(server, cycle).index(case_id),
        'local_baseline_run_id': run_id(server, cycle, 'BASE'),
        'pair_group_id': f'{CAMPAIGN}_{server}_C{cycle:06d}_S{seed}',
        'shared_seed_block_id': f'{CAMPAIGN}_C{cycle:06d}_S{seed}',
        'sweep_axis': axis or 'baseline', 'sweep_value': value,
        'parameters': p,
        'resolved_values': {'q_ref': qref, 'tau_R_used': tau, 'A_peak_lr': alr,
                            'w_at_original_q_median': qref/(qref+Q0)},
        'fixed': copy.deepcopy(FIXED),
        'backend_overrides': config_overrides(p, seed),
        'parameter_signature_sha256': parameter_signature,
        'run_display': ('PANDA G23 | P0 W104D121 | '
                        f'alpha={p["alpha"]:g} beta={p["beta"]:g} edge={p["lambda_E"]:g} | '
                        f'qref*x{p["q_ref_scale"]:g} tau*x{p["tau_R_scale"]:g} rA={p["r_A"]:g} | fresh50K'),
        'runtime_requirements': {
            'new_U_init_snapshot_per_seed': True,
            'same_local_U_init_sha_as_baseline': True,
            'same_local_initial_A_sha_as_baseline': True,
            'same_local_sample_and_augmentation_stream_as_baseline': True,
            'no_parent_student_checkpoint': True,
            'teacher_q_and_data_hashes_must_match_locked_bindings': True,
            'production_adapter_and_preflight_required': True,
            'interserver_dependency': False,
        },
    }
    payload['case_spec_sha256'] = digest(payload)
    return payload


def cycle_cases(server: str, cycle: int) -> list[dict[str, Any]]:
    return [make_case(server, cycle, code) for code in order_for(server, cycle)]


def iter_cases(server: str, start_cycle: int = 0,
               start_position: int = 0) -> Iterator[dict[str, Any]]:
    check_cycle(start_cycle)
    if server not in GROUPS or not 0 <= start_position < 7:
        raise ValueError('invalid server or cursor position')
    for cycle in itertools.count(start_cycle):
        cases = cycle_cases(server, cycle)
        yield from cases[start_position if cycle == start_cycle else 0:]


def validate_case(case: dict[str, Any]) -> None:
    expected = make_case(case['server'], case['cycle'], case['case_id'])
    if case != expected:
        raise ValueError('case differs from immutable v1 specification; create a new campaign revision')


def next_cursor(cycle: int, position: int) -> dict[str, int]:
    check_cycle(cycle)
    if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < 7:
        raise ValueError('position must be an integer in [0, 6]')
    return {'cycle': cycle + (position == 6), 'position': (position + 1) % 7}


def verify_receipt(case: dict[str, Any], receipt: dict[str, Any]) -> None:
    """Structural check for an integration receipt. Does NOT inspect GPU artifacts.

    Artifact hash rechecks and full protocol verification are production runner
    duties. Call this only after such checks; never invent a completion receipt.
    """
    validate_case(case)
    for key in ('campaign_id', 'run_id', 'case_spec_sha256'):
        if receipt.get(key) != case[key]:
            raise ValueError(f'receipt {key} mismatch')
    if receipt.get('training_status') != 'COMPLETE_EXACT_50000' or receipt.get('actual_updates') != 50000:
        raise ValueError('training has not reached exact 50000')
    if receipt.get('evaluation_status') != 'COMPLETE' or receipt.get('primary_selection') != 'EXACT_50000':
        raise ValueError('primary evaluation is incomplete or selection is wrong')
    if receipt.get('protocol_verified') is not True:
        raise ValueError('runtime protocol verification is required')
    h = receipt.get('checkpoint_sha256', '')
    if len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
        raise ValueError('checkpoint_sha256 must be a full lowercase sha256')
    if receipt.get('rr_checkpoint_sha256') != h or receipt.get('fr_checkpoint_sha256') != h:
        raise ValueError('RR and FR checkpoints differ')
    metrics = receipt.get('metrics', {})
    for k in ('ergas', 'hqnr', 'd_lambda', 'd_s'):
        value = metrics.get(k)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'missing or non-finite primary metric: {k}')


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text('utf-8') == text:
            return
        raise FileExistsError(f'refusing to overwrite a different artifact: {path}')
    with path.open('x', encoding='utf-8') as f:
        f.write(text)


def flat(case: dict[str, Any]) -> dict[str, Any]:
    return {
        'server': case['server'], 'cycle': case['cycle'],
        'position': case['position_in_cycle'], 'seed': case['seed'],
        'case_id': case['case_id'], 'run_id': case['run_id'],
        'baseline_run_id': case['local_baseline_run_id'],
        'axis': case['sweep_axis'], 'value': case['sweep_value'],
        **case['parameters'], **case['resolved_values'],
        'updates': 50000, 'primary': 'EXACT_50000',
        'case_spec_sha256': case['case_spec_sha256'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('emit', help='Emit a finite preview; the campaign itself has no count limit')
    p.add_argument('--server', choices=['s4', 's5', 'both'], default='both')
    p.add_argument('--start-cycle', type=int, default=0)
    p.add_argument('--cycles', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    n = sub.add_parser('next', help='Emit one case at a persisted server-local cursor')
    n.add_argument('--server', choices=['s4', 's5'], required=True)
    n.add_argument('--cycle', type=int, required=True)
    n.add_argument('--position', type=int, choices=range(7), required=True)
    v = sub.add_parser('validate', help='Validate one generated case file')
    v.add_argument('path', type=Path)
    a = parser.parse_args()
    if a.command == 'validate':
        validate_case(json.loads(a.path.read_text('utf-8')))
        print('CASE_SPEC_VALID (not a GPU execution check)')
    elif a.command == 'next':
        case = next(iter_cases(a.server, a.cycle, a.position))
        print(json.dumps(case, ensure_ascii=False, indent=2))
    else:
        if a.cycles < 1:
            parser.error('--cycles must be positive')
        check_cycle(a.start_cycle)
        servers = ['s4', 's5'] if a.server == 'both' else [a.server]
        cases = [x for c in range(a.start_cycle, a.start_cycle + a.cycles)
                 for s in servers for x in cycle_cases(s, c)]
        a.out.mkdir(parents=True, exist_ok=True)
        for case in cases:
            write_json(a.out / case['server'] / f"{case['run_id']}.json", case)
        rows = [flat(case) for case in cases]
        csvpath = a.out / 'cases.csv'
        import io
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        text = buffer.getvalue()
        if csvpath.exists() and csvpath.read_text('utf-8-sig').replace('\r\n','\n') != text.replace('\r\n','\n'):
            raise FileExistsError('refusing to overwrite a different cases.csv')
        if not csvpath.exists():
            csvpath.write_text(text, encoding='utf-8-sig', newline='')
        print(json.dumps({'emitted': len(cases), 'servers': servers,
                          'preview_cycles': a.cycles, 'out': str(a.out),
                          'training_launched': False}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, FileExistsError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
