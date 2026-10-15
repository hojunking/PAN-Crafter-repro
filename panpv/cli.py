"""Intake/preparation CLI; deliberately does not auto-start an unverified campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

from .common import atomic_json, canonical_sha, file_sha, load_donor, load_json

DONOR_SHA = '6ad7cff8fc60542ab687a405f7b55430926942f008a65345de99fa69fa8b3846'


def prepare(args):
    from .data import make_split_candidate, prepare_lpan_cache
    receipt = load_json(args.intake)
    root, work = args.input_root.resolve(strict=True), args.work_root.resolve()
    if work == root or work.is_relative_to(root) or root.is_relative_to(work):
        raise ValueError('Input and work roots must be separate, non-nested trees')
    work.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in receipt['scene_allowlist']:
        name = entry['rr_file']
        if Path(name).name != name:
            raise ValueError('Input manifest must use explicit simple basenames')
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing/linked allowlisted RR file: ' + name)
        scene_work = work / 'scenes' / entry['scene_id']
        split_path = scene_work / 'split_candidate.json'
        candidate = make_split_candidate(path, seed=261002, guard_rr=args.guard_rr,
                                         block_rr=args.block_rr)
        if candidate['source']['scene_id'] != entry['scene_id']:
            raise ValueError('Scene ID does not match H5 source mapping')
        if candidate['source']['count'] != entry['rr_rows']:
            raise ValueError('Allowlisted RR row count changed')
        if split_path.exists():
            if load_json(split_path) != candidate:
                raise ValueError('Existing split differs; create an explicit revision')
        else:
            atomic_json(split_path, candidate)
        record = dict(scene_id=entry['scene_id'], category=entry['category'],
                      rr_sha256=candidate['source']['sha256'],
                      split_candidate=str(split_path),
                      split_candidate_sha256=file_sha(split_path),
                      counts=candidate['actual_counts'],
                      excluded=len(candidate['excluded_ids']), split_frozen=False,
                      fr_scope='PROVIDED_NATIVE_FR_TILES' if entry.get('fr_file') else 'NOT_AVAILABLE',
                      full_scene_fr=False)
        if args.make_lpan:
            cache = prepare_lpan_cache(path, scene_work / 'cache')
            lp_path = scene_work / 'lpan_manifest.json'
            if lp_path.exists() and load_json(lp_path) != cache:
                raise ValueError('LP cache identity changed')
            atomic_json(lp_path, cache)
            record['lpan_manifest'] = str(lp_path)
            record['lpan_sha256'] = cache['lpan_sha256']
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        # Keep partial preparation visible if a later scene fails. Not a gate pass.
        atomic_json(work / 'preparation_status.json', dict(
            schema='PANPV_PREPARATION_v1', updated_at=datetime.now(timezone.utc).isoformat(),
            state='PREPARED_PROVISIONAL_SPLITS_NOT_TRAINING_ADMISSION',
            input_root=str(root), intake_sha256=file_sha(args.intake),
            scope_revision=receipt['scope_revision'], scenes=records,
            training_started=False, brovey_evaluated=False, sheet_writes=0,
            blocking=['Exact donor delivery/strict-load check',
                      'RR generation/support/coordinate recipe evidence for split freeze',
                      'Model context and real-data inference/evaluation validation',
                      'End-to-end campaign runner and dedicated report sink integration']))


def check_donor(args):
    model = load_donor(args.weights, DONOR_SHA)
    manifest = dict(schema='PANPV_DONOR_STRICT_LOAD_v1',
                    weights=str(args.weights.resolve()), sha256=file_sha(args.weights),
                    strict_load=True, target_sensor='WV2', donor_state_namespace='WV3',
                    architecture=model.architecture_manifest(),
                    ft0_zs_numerical_parity=False,
                    note='Strict load only; not a training admission')
    if args.output:
        if args.output.exists() and load_json(args.output) != manifest:
            raise ValueError('Refusing to overwrite a different donor receipt')
        atomic_json(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cmd = sub.add_parser('prepare', help='Write provisional spatial splits; optionally contextual LP caches')
    cmd.add_argument('--intake', required=True, type=Path)
    cmd.add_argument('--input-root', required=True, type=Path)
    cmd.add_argument('--work-root', required=True, type=Path)
    cmd.add_argument('--guard-rr', type=int, default=64)
    cmd.add_argument('--block-rr', type=int, default=512)
    cmd.add_argument('--make-lpan', action='store_true')
    cmd.set_defaults(func=prepare)
    cmd = sub.add_parser('check-donor', help='Verify the exact supplied donor SHA and strict architecture load')
    cmd.add_argument('--weights', required=True, type=Path)
    cmd.add_argument('--output', type=Path)
    cmd.set_defaults(func=check_donor)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
