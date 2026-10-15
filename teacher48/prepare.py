"""Offline preparation CLI; never launches training or admits a source/prior.

Run ``python -m teacher48.prepare --help``. JSON destinations must be new. Pixel
limits are explicit safety bounds, not instructions to resize or subsample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ta2.evaluation import file_sha
from .data import SENSORS
from .prepare_assets import (SCHEMA, LMS_RECIPE, NativeScene, Raster, crop_native,
                             exact_gt_candidates)
from .prepare_inventory import inventory
from .prepare_prior import propose_scene_prior


def _new_output(path):
    path = Path(path).expanduser().absolute()
    if os.path.lexists(path):
        raise FileExistsError('Preparation output must not already exist: ' + str(path))
    return path


def _publish_new_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path = _new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication, refusing an intervening writer rather than replace.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _catalog(path):
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    spec = json.loads(payload)
    if spec.get('schema') != SCHEMA or spec.get('dataset') not in SENSORS:
        raise ValueError('Native source catalog schema/dataset mismatch')
    scenes = spec.get('scenes')
    if (not isinstance(scenes, list) or not scenes or
            len({s['source_scene_id'] for s in scenes}) != len(scenes)):
        raise ValueError('Nonempty unique real scene records required')
    if not isinstance(spec.get('split_basis'), str) or not spec['split_basis'].strip():
        raise ValueError('Explicit split_basis required')
    if not isinstance(spec.get('heldout_crops'), list):
        raise ValueError('Explicit heldout_crops list required, even if empty')
    for row in spec['heldout_crops']:
        crop = row['source_crop_native_pan_yxhw']
        if (not isinstance(row.get('acquisition_id'), str) or not row['acquisition_id'].strip()
                or not isinstance(crop, list) or len(crop) != 4
                or any(type(v) is not int or v < 0 for v in crop) or min(crop[2:]) <= 0):
            raise ValueError('Invalid held-out acquisition/crop evidence')
    return path, spec, hashlib.sha256(payload).hexdigest()


def _unchanged_catalog(path, sha):
    if file_sha(path) != sha:
        raise ValueError('Native source catalog changed during preparation')


def _positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Explicit positive pixel limit required')
    return value


def prepare_prior(catalog, scene_id, config_path, max_pixels):
    """Read a whole verified training scene only under an explicit PAN-pixel cap."""
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError('Explicit positive maximum native PAN pixels required')
    path, spec, sha = _catalog(catalog)
    matches = [entry for entry in spec['scenes'] if entry['source_scene_id'] == scene_id]
    if len(matches) != 1:
        raise ValueError('scene-id must identify exactly one catalog native training source')
    entry = matches[0]
    # A scene-global fit consumes the whole scene, unlike heldout-filtered crops.
    if any(row['acquisition_id'] == entry['acquisition_id'] for row in spec['heldout_crops']):
        raise ValueError('Whole-scene prior blocked: acquisition has held-out pixels; region-only prior protocol is not implemented')
    if 'pixel_scale' not in entry or 'origin_phase' not in entry:
        raise ValueError('Prior needs explicit canonical pixel_scale and origin_phase in scene catalog')
    config_path = Path(config_path).resolve(strict=True)
    config_bytes = config_path.read_bytes()
    policy = json.loads(config_bytes)
    with NativeScene(path.parent, entry, spec['dataset']) as scene:
        _, height, width = scene.pan.shape
        if height * width > max_pixels:
            raise ValueError('Native PAN exceeds --max-pixels; no resizing/subsampling allowed')
        arrays = scene.crop([0, 0, height, width])
        provenance = dict(source_catalog_path=str(path), source_catalog_sha256=sha,
            dataset=spec['dataset'], source_scene=entry, split_basis=spec['split_basis'],
            whole_scene_heldout_policy='reject any heldout crop in this acquisition',
            max_native_pan_pixels=max_pixels,
            config_file={'path': str(config_path), 'sha256': hashlib.sha256(config_bytes).hexdigest()},
            lms_mode=scene.lms_mode, native_pan_modified=False, gt_used=False)
        if scene.lms_mode == LMS_RECIPE:
            provenance['lms_recipe_source_sha256'] = file_sha(Path(__file__).resolve().parents[1]/'align/resample.py')
        result = propose_scene_prior(arrays['pan'][0], arrays['lms'],
            source_scene_id=entry['source_scene_id'], prior_grid_name=entry['native_pan_grid'],
            pixel_scale=entry['pixel_scale'], origin_phase=entry['origin_phase'],
            config=policy, provenance=provenance)
        if scene.lms_mode == LMS_RECIPE:
            result['reference_type'] = 'generated_native_ms_interp23tap_bandmean'
            result['reference_warning'] = 'Explicit repository recipe, not claimed identical to unknown dataset-provided LMS generation'
        scene.verify()
    _unchanged_catalog(path, sha)
    if config_path.read_bytes() != config_bytes:
        raise ValueError('Prior config changed during estimation')
    return result


def match_rr(rr_h5, gt_key, row, catalog, max_scene_pixels):
    """Find exact RR-GT/native-MS candidates across every catalog scene.

    ``max_scene_pixels`` counts native MS spatial pixels (not bands). All scenes
    must be checked; size excess fails rather than silently claiming uniqueness.
    Native PAN degradation and RR sampling phase remain unverified.
    """
    if type(max_scene_pixels) is not int or max_scene_pixels <= 0:
        raise ValueError('Explicit positive maximum native MS pixels required')
    if type(row) is not int or row < 0:
        raise ValueError('RR row must be a nonnegative integer')
    path, spec, sha = _catalog(catalog)
    rr = Path(rr_h5).resolve(strict=True)
    rr_sha = file_sha(rr)
    rr_ref = dict(path=str(rr), sha256=rr_sha, key=gt_key, layout='NCHW', index=row)
    scenes, candidates, truncated = [], [], False
    with Raster(rr.parent, rr_ref) as gt_raster:
        if gt_raster.shape[1] * gt_raster.shape[2] > max_scene_pixels:
            raise ValueError('RR GT exceeds explicit pixel limit')
        gt = gt_raster.read(0, 0, *gt_raster.shape[1:])
        for entry in spec['scenes']:
            with NativeScene(path.parent, entry, spec['dataset']) as scene:
                _, height, width = scene.ms.shape
                if height * width > max_scene_pixels:
                    raise ValueError('Native MS exceeds --max-scene-pixels; cannot skip scenes or infer uniqueness')
                source_ms = scene.ms.read(0, 0, height, width)
                found = exact_gt_candidates(gt, source_ms)
                for candidate in found['candidates']:
                    y, x, h, w = candidate['native_ms_crop_yxhw']
                    candidates.append(dict(candidate, source_scene_id=entry['source_scene_id'],
                        acquisition_id=entry['acquisition_id'], native_ms_ref=entry['ms'],
                        candidate_native_pan_crop_yxhw=[scene.origin[0]+4*y, scene.origin[1]+4*x, 4*h, 4*w],
                        candidate_pan_mapping_status='RATIO4_COORDINATE_CANDIDATE_ONLY_DEGRADATION_PHASE_UNVERIFIED'))
                scenes.append(dict(source_scene_id=entry['source_scene_id'], native_ms_ref=entry['ms'],
                                   native_pan_ref=entry['pan'], result=found))
                truncated = truncated or found['truncated']
                scene.verify()
        gt_raster.verify()
    _unchanged_catalog(path, sha)
    if file_sha(rr) != rr_sha:
        raise ValueError('RR source changed during exact matching')
    return dict(schema='TEACHER48_RR_EXACT_MATCH_PROPOSAL_V1', accepted=False,
        status='UNMATCHED' if not candidates else 'AMBIGUOUS' if len(candidates)>1 or truncated else 'EXACT_GT_ONLY_CANDIDATE',
        rr_source=rr_ref, source_catalog={'path': str(path), 'sha256': sha}, dataset=spec['dataset'],
        scenes=scenes, candidates=candidates, truncated=truncated,
        max_native_ms_pixels=max_scene_pixels, all_catalog_scenes_checked=True,
        remaining_verification=['RR PAN degradation and native PAN counterpart',
                                'RR grid pixel scale and sampling origin/phase',
                                'source acquisition identity and heldout/geographic separation',
                                'reviewed admission; matching does not authorize training'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inv = commands.add_parser('inventory', help='Read-only HDF5 metadata inventory')
    inv.add_argument('--root', action='append', required=True)
    inv.add_argument('--output', required=True)
    crop = commands.add_parser('crop-native', help='Stage native FR crops; never admit for training')
    crop.add_argument('--catalog', required=True)
    crop.add_argument('--output', required=True)
    crop.add_argument('--patch', type=int, nargs=2, required=True, metavar=('H','W'))
    crop.add_argument('--stride', type=int, nargs=2, required=True, metavar=('H','W'))
    crop.add_argument('--split-margin', type=int, required=True)
    prior = commands.add_parser('prior', help='Review-only whole-native-scene prior proposal, no GT')
    prior.add_argument('--catalog', required=True)
    prior.add_argument('--scene-id', required=True)
    prior.add_argument('--config', required=True)
    prior.add_argument('--output', required=True)
    prior.add_argument('--max-pixels', type=_positive, required=True, help='maximum whole native PAN spatial pixels')
    match = commands.add_parser('match-rr', help='Exact GT/native-MS coordinate candidates; no admission')
    match.add_argument('--rr-h5', required=True)
    match.add_argument('--gt-key', default='gt')
    match.add_argument('--row', type=int, required=True)
    match.add_argument('--catalog', required=True)
    match.add_argument('--max-scene-pixels', type=_positive, required=True, help='maximum native MS spatial pixels per scene')
    match.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = _new_output(args.output)
    if args.command == 'inventory':
        result = inventory(args.root)
    elif args.command == 'crop-native':
        result = crop_native(args.catalog, output, patch_hw=args.patch, stride_hw=args.stride,
                             margin=args.split_margin)
    elif args.command == 'prior':
        result = prepare_prior(args.catalog, args.scene_id, args.config, args.max_pixels)
    else:
        result = match_rr(args.rr_h5, args.gt_key, args.row, args.catalog, args.max_scene_pixels)
    if args.command != 'crop-native':
        _publish_new_json(output, result)
    print(json.dumps({'command': args.command, 'output': str(output), 'training_admitted': False}))
    return result


if __name__ == '__main__':
    main()
