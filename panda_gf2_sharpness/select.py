"""Select GF2 qualitative candidates from saved exports on CPU, without training."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from PIL import Image, ImageDraw, ImageFont
import PIL

from panda_sensor_visuals.common import ROOT, read, sha, write
from panda_sensor_visuals.metrics import gradient, local_std
from .selection_metrics import classify, measure_saved, shared_rgb

QUAL_RUN = 'PSFC_GF2_S3_GF2_C60LR150_NOSOFT_R1_S261018101'
QUANT_RUN = 'PSFC_GF2_S3_GF2P_C60LR200_NOSOFT_R2_S261018102'
RELEASE = ROOT / 'config/panda_gf2_sharpness/selection_release.json'
REFERENCE = ROOT / 'config/panda_gf2_sharpness/reference_metrics.json'


def verify_release():
    value = read(RELEASE)
    core = {k:v for k,v in value.items() if k != 'source_sha256'}
    digest = hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if value.get('source_sha256') != digest:
        raise ValueError('Selection release manifest digest differs')
    for path, digest in value['files'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Selection source differs; pull complete commit: ' + path)
    return value


def choose_baseline(candidates, explicit=None):
    if explicit:
        requested = inspect(Path(explicit).resolve())
        key = (requested['manifest']['run_id'], requested['manifest']['step'])
        choices = [(key, 'EXPLICIT_REFERENCE')]
    else:
        choices = [((QUAL_RUN,47000), 'QUALITATIVE_REFERENCE'),
                   ((QUANT_RUN,47000), 'QUANTITATIVE_REFERENCE_FALLBACK_QUALITATIVE_EXPORT_MISSING')]
    for key, role in choices:
        found = next((x for x in candidates if (x['manifest']['run_id'],x['manifest']['step']) == key),None)
        if found:
            return found, role
    raise ValueError('No verified 47K reference export. Supply C60LR150 R1 or C60LR200 R2, or explicitly --baseline EXPORT_DIR.')


def inside(folder, name):
    path = (folder / name).resolve(strict=True)
    if not path.is_relative_to(folder.resolve()):
        raise ValueError('Export path escapes its directory: ' + name)
    return path


def checked(folder, name, ledger):
    path = inside(folder, name)
    if name not in ledger or sha(path) != ledger[name]:
        raise ValueError('Export file missing from ledger or modified: ' + name)
    return path


def discover(roots):
    folders = set()
    for root in roots:
        root = Path(root).resolve(strict=True)
        if (root / 'manifest.json').is_file():
            folders.add(root)
        else:
            folders.update(p.parent for p in root.rglob('manifest.json'))
    return sorted(folders)


def inspect(folder):
    if (folder / 'INCOMPLETE.json').exists():
        raise ValueError('Incomplete export')
    ledger = read(folder / 'files_sha256.json')
    m = read(checked(folder, 'manifest.json', ledger))
    if (m.get('sensor') != 'GF2' or m.get('layout') != 'CHW' or
            m.get('bands') != ['Blue', 'Green', 'Red', 'NIR'] or m.get('original_peak_DN') != 1023):
        raise ValueError('Expected GF2 CHW4 float32 DN export with peak 1023')
    step = m['step']
    ev_path = checked(folder, f'provenance/evaluation_{step:07d}.json', ledger)
    ev = read(ev_path)
    cfg = read(checked(folder, 'provenance/config.json', ledger))
    side = read(checked(folder, f'provenance/step_{step:07d}.json', ledger))
    identity = ev['state_before']
    if (identity != ev['state_after'] or identity != ev['rr_state'] or identity != ev['fr_state']
            or identity != side['artifact']['identity'] or side['config'] != cfg
            or cfg['run_id'] != m['run_id'] or cfg['dataset'] != 'GF2'
            or identity['run_id'] != m['run_id'] or identity['completed_update'] != step
            or identity['dataset'] != 'GF2' or side['artifact']['completed_update'] != step
            or side['artifact']['file_sha256'] != m['checkpoint_sha256']
            or m['export_binding']['checkpoint_sha256'] != m['checkpoint_sha256']
            or m['export_binding']['evaluation_sha256'] != sha(ev_path)):
        raise ValueError('Mixed checkpoint, configuration or RR/FR evaluation identity')
    for split in ('rr', 'fr'):
        scenes = ev[split]['per_scene']
        if len(scenes) != 20 or [s['scene_id'] for s in scenes] != [f'GF2:{split}:{i}' for i in range(20)]:
            raise ValueError('Incomplete or reordered original RR20/FR20 evaluation')
    exported = {}
    for scene in m['scenes']:
        key = (scene['split'], scene['scene'])
        if key in exported:
            raise ValueError('Duplicate exported scene')
        if key[0] not in ('rr', 'fr') or key[1] not in range(20):
            raise ValueError('Invalid exported scene identifier')
        if scene['raw_prediction_sha256'] != ev[key[0]]['per_scene'][key[1]]['raw_prediction_sha256']:
            raise ValueError('Export scene SHA differs from original evaluation')
        exported[key] = scene
    if {i for s, i in exported if s == 'fr'} != set(range(20)):
        raise ValueError('All 20 saved FR predictions are required')
    rr = {i for s, i in exported if s == 'rr'}
    if rr and rr != set(range(20)):
        raise ValueError('Partial RR prediction export')
    metrics = ev['rr']['metrics'] | ev['fr']['metrics']
    if any(not np.isfinite(metrics[k]) for k in ('HQNR', 'ERGAS', 'PSNR')):
        raise ValueError('Non-finite quantitative metrics')
    token = hashlib.sha256((m['run_id'] + ':' + str(step) + ':' + m['checkpoint_sha256']).encode()).hexdigest()[:16]
    return dict(folder=folder, manifest=m, ledger=ledger, evaluation=ev, exported=exported,
                candidate_id=token, config=cfg)


def load_prediction(item, split, scene):
    entry = item['exported'][(split, scene)]
    p = checked(item['folder'], entry['prediction_npy'], item['ledger'])
    x = np.load(p, allow_pickle=False)
    size = 512 if split == 'fr' else 256
    if (x.dtype != np.float32 or x.shape != (4, size, size) or not np.isfinite(x).all()
            or hashlib.sha256(x.tobytes()).hexdigest() != entry['raw_prediction_sha256']
            or sha(p) != entry['npy_sha256']):
        raise ValueError(f'{split} scene {scene}: invalid or modified NPY')
    return x


def render_inputs(item, split, scene):
    prefix = split.upper()
    name = f'{prefix}/render_metadata/scene_{scene:02d}.json'
    meta = read(checked(item['folder'], name, item['ledger']))
    bounds = np.asarray(meta['rgb_bounds_dn'], float)
    expected_ref = 'stored LMS, not GT' if split == 'fr' else 'GT'
    if (meta['scene'] != scene or meta['split'] != split or meta['rgb_indices'] != [2, 1, 0]
            or meta['peak_DN'] != 1023 or meta['shared_reference'] != expected_ref
            or bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError('Invalid common-reference rendering metadata')
    p = checked(item['folder'], f'{prefix}/PAN/scene_{scene:02d}.png', item['ledger'])
    with Image.open(p) as im:
        pan = np.asarray(im.convert('RGB')).copy()
    size = 512 if split == 'fr' else 256
    if pan.shape != (size, size, 3):
        raise ValueError('Wrong PAN dimensions')
    return pan, bounds


def fr_context(item):
    return {i: render_inputs(item, 'fr', i) for i in range(20)}


def measure_candidate(item, context):
    per_scene = []
    for i in range(20):
        pan, bounds = render_inputs(item, 'fr', i)
        ref_pan, ref_bounds = context[i]
        if not np.array_equal(pan, ref_pan) or not np.allclose(bounds, ref_bounds, rtol=0, atol=1e-9):
            raise ValueError('PAN/reference bounds differ across candidates: scene ' + str(i))
        raw = load_prediction(item, 'fr', i)
        per_scene.append(dict(candidate_id=item['candidate_id'], scene=i,
                              **measure_saved(raw, pan, ref_bounds, 1023)))
    rr_errors = []
    rr_present = ('rr', 0) in item['exported']
    if rr_present:
        for i in range(20):
            raw = load_prediction(item, 'rr', i)
            name = f'RR/references_npz/scene_{i:02d}.npz'
            if name in item['ledger']:
                with np.load(checked(item['folder'], name, item['ledger']), allow_pickle=False) as f:
                    gt = f['gt'].astype(np.float64)
                if gt.shape != raw.shape or not np.isfinite(gt).all():
                    raise ValueError('RR GT shape or values invalid')
                from scipy.ndimage import sobel
                difference = raw.astype(np.float64) / 1023 - gt / 1023
                # Per-band derivatives: do not smooth across the spectral axis.
                error = np.mean([np.abs(sobel(b, 0, mode='nearest')[20:-21, 20:-21]).mean()
                                 + np.abs(sobel(b, 1, mode='nearest')[20:-21, 20:-21]).mean() for b in difference])
                rr_errors.append(float(error))
    m, cfg = item['manifest'], item['config']
    means = {k: (float(np.mean([r[k] for r in per_scene])) if all(r[k] is not None for r in per_scene) else None)
             for k in per_scene[0] if k not in ('candidate_id', 'scene')}
    if means['raw_edge_pm2'] <= 1e-12 or means['shared_edge_pm2'] <= 1e-12:
        raise ValueError('Degenerate constant output')
    metrics = item['evaluation']['rr']['metrics'] | item['evaluation']['fr']['metrics']
    row = dict(candidate_id=item['candidate_id'], run_id=m['run_id'], step=m['step'],
               checkpoint_sha256=m['checkpoint_sha256'], checkpoint_path=m['checkpoint_path'],
               export_path=str(item['folder']), width=cfg.get('stage2_spec', {}).get('student_width'),
               rr_npy_verified=rr_present, rr_gt_gradient_mae=float(np.mean(rr_errors)) if len(rr_errors) == 20 else None,
               rr_gt_diagnostic_scenes=len(rr_errors), **metrics, **means)
    return row, per_scene


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def auto_rois(context):
    """Pick PAN-only edge-rich and flat regions, without consulting model outputs."""
    windows = []
    for i, (pan, _) in context.items():
        lum = pan.astype(float).mean(2) / 255
        g, std = gradient(lum), local_std(lum, 9)
        for y in range(16, 401, 48):
            for x in range(16, 401, 48):
                patch = lum[y:y+96, x:x+96]
                if np.std(patch) < 1e-5:
                    continue
                windows.append(dict(scene=i, x=x, y=y, size=96,
                                    edge=float(g[y:y+96, x:x+96].mean()),
                                    flat=float(std[y:y+96, x:x+96].mean())))
    rois, used = [], set()
    for label, key in [('PAN-edge', lambda r: -r['edge']), ('PAN-flat', lambda r: r['flat'])]:
        count = 0
        for row in sorted(windows, key=key):
            if row['scene'] in used:
                continue
            rois.append(dict(scene=row['scene'], x=row['x'], y=row['y'], size=row['size'], label=label))
            used.add(row['scene'])
            count += 1
            if count == 4:
                break
    return rois


def panels(output, items, context, ids, rois, seed):
    lookup = {x['candidate_id']: x for x in items}
    rng = np.random.default_rng(seed)
    order = [ids[i] for i in rng.permutation(len(ids))]
    key = {f'B{i+1:02d}': dict(candidate_id=cid, run_id=lookup[cid]['manifest']['run_id'],
                               step=lookup[cid]['manifest']['step']) for i, cid in enumerate(order)}
    write(output / 'blind_key.json', key)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path),14) if font_path.exists() else ImageFont.load_default()
    cell, label_width, header, row_height = 192, 150, 40, 220
    canvas = Image.new('RGB', (label_width + (len(order)+1)*cell, header + len(rois)*row_height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((label_width+8, 10), 'PAN', fill='black', font=font)
    for j in range(len(order)):
        draw.text((label_width+(j+1)*cell+8, 10), f'B{j+1:02d}', fill='black', font=font)
    for row_index, roi in enumerate(rois):
        scene, x, y, size = (int(roi[k]) for k in ('scene', 'x', 'y', 'size'))
        if scene not in range(20) or size < 8 or min(x, y) < 0 or max(x+size, y+size) > 512:
            raise ValueError('Invalid fixed FR ROI')
        top = header + row_index*row_height
        label = str(roi.get('label', ''))
        if not font_path.exists():
            label = label.encode('ascii',errors='replace').decode()
        draw.text((5, top+8), f"FR {scene:02d} {label}\n({x},{y}) {size}px", fill='black', font=font)
        pan, bounds = context[scene]
        arrays = [pan]
        for cid in order:
            raw = load_prediction(lookup[cid], 'fr', scene)
            arrays.append(np.rint(shared_rgb(raw, bounds)*255).astype(np.uint8))
        for j, arr in enumerate(arrays):
            crop = Image.fromarray(arr[y:y+size, x:x+size]).resize((cell, cell), Image.Resampling.NEAREST)
            canvas.paste(crop, (label_width+j*cell, top))
    canvas.save(output / 'FR_blind.png')
    named = canvas.copy()
    draw = ImageDraw.Draw(named)
    for j, cid in enumerate(order):
        m = lookup[cid]['manifest']
        draw.text((label_width+(j+1)*cell+42, 10), f"{m['step']//1000}K {cid[:6]}", fill='black', font=font)
    named.save(output / 'FR_named.png')
    # Every scene is also available; fixed ROI sheets are not the full decision set.
    scenes_dir = output / 'FR_all20'
    scenes_dir.mkdir()
    for scene in range(20):
        pan, bounds = context[scene]
        sheet = Image.new('RGB', ((len(order)+1)*256, 280), 'white')
        d = ImageDraw.Draw(sheet)
        for j, cid in enumerate([None] + order):
            arr = pan if cid is None else np.rint(shared_rgb(load_prediction(lookup[cid], 'fr', scene), bounds)*255).astype(np.uint8)
            d.text((j*256+5, 3), 'PAN' if cid is None else f'B{j:02d}', fill='black', font=font)
            sheet.paste(Image.fromarray(arr).resize((256,256), Image.Resampling.LANCZOS), (j*256,24))
        sheet.save(scenes_dir / f'scene_{scene:02d}.png')
    rr_availability = []
    rr_ids = [cid for cid in order if ('rr', 0) in lookup[cid]['exported']]
    for cid in order:
        rr_availability.append(dict(candidate_id=cid, status='RR20_VERIFIED' if cid in rr_ids else 'RR_NPY_NOT_EXPORTED'))
    if rr_ids:
        rr_dir = output / 'RR_all20'
        rr_dir.mkdir()
        ref_item = next((lookup[cid] for cid in rr_ids if 'RR/references_npz/scene_00.npz' in lookup[cid]['ledger']), lookup[rr_ids[0]])
        for scene in range(20):
            pan, bounds = render_inputs(ref_item, 'rr', scene)
            arrays = [('PAN', pan)]
            gt_name = f'RR/references_npz/scene_{scene:02d}.npz'
            if gt_name in ref_item['ledger']:
                with np.load(checked(ref_item['folder'], gt_name, ref_item['ledger']), allow_pickle=False) as ref:
                    gt = ref['gt']
                arrays.append(('GT', np.rint(shared_rgb(gt, bounds)*255).astype(np.uint8)))
            for cid in rr_ids:
                candidate_pan, candidate_bounds = render_inputs(lookup[cid], 'rr', scene)
                if not np.array_equal(candidate_pan, pan) or not np.allclose(candidate_bounds, bounds, rtol=0, atol=1e-9):
                    raise ValueError(f'RR scene {scene}: comparison inputs differ between candidates')
                raw = load_prediction(lookup[cid], 'rr', scene)
                arrays.append((f'B{order.index(cid)+1:02d}', np.rint(shared_rgb(raw,bounds)*255).astype(np.uint8)))
            sheet = Image.new('RGB', (len(arrays)*256, 280), 'white')
            d = ImageDraw.Draw(sheet)
            for j, (label, arr) in enumerate(arrays):
                d.text((j*256+5, 3), label, fill='black', font=font)
                sheet.paste(Image.fromarray(arr), (j*256, 24))
            sheet.save(rr_dir/f'scene_{scene:02d}.png')
    write(output / 'RR_availability.json', rr_availability)
    return key


def run(args):
    release = verify_release()
    folders = discover(args.exports)
    if args.baseline:
        folders = sorted(set(folders + [Path(args.baseline).resolve(strict=True)]))
    if not folders:
        raise ValueError('No manifest.json exports found; expected panda_sensor_visuals.export format')
    out = Path(args.output).resolve()
    if any(out == f or out.is_relative_to(f) for f in folders):
        raise ValueError('Selection output must be outside every original export directory')
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'INCOMPLETE.json', dict(status='RUNNING', originals_read_only=True))
    excluded, candidates, duplicates = [], [], []
    for folder in folders:
        try:
            candidates.append(inspect(folder))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            excluded.append(dict(path=str(folder), reason=str(exc)))
    # Prefer a complete RR+FR mirror over an otherwise identical FR-only copy.
    candidates.sort(key=lambda x: (-len(x['exported']), str(x['folder'])))
    unique = {}
    conflicts = set()
    for item in candidates:
        k = (item['manifest']['run_id'], item['manifest']['step'])
        if k in unique:
            if item['manifest']['checkpoint_sha256'] != unique[k]['manifest']['checkpoint_sha256']:
                conflicts.add(k)
                excluded.append(dict(path=str(item['folder']), reason='Conflicting checkpoint SHA for same run/step'))
            else:
                duplicates.append(dict(path=str(item['folder']), preferred=str(unique[k]['folder'])))
        else:
            unique[k] = item
    for k in conflicts:
        excluded.append(dict(path=str(unique[k]['folder']), reason='Conflicting checkpoint SHA for same run/step'))
        unique.pop(k)
    candidates = list(unique.values())
    write(out / 'exclusions.json', excluded)
    write(out / 'duplicates.json', duplicates)
    baseline, baseline_role = choose_baseline(candidates,args.baseline)
    print('Paired reference:', baseline_role, baseline['manifest']['run_id'], flush=True)
    context = fr_context(baseline)
    rows, scenes, valid = [], [], []
    ordered = [baseline] + [x for x in candidates if x is not baseline]
    for item in ordered:
        print('Reading', item['manifest']['run_id'], item['manifest']['step'], flush=True)
        try:
            row, per_scene = measure_candidate(item, context)
            rows.append(row)
            scenes.extend(per_scene)
            valid.append(item)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            excluded.append(dict(path=str(item['folder']), reason=str(exc)))
            write(out / 'exclusions.json', excluded)
            if item is baseline:
                raise ValueError('Baseline failed NPY/reference verification: ' + str(exc)) from exc
    base_row = rows[0]
    if base_row['shared_flat_grain'] <= 1e-12 or base_row['raw_flat_grain'] <= 1e-12:
        raise ValueError('Baseline grain is zero; relative noise comparisons undefined')
    ref = read(REFERENCE)
    shortlists = classify(rows, scenes, baseline['candidate_id'], ref['methods'], min_wins=args.min_wins,
                          grain_ratio_max=args.grain_ratio_max, edge_retention=args.edge_retention, top_k=args.top_k)
    for r in rows:
        same_run = sorted((x for x in rows if x['run_id'] == r['run_id']), key=lambda x:x['step'])
        at = same_run.index(r)
        neighbors = same_run[max(0,at-1):at] + same_run[at+1:at+2]
        r['neighbor_steps'] = ','.join(str(x['step']) for x in neighbors)
        r['neighbor_visual_passes'] = sum(int(x['visual_gate']) for x in neighbors)
    rows.sort(key=lambda r: (-r['shared_edge_pm2'], r['shared_flat_grain'], r['candidate_id']))
    quantitative_reference = next((r for r in rows if (r['run_id'], r['step']) == (QUANT_RUN, 47000)), None)
    representatives = {k: (shortlists[k][0] if shortlists[k] else None) for k in ('visual','quantitative','compromise')}
    chosen = [baseline['candidate_id']]
    if quantitative_reference:
        chosen.append(quantitative_reference['candidate_id'])
    for cid in list(representatives.values()) + shortlists['visual'] + shortlists['provisional_frontier']:
        if cid and cid not in chosen and len(chosen) < args.panel_models:
            chosen.append(cid)
    rois = read(Path(args.rois)) if args.rois else auto_rois(context)
    if not isinstance(rois, list) or not rois:
        raise ValueError('ROI JSON must be a nonempty list of {scene,x,y,size,label}')
    write(out / 'rois.json', rois)
    key = panels(out, valid, context, chosen, rois, args.blind_seed)
    protocol = dict(schema='GF2_SAVED_OUTPUT_SELECTION_V1', source_release=release,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pillow=PIL.__version__),
        exports=[str(Path(p).resolve()) for p in args.exports], baseline=str(baseline['folder']),
        baseline_id=baseline['candidate_id'], baseline_role=baseline_role,
        quantitative_reference_id=None if not quantitative_reference else quantitative_reference['candidate_id'],
        target=dict(HQNR_min=.9645, ERGAS_max=.522, PSNR_min=45.08),
        rules=dict(min_joint_improved_scenes=args.min_wins, grain_ratio_max=args.grain_ratio_max,
                   compromise_edge_retention=args.edge_retention, top_k=args.top_k,
                   compromise_order='ERGAS ascending, HQNR descending, PSNR descending among near-best clean consistent Pareto candidates',
                   visual_gate='legacy external threshold AND raw/shared guard AND scene consistency',
                   external_thresholds='Only legacy own-stretch PNG protocol; supplied comparison records, not independently rerun',
                   raw='RGB arithmetic mean /1023, unclipped float64; masks from same archived PAN',
                   shared='RGB bounds from stored LMS, identical per scene across candidates; float64 [0,1], no uint8 quantization',
                   rr_gt_gradient_mae='optional four-band Sobel gradient error on raw/1023, crop [20:-21], diagnostic not official metric'),
        selection='Exploratory test-output selection; one checkpoint across all RR/FR scenes',
        final_selection='PENDING_HUMAN_REVIEW', training=False, inference=False, sheet_write=False,
        comparison_reference=ref, blind_seed=args.blind_seed,
        limitations=['PAN is a spatial proxy, not FR spectral GT.',
                     'Flat-grain is local variation, not a pure noise estimator.',
                     'Numeric gates cannot certify absence of ringing or color artifacts.',
                     'RR NPY/GT diagnostics are available only where exported; missing evidence is explicit.'])
    write_csv(out / 'ranking.csv', rows)
    write_csv(out / 'per_scene.csv', scenes)
    write(out / 'ranking.json', dict(protocol=protocol, rows=rows))
    write(out / 'shortlist.json', dict(representatives=representatives, shortlists=shortlists,
          panel_candidates=chosen, final_selection='PENDING_HUMAN_REVIEW'))
    write(out / 'exclusions.json', excluded)
    report = ['# GF2 saved-output selection', '', f'Verified candidates: {len(rows)}; excluded: {len(excluded)}.', '',
              'CPU postprocessing only. Original exports, checkpoints, training and Sheets are unchanged.', '',
              f"Paired reference: {baseline['manifest']['run_id']} @{baseline['manifest']['step']} — {baseline_role}.",
              'If the original qualitative export is absent, the fixed quantitative 47K reference is used; paired improvements then refer to that model, not the old qualitative Ours.', '',
              '## Preliminary candidates (human review required)', '']
    lookup = {r['candidate_id']:r for r in rows}
    for group, cid in representatives.items():
        report.append(f'- {group}: ' + ('NONE — no candidate satisfies this category.' if cid is None
                      else f"{lookup[cid]['run_id']} @{lookup[cid]['step']:,} ({cid})"))
    report += ['', 'If visual is NONE, provisional_frontier is an inspection list, not a successful sharpness selection.',
               'The quantitative category passes all three original goals; it may fail the visual gates.',
               'Inspect FR_blind.png before blind_key.json / FR_named.png. FR_all20 contains every scene.',
               'RR_all20 provides shared-reference comparisons where RR NPYs are available; see RR_availability.json. RR and FR scene numbers are independent.',
               'Only historical own-stretch legacy_* metrics can be compared to the supplied U-Know/PAN-Crafter thresholds.',
               'Shared/raw metrics use their own scales; compare them to the same local baseline.', '',
               '## Comparison columns', '', '| ID | Run / step | HQNR | ERGAS | PSNR | Shared edge | Shared grain | Wins/20 | Visual | Target |',
               '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        report.append(f"| {r['candidate_id']} | {r['run_id']} / {r['step']} | {r['HQNR']:.4f} | {r['ERGAS']:.4f} | {r['PSNR']:.4f} | {r['shared_edge_pm2']:.6f} | {r['shared_flat_grain']:.6f} | {r['improvement_scenes']} | {r['visual_gate']} | {r['quantitative_gate']} |")
    report += ['', '## Blind key', ''] + [f"- {label}: {v['run_id']} @{v['step']}" for label,v in key.items()]
    (out / 'report.md').write_text('\n'.join(report)+'\n')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>GF2 selection</title>'
        '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>GF2 candidate review</h1><p>Human review pending; saved outputs only.</p>'
        '<p><a href="ranking.csv">Ranking CSV</a> · <a href="shortlist.json">Shortlist</a> · '
        '<a href="report.md">Named report</a> · <a href="blind_key.json">Reveal identities</a></p>'
        '<h2>Shared-reference fixed crops</h2><img src="FR_blind.png">'
        '<h2>All 20 FR scenes</h2>' + ''.join(f'<p>Scene {i:02d}</p><img loading="lazy" src="FR_all20/scene_{i:02d}.png">' for i in range(20)) +
        ('<h2>RR comparisons (independent scenes)</h2>' + ''.join(f'<p>RR {i:02d}</p><img loading="lazy" src="RR_all20/scene_{i:02d}.png">' for i in range(20)) if (out/'RR_all20').exists() else '<p>RR images were not exported for the panel candidates; quantitative metrics remain in the ranking.</p>'))
    write(out / 'completion.json', dict(status='COMPLETED', verified=len(rows), excluded=len(excluded),
                                       baseline=baseline['candidate_id'], final_selection='PENDING_HUMAN_REVIEW'))
    (out / 'INCOMPLETE.json').unlink()
    print(json.dumps(dict(output=str(out), candidates=len(rows), representatives=representatives), indent=2))
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports', nargs='+', default=[str(ROOT/'work_dir/panda_gf2_sharpness/exports')])
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    p.add_argument('--output', default=str(ROOT/'work_dir/panda_gf2_sharpness/selections'/stamp))
    p.add_argument('--baseline', help='Explicit reference export; default C60LR150 R1@47K, else fixed C60LR200 R2@47K')
    p.add_argument('--rois', help='Optional fixed FR ROI JSON; otherwise selected only from PAN')
    p.add_argument('--min-wins', type=int, default=14)
    p.add_argument('--grain-ratio-max', type=float, default=1.0)
    p.add_argument('--edge-retention', type=float, default=.95)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--panel-models', type=int, default=6)
    p.add_argument('--blind-seed', type=int, default=20261109)
    p.add_argument('--verify', action='store_true', help='Only verify selector source release')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify_release(), indent=2))
        return
    if not (1 <= args.min_wins <= 20 and np.isfinite(args.grain_ratio_max) and args.grain_ratio_max > 0
            and 0 < args.edge_retention <= 1 and args.top_k > 0 and args.panel_models >= 3):
        p.error('Invalid selection thresholds or panel size')
    run(args)


if __name__ == '__main__':
    main()
