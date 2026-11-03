"""Local RGB donor selection and exact train-only, same-donor calibration.

The old tune30 donor adapter rejects RGB references by design. Reuse its
numerical frozen prediction operators, but never relabel RGB as mean8.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from directshift48.method import ScharrAbsolute
from directshift48.reporting import atomic_json
from panda_band30.reconstruction import BandReconstructionSpec, upsampler_manifest
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_stage2_condq.calibration import CalibrationBuilder, CalibrationStats, validate_binding
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.sources import validate_training_batch
from panda_stage2_condq.spec import MASK2_POLICY
from panda_tune30.method import frozen_predictions
from panda_tune30.warp import warp_manifest
from panda_wv3_ablation.assets import sources
from panda_wv3_ablation_rgb import selection
from ta2.evaluation import file_sha
from . import plan
from .model import Stage1


def _read(path):
    return json.loads(Path(path).read_text())


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError('Immutable donor/calibration evidence changed: ' + str(path))
    else:
        atomic_json(path, value)


def _select(cfg, registered, root, pins):
    from .evaluation import verify_record
    root = Path(root)
    donor_cfg = plan.resolve('s1', 'S1_RGB_FULL', cfg['repeat'])
    if cfg['server'] != 's1' or cfg['donor_run_id'] != donor_cfg['run_id']:
        raise ValueError('Same-server and same-repeat RGB Stage1 donor required')
    state = _read(root/'campaign_state.json')
    jobs = [j for j in state['runs'] if j['run_id'] == donor_cfg['run_id']]
    if (state.get('campaign_id') != donor_cfg['campaign_id'] or state.get('server') != 's1'
            or len(jobs) != 1 or jobs[0]['status'] != 'COMPLETED' or jobs[0]['step'] != 50000):
        raise ValueError('RGB donor selection requires completed local 50K Stage1')
    directory = root/'runs'/donor_cfg['run_id']
    if _read(directory/'config.json') != donor_cfg:
        raise ValueError('Source Stage1 configuration differs from its registered recipe')
    inventory = _read(root/'validation_inventory.json')
    selection.validate_inventory(inventory, registered.sources)
    if pins['train_inventory'] != registered.verify().manifest():
        raise ValueError('Donor training inventory differs from current admission')
    rows = []
    for step in donor_cfg['evaluation_steps']:
        evpath = directory/f'evaluation_{step:07d}.json'
        valpath = directory/f'validation_{step:07d}.json'
        record, val = _read(evpath), _read(valpath)
        verify_record(record, donor_cfg)
        if record['step'] != step or record['eval_protocol_id'] != plan.digest(pins['evaluator']):
            raise ValueError('Donor evaluation step/evaluator mismatch')
        if val.get('protocol') != selection.PROTOCOL:
            raise ValueError('Donor validation must use the registered RGB protocol')
        aggregate = selection.aggregate_rows(val['per_sample'], inventory)
        if any(val.get(k) != v for k, v in aggregate.items()):
            raise ValueError('RGB validation summary differs from its full population')
        summary = {k:v for k,v in val.items() if k not in ('per_sample', 'protocol')}
        if record.get('validation') != summary:
            raise ValueError('Evaluation and RGB validation evidence disagree')
        if not all(math.isfinite(float(val[k])) and val[k] >= 0
                   for k in ('RRE50', 'RRE90', 'Val_ERGAS')):
            raise ValueError('Finite nonnegative RGB validation ranking required')
        checkpoint = directory/'models'/f'step_{step:07d}.pt'
        meta = _read(checkpoint.with_suffix('.json'))
        if (meta != record['checkpoint'] or meta['step'] != step
                or plan.local_path(meta['path']).resolve() != checkpoint.resolve()
                or meta['config_sha256'] != donor_cfg['config_sha256']
                or meta['model_sha256'] != record['model_sha256']
                or file_sha(checkpoint) != meta['file_sha256']):
            raise ValueError('Donor retained checkpoint hash/identity mismatch')
        rows.append({'step':step, 'validation':summary,
            'checkpoint_path':str(checkpoint.relative_to(root)),
            'file_sha256':meta['file_sha256'], 'model_sha256':meta['model_sha256'],
            'evaluation_sha256':file_sha(evpath), 'validation_sha256':file_sha(valpath)})
    if not rows or rows[-1]['step'] != 50000:
        raise ValueError('All scheduled Stage1 candidates through 50K are required')
    chosen = min(rows, key=lambda r:(r['validation']['RRE50'], r['validation']['RRE90'],
                                     r['validation']['Val_ERGAS'], r['step']))
    receipt = {'schema':'WV3_MAIN_RGB_DONOR_SELECTION_V1', 'server':'s1',
        'repeat':cfg['repeat'], 'donor_run_id':donor_cfg['run_id'],
        'source_config_sha256':donor_cfg['config_sha256'], 'source_release':pins['source_release'],
        'selector':'RGB_VAL_RRE50_RRE90_VAL_ERGAS_STEP', 'uses_test_for_selection':False,
        'validation_inventory_sha256':inventory['inventory_sha256'],
        'selector_protocol_sha256':canonical_sha256(selection.PROTOCOL),
        'candidate_steps':[r['step'] for r in rows], 'candidates':rows, 'selected':chosen}
    receipt['receipt_sha256'] = plan.digest(receipt)
    return donor_cfg, receipt


def _load_selected(donor_cfg, receipt, registered, root, pins, *, device):
    from .training import model_hash
    chosen = receipt['selected'];path = Path(root)/chosen['checkpoint_path']
    before = path.stat()
    if file_sha(path) != chosen['file_sha256']:
        raise ValueError('Selected donor checkpoint SHA mismatch')
    value = torch.load(path, map_location='cpu', weights_only=False)
    if path.stat() != before or file_sha(path) != chosen['file_sha256']:
        raise ValueError('Donor checkpoint changed while loading')
    complete_schema = 'WV3_MAIN_RGB_COMPLETE_CHECKPOINT_V1'
    evaluated_schema = 'WV3_MAIN_RGB_EVALUATED_MODEL_V1'
    if (value.get('schema') not in (complete_schema, evaluated_schema)
            or value.get('config') != donor_cfg or value.get('pins') != pins
            or value.get('updates') != chosen['step']
            or value.get('model_sha256') != chosen['model_sha256']
            or (value['schema'] == complete_schema
                and any(k not in value for k in ('optimizer','stream','rng')))):
        raise ValueError('Verified same-source RGB donor checkpoint required')
    with torch.random.fork_rng(devices=[]):
        model = Stage1(donor_cfg, registered.inventory.manifest())
        model.load_state_dict(value['model'], strict=True)
    if model_hash(model) != chosen['model_sha256']:
        raise ValueError('Loaded donor A/T state differs from evaluation evidence')
    for module in (model.aligner, model.teacher):
        if any(not bool(torch.isfinite(p).all()) for p in module.parameters()):
            raise ValueError('Nonfinite donor parameters')
        module.requires_grad_(False).eval().to(device)
    inventory = registered.verify().manifest()
    binding = {k:v for k,v in inventory.items() if k != 'sample_ids'}
    binding.update(aligner_state_sha256=state_dict_sha256(model.aligner),
        teacher_state_sha256=state_dict_sha256(model.teacher),
        donor_checkpoint_id=chosen['file_sha256'], donor_checkpoint_sha256=chosen['file_sha256'],
        donor_run_id=donor_cfg['run_id'], donor_step=chosen['step'],
        alignment_target='SCHARR_OF_RGB_MEAN_GT',
        reconstruction_spec={'bands':8,'width':112,'depth':[1,2,3],
                             'input_layout':'P0','base_reference':'BICUBIC_MS'},
        scharr={'kernel':'absolute_xy_/32','padding':'reflect1','rgb_indices':[4,2,1],'roi_margin':8},
        warp='bicubic_border_FP32_align_corners_false_current_pan_px',
        main_rgb_geometry={'schema':'WV3_MAIN_RGB_FULL_GEOMETRY_V1','scale':1.,
            'student_warp':warp_manifest(), 'teacher_warp':'BICUBIC_BORDER_FULL_U',
            'base':upsampler_manifest('interp23'), 'gt_and_output_coordinates':'ORIGINAL_MS',
            'q_target':'RGB_MEAN_GT_4_2_1','q_pan':'LANCZOS4_CORRECTED_PAN',
            'loss_mask_policy':MASK2_POLICY},
        band_reconstruction_spec=BandReconstructionSpec(8,112,(1,2,3),'P0','BICUBIC_MS',1).manifest(),
        selector_receipt_sha256=receipt['receipt_sha256'], source_release=pins['source_release'])
    binding = validate_binding(binding)
    donor = LoadedDonor(model.aligner, model.teacher, binding,
        {'schema':'WV3_MAIN_RGB_IMPORTED_DONOR_V1','source_config':deepcopy(donor_cfg),
         'selection':deepcopy(receipt),'checkpoint_sha256':chosen['file_sha256']})
    donor.selection_receipt = deepcopy(receipt)
    donor.assert_frozen()
    return donor


def calibration(donor, registered, folder, *, device='cpu', batch_size=48):
    """Fresh exact full-train ECDF; cache identity includes truthful RGB A/T."""
    folder = Path(folder);folder.mkdir(parents=True, exist_ok=True)
    path, receipt_path = folder/'full_s100.npz', folder/'full_s100.json'
    donor.assert_frozen();registered.verify()
    binding = validate_binding(donor.binding)
    if path.exists():
        # A completed immutable NPZ can survive interruption before JSON receipt.
        # Its embedded binding/content hash must still pass before reconstruction.
        receipt = _read(receipt_path) if receipt_path.exists() else None
        if receipt is not None and receipt['donor_binding'] != binding:
            raise ValueError('Calibration receipt belongs to another RGB donor')
        stats = CalibrationStats.load(path, expected_binding=binding,
            expected_file_sha256=None if receipt is None else receipt['file_sha256'],
            expected_mask_policy=MASK2_POLICY,memory_budget_bytes=4*1024**3)
        if receipt is None:
            receipt = {'schema':'WV3_MAIN_RGB_CALIBRATION_RECEIPT_V1',
                'path':str(path),'file_sha256':file_sha(path),'artifact_sha256':stats.artifact_sha256,
                'donor_binding':binding,'sample_count':len(registered.sample_ids),
                'tau_e':stats.tau_e,'recovered_from_verified_immutable_artifact':True}
            _immutable_json(receipt_path, receipt)
    else:
        if receipt_path.exists():
            raise ValueError('Calibration receipt exists without its retained artifact')
        collector = CalibrationBuilder(registered.sample_ids,binding,k=5,
            memory_budget_bytes=4*1024**3,include_condq=True,mask_policy=MASK2_POLICY)
        scharr = ScharrAbsolute('WV3').to(device);started = time.monotonic();seen = 0
        for rr in registered.calibration_batches(batch_size, device=device):
            validate_training_batch(rr,binding,expected_sample_ids=registered.sample_ids,
                                    require_unaugmented=True)
            f = frozen_predictions(donor.aligner,donor.teacher,rr,geometry_key='s100')
            e,g,q = reference_maps(scharr,f['teacher_y'],f['gt_loss'],f['pan_loss'])
            collector.add(rr.lineage.sample_ids,e,g,q,sensor=rr.lineage.sensor,
                split=rr.lineage.split,resolution=rr.lineage.resolution,
                unaugmented=all(a==(False,False,0) for a in rr.lineage.augmentations),
                support_ok=bool(f['spatial_support_ok'].all()),loss_mask=f['loss_mask'])
            seen += len(rr.pan)
        donor.assert_frozen();registered.verify()
        stats = collector.finalize();stats.assert_integrity(full_hash=True)
        saved = stats.save(path)
        receipt = {**saved,'schema':'WV3_MAIN_RGB_CALIBRATION_RECEIPT_V1',
            'donor_binding':binding,'sample_count':seen,'tau_e':stats.tau_e,
            'elapsed_seconds':time.monotonic()-started,'mask_policy':MASK2_POLICY}
        _immutable_json(receipt_path,receipt)
    stats.validate_binding(binding);stats.assert_integrity(full_hash=True)
    donor.assert_frozen();registered.verify()
    return stats.to(device,memory_budget_bytes=1024**3)


def prepare_reference(cfg, registered, root, pins, device='cpu'):
    if cfg['stage'] != 'stage2':
        raise ValueError('Only Stage2 consumes a frozen RGB donor')
    donor_cfg, receipt = _select(cfg,registered,root,pins)
    donor = _load_selected(donor_cfg,receipt,registered,root,pins,device=device)
    folder = Path(root)/'donors'/f"R{cfg['repeat']}"
    _immutable_json(folder/'selection.json',receipt)
    stats = calibration(donor,registered,folder/'calibration',device=device)
    return donor,stats
