"""Existing local QB and fresh GF2 donors; exact common-domain calibration."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from directshift48.reporting import atomic_json
from panda_unified.core import upsampler_manifest, ReferenceSpatial, reference_maps
from panda_method.calibration import canonical_sha256, state_dict_sha256
from .calibration import CalibrationBuilder, CalibrationStats, validate_binding
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.sources import validate_training_batch
from .geometry import MASK2_POLICY, WARP_CONTRACT
from .geometry import frozen_predictions
from .geometry import warp_manifest
from . import selection
from ta2.evaluation import file_sha
from . import plan
from .model import Stage1


def sources(server='s2'):
    """Preserve original QB Aligner LMS and registered repaired LRMS separately."""
    from panda_method.edge_data import EdgeDataSources
    from panda_stage2_condq.sources import RegisteredRRTrain
    cfg = plan.donor_config(server, 1)
    catalog = plan.local_path(cfg['source_registry'])
    if file_sha(catalog) != cfg['source_registry_sha256']:
        raise ValueError('Registered sensor data catalog changed')
    overrides = {}
    if cfg['sensor'] == 'QB':
        value = _read(plan.ROOT/'config/bglobal_rr/qb_original_lms.json')
        if value['sensor'] != 'QB':
            raise ValueError('QB original stored-LMS override required')
        overrides = deepcopy(value['splits'])
    source = EdgeDataSources(catalog, root=plan.ROOT, sensor=cfg['sensor'], lms_overrides=overrides)
    for split in ('train', 'val', 'rr', 'fr'):
        source.raw(split, [0])
    return source, RegisteredRRTrain(source)


def _read(path):
    return json.loads(Path(path).read_text())


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError('Immutable donor/calibration evidence changed: ' + str(path))
    else:
        atomic_json(path, value)


def _select(cfg,registered,root,pins):
    from .ranking import choose
    from .evaluation import verify_record as verify_new
    from panda_unified_campaign.evaluation import verify_record as verify_old
    plan.validate(cfg)
    donor_cfg=plan.donor_config(cfg['server'],cfg['repeat'])
    source_root=plan.donor_root(cfg['server']);oldpins=_read(source_root/'admission.json')
    if cfg['donor_run_id']!=donor_cfg['run_id']:
        raise ValueError('Registered local donor identity mismatch')
    for key in ('evaluator','train_inventory','split_manifests'):
        if oldpins[key]!=pins[key]:raise ValueError('Donor/current data or evaluator mismatch: '+key)
    if registered.verify().manifest()!=oldpins['train_inventory']:raise ValueError('Training inventory changed')
    state=_read(source_root/'campaign_state.json')
    jobs=[r for r in state['runs'] if r['run_id']==donor_cfg['run_id']]
    if (state['campaign_id']!=donor_cfg['campaign_id'] or state['server']!=cfg['server']
        or len(jobs)!=1 or jobs[0]['status']!='COMPLETED' or jobs[0]['step']!=50000):
        raise ValueError('Completed local RGBN50K Stage1 required')
    directory=source_root/'runs'/donor_cfg['run_id']
    if _read(directory/'config.json')!=donor_cfg:raise ValueError('Donor config differs from recipe')
    rows=[];verify=verify_old if cfg['server']=='s2' else verify_new
    for step in donor_cfg['evaluation_steps']:
        ev=directory/f'evaluation_{step:07d}.json';e=_read(ev);verify(e,donor_cfg)
        ck=directory/'models'/f'step_{step:07d}.pt';meta=_read(ck.with_suffix('.json'))
        if (e['step']!=step or e['eval_protocol_id']!=plan.digest(oldpins['evaluator'])
            or meta!=e['checkpoint'] or meta['model_sha256']!=e['model_sha256']
            or meta['config_sha256']!=donor_cfg['config_sha256'] or meta['step']!=step
            or plan.local_path(meta['path']).resolve()!=ck.resolve() or file_sha(ck)!=meta['file_sha256']):
            raise ValueError('Donor checkpoint/evaluation identity mismatch')
        rows.append(dict(step=step,rr=e['rr']['metrics'],fr=e['fr']['metrics'],
            checkpoint_path=str(ck.relative_to(source_root)),file_sha256=meta['file_sha256'],
            model_sha256=meta['model_sha256'],evaluation_sha256=file_sha(ev)))
    chosen=choose(rows,**cfg['donor_selection'])
    receipt=dict(schema='PANDA_SENSOR_EQ40_DONOR_SELECTION_V1',server=cfg['server'],sensor=cfg['sensor'],
        reference_mode='RGBN',repeat=donor_cfg['repeat'],donor_run_id=donor_cfg['run_id'],
        source_config_sha256=donor_cfg['config_sha256'],source_release=oldpins['source_release'],
        uses_test_for_selection=True,selector='ERGAS_GATE_THEN_'+cfg['donor_selection']['order'],
        thresholds=cfg['donor_selection'],candidates=rows,candidate_steps=[r['step'] for r in rows],selected=chosen)
    receipt['receipt_sha256']=plan.digest(receipt)
    return donor_cfg,receipt,source_root,oldpins


def _load_selected(donor_cfg, receipt, registered, root, pins, *, device):
    from .training import model_hash
    chosen = receipt['selected'];path = Path(root)/chosen['checkpoint_path']
    before = path.stat()
    if file_sha(path) != chosen['file_sha256']:
        raise ValueError('Selected donor checkpoint SHA mismatch')
    value = torch.load(path, map_location='cpu', weights_only=False)
    if path.stat() != before or file_sha(path) != chosen['file_sha256']:
        raise ValueError('Donor checkpoint changed while loading')
    complete_schema = 'PANDA_SENSOR_EQ40_COMPLETE_CHECKPOINT_V1'
    evaluated_schema = 'PANDA_SENSOR_EQ40_EVALUATED_MODEL_V1'
    if (value.get('schema') not in (complete_schema, evaluated_schema)
            or value.get('config') != donor_cfg or value.get('pins') != pins
            or value.get('updates') != chosen['step']
            or value.get('model_sha256') != chosen['model_sha256']
            or (value['schema'] == complete_schema
                and any(k not in value for k in ('optimizer','stream','rng')))):
        raise ValueError('Verified same-source unified donor checkpoint required')
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
        alignment_target=donor_cfg['alignment_target'],
        reference_mode=donor_cfg['reference_mode'], mask_policy=MASK2_POLICY,
        stage1_spatial_reference={'bands':donor_cfg['reference_bands'],
            'aggregation':'arithmetic_mean_before_Scharr',
            'weights':[1/len(donor_cfg['reference_bands'])]*len(donor_cfg['reference_bands'])},
        reconstruction_spec={'bands':donor_cfg['bands'],'width':donor_cfg['width'],
            'depth':[1,2,3], 'sampling_variant':'UBIL',
            'input_layout':'P0','base_reference':'INTERP23_MS'},
        scharr={'kernel':'absolute_xy_/32','padding':'reflect1',
            'reference_indices':donor_cfg['reference_bands'],'comparison_domain':MASK2_POLICY},
        warp=WARP_CONTRACT,
        unified_geometry={'schema':'PANDA_SENSOR_EQ40_GEOMETRY_V1','scale':1.,
            'student_warp':warp_manifest(), 'teacher_warp':warp_manifest(),
            'base':upsampler_manifest('interp23'), 'gt_and_output_coordinates':'ORIGINAL_MS',
            'q_target':donor_cfg['reference_mode']+'_MEAN_GT',
            'q_pan':'LANCZOS4_CORRECTED_PAN','loss_mask_policy':MASK2_POLICY},
        band_reconstruction_spec=model.teacher.spec.manifest(),
        selector_receipt_sha256=receipt['receipt_sha256'], source_release=pins['source_release'])
    binding = validate_binding(binding)
    donor = LoadedDonor(model.aligner, model.teacher, binding,
        {'schema':'PANDA_SENSOR_EQ40_IMPORTED_DONOR_V1','source_config':deepcopy(donor_cfg),
         'selection':deepcopy(receipt),'checkpoint_sha256':chosen['file_sha256']})
    donor.selection_receipt = deepcopy(receipt)
    donor.assert_frozen()
    return donor


def calibration(donor, registered, folder, *, device='cpu', batch_size=48):
    """Fresh exact full-train ECDF; cache identity includes matching sensor/reference A/T."""
    folder = Path(folder);folder.mkdir(parents=True, exist_ok=True)
    path, receipt_path = folder/'full_s100.npz', folder/'full_s100.json'
    donor.assert_frozen();registered.verify()
    binding = validate_binding(donor.binding)
    if path.exists():
        # A completed immutable NPZ can survive interruption before JSON receipt.
        # Its embedded binding/content hash must still pass before reconstruction.
        receipt = _read(receipt_path) if receipt_path.exists() else None
        if receipt is not None and receipt['donor_binding'] != binding:
            raise ValueError('Calibration receipt belongs to another donor')
        stats = CalibrationStats.load(path, expected_binding=binding,
            expected_file_sha256=None if receipt is None else receipt['file_sha256'],
            expected_mask_policy=MASK2_POLICY,memory_budget_bytes=8*1024**3)
        if receipt is None:
            receipt = {'schema':'PANDA_SENSOR_EQ40_CALIBRATION_RECEIPT_V1',
                'path':str(path),'file_sha256':file_sha(path),'artifact_sha256':stats.artifact_sha256,
                'donor_binding':binding,'sample_count':len(registered.sample_ids),
                'tau_e':stats.tau_e,'recovered_from_verified_immutable_artifact':True}
            _immutable_json(receipt_path, receipt)
    else:
        if receipt_path.exists():
            raise ValueError('Calibration receipt exists without its retained artifact')
        collector = CalibrationBuilder(registered.sample_ids,binding,k=5,
            memory_budget_bytes=8*1024**3,include_condq=True,mask_policy=MASK2_POLICY)
        scharr = ReferenceSpatial(binding['sensor'],binding['reference_mode']).to(device);started = time.monotonic();seen = 0
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
        receipt = {**saved,'schema':'PANDA_SENSOR_EQ40_CALIBRATION_RECEIPT_V1',
            'donor_binding':binding,'sample_count':seen,'tau_e':stats.tau_e,
            'elapsed_seconds':time.monotonic()-started,'mask_policy':MASK2_POLICY}
        _immutable_json(receipt_path,receipt)
    stats.validate_binding(binding);stats.assert_integrity(full_hash=True)
    donor.assert_frozen();registered.verify()
    return stats.to(device,memory_budget_bytes=1024**3)


def prepare_reference(cfg,registered,root,pins,device='cpu'):
    if cfg['stage']!='stage2':raise ValueError('Only Stage2 consumes a frozen donor')
    dc,receipt,source_root,source_pins=_select(cfg,registered,root,pins)
    if cfg['server']=='s2':
        from panda_unified_campaign.assets import _load_selected as load_original
        donor=load_original(dc,receipt,registered,source_root,source_pins,device=device)
    else:
        donor=_load_selected(dc,receipt,registered,source_root,source_pins,device=device)
    folder=Path(root)/'donors'/f"RGBN_R{cfg['repeat']}"
    _immutable_json(folder/'selection.json',receipt)
    # QB repeats share the same frozen A/T and identical full-train calibration.
    cache=Path(root)/'calibration'/donor.binding['donor_checkpoint_sha256']
    stats=calibration(donor,registered,cache,device=device)
    return donor,stats
