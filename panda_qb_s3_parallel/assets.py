"""Fresh same-server donor; byte identity and calibration binding retained."""
from copy import deepcopy
import json
from pathlib import Path
import torch
from directshift48.reporting import atomic_json
from ta2.evaluation import file_sha
from panda_unified.core import upsampler_manifest,warp_manifest
from panda_method.calibration import state_dict_sha256
from panda_sensor_eq40.calibration import validate_binding
from panda_stage2_condq.donor import LoadedDonor
from panda_sensor_eq40.assets import sources as qb_sources,calibration
from panda_donor_read_guard.guard import fingerprint
from .geometry import MASK2_POLICY,WARP_CONTRACT
from .model import Stage1
from . import plan

def sources(server='s3'):
    plan.sensor_for(server)
    # Sensor registry, stored original QB LMS, and repaired LRMS match s2 exactly.
    return qb_sources('s2')

def select(root,server='s3'):
    from .evaluation import verify_record
    root=Path(root);dc=plan.donor_config(server);directory=root/'runs'/dc['run_id']
    state=json.loads((root/'campaign_state.json').read_text())
    job=next(r for r in state['runs'] if r['run_id']==dc['run_id'])
    if job['status']!='COMPLETED' or job['step']!=50000:raise ValueError('Complete this server Stage1 first')
    rows=[]
    for step in dc['evaluation_steps']:
        path=directory/f'evaluation_{step:07d}.json';e=json.loads(path.read_text());verify_record(e,dc)
        meta=e['checkpoint'];ck=plan.local_path(meta['path'])
        if (ck.parent!=directory/'models' or file_sha(ck)!=meta['file_sha256']
            or meta!=json.loads(ck.with_suffix('.json').read_text())
            or meta['step']!=step or meta['model_sha256']!=e['model_sha256']):raise ValueError('Local donor evidence mismatch')
        rows.append(dict(step=step,checkpoint_path=str(ck.relative_to(root)),file_sha256=meta['file_sha256'],
            model_sha256=meta['model_sha256'],evaluation_sha256=file_sha(path),
            rr=e['rr']['metrics'],fr=e['fr']['metrics'],fr_corrected_pan=e['fr_corrected_pan']['metrics']))
    chosen=plan.pick_donor(rows)
    receipt=dict(schema='PANDA_QB_S3_PARALLEL_DONOR_V1',server=server,sensor='QB',reference_mode='RGBN',repeat=1,
        donor_run_id=dc['run_id'],selected=chosen,candidates=rows,source_config_sha256=dc['config_sha256'],
        uses_test_for_selection=True,selector='ERGAS_LE_3P5_MAX_PSNR_CORRECTED_H_ELSE_MIN_E',
        thresholds_are_admission_gates=False,selection_reference='CORRECTED_PAN')
    receipt['receipt_sha256']=plan.digest(receipt)
    return dc,receipt

def prepare_reference(cfg,registered,root,pins,device='cpu'):
    plan.validate(cfg);dc,receipt=select(root,cfg['server'])
    path=Path(root)/'donor/selection.json'
    if path.exists() and json.loads(path.read_text())!=receipt:raise ValueError('Frozen donor changed')
    atomic_json(path,receipt)
    donor=_load_selected(dc,receipt,registered,root,pins,device=device)
    stats=calibration(donor,registered,Path(root)/'donor/calibration',device=device)
    return donor,stats

def _load_selected(donor_cfg, receipt, registered, root, pins, *, device):
    from .training import model_hash
    chosen = receipt['selected'];path = Path(root)/chosen['checkpoint_path']
    before = fingerprint(path)
    if file_sha(path) != chosen['file_sha256']:
        raise ValueError('Selected donor checkpoint SHA mismatch')
    value = torch.load(path, map_location='cpu', weights_only=False)
    if fingerprint(path) != before or file_sha(path) != chosen['file_sha256']:
        raise ValueError('Donor checkpoint changed while loading')
    complete_schema = 'PANDA_QB_S3_PARALLEL_COMPLETE_CHECKPOINT_V1'
    evaluated_schema = 'PANDA_QB_S3_PARALLEL_EVALUATED_MODEL_V1'
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
        unified_geometry={'schema':'PANDA_QB_S3_PARALLEL_GEOMETRY_V1','scale':1.,
            'student_warp':warp_manifest(), 'teacher_warp':warp_manifest(),
            'base':upsampler_manifest('interp23'), 'gt_and_output_coordinates':'ORIGINAL_MS',
            'q_target':donor_cfg['reference_mode']+'_MEAN_GT',
            'q_pan':'LANCZOS4_CORRECTED_PAN','loss_mask_policy':MASK2_POLICY},
        band_reconstruction_spec=model.teacher.spec.manifest(),
        selector_receipt_sha256=receipt['receipt_sha256'], source_release=pins['source_release'])
    binding = validate_binding(binding)
    donor = LoadedDonor(model.aligner, model.teacher, binding,
        {'schema':'PANDA_QB_S3_PARALLEL_IMPORTED_DONOR_V1','source_config':deepcopy(donor_cfg),
         'selection':deepcopy(receipt),'checkpoint_sha256':chosen['file_sha256']})
    donor.selection_receipt = deepcopy(receipt)
    donor.assert_frozen()
    return donor
