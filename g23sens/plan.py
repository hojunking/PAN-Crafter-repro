"""Pinned supplied case generator; no manually maintained second case catalog."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import re
import yaml

from g23sens.common import ROOT, camp, run_dir, object_sha, sha256, read_json, source_identity

BUNDLE = 'research_log/PANDA_G23_SENS_S45_UNLIMITED_2026-09-23'
MANIFEST_SHA256 = '759a54764ec3117cc0c7197b7079bc82778cfa115fa078a12c74ae500fca6243'
BASE_CONFIG = 'config/PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S1234_FRESH50_v2.yaml'
BASE_CONFIG_SHA256 = 'eb162d7a8c751c1ef590be266dd762ddcd609c824174070d4616e7033ae091c8'


def verify_sources(root=ROOT):
    root = Path(root); bundle = root/BUNDLE
    manifest = bundle/'SHA256SUMS.txt'
    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError('Supplied G23 plan manifest changed')
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip('*')
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha256(bundle/path) != digest:
            raise ValueError('Supplied G23 bundle differs: '+name)
    if sha256(root/BASE_CONFIG) != BASE_CONFIG_SHA256:
        raise ValueError('Legacy G23 reference config changed')
    blobs = {'kdv/qrecon.py':'7323bbd1c8b9863a000f33b913779386466d2e03',
        'kdv/losses_rec.py':'8dfb86629c7ed98f39230c98f998a290fba9899d',
        'kdv/registry.py':'7c9566c9e31dcf6488218e888a9c0aed254576f8'}
    for name, expected in blobs.items():
        raw = (root/name).read_bytes()
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != expected:
            raise ValueError('Pinned G23 primitive changed: '+name)
    return True


verify_sources()
_spec = importlib.util.spec_from_file_location('_g23sens_pinned_casegen', ROOT/BUNDLE/'tools/casegen.py')
_generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generator)
CAMPAIGN = CAMPAIGN_ID = _generator.CAMPAIGN
TEACHER_SHA256 = _generator.TEACHER_SHA256
Q0, TAU0 = _generator.Q0, _generator.TAU0
make_case, cycle_cases = _generator.make_case, _generator.cycle_cases
order_for, seed_for = _generator.order_for, _generator.seed_for
iter_cases, next_cursor = _generator.iter_cases, _generator.next_cursor
validate_case, verify_receipt = _generator.validate_case, _generator.verify_receipt


def case_for(run_id):
    match = re.fullmatch(r'SENS_G23_WV3_(s[45])_C(\d{6,})_([A-Z0-9]+)_S(\d+)_F50K_v1', run_id)
    if not match: raise ValueError('Not a G23 SENS run ID')
    server, cycle, code, _seed = match.groups()
    case = make_case(server, int(cycle), code)
    if case['run_id'] != run_id: raise ValueError('Run ID does not match canonical seed/cycle')
    return case


def build_config(case, root=ROOT, bindings=None, attempt=0):
    validate_case(case); verify_sources(root); root = Path(root)
    cfg = yaml.safe_load((root/BASE_CONFIG).read_text())
    for dotted, value in case['backend_overrides'].items():
        node = cfg; parts = dotted.split('.')
        for part in parts[:-1]: node = node.setdefault(part,{})
        node[parts[-1]] = deepcopy(value)
    cfg.update(work_dir=str(run_dir(case,root,attempt).resolve()), trainer='g23sens',
               mixed_precision='no', resume=False)
    for key in ('select_on','fr_select_indices','weights','parent_run','budget'):
        cfg.pop(key, None)
    kdv = cfg['kdv']
    for key in ('budget','qrc24','control_runs','parent_campaign_id','lineage_campaign_ids'):
        kdv.pop(key,None)
    kdv.update(campaign_id=CAMPAIGN, case_id=case['case_id'], version='v1',
        experiment_branch_id=CAMPAIGN, baseline_run=case['local_baseline_run_id'])
    # This legacy registry field is never a checkpoint selector in this trainer.
    kdv['select'] = dict(primary='best_hqnr',secondary=['best_rr_val','last'],retain_all_candidates=True)
    if bindings is not None:
        if bindings['server'] != case['server'] or bindings['campaign_id'] != CAMPAIGN:
            raise ValueError('Runtime bindings belong to a different campaign/server')
        for split,key in dict(train='train_feeder_args',val='val_feeder_args',
                             rr='test_reduced_feeder_args',fr='test_full_feeder_args').items():
            cfg[key]['dataroot'] = bindings['data'][split]['path']
        kdv['teacher']['run'] = bindings['teacher']['run_path']
        kdv['donor']['source'] = str(Path(bindings['teacher']['path']).parent)
        kdv['qrecon']['asset'] = bindings['cue']['json_path']
    else:
        prefix = '/home/knuvi/Desktop/song/PAN-Crafter/'
        for key in ('train_feeder_args','val_feeder_args','test_reduced_feeder_args','test_full_feeder_args'):
            cfg[key]['dataroot'] = str(root/cfg[key]['dataroot'].removeprefix(prefix))
    cfg['g23sens'] = dict(case=deepcopy(case),
        bindings_path=str(camp(root,case['server'])/'runtime_bindings.json'),
        binding_sha256=object_sha(bindings) if bindings is not None else None,
        source_identity=source_identity(root) if bindings is not None else None,
        attempt=attempt, primary_selection='EXACT_50000',secondary_selection='RR_VAL_ERGAS_MIN',
        candidate_grid='GRID1010_50K_v1',raw_max_selection=False)
    return cfg


def validate_config(cfg, root=ROOT, bindings=None, require_bound=False):
    meta = cfg['g23sens']; case = meta['case']; validate_case(case)
    if require_bound and meta.get('binding_sha256') is None:
        raise ValueError('Actual verified runtime bindings required')
    if bindings is None and meta.get('binding_sha256') is not None:
        bindings = read_json(meta['bindings_path'])
    if cfg != build_config(case,root,bindings,meta['attempt']):
        raise ValueError('Resolved config differs from the pinned one-factor case')
    return case
