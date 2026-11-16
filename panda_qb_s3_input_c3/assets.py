"""Verify the completed local Stage1 and load only its predeclared34K checkpoint."""
import json
from pathlib import Path
from directshift48.reporting import atomic_json
from ta2.evaluation import file_sha
from panda_qb_s3_parallel import assets as old
from panda_qb_s3_parallel.evaluation import verify_record
from panda_donor_read_guard.guard import protect
from . import plan


def sources(server):
    plan.sensor_for(server)
    return old.sources('s3')


def read(path): return json.loads(Path(path).read_text())


def verify_receipt(receipt):
    fixed = plan.FIXED; selected = receipt.get('selected', {})
    expected = {'schema':'PANDA_QB_S3_PARALLEL_DONOR_V1', 'server':'s3', 'sensor':'QB',
        'reference_mode':'RGBN', 'donor_run_id':fixed['run_id'],
        'source_config_sha256':fixed['source_config_sha256'], 'receipt_sha256':fixed['receipt_sha256']}
    if (any(receipt.get(k) != v for k,v in expected.items())
            or plan.digest({k:v for k,v in receipt.items() if k!='receipt_sha256'}) != fixed['receipt_sha256']
            or any(selected.get(k) != fixed[v] for k,v in
                [('step','step'),('file_sha256','checkpoint_file_sha256'),
                 ('model_sha256','model_sha256'),('evaluation_sha256','evaluation_sha256')])):
        raise ValueError('Must use the existing same-server RGBN donor34K and receipt')


def selected_reference(server, registered, pins):
    plan.sensor_for(server); root = plan.donor_root(server); cfg = plan.donor_config(server)
    source_pins = read(root / 'admission.json')
    for key in ('evaluator','train_inventory','split_manifests'):
        if source_pins[key] != pins[key]: raise ValueError('Donor data/evaluator differs: ' + key)
    if registered.verify().manifest() != source_pins['train_inventory']:
        raise ValueError('QB original stored LMS / repaired MS inventory changed')
    state = read(root / 'campaign_state.json')
    jobs = [r for r in state['runs'] if r['run_id'] == plan.FIXED['run_id']]
    if (state.get('campaign_id') != cfg['campaign_id'] or state.get('server') != server
            or len(jobs)!=1 or jobs[0]['status']!='COMPLETED' or jobs[0]['step']!=50000):
        raise ValueError('Existing s3 QB Stage1 must have completed50K; no Stage1 is queued here')
    directory = root / 'runs' / cfg['run_id']; step = plan.FIXED['step']
    if read(directory / 'config.json') != cfg or cfg['config_sha256'] != plan.FIXED['source_config_sha256']:
        raise ValueError('Historical Stage1 recipe differs')
    receipt = read(root / 'donor/selection.json'); verify_receipt(receipt)
    path = directory / f'evaluation_{step:07d}.json'; record = read(path)
    verify_record(record, cfg)
    checkpoint = directory / 'models' / f'step_{step:07d}.pt'
    meta = read(checkpoint.with_suffix('.json'))
    if (file_sha(path) != plan.FIXED['evaluation_sha256'] or record['step'] != step
            or record['eval_protocol_id'] != plan.digest(source_pins['evaluator'])
            or record['checkpoint'] != meta or meta['config_sha256'] != cfg['config_sha256']
            or meta['step'] != step or meta['model_sha256'] != plan.FIXED['model_sha256']
            or record['model_sha256'] != plan.FIXED['model_sha256']
            or meta['file_sha256'] != plan.FIXED['checkpoint_file_sha256']
            or file_sha(checkpoint) != plan.FIXED['checkpoint_file_sha256']
            or plan.local_path(meta['path']).resolve() != checkpoint.resolve()
            or (root / receipt['selected']['checkpoint_path']).resolve() != checkpoint.resolve()):
        raise ValueError('Pinned s3 QB34K checkpoint/evaluation SHA mismatch')
    return cfg, receipt, root, source_pins


def prepare_reference(cfg, registered, root, pins, device='cpu'):
    plan.validate(cfg)
    dc, receipt, source_root, source_pins = selected_reference('s3', registered, pins)
    donor = protect(old._load_selected)(dc, receipt, registered, source_root, source_pins, device=device)
    folder = Path(root) / 'donor'; saved = folder / 'selection.json'
    if saved.exists() and read(saved) != receipt: raise ValueError('New queue donor receipt changed')
    atomic_json(saved, receipt)
    previous = source_root / 'donor/calibration'
    cache = previous if all((previous / f).exists() for f in ('full_s100.npz','full_s100.json')) else folder/'calibration'
    stats = old.calibration(donor, registered, cache, device=device)
    atomic_json(folder / 'calibration_source.json', {'path':str(cache),
        'reused_existing':cache==previous,'artifact_sha256':stats.artifact_sha256,
        'donor_checkpoint_sha256':plan.FIXED['checkpoint_file_sha256']})
    return donor, stats
