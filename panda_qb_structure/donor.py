"""Read authentic local selection receipts; never transfer or relabel Stage1."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import json
from ta2.evaluation import file_sha
from . import registry as G

ROOT = Path(__file__).resolve().parents[1]


def local(path, root=ROOT):
    path = Path(path)
    return Path(root) / path.relative_to('/workspace') if path.parts[:2] == ('/', 'workspace') else (
        path if path.is_absolute() else Path(root) / path)


def receipt_path(server, root=ROOT):
    source = G.SOURCES[server]
    return (Path(root) / 'work_dir' / source['package'] / source['campaign_id'] /
            server / 'QB/donors' / G.DONOR_CASE / 'rre_selection.json')


def verify_provenance(receipt, server, root=ROOT):
    """Additional local-origin checks after the source's mathematical selector validation."""
    source = G.SOURCES[server]; selected = receipt['selected']; cfg = selected['source_config']
    expected_root = receipt_path(server, root).parents[2]
    if (receipt.get('server') != server or receipt.get('campaign_id') != source['campaign_id']
            or cfg.get('server') != server or cfg.get('dataset') != 'QB'
            or cfg.get('campaign_id') != source['campaign_id'] or cfg.get('case') != G.DONOR_CASE
            or cfg.get('seed') != source['seed'] or cfg.get('phase') != 'donor'
            or cfg.get('planned_horizon') != 50000 or cfg.get('lambda_abs') != 2.
            or cfg.get('lambda_epsilon') != .03 or cfg.get('epsilon_radius_rr', 3.) != 3.):
        raise ValueError('Donor is not the registered independently trained local Stage1')
    expected = expected_root / 'runs' / cfg['run_id'] / 'attempt_01/models' / f"step_{selected['completed_update']:07d}.pt"
    if local(selected['checkpoint_path'], root).resolve() != expected.resolve():
        raise ValueError('Selected checkpoint is outside its own local Stage1 run')
    return selected


@lru_cache(maxsize=4)
def _metadata(server, root):
    source = {**G.SOURCES[server], 'lambda_abs': 2., 'lambda_epsilon': .03, 'epsilon_radius_rr': 3.,
              'selected_step': None, 'checkpoint_sha256': None, 'receipt_sha256': None}
    path = receipt_path(server, root)
    if path.exists():
        receipt = json.loads(path.read_text()); selected = verify_provenance(receipt, server, root)
        source.update(selected_step=selected['completed_update'], checkpoint_sha256=selected['file_sha256'],
                      receipt_sha256=receipt['receipt_sha256'])
    return source


def metadata(server, root=ROOT):
    # A catalog may be rendered on the other server. Launch/check requires files.
    return deepcopy(_metadata(server, str(Path(root).resolve())))


def validate_source(server, root=ROOT, *, verify_checkpoint=True):
    from panda_s1loss9.selection import validate_receipt
    from panda_s1loss9.donor import complete_history
    receipt = validate_receipt(json.loads(receipt_path(server, root).read_text()))
    selected = verify_provenance(receipt, server, root)
    checkpoint = local(selected['checkpoint_path'], root)
    state = json.loads((checkpoint.parent.parent / 'run_state.json').read_text())
    if (state.get('status') != 'COMPLETED' or state.get('step') != 50000
            or state.get('run_id') != selected['source_config']['run_id']):
        raise ValueError('Own Stage1 must have completed its full selection horizon')
    if verify_checkpoint and (not checkpoint.is_file() or file_sha(checkpoint) != selected['file_sha256']):
        raise ValueError('Selected local Stage1 checkpoint is missing or changed')
    # The source receipt already contains all per-patch RRE evidence; also match
    # its complete retained candidate inventory, without rerunning selection.
    if verify_checkpoint:
        history = complete_history(sorted(checkpoint.parent.glob('step_*.pt')), server=server)
        if history != receipt['source_history']:
            raise ValueError('Local full50K history differs from the selected source receipt')
    source = metadata(server, root)
    if (source['checkpoint_sha256'] != selected['file_sha256']
            or source['receipt_sha256'] != receipt['receipt_sha256']):
        raise ValueError('Local donor changed since configuration resolution')
    return receipt


def load_donor(path, expected_sha256, registered, *, server='s1', device='cpu',
               selector_receipt=None, include_teacher=True):
    from panda_s1loss9.donor import load_donor as original
    receipt = validate_source(server, verify_checkpoint=False)
    selected = receipt['selected']
    if (selector_receipt != receipt or expected_sha256 != selected['file_sha256']
            or local(path).resolve() != local(selected['checkpoint_path']).resolve()):
        raise ValueError('Caller donor differs from the local validation-selected source')
    return original(path, expected_sha256, registered, server=server, device=device,
                    selector_receipt=receipt, include_teacher=include_teacher)


def verify(root, server):
    receipt = validate_source(server, root)
    selected = receipt['selected']
    return {'schema': 'QB_STRUCTURE_LOCAL_STAGE1_V1', 'server': server, 'dataset': 'QB',
            'source_campaign_id': receipt['campaign_id'], 'source_run_id': selected['source_run_id'],
            'seed': selected['source_config']['seed'], 'selected_step': selected['completed_update'],
            'checkpoint_sha256': selected['file_sha256'], 'receipt_sha256': receipt['receipt_sha256'],
            'full_selection_horizon': 50000, 'cross_server_transfer': False, 'stage1_retrained': False}
