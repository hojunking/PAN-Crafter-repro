"""Independent full-horizon local Stage1 selection proof, without GPU work."""
from pathlib import Path
from . import registry as G
from .prerequisites import read, sha


def verify(root):
    from panda_s1loss9 import plan
    from panda_s1loss9.selection import validate_receipt
    from panda_s1loss9.donor import complete_history
    root = Path(root)
    folder = root / G.OUTPUT / G.CAMPAIGN_ID / 's3/QB'
    cfg = plan.resolve_case('s3', G.DONOR_CASE)
    directory = folder / 'runs' / cfg['run_id'] / 'attempt_01'
    receipt_path = folder / 'donors' / G.DONOR_CASE / 'rre_selection.json'
    receipt = validate_receipt(read(receipt_path))
    history = complete_history(sorted((directory / 'models').glob('step_*.pt')), server='s3')
    selected = receipt['selected']
    if (receipt['server'] != 's3' or selected['source_config']['seed'] != G.STAGE1_SEED
            or selected['source_config']['case'] != G.DONOR_CASE
            or selected['source_config']['campaign_id'] != G.CAMPAIGN_ID
            or history != receipt['source_history']):
        raise ValueError('Selected donor is not the complete independently trained local Stage1')
    path = Path(selected['checkpoint_path'])
    if path.is_absolute() and str(path).startswith('/workspace/'):
        path = root / path.relative_to('/workspace')
    if path.resolve() != (directory / 'models' / f"step_{selected['completed_update']:07d}.pt").resolve():
        raise ValueError('Selected donor path escapes its local run')
    if sha(path) != selected['file_sha256']:
        raise ValueError('Selected local donor checkpoint changed')
    return {'schema': 'QB_DEEPEN_LOCAL_STAGE1_SELECTED_V1', 'server': 's3', 'seed': G.STAGE1_SEED,
            'source_run_id': cfg['run_id'], 'completed_horizon': 50000,
            'selected_step': selected['completed_update'], 'checkpoint_sha256': selected['file_sha256'],
            'selection_receipt_sha256': receipt['receipt_sha256'], 'external_donor_reused': False}
