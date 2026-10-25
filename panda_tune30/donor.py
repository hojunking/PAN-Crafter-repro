"""Strict same-server Mature30 receipt reuse, no retraining or fallback."""
import json
from pathlib import Path
from copy import deepcopy

from panda_mature30.donor import load_donor as original_load_donor
from panda_mature30.selection import validate_receipt, validate_inventory
from ta2.evaluation import file_sha
from .contract import CAMPAIGN_ID


def source_receipt(server, root):
    from .plan import donor_receipt_path
    path = Path(root) / donor_receipt_path(server)
    receipt = validate_receipt(json.loads(path.read_text()))
    if receipt['server'] != server:
        raise ValueError('Only same-server RRE-selected Mature30 donor is allowed')
    return path, receipt


def inspect_source_artifact(receipt, root):
    """Check the original receipt's bytes on the host without rewriting it.

    Receipts were recorded inside /workspace. Full donor tensor/config checks
    still run through load_donor inside that same container path at launch.
    """
    receipt = validate_receipt(receipt)
    original = Path(receipt['selected']['checkpoint_path'])
    if original.is_absolute() and original.parts[:2] == ('/', 'workspace'):
        local = Path(root) / original.relative_to('/workspace')
    else:
        local = original if original.is_absolute() else Path(root) / original
    if not local.is_file() or file_sha(local) != receipt['selected']['file_sha256']:
        raise ValueError('Selected donor artifact missing or SHA mismatch')
    return local


def load_donor(path, expected_sha256, registered, *, server='s1', device='cpu',
               include_teacher=True, selector_receipt=None):
    if selector_receipt is None:
        raise ValueError('Original validation-only selection receipt is mandatory')
    validate_inventory(selector_receipt['validation_inventory'], registered.sources)
    donor = original_load_donor(path, expected_sha256, registered, server=server,
        device=device, include_teacher=include_teacher, selector_receipt=selector_receipt)
    donor.checkpoint_metadata['consumer_campaign_id'] = CAMPAIGN_ID
    return donor
