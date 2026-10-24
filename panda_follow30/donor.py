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


def load_donor(path, expected_sha256, registered, *, server='s1', device='cpu',
               include_teacher=True, selector_receipt=None):
    if selector_receipt is None:
        raise ValueError('Original validation-only selection receipt is mandatory')
    validate_inventory(selector_receipt['validation_inventory'], registered.sources)
    donor = original_load_donor(path, expected_sha256, registered, server=server,
        device=device, include_teacher=include_teacher, selector_receipt=selector_receipt)
    donor.checkpoint_metadata['consumer_campaign_id'] = CAMPAIGN_ID
    return donor
