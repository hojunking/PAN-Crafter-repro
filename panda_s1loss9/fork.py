"""TUNE30 is fresh-only; no predecessor Student checkpoint is imported."""
def audit_sources(server):
    from . import plan
    plan.baseline(server)
    return {'server':server,'fresh_only':True,'continuation_sources':[],
            'donor_receipt':plan.donor_receipt_path(server)}


def initialize_from_source(*args, **kwargs):
    raise ValueError('TUNE30 forbids Student continuation and weight averaging')
