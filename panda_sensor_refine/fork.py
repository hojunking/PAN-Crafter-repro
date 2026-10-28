"""Only fresh Students are registered; no weight or optimizer forks."""
def audit_sources(server):
    from . import plan
    return {'server':server,'fresh_only':True,'continuation_sources':[],
            'frozen_external_donor':plan.EXTERNAL_DONOR,
            'fresh_donor_receipts':{case:plan.donor_receipt_path(server,case) for case in plan.DONORS}}

def initialize_from_source(*args, **kwargs):
    raise ValueError('Sensor refinement registers no continuation source')
