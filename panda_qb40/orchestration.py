"""No predecessor campaign, shared donor, or cross-server approval prerequisite."""
def predecessor_receipt(root,server):
    if server not in ('s1','s3'): raise ValueError('Registered local lane required')
    return {'server':server,'dataset':'QB','predecessor_queue_gate':False,
            'cross_server_donor_transfer':False,'successor_skips':[]}
