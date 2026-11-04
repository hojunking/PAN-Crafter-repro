"""Independent local consumer: no source queue completion gate or server handoff."""
def predecessor_receipt(root,server):
    if server!='s2': raise ValueError('s2 only')
    return dict(server=server,dataset='QB',predecessor_queue_gate=False,
                cross_server_donor_transfer=False,stage1_training_runs=0)
