"""WV3 inputs on s3, independent of the server's former GF2/QB campaigns."""
from panda_method.edge_data import EdgeDataSources
from panda_stage2_condq.sources import RegisteredRRTrain
from ta2.evaluation import file_sha
from . import plan


def sources(server='s3'):
    cfg = plan.resolve(server, 'OURS', 1)
    catalog = plan.local_path(cfg['source_registry'])
    if file_sha(catalog) != cfg['source_registry_sha256']:
        raise ValueError('Registered WV3 data catalog changed')
    source = EdgeDataSources(catalog, root=plan.ROOT, sensor='WV3')
    for split in ('train', 'val', 'rr', 'fr'):
        source.raw(split, [0])
    return source, RegisteredRRTrain(source)
