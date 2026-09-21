"""Local numerical/data readiness only; no performance acceptance threshold."""
import io
import unittest

from ablr2.common import (ROOT, camp, apply_runtime_policy, source_identity, atomic_json,
                         read, read_json, object_sha, utcnow, check_deadline)


def method_checks():
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'ablr2'), 'test_*.py', top_level_dir=str(ROOT))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return dict(complete=result.wasSuccessful(), tests_run=result.testsRun, output=stream.getvalue())


def run(root=ROOT, server='s1', device='cuda', deadline_utc=None, manifest_path=None):
    import torch
    import importlib.util
    from ablr2.data import prepare_data, build_dataset
    from ablr2.model import build_model
    from ablr2.evaluation import FRMetrics
    from ablr2.plan import verify_sources, verify_lane
    sensor = verify_lane(server)
    verify_sources(root)
    runtime = apply_runtime_policy(root)
    if importlib.util.find_spec('matplotlib') is None:
        raise ValueError('matplotlib is required for complete-panel figures before admitting training')
    if device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Production ABLR2 requires local CUDA; CPU is testing only')
    release = source_identity(root)
    previous = read(camp(root,server) / 'preflight.json')
    if previous and previous['source_identity'] != release:
        raise ValueError('Numerical source/runtime changed; explicit new revision required')
    if len(release['files'].get('external/DLPan/wald_utilities.py','')) != 64:
        raise ValueError('Pinned DLPan metric dependency unavailable')
    data_path = prepare_data(root, server, deadline_utc, manifest_path)
    data = read_json(data_path)
    if previous and previous.get('dataset_manifest_sha256') != object_sha(data):
        raise ValueError('Data changed after preflight')
    if previous:
        if previous.get('complete') is not True or previous.get('server_id')!=server or previous.get('sensor')!=sensor:
            raise ValueError('Incomplete or wrong-lane preflight receipt')
        return previous
    checks = method_checks()
    atomic_json(camp(root,server) / 'method_regression.json', checks)
    if not checks['complete']: raise ValueError('ABLR2 CPU regression failed')
    fr = build_dataset(data,'fr',root=root,server=server)
    FRMetrics(fr)
    del fr
    train = build_dataset(data,'train',root=root,server=server)
    batch = train.base(0)
    model,_ = build_model(bands=train.bands,seed=0,role='T')
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(batch[4][None].to(device),batch[2][None].to(device),batch[3][None].to(device))
    if out['y'].shape != batch[0][None].shape or not torch.isfinite(out['y']).all():
        raise ValueError('Local native forward shape/finiteness failed')
    del train,model,out
    torch.cuda.empty_cache()
    check_deadline(deadline_utc)
    receipt = dict(complete=True,server_id=server,sensor=sensor,at_utc=utcnow(),
        source_identity=release,runtime=runtime,gpu=torch.cuda.get_device_name(0),
        dataset_manifest_sha256=object_sha(data),method_tests=checks['tests_run'],
        performance_gate=False,cross_server_barrier=False,real_native_forward=True)
    atomic_json(camp(root,server) / 'preflight.json',receipt)
    return receipt
