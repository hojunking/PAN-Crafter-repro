"""Local integrity checks, not cross-server or HQNR performance gates."""
import io
import subprocess
import unittest
from pathlib import Path

from l100.common import (ROOT, apply_runtime_policy, atomic_json, camp, check_deadline,
                         object_sha, read, read_json, source_identity, utcnow)


def method_checks():
    modules = ['l100.test_plan', 'l100.test_policy', 'l100.test_training',
               'l100.test_calibration', 'l100.test_references',
               'l100.test_evaluation', 'l100.test_data', 'g20.test_evaluation']
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return dict(complete=result.wasSuccessful() and result.testsRun > 0,
                tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                detail=stream.getvalue())


def execute(root, server, deadline_utc, device='cuda', manifest_path=None):
    import torch
    from l100.data import prepare_data, build_dataset
    from g20.model import build_model
    from l100.evaluation import FRMetrics
    folder = camp(root, server)
    check_deadline(deadline_utc)
    runtime = apply_runtime_policy(root)
    if device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Production LOCAL-T P0 requires local CUDA; no CPU training fallback')
    gpu = torch.cuda.get_device_name(0)
    if runtime['policy']['expected_gpu_name_contains'] not in gpu:
        raise ValueError('LOCAL-T plan requires the assigned 5090 servers; got ' + gpu)
    release = source_identity(root)
    if len(release['files'].get('external/DLPan/wald_utilities.py', '')) != 64:
        raise ValueError('The pinned DLPan evaluation dependency is missing')
    previous = read(folder / 'preflight.json')
    if previous and previous.get('source_identity') != release:
        raise ValueError('LOCAL-T source/runtime changed after P0; do not silently migrate a live campaign')
    data_path = prepare_data(root, server, deadline_utc, manifest_path)
    data = read_json(data_path)
    # This is a real import/geometry check of the native FR evaluation path.
    fr = build_dataset(data, 'fr', root=root)
    FRMetrics(fr)
    del fr
    checks = method_checks()
    atomic_json(folder / 'method_regression.json', checks)
    if not checks['complete']:
        raise ValueError('LOCAL-T numerical/routing regression failed; see method_regression.json')
    check_deadline(deadline_utc)
    from qg40.phase_audit import phase_coordinate_checks
    atomic_json(folder / 'phase_coordinates.json', phase_coordinate_checks())
    train = build_dataset(data, 'train', root=root)
    row = train.base(0)
    teacher, _ = build_model('P0', 112, (1, 2, 3), 94001, role='T', num_bands=4)
    teacher = teacher.to(device).eval()
    with torch.no_grad():
        out = teacher(row[4][None].to(device), row[2][None].to(device), row[3][None].to(device))
    if out['y'].shape != row[0][None].shape or not torch.isfinite(out['y']).all():
        raise ValueError('Local native GF2 forward shape/finite check failed')
    del train, teacher, out
    torch.cuda.empty_cache()
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
            '--format=csv,noheader'], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        driver = 'UNAVAILABLE'
    receipt = dict(complete=True, status='LOCAL_READY', server_id=server, at_utc=utcnow(),
        source_identity=release, runtime=runtime, gpu=gpu, driver=driver,
        dataset_manifest_sha256=object_sha(data), method_tests=checks['tests_run'],
        real_native_forward=True, performance_gate=False, cross_server_barrier=False)
    atomic_json(folder / 'preflight.json', receipt)
    return receipt
