"""Local numeric/data readiness, never a cross-server performance barrier."""
import io
from pathlib import Path
import unittest

from g20.common import (ROOT, atomic_json, camp, check_deadline, object_sha,
                        read, read_json, source_identity, utcnow)


def method_checks(root=ROOT):
    """Run installed contract regression tests, recording their exact source identity."""
    modules = ['g20.test_training', 'g20.test_evaluation', 'g20.test_data', 'g20.test_plan', 'g20.test_policy']
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return dict(complete=result.wasSuccessful() and result.testsRun > 0,
                tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                detail=stream.getvalue(), source_identity=source_identity(root),
                scope='C4/C8 numerical/metric/data CPU regression; not a performance gate')


def execute(root, server, spec_path, deadline_utc, device='cuda'):
    import torch
    from g20.data import prepare_data, build_dataset
    from g20.model import build_model
    from g20.plan import SensorSpec
    check_deadline(deadline_utc)
    directory = camp(root, server)
    previous = read(directory / 'preflight.json')
    release = source_identity(root)
    external_sha = release.get('files', {}).get('external/DLPan/wald_utilities.py')
    if not isinstance(external_sha, str) or len(external_sha) != 64:
        raise ValueError('Verified DLPan evaluation dependency is required for LOCAL_READY')
    bound = read_json(spec_path)
    # P0 independent analytic coordinate checks; historical phase wording is
    # not accepted as a pixel-unit test. Receipts do not modify dataset bytes.
    from qg40.phase_audit import phase_coordinate_checks
    atomic_json(directory / 'phase_coordinates.json', phase_coordinate_checks())
    if previous:
        if previous.get('source_identity') != release or previous.get('sensor_spec_sha256') != object_sha(bound):
            raise ValueError('Existing local preflight source/sensor binding changed')
        if previous.get('complete'):
            # Reader preparation revalidates source/cache bytes; this is not a bypass.
            prepared = prepare_data(root, server, SensorSpec.from_dict(bound), deadline_utc=deadline_utc)
            if isinstance(prepared, (str, Path)):
                prepared = read_json(prepared)
            if (object_sha(prepared) != previous.get('dataset_manifest_sha256')
                    or read_json(directory / 'dataset_manifest.json') != prepared):
                raise ValueError('Cached local readiness dataset manifest changed')
            from g20.input_distribution import record_input_distributions
            record_input_distributions(root, server, prepared, deadline_utc=deadline_utc)
            return previous
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Local CUDA device is unavailable; no CPU production fallback')
    spec = SensorSpec.from_dict(bound)
    data = prepare_data(root, server, spec, deadline_utc=deadline_utc)
    data_path = directory / 'dataset_manifest.json'
    if isinstance(data, (str, Path)):
        data = read_json(data)
    if data_path.is_file() and read_json(data_path) != data:
        raise ValueError('Prepared dataset manifest differs from existing campaign data')
    atomic_json(data_path, data)
    from g20.evaluation import FRMetrics
    FRMetrics(build_dataset(data, 'fr', root=root))  # Real full512 references and external evaluator import.
    checks = method_checks(root)
    atomic_json(directory / 'method_regression.json', checks)
    if not checks['complete']:
        raise ValueError('Local C4/C8 regression failed; see method_regression.json')
    check_deadline(deadline_utc)
    dataset = build_dataset(data, 'train', root=root)
    row = dataset.base(0)
    teacher, _ = build_model('P0', 112, (1, 2, 3), 1234, role='T', num_bands=4)
    teacher = teacher.to(device).eval()
    with torch.no_grad():
        out = teacher(row[4][None].to(device), row[2][None].to(device), row[3][None].to(device))
    if out['y'].shape != row[0][None].shape or not torch.isfinite(out['y']).all():
        raise ValueError('Real sensor C4 native model smoke failed')
    del teacher, out
    if device == 'cuda':
        torch.cuda.empty_cache()
    receipt = dict(complete=True, status='LOCAL_READY', server_id=server, sensor=spec.sensor,
                   at_utc=utcnow(), source_identity=release, sensor_spec_sha256=object_sha(bound),
                   dataset_manifest_sha256=object_sha(data), method_tests=checks['tests_run'],
                   real_native_forward=True, device=device, performance_gate=False,
                   cross_server_barrier=False)
    atomic_json(directory / 'preflight.json', receipt)
    # P1 distribution evidence is separate and nonblocking. Incomplete analysis
    # must neither revoke LOCAL_READY nor gate an unrelated sensor/reference.
    from g20.input_distribution import record_input_distributions
    record_input_distributions(root, server, data, deadline_utc=deadline_utc)
    return receipt
