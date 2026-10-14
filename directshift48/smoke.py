"""Reproducible CPU contracts, deliberately separate from GPU throughput admission."""
import unittest


def verify_method_contracts():
    """Disposable CPU numerical gate, also run inside the 48h preflight clock."""
    import io
    import torch
    previous_threads = torch.get_num_threads()
    output = io.StringIO()
    try:
        suites = unittest.TestLoader().loadTestsFromNames([
            'directshift48.test_method', 'directshift48.test_training',
            'directshift48.test_plan',
        ])
        result = unittest.TextTestRunner(stream=output, verbosity=1).run(suites)
        report = {'status': 'COMPLETE' if result.wasSuccessful() else 'FAILED',
                  'tests_run': result.testsRun, 'failures': len(result.failures),
                  'errors': len(result.errors), 'output': output.getvalue(),
                  'scope': 'SYNTHETIC_CPU_NUMERICAL_CONTRACTS_NOT_GPU_THROUGHPUT'}
        return report
    finally:
        torch.set_num_threads(previous_threads)


def run_smoke(*, verbose=False):
    suites = unittest.TestLoader().loadTestsFromNames([
        'directshift48.test_method', 'directshift48.test_training',
        'directshift48.test_plan', 'directshift48.test_evaluation',
        'directshift48.test_diagnostics', 'directshift48.test_runtime',
        'directshift48.test_reporting', 'directshift48.test_publication',
        'directshift48.test_launchers',
        'directshift48.test_infra_integration',
    ])
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suites)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(run_smoke(verbose=True))
