"""Install a registry in this dedicated process; never edit pinned focus sources.

The previous campaign remains importable unchanged in all other processes.
Install before importing its runtime/reporting modules. Module globals are used
by the existing validated training, calibration, evaluation and Sheet machinery.
"""
from copy import deepcopy
from pathlib import Path
import sys

from . import registry as G

_installed = False


def install():
    global _installed
    if _installed:
        from panda_sensor_focus import plan
        return plan
    unsafe = [name for name in sys.modules if name.startswith('panda_sensor_focus.')
              and name not in ('panda_sensor_focus.contract',)]
    if unsafe:
        raise RuntimeError('Use a dedicated GF2 process; focus modules already loaded: ' + ','.join(unsafe))
    from panda_sensor_focus import contract as C
    C.CASES = G.cases()
    C.REGISTERED_RUNS = {'s3': G.RUNS}
    C.CAMPAIGN_ID = C.NEXT_CAMPAIGN_ID = C.CONTROL_CAMPAIGN = G.CAMPAIGN_ID
    C.METHOD_REVISION = G.METHOD_REVISION
    from panda_sensor_focus import plan as P
    original_default, original_resolve = P.default_config, P.resolve_case

    def default_config(server):
        if server != 's3':
            raise ValueError('This followup is registered only for s3/GF2')
        cfg = original_default(server)
        cfg.update(joint_target=deepcopy(G.TARGET), after_campaign='FINISH_GF2_PSNR_QUEUE',
                   publication_target={'HQNR': .9645, 'ERGAS': .522},
                   additional_target={'PSNR': 45.08})
        return cfg

    def resolve_case(server, case, replica='R1', *, attempt=1):
        cfg = original_resolve(server, case, replica, attempt=attempt)
        cfg.update(run_display=f'GF2_PSNR4508 {case} {replica}', joint_target=deepcopy(G.TARGET),
                   psnr_min=45.08, question='GF2 H>=.9645 E<=.522 P>=45.08; fixed donor, fresh Student, one knob vs C60LR150')
        return cfg

    P.default_config, P.resolve_case = default_config, resolve_case
    from panda_sensor_focus import runtime, orchestration
    original_pin = runtime.code_pin

    def code_pin():
        from ta2.evaluation import file_sha
        files = [p for p in (P.ROOT / 'panda_gf2_psnr').rglob('*.py') if 'tests' not in p.parts]
        return {**original_pin(), **{str(p.relative_to(P.ROOT)): file_sha(p) for p in files}}

    def predecessor_receipt(root, server):
        from .prerequisites import evidence
        if server != 's3':
            raise ValueError('GF2-only predecessor')
        return evidence(Path(root), expected_code=original_pin())

    runtime.code_pin = code_pin
    orchestration.predecessor_receipt = predecessor_receipt
    _installed = True
    return P
