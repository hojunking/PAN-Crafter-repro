"""Configure the unchanged focus engine in a dedicated QB followup process.

No existing source file is modified. Install before importing focus modules;
other processes retain their original registry, source pin and live queues.
"""
from copy import deepcopy
from pathlib import Path
import sys

from . import registry as G

_installed = False


def own_files(root):
    directory = Path(root) / 'panda_qb_refit'
    return sorted([p for p in directory.rglob('*.py') if 'tests' not in p.parts] +
                  [directory / 'experiment_manifest.json', directory / 'cases_s1.csv'])


def install():
    global _installed
    if _installed:
        from panda_sensor_focus import plan
        return plan
    unsafe = [name for name in sys.modules if name.startswith('panda_sensor_focus.')
              and name not in ('panda_sensor_focus.contract',)]
    if unsafe:
        raise RuntimeError('Use a dedicated QB process; focus modules already loaded: ' + ','.join(unsafe))
    from panda_sensor_focus import contract as C
    C.CASES = G.cases()
    C.REGISTERED_RUNS = {'s1': G.RUNS}
    C.CAMPAIGN_ID = C.NEXT_CAMPAIGN_ID = C.CONTROL_CAMPAIGN = G.CAMPAIGN_ID
    C.METHOD_REVISION = G.METHOD_REVISION
    from panda_sensor_focus import plan as P
    original_default, original_resolve = P.default_config, P.resolve_case

    def default_config(server):
        if server != 's1':
            raise ValueError('This followup is registered only for s1/QB')
        cfg = original_default(server)
        cfg.update(joint_target=deepcopy(G.TARGET), after_campaign='FINISH_QB_REFIT_QUEUE',
                   matched_seed_controls=deepcopy(G.CONTROLS), comparisons=deepcopy(G.CHANGES),
                   queue_change_policy='USER_REQUESTED_REVIEW_THEN_VERSIONED_CHECKPOINT_BOUNDARY_MIGRATION')
        return cfg

    def resolve_case(server, case, replica='R1', *, attempt=1):
        cfg = original_resolve(server, case, replica, attempt=attempt)
        control = deepcopy(G.CONTROLS[replica])
        control.update(server='s1', fresh_student_no_weights_reused=True,
                       same_exact_donor_checkpoint=True, student_seed_differs_from_control=False,
                       changes=deepcopy(G.CHANGES[case]))
        cfg.update(run_display=f'QB_REFIT {case} {replica}', joint_target=deepcopy(G.TARGET),
                   comparison_control=control,
                   question='QB H>=.9315 E<=3.5; fixed exact donor, fresh matched-seed Student: ' + G.CHANGES[case]['purpose'])
        return cfg

    P.default_config, P.resolve_case = default_config, resolve_case
    from panda_sensor_focus import runtime, orchestration
    original_pin = runtime.code_pin

    def code_pin():
        from ta2.evaluation import file_sha
        return {**original_pin(), **{str(p.relative_to(P.ROOT)): file_sha(p) for p in own_files(P.ROOT)}}

    def predecessor_receipt(root, server):
        from .prerequisites import evidence
        if server != 's1':
            raise ValueError('QB-only predecessor')
        return evidence(Path(root), expected_code=original_pin())

    runtime.code_pin = code_pin
    orchestration.predecessor_receipt = predecessor_receipt
    _installed = True
    return P
