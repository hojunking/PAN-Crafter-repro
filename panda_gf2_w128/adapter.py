"""Process-local registry extension; original running campaigns stay unchanged."""
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
    unsafe = [n for n in sys.modules if n.startswith('panda_sensor_focus.')
              and n != 'panda_sensor_focus.contract']
    if unsafe:
        raise RuntimeError('Use a dedicated GF2 W128 process: ' + ', '.join(unsafe))
    from panda_sensor_focus import contract as C
    C.CASES = G.cases()
    C.REGISTERED_RUNS = {'s3': G.RUNS}
    C.CAMPAIGN_ID = C.NEXT_CAMPAIGN_ID = C.CONTROL_CAMPAIGN = G.CAMPAIGN_ID
    C.METHOD_REVISION = G.METHOD_REVISION
    from panda_sensor_focus import plan as P
    base_default, base_resolve = P.default_config, P.resolve_case

    def default_config(server):
        if server != 's3':
            raise ValueError('GF2 W128 belongs to s3 only')
        cfg = base_default(server)
        cfg.update(joint_target=deepcopy(G.TARGET), after_campaign='FINISH_REGISTERED_GF2_W128_QUEUE',
                   publication_target={'HQNR': .9645, 'ERGAS': .522}, additional_target={'PSNR': 45.08})
        return cfg

    def resolve_case(server, case, replica='R1', *, attempt=1):
        cfg = base_resolve(server, case, replica, attempt=attempt)
        cfg.update(run_display=f'GF2 W128 {case} {replica}', joint_target=deepcopy(G.TARGET), psnr_min=45.08,
                   question='W128 GF2: LR 1.5/2.0/2.5e-4 x full/half adaptive Hard; same local 42K donor')
        cfg['comparison_control'] = {
            'campaign_id': G.BASELINE_CAMPAIGN, 'case': 'GF2P_C60LR200', 'server': 's3',
            'replica': replica, 'seed': P.SEEDS[replica], 'fresh_student_no_weights_reused': True,
            'same_exact_donor_checkpoint': True, 'student_seed_differs_from_control': False}
        return cfg

    P.default_config, P.resolve_case = default_config, resolve_case
    from panda_sensor_focus import runtime, orchestration
    base_pin = runtime.code_pin

    def code_pin():
        from ta2.evaluation import file_sha
        files = [p for p in (P.ROOT / 'panda_gf2_w128').rglob('*.py') if 'tests' not in p.parts]
        files += [p for p in (P.ROOT / 'config/panda_gf2_w128').glob('*')
                  if p.is_file() and p.name != 'release.json']
        # Imported by the shared reporting layer even during read-only checks.
        files += [P.ROOT / 'research_log/gspread_rules/PAN_FIXED64_V1.schema.json']
        files += list((P.ROOT / 'tools/metrics').glob('*.py'))
        files += [p for p in (P.ROOT / 'experiment_cost').rglob('*.py') if 'tests' not in p.parts]
        files += [P.ROOT / name for name in ('tools/eval_dlpan.py', 'Dockerfile.env',
                                           'requirements.txt', 'ablr2/sensor_sources.json',
                                           'edgeres48/docker_mounts.py', 'model/swin.py')]
        return {**base_pin(), **{str(p.relative_to(P.ROOT)): file_sha(p) for p in sorted(files)}}

    def predecessor(root, server):
        if server != 's3':
            raise ValueError('GF2 W128 is s3 only')
        from .prerequisites import evidence
        return evidence(Path(root))

    runtime.code_pin = code_pin
    orchestration.predecessor_receipt = predecessor
    _installed = True
    return P
