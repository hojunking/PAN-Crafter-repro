"""Configure the unchanged INPUT3 engine in an isolated C30 worker process."""
from . import recipes
from .release import verify

_installed = False


def install():
    global _installed
    if _installed: return
    verify()
    from panda_input3 import plan, release, assets
    # All tensors, optimizer/evaluator code and publication gates remain shared.
    plan.CAMPAIGN_ID = recipes.CAMPAIGN_ID
    plan.campaign_root = recipes.campaign_root
    plan.resolve, plan.queue, plan.validate = recipes.resolve, recipes.queue, recipes.validate
    release.verify = verify
    prepare_reference = assets.prepare_reference

    def shared_reference(cfg, registered, root, pins, device='cpu'):
        # Reuse exactly the original frozen donor and cached calibration, not a
        # freshly selected checkpoint or a different q/g reference distribution.
        return prepare_reference(cfg, registered, recipes.ORIGINAL_ROOT(cfg['server']), pins, device=device)

    assets.prepare_reference = shared_reference
    from panda_input3_targets.reporting import install as reporting
    reporting()
    from panda_input3 import preflight
    original_preflight = preflight.run

    def run_preflight(server='s2', device='cuda:0'):
        # The B/C initial-function parity probe requires B before C; this is
        # independent of the scheduling order recorded in cutover.json.
        order = plan.queue
        try:
            plan.queue = recipes.canonical_queue
            return original_preflight(server, device)
        finally:
            plan.queue = order

    preflight.run = run_preflight
    _installed = True
