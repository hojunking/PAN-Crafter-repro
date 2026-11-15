"""Use the unchanged model, trainer, evaluator and original-PAN publisher."""
from . import recipes
from .release import verify

_installed = False


def install():
    global _installed
    if _installed: return
    verify()
    from panda_input3 import plan, release, assets
    plan.CAMPAIGN_ID = recipes.CAMPAIGN_ID
    plan.campaign_root = recipes.campaign_root
    plan.resolve, plan.queue, plan.validate = recipes.resolve, recipes.queue, recipes.validate
    release.verify = verify
    original_prepare = assets.prepare_reference

    def shared_reference(cfg, registered, root, pins, device='cpu'):
        return original_prepare(cfg, registered, recipes.previous.ORIGINAL_ROOT(cfg['server']),
                                pins, device=device)

    assets.prepare_reference = shared_reference
    from panda_input3_original.reporting import install as reporting
    reporting()
    _installed = True
