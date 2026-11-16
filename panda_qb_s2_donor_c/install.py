"""Process-local binding; never rewrite the original INPUT3 training source."""
from . import plan
_installed=False


def install():
    global _installed
    if _installed:return
    from .release import verify
    verify()
    from panda_input3 import plan as shared,assets as shared_assets,release
    from panda_input3 import publication,transport
    # The generic transport validates its three-server destination registry.
    # Capture that registry before binding the actual runner to s2 only.
    destinations={server:publication.destination(server) for server in ('s1','s2','s3')}
    from . import assets
    for key in ('CAMPAIGN_ID','sensor_for','targets','campaign_root','resolve','queue','validate','catalog'):
        setattr(shared,key,getattr(plan,key))
    transport.destination=destinations.__getitem__
    release.verify=verify
    for key in ('sources','selected_reference','prepare_reference'):
        setattr(shared_assets,key,getattr(assets,key))
    from panda_qb_repfix.reporting import install as reporting
    reporting()
    from panda_input3 import preflight
    from .preflight import run
    preflight.run=run
    _installed=True
