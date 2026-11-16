"""Bind the shared INPUT3 engine to QB on s3, without changing its method."""
import json
from directshift48.reporting import INDEX, canonical_json
from . import plan
from .release import verify

_installed = False


def install():
    global _installed
    if _installed:return
    verify()
    from panda_input3 import plan as base, release, assets as base_assets
    from . import assets
    for key in ('CAMPAIGN_ID','sensor_for','targets','campaign_root','resolve','queue','validate','catalog'):
        setattr(base,key,getattr(plan,key))
    release.verify=verify
    base_assets.sources=assets.sources
    base_assets.selected_reference=assets.selected_reference
    base_assets.prepare_reference=assets.prepare_reference
    from panda_input3_original.reporting import install as reporting
    reporting()
    from panda_input3 import publication
    make_row = publication.make_row

    def three_repeat_row(*args, **kwargs):
        row, raw = make_row(*args, **kwargs)
        note = json.loads(raw)
        note.update(reference_repeats=3,
                    reference_repeats_scope='THREE_FRESH_STUDENTS_ONE_FROZEN_LOCAL_STAGE1')
        raw = canonical_json(note); row[INDEX['Original description / notes']] = raw
        return row, raw

    publication.make_row = three_repeat_row
    from panda_input3 import preflight as shared_preflight
    from .preflight import run
    shared_preflight.run=run
    _installed=True
