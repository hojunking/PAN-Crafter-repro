"""Use the shared atomic FR67 writer with this campaign's own namespace."""
import json
from directshift48.reporting import canonical_json
from panda_fr_columns import publication as fr
from . import publication as pub

_installed=False
def install():
    global _installed
    if _installed:return
    fr.PACKAGES=(*fr.PACKAGES,'panda_qb_s3_parallel')
    fr.install()
    original=fr.extras
    def extras(value):
        result=original(value)
        if value['policy']==pub.POLICY:
            for cells in result.values():
                try:note=json.loads(cells[0]['note'])
                except json.JSONDecodeError:continue
                note.update(official_selection_reference='CORRECTED_PAN',
                    evaluation_reference='SAME_FIXED_DONOR_FOR_ALL_CASES',
                    selection_policy=pub.POLICY,original_PAN_benchmark_comparable=False)
                cells[0]['note']=canonical_json(note)
        return result
    fr.extras=extras
    _installed=True
