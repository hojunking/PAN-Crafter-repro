"""Sanity checks for the finished report and source/checkpoint provenance."""
import json
import re
from pathlib import Path
import numpy as np
from analyze import ROOT, OUT, CASES, sha, dump

sources=['pa/aligner.py','pa/warp.py','fh12/model.py','qg40/model.py',
         'model/pancrafter_paper.py','fh12/evaluation.py','qg40/evaluation.py']
checks={}
for sensor,(run,step,_,_,_) in CASES.items():
    official=json.loads((ROOT/'work_dir'/run/'official'/f'candidate_{step}.json').read_text())
    diag=json.loads((OUT/(sensor+'_diagnostics.json')).read_text())
    quality=json.loads((OUT/(sensor+'_quality.json')).read_text())
    assert official['checkpoint_identity']['model_sha256']==diag['checkpoint_sha256']==quality['checkpoint_sha256']
    assert len(quality['alphas'])==6
    for k,values in quality['alphas'].items():
        assert len(values['rows'])==20 and all(np.isfinite(v) for v in values['mean'].values())
    assert all(len(diag['splits'][s]['rows'])==20 for s in ['train','rr','fr'])
    sf=official['checkpoint_identity']['source_identity']['files']
    matched={s:sha(ROOT/s)==sf[s] for s in sources if s in sf}
    assert all(matched.values())
    checks[sensor]=dict(checkpoint_matches_official=True,quality_groups=6,scenes_per_group=20,
                        source_byte_parity_with_official_record=matched)
report=ROOT/'results_log/2026-10-06_PAN_Aligner_RR_FR_effectiveness_report.md'
text=report.read_text()
assert 'QUALITY_TABLES' not in text
for link in re.findall(r'\]\(([^)]+)\)',text):
    if not link.startswith('http'):
        if (report.parent/link).resolve()==OUT/'artifact_validation.json': continue
        assert (report.parent/link).exists(),link
assert len(list(OUT.glob('*.png')))==10
dump('artifact_validation.json',dict(status='PASS',sensors=checks,local_report_links='PASS',figures=10))
print('PASS: checkpoint/source provenance; 3 sensors x 6 quality groups x 20 scenes; 10 figures; report links')
