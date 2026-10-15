import json, os, pathlib, shutil, subprocess, sys, tempfile
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'QRECON24_Implementation_Audit_2026-09-16'))
from audit_common import ROOT, PY, ENV, fixture
with tempfile.TemporaryDirectory(prefix='noa-unit-audit-') as td:
    p=pathlib.Path(td); fixture(p,'s1')
    for folder in ('tools','gspread','research_log'):
        for src in (HERE/'source'/folder).rglob('*'):
            if src.is_file():
                dst=p/folder/src.relative_to(HERE/'source'/folder);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    # K50 checks existence of this prior implementation note as well.
    note='2026-09-17_qrecon24-adjustment-r1-implementation.md'
    shutil.copy2(ROOT/'research_log'/note,p/'research_log'/note)
    tag='PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v2'
    for rel in ('checkpoint_metrics.csv','best_hqnr_meta.json','results/fr_mat20.json'):
        dst=p/'work_dir'/tag/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'work_dir'/tag/rel,dst)
    suffix=os.environ.get('NOA_AUDIT_SUFFIX','')
    with (HERE/('unit_run'+suffix+'.log')).open('w') as f:
        r=subprocess.run([PY,'-B','tools/pakd50_unit_tests.py'],cwd=p,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=360)
    lines=(HERE/('unit_run'+suffix+'.log')).read_text().splitlines()
    result=dict(returncode=r.returncode,passed=sum(l.startswith('  OK ') for l in lines),failures=[l for l in lines if l.startswith('  FAIL')],tail=lines[-5:])
    (HERE/('unit_results'+suffix+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,ensure_ascii=False))
