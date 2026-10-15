import pathlib,tempfile,subprocess,json,shutil
from audit_common import *
with tempfile.TemporaryDirectory(prefix='qrc24-audit-unit-') as td:
 p=pathlib.Path(td);fixture(p,'s1')
 with (OUT/'updated_units_no_history.log').open('w') as log:
  z=subprocess.run([PY,'-B','tools/pakd50_unit_tests.py'],cwd=p,env=ENV,stdout=log,stderr=subprocess.STDOUT,text=True,timeout=360)
 lines=(OUT/'updated_units_no_history.log').read_text().splitlines()
 result=dict(rc=z.returncode,checks=sum(l.startswith('  OK ') for l in lines),failures=[l for l in lines if l.startswith('  FAIL')])
 (OUT/'updated_units_no_history.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
 # Copy only the three read-only history files required by K48c, not checkpoint state or production outputs.
 tag='PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v2'
 for rel in ('checkpoint_metrics.csv','best_hqnr_meta.json','results/fr_mat20.json'):
  dst=p/'work_dir'/tag/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'work_dir'/tag/rel,dst)
 with (OUT/'updated_units_other_history.log').open('w') as log:
  z=subprocess.run([PY,'-B','tools/pakd50_unit_tests.py'],cwd=p,env=ENV,stdout=log,stderr=subprocess.STDOUT,text=True,timeout=360)
 lines=(OUT/'updated_units_other_history.log').read_text().splitlines(); result=dict(rc=z.returncode,checks=sum(l.startswith('  OK ') for l in lines),failures=[l for l in lines if l.startswith('  FAIL')])
 (OUT/'updated_units_other_history.json').write_text(json.dumps(result,indent=2));print(result)
