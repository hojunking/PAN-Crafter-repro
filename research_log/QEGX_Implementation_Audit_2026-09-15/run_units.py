import pathlib,tempfile,subprocess,json
from audit_common import *
with tempfile.TemporaryDirectory(prefix='qegx-audit-unit-') as td:
    p=pathlib.Path(td);fixture(p,'s1')
    with (OUT/'existing_units_host.log').open('w') as log:
        z=subprocess.run([PY,'-B','tools/pakd50_unit_tests.py'],cwd=p,env=ENV,stdout=log,stderr=subprocess.STDOUT,text=True,timeout=240)
    lines=(OUT/'existing_units_host.log').read_text().splitlines()
    result=dict(rc=z.returncode,checks=sum(l.startswith('  OK ') for l in lines),note='CPU isolated fixture; unsandboxed local IPC allowed for DataLoader workers')
    (OUT/'units_summary_host.json').write_text(json.dumps(result));print(result)
