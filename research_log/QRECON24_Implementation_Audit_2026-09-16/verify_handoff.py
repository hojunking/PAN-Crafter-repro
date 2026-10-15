"""Actual switch in isolated fixture; runner queue loop extracted verbatim.
Only expensive unit/cue checks and launch/cron/process providers are stubbed.
No production shell launcher or GPU job runs.
"""
import json,os,pathlib,subprocess,tempfile,time
from audit_common import *
R=[]
with tempfile.TemporaryDirectory(prefix='qrc24-handoff-') as td:
 p=pathlib.Path(td);fixture(p,'s2')
 for f,code in [('pakd50_unit_tests.py',"print('ALL OK: separately audited')\n"),('qedge9_cue.py',"print('cue verified separately')\n")]: (p/'tools'/f).write_text(code)
 for f in ['qrecon24_waiter.sh','_watchdog.sh','campaign_start.sh']:(p/'tools'/f).write_text('#!/bin/bash\nexit 0\n');(p/'tools'/f).chmod(0o755)
 fake=p/'bin';fake.mkdir();(fake/'ps').write_text('#!/bin/bash\necho "bash tools/_run_cases.sh"\n');(fake/'ps').chmod(0o755);env={**ENV,'PYTHON':PY,'PATH':str(fake)+':'+os.environ['PATH']}
 # Check real dry-run leaves operational files intact (stub heavyweight checks only).
 watch=['work_dir/_pakd50/mandatory_runs.txt','work_dir/_qrecon24/mandatory_runs.txt','work_dir/_pakd50/extra_priority.txt','work_dir/_pakd50/reservations.json','work_dir/cases_queue.txt','work_dir/cases_deadline.txt'];before={x:(p/x).read_bytes() if (p/x).exists() else None for x in watch}
 dry=subprocess.run(['bash','tools/qrecon24_switch.sh','--dry-run'],cwd=p,env=env,text=True,capture_output=True,timeout=90);(OUT/'switch_s2_dry.log').write_text(dry.stdout+dry.stderr);same=all(((p/x).read_bytes() if (p/x).exists() else None)==before[x] for x in watch);R.append(dict(name='dry_run_operational_noop',passed=dry.returncode==0 and same,kind='conformance',rc=dry.returncode))
 # Existing ORDER already in runner memory. First fake case blocks until switch has returned.
 olds=['OLD_CURRENT','OLD_PENDING_1','OLD_PENDING_2'];(p/'work_dir/cases_queue.txt').write_text('\n'.join(olds)+'\n');src=(p/'tools/_run_cases.sh').read_text();loop=src[src.index('i=0\n'):src.index('\n# 본 큐 종료 후:')]
 shell='QUEUE_FILE=work_dir/cases_queue.txt\nmapfile -t ORDER < <(grep -vE "^[[:space:]]*(#|$)" "$QUEUE_FILE")\nrun_case(){ echo "$1" >> work_dir/audit_executed; if [ "$1" = OLD_CURRENT ]; then touch work_dir/audit_ready; while [ ! -f work_dir/audit_release ]; do sleep 0.05; done; fi; }\n'+loop
 runner=subprocess.Popen(['bash','-c',shell],cwd=p,env=env)
 for _ in range(100):
  if (p/'work_dir/audit_ready').exists():break
  time.sleep(.05)
 try:
  switch=subprocess.run(['bash','tools/qrecon24_switch.sh'],cwd=p,env=env,text=True,capture_output=True,timeout=90);(OUT/'switch_with_alive_runner.log').write_text(switch.stdout+switch.stderr)
 finally:(p/'work_dir/audit_release').touch()
 runner.wait(timeout=15);executed=(p/'work_dir/audit_executed').read_text().splitlines()
 R.append(dict(name='switch_keeps_old_pending_after_current',passed=switch.returncode==0 and executed==olds,kind='defect_reproduction',switch_rc=switch.returncode,executed=executed,queue_after=(p/'work_dir/cases_queue.txt').read_text().splitlines(),note='Actual switch + original runner ORDER loop; only run_case is a no-training stub. At least remaining current ORDER/EXTRA pass is retained before a new gate pass.'))
(OUT/'handoff_results.json').write_text(json.dumps(R,indent=2));print(json.dumps(R,indent=2))
