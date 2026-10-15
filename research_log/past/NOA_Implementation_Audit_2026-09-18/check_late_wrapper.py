"""Exercise the added shell wrapper in /tmp. No training or Sheet tool is run."""
import json, os, pathlib, shutil, subprocess, tempfile
OUT=pathlib.Path(__file__).resolve().parent
PY='/home/knuvi/miniconda3/envs/pancrafter/bin/python'
with tempfile.TemporaryDirectory(prefix='noa-wrapper-audit-') as td:
    p=pathlib.Path(td);(p/'tools').mkdir();phase=p/'work_dir/_eval_phase';phase.mkdir(parents=True)
    shutil.copy2(OUT/'late_snapshot/tools/noa_eval_switch.sh',p/'tools/noa_eval_switch.sh')
    hold=phase/'hold.json';hold.write_text(json.dumps(dict(phase='LOCAL_STUDENT_EVAL',server='s1')))
    shim=p/'python_shim'
    shim.write_text('#!'+PY+'\n'+'''import importlib.util, pathlib, sys, types
if sys.argv[1]=='-c': print('s1')
elif sys.argv[1:]==['tools/eval_phase.py','release']:
    s=importlib.util.spec_from_file_location('audit_ep',SOURCE)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.ROOT=str(pathlib.Path.cwd());m.cmd_release(types.SimpleNamespace(dry_run=False))
else: print('fixture status: required incomplete')
'''.replace('SOURCE',repr(str(OUT/'source/tools/eval_phase.py'))))
    shim.chmod(0o755)
    r=subprocess.run(['bash','tools/noa_eval_switch.sh','--release','--dry-run'],cwd=p,env=dict(os.environ,PYTHON=str(shim),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=20)
    result=dict(command='noa_eval_switch.sh --release --dry-run',returncode=r.returncode,hold_removed=not hold.exists(),stdout=r.stdout,stderr=r.stderr)
    (OUT/'late_wrapper_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,ensure_ascii=False))
