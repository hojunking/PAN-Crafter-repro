"""Test only the Git-indexed release closure; local untracked packages are unavailable."""
import hashlib,json,os,re,subprocess,tempfile
from pathlib import Path
repo=Path.cwd();root=Path(tempfile.mkdtemp(prefix='wv3-hyper-index-'))
m=json.loads(subprocess.check_output(['git','show',':config/panda_wv3_hyper/release.json']))
for path,sha in m['files'].items():
    data=subprocess.check_output(['git','show',':'+path])
    if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Index/release mismatch '+path)
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
p=root/'config/panda_wv3_hyper/release.json';p.write_bytes(subprocess.check_output(['git','show',':config/panda_wv3_hyper/release.json']))
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root),PANCRAFTER_DLPAN=str(repo.parent/'DLPan-Toolbox'))
python='/home/knuvi/miniconda3/envs/pancrafter/bin/python'
print('Clean indexed closure',len(m['files']),str(root),flush=True)
subprocess.run([python,'-m','panda_wv3_hyper.release'],cwd=root,env=env,check=True)
result=subprocess.run([python,'-m','unittest','discover','-s','panda_wv3_hyper/tests','-v'],cwd=root,env=env,text=True,capture_output=True)
log=result.stdout+result.stderr
out=repo/'results_log/assets/2026-11-08-wv3-hyperparameter-implementation'
(out/'clean-index-test.log').write_text(log)
print(log[-6000:],flush=True)
result.check_returncode()
(out/'clean-index.json').write_text(json.dumps({'status':'VERIFIED','files':len(m['files']),'source_release':m['source_sha256'],'tests':int(re.search(r'Ran (\d+) tests',log)[1]),'test_scope':'clean indexed source closure without local untracked packages/data/weights'},indent=2)+'\n')
