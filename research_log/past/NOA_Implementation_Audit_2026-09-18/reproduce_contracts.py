"""Read-only production audit: real orchestration code, /tmp files and fake Sheets/inference."""
import contextlib, importlib.util, io, json, os, pathlib, shutil, sys, tempfile, types
from unittest.mock import patch
import numpy as np

ROOT = pathlib.Path('/home/knuvi/Desktop/song/PAN-Crafter')
OUT = pathlib.Path(__file__).resolve().parent
SNAP = OUT / 'source'
sys.path.insert(0, str(ROOT))
RESULTS = {}

def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, SNAP / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m

def put(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))

NE = load('tools/noa_eval.py', 'audit_noa')
EP = load('tools/eval_phase.py', 'audit_phase')
UP = load('gspread/noa_upload.py', 'audit_upload')
SA = load('tools/s1_aligner_analysis.py', 'audit_s1')

with tempfile.TemporaryDirectory(prefix='noa-contract-audit-') as td:
    p = pathlib.Path(td)
    NE.ROOT = EP.ROOT = SA.ROOT = str(p)
    phase = p / NE.PHASE_DIR; phase.mkdir(parents=True)
    EP.chain_alive = lambda: (False, [])
    put(p / EP.RESUME, dict(resume=dict(next_case='APPROVED_RUN'), approved_main_queue_revision='R2_APPROVED'))
    put(p / EP.HOLD, dict(phase='LOCAL_STUDENT_EVAL', server='s1'))
    EP.cmd_resume(types.SimpleNamespace(dry_run=True))
    RESULTS['dry_run_removed_hold'] = not (p / EP.HOLD).exists()
    put(p / EP.HOLD, dict(phase='S1_ALIGNER_ANALYSIS', server='s1'))
    EP.cmd_release(types.SimpleNamespace(dry_run=False))
    RESULTS['release_without_metrics_upload_or_analysis'] = not (p / EP.HOLD).exists()

    run = 'PAKD50_QRC24_S1_FIXTURE'; current = 'PAKD50_QRC24_S1_CURRENT'
    wd = p / 'work_dir' / run
    put(wd / 'best_hqnr_meta.json', dict(step=100))
    ck = wd / 'best_hqnr/model.safetensors'; ck.parent.mkdir(parents=True); ck.write_bytes(b'checkpoint-A')
    for rel in ['results/reduced_best_hqnr.mat', 'results/full_best_hqnr.mat']:
        f = wd / rel; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(b'result')
    # A foreign training source deliberately disagrees with the current host.
    put(wd / 'meta/execution_manifest.json', dict(source_train_server='s2'))
    NE.audit_run = lambda *a, **kw: dict(verdict='single_definition_factor1')
    NE.server_id = lambda: 's1'
    NE.running_runs = lambda: [current]
    cohort = NE.capture_cohort()
    RESULTS['cohort_source_taken_from_current_host'] = cohort['records'][0]['source_train_server']
    for rel in ['results/reduced_best_hqnr.mat', 'results/full_best_hqnr.mat']:
        f = p / 'work_dir' / current / rel; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(b'complete-now')
    evaluated = []
    def eval_stub(r, **kw):
        evaluated.append(r); return {}
    with patch.object(NE, 'evaluate_run', eval_stub), patch.object(sys, 'argv', ['noa_eval.py', '--all']):
        NE.main()
    RESULTS['pending_current_never_evaluated'] = dict(pending=cohort['pending_current_run_id'], now_completed=current in NE.completed_runs(), evaluated=evaluated)

    # Real identity/cache/evaluate_run logic; only expensive inference and metrics are fake.
    for rel in ['tools/metrics/eval_rr.py', 'tools/metrics/eval_fr.py', 'tools/eval_dlpan.py', 'gspread/gspread_upload.py', 'kdv/eval_modes.py', 'pa/model.py']:
        dest = p / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2((SNAP / rel) if (SNAP / rel).exists() else (ROOT / rel), dest)
    fr = p / NE.FR_H5; fr.parent.mkdir(parents=True); fr.write_bytes(b'fr-v1')
    rr = p / 'rr.h5'; rr.write_bytes(b'rr-v1')
    cfg = wd / 'meta/config.yaml'; cfg.write_text('test_reduced_feeder_args:\n  dataroot: ' + str(rr) + '\nversion: 1\n')
    infers = []
    def infer(run, ck, mode, device, cfg, split):
        infers.append((mode, device, cfg['version'], split))
        sr = np.full((1, 8, 4, 4), cfg['version'], np.float32)
        return sr, 'fixture', NE.expected_calls(mode, 1), True, [dict(delta=[0, 0])]
    NE._infer = infer
    NE.rr_metrics = lambda *a: dict(ergas=2.03, n=1)
    NE.fr_metrics = lambda sr: dict(hqnr_raw=.96, n=1, fixture_version=float(sr[0, 0, 0, 0]), per_scene_hqnr=[.96], per_scene_d_lambda=[.02], per_scene_d_s=[.02])
    NE.ledger_add = lambda *a: None
    initial = NE.evaluate_run(run, device='cpu')
    infers.clear(); NE.evaluate_run(run, device='cuda')
    RESULTS['device_change_reused_cache_without_inference'] = not infers
    rr.write_bytes(b'rr-v2'); infers.clear(); NE.evaluate_run(run, device='cpu')
    RESULTS['rr_data_change_reused_cache_without_inference'] = not infers
    cfg.write_text(cfg.read_text().replace('version: 1', 'version: 2'))
    NE.evaluate_run(run, device='cpu', splits=('rr',))
    infers.clear(); stale = NE.evaluate_run(run, device='cpu')
    RESULTS['partial_refresh_accepts_old_fr_as_new_identity'] = dict(inferences=list(infers), new_config_version=2, fr_versions={k:v['fr']['fixture_version'] for k,v in stale['modes'].items()})
    rp = phase / 'records' / (run + '.json'); rec = json.loads(rp.read_text()); rec['uploaded'] = 'verified-before-restart'; put(rp, rec)
    NE.evaluate_run(run, device='cpu')
    RESULTS['cache_rerun_loses_upload_marker'] = 'uploaded' not in json.loads(rp.read_text())
    RESULTS['cache_rerun_loses_elapsed_seconds'] = all('seconds' not in v for v in json.loads(rp.read_text())['modes'].values())
    ck.write_bytes(b'checkpoint-B'); infers.clear(); rec = NE.evaluate_run(run, device='cpu')
    RESULTS['cohort_checkpoint_sha_not_enforced'] = dict(captured=cohort['records'][0]['checkpoint_sha256'], evaluated=rec['checkpoint_sha256'])
    partial = dict(run=run, modes={m:dict(rr=dict(ergas=2.03)) for m in NE.DEFAULT_MODES}, delta_rr_ergas=0)
    put(rp, partial)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): NE.status()
    RESULTS['rr_only_status'] = buf.getvalue()

    # Actual uploader with fake network backend. No Google service is contacted.
    UP.RECORDS = str(phase / 'records'); UP.LOCK = str(p / 'upload.lock'); UP._server = lambda:'s1'
    UP._load_local = lambda mod: types.SimpleNamespace(run_tag=lambda t:t.split(' | ')[0])
    def col(n):
        out = ''
        while n: n, rem = divmod(n-1,26); out = chr(65+rem)+out
        return out
    gu = types.SimpleNamespace(CRED='FAKE', SHEET='FAKE', ORIGIN_ROW=2, ORIGIN_COL=2, _col=col, _a1=lambda r,c:col(c)+str(r), _retry=lambda fn,*a,**kw:fn(*a,**kw), sheet_name=lambda ds,s:ds+'-'+s)
    UP._gu = lambda:gu
    class WS:
        col_count = 100
        def __init__(self): self.writes=[];self.reads=[]
        def row_values(self,r): self.reads.append(('header',r)); return ['old']*24
        def get(self,r): self.reads.append(('range',r));return [[run+' | step=100'],[run+' | step=200']]
        def batch_update(self,ops):self.writes.extend(ops);return {'updatedCells':0} # no persisted values/readback
    ws=WS(); fake=types.SimpleNamespace(service_account=lambda **kw:types.SimpleNamespace(open=lambda name:types.SimpleNamespace(worksheet=lambda name:ws)))
    rec=dict(run=run,checkpoint_step=100,checkpoint_sha256='a'*64,modes={m:dict(rr=dict(ergas=2.03),fr=dict(hqnr_raw=.96)) for m in NE.DEFAULT_MODES})
    put(rp,rec)
    with patch.dict(sys.modules,{'gspread':fake}): UP.upload([rec])
    RESULTS['upload_aliases_with_different_steps_both_written'] = [op['range'] for op in ws.writes[2:]]
    RESULTS['upload_with_unknown_source_accepted'] = len(ws.writes)>2
    RESULTS['upload_marked_done_without_readback'] = bool(json.loads(rp.read_text()).get('uploaded'))
    RESULTS['upload_read_requests'] = ws.reads

    calls=[]
    for r in SA.S1_ASSETS: put(p/'work_dir'/r/'meta/config.yaml', {})
    SA.run_size_probe=lambda *a:calls.append(a)
    with patch.object(sys,'argv',['s1_aligner_analysis.py','--assets']):SA.main()
    RESULTS['s1_assets_default_execution']=[dict(run=r,split=s,scenes=sc,device=d) for r,s,sc,d in calls]
    with patch.object(sys,'argv',['s1_aligner_analysis.py','--teacher']),contextlib.redirect_stderr(io.StringIO()):
        try:SA.main()
        except SystemExit as e:RESULTS['documented_teacher_option_exit']=e.code

(OUT/'contract_results.json').write_text(json.dumps(RESULTS,indent=2,ensure_ascii=False))
print(json.dumps(RESULTS,indent=2,ensure_ascii=False))
