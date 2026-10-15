"""Audit completion/reuse admission with isolated incomplete artifacts only."""
import json, pathlib, sys, tempfile
from unittest.mock import patch
ROOT = pathlib.Path(sys.argv[1]).resolve()
OUT = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT))
from tools import campaign_gate as C
from tools import gen_pakd50_configs as G
tag = G.run_name('J0', 1234, arch='W104_D121')
with tempfile.TemporaryDirectory(prefix='qedge9-incomplete-control-') as td:
    p = pathlib.Path(td) / 'work_dir' / tag
    (p / 'results').mkdir(parents=True)
    for name in ('reduced_best_hqnr.mat', 'full_best_hqnr.mat'):
        (p / 'results' / name).touch()
    (p / 'last_meta.json').write_text(json.dumps({'step': 1010}))
    (p / 'initialization_hashes.json').write_text(json.dumps({'unet_init_sha256_16': 'wrong_fixture_init'}))
    with patch.object(C, 'ROOT', td):
        result = dict(name='incomplete_control_is_terminal', kind='defect_reproduction',
                      passed=C.complete(tag) and C.terminal(tag),
                      step=1010, checkpoint_present=False, candidate_grid_present=False,
                      note='Empty marker fixtures expose existence-only admission; no actual result is alleged incomplete. qedge9_switch lines 40-42 then skips config checks for this terminal run.')
(OUT / 'completion_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
sys.exit(0 if result['passed'] else 1)
