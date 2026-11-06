"""Developer-only manifest generation; queued workers verify the saved manifest."""
from .common import ROOT, CONFIG, read, sha, digest, write


if __name__=='__main__':
    files=dict(read(ROOT/'config/panda_gf2_w128/release.json')['files'])
    for folder in ('panda_wv3_ablation','panda_wv2_eval'):
        for p in (ROOT/folder).rglob('*.py'):
            if 'tests' not in p.parts:files[str(p.relative_to(ROOT))]=sha(p)
    files['config/panda_wv2_eval/plan.json']=sha(CONFIG/'plan.json')
    write(CONFIG/'release.json',{'schema':'WV2_ZERO_SHOT_RELEASE_V1','files':files,'sha256':digest(files)})
    print(len(files),digest(files))
