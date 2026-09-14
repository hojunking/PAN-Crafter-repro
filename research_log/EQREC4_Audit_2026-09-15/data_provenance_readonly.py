"""Bounded duplicate/proximity audit; CPU only; output remains in this audit folder.
Exact input/tile equality and feature neighbors do not establish scene independence.
"""
import hashlib,json,pathlib
import h5py,numpy as np,pandas as pd
from scipy.spatial import cKDTree
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=pathlib.Path(__file__).resolve().parent
man=pd.read_csv(ROOT/'work_dir/_eqrec4_s1_campaign/data_manifest.csv').sort_values('sample_id')
ids=man.sample_id.to_numpy();roles=man.split_role.to_numpy();groups=man.source_group_id.to_numpy()
whole={};tiles={};features=[]
with h5py.File(ROOT/'data/PanCollection/WV3/train_wv3.h5') as f:
    for start in range(0,len(ids),64):
        p=f['pan'][ids[start:start+64]];m=f['ms'][ids[start:start+64]]
        for j,(pan,ms) in enumerate(zip(p,m)):
            k=start+j;h=hashlib.sha256(pan.tobytes()+ms.tobytes()).hexdigest();whole.setdefault(h,[]).append(k)
            for y,x in [(0,0),(0,32),(32,0),(32,32)]:
                a=pan[0,y:y+32,x:x+32]
                if a.std()>1.:tiles.setdefault(hashlib.sha256(a.tobytes()).hexdigest(),[]).append(k)
            # Intensity-normalized 8x8 PAN descriptor + per-band input means/stds.
            v=pan[0].reshape(8,8,8,8).mean((1,3)).reshape(-1);v=(v-v.mean())/(v.std()+1e-6)
            features.append(np.r_[v,ms.mean((1,2))/2047,ms.std((1,2))/2047])
def pairs(mapping):
    out=[]
    for lst in mapping.values():
        u=sorted(set(lst))
        for ai,a in enumerate(u):
            for b in u[ai+1:]:
                if roles[a]!=roles[b]:out.append([int(ids[a]),int(ids[b]),str(roles[a]),str(roles[b])])
    return out
features=np.asarray(features);dist,near=cKDTree(features).query(features,k=2)
order=np.argsort(dist[:,1])[:20]
r=dict(n_selected=len(ids),known_original_source_ids=False,exact_pan_plus_ms_cross_role_pairs=pairs(whole),exact_nonflat_pan_32x32_tile_cross_role_pairs=pairs(tiles),
       feature_nearest_neighbor_cross_role_fraction=float(np.mean(roles!=roles[near[:,1]])),feature_nearest_neighbor_same_proxy_block_fraction=float(np.mean(groups==groups[near[:,1]])),
       closest_feature_pairs=[dict(sample_id=int(ids[i]),neighbor_id=int(ids[near[i,1]]),role=str(roles[i]),neighbor_role=str(roles[near[i,1]]),descriptor_l2=float(dist[i,1])) for i in order],
       limitation='A coarse image-feature proximity screen, not geolocation or overlap reconstruction. No independent scene count or scene-level CI can be certified from these checks.')
(OUT/'data_provenance.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='closest_feature_pairs'},indent=2))
