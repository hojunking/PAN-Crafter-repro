"""Use native dataset LMS as an independent upsampling-phase control."""
import json
import numpy as np
from analyze import CASES, paths, split_data, registration, register_one, dump, sha, ROOT, OUT, warp_np

def main():
    out={}
    for sensor in CASES:
        _,manifest,bands,dn=paths(sensor);result={}
        diagnostic=json.loads((OUT/(sensor+'_diagnostics.json')).read_text())
        for split in ['rr','fr']:
            data=split_data(manifest,split,dn);rows=[]
            assert sha(manifest['splits'][split]['dataroot'])==manifest['splits'][split]['sha256']
            assert sha(manifest['splits'][split]['lpan_path'])==manifest['splits'][split]['lpan_sha256']
            for i in range(len(data['pan'])):
                r=dict(index=i,native_lms=registration(data['pan'][i,0],data['lms'][i]))
                c=np.asarray(diagnostic['splits'][split]['rows'][i]['delta'])
                for alpha in [1,4]:
                    r['after_'+str(alpha)]=registration(warp_np(data['pan'][i,0],c*alpha),data['lms'][i])['shift']
                if split=='rr':
                    r['lms_gt_phase']=[register_one(data['lms'][i,b],data['gt'][i,b],1.98,1.98)['shift'] for b in range(bands)]
                rows.append(r)
            result[split]=rows
            print(sensor,split,'source/LP hashes PASS; native LMS offset',np.mean([np.linalg.norm(r['native_lms']['shift']) for r in rows]),flush=True)
        out[sensor]=result
        dump('reference_sensitivity.json',out)
    source=['pa/aligner.py','pa/warp.py','pa/offset.py','fh12/model.py','qg40/model.py',
            'fh12/losses.py','qg40/losses.py','g20/losses.py','fh12/evaluation.py',
            'qg40/evaluation.py','model/pancrafter_paper.py','tools/metrics/eval_fr.py','tools/metrics/eval_rr.py']
    dump('source_hashes.json',{p:sha(ROOT/p) for p in source})

if __name__=='__main__':main()
