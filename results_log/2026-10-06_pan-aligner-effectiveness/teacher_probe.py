"""Exact donor A controls and in-training-radius cross-scale response probes."""
import json
import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from analyze import (ROOT, CASES, EPS, dump, sha, paths, split_data, tensors,
                     predict, reduce4, load_aligner, PANGlobalAligner, warp_pan, F)


@torch.no_grad()
def main():
    for sensor in CASES:
        run=CASES[sensor][0]
        config=yaml.safe_load((ROOT/'work_dir'/run/'meta/config.resolved.yaml').read_text())
        spec=config[config['trainer']]
        ck=ROOT/spec['teacher_checkpoint']
        _,manifest,bands,dn=paths(sensor)
        state=load_file(str(ck),device='cpu')
        a=PANGlobalAligner(ms_bands=bands).eval()
        a.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('aligner.')},strict=True)
        out=dict(teacher_checkpoint=str(ck.relative_to(ROOT)),sha256=sha(ck),splits={})
        student,_,_=load_aligner(sensor)
        for split in ['train','rr','fr']:
            data=split_data(manifest,split,dn);p,m,base=tensors(data,dn);rows=[]
            for i in range(len(p)):
                c=predict(a,p[i:i+1],base[i:i+1])[0].numpy()
                responses=[]
                for eps in EPS:
                    pe=warp_pan(p[i:i+1],torch.from_numpy(eps[None]))
                    responses.append(predict(a,pe,base[i:i+1])[0].tolist())
                row=dict(index=int(data['indices'][i]),delta=c.tolist(),epsilon_prediction=responses)
                if split=='fr':
                    row['paired_R2']={}
                    for label,aligner in [('teacher',a),('student',student)]:
                        pr,mr=reduce4(p[i:i+1]),reduce4(m[i:i+1])
                        br=F.interpolate(mr,scale_factor=4,mode='bicubic',align_corners=False)
                        cf=predict(aligner,p[i:i+1],base[i:i+1])[0].numpy()
                        cr=predict(aligner,pr,br)[0].numpy()
                        paired=[]
                        for eps in [[2.,0.],[-2.,0.],[0.,2.],[0.,-2.]]:
                            pe=warp_pan(p[i:i+1],torch.tensor([eps]))
                            cfe=predict(aligner,pe,base[i:i+1])[0].numpy()
                            cre=predict(aligner,reduce4(pe),br)[0].numpy()
                            paired.append(dict(epsilon_fr=eps,fr_response=(cfe-cf).tolist(),reduced_response=(cre-cr).tolist()))
                        row['paired_R2'][label]=dict(fr_delta=cf.tolist(),reduced_delta=cr.tolist(),responses=paired)
                rows.append(row)
            out['splits'][split]=rows
            print(sensor,'Teacher',split,flush=True)
            dump(sensor+'_teacher.json',out)


if __name__=='__main__': main()
