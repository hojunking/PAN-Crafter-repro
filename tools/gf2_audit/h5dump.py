import h5py, numpy as np, sys, os, json
root='/home/knuvi/Desktop/song/PAN-Crafter/data/PanCollection'
files = {
 'GF2': ['train_gf2.h5','train_gf2_pan.h5','valid_gf2.h5','valid_gf2_pan.h5',
         'reduced_examples_h5/test_gf2_multiExm1.h5','reduced_examples_h5/test_gf2_multiExm1_pan.h5',
         'full_examples_h5/test_gf2_OrigScale_multiExm1.h5','full_examples_h5/test_gf2_OrigScale_multiExm1_pan.h5',
         'full_examples_mat20/test_gf2_OrigScale_mat20.h5','full_examples_mat20/test_gf2_OrigScale_mat20_pan.h5'],
 'WV3': ['train_wv3.h5','train_wv3_pan.h5','valid_wv3.h5','valid_wv3_pan.h5',
         'reduced_examples_h5/test_wv3_multiExm1.h5','reduced_examples_h5/test_wv3_multiExm1_pan.h5',
         'full_examples_h5/test_wv3_OrigScale_multiExm1.h5','full_examples_h5_repaired/test_wv3_OrigScale_multiExm1_pan.h5',
         'full_examples_mat20/test_wv3_OrigScale_mat20.h5','full_examples_mat20/test_wv3_OrigScale_mat20_pan.h5'],
 'QB':  ['train_qb.h5','train_qb_msfix.h5','train_qb_pan.h5','valid_qb.h5','valid_qb_msfix.h5','valid_qb_pan.h5',
         'reduced_examples_h5/test_qb_multiExm1.h5','reduced_examples_h5/test_qb_multiExm1_pan.h5',
         'full_examples_h5/test_qb_OrigScale_multiExm1.h5','full_examples_h5_repaired/test_qb_OrigScale_multiExm1_pan.h5',
         'full_examples_mat20/test_qb_OrigScale_mat20.h5','full_examples_mat20/test_qb_OrigScale_mat20_pan.h5'],
}
out={}
for sensor, fl in files.items():
    for f in fl:
        p=os.path.join(root,sensor,f)
        if not os.path.exists(p):
            print(f'[{sensor}] {f}: MISSING'); continue
        with h5py.File(p,'r') as h:
            print(f'\n[{sensor}] {f}  (attrs={dict(h.attrs)})')
            for k in h.keys():
                d=h[k]
                n=d.shape[0]
                # sample up to 256 patches evenly for stats
                idx=np.linspace(0,n-1,min(n,256)).astype(int)
                idx=np.unique(idx)
                arr=d[idx] if n>256 else d[()]
                arr=np.asarray(arr).astype(np.float64)
                pct=np.percentile(arr,[0,0.1,1,50,99,99.9,100])
                fullmax=None
                print(f'   {k:6s} shape={d.shape} dtype={d.dtype} chunks={d.chunks} attrs={dict(d.attrs)} '
                      f'| sampled({len(idx)}) min={pct[0]:.1f} p0.1={pct[1]:.1f} p1={pct[2]:.1f} med={pct[3]:.1f} p99={pct[4]:.1f} p99.9={pct[5]:.1f} max={pct[6]:.1f} mean={arr.mean():.2f}')
                out[f'{sensor}/{f}/{k}']=dict(shape=list(d.shape),dtype=str(d.dtype),pct=pct.tolist(),mean=float(arr.mean()),nsample=int(len(idx)))
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'h5dump.json'),'w'),indent=1)
