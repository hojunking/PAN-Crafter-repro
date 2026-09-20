import h5py, numpy as np, os, sys
D='data/PanCollection'
files = {
 'GF2 train': f'{D}/GF2/train_gf2.h5',
 'GF2 valid': f'{D}/GF2/valid_gf2.h5',
 'GF2 RR test': f'{D}/GF2/reduced_examples_h5/test_gf2_multiExm1.h5',
 'GF2 FR mat20': f'{D}/GF2/full_examples_mat20/test_gf2_OrigScale_mat20.h5',
 'QB FR mat20': f'{D}/QB/full_examples_mat20/test_qb_OrigScale_mat20.h5',
 'WV3 FR mat20': f'{D}/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5',
}
def stats(a):
    a=np.asarray(a); return f'dtype={a.dtype} shape={a.shape} min={a.min():.1f} max={a.max():.1f} mean={a.mean():.2f}'
for name,p in files.items():
    if not os.path.exists(p): print('MISSING',name,p); continue
    print('=====',name,p)
    with h5py.File(p,'r') as f:
        for k in f.keys():
            ds=f[k]; n=ds.shape[0]; sel=slice(0,min(n,400))
            print(f'  {k}: {stats(ds[sel])}  (first {min(n,400)} of {n})')
    pp=p.replace('.h5','_pan.h5')
    if os.path.exists(pp):
        with h5py.File(pp,'r') as f:
            for k in f.keys():
                ds=f[k]; n=ds.shape[0]; print(f'  [pan.h5] {k}: {stats(ds[:min(n,400)])} of {n}')
    else: print('  no _pan.h5')
# lpan vs PAN scene match on FR/RR sets (F-1 style): corr(lpan, gaussian-lowpass-decimated pan)
import scipy.ndimage as ndi
def lpan_check(p):
    with h5py.File(p,'r') as f: pan=f['pan'][:,0].astype(np.float64)
    with h5py.File(p.replace('.h5','_pan.h5'),'r') as f: lpan=f['lpan'][:,0].astype(np.float64)
    out=[]
    for i in range(pan.shape[0]):
        lp=ndi.gaussian_filter(pan[i],1.98,mode='nearest')[2::4,2::4]
        c=np.corrcoef(lp.ravel(),lpan[i].ravel())[0,1]; rmse=np.sqrt(np.mean((lp-lpan[i])**2))
        out.append((round(c,6),round(rmse,3)))
    return out
for name in ['GF2 FR mat20','GF2 RR test']:
    r=lpan_check(files[name]); cs=[x[0] for x in r]; print(f'== lpan check {name}: corr min {min(cs):.5f} max {max(cs):.5f}; rmse range {min(x[1] for x in r)}..{max(x[1] for x in r)}')
# ms vs gt decimation phase on GF2 train/valid (F-3 style) using genMTF default 0.3 kernel is DLPan-specific; use simple test: which of 4x4 phases of blurred gt best matches ms
def phase_check(p, n=300):
    with h5py.File(p,'r') as f:
        gt=f['gt'][:n].astype(np.float64); ms=f['ms'][:n].astype(np.float64)
    # approximate MTF blur with gaussian sigma from GNyq=0.3 (N=41, ratio 4): alpha = sqrt(((N-1)*(fcut/2))^2/(-2 ln GNyq))
    N=41; fcut=0.25; alpha=np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(0.3)))
    from collections import Counter
    cnt=Counter(); mads=[]
    for i in range(gt.shape[0]):
        best=None
        for dy in range(4):
            for dx in range(4):
                b=np.stack([ndi.gaussian_filter(gt[i,c],alpha,mode='nearest')[dy::4,dx::4] for c in range(gt.shape[1])])
                mad=np.mean(np.abs(b-ms[i]))
                if best is None or mad<best[0]: best=(mad,dy,dx)
        cnt[(best[1],best[2])]+=1; mads.append(best[0])
    return cnt, np.median(mads), alpha
for name in ['GF2 train','GF2 valid']:
    cnt,med,alpha=phase_check(files[name]); print(f'== ms/gt phase {name} (gauss alpha {alpha:.3f}, n=300): {dict(cnt)} median MAD {med:.3f} DN')
