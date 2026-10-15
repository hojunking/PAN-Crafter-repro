from common import *
from scipy.ndimage import gaussian_filter, shift as image_shift
from scipy.optimize import minimize
from pa.warp import warp_pan

def plane(x):
 x=gaussian_filter(x.astype(np.float64),sigma=(0,2,2))[:,8:-8,8:-8]
 x=(x-x.mean((1,2),keepdims=True))/np.maximum(x.std((1,2),keepdims=True),1e-8)
 return x.mean(0)
def shift(a,b):
 ref,mov=plane(a),plane(b)
 def objective(c):
  shifted=image_shift(mov,c,order=3,mode='nearest',prefilter=True)
  return float(np.mean((ref[8:-8,8:-8]-shifted[8:-8,8:-8])**2))
 result=minimize(objective,[0.,0.],method='Powell',bounds=[(-2,2),(-2,2)],options={'xtol':1e-5,'ftol':1e-8,'maxiter':100})
 return result.x.tolist()
def main():
 setup();d=dataset('val');ids=np.linspace(0,len(d['pan'])-1,64,dtype=int);rows=[];syn=[]
 with torch.no_grad():
  for i in ids:
   b=tensors(d,slice(i,i+1));bic=dn(torch.nn.functional.interpolate(b['ms'],scale_factor=4,mode='bicubic'))[0]
   # Values above are an un-clipped reference below; no model output convention is applied.
   bic=((torch.nn.functional.interpolate(b['ms'],scale_factor=4,mode='bicubic').cpu().numpy()[0]+1)/2*2047.)
   gt=d['gt'][i];lms=d['lms'][i]
   rows.append(dict(index=int(i),gt_to_bic=shift(gt,bic),gt_to_lms=shift(gt,lms),lms_to_bic=shift(lms,bic),
     gt_bic_rmse_blurred=float(np.sqrt(np.mean((plane(gt)-plane(bic))**2))),gt_lms_rmse_blurred=float(np.sqrt(np.mean((plane(gt)-plane(lms))**2)))))
   for eps in ((.5,-.5),(-.5,.5),(1.,-1.)):
    moved=warp_pan(torch.from_numpy(gt[None]).to('cuda:0'),torch.tensor([eps],device='cuda:0')).cpu().numpy()[0]
    c=shift(gt,moved);syn.append(dict(index=int(i),known_sampling_offset=list(eps),estimated_shift_to_reference=c,error_l2=float(np.linalg.norm(np.array(c)-eps))))
 out=dict(protocol='reference-only proxy; band standardize/common Gaussian sigma2, fit constant content translation on interior32 by cubic/Powell within +-2px',
          identity=d['identity'],indices=ids.tolist(),per_patch=rows,synthetic=syn,
          summary={k:dict(median=np.median([r[k] for r in rows],axis=0).tolist(),mean_norm=float(np.linalg.norm([r[k]for r in rows],axis=1).mean())) for k in ('gt_to_bic','gt_to_lms','lms_to_bic')},
          synthetic_mean_epe=float(np.mean([r['error_l2']for r in syn])),synthetic_median_epe=float(np.median([r['error_l2']for r in syn])),
          warning='Estimated shifts are blur/content-dependent proxies; LMS is not geometric ground truth. No claim that decimation phase is proven.',at=stamp())
 write(OUT/'reference_check.json',out);print(json.dumps({k:v for k,v in out.items()if k in ('summary','synthetic_mean_epe','synthetic_median_epe')},indent=2),flush=True)
if __name__=='__main__':main()
