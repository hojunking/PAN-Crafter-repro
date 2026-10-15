from common import *
import argparse,itertools
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args();setup()
 for seed in SEEDS:
  print('SEED',seed,flush=True)
  # Selection is frozen using validation ONLY, before any new RR or FR scores.
  vp=OUT/'validation'/f'{seed}.json'
  if not vp.exists():
   vals={}
   for t in (0.,.25,.5,.75,1.):
    m=model_for(seed,u_t=t);vals[f'Ab_U{t:.2f}']=validation(m);del m
   m=model_for(seed);bvals={}
   for by,bx in itertools.product((-.1,0.,.1),repeat=2):
    name=f'b{by:+.1f}_{bx:+.1f}';bvals[name]=dict(bias=[by,bx],**validation(m,(by,bx)))
   vref=validation(m,ref='lms');del m
   ts=min(vals,key=lambda k:vals[k]['mean_ergas']);bs=min(bvals,key=lambda k:bvals[k]['mean_ergas'])
   write(vp,dict(at=stamp(),val_identity=dataset('val')['identity'],interpolation=vals,bias=bvals,lms_ref=vref,selected_t=ts,selected_bias=bs,selection_before_test=True))
  vr=read(vp)
  for name,al,ut in [('AbUb',False,0.),('AlUl',True,1.),('AbUl',False,1.),('AlUb',True,0.)]:
   m=model_for(seed,a_last=al,u_t=ut);evaluate(m,seed,name,description='same-run A/U crossed checkpoint');del m
   if args.pilot:return
  for t in (.25,.5,.75):
   m=model_for(seed,u_t=t);evaluate(m,seed,f'Ab_U{t:.2f}',validation_result=vr['interpolation'][f'Ab_U{t:.2f}'],description='U weight interpolation, A fixed to best');del m
  m=model_for(seed)
  for name,v in vr['bias'].items():
   b=tuple(v['bias'])
   if b==(0.,0.):continue
   evaluate(m,seed,name,bias=b,validation_result=v,description='bias sensitivity; deploy candidate chosen on val before RR')
  evaluate(m,seed,'LMS_Aref',ref='lms',validation_result=vr['lms_ref'],description='A reference only; U input/residual base unchanged')
  del m;torch.cuda.empty_cache()
 print('INFERENCE COMPLETE',flush=True)
if __name__=='__main__':main()
