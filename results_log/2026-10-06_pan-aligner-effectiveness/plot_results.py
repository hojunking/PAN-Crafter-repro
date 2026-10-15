"""Measured-data figures and summary tables; never generates image content."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import OUT, CASES, EPS, paths, split_data, tensors, features, warp_np

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':140,'savefig.dpi':180})
SENSORS=list(CASES)
COLORS=['#517caa','#d28237','#448570']


def read(sensor,kind): return json.loads((OUT/f'{sensor}_{kind}.json').read_text())


def gain(rows):
    c=np.array([r['delta'] for r in rows]); ce=np.array([r['epsilon_prediction'] for r in rows])
    return -np.sum((ce-c[:,None])*EPS,axis=(1,2))/np.sum(EPS**2)


def interval(x):
    x=np.asarray(x);rng=np.random.default_rng(20261006)
    boot=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return [float(v) for v in np.percentile(boot,[2.5,97.5])]


def summarize():
    out={};flat=[]
    for sensor in SENSORS:
        d=read(sensor,'diagnostics'); t=read(sensor,'teacher');ss={}
        for split,sp in d['splits'].items():
            rows=sp['rows'];c=np.array([r['delta'] for r in rows]);norm=np.linalg.norm(c,axis=1)
            summary=dict(mean_delta=c.mean(0).tolist(),mean_norm=norm.mean(),mean_norm_ci=interval(norm),
                         norm_min=float(norm.min()),norm_max=float(norm.max()),
                         gain=float(gain(rows).mean()),gain_ci=interval(gain(rows)),
                         teacher_norm=float(np.mean([np.linalg.norm(r['delta']) for r in t['splits'][split]])),
                         teacher_gain=float(gain(t['splits'][split]).mean()))
            for key in ['input','gt']:
                if key not in rows[0]: continue
                raw=np.array([r[key]['shift'] for r in rows]);cur=np.array([r[key]['after_native'] for r in rows]);four=np.array([r[key]['after_x4'] for r in rows]);
                nr=np.linalg.norm(raw,axis=1);nc=np.linalg.norm(cur,axis=1);nf=np.linalg.norm(four,axis=1)
                intensity=np.array([r[key]['intensity_check']['shift'] for r in rows])
                summary[key]=dict(raw_mean=float(nr.mean()),raw_ci=interval(nr),raw_median=float(np.median(nr)),
                                  raw_vector=raw.mean(0).tolist(),after_mean=float(nc.mean()),after4_mean=float(nf.mean()),
                                  improved=int((nc<nr).sum()),improved4=int((nf<nr).sum()),
                                  change_ci=interval(nc-nr),band_spread=float(np.mean([r[key]['spread'] for r in rows])),
                                  intensity_norm=float(np.linalg.norm(intensity,axis=1).mean()),
                                  method_disagreement=float(np.linalg.norm(raw-intensity,axis=1).mean()),
                                  boundary_hits=sum(r[key]['boundary'] for r in rows),
                                  ncc={str(a):float(np.mean([r[key]['ncc_alpha'][str(a)] for r in rows])) for a in [0,1,4]})
            if split=='rr': summary['base_gt_phase_vector']=np.mean([np.median(r['base_gt_phase'],axis=0) for r in rows],axis=0).tolist()
            if split=='fr':
                paired={}
                for role in ['student','teacher']:
                    rr=[r['paired_R2'][role] for r in t['splits']['fr']]
                    f=np.array([r['fr_delta'] for r in rr]);r=np.array([r['reduced_delta'] for r in rr])
                    ef=np.array([z['epsilon_fr'] for z in rr[0]['responses']]);er=ef/4
                    rf=np.array([[z['fr_response'] for z in x['responses']] for x in rr]);rs=np.array([[z['reduced_response'] for z in x['responses']] for x in rr])
                    gf=-(rf*ef).sum((1,2))/(ef**2).sum();gr=-(rs*er).sum((1,2))/(er**2).sum()
                    paired[role]=dict(fr_norm=float(np.linalg.norm(f,axis=1).mean()),reduced_norm=float(np.linalg.norm(r,axis=1).mean()),
                                      ratio_norm=float(np.linalg.norm(f,axis=1).mean()/np.linalg.norm(r,axis=1).mean()),
                                      fr_gain=float(gf.mean()),reduced_gain=float(gr.mean()),
                                      vector_scale_error=float(np.linalg.norm(f-4*r,axis=1).mean()))
                summary['paired']=paired
            ss[split]=summary
            for r in rows:
                rec=dict(sensor=sensor,split=split,scene_index_0based=r['index'],dy=r['delta'][0],dx=r['delta'][1])
                for key in ['input','gt']:
                    for field in ['shift','after_native','after_x4']:
                        rec[key+'_'+field+'_norm']=np.linalg.norm(r[key][field]) if key in r else ''
                flat.append(rec)
        if (OUT/f'{sensor}_quality.json').exists():
            ss['quality']=read(sensor,'quality')['alphas']
        out[sensor]=ss
    (OUT/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    with (OUT/'per_scene.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    return out


def save(fig,name):
    fig.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close(fig)


def plots(summary):
    fig,ax=plt.subplots(1,3,figsize=(13,4.2))
    for a,sensor in zip(ax,SENSORS):
        d=read(sensor,'diagnostics');vals=[]
        for split,key in [('rr','input'),('rr','gt'),('fr','input')]:
            rows=d['splits'][split]['rows'];vals.append([np.linalg.norm(r[key]['shift']) for r in rows])
        a.boxplot(vals,labels=['RR vs input','RR vs GT','FR vs input'],showfliers=True)
        for j,v in enumerate(vals):a.scatter(j+1+np.linspace(-.1,.1,len(v)),v,s=13,alpha=.6,color=COLORS[j])
        a.set_title(sensor);a.set_ylabel('Estimated shift (current PAN pixels)');a.tick_params(axis='x',rotation=15)
    fig.suptitle('Structural registration proxies, 20 scenes per split; not physical ground truth',y=1.04)
    save(fig,'01_native_registration.png')
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for a,sensor in zip(ax,SENSORS):
        s=summary[sensor];x=np.arange(3)
        a.bar(x-.18,[s[k]['gain'] for k in ['train','rr','fr']],.34,label='Selected Student',color=COLORS[0])
        a.bar(x+.18,[s[k]['teacher_gain'] for k in ['train','rr','fr']],.34,label='Exact donor Teacher',color=COLORS[1])
        a.axhline(1,color='gray',ls='--',label='Ideal correction');a.set_ylim(0,1.05)
        a.set_xticks(x,['Train 64','RR 256','FR 512']);a.set_title(sensor);a.set_ylabel('Response gain (ideal = 1)')
    ax[0].legend(fontsize=8);fig.suptitle('Known perturbations: four axes × radii 0.25, 0.5, 1, 2 px',y=1.02)
    save(fig,'02_offset_response.png')
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for a,sensor in zip(ax,SENSORS):
        rows=read(sensor,'diagnostics')['splits']['fr']['rows'];d=np.array([r['input']['shift'] for r in rows]);c=np.array([r['delta'] for r in rows])
        a.scatter(d[:,1],d[:,0],label='PAN-to-input proxy',color=COLORS[1]);a.scatter(c[:,1],c[:,0],label='Applied sampling shift',color=COLORS[0])
        for v,w in zip(d,c):a.plot([v[1],w[1]],[v[0],w[0]],color='gray',alpha=.2,lw=.7)
        a.axhline(0,color='gray',lw=.5);a.axvline(0,color='gray',lw=.5)
        a.set_title(sensor+' FR');a.set_xlabel('dx (PAN px)');a.set_ylabel('dy (PAN px)');a.set_aspect('equal',adjustable='datalim')
    ax[0].legend(fontsize=8);save(fig,'03_fr_vectors.png')
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for a,sensor in zip(ax,SENSORS):
        labels=['RR/input','RR/GT','FR/input'];x=np.arange(3)
        for j,(field,alpha) in enumerate([('raw_mean','0'),('after_mean','1'),('after4_mean','4')]):
            vals=[summary[sensor][split][key][field] for split,key in [('rr','input'),('rr','gt'),('fr','input')]]
            a.bar(x+(j-1)*.25,vals,.24,label='alpha='+alpha,color=COLORS[j])
        a.set_title(sensor);a.set_xticks(x,labels);a.set_ylabel('Re-estimated residual shift (px)')
    ax[0].legend();save(fig,'04_residual_registration.png')
    fig,ax=plt.subplots(3,3,figsize=(12,9))
    for i,sensor in enumerate(SENSORS):
        q=summary[sensor].get('quality',{})
        for j,(split,key,label) in enumerate([('rr','ergas','RR ERGAS (lower better)'),('rr','scc','RR SCC (higher better)'),('fr','hqnr','FR HQNR (higher better)')]):
            a=ax[i,j]
            if not all(f'{split}_{v}' in q for v in [0,1,4]):continue
            values=[q[f'{split}_{v}']['mean'][key] for v in [0,1,4]]
            a.plot([0,1,4],values,'o-',color=COLORS[i]);a.set_title(sensor+' '+label)
            a.set_xticks([0,1,4]);a.ticklabel_format(useOffset=False,axis='y')
            a.set_xlabel('alpha: applied shift = alpha × prediction')
    fig.tight_layout();save(fig,'05_frozen_checkpoint_quality.png')
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    for a,sensor in zip(ax,SENSORS):
        z=summary[sensor]['fr']['paired'];x=np.arange(2)
        a.bar(x-.18,[z[k]['fr_gain'] for k in ['student','teacher']],.35,label='Native FR (512)',color=COLORS[0])
        a.bar(x+.18,[z[k]['reduced_gain'] for k in ['student','teacher']],.35,label='Same-FOV reduced (128)',color=COLORS[1])
        a.axhline(1,color='gray',ls='--');a.set_ylim(0,1.05);a.set_title(sensor);a.set_xticks(x,['Student','Teacher']);a.set_ylabel('Known-shift correction gain')
    ax[0].legend(fontsize=8);fig.suptitle('Paired scale probe: injected 2 FR px = 0.5 reduced px',y=1.02)
    save(fig,'06_paired_scale.png')


def stretch(x,lo=None,hi=None):
    if lo is None:lo,hi=np.percentile(x,[1,99])
    return np.clip((x-lo)/max(hi-lo,1e-8),0,1)


def visuals():
    selection={}
    for sensor in SENSORS:
        _,manifest,bands,dn=paths(sensor);d=read(sensor,'diagnostics')
        fig,ax=plt.subplots(2,5,figsize=(16,8));sel={}
        for rowidx,split in enumerate(['rr','fr']):
            rows=d['splits'][split]['rows'];key='gt' if split=='rr' else 'input'
            order=np.argsort([np.linalg.norm(r[key]['shift']) for r in rows]);idx=int(order[len(order)//2]);r=rows[idx]
            data=split_data(manifest,split,dn);p=data['pan'][idx,0];_,_,base=tensors(data,dn)
            ref=data['gt'][idx] if split=='rr' else (base[idx].numpy()+1)*(dn/2)
            b=r[key]['selected_bands'][-1];sp,sr=(.8,.8) if split=='rr' else (1.98,.7)
            reference=np.linalg.norm(features(ref[b],sr),axis=-1)
            panedge=np.linalg.norm(features(p,sp),axis=-1)
            # ROI is maximum reference edge energy, not best correction outcome.
            size=80;energy=cv2.boxFilter(reference,-1,(size,size),normalize=False)
            margin=size//2+16;yy,xx=np.unravel_index(energy[margin:-margin,margin:-margin].argmax(),energy[margin:-margin,margin:-margin].shape);yy+=margin;xx+=margin
            sl=(slice(yy-size//2,yy+size//2),slice(xx-size//2,xx+size//2))
            rgbids=[4,2,1] if bands==8 else [2,1,0]
            rgb=np.stack([stretch(ref[b0]) for b0 in rgbids],-1)
            ax[rowidx,0].imshow(rgb[sl]);ax[rowidx,0].set_title(f'{split.upper()} #{idx+1}: '+('GT RGB' if split=='rr' else 'Bicubic MS RGB'))
            ax[rowidx,1].imshow(stretch(p)[sl],cmap='gray');ax[rowidx,1].set_title('Native PAN, same ROI')
            rscale=np.percentile(reference,99);pscale=np.percentile(panedge,99)
            for col,alpha in enumerate([0,1,4],2):
                edge=np.linalg.norm(features(warp_np(p,np.array(r['delta'])*alpha),sp),axis=-1)
                overlay=np.zeros((*p.shape,3));overlay[...,0]=np.clip(reference/rscale,0,1);overlay[...,1]=overlay[...,2]=np.clip(edge/pscale,0,1)
                ax[rowidx,col].imshow(overlay[sl],interpolation='nearest')
                residual=np.linalg.norm(r[key][['shift','after_native','after_x4'][col-2]])
                ax[rowidx,col].set_title(f'alpha={alpha}; residual {residual:.3f}px')
            for a in ax[rowidx]:a.axis('off')
            sel[split]=dict(scene_1based=idx+1,reference=key,edge_band_0based=b,roi_yx=[int(yy-size//2),int(xx-size//2),size,size],delta=r['delta'])
        fig.suptitle(sensor+' | Red = reference edges; cyan = PAN edges; white = overlap\nReal pixels; fixed edge contrast across alpha; residual estimated on full interior',fontsize=12)
        fig.subplots_adjust(top=.84,bottom=.02,hspace=.24,wspace=.04)
        save(fig,f'07_{sensor}_real_overlays.png');selection[sensor]=sel
    (OUT/'visual_selection.json').write_text(json.dumps(selection,indent=2)+'\n')


def phase_plot():
    native=json.loads((OUT/'reference_sensitivity.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for a,sensor in zip(axes,SENSORS):
        diag=read(sensor,'diagnostics')['splits']['rr']['rows']
        b=np.array([np.median(r['base_gt_phase'],axis=0) for r in diag])
        l=np.array([np.median(r['lms_gt_phase'],axis=0) for r in native[sensor]['rr']])
        a.scatter(b[:,1],b[:,0],s=25,label='Torch bicubic base -> GT',color=COLORS[0])
        a.scatter(l[:,1],l[:,0],s=25,label='Dataset LMS -> GT',color=COLORS[1])
        a.axhline(0,color='gray',lw=.5);a.axvline(0,color='gray',lw=.5)
        a.set_xlim(-.65,.15);a.set_ylim(-.65,.15);a.set_aspect('equal');a.set_title(sensor)
        a.set_xlabel('dx sampling correction (RR px)');a.set_ylabel('dy sampling correction (RR px)')
    axes[0].legend(fontsize=8,loc='lower right');fig.suptitle('Same-band phase control: no PAN or spectral mixing in this comparison',y=1.03)
    save(fig,'08_upsampling_phase.png')


if __name__=='__main__':
    import cv2
    summary=summarize();plots(summary);visuals();phase_plot()
