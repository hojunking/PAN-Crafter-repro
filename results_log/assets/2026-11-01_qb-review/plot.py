"""Standalone publication-style summary of local QB evidence and s3 Sheet rows."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT=Path(__file__).resolve().parent
local=json.loads((OUT/'local_analysis.json').read_text())
sheet=json.loads((OUT/'sheet_snapshot.json').read_text())
runs={(r['case'],r['replica']):r for r in local['runs']}
control={'R1':runs['QB_A200_E030','R1'],'R2':runs['QB_BASE','R2'],'R3':runs['QB4_CONTROL','R3']}
colors={'R1':'#2274A5','R2':'#DC7F27','R3':'#45976B'}
markers={'R1':'o','R2':'s','R3':'^'}

s3={}
for row in sheet['early_metrics']['values']+sheet['latest_metrics']['values']:
    if len(row)<30 or row[1]!='s3':continue
    match=re.search(r'_(QB[34]_(?:CONTROL|WD000))_NOSOFT_(R\d)_',row[29])
    if match:s3[match.group(1),match.group(2)]={'HQNR':row[3],'ERGAS':row[7]}
assert all((('QB4_CONTROL' if r=='R3' else 'QB3_CONTROL'),r) in s3 for r in colors)
assert all((('QB4_WD000' if r=='R3' else 'QB3_WD000'),r) in s3 for r in colors)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                    'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
                    'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(12.4,8.2))
fig.subplots_adjust(left=.08,right=.985,top=.90,bottom=.12,hspace=.41,wspace=.25)
fig.suptitle('QB: original-PAN information is critical; new tuning has not reached the joint target',
             fontsize=14,fontweight='semibold',x=.08,ha='left',y=.973)

cells=['DIFF_UBIL','DIFF_BASE','P0_UBIL','P0_BASE']
labels=['DIFF\nUBIL','DIFF\nBASE','P0\nUBIL','P0\nBASE']
factorial={f['replica']:f['cells'] for f in local['factorial_10K']}
for ax,metric,title in [(axes[0,0],'HQNR','A  Removing DIFF lowers HQNR at 10K'),
                        (axes[0,1],'D_s','B  The loss is predominantly spatial')]:
    for rep,offset in [('R1',-.09),('R2',.09)]:
        values=[factorial[rep][c][metric] for c in cells]
        ax.scatter(np.arange(4)+offset,values,color=colors[rep],marker=markers[rep],
                   s=62,edgecolors='white',linewidths=.6,label=rep,zorder=4)
    ax.axvspan(1.5,3.5,color='#f1f3f5',zorder=0)
    ax.set(xticks=range(4),xticklabels=labels,xlim=(-.4,3.4),ylabel=metric,title=title)
    ax.text(.97,.97,'Same donor, shifts, calibration and seeds',transform=ax.transAxes,
            ha='right',va='top',fontsize=8.3,color='#555555')
    if metric=='HQNR':
        ax.axhline(.9315,color='#A52B37',linestyle='--',linewidth=1.1)
        ax.text(3.34,.934,'HQNR target .9315',ha='right',va='bottom',fontsize=8.5,color='#A52B37')
        ax.set_ylim(.72,.975)
        ax.legend(loc='lower left',ncol=2,frameon=False,fontsize=9)
    else:
        ax.set_ylim(0,.27)
        ax.text(2.5,.12,'DIFF removal worsens HQNR\nin all 20 FR scenes\nfor both seeds and samplers',
                ha='center',va='center',fontsize=9,color='#555555')

ax=axes[1,0]
groups=[('CONTROL',lambda rep:control.get(rep)),
        ('DIFF + BASE',lambda rep:runs.get(('QB4_DIFFBASE',rep))),
        ('WD = 0',lambda rep:runs.get(('QB4_WD000',rep)))]
for rep,offset in [('R1',-.16),('R2',0),('R3',.16)]:
    for x,(_,lookup) in enumerate(groups):
        record=lookup(rep)
        if record is None:continue
        v=record['representative'];assert v['HQNR']>=.9315
        ax.scatter(x+offset,v['ERGAS'],color=colors[rep],marker=markers[rep],s=62,
                   edgecolors='white',linewidths=.6,zorder=4)
ax.axhline(3.5,color='#A52B37',linestyle='--',linewidth=1.1)
ax.text(2.37,3.5005,'ERGAS target 3.5',ha='right',va='bottom',fontsize=8.5,color='#A52B37')
ax.set(xticks=range(3),xticklabels=[g[0] for g in groups],xlim=(-.4,2.4),ylim=(3.496,3.528),
       ylabel='Representative RR ERGAS',title='C  s1: existing control remains best')
ax.text(.03,.97,'All plotted representatives pass HQNR',transform=ax.transAxes,
        ha='left',va='top',fontsize=8.3,color='#555555')
ax.legend(handles=[Line2D([],[],color=colors[r],marker=markers[r],linestyle='None',label=r) for r in colors],
          loc='lower left',bbox_to_anchor=(0,.15),frameon=False,ncol=3,fontsize=9)

ax=axes[1,1]
for rep,offset in [('R1',-.12),('R2',0),('R3',.12)]:
    a=runs['QB4_WD000',rep]['representative'];b=control[rep]['representative']
    name='QB4' if rep=='R3' else 'QB3'
    aa=s3[name+'_WD000',rep];bb=s3[name+'_CONTROL',rep]
    for x,delta in [(0,a['ERGAS']-b['ERGAS']),(1,aa['ERGAS']-bb['ERGAS'])]:
        ax.vlines(x+offset,0,delta,color=colors[rep],alpha=.45,linewidth=1.2)
        ax.scatter(x+offset,delta,color=colors[rep],marker=markers[rep],s=62,
                   edgecolors='white',linewidths=.6,zorder=4)
ax.axhline(0,color='#555555',linewidth=.9)
ax.set(xticks=[0,1],xticklabels=['s1: local Stage 1','s3: local Stage 1'],xlim=(-.4,1.4),
       ylim=(-.018,.018),ylabel='Representative ERGAS: WD0 minus CONTROL',
       title='D  WD0 effects change with seed and server')
ax.text(.03,.96,'Above zero: worse',transform=ax.transAxes,va='top',fontsize=8.5,color='#555555')
ax.text(.03,.04,'Below zero: better',transform=ax.transAxes,va='bottom',fontsize=8.5,color='#555555')

for ax in axes.flat:
    ax.grid(axis='y',alpha=.18,linewidth=.7);ax.set_axisbelow(True)
    ax.title.set_position((.0,1.02));ax.title.set_ha('left')

r3=runs['QB4_WD000','R3']
status=f"s1 WD0 R3: {r3['status']} through {r3['latest']['step']//1000}K."
fig.text(.08,.055,'P0 = corrected PAN; DIFF adds corrected minus original PAN. Both retain the MS base. Aligner scale = 1.\n'
         'Representatives are individual HQNR-qualified checkpoints, not model averages; within-server seed pairing. '+status,
         fontsize=8.5,ha='left',va='center',color='#555555')
fig.savefig(OUT/'qb_summary.png',dpi=180)
fig.savefig(OUT/'qb_summary.pdf')
print(json.dumps({'png':str(OUT/'qb_summary.png'),'pdf':str(OUT/'qb_summary.pdf'),
                  'local_snapshot':local['captured_utc'],'sheet_snapshot':sheet['captured_utc'],'r3_status':status}))
