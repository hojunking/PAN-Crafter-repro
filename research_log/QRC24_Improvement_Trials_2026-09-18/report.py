"""Build an evidence-complete report only after all predeclared cases finish."""
from pathlib import Path
import base64,csv,datetime,hashlib,html,itertools,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
SEEDS=(1234,3407,41001)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['AbUb','AlUl','AbUl','AlUb','Ab_U0.25','Ab_U0.50','Ab_U0.75']
names += [f'b{y:+.1f}_{x:+.1f}' for y,x in itertools.product((-.1,0.,.1),repeat=2) if (y,x)!=(0.,0.)]
names += ['LMS_Aref','FT_bicubic_joint','FT_lms_Aref_joint','FT_bicubic_Afreeze']
R={};missing=[]
for s,n in itertools.product(SEEDS,names):
 f=P/'results'/str(s)/n/'result.json'
 if not f.exists():missing.append((s,n));continue
 r=read(f);assert r['complete'] and r['state_unchanged'] and r['network_disabled'];R[s,n]=r
if missing:raise SystemExit('Pending cases: '+str(missing))
assert len(R)==57
M=read(P/'manifest.json')
for rel,h in M['source_hashes'].items():assert sha(P/'source'/rel)==h
def e(s,n):return R[s,n]['results']['rr']['mean']['ergas']
def h(s,n):return R[s,n]['results']['fr']['mean']['hqnr']
def v(s,n):
 r=R[s,n]['validation']
 return None if r is None else r['mean_ergas']
def mean(n,metric='e'):return float(np.mean([(e if metric=='e' else h)(s,n) for s in SEEDS]))
def paired(s,n):
 a=R[s,n]['results']['rr']['per_scene'];b=R[s,'AbUb']['results']['rr']['per_scene']
 d=np.array([x['ergas']-y['ergas'] for x,y in zip(a,b)]);rng=np.random.default_rng(20260918)
 ci=np.percentile(d[rng.integers(0,20,(10000,20))].mean(1),[2.5,97.5])
 return {'delta_ergas':float(d.mean()),'improved_scenes':int((d<0).sum()),'scene_ci95':ci.tolist()}
flat=[];scenes=[]
for s,n in itertools.product(SEEDS,names):
 r=R[s,n];d=paired(s,n)
 row=dict(seed=s,name=n,**r['results']['rr']['mean'],**r['results']['fr']['mean'],val_ergas=v(s,n),
          delta_ergas=d['delta_ergas'],delta_hqnr=h(s,n)-h(s,'AbUb'),improved_scenes=d['improved_scenes'],
          ci95_low=d['scene_ci95'][0],ci95_high=d['scene_ci95'][1],joint_pass=r['joint_pass'])
 flat.append(row)
 for split in ('rr','fr'):
  for x in r['results'][split]['per_scene']:
   scenes.append(dict(seed=s,name=n,split=split,**{k:x.get(k)for k in ('scene','ergas','sam','psnr','ssim','scc','q8','hqnr','d_lambda','d_s')},dy=x['delta'][0],dx=x['delta'][1]))
for name,rows in [('all_variants.csv',flat),('all_scenes.csv',scenes)]:
 with (P/name).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

factorial=[]
for s in SEEDS:
 factorial.append(dict(seed=s,A_ergas=e(s,'AlUb')-e(s,'AbUb'),U_ergas=e(s,'AbUl')-e(s,'AbUb'),
  interaction_ergas=e(s,'AlUl')-e(s,'AbUl')-e(s,'AlUb')+e(s,'AbUb'),
  A_hqnr=h(s,'AlUb')-h(s,'AbUb'),U_hqnr=h(s,'AbUl')-h(s,'AbUb'),
  interaction_hqnr=h(s,'AlUl')-h(s,'AbUl')-h(s,'AlUb')+h(s,'AbUb')))
with (P/'factorial_effects.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(factorial[0]));w.writeheader();w.writerows(factorial)
valsel={s:read(P/'validation'/f'{s}.json')for s in SEEDS}
ft=[]
for s in SEEDS:
 for arm in ('bicubic_joint','lms_Aref_joint','bicubic_Afreeze'):
  p=P/'finetune'/arm if s==1234 else P/'finetune'/str(s)/arm
  meta=read(p/'complete.json');assert meta['updates']==1000 and meta['teacher_unchanged'];assert sha(p/'model.safetensors')==meta['model_sha256']
  assert meta['A_unchanged']==(arm=='bicubic_Afreeze');ft.append(dict(seed=s,**meta))
 assert len({x['initial_hash'] for x in ft if x['seed']==s})==1
old=P.parent/'QRC24_RR_Crop64_Report_2026-09-18';cohort=read(old/'cohort.json');baseline=[]
for a in cohort['assets']:
 if a['profile']!='G23':continue
 assert sha(old/a['snapshot']/'model.safetensors')==a['checkpoint_sha256']
 x=read(old/'results'/a['run']/a['checkpoint']/'result.json');name='AbUb'if a['checkpoint']=='best_hqnr'else'AlUl'
 baseline.append(dict(seed=a['seed'],name=name,delta_ergas=e(a['seed'],name)-x['modes']['A_ON']['summary']['ergas']['mean']))
checks=dict(n_variants=len(R),n_outputs=len(scenes),n_finetune=len(ft),updates_per_arm=1000,baseline_reproduction=baseline,
 frozen_source_hashes_match=True,source_checkpoint_hashes_match=True,finetune_same_initial_within_seed=True,
 teacher_unchanged=True,A_freeze_unchanged=True,no_sheet_network=True)
write(P/'checks.json',checks)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(12,4.6));ts=[0,.25,.5,.75,1];tn=['AbUb','Ab_U0.25','Ab_U0.50','Ab_U0.75','AbUl']
for s in SEEDS:
 axs[0].plot(ts,[e(s,n)for n in tn],'.-',label=f'seed {s}')
 axs[1].plot(ts,[h(s,n)for n in tn],'.-',label=f'seed {s}')
axs[0].axhline(2.04,c='gray',ls='--');axs[1].axhline(.9585,c='gray',ls='--')
axs[0].set_ylabel('RR ERGAS (lower is better)');axs[1].set_ylabel('FR raw HQNR (higher is better)')
for ax in axs:ax.set_xlabel('Late U weight t (A fixed to selected checkpoint)');ax.set_xticks(ts);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('U weight interpolation: RR gain and HQNR trade-off');fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(P/f'interpolation.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
grids=[]
for s in SEEDS:
 grids.append(np.array([[0 if (y,x)==(0,0)else e(s,f'b{y:+.1f}_{x:+.1f}')-e(s,'AbUb')for x in (-.1,0,.1)]for y in (-.1,0,.1)]))
for ax,s,g in zip(axs,SEEDS,grids):
 im=ax.imshow(g,cmap='RdYlBu_r',vmin=0,vmax=max(x.max()for x in grids),origin='lower')
 for y,x in itertools.product(range(3),repeat=2):ax.text(x,y,f'{g[y,x]:+.3f}',ha='center',va='center',fontsize=10)
 ax.set_xticks(range(3),['-0.1','0','+0.1']);ax.set_yticks(range(3),['-0.1','0','+0.1']);ax.set_xlabel('dx bias (RR PAN px)');ax.set_ylabel('dy bias (RR PAN px)');ax.set_title(f'seed {s}')
fig.colorbar(im,ax=axs,label='ERGAS change from unmodified model');fig.suptitle('Constant offset: 3 x 3 predeclared sensitivity grid')
for ext in ('png','pdf'):fig.savefig(P/f'bias_grid.{ext}',dpi=180)
plt.close(fig)

md=[];web=[]
def title(t):md.append('**'+t+'**\n');web.append('<h2>'+html.escape(t)+'</h2>')
def para(t):md.append(t+'\n');web.append('<p>'+html.escape(t)+'</p>')
def table(head,rows):
 md.append('| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |'for r in rows)+'\n')
 web.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>'for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in r)+'</tr>'for r in rows)+'</tbody></table></div>')
def figure(name,caption):
 md.append(f'![{caption}]({P.name}/{name})\n');enc=base64.b64encode((P/name).read_bytes()).decode();web.append(f'<figure><img src="data:image/png;base64,{enc}"><figcaption>{html.escape(caption)}</figcaption></figure>')
def link(label,name):md.append(f'[{label}]({P.name}/{name})\n');web.append('<p><a href="'+name+'">'+html.escape(label)+'</a></p>')
def metric_table(selected):
 table(['seed','조합','ERGAS↓','ΔERGAS','HQNR↑','ΔHQNR','HQNR 하한'],[
  [s,n,f'{e(s,n):.6f}',f'{e(s,n)-e(s,"AbUb"):+.6f}',f'{h(s,n):.7f}',f'{h(s,n)-h(s,"AbUb"):+.7f}','통과'if h(s,n)>=.9585 else '미달']for s,n in selected])

# Narrative is appended below so the script also documents the interpretation.
title('QRC24 개선 후보 전수 비교 — A/U 교차·U 보간·상수 편향·MS 참조와 예비학습')
para('작성: '+datetime.datetime.now().astimezone().isoformat(timespec='seconds')+' · G23 seed1234/3407/41001 · Sheet 업로드 없음')
title('결론')
para('RR 개선과 HQNR 하락은 주로 U-Net의 후반 학습 상태를 따라갔다. A/U 교환으로 후반 U의 RR 이득은 유지됐지만 HQNR을 되찾지는 못했다. U 보간은 두 성능 사이의 절충점을 제공했고, 상수 편향과 LMS 참조의 단순 교체는 이번 범위에서 RR 개선책이 되지 못했다.')
para(f"세 seed의 기존 선택 평균 ERGAS {mean('AbUb'):.6f} → 기존 A+50K U {mean('AbUl'):.6f}; 평균 HQNR {mean('AbUb','h'):.7f} → {mean('AbUl','h'):.7f}. A만 50K 것으로 바꾼 평균 ERGAS 차이는 {mean('AlUb')-mean('AbUb'):+.6f}다.")
para(f"seed1234의 사전 지정 t=.25 U 보간은 ERGAS {e(1234,'AbUb'):.6f} → {e(1234,'Ab_U0.25'):.6f}, HQNR {h(1234,'AbUb'):.7f} → {h(1234,'Ab_U0.25'):.7f}였다. HQNR .9585 하한 여유는 {h(1234,'Ab_U0.25')-.9585:.7f}에 불과하다. 이 후보를 전체 seed에서 안정된 개선이나 최종 채택 모델로 선언하지 않는다.")
joint=sum(r['joint_pass'] for r in R.values());para(f"측정한 57개 조합 중 ERGAS<2.040 및 raw HQNR≥.9585를 동시에 통과한 후보는 {joint}개다. 이번 비교에서 원본 모델이나 실행 중 학습을 자동 교체하지 않았다.")

title('실행 범위와 선택 규칙')
table(['항목','실행'],[
 ['A/U 교차','3 seed × 4 조합 = 12개'],['U 가중치 보간','3 seed × t=.25/.5/.75 = 신규 9개; endpoint는 교차 조합 공유'],
 ['상수 편향','3 seed × 3×3 grid = 27개; b=(0,0) 3개는 기존 기준과 공유 → 신규 24개'],
 ['A의 LMS 참조 직접 교체','3 seed × 1 = 3개'],['예비학습','3 seed × 기존 참조 joint / LMS 참조 joint / 기존 참조 A 동결 = 9개'],
 ['합계','중복 제외 57개 모델/정책; 각 RR20+FR20 = 2,280개 주 평가 출력'],
 ['validation','1080개 native64 patch의 center32 ERGAS; seed별 보간/편향을 RR·FR 계산 전에 선택'],
 ['RR','20장, 20:-21 crop, DN2047, Q8 block32; FP32 추론'],['FR','배포 mat20 전체20장, 원 PAN·LMS 기준 raw-original HQNR; reference warp 없음'],
 ['격리','고정 소스 181개·기존 checkpoint snapshot 사용; 네트워크 차단·Sheet 모듈 import 금지'],
])
para('보간과 bias의 validation 선택은 고정된 모델별로 수행했다. native64 validation은 큰 영역 RR256과 동일 분포를 보장하지 않는다. 아래에서 HQNR 하한을 보고 t=.25를 강조한 것은 test FR20에 대한 탐색적 해석이며, 독립 validation으로 공동목표 정책이 확정된 결과는 아니다. RR20 최적값으로 영상별 보정량을 고르는 처리는 하지 않았다.')
table(['seed','validation이 고른 U 비율','validation이 고른 공통 bias','선택 시각'],[[s,valsel[s]['selected_t'],valsel[s]['selected_bias'],valsel[s]['at']]for s in SEEDS])

title('1. A/U 교차: 후반 U가 주된 차이를 만든다')
para('Ab/Ub는 기존 best_hqnr의 A/U, Al/Ul은 같은 run의 last@50000이다. 서로 다른 seed 간 교환은 하지 않았다.')
metric_table([(s,n)for s in SEEDS for n in ('AbUb','AlUl','AbUl','AlUb')])
table(['seed','A만 교체 ΔERGAS','U만 교체 ΔERGAS','상호작용 ΔERGAS','A만 교체 ΔHQNR','U만 교체 ΔHQNR'],[
 [r['seed'],f"{r['A_ergas']:+.6f}",f"{r['U_ergas']:+.6f}",f"{r['interaction_ergas']:+.6f}",f"{r['A_hqnr']:+.7f}",f"{r['U_hqnr']:+.7f}"]for r in factorial])
para('상호작용은 AlUl−AbUl−AlUb+AbUb로 계산했다. 이 수치는 이미 학습된 두 시점의 교환 효과를 나타낸다. A가 전체 학습에서 필요 없다는 뜻이나, U가 기하 문제의 유일한 원인이라는 뜻은 아니다. 기존 NOA 결과와도 모순되지 않는다.')

title('2. U 보간: RR과 HQNR의 절충')
metric_table([(s,n)for s in SEEDS for n in ('Ab_U0.25','Ab_U0.50','Ab_U0.75')])
figure('interpolation.png','기존 A 고정, 후반 U 비율에 따른 RR ERGAS와 FR HQNR')
r=paired(1234,'Ab_U0.25');lo,hi=r['scene_ci95']
para(f"seed1234 t=.25의 RR 개선 장면은 {r['improved_scenes']}/20이다. 20장 대응 차이의 descriptive bootstrap 95% 구간은 [{lo:+.6f}, {hi:+.6f}]이다. 장면을 10,000회 재표집했으며 독립적인 3-seed 일반화 검정으로 사용하지 않는다.")
para('U 비율을 높이면 RR이 좋아지는 반면 HQNR이 낮아지는지 각 seed의 곡선으로 판단한다. 다른 seed의 HQNR 자체가 하한 미달이라면 seed1234의 보간 성공을 공통 recipe 성공으로 옮겨 해석할 수 없다. 두 checkpoint의 가중치 보간은 새 학습이나 논문의 SWA 실행과 구분한다.')

title('3. 상수 편향: 사전 지정 ±.1px grid')
figure('bias_grid.png','각 셀은 같은 seed의 원본 대비 RR ERGAS 변화; 중심이 b=0')
bias_names=[n for n in names if n.startswith('b')]
table(['seed','b=0 ERGAS','nonzero bias 중 최저 ERGAS','nonzero bias 악화 수','validation 선택'],[
 [s,f'{e(s,"AbUb"):.6f}',f'{min(e(s,n)for n in bias_names):.6f}',f'{sum(e(s,n)>e(s,"AbUb")for n in bias_names)}/8',valsel[s]['selected_bias']]for s in SEEDS])
para('이 grid에서 개선이 없다는 결과는 ±.1px의 미리 정한 공통 보정에 대한 것이다. 더 작은 편향, 다른 센서, 지역별 변위를 모두 기각하지 않는다. 결과를 본 뒤 grid를 더 촘촘하게 확장하지 않았다. q가 절대 편향을 식별하지 못한다는 수학적 사실과, 현재 모델에 보정할 만한 공통 편향이 실제로 있다는 주장은 구분된다.')

title('4. MS 참조 점검과 직접 교체')
ref=read(P/'reference_check.json');failed=read(P/'reference_check_phase_failed.json')
table(['비교','추정 content shift 중앙값 (dy,dx)','평균 norm px'],[[k,str([round(x,5)for x in a['median']]),f"{a['mean_norm']:.5f}"]for k,a in ref['summary'].items()])
para(f"validation에서 사전 지정한 64 patch를 사용했다. 같은 band의 Gaussian σ2 공통 저역통과 후 band별 표준화를 하고, 중앙32에서 ±2px translation을 추정했다. 표의 부호는 moving 영상을 reference에 맞추기 위한 content translation이며 sampler offset 부호와 반대다. 알려진 ±.5/1px 이동 sanity의 평균 EPE는 {ref['synthetic_mean_epe']:.5f}px, 중앙값은 {ref['synthetic_median_epe']:.5f}px였다.")
para(f"초기 FFT phase-correlation은 알려진 이동에서 평균 EPE {failed['synthetic_mean_epe']:.5f}px로 부정확해 참조 결론에 사용하지 않았다. 공간영역의 내부 support로 수정한 뒤 합성 이동 검사를 통과한 결과를 위에 사용했다. 실패 원자료도 보존했다. 참조 사이의 약 .7px 상대 차이는 관찰됐지만, blur/내용/보간 커널에 의존하는 proxy이므로 센서 오정합이나 decimation 위상 버그의 확정 증거는 아니다.")
metric_table([(s,'LMS_Aref')for s in SEEDS])
para('LMS를 A의 MS 참조로만 사용했다. U의 MS 입력·bicubic 확대·residual base·GT·평가 reference는 그대로다. 추론 시 참조만 바꾸는 것은 학습 분포 변경이므로 이 악화만으로 LMS 참조를 사용한 재학습 전체를 기각하지 않았다.')

title('5. 동일 조건 1,000-update 예비학습: 3 seed × 3 arm')
para('각 seed의 기존 선택 checkpoint에서 시작했다. 새로운 AdamW 상태, U LR=1e−5, A LR=3e−7, constant LR, betas=(.9,.999), eps=1e−8, weight decay=.01, batch8, FP32, 1,000 update/8,000 patch view다. 원본 50K의 exact resume 또는 같은 학습량의 fresh run이 아니다. 사용자 승인 범위에서 짧은 비교로 제한했고, 결과를 보고 조기 종료하거나 arm별 update 수를 바꾸지 않았다.')
para('모든 arm에서 같은 sample index와 rot 순서를 사용했다. 원 feeder와 같이 hflip/vflip을 고정 적용하고 rot0–3을 적용했으며 q cache도 같은 (index,rot)로 조회했다. Teacher T0와 q 자산의 identity를 검증했다. U는 H+K+.002·wE·E, A는 wA·H만으로 autograd.grad를 별도로 계산했다. A 동결 arm은 원 checkpoint의 A를 그대로 유지했다.')
metric_table([(s,n)for s in SEEDS for n in ('FT_bicubic_joint','FT_lms_Aref_joint','FT_bicubic_Afreeze')])
table(['seed','LMS−기존 참조 ΔERGAS','LMS−기존 참조 ΔHQNR','동결−joint ΔERGAS','동결−joint ΔHQNR'],[
 [s,f"{e(s,'FT_lms_Aref_joint')-e(s,'FT_bicubic_joint'):+.6f}",f"{h(s,'FT_lms_Aref_joint')-h(s,'FT_bicubic_joint'):+.7f}",f"{e(s,'FT_bicubic_Afreeze')-e(s,'FT_bicubic_joint'):+.6f}",f"{h(s,'FT_bicubic_Afreeze')-h(s,'FT_bicubic_joint'):+.7f}"]for s in SEEDS])
para('LMS arm의 개선 여부는 학습 전 직접 교체와의 차이뿐 아니라 동일 update의 기존 참조 대조군과 비교해야 한다. A 동결과 joint의 차이가 작다면 이 짧은·낮은 LR 구간에서 동결의 추가 효과가 작다는 결론까지만 가능하다. 더 긴 학습이나 처음부터 새 참조를 사용한 모델에 대한 결론은 아니다.')

title('검증·비용·한계')
table(['검사','결과'],[
 ['source/checkpoint','고정 소스 181개 SHA 및 G23 원 snapshot SHA 일치'],['기준 RR 재현','기존 6 checkpoint A_ON ERGAS와 최대 절댓값 차이 '+f"{max(abs(x['delta_ergas'])for x in baseline):.9g}"],
 ['실험 결과','57개 complete, 2,280 scene 결과; 평가 전후 model state hash 불변'],
 ['예비학습','9개 모두 같은 seed 내 동일 초기 hash, 1,000 update, Teacher 불변; A 동결 hash 불변'],
 ['메모리','프로세스별 CUDA allocator 상한20%; 예비학습 peak allocated 최대 '+f"{max(x['peak_allocated_mib']for x in ft):.1f} MiB"],
 ['학습 소요','9개 예비학습 optimizer loop wall time 합계 '+f"{sum(x['seconds']for x in ft):.1f}초 (RR/FR/validation·입출력 제외)"],
 ['Sheet','socket 접속 차단 및 Sheet import 없음 검사; 업로드 없음'],
])
para('RR test20과 FR20은 이전 연구에도 사용된 개발 benchmark다. 이 표에서 후보를 선택하면 독립 test 검증이 되지 않는다. 특히 HQNR .9585에 매우 가까운 값은 다른 실행 환경·센서·seed의 통과를 보장하지 않는다. 이번 결과는 s1의 3 seed에 한정되며 s2–s5를 측정한 결과로 확장하지 않는다.')
para('현 단계에서 보존할 후보는 기존 A를 유지한 U 보간이고, 즉시 채택할 근거는 부족하다. A 입력 크기·단순 공통 편향보다 U의 학습 목적 및 RR/HQNR의 절충을 다음 연구의 중심으로 둘 근거가 강화됐다. 이번 작업에서 임의의 추가 loss·정규화 grid나 원본 모델 자동 교체는 실행하지 않았다.')

title('원자료와 재현')
for label,name in [('57개 모델/정책 전체 지표','all_variants.csv'),('2,280개 장면별 결과','all_scenes.csv'),('A/U 교차 효과','factorial_effects.csv'),
 ('사전 지정 protocol·source hash','manifest.json'),('완료 검증','checks.json'),('참조 점검','reference_check.json'),
 ('추론 평가','inference.py'),('예비학습','finetune.py'),('공통 평가 규약','common.py'),('보고서 생성','report.py')]:link(label,name)
para('results/<seed>/<variant>/result.json에 validation 결과·model hash·RR/FR per-scene·data/source identity가 있다. validation/<seed>.json은 test 계산 전 선택 기록이다. finetune 아래에는 새 모델, routing 검사, progress, complete metadata가 있다. 원본 모델은 이전 crop 보고서의 frozen assets를 참조하며 수정하지 않았다.')
para('사전 지정 t=.25 보간 모델은 세 seed 모두 candidates/S<seed>_Ab_U0.25/model.safetensors에 별도 저장했고 평가 당시 model hash와 대조했다. source_config.yaml은 구조 참조용으로, 원래 work_dir/phase를 가진 config를 그대로 실행하라는 뜻이 아니다. 기존 모델로 배포하거나 학습 큐에 등록하지 않았다.')
link('seed1234 t=.25 후보 manifest','candidates/S1234_Ab_U0.25/manifest.json')
path=P.parent/'PAN_QRC24_Improvement_Trials_Results_2026-09-18.md';path.write_text('\n'.join(md))
css='body{font:16px/1.65 system-ui,sans-serif;color:#20303a;max-width:1200px;margin:32px auto;padding:0 24px}h2{margin-top:2em;color:#115d70}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ccd7dd;padding:8px;text-align:left}th{background:#edf5f7}.table{overflow:auto}img{max-width:100%}figcaption{font-size:13px;color:#526771}a{color:#126e82}@media print{body{font-size:11px}table{font-size:8px}figure{break-inside:avoid}}'
(P/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QRC24 improvement trials</title><style>'+css+'</style><body>'+''.join(web)+'</body></html>')
overview=dict(n_variants=57,n_scene_outputs=len(scenes),n_finetune=9,joint_pass_count=joint,
 mean_A_swap_delta_E=mean('AlUb')-mean('AbUb'),mean_U_swap_delta_E=mean('AbUl')-mean('AbUb'),
 candidate_seed1234_t025=dict(ergas=e(1234,'Ab_U0.25'),hqnr=h(1234,'Ab_U0.25'),delta_ergas=e(1234,'Ab_U0.25')-e(1234,'AbUb')),
 mean_LMS_FT_minus_control_E=mean('FT_lms_Aref_joint')-mean('FT_bicubic_joint'),
 mean_Afreeze_FT_minus_control_E=mean('FT_bicubic_Afreeze')-mean('FT_bicubic_joint'))
write(P/'overview.json',overview);print(json.dumps(overview,indent=2));print(path)
