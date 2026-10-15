"""Aggregate the frozen report-only RR cohort. Writes Markdown, standalone HTML, CSV, figures."""
from pathlib import Path
import base64
import csv
import datetime
import hashlib
import html
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MD = OUT.parent/'PAN_QRC24_RR_Full_vs_Crop64_Results_2026-09-18.md'
METRICS = ('ergas', 'sam', 'psnr', 'ssim', 'scc', 'q8', 'rmse', 'cc')
cohort = json.loads((OUT/'cohort.json').read_text())
results = []
for a in cohort['assets']:
    p = OUT/'results'/a['run']/a['checkpoint']/'result.json'
    r = json.loads(p.read_text())
    assert r['complete'] and r['asset'] == a and r['state_unchanged']
    assert r['modes']['A_ON']['calls'] == {'aligner_calls':20,'warp_calls':20}
    assert r['modes']['A_CROP64_MED']['calls'] == {'aligner_calls':320,'warp_calls':20}
    for rel, digest in r['identity']['sources'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    assert len(r['scene_comparisons']) == 20
    results.append(r)
assert len(results) == cohort['n_assets']
primary = [r for r in results if r['asset']['checkpoint']=='best_hqnr']
g23 = [r for r in primary if r['asset']['profile']=='G23']
last = [r for r in results if r['asset']['checkpoint']=='last']
assert len(primary)==15 and len(g23)==3 and len(last)==3

def val(r, mode, k='ergas'):
    return r['modes'][mode]['summary'][k]['mean']

def avg(rs, mode, k='ergas'):
    return float(np.mean([val(r,mode,k) for r in rs]))

def diff(rs, k='ergas'):
    return float(np.mean([r['delta_crop_minus_on'][k] for r in rs]))

def csvout(name, rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

flat=[];scenes=[]
for r in results:
    a=r['asset']; row={k:a[k] for k in ('run','server','profile','seed','checkpoint','step','panel','checkpoint_sha256')}
    for k in METRICS:
        for m in ('A_ON','A_CROP64_MED'):row[f'{m}_{k}']=val(r,m,k)
        row[f'delta_{k}']=r['delta_crop_minus_on'][k]
    row.update(mean_shift_difference_px=r['mean_shift_difference'],max_shift_difference_px=r['max_shift_difference'],
               improved_scenes=r['ergas_improved_scenes'],worsened_scenes=r['ergas_worsened_scenes'],
               scene_ci95_low=r['ergas_scene_bootstrap_ci95'][0],scene_ci95_high=r['ergas_scene_bootstrap_ci95'][1])
    flat.append(row)
    for s in r['scene_comparisons']:
        scenes.append(dict(run=a['run'],profile=a['profile'],seed=a['seed'],checkpoint=a['checkpoint'],**s))
csvout('all_checkpoint_metrics.csv',flat);csvout('all_scene_pairs.csv',scenes)

ordered=sorted(primary,key=lambda r:(r['asset']['profile']!='G23',r['asset']['profile'],r['asset']['seed']))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10,7))
for i,r in enumerate(ordered):
    d=r['delta_crop_minus_on']['ergas'];lo,hi=r['ergas_scene_bootstrap_ci95']
    color='#126e82' if r['asset']['profile']=='G23' else '#717e8a'
    ax.plot([lo,hi],[i,i],c=color,alpha=.65,lw=2)
    ax.scatter([d],[i],c=color,s=36,zorder=3)
ax.axvline(0,c='#b34132',lw=1)
ax.set_yticks(range(len(ordered)),[f"{r['asset']['profile']} / seed {r['asset']['seed']}" for r in ordered])
ax.invert_yaxis();ax.set_xlabel('ERGAS difference: crop64 median - full256 (negative = improvement)')
ax.set_title('Existing selected checkpoints: paired RR20 evaluation')
ax.grid(axis='x',alpha=.18)
fig.text(.02,.012,'Bars: descriptive 95% scene-bootstrap interval; same benchmark images, not independent seed repetitions.',fontsize=8)
fig.tight_layout(rect=(0,.025,1,1))
for ext in ('png','pdf'):fig.savefig(OUT/f'ergas_effects.{ext}',dpi=180)
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.4))
for r in g23:
    s=r['scene_comparisons']; label=f"seed {r['asset']['seed']}"
    axs[0].plot(range(1,21),[x['delta_ergas'] for x in s],marker='.',label=label,alpha=.85)
    axs[1].scatter([x['delta_shift_l2'] for x in s],[x['delta_ergas'] for x in s],label=label,s=24,alpha=.8)
axs[0].set_xlabel('RR scene (1-based)');axs[0].set_ylabel('ERGAS: crop - full');axs[0].set_xticks([1,5,10,15,20])
axs[1].set_xlabel('Difference between predicted shifts (L2, RR PAN pixels)')
axs[0].set_title('G23: paired differences by scene');axs[1].set_title('G23: shift change and reconstruction effect')
for ax in axs:ax.axhline(0,c='#b34132',lw=1);ax.grid(alpha=.18)
axs[0].legend(fontsize=8)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(OUT/f'g23_scene_effects.{ext}',dpi=180)
plt.close(fig)

md=[]; web=[]
def para(text):
    md.append(text+'\n');web.append('<p>'+html.escape(text)+'</p>')
def title(text):
    md.append('**'+text+'**\n');web.append('<h2>'+html.escape(text)+'</h2>')
def table(headers, rows):
    md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+
              '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')
    web.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+
               '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>')
def figure(name, caption):
    md.append(f'![{caption}]({OUT.name}/{name})\n')
    enc=base64.b64encode((OUT/name).read_bytes()).decode()
    web.append(f'<figure><img src="data:image/png;base64,{enc}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
def link(label,path):
    rel=str(path.relative_to(OUT.parent));md.append(f'[{label}]({rel})\n')
    web.append('<p><a href="'+html.escape(path.name if path.parent==OUT else '../'+rel)+'">'+html.escape(label)+'</a></p>')

title('QRC24 fixed method — RR full256 정합 vs crop64 중앙값 정합 실측 보고서')
para(f"작성: {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} · cohort 고정: {cohort['captured_at']} · source HEAD: {cohort['head']}")
title('판정과 적용 범위')
n_good=sum(r['delta_crop_minus_on']['ergas']<0 for r in primary)
para(f"로컬 s1의 완료 QRC24 15개 실행을 모두 측정했다. 기존 best_hqnr 15개와 G23의 last@50K 3개, 총 18 checkpoint에 같은 RR 20장으로 두 추론 방식을 적용했다(18×20×2=720개 주 평가 출력; 반복 검사는 별도). Sheet에는 기록하지 않았다.")
para(f"주력 G23의 기존 선택 checkpoint 3 seed 평균 ERGAS는 {avg(g23,'A_ON'):.6f} → {avg(g23,'A_CROP64_MED'):.6f}, 차이는 {diff(g23):+.6f} ({100*diff(g23)/avg(g23,'A_ON'):+.4f}%)다. 전체 15개 중 {n_good}개에서 평균 ERGAS가 개선됐다. 이 실험은 실제 복원 점수에서 입력 영역 크기의 영향을 분리해 확인한다는 의미가 있다. 효과 크기와 장면별 불확실성을 함께 판단해야 한다.")
para('판정: 작은 평균 개선은 관찰됐고 진단 가치는 있다. 그러나 성능을 크게 끌어올리는 해결책이라는 근거는 약하다. G23의 선택 checkpoint 3개 모두 장면 bootstrap 95% 구간이 0을 포함하며, 일부 장면에서는 악화된다. 따라서 강한 통계적 유의성이나 모든 영상에서의 개선을 주장하지 않는다.')
para('“전부”의 실측 범위는 이 서버에서 접근 가능한 완료 QRC24 실행이다. s2–s5 checkpoint와 접속 정보가 없어 전 서버 전체를 측정한 결과는 아니다. G23 seed41006은 cohort 고정 당시 학습 미완료여서 제외했다. 과거 N2/T0/PAKD50 전체 캠페인까지 최종 fixed method 반복으로 합치지 않았다.')
para('현재 추론 정책과 학습 코드는 변경하지 않았다. 보고서의 best_hqnr는 기존 결과가 사용한 저장 checkpoint 이름이며, 새 HQNR9585_ERGAS2040_v2 target selector를 전 실행에 적용하거나 이 비교를 보고 checkpoint를 다시 고른 것이 아니다. exact50K 보조 결과는 선택 checkpoint 결과와 분리한다.')

title('비교 조건과 해석할 수 있는 범위')
table(['항목','고정 조건'],[
 ['데이터','WV3 RR20: PAN/GT 256×256, LRMS 64×64, 8 bands'],
 ['A_ON','전체 PAN256와 전체 bicubic MS256로 aligner 예측'],
 ['A_CROP64_MED','전체 PAN/MS-up에서 64×64 비중첩 16개 crop → 각 (dy,dx)의 성분별 중앙값'],
 ['중앙값','짝수 16개: 가운데 두 값 평균; GT나 지표를 이용한 crop 선택 없음'],
 ['aligner margin','양쪽 모두 기존 margin4를 한 번 적용: 실제 A view는 248×248 또는 56×56'],
 ['복원/warp','원 PAN256을 대표 이동량으로 한 번 warp; U-Net은 양쪽 모두 전체256 처리'],
 ['단위','crop과 full은 동일 RR PAN pixel grid; resize나 이동량 ×4 없음'],
 ['지표','양쪽 동일 20:-21 경계 crop(215×215), peak=2047, ERGAS ratio4, Q8 block32'],
 ['출력','FP32 추론, clip[-1,1] 후 FP32 DN 변환, 장면별 지표를 20장 산술평균'],
 ['불확실성','같은 20장 내 대응 차이를 10,000회 재표집한 95% 구간; 독립 센서/seed 반복의 검정은 아님'],
])
para('이 비교는 입력 영역을 제한한 추론 정책의 효과를 측정한다. crop별 정규화·GroupNorm·경계·pooling 통계와 중앙값 집계가 함께 달라지므로, 입력 크기 하나의 순수 인과 효과라고 단정하지 않는다. q 또는 synthetic-shift response를 새로 측정한 실험도 아니며, absolute 정합 정확도의 ground truth를 얻은 것은 아니다. 복원 점수 개선을 native shift의 물리적 정답으로 해석하지 않는다.')

def checkpoint_table(rs):
    rows=[]
    for r in rs:
        a=r['asset'];lo,hi=r['ergas_scene_bootstrap_ci95']
        rows.append([a['profile'],a['seed'],a['step'],f"{val(r,'A_ON'):.6f}",f"{val(r,'A_CROP64_MED'):.6f}",
                     f"{r['delta_crop_minus_on']['ergas']:+.6f}",f"[{lo:+.6f}, {hi:+.6f}]",
                     f"{r['ergas_improved_scenes']}/{r['ergas_worsened_scenes']}",f"{r['mean_shift_difference']:.5f}"])
    table(['설정','seed','step','Full ERGAS↓','Crop ERGAS↓','Δ Crop−Full','장면 bootstrap 95%','개선/악화 장면','평균 이동량 차이 px'],rows)

title('주력 G23: 기존 선택 checkpoint의 3 seed 결과')
para('G23: λE=.002, rA=.03, α=1, β=.1, U LR=1e−4, A LR=3e−6. 저장 설정은 qref/(qref+q_T), uniform 기준 .5인 factor1 정의이며, 완료 실행 전부 definition audit를 통과했다.')
checkpoint_table(g23)
table(['지표','Full: 3 seed 평균','Crop: 3 seed 평균','Δ Crop−Full'],[
    [k+('↓' if k in ('ergas','sam','rmse') else '↑'),f"{avg(g23,'A_ON',k):.8f}",f"{avg(g23,'A_CROP64_MED',k):.8f}",f"{diff(g23,k):+.8f}"] for k in METRICS])
base_sd=float(np.std([val(r,'A_ON') for r in g23],ddof=1))
para(f"3 seed Full ERGAS의 sample SD는 {base_sd:.6f}이며, 대응 추론 변경의 평균 효과는 {diff(g23):+.6f}다. seed마다 다른 학습 결과의 절대 점수 차이와 동일 checkpoint 안의 추론 효과를 구분해야 한다. 같은 20개 scene을 3 seed에서 반복했으므로 60개의 독립 scene 표본으로 세지 않는다.")
gap=avg(g23,'A_ON')-2.04
para(f"평균 Full ERGAS와 2.04 사이의 차이는 {gap:.6f}다. crop 변경이 줄인 차이는 {-diff(g23):.6f}, 이 간격의 {-100*diff(g23)/gap:.2f}%다. RR 결과만으로 HQNR 목표 달성이나 FR 품질 유지를 주장하지 않는다.")
figure('g23_scene_effects.png','G23: 장면별 효과 및 이동량 차이와 복원 효과의 관계')

title('G23: exact50K 보조 비교')
checkpoint_table(last)
para(f"last@50000의 3 seed 평균 ERGAS는 {avg(last,'A_ON'):.6f} → {avg(last,'A_CROP64_MED'):.6f}, Δ={diff(last):+.6f}다. 동일 seed의 best와 last는 한 학습 trajectory에서 나온 checkpoint이므로 독립 6-seed 반복이 아니다.")

title('완료된 QRC24 기존 선택 결과 전체')
checkpoint_table(ordered)
figure('ergas_effects.png','완료된 15개 실행의 ERGAS 대응 차이와 장면 bootstrap 구간')
para('대조군은 입력 영역 효과가 주력 설정에만 나타나는지 확인하기 위한 보조 자료다. 여러 profile의 15개 값을 하나의 독립 반복 집단으로 합쳐 통계적 유의성을 주장하지 않는다. 각 실행의 8개 지표 전체와 20장 원수치는 아래 CSV/JSON에 보존했다.')

title('이동량과 결과 신뢰성 확인')
shifts=[x['delta_shift_l2'] for r in g23 for x in r['scene_comparisons']]
full_norm=[np.hypot(x['full_dy'],x['full_dx']) for r in g23 for x in r['scene_comparisons']]
crop_norm=[np.hypot(x['crop_dy'],x['crop_dx']) for r in g23 for x in r['scene_comparisons']]
legacy=[x for r in primary for x in r['legacy_on_comparison']]
repeats=[x for r in results for x in r['repeated_scene0'].values()]
max_mean_legacy=max(abs(np.mean([x['ergas_delta'] for x in r['legacy_on_comparison']])) for r in primary)
peak=max(r['gpu_peak_allocated_mib'] for r in results)
para(f"G23 선택 결과의 평균 shift 크기는 Full {np.mean(full_norm):.6f}px, Crop {np.mean(crop_norm):.6f}px다. 두 방식의 벡터 차이 L2는 평균 {np.mean(shifts):.6f}px, 최대 {max(shifts):.6f}px다. 여기서 px는 원 RR PAN256 grid의 픽셀이며, 작은 crop의 픽셀도 같은 단위다. 이 값들로 4배 수동 보정이 필요하다는 근거는 얻을 수 없다.")
table(['검사','결과'],[
 ['호출 수 / checkpoint / 20장','Full A 20회·warp 20회, Crop A 320회·warp 20회; 18개 모두 통과'],
 ['모델 변경','strict checkpoint load; 평가 전후 parameter/buffer hash 동일, gradient 없음, sampler 복원'],
 ['반복 검사','각 checkpoint·방식의 scene0 반복: 최대 DN 차이 '+f"{max(x['max_abs_dn'] for x in repeats):.8g}, 최대 ERGAS 차이 {max(x['abs_ergas_delta'] for x in repeats):.8g}"],
 ['기존 저장 Full 결과 재현',f"15개 중 평균 ERGAS 차이 최대 절댓값 {max_mean_legacy:.9g}; 전체 pixel 최대 DN 차이 {max(x['max_abs_dn'] for x in legacy):.6g}"],
 ['GPU','RTX 4090, batch1, 4 CPU threads, 메모리 상한20%; PyTorch peak allocated 최대 '+f'{peak:.1f} MiB'],
 ['출력 경로','research_log/QRC24_RR_Crop64_Report_2026-09-18 및 연결 보고서만 생성'],
 ['Sheet','업로더 import 없음 검사 통과; socket connect 차단; 전송 없음'],
])
para('과거 MAT와 이번 Full 출력은 bitwise 동일하지 않다. 위 수치는 그 차이를 숨기지 않고 기록한 재현 검사다. 핵심 효과는 과거 MAT와 새 Crop의 차이가 아니라 같은 이번 실행 조건으로 다시 계산한 Full 대 Crop의 대응 차이다. scene0 반복 검사가 모든 장면·모든 하드웨어의 결정론을 보장하지는 않는다.')
para('SSIM은 Gaussian 11×11, σ1.5; SCC는 zero-padding Sobel cosine; PSNR은 전 밴드 통합 MSE를 사용한다. 이는 현재 저장소 보고 규약과 일치한다. PSNR/SSIM을 DLPan 공식 표준 자체라고 부르지 않는다. GPU를 학습과 공유한 실행이며 forward hook 및 반복·계측 오버헤드가 있으므로 저장된 시간은 정밀 성능 벤치마크가 아니다.')

title('권장 처리')
para('이 결과는 RR에서 full 입력 때문에 이동량을 ×4 해야 한다는 처방을 뒷받침하지 않는다. 학습 patch 크기의 정합을 적용했을 때의 실제 복원 효과를 확인하는 진단으로 활용한다. 현재 정책의 자동 전환 대신 효과 크기·seed 방향·장면별 분산을 함께 판단한다. test20을 보고 정책을 선택한다면 이후 별도 validation/holdout으로 재확인해야 한다.')
para('현재 실측 범위에서는 train64→RR256의 영역 크기 차이가 RR 오차의 주된 원인이라는 증거가 약하다. crop 중앙값만으로 ERGAS 2.04에 도달한 G23 checkpoint는 없다. 작은 개선 후보로 보존하되, 정합의 학습 목표·A와 U의 공동 적응·checkpoint 선택 효과를 별도로 살펴보는 것이 다음 우선순위다. best보다 last의 RR이 낮다는 사실만으로 HQNR 조건까지 확인하지 않고 last를 최종 모델로 교체해서는 안 된다.')
para('전 서버 완료 결과에 동일 비교를 적용하는 것은 기술적으로 가능하다. 필요한 것은 각 서버의 완료 checkpoint/config 및 같은 RR 데이터 접근이다. 현재 접근하지 못한 s2–s5는 미측정으로 유지한다. 원격 checkpoint 없이 Sheet에 적힌 점수만으로 crop 추론 결과를 복원할 수는 없다.')

title('원자료와 재현')
for label,name in [('18 checkpoint의 전체 지표 CSV','all_checkpoint_metrics.csv'),('360 checkpoint-scene 쌍의 원수치 CSV','all_scene_pairs.csv'),
                   ('고정 cohort / 선택 / checksum','cohort.json'),('실행 환경','environment.json'),
                   ('보고서 전용 평가 스크립트','evaluate.py'),('보고서 생성 스크립트','report.py')]:link(label,OUT/name)
para('각 results/<run>/<checkpoint>/result.json에 16개 crop의 예측, 장면별 8개 지표, 반복/호출/재현 검사를 저장했다. G23의 6 checkpoint는 두 방식의 SR 배열도 sr_pair.npz로 보존했다. assets에는 비교 당시 checkpoint/config/선택 이력, source에는 주요 구현 snapshot이 있다. 초기 GPU 장치 문자열 지정 오류는 추론 전에 발생했고 cuda:0으로 재실행했다. errors의 해당 기록은 초기 실패 이력이며 모든 18개 complete 결과의 성공 여부는 본 집계에서 별도로 검증했다.')
para('재현 명령: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/knuvi/miniconda3/envs/pancrafter/bin/python -B research_log/QRC24_RR_Crop64_Report_2026-09-18/evaluate.py --device cuda:0. 기존 cohort와 동일 source/checkpoint/data identity일 때만 완료 캐시를 재사용한다. 집계는 같은 Python으로 report.py를 실행한다.')

MD.write_text('\n'.join(md))
style='body{font:16px/1.65 system-ui,sans-serif;color:#20303a;max-width:1200px;margin:36px auto;padding:0 24px}h2{margin-top:2em;color:#115d70}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ccd7dd;padding:8px;text-align:left}th{background:#edf5f7}.table{overflow:auto}img{max-width:100%}figcaption{font-size:13px;color:#526771}a{color:#126e82}@media print{body{font-size:11px;max-width:none}table{font-size:8px}figure{break-inside:avoid}}'
(OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QRC24 RR full vs crop64 report</title><style>'+style+'</style><body>'+''.join(web)+'</body></html>')
overview=dict(n_runs=len(primary),n_checkpoints=len(results),n_main_outputs=len(results)*40,
              g23_selected_ergas_full=avg(g23,'A_ON'),g23_selected_ergas_crop=avg(g23,'A_CROP64_MED'),
              g23_selected_delta_ergas=diff(g23),g23_last_delta_ergas=diff(last),
              primary_improved=n_good,missing_servers=cohort['missing_source_servers'],
              legacy_max_run_mean_ergas_difference=max_mean_legacy,all_checks_passed=True)
(OUT/'overview.json').write_text(json.dumps(overview,indent=2,ensure_ascii=False))
print(json.dumps(overview,indent=2,ensure_ascii=False))
print(MD)
