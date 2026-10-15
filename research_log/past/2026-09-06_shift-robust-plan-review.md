# Shift-robust conditioning + M-frame PAN guidance 30h 계획 — 검토와 구현 보고 (2026-09-06)

대상: [s1_w168_d123_shift_robust_alignment_30h_plan.md](s1_w168_d123_shift_robust_alignment_30h_plan.md)
구현: `sr/{jitter,pan_align,forward}.py` · `train_sr.py` · `tools/{calibrate_blur,test_shift_robust,sr_infer_diag,local_align_diag}.py`
· `config/SR_*.yaml`·`AF_G1_*.yaml` (11벌: 본 5 + seed 1234 반복 4 + radius refinement 2) · `config/queues/s1_shift_robust.txt`
· `tools/campaign_gate.py::gate_sr` · `model/pancrafter_paper.py forward(f_in=)`
판정: **best checkpoint HQNR(12-19, 장면별 곱 평균, running-max tie 1e-4) → fSCC → 나중 iteration**. 이미 main.py 에 있다.

---

## 0. 요약

- **원 파이프라인으로 완전히 복귀**했다: 원 `PanFeeder`(LR 증강), 원 bicubic(`F.interpolate`), M-frame 잔차 base·GT·출력,
  PAN mode 는 inverse/warp/mask 없음. 지난 캠페인의 interp23tap·cache·inverse 는 전부 뺐다. 바뀌는 것은 **네트워크가 보는
  MS 조건 채널**(J 계열) 또는 **first conv 의 PAN 기여**(G1) 뿐이다.
- 전제 3건을 데이터로 확인했다 (§1): anchor 실측치, 원 bicubic 이 LR flip/rot 과 가환(1e-13 — 지난번 버그 재발 없음),
  TV-L1 부재(→ DIS).
- **계획의 한 전제가 데이터와 어긋난다** (§2): sub-pixel bicubic warp 는 gradient energy 를 **줄이지 않는다**(r_jit = 1.0056).
  계획 §8.1 의 grad-energy 매칭은 σ* = 0.10 ≈ 항등을 골라 J3 가 anchor 재실행이 된다. J3 는 **perturbation 크기(MSE) 매칭
  σ* = 1.225 HR px** 로 돌리고, 두 값을 모두 보정 파일에 남겼다(`blur.match` 로 전환 가능).
- 검증: 단위 T01–T24 **24/24**, smoke 5/5(peak J1/J2/J3 8.6 GB · J4 12.8 GB · G1 9.8 GB), 등가 실행·통합 210 iter (§5).
- 기동은 진행 중인 대조 run(`GA_CTRL_C2_BICUBIC15_A100`, ≈03:15 종료) 뒤에 한다.

---

## 1. 전제 확인

| 전제 | 실측 |
|---|---|
| anchor `S1_T05_W168_D123_DUAL` | HQNR 0.9571(장면별 평균 0.9571) · fSCC 0.8785 · D_λ 0.0231 · D_s 0.0202 · 학습 3.46h (계획 3.5h 추정 정확) |
| 원 bicubic + 원 feeder 의 LR 증강 | `F.interpolate(bicubic)` 은 phase 1.5(중심 대칭)라 `up(aug_LR) == aug_HR(up)` 이 **1e-13** — phase-2 격자에서 났던 1px 오정렬이 원리적으로 없다. §4.1 "jitter 는 증강·bicubic 뒤 ms_base 복사본에만" 도 그대로 성립 |
| local field 추정기 | opencv-python 본체에 TV-L1(`cv2.optflow`) 없음 → **DIS optical flow** 로 대체(§12 는 추론 진단만이라 영향 한정) |
| FR audit Δ | `outputs/global_shift_cache/wv3_fr.csv` 12-19 ≈ (−0.16, +0.18) LR px. G1 진단의 **비교 target 으로만** 쓴다(§11.6) — 학습 label 아님 |

---

## 2. 계획과 다르게 한 것 (이유 포함)

| 항목 | 계획 | 구현 | 이유 |
|---|---|---|---|
| **J3 blur σ 매칭 통계** | gradient-energy 비 (§8.1) | **MSE(perturbation 크기) 매칭, σ* = 1.225 HR px**. grad-energy 값(σ* 0.10)도 기록 | 500 표본 실측 r_jit = **1.0056 > 1** — Keys bicubic(a=−0.75) warp 는 Scharr 에너지를 오히려 0.6% 키운다. 원안대로면 σ→0.10 으로 사실상 항등(r_blur 1.0000)이라 H3 를 못 답한다. 이 사실 자체가 "jitter 의 smoothing 성분은 없다" 는 H3 의 절반 답이다. MSE 매칭은 "같은 입력 perturbation 에너지, 변위 없음" 대조가 된다 — 다만 σ 1.2 px 는 시각적으로 강한 blur 라 J3 를 "강한 low-pass 조건" 대조로 읽어야 한다 |
| local flow | global-corrected TV-L1 | DIS(MEDIUM preset), forward-backward ≤0.5px·edge 상위 30%·|flow|≤2px 게이트 | TV-L1 부재. 진단 전용 |
| 실행 산출물 이름 | `train_log.jsonl`, `checkpoint_metrics.csv`, `best_hqnr.pt`, `last.pt` | 저장소 규약: `train_log.txt`, `metrics.csv`, `best_hqnr/`(accelerate), `lastest/`, `best_hqnr_meta.json` | 기존 도구(시트 업로드·plateau_report·sweep)가 그 이름을 읽는다 |
| G1 edge weight `normalize(|PAN-HF|)` | 정의 미기재 | 표본별 99 분위수로 나눠 [0,1] clip, w = 0.25 + 0.75·그 값 | 최대값 정규화는 outlier 에 흔들린다 |
| G1 `SmoothL1(Δ̂, −ε_g)` | β 미기재 | torch 기본 β=1.0 (|오차|<1 이면 0.5·L2) | 목표 범위 ±1 px 안에서 사실상 L2 |
| G1 soft-argmax | — | 그대로 | 단, 매끈한 특징에서는 이웃 후보 cosine ≈ 0.99 라 τ=0.07 softmax 가 퍼져 **Δ̂ 가 0 쪽으로 수축**한다(합성 검사에서 ε=1.0 → Δ̂ 0.61). 학습 전 descriptor 의 성질이고 L_shift 가 이를 날카롭게 만드는 것이 학습 목표. 진단에서 |Δ̂| 와 audit 대비 medErr 로 확인 |
| G1 진단 "beta sweep" | 정의 미기재 | `g1_scale` β ∈ {0, 0.5, 1.0} — 적용 보정량 스케일 | 가장 자연스러운 해석 |
| J4 batch | "MS branch 하나 추가, 1.5배" | 한 forward 에 [PAN clean \| MS clean \| MS jitter] 3×48 = 144 | 가중치 공유·한 번의 backward. peak 12.8 GB (24 GB GPU) 라 gradient accumulation 불필요 |
| 마지막 슬롯(seed repeat / refinement) | 사람이 고른다 | `gate_sr` 가 자동: winner(HQNR→fSCC 1e-4→plateau→final) 의 seed 1234 config 를 열고, 전 case 가 anchor−0.002 이하·fSCC 열위면 J1 radius refinement(R075/R025) | 30h 안에 사람 개입 없이 끝나게. 4+2 벌 config 를 미리 만들어 뒀다 |
| 평가 격자 | "촘촘한 주기 + 50K" | eval_epoch 5 (49점) + 종료 시 50K 추가 평가 | 이미 구현돼 있음 |
| s2 병렬(§24) | 선택 | **하지 않음** | s1 30h 예산 밖 |

---

## 3. 구현 요지

### 3.1 forward 한 곳 (`sr/forward.py::sr_forward`) — Accelerator 없이 검사 가능

```
ms_base, lpan_u, pan_hf = bicubic(ms), bicubic(lpan), pan − bicubic(lpan)      # 원 코드와 동일
j1 : cond = T_ε(ms_base)  →  x_pan = x_ms = [pan, lpan_u, pan_hf, cond]
j2 : x_pan = [.., ms_base],  x_ms = [.., T_ε(ms_base)]
j3 : cond = G_σ*(ms_base) 두 mode
j4 : [x_pan(clean) | x_ms(clean) | x_ms(jitter)] 3B,  L_MS = ½L1(Ŷ0)+½L1(Ŷε),  L_cons = |Rε − sg(R0)|₁,  λ(t) 0→0.1 (5K)
g1 : F_P, F_M = split(first conv, x_clean);  MS: F_P^syn = W(F_P, ε_g);  Δ̂,g = correlator(F_P^syn, F_M, w_edge)
     F̃_P = F_P^syn + g[W(F_P^syn, Δ̂) − F_P^syn];  backbone(f_in = F_M + F̃_P);  PAN: backbone(f_in = conv(x_clean))
     L += 0.1·SmoothL1(Δ̂, −ε_g)
공통: y = ms_base + R_MS,  p̂ = lpan_u·rep + R_PAN,  L1.  추론: ε=0, ε_g=0, g1 은 실제 쌍에 correlator 적용
```

ε 는 `sample_jitter` U(−0.5,0.5)² HR px, 표본당 1개, 8밴드 동일(T02), `translate_hr` 는 `align.resample.warp_hr`
(부호 out[y,x]=src[y+dy,x+dx], T10) 재사용. ε=0 이면 warp 를 부르지 않고 **같은 객체**를 돌려준다(T01 비트 동일).

### 3.2 backbone 변경
`forward(..., x_in=None, f_in=None)`: `f_in` 이 오면 `self.input` 을 건너뛴다. 기본 경로는 그대로(이미 등가 확인된 x_in 과 같은 방식).

### 3.3 큐·게이트
`s1_shift_robust.txt`: J1 → J3 → J4 → J2 → G1 (계획 §13.1 순서). 본 큐 뒤 `gate_sr` 가 seed repeat 또는 refinement 1벌을 연다.
예상: 3.5 + 3.5 + 5.2 + 3.5 + 4.0 + 3.5 ≈ 23.2h + 평가 → 30h 안.

### 3.4 진단 도구
- `tools/sr_infer_diag.py`: J 계열 추론 jitter 0 / ±0.5 조합 / 무작위, G1 β 0/0.5/1 · wrong-sign · confidence P10/P90 · center/boundary 확률 · audit(−4Δ_LR) 대비 corr/medErr.
- `tools/local_align_diag.py`: §12 L0~L4 (DIS field, 게이트) + §12.6 gate 판정.
- 시트: SR 범주(⑬) 구분행 + case 질문·세팅·판정·anchor 서술 Notes.

---

## 4. 실행 전 알게 된 것

1. **jitter 는 smoothing 이 아니다.** r_jit(grad-energy) = 1.0056. H3 의 "sub-pixel 보간의 약한 smoothing" 가설은 데이터에서 성립하지 않는다.
   J3 는 그래서 "변위 없는 등에너지 perturbation" 대조로 재정의됐다(§2).
2. G1 의 soft-argmax 는 학습 전 0 쪽으로 수축한다(§2). G1 학습 로그의 `err`(|Δ̂+ε_g|)·`conf`·`pB` 로 수렴을 본다.
3. J4 는 J1 의 1.5배(3×48 forward), peak 12.8 GB.

---

## 5. 검증

- `tools/test_shift_robust.py` **T01–T24 24/24** — T01 ε=0 비트 동일(원 모델 forward 와 `torch.equal`), T05 출력 frame, T07/T08 mode 별 조건,
  T09 PAN 출력 무warp, T15 stop-gradient(autograd 로 clean 잔차에 grad None), T16 양쪽 GT grad, T17 PAN 무consistency,
  T18 split conv 1e-6, T19 bias 1회, T21 synthetic +ε → argmax 정확히 −ε·부호 일치, T22 PAN mode aligner 0회, T24 저신뢰 gate 항등.
- smoke 5/5 (실배치 forward+backward+AdamW, FR 512² 추론, jitter 통계).
- 등가 실행·통합 210 iter: §6 에 기록.

## 6. 등가·통합 실행 (아래 갱신)

- **등가 실행** (W168, 210 iter, num_worker=0): 원 Trainer 0.111351 / SR j1 r=0 0.111351 — iter 10 까지 완전 동일. 이후 차이는
  같은 설정을 두 번 돌린 원 Trainer 끼리의 차이(iter 210 2.0e-4, ep2 HQNR 0.8631 vs 0.8648)와 같은 크기(sr_r0 는 그 사이 0.8641).
  → SR 경로의 ε=0 은 원 Trainer 와 동작이 같다. 부수: 같은 설정 두 run 이 210 iter 뒤 HQNR 1.7e-3 갈린다(비결정성).
- **통합 210 iter** J1·J3·J4·G1 전부 완주(rc=0, mat 2종). 로그 항목 확인: J1 `eps y/x mean±std |max| geR`, J3 `sigma geR`,
  J4 `cons/lam/ratio`(+종료 시 ratio 판정 줄), G1 `shift/delta/err/conf/pB/pC` 와 FR `[G1]` 진단(audit 대비 corr·medErr).
  G1 은 210 iter 시점 conf 0.001(균등 posterior)·gate≈0 이라 사실상 항등에서 출발한다 — L_shift 로 날카로워지는지를 로그로 본다.
- 기동: 대조 run(`GA_CTRL_C2_BICUBIC15_A100`) DONE 직후 자동 (`--hours 30`).
