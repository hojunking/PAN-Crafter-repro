# PAN QRECON24 — 전 서버 20시간 재편 계획
## HQNR ≥ 0.9585를 먼저 만족시키고, 그 안에서 ERGAS 최소화

**작성일:** 2026-09-18  
**Campaign ID:** `QRC24_MIX20H_20260918_v1`  
**Queue revision:** `QRC24_MIX20H_Q1_20260918`  
**학습 method:** 기존 `qrecon_continuous_v1` 유지  
**주 평가 selector:** `HQNR9585_ERGAS2040_v2`  
**실행 방식:** 각 서버에서 명시된 `(profile, seed, version)`을 순차 실행. 서버 간 결과 대기·공통 recipe lock 없음.  
**문서 상태:** 실행 case·seed·수식·예산·전환·검사 명세를 확정한 계획. 이 파일과 동봉 case CSV를 만들었으며, 원격 YAML 생성/registry 등록/큐 적용/학습 기동까지 수행한 것은 아니다.

> **이번 계획의 핵심**
>
> 1. A_ON의 raw-original HQNR가 **0.9585 이상인 checkpoint만** 주 목표 후보로 인정한다.
> 2. 하한을 통과한 후보에서는 **HQNR을 더 높이는 것보다 ERGAS를 낮추는 것이 우선**이다. ERGAS 목표는 **< 2.040**이다.
> 3. G23(β=0.1)과 B20A03(β=0.2)만 남기고 **같은 서버·같은 seed의 두 설정을 짝지어** 실행한다. 모든 서버가 두 설정을 섞어 돌린다.
> 4. 현재 실행 중인 run은 이름·seed·계수·학습 경로를 바꾸지 않는다. 새 편성은 **520xx seed / v4 / 별도 campaign**으로 분리한다.
> 5. 모든 신규 run은 fresh U-Net 50K다. HQNR를 처음 넘었다고 학습을 중단하거나 aligner를 동결하지 않는다. 정상 A_ON을 유지한다.
> 6. **20시간은 다섯 서버가 병렬로 사용하는 하나의 운영 창**이다. 1 GPU/서버를 기준으로 최대 100 GPU-hours에 해당하지만, 실제 학습량은 잔여 작업·준비·평가 시간에 따라 줄어든다.

---

## 0. 무엇을 유지하고 무엇을 교체하는가

| 구분 | 이번 계획의 결정 |
|---|---|
| 학습 수식·Teacher·Student 골격 | 기존 QRECON24 그대로 |
| 주 추론 | `A_ON`: Student aligner + sampler + Student U-Net |
| 마지막 β 비교 후 공통 C*를 기다리던 절차 | **폐기**. β 두 값을 같은-server/seed 쌍으로 미리 배정 |
| `recipe_lock` 존재/다른 서버 결과/목표 통과 횟수에 따른 학습 허용 | **사용하지 않음** |
| 로컬 lock 파일에서 실행 profile을 자동 선택 | **금지**. 매 case의 명시 profile이 유일한 선택값 |
| 41xxx seed 고정 G23 큐의 미시작 부분 | 이번 520xx 혼합 큐로 대체. 이미 시작된 부분은 별도 carry-over |
| 원 ADJ-R1의 넓은 G/H/L grid 및 자동 extra | 미시작분 보류. 새 큐 끝에서 자동 복구하지 않음 |
| 원 캠페인의 서버당 최소24h / `no_hard_limit` | **이번 20h 운영 창에 적용하지 않음**. 과거 비용·이력은 보존 |
| 체크포인트 선택 | HQNR 하한 → ERGAS → SCC → PSNR → 나머지 tie-break |
| Aligner C00–C05 독립 진단 | 별도 작업공간·자원으로 계속. 본 학습 시작/완료 조건이 아님 |
| 평가 기준·정의 검사와 자원 검사 | 유지. 무효 입력·잘못된 loss·중복 실행까지 허용한다는 뜻이 아님 |

**공통 방법 유지와 공통 profile 선택은 다르다.** 두 profile은 같은 수식의 β만 다르다. 서로 다른 β를 사용한 모든 run을 나중에 ‘동일 설정의 seed 반복’으로 합산하지 않는다.

## 1. 계획의 근거와 확인 범위

### 1.1 기존 자료에서 확인한 사실

이 절은 기존 감사 결과 [A]의 요약이고, 이후 절의 seed 배정·예산은 이번에 제안하는 새 운영 설계다.

- 기존 감사 snapshot: QRC24 고유80run, NOA pair64run. 당시 등록 본 행에는 ERGAS <2.040이 없었다.
- s1 G23·1234: 같은-checkpoint pair **HQNR 0.959148 / ERGAS 2.071234 / step31310**.
- s3 G23·41003: 본 행 **HQNR 0.9596 / ERGAS 2.0824**. 최고 HQNR라고 해서 기존 s1 자산보다 우선인 것은 아니다.
- s4 H31·1234: pair **HQNR 0.958531 / ERGAS 2.139965**. H 하한은 통과하지만 E 비용이 크다.
- β=0.2의 이득은 현재 대응 결과에서 일관되지 않다. G23/B20A03 두 설정을 남기는 근거이지 β=0.2의 우위가 확정됐다는 뜻이 아니다.
- 64개 NOA pair는 모두 ERGAS 상승·HQNR 하락이었다. 이번 본 실험을 NOA로 바꾸지 않는다.
- 기존 자동 postrun은 `--selector` 미지정으로 SCC 우선 v1을 호출하는 문제가 있었다. v2 함수의 존재만으로 자동 적용이 끝난 것으로 판단하지 않는다.
- 기존 선택 행의 ERGAS는 선택 step과 강한 상관이 있었다. 계수 효과를 보기 위해 **target와 exact50K를 함께 보존**한다.

근거: [A] §2–6. 이 결과는 목표 달성 가능성을 보장하지 않는다. 현재 좋은 자산의 E=2.071234에서 2.040 미만까지는 약 0.031234 이상의 감소가 필요하다.

### 1.2 이번 작성에서 추가 확인한 범위

- 연결 저장소 `main` 조회 ref는 `a565cafbb1d1f124dbb34450207b0374ade5aa42`다. 이것은 **조회 ref**이지 향후 모든 서버에서 그대로 실행해야 하는 검증 완료 배포 ref는 아니다. selector/queue 연결을 보완한 실제 배포 ref를 따로 남긴다.
- Sheet 전체 결과를 다시 집계한 것이 아니라 각 서버의 최근 `Train(h)`와 추가 두 행을 직접 재확인했다.
- 추가 확인: **s2 G23·41002, H=0.9550 / E=2.0657 / Train=2.55h**, **s4 G23·41014, H=0.9570 / E=2.1407 / Train=1.53h**. 둘 모두 H 하한 미달이다. 이 두 행을 새 520xx cohort로 재분류하지 않는다.
- 최근 시간 범위: s1 약2.36–2.42h(동결 대조2.15h 별도), s2 약2.54–2.56h, s3 약1.45–1.47h, s4 약1.50–1.53h. s5 최근 기록1.64/2.69/3.40h로 변동이 크다.
- 서버의 현재 PID·step·실제 큐·로컬 미push 변경·유휴 GPU는 조회하지 않았다. **현재 무슨 run이 돌고 있는지와 적용 시 남은 시간을 서버에서 확인해야 한다.**

이번 Sheet 재조회 범위: s1 U73:V95, s2 U87:V112 및 B100:V100, s3 U119:V145, s4 U49:V73 및 B60:V60, s5 U41:V67. source tab/gid는 §10에 적는다. 원 XLSX export는 확보했지만 본 계획의 시간·증분 근거는 직접 읽은 위 cell range다.

---

## 2. 목표의 정확한 정의 — HQNR은 하한, ERGAS는 최적화 순위

H는 **FR paper mat20의 장면평균 raw-original HQNR**, E는 **공식 RR 전체 평가셋의 장면평균 ERGAS**다. 장면 하나가 H를 넘는지로 판단하지 않는다. V64 HQNR나 aligned reference HQNR로 대체하지 않는다.

run r의 후보 checkpoint 집합을 다음으로 고정한다.

$$
\mathcal K=\{1010j:j=1,\ldots,49\}\cup\{50000\}.
$$

즉 기존 `GRID1010_50K_v1`의 50개 후보다. 특정 profile이나 좋은 run만 후보를 더 촘촘히 만들지 않는다.

$$
\mathcal F_r=\{k\in\mathcal K:H(r,k)\ge0.9585\}.
$$

공식 RR가 완료된 동일-checkpoint 결과로 다음을 선택한다.

$$
 k_r^*=\operatorname*{arg\,min}_{k\in\mathcal F_r}
 \left(E,-\mathrm{SCC},-\mathrm{PSNR},\mathrm{SAM},-\mathrm{Q8},-\mathrm{SSIM},-H,k\right).
$$

여기서 **ERGAS가 먼저**이고, 뒤 항목은 값이 정확히 같을 때의 결정적 tie-break다. 임의 epsilon tie band나 H와 E의 가중합을 넣지 않는다.

$$
 \operatorname{joint\_pass}(r)
 = [H(r,k_r^*)\ge0.9585]\land[E(r,k_r^*)<2.040].
$$

| 가상 값 | 판정 |
|---|---|
| H=.9586 / E=2.039 vs H=.9602 / E=2.070 | **첫 번째 선택** |
| H=.9586 / E=2.055 vs H=.9602 / E=2.071 | 아직 E 목표는 미달이지만 **첫 번째 선택** |
| H=.95849 / E=2.030 | H 하한 미달. 반올림으로 통과 처리하지 않음 |
| H=.9590 / E=2.040000 | H 통과, strict E 목표 미달 |
| 적격 후보 없음 | `no_eligible`; target 수치는 빈 값. 정상 평가 종결이며 다음 학습 진행 |
| 적격 후보가 있으나 일부 공식 RR 미계산 | `incomplete_official_rr`; 확정 target와 공동목표 판정 보류 |

**첫 H 통과 시점에 학습을 멈추지 않는다.** 50K까지 같은 schedule로 진행해, 이후에도 H를 지키면서 E가 내려가는 후보를 수집한다. H를 넘었다고 loss·LR·aligner 동결·평가 모드를 중간 변경하지 않는다.

### 2.1 target, raw-max, exact50K는 반드시 분리

- **Target:** H 하한을 만족하는 후보 중 E 최소. 이번 주 결과다.
- **Legacy/raw-max:** 이전 결과와 연결하기 위한 최고 H 관련 값. 주 target를 덮어쓰지 않는다.
- **Exact50K:** H 통과 여부와 무관하게 같은 학습 지점의 공식 RR·FR를 보조 기록한다. profile 비교의 선택-step 영향을 점검한다.

어느 경우도 서로 다른 step의 H와 E를 결합하지 않는다. ‘E가 가장 낮은 50K’의 H가 하한 미달이면 주 target가 아니다. Target H가 높아도 E가 나쁘면 더 낮은 E의 H-적격 후보보다 우선하지 않는다.

### 2.2 ERGAS를 우선하기 위한 이번 변경의 범위

**선택 정책과 보고를 E 우선으로 연결하는 것이 확정 변경**이다. 학습에 새 ERGAS loss/L2 loss를 넣는 것은 아니다. β=.1/.2 중 어느 값이 H 하한 안에서 더 낮은 E를 주는지는 실험으로 확인한다. 계수 변경만으로 E<2.040을 달성한다고 보장하지 않는다.

---

## 3. 학습 방법 — 모든 서버의 불변 기준

이 절은 [H]의 확정 계산을 계승한다. 파일명·과거 campaign 주석보다 **실제 resolved config와 실행 경로**가 이 정의를 만족해야 한다.

### 3.1 구조·자산·초기화

| 항목 | 고정 정의 |
|---|---|
| Dataset | WV3, 기존 train/validation/RR 및 FR paper mat20 동일 bytes·전처리 |
| Teacher | T0 W112D123의 기존 selected A/U 모두 frozen |
| T0 checkpoint SHA256 | `16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32` — 실제 파일과 대조 |
| Student U-Net | W104D121, depth[1,2,1], 입력 PAN+MS 9ch, `in_mode=paper`, `attn_locations=[]`, `norm=ln`, `mode_modulation=false` |
| Student aligner | 기존 global (dy,dx) CNN. T0 A 전체 state를 별도 clone하여 trainable로 사용 |
| U-Net 초기화 | 신규 seed로 fresh. 같은 seed의 두 profile은 **동일한 저장 초기 U state를 재사용** |
| Optimizer/scheduler | 각 run에서 fresh. 앞선 profile의 학습 가중치나 optimizer를 이어받지 않음 |
| PAN/MS 입력 | `I-NATIVE-TRANSFER`; 추가 Student jitter/offset 훈련 없음 |
| Aligner view | PAN·MS base 양쪽 margin4 후 기존 spatial z-score와 GroupNorm |
| Warp | PAN만 bicubic / border / align_corners=False. dy,dx는 현재 PAN pixel |
| 잔차 base | 기존 bicubic×4 MS base를 최종 출력에 정확히 한 번 더함 |
| 주 추론 | 같은 checkpoint의 Student A/U를 함께 사용. Teacher·GT·q probe는 추론에 없음 |

T0를 새로 학습하거나 checkpoint를 바꾸지 않는다. N2 donor를 T0 대신 넣지 않는다. T0 U-Net을 Student에 partial load하거나 완료 Student를 tail fine-tuning하지 않는다.

### 3.2 Forward

LRMS X, 원 PAN P, GT Y, bicubic MS base M, Teacher/Student 첨자 T/S를 사용한다.

$$
 M_i=\mathcal U_4(X_i),\quad
 \mathbf c_{T,i}=A_{\phi_T}(P_i,M_i),\quad
 \mathbf c_{S,i}=A_{\phi_S}(P_i,M_i),
$$
$$
 Z_{T,i}=M_i+F_{\theta_T}([\mathcal W(P_i,\mathbf c_{T,i}),M_i]),
$$
$$
 Z_{S,i}=M_i+F_{\theta_S}([\mathcal W(P_i,\mathbf c_{S,i}),M_i]).
$$

**native는 추가 jitter가 없다는 뜻이며 aligner를 끈다는 뜻이 아니다.** MS/GT/최종 출력 좌표를 옮기거나 inverse warp하지 않는다. 수학식의 concat을 이유로 원 backbone API의 LRMS 인자에 이미 확대된 M을 넣지 않는다.

### 3.3 Teacher error·Student error·adaptive KD

$$
 e_{T,i}(p)=\frac1C\sum_b|Z_{T,i,b}(p)-Y_{i,b}(p)|,\quad
 e_{S,i}(p)=\frac1C\sum_b|Z_{S,i,b}(p)-Y_{i,b}(p)|,
$$
$$
 k_i(p)=\frac1C\sum_b|Z_{S,i,b}(p)-\operatorname{sg}(Z_{T,i,b}(p))|.
$$

$$
 d_{T,i}(p)=\operatorname{sg}\frac{e_{T,i}(p)}{e_{T,i}(p)+\tau_R},\quad
 \tau_R=0.012463942170143127,
$$
$$
 a_{T,i}(p)=\operatorname{sg}\left[\operatorname{clip}_{[0,1]}
 \frac{[e_{S,i}(p)-e_{T,i}(p)]_+}{e_{S,i}(p)+10^{-6}}\right].
$$

$$
 \ell_{H,i}=\operatorname{mean}_{p\in\Omega}[(1+\alpha d_{T,i}(p))e_{S,i}(p)],
$$
$$
 \ell_{K,i}=\operatorname{mean}_{p\in\Omega}[\beta(1-d_{T,i}(p))a_{T,i}(p)k_i(p)].
$$

평균은 고정된 전체 pixel 영역에서 계산한다. 활성 soft pixel 수나 가중치 합으로 다시 나누지 않는다. β는 ℓK 안에 이미 한 번 포함되므로 최종 loss에서 다시 곱하지 않는다. Weight 계산은 detach이고 hard의 eS와 soft의 k는 live다.


### 3.4 q 가중과 GT edge

T0의 AXIS16 raw q cache를 sample index·실제 augmentation view로 조회한다.

$$
 q_{T,i}=\frac{1}{2\cdot16}\sum_{j=1}^{16}
 \|\mathbf c_{T,i,j}+\boldsymbol\epsilon_j-\mathbf c_{T,i,0}\|_1,
$$
$$
 s_i=\operatorname{sg}\frac{q_{\mathrm{ref}}}{q_{\mathrm{ref}}+q_{T,i}},\qquad
 q_{\mathrm{ref}}=0.3276133416220546.
$$

분자2, cutoff, batch 재정규화, temperature를 추가하지 않는다. 이번 두 profile은 A/E 모두 q를 사용한다. 과거 uniform 대조를 확인할 때만 uniform=.5가 정확한 기준이다. 학습 중 q를 Student로 다시 계산하지 않는다.

Edge는 band별 signed Scharr/32, reflect1, 상하좌우1px를 제외한 고정 interior ΩE다.

$$
 \ell_{E,i}=\tfrac12\operatorname{mean}_{b,p\in\Omega_E}|D_xZ_{S,i,b}(p)-D_xY_{i,b}(p)|
 +\tfrac12\operatorname{mean}_{b,p\in\Omega_E}|D_yZ_{S,i,b}(p)-D_yY_{i,b}(p)|.
$$

Target는 GT만이다. Teacher/PAN edge, 5×5 variance, gradient magnitude loss로 교체하지 않는다. λE는 **절대계수 .002**이며 과거 환산을 다시 적용해 .004로 만들지 않는다.

### 3.5 최종 loss와 gradient routing

$$
 L_U=\frac1B\sum_i[\ell_{H,i}+\ell_{K,i}+\lambda_Es_i\ell_{E,i}],\qquad
 L_A=\frac1B\sum_i s_i\ell_{H,i}.
$$

$$
 G_U=\nabla_{\theta_S}L_U,\qquad G_A=\nabla_{\phi_S}L_A.
$$

**같은 forward·같은 pre-step 상태에서 두 gradient를 따로 계산한 뒤 optimizer를 한 번 step한다.** `L_U+L_A`의 전체 backward 금지. A gradient 계산은 U-Net의 입력 Jacobian을 통과하지만 그 gradient를 U parameter에 누적하지 않는다. A는 q-weighted **재가중 hard**를 받으며 plain L1로 바꾸지 않는다. Soft·edge·Student offset이 A의 직접 목적함수로 들어가지 않는다.

### 3.6 학습·데이터 실행 설정

| 항목 | 값/검사 |
|---|---|
| 업데이트 | optimizer update 50,000; epoch 수나 20시간 자체를 학습 step으로 대신하지 않음 |
| Batch | 48, single MS task; MARs용 batch 복제 없음 |
| U peak LR / A peak LR | 1e−4 / 3e−6, rA=.03 |
| Optimizer | AdamW, weight_decay=.01, 기존 두 parameter group 유지 |
| Scheduler | 기존 cosine, warmup100, 전체50K 기준. 중간 H 통과 때 재시작 금지 |
| Augmentation | 기존 crop=False, hflip=True, vflip=True, rot=True, return_meta=True |
| Aligner/U dropout·norm | 원 config 그대로. U dropout0, channel-LN; A 기존 GN |
| Evaluation | 기존 1010-step grid 및 exact50K, test batch1 |
| Resume | `kdv.exact_resume=true`; 실제 중단 run에만 같은 ID·원 상태로 재개 |
| Precision 등 암묵 default | **참조 run의 resolved 값으로 명시**. 아래 검사로 채우며 새 환경 default를 추정해 넣지 않음 |

현재 template에는 AdamW의 betas/eps, mixed_precision 등의 일부 값이 명시돼 있지 않다. 코드가 default를 사용하는 항목은 적용 시 실제 optimizer/scheduler/Accelerator에서 betas·eps·AMSGrad·minimum LR·precision·gradient accumulation·gradient clipping·framework version을 출력해 `resolved_recipe.json`에 남긴다. 기존 정의와 맞지 않으면 해당 서버의 새 실행만 보류한다. 임의의 숫자를 본 계획에서 새 기본값으로 만들지 않는다.

---

## 4. 파라미터 범위 — 두 profile을 모든 서버에서 교차

| profile | λE | rA | α | **β** | U LR | A LR | A/E q | 목적 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **G23** | .002 | .03 | 1.0 | **.1** | 1e−4 | 3e−6 | q/q | 이미 H 하한 자산을 만든 기준 |
| **B20A03** | .002 | .03 | 1.0 | **.2** | 1e−4 | 3e−6 | q/q | H 하한 내 E 개선 가능성을 같은 seed에서 확인 |

새 λE/rA/α/LR grid, β=.15, architecture sweep, GT-loss 추가는 이번 20h 기본 편성에 넣지 않는다. ‘ERGAS 우선’이 근거 없이 여러 축을 동시에 바꾸라는 뜻은 아니다.

**쌍의 의미:** 한 seed의 두 run은 동일 Student U 초기값·T0 A 초기값·학습 데이터 순서·augmentation RNG 정책을 사용하고 β만 달라야 한다. 서로 다른 서버의 같은 seed를 주 대응으로 만들지 않는다.

- 각 쌍의 실행 순서를 AB/BA로 교대해 첫/두 번째 실행 위치가 한 profile에만 편중되지 않게 한다.
- β를 학습 중 번갈아 바꾸는 것이 아니다. 한 run의 β는 50K 내내 고정이다.
- 두 번째 run은 첫 번째 결과와 무관하게 실행한다. H 실패를 이유로 짝을 취소하지 않는다.
- 다른 쌍·다른 서버의 결과를 기다리지 않는다. 대기는 **자기 자원·코드 무결성·남은 시간**만으로 결정한다.
- 같은 seed의 두 run에서 init hash나 처음 batch-meta sequence가 다르면 `pair_unverified`로 남긴다. 해당 결과 자체를 삭제하지 않는다.

---

## 5. 서버별 신규 case — 520xx / v4로 기존 실행과 분리

**새로 정의한 기본 목표는 36run = 18개 같은-server/seed 쌍이다.** 추가 10run = 5쌍은 시간 예비 목록이다. 따라서 CSV의46행은 모두 반드시20시간 안에 학습한다는 뜻이 아니다. §7의 시간 검사로 시작 가능한 **서버별 목록의 유한한 앞부분**만 실행한다.

기본 36run도 현재 run 잔여·준비·평가가 길면 모두 못 끝날 수 있다. 건수를 맞추기 위해20시간을 연장하거나 이미 시작한 run을50K 완료처럼 가장하지 않는다. ‘기본’은 시간 여유가 있을 때의 목표이지 예산을 무시하는 `mandatory=True`가 아니다.

| 서버 | 기본 seed 쌍 | 기본 run | 예비 seed 쌍 | 예비 run |
|---|---|---:|---|---:|
| s1 | 52001 / 52006 / 52011 | 6 | 52016 | 2 |
| s2 | 52002 / 52007 / 52012 | 6 | 52017 | 2 |
| s3 | 52003 / 52008 / 52013 / 52018 / 52023 | 10 | 52028 | 2 |
| s4 | 52004 / 52009 / 52014 / 52019 / 52024 | 10 | 52029 | 2 |
| s5 | 52005 / 52010 | 4 | 52015 | 2 |

각 seed에 G23와 B20A03 두 run을 모두 정의한다. 새 namespace는 기존 1234/3407/777/2026/4321 및41xxx 이력과 섞이지 않도록 선택한 **명시 배정**이다. 실제 적용 시 현재 로컬 자산/큐/Sheet와 충돌을 검사한다. 충돌이 있으면 조용히 다른 seed로 대체하지 말고 manifest revision을 갱신한다.

### 5.1 정확한 실행 순서

ID 약칭이 아니라 아래의 `run_id`를 config 파일명·work_dir·로그·업로드 원본 key로 사용한다. 동봉 CSV에도 동일46개가 있다. 가짜 G22 control ID를 생성하지 않는다.

#### s1

| 순서 | 구분 | Profile | Seed | 정확한 run ID |
|---:|---|---|---:|---|
| 1 | 기본 | G23 | 52001 | `PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S52001_FRESH50_v4` |
| 2 | 기본 | B20A03 | 52001 | `PAKD50_QRC24_S1_B20A03_W104_D121_WV3_T0_S52001_FRESH50_v4` |
| 3 | 기본 | B20A03 | 52006 | `PAKD50_QRC24_S1_B20A03_W104_D121_WV3_T0_S52006_FRESH50_v4` |
| 4 | 기본 | G23 | 52006 | `PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S52006_FRESH50_v4` |
| 5 | 기본 | G23 | 52011 | `PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S52011_FRESH50_v4` |
| 6 | 기본 | B20A03 | 52011 | `PAKD50_QRC24_S1_B20A03_W104_D121_WV3_T0_S52011_FRESH50_v4` |
| 7 | 시간 예비 | B20A03 | 52016 | `PAKD50_QRC24_S1_B20A03_W104_D121_WV3_T0_S52016_FRESH50_v4` |
| 8 | 시간 예비 | G23 | 52016 | `PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S52016_FRESH50_v4` |

#### s2

| 순서 | 구분 | Profile | Seed | 정확한 run ID |
|---:|---|---|---:|---|
| 1 | 기본 | B20A03 | 52002 | `PAKD50_QRC24_S2_B20A03_W104_D121_WV3_T0_S52002_FRESH50_v4` |
| 2 | 기본 | G23 | 52002 | `PAKD50_QRC24_S2_G23_W104_D121_WV3_T0_S52002_FRESH50_v4` |
| 3 | 기본 | G23 | 52007 | `PAKD50_QRC24_S2_G23_W104_D121_WV3_T0_S52007_FRESH50_v4` |
| 4 | 기본 | B20A03 | 52007 | `PAKD50_QRC24_S2_B20A03_W104_D121_WV3_T0_S52007_FRESH50_v4` |
| 5 | 기본 | B20A03 | 52012 | `PAKD50_QRC24_S2_B20A03_W104_D121_WV3_T0_S52012_FRESH50_v4` |
| 6 | 기본 | G23 | 52012 | `PAKD50_QRC24_S2_G23_W104_D121_WV3_T0_S52012_FRESH50_v4` |
| 7 | 시간 예비 | G23 | 52017 | `PAKD50_QRC24_S2_G23_W104_D121_WV3_T0_S52017_FRESH50_v4` |
| 8 | 시간 예비 | B20A03 | 52017 | `PAKD50_QRC24_S2_B20A03_W104_D121_WV3_T0_S52017_FRESH50_v4` |

#### s3

| 순서 | 구분 | Profile | Seed | 정확한 run ID |
|---:|---|---|---:|---|
| 1 | 기본 | G23 | 52003 | `PAKD50_QRC24_S3_G23_W104_D121_WV3_T0_S52003_FRESH50_v4` |
| 2 | 기본 | B20A03 | 52003 | `PAKD50_QRC24_S3_B20A03_W104_D121_WV3_T0_S52003_FRESH50_v4` |
| 3 | 기본 | B20A03 | 52008 | `PAKD50_QRC24_S3_B20A03_W104_D121_WV3_T0_S52008_FRESH50_v4` |
| 4 | 기본 | G23 | 52008 | `PAKD50_QRC24_S3_G23_W104_D121_WV3_T0_S52008_FRESH50_v4` |
| 5 | 기본 | G23 | 52013 | `PAKD50_QRC24_S3_G23_W104_D121_WV3_T0_S52013_FRESH50_v4` |
| 6 | 기본 | B20A03 | 52013 | `PAKD50_QRC24_S3_B20A03_W104_D121_WV3_T0_S52013_FRESH50_v4` |
| 7 | 기본 | B20A03 | 52018 | `PAKD50_QRC24_S3_B20A03_W104_D121_WV3_T0_S52018_FRESH50_v4` |
| 8 | 기본 | G23 | 52018 | `PAKD50_QRC24_S3_G23_W104_D121_WV3_T0_S52018_FRESH50_v4` |
| 9 | 기본 | G23 | 52023 | `PAKD50_QRC24_S3_G23_W104_D121_WV3_T0_S52023_FRESH50_v4` |
| 10 | 기본 | B20A03 | 52023 | `PAKD50_QRC24_S3_B20A03_W104_D121_WV3_T0_S52023_FRESH50_v4` |
| 11 | 시간 예비 | B20A03 | 52028 | `PAKD50_QRC24_S3_B20A03_W104_D121_WV3_T0_S52028_FRESH50_v4` |
| 12 | 시간 예비 | G23 | 52028 | `PAKD50_QRC24_S3_G23_W104_D121_WV3_T0_S52028_FRESH50_v4` |

#### s4

| 순서 | 구분 | Profile | Seed | 정확한 run ID |
|---:|---|---|---:|---|
| 1 | 기본 | B20A03 | 52004 | `PAKD50_QRC24_S4_B20A03_W104_D121_WV3_T0_S52004_FRESH50_v4` |
| 2 | 기본 | G23 | 52004 | `PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S52004_FRESH50_v4` |
| 3 | 기본 | G23 | 52009 | `PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S52009_FRESH50_v4` |
| 4 | 기본 | B20A03 | 52009 | `PAKD50_QRC24_S4_B20A03_W104_D121_WV3_T0_S52009_FRESH50_v4` |
| 5 | 기본 | B20A03 | 52014 | `PAKD50_QRC24_S4_B20A03_W104_D121_WV3_T0_S52014_FRESH50_v4` |
| 6 | 기본 | G23 | 52014 | `PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S52014_FRESH50_v4` |
| 7 | 기본 | G23 | 52019 | `PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S52019_FRESH50_v4` |
| 8 | 기본 | B20A03 | 52019 | `PAKD50_QRC24_S4_B20A03_W104_D121_WV3_T0_S52019_FRESH50_v4` |
| 9 | 기본 | B20A03 | 52024 | `PAKD50_QRC24_S4_B20A03_W104_D121_WV3_T0_S52024_FRESH50_v4` |
| 10 | 기본 | G23 | 52024 | `PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S52024_FRESH50_v4` |
| 11 | 시간 예비 | G23 | 52029 | `PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S52029_FRESH50_v4` |
| 12 | 시간 예비 | B20A03 | 52029 | `PAKD50_QRC24_S4_B20A03_W104_D121_WV3_T0_S52029_FRESH50_v4` |

#### s5

| 순서 | 구분 | Profile | Seed | 정확한 run ID |
|---:|---|---|---:|---|
| 1 | 기본 | G23 | 52005 | `PAKD50_QRC24_S5_G23_W104_D121_WV3_T0_S52005_FRESH50_v4` |
| 2 | 기본 | B20A03 | 52005 | `PAKD50_QRC24_S5_B20A03_W104_D121_WV3_T0_S52005_FRESH50_v4` |
| 3 | 기본 | B20A03 | 52010 | `PAKD50_QRC24_S5_B20A03_W104_D121_WV3_T0_S52010_FRESH50_v4` |
| 4 | 기본 | G23 | 52010 | `PAKD50_QRC24_S5_G23_W104_D121_WV3_T0_S52010_FRESH50_v4` |
| 5 | 시간 예비 | G23 | 52015 | `PAKD50_QRC24_S5_G23_W104_D121_WV3_T0_S52015_FRESH50_v4` |
| 6 | 시간 예비 | B20A03 | 52015 | `PAKD50_QRC24_S5_B20A03_W104_D121_WV3_T0_S52015_FRESH50_v4` |

### 5.2 식별 및 재현 metadata

명시 tuple의 예시는 다음과 같다. 이는 **case registry에 넣을 데이터 구조**이며 완전한 trainer YAML이 아니다.

```yaml
campaign_id: QRC24_MIX20H_20260918_v1
queue_revision: QRC24_MIX20H_Q1_20260918
server: s1
pair_id: M20_S1_S52001
runs:
  - profile: G23
    seed: 52001
    version: v4
  - profile: B20A03
    seed: 52001
    version: v4
selector: HQNR9585_ERGAS2040_v2
requires_recipe_lock: false
requires_other_server_results: false
```

두 번째 항목을 local lock·기본 profile·서버 기본 seed로 다시 해석하지 않는다. §3의 전체 학습 설정을 합쳐 실제 YAML을 생성하고 원 데이터 경로와 로컬 경로 매핑을 따로 기록한다.

필수 필드: `campaign_id`, `queue_revision`, `original_run_id`, `server_id`, `pair_id`, `profile`, `seed`, `version`, `method`, `selector`, `eval_mode`, `source_train_code_ref`, 실제 `release_sha`, `dirty_diff_sha`, `init_unet_sha256`, `init_aligner_sha256`, `teacher_sha256`, `cue_sha256`, `dataset_hashes`, `resolved_config_sha256`, `recipe_hash`, `started_at_utc`, `deadline_at_utc`.

- `seed` 하나만 key로 사용하지 않는다. 같은 seed를 두 profile에 의도적으로 사용한다.
- `recipe_hash`는 학습 정의의 fingerprint일 뿐, 다른 서버의 hash·결정 대기를 만드는 lock이 아니다.
- profile이 다른 두 run은 β 때문에 recipe hash가 다를 수 있다. 비교는 β를 제외한 허용 diff와 같은 초기값·batch 정책으로 확인한다.
- `campaign_id + server + profile + seed + version`의 ID와 실제 config가 맞지 않으면 실행 금지한다.
- 현재 52xxx를 기존 41xxx용 resolver에 넣었을 때 profile을 G23로 강제하거나 가상 G22를 요구하는 경로가 없어야 한다. **해당 행의 explicit profile을 우선**한다.
- 새 파일명 suffix `v4`는 편성/metadata 버전이다. factor4나 새로운 loss 버전이 아니다.

---

## 6. 현재 서버별로 꼬인 큐·seed·case를 정리하는 절차

### 6.1 적용 전 로컬 inventory — 날짜/Sheet만 보고 상태를 추정하지 않는다

각 서버 담당자는 `migration_manifest.json`에 다음을 한 번 저장한다.

1. 실제 host와 source server, PID·현재 run·step·정상학습/후처리/중단 상태.
2. 원 active/effective/persistent queue, handover, mandatory, extra_priority, reservations, watchdog/waiter 설정과 각 hash.
3. 존재하는 완료 checkpoint·중단 checkpoint·init cache, 원 run key, config/코드/data/Teacher/cue hash.
4. 기존 `_eval_phase/hold.json`, `recipe_lock.json`, deadline 파일의 존재·내용·역할.
5. 새 campaign의 시작/마감 시각, 신규 46개 중 이 서버 행, 중복/충돌 검사, 최종 admission 대상.

이 단계는 **다른 서버의 inventory가 끝나기를 기다리지 않는다.** 한 서버의 데이터 누락이 다른 서버 학습을 중단시키지 않는다.

### 6.2 상태별 처리

| 기존 항목 상태 | 처리 |
|---|---|
| 완료 | 원 ID/seed/profile/step/기존 품질 보존. 새 cohort로 이름 변경하거나 재학습하지 않음 |
| 정상 실행 중 | 원 코드·계수·seed·50K 정의로 계속. 안전한 case 경계에서 새 큐로 넘어감 |
| 후처리 중 | 기존 파일 완성 후 새 큐 연결. 신규 selector 결과는 별도 파일로 추가 |
| checkpoint가 있는 중단 run | 미시작으로 분류하지 않음. 원 resume 상태를 보존하고 재개 필요/비용을 명시. 새 seed의 초기값으로 쓰지 않음 |
| 완전 미시작의 원 grid/41xxx/β-close | `superseded_pending_M20`로 이력만 보존하고 이번 active 큐에서 제외 |
| 실패/NaN | 자동 새 seed·새 version으로 숨기지 않음. 오류/원비용 유지 |
| Sheet에 없지만 로컬 결과 있음 | 미학습으로 재실행하지 않음. 업로드 backlog로 분리 |
| 긴 이름/단축 이름의 중복 | canonical key로 연결. 같은 run의 동일 checkpoint 계산을 새 반복으로 세지 않음 |

원래 β-close v3 네 개를 모두 채워야 새 큐를 시작할 수 있다는 조건은 없다. 현재 s2 B20A03·777처럼 이미 끝난 비교는 reference로 보존한다. 현재 진행 중인 원 β-close가 있으면 원 정의로 마무리한다.

### 6.3 큐·writer를 한 번만 연결

새 로컬 실행 목록의 유일한 기준은 **이 서버의 manifest에 적힌 명시 tuple 순서**다. 구현할 경로 예시는 다음과 같으며 아직 배포됐다는 뜻은 아니다.

```text
config/queues/qrc24_mix20h_s1.txt       # 서버별로 s1…s5
work_dir/_qrc24_mix20h/
    migration_manifest.json
    plan_manifest.json
    queue_effective.txt
    pair_reservations.json
    status.json
    events.jsonl
    budget.json
```

Generator, 정적·effective·persistent queue, handover, mandatory, waiter, reservations를 같은 revision으로 맞춘다. 원 큐를 단순 append하지 않는다. 새 queue가 끝난 뒤 기존 `--extend`·최소24h 보충 규칙이 원 grid나 41xxx를 다시 기동해서는 안 된다.

**기존 lock 파일은 과거 이력으로 보존하되, 이번 run의 profile 선택·생성·admission에서는 읽지 않는다.** 정상 실행을 위한 file-writing mutex·단일 GPU 자원 보호는 계속 사용할 수 있다. 이것은 공통 결과를 기다리는 recipe lock과 다르다.

### 6.4 NOA와 독립 ASV 처리

- 완료한 선행 NOA cohort를 새 52xxx 결과가 생길 때마다 다시 열지 않는다. 새로운 모든 run의 NOA 재평가는 이번 20h 필수 항목이 아니다.
- s5의 과거 NOA 미완료/미업로드, s1 G21·3407의 cohort 포함 여부를 로컬에서 확인한다. **이미 계산됐다면 업로드만** 하고, 미계산 finite cohort이면 해당 서버 준비 작업에서 보완하되 다른 서버를 기다리게 하지 않는다.
- 오래된 eval hold가 남아 있다면 원인·cohort완료·현재 writer를 확인한 뒤 담당자가 안전하게 해소한다. `rm hold.json`만으로 의미를 바꾸지 않는다.
- ASV C00–C05는 별도 작업공간·가용 자원으로 실행한다. 새 학습의 완료 조건이 아니다. 같은GPU를 사용하면 차지한 시간을20h에서 차감하며 자원이 없을 때 **진단만** 대기한다.
- NOA/CROP/GN 개입을 본 학습 인스턴스의 flag나 norm에 남기지 않는다. 본 학습/정상평가 forward는 A_ON으로 새 인스턴스에서 확인한다.

---

## 7. 20시간 예산과 서버별 실행량

### 7.1 시간의 의미

적용 담당자가 `campaign_started_at_utc=T0`, `deadline_at_utc=T0+20h`를 정해 모든 서버 manifest에 넣는다. UTC와 로컬 offset을 함께 표시한다. 다른 서버의 결과가 모여야 timestamp를 정하는 것이 아니다.

각 서버는 같은 20시간 창에서 독립적으로 실행한다. 늦게 적용된 서버의 deadline을 임의로 20시간 더 미루지 않는다. T0 이후의 현재 run 잔여, 준비, 평가, 자원 경합, 재시작 비용을 포함하되 과거 캠페인의 누적 비용을 다시 합산하지 않는다. 재시작이나 평가 대기 때문에 deadline을 0부터 다시 세지 않는다.

**‘GPU 학습 20시간 + 준비·평가 추가 시간’이 아니라, 준비와 마무리를 포함한 20시간 운영 창**이다. 기준은 서버당 학습 GPU 한 개이며, 여러 GPU를 추가로 동원하는 비용은 별도 승인 없이 숨겨 합산하지 않는다.

### 7.2 시작 시 예약값

Sheet의 Train(h)는 기존 보고값이며, 이번에 추가할 공식 target·exact50K 처리비 전체가 포함됐다고 가정하지 않는다. 아래 R은 계획용 보수 기준이고 실제 시간을 측정해 갱신한다.

$$\widehat t_s=1.10R_s+0.10\ \mathrm{hours/run}.$$

0.10h는 새 target/exact50K·업로드 증가분의 **초기 대용값**이다. 정기 grid 평가 등 R에 이미 포함된 시간은 다시 더하지 않는다. 준비 0.5h와 종료·잔여 평가 0.5h, 합계 1.0h를 기본 예약한다. 현재 run 잔여와 추가 backlog 비용은 이와 별도로 남은 시간에서 차감한다.

| 서버 | 기준 R(h/run) | 신규 run 예약(h) | 기본 run 목표 | 기본 run + 준비·종료 1h(h) | 예비 run |
|---|---:|---:|---:|---:|---:|
| s1 | 2.45 | 2.795 | 6 | 17.77 | 2 |
| s2 | 2.60 | 2.960 | 6 | 18.76 | 2 |
| s3 | 1.50 | 1.750 | 10 | 18.50 | 2 |
| s4 | 1.55 | 1.805 | 10 | 19.05 | 2 |
| s5 | 3.40 | 3.840 | 4 | 16.36 | 2 |

이 표는 **현재 run 잔여가 0이고 준비가 계획 범위인 경우**다. s2·s3·s4는 잔여 작업이 길면 마지막 기본 쌍이 이번 예산 밖으로 밀릴 수 있다. s5는 최근 3.40h를 반영해 4run만 기본에 두며, 새 실측이 충분히 짧을 때만 예비 쌍까지 시작한다. 과거 빠른 기록인 1.46h를 일괄 적용하지 않는다.

### 7.3 Pair 단위 자원 admission — 성능 gate가 아니다

쌍의 첫 run을 시작하기 전에 두 run과 종료 평가비를 함께 예약한다.

$$
\mathrm{remaining\_wallclock}
\ge \widehat t_{\mathrm{first}}+\widehat t_{\mathrm{second}}
 +\mathrm{known\_evaluation\_backlog}+\mathrm{close\_reserve}.
$$

close reserve의 초기값은 0.5h다. 이미 누적된 target 평가가 많다면 backlog 비용을 더한다. 부족하면 새 쌍을 시작하지 않고 남은 평가·업로드·요약을 수행한다. 첫 profile만 성과가 좋아서 실행하고 짝을 버리는 방식으로 줄이지 않는다.

한 쌍이 오류나 예상 밖의 자원 변동으로 불완전해지면 상태와 이유를 남긴다. 완결 쌍 분석에서는 제외하되 단일 run의 결과는 보존한다. 첫 두 새 run 이후에는 **이번 서버·이번 방법의 최근 실제 end-to-end 비용**으로 예약을 갱신한다. 표본이 적으면 느린 쪽을 사용하고, 충분한 기록이 생기면 최근 상위분위와 10% 여유를 사용한다. End-to-end 실측에 이미 포함된 후처리 시간을 다시 0.10h로 더하지 않는다.

**완료 건수, HQNR, ERGAS는 시작 허용 조건이 아니다.** H 미달이나 공동목표 0건이어도 시간상 가능한 다음 쌍을 진행한다. 기본 목록을 마쳤고 시간이 남으면 서버별 예비 쌍 하나만 실행한다. 추가 무한 seed 생성은 없다.

**기존 예산 gate 주의:** 전체 미실행 46개를 모두 ‘필수 remaining’으로 합산하면 첫 run부터 차단될 수 있다. 전체 case 목록은 후보 목록이지 동시 예약 목록이 아니다. 현재 허용한 **로컬 쌍과 미결 평가만** 예약하도록 adapter를 연결한다. 반대로 `required=True`로 모든 예산 초과를 경고만 하고 계속 실행하는 것도 금지한다. 최종 wall-clock 검사는 runner가 적용한다.

### 7.4 20시간 종료

Deadline에 새 학습, 예비 쌍, 자동 extra를 시작하지 않는다. 사전 예약으로 각 정상 run의 50K 완료를 목표로 하지만 실행시간은 보장값이 아니다.

예상 밖으로 진행 중 run이 deadline을 넘기게 되면 **강제 kill하지 말고 안전한 optimizer-update 경계에서 완전한 resume checkpoint를 저장해 정지**한다. `stopped_budget`과 실제 완료 step을 기록하고, 이를 FRESH50 완료 또는 주 paired 표의 완료 run으로 세지 않는다. 원 정의로 재개할 수 있게 보존하되 이번 예산 뒤 자동 재개하지 않는다. 안전 저장을 위한 최소 초과가 발생하면 초과시간도 명시한다.

나쁜 성능을 이유로 중도 중단하는 것과 예산 만료에 따른 안전 정지는 구분한다. 적용 전에 필요한 안전 저장·정지 기능이 실제로 작동하는지 확인해야 한다. 새 예산이나 기간 연장은 별도 결정이다.

---

## 8. 평가 실행 연결 — v2를 실제 자동 경로에 넣는다

### 8.1 기존 자산을 먼저 읽되 전체 backlog가 모든 학습을 막지는 않는다

우선 보존할 목표 자산은 s1 G23·1234와 s3 G23·41003이다. s4 H31·1234도 경계값·공식 target 확인용으로 남긴다. 유효한 cache는 재사용하고 재학습하지 않는다.

각 서버는 자기 새 run의 필수 후처리를 우선한다. 기존 완료 run의 H 적격 후보 v2 보완은 로컬 평가 backlog로 처리한다. 적격이 없는 run은 FR 로그·후보 identity 확인 후 `no_eligible`로 종결한다. 정의가 불명확하거나 candidate가 유실된 경우에는 사유를 기록한다. 기대 성능을 숫자로 채우지 않는다.

과거 전체 backlog 완료를 새 학습의 전역 장벽으로 사용하지 않는다. 시간 내 완료하지 못한 작업은 pending으로 보고한다. 필요하면 가용 CPU·별도 평가 GPU에서 진행하되 자원·비용을 따로 기록한다.

### 8.2 매 신규 run의 필수 후처리

1. 원 grid 50개가 실제로 저장됐는지, checkpoint_metrics의 H가 raw-original protocol인지 확인한다.
2. Full-precision H≥0.9585인 후보를 추린다. 로그 H와 legacy 공식 FR의 identity·일치도 확인한다.
3. 모든 적격 후보의 공식 RR 6지표를 채운다. Proxy RR와 다른 evaluator 값을 섞어 정렬하지 않는다.
4. `HQNR9585_ERGAS2040_v2`로 E 우선 선택한다. 일부 적격 후보의 공식 RR가 없으면 임시값만 남기고 target를 확정하지 않는다.
5. Exact50K의 공식 RR·FR를 기록한다. 저장된 예측 출력의 identity가 맞으면 재추론 없이 metric만 계산한다.
6. 기존 legacy 행은 보존하고 v2 target·50K 영역을 업로드한다. 원본 key와 checkpoint identity를 read-back으로 확인한다.
7. H 적격 후보가 없거나 E<2.040이 없어도 실행 실패가 아니다. 다음 시간상 가능한 쌍으로 진행한다.

기존 CLI에서 지원하는 selector 호출 형태는 아래와 같다. `<run_id>`는 §5의 실제 ID로 치환한다. 이 명령은 평가 자원을 사용하므로 학습 중인 동일 GPU에 무조건 추가 투입하지 않는다.

```bash
python tools/qrecon24_select.py <run_id> \
  --official --threshold 0.9585 \
  --selector HQNR9585_ERGAS2040_v2 \
  --device cuda
```

이번에 필요한 것은 위 호출을 **새 campaign의 postrun·backlog·완료 판정·업로더까지 연결**하는 것이다. 기존 postrun의 기본 v1 호출을 그대로 두고 v2가 실행됐다고 기록하지 않는다. Selector가 새로 선택한 step 때문에 50K 학습 경로를 다시 시작하거나 바꾸지 않는다.

### 8.3 캐시·완료 key

완료 key는 `(run_id, checkpoint/grid identity, selector_id, dataset hashes, evaluator hashes, A_ON mode, precision)`다. V1 official 완료를 v2 완료로 대신하지 않는다. Selector만 바뀌었고 같은 후보의 공식 RR cache가 완전하다면 그 cache로 재정렬한다.

```text
results/qrecon24_target_selection_HQNR9585_ERGAS2040_v2.json
results/exact50k_official_AON.json       # 새로 연결/정규화할 명세 파일명
meta/resolved_recipe.json
meta/mix20h_run_manifest.json
```

`training_complete`, `raw_grid_complete`, `official_target_complete`, `exact50k_complete`, `sheet_uploaded`를 별도로 기록한다. `no_eligible`는 target 평가의 정상 종결이고, 자료 유실·공식 RR 일부 누락·업로드 실패는 각각 다른 상태다.

### 8.4 지표 해석

GT와 PAN/reference를 결과에 맞춰 이동시키지 않는다. RR GT는 metric 계산에 사용하며 inference shift 선택에는 사용하지 않는다. Candidate 선택에 RR·FR 세트를 반복 사용하므로 이 수치는 개발·모델 선택 결과다. 이를 미사용 test set의 독립 평가나 SOTA 확증으로 소개하지 않는다.

---

## 9. 구현 case와 담당자가 남길 실행 증거

파일에 case 이름을 적는 것과 runner가 실행할 수 있는 상태를 구분한다. 아래 항목은 **이 계획 적용 시 구현·연결할 작업**이며 이 MD 작성으로 완료됐다는 뜻은 아니다.

| 운영 case | 범위 | 완료 증거 | 다른 서버 대기 |
|---|---|---|---|
| **M20-PREP-sN** | Inventory·현재 run 보호·520xx 충돌·T0/cue/data/학습 정의 검사 | migration manifest, resolved template, inventory 표 | 없음 |
| **M20-PLAN-sN** | Explicit tuple에 따른 YAML·registry·큐 생성과 검증 | 실제 YAML 목록, parser 검사, queue hash | 없음 |
| **M20-TRAIN-실제ID** | 신규 run의 fresh50K | config/init/batch-meta/hash/step 로그 | 없음 |
| **M20-EVAL-실제ID** | v2 target 및 exact50K 공식 평가 | selector ID, 후보 hash, 공식 metric | 없음 |
| **M20-SHEET-sN** | 원 탭의 새 영역 업로드 | canonical key별 read-back | 없음 |
| **M20-CLOSE-sN** | 예산 종료·미완료 표·서버별 결론 | status, 비용·초과 기록, 완료 보고서 | 없음 |

### 9.1 최소 구현 산출물

명시 case CSV를 읽는 generator 또는 그와 동등한 tuple registry, 해당 서버의 profile·seed·version이 실제로 반영된 YAML, 같은 seed의 pairmate 실제 ID를 저장하는 metadata, 새로운 유한 큐와 runtime handover가 필요하다. 기존 v1/v2/v3 config는 덮어쓰지 않는다.

새 postrun adapter는 v2를 명시하고, 원본 case·hash를 기준으로 완료 여부를 판정해야 한다. **H 적격 없음, 자료 미완료, 예산 부족, code/config 불일치를 ‘실험 case 없음’이라는 한 상태로 묶지 않는다.** C*나 존재하지 않는 canonical G22를 대조군 key로 생성하지 않는다.

### 9.2 사전 검사 — 성능을 통과 조건으로 쓰지 않는다

| 검사 | 통과 조건 |
|---|---|
| CASE-LIST | 이 서버의 기본·예비 run 수, 순서, seed, profile이 CSV와 같음 |
| CASE-RESOLVE | 같은 520xx seed의 G23/B20A03 둘 다 생성 가능. Lock 존재·내용이 결과를 바꾸지 않음 |
| CASE-HISTORY | 기존 완료·진행 run의 정의 보존. 41xxx를 520xx로 rename하지 않음 |
| INIT-PAIR | 쌍의 U/A 초기 state hash 일치, optimizer·scheduler는 각 run에서 fresh |
| RECIPE | 허용 학습 diff는 β뿐. T0/cue/data/전처리/50K/정상 forward 일치 |
| GRAD | 같은 pre-step 상태에서 β 변경이 A의 직접 gradient를 바꾸지 않음. U/A 별도 미분 |
| FORWARD | A_ON에서 A 1회·warp 1회, NOA·GN 실험 flag 잔류 없음 |
| SELECT-ORDER | H=.9586/E=2.039를 H=.9602/E=2.070보다 우선 선택 |
| SELECT-BOUNDARY | H=.958499 제외, E=2.040000은 strict pass 아님. 반올림 판정 금지 |
| SELECT-COMPLETE | v1 official을 v2 완료로 오인하지 않음. 공식 RR 일부 누락은 incomplete |
| SELECT-SAME-CKPT | Target H/E 및 나머지 지표가 동일 checkpoint·동일 A_ON |
| QUEUE | 미시작 옛 extra/41xxx/ADJ가 재시작·watchdog로 되살아나지 않음 |
| BUDGET | 다음 쌍만 예약. 재시작으로 deadline이 바뀌지 않음. 최소24h 자동 연장 없음 |
| SHEET | Legacy/NOA/target/50K 영역 보존, ID 왕복 연결, 재업로드 중복 없음 |

수정 파일이 실행 중 trainer의 정의를 바꾸지 않도록 별도 worktree 또는 안전한 case 경계 배포를 사용한다. 모든 서버가 같은 검사를 통과할 때까지 서로 기다리라는 의미가 아니다. 문제가 있는 서버만 로컬 상태를 해소한다.

### 9.3 실행 준비 완료의 보고 양식

각 서버는 실행 전 다음 표를 실제 값으로 남긴다. 누락 상태를 `READY`로 표시하지 않는다.

```text
server / applied_release / queue_revision / deadline_utc
current_carryover_run / current_step / estimated_remaining_h
planned_base_count / reserve_count
actual_yaml_count / registry_resolved_count / next_two_real_run_ids
init_pair_check / teacher_hash_check / data_hash_check
selector_auto_id / target_completion_key / sheet_schema_version
old_queue_removed_count / preserved_running_count / remaining_budget_h
status = READY | BLOCKED_LOCAL_<reason> | WAIT_RESOURCE | BUDGET_CLOSED
```

계획·CSV를 전달한 것만으로 이 표가 채워졌다고 보고하지 않는다. 새 `--mix20` 같은 CLI가 이미 있다고 가정한 실행 지시를 만들지 않는다. 담당자가 실제 지원되는 생성·적용 명령을 구현 노트에 기록한다.

---

## 10. Sheet 기록 — target와 50K를 분리하고 과거 열은 보존

| 서버 | 원본 탭 | gid |
|---|---|---:|
| s1 | WV3-s1 | 994031662 |
| s2 | WV3-s2 | 991648123 |
| s3 | WV3-s3(5090) | 284220763 |
| s4 | WV3-s4 | 2026091404 |
| s5 | WV3-s5 | 823586191 |

적용 시 실제 header·gid를 다시 확인한다. 새 run은 예를 들어 **`M20 G23 S52001 v4`**로 짧게 표시할 수 있지만 원 run_id를 Notes/명시 ID 열에 보존해야 한다. 기존 canonical parser가 M20 표시를 지원하기 전에는 임의 축약을 적용하지 않는다. 기존 QRC24 단축 표시를 쓰고 campaign 열에 M20을 붙이는 방법도 가능하다.

| 영역 | 필수 정보 |
|---|---|
| M20 identity | campaign, queue_revision, server, profile, seed, pair_id, run_id, init SHA, release |
| **Target v2** | selector, step, ckpt SHA, **H_raw**, **ERGAS**, SCC, PSNR, SAM, Q8, SSIM, n_eligible, status, joint_pass |
| **Exact50K A_ON** | step50000, ckpt SHA, H_raw, ERGAS, SCC, PSNR, 공식 평가 완료 |
| Legacy/raw-max | 기존 값과 step 보존. 새 target로 몰래 교체하지 않음 |
| Runtime | 실제 updates, 학습·추가 평가시간, UTC 시각, finished/partial/failed 상태 |

H와 E가 한 checkpoint의 값임을 step·hash로 검증한다. 표시는 소수6자리 이상을 권장하지만 판정은 원 JSON full-precision 값으로 한다. 미적격 target에 0을 넣지 않고 빈 값과 `no_eligible`를 기록한다.

업로드는 기존 gspread 운영 경로를 확장한다. 각 서버는 자기 탭만 쓰고 기존 NOA 열·Date·Train(h)·Cost를 불필요하게 덮어쓰지 않는다. Header는 실제 라벨로 찾으며 고정 열 위치를 가정하지 않는다. 로컬 학습·평가 writer는 같은 쓰기 잠금을 공유하고 전체 row-clear/replace를 피한다.

업로드 장애는 `EVAL_COMPLETE_UPLOAD_PENDING`으로 분리한다. 이미 나온 metric 때문에 GPU 학습을 다시 하지 않는다. 로컬 산출물이 완전하고 충돌 방지 backlog가 있으면 단순 네트워크 오류가 다른 서버의 정지 조건이 되지 않는다.

---

## 11. 20시간 종료 후 분석과 다음 결정

### 11.1 필수 보고

Profile별로 **완료 seed 쌍 수, H 통과 run 수, 공동목표 run 수, H 적격 target의 E 분포**를 제시한다. 모든 run의 H 최댓값과 exact50K E/H를 함께 보존한다.

**주자산:** 새 run과 보존 기존 자산 중 H≥.9585를 지킨 가장 낮은 공식 E의 checkpoint. 기존 자산보다 못하면 그대로 기록한다.

**Paired 비교:** 같은 server/seed의 G23와 B20A03 차이를 target 기준과 exact50K 기준으로 분리한다. 한쪽이 H 미달이면 조건부 target E를 직접 비교할 수 없음을 표시하고 H 통과 여부의 차이를 기록한다. 시간 예비 쌍도 같은 정의를 사용한다.

**완료율:** 예산 미입장, partial50K, 오류, definition/identity 미확정, upload pending을 숨기지 않는다. 기본 36run 대비 실제 완료 수를 보고하며, 예비를 포함해 실행한 전체 결과도 공개한다.

**선택 효과:** Target 선택 step과 E, late6/50K 관계를 본다. 서로 다른 selected step의 차이를 순수 β 효과라고 부르지 않는다.

**서버 효과:** 서버마다 다른 seed 집합을 배정하므로 hardware와 seed의 기여를 완전히 분리할 수 없다. 주 비교는 같은 서버 내 쌍이다. 서로 다른 profile의 run 전체를 동일 설정의 독립 seed 반복으로 합산하지 않는다.

장면별 지표가 있다면 원 장면 단위로 대응 차이를 요약한다. 한 장면의 crop 수나 한 run의 checkpoint 50개를 독립 실험 수로 세지 않는다.

### 11.2 성과 판정

| 관측 | 처리 |
|---|---|
| **H≥.9585 & E<2.040** | 공동목표 자산 보존. 시간이 남으면 사전 큐를 계속해 반복 결과도 확보 |
| H≥.9585, E가 기존2.071234보다 낮지만 ≥2.040 | E 개선으로 기록하되 공동목표 달성으로 표시하지 않음 |
| H 통과는 있으나 E 개선 없음 | Selector·exact50K·ASV 결과로 원인 분리. 자동 무한 seed 추가 금지 |
| E<2.040이나 H 미달 | RR 측 참고 결과. 현재 우선목표의 성공은 아님 |
| 모든 E가2.04 이상 | 현재 방법·β 두 값·이번 예산에서 미달로 보고. 다음 loss/구조/학습량 변경은 별도 결정 |

좋은 값 하나를 보고 나머지 결과를 숨기지 않는다. 이번 20시간 도중 별도 성능 gate나 C* lock을 다시 만들지 않는다. 이후 한 profile로 더 좁힐 수는 있지만, 현재 run의 정의를 소급 변경하지 않는다.

---

## 12. 근거 자료 및 적용 우선순위

**최상위 운영 의도:** 이번 사용자 지시 — 전 서버20h, 좁은 파라미터 범위×seed 독립 실행, H≥.9585 이후 E 최소. 과거 공통 lock, 최소24h, 원41xxx 순서는 이에 우선하지 않는다.

**학습 계산의 근거:** [H]의 확정 수식과 실제 training-start metadata. 이번 편성 개편을 loss 정의 변경으로 읽지 않는다.

- **[A]** `PAN_Results_and_Plan_Implementation_Audit_2026-09-18.md`: 80run 감사, selector 자동 연결, 혼합 편성 차이, 서버 재현성 주의.
- **[H]** `PAN_New_Session_Handoff_2026-09-17.md` §2–3·§12.3: forward, loss, gradient, Teacher 계보와 정오.
- **[ASV]** `PAN_Aligner_Size_Norm_GAP_Standalone_Validation_2026-09-18.md`: 독립 C00–C05. 본 학습의 대기 조건이 아님.
- **[OLD-R2]** `PAN_QRC24_Narrow_R2_SeedLock_ERGAS_2026-09-17.md`: 과거 편성 이력. 이번과 충돌하는 lock·seed20 조건은 계승하지 않음.
- **[SHEET]** `pan-cvpr27`, ID `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`: §1.2 범위를 이번에 직접 재확인.

조회된 저장소 ref: `a565cafbb1d1f124dbb34450207b0374ade5aa42`.

```text
https://github.com/hojunking/PAN-Crafter-repro/blob/a565cafbb1d1f124dbb34450207b0374ade5aa42/train_kdv.py
https://github.com/hojunking/PAN-Crafter-repro/blob/a565cafbb1d1f124dbb34450207b0374ade5aa42/kdv/qrecon.py
https://github.com/hojunking/PAN-Crafter-repro/blob/a565cafbb1d1f124dbb34450207b0374ade5aa42/kdv/losses_rec.py
https://github.com/hojunking/PAN-Crafter-repro/blob/a565cafbb1d1f124dbb34450207b0374ade5aa42/config/PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S1234_FRESH50_v1.yaml
https://github.com/hojunking/PAN-Crafter-repro/blob/a565cafbb1d1f124dbb34450207b0374ade5aa42/tools/gen_pakd50_configs.py
https://github.com/hojunking/PAN-Crafter-repro/blob/a565cafbb1d1f124dbb34450207b0374ade5aa42/tools/qrecon24_select.py
https://github.com/hojunking/PAN-Crafter-repro/blob/a565cafbb1d1f124dbb34450207b0374ade5aa42/tools/qrecon24_postrun.py
https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit
```

[A]의 과거 결과와 이번 직접 재조회 범위를 구분했다. 현재 서버 프로세스, 모든 후보 checkpoint, 배포 상태를 검증 완료했다는 뜻은 아니다. 실제 실행 정의가 달랐던 과거 결과를 같은 방법으로 조용히 합치지 않는다.

---

## 부록 A. 동봉 case CSV와 적용 상태

`PAN_MIX20H_Server_Cases_2026-09-18.csv`는 §5와 동일한 **46개 exact run ID, 서버별 순서, 기본/시간 예비, profile, seed, β, 학습 정의 식별자, selector, 시간 예약값**을 포함한다.

기본 36개·예비 10개이며 중복 run ID는 없다. 총 23개 seed가 각각 두 profile에 배정된다. 모든 `implementation_status`는 `DEFINED_NOT_DEPLOYED`다. 실제 YAML 생성·registry·큐 반영 뒤 담당자가 서버별 준비 완료표를 별도로 작성한다.

CSV는 generator adapter 또는 registry 정의의 기준표다. 현재 학습 runner가 이 CSV를 수정 없이 직접 읽는다고 가정하지 않는다. **이번 학습 허용 조건은 명시 case, 정의 무결성, 로컬 자원, 남은 시간이며 다른 서버의 성능 결과나 공통 lock이 아니다.**

CSV SHA256: `6197fcbbc9eda4bca6ae829a91440d5b3534dabbac0e8f1d657ea1cbc87693cb`
