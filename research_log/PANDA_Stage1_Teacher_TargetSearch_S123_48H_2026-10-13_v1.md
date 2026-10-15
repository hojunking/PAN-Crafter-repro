# PANDA MethodPlan-v1.2 — Stage 1 Teacher 고성능 탐색 / 48시간 실험계획

**작성일:** 2026-10-13  
**실험계획 버전:** `S1-Teacher48H-v1.0`  
**Campaign ID:** `PANDA_S1_M12_TEACHER48H_S123_20261013_v1`  
**범위:** 실험 case·탐색 순서·평가·보존·기록 방식. 모델/runner/uploader 구현은 실행 서버 담당.  
**현재 상태:** `PLAN_ONLY`. 본 문서 작성으로 서버가 기동되거나 시트가 변경된 것은 아니다.

> **최우선 목표:** 새 방법의 **Stage 1 전체 구성**을 유지하면서, 각 데이터셋에서 **같은 Teacher+Aligner checkpoint가 HQNR 하한과 ERGAS 상한을 동시에 만족**하도록 탐색한다. 학습 파라미터 파일과 정성 산출물은 그 조건을 통과할 때만 영구 보존한다. 미달·중단·실패 실험의 수치와 설정은 전부 로컬에 기록하고 시트에 업로드한다.
>
> 이 문서는 기존 TA2의 52조건×3회 반복 계획을 그대로 이어가는 문서가 아니다. **새 Stage 1의 48시간 성능 탐색 캠페인**이다. 과거 TA2의 loss, Aligner, donor, 서버 매핑을 그대로 재사용하지 않는다.

---

## 0. 실행 담당자가 먼저 볼 결정사항

| 항목 | 이번 실행 규칙 |
|---|---|
| 서버 | **s1=WV3 / s2=QB / s3=GF2**. s4·s5는 대상 아님 |
| 학습 대상 | 새 Stage 1의 PAN Aligner `A` + Teacher reconstruction U-Net `T` |
| Stage 2 | Student 학습, KD, `e/q` fitting 및 calibration 모두 제외 |
| 방법 기준 | 첨부 `PANDA_Method_Implementation_Validation_Plan_2026-10-13_v1_2.md` |
| 변경 허용 | loss의 **양수 scalar 계수**, A/U 학습률, weight decay, 학습 길이·scheduler 예산 |
| 변경 금지 | scene prior 의미/부호를 성능에 맞춰 변조, U-Net 구조/입력/base 변경, loss 종류 변경, FR test 학습, ε radius·주기 변경 |
| 기준 objective | `L_rec_RR + λ_abs·L_abs_RR + λ_eps·(L_eps_RR + L_eps_FR)` |
| 시작 계수 | `λ_abs=1`, `λ_eps=1e-3`; **RR와 FR에 같은 λ_eps** |
| ε | 반경 2 current PAN pixels의 면적 균등 원판, RR/FR 각각 매 update |
| 시간 | 서버별 독립 **48시간**. 실제 campaign 시작 후 preflight·학습·평가·저장 포함 |
| 후보 보존 | **두 목표를 동시에 통과한 동일 A/T snapshot만 저장** |
| 미달 가중치 | 영구 보존하지 않음. strict 기본 모드에서는 `last/resume/temp_eval`도 디스크에 쓰지 않음 |
| 기록 | 미달 포함 **모든 공식 평가 수치·설정·선정/중단 사유는 로컬 우선 기록** |
| 시트 | 기존 `analysis`에 별도 campaign으로 append. 시트 실패는 학습을 멈추지 않음 |
| 반복 | 탐색은 고정 R1, 우수 recipe는 R2/R3로 확인. **모든 후보를 3회 수행하는 예산은 아님** |
| 실행 방식 | 서버 간 대기·공유 selection lock 없음. 서버 내부 중복 실행 방지만 허용 |

### 0.1 서버 재배치 주의

이전 TA2는 `s1/WV3, s3/QB, s5/GF2`였다. **이번 문서는 `s1/WV3, s2/QB, s3/GF2`로 배정**한다. s2/s3의 데이터 경로와 evaluator를 새 배정에 맞춰 확인하며, 과거 서버 이름에서 데이터셋을 추정하지 않는다. 기존 다른 작업의 중단·파일 이동은 실제 실행 담당자가 확인한 범위에서만 수행한다. 본 문서는 전역 kill, 기존 결과 삭제 또는 s4/s5 작업 변경을 지시하지 않는다.

---

## 1. 근거와 이번 계획의 변경 권한

### 1.1 방법론의 기준 자료

- 기준 파일: `PANDA_Method_Implementation_Validation_Plan_2026-10-13_v1_2.md`
- SHA256: `7a4408e3294db467e3f503cfe6cf2b5d3a8a035267923b2530f825b7a3bd6f2e`
- 핵심 참조: §0·§3(scene prior), §4·§8(Aligner/공통 ResBlock/U 유지), §6–7(RR/FR ε), §9–10(absolute gradient L1/Stage 1 objective), §16·§18(수치 고정 및 preflight).
- 이 자료는 설계·구현검증 명세다. **현재 서버 구현이 완료됐다는 증거가 아니므로**, 서버의 실제 resolved implementation과 함께 pin해야 한다.

### 1.2 무엇을 새로 제안하는가

사용자의 이번 지시에 따라 **성능 탐색용 scalar tuning과 48시간 운영 규칙**을 구체화한다. 원자료의 `λ_eps=1e-3`, `λ_abs=1`은 F00의 시작점으로 유지하고, 다른 값은 **본 캠페인의 명시적 튜닝 조건**이다. 센서별로 우수한 scalar가 다를 수 있다. 이는 원자료의 기본값을 이미 최적이라고 주장하거나 몰래 교체하는 것이 아니다.

단, scalar tuning 허용을 **새 loss/branch/architecture 허용**으로 해석하지 않는다. 모든 F/T/H 후보는 full Stage 1이다. ε를 0으로 만들거나 FR branch를 없애는 방법 제거 ablation은 이번 고성능 탐색의 후보에 넣지 않는다.

### 1.3 과거 결과를 사용하는 범위

2026-10-13 TA2 집계에서는 QB의 강한 정합 조건에서 후기 FR 악화가 반복됐고, GF2에서는 ε와 구조 지도를 함께 조절한 결과가 상대적으로 유망했다. 이 관측은 **시도 순서의 근거**로만 사용한다.

이번에는 **scene direction+magnitude prior, residual head, Aligner 공통 LN ResBlock, FR ε branch, RGB absolute-gradient L1**이 들어가므로 TA2와 다른 방법이다. 과거의 NCC/NGF `λ_struct=0.1`과 이번 `λ_abs=0.1`의 의미나 gradient scale은 동일하지 않다. 과거 점수를 새 방법의 성능으로 간주하거나 TA2 가중치를 새 A에 조용히 이식하지 않는다. 목표 달성은 미검증이며 48시간 내 달성을 보장하지 않는다.

---

## 2. 성능 목표와 Teacher 선정 규칙

### 2.1 사용자 목표 — 변경 금지

| 서버 | 데이터셋 | HQNR 하한 `H*` | ERGAS 상한 `E*` | 통과 조건 |
|---|---|---:|---:|---|
| s1 | WV3 | **0.958** | **2.040** | `HQNR >= 0.958 AND ERGAS <= 2.040` |
| s2 | QB | **0.923** | **3.500** | `HQNR >= 0.923 AND ERGAS <= 3.500` |
| s3 | GF2 | **0.965** | **0.522** | `HQNR >= 0.965 AND ERGAS <= 0.522` |

**반올림 전 full-precision 값**으로 판단한다. `HQNR=0.9579999`를 0.958로 표시해 통과시키거나, `ERGAS=0.5220001`을 0.522로 표시해 통과시키지 않는다. 목표의 `<`/`>`를 사용자가 지정한 `<=`/`>=`와 혼동하지 않는다.

### 2.2 ‘같은 checkpoint’의 의미

다음이 모두 같은 상태여야 한다.

1. `A_state`와 `T_state`의 동일한 snapshot.
2. 같은 `run_id / attempt / seed / completed_update`.
3. 같은 논리적 inference policy와 고정 evaluator revision.
4. RR/FR 각각의 올바른 grid에 변환된, 동일한 규칙의 scene prior. **RR/FR prior의 숫자가 같아야 한다는 뜻은 아니다.**
5. 각 데이터셋의 고정 RR/FR 평가 목록 전체를 완료하고, NaN/누락/중복 장면이 없는 결과.

HQNR는 12K 모델, ERGAS는 50K 모델에서 가져오는 조합은 금지한다. scene마다 다른 checkpoint를 선택하거나, FR/RR에서 서로 다른 모델·임의 correction 배율을 쓰는 것도 금지한다. 정상적인 좌표·prior 단위 변환은 그대로 수행한다.

### 2.3 최고 Teacher 선택

**영구 보존 자격:** 위 AND 조건을 통과한 checkpoint 전부.

**최종 대표 선택:** 통과한 후보 중 `HQNR 내림차순 → ERGAS 오름차순 → step 오름차순 → run_id`의 순서. 앞 순위가 exact tie인 경우에만 다음 항목을 쓴다. ERGAS 상한이 충족된 뒤 HQNR를 우선하는 **제약식 기반 선정**이다.

별도로 통과 후보 중 ERGAS 최소인 RR 중심 예비 Teacher도 표시한다. 대표가 아니라는 이유로 이미 통과해 저장한 checkpoint를 자동 삭제하지 않는다. 같은 A/T state hash의 파일은 동일본으로 연결할 수 있지만 선택 이유와 평가 기록은 각각 보존한다.

**미달 후보:** 점수가 가까워도 영구 weight 파일은 없다. `BEST_OBSERVED_NOT_QUALIFIED`처럼 **수치상 근접 후보**로만 기록한다. 성공이 없으면 `NO_TARGET_QUALIFIED_TEACHER`가 정식 결과이며, 목표를 완화하지 않는다.

### 2.4 이 평가의 연구적 범위

기존 benchmark RR/FR 결과로 목표를 판정하고 hyperparameter를 고르므로 **test-aware 개발·모델 선택 실험**이다. FR test 영상에 gradient를 흘리는 것은 금지하지만, 그 사실만으로 test-independent라고 부를 수는 없다. 추후 논문에서 독립 검증을 제시하려면 별도 validation/holdout protocol이 필요하다.

이번 gate는 **저장·개발 목표**이지 기하학의 물리적 정확도 인증, 전체 seed의 성공 또는 statistically significant improvement를 의미하지 않는다.

---

## 3. 모든 성능 후보에서 유지할 새 Stage 1

### 3.1 Scene prior + patch residual

\[
\mathbf c_{s,i}=\mathbf b_s+\alpha_{s,i}\mathbf d_s+\beta_{s,i}\mathbf n_s.
\]

- `b_s`는 source scene의 고정 **방향+크기** prior. GT/test score로 튜닝하는 상수가 아니다.
- `d_s,n_s`는 prior에 따른 좌표 basis. `α,β`는 patch당 residual의 두 성분이며, loss의 coefficient와 다른 변수다.
- 원본 PAN에 **총 correction c로 한 번 warp**한다. 먼저 prior로 warp하고 residual로 다시 warp하는 구성으로 바꾸지 않는다.
- MS/GT/reference frame은 유지하고 PAN만 MS frame으로 이동한다.
- ε branch에서도 prior는 native branch와 동일하다. ε를 만들었다고 prior를 다시 추정하거나 `b_s-ε`를 제공하지 않는다. 그러면 알려진 이동을 A가 배워야 한다는 검정이 사라진다.
- source-scene prior의 픽셀 단위·부호·위상·augmentation 변환은 데이터 근거로 결정한다. `×0.25`, `×4`, shrink factor를 HQNR에 맞춰 적용하지 않는다.

### 3.2 Architecture / 입력

| 구성 | 유지 조건 |
|---|---|
| A 입력 | PAN + **native LMS**; 별도 aligner 입력 정규화는 pin된 구현 유지 |
| A ResBlock | **기존 U-Net과 동일한 64ch common ResBlock**, channel LN/SiLU/Conv×2/second Conv zero-init/no post-add activation |
| A stem/joint | 기존 GN 유지. Aligner 전체를 LN으로 바꾸는 것이 아님 |
| A head | 서버에서 검증한 scene-conditioned residual head를 pin. source의 66→32→2는 구현 제안이므로 실제 채택 차원을 manifest에 확정 |
| T U-Net | 현재 승인된 Teacher width/depth/body/input layout 유지. source의 112/(1,2,3)은 concrete candidate이지 임의 교체 허가가 아님 |
| T MS base | **현재 승인된 reconstruction base 유지**. A를 native LMS로 바꿨다는 이유로 T까지 자동 변경하지 않음 |
| 초기화 | 새 A와 T를 fresh 초기화. old A GN checkpoint의 묵시적 migration 금지 |
| prior 초기값 | zero-init residual head에서 `c0=b_s`가 성립해야 함 |

데이터셋의 밴드 수에 따른 입력·출력 channel 차이는 허용되지만, 나머지 U/A 구조는 같은 method revision으로 유지한다. 본 48시간 탐색에는 U-Net width/depth, attention, LPAN/HPAN 추가, PAN reconstruction head, PANMIX, EMA/SWA, ensemble/TTA 탐색을 넣지 않는다.

### 3.3 Loss

\[
L=L_{rec}^{RR}+\lambda_{abs}L_{abs}^{RR}
+\lambda_{\epsilon}\big(L_{\epsilon}^{RR}+L_{\epsilon}^{FR}\big).
\]

`L_abs`는 **native corrected PAN ↔ GT HRMS의 RGB 단순 평균 luminance**, 고정 절댓값 2-D gradient 사이의 L1이다. 기존 TA2 NCC/NGF를 이름만 바꿔서 사용하지 않는다. derivative kernel, scaling, per-axis/magnitude 정의, normalization, mask/ROI 및 denominator는 서버가 검증하여 **한 번 고정**한다. 학습 후보별로 loss kernel이나 radiometry를 바꾸지 않는다.

`L_eps = mean_batch,xy |c_eps + eps - stopgrad(c0)|`. RR와 FR의 ε loss는 **각 branch 내부 mean을 구한 뒤 더한다**. RR/FR sample을 이어 붙여 한 번 평균하거나 두 branch 평균에 다시 1/2를 곱하지 않는다. sample 수가 다르다는 이유로 효과적인 FR 계수가 바뀌지 않도록 gradient accumulation을 검증한다.

- `L_rec`: A와 T 모두 update.
- `L_abs`: A만 직접 update.
- `L_eps_RR`, `L_eps_FR`: 같은 A만 update.
- shifted PAN은 T에 넣지 않는다. FR에서는 reconstruction/GT absolute loss를 만들지 않는다.
- ε는 RR/FR 각각 radius 2, 면적 균등 원판, 독립 stream, 매 update. **ε ramp·격 update·sensor별 radius·L_scale는 추가하지 않는다.**

---

## 4. 시작 전 blocking gate — 구현 담당자가 확인할 최소 항목

### 4.1 데이터와 source scene

다음 중 하나라도 미확인이라면 `BLOCKED_DATA_PROTOCOL`로 기록하고 해당 full-method training을 시작하지 않는다.

- RR patch → 실제 source scene ID/crop/augmentation 연결.
- FR ε 학습용 **training-only native PAN/LRMS/LMS**의 존재와 provenance.
- FR/RR 평가 scene 및 근접·중복 crop이 학습에 섞이지 않는 split 검증.
- native LMS와 T base의 생성 경로, band 순서/RGB index, DN scaling.
- prior estimator의 sign/unit/phase 변환과 RR/FR별 prior-cache digest.
- zero/low-confidence prior의 처리 정책. 미확정이면 임의 `b=0,d=(1,0)`로 조용히 대체하지 않는다.

**금지:** source scene이 없는데 patch index를 scene ID로 재명명, FR training data가 없는데 FR20 test를 사용, RR 확대본을 native FR training이라 기재, prior estimator가 실패한 scene을 test metric 계산에서 제외.

Low-confidence scene을 어떻게 다룰지는 첨부 명세에서 미정이다. 서버가 결정한 정책은 별도 implementation decision으로 기록·검증하고 탐색 전에 고정한다. 다른 정책을 쓰면 `implementation_revision`을 나누며 결과를 같은 scalar sweep으로 합치지 않는다.

### 4.2 방법과 gradient

- `c0=b_s` 초기값, known shift 부호, 모든 augment의 vector 변환.
- source에서 요구한 common ResBlock 교체가 실제 수행됐는지.
- RR/FR shared A, native LMS ref와 T base 분리.
- `L_rec / L_abs / L_eps_RR / L_eps_FR` 각각의 backward 대상.
- ε target에만 stop-gradient; rec/abs의 c0는 detach하지 않음.
- `L_abs` 및 큰 prior의 유효 support 처리. empty ROI·NaN을 0 loss로 숨기지 않음.
- full-resolution inference가 임의 resize/shift multiplier를 몰래 적용하지 않는지.

### 4.3 반드시 pin할 실행값

서버가 이미 승인한 Teacher training recipe의 다음 값을 `baseline_resolved.json`에 확정한다. **이 계획은 미확인 값을 현재 서버 설정인 것처럼 만들어 넣지 않는다.**

`teacher_architecture`, `aligner_architecture`, `teacher_input_layout`, `teacher_base`, `optimizer`, `optimizer_betas/eps`, `lr_U0`, `lr_A0`, `weight_decay0`, `warmup_updates0`, `effective_batch_RR`, `effective_batch_FR`, `microbatch/accumulation`, `RR_train_HW`, `FR_train_HW`, `precision`, `augmentation`, `evaluator_revision`.

학습률 case는 이 **확정된 기준값의 배율**로 정의한다. 예를 들어 baseline이 `lr_U0=1e-4`, `lr_A0=1e-5`라면 T01은 A `3e-6`가 되지만, 이 숫자가 현 서버의 실제 설정이라고 가정하지 않는다. 기준값이 비어 있으면 실행 config가 완성된 것이 아니다.

FR batch/크기는 실제 native support와 VRAM을 확인해 정한다. RR:FR 샘플비는 source에서 확정하지 않았으므로 임의의 센서 간 공통 숫자를 이 문서에서 사실로 두지 않는다. **선정된 batch·shape 정책은 데이터셋 내 scalar 비교군 전체에서 고정**하고, 두 branch mean과 update 횟수는 동일하게 유지한다.

### 4.4 일회성 성능 측정

실제 시작 후 disposable smoke와 300–500 update 규모의 timing을 수행한다. 초기 compile/cache warmup은 분리한다. U/A가 오염되지 않도록 smoke 가중치를 본 실험 초기값으로 이어 쓰지 않는다.

측정: update latency의 median/P90, RR 전체 평가 시간, FR 전체 평가 시간, VRAM peak, native FR A branch의 추가 시간, qualifying checkpoint 저장/정성 export 예상 시간, 로컬 여유 공간.

**기존 TA2의 sec/step이나 과거 ‘5090 서버’ 명칭을 이번 서버의 속도로 대신하지 않는다.**

---

## 5. 48시간 운영 — 탐색 가능한 수만 실행

### 5.1 서버별 시간표

`t0`는 구현 revision을 넘겨받아 **해당 서버 campaign을 실제 시작한 시각**이다. 구현 개발 자체를 이 문서에서 시작하거나 완료한다고 가정하지 않는다. t0 이후의 preflight·prior 준비·평가·기록·저장은 48시간에 포함한다.

| 경과 시간 | 작업 | 우선순위 |
|---|---|---|
| **0–2h** | Gate 확인, prior/cache 검증, timing, baseline pin | 정확성/throughput 확인 |
| **2–22h** | F00 + 센서별 우선 loss 조합 탐색 | full method를 유지한 후보 확보 |
| **22–36h** | 상위 recipe의 LR/decay 조정 또는 긴 학습 | 현재 부족한 지표 개선 |
| **36–46h** | 통과 recipe의 R2/R3 재현, 또는 가장 근접한 recipe 추가 확인 | 최고 Teacher와 반복성 |
| **46–48h** | 새 학습 금지, 마지막 평가·통과 모델/정성 검증·기록 정리 | deadline 내 마감 |

이 배분은 운영 제안이다. phase 경계에서는 **현재 살아 있는 run의 완료 예상 시간과 남은 확인 예산**을 보고 다음 작업을 선택한다. 경계를 맞추려고 미달 checkpoint를 디스크에 임시 저장해 pause하지 않는다. 46시간 전까지의 live run도 마지막 평가/저장에 필요한 여유를 남기고 종료한다.

각 서버는 자신의 t0와 로컬 결과만 사용한다. 다른 데이터셋의 결과를 기다리지 않는다. 네트워크 지연도 다른 서버의 실험을 막지 않는다.

### 5.2 고정 ‘N개 완료’를 약속하지 않는 이유

새 FR ε branch와 prior 준비 비용은 아직 실제 측정 전이다. 16개 loss 조건과 9개 optimizer 조건을 **모든 서버에서 3회씩 전부 끝내는 계획이 아니다.** 아래 registry는 후보 목록이고, 48시간 안에서 우선순위에 따라 가능한 subset을 수행한다.

실행마다 다음 보수적 예상치를 갱신한다.

`ETA = 남은 update 수 × 관측 update P90 + 예정 RR/FR 전체 평가 시간 + 필요한 저장/정성 시간`.

deadline과 마감 예산 안에 들어오는 작업만 시작한다. 새로운 후보를 시작하려면 최소 25K 판단점까지 도달할 예상 예산이 있어야 한다. baseline조차 매우 느리면 후보 수를 줄이지, FR branch·평가 장면·방법 구성을 몰래 줄이지 않는다.

완료 예상이 불확실하면 안전 여유 20%를 더하는 것을 운영 기본으로 한다. 이것은 성능 gate의 tolerance가 아니라 **시간 계획용 여유**다.

### 5.3 지연/부재의 처리

- 준비 2시간 안에 blocking gate가 해결되지 않으면 해결되지 않은 항목을 즉시 기록. 검증 없이 training으로 넘어가지 않는다.
- 구현 변경이 필요하면 새 source/config hash와 그 이유를 남긴다. 변경 전·후를 같은 run으로 이어 붙이지 않는다.
- 48시간이 끝나면 추가 run을 자동 생성하거나 ‘성공할 때까지’ seed를 바꾸지 않는다.
- 최종 report에 `PLANNED / STARTED / COMPLETE / PRUNED / FAILED / NOT_STARTED_BUDGET / BLOCKED`를 모두 구분한다.

---

## 6. Phase A — loss coefficient 후보 16개

모든 조건은 **scene prior+residual, native LMS A, RR rec+absolute, RR ε, FR ε**를 갖춘 full Stage 1이다. `λ_rec=1`, RR와 FR의 λ_eps는 같은 값, 다른 모든 설정은 baseline과 동일하다.

| Case | `λ_eps` — RR/FR 공통 | `λ_abs` | 변경 목적 |
|---|---:|---:|---|
| **F00** | 0.001 | 1 | MethodPlan-v1.2 기본 계수 |
| **F01** | 0.0003 | 1 | ε 0.3배 |
| **F02** | 0.0001 | 1 | ε 0.1배 |
| **F03** | 0.003 | 1 | ε 3배 |
| **F04** | 0.001 | 0.3 | absolute 구조항 0.3배 |
| **F05** | 0.001 | 0.1 | absolute 구조항 0.1배 |
| **F06** | 0.001 | 3 | absolute 구조항 3배 |
| **F07** | 0.0003 | 0.3 | 약한 ε × 약한 구조항 |
| **F08** | 0.0001 | 0.3 | 더 약한 ε × 약한 구조항 |
| **F09** | 0.003 | 0.3 | 강한 ε × 약한 구조항 |
| **F10** | 0.0003 | 0.1 | 약한 ε × 매우 약한 구조항 |
| **F11** | 0.0001 | 0.1 | 두 항 모두 0.1배 |
| **F12** | 0.003 | 0.1 | 강한 ε × 매우 약한 구조항 |
| **F13** | 0.0003 | 3 | 약한 ε × 강한 구조항 |
| **F14** | 0.0001 | 3 | 매우 약한 ε × 강한 구조항 |
| **F15** | 0.003 | 3 | 두 항 모두 3배 |

**그리드 전체를 돌리는 것보다 F00과 데이터셋별 첫 6개를 우선**한다. scalar가 작아도 두 ε branch와 absolute branch는 매 update 그대로 계산한다. 0 계수, branch bypass 또는 radius 변경은 없다.

### 6.1 서버별 탐색 순서

| 서버 | 우선 6개 — 시간 허용 시 순서대로 | 남는 시간의 후보 순서 |
|---|---|---|
| s1 | F00 → F01 → F04 → F07 → F02 → F06 | F08 → F05 → F10 → F11 → F13 → F14 → F03 → F09 → F12 → F15 |
| s2 | F00 → F02 → F08 → F11 → F01 → F07 | F04 → F10 → F05 → F06 → F14 → F13 → F03 → F09 → F12 → F15 |
| s3 | F00 → F06 → F13 → F01 → F14 → F04 | F07 → F02 → F08 → F05 → F10 → F11 → F03 → F15 → F09 → F12 |

- **WV3:** 현재 반복 근거가 적으므로 F00와 단일축 완화 비교부터 확보한다. 과거 한 seed의 순위를 최적 조건으로 고정하지 않는다.
- **QB:** 과거 강한 ε의 FR 악화를 고려해 약한 ε 조합을 빨리 검정한다. 다만 새 방법에서는 FR ε와 prior가 추가됐으므로, 약한 값의 우위를 미리 확정하지 않는다.
- **GF2:** λ_abs를 높이거나 ε와 균형을 바꾸는 후보를 우선한다. 과거 G05 숫자를 새 absolute-gradient loss에 복사했다는 의미는 아니다.

### 6.2 공정한 기준 run

F00/R1은 원칙적으로 50K까지 실행하고 모든 지정 평가를 남긴다. 예외는 수치/데이터 실패 또는 명확한 시간 상한뿐이다. 다른 F 조건도 **같은 초기화 seed·data/ε stream·학습 스케줄**로 fresh 시작한다. 앞 case의 학습된 파라미터를 다음 case로 넘기지 않는다.

λ_abs 변화가 단순히 A의 gradient를 지배하는지 확인하도록 raw/weighted gradient를 따로 기록한다. 학습 중 자동 loss normalization, adaptive weighting, 새 clipping을 넣어 계수 차이를 상쇄하지 않는다.

---

## 7. Phase B — 상위 full-method recipe의 optimizer 튜닝

### 7.1 Anchor 선정

Phase A에서 **최대 2개 recipe**를 고른다.

1. 통과 checkpoint가 있으면 그중 최선의 recipe.
2. 두 목표를 동시에 만족하지 못했으면 §10의 동일-checkpoint target-gap이 가장 작은 recipe.
3. 두 번째 recipe는 다른 trade-off를 대표하는 후보로 선택할 수 있다. 단, 후보/선정점/이유를 숫자와 함께 기록한다.

Anchor는 **hyperparameter 설정**이지 미달 weight checkpoint가 아니다. 미달 anchor는 config와 seed만 재사용하여 fresh 학습한다. 이미 멈춘 미달 가중치를 로드하는 다단계 탐색은 쓰지 않는다.

### 7.2 튜닝 후보

아래 배율은 **baseline_resolved의 lr/decay**에 적용한다. Anchor에서 F loss 계수만 상속한다. T 조합을 연달아 수행한다고 배율을 누적해서 곱하지 않는다.

| Case | `lr_A/lr_A0` | `lr_U/lr_U0` | `WD/WD0` | 검정 목적 |
|---|---:|---:|---:|---|
| **T00** | ×0.1 | ×1 | ×1 | A 업데이트를 크게 완화 |
| **T01** | ×0.3 | ×1 | ×1 | A 업데이트를 완만하게 완화 |
| **T02** | ×3 | ×1 | ×1 | A 학습이 부족한 경우의 반대 대조 |
| **T03** | ×1 | ×0.5 | ×1 | U learning rate 감소 |
| **T04** | ×1 | ×1.5 | ×1 | U learning rate 증가 |
| **T05** | ×1 | ×1 | ×0.1 | weight decay 감소 |
| **T06** | ×1 | ×1 | ×0 | weight decay 제거 |
| **T07** | ×0.3 | ×1.5 | ×1 | 느린 A + 빠른 U |
| **T08** | ×0.1 | ×0.5 | ×1 | 매우 느린 A + 느린 U |

- `WD0=0`인 경우 T05/T06는 중복 조건이므로 실행하지 않고 `SKIP_DUPLICATE_CONFIG`로 기록한다.
- optimizer 종류·betas·eps·warmup 규칙은 그대로 두고, 이 표에 적힌 항목만 바꾼다.
- A의 LR 감소는 **A freeze가 아니다.** 모든 loss 경로와 업데이트가 유지된다.
- 데이터 batch/augmentation·T base·출력 clipping·evaluation은 바꾸지 않는다.

### 7.3 센서별 우선 튜닝

| 데이터셋 | 우선 순서 | 배정 의도 |
|---|---|---|
| WV3 | T01 → T03 → T00 → T07 | A/U 업데이트 속도에 따른 후기 품질 균형 |
| QB | T00 → T01 → T07 → T03 | residual A가 초기 prior에서 과도하게 변하는지, U와 학습 속도 불균형인지 |
| GF2 | T04 → T07 → T05 → T01 | RR 오차를 더 줄이면서 FR 목표를 유지하는 조합 |

이는 가설에 따른 시도 순서다. 실제 우위나 실패 원인을 미리 단정하지 않는다. 48시간 내 모든 T case를 돌리기보다, 한 anchor의 핵심 비교를 끝낸 뒤 추가 조건을 고려한다.

---

## 8. 학습 길이: 50K 기본, 100K 우선 확장, 150K 조건부

| Schedule ID | 목표 optimizer updates | 용도 | 초기화/비교 주의 |
|---|---:|---|---|
| **H50** | **50,000** | 모든 F/T 기본 비교 | fresh; cosine horizon 50K |
| **H100** | **100,000** | 우수 recipe의 RR 추가 수렴 확인 | **fresh; 처음부터 horizon 100K** |
| **H150** | **150,000** | 속도·잔여 시간에 여유가 있고 H100 경향이 유망할 때 | fresh; 처음부터 horizon 150K |

Scheduler family는 이번 실행 제안으로 **고정 warmup 후 cosine**을 사용한다. warmup update 수와 최소 LR은 서버의 승인된 기준값을 pin한다. 기존 승인 recipe가 다른 family라면 새 baseline을 그 recipe로 명시적으로 확정한 뒤 전 조건을 동일하게 적용한다. 서로 다른 scheduler를 같은 H50으로 기재하지 않는다.

**H50의 종료 weight에 horizon을 100K로 바꾸고 fresh100K라 부르지 않는다.** H100의 50K 결과도 H50의 최종점과 scheduler 이력이 다르다. 각각 기록한다.

미달 checkpoint를 저장하지 않으므로, H100/H150은 종료된 미달 H50을 불러오는 연장학습이 아니라 **선정 recipe로 다시 시작하는 긴 학습**이다. 처음부터 H100으로 등록한 live run을 25K/50K에서 계속 진행하는 것은 가능하다. 파라미터는 메모리에 그대로 있고, horizon은 바꾸지 않는다.

### 8.1 긴 학습을 우선할 조건

- 한 목표는 충족했고 다른 지표가 후반까지 개선 중이며, 반대 지표가 지속적으로 무너지지 않는 경우.
- RR 정확도가 병목인 GF2에서는 H100을 우선 고려하되, HQNR가 초반부터 지속 하락하면 단순 장기학습으로 해결된다고 가정하지 않는다.
- QB/WV3에서 후기 HQNR 악화가 재현되면 H100보다 먼저 약한 ε/느린 A 후보를 본다.
- actual ETA가 남은 시간을 넘으면 시작하지 않는다. H150는 필수 수행항목이 아니다.

이번 캠페인에는 schedule 중간의 A freeze, ε ramp, term 제거, 기존 checkpoint 기반 임의 fine-tuning을 넣지 않는다. 그런 방식이 필요하면 별도 method-revision 실험으로 제안하고 다음 예산에서 다룬다.

---

## 9. Seed와 반복 방식

### 9.1 고정 seed registry — 이번 계획에서 새로 배정

| 데이터셋 / 서버 | R1 — 탐색 | R2 — 확인 | R3 — 확인 |
|---|---:|---:|---:|
| WV3 / s1 | 261013101 | 261013102 | 261013103 |
| QB / s2 | 261013201 | 261013202 | 261013203 |
| GF2 / s3 | 261013301 | 261013302 | 261013303 |

진단 전용 probe seed는 `261013900`을 사용하며 training/data/ε stream과 분리한다. 각 네트워크와 stream의 RNG seed 파생 규칙을 manifest에 남긴다.

- 같은 데이터셋의 F/T 탐색은 R1을 공유하고 A/T를 각각 독립 RNG로 초기화한다.
- initial A/T는 **RAM에서 생성 후 hash만 기록**한다. 미달 초기 weight를 디스크 snapshot으로 저장하지 않는다.
- 같은 R1이라고 데이터 순서·augmentation·RR/FR ε stream이 달라지지 않게 정의한다. term 강도와 무관하게 각 stream 소비를 고정한다.
- 기술적 재시도는 같은 run/seed의 새 `attempt`다. 독립 seed 한 번으로 추가 집계하지 않는다.

### 9.2 3회 반복의 범위

이번 목적은 **48시간 안에서 최고 Teacher 확보**다. 따라서 모든 F/T 조합을 3회 반복하지 않고, 우수 **recipe 1개를 우선 R1/R2/R3로 확인**한다. 시간과 throughput이 충분하면 2번째 recipe도 반복한다.

통과한 seed만 평균에 넣지 않는다. 동일 recipe의 미달 seed도 성능 평균·표준편차·목표 달성률의 분모에 포함한다. 가중치를 저장하지 않는 것과 통계에서 제외하는 것은 다른 결정이다.

R2/R3가 미달해도 다른 seed로 교체해 ‘3/3 성공’을 만들지 않는다. 시간상 R1만 끝났으면 `n=1`, R2까지면 `n=2`로 보고한다. 반복 전용 run은 수치/시간 실패 외에는 예정 endpoint까지 진행해 seed 간 비교를 확보하는 것을 우선한다.

---

## 10. 실시간 평가, 후보 순위, 불필요한 학습의 조기 중단

### 10.1 평가 cadence — 새 캠페인 전용

공식 RR/FR 전체 평가는 다음 update 집합에서 수행한다.

`{1,000} ∪ {2,000, 4,000, ..., horizon} ∪ {25,000} ∪ {등록 horizon의 정확한 endpoint}`.

중복 step은 한 번만 평가한다. H50은 27개, H100은 52개, H150은 77개 예정점이다. 조기 중단·deadline 마지막 상태는 필요할 때 `PRUNED_LAST` 또는 `TIMEBOX_LAST`로 별도 평가하며 실제 count에 포함한다.

**이 계획은 과거 TA2의 50개 grid와 다르다.** `HQNR_MAX50` 라벨을 재사용하지 않는다. 새 label은 `TARGET_QUALIFIED`, `HQNR_MAX_OBSERVED_NUMERIC`, `EXACT_FINAL`, `TIMEBOX_LAST` 등이다. 관측 최대값의 후보 개수를 모두 보고한다.

### 10.2 같은 A/T snapshot의 RR/FR 평가

기본 실행은 다음 순서다.

1. optimizer update 경계에서 학습을 잠시 멈춘다.
2. 현재 A/T의 canonical in-memory state hash를 계산하고 `eval mode`로 고정한다.
3. RR와 FR 전체 목록을 그 상태로 평가한다. 추론 결과는 장면 단위로 수치만 집계하고 버린다.
4. metric completeness, finite 값, evaluator/input digest를 확인한다.
5. 로컬 numeric record를 먼저 durable하게 쓴다.
6. 두 gate를 모두 통과했을 때만 weight 파일을 쓰고, 이어서 정성 export를 수행한다.
7. mode/RNG를 복원하고 다음 update로 진행한다.

공식 평가를 async worker가 수행하려면 동일 snapshot을 RAM에 보유하는 구현이 필요하다. **미달 여부를 확인하기 위해 모든 weight를 디스크 큐에 먼저 저장하는 구조는 이 정책과 맞지 않는다.** 메모리 비용과 구현 복잡도를 줄이려면 위의 synchronous in-memory 평가를 사용한다.

### 10.3 미달 후보의 탐색용 target-gap

미달 checkpoint도 다음 **검색용 수치**를 기록한다.

\[
v_H=\max(0,(H^*-H)/H^*),\quad
v_E=\max(0,(E-E^*)/E^*),\quad
D_\infty=\max(v_H,v_E),\quad D_1=v_H+v_E.
\]

- 통과 후보가 있으면 §2.3의 제약식 순위 사용.
- 미달 후보는 `D_inf 오름차순 → D1 오름차순 → HQNR 내림차순 → ERGAS 오름차순`으로 탐색 우선순위 설정.
- **한 checkpoint의 H/E로만** 계산. 서로 다른 step의 최고 H와 최저 E를 섞지 않는다.
- 이 점수는 새 학습 loss나 과학적 유의성 척도가 아니며, gate를 완화하지 않는다.
- 별도 Pareto 수치 목록도 남겨 한쪽 지표 개선 방향을 놓치지 않는다. Pareto 미달 모델도 weight는 저장하지 않는다.

### 10.4 조기 중단은 ‘목표 미달’ 자체와 다르다

**저장 gate에 미달했다고 그 즉시 training을 중단하지 않는다.** 초기에는 ERGAS가 높아도 후반에 목표에 접근할 수 있다.

기본 운영:

- F00는 예정 50K를 우선 완결한다.
- 다른 탐색 후보는 원칙적으로 **25K까지 관측**한다.
- 25K 이후 탐색 후보에 한해, 동일 dataset/schedule/R1의 앞선 비교군 **2개 이상**이 공통 평가 step 3개에서 HQNR≥현재·ERGAS≤현재로 일관되게 우세하고, 현재 run의 target-gap이 초기 10K 이후 줄지 않는 경우 `PRUNED_DOMINATED`로 중단할 수 있다.
- 두 비교군이 없거나 개선 추세가 남으면 성능상 pruning을 적용하지 않는다.
- 공통 평가점이 부족하면 판단하지 않는다. 서로 다른 horizon/seed의 좋은 점 하나로 중단시키지 않는다.
- 수치 오류·데이터 오류는 즉시 `FAILED_*`; deadline은 `TIMEBOX_*`로 분리한다.

이 pruning은 48시간 효율을 위한 **휴리스틱**이며 장기학습 실패를 증명하지 않는다. PRUNED run을 완결 50K와 같은 성능 ablation 행으로 평균내지 않는다. 미달 파라미터가 사라지므로, 재검토가 필요하면 설정/seed를 이용한 fresh 재실행만 가능하다.

### 10.5 Phase 간 선택과 live run

현재 RAM에 살아 있는 run은 예정 horizon까지 계속할 수 있다. 그러나 다음 case를 실행하기 위해 **미달 run을 저장해 나중에 복귀하는 successive-halving 방식은 사용하지 않는다.**

Phase B/C에서 이전의 미달 recipe를 다시 선택하면 처음부터 재학습한다. 이 재실행 비용까지 ETA에 포함한다. 논문용 반복은 가능하면 pruning 없이 완료하며, 시간으로 중단한 경우 그 사실을 명시한다.

---

## 11. 체크포인트 보존 정책 — STRICT_TARGET_ONLY

### 11.1 파일을 쓸 수 있는 조건

다음 논리 조건을 모두 만족해야 한다.

`full_method_eligible AND metrics_complete AND metrics_finite AND same_state_verified AND HQNR>=H* AND ERGAS<=E*`.

- `full_method_eligible`: 이 문서의 full Stage 1, source/data/protocol gates가 충족됨.
- `metrics_complete`: RR/FR 고정 평가 장면 전체가 평가됨.
- `same_state_verified`: A/T가 두 평가 사이에서 바뀌지 않음.

Gate 미달 또는 incomplete 상태에서는 checkpoint의 영구 파일을 쓰지 않는다. HQNR만 높거나 ERGAS만 낮은 파일도 저장하지 않는다. 디스크 임시 파일 후 삭제하는 동작도 기본 모드에서는 사용하지 않는다.

### 11.2 저장하지 않는 항목

- 미달 `model.pt`, `teacher.safetensors`, `aligner.safetensors`.
- 주기적 `last`, `best_hqnr`, `best_ergas`, optimizer resume snapshot에 포함된 미달 가중치.
- 미달 초기화 snapshot, 평가 대기용 temp weight, crash dump의 parameter tensor.
- hidden cache, 다른 서버 복사, 외장하드 이관으로 미달 weight를 우회 보존하는 행위.

**RAM의 현재 training state를 유지하는 것은 허용**한다. 다음 update를 위해 필요한 정상 학습 상태이지, 영구 보존 checkpoint가 아니다. configuration/seed/RNG 또는 sampler의 비모델 메타데이터와 수치 prior cache는 기록할 수 있다. 단 그것만으로 모델 파라미터나 exact resume를 복구할 수 있다고 설명하지 않는다.

### 11.3 운영상 trade-off

엄격한 미달 weight 미저장 조건에서는 프로세스/서버가 죽으면 **그 미달 run은 exact resume가 불가능**하다. 동일 config·seed로 fresh 재시도하고 새 attempt를 남긴다. GPU 비결정성까지 포함한 bitwise 동일 재현을 보장하지 않는다.

이를 피하려고 임의의 `rolling_resume.pt`를 추가하지 않는다. 운영용 1개 resume 파일도 사용자 지시의 예외이므로 **별도 승인이 생기기 전에는 OFF**다. 멈췄거나 deadline에 도달한 미달 모델을 `TIMEBOX_LAST` 이름으로 저장하지 않는다.

성능을 통과한 시점의 snapshot에는 **그 동일 시점**의 optimizer/scheduler/RNG state를 함께 저장할 수 있다. 이후 미달 상태의 optimizer/weights로 해당 파일을 갱신해서는 안 된다.

### 11.4 통과 모델 패키지

최소한 다음을 묶는다.

- A와 T의 동일 snapshot, canonical model-state hash와 실제 파일 SHA256.
- resolved architecture/input/base, source/environment/evaluator digests.
- `b_s` estimator·prior-cache·scene-map·grid conversion digest 및 필요한 inference prior 자산 참조.
- seed/attempt/step/schedule, loss coefficient와 optimizer 설정.
- 해당 checkpoint의 **RR/FR 전체 수치**와 목표 판정.
- qual exporter configuration과 결과 목록.
- `record_id`, `model_state_sha256`, `weights_file_sha256`, 저장 검증 결과.

state hash는 정렬된 tensor 이름·shape·dtype·bytes에 대한 logical hash, 파일 SHA는 저장 컨테이너 bytes의 hash로 구분한다. 서로 같은 값이어야 한다고 가정하지 않는다.

파일 저장은 qualified 상태에서만 atomic하게 수행하고 load/hash 검증을 한다. 저장 실패는 `TARGET_PASS_ARTIFACT_PENDING/FAILED`로 남기고 성공 보존이라고 표시하지 않는다. 새 패키지가 stage2에서 재사용 가능한지는 이 metadata와 A/T 동시 load를 확인하되, 이번 예산에 Stage 2 calibration/학습을 시작하지 않는다.

### 11.5 적용 범위

이 정책은 **새 campaign 디렉터리에서 생성할 모델**에만 적용한다. 기존 paper, TA2, ablation, PAN-Crafter 재현 가중치에 대한 삭제/정리 명령이 아니다. 과거 미달 결과를 소급 삭제하지 않는다.

---

## 12. 정성평가 — 수치 gate 통과 모델만

### 12.1 미달일 때

공식 metric 계산을 위한 transient prediction은 필요하지만, 계산 후 RAM에서 버린다. **PNG/JPEG/TIFF/MAT/NPY prediction, overlay, heatmap, TensorBoard image를 저장하지 않는다.**

미달 모델의 correction 평균·분산, loss, gradient norm, 장면별 scalar quality, source IDs 등 숫자 기록은 저장한다. 숫자 분석을 위해 이미지가 반드시 필요한 예외를 이 계획에서 자동 허용하지 않는다.

### 12.2 통과일 때

checkpoint 저장·identity 검증 후 **그 동일 snapshot으로** 정성 결과를 생성한다. 학습이 다시 진행된 모델로 이미지를 뽑아 이전 수치와 붙이지 않는다.

- RR/FR의 고정 평가 scene 전체의 reconstruction export를 우선 보존.
- RGB 시각화 band index, DN→display stretch, clipping, crop 좌표를 고정.
- RR은 GT와 동일 ROI의 예측/오차, FR은 PAN/LRMS 또는 native LMS와 예측을 표시.
- PAN native / corrected PAN / GT RGB 또는 LMS edge overlay를 같은 ROI·contrast로 비교 가능하게 저장.
- 대표 ROI는 결과가 나오기 전에 input/source만으로 정하거나, 이미 고정된 목록을 재사용. 좋은 scene만 골라낸 것처럼 전체 성능을 설명하지 않는다.
- 전 밴드 raw output의 저장 형식은 서버의 기존 exporter를 pin한다. 보기용 RGB만으로 전체 spectral 결과를 대체하지 않는다.

출력 전체가 통과했더라도 per-scene artifact 유무는 별도 문제다. ghost/blur/과도한 이동/spectral distortion을 확인할 수 있게 export한다. FR에서 존재하지 않는 HRMS GT를 만들어 비교하지 않는다.

새로 통과한 각 checkpoint는 개별 record를 남긴다. state가 동일한 alias는 한 번 export하고 여러 selector가 같은 산출물을 참조할 수 있다. 시간/저장 오류로 export가 끝나지 않으면 `QUAL_PENDING`으로 정직하게 표시하고 46–48h 마감에 처리한다.

---

## 13. 분석용 numeric 진단 — 모든 후보에 최소량, 우수 후보에 확장

목표는 성능 높은 Teacher 확보지만, 실패 원인을 시트 숫자로 확인할 수 있어야 한다. 대규모 TA2 full-diagnostic을 매 평가점마다 반복하지 않는다.

### 13.1 모든 update/주기 로그

- RR `L_rec`, `L_abs`, `L_eps_RR`; FR `L_eps_FR`의 raw/weighted 값.
- U/A learning rate, effective batch와 RR/FR branch 발생 횟수.
- `||b_s||`, `||residual||`, `||c||` 요약; `α,β` 요약.
- finite 비율, warp support 비율, low-confidence scene count, empty-ROI count.
- updates/sec, GPU memory, data wait, eval/train/qual 각 시간.

update별 자세한 기록은 로컬에 남기고, 중앙에는 일정 구간 평균과 상태 전이만 보낸다. 시트에 batch마다 STATUS 행을 쌓지 않는다.

### 13.2 고정 probe 시점

기본 `10K / 25K / 50K`, 긴 run은 `100K / 150K` endpoint에 숫자 진단을 한다. 같은 checkpoint가 통과하면 저장된 모델에서 추가 계산할 수 있다. 미달 모델은 live state에서 처리하며 나중 재계산용 weight를 남기지 않는다.

1. RR/FR 각각 AXIS16 gain, MAE, EPE, axis별 값. 반경 0.25/0.5/1/2를 분리.
2. before/prior-only/current correction의 구조 discrepancy. prior와 같은 matcher만 쓸 때는 독립 정합 증거가 아님을 표기. MIND는 구현이 검증되어 있을 때 **평가에만** 사용.
3. A에 들어가는 rec/abs/eps_RR/eps_FR gradient norm과 cosine; raw/weighted 분리.
4. 고정 소규모 probe에서 동일 T에 current / prior-only / zero correction을 적용한 숫자 비교. **추론 개입이며 재학습 ablation이나 공식 checkpoint 결과가 아님**을 표시.

시험 전에 probe IDs·표본수·geometry·budget을 한 번 정한다. FR/RR 평가셋에서 측정한 진단은 evaluation-only이며 training에는 쓰지 않는다. 간소 probe 값이 공식 20장 전체 평균을 대신하지 않는다. 모든 후보에 같은 표본을 쓰고 누락 이유를 기록한다.

이번 48시간에서는 no-aligner/prior-only 새 U 학습에 예산을 별도 배정하지 않는다. 핵심은 full-method scalar search다. 정식 component ablation은 Teacher 후보를 확보한 뒤 분리한다.

---

## 14. 로컬 기록 — 시트보다 먼저 남기는 실제 원장

### 14.1 디렉터리 예시

아래는 상대적 구조 제안이며 실제 mount 경로는 실행 서버에서 확인한다.

```text
<run_root>/PANDA_S1_M12_TEACHER48H_S123_20261013_v1/<server>/<dataset>/
  campaign.json
  baseline_resolved.json
  admission/
    source_manifest.json
    data_split_scene_manifest.json
    prior_manifest.json
    implementation_decisions.json
    preflight_results.json
    timing.json
  runs/<run_id>/attempt_<n>/
    config_resolved.json
    events.jsonl
    train_metrics.jsonl
    evaluations.jsonl
    evaluations_per_scene.jsonl
    diagnostics_numeric.jsonl
    final_status.json
    # 미달 run에는 weight/qual 디렉터리를 만들지 않음
  qualified/<run_id>/step_<u>/<model_state_sha>/
    teacher_aligner.<server_selected_format>
    model_manifest.json
    metrics.json
    metrics_per_scene.json
    qual/
  outbox/<record_id>.json
  receipts/<record_id>.json
  summaries/
    case_progress.json
    all_evaluations.jsonl
    target_passes.jsonl
    best_teacher.json
    final_report.md
```

원본 데이터셋/공용 source-scene/prior 자산은 불필요하게 run마다 복사하지 않고 immutable digest와 경로로 연결한다. `qualified`에도 A/T만 있고 prior 자산이 사라진 상태가 되지 않도록 복구 가능성을 확인한다.

### 14.2 매 공식 평가의 최소 local record

| 범주 | 필수 필드 |
|---|---|
| 실행 ID | campaign, method_revision, dataset, server, case, parent_recipe, run, attempt, seed, replica |
| 학습 상태 | completed_update, planned_horizon, actual_horizon, schedule, lr_A/U, elapsed_train/eval/wall |
| 방법 설정 | λ_abs, λ_eps_RR, λ_eps_FR, ε_radius, ε_every, input/base, A_resblock/head, method_eligible |
| 근거 | source/config/data/evaluator/scene_map/prior/model_state SHA, initial_A/T hash, RNG/stream 정의 |
| 품질 | HQNR, D_s, D_lambda, JQM, ERGAS, SCC, SAM, PSNR, SSIM, Q4/Q8, RMSE, CC, per-scene metrics |
| 완결성 | expected/completed RR/FR scene count, missing/duplicate/NaN, eval_protocol_id |
| gate | H_target, E_target, h_pass, e_pass, joint_pass, v_H, v_E, D_inf, D1 |
| 보존 | weights_saved, weights_file_sha/path 또는 null, qualitative_saved/path 또는 null |
| 상태 | RUNNING/COMPLETE/PRUNED/FAILED/TIMEBOX, failure/retry reason, source timestamps |
| 업로드 | PENDING/UPLOADED/READBACK_VERIFIED/FAILED_RETRY, destination, receipt, last_attempt |

`missing`은 0이 아니다. 미측정 cost나 gain은 null로 유지한다. 실패한 공식 평가도 시도 기록을 남기되 finite하지 않은 값으로 pass를 계산하지 않는다.

### 14.3 쓰기 순서와 복구

1. 수치를 완성하면 로컬 append-only record와 durable outbox를 먼저 생성한다.
2. atomic write/flush로 반쯤 쓴 JSON을 완료 결과처럼 읽지 않게 한다.
3. 학습 프로세스는 Google 접근 성공을 기다리지 않는다.
4. 별도 publisher가 outbox를 전송하고 readback을 확인한다.
5. `record_id`를 기준으로 retry를 idempotent하게 처리한다. 네트워크 timeout 뒤 blind append로 중복 생성하지 않는다.
6. 수정이 필요하면 기존 숫자를 조용히 덮어쓰지 않고 `supersedes_record_id`, correction reason을 기록한다.
7. 성공 receipt를 얻기 전에는 PENDING 원본을 지우지 않는다. 새 launch는 먼저 미전송 record를 다시 읽는다.

시트가 48시간 내내 실패해도 **config·전체 평가값·장면별 결과·종료 상태·통과 모델의 local path**가 로컬만으로 재구성되어야 한다. 미달 weight가 없으므로 미래의 새로운 지표 재평가는 재학습 없이는 불가능하다는 한계도 기록한다.

---

## 15. Google Sheet 업로드 지시

### 15.1 목적지 및 분리

기존 workbook `pan-cvpr27`의 **`analysis`**를 사용한다. 알려진 workbook ID는 `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`이며, 실제 연결 권한·header·목적지를 서버가 read-only admission으로 확인한다.

**별도 Campaign_ID 구획**에 추가하고 TA2/과거 계획·paper/ablations/대표값은 수정하지 않는다. 이번 MD 작성 과정에서 실제 시트 추가/변경은 하지 않았다.

권장 구획 제목:

`Stage 1 Teacher — MethodPlan-v1.2 — 48h Target Search — 2026-10-13`

### 15.2 보이는 표의 순서

`Experiment | Dataset | Server | Case | Seed/Replica | Selection@Step | HQNR | D_s | D_lambda | JQM | ERGAS | SCC | SAM | PSNR | 나머지 RR | Infer(ms) | Mem(MB) | Params(M) | FLOPs(G) | Train/Eval/Wall | Goal status | Weight/Qual status`.

실험 설명은 개행 없는 한 줄로 한다. loss 계수·LR 배율·horizon을 짧게 표시하고 전체 config는 note/로컬 근거로 연결한다. Seed는 분석 원장에서 반드시 유지한다.

### 15.3 기존 analysis152 호환

기존 `analysis`의 152개 header 계약을 실행 전에 확인하고 **열을 임의로 삽입/재정렬하지 않는다.** 기존 필드 `Campaign_ID / Record_type / Metric_name / Estimate / Reason / Raw_results_URI / Asset_manifest_URI` 등을 통해 새 campaign을 구분한다.

- checkpoint별 SUMMARY에는 모든 공식 품질 수치와 model_state identity를 올린다.
- 새 `λ_abs`는 이번 의미를 label로 명확히 표시하고 기존 `Lambda_struct`를 활용할 경우 descriptor를 `RGB_ABS_GRAD_L1`로 고정하여 NCC와 혼동하지 않게 한다.
- `λ_eps_RR=λ_eps_FR=λ_eps`는 local manifest에 각각 기록한다. 두 branch별 gradient·loss는 Metric_name/Estimate의 long-form 숫자행으로 분리한다.
- gate 필드가 없는 기존 schema에는 `target_joint_pass` 0/1의 수치행과 SUMMARY의 한 줄 Reason을 연결한다. 해석 가능한 숫자를 JSON 링크로만 대신하지 않는다.
- prior norm/residual norm·valid fraction·branch count도 같은 long-form 구조를 쓴다.
- 별도 요약 view가 필요하면 raw append 영역을 뒤섞지 않는 읽기용 view로 만든다. 기존 uploader의 정해진 header를 깨뜨리지 않는다.

구조 확장이 정말 필요하면 서버에서 schema revision을 명시적으로 올리고 기존 데이터와 호환 검증 후 수행한다. 새 metadata를 의미가 다른 오래된 필드에 조용히 끼워 넣지 않는다.

### 15.4 업로드 빈도와 색상

- **공식 평가마다:** gate 통과/미달과 관계없이 1 SUMMARY + 필요한 장면별/진단 수치.
- **상태:** run 시작/종료/중단/오류. 장기 run은 대략 1시간 간격 진행 요약. 매 update STATUS 업로드 금지.
- **실험군 구분:** Full reference / Loss sweep / Optimizer sweep / Long schedule / Confirmation.
- **색상:** 통과=녹색, 한 목표만 충족=노랑, 둘 다 미달=연회색, 실행 오류/데이터 불명=연한 빨강, 진행=파랑, 진단=보라.

붉은 실행 오류와 낮은 성능의 유효 결과를 같은 상태로 처리하지 않는다. 시트에 업로드됐다는 것과 weight가 저장됐다는 것은 별도 값이다.

Uploader는 서비스의 실제 quota에 맞춰 retry/backoff를 적용한다. 다른 서버 결과나 global shared lock을 기다리지 않는다. permission denied와 rate-limit을 분리해 로컬에 남기고 현재 공개 범위를 임의로 넓히지 않는다.

### 15.5 비용 수치

추론은 **scene prior 계산 + A + T** 경로의 실제 비용을 기록한다. scene prior는 일회성/타일당 amortized 비용을 별도로 남긴다. RR/FR shape·batch1·precision·GPU를 구분한다.

`Infer(ms)`에 학습시간을 넣거나, A가 바뀌었는데 과거 parameter/FLOPs 값을 재사용하지 않는다. 직접 측정되지 않은 값은 공란이다. 성능 목표에는 cost gate를 추가하지 않는다.

---

## 16. 결과에 따른 다음 실행 선택

### 16.1 네 가지 상황

| 관측 | 다음 우선 작업 | 하지 않을 것 |
|---|---|---|
| HQNR 통과, ERGAS 미달 | 우수 recipe의 U LR/긴 horizon, λ_abs 조절 검정 | 다른 step의 낮은 ERGAS를 결합 |
| ERGAS 통과, HQNR 미달 | ε 강도·A LR 완화, loss interaction 검정 | FR에서만 A를 꺼서 full method로 표기 |
| 두 지표 모두 미달 | F00 대비 gap/curve/gradient를 보고 다른 full scalar recipe | 빠른 seed 갈아끼우기, method 항 삭제 |
| 두 지표 동시 통과 | 즉시 저장/정성 export, 이어 R2/R3 확인 및 더 좋은 feasible Teacher 탐색 | 최초 성공만 남기고 다른 실패 기록 삭제 |

모든 규칙은 새 방법의 성공 가설을 검정하기 위한 것이다. prior-only가 나쁘면 estimator의 부호/원천/단위를 확인하되, test score에 맞춰 scene shift를 직접 최적화하지 않는다.

### 16.2 후보가 끝내 없을 때

48시간 마감에 통과가 없으면:

- 최고 수치 후보와 각 지표의 목표 차이를 **같은 checkpoint 기준**으로 기록한다.
- scalar별 추세, 초기/최종 response·residual, loss gradient imbalance를 남긴다.
- `weights_saved=0`, `qualitative_saved=0`을 정상적인 gate 결과로 보고한다.
- 가장 근접한 recipe의 config/seed/step을 다음 캠페인의 재실행 대상으로 제안할 수 있지만 가중치가 존재한다고 표시하지 않는다.
- 구조 변경이나 test-time adaptation이 필요한지의 논의는 다음 버전에서 별도로 한다.

---

## 17. 48시간 종료 시 제출할 결과

서버별 다음 여섯 항목을 로컬에 만들고 가능한 것은 시트에 올린다.

1. **진행 목록:** planned/started/complete/pruned/failed/not-started와 실제 time budget.
2. **모든 checkpoint 수치:** same-state RR/FR 결과, 평가 scene completeness, target pass/fail.
3. **최고 통과 Teacher:** 선택 기준, A/T/model+file SHA, inference prior 자산, qualifier metrics, 정성 산출물.
4. **Recipe 반복 결과:** 성공 seed만이 아닌 전체 n의 평균/표준편차/target-hit count. best single과 repeat mean을 구분.
5. **기전 요약:** RR/FR response, prior/residual 크기, rec/abs/ε_RR/ε_FR gradient의 변화. 물리적 변위 GT 주장 금지.
6. **운영 미완료 목록:** local-only/PENDING uploads, artifact/qual pending, technical failures. 학습 미완료와 업로드 미완료 분리.

최종 서버 요약 한 줄:

`dataset | cases_started | full_runs | evaluated_states | pass_states | pass_runs | best_HQNR | same_ckpt_ERGAS | run/step | weight_verified | qual_complete | uploads_verified/pending | wall_h`.

캠페인 전체 결론은 `WV3 달성 여부 / QB 달성 여부 / GF2 달성 여부`를 각각 적는다. 하나의 센서 성공을 다른 센서에도 달성한 것처럼 합치지 않는다.

---

## 18. 실행 담당자용 최종 체크리스트

- [ ] s1/WV3, s2/QB, s3/GF2로 정확히 연결했고 s4/s5를 건드리지 않았다.
- [ ] 구현은 MethodPlan-v1.2의 Stage 1이고 TA2 NCC/NGF·old A를 재사용한 다른 방법이 아니다.
- [ ] actual source scene mapping과 FR training-only inputs가 확보됐다.
- [ ] baseline architecture/input/base/optimizer/batch/precision/ROI 및 실제 source hash가 고정됐다.
- [ ] RR/FR ε는 radius2·매 update·동일 λ이며, branch별 mean의 합이다.
- [ ] λ_abs/λ_eps 변화는 등록 case scalar만이며 loss definition은 동일하다.
- [ ] 서버별 실제 throughput으로 48시간 내 작업 수를 산정했다.
- [ ] F00/R1 후 센서별 loss 탐색, 상위 recipe LR/긴 학습, 우수 recipe 반복 순서를 따른다.
- [ ] 공식 H/E를 같은 A/T 상태에서 전체 평가하고 반올림 전 AND gate를 적용한다.
- [ ] 미달 weight, periodic last/resume, 평가용 임시 weight를 쓰지 않는다.
- [ ] 미달 model의 numeric 결과는 로컬과 시트에 남기지만 정성 파일은 만들지 않는다.
- [ ] 통과 모델은 A/T·prior/source/evaluator provenance와 정성 산출물을 함께 검증한다.
- [ ] local-first durable outbox와 idempotent upload/readback을 사용한다.
- [ ] 멈춘 미달 모델의 재개는 불가능함을 알고 새 attempt fresh 재시도로 구분한다.
- [ ] deadline에서 학습을 늘리거나 목표/평가 방식을 바꾸지 않는다.
- [ ] 기존 모델·paper·ablation·재현 결과의 파일과 시트는 보존한다.

---

## 부록 A. 사람·구현 모두 읽을 수 있는 캠페인 핵심 명세

아래는 **실험 규칙 예시 YAML**이며 현재 서버 CLI나 이미 구현된 config schema라고 주장하지 않는다. 실제 loader에 연결하기 전 baseline pin 항목을 채워야 한다.

```yaml
campaign_id: PANDA_S1_M12_TEACHER48H_S123_20261013_v1
method_revision: MethodPlan-v1.2
scope: STAGE1_ONLY
model_scope: FULL_STAGE1_TEACHER_AND_ALIGNER
implementation_owner: execution_server
experiment_plan_status: PLANNED

lanes:
  s1: {dataset: WV3, hqnr_min: 0.958, ergas_max: 2.04, seeds: [261013101, 261013102, 261013103]}
  s2: {dataset: QB, hqnr_min: 0.923, ergas_max: 3.5, seeds: [261013201, 261013202, 261013203]}
  s3: {dataset: GF2, hqnr_min: 0.965, ergas_max: 0.522, seeds: [261013301, 261013302, 261013303]}

method:
  scene_prior: FIXED_DIRECTION_AND_MAGNITUDE
  correction: prior_plus_alpha_direction_plus_beta_orthogonal
  aligner_reference: NATIVE_LMS
  aligner_resblock: SHARED_UNET_RESBLOCK_64_LN
  aligner_stem_joint: KEEP_EXISTING_GN
  teacher_architecture: MUST_PIN_CURRENT_APPROVED
  teacher_input_and_base: MUST_PIN_CURRENT_APPROVED
  reconstruction: RR_NATIVE_ONLY
  absolute_loss: RGB_MEAN_ABSOLUTE_2D_GRADIENT_L1_RR_NATIVE_ONLY
  epsilon_rr: true
  epsilon_fr: true
  epsilon_radius_rr: 2.0
  epsilon_radius_fr: 2.0
  epsilon_distribution: UNIFORM_AREA_DISK
  epsilon_every_update: true
  epsilon_shared_coefficient_rr_fr: true
  epsilon_target: STOPGRAD_NATIVE_CORRECTION
  scene_prior_changed_by_epsilon: false
  shifted_pan_to_teacher: false
  fr_gt_loss: false
  cross_scale_loss: false
  stage2: false

baseline:
  lambda_rec: 1.0
  lambda_abs: 1.0
  lambda_epsilon: 0.001
  lr_U0: MUST_PIN_FROM_APPROVED_BASELINE
  lr_A0: MUST_PIN_FROM_APPROVED_BASELINE
  weight_decay0: MUST_PIN_FROM_APPROVED_BASELINE
  effective_batch_rr: MUST_PIN
  effective_batch_fr: MUST_PIN
  gradient_kernel_roi_normalization: MUST_PIN

search:
  loss_grid_epsilon: [0.0001, 0.0003, 0.001, 0.003]
  loss_grid_absolute: [0.1, 0.3, 1.0, 3.0]
  horizon_updates: [50000, 100000, 150000]
  all_candidates_three_seeds: false
  shortlisted_recipe_confirmations: 3
  complete_all_candidates_guaranteed: false
  use_previous_ta2_weights: false
  pruned_unqualified_resume_from_disk: false

budget:
  per_server_hours: 48
  independent_server_clock: true
  no_new_training_after_hour: 46
  include_preflight_eval_artifacts: true
  cross_server_wait: false

selection:
  qualification: SAME_STATE_HQNR_GE_MIN_AND_ERGAS_LE_MAX
  representative: HQNR_DESC_THEN_ERGAS_ASC_THEN_STEP_ASC
  comparison_uses_unrounded_values: true
  all_eval_scenes_required: true
  test_aware_development: true
  use_raw_original_fr_reference: true

retention:
  policy: STRICT_TARGET_ONLY
  persist_unqualified_weights: false
  persist_unqualified_initial_weights: false
  persist_unqualified_rolling_resume: false
  persist_unqualified_eval_temp_weights: false
  persist_unqualified_qualitative: false
  retain_live_training_state_in_ram: true
  preserve_all_numeric_records: true
  preserve_all_qualified_checkpoints: true
  preserve_existing_campaign_artifacts: true

logging:
  local_first: true
  durable_outbox: true
  spreadsheet: pan-cvpr27
  sheet: analysis
  isolate_by_campaign: true
  append_and_readback: true
  idempotent_record_id: true
  upload_failure_blocks_training: false
  silently_delete_pending_results: false
```

## 부록 B. 실행 ID 및 traceability 예시

실제 run ID 예:

`S1M12_48H_WV3_S1_F01_H50_R1_S261013101_v1`

튜닝 예:

`S1M12_48H_QB_S2_F08_T00_H50_R1_S261013201_v1`

긴 확인 예:

`S1M12_48H_GF2_S3_F06_T04_H100_R2_S261013302_v1`

위 이름은 **예시 조합**이지 이미 선정된 우수 recipe나 실행 완료 기록이 아니다. ID 외에 실제 full resolved config를 반드시 저장한다. 다른 물리 서버로 이동하면 origin_server와 current_server를 구분하고, 다른 데이터셋으로 바뀐 것처럼 이름만 재사용하지 않는다.

**최종 원칙:** 목표가 높은 Teacher를 찾는 것이지, 목표를 맞추기 위해 평가 기준이나 method를 바꾸는 것이 아니다. 성공 모델만 파일로 남기되, **성공하지 못한 실험도 판단 가능한 기록으로 모두 남긴다.**
