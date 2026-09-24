# PANDA WV3 — Table A supervision-coefficient experiments
## s4·s5 전환 / 메인 PLH–F1 기준 / HQNR checkpoint 선택 / 독립 연속 반복

- 문서 작성일: **2026-09-25**
- Campaign ID: `PANDA_MAINA_WV3_S45_HQNR_20260925_v1`
- 대상 서버: **s4, s5만**. s1·s2·s3의 component ablation cycle은 변경하지 않는다.
- 사용자 승인 범위: Table A의 **α·β·λE**만 실험하고, 모델 선택 기준을 **ERGAS 최소에서 HQNR 최대로 변경**한다. 기존 s4·s5 sensitivity 작업은 아래 안전 전환 절차로 중단·보존한 뒤 새 campaign으로 교체한다.
- 이 문서의 상태: **실험 명세와 인계자료 작성 완료 / 원격 배포·중단·학습 기동은 미수행**. 새 case의 실제 실행·완료를 뜻하는 문서가 아니다.

> 실행 요약: 각 서버에서 기존 G23 신규 작업 진입을 막고, 현재 작업을 안전하게 checkpoint·보존하여 GPU를 반납받는다. **PLH/W104D122 + 동일 F1** 자산을 검증한 뒤, **BASE+6변형을 같은 초기 상태로 각각 fresh50K** 학습한다. 모든 조건은 동일한 50개 후보의 **native FR20 raw-original HQNR 최대 checkpoint**를 주 결과로 선택한다. 다음 cycle에서는 새 Student seed로 같은 7조건을 반복한다. 서버 간 대기·공유 selection lock은 없다.

---

## 0. 이번 변경으로 확정하는 것

| 항목 | 기존 SENS-G23 | 새 MAIN-A / HQNR campaign |
|---|---|---|
| 논문용 탐색 범위 | Table A + Table B에 해당하는 6축 | **Table A의 α, β, λE 3축만** |
| Student | P0 / W104 / D[1,2,1] | **메인 Ours와 같은 PLH / W104 / D[1,2,2]** |
| Teacher | T0 / step24240 | **메인 Ours의 F1 / exact50K** |
| Calibration | T0 원값 | **F1 원값 고정** |
| 주 checkpoint 선택 | 기존 문서는 Exact50K 주 결과, val-ERGAS 최소 보조 | **HQNR_MAX50** |
| 보조 checkpoint | val-ERGAS 최소 | **EXACT_50000** |
| 서버 역할 | s4: loss / s5: calibration·A LR | **s4·s5 모두 Table A 전체 7조건, 서로 다른 seed block** |
| 반복 | 서버별 독립 cycle | 유지: 각 서버 독립 무기한 cycle |
| 기존 결과 | G23 원본 | 삭제·재라벨링하지 않고 보존, 새 표와 분리 |

사용자가 제외한 Table B의 `q_ref` 배율, `tau_R` 배율, Aligner LR 비율은 **실험 축과 논문 sensitivity 표에서 제외**한다. 이 설정들이 method에서 사용되지 않는다고 새로 주장하는 것은 아니다. 해당 calibration과 Aligner update는 기존 방법의 일부로 그대로 유지한다.

이번 전환에서는 새 Teacher, 새로운 loss, attention, PAN reconstruction task, PANMIX, 학습 길이 탐색을 추가하지 않는다. 또한 checkpoint 선택 변경을 s1·s2·s3의 기존 component study에 소급 적용하지 않는다.

## 1. 기준 모델: paper의 WV3 Ours

### 1.1 실제 기준 run

| 항목 | 고정할 identity |
|---|---|
| Source | `paper!B17:N17`, `유의미한결과!B16:U16`, `WV3-s1` row96 |
| 원 run | `FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1` |
| Student | PLH / W104 / D[1,2,2], attention-free, 단일 HRMS 복원 |
| Teacher | F1 / P0 / W112 / D[1,2,3], Teacher seed71001, exact50K |
| 과거 main 선택 | TARGET, step38380 |
| 과거 main HQNR / ERGAS | **0.9587325529698706 / 2.056060193721197** |
| 원본 실행 release | `cccedeeeffd7ed19686e5ea684489ceee23cf313` |
| 초기화 규약 | `FH12_named_tensor_fresh9_zero_extra_v1` |
| 기본 계수 | **α=1, β=0.1, λE=0.002** |

위 identity는 이번 문서 작성 중 live Sheet의 메인 행·메모와 원본 row96 metadata를 재조회하여 확인했다. 새 실험은 이 recipe를 기준으로 하되 **아래의 checkpoint 선택·실행 관리 규칙만 명시적으로 교체**한다. [S1–S3]

### 1.2 HQNR 선택으로 바뀌면 과거 main의 비교점도 달라진다

과거 main은 `HQNR≥0.9585 후보 중 ERGAS 최소`인 TARGET 선택점이다. 이번에는 그러한 조건부 ERGAS 선택을 사용하지 않는다.

같은 과거 run의 저장된 raw-HQNR 최대점은 **step22220 / HQNR 0.9612164300901325 / ERGAS 2.090374252705049**다. 이 값은 이번에 새로 재현한 결과가 아니라, 기존 run의 별도 선택점이다. 현재 paper의 TARGET 값과 바꿔 읽으면 안 된다. [S1]

따라서 다음을 지킨다.

- 새 sensitivity의 기본값 행은 **새 BASE에서 HQNR 규칙으로 고른 checkpoint**로 채운다.
- 과거 TARGET 수치 0.958733을 새 seed의 중앙값·평균 대신 넣지 않는다.
- paper의 기존 Ours 행은 이 계획만으로 자동 교체하지 않는다.
- HQNR가 높은 checkpoint의 ERGAS가 덜 좋더라도, 다른 checkpoint의 ERGAS를 붙이지 않는다.
- HQNR 최대점과 최종50K를 함께 보존하여 선택 기준 변화와 실제 최종 모델의 차이를 볼 수 있게 한다.

## 2. Table A: 실행할 7개 조건

모든 조건은 **입력·구조·Teacher·calibration·데이터·optimizer·학습 길이가 같다.** BASE 대비 표시된 계수 하나만 바꾼다.

| 실행 Case ID | 논문상 역할 | α / λ_hard | β / λ_soft | λE / λ_edge | BASE 대비 변경 |
|---|---|---:|---:|---:|---|
| **BASE** | Main default | **1.0** | **0.10** | **0.0020** | 없음 |
| **AL05** | Hard emphasis — low | **0.5** | 0.10 | 0.0020 | α만 감소 |
| **AL15** | Hard emphasis — high | **1.5** | 0.10 | 0.0020 | α만 증가 |
| **BE005** | Soft distillation — low | 1.0 | **0.05** | 0.0020 | β만 감소 |
| **BE020** | Soft distillation — high | 1.0 | **0.20** | 0.0020 | β만 증가 |
| **ED0006** | GT edge supervision — low | 1.0 | 0.10 | **0.0006** | λE만 감소 |
| **ED006** | GT edge supervision — high | 1.0 | 0.10 | **0.0060** | λE만 증가 |

범위는 사용자가 동의한 기존 Table A 제안에서 그대로 가져왔다. 결과에 따라 범위를 줄이거나, 좋아 보이는 조건만 다음 cycle에서 반복하지 않는다. [S2]

논문 표는 다음 세 블록으로 만든다.

| 블록 | Low | Default | High |
|---|---|---|---|
| α | AL05 | BASE | AL15 |
| β | BE005 | BASE | BE020 |
| λE | ED0006 | BASE | ED006 |

**세 블록의 Default는 동일한 BASE 결과 집합을 공유한다.** BASE를 블록마다 새 독립 관측처럼 세 번 집계하지 않는다. λE=0 등의 component 제거 실험도 이번 표에 새로 끼우지 않는다.

## 3. 고정 training 및 method 규약

### 3.1 학습 설정

| 항목 | 설정 |
|---|---|
| Dataset | WV3, 기존 F1/main에 사용한 train·val·RR·FR 파일 |
| Bands / normalization | 8 bands / maxDN2047 |
| Student U | W104, depth[1,2,2], norm=LN, dropout=0, attention locations=[], mode modulation=False |
| Student 입력 | **PAN·LPAN·HPAN + upsampled MS**, 기존 PLH 채널 배치 유지 |
| Student A | F1 Teacher A에서 clone, **trainable** |
| Teacher | F1의 U와 A 모두 frozen, inference/eval mode |
| 학습 길이 | **각 run 50,000 optimizer updates** |
| Batch / test batch | 48 / 1 |
| Optimizer | AdamW, wd0.01, betas[0.9,0.999], eps1e-8 |
| U / A peak LR | **1e-4 / 3e-6** |
| Scheduler | warmup100 + `COSINE_W100_BASE_v1` |
| Precision | 원 main 규약: FP32, `mixed_precision=no` |
| Data loader | 원 FH20R1 feeder와 동일한 동작, num_worker4 |
| Student 추가 shift loss | offset_weight=0, synthetic shift radius=0 |
| View margin | HR4 |
| α·β·λE | §2의 해당 case 값 |
| q_ref / tau_R | **0.4532603621482849 / 0.012118559330701828**, 전 case 고정 |
| q_ref 배율 / tau_R 배율 / rA | **1 / 1 / 0.03**, 전 case 고정 |

Optimizer·학습 규약은 원 main builder와 그 inherited FH12 config를 기준으로 한다. 신버전에서 base 설정을 복원할 때 현재 G23 config를 대충 수정하지 말고, **원 main의 계산 경로와 실제 설정을 출발점으로 사용**한다. [S2–S4]

특히 서버가 s4 또는 s5라는 이유로 `teacher_for(server)`를 호출하여 F4/F5로 바꾸지 않는다. 실행 위치만 달라지며 Teacher는 두 서버 모두 **F1**이다.

### 3.2 LPAN/HPAN·geometry는 그대로 유지

원 main의 `gauss41-offset2-up4-bicubic-align_corners_false` LP 규약과 파일을 사용한다. PAN과 LPAN에는 같은 Student correction을 적용하며 HPAN은 기존 synchronized frontend에서 계산한다. MS는 기준 좌표계에 유지한다.

G23의 P0 입력 경로에 LPAN/HPAN label만 붙여 실행하지 않는다. 기존 component study의 scratch-A·frozen-A·masked-input switches도 이번 모든 run에서 사용하지 않는다.

Crop, flip, rotation은 **원 feeder의 실제 동작과 train-view 인덱싱**을 그대로 사용한다. flag 이름만 보고 새로운 augmentation 확률을 구현하지 않는다. 이번 전환 중 데이터 생성, 보간 위상 보정, 평가 crop 수정, LP 재생성은 수행하지 않는다. 데이터 프로토콜을 바꿔야 하면 별도 revision으로 분리한다.

### 3.3 Table A 계수가 제어하는 경로

다음은 method와 실행 변수의 대응을 설명하는 식이다. 이를 근거로 loss 구현을 새로 만들지 말고 원 main criterion의 reduction·detach·epsilon·edge support를 재사용한다. [S5]

\[
d_T=\frac{e_T}{e_T+\tau_R},\qquad
 a_T=\operatorname{sg}\!\left[\frac{\max(e_S-e_T,0)}{e_S+\epsilon}\right],\qquad
 w_i=\operatorname{sg}\!\left[\frac{q_{\rm ref}}{q_{\rm ref}+q_{T,i}}\right].
\]

\[
H_i=\operatorname{mean}_p[(1+\alpha d_T)e_S],\qquad
K_i=\operatorname{mean}_p[(1-d_T)a_T\Delta_{TS}].
\]

\[
L_U=\operatorname{mean}_i[H_i+\beta K_i+\lambda_E w_i E_i],\qquad
L_A=\operatorname{mean}_i[w_iH_i].
\]

`sg`는 stop-gradient이며 reference/weight의 detach는 원 구현대로 유지한다. β가 이미 들어간 soft loss를 반환하는 함수라면 바깥에서 β를 다시 곱하지 않는다. λE도 실제 gradient 경로에 정확히 한 번만 적용한다.

| 변경 | 직접 바뀌는 학습 신호 | 바꾸지 않는 것 |
|---|---|---|
| α | U의 hard 및 A의 q-weighted hard | Teacher·q·A LR |
| β | U의 selective soft | A로의 직접 soft gradient 없음 |
| λE | U의 q-weighted GT edge | A로의 직접 edge gradient 없음 |

β·λE 변경이 학습 중 U를 바꾸면서 A에 간접 영향을 줄 수는 있다. 이를 금지된 직접 gradient 유입과 혼동하지 않는다.

## 4. F1 및 데이터 자산: 기동 전 필수 확인

### 4.1 고정 identity

아래는 `WV3-s1!FP96:HH96`에서 다시 읽은 **원 metadata의 SHA**다. 각 manifest가 정의한 hash 규약으로 검증한다. 파일 SHA, tensor SHA, JSON manifest SHA를 서로 바꿔 비교하지 않는다.

| 자산 | 원 보고 SHA256 |
|---|---|
| F1 Teacher checkpoint | `04ef8e756ae8b229c67ef14daf1912f0658f2e5c8bfe30afa9e0b476486fc519` |
| F1 bridge | `b902e12b376c2cfdbd69d89c21b732224fec2a52183eca4d5983147e7598c92d` |
| F1 calibration | `45cae955010ab044790442885206b1ffd4d8cecaf5bdf47121747c2fbcc51658` |
| F1 q-cache | `96774f7a7ddc73e151d03009066601d61c40b0b4f9d8bbc88cfc2e41a5ac1c66` |
| Main data manifest | `d2d49536a1e692343bb8c9f2dd64cfebe9e331d15df455b9941ece1c0128a4c2` |
| Original evaluator identity | `79a7e4a154547b355e94e4198a5d85a5a4ecf220ebc039da1b3a3404a91bbc89` |
| 원 initial A tensor-report | `1e01b69360dbd7526b77e55718050fad405ade38eb59f56c215ae07edabc65cd` |

LP file identity:

| Split | SHA256 |
|---|---|
| Train | `53ea13dd0a0de8c4f69cbbcf6fa6f7d9b70eb0362af22ff4d55f16f978a255f3` |
| Val | `5410316e91f801ceb7684174c7675ec1f501aaddcc6cb2b3888735d50856ad0e` |
| RR | `2b81b4c5bf27137a0030840304533d4bc40fb16da586414627123348fe098371` |
| FR | `83e969b1f25234f3908ca2dfeb2fc725fa5eb0fca401aa279b6b9093e927db41` |

파일의 접근 경로는 서버별 로컬 경로로 binding하되 canonical 자산과 내용은 같아야 한다. 경로 필드 때문에 manifest bytes가 달라지는 경우 canonical 원본과 path-only mapping을 함께 보존하고, 내용의 동등성을 별도로 증명한다. SHA 차이를 단순히 무시해서는 안 된다.

### 4.2 두 서버가 독립 실행할 수 있게 자산을 먼저 로컬에 둔다

F1이 현재 로컬에 없다면 검증된 기존 artifact 저장소에서 **한 번만 복제**한다. 이는 새 Teacher 학습이 아니다. s4 결과나 s5 결과를 기다려 reference를 선택하는 과정도 없다.

F1 또는 F1 q-cache가 준비되지 않은 서버는 `PAUSED_ASSET_MISSING`으로 남긴다. 그 서버의 기존 T0, GF2 Teacher, F4/F5, 임의로 재산출한 q-cache로 대체하지 않는다. 다른 서버는 자기 preflight가 끝나면 독립 시작한다.

새 seed마다 calibration을 다시 구하지 않는다. F1 native train-view cache와 원 tau/q scale을 그대로 유지한다. 결과만 보고 calibration을 다시 맞추는 경로를 넣지 않는다.

## 5. Checkpoint 선택을 HQNR로 통일

### 5.1 주 selector

- 표시 이름: **`HQNR_MAX50`**
- 규칙 ID: `MAX_RAW_ORIGINAL_FR20_MEAN_HQNR_THEN_LOWER_STEP_V1`
- 주 지표: 기존 공식 evaluator의 **native FR20 raw-original PAN 기준, 장면별 HQNR의 평균**
- 선택 대상: 한 run의 미리 정한 **동일 50개 A/U checkpoint**
- 동률: 저장된 full-precision HQNR가 같을 때 **더 이른 update**
- **ERGAS·SAM·SCC·PSNR는 checkpoint 선택 또는 동률 해소에 사용하지 않는다.**

\[
\mathcal T=\{1010,2020,\ldots,49490,50000\},\qquad |\mathcal T|=50.
\]

\[
t^*=\operatorname*{argmax}_{t\in\mathcal T}\frac{1}{20}\sum_{j=1}^{20}\operatorname{HQNR}_{j,t}.
\]

표시용 반올림값이 아니라 evaluator가 저장한 full precision으로 정렬한다. HQNR를 평균 Dλ·Ds의 곱에서 다시 계산하지 않는다. `HQNR(V64)`, aligned-PAN reference HQNR, PAN 이동/blur/출력 resampling으로 바꾼 HQNR는 선택에 사용하지 않는다.

이름을 HQNR로만 바꾸고 실제 `min(val_ergas)` 경로를 남기는 구현은 허용하지 않는다. `best_*` directory, checkpoint pointer, summary, uploader, selector alias가 모두 같은 규칙을 따라야 한다.

### 5.2 후보 평가와 보존

50개 후보 모두 같은 20개 FR 장면·같은 preprocessing·같은 evaluator로 평가한다. 원 run을 50K까지 학습하는 규칙은 유지하며, HQNR가 일찍 높게 나와도 학습을 조기 종료하지 않는다. 평가 결과를 이용해 LR·loss·augmentation을 중간에 바꾸지 않는다.

FR 평가는 후보 저장 후 별도 postrun으로 처리해도 된다. 이 경우 **모든 후보 A/U weight를 postrun 완료 전까지 보존**해야 한다. 원 G23의 val-selected/Exact 두 점만 남기는 로직을 먼저 실행하여 나머지 후보를 지워서는 안 된다.

주 결과의 상태는 다음 조건에서만 `COMPLETE`로 승격한다.

1. actual_updates=50000.
2. 등록한 50개 step 전부 존재하고 중복이 없다.
3. 각 step의 FR20 평가가 전 장면 완료되어 있다.
4. NaN/Inf·잘못된 reference·다른 데이터 protocol이 없다.
5. 선택된 A와 U의 checkpoint identity가 일치한다.
6. 선택 checkpoint의 RR20 지표도 전부 계산하여 동일 행에 연결한다.

평가 파일만 누락됐으면 학습을 다시 하지 말고 해당 평가를 복구한다. 이때 `PENDING_EVAL_NOT_COMPARABLE`로 표시하고, 후보 몇 개만 평가한 임시 최고값을 완전한 run과 비교하지 않는다. 학습 자체의 수치 발산은 `DIVERGED`로 남기며 seed를 바꿔 실패를 덮지 않는다.

### 5.3 보조 selector

**`EXACT_50000`**을 모든 run에서 함께 보고한다. HQNR_MAX50와 checkpoint가 같으면 같은 SHA로 alias 처리한다. 두 선택점은 두 번의 독립 학습이 아니다.

기존 val ERGAS는 데이터 검증·학습 log로 남겨도 되지만 이번 campaign에서 선택 권한이 없다. 원 G23나 원 FH20R1의 `RR_VAL_SELECTED`/TARGET 결과는 과거 기록으로 보존할 뿐, 새 sensitivity의 중간점으로 가져오지 않는다.

### 5.4 평가 범위의 정확한 명칭

현재 사용하는 FR20은 공식 FR 평가 세트이므로, 그 HQNR로 checkpoint를 선택한 이번 결과는 **test-aware 개발 선택**이다. 새 validation-HQNR 세트를 확보했다는 뜻이 아니다.

필수 metadata:

```text
primary_selection = HQNR_MAX50
selection_split = FR20
test_aware = true
independent_test = false
expected_candidate_count = 50
hqnr_reference = RAW_ORIGINAL_PAN
checkpoint_tie_breaker = LOWER_COMPLETED_UPDATES
ergas_used_for_selection = false
```

이를 이유로 사용자 요청과 다른 ERGAS selector로 되돌리지 않는다. 대신 표·메모·캡션에 선택 기준을 정확히 남긴다. 추후 별도 validation FR를 만들면 새로운 protocol revision으로 수행한다.

## 6. s4·s5 반복 구성: 전 서버 Table A, 서버 간 대기 없음

### 6.1 매 cycle의 실행 단위

**서버당 BASE1 + 변형6 = 7개의 fresh50K run**이다. 두 서버 모두 세 축을 전부 실행한다. 따라서 어느 축의 결과가 특정 서버의 BASE와만 묶이는 구조를 피하고, 다른 서버가 늦어져도 완전한 local 비교를 얻는다.

같은 서버·cycle의 7조건은 다음을 공유한다.

- 같은 fresh Student U snapshot 및 F1-cloned A snapshot.
- 같은 50K 전체 sample/augmentation 순서와 train-view index 규약.
- 같은 F1 prediction/q 자산, tau/q scale.
- 같은 source·data·evaluator 및 후보 grid.

단, 모델·optimizer·scheduler state는 매 case 독립 생성한다. BASE의 **학습된** weight를 다음 case로 넘기지 않는다. 복사하는 것은 초기 snapshot뿐이다.

### 6.2 Seed 규칙

새 seed 배정은 이 문서의 설계값이다. 과거 메인 seed는 s4 첫 block에서만 재현 anchor로 사용한다.

| Local cycle | s4 seed | s4 목적 | s5 seed | s5 목적 |
|---:|---:|---|---:|---|
| 0 | **73101** | Main recipe anchor: 7조건 모두 실행 | **2100001** | 새 paired block |
| 1 | **2100000** | 새 paired block | **2100003** | 새 paired block |
| 2 | **2100002** | 새 paired block | **2100005** | 새 paired block |
| 3 | 2100004 | 계속 반복 | 2100007 | 계속 반복 |

생성 규칙:

```text
s4, cycle=0: seed=73101
s4, cycle>=1: seed=2100000 + 2*(cycle-1)
s5, cycle>=0: seed=2100001 + 2*cycle
```

Seed 표현 범위가 끝나면 wrap하지 않고 명시적으로 정지한다. 같은 서버/cycle의 retry는 원 seed를 유지한다. s4·s5 사이에 동일 seed를 중복 사용하지 않는다.

s4 cycle0은 이미 선택된 과거 seed의 anchor이므로 **미관측 새 seed 평균에서 제외**하고 별도로 보고한다. 처음 3개 완료 fresh-seed block에서 초기 요약, 6개에서 다음 요약을 내되 이것은 보고 시점이며 성공·실패에 따른 종료 규칙이 아니다.

### 6.3 실행 순서

첫 cycle:

```text
s4: BASE → AL05 → AL15 → BE005 → BE020 → ED0006 → ED006
s5: BASE → ED006 → ED0006 → AL15 → AL05 → BE020 → BE005
```

다음 cycle은 3개 축 묶음의 순서를 한 칸 회전하고 low/high 순서를 번갈아 뒤집는다. BASE는 항상 먼저 실행한다. 계수·seed·구성은 결과를 보고 바꾸지 않는다.

명세 생성 규칙:

```python
pairs = [["AL05", "AL15"], ["BE005", "BE020"], ["ED0006", "ED006"]]
offset = (cycle + (0 if server == "s4" else 2)) % 3
pairs = pairs[offset:] + pairs[:offset]
if (cycle + (0 if server == "s4" else 1)) % 2:
    pairs = [pair[::-1] for pair in pairs]
order = ["BASE"] + [case for pair in pairs for case in pair]
```

### 6.4 연속 운영

- 전체 시간·run·cycle 상한은 두지 않고 사용자 중지까지 반복한다.
- 한 Student를 무한히 이어 학습하는 것이 아니라, **50K run 7개를 완료한 뒤 다음 새 seed cycle**로 넘어간다.
- 서버 간 공유 lock, MSTAR 선택/수신, 공통 t0, 다른 서버의 결과 대기는 없다.
- 서버 내부의 service 중복 실행 방지와 GPU 사용권 보호만 유지한다.
- 다른 서버의 미완료 때문에 정상 서버의 local cycle을 정지시키지 않는다.

최초 3 local cycles씩의 미리보기는 **42개 학습 = 2,100,000 optimizer updates**다. 그중 7개는 과거 seed anchor, 35개는 5개의 새 seed block이다. 42개는 종료 상한이 아니다.

## 7. 기존 s4·s5 작업을 바로 전환하는 절차

### 7.1 전환 대상 식별

기존 대상은 `PANDA_G23_SENS_WV3_S45_20260923_v1`의 해당 서버 controller 및 worker다. **실제 PID, cwd, Docker mount, GPU UUID, `state.json` 위치를 먼저 확인**한다. Sheet의 마지막 완료행만으로 현재 PID나 안전 중단 여부를 단정하지 않는다.

기존 README와 runner에는 `stop --safe-now`가 구현되어 있다. 명령은 현재 실행 release와 **같은 실제 work_dir에 접근하는 환경에서** 사용한다. 다른 checkout의 빈 work_dir에 stop marker를 만들면 실행 중인 프로세스가 멈추지 않을 수 있다. [S6–S7]

확인된 기존 명령 형태:

```bash
# s4의 실제 활성 release / 올바른 Docker 또는 host 환경에서
python tools/g23sens_runner.py status --server s4
python tools/g23sens_runner.py stop --server s4
python tools/g23sens_runner.py stop --server s4 --safe-now

# s5도 해당 서버의 실제 활성 환경에서 독립 실행
python tools/g23sens_runner.py status --server s5
python tools/g23sens_runner.py stop --server s5
python tools/g23sens_runner.py stop --server s5 --safe-now
```

위 명령은 **기존 runner의 안전 중단 요청**이지, 이 문서에서 실제로 호출한 기록이 아니다. `stop`은 후속 run admission을 막고, `--safe-now`는 지원되는 안전 지점에서 현재 작업을 보존·중단하도록 요청한다. 프로세스가 즉시 종료됐다는 의미는 아니다.

### 7.2 즉시 전환의 의미

기존 7조건 cycle이 끝날 때까지 기다리지 않는다. 새 작업 진입을 즉시 막고, **현재 training update 경계에서 full-state를 저장한 후 중단**하는 것을 우선한다. 진행 중인 postrun도 지원되는 안전 경계에서 멈추고 평가 ledger를 보존한다.

현재 활성 release가 safe-now를 지원하지 않거나 동작이 검증되지 않으면 신호를 추측해서 보내지 않는다. 현재 run 경계에서 멈추는 지원 절차를 사용하고 그 지연 사유를 기록한다. 광범위한 `pkill`, 무관한 컨테이너 삭제, 임의 `kill -9`는 사용하지 않는다.

중단 artifact에는 최소한 다음이 있어야 한다.

```text
old_campaign / server / run_id / attempt / actual_updates
A / U weights + optimizer + scheduler
Python / NumPy / Torch / CUDA RNG
sampler cursor / stream identity / accumulation state(if any)
candidate files + evaluation ledger + upload outbox
active release / container image / data & reference identities
pause_reason=OPERATOR_MIGRATION_TO_MAINA_HQNR
paused_at_utc
```

운영상 중단을 성능 발산이나 학습 완료로 표시하지 않는다. 기존 성공 결과, 부분 학습, 재시도 기록을 삭제하지 않는다.

### 7.3 새 campaign 활성화

1. 기존 controller의 후속 admission과 watchdog 재기동이 차단됐는지 확인한다.
2. 기존 full-state와 outbox 보존을 확인한다.
3. 해당 worker와 controller가 안전 종료하여 GPU·로컬 service lock을 반납했는지 확인한다.
4. 별도 frozen release와 **새 namespace `work_dir/maina_hqnr/<server>/`**를 만든다.
5. F1·data·PLH frontend·loss·HQNR selector에 대한 §8 preflight를 수행한다.
6. 새 `BASE`를 fresh initialization에서 시작하고, 시작 receipt를 기록한다.
7. 로그에서 **`PLH / W104D122 / F1 / HQNR_MAX50`**와 actual update 증가를 확인한 후에만 `RUNNING`으로 표시한다.

신규 실행 파일명과 CLI는 구현자가 확정해야 한다. 기존 `g23sens_start.sh`를 이름만 바꾸거나 최신 `main`을 pull했다는 이유만으로 새 학습이 시작됐다고 기록하지 않는다. **이 번들은 새 CUDA trainer/launcher를 배포하는 실행 스크립트가 아니다.**

새 자산 검증에 실패해도 T0 fallback이나 기존 G23 자동 재시작은 하지 않는다. 해당 서버만 명확한 PAUSED 사유를 남긴다. s1·s2·s3는 건드리지 않는다.

## 8. 구현 변경과 시작 전 acceptance

### 8.1 변경할 코드 범위

| 모듈 역할 | 필요한 변경 |
|---|---|
| Case registry | §2의 7조건만, 독립 seed·cycle 생성 |
| Model/input loader | 원 main PLH/W104D122, synchronized frontend |
| Teacher/reference loader | 두 서버 모두 F1 + 해당 calibration/q-cache |
| Config builder | old G23/T0/9ch/D121 강제값 제거; 메인 init policy 복원 |
| Trainer | 원 loss/gradient 경로 그대로, α·β·λE만 override |
| Candidate writer | 50개 A/U checkpoint 보존 및 SHA ledger |
| Postrun evaluator | 후보50 × FR20 HQNR 평가 후 동일 체크포인트 RR 평가 |
| Selector | `argmax(raw_original_HQNR)` + lower-step tie, ERGAS 선택 제거 |
| Reporter/uploader | HQNR_MAX50 주행 + Exact50K 보조행, test-aware 명시 |
| Controller | 새 namespace, 서버별 7조건 cycle, 기존 작업 safe cutover |

원 main의 `hqnr_threshold=.9585`와 `ergas_goal=2.040`가 config에 남더라도 이번 selector의 gate·필터·중단 조건에 사용하면 안 된다. metadata 목적이라도 과거 항목으로 명확히 분리하고 새 selector는 이 항목을 참조하지 않는다.

### 8.2 계산 경로 acceptance

실행 담당자는 최소 다음을 검증하고 결과를 남긴다.

- PLH가 실제로 11채널이고 P0는 frozen Teacher 입력에만 쓰이는지.
- Student width/depth, head·attention·mode modulation이 기준과 같은지.
- F1 frozen U/A의 identity, trainable Student A clone 동일성.
- BASE의 초기 PLH extra-slot policy와 원 main의 named-tensor initialization이 같은지.
- 동일 sample/view에서 새 BASE와 원 main의 입력·warp·output·loss·U/A gradient가 같은 계산을 하는지.
- soft·edge loss가 Student A로 **직접** 전달되지 않는지.
- case별 실제 coefficient가 정확히 한 번 적용되고 fixed q/tau/LR가 변하지 않는지.
- 같은 cycle의 7조건 initial U/A와 data-stream SHA가 같은지.
- checkpoint save/resume가 optimizer·RNG·sampler까지 복원하는지.

부동소수점 비교 tolerance는 수치 모드와 비교대상 장치를 명시해 **비교 전에 고정**한다. 결과가 불편하다는 이유로 tolerance나 모델 설정을 나중에 완화하지 않는다. 동일 seed의 과거 성능을 정확히 재현한다는 주장은 GPU 재현 후에만 한다. 성능값 자체를 성공 launch 조건으로 삼지 않는다.

### 8.3 selector acceptance

최소 테스트:

1. ERGAS가 낮은 checkpoint와 HQNR가 높은 checkpoint가 다를 때 **HQNR가 높은 쪽**을 선택한다.
2. full-precision HQNR가 같으면 더 이른 step을 선택한다; ERGAS는 tie-break에 참여하지 않는다.
3. 반올림 후 같아 보이는 두 HQNR도 원 precision으로 구분한다.
4. HQNR_MAX50 선택 checkpoint의 RR/FR·A/U SHA가 모두 동일하다.
5. HQNR(V64)만 있는 기록은 raw-original 후보로 사용하지 않는다.
6. 후보 49개·중복 step·FR19장·NaN·SHA 불일치이면 COMPLETE 승격을 막는다.
7. HQNR 최대가 일찍 나오더라도 actual_updates=50000까지 학습한다.
8. Exact50K와 같은 checkpoint일 때 alias 처리한다.
9. 기존 G23/TARGET/VAL 결과를 새 BASE로 읽어오는 경로가 없다.
10. 평가 호출이 학습 RNG·stream을 변경하지 않는다.

## 9. 결과·Sheet 정리 규약

### 9.1 원본 결과 탭

새 탭 이름을 사용한다.

- s4: **`SENS-MAINA-WV3-s4`**
- s5: **`SENS-MAINA-WV3-s5`**

각 run의 표 행은 `HQNR_MAX50`와 `EXACT_50000`로 구분한다. 같은 checkpoint면 alias 필드로 연결한다. 후보50개의 전체 곡선과 장면별 지표는 별도 JSON/CSV artifact에 보존한다.

필수 field:

```text
campaign_id, server, cycle, seed, case_id, axis, value, run_id, attempt
student_layout, width, depth, teacher_alias, teacher_sha, calibration_sha, q_cache_sha
alpha, beta, lambda_E, fixed_q_ref, fixed_tau_R, U_peak_lr, A_peak_lr
initial_U_sha, initial_A_sha, stream_sha, data_sha, source_sha, evaluator_sha
actual_updates, candidate_grid_sha, expected_candidates, evaluated_candidates
selection, selection_split, test_aware, independent_test, selected_step
checkpoint_AU_sha, evaluation_manifest_sha, selection_alias_of
hqnr, d_lambda, d_s, ergas, sam, psnr, scc, ssim, q8, rmse, cc, jqm
baseline_run_id, baseline_selection, delta_hqnr, delta_ergas, delta_d_s, delta_d_lambda
training_seconds, evaluation_seconds, wall_seconds, status, readback_status
```

기존 `PAPERSET_IDENTITY_UNVERIFIED`와 같은 평가 대응 경고가 실제로 해소되지 않았다면 없애지 않는다. 파일 SHA 일치가 논문 장면 대응 검증까지 자동으로 뜻하지는 않는다.

### 9.2 ablations 수집 표

사용자가 만든 `Hyper parameter experiements all`에는 **새 MAIN-A 별도 구획**으로 넣는다. 구획명은 `WV3 — MAIN-A | PLH/W104D122 | F1 | HQNR_MAX50`로 한다. 기존 G23 결과·실패 기록과 사용자 main component 대표표는 그대로 둔다.

정렬: **baseline → α → β → λE**, 각 계수값 안에서 server/cycle 순서. 우수한 run만 남기거나 historical main을 새 BASE 칸에 복사하지 않는다.

Methods 예시 — 개행 없이 한 줄:

```text
α=0.5 | PLH/F1 | s4 cycle1 | HQNR@22220
β=0.2 | PLH/F1 | s5 cycle1 | HQNR@24240
λE=0.006 | PLH/F1 | s4 cycle2 | HQNR@33330
```

위 step들은 **표시 형태 예시이지 실제 선택 결과가 아니다.** 실제 step으로 채운다. seed와 전체 run ID는 별도 metadata·셀 note에 남기고 표시용 methods에는 seed를 넣지 않는다.

주행의 HQNR·Ds·ERGAS·SCC·SAM·PSNR는 같은 checkpoint의 값이다. Inference는 실제 profiling 시 **초(s)**, 비용은 원 convention을 명시한 **FLOPs(G)**로 기록한다. 새 profiling이 없으면 공란으로 두며 학습시간이나 다른 run의 값을 대신 넣지 않는다.

### 9.3 Table A 집계

각 변형은 **같은 서버·같은 cycle·같은 selector의 BASE**와 비교한다.

\[
\Delta H_s=H_s(\text{variant})-H_s(\text{BASE}),\qquad
\Delta E_s=E_s(\text{variant})-E_s(\text{BASE}).
\]

새 seed의 raw 값, 평균±표준편차, paired Δ, 완료/발산/기술실패 수를 함께 남긴다. s4 과거-seed anchor는 별도 block으로 표시한다. 두 서버에 서로 다른 seed를 배정했어도 **Teacher는 같은 F1 하나**이므로 Teacher-seed 일반화라고 주장하지 않는다.

한 축의 low/default/high 평균은 같은 완료 seed 집합으로 계산한다. 완료가 느린 불리한 variant를 빼서 평균을 먼저 확정하지 않는다. 불완전 block의 raw 값은 공개 수집에 남기되 summary의 완결 여부를 표시한다.

기본값이 가장 좋은지는 이 비교의 결과로 판정한다. 다른 값이 반복적으로 더 좋으면 그 결과도 그대로 보존하며 자동으로 약한 대조군으로 바꾸지 않는다. 이번에는 HQNR가 주 선택 지표이고 ERGAS는 반드시 함께 보고하는 복원 정확도 지표다.

## 10. 무기한 운영의 실패·복구·보존

- **BASE 기술실패:** 해당 local block을 pause하고 원인 수정 후 같은 case·seed의 새 attempt로 재시도한다. BASE 없는 variant 비교를 진행하지 않는다.
- **Variant 기술실패:** 원 attempt를 보존한다. 동일 case의 안전 재개 또는 fresh retry를 별도 attempt로 기록하며, 총 2회 retry 후 서버를 pause한다.
- **수치 발산:** `DIVERGED`로 남기고 바꾼 seed로 대체하지 않는다. BASE가 정상인 경우 다른 variant는 계속할 수 있지만 block summary의 발산을 누락하지 않는다.
- **평가·업로드 실패:** 학습을 다시 하지 않고 저장한 candidate/outbox에서 복구한다.
- **Source 변경:** block 중간에 계산 경로가 바뀌면 mixed-source 완결 block으로 집계하지 않는다. 수정 후 새 block/revision에서 BASE 포함 7조건을 다시 맞춘다.
- **저장공간 부족:** pause한다. 자동 batch 축소·precision 변경·후보 수 축소·비검증 checkpoint 삭제를 하지 않는다.
- **사용자 중지:** 해당 서버의 로컬 stop marker로 다음 admission을 막고 안전 보존한다.

모든50 후보는 HQNR 선택 완료 전까지 유지한다. 이후에는 원자료와 정확한50K, HQNR-selected, full-state, config/seed/init/stream/eval identity를 반드시 보존한다. 중간 후보 삭제는 별도 retention 정책을 승인한 경우에만 한다.

## 11. 계산량과 초기 확인 일정

현재 시간 상한은 없다. 50K 학습시간은 기존 G23/P0에서 측정한 시간을 그대로 보장값으로 사용하지 않는다. 새로운 PLH/F1 모델에서 첫 BASE의 실측 시간을 얻어 예약량을 갱신한다.

**HQNR 선택 변경은 평가량도 바꾼다.** 최종 두 checkpoint만 FR 평가하는 방식이 아니라, 최소 **50후보 × FR20 = 1,000장면-forward/run**이 필요하다. 선택된 후보와 Exact50K에는 RR20 평가도 필요하다. Exact가 selected와 같으면 재사용한다.

서버별 첫 BASE 이후 training/eval/I/O 시간을 따로 기록하고, 후보 retention 용량과 outbox 공간을 확인한다. 시간을 줄이려고 특정 case만 후보 수나 FR 장면 수를 줄이지 않는다. candidate50 평가의 비용을 숨기고 Train(h)만 전체 실험시간처럼 기록하지 않는다.

초기 확인 우선순위는 **각 서버 7조건 한 block 완결 → 새 seed block 누적**이다. s4·s5 중 한 서버만 준비됐으면 해당 서버에서 전부 실행할 수 있다.

## 12. 실행 담당자용 최종 인계 체크리스트

1. 대상이 s4·s5인지, 실제 기존 G23 PID/release/work_dir가 무엇인지 확인한다.
2. 기존 신규 admission 차단 → safe-now checkpoint → 실제 GPU 반납을 확인한다.
3. 기존 G23 데이터·결과·부분학습·outbox를 보존한다.
4. F1 U/A + F1 q-cache + F1 calibration + 원 WV3/LP 자산을 로컬 검증한다.
5. 새 PLH/W104D122 builder와 원 main의 계산 경로를 맞춘다. T0/P0/D121 fallback은 없다.
6. α·β·λE 외 설정은 고정한다. Table B case는 queue에 없다.
7. `HQNR_MAX50` 및 동률 lower-step, FR20·raw-original, 50후보를 검증한다.
8. 각 서버 BASE부터 실제 학습을 시작하고 PID·첫 update·run/config SHA receipt를 남긴다.
9. BASE+6조건을 local paired block으로 마친 뒤 새 seed cycle로 계속 진행한다.
10. new raw tabs / ablations 새 구획에 기록하고 readback을 확인한다. 사용자 대표값은 자동 교체하지 않는다.

**구현 완료·기동 완료·50K 학습 완료·50후보 평가 완료·Sheet readback 완료는 서로 다른 상태다. 각각의 증거가 있는 단계까지만 완료로 표시한다.**

## 13. 동봉 자료

- `TableA_7_Case_Templates.csv`: 7개 조건의 고정 입력. 결과 수치 없음.
- `First3Cycles_PerServer_42Runs_PREVIEW.csv`: s4·s5 각 최초3 cycle의 ID·seed·순서·BASE 연결. 42개로 종료하라는 뜻이 아님.
- `campaign_spec.json`: 계수, Teacher/data identity, selector, 반복, 전환 규약의 기계 판독 명세. **기존 main.py에 그대로 넣는 executable config가 아님.**
- `spec_validation.json`: 7조건/OAT/50후보/seed·ID·BASE 연결 등 정적 명세 검증 결과.

동봉 검증은 **명세 일관성 18항목**이다. GPU 수치 동등성, checkpoint resume, 실제 중단·기동, 새 trainer의 학습 성공을 검증한 테스트가 아니다.

## 14. 근거와 새 설계의 구분

**Source에서 확인한 것:** 메인 Ours identity·수치·Teacher/calibration/data metadata, 기존 α·β·λE 중심값과 원 recipe, 기존 G23의 P0/T0 범위와 안전 중단 CLI.

**이번 사용자의 지시 및 이 문서에서 확정한 것:** Table B 제외, 메인 PLH/F1로 기준 전환, HQNR_MAX50 주 선택, s4·s5 모두 7조건 실행, 새 seed 배정·순서, 새 namespace/tab 및 즉시 안전 전환 절차. 아직 새 학습 결과는 없다.

### Sources

- **[S1] Live workbook / 이번 재조회:** `paper!B17:N17`, `유의미한결과!B16:U16`의 값·notes, `WV3-s1!FP3:HI3`와 `FP96:HH96`. https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit
- **[S2] 직전 검토 MD:** `WV3_Main_Ours_Hyperparameter_Review_and_Comparison_Plan.md`, §§1, 4–5. 동봉된 앞선 13조건에서 Table A 7조건만 유지하고 selector·server 분담을 이번 지시로 교체했다.
- **[S3] Original main builder:** https://github.com/hojunking/PAN-Crafter-repro/blob/cccedeeeffd7ed19686e5ea684489ceee23cf313/fh20r1/plan.py — `build_config`, original init policy / F1 transfer / α·β·λE / scheduler. 이번 직접 fetch에서 blob SHA `9772235a8c9b1a63048feabcfc381b1636653da9` 확인.
- **[S4] Inherited optimizer/feeder/model defaults:** https://github.com/hojunking/PAN-Crafter-repro/blob/cccedeeeffd7ed19686e5ea684489ceee23cf313/fh12/plan.py — 직전 source review에서 확인한 기준 builder. 실제 배포에서는 동일 frozen revision을 재검증한다.
- **[S5] 제공된 Method:** `PAN_Final_Method_qe_AlignmentAware_Fitting_2026-09-20_KR_v4_notation.md`, §§5.1–5.4, 변수 대응표. hard·soft·edge coefficient와 분리 gradient 경로의 정의.
- **[S6] 기존 G23 운영 문서 / 이번 직접 fetch:** https://github.com/hojunking/PAN-Crafter-repro/blob/main/g23sens/README.md — blob SHA `9b5aee925eb833b02ffe3a455c33e015a10a86eb`.
- **[S7] 기존 stop/status CLI / 이번 직접 fetch:** https://github.com/hojunking/PAN-Crafter-repro/blob/main/tools/g23sens_runner.py — blob SHA `2b6cde90635e8545cdddad955f9e5f4e18676ee4`. `ROOT`에 따른 state 경로, `STOP_AFTER_RUN`, `STOP_NOW_SAFE` 분기 확인. 실제 실행 release가 동일 기능을 지원하는지 별도 확인한다.

---

### 다음 실행 세션에 전달할 짧은 지시문

> s4·s5의 현재 G23 sensitivity를 이 문서 §7 절차로 안전 중단·보존하고, 메인 Ours PLH/W104D122 + F1 기반 MAIN-A campaign으로 전환하라. Table A α·β·λE만 변경하고 Table B는 실행하지 마라. 모든 case는 fresh50K이며 동일한 후보50개 중 FR20 raw-original HQNR 최대 checkpoint를 선택하고 RR/FR는 같은 A/U checkpoint로 보고하라. ERGAS는 선택에 쓰지 않는다. 각 서버에서 BASE+6변형을 같은 초기화·stream으로 완료한 뒤 새 seed cycle을 무기한 반복하라. 공유 lock이나 다른 서버의 결과 대기는 금지한다. F1/T0·PLH/P0·D122/D121을 혼용하지 마라. 실제 기동 여부는 PID·첫 update·config/asset SHA로 확인하고, 작성된 명세를 실행 완료로 보고하지 마라.
