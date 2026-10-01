# PANDA 후속 ablation 실행 명세 — s1·s3·s5 각 4회 / case당 12개 Student

작성일: **2026-10-01**  
Campaign: **`PANDA_REBUTTAL_B02_M12_WV3_S135_20261001_v1`**  
대상: **WV3**, 서버 **s1·s3·s5**, 독립 Student seed **서버별 4개**  
문서 상태: **구현·실행 지시서. 이 파일 작성 단계에서는 서버 기동·코드 배포·Sheet 변경을 수행하지 않았다.**

## 0. 실행 요약 — 먼저 읽을 내용

이 작업은 제출본을 수정하는 작업이 아니다. 제출본의 설명에 대응하는 질문을 **q의 sample 정보, geometry 경로, hard 강조, selective soft gate**로 나누어, 반복 결과의 방향과 조건을 확인한다. 최고 점수를 얻을 때까지 seed를 바꾸는 것이 아니라, 사전에 정한 모든 대조를 같은 12개 seed block에서 완료한다.

**이번 실행 범위는 준비 작업 + 두 학습 step이다.** 기존의 RB04(input/feature alignment 위치 대조), 새 Teacher consistency 실험, QB/GF2 확장, 광범위 sensitivity는 이번 자동 실행에 포함하지 않는다. `STEP1/STEP2`는 이 캠페인의 실행 순서이며, 논문 Stage I/II 또는 기존 RB04의 의미를 바꾸지 않는다.

| 순서 | 작업 | 학습량 | 실행 여부 |
|---|---|---:|---|
| 준비 | 기존 B01/ABLR2 결과 회수, 자산 검증, shift response 사전 진단 | 새 학습 0 | 먼저 수행. 미수집 서버 때문에 다른 서버를 무기한 대기시키지 않음 |
| **STEP1_QROUTE** | QFULL/QMEAN/QSHUF/QESUR/QEDGE/QALIGN | **6 case × 12 = 72 run** | 기술 검증 통과 후 실행 |
| **STEP2_FITTING** | H0/HSPMEAN/NOADV/ADVMEAN | **4 case × 12 = 48 run** | STEP1과 같은 초기값·seed를 사용하여 이어 실행 |
| 마무리 | native, paired 통계, 진단, 업로드 검증 | 추가 학습 0 | 완료 후 `STOP_FOR_REVIEW` |

**총 fresh Student 120개, 서버당 40개, 각 50,000 update.** 서버별 2,000,000 update, 전체 6,000,000 update다. 학습시간 제한은 두지 않지만 **run 수는 유한**하며, 종료 후 무한 cycle을 시작하지 않는다. STEP1 결과가 기대와 다르다는 이유로 STEP2를 취소하거나 변형하지 않는다.

### 0.1 4회는 “추론 4번”이 아니라 “다른 seed로 새 학습 4번”이다

이전 B01 R1/R2는 그대로 보존한다. 이번에는 **새 seed 4개/서버로 10개 case를 공통으로 비교**한다. 이전 결과 2개에 좋은 신규 결과만 골라 붙여 4개를 만드는 방식이 아니다. 원 B01은 별도 exploratory/역사적 cohort로 보고하고 이번 M12는 새 Student-seed cohort로 보고한다. 단, 같은 기존 test 장면을 쓰므로 새로운 untouched test-set 검증이라고 부르지 않는다.

새 캠페인의 자기 run이 이미 완료되어 있으면 identity 검증 후 재사용한다. 같은 신규 run을 재시작하여 다른 결과를 만들지는 않는다. 기존 B01을 현재 4회 registry로 확장하거나 원 run ID를 재사용하지 않는다.

## 1. 근거와 이번에 새로 정한 사항

### 1.1 원고에 명시된 기준

제출본은 reconstruction reliability와 geometric reliability를 서로 다른 감독에 배분한다(p2, p4). Eq.(9)는 pixel-wise hard 강조와 trust/advantage 기반 soft fitting을, Eq.(10)–(12)는 sample-wise q weight와 U/A 분리 업데이트를 정의한다(p6–7). **LPAN/HPAN 입력은 reliability weighting과 독립적인 설정**이다(p6). 이 구조를 비교의 기준으로 유지한다. [S1]

### 1.2 기존 분석에서 해결되지 않은 질문

2026-10-01 분석에서는 q가 mean weight보다 유리한 B01 결과가 있지만 shuffle/e-surrogate 대비 고유한 이득은 확정되지 않았다. ABLR2에서는 hard 강조와 adaptive fitting의 RR 악화가 관찰되어 공간적 가중 배분과 총 감독 강도의 영향을 분리할 필요가 있다. Controlled shift는 correction 사용의 이득과 shift에 대한 실제 반응을 구별해야 한다. [S2]

이 문서의 **10개 case, 12개 seed, 두 실행 단계, HSPMEAN/ADVMEAN의 구체식, 추가 inference modes, fixed160 진단 ROI**는 이번 새 protocol이다. 원 논문이나 기존 완료 실험에 이미 포함되어 있던 설정이라고 기록하지 않는다.

### 1.3 논문·기존 결과 보호

`paper`, 이름이 정확히 `ablations`인 논문용 탭, 기존 제출값, `유의미한결과`의 현재 수동 검토표를 자동으로 수정하지 않는다. 원본 B01/ABLR2 JSON·CSV·NPY·checkpoint·source hash도 변경하지 않는다. 이번 결과는 **`WV3-ablations`의 별도 M12 그룹**으로 보낸다. [S3]

## 2. 고정 자산과 재현 조건

| 항목 | 고정값/정책 |
|---|---|
| 데이터셋 | WV3, 동일 train/validation/RR20/FR20 manifest와 band/order/LP identity |
| Student U | **PLH, W104, depth [1,2,2], LN, attention off, mode modulation off** |
| Reference F1 | `FH12_S1_T_P0_W112_D123_WV3_S71001_FRESH50_v1`, **exact50K** |
| 원 설정 anchor | `FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1`의 config |
| 초기 U | 새 random initialization. 동일 block의 10개 case가 같은 state를 복사 |
| 초기 A | F1 Aligner clone. case별 독립 복사 후 trainable |
| Teacher | 동일 F1, eval + requires_grad=False. 새 Teacher 학습 없음 |
| 학습 | fresh50K, batch48, FP32, AdamW |
| Optimizer | U LR=1e-4, A LR=3e-6, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01 |
| Schedule | 기존 warmup100 + cosine 구현 유지 |
| 기본 계수 | α=1, β=0.1, λE=0.002. 등록한 H0만 hard α=0 |
| 수치 안정화 | advantage 분모의 δrec=1e-6, 기존 normalized 학습 단위 유지 |
| Calibration | **실제 F1 binding의 q_ref와 τrec**. G23의 숫자로 대체 금지 |
| Augmentation | 원 B01/FH12의 64/16 입력, BatchStream와 실제 view 규칙 그대로. 임의 random flip/crop 교정 금지 |
| Selector | EXACT_50000 primary; RR_VAL_ERGAS_MIN secondary |
| Validation grid | **1010, 2020, …, 49490, 50000**. 1000 간격으로 바꾸지 않음 |

위 optimizer·grid·gradient 기준은 현재 `panda_rb/plan.py`, `panda_rb/training.py`, `fh12/losses.py`의 실제 구현에서 확인했다. [S4–S6]

### 2.1 반드시 서버에서 바인딩할 값

이 파일은 원격 디스크의 현재 경로나 checkpoint bytes를 인증한 문서가 아니다. `planning/binding.template.json`의 미확정 항목은 **실제 기존 B01 binding을 읽어** 채운다.

- F1 전체 checkpoint SHA256 및 source/config/reference manifest.
- q_ref, τrec, raw q/e_bar cache SHA, 전체 train×view 대응과 calibration identity.
- 원본 H5/LP bytes, train/val/test 분할 및 scene 순서.
- 실제 frozen numerical runtime와 evaluator identity, Docker image digest, GPU와 precision 정책.
- 12개 block의 초기 U/A state SHA와 50K consumed sample/view stream SHA.

`q_ref`/`τrec`/Teacher SHA가 null인 상태에서는 학습하지 않는다. 다른 F1이나 T0를 찾아 빈 값을 채우지 않는다. 경로만 서버에 맞게 바꾸는 것은 가능하지만, bytes와 내용 identity를 검증해야 한다.

**이 설계의 n=12는 고정 F1에 대한 Student 반복이다. 12개 독립 Teacher–Student 전체 파이프라인 반복이 아니다.**

## 3. seed·block·순서

| 서버 | R1 | R2 | R3 | R4 | 담당 |
|---|---:|---:|---:|---:|---|
| s1 | 261001101 | 261001102 | 261001103 | 261001104 | 10개 case 전부 |
| s3 | 261001301 | 261001302 | 261001303 | 261001304 | 10개 case 전부 |
| s5 | 261001501 | 261001502 | 261001503 | 261001504 | 10개 case 전부 |

각 block은 `(server, repeat, Student seed)`로 식별한다. 같은 block의 10개 case는 **초기 U, 초기 A, data/view stream, Teacher, optimizer, update budget**이 같아야 한다. 학습된 이전 case에서 이어 시작하지 않는다. QSHUF permutation과 진단 RNG는 sampler/model RNG와 분리한다.

STEP1의 6개 case 순서는 12개 block에 대해 cyclic rotation으로 배정하여 각 case가 각 순서에 두 번씩 배치되게 했다. STEP2의 4개 신규 case도 각 순서에 세 번씩 배치된다. 정확한 순서는 `planning/server_schedule.csv`를 따른다. case 이름의 알파벳 순서나 결과 순서로 재정렬하여 실행하지 않는다.

서버는 자신의 STEP1 24개 run을 완료한 뒤 STEP2 16개 run을 수행한다. 다른 서버의 완료는 기다리지 않는다. **STEP1의 QFULL이 STEP2의 shared baseline**이므로 STEP2에서 QFULL을 다시 학습할 필요가 없다. STEP2도 그 block의 원 초기 U/A에서 시작한다.

## 4. 공통 수식과 gradient contract

**명칭 주의:** `NOADV`/`ADVMEAN`의 ADV는 reference advantage gate를 뜻한다. Aligner를 끄는 case가 아니다.

이하 모든 error는 기존 학습과 같은 normalized 단위로 계산한다. 평가용 DN/clamp 수식을 학습 loss에 섞지 않는다.

\[
e_S(n,p)=\operatorname{mean}_c|\hat Y_S-Y|,\quad
 e_T(n,p)=\operatorname{mean}_c|\hat Y_T-Y|,
\]
\[
d=\operatorname{sg}\left(\frac{e_T}{e_T+\tau_{rec}}\right),\quad
 t=1-d,\quad
 a=\operatorname{sg}\left(\frac{\max(e_S-e_T,0)}{e_S+10^{-6}}\right),
\]
\[
\Delta_{TS}=\operatorname{mean}_c|\hat Y_S-\hat Y_T|,\quad
 s_n=\frac{q_{ref}}{q_{ref}+q_n},\quad h=1+\alpha d.
\]

case에서 지정한 `h_case`, `a_case`, `w_E`, `w_A`로

\[
H_n=\operatorname{mean}_p[h_{case}(n,p)e_S(n,p)],\quad
K_n=\beta\operatorname{mean}_p[t(n,p)a_{case}(n,p)\Delta_{TS}(n,p)],
\]
\[
L_U=\operatorname{mean}_n[H_n+K_n+\lambda_Ew_{E,n}E_n],\quad
L_A=\operatorname{mean}_n[w_{A,n}H_n].
\]

`E_n`은 기존 `output_edge_loss_per_sample`의 signed gradient loss다. 새 magnitude-only edge나 다른 padding으로 바꾸지 않는다.

**실제 적용 gradient는 `grad(U)=∂L_U/∂U`, `grad(A)=∂L_A/∂A`다.** 하나의 `total.backward()`로 대체하지 않는다. Teacher, GT, d, a, h의 재배분 계수, q/geometry weight는 detached이며 live eS와 ΔTS에만 해당 loss의 gradient가 흐른다. **U loss의 계산 그래프에 A가 포함되어 있어도, A에 적용할 gradient는 L_A에서만 계산한다.** 두 gradient를 구하기 전에 optimizer.step하지 않는다. β와 λE는 한 번만 곱한다. [S1, S6]

## 5. STEP1 — reliability cue와 q 경로 분리 / 72회 학습

| Case | w_E | w_A | Hard / Soft | 비교 질문 |
|---|---|---|---|---|
| **QFULL** | 원 s | 원 s | 기존 식 | 전체 방법의 공통 baseline |
| **QMEAN** | s̄ | s̄ | 기존 식 | sample별 정보 없이 평균 강도만 필요한가? |
| **QSHUF** | s의 조건부 permutation | 같은 permutation | 기존 식 | sample와 q의 대응 자체가 중요한가? |
| **QESUR** | e-ranked s | 같은 e-ranked s | 기존 식 | geometry에 별도 q 대신 e로 충분한가? |
| **QEDGE** | 원 s | s̄ | 기존 식 | sample q의 이득이 U edge 경로에 있는가? |
| **QALIGN** | s̄ | 원 s | 기존 식 | sample q의 이득이 A hard 경로에 있는가? |

### 5.1 각 weight의 정의

- **s̄**: 원 F1 raw q에서 계산한 s의 **전체 training sample×실제 4개 view 평균**. calibration subset의 median이나 임의 0.5로 대체하지 않는다.
- **QSHUF**: 기존 B01 구현을 재사용한다. 같은 augmentation view와 train e_bar quintile 안에서 sample 대응을 순열화하고, 그룹 내 s의 multiset을 정확히 보존한다. 가능한 그룹에서는 자기 자신과 연결하지 않는 permutation을 사용한다. 각 block에서 별도 RNG/seed를 고정하며 singleton 처리도 기록한다.
- **QESUR**: 기존 B01의 e_bar 순위와 s의 값들을 대응한다. 낮은 teacher e_bar에 큰 geometry weight를 주며 원 구현의 view/동점 처리와 주변분포 보존 규칙을 그대로 유지한다. 원 pixel-level d/a는 바꾸지 않는다.
- QEDGE/QALIGN은 **해당 경로를 삭제하는 실험이 아니다.** sample별 weight를 train 평균으로 대체한다. loss 계수·A LR로 이를 보상하지 않는다.

QSHUF의 차이는 q와 해당 sample의 연결에 조건부 정보가 있는지 묻는 대조다. 모든 q 분포를 완전히 독립 난수로 바꾸는 실험과 다르다. QESUR 역시 모든 가능한 reconstruction cue의 대표가 아니라 **이번에 명시한 e-based 대체**다. [S7]

### 5.2 만들 결과표

**표 Q1:** QFULL/QMEAN/QSHUF/QESUR, 각 n=12의 HQNR/Ds/Dλ/ERGAS/SAM/PSNR 평균±SD.

**표 Q2:** QMEAN, QEDGE, QALIGN, QFULL의 2×2 경로 표. 같은 block에서 다음 효과를 계산한다.

\[
\Delta_E=\tfrac12[(QEDGE-QMEAN)+(QFULL-QALIGN)],
\]
\[
\Delta_A=\tfrac12[(QALIGN-QMEAN)+(QFULL-QEDGE)],
\]
\[
I_{EA}=QFULL-QEDGE-QALIGN+QMEAN.
\]

위 식은 각 metric에 대한 결과 차이다. HQNR는 양수, ERGAS/Ds/Dλ는 음수가 개선이다. interaction이 있다는 사실만으로 FULL이 가장 좋다고 해석하지 않는다.

## 6. STEP2 — hard/soft 원인 분리 / 신규 48회 학습

모든 case에서 geometry는 `w_E=w_A=s`, β=0.1, λE=0.002를 유지한다. QFULL baseline과 같은 12개 seed·초기값을 사용한다.

| Case | 바꾸는 항목 | 유지하는 항목 | 질문 |
|---|---|---|---|
| **H0** | hard에서 α=0 → h=1 | soft의 d와 a, q, edge 그대로 | hard 강조 자체가 도움이 되는가? |
| **HSPMEAN** | h를 **sample 안의 공간 평균** h̄_n으로 대체 | sample별 hard weight 총량, soft/q/edge | 어려운 pixel에 배분하는 것이 총량보다 중요한가? |
| **NOADV** | soft에서 a=1 | trust t=1-d, β, hard/q/edge | Student 대비 advantage gate를 없애면 어떻게 되는가? |
| **ADVMEAN** | a를 **trust-weighted 공간 평균** ā_n으로 대체 | 현재 forward의 effective soft-weight 총량, trust/hard/q/edge | gate의 공간적 선택성이 총량보다 중요한가? |

### 6.1 HSPMEAN — per-sample mass matched

\[
\bar h_n=\operatorname{sg}\left(\operatorname{mean}_p[1+d(n,p)]\right),\quad
H_n^{HSPMEAN}=\bar h_n\operatorname{mean}_p e_S(n,p).
\]

원 h와 대체 h는 같은 sample에서 \(\sum_p h\)가 같다. 따라서 이 대조는 **sample 사이의 평균 감독 강도를 유지하면서 pixel별 hard 배분만 제거**한다. global train 평균 1개로 대체하거나 h를 평균으로 나누는 실험이 아니다.

U와 A에 사용되는 H_n을 함께 이 식으로 바꾼다. `d`는 soft trust에 원래대로 남고, `a`도 원래대로다. matching은 weight 총량에 대한 것이며 weighted error 또는 실제 gradient/AdamW update 크기까지 동일하게 만든다는 뜻은 아니다.

### 6.2 ADVMEAN — trust-weighted advantage mass matched

\[
\bar a_n=\operatorname{sg}\left(
\frac{\sum_p t(n,p)a(n,p)}{\sum_p t(n,p)}\right),\quad
K_n^{ADVMEAN}=\beta\operatorname{mean}_p[t(n,p)\bar a_n\Delta_{TS}(n,p)].
\]

`sum(t)==0`이면 해당 sample의 soft를 0으로 둔다. 양수이면 원 비율을 계산하고 finite를 확인한다. 이로써 **같은 forward에 대해** \(\sum_p t\bar a_n=\sum_p ta\)가 성립한다. ā는 broadcast된 상수이며 detach한다. a만 단순 산술평균하면 t와 결합된 총량이 보존되지 않으므로 사용하지 않는다.

ADVMEAN의 계수는 **해당 ADVMEAN Student의 현재 eS**로 계산한다. 별도로 학습한 QFULL 모델을 매 update 따라가는 teacher-forcing 또는 외부 trace는 사용하지 않는다. 따라서 서로 다른 run의 전체 학습 trajectory에 걸친 soft 총량까지 동일한 것은 아니다.

NOADV는 a=1이므로 effective KD 강도가 커질 수 있다. **NOADV만으로 gate의 위치 정보가 유효하다고 단정하지 않고 ADVMEAN과 함께 해석**한다. β를 사후 tuning하여 맞추지 않는다.

### 6.3 만들 결과표

**표 H:** QFULL/H0/HSPMEAN — hard 강조의 존재, 평균 강도, 공간적 배분을 구분한다.

**표 K:** QFULL/NOADV/ADVMEAN — advantage gate의 존재와 총량을 통제한 위치 선택을 구분한다.

기존 ABLR2의 C12/C13/C05와 동일한 이름의 효과처럼 합쳐 평균내지 않는다. 이번 H0 등은 fixed-F1 M12 cohort의 새 측정이다. Soft 전체 삭제나 hard×soft 전 요인 조합은 이번 실행 범위 밖이다.

## 7. 준비 작업 — 결과 회수와 Aligner 진단

### 7.1 기존 완료 자료 우선 회수

`s1/s3/s5`에서 기존 B01의 frozen `report`와 evidence package를 회수한다. 원 B01의 기대량은 **서버당 Student8/native16/curve16** 그대로다. 본 M12의 12-repeat 수량으로 고치지 않는다.

`유의미한결과!B330:U332`의 미수집 목록과 실제 파일을 대조한다. WV3 ABLR2의 P03/C14와 P05/C10–C15는 **과거 snapshot의 누락 후보**이며, 실제로 완료 파일이 있는지부터 확인한다. 자동으로 새 seed를 추가하지 않는다. 원등록 학습까지 없는 항목은 `OLD_SLOT_NOT_FOUND`로 반환하고, 이번 120개 실행과 별개의 보완 목록으로 둔다.

### 7.2 새 학습 전 response-only 검사

원 F1 Aligner, 각 block의 초기 A clone, 회수 가능한 기존 B01 최종 A에 대해, 알려진 ε를 주었을 때

\[
R(\epsilon)=\|c_\epsilon+\epsilon-c_0\|_1
\]

을 기록한다. 원 F1 1개를 여러 서버에서 추론했다고 여러 Teacher seed로 세지 않는다. 초기 A clone은 parameter와 eval state가 동일하면 F1 A와 같은 입력에서 일치해야 한다.

회수한 RR20에서 size64/128은 center+네 모서리, size256은 전체를 사용하는 **고정 crop manifest**를 먼저 저장한다. 이 검사는 aligner의 response만 비교하며, 작은 crop의 RR reconstruction score를 full256 지표에 합치지 않는다. MS upsampled reference와 PAN은 같은 위치를 crop하고 padding/warp 부호/scale 단위를 기록한다.

의미 있는 shift cancellation이 약하다는 이유만으로 setup 실패나 새 Teacher 선택을 하지 않는다. **부호 오류, 잘못된 checkpoint, 틀린 입력 차원, source 불일치**는 기술적 중단 사유지만, 약한 response는 측정 결과다.

## 8. STEP1 모든 모델의 추가 inference — 같은 12개 모델, 새로운 학습 0

STEP1의 72개 모델을 각 4개 mode로 평가한다. **288개 model-mode curve**, server당 96개다. 각 curve는 49 shift×RR20장이다. STEP2의 네 fitting 변형에는 이 stress를 기본으로 추가하지 않으며 native와 training diagnostics까지 수행한다.

| Mode | 적용할 correction | 분리하는 효과 |
|---|---|---|
| **A_ON** | cε | 정상적으로 재추정한 correction |
| **A_ZERO_INFERENCE_ONLY** | 0 | correction의 전체 사용 효과 |
| **A_NATIVE_FIXED** | c0 | native pair에서 얻은 correction을 고정한 효과 |
| **KNOWN_SHIFT_INVERSE** | c0−ε | 알려진 추가 shift만 상쇄하는 참조 |

`KNOWN_SHIFT_INVERSE`는 native 정합 GT를 아는 절대 oracle이 아니다. 보간·LP 재생성에 의한 잔차는 남는다. `A_ON−A_NATIVE_FIXED`가 새로운 shift에 반응한 correction의 기여를 구분하고, `A_ON−A_ZERO`는 native correction 사용 효과까지 포함한다.

### 8.1 고정 grid와 입력

`planning/shift_grid.json`: D000=0, 반경 `{0.25,0.5,1,2,3,4}` HR pixel × 방향 `{0,45,...,315}` 총49점. 좌표는 `(dy,dx)`. 원 MS/GT는 고정, PAN을 한 번 shift한 뒤 **그 shifted PAN에서 원 frozen recipe로 LP를 재생성**한다. 이후 PAN/LP/HP 전체 synchronized frontend에 mode별 correction을 적용한다. stale LP 또는 PAN만 bypass하는 mode를 만들지 않는다.

D000에서는 원 native 입력을 그대로 재사용한다. zero shift에서 A_ON, A_NATIVE_FIXED, KNOWN_SHIFT_INVERSE가 동일한 forward/출력을 내는지 검사한다. test GT로 correction을 최적화하지 않는다.

### 8.2 ROI와 coverage — 기존 실패를 덮어쓰지 않는다

기존 RB02의 fixed192(`32:-32`) 결과와 radius4 support 경고는 보존한다. **새 M12 primary stress는 사전에 정한 fixed160(`48:-48`)**으로 측정하고, 같은 full256 prediction에서 fixed192를 auxiliary로 함께 계산한다. crop 크기는 결과 score를 보기 전에 고정한다. 이 신규 ROI는 기존 B01과 다른 protocol로 표시한다.

모든 case·seed·mode·shift에 같은 ROI를 적용한다. correction이 크면 fixed160도 support가 부족할 수 있으므로 안전성을 가정하지 않는다. PAN과 LP/HP 경로의 실제 sampling/filter support를 검사하고 `coverage<1`, nonfinite, signed-edge 오류를 남긴다. U의 전체 receptive field까지 보장했다고 표현하지 않는다.

Support 부족으로 ROI를 장면마다 줄이거나 correction을 clamp하지 않는다. raw point와 metric은 보존하되 clean aggregate와 flagged aggregate를 구분한다. clean 반경 요약은 그 Student에서 **모든 예정 방향·장면이 유효한 경우만** 형성하며, 일부 방향을 조용히 버려 평균내지 않는다. `N_valid/N_expected`, failure 수, flagged 전체 요약도 같이 보고한다. 임의 ROI 변경이 필요하면 v2로 따로 승인·등록하고 모든 대조에 동일하게 적용한다.

### 8.3 저장과 진단

per-scene: ERGAS/PSNR/SAM/signed-edge error, c0/cε/applied correction, response L1 sum, coverage, prediction digest, scene/shift ID, ROI/protocol ID.

서로 다른 reduction을 혼동하지 않도록 원 q의 coordinate reduction과 신규 response의 `abs(dy)+abs(dx)`를 명시한다. q는 frozen Teacher의 상대 consistency cue이지 절대 registration error가 아니다. [S1]

반경 요약은 **scene → 해당 Student의 방향 평균 → 12 Student 평균/SD** 순서다. mode·방향·scene를 seed 반복으로 세지 않는다. 전체 raw 영상 저장 대신 사전 지정 scene `{0,10}`·shift `{D000,D017,D042}`와 최초 실패 8개/curve를 저장하고, 모든 다른 point의 scalar·digest·재생성 identity는 보존한다.

## 9. 학습 중 반드시 남길 진단

원 B01과 같은 고정 probe와 RNG 보존 규칙을 사용한다. 상세 진단은 update `{0,1000,10000,25000,50000}`, 경량 log는 매100 update를 기본으로 한다. probe가 optimizer·sampler RNG, buffer, .grad 또는 train/eval 상태를 바꾸지 않아야 한다.

| 진단 | 필요한 값 |
|---|---|
| 감독 강도 | hard/soft/edge의 raw/weighted 값, mean(h), mean(βta), mean(wE), mean(wA) |
| Gate 동작 | a>0 비율, eS<eT 비율, d/a/weight quantile, sample·pixel 분포 |
| 경로 | U/A별 실제 배정 gradient norm, term별 diagnostic gradient, 실제 optimizer update norm |
| Geometry | 초기·최종 A의 correction/drift/response, wE와 wA 실제 consumed 값 |
| Matching | HSPMEAN의 per-sample hard 총량 보존, ADVMEAN의 per-sample trust-weighted soft 총량 보존 |
| 순서 | 실제 consumed sample/view stream digest, resume 전후 이어짐 |

가능하면 term-gradient cosine도 고정 probe에서 기록한다. 이는 loss 간 최적화 충돌을 진단하는 보조값이며, 하나의 gradient norm이 크다는 사실을 바로 일반화 성능 기여로 해석하지 않는다.

## 10. 평가·반복·해석 규칙

### 10.1 정량평가

모든 120 run의 native RR20·FR20을 **EXACT_50000**에서 평가한다. validation winner는 앞서 고정한 grid에서 ERGAS 최소, 동률이면 낮은 step이다. 두 selector의 SHA가 같으면 alias로 표시하며 서로 다른 Student로 세지 않는다. 가능하면 두 selector 모두 raw 및 per-scene metrics를 저장한다.

RR은 원 evaluator의 border/band/normalization 규약, FR은 WV3 sensor/MTF를 그대로 사용한다. stress의 fixed160/192 metric은 native RR 값과 같은 열·평균에 혼합하지 않는다. checkpoint를 FR HQNR로 선택하거나 모드별 best checkpoint를 고르지 않는다.

### 10.2 주 분석과 보조 분석

- 주 분석: **M12의 새 12개 seed**만, fixed-F1/동일 numerical/evaluator/data cohort.
- 보조 분석: 서버별 4개, 기존 B01 6개 계획 슬롯의 회수 가능한 결과, 기존 ABLR2. 서로 합산하지 않는다.
- 차이는 같은 block의 `case−QFULL` 및 §5.2의 factorial contrasts로 계산한다. 교집합 n을 명시한다.
- mean/sample SD(ddof=1), paired delta, 95% paired CI, seed별 개선/동률/악화 수를 모두 남긴다. 장면 수를 n으로 사용하지 않는다.
- CI는 사전에 고정한 분석 함수로 계산한다. 기본은 seed-level paired t interval(df=n−1; 분포 가정 명시), 보조는 서버별 4개 block을 resample하는 stratified paired bootstrap(10,000회, seed20261001)이다. bootstrap도 새 관측을 만드는 것은 아니다.
- 순수한 server/hardware 효과는 seed와 교차되어 있지 않으므로 추정하지 않는다. server별 결과는 seed cohort의 일관성 점검이다.
- n<12 또는 기술 실패/수치 발산이 있으면 완료 수·유효 수·사유를 같이 보고한다. 나쁜 run을 새 seed로 교체하지 않는다.

Formal significance를 쓰려면 primary contrasts와 endpoint family를 먼저 고정한다. 이 계획의 primary 정보성 대조는 **QFULL−QSHUF, QFULL−QESUR, QFULL−HSPMEAN, QFULL−ADVMEAN**, endpoint는 HQNR와 ERGAS의 8개 조합이다. p-value를 보고할 때는 이 family의 Holm 보정을 함께 보고하고, 나머지 계획 대조는 effect-size/CI 중심의 보조로 구분한다. CI가 0을 포함한다는 이유만으로 동등함을 주장하지 않는다.

### 10.3 결과에 따른 서술 범위

| 관찰 | 가능한 설명 | 바로 주장할 수 없는 내용 |
|---|---|---|
| FULL이 MEAN보다 좋고 SHUF보다도 반복적으로 좋음 | sample와 reliability 연결의 효용 | 모든 센서·모든 Teacher에 동일 |
| QEDGE만 개선 | U edge 경로에서 sample q가 더 유효 | A refinement까지 필수 |
| QALIGN만 개선 | A update 경로에서 sample q가 더 유효 | edge의 q도 반드시 필요 |
| HSPMEAN이 FULL과 비슷하거나 우세 | hard 총량/배분을 다시 해석할 필요 | 어려운 pixel 강조가 입증됨 |
| ADVMEAN이 FULL과 비슷함 | advantage 위치 선택의 추가 이득 불확실 | gate가 불필요하다는 확정 |
| E surrogate가 FULL과 경쟁적 | 해당 fixed-F1 조건에서 대체 가능성 | q의 보편적 불필요성 |
| 특정 축만 개선 | RR/FR 또는 route별 조건부 효과 | 전 지표 일괄 우위 |

**12회는 더 정밀한 추정을 위한 예산이지, 기대한 방향의 결론을 보장하는 장치가 아니다.** 전체 완료 후 지지/조건부/미확립을 판정한다. 해석이 약하다고 같은 설정의 반복을 무한히 추가하지 않고 다음 revision의 가설을 명시한다.

## 11. 구현 지시 — B01 frozen run을 바꾸지 않는다

현재 B01 코드는 4 case, 서버당2 seed, 총24 Student/48curve와 고정 namespace를 검증한다. 그 registry/validator를 소급 확장하면 기존 evidence 검증을 깨뜨릴 수 있다. [S4]

권장 신규 위치:

```text
panda_rb_m12/                  # 새 training/plan/loss/evaluation/controller
    plan.py
    losses.py
    training.py
    evaluation.py
    controller.py
    reporting.py

tools/rb_m12_runner.py         # 새 명령; 기존 panda_rb_runner.py는 그대로
reporting_bridge/rb_m12_*.py   # 새 reporting adapter/contract

work_dir/_panda_rb/20261001/B02_M12/
    common/                   # immutable binding, cache, initial state, stream
    train/<server>/R<1..4>/<case>/
    control/<server>/
    reporting/outgoing/<server>/
```

기존 `panda_rb/*`, `tools/panda_rb_*`, `fh12`/`fh20r1`의 frozen 파일을 in-place 편집하지 않는다. 새 loss adapter에서 필요한 intervention만 구현하고, 원 수치 함수는 원 release에서 재사용한다. 새 campaign의 source identity는 **새 모듈과 실제 호출된 기존 파일을 모두 포함**하는 독립 manifest로 만든다. 원 B01의 numerical source SHA를 새 코드의 SHA인 것처럼 복사하지 않는다.

code source hash가 모든 case에서 같고, 의도한 intervention/config/map만 다르게 해야 한다. case별 의도된 config 차이를 이유로 paired 비교를 전부 다른 cohort로 분할하거나, 반대로 Teacher/runtime/data 차이를 의도된 case 차이로 숨기지 않는다.

## 12. gspread 등록 — 현재 데이터셋별 ablations 정책 준수

대상 Spreadsheet ID: `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`. 공개 결과는 **`WV3-ablations`(gid261001010)**, 64열 기존 양식으로 표시한다. [S3]

1. 원 B01의 `_rb01_s1/s3/s5` 16개 slot, registry, receipt를 덮어쓰거나 M12 관측을 거기 끼워 넣지 않는다.
2. 새 hidden source `_rb_m12_s1`, `_rb_m12_s3`, `_rb_m12_s5`를 **80개 native 관측/서버 + 헤더** 규격으로 한 번만 준비한다. schema/version/소유권 확인 후 재사용한다. 새 공개 탭은 만들지 않는다.
3. `Dataset=WV3`, `Bucket=Ablations`, `Approach=06 | Rebuttal M12 | 01 q cue and routing` 또는 `06 | Rebuttal M12 | 02 hard-soft fitting`을 명시한다. `06 | `는 현 routing 규칙이 인식한다.
4. `_records!A2`의 기존 parser/B01 union/AA routing을 보존하면서 **M12 hidden source union만 최소 추가**한다. 최신 live formula를 snapshot/diff한 후 적용한다. 과거 전체 수식을 복원하지 않는다.
5. native Result_ID는 campaign/run/selection/checkpoint/protocol을 포함하고, 같은 run의 여러 관측을 Student 반복으로 세지 않는다. Source_URL은 실제 source row를 가리킨다.
6. stress는 새 hidden `_rb_m12_points`에 별도 protocol/schema로 보존한다. primary fixed160의 최대14112 point만 online 표의 기본으로 사용하고 fixed192 auxiliary는 sidecar CSV/JSON으로 연결한다. 이 값들을 64열 native RR/FR에 넣지 않는다.
7. 기존 `RB02-curves`를 확장할 때는 Campaign/ROI/Mode 구분을 보존한 별도 M12 구역/union으로 읽게 한다. 기존 B01 구역·43열 의미·관측은 변경하지 않는다. 현재 uploader의 고정24/49/48 guard를 새 protocol에 그대로 적용하지 않는다. 신규 source/표시 영역의 행 용량은 실제 사용량에 맞춰 사전 확장한다(새 primary point만 최대14112행). 기존 2600행 view에 고정해 뒤쪽을 누락하지 않으며, 문서 전체 cell 한도와 수식 성능을 setup 전에 점검한다. extension이 준비되지 않았으면 stress만 upload pending으로 두고 native 학습/업로드를 막지 않는다.
8. 120개 run 상태는 새 hidden `_rb_m12_status`와 로컬 ledger로 관리한다. `paper`, 논문용 `ablations`, `유의미한결과`, `배포용 모델` 자동 수정은 금지한다. 검토용 summary CSV/MD는 반환만 하고 수동 요약표 반영은 별도 요청을 따른다.
9. **s1 단일 writer**가 구조 변경/누적 업로드를 맡는다. s3/s5는 verified package를 승인된 내부 경로로 전달한다. 공유 Sheet에 세 서버가 동시에 setup하지 않는다.
10. source RAW numeric readback → `_records` → `WV3-ablations`의 동일 Result_ID 확인, `WV3-main`에서 부재 확인. 재전송은 신규 관측0이어야 한다. credentials·weights·raw 이미지를 Sheet에 넣지 않는다.

B01 collect/bridge는 본래 B01에만 쓰고, M12용 별도 adapter를 구현한다. 현재 64열의 Repeat=1..4, Seed, Campaign_ID, Case_ID, source/protocol/alias 정보를 보존한다. header slot을 추측해 `append_row()`로 view 아래에 쓰지 않는다.

## 13. 서버 운용·resume·저장 용량

사용자가 지정한 가용 서버는 s1·s3·s5다. 실제 GPU owner/process를 확인한다. s2/s4, 알 수 없는 기존 학습, 배포 작업을 종료하지 않는다. 충돌 시 `WAITING_FOR_GPU_OWNER`로 기록하고 승인된 자원 인계 경로로 대기한다.

각 run은 fresh50K 또는 자기 run의 full-state resume만 허용한다. 모델·optimizer moments·scheduler·RNG·sampler cursor·augmentation stream·actual update를 함께 저장한다. SIGTERM 등 안전 중단은 현재 update 경계에서 저장하고 다음 update부터 이어간다. Weight-only resume를 exact resume로 표시하지 않는다.

기술 오류의 retry는 같은 run/seed와 별도 attempt다. 손실 발산이나 낮은 성능을 기술 오류로 바꾸어 숨기지 않는다. API/upload 장애는 별도 spool debt로 남기고 모델을 다시 학습/평가하지 않는다.

Native raw 배열 본체만 exact120개 모델 전체에서 약 **23.44 GiB**, secondary가 모두 별도 checkpoint면 합계 약 **46.88 GiB**다. 여기에 checkpoint·optimizer 상태·diagnostics·stress raw가 추가된다. 이는 저장 예시 계산이며 원격 free space 측정값이 아니다. 현재 checkpoint 보존 정책을 읽어 실제 예상 용량을 산출하고 디스크 부족이면 시작 전 중단한다. 저장 공간 때문에 원래 증거를 자동 삭제하지 않는다.

학습 runtime은 첫 완료 run에서 train/eval/diagnostic/IO를 나눠 측정하고 남은 동일 workload로 예측한다. 특정 시간 안에 끝난다고 보장하지 않는다. GPU 진단이 장시간 막혀도 승인된 다른 case의 학습은 계속할 수 있게 별도 retry queue를 둔다. 단 최종 완료 시 진단 미완료는 명시한다.

## 14. 실행 전 테스트와 완료 기준

### 14.1 구현 담당자의 필수 검사

| 검사 | 통과 조건 |
|---|---|
| Registry | 10 case×3server×4seed=120, ID 중복0, old B01 ID 중복0 |
| Counterbalance | STEP1 각 case/position 2회, STEP2 3회 |
| FULL parity | 원 FH12/B01과 prediction/loss/routed gradient/비영 LR optimizer update 일치 |
| Routing | wE만 바꾸면 동일 상태의 LA는 같고, wA만 바꾸면 LU는 같음 |
| Gate detach | d/a/ā/weight 경로를 통한 Student gradient 없음; residual gradient는 존재 |
| HSPMEAN | sample별 Σh 보존; h가 상수이면 FULL과 동일 |
| ADVMEAN | sample별 Σta 보존; a가 상수이면 FULL과 동일; a=0이면 soft=0 |
| NOADV/H0 | NOADV는 trust/β 유지; H0는 soft의 d/a 유지 |
| Q maps | 원 train-view mean, shuffle multiset/stratum, e-surrogate 순위/분포 검사 |
| Resume | 같은 짧은 CUDA trajectory의 uninterrupted/resumed state 및 stream 비교 |
| Probe | 진단 전후 RNG/modes/buffers/.grad 불변 |
| Stress | 부호·단위·동기 frontend·D000 parity·fixed160/192·coverage 경고 보존 |
| Reporting | M12와 B01/ABLR2 분리, selector alias 처리, n=12와 coverage failure 구분 |
| Sheet | frozen 원본·논문용 탭 불변, Dataset-ablations에만 native 표시, 재전송 no-op |

이 번들의 `verification/test_plan.py`는 **계획 수량·배치·loss 정의의 대수적 계약만** 검사한다. 위의 실제 모델/CUDA/optimizer/resume/Sheet 테스트를 대신하지 않는다.

### 14.2 정상 수량

| 산출물 | 서버당 | 전체 |
|---|---:|---:|
| Fresh Student | 40 | 120 |
| Native selection 관측 | 80 | 240 |
| STEP1 stress curve | 96 | 288 |
| Primary stress points | 4704 | 14112 |
| STEP1 stress scene-forward | 94080 | 282240 |
| Case당 독립 Student | 4 | 12 |

Alias이면 추론/raw를 재사용하되 두 selector 관측의 관계는 보존한다. 기존 F1의 반복 계산, 초기 A response, 두 ROI 재채점은 새 Student 수에 넣지 않는다. 수치 실패·support 경고가 남으면 `PROCESSED_WITH_FLAGS`이지 clean pass가 아니다.

### 14.3 새 CLI 계약 — 구현 후 사용할 명령

아래는 **새로 구현할 인터페이스**이며 현재 존재하는 명령이라고 가정하지 않는다.

```bash
# 각 서버에서 S는 실제 원 서버 s1/s3/s5.
S=s1
PLAN=/absolute/path/to/PANDA_REBUTTAL_M12_S135_2026-10-01/planning/experiment_registry.json
BINDING=/absolute/path/to/verified_M12_bindings.json

python3 -B tools/rb_m12_runner.py validate-plan --plan "$PLAN"
python3 -B tools/rb_m12_runner.py preflight --server "$S" --plan "$PLAN" --binding "$BINDING"
python3 -B tools/rb_m12_runner.py run --server "$S" --plan "$PLAN" --binding "$BINDING" --activate
python3 -B tools/rb_m12_runner.py report --server "$S" --plan "$PLAN"
python3 -B tools/rb_m12_runner.py package --server "$S" --plan "$PLAN" --output /actual/path/M12_${S}_evidence.zip
```

`run --activate`는 두 step의 지정 case만 유한 실행한다. `report/package`는 실행 재개·추론·Sheet 변경을 유발하지 않는다. uploader의 쓰기는 별도 explicit activation으로만 한다. setup 승인 범위는 이 문서의 신규 hidden source와 기존 표시 연결에 한정된다.

**기술 검증이 통과하면 이번 두 step은 진행한다. 중간 score 승인이 필요하다는 이유로 각 case에서 멈추지 않는다.** 단 identity/자원/파일 안전성 충돌은 중단·보고한다. 완료 후 다음 구조 실험을 자동 시작하지 않는다.

## 15. 반환 보고서

1. 서버별 completed/failed/pending Student, native selector 수, curve 처리/clean/flagged 수.
2. 원 F1/data/runtime/evaluator/protocol identity와 각 run initial/stream/checkpoint SHA.
3. 표 Q1, Q2, H, K와 Student별 paired delta, n/SD/CI/seed별 승패.
4. 네 inference mode의 fixed160 curve, fixed192 auxiliary, c0/cε/coverage 원자료.
5. hard/soft/edge routing 및 weight-mass/gradient/update diagnostics.
6. 새 cohort 결과와 과거 B01/ABLR2를 분리한 해석: 지지/조건부/미확립 및 실제 효과 크기.
7. Sheet source와 `WV3-ablations` readback receipt, upload debt, 다운로드 가능한 evidence ZIP과 SHA256.

실제 실행 중인 code revision, 원본 evidence 검증 완료, 학습 완료, Sheet 표시 완료는 별도 상태로 보고한다. 누락 파일이나 낮은 score를 추정으로 채우지 않는다.

## 16. 참고 근거와 보관

- **[S1] 제출본** `59391_PANDA_PAN_Alignment_Awar.pdf`, p2/p4 주장, p6 Eq.(7)–(8)와 LPAN/HPAN 설명, p7 Eq.(9)–(12), p8 implementation details. 원고 수정 없음.
- **[S2] 이전 분석** `ABLATION_ANALYSIS_AND_PAPER_TABLES_KR.md`(2026-10-01)와 `SELECTED_ABLATIONS_REVIEW_2026-10-01.zip`; live `유의미한결과!B328:U336`의 후속 작업을 2026-10-01 재확인했다.
- **[S3] Sheet 방침** `DATASET_ABLATIONS_GSPREAD_HANDOFF_KR.md`(2026-10-01). 논문용 `ablations`와 Dataset-ablations는 다름.
- **[S4] 코드** `panda_rb/plan.py`, 조회 blob SHA `1794d20952c479ac64ff7a2f2efc905a4f09501d`: 24run/48curve/2repeat 가드, F1 anchor, validation grid.
- **[S5] 코드** `panda_rb/training.py`: geometry_student_losses, one_batch_parity, stream/probe/optimizer/scheduler 구현. 조회 결과는 원 서버 frozen release 인증을 대신하지 않음.
- **[S6] 코드** `fh12/losses.py`, 조회 blob SHA `541645adf666138d8ce859a9a236e5065213214c`: detached weights/d/a, α1/β.1/λE.002, 분리 backward.
- **[S7] 코드** `panda_rb/weights.py`, 검색에서 확인한 commit `ea71b68be637a1d1f4d61bf89cb4e023cbf00385`: conditional multiset/derangement와 e surrogate. 구현 시 해당 frozen 파일 전체를 재검증할 것.

저장소: `hojunking/PAN-Crafter-repro`. 참고 파일은 archive manifest에 SHA를 기록한다. 이 번들의 문서/registry는 로컬 생성본이며 원격 Git/Drive에 이미 보관됐다고 주장하지 않는다.
