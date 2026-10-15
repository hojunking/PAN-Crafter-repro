# 팬샵 연구 진행 — 새 모델 기준과 연구 방향

**기록일:** 2026-09-09  
**문서 성격:** 사용자 확정 결정 및 연구 상태 인계 문서  
**현재 단계:** 방향 1 — 정합 문제 조정을 통한 모달리티 간 fitting  
**새 기준:** W96–D124 U-Net · MS+PAN 입력 · HRMS 단일 복원 task

> **이 문서의 새 고정 결정은 과거 W168–D123 / 11ch / dual MARs 기준보다 우선한다.**
> PAN reconstruction과 LPAN/HPAN 입력은 제거한다. 앞으로의 방법론은 이 기준에서 개발한다.
> 현재는 정합 기반 fitting을 먼저 연구하며, 동일 구조 Teacher–Student KD는 후속 연구 방향으로 둔다.

---

## 1. 이번에 확정한 모델 기준

아래 항목은 사용자 최신 결정이다. 성능 비교를 위해 열어 둔 선택지가 아니라, 새 연구의 고정 조건이다. [U1]

| 항목 | 새 고정 기준 | 적용 의미 |
|---|---|---|
| Backbone | U-Net, **width=96** | 기존 W168을 mainline 기준으로 사용하지 않음 |
| Depth | **[1,2,4]** | W96–D124 골격으로 고정 |
| 입력 모달리티 | **MS + PAN만 사용** | 별도의 LPAN/HPAN 조건 채널을 넣지 않음 |
| LPAN | **제거** | 저주파 PAN을 명시적 입력으로 제공하지 않음 |
| HPAN / PAN-HF | **제거** | 기존 `PAN−LPAN` 채널을 제공하지 않음 |
| PAN reconstruction loss | **제거** | MARs의 PAN back-reconstruction을 학습 목표에서 제외 |
| 복원 task | **HRMS reconstruction 단일 task** | PAN 재구성을 위한 별도 forward·target·batch 복제는 새 학습 구성에 포함하지 않음 |
| 향후 KD | **Teacher와 Student의 네트워크 구조 동일** | 큰 Teacher를 작은 Student로 압축하는 구조를 기본 전제로 삼지 않음 |

MS band 수를 `C_MS`라고 할 때 입력 채널 수는 `C_MS + 1`이다. 현재 WV3의 8-band MS에서는 **9ch**, 4-band MS에서는 **5ch**에 해당한다. 이는 입력 구성에서 따라오는 채널 수이며, 모든 센서를 9ch로 강제한다는 뜻은 아니다. [U1, S1 §I.0]

**PAN 입력 제거가 아니다.** PAN은 HRMS 복원의 입력으로 계속 사용하고, 제거하는 것은 PAN 자체를 다시 복원하도록 감독하는 auxiliary task이다. 또한 단일 task라는 표현은 MS 영상만 입력한다는 뜻이 아니다.

### 변경 목적

PAN-Crafter의 MARs와 PAN 파생 입력에 의존하던 구성을 벗어나, **정합 기반 fitting과 KD를 중심으로 방법론을 구성**한다. 일반적인 U-Net 골격과 기존 코드의 계보는 구분해 기록하며, 과거 재현·분석 이력은 삭제하지 않는다. [U1]

---

## 2. 과거 기준에서 무엇이 대체되는가?

| 과거 기준·논의 | 새 기준에서의 상태 |
|---|---|
| W168–D123 mainline | **W96–D124로 대체** |
| `[PAN, LPAN, PAN−LPAN, ↑MS]` 11ch | **MS+PAN 입력으로 대체** |
| Dual MARs, PAN loss weight=1.0 | **PAN reconstruction task와 loss 제거** |
| J1/J2의 MS/PAN mode별 jitter 비교 | PAN mode가 없어져 **그 구분은 더 이상 적용되지 않음** |
| J4의 clean MS + jitter MS + clean PAN 세 경로 | J4 아이디어를 재사용하더라도 **PAN 경로는 제외** |
| LPAN을 PAN mode의 residual base로 사용 | PAN task 제거에 따라 **해당 경로 불필요** |
| 11ch↔10ch↔9ch 입력 ablation 제안 | 새 기준에서는 **MS+PAN만 사용하도록 확정** |
| Robust dual-MARs Teacher를 먼저 확보 | 후속 Teacher도 **새 기준 모델에서 정의** |
| 기존 J4의 ±0.5 px, consistency 가중치·스케줄 | **과거 실험 설정**. 새 방법의 고정 조건으로 자동 승계하지 않음 |
| 과거 모델의 성능·학습 시간·연산량 | 역사적 참조. **새 모델의 실측값으로 사용하지 않음** |

과거 캠페인의 구성은 `s1_w168_d123_shift_robust_alignment_30h_plan.md`에 보존되어 있다. 그 문서에 적힌 “유지”나 “고정”은 당시 캠페인의 결정이며, 충돌하는 항목에는 이번 사용자 결정이 우선한다. [U1, S2]

---

## 3. 새 baseline의 계산 관계

### 3.1 이번에 바꾸지 않은 기존 작업 원칙

다음은 이번에 새로 확정한 항목이 아니라, **기존 합의 중 변경 지시가 없었던 작업 기준**이다. 새 방법에서 변경이 필요하면 별도로 기록한다. [C1, S2 §0·§4]

| 항목 | 기존 작업 기준 |
|---|---|
| Attention / CM3A | 미사용 |
| Crop / scale jitter | 기존 nocrop 기준 |
| MS upsampling | 기존 bicubic 경로 |
| Residual base | Clean upsampled MS |
| GT·최종 HRMS 좌표 | M-frame |
| Output inverse warp | 사용하지 않음 |
| Mutual learning | 종료 상태 유지 |
| 작은 train patch의 noisy shift cache | 실제 alignment label로 사용하지 않음 |

PAN-frame output과 target에 관한 과거 규정은 **PAN task 자체가 제거됐으므로 새 모델의 출력 경로에는 적용 대상이 없다.**

### 3.2 입력과 출력

`S`를 원래 LRMS, `P`를 PAN, `U_r`을 기존 MS upsampling 연산이라고 두면,

\[
M=U_r(S),\qquad X=\operatorname{Concat}(P,M)
\]

\[
R_\theta=F_\theta(X),\qquad \hat Y=M+R_\theta
\]

이다. `F_θ`는 W96–D124 U-Net이다. 위 concat 순서는 설명용이며, 실제 채널 순서는 구현 config에 명시해야 한다. 최종 잔차는 MS와 같은 band 수를 갖는다. [U1, C1]

일반 HRMS 복원의 기준 loss를 기존 L1로 이어 쓰는 경우에는,

\[
L_{\mathrm{rec}}=\|\hat Y-Y_{\mathrm{GT}}\|_1
\]

이며, **`L_PAN`은 더하지 않는다.** 이후 fitting loss나 KD loss는 별도로 설계한다. 이 문서는 새 loss의 최종 수식·계수·스케줄을 확정하지 않는다.

### 3.3 MARs 제거에 따른 구현 정리

PAN loss의 계수만 0으로 만들고 불필요한 PAN forward를 계속 계산하는 구성과, PAN task를 제거한 single-task 구성은 구분한다. 새 기준의 학습 흐름에는 PAN reconstruction용 batch 복제·target 생성·출력 경로를 포함하지 않는다.

다만 기존 코드에 남아 있을 수 있는 **mode-dependent γ/β, mode 인자, routing을 정확히 어떻게 제거하거나 일반화할지는 코드 확인 후 기록할 사항**이다. 사용자가 이번 메시지에서 그 세부 변환 방식까지 지정한 것은 아니다. 일반 normalization의 affine parameter와 MARs mode 전용 parameter도 혼동하지 않는다.

LPAN/HPAN을 다른 이름의 condition이나 숨은 우회 입력으로 재도입하지 않는다. 향후 해상도 분석이나 loss 계산을 위해 저역통과·downsampling을 검토하는 것은 별도 설계 항목이며, 제거한 PAN 파생 입력 채널을 다시 허용한다는 뜻은 아니다.

---

## 4. 연구 방향 1 — 정합 문제 조정을 통한 모달리티 간 fitting

**상태: 현재 우선 진행 중인 문제.** [U1]

### 4.1 목적

PAN과 MS 사이의 위치 불일치가 복원 학습에 미치는 영향을 다루고, 두 모달리티의 정보를 더 적절하게 결합해 HRMS를 복원하는 방법을 찾는다.

현재의 핵심 질문은 다음과 같다.

> **MS와 PAN의 좌표 대응이 완벽하지 않은 조건에서, 무엇을 이동·복원·비교해야 모달리티 간 fitting이 개선되는가?**

여기서 “fitting 개선”을 곧바로 명시적 정합 성공으로 간주하지 않는다. 구분할 대상은 아래와 같다.

| 구분 | 의미 |
|---|---|
| 정합 조정 | 실제 상대 위치 대응을 추정하거나 보정하는 것 |
| 변위 강건성 | 입력 위치가 조금 달라도 복원 결과가 안정적인 것 |
| 보간·smoothing 효과 | 공간 변화에 민감한 성분이 약해져 예측 차이가 줄어드는 것 |

기존 audit과 jitter 캠페인도 실제 정합, perturbation, smoothing을 분리하는 문제를 제기했다. 새 기준에서도 어느 효과가 발생했는지 구분해야 한다. [S1, S2 §1·§2·§8]

### 4.2 현재까지 논의한 후보 — 채택 미확정

| 후보 | 논의한 아이디어 | 현재 상태 |
|---|---|---|
| MS 조건 jitter | MS 조건의 복사본에 작은 global translation 적용 | 새 baseline에서 재검토 |
| Clean–jitter consistency | 두 입력의 복원 결과 또는 residual을 비교 | 기존 J4의 아이디어. 최종 채택 미정 |
| 해상도별 supervision | 저해상도에서 비교적 안정적인 내용을 복원 목표로 활용 | 개념 검토 단계 |
| 추가 corruption → 이전 관측 복원 | 변위를 추가한 입력에서 추가 전 관측을 복원하도록 학습 | Noise-to-noise 관점의 검토 아이디어 |
| 저해상도 관측 consistency | 복원 결과를 낮춰 원래 LRMS와 비교 | 앞선 대화의 제안. 구현·효과 미확인 |
| PAN/MS 정합 단서 활용 | 명시적 또는 암묵적 위치 대응 개선 | 대상·모듈·loss 미확정 |

이 표는 앞선 대화의 후보를 보존한 것이지, 모든 항목을 함께 사용하는 모델을 확정한 것이 아니다. **새 baseline을 확정했다는 사실과, 정합 방법이 확정됐다는 사실은 다르다.** [C2]

### 4.3 0.5 HR px와 0.125 LR px의 의미를 보존한다

기존 jitter 계획은 각 step·sample에서 가로와 세로 이동량을 각각 뽑아, upsampling된 MS 조건 전체에 동일하게 적용하는 구성이었다. [S2 §4–5]

\[
\epsilon_x,\epsilon_y\sim U(-0.5,0.5)\quad\text{현재 입력 PAN/HR pixel}
\]

배율이 4인 pair에서,

\[
0.5\ \mathrm{HR\ px}=0.125\ \mathrm{LRMS\ px}
\]

이다. 아래 구분을 유지한다.

- 이 값은 **추가 perturbation의 축별 상한**이지 실제 misalignment의 측정된 최대값이 아니다.
- 같은 변위를 서로 다른 격자 단위로 표시하는 것만으로 물리적 위치 오차가 줄어들지는 않는다.
- 실제 low-pass·downsampling 뒤 비교값이 작은 변위에 덜 민감해지는지는 별도로 검증할 연구 가설이다.
- ±0.5는 기존 C2 적용값의 표준편차 약 0.076 LR px와 비슷한 perturbation을 만들기 위한 설정이었다. [S2 §5.1, S3 §1.2]
- Audit의 bicubic–제공 `lms` 사이 반 픽셀 phase 차이와, random jitter 범위 ±0.5는 서로 다른 양이다. [S1 §II.4]

**따라서 새 방법의 jitter 범위를 0.5로 자동 고정하지 않는다.** 분포, 범위 결정 규칙, 적용 해상도와 대상은 방향 1에서 결정할 항목이다.

### 4.4 Noise-to-noise 관점의 위치

현재 보존할 아이디어는 **“완벽하지 않을 수 있는 관측 pair에 크기를 아는 추가 corruption을 가하고, 추가 전 관측 또는 그 안정적인 성분을 복원하도록 학습한다”**는 것이다. [C2]

이 문서에서는 이를 검토 컨셉으로만 기록한다. 원래 관측이 완벽한 정합 GT라는 주장, 원래 Noise2Noise의 조건이 그대로 성립한다는 주장, GT를 제거한 학습이 이미 확정됐다는 주장은 하지 않는다. 복원 target의 해상도·좌표·정보량과 HR 세부 supervision의 유지 방식은 아직 미정이다.

---

## 5. 연구 방향 2 — 동일 구조 KD를 통한 overfitting 문제 대응

**상태: 후속 연구 방향. 현재 방향 1을 대체하거나 동시에 확정하지 않는다.** [U1]

사용자가 말한 “KD를 통한 overfitting”은 이 문서에서 **overfitting 문제의 대응·완화와 일반화 검증을 다루는 방향**으로 정리한다. KD가 이미 overfitting을 줄였다는 실험 결론은 아니다.

### 5.1 Teacher–Student 구조 원칙

Teacher와 Student는 **동일한 네트워크 구조**를 사용하며, 모두 새 W96–D124·MS+PAN·단일 HRMS task 기준을 따른다. [U1]

“네트워크가 동일하다”는 것은 구조에 대한 결정이다. 같은 가중치를 반드시 공유한다는 결정은 아니다. Teacher/Student의 학습 순서, 초기화, checkpoint 선택, frozen/EMA 등의 갱신 방식은 아직 확정되지 않았다.

특히 다음 두 구성을 구분한다.

| 구성 | 의미 |
|---|---|
| Clean/jitter 두 branch | 같은 모델에 서로 다른 입력을 넣어 비교하는 기존 consistency 구성 |
| 동일 구조 Teacher–Student KD | 같은 architecture에서 Teacher와 Student 역할을 나누어 지식을 전달하는 후속 구성 |

두 모델의 구조가 같더라도 Teacher의 정합 지도 신호가 Student에 유효한지는 검증해야 한다. Teacher가 먼저 학습됐다는 이유만으로 실제 정합을 해결했다고 전제하지 않는다.

### 5.2 KD에서 다룰 세 가지 요소

| 요소 | 연구 내용 | 미확정 세부 |
|---|---|---|
| **Teacher uncertainty 유무** | Uncertainty 없이 전달하는 경우와 사용하는 경우의 차이 | 추정 방식, calibration, loss 가중 방식 |
| **Teacher의 alignment-leading loss** | Teacher가 Student에게 정합 문제의 단서를 전달하는 loss | 전달 대상이 출력·잔차·feature·변위 중 무엇인지, 좌표·부호·가중치 |
| **GT variance** | GT variance 정보를 supervision 또는 distillation에 활용 | variance의 정의, 계산 영역, band 처리, 정규화, 적용 위치 |

세 요소를 처음부터 모두 켜는 것으로 확정하지 않는다. 각각의 기여와 상호작용을 분리해서 확인할 대상이다. [U1]

Teacher uncertainty와 GT variance를 동일한 양으로 취급하지 않는다. 또한 GT variance를 곧바로 정합 불확실성으로 정의하지 않는다. **GT variance의 정확한 계산식은 이번 사용자 결정에 포함되지 않았다.**

Uncertainty 산출에 보조 head가 필요한 경우에도 **Teacher–Student 구조 동일성 조건과 어떻게 양립할지 먼저 정해야 한다.** Teacher만 더 큰 복원 backbone을 사용하는 변경을 자동으로 허용하지 않는다. Uncertainty head의 도입·공통 탑재·외부 추정 중 어떤 방식을 택할지는 미정이다.

### 5.3 방향 1과의 연결

연구의 연결은 다음과 같이 기록한다.

**새 baseline → 정합 기반 fitting의 독립 효과 확인 → 동일 구조 Teacher의 지도 신호 정의 → Student의 overfitting·일반화 검증**

방향 1에서 유효한 정합 정보나 perturbation에 대한 복원 특성이 확인되면, 이를 Teacher가 Student에게 전달하는 대상으로 검토한다. 단순히 Teacher 출력 전체를 따라 하는 효과와, 정합 관련 지도 신호의 추가 효과는 구분한다.

실제 Teacher 사전학습 절차와 KD 실행 큐는 아직 정해지지 않았다.

---

## 6. 현재 단계와 연구 범위

| 항목 | 현재 상태 |
|---|---|
| W96–D124 / MS+PAN / PAN loss 없음 | **고정** |
| 정합 기반 모달리티 fitting | **현재 집중 연구** |
| 최종 jitter·복원 target·consistency 설계 | **미확정** |
| 저해상도 관계를 활용한 복원 학습 | **방향 1의 검토 후보** |
| 동일 구조 Teacher–Student KD | **후속 방향으로 확정, 세부 설계 미정** |
| Teacher uncertainty·alignment-leading loss·GT variance | **방향 2의 검증 요소** |
| 정합을 반영해 HQNR 참조·계산을 바꾸는 평가 | **보류** |

**HQNR 정합 반영 평가를 보류한다는 것은 HQNR 자체의 평가를 중단한다는 뜻이 아니다.** 기존 metric 평가와 정합 반영 평가를 구분한다. 이 문서에서는 새로운 평가 subset이나 checkpoint 선택 규칙을 확정하지 않는다. 이전 FR 12–19와 별도 보고 범위에 관한 논의도 새 모델 변경만으로 자동 해결된 것으로 취급하지 않는다. [C1]

과거 30시간 예산, optimizer·batch·iteration·seed·평가 주기 또한 이번 새 기준에 포함된 확정사항이 아니다. 실험 실행 문서를 만들 때 기존 설정의 유지 여부를 따로 적는다.

---

## 7. 다음 실험을 해석할 때의 기준

다음은 새 baseline에서 효과를 분리하기 위한 **기록·비교 원칙**이다. 실행 큐나 시간 예산을 확정한 것은 아니다.

### 현재 방향 1

먼저 새 기준의 일반 HRMS 복원 baseline과 정합/fitting 후보를 비교한다. 폭·깊이·입력·PAN task 유무를 동시에 다시 바꾸지 않는다. 과거 W168–D123 dual-MARs 결과를 새 방법의 직접 대조군으로 대신하지 않는다.

Clean/jitter 두 branch를 사용한다면 **같은 두 branch에 GT loss만 적용한 대조군**과 비교해, 추가 consistency의 기여를 구분한다. 저해상도 loss를 검토한다면 smoothing 효과와 실제 위치 대응 개선을 구분할 진단을 함께 기록한다. [C2]

### 후속 방향 2

독립 학습, KD, uncertainty 사용 여부, alignment-leading loss 추가, GT variance 추가를 구분해 기록한다. 구체적인 조합 수와 실행 순서는 방향 1 결과 이후 별도로 정한다. 동일한 구조라는 이유로 Student가 Teacher의 도움 없이도 얻는 개선을 KD 효과로 섞지 않는다.

이 문서는 **새 baseline에서의 구현 완료, 새 실험 실행, 성능 향상 또는 정합 개선을 보고하지 않는다.** 해당 사실은 실제 config·코드·로그·결과가 확보된 뒤 추가한다.

---

## 8. 미확정 항목

| 영역 | 다음 설계에서 정할 내용 |
|---|---|
| 코드 정리 | MARs mode routing·γ/β의 제거/일반화, 입력 순서, 새 모델 구현 확인 |
| Perturbation | MS 또는 PAN 중 대상, 적용 격자, 크기 분포, sample별 재추출 방식 |
| 해상도 활용 | 단순 좌표 단위 환산과 실제 downsampling의 구분, 필터·sampling phase |
| 복원 target | HRMS GT, 원래 LRMS, clean 예측 등 각 목표의 역할·좌표 |
| Fitting loss | HR/LR 비교 범위, consistency 여부, 가중치·스케줄 |
| KD 절차 | Teacher/Student 초기화·학습 순서·갱신·checkpoint |
| Uncertainty | 구조 동일성을 지키는 산출 방식과 사용 여부 비교 |
| Alignment-leading loss | 지도 신호의 정의, 실제 정합과의 관계, 잘못된 cue 대조 |
| GT variance | 무엇의 분산인지, 계산·정규화·적용 방식 |
| 실행 조건 | 학습 budget, seed, batch, optimizer, 평가 subset·선정 기준 |

위 항목을 이전 제안의 수식이나 hyperparameter로 임의 확정하지 않는다.

---

## 9. 다음 세션용 핵심 인계

> 새 모델 기준은 **W96–D124 U-Net, MS+PAN 입력만 사용, LPAN/HPAN 제거, PAN reconstruction 및 dual MARs 제거**이다.
> WV3에서는 입력이 9채널이며, PAN은 복원 target이 아니라 HRMS 생성을 위한 입력이다.
> 연구는 **① 정합 문제 조정을 통한 모달리티 간 fitting, ② 동일 구조 Teacher–Student KD를 통한 overfitting 문제 대응**으로 진행한다.
> KD에서는 Teacher uncertainty 유무, Teacher의 alignment-leading loss, GT variance를 검토한다.
> **현재는 ①을 푸는 과정이다.** Jitter와 저해상도 복원 supervision은 후보이며, 최종 method는 아직 확정되지 않았다.
> 기존 ±0.5 HR px는 실제 misalignment의 측정된 최대값이 아니며, 새 모델의 고정 hyperparameter로 자동 승계하지 않는다.
> HQNR 정합 반영 평가 논의는 보류한다. 과거 문서와 충돌하면 이번 새 고정 기준을 우선한다.

---

## 10. 근거와 문서 적용 범위

- **[U1] 사용자 최신 결정 — 이 대화.** PAN reconstruction loss 제거, LPAN/HPAN 제거, MS+PAN 입력, W96–D124 고정, 동일 구조 KD와 두 연구 방향, 현재 방향 1 우선. 새 고정 결정의 1차 근거이다.
- **[C1] 이 대화의 기존 합의.** Attention/CM3A 미사용, nocrop, 기존 bicubic과 clean MS residual base, M-frame 출력, output inverse 미사용, mutual learning 종료, HQNR 정합 반영 평가 보류. 이번 사용자 변경과 충돌하지 않는 범위에서만 이어 쓴다.
- **[C2] 이 대화의 이전 아이디어 논의.** Additional corruption 복원, clean–jitter consistency, 해상도별 supervision, LR observation consistency와 대조군 제안. 확정 설정이나 검증 결과가 아니라 검토 이력이다.
- **[S1] `2026-09-03_alignment-audit-s2-detail.md`.** PanCollection 공간 정합 audit 상세본. 측정 격자·보고 단위, edge 추출, sampling phase, 측정 한계에 관한 근거이다. 특히 §I.0, §II.4, §II.8.5, §III를 참조한다.
- **[S2] `s1_w168_d123_shift_robust_alignment_30h_plan.md` (2026-09-06).** 과거 W168–D123/11ch/dual-MARs 캠페인의 설계·jitter 정의·J4 loss. 현재 고정 기준의 근거가 아니라 변경 전 이력이다. 특히 §0, §4–5, §8–9를 참조한다.
- **[S3] `2026-09-04_global-alignment-plan-review (1).md`.** 기존 C2 cache의 학습 적용값 통계와 정합 추정기의 한계. ±0.5 jitter 선택 배경은 §1.2를 참조한다.

**출처 우선순위:** 최신 사용자 결정 → 충돌하지 않는 기존 사용자 합의 → 첨부 문서의 기록 → 이전 대화의 미확정 제안. 이 문서는 새 기준을 기록한 것이며, 외부 문헌 조사나 최신 Sheet 성능 재평가를 대신하지 않는다.
