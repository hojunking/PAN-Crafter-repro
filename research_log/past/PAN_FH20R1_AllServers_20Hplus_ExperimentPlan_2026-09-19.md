# FH20R1 — WV3 전 서버 20시간 이상 후속 실험 계획

**작성일:** 2026-09-19  
**Campaign:** `WV3_FH20R1_20260919_v1`  
**Registry:** `FH20R1_REGISTRY_20260919_v1`  
**대상:** s1–s5 모두 WV3. QB/GF2 계획은 계속 보류한다.  
**작업 상태:** 실행 명세·구체적 case 정의를 작성한 단계. 이 문서 작성으로 서버 큐·코드·Sheet·GPU 작업이 변경되거나 시작되지는 않았다.

> **이번 20시간은 상한이 아니라 서버별 최소 유효 작업시간이다.** FH12의 12시간 기록을 고치거나 재시작하지 않고, 새 campaign에서 각 서버가 학습·공식 평가·필요 진단을 20시간 이상 수행한다. 기본 비교를 완료하고 최소시간도 충족하면 현재 비교 block을 마친 뒤 종료한다. 서버별 결과를 기다리는 공통 recipe lock은 없다.

## 0. 먼저 읽을 실행 요약

이번 목적은 실험 범위를 다시 크게 넓히는 것이 아니다. **LPAN 입력과 D122의 조합을 반복 검증하고, 후반 공간 왜곡의 원인과 Teacher Aligner 초기화의 영향을 분리**한다. 새 attention, PAN reconstruction, Student offset loss, 직접 shift KD, ERGAS 전용 loss는 넣지 않는다.

| 서버 | 중심 역할 | 재사용 Teacher | 정상 경로 기본 신규 학습 | 시간 부족이 아닌 ‘시간 미달’ 때의 예비 |
|---|---|---|---:|---:|
| **s1** | A/U 교차 진단 → 근거가 있으면 A의 후반 LR 대조. 미지지면 depth 반복으로 전환 | F1: P0·T71001 | **Student 14개** | Student 8개 |
| **s2** | N2 A 초기화 Teacher 1개 → 기존 fresh Teacher와 Student 대응 비교 | F2: PL·T71001 + 새 N2PL | **Teacher 1개 + Student 10개** | Student 8개 |
| **s3** | PH의 D121/D122 반복 → D122에서 W104/W112 비교 | F3: PH·T71001 | **Student 20개** | Student 8개 |
| **s4** | PL×D122 누락 대조 → PL/PLH×D121/D122 반복 → 제한적 W112D122 | F4: PLH·T71001 | **Student 20개** | Student 8개 |
| **s5** | 두 번째 Teacher 아래 PL/PLH×D 비교 및 반복 | F5: PLH·T71002 | **Student 20개** | Student 8개 |
| **합계** | | | **Teacher 1 + Student 84 = 85run** | **최대 40run** |

**85개는 85종의 하이퍼파라미터 설정이 아니다.** 대부분은 소수 설정을 같은-seed pair/quartet으로 반복하는 실험이다. FH12 실측에서 Student 50K+평가가 서버별 약 1.07–1.89시간이었기 때문에, 직전 제안의 신규 6개만으로는 다섯 서버를 각각 20시간 운영할 수 없다. 본 편성은 그 6개를 먼저 수행하면서 반복 근거를 늘린다. [R1 §10.2; R2; R3]

s1에는 두 가지 **상호 배타적** 실행 경로가 있고, s2에는 donor 미확보 시 대체 경로가 있다. CSV/JSON에 정의된 145개 학습 항목을 전부 순서대로 실행해서는 안 된다. 정상 경로의 기본은 85개, 정상 경로에서 모든 시간예비까지 도달해도 125개다. 조건부 대안 20개가 정의 수에 추가돼 145개가 된다. 준비·진단·calibration 11개 stage는 학습 run 수에 포함하지 않는다.

### 0.1 첫 번째로 완결할 기존 제안 6개

| 서버 | 가장 먼저 시작할 새 학습 |
|---|---|
| s1 | 새 학습보다 로컬 A/U·q 진단을 먼저 수행한다. 이후 §7.1의 pair를 시작한다. |
| s2 | `N2INIT Teacher PL W112D123 T71001`, 이후 exact50K calibration |
| s3 | `PH W104D121 S72002`와 `PH W104D122 S72002` |
| s4 | `PL W104D122 S72001` |
| s5 | `PL W104D121 S72002`와 `PL W104D122 S72002` |

s3–s5는 s1 진단이나 s2 새 Teacher를 기다리지 않는다. s1 역시 다른 서버에서 새 모델을 완료할 때까지 기다리지 않는다. 모든 필요한 기존 reference는 이미 완료된 FH12 자산이다.

## 1. 근거와 새 가설을 구분한다

### 1.1 FH12에서 관측된 사실

이 문서의 결과 기준은 **첨부된 2026-09-19 FH12 21run snapshot**이다. 새 live Sheet 전체를 다시 분석하거나, 이후 추가 run이 없다고 인증한 것은 아니다.

| 관측 | 후속 편성에 반영한 내용 |
|---|---|
| 동일 Teacher의 6개 입력 대조에서 Exact50K HQNR 상승·ERGAS 감소. 공유 P0 기준선이 있어 6개 독립 seed는 아님 | L/H 입력을 유지하고 PL/PLH 비교를 계속한다. |
| PH 1개·PLH 2개에서 D122의 Exact50K ERGAS가 2.0444–2.0452로 개선 | D122를 중심 구조 후보로 둔다. D121 대조는 유지한다. |
| s4 같은 Teacher/seed에서 PL D121의 E=2.049526, H=0.954620. PLH D121은 E=2.052659, H=0.954629 | **PL D122**를 우선 채운다. PLH를 자동 최종 입력으로 선택하지 않는다. |
| 모든 Student에서 RAW_MAX→50K 사이 E 감소와 D_s 증가가 동반됨 | A/U 시점 교차 평가로 원인을 분리한다. |
| 새 Teacher q_ref≈0.4533–0.4566, AXIS16 상수-A 기준 q=0.46875 | q 분포·반경별 반응을 읽고 N2 A 초기화 대조를 제한적으로 만든다. |
| FH12 Student의 50후보 최저 E도 2.044073, 공동목표 달성 없음 | selector 변경만으로 해결됐다고 하지 않는다. |

원 관측·제한·전체 표: [R1 §§3–8], 정확한 값: [R2].

### 1.2 이번에 새로 시험하는 가설 — 아직 확인된 효과가 아님

1. **PL×D122:** LPAN을 추가한 입력의 균형과 D122의 RR 개선이 함께 나타날 수 있는가?
2. **후반 A-LR:** A의 업데이트를 늦추면 후반 U의 RR 개선을 유지하면서 D_s 악화를 줄일 수 있는가? 진단이 이를 지지할 때만 s1에서 시험한다.
3. **Teacher A 초기화:** 같은 PL Teacher에서 N2 donor A로 시작하면 fresh-A의 약한 상대 이동 반응을 개선하는가? 그 reference를 사용하는 Student에도 이득이 있는가?
4. **W112D122:** D122에 소폭 width 증가를 더하면 E<2.040과 HQNR 하한을 함께 만족할 여지가 있는가?

어느 가설도 성공을 전제하지 않는다. 특정 seed 하나에서의 최고값을 근거로 다른 서버의 실행 profile을 자동 변경하지 않는다.

## 2. 목표와 checkpoint 선택은 그대로 유지한다

주 목표는 **동일 checkpoint·정상 A_ON의 raw-original HQNR≥0.9585를 먼저 충족하고, 그 안에서 공식 RR ERGAS 최소**다. 공동목표는 E<2.040이다. PAN-Crafter의 WV3 보고값 H=.958/E=2.040과, 사용자 개발 하한 .9585를 혼동하지 않는다. [R4 Table 1/6]

\[
\mathcal K_H=\{k:H_k\ge0.9585\},\qquad
k^*=\arg\min_{k\in\mathcal K_H}(E_k,-\mathrm{SCC}_k,-\mathrm{PSNR}_k,k).
\]

- \(\mathcal K_H=\varnothing\)이면 `no_eligible`, Target 수치는 빈칸이다. 다른 checkpoint의 HQNR와 ERGAS를 합치지 않는다.
- H=.9586/E=2.039는 H=.9602/E=2.070보다 우선이다. 2.040 미만에 도달하지 못했어도 H 하한을 통과한 후보 안에서는 E가 낮은 쪽을 선택한다.
- 반올림 전 값으로 비교한다. 원본 precision을 보존한다.
- U/A 교차 진단 결과를 정상 Target에 넣지 않는다. 정상 모델은 같은 run·같은 update의 A/U를 쓴다.
- 후보 50개: `{1010*k | k=1,…,49} ∪ {50000}`. 특정 좋은 run에만 후보를 더 촘촘하게 추가하지 않는다.

다섯 선택점을 계속 저장한다.

| 선택점 | 의미 |
|---|---|
| RAW_MAX | 50후보 중 raw HQNR 최대. 기존 Sheet 본 행과 연결 |
| TARGET | H≥.9585 후보 중 E→SCC→PSNR→낮은 step |
| EXACT50K | 고정된 학습량 비교 및 Teacher reference |
| RR_VAL_SELECTED | 기존 validation 1080 patch의 ERGAS로 선택한 checkpoint의 공식 RR/FR |
| E_MIN_DIAG50 | 공식 RR test 50후보의 최저 E. **개발용 oracle, 독립 test 성능이 아님** |

고정된 50후보 전체에서 H/E/Dλ/Ds와 장면별 결과를 읽어 적격 구간 개수·후반 H 하락을 함께 보고한다. 후보 50개를 독립 seed 50개로 세지 않는다. 반복 seed가 시험셋의 독립성을 복원하는 것도 아니다.

## 3. 유지하는 method와 이번에만 달라지는 조건

### 3.1 Forward·입력·좌표

\[
M=U_4(MS),\quad L=U_4(LPAN),\quad c=A(P_{m4},M_{m4}),
\]
\[
\widetilde P=\mathcal W(P,c),\quad
\widetilde L=\mathcal W(L,c),\quad
\widetilde H=\widetilde P-\widetilde L,\quad Z=M+F(X).
\]

| Layout | U-Net 입력 | 채널 |
|---|---|---:|
| P0 | `[P_tilde,M]` | 9 |
| PL | `[P_tilde,L_tilde,M]` | 10 |
| PH | `[P_tilde,H_tilde,M]` | 10 |
| PLH | `[P_tilde,L_tilde,H_tilde,M]` | 11 |

**Aligner는 PAN1/MS8만 본다.** LP/HP를 A stem에 넣지 않고 c를 따로 만들지 않는다. PAN과 HR로 올린 LP를 동일 grid로 warp하고 HP는 그 차이로 만든다. c=(dy,dx)는 HR PAN pixel이다. Warp/frontend FP32, bicubic, border, align_corners=False, Aligner-only margin4, U-Net과 평가 full image를 유지한다. MS base와 GT는 움직이지 않고 base는 한 번만 더한다.

각 branch는 자기 c를 쓴다. Teacher c를 Student inference에 직접 주는 구조가 아니다. c를 LP/HP 경로에서 detach하지 않는다. 고정 grid에서 \(W(P-L,c)=W(P,c)-W(L,c)\)라는 선형성은 이용하지만, filtering/decimation/warp의 교환 가능성까지 가정하지 않는다.

**FH12 LP cache를 그대로 재사용**한다. Gaussian σ1.98/k41/replicate/[2::4,2::4] recipe, split별 hash·phase를 유지한다. 이번에는 LP kernel·decimation phase·HP normalization을 새로 튜닝하거나 다른 원본 cache로 바꾸지 않는다. P=L+H이므로 PL/PH의 정보 span과 parameterization 차이를 구분한다. [R1 §4/9; R5]

### 3.2 Teacher consistency와 s2 초기화 대조

\[
c_0=A_T(P,M),\quad P_\epsilon=W(P,\epsilon),\quad c_\epsilon=A_T(P_\epsilon,M),
\]
\[
L_{\mathrm{off}}=\frac1{2B}\sum_i\|c_{\epsilon,i}+\epsilon_i-\mathrm{sg}(c_{0,i})\|_1,
\]
\[
L_T(t)=\mathrm{mean}|Z_T-Y|+\mathbf1[t\bmod2=1]10^{-4}L_{\mathrm{off}}.
\]

`t`는 0-based optimizer update다. Native reconstruction은 매 update에 A/U를 갱신한다. 반경2 HR-pixel 원판의 면적 균등 ε를 별도 RNG로 생성하며, shifted PAN은 A-only consistency branch에만 넣는다. U-Net에는 synthetic shifted PAN을 주지 않는다. Native reconstruction의 c0는 detach하지 않는다.

**새 Teacher는 s2의 N2INIT 한 개만 기본 편성한다.** 나머지 서버는 기존 FH12 Teacher를 재사용한다. s2에서는 Teacher W112D123·PL·seed71001의 fresh U를 기존 F2 초기 U와 동일하게 만들고 A 초기값만 N2 donor로 바꾼다. λ_off, 홀수 적용, 반경, U/A LR, 데이터, LP cache는 기존 F2와 동일하다.

- donor U-Net/optimizer/scheduler는 가져오지 않는다.
- donor A의 마지막 head를 0으로 다시 초기화하지 않는다.
- 이를 ‘A와 U 모두 from scratch’라고 쓰지 않는다. `N2INIT50`으로 명시한다.
- N2 단계 자체를 이번에 다시 학습하지 않는다. donor를 검증할 수 없으면 s2의 대체 입력 대조를 실행한다.
- Teacher가 H 하한을 못 넘거나 q 개선이 작다는 이유만으로 Student를 막지 않는다. 무결성 오류·non-finite·자산 미확보는 별도 실패다.

Stage1의 reconstruction/consistency 역할 분리는 프로젝트 연구자료 p.8–10과 동일하다. [R6]

### 3.3 Student KD·edge·Aligner adjustment

\[
e_T=\mathrm{mean}_b|Z_T-Y|,\quad e_S=\mathrm{mean}_b|Z_S-Y|,
\]
\[
d_T=\mathrm{sg}\frac{e_T}{e_T+\tau_R},\quad
 a_T=\mathrm{sg}\frac{[e_S-e_T]_+}{e_S+10^{-6}},\quad
 s_i=\mathrm{sg}\frac{q_{ref}}{q_{ref}+q_{T,i}},
\]
\[
\ell_{H,i}=\mathrm{mean}_p[(1+d_T)e_S],\qquad
\ell_{K,i}=\mathrm{mean}_p[0.1(1-d_T)a_T\,\mathrm{mean}_b|Z_S-Z_T|],
\]
\[
\boxed{L_U=\mathrm{mean}_i[\ell_{H,i}+\ell_{K,i}+0.002s_i\ell_{E,i}]},\qquad
\boxed{L_A=\mathrm{mean}_i[s_i\ell_{H,i}]}.
\]

| 역할 | 받는 gradient |
|---|---|
| Teacher U | Teacher native reconstruction |
| Teacher A | Teacher native reconstruction + 정해진 consistency |
| Student U | hard + adaptive soft KD + q-weighted GT edge |
| Student A | **q-weighted hard만** |

Teacher A/U·GT·d/a/q는 detach/frozen이다. Live eS와 soft discrepancy는 미분 가능하다. β는 이미 soft 항 안에 한 번만 들어간다. Edge는 최종 HRMS와 GT의 signed Scharr x/y 차이, 두 방향 0.5 평균, 경계1px 제외다. 입력 HP를 edge target으로 바꾸지 않는다. q 분자2, batch 재정규화, 새로운 temperature/threshold를 넣지 않는다.

U/A parameter 집합에 `autograd.grad`를 따로 적용한다. 두 optimizer를 쓰더라도 한쪽 update 전에 양쪽 gradient 계산을 끝낸다. Student A에 soft·edge·consistency를 직접 주거나 c_T-c_S loss를 추가하지 않는다. **Student A는 BASE와 A10 모두 trainable이며 freeze case는 이번 기본 편성에 없다.** [R1 §9; R6 pp.12–15]

### 3.4 모든 새 Student의 공통 설정

| 항목 | 설정 |
|---|---|
| Dataset | WV3, train9714 / val1080 / RR20 / FR20 — 실제 source hash로 확인 |
| Shape | train PAN64/MS16; RR PAN256; FR PAN512 |
| Band/정규화 | 8 bands, max_pixel2047, 기존 2DN/2047−1 |
| Task/구조 | MS-only, LN, attention OFF, mode modulation OFF |
| 학습 | 50K, batch48, workers4, FP32/no-AMP |
| Optimizer | AdamW β=(.9,.999), eps1e−8, wd=.01 |
| Schedule | 기존 cosine, warmup100, restart 없음 |
| LR | U peak1e−4 / A peak3e−6; A10만 §6의 multiplier |
| 초기화 | U fresh; 해당 Teacher A의 독립 clone |
| 초기 tensor | `FH12_named_tensor_fresh9_zero_extra_v1`를 그대로 사용 |
| 증강/RNG | FH12의 실제 HV/rot4 계약, model/data/corruption RNG 분리 |
| Teacher reference | 지정된 exact50K의 A/U 전체; 다른 checkpoint를 혼합하지 않음 |

초기화 key에 새 campaign 이름을 넣어 기존 named tensor가 달라지지 않게 한다. 같은 Teacher/seed/layout/W/D의 BASE와 A10은 공통 step0 tensor가 같아야 한다. 다른 입력에서는 공통 P/MS kernel·body를 맞추고 추가 L/H kernel=0을 유지한다. W/D가 달라진 경우 전체 함수가 같다고 주장하지 않는다.

## 4. Reference 자산과 calibration 연결

다음 표는 **새 학습할 목록이 아니라 재사용할 기존 자산**이다. 아래 숫자는 감사용 snapshot이며 config에 손으로 입력할 값이 아니다. 원 reference_manifest와 cache를 읽고 SHA로 검증한다.

| Alias | 서버·Teacher 입력 | T seed | exact50K SHA 앞16자리 | τ_R(감사용) | q_ref(감사용) |
|---|---|---:|---|---:|---:|
| **F1** | s1 · P0 W112D123 | 71001 | `04ef8e756ae8b229` | 0.012118559331 | 0.453260362148 |
| **F2** | s2 · PL W112D123 | 71001 | `d2832f5a2d40536a` | 0.012083493173 | 0.454883515835 |
| **F3** | s3 · PH W112D123 | 71001 | `8a971c857a1f386f` | 0.012089714408 | 0.455190271139 |
| **F4** | s4 · PLH W112D123 | 71001 | `9b6fd8ccc0e3e567` | 0.012065213174 | 0.455461412668 |
| **F5** | s5 · PLH W112D123 | 71002 | `dff2b0c1b4e00422` | 0.012087114155 | 0.456601321697 |

SHA 전체는 JSON에 보존한다. N2PL의 SHA/τ_R/q_ref는 아직 존재하지 않으며 새 Teacher·calibration 완료 뒤 실제 값으로 기록해야 한다.


### 4.1 경로와 identity

F1–F5의 실제 run ID, checkpoint SHA 전체, 원 Git/content SHA 및 경로 패턴은 JSON `reference_assets`에 있다. 새 campaign에서 기존 manifest를 변경하지 않는다.

기존 FH12 loader는 `source_identity` 전체와 immutable 12h window/등록 config를 강하게 검사한다. 따라서 새 run 이름만 기존 `fh12_start.sh`에 주거나 12h 숫자만 20으로 고치면 안 된다. **읽기 전용 reference import와 새 campaign runner를 구현**한다. [R7]

Import는 원 model bytes/config/source/calibration을 보존한 상태에서, 새 consumer가 원 Teacher forward·loss·q를 재현하는지 확인하는 bridge record를 따로 만든다. 원 artifact의 Git SHA나 `source_identity`를 현재 값으로 덮어쓰거나, 검사를 통째로 삭제하지 않는다. 같은 수치의 모델을 새 버전에서 읽는 것과 과거 학습을 새 코드로 exact resume하는 것은 별개다.

검증에는 최소 train64 8개, RR256 2장, FR512 2장으로 원/신규 forward의 y/c를 비교하고 동일 공식 평가 함수를 사용한다. 동일 FP32 runtime에서는 초기 기준 atol=3e−6, rtol=2e−5를 사용한다. GPU/runtime 차이로 초과하면 허용오차를 조용히 키우지 말고 환경을 맞추거나 원 reader를 보존한다. 이 tolerance는 신규 구현 확인 기준이지 성능 개선의 유의성 기준이 아니다.

### 4.2 N2 donor의 알려진 reference

- Source run: `PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT`.
- Source checkpoint: `last`, step50000.
- 과거 config에 기록된 **원 checkpoint container SHA256**: `9d4cbf218cd8f6743942115e882d87d664d663935a1a1ba295a9903e27f3f816`.
- 이 SHA를 A tensor-state hash 또는 변환된 safetensors hash로 잘못 비교하지 않는다. 원 파일과 추출 A의 tensor SHA를 각각 기록한다.
- 8-band A 구조·margin4·train data lineage를 검증한다. 이 문서 작성 시 서버 로컬 donor 파일 존재를 새로 확인한 것은 아니다. [R8]

s2 donor 확인은 초기에 실시한다. 접근/전송/검증이 해결되지 않으면 그 대조만 `BLOCKED_DONOR`로 기록하고 §7.2의 F2 입력 비교로 진행한다. 자동으로 다른 donor나 새 랜덤 A를 같은 run ID에 넣지 않는다.

### 4.3 τ_R/q: 재사용과 새 측정

F1–F5 아래에서 Student input/W/D/seed/BASE-A10만 바뀌면 **원 τ_R/q_ref/cache를 그대로 재사용**한다. 각 run에서 다시 추정하거나 좋은 성능이 나오도록 조정하지 않는다.

N2PL Teacher만 exact50K에서 새로 측정한다.

\[
\tau_R=\max(\operatorname{median}_{i,p\in\mathcal C}e_T(i,p),10^{-6}),\quad
q_{i,r}=\frac1{2K}\sum_j\|A_T(W(P_{i,r},\epsilon_j),M_{i,r})+\epsilon_j-A_T(P_{i,r},M_{i,r})\|_1.
\]

K=16, 반경 .25/.5/1/2 × 네 축 방향. τ_R는 무증강 train3072, q_ref는 동일 calibration base IDs×rot4의 중앙값, q cache는 전체 train×4 view다. **F2가 실제로 사용한 calibration index 배열을 재사용**한다. seed1234라는 숫자만 같게 주고 다른 RNG API로 subset을 다시 뽑지 않는다. q에는 LP/HP나 U-Net이 들어가지 않지만, τ_R에는 Teacher 자신의 PL forward가 들어간다.

q_ref>0·finite와 cache covering을 검사한다. q_ref가 .46875에 가깝다는 이유로 임의의 calibration failure로 만들지 않는다. absolute q, std/분위수, 반경별 slope, native c, 실제 s 분포를 보고한다.

## 5. 기존 checkpoint 진단 — 새 학습과 최종 성능표를 구분

### 5.1 서버별 로컬 대상

| 서버 | 우선 진단할 기존 자산 |
|---|---|
| s1 | F1 아래 기존 P0·PLH W104D121 두 run, F1 q; 기존 T0가 로컬에 있으면 같은 probe로 참고 비교 |
| s2 | F2 q/native c와 N2 donor. 새 N2PL이 끝나면 같은 calibration view로 비교 |
| s3 | F3 아래 기존 PH W104D121/D122 |
| s4 | F4 아래 기존 PLH W104D122, PLH W112D121 |
| s5 | F5 아래 기존 PLH W104D121/D122 |

s3/s4의 대표 진단을 s1으로 모아야만 시작하는 구조는 아니다. 각자 같은 스키마로 저장하고 결과를 추후 합친다. 아직 없는 새 모델을 기다리지 않는다.

### 5.2 A/U 3×3

한 run에서 step `{10100,24240,50000}`의 A와 U를 각각 조합한다. **다른 run·layout·W/D를 섞지 않는다.** 동일시점 세 조합이 먼저 원 평가를 재현해야 한다. 그다음 나머지 여섯 조합에서 RR20/FR20 전체를 평가한다.

\[
Z_{a,u}=M+F_{\theta_u}(\operatorname{concat}[W(P,c_a),W(L,c_a),W(P,c_a)-W(L,c_a),M]),
\quad c_a=A_{\phi_a}(P,M).
\]

실제 concat 항은 해당 layout에 맞춘다. 모든 조합에서 U는 full RR256/FR512로 실행한다. Aligner margin만4다. GT 없이 추론한 후 별도 RR metric을 계산한다. 모델 weight는 frozen이고 원 checkpoint를 덮어쓰지 않는다.

두 source SHA, step_A/step_U, layout/W/D, split/sample ID, H/E/Dλ/Ds, native c, 각 장면 차이를 `diagnostics/au_cross/`에 저장한다. 같은 source의 정식 diagonal cache가 identity까지 일치하면 재사용하고 반복 추론으로 시간을 채우지 않는다.

### 5.3 s1 A10 실행 분기 — 이번에 제안한 운영 기준

다음 숫자는 **FH12에서 최적화해 얻은 정답이 아니라 이번 진단의 사전 지정 기준**이다. s1의 기존 P0/PLH 두 run에서 U50000을 고정하고 A10100/A24240을 각각 적용한 네 비교를 검사한다. 비교 기준은 각 run의 A50000/U50000이다.

한 비교의 `support=true` 조건:

- 평균 raw HQNR 변화 ≥ +0.0010;
- 평균 D_s 변화 ≤ −0.0010;
- 공식 RR ERGAS 악화 ≤ +0.0030;
- FR20 중 적어도 12장에서 D_s 개선.

네 비교 중 두 개 이상이 support이고, 그중 PLH run의 비교가 적어도 하나면 `S1_A_SUPPORT`. 정상 diagonal 재현이 통과했지만 기준 미달·상충·진단 산출물 불충분이면 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`로 정한다. **반대/불충분을 ‘A는 원인이 아니다’라고 단정하지 않는다.**

분기는 첫 새 Student pair 전 한 번 기록한다. s1의 뒤쪽 pair 결과를 보고 유리한 branch로 재선택하지 않는다. 진단의 초기 처리 예약은 약1시간이고, 추가 자산 전송·확장 진단 때문에 다른 서버나 s1 학습이 무한 대기하지 않게 한다. 기초 reference/forward 무결성 실패는 이 과학적 미지지 분기와 다르며 해당 서버의 학습을 안전하게 중지해야 한다.

A10 미지지 시에는 **PLH W104D121/D122 BASE pair**를 같은 seed 목록으로 실행한다. 일하지 않고 다른 서버 결과를 기다리는 대기는 없다. A10이 지지돼도 나머지 서버의 기본 LR를 자동 변경하지 않는다.

이 진단은 RR/FR 개발 자료를 사용한 가설 선택이므로 최종 독립 test 검증이라고 보고하지 않는다.

## 6. 유일한 Student 학습정책 변형: A10

BASE는 기존 G23과 같다. A10은 **loss가 아니라 A의 LR schedule만** 바꾼다.

FH12 cosine/warmup이 만든 multiplier를 g(t)라고 하자. 해당 run의 첫 update 직전 t=0이고, 완료한 update 수가 t다.

\[
\eta_U(t)=10^{-4}g(t),\qquad
\eta_A^{BASE}(t)=3\times10^{-6}g(t),
\]
\[
\eta_A^{A10}(t)=3\times10^{-6}g(t)
\begin{cases}1,&t<10000\\1/3,&t\ge10000.\end{cases}
\]

10,000회 update를 마친 뒤 **10,001번째 update부터** A multiplier를 1/3로 한다. nominal A peak가 1e−6인 후반 경로이며 실제 LR는 여전히 cosine에 의해 줄어든다. U LR·optimizer state·weight decay·warmup·학습량은 바꾸지 않는다. A를 freeze하거나 optimizer를 재생성하지 않는다. LR와 함께 AdamW update 크기가 달라지는 것은 이 optimizer-policy 대조에 포함된다.

**후반에 step마다 LR에 1/3을 누적 곱하지 않는다.** 항상 base LR×cosine×정해진 multiplier로 계산해야 한다. 기존 single scheduler가 다음 update에서 A LR를 다시 덮어쓰지 않도록 param-group별 schedule test가 필요하다.

BASE/A10은 모두 fresh50K다. 기존 24K weights에서 새 optimizer로 이어 학습한 것을 exact continuation으로 표시하지 않는다. 새 run에는 10K·24,240·50K의 full optimizer/RNG state를 추가 보존하되, 10K를 공식 후보 격자에 끼워 넣지 않는다. 두 profile의 10K 이전 data order·init·LR 계약이 같음을 검사한다.

## 7. 서버별 정확한 학습 순서

표의 모든 새 Student는 50K이며, Teacher A clone+fresh U다. BASE/A10 이외 계수는 §3과 같다. `Bxx`는 atomic block이고, block 안의 pair/quartet을 성능을 이유로 중도 취소하지 않는다.

### 7.1 s1 — 진단 + 후반 A LR 또는 depth 대조

Reference는 **F1(P0 Teacher T71001 exact50K)** 고정이다. Teacher를 바꾸지 않는다.

| Tier / block | Student seed | A-support 경로 | A 미지지·불충분 경로 | 새 run 수 |
|---|---|---|---|---:|
| CORE B01–B07 | **73101 / 73102 / 73103 / 73104 / 73105 / 73106 / 73107** | PLH W104D122: BASE와 A10 | PLH W104: D121 BASE와 D122 BASE | 14 |
| RESERVE B08–B11 | **73108 / 73109 / 73110 / 73111** | 같은 비교 | 같은 비교 | 최대8 |

각 seed의 공통 D122 BASE는 한 번만 실행한다. 두 branch를 모두 실행하거나 같은 BASE를 이름만 바꿔 두 번 돌리지 않는다. s1의 절대 성능을 s4의 다른 Teacher 결과와 바로 비교해 LR 효과라고 부르지 않는다.

### 7.2 s2 — N2 Teacher 초기화 + 같은 Student의 reference 대조

첫 Teacher:

`FH20R1_S2_T_PL_W112_D123_WV3_TS71001_N2INIT50_v1`

완료 후 N2PL calibration을 발행한다. F2는 원 fresh Teacher 그대로 보존한다. **Teacher 초기화 대조의 독립 Teacher 반복은 1쌍**이며, 아래의 여러 Student seed가 이를 여러 Teacher 반복으로 바꾸지는 않는다.

| Tier / block | Student seed | 정상 경로: 같은 Student PL W104D122 | donor/reference 불가 대체 경로 | 새 run 수 |
|---|---|---|---|---:|
| CORE B01 | **72001** | F2 vs N2PL | F2 아래 PL vs PLH, W104D122 | 2 |
| CORE B02–B05 | **73201 / 73202 / 73203 / 73204** | F2 vs N2PL | 동일 대체 비교 | 8 |
| RESERVE B06–B09 | **73205 / 73206 / 73207 / 73208** | F2 vs N2PL | 동일 대체 비교 | 최대8 |

F2-vs-N2PL의 Student 비교에서는 Teacher output, A initialization, τ_R, q가 함께 달라진다. 이는 **전체 Teacher reference의 효과**다. 두 Teacher 학습에서의 유일한 의도적 변경이 A 초기화라는 것과 구분한다.

N2 donor가 없으면 N2 Teacher를 fresh A로 바꿔 같은 ID로 실행하지 않는다. N2 reference가 준비되지 않은 새 Student만 대체로 보내고 이미 끝난 F2 PL baseline은 재사용한다. 대체 run 이름은 `TF2`·`PLH`로 분리돼 있다. A-support 결과를 기다리지 않는다.

### 7.3 s3 — PH의 depth 반복과 작은 width 결합

Reference **F3(PH Teacher T71001 exact50K)** 고정.

| Block | Seed | 비교 | 수 |
|---|---|---|---:|
| CORE B01 | **72002** | PH W104 D121 / D122 | 2 |
| CORE B02–B05 | **73301 / 73302 / 73303 / 73304** | PH W104 D121 / D122 | 8 |
| CORE B06–B10 | **73305 / 73306 / 73307 / 73308 / 73309** | PH D122에서 W104 / W112 | 10 |
| RESERVE B11–B14 | **73310 / 73311 / 73312 / 73313** | PH W104 D121 / D122 | 최대8 |

기존 seed72001의 PH D121/D122/W112D121은 재학습하지 않는다. seed72002의 첫 pair가 직전 제안의 반복 검증이다. W112D122는 새 가설이므로 D122-only의 결과와 분리한다. D123/124 또는 W128 이상으로 확장하지 않는다.

### 7.4 s4 — PL×D122를 먼저, 입력×depth의 중심 비교

Reference **F4(PLH Teacher T71001 exact50K)** 고정.

| Block | Seed | 새로 실행할 case | 수 |
|---|---|---|---:|
| CORE B01 | **72001** | **PL W104D122** | 1 |
| CORE B02–B05 | **73401 / 73402 / 73403 / 73404** | 각 seed에 PL/PLH × W104D121/D122 네 조건 | 16 |
| CORE B06 | **72001** | PLH W112D122 | 1 |
| CORE B07 | **73405** | PLH W112D121 / W112D122 | 2 |
| RESERVE B08–B09 | **73406 / 73407** | 각 seed에 PL/PLH × W104D121/D122 | 최대8 |

B01은 기존 PL D121·PLH D121/D122와 함께 입력×depth 2×2를 완성한다. B06은 기존 W112D121·W104D122와 비교할 capacity 결합점이다. **이미 있는 셋을 새 이름으로 재실행하지 않는다.** 새 seed quartet은 네 조건을 모두 같은 서버·Teacher·seed로 진행하므로 주요 통계 단위가 된다.

### 7.5 s5 — 두 번째 Teacher 아래 입력×depth 확인

Reference **F5(PLH Teacher T71002 exact50K)** 고정. s4와 Teacher가 다르다는 사실을 보존한다.

| Block | Seed | 새로 실행할 case | 수 |
|---|---|---|---:|
| CORE B01 | **72002** | **PL W104D121 / PL W104D122** | 2 |
| CORE B02–B05 | **73501 / 73502 / 73503 / 73504** | 각 seed에 PL/PLH × W104D121/D122 | 16 |
| CORE B06 | **72002** | PLH W112D121 / W112D122 | 2 |
| RESERVE B07–B08 | **73505 / 73506** | 각 seed에 PL/PLH × W104D121/D122 | 최대8 |

기존 PLH W104D121/D122 S72002를 재사용해 B01에서 기존 2×2를 완성한다. B06은 같은 seed에 W104/W112×D121/D122의 PLH 2×2를 완성한다. s5 내 신규 Student seed 비교는 Teacher를 고정한 반복이며, s4-vs-s5 자체는 Teacher/서버 차이가 함께 있는 비교다.

### 7.6 seed와 block의 일반 규칙

- `72001`, `72002`는 **누락 조합을 메우거나 기존 대응을 연결할 때만** 재사용한다. `73xxx`는 이번 신규 반복 seed다. 서로 다른 값을 별칭으로 취급하지 않는다.
- 같은 seed pair/quartet은 같은 초기 backbone 텐서·배치 순서·증강 순서를 공유한다. 각 run optimizer는 fresh다.
- pair 순서와 quartet 순서를 block별로 반대로 배치해 특정 설정이 항상 먼저 실행되는 편향을 줄인다. 정확한 순서는 JSON `primary_order`/`alternative_orders`와 부록 A가 기준이다.
- Student A의 학습된 결과는 run마다 다르다. ‘같은 Teacher를 썼으니 학습된 A도 같다’고 가정하지 않는다.
- 통계 집계는 Teacher/서버/seed block 안의 대응 차이를 먼저 계산한다. 여러 checkpoint나 같은 baseline을 공유하는 차분을 독립 표본으로 부풀리지 않는다.

## 8. 각 서버 20시간 이상을 실제로 채우는 운영 방식

### 8.1 새 누적시간의 정의

`effective_seconds`는 이번 campaign에서 수행한 **유효한 학습, 공식 평가, 필요한 calibration, 사전 지정 진단 작업의 중복 없는 실행시간**이다. 서버별 기본 장치 한 개를 기준으로 기록한다.

포함: 새 Teacher/Student 학습, 후보 RR/FR/validation 공식 평가, 검증에 사용한 q calibration 및 A/U 진단. 실제 작업 중 필수 local I/O는 해당 job의 일부로 기록하되 별도 항목도 남긴다.

제외: 이전 FH12/MIX20H 시간, SSH/다운로드 대기, code edit, 수동 대기, network 업로드 재시도, sleep, 실패한 smoke만 반복한 시간, 같은 checkpoint를 불필요하게 다시 평가한 시간. 유효 full-state로 보존한 진행분은 정확히 한 번만 누적한다.

학습 loop 시간과 그 안에서 호출된 평가 시간, parent process와 child process 시간을 이중 합산하지 않는다. 겹친 작업 구간은 interval union으로 집계한다. 별도로 `train_seconds`, `eval_seconds`, `calibration_seconds`, `diagnostic_seconds`, `waiting_seconds`, `wall_elapsed_seconds`도 보고한다.

> 이것은 엄밀한 GPU kernel busy-time 20시간이나 ‘optimizer update만 20시간’이라는 뜻은 아니다. 사용자 요청의 **실험 유효 가동시간**을 20시간 이상 확보하는 운영 정의다. 준비·대기만으로 시간을 채워 완료로 처리하지 않는다.

### 8.2 실측 근거와 최초 편성값

기존 FH12의 `Train(h)`는 학습 loop 시간이다. 아래의 Student 시간은 **completed UTC−started UTC**를 계산한 값으로 평가·후처리도 포함한다. 새 캠페인의 실제 유효시간 대신 초기 예약 추정에만 쓴다. D122/W112D122·진단·donor import에 따른 추가 비용은 첫 완결 block에서 갱신한다. [R2; R3]

| 서버 | FH12 Student 수 | Student 학습시간 중앙값(h) | 학습+평가 경과 중앙값(h) | 기본 신규 T/S | 기본 예상 유효작업(h, 추정) |
|---|---:|---:|---:|---|---:|
| s1 | 2 | 1.267 | 1.568 | 0T + 14S | 약 22.9 |
| s2 | 2 | 1.639 | 1.893 | 1T + 10S | 약 21.1 |
| s3 | 4 | 0.940 | 1.070 | 0T + 20S | 약 21.9 |
| s4 | 6 | 0.998 | 1.126 | 0T + 20S | 약 23.0 |
| s5 | 2 | 0.964 | 1.091 | 0T + 20S | 약 22.3 |


s1 진단1h, s2 진단/calibration0.75h, s3–s5 각0.5h를 예약값으로 더했다. 이는 측정된 소요시간이 아니며 그만큼 강제로 실행하라는 뜻도 아니다. 새 W112D122의 시간도 아직 실측되지 않았으므로 위 값에 완결 보장은 없다.

### 8.3 종료·예비 진입 규칙

1. **명시적 새 campaign 시작**에서 각 서버별 ledger를 생성한다. 기존 FH12의 12h deadline을 연장/초기화하지 않는다.
2. 해당 서버의 적용 가능한 CORE block을 모두 완결한다. 20시간이 먼저 지나도 이미 지정된 핵심 비교를 임의로 버리지 않는다.
3. CORE가 끝났고 `effective_hours>=20`이면 현재 block의 공식 평가·로컬 저장을 마친 뒤 종료한다. 모든 RESERVE를 돌릴 필요는 없다.
4. CORE가 끝났는데 `<20`이면 다음 RESERVE block을 성능과 무관하게 순서대로 실행한다. 두 run pair 또는 네 run quartet을 통째로 완결한다.
5. RESERVE까지 모두 소진했는데도 `<20`이면 `CAPACITY_EXHAUSTED_BELOW20`으로 정확하게 보고한다. 같은 seed 재실행·sleep·가짜 cost로 시간을 채우거나 seed를 무한 자동 생성하지 않는다.
6. 하나의 좋은 결과가 나왔다고 일찍 전체 서버를 종료하지 않는다. 반대로 threshold 미달 때문에 같은 run을 반복하지 않는다.

최소20h와 block 완결 조건 때문에 실제 종료는 보통 **21–25h 정도를 계획값**으로 보며, quartet 예비가 필요하면 그보다 길어질 수 있다. 이 범위는 보장이나 하드 deadline이 아니다. 누적20h 도달 시 실행 중인 정상 50K 학습을 강제 종료하지 않는다.

각 서버의 실패·asset 문제가 생기면 그 서버의 가능한 독립 branch를 진행하고, 다른 서버는 계속한다. 기본 reference 자체가 손상됐거나 공통 evaluator 오류라면 안전한 정지가 우선이고, 이를 시간 목표보다 우선하는 예외로 기록한다.

## 9. 구현 인계 — ‘case가 문서에 있음’과 ‘서버 실행 가능’을 분리

### 9.1 새 runner가 필요한 이유

기존 FH12는 run ID 21개, 12h window, local Teacher-from-scratch 규칙을 고정한 전용 경로다. 본 계획에는 새 seed, 기존 reference 재사용, N2-initialized Teacher, 최소20h, A10이 있으므로 **기존 FH12 runner에 ID만 추가해 실행됐다고 간주할 수 없다.** [R7; R9]

다음 작업이 구현돼야 한다.

| 항목 | 요구 사항 |
|---|---|
| Registry/config generator | 제공 JSON의 정확한 case별 값을 실제 YAML로 생성. 두 branch를 전부 실행하지 않음 |
| Model builder | FH12 sync frontend/입력10ch/공통 init 유지. N2INIT는 명시적 A-only load 정책으로 분리 |
| Training | 기존 loss/routing 유지, A10 param-group schedule, 50K/full-state 저장 |
| Import | 구 FH12 자산을 읽기 전용으로 검증해 새 reference bridge를 발행 |
| Runner | CORE→시간예비, 서버별 유효시간 최소20h, local branch 결정, 재시작 멱등 |
| Evaluation | 50후보/다섯 selector, 정상 A_ON와 A/U 진단 분리 |
| Upload | 기존 WV3 탭 유지, 새 campaign ID/run ID upsert, readback |

`tools/fh20r1_start.sh` 등 새 entrypoint 이름은 **구현할 인터페이스 제안**이다. 현재 존재하거나 실행 가능한 명령으로 보장하지 않는다. 기존 `tools/fh12_start.sh`나 `_run_cases.sh`를 그대로 호출해 본 campaign이 돌아간다고 안내하지 않는다.

CSV는 145개 **학습 정의**의 목록이다. JSON에는 11개 준비/진단/calibration stage와 atomic block·의존성이 추가돼 있다. `planned_config_path`는 구현 목표 경로이며, 이 인계본에 실제 trainer-ready YAML이 생성됐다는 의미가 아니다.

### 9.2 시작 전 필수 검증

- 원 FH12 파일·checkpoint·LP cache·Sheet 행을 덮어쓰지 않았는가?
- snapshot 이후 이미 완료된 동일 의미의 run(Teacher SHA/layout/W/D/seed/profile/data/학습 정의)이 로컬이나 Sheet에 있는가? 완전한 identity·평가 동등성을 검증한 경우만 결과를 연결해 재사용하고 신규 계산시간은 0으로 기록한다. 이름만 비슷하거나 seed만 같은 경우는 재사용하지 않는다. 같은 FH20R1 ID의 진행 중 자산이 있으면 덮어쓰지 않고 원 full-state의 exact resume로만 이어간다.
- 해당 서버 ID/GID와 새 campaign 경로가 정확한가? s3는 `WV3-s3(5090)`다.
- F1–F5의 whole Teacher checkpoint SHA, layout, reference step, τ/q cache가 일치하는가?
- BASE branch의 수치 forward와 loss/routing이 기존 FH12와 일치하는가?
- N2INIT는 A만 load하고 U fresh/init hash·offset RNG는 F2와 같게 유지되는가?
- 추가 input stem kernel mapping과 0 초기화, c의 모든 PAN-derived 경로 gradient가 맞는가?
- 같은-seed pair에서 body init/data order가 일치하고, A10의 LR switch가 10,001번째 update인가?
- scheduled LR가 매번 1/3씩 줄어드는 누적 곱 버그가 없는가?
- RR 공식 evaluator의 support·range·clipping과 FR raw PAN reference가 유지되는가?
- local donor/diagnostic 미확보 시 fallback 큐가 실제로 연결되며, 다른 서버 결과를 읽어야만 진행하는 gate가 없는가?
- 소모성 smoke가 actual Teacher/Student init·RNG에 재사용되지 않는가?
- disk 여유가 50 model candidates + full-state checkpoints + diagnostics + resume 저장을 감당하는가?

### 9.3 실제 실행 준비 완료를 알리는 서버 보고 형식

각 서버는 최초 시작 전에 아래와 같은 record를 생성한다. 값은 실제 결과로 채우고 임의의 `true`나 SHA placeholder로 통과시키지 않는다.

```text
server / campaign_id / execution_release / numeric_method_revision
reference_alias / origin_teacher_run / exact50k_sha / reference_bridge_sha
selected_local_branch / branch_reason / branch_report_sha
number_of_active_core_runs / number_of_active_reserve_runs
next_block_id / next_two_or_four_run_ids
resolved_yaml_paths / queue_sha / loss_routing_test_status
historical_step0_parity_status / actual_batch48_smoke_status
min_effective_hours=20 / credited_hours=0 / counted_categories
actual_start_authorized / old_campaign_drain_status / uploader_schema
```

‘MD를 읽었다’, ‘config generator가 있다’, ‘registry에 이름이 있다’만으로 실행준비 완료로 보고하지 않는다. 본 문서는 서버 적용·학습 기동 명령을 이미 실행했다고 주장하지 않는다.

## 10. Sheet·산출물·재현 기록

### 10.1 탭과 열

기존 WV3-s1/s2/s3(5090)/s4/s5 탭의 **RR·FR·Cost·NOA·Target·Exact50K·RR_VAL_SELECTED·E_MIN** 구조를 유지한다. Q8은 WV3이므로 바꾸지 않는다. 새로운 FH20R1 provenance group만 오른쪽에 추가한다.

필수 provenance: campaign/run/block, server, layout/W/D, T alias/T layout/T seed/whole SHA, Student seed, profile(BASE/A10/N2INIT), LR schedule ID, init/P-MS/body/A hashes, LP split SHA/phase, data/release/evaluator SHA, reference τ/q/calibration SHA, actual updates, effective-time breakdown, target eligible count, branch type/reason, upload status.

본 행 RR/FR은 같은 RAW_MAX checkpoint다. TARGET과 다른 checkpoint의 metric을 혼합하지 않는다. JQM/V64/NOA를 실제로 계산하지 않았다면 빈칸을 유지한다. 다른 모델의 값으로 채우거나 0으로 표시하지 않는다. 기존 NOA 열을 전체 신규 run의 선행 필수 gate로 만들지 않는다.

A/U 교차 결과는 전용 진단 파일/별도 진단 표에만 기록하고 본 성능행에 넣지 않는다. 업로드는 기존 프로젝트 gspread 경로를 확장하고 `(campaign,run)` upsert/readback을 사용한다. upload failure는 학습 실패가 아니며 로컬 결과를 보존하고 upload만 재시도한다.

### 10.2 저장 규칙

```text
work_dir/_fh20r1/<server>/
  campaign_budget.json           # 새 최소20h와 ledger 정의
  admission_events.jsonl         # case/block 시작·종료·재개
  active_intervals.jsonl         # 중복 없는 유효시간 근거
  branch_record.json             # s1/s2 로컬 결정; 공통 recipe lock 아님
  imported_refs/<alias>/
    bridge_manifest.json         # 원 자산은 읽기 전용
  diagnostics/
    au_cross/<source_run>/...
    q_response/<teacher_alias>/...
  completion_report.json

work_dir/<FH20R1_run_id>/
  meta/config.resolved.yaml
  meta/training_start_manifest.json
  init_manifest.json
  diagnostics/loss_routing.json
  diagnostics/loss_terms_and_activity.csv
  diagnostics/gradient_norms.csv
  diagnostics/frequency_shift.csv
  diagnostics/lr_groups.csv
  candidates/<fixed_grid_step>/model.safetensors
  restart_fullstates/{10000,24240,50000}/...
  official/raw_grid.json
  official/{raw_max,target_selection,exact50k,rr_val_selected,e_min_diag}.json
  official/profile.json
  official/postrun_status.json
```

새 diagnostics는 모델 forward·RNG·optimizer를 바꾸지 않게 구현한다. 별도 gradient norm 계산 때문에 실제 optimizer gradient가 합산되거나 추가 step이 실행되면 안 된다. gradient 진단 주기는 매5000 update의 고정 calibration mini-batch로 제한하고, 실제 학습 RNG를 복원한다. 임의 train sample을 추가 소비하지 않는다. hard/soft/edge loss, soft-positive 비율, Student-better 비율, q/s 분포를 함께 남긴다.

## 11. 결과 해석과 다음 채택 규칙

### 11.1 무엇이 확인돼야 하는가

| 비교 | 핵심 질문 | 비교 단위 |
|---|---|---|
| PL/PLH × D121/D122 | PL의 공간 균형과 depth의 RR 개선이 같이 유지되는가? | 같은 Teacher/서버/seed의 quartet |
| PH D121/D122 | FH12 D122 이득이 Student seed를 바꿔도 유지되는가? | 같은 Teacher/seed pair |
| W104/W112 at D122 | 소폭 width가 남은 ERGAS 차이를 줄이는가? | 같은 Teacher/layout/D/seed pair |
| BASE/A10 | 후반 A update 감소가 낮은 E 구간에서 H를 보존하는가? | 같은 Teacher/layout/W/D/seed pair |
| F2/N2PL | 개선된 Teacher A 초기화가 q 및 Student의 목표 성능에 기여하는가? | Teacher 쌍 + 각 Student seed의 reference pair |

각 비교는 TARGET과 EXACT50K, validation-selected를 모두 보고한다. H 하한 통과 수, 공동목표 통과 수, 적격 checkpoint 수/연속 구간, Dλ/Ds 변화, Params/추론비용을 병기한다. 유효 pair 수와 실패·미시작 수를 따로 표시하고 성공 seed만 남기지 않는다.

q가 좋아졌지만 RR/FR가 나빠지면 Teacher를 자동 채택하지 않는다. A10이 H만 올리고 E를 크게 악화시키면 공동 해결책이 아니다. W112D122가 E<2.040을 달성해도 H가 미달이면 RR 개선 후보로 보존하되 최종 성공으로 표시하지 않는다.

### 11.2 이번 round에서 하지 않는 것

PAN auxiliary/MARs, attention, Student offset consistency, direct c KD, loss 합산 후 A/U 통합 backward, HP target edge, α/β/λE grid, τ/q reference 숫자 수동 조절, test별 최적 phase/warp, 학습 후 output shift·band 보정, 비공식 RR metric을 공식값으로 대체하는 것. W/D는 이 문서의 W104/W112, D121/D122 범위만 사용한다.

### 11.3 종료 보고

서버별 실제 유효시간≥20, 적용 CORE 완결, 시간예비 실행/미실행 이유, run별50K/후보50/정식평가·업로드 상태, 중단·대체 branch와 계수/방법 변경 여부를 보고한다. 모두 충족했을 때만 `COMPLETE_20HPLUS`로 표시한다. donor 진단이 불가능했지만 fallback으로 시간을 충족했다면 `COMPLETE_20HPLUS_WITH_BLOCKED_DONOR`로 구분하고 Teacher 가설이 검증됐다고 쓰지 않는다.

작업 종료 후 QB/GF2나 옛 FH12/MIX20H 큐를 자동으로 다시 시작하지 않는다. 새 결과를 검토한 다음 방향을 결정한다.

## 12. 인계 파일과 검증 범위

- 본 MD: 목적·수식·server별case·예산·구현·평가의 단일 명세.
- `PAN_FH20R1_Cases_2026-09-19.csv`: 145개 학습 정의. condition/tier를 먼저 해석해야 한다.
- `PAN_FH20R1_Registry_2026-09-19.json`: references, 11 stages, block primary/alternative 순서, case별 수치 설정.
- `PAN_FH20R1_RuntimeBudget_2026-09-19.csv`: FH12 실측시간과 이번 편성의 초기 계산.
- `queues/*.plan.txt`: 서버별 읽기용 순서. shell로 직접 실행하는 명령 파일이 아니다.
- `plan_audit.json`: ID/seed/dependency/상호배타조건/기존 의미상 중복 및 수식계수 검사 결과.

인계본에서 검증한 것은 **정의의 일관성**이다. 원격 YAML 생성, 실제 batch48 GPU 실행, donor 로컬 존재, 새 reference importer, runner/Sheet writes의 실동작은 아직 확인하지 않았다.

## 출처

[R1] `PAN_FH12_Results_Review_and_NextDirection_2026-09-19.md`, 특히 §§4–10. 관측 결과와 직전 후속6개 제안의 근거.  
[R2] `PAN_FH12_21Run_Results_2026-09-19.csv`, 21run의 metric·reference SHA·τ/q·Train(h).  
[R3] 첨부 분석 ZIP의 `FH12_21runs_full_2026-09-19.csv`, started/completed UTC 및 original row metadata. 이번 시간표는 이 값에서 계산.  
[R4] PAN-Crafter, 제공 PDF, Table 1/6의 WV3 보고 H=.958/E=2.040. 이 목표를 이긴 것만으로 최신 모든 방법의 SOTA라고 주장하지 않는다.  
[R5] `PAN_FH12_WV3_LPAN_HPAN_FreshTeacher_12H_2026-09-18.md`, §§2–6의 frontend/초기화/calibration 정의.  
[R6] 제공 `Pansharpening_research.pdf`, p.8–15의 Teacher consistency→frozen reference→Student hard/soft/edge/A adjustment 구조.  
[R7] 저장소 `hojunking/PAN-Crafter-repro`, `a5e9f858ed33349f8d67a3d30007faa9977cb064`, `fh12/training.py`의 validate_config/runtime_context/reference/source/resume 검사. 이번에 해당 revision을 다시 읽음.  
[R8] 같은 revision, `config/PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1.yaml`, N2 donor run·container SHA·margin·step. 이번에 다시 읽음.  
[R9] 같은 revision, `fh12/plan.py`의 21case/12h 원 registry와 FH12 config 정의. 이번에 다시 읽음.

**[R1–R9]는 출처이고, 본 문서의 20h 최소 예산·신규 seed/배정·s1 진단 기준·A10·fallback은 이번에 새로 제안한 설계다. 관측된 사실과 혼동하지 않는다.**

---

# 부록 A. 정확한 학습 run ID와 서버별 순서

아래는 `Registry JSON`에서 생성한 목록이다. CORE를 먼저, 시간 미달이면 RESERVE를 순서대로 실행한다. 각 block의 alternative는 primary와 상호 배타적이다. 새 teacher와 calibration은 s2 block 전에 수행한다. 이 목록의 존재는 원격 YAML/runner 등록 완료를 뜻하지 않는다.

## s1: 읽기용 queue plan

Import: `FH20R1_S1_IMPORT` → diagnostic: `FH20R1_S1_DIAG`

### FH20R1_S1_B01 / CORE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_A10_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

### FH20R1_S1_B02 / CORE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73102_A10_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73102_BASE_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73102_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73102_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

### FH20R1_S1_B03 / CORE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73103_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73103_A10_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73103_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73103_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

### FH20R1_S1_B04 / CORE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73104_A10_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73104_BASE_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73104_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73104_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

### FH20R1_S1_B05 / CORE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73105_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73105_A10_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73105_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73105_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

### FH20R1_S1_B06 / CORE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73106_A10_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73106_BASE_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73106_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73106_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

### FH20R1_S1_B07 / CORE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73107_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73107_A10_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73107_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73107_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

### FH20R1_S1_B08 / RESERVE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73108_A10_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73108_BASE_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73108_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73108_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

### FH20R1_S1_B09 / RESERVE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73109_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73109_A10_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73109_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73109_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

### FH20R1_S1_B10 / RESERVE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73110_A10_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73110_BASE_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73110_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73110_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

### FH20R1_S1_B11 / RESERVE / BASE vs A10; fallback D121 vs D122

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73111_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73111_A10_FRESH50_v1
```

대체 조건 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`일 때 **위 목록 대신**:
```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73111_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73111_BASE_FRESH50_v1
```

Select branch once from local diagnostics before any block training; do not run both alternatives.

## s2: 읽기용 queue plan

Import: `FH20R1_S2_IMPORT` → diagnostic: `FH20R1_S2_DIAG`

Donor 확인 정상일 때만 다음 Teacher 및 calibration을 먼저 수행한다:
`FH20R1_S2_T_PL_W112_D123_WV3_TS71001_N2INIT50_v1` → `FH20R1_S2_CAL_N2PL`

### FH20R1_S2_B01 / CORE / Fresh-vs-N2 Teacher reference; fallback PL-vs-PLH

```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS72001_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TN2PL_TS71001_SS72001_BASE_FRESH50_v1
```

대체 조건 `S2_DONOR_OR_REFERENCE_UNAVAILABLE`일 때 **위 목록 대신**:
```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS72001_BASE_FRESH50_v1
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS72001_BASE_FRESH50_v1
```

Primary compares complete Teacher references. q/tau recalibrated for N2PL. Donor failure does not wait for another server.

### FH20R1_S2_B02 / CORE / Fresh-vs-N2 Teacher reference; fallback PL-vs-PLH

```text
FH20R1_S2_S_PL_W104_D122_WV3_TN2PL_TS71001_SS73201_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73201_BASE_FRESH50_v1
```

대체 조건 `S2_DONOR_OR_REFERENCE_UNAVAILABLE`일 때 **위 목록 대신**:
```text
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73201_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73201_BASE_FRESH50_v1
```

Primary compares complete Teacher references. q/tau recalibrated for N2PL. Donor failure does not wait for another server.

### FH20R1_S2_B03 / CORE / Fresh-vs-N2 Teacher reference; fallback PL-vs-PLH

```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73202_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TN2PL_TS71001_SS73202_BASE_FRESH50_v1
```

대체 조건 `S2_DONOR_OR_REFERENCE_UNAVAILABLE`일 때 **위 목록 대신**:
```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73202_BASE_FRESH50_v1
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73202_BASE_FRESH50_v1
```

Primary compares complete Teacher references. q/tau recalibrated for N2PL. Donor failure does not wait for another server.

### FH20R1_S2_B04 / CORE / Fresh-vs-N2 Teacher reference; fallback PL-vs-PLH

```text
FH20R1_S2_S_PL_W104_D122_WV3_TN2PL_TS71001_SS73203_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73203_BASE_FRESH50_v1
```

대체 조건 `S2_DONOR_OR_REFERENCE_UNAVAILABLE`일 때 **위 목록 대신**:
```text
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73203_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73203_BASE_FRESH50_v1
```

Primary compares complete Teacher references. q/tau recalibrated for N2PL. Donor failure does not wait for another server.

### FH20R1_S2_B05 / CORE / Fresh-vs-N2 Teacher reference; fallback PL-vs-PLH

```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73204_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TN2PL_TS71001_SS73204_BASE_FRESH50_v1
```

대체 조건 `S2_DONOR_OR_REFERENCE_UNAVAILABLE`일 때 **위 목록 대신**:
```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73204_BASE_FRESH50_v1
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73204_BASE_FRESH50_v1
```

Primary compares complete Teacher references. q/tau recalibrated for N2PL. Donor failure does not wait for another server.

### FH20R1_S2_B06 / RESERVE / Fresh-vs-N2 Teacher reference; fallback PL-vs-PLH

```text
FH20R1_S2_S_PL_W104_D122_WV3_TN2PL_TS71001_SS73205_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73205_BASE_FRESH50_v1
```

대체 조건 `S2_DONOR_OR_REFERENCE_UNAVAILABLE`일 때 **위 목록 대신**:
```text
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73205_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73205_BASE_FRESH50_v1
```

Primary compares complete Teacher references. q/tau recalibrated for N2PL. Donor failure does not wait for another server.

### FH20R1_S2_B07 / RESERVE / Fresh-vs-N2 Teacher reference; fallback PL-vs-PLH

```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73206_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TN2PL_TS71001_SS73206_BASE_FRESH50_v1
```

대체 조건 `S2_DONOR_OR_REFERENCE_UNAVAILABLE`일 때 **위 목록 대신**:
```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73206_BASE_FRESH50_v1
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73206_BASE_FRESH50_v1
```

Primary compares complete Teacher references. q/tau recalibrated for N2PL. Donor failure does not wait for another server.

### FH20R1_S2_B08 / RESERVE / Fresh-vs-N2 Teacher reference; fallback PL-vs-PLH

```text
FH20R1_S2_S_PL_W104_D122_WV3_TN2PL_TS71001_SS73207_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73207_BASE_FRESH50_v1
```

대체 조건 `S2_DONOR_OR_REFERENCE_UNAVAILABLE`일 때 **위 목록 대신**:
```text
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73207_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73207_BASE_FRESH50_v1
```

Primary compares complete Teacher references. q/tau recalibrated for N2PL. Donor failure does not wait for another server.

### FH20R1_S2_B09 / RESERVE / Fresh-vs-N2 Teacher reference; fallback PL-vs-PLH

```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73208_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TN2PL_TS71001_SS73208_BASE_FRESH50_v1
```

대체 조건 `S2_DONOR_OR_REFERENCE_UNAVAILABLE`일 때 **위 목록 대신**:
```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73208_BASE_FRESH50_v1
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73208_BASE_FRESH50_v1
```

Primary compares complete Teacher references. q/tau recalibrated for N2PL. Donor failure does not wait for another server.

## s3: 읽기용 queue plan

Import: `FH20R1_S3_IMPORT` → diagnostic: `FH20R1_S3_DIAG`

### FH20R1_S3_B01 / CORE / PH D121 vs D122

```text
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS72002_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS72002_BASE_FRESH50_v1
```

### FH20R1_S3_B02 / CORE / PH D121 vs D122

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73301_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73301_BASE_FRESH50_v1
```

### FH20R1_S3_B03 / CORE / PH D121 vs D122

```text
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73302_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73302_BASE_FRESH50_v1
```

### FH20R1_S3_B04 / CORE / PH D121 vs D122

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73303_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73303_BASE_FRESH50_v1
```

### FH20R1_S3_B05 / CORE / PH D121 vs D122

```text
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73304_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73304_BASE_FRESH50_v1
```

### FH20R1_S3_B06 / CORE / PH W104 vs W112 at D122

```text
FH20R1_S3_S_PH_W112_D122_WV3_TF3_TS71001_SS73305_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73305_BASE_FRESH50_v1
```

Width/depth combination is a new hypothesis, not an observed success.

### FH20R1_S3_B07 / CORE / PH W104 vs W112 at D122

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73306_BASE_FRESH50_v1
FH20R1_S3_S_PH_W112_D122_WV3_TF3_TS71001_SS73306_BASE_FRESH50_v1
```

Width/depth combination is a new hypothesis, not an observed success.

### FH20R1_S3_B08 / CORE / PH W104 vs W112 at D122

```text
FH20R1_S3_S_PH_W112_D122_WV3_TF3_TS71001_SS73307_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73307_BASE_FRESH50_v1
```

Width/depth combination is a new hypothesis, not an observed success.

### FH20R1_S3_B09 / CORE / PH W104 vs W112 at D122

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73308_BASE_FRESH50_v1
FH20R1_S3_S_PH_W112_D122_WV3_TF3_TS71001_SS73308_BASE_FRESH50_v1
```

Width/depth combination is a new hypothesis, not an observed success.

### FH20R1_S3_B10 / CORE / PH W104 vs W112 at D122

```text
FH20R1_S3_S_PH_W112_D122_WV3_TF3_TS71001_SS73309_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73309_BASE_FRESH50_v1
```

Width/depth combination is a new hypothesis, not an observed success.

### FH20R1_S3_B11 / RESERVE / PH D121 vs D122 reserve

```text
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73310_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73310_BASE_FRESH50_v1
```

### FH20R1_S3_B12 / RESERVE / PH D121 vs D122 reserve

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73311_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73311_BASE_FRESH50_v1
```

### FH20R1_S3_B13 / RESERVE / PH D121 vs D122 reserve

```text
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73312_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73312_BASE_FRESH50_v1
```

### FH20R1_S3_B14 / RESERVE / PH D121 vs D122 reserve

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73313_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73313_BASE_FRESH50_v1
```

## s4: 읽기용 queue plan

Import: `FH20R1_S4_IMPORT` → diagnostic: `FH20R1_S4_DIAG`

### FH20R1_S4_B01 / CORE / Bridge PL D122 seed72001

```text
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS72001_BASE_FRESH50_v1
```

Compare against existing FH12 PL D121/PLH D121/D122; do not retrain those assets.

### FH20R1_S4_B02 / CORE / PL/PLH x D121/D122

```text
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73401_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73401_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73401_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73401_BASE_FRESH50_v1
```

### FH20R1_S4_B03 / CORE / PL/PLH x D121/D122

```text
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73402_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73402_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73402_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73402_BASE_FRESH50_v1
```

### FH20R1_S4_B04 / CORE / PL/PLH x D121/D122

```text
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73403_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73403_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73403_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73403_BASE_FRESH50_v1
```

### FH20R1_S4_B05 / CORE / PL/PLH x D121/D122

```text
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73404_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73404_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73404_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73404_BASE_FRESH50_v1
```

### FH20R1_S4_B06 / CORE / Bridge W112D122 PLH seed72001

```text
FH20R1_S4_S_PLH_W112_D122_WV3_TF4_TS71001_SS72001_BASE_FRESH50_v1
```

Paired with existing W112D121 and W104D122 of the same Teacher/seed.

### FH20R1_S4_B07 / CORE / PLH W112 D121 vs D122

```text
FH20R1_S4_S_PLH_W112_D121_WV3_TF4_TS71001_SS73405_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W112_D122_WV3_TF4_TS71001_SS73405_BASE_FRESH50_v1
```

### FH20R1_S4_B08 / RESERVE / PL/PLH x D121/D122 reserve

```text
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73406_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73406_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73406_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73406_BASE_FRESH50_v1
```

### FH20R1_S4_B09 / RESERVE / PL/PLH x D121/D122 reserve

```text
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73407_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73407_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73407_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73407_BASE_FRESH50_v1
```

## s5: 읽기용 queue plan

Import: `FH20R1_S5_IMPORT` → diagnostic: `FH20R1_S5_DIAG`

### FH20R1_S5_B01 / CORE / Bridge PL D121/D122 seed72002

```text
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1
```

Reuse existing PLH D121/D122 at the same seed; not a new Teacher-seed repeat.

### FH20R1_S5_B02 / CORE / PL/PLH x D121/D122

```text
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73501_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73501_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73501_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73501_BASE_FRESH50_v1
```

### FH20R1_S5_B03 / CORE / PL/PLH x D121/D122

```text
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73502_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73502_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73502_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73502_BASE_FRESH50_v1
```

### FH20R1_S5_B04 / CORE / PL/PLH x D121/D122

```text
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73503_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73503_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73503_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73503_BASE_FRESH50_v1
```

### FH20R1_S5_B05 / CORE / PL/PLH x D121/D122

```text
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73504_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73504_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73504_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73504_BASE_FRESH50_v1
```

### FH20R1_S5_B06 / CORE / PLH W112 D121/D122 bridge

```text
FH20R1_S5_S_PLH_W112_D121_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W112_D122_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1
```

Together with existing W104 runs forms W104/W112 x D121/D122 at fixed Teacher/seed.

### FH20R1_S5_B07 / RESERVE / PL/PLH x D121/D122 reserve

```text
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73505_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73505_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73505_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73505_BASE_FRESH50_v1
```

### FH20R1_S5_B08 / RESERVE / PL/PLH x D121/D122 reserve

```text
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73506_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73506_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73506_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73506_BASE_FRESH50_v1
```

