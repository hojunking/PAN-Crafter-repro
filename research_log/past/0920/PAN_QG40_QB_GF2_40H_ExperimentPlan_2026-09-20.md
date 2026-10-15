# PANDA QG40 — QB·GF2 40시간 실험 Case 및 실행 인계 명세

**작성일:** 2026-09-20  
**Campaign:** `PANDA_QG40_20260920_v1`  
**기준 Method:** WV3 종결 조건의 `P0·W112D123 Teacher → PLH·W104D122 Student·BASE`  
**범위:** s1–s5, QB 3대 / GF2 2대. WV3 신규 탐색은 재개하지 않는다.  
**작성 상태:** 실험 정의·JSON 계약·CSV·서버별 순서·검사 목록 작성. 원격 trainer/YAML/registry/queue/Sheet 배포 및 실제 학습은 수행하지 않았다.

> 이번 **40시간은 다섯 서버가 동시에 작업하는 하나의 경과시간 창**으로 편성한다. 합산 40 GPU-hour나 서버를 차례로 40시간씩 사용하는 뜻이 아니다. 한 서버당 GPU 슬롯 1개를 가정하면 명목 상한은 200 server-slot-hour이며, 실제 GPU 수·점유시간은 따로 기록한다.  
> **준비·포팅·Teacher·calibration·Student·평가·보존을 모두 포함**한다. 종료한 WV3의 최소 20시간 조건을 승계하지 않는다. 40시간은 상한·계획 창이지 불필요한 실행으로 채워야 하는 최소시간이 아니다.

## 0. 실행자가 먼저 확인할 결정

| 구분 | 이번 결정 |
|---|---|
| 기본 접근 | 센서별 새 P0 W112D123 Teacher, exact50K reference, PLH W104D122 Student |
| Method | Teacher consistency, adaptive hard/soft KD, q-weighted GT edge, q-weighted trainable Student Aligner adjustment 모두 유지 |
| 기본계수 | Student α=1, β=0.1, λE=0.002, U peak LR=1e−4, A peak LR=3e−6 |
| 데이터 대응 | C=4, QB maxDN2047 / GF2 maxDN1023, 실제 Q4, 센서별 MTF·band order·source identity |
| 측정값 | τR·qref·q cache는 센서×Teacher마다 새로 측정; 측정 전 null. WV3 숫자 복사 금지 |
| 1차 핵심 | 이전 준비안의 **Teacher 3개 + Student 8개 = 11개 50K**를 가장 먼저 확보 |
| 확장 | 기본 seed 반복 + 같은 Teacher/seed에서 제한적인 한 축 대조. 입력·W/D grid를 다시 열지 않음 |
| 조건부 Student 시험 | s2(QB), s5(GF2)에서 각각 **A24R/B20/E10 중 최대 1개**. 근거 없으면 BASE 반복 |
| 조건부 Teacher 시험 | s1(QB TB), s3(GF2 TA)에서만 **C3: consistency 계수 1e−4→3e−4**. 별도 reference; 기본 reference 교체 금지 |
| 종료 | 공통 t0+40h. 36h 이후 새 학습 입장 금지, 마지막 4h는 평가·보존. 수량 목표보다 deadline 우선 |
| 공통대기 금지 | 다른 센서 성능·전체 seed 결과·공통 recipe lock을 기다리지 않음. 자기 Teacher/reference와 수치 무결성은 실제 필수 의존성 |

**중요한 수량 구분:** 89개는 등록용으로 정의한 *유한 학습 case 전체*다. 상호 배타적인 대조 후보를 포함하므로 전부 실행하지 않는다. 핵심 11개가 최우선이며, 주 순서의 실제 활성 case는 분기에 따라 **31–39개**다. 조건부 Teacher 묶음은 최대 6개 학습, 시간 예비는 28개다. 전 분기를 풀어 실제로 선택될 수 있는 상한도 73개이며, 이것은 **40시간 내 완료 보장이 아니다.** 실제 실행 수는 서버별 완료시간과 평가부채를 반영해 더 줄어들 수 있다.

## 1. 기준 자료와 이번에 새로 설계한 부분

### 1.1 자료에서 유지한 사항

이전 `PAN_QB_GF2_DistributionAware_Preparation_2026-09-20.md`의 QG00–QG19, 서버별 TA/TB·seed 배정, 센서별 4밴드·DN·Q4, fresh Teacher·exact50K reference, train-only calibration, 주파수 동기 정합을 계승한다. WV3 종결문서의 좋은 조건을 이식하는 것이며, WV3의 선택 seed73101·step38380을 다른 센서의 최적값으로 가정하지 않는다. [S1,S2]

연구자료 pp.8–10은 native reconstruction과 A-only synthetic consistency를, pp.12–15는 frozen Teacher reference, adaptive reconstruction supervision, q-edge와 q-hard A adjustment를 구분한다. 이 역할을 바꾸지 않는다. 도식의 축약된 합/평균 표기보다 아래의 기존 수치 계약을 적용한다. [S3]

### 1.2 이번에 추가한 설계 — 아직 관측된 성능이 아니다

**공통 40h 상한, 36h 신규입장 cutoff, 4h 보존 여유, 확장 seed, A24R/B20/E10/C3의 정확한 값, 진단 분기 기준, 예산 예약값**은 이번에 제안한 운영·대조 설계다. QB/GF2에서 이미 효과를 확인한 최적값으로 쓰지 않는다.

이번 작성에서는 원격 GPU·H5·새 Teacher 결과·새 τ/q를 직접 측정하지 않았다. 이전 준비문서의 코드 포팅 장애도 그 문서의 조회 시점 근거다. 서버에 이미 수정본이 있다면 QG04/05의 실제 receipt로 확인하고, 검사 없이 구버전 장애가 여전히 있다고 단정하지 않는다.

## 2. 시간 계약: 하나의 40시간 창

### 2.1 시작·종료와 실제 자원

운영자가 최초 QG40 준비 작업을 시작한 UTC 시각을 `t0_utc`로 기록하고 모든 서버에 배포한다. `deadline_utc = t0_utc + 40h`다. **현재 문서의 생성시각이 실제 기동시각은 아니다.** 이 공유 timestamp는 시간 기준이지 성능 결정 lock이 아니다.

포팅, 데이터 생성, 기존 WV3 안전정리, reference 전송에 시간이 걸려도 시계를 다시 시작하지 않는다. 뒤늦게 참여한 서버도 같은 deadline을 사용한다. 실제 GPU·VRAM·디스크 부족은 local inventory에 기록하고 해당 block만 보류한다. 기존 실행을 강제 kill하거나 미시작 WV3 pair를 추가로 시작하지 않는다.

| 공통 경과시간 | 우선 작업 | 시간 지연 시 원칙 |
|---|---|---|
| 0–6h | WV3 안전정리, 4밴드 포팅, QB phase·GF2 DN, 데이터·LP·metric·gradient 검사 | 수치 검사를 생략하지 않음. 준비가 늦으면 후반 대조 수를 줄임 |
| 준비 통과 후–약14h | s1/s3/s4의 Teacher50K·τ/q·reference 배포; s2/s5의 local 검증 | 전체 서버 barrier 없음. 준비된 경로만 즉시 진행 |
| reference 준비 후–약26h | 11개 기본군 우선, baseline 진단·로컬 screen·기본 반복 | 낮은 HQNR·높은 q 자체는 BASE 실행 차단 사유가 아님 |
| 약26–36h | 새 seed 확인, 이미 승인한 한 축 대조, 유한 BASE 예비 | 실패한 후보를 다른 후보로 끝없이 교체하지 않음 |
| 36–40h | 미완 공식평가, 선택점 확인, Sheet readback, full-state·backup·종결표 | 신규 학습 없음. 추가 업데이트로 deadline을 연장하지 않음 |

이 구간은 순차 barrier가 아니라 우선순위다. 예를 들어 s4의 reference가 먼저 준비되면 s2는 s3의 GF2 Teacher를 기다리지 않는다.

### 2.2 Atomic block 입장

새 block은 다음을 **모두** 만족할 때만 시작한다.

1. 데이터·method·reference의 필요한 P0 검사 통과.
2. 현재 시각이 t0+36h 이전.
3. `남은시간 > block 전체의 보수적 잔여시간 + 이미 발생한 공식평가부채 + 4h 종결여유`.
4. 새 C3 Teacher 묶음은 추가로 t0+22h 이전에 입장하고, **Teacher50K + calibration + Student 2개 + 필요한 평가**의 전체 시간을 예약.

Pair는 BASE와 한 대조를 모두 포함해 예약한다. C3 패키지는 Teacher만 돌리고 Student 비교를 못 하는 상황을 피하기 위해 전체 체인을 예약한다. 이미 완료된 동일조건 BASE는 checksum·reference·소스·초기화·평가 동등성 확인 후 재사용하며, 시간을 채우려고 재학습하지 않는다.

예상시간은 초기에는 §2.3의 예약 가정을 쓰고, 완료 run이 생기면 **같은 서버/센서/역할/구조의 실측값**으로 갱신한다. 유사조건 p90의 1.25배와 잔여 평가비용을 보수적으로 사용한다. Teacher와 Student 시간, train과 calibration/eval 시간은 섞지 않는다. 손상된 evaluation 때문에 비용이 줄어든 실행을 속도 개선으로 취급하지 않는다.

예상보다 길어져 deadline에 걸리면 optimizer update의 안전한 경계에서 latest full-state를 저장하고 중단한다. 50K 미달은 `INCOMPLETE_AT_DEADLINE`이다. 이 가중치를 50K 완료 결과와 합산하지 않는다. 파일 flush에 필요한 최소시간을 별도 기록하되 새 학습 허가로 쓰지 않는다.

### 2.3 최초 시간 예약 예시 — 새 센서 실측값이 아님

단위 h. Student 비용에는 50K와 공식 50후보 평가를 포함한다. 아래는 case 수를 무조건 약속하지 않기 위한 **초기 예약 가정**이다.

| 서버 | 준비 또는 공유 reference 예상 준비시점 | Teacher+평가 | calibration | Student 1개+평가 | 추가 진단 | 종결 |
|---|---:|---:|---:|---:|---:|---:|
| s1 | 준비5.0 | 3.5 | 2.5 | 2.2 | 2.0 | 4.0 |
| s2 | reference 약t0+10h | 없음 | import 포함 | 2.5 | 1.5 | 4.0 |
| s3 | 준비4.0 | 3.0 | 2.5 | 1.8 | 2.0 | 4.0 |
| s4 | 준비4.0 | 3.0 | 2.0 | 2.0 | 2.0 | 4.0 |
| s5 | reference 약t0+10h | 없음 | import 포함 | 2.5 | 1.5 | 4.0 |

C3·예비 제외, Student 주 순서 전부를 실행할 때의 예시 총시간은 s1 약30.2h, s3 약26.3h, s4 약27.0h다. s2/s5는 대조를 screen·confirm 두 seed씩 모두 수행하면 약38.0h, BASE만 반복하면 약28.0h다. **실제 준비/teacher/q/평가가 더 길면 이 수치는 성립하지 않는다.** reference 예상 준비시각도 보장이 아니다.

남는 시간은 아래의 유한 BASE seed와 기존 자산 진단에 사용한다. C3에 시간을 쓰면 뒤의 BASE 반복·예비 일부가 빠질 수 있다. 모든 정의 수를 완료한 후 sleep/중복평가/무한 seed 생성으로 40시간을 채우지 않는다.

## 3. 모든 서버 배정·Reference 계통

| 서버 | 센서 / 주 reference | 기본 흐름 | 제한적 확장 |
|---|---|---|---|
| **s1** | QB / **TB**, T81002 | 포팅·QB 감사 → 새 Teacher → calibration → PLH/D122 BASE 반복 | 증거가 있으면 TB_C3 Teacher 묶음 1개 |
| **s2** | QB / **TA**, T81001, s4에서 수령 | QB metric·parity 점검 → TA import → S82003 BASE | 같은 TA에서 Student 한 축 screen·새 seed 확인 |
| **s3** | GF2 / **TA**, T91001 | 새 Teacher → calibration → PLH/D122 BASE 반복 | 증거가 있으면 TA_C3 Teacher 묶음 1개 |
| **s4** | QB / **TA**, T81001 | QB 주 Teacher → calibration 배포 → PLH/D122 BASE 반복 | 계수 변경 없이 seed 재현성 강화 |
| **s5** | GF2 / **TA**, T91001, s3에서 수령 | GF2 DN·MTF·parity 점검 → TA import → S92003 BASE | 같은 TA에서 Student 한 축 screen·새 seed 확인 |

TA와 TB는 별도 Teacher 계통이다. 같은 센서의 같은 Student seed를 쓰더라도 TA/TB 결과를 하나의 Student-seed 반복 평균으로 섞지 않는다. QB에서 s1 TB와 s4 TA의 차이는 Teacher뿐 아니라 서버 환경도 포함할 수 있으므로 Teacher seed의 순수 인과효과라고 단정하지 않는다.

s2/s5가 자기 TA를 기다리는 동안에는 QG01–07/15–17, 기존 checkpoint의 센서 metric 검증, 저장공간·결과 schema 검사를 한다. 필요한 Teacher를 임의로 대체하거나 WV3 A를 부분 로드하지 않는다. reference가 준비되면 다른 서버 결과를 기다리지 않고 import·Student로 이어간다. 준비 작업이 끝났는데 reference가 아직 없으면 `WAIT_REFERENCE`를 솔직히 기록한다. 존재하지 않는 독립 작업으로 GPU를 강제 채우지 않는다.

### 3.1 기존 11개 ID 보존

직전 준비안의 11개 `QGBASE_...` ID는 그대로 유지하고 이번 campaign manifest에서 참조한다. 새 확장만 `QG40_...` ID를 사용한다. 이것은 이전에 실행되었다는 뜻이 아니다. 현장에 이미 동일한 run이 있다면 먼저 inventory·signature를 대조하고 중복 실행을 막는다. 이전 run의 정규화나 method가 다르면 이름만 같아도 재사용하지 않는다.

| 서버 | Teacher seed / reference | 1차 Student seed | 우선 50K 학습 수 |
|---|---|---|---:|
| s1 | 81002 / QB_TB | 82001, 82002 | 3 |
| s2 | s4의 81001 / QB_TA | 82003 | 1 |
| s3 | 91001 / GF2_TA | 92001, 92002 | 3 |
| s4 | 81001 / QB_TA | 82001, 82002 | 3 |
| s5 | s3의 91001 / GF2_TA | 92003 | 1 |
| 합계 | 새 Teacher3 | Student8 | **11** |

1차11개는 최우선 cohort이지, 수치 오류나40h deadline을 무시하고 강제로 끝내라는 뜻은 아니다.

## 4. 데이터·4밴드 구현: 학습 전에 통과해야 하는 계약

| 항목 | QB | GF2 |
|---|---|---|
| bands | 4, 실제 source band order 명시 | 4, 실제 source band order 명시 |
| Teacher / Student 입력 | P0=5ch / PLH=7ch | 동일 |
| 출력 / Aligner MS stem | 4 / 4 | 동일 |
| DN 최대·정규화 | 2047, 2DN/2047−1 | 1023, 2DN/1023−1 |
| 역정규화 | (y+1)×1023.5 | (y+1)×511.5 |
| train / RR / FR PAN | 64 / 256 / 512 | 동일 |
| LRMS ratio | 4 | 4 |
| RR metric | Q4 (4밴드 Q2n), 기존 나머지 지표 | 동일 |
| FR metric filter | QB preset, 원 band order | GF2 preset, 원 band order |

논문의 band·bit·nominal N은 데이터 계약의 참고값이다. 실제 H5의 단위·dtype·source count·no-data·saturation·split correspondence를 읽어 검증한다. observed max로 sample-wise min–max 정규화하거나 hist-matching을 추가하지 않는다. [S4,S5]

**QB:** 검증된 `train_qb_msfix.h5`, `valid_qb_msfix.h5` 후보를 사용한다. 파일명만으로 PASS하지 않는다. raw/msfix source hash와 GT/PAN 불변, MTF/decimation/lms 레시피를 대조한다. 과거 기록의 LR1px/HR4px 표현과 phase index 차이는 실제 좌표계로 확인하고 미검증 숫자를 synthetic 반경 확대의 근거로 쓰지 않는다. test GT로 test 입력을 재생성하거나 모델 성능에 맞게 test를 이동시키지 않는다. 복구 사실은 논문·metadata에 명시한다.

**GF2:** 1023 정규화와 511.5 역변환, PSNR·SSIM의 data range, sensor MTF를 묶어 검사한다. DN2047이나 q8 key가 숨은 기본값으로 fallback하면 실패다.

**공통:** model input/output/head/A stem, loss band guard, calibration, RGB 시각화, metric 및 cost, uploader까지 명시적 `C`를 사용한다. C=8 회귀는 기존 WV3 예측·loss·metric의 동등성 검사이지 새 WV3 학습이 아니다. C=4에서 검사를 없애지 말고 실제 sensor contract로 바꾼다.

Split 경로는 SensorSpec에 실제 resolve된 경로·SHA로 고정한다. 기본 후보는 `data/PanCollection/{QB,GF2}` 아래 train/valid, `reduced_examples_h5`, `full_examples_mat20`이다. 원본 H5 test와 mat20이 동일하다고 가정하지 않는다. 없는 경로로 추론하거나 WV3 파일로 fallback하지 않는다.

## 5. 주파수 입력·Teacher·Student 수치 Method

### 5.1 Forward

\[
M=U_4(MS),\quad L=U_4(LPAN),\quad c=A(P_{m4},M_{m4}),
\]
\[
\widetilde P=W(P,c),\quad\widetilde L=W(L,c),\quad
\widetilde H=\widetilde P-\widetilde L.
\]

Teacher P0는 `Z_T=M+F_T([P_tilde,M])`, Student PLH는 `Z_S=M+F_S([P_tilde,L_tilde,H_tilde,M])`다. 각각 자기 A가 만든 c를 쓴다. **Student inference에 Teacher c를 넣지 않는다.**

A는 native PAN1/MS4만 입력받고 Aligner-only margin4를 유지한다. c는 `(dy,dx)` HR PAN pixel 단위, warp는 FP32/bicubic/border/align_corners=False다. P와 upsampled L을 한 grid로 함께 warp하고 signed H를 차이로 만든다. c를 ±2로 clamp하거나 LP에만 detach하지 않는다. MS/GT/출력 좌표계는 고정하고 MS base는 정확히 한 번 더한다.

LP는 각 split의 실제 native PAN에서 **Gaussian σ1.98/k41/replicate/[2::4,2::4]**, float64 생성·float32 cache로 준비한다. 증강 전에 생성해 data와 같은 view를 취한다. 이 알고리즘용 LP는 sensor의 물리적 MTF라는 주장이 아니다. 평가 MTF와 분리한다. 첫40h에는 LP filter/phase를 성능에 맞춰 바꾸지 않는다.

Train에서는 PAN·MS·LP를 정규화한 뒤 H를 계산한다. H에 abs·clip·추가 영상 정규화를 적용하지 않는다. 모델 최종 출력의 공식 metric 입력 clamp만 기존 계약을 유지하며, 학습/τ calibration 출력을 metric처럼 미리 clip하지 않는다.

### 5.2 Teacher: fresh U/A, exact50K

Teacher는 W112D123, MS-only, LN, attention OFF, mode modulation OFF. A 마지막 Linear만0 초기화하고 나머지는 fresh다. **N2 donor·WV3 checkpoint를 사용하지 않는다.**

\[
c_0=A_T(P,M),\qquad c_\epsilon=A_T(W(P,\epsilon),M),
\]
\[
L_{\rm con}=\frac{1}{2B}\sum_i
\|c_{\epsilon,i}+\epsilon_i-\operatorname{sg}(c_{0,i})\|_1,
\]
\[
L_T(t)=\operatorname{mean}|Z_T-Y|
+\mathbf1[t\bmod 2=1]10^{-4}L_{\rm con}.
\]

`t`는0-based optimizer update. ε는 HR PAN pixel 반경2 원판에서 면적균등 샘플, 별도 CPU RNG. Synthetic PAN은 A-only branch에만 들어가며 U로 가지 않는다. stop-gradient는 consistency의 native 기준에만 적용한다. Native reconstruction의 c는 live이므로 A/U 모두 갱신된다.

기본 Teacher peak LR는 U1e−4/A1e−5. 50K, batch48, AdamW β(.9,.999), eps1e−8, wd.01, warmup100+cosine, FP32/no AMP를 유지한다. Teacher reference는 성능을 보고 고른 best가 아니라 **exact50K의 A/U 전체**다. 낮은 Teacher HQNR은 단독으로 Student 차단 사유가 아니다.

C3 조건부 시험에서만 `10−4→3×10−4`로 바뀐다. 수식·gradient·빈도·반경·초기화·seed·U/A LR는 그대로다. BASE reference와 혼합하지 않는다.

### 5.3 Calibration: error와 consistency를 분리

같은 센서의 Teacher들은 train에서 seed1234로 고정 선택한 **동일3072 base ID**를 공유하되, 각 Teacher 출력으로 τ/q를 새로 측정한다. Student seed마다 subset을 다시 뽑지 않는다.

\[
e_T(i,p)=\operatorname{mean}_b|Z_{T,i,b}(p)-Y_{i,b}(p)|,
\quad\tau_R=\max(\operatorname{median}_{i\in I,p}e_T(i,p),10^{-6}).
\]

Frozen/eval/no_grad/FP32, 무증강 train3072, 최종HRMS·full-pixel 오차다. patch별 median의 평균이나 RR ERGAS로 대체하지 않는다. 3072 미만 자료는 프로토콜 불일치이므로 조용히 표본수를 줄이지 않는다.

q는 **전체 train×실제 fixed-HV/ROT4 view**에서 계산한다. rotation 전 고정 H/V 두 flip이라는 현재 augmentation을 임의 random flip으로 바꾸지 않는다. cache key는 `(source_sample_id,view_id)`다.

\[
q_{i,r}=\frac1{2K}\sum_{j=1}^{K}
\|c_{i,r,j}+\epsilon_j-c_{i,r,0}\|_1,\qquad K=16,
\]
\[
q_{\rm ref}=\operatorname{median}_{i\in I,r\in\{0,1,2,3\}}q_{i,r},\quad
s_{i,r}=\operatorname{sg}\frac{q_{\rm ref}}{q_{\rm ref}+q_{i,r}}.
\]

AXIS16은 [.25,.5,1,2]의4방향이다. 전체 train median은 별도진단이며 qref로 대체하지 않는다. 코드의 분자에2를 넣지 않는다. cache 일부만 만들어 전체완료로 표시하지 않는다.

**τR/qref/cache/Teacher SHA가 측정·검증되지 않은 Student는 실행할 수 없다.** 이것은 과거 recipe-lock처럼 모든 서버의 성능을 기다리는 조건이 아니라 자신의 입력 reference 무결성 조건이다. finite-positive qref가 높아도 BASE는 진행한다. qref=0이면 현재 수식의 프로토콜 문제를 보고하며 임의 epsilon·uniform으로 바꾸지 않는다.

### 5.4 Student: 기존 KD·edge·A adjustment 유지

\[
e_S=\operatorname{mean}_b|Z_S-Y|,\quad
 d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\quad
 a_T=\operatorname{sg}\frac{[e_S-e_T]_+}{e_S+10^{-6}},
\]
\[
\ell_{H,i}=\operatorname{mean}_p[(1+\alpha d_T)e_S],\quad
\ell_{K,i}=\operatorname{mean}_p[\beta(1-d_T)a_T\operatorname{mean}_b|Z_S-Z_T|],
\]
\[
L_U=\operatorname{mean}_i[\ell_{H,i}+\ell_{K,i}+\lambda_Es_i\ell_{E,i}],
\qquad L_A=\operatorname{mean}_i[s_i\ell_{H,i}].
\]

GT edge는 최종HRMS/GT의 signed Scharr 차이, 방향 각각0.5·경계1px 제외의 기존 정의다. HPAN을 edge 정답으로 쓰지 않는다. C4에서도 band mean이며 C8→C4라고 계수를2배 보정하지 않는다.

U의 gradient는 LU로, A의 gradient는 LA로 따로 계산한 뒤 step한다. `backward(LU+LA)`나 LU의 전체model backward는 이 Method가 아니다. Teacher A/U는 frozen, Student A는독립 clone이자trainable, Student U는fresh다. Student에 offset/direct shift KD를 추가하지 않는다.

기본 초기화는 이름·shape 기반 fresh P/MS/body와 추가 L/H kernel=0의 정책을 C4로 일반화한다. **같은 Teacher·Student seed의 BASE/변경안은 profile명 때문에 초기 tensor가 달라지면 안 된다.** C8 회귀에서는 기존 키/seed 의미를 보존한다. 실제 data stream/augmentation/corruption RNG도 role별로 격리하고 기록한다. 공통 tensor가 같다는 사실이 서로다른 Teacher·구조의 전체함수까지 같다는 뜻은 아니다.

## 6. 분포·정합·gradient 진단: 수치 오류와 성능가설을 구분

QG00–QG19 원 항목을 보존하고, QG20–23으로 시간·등록·분기·종결을 추가했다. 전체24개 체크리스트의 상태는 처음에는 TO_VERIFY/TO_MEASURE다.

| 구분 | 필수 확인·산출물 | 오류/미확인 시 처리 |
|---|---|---|
| P0 데이터 | QB phase/MS–GT, GF2 DN, band order, GT/PAN/source 불변, LP correspondence | 영향받은 데이터 학습만 차단 |
| P0 Method | C4 전체경로·C8 회귀, consistency/KD/edge/q/A gradient, Teacher frozen | 해당 release 신규입장 차단. loss 제거 우회 금지 |
| P0 Reference | exact50K A/U, τ/q/cache/source/indices/증강hash, import 재현 | 해당 reference 소비만 차단 |
| P0 평가 | C4/Q4, sensor MTF, DN inverse, RR20/FR20·지표정의, 같은 checkpoint | 공식값 보류·평가수정; 학습 재실행은 영향분석 후 |
| P1 분포 | eT/eS의tail·band bias·texture·HP/edge energy, c/q/s 분포 | warning. 임의 표본삭제·tau조정 금지 |
| P1 원인 | 크기별 c/q, A/U 교차, gradient norm/cosine, late Ds | 미확인은 BASE 지속. 가설대조를 강제로 열지 않음 |

진단 sample ID는 train에서 고정128개를 별도로 뽑고 calibration3072와 관계를 기록한다. 추출에 성능·test 정보를 쓰지 않는다. gradient 측정은 이 중고정24개, P0 실제오류 검사는 전체source metadata/shape 및필요값scan을 한다. 표본개수는 이번 진단용 설계이며 τ/calibration 정의를 대체하지 않는다.

기록 시점은 Teacher/Student completed update10,000·24,240·50,000. 모든 profile에서 같은 budget/표본을 사용한다. 초기 zero head 때문에 첫 update A gradient가0일 수 있으므로 초기시점 하나만으로 경로단절을 판정하지 않는다. Controlled active synthetic data와 실제학습의 gate-zero를 구분한다.

**해석 불변:** `c 크기 ≠ 실제 native displacement GT`, `q ≠ native 정합 절대오차`, `e ≠ 순수 misalignment 오차`다. 상수 A의 AXIS16 q는0.46875이고, c에공통bias를더하면q에서소거된다. 자기qref로정규화된s≈0.5는 정합정확도인증이아니다. 저texture·분광차이·반복패턴이proxy shift를오염시킬수있으므로 proxy를GT처럼쓰지않는다.

Train64·crop128·RR256·FR512 크기진단에서는 U-Net을full frame으로 유지한다. crop-consensus나 cross A/U는 진단만이며정상Target에넣지않는다. FR에GT를가정하지않고 data/evaluation reference를warp하지않는다.

## 7. Student의 제한적 조정: 센서당 한 가지, 같은 seed 대응

**기본 구조·입력·Teacher·τ/q·LP·50K는 고정**한다. s2는QB TA, s5는GF2 TA에서만 아래 중한개를선택한다. 이 표의값은이번대조제안이며효과가검증된값이아니다.

| Profile | α | β | λE | Student A LR 정책 | BASE에서 달라지는 한 축 |
|---|---:|---:|---:|---|---|
| BASE | 1 | .1 | .002 | peak3e−6 × 기존 cosine | 없음 |
| B20 | 1 | **.2** | .002 | BASE | soft KD 계수 |
| E10 | 1 | .1 | **.001** | BASE | GT edge 계수; 0 아님 |
| A24R | 1 | .1 | .002 | **24,240회 완료 뒤 BASE A LR×1/3** | A LR의후반정책; freeze 아님 |

A24R의 0-based update t=0…24239는BASE와같고, t≥24240부터multiplier1/3이다. U LR·optimizer/RNG/scheduler state·weight decay는 유지한다. fresh run에서 사전정의하며 초기weights-only에서재시작하지않는다. cosine끝의LR0은기존schedule의끝값이지조기freeze가아니다.

### 7.1 진단 분기 규칙

첫 local BASE 50K와 P0 검사가 끝난 뒤 `student_trial[s2 or s5]`를 한 번 정한다. 아래 숫자는 **운영상 사전 정의한 탐색 heuristic**이며, 물리적 정합 정확도나 통계적 유의성의 절대 기준이 아니다.

**A24R 후보:** 같은 run의 정상 A/U 24,240·50,000과 `A24240+U50000`을 비교했을 때, 후반 U의 RR를 보존하면서 과거 A로 H와 Ds가 개선되는 근거가 있어야 한다. 구체적으로 ΔH≥.001, ΔDs≤−.001, ΔE/E≤+.005를 동시에 만족하고 native c의 변화가 확인되어야 한다. 대각 checkpoint 재현, 자산 SHA, LP 동기 검사를 통과해야 한다. 비대각 조합 자체는 정상 모델 성능이 아니며, 원인의 확정도 아니다.

**B20 후보:** 해당 BASE에 H 적격 후보가 있으나 E 목표는 미달이고, 24,240과 50K의 train 진단에서 advantage>0 비율≥.10, soft-gradient norm / hard-gradient norm의 중앙값이 (0,.20]여야 한다. β 상향의 개선은 보장하지 않는다. 정상 gate-zero와 구현 누락을 먼저 구분한다.

**E10 후보:** late Ds뿐 아니라 Dλ도 24,240→50K에 .001 이상 악화하고, signed edge와 hard의 U-gradient cosine이 두 진단 시점에서 <0, weighted-edge/hard gradient norm이 ≥.10이어야 한다. LP phase·band order·metric은 정상이어야 한다. 이것도 edge가 원인이라는 증명이 아니라, 감소 대조를 여는 기준이다.

둘 이상 충족하면 **A24R → B20 → E10** 순으로 단 하나를 선택한다. 모두 미충족이거나 진단이 미완료·예산 부족이면 **BASE**다. 첫 성능이 낮다는 것만으로 아무 후보나 넣지 않는다. 선택 시각·근거 파일·원 측정값을 local `branch_receipt.json`에 남기고 다른 서버 결과를 기다리지 않는다.

### 7.2 Screen과 확인 seed

| 서버 | 최초 BASE | Screen: BASE vs 선택 후보 | 확인: BASE vs 승격 후보 |
|---|---:|---|---|
| s2 QB TA | 82003 | **82006, 82007** | **82012, 82013** |
| s5 GF2 TA | 92003 | **92006, 92007** | **92012, 92013** |

첫 screen seed는 BASE→대조, 두 번째는 대조→BASE 순으로 실행한다. 확인 seed도 같은 교차 순서를 사용한다. 두 run은 각각 fresh이며, 같은 seed의 공통 초기화·data stream을 유지한다. 선택이 BASE이면 해당 seed마다 BASE를 **한 번만** 실행한다. 별칭 BASE 대조를 만들어 중복 실행하지 않는다.

Screen 2개 seed 완료 후 순위는 `(H 적격 seed 수 내림, 적격 seed의 Target E 중앙값 오름, RAW_MAX H 중앙값 내림, E50 중앙값 오름)`으로 정한다. Target 미적격 E를 0이나 자의적인 큰 숫자로 채우지 않는다. 적격 수가 같아도 서로 다른 seed가 적격이면 그 사실을 별도로 표시한다.

두 profile 모두 H 적격 0이면, 후보의 RAW H 중앙값이 BASE보다 ≥.001 높고 E50 중앙값의 악화가 ≤.5%일 때만 승격한다. 동률·불충분·무효 평가이면 BASE를 유지한다. 승격하면 사전에 배정한 새 2개 seed에서 BASE와 같은 후보를 비교한다. 탈락하면 확인 seed의 BASE만 실행하고 다른 후보로 교체하지 않는다.

이는 test-aware 개발 screen이다. 새 seed 확인이 같은 test set을 독립 held-out set으로 만드는 것은 아니다. 모든 seed·no_eligible·탈락 결과를 보존한다. 최종 보고에서 고정 BASE 이식군과 sensor-tuned 군을 분리한다.

## 8. Teacher C3 묶음: 정합반응이 약한 경우에만

| 서버 | BASE parent | 조건부 Teacher | 조건부 Student 2개 |
|---|---|---|---|
| s1 | QB_TB, TS81002 | **QB_TB_C3**, 같은TS/초기화/U/A LR, λcon만3e−4 | BASE PLH/D122 **SS82001,82002**, 새TB_C3 reference |
| s3 | GF2_TA, TS91001 | **GF2_TA_C3**, 같은TS/초기화/U/A LR, λcon만3e−4 | BASE PLH/D122 **SS92001,92002**, 새TA_C3 reference |

이변경은**새 Teacher를그대로승격하는절차가아니라Teacher학습계수의민감도대조**다. parent의2개Student 결과를보존하고같은 seed로C3의전체학습경로를비교한다. τ/q/cache/Student A초기값이달라질수있으므로그차이는전체 reference 변경 효과다. C3Teacher의q 개선만으로최종정합이좋아졌다고단정하지않는다.

### 8.1 조건

parent BASE Teacher 50K와그reference의첫 Student 2개를완료한뒤,local 검증을통과하고예산이있을때만입장한다. 정합반응의진단기준은다음이다.

- qref/0.46875≥.90.
- 고정 train표본에서 response gain `g=−Σ εᵀ(cε−c0)/Σ||ε||²`의median≤.20. 이상적상쇄g=1, 상수반응g=0.
- Teacher24,240/50K진단에서 weighted-consistency/rec의A-gradient norm ratio<.10.
- sign/units/cache/augmentation/LP·MS phase가정상이며수치 오류로약한반응이생긴것이아님.

위수치는**이번시험을여는보수적heuristic**이며과거자료에서학습된최적gate가아니다. 높은q단독은부족하다. 진단이없거나불충분하면C3를실행하지않고정해진BASE반복으로이어간다.

C3는새U/A를같은name-keyedseed에서초기화해50K를처음부터학습한다. parentcheckpoint이어학습금지. synthetic draw빈도·반경·RNG와data stream은parent와같게하고weight만변경한다. exact50K에서같은calibrationbaseID로**새 τ/q cache**를만든다. parentmanifest를덮어쓰지않고s2/s5의주TA도교체하지않는다.

예산은전체`Teacher+cal+Student2+평가`로예약한다. 기술적으로유효하다면낮은Teacher성능만보고Student단계를임의취소하지않는다. 반대로NaN/손상/잘못된data/reference는그룹중단사유다. 배포·시간·수치문제를`성능실패`로기록하지않는다.

## 9. 서버별 주 순서와 유한 예비

### 9.1 s1 — QB TB 기준 재현성 + 선택적 C3

`준비 → T81002 BASE → calibration → S82001 BASE → S82002 BASE → [C3 전체묶음, 조건/시간 충족시] → S82004/82005 BASE → S82008/82009 BASE`.

시간 예비는 같은TB/PLH/D122/BASE로 **82010,82011,82016,82017**. 2개seed씩유한block이다. C3가시간을썼으면뒤의반복·예비는줄인다. 소스에없는새 seed를생성하지않는다.

### 9.2 s2 — QB TA 아래 Student one-axis

`local검사 → s4 TA import → S82003 BASE → trial한개결정 → screen82006/82007 → 확인82012/82013`.

각 screen/확인seed의BASE는정의되어있고ALT는§7조건에따라한개만활성화된다. 시간 예비 **82014,82015,82022,82023**은항상원래TA/BASE다. tuned profile의무한탐색에쓰지않는다.

### 9.3 s3 — GF2 TA 주 파이프라인 + 선택적 C3

`준비 → T91001 BASE → calibration·s5배포 → S92001/92002 BASE → [C3 전체묶음] → S92004/92005 BASE → S92008/92009 BASE`.

시간 예비 **92010,92011,92016,92017,92018,92019,92020,92021**. C3를실행해도s5의TAconsumption은원 BASE Teacher를유지한다.

### 9.4 s4 — QB TA 주 파이프라인·기본조건 집중

`준비 → T81001 BASE → calibration·s2배포 → S82001/82002 BASE → S82004/82005 BASE → S82008/82009 BASE`.

시간 예비 **82010,82011,82016,82017,82018,82019,82020,82021**. s4에는새 Teacherprofile·Student계수 변경을넣지않는다. 고정 이식군의재현성기준을확보한다.

### 9.5 s5 — GF2 TA 아래 Student one-axis

`local검사 → s3 TA import → S92003 BASE → trial한개결정 → screen92006/92007 → 확인92012/92013`.

시간 예비 **92014,92015,92022,92023**은원TA/BASE. GF2DN/MTF·fullFR경로검증을선행하고다른 센서로학습을바꾸지않는다.

### 9.6 실제 우선순위

**P0 검사 → 핵심 11개 중 자기 의존경로 → 자기screen/Teacher대조의근거확인 → 사전지정확인seed·기본반복 → 유한시간 예비**다. 단,예산이부족하면먼저C3/ALT/예비를줄이고현재 admitted block의정상평가를우선한다. 다른 서버의11개전체완료를기다리는gate는없다.

학습중계수·seed·Teacher를바꾸지않는다. Method수치코드가수정되면새 numerical revision과영향run목록을남기고서로다른release를짝비교한것처럼표시하지않는다.

## 10. 평가·목표·Sheet

### 10.1 센서별 목표

| 센서 | HQNR 선행 조건 | 그 안에서 Target 선택 | 공동목표 |
|---|---|---|---|
| QB | **raw-original H > .920** | E최소 → SCC최대 → PSNR최대 → 낮은step | **E < 3.570** |
| GF2 | **raw-original H > .964** | 동일 | **E < .552**, 별도강한기준 **E < .522** |

strict `>`는보고값을넘겠다는이번센서정책이다. 표시반올림으로판정하지않고WV3의.9585를재사용하지않는다. GF2본문Table2의.522와보충Table7의.552는원문안불일치이므로두기준을보존한다. 하나를오타로확정하지않는다. 이들은PAN-Crafter보고값비교이며모든연구에대한SOTA보장은아니다. [S4]

같은 50개 후보`1010,2020,…,49490,50000`를고정한다. 센서별epoch길이와관계없이update기준이다. 50K도달후성능이낮다고추가75K/100K를동일case안에서실행하지않는다.

### 10.2 선택점을 혼합하지 않는다

| Selection | 정의 | 기록 주의 |
|---|---|---|
| RAW_MAX | rawHQNR최대점, 동률낮은step | 본행의정의명시 |
| TARGET | H조건충족후E최소 | 적격없으면no_eligible·빈값, test-aware |
| EXACT50K | 완료update50000 | 같은학습량대조 |
| RR_VAL_SELECTED | 고정validation ERGAS최소후공식 RR/FR | 선택과test평가구분 |
| E_MIN_DIAG50 | 공식 RR test50후보중최소 | 개발용oracle진단, 독립test성과아님 |

공식 RR은센서C4·DN·GT·support20:-21·Qblock32, FR은nativePAN/full512·원LMS·센서MTF·mean-per-scene HQNR를사용한다. `(1−mean Dλ)(1−mean Ds)`로HQNR를대체하지않는다. sourcePAN을A의c로옮기거나testmask/crop을성능에맞춰변경하지않는다.

50후보평가가deadline안에끝나지않으면실제n_evaluated와pending상태를남긴다. 일부후보최저를50개전체최저라고쓰지않는다. 평가부채를입장예약에서미리제외해이상황을줄인다. Teacherreference배포는exact50Kidentity+cal완료후허용하고,Teacher의나머지평가부채는삭제하지않는다.

### 10.3 Sheet 동선

| 서버 | tab |
|---|---|
| s1 | QB-s1 |
| s2 | QB-s2 |
| s3 | GF2-s3(5090) |
| s4 | QB-s4 |
| s5 | GF2-s5 |

기존과거행·benchmark행은보존한다. 최신WV3의**의미별header**를실제로조회해RR/FR/Cost·RAW/TARGET/EXACT/VAL/E_MIN·provenance를매핑한다. 열번호88/122/246을하드코딩하지않는다. 모든선택점에서Q8label/key를실제 Q4로맞춘다.

추가필수필드: `campaign/run/role/sensor/C/maxDN/TA-TB/Teacher seed/Student seed/Teacher SHA/calibration ID/τR/qref/cacheSHA/LPphase/dataSHA/numericalrevision/profile/branch_evidence/selected_step/selected_SHA/n_evaluated/actual_updates/status`.

Params/MAC(orFLOPs)/latency/memory는새4밴드전체Student(A+syncfrontend+U)에서실측한다. WV3비용을복사하지않는다. JQM은variant와추정자료를기록하며SIPSA-equivalent라고가정하지않는다. 미측정JQM/NOA/V64는빈칸이다. 정상Target의지표는같은checkpoint로묶고runprofilecost는별도scope로명시한다.

Upsert는`(sensor,campaign,run_id)`이고,이전QGBASE의동일run이있으면alias/crosswalk로중복을차단한다. 업로드실패는로컬결과를보존하고재전송만한다. 같은학습을재실행하지않는다. readbackverified와officialcomplete/targeteligible은각각다른상태다.

## 11. 구현 인계: Case 정의와 실행 가능한 등록을 구분

**패키지의 `definitions/*.json`은실험계약이다. 기존FH12/FH20R1runner에그대로넣을수있는config라고주장하지않는다.** 현재실제entrypoint와schema에맞추어4밴드지원코드·YAML·registry·runner연결이필요하다. 다음을구현자가완료해야한다.

1. `QG40_Registry.json`을읽어명시된89개계약을확인하고,조건부분기를resolve한뒤실제config를생성한다. oldQGBASE11ID는그대로쓴다.
2. actualsensor/band order/data/MTF/normalization/source/runtime경로를binding한다. Teacher/calibration미생성시값은null로보존하고,Studentlaunch직전에는actualreference에서검증·주입한다. 구WV3상수로null을메우지않는다.
3. Teacher50K→CAL→IMPORT→Student의상태를분리한다. CAL은동일Teacherexact50K검증완료가필수지만HQNR목표달성을기다리지않는다. reference경로relocation과원수치source identity를분리하고검사를제거하지않는다.
4. epoch기반旧queue·12h/20h종료·GPU재시도루프를승계하지않는다. 공통40h·atomic예약·36hcutoff를실제runner에연결한다.
5. 각 서버dry-run에서next2runID,단일activeprofile,teacherref,case개수,remaining시간,평가부채,Sheetroute를출력한다. 실행파일존재뿐아니라registrylookup이성공해야한다.
6. run시작시resolvedconfig/initialization/data/RNG/Teacher/sampler해시를보존한다. 소스파일한개hash를전체runtime동등성으로취급하지않는다.
7. `LOCAL_READY`는수치/자산의local 검사이지성능을보고승인하는전서버lock이아니다. 다른 센서의check완료를요구하지않는다.

`validate_qg40.py`는정의의중복·수식계수·dependency참조·분기별case수를검사하는**로컬정적검증**이다. 그PASS가포팅·GPU학습·센서데이터검증PASS는아니다. 실제실행로그없이는`RUNNING`/`COMPLETED`로표시하지않는다.

## 12. 종료 보고·해석

마감에는센서별로**고정 BASE이식군**과**조건부profile군**을따로보고한다. 핵심 TA Student 3개 seed를먼저표시하고추가seed·TB·C3를분리한다. 같은 Teacher반복과다른 Teacher반복을섞어표본수를늘리지않는다.

필수표는모든run의상태,Target적격여부·H/E·동일stepSHA,Exact50K/VAL,seed별차이,학생screen/확인성공여부,Teacherq/e/c분포·reference해시,실제시간·누락사유다. condition이안맞아미실행인C3를성능 실패로쓰지않고,reference미확보를Student실패로쓰지않는다.

**미달시에도기존Method를없애지않는다.** KD·edge·q·Aadjustment가유지된baseline을보존하고어느지표가남았는지기록한다. 큰architecturegrid,새PANauxiliary,uniformq,τ수동조정,NOA주추론으로결과를덮지않는다.

## 13. 산출물과 출처

- 본MD: 시간·Method·서버·분기·평가·구현 계약 및 전체case부록.
- `PAN_QG40_Cases_2026-09-20.csv`: 89개정확한학습case,seed,reference,조건,block,순서.
- `PAN_QG40_Baseline11_2026-09-20.csv`: 이전준비안의ID를보존한핵심 11개.
- `QG40_Registry.json`, `definitions/*.json`: machine-readable계약. 실행등록완료를뜻하지않음.
- `QG40_Actions.json`: 기본calibration3,조건부calibration2,referenceimport2. 학습run수에합산하지않음.
- `PAN_QG40_Checklist_2026-09-20.csv`: 기존20개+새4개실행검사.
- `QG40_BudgetContract.json`, `QG40_BranchContract.json`: 시간·조건부규칙.
- `queues/*_definition_order.txt`: branch전후순서를확인하는명세,legacyrunner입력아님.
- `validate_qg40.py`, `QG40_Validation_Result.json`: 실제수행한정적정의검사와한계.

### 출처

[S1] `PAN_QB_GF2_DistributionAware_Preparation_2026-09-20.md`, §§2–12. 데이터·reference·계수·서버계통·목표·미확인항목.  
[S2] `PAN_WV3_Closeout_Review_FrozenRecipe_2026-09-20.md` 및 `PAN_WV3_FrozenRecipe_2026-09-20.json`, 고정구조·Teacher·Student·종결정책.  
[S3] 제공 `Pansharpening_research.pdf`, pp.8–10 consistency와native rec, p.11 q/e구분, pp.12–15 KD/edge/A routing.  
[S4] 제공 `pancrafter.pdf`, p.6 데이터/patch/metrics, p.7 Table2 및p.13 Table7·8의QB/GF2 benchmark. GF2 .522/.552불일치보존.  
[S5] 제공 `uknowdiff.pdf`, p.6 Table1의C/bitdepth/nominalN. 그논문의τ=1·diffusion/SWT·loss를우리Method에이식한것이아님.  
[S6] `PAN_QB_GF2_Preflight_Checklist_2026-09-20.csv`의QG00–QG19. 원문과새40h상세의관계는새CSV origin열에표시.  
[S7] [S1]이이미조회한`fh12/model.py`, `data.py`, `losses.py`, `calibration.py`, `evaluation.py`, `pa/aligner.py`, `tools/repair_qb_ms.py`. 이번문서에서최신서버동작을인증한것은아님.

본패키지`source_basis/`에[S1,S2,S6]원문과SHA를보존했다. 아래case값·분기threshold·시간예약은이번설계이며위논문의실험결과에서그대로가져온값이아니다.

---

## 부록 A. 기본 11개 — exact run ID

| 서버 | 역할 | Teacher reference | TS | SS | 정확한 run ID |
|---|---|---|---:|---:|---|
| s1 | Teacher | QB_TB | 81002 | — | `QGBASE_QB_TB_P0_W112_D123_TS81002_FRESH50_v1` |
| s1 | Student | QB_TB | 81002 | 82001 | `QGBASE_QB_S_PLH_W104_D122_TB_TS81002_SS82001_BASE_FRESH50_v1` |
| s1 | Student | QB_TB | 81002 | 82002 | `QGBASE_QB_S_PLH_W104_D122_TB_TS81002_SS82002_BASE_FRESH50_v1` |
| s2 | Student | QB_TA | 81001 | 82003 | `QGBASE_QB_S_PLH_W104_D122_TA_TS81001_SS82003_BASE_FRESH50_v1` |
| s3 | Teacher | GF2_TA | 91001 | — | `QGBASE_GF2_TA_P0_W112_D123_TS91001_FRESH50_v1` |
| s3 | Student | GF2_TA | 91001 | 92001 | `QGBASE_GF2_S_PLH_W104_D122_TA_TS91001_SS92001_BASE_FRESH50_v1` |
| s3 | Student | GF2_TA | 91001 | 92002 | `QGBASE_GF2_S_PLH_W104_D122_TA_TS91001_SS92002_BASE_FRESH50_v1` |
| s4 | Teacher | QB_TA | 81001 | — | `QGBASE_QB_TA_P0_W112_D123_TS81001_FRESH50_v1` |
| s4 | Student | QB_TA | 81001 | 82001 | `QGBASE_QB_S_PLH_W104_D122_TA_TS81001_SS82001_BASE_FRESH50_v1` |
| s4 | Student | QB_TA | 81001 | 82002 | `QGBASE_QB_S_PLH_W104_D122_TA_TS81001_SS82002_BASE_FRESH50_v1` |
| s5 | Student | GF2_TA | 91001 | 92003 | `QGBASE_GF2_S_PLH_W104_D122_TA_TS91001_SS92003_BASE_FRESH50_v1` |

## 부록 B. 서버별 전체 학습 case — 상호 배타적 후보 포함

이 표를 그대로 전체 mandatory 큐로 실행하지 않는다. `SCREEN/CONFIRM_EQ_*`는 local 선택한 한 profile만 활성화한다. `TC3_ENABLE`은 별도 증거·전체예산이 있는 경우만 활성화한다. TIME_RESERVE는 유한 BASE 목록이다.

### s1

| Case | 순서/block | 역할·profile | TS/SS | Tier·조건 | 정확한 run ID |
|---|---|---|---|---|---|
| QG40-C001 | 10/s1_T_BASE/1 | Teacher/BASE | 81002/— | P0_BASELINE; ALWAYS | `QGBASE_QB_TB_P0_W112_D123_TS81002_FRESH50_v1` |
| QG40-C002 | 30/s1_BASE_01/1 | Student/BASE | 81002/82001 | P0_BASELINE; ALWAYS | `QGBASE_QB_S_PLH_W104_D122_TB_TS81002_SS82001_BASE_FRESH50_v1` |
| QG40-C003 | 30/s1_BASE_01/2 | Student/BASE | 81002/82002 | P0_BASELINE; ALWAYS | `QGBASE_QB_S_PLH_W104_D122_TB_TS81002_SS82002_BASE_FRESH50_v1` |
| QG40-C004 | 45/s1_TC3_PACKAGE/1 | Teacher/C3 | 81002/— | P2_TEACHER_CONDITIONAL; TC3_ENABLE_s1 | `QG40_QB_TB_C3_P0_W112_D123_TS81002_C3_FRESH50_v1` |
| QG40-C005 | 45/s1_TC3_PACKAGE/3 | Student/BASE | 81002/82001 | P2_TEACHER_CONDITIONAL; TC3_ENABLE_s1 | `QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1` |
| QG40-C006 | 45/s1_TC3_PACKAGE/4 | Student/BASE | 81002/82002 | P2_TEACHER_CONDITIONAL; TC3_ENABLE_s1 | `QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82002_BASE_FRESH50_v1` |
| QG40-C007 | 50/s1_BASE_02/1 | Student/BASE | 81002/82004 | P1_FIXED_REPEAT; ALWAYS | `QG40_QB_S_PLH_W104_D122_TB_TS81002_SS82004_BASE_FRESH50_v1` |
| QG40-C008 | 50/s1_BASE_02/2 | Student/BASE | 81002/82005 | P1_FIXED_REPEAT; ALWAYS | `QG40_QB_S_PLH_W104_D122_TB_TS81002_SS82005_BASE_FRESH50_v1` |
| QG40-C009 | 70/s1_BASE_03/1 | Student/BASE | 81002/82008 | P1_FIXED_REPEAT; ALWAYS | `QG40_QB_S_PLH_W104_D122_TB_TS81002_SS82008_BASE_FRESH50_v1` |
| QG40-C010 | 70/s1_BASE_03/2 | Student/BASE | 81002/82009 | P1_FIXED_REPEAT; ALWAYS | `QG40_QB_S_PLH_W104_D122_TB_TS81002_SS82009_BASE_FRESH50_v1` |
| QG40-C011 | 100/s1_RESERVE_01/1 | Student/BASE | 81002/82010 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TB_TS81002_SS82010_BASE_FRESH50_v1` |
| QG40-C012 | 100/s1_RESERVE_01/2 | Student/BASE | 81002/82011 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TB_TS81002_SS82011_BASE_FRESH50_v1` |
| QG40-C013 | 101/s1_RESERVE_02/1 | Student/BASE | 81002/82016 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TB_TS81002_SS82016_BASE_FRESH50_v1` |
| QG40-C014 | 101/s1_RESERVE_02/2 | Student/BASE | 81002/82017 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TB_TS81002_SS82017_BASE_FRESH50_v1` |

### s2

| Case | 순서/block | 역할·profile | TS/SS | Tier·조건 | 정확한 run ID |
|---|---|---|---|---|---|
| QG40-C015 | 30/s2_BASE_01/1 | Student/BASE | 81001/82003 | P0_BASELINE; ALWAYS | `QGBASE_QB_S_PLH_W104_D122_TA_TS81001_SS82003_BASE_FRESH50_v1` |
| QG40-C016 | 50/s2_SCREEN_82006/1 | Student/BASE | 81001/82006 | P1_SCREEN_BASE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82006_BASE_FRESH50_v1` |
| QG40-C017 | 50/s2_SCREEN_82006/2 | Student/A24R | 81001/82006 | P1_SCREEN_ALT; SCREEN_s2_EQ_A24R | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82006_A24R_FRESH50_v1` |
| QG40-C018 | 50/s2_SCREEN_82006/2 | Student/B20 | 81001/82006 | P1_SCREEN_ALT; SCREEN_s2_EQ_B20 | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82006_B20_FRESH50_v1` |
| QG40-C019 | 50/s2_SCREEN_82006/2 | Student/E10 | 81001/82006 | P1_SCREEN_ALT; SCREEN_s2_EQ_E10 | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82006_E10_FRESH50_v1` |
| QG40-C020 | 51/s2_SCREEN_82007/1 | Student/A24R | 81001/82007 | P1_SCREEN_ALT; SCREEN_s2_EQ_A24R | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82007_A24R_FRESH50_v1` |
| QG40-C021 | 51/s2_SCREEN_82007/1 | Student/B20 | 81001/82007 | P1_SCREEN_ALT; SCREEN_s2_EQ_B20 | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82007_B20_FRESH50_v1` |
| QG40-C022 | 51/s2_SCREEN_82007/1 | Student/E10 | 81001/82007 | P1_SCREEN_ALT; SCREEN_s2_EQ_E10 | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82007_E10_FRESH50_v1` |
| QG40-C023 | 51/s2_SCREEN_82007/2 | Student/BASE | 81001/82007 | P1_SCREEN_BASE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82007_BASE_FRESH50_v1` |
| QG40-C024 | 70/s2_CONFIRM_82012/1 | Student/BASE | 81001/82012 | P1_CONFIRM_BASE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82012_BASE_FRESH50_v1` |
| QG40-C025 | 70/s2_CONFIRM_82012/2 | Student/A24R | 81001/82012 | P1_CONFIRM_ALT; CONFIRM_s2_EQ_A24R | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82012_A24R_FRESH50_v1` |
| QG40-C026 | 70/s2_CONFIRM_82012/2 | Student/B20 | 81001/82012 | P1_CONFIRM_ALT; CONFIRM_s2_EQ_B20 | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82012_B20_FRESH50_v1` |
| QG40-C027 | 70/s2_CONFIRM_82012/2 | Student/E10 | 81001/82012 | P1_CONFIRM_ALT; CONFIRM_s2_EQ_E10 | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82012_E10_FRESH50_v1` |
| QG40-C028 | 71/s2_CONFIRM_82013/1 | Student/A24R | 81001/82013 | P1_CONFIRM_ALT; CONFIRM_s2_EQ_A24R | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82013_A24R_FRESH50_v1` |
| QG40-C029 | 71/s2_CONFIRM_82013/1 | Student/B20 | 81001/82013 | P1_CONFIRM_ALT; CONFIRM_s2_EQ_B20 | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82013_B20_FRESH50_v1` |
| QG40-C030 | 71/s2_CONFIRM_82013/1 | Student/E10 | 81001/82013 | P1_CONFIRM_ALT; CONFIRM_s2_EQ_E10 | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82013_E10_FRESH50_v1` |
| QG40-C031 | 71/s2_CONFIRM_82013/2 | Student/BASE | 81001/82013 | P1_CONFIRM_BASE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82013_BASE_FRESH50_v1` |
| QG40-C032 | 100/s2_RESERVE_01/1 | Student/BASE | 81001/82014 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82014_BASE_FRESH50_v1` |
| QG40-C033 | 100/s2_RESERVE_01/2 | Student/BASE | 81001/82015 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82015_BASE_FRESH50_v1` |
| QG40-C034 | 101/s2_RESERVE_02/1 | Student/BASE | 81001/82022 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82022_BASE_FRESH50_v1` |
| QG40-C035 | 101/s2_RESERVE_02/2 | Student/BASE | 81001/82023 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82023_BASE_FRESH50_v1` |

### s3

| Case | 순서/block | 역할·profile | TS/SS | Tier·조건 | 정확한 run ID |
|---|---|---|---|---|---|
| QG40-C036 | 10/s3_T_BASE/1 | Teacher/BASE | 91001/— | P0_BASELINE; ALWAYS | `QGBASE_GF2_TA_P0_W112_D123_TS91001_FRESH50_v1` |
| QG40-C037 | 30/s3_BASE_01/1 | Student/BASE | 91001/92001 | P0_BASELINE; ALWAYS | `QGBASE_GF2_S_PLH_W104_D122_TA_TS91001_SS92001_BASE_FRESH50_v1` |
| QG40-C038 | 30/s3_BASE_01/2 | Student/BASE | 91001/92002 | P0_BASELINE; ALWAYS | `QGBASE_GF2_S_PLH_W104_D122_TA_TS91001_SS92002_BASE_FRESH50_v1` |
| QG40-C039 | 45/s3_TC3_PACKAGE/1 | Teacher/C3 | 91001/— | P2_TEACHER_CONDITIONAL; TC3_ENABLE_s3 | `QG40_GF2_TA_C3_P0_W112_D123_TS91001_C3_FRESH50_v1` |
| QG40-C040 | 45/s3_TC3_PACKAGE/3 | Student/BASE | 91001/92001 | P2_TEACHER_CONDITIONAL; TC3_ENABLE_s3 | `QG40_GF2_S_PLH_W104_D122_TA_C3_TS91001_SS92001_BASE_FRESH50_v1` |
| QG40-C041 | 45/s3_TC3_PACKAGE/4 | Student/BASE | 91001/92002 | P2_TEACHER_CONDITIONAL; TC3_ENABLE_s3 | `QG40_GF2_S_PLH_W104_D122_TA_C3_TS91001_SS92002_BASE_FRESH50_v1` |
| QG40-C042 | 50/s3_BASE_02/1 | Student/BASE | 91001/92004 | P1_FIXED_REPEAT; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92004_BASE_FRESH50_v1` |
| QG40-C043 | 50/s3_BASE_02/2 | Student/BASE | 91001/92005 | P1_FIXED_REPEAT; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92005_BASE_FRESH50_v1` |
| QG40-C044 | 70/s3_BASE_03/1 | Student/BASE | 91001/92008 | P1_FIXED_REPEAT; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92008_BASE_FRESH50_v1` |
| QG40-C045 | 70/s3_BASE_03/2 | Student/BASE | 91001/92009 | P1_FIXED_REPEAT; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92009_BASE_FRESH50_v1` |
| QG40-C046 | 100/s3_RESERVE_01/1 | Student/BASE | 91001/92010 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92010_BASE_FRESH50_v1` |
| QG40-C047 | 100/s3_RESERVE_01/2 | Student/BASE | 91001/92011 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92011_BASE_FRESH50_v1` |
| QG40-C048 | 101/s3_RESERVE_02/1 | Student/BASE | 91001/92016 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92016_BASE_FRESH50_v1` |
| QG40-C049 | 101/s3_RESERVE_02/2 | Student/BASE | 91001/92017 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92017_BASE_FRESH50_v1` |
| QG40-C050 | 102/s3_RESERVE_03/1 | Student/BASE | 91001/92018 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92018_BASE_FRESH50_v1` |
| QG40-C051 | 102/s3_RESERVE_03/2 | Student/BASE | 91001/92019 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92019_BASE_FRESH50_v1` |
| QG40-C052 | 103/s3_RESERVE_04/1 | Student/BASE | 91001/92020 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92020_BASE_FRESH50_v1` |
| QG40-C053 | 103/s3_RESERVE_04/2 | Student/BASE | 91001/92021 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92021_BASE_FRESH50_v1` |

### s4

| Case | 순서/block | 역할·profile | TS/SS | Tier·조건 | 정확한 run ID |
|---|---|---|---|---|---|
| QG40-C054 | 10/s4_T_BASE/1 | Teacher/BASE | 81001/— | P0_BASELINE; ALWAYS | `QGBASE_QB_TA_P0_W112_D123_TS81001_FRESH50_v1` |
| QG40-C055 | 30/s4_BASE_01/1 | Student/BASE | 81001/82001 | P0_BASELINE; ALWAYS | `QGBASE_QB_S_PLH_W104_D122_TA_TS81001_SS82001_BASE_FRESH50_v1` |
| QG40-C056 | 30/s4_BASE_01/2 | Student/BASE | 81001/82002 | P0_BASELINE; ALWAYS | `QGBASE_QB_S_PLH_W104_D122_TA_TS81001_SS82002_BASE_FRESH50_v1` |
| QG40-C057 | 50/s4_BASE_02/1 | Student/BASE | 81001/82004 | P1_FIXED_REPEAT; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82004_BASE_FRESH50_v1` |
| QG40-C058 | 50/s4_BASE_02/2 | Student/BASE | 81001/82005 | P1_FIXED_REPEAT; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82005_BASE_FRESH50_v1` |
| QG40-C059 | 70/s4_BASE_03/1 | Student/BASE | 81001/82008 | P1_FIXED_REPEAT; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82008_BASE_FRESH50_v1` |
| QG40-C060 | 70/s4_BASE_03/2 | Student/BASE | 81001/82009 | P1_FIXED_REPEAT; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82009_BASE_FRESH50_v1` |
| QG40-C061 | 100/s4_RESERVE_01/1 | Student/BASE | 81001/82010 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82010_BASE_FRESH50_v1` |
| QG40-C062 | 100/s4_RESERVE_01/2 | Student/BASE | 81001/82011 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82011_BASE_FRESH50_v1` |
| QG40-C063 | 101/s4_RESERVE_02/1 | Student/BASE | 81001/82016 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82016_BASE_FRESH50_v1` |
| QG40-C064 | 101/s4_RESERVE_02/2 | Student/BASE | 81001/82017 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82017_BASE_FRESH50_v1` |
| QG40-C065 | 102/s4_RESERVE_03/1 | Student/BASE | 81001/82018 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82018_BASE_FRESH50_v1` |
| QG40-C066 | 102/s4_RESERVE_03/2 | Student/BASE | 81001/82019 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82019_BASE_FRESH50_v1` |
| QG40-C067 | 103/s4_RESERVE_04/1 | Student/BASE | 81001/82020 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82020_BASE_FRESH50_v1` |
| QG40-C068 | 103/s4_RESERVE_04/2 | Student/BASE | 81001/82021 | TIME_RESERVE; ALWAYS | `QG40_QB_S_PLH_W104_D122_TA_TS81001_SS82021_BASE_FRESH50_v1` |

### s5

| Case | 순서/block | 역할·profile | TS/SS | Tier·조건 | 정확한 run ID |
|---|---|---|---|---|---|
| QG40-C069 | 30/s5_BASE_01/1 | Student/BASE | 91001/92003 | P0_BASELINE; ALWAYS | `QGBASE_GF2_S_PLH_W104_D122_TA_TS91001_SS92003_BASE_FRESH50_v1` |
| QG40-C070 | 50/s5_SCREEN_92006/1 | Student/BASE | 91001/92006 | P1_SCREEN_BASE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92006_BASE_FRESH50_v1` |
| QG40-C071 | 50/s5_SCREEN_92006/2 | Student/A24R | 91001/92006 | P1_SCREEN_ALT; SCREEN_s5_EQ_A24R | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92006_A24R_FRESH50_v1` |
| QG40-C072 | 50/s5_SCREEN_92006/2 | Student/B20 | 91001/92006 | P1_SCREEN_ALT; SCREEN_s5_EQ_B20 | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92006_B20_FRESH50_v1` |
| QG40-C073 | 50/s5_SCREEN_92006/2 | Student/E10 | 91001/92006 | P1_SCREEN_ALT; SCREEN_s5_EQ_E10 | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92006_E10_FRESH50_v1` |
| QG40-C074 | 51/s5_SCREEN_92007/1 | Student/A24R | 91001/92007 | P1_SCREEN_ALT; SCREEN_s5_EQ_A24R | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92007_A24R_FRESH50_v1` |
| QG40-C075 | 51/s5_SCREEN_92007/1 | Student/B20 | 91001/92007 | P1_SCREEN_ALT; SCREEN_s5_EQ_B20 | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92007_B20_FRESH50_v1` |
| QG40-C076 | 51/s5_SCREEN_92007/1 | Student/E10 | 91001/92007 | P1_SCREEN_ALT; SCREEN_s5_EQ_E10 | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92007_E10_FRESH50_v1` |
| QG40-C077 | 51/s5_SCREEN_92007/2 | Student/BASE | 91001/92007 | P1_SCREEN_BASE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92007_BASE_FRESH50_v1` |
| QG40-C078 | 70/s5_CONFIRM_92012/1 | Student/BASE | 91001/92012 | P1_CONFIRM_BASE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92012_BASE_FRESH50_v1` |
| QG40-C079 | 70/s5_CONFIRM_92012/2 | Student/A24R | 91001/92012 | P1_CONFIRM_ALT; CONFIRM_s5_EQ_A24R | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92012_A24R_FRESH50_v1` |
| QG40-C080 | 70/s5_CONFIRM_92012/2 | Student/B20 | 91001/92012 | P1_CONFIRM_ALT; CONFIRM_s5_EQ_B20 | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92012_B20_FRESH50_v1` |
| QG40-C081 | 70/s5_CONFIRM_92012/2 | Student/E10 | 91001/92012 | P1_CONFIRM_ALT; CONFIRM_s5_EQ_E10 | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92012_E10_FRESH50_v1` |
| QG40-C082 | 71/s5_CONFIRM_92013/1 | Student/A24R | 91001/92013 | P1_CONFIRM_ALT; CONFIRM_s5_EQ_A24R | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92013_A24R_FRESH50_v1` |
| QG40-C083 | 71/s5_CONFIRM_92013/1 | Student/B20 | 91001/92013 | P1_CONFIRM_ALT; CONFIRM_s5_EQ_B20 | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92013_B20_FRESH50_v1` |
| QG40-C084 | 71/s5_CONFIRM_92013/1 | Student/E10 | 91001/92013 | P1_CONFIRM_ALT; CONFIRM_s5_EQ_E10 | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92013_E10_FRESH50_v1` |
| QG40-C085 | 71/s5_CONFIRM_92013/2 | Student/BASE | 91001/92013 | P1_CONFIRM_BASE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92013_BASE_FRESH50_v1` |
| QG40-C086 | 100/s5_RESERVE_01/1 | Student/BASE | 91001/92014 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92014_BASE_FRESH50_v1` |
| QG40-C087 | 100/s5_RESERVE_01/2 | Student/BASE | 91001/92015 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92015_BASE_FRESH50_v1` |
| QG40-C088 | 101/s5_RESERVE_02/1 | Student/BASE | 91001/92022 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92022_BASE_FRESH50_v1` |
| QG40-C089 | 101/s5_RESERVE_02/2 | Student/BASE | 91001/92023 | TIME_RESERVE; ALWAYS | `QG40_GF2_S_PLH_W104_D122_TA_TS91001_SS92023_BASE_FRESH50_v1` |

## 부록 C. 실행 상태 용어

`DEFINED_NOT_DEPLOYED` → `REGISTERED` → `LOCAL_READY` → `WAIT_REFERENCE/READY` → `RUNNING` → `TRAIN50K_COMPLETE` → `OFFICIAL_EVAL_COMPLETE` → `UPLOAD_VERIFIED` → `ARCHIVED`.

조건부 미충족은 `NOT_ADMITTED_CONDITION`, 시간 부족은 `NOT_ADMITTED_BUDGET`, 마감 중단은 `INCOMPLETE_AT_DEADLINE`, 수치/자산 오류는 `BLOCKED_INTEGRITY`다. `no_eligible`은 완료된 성능 판정이지 학습 실패 상태가 아니다. 실제 실행하지 않은 단계는 완료로 바꾸지 않는다.
