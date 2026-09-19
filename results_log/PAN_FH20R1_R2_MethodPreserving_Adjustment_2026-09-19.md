# FH20R1-R2 — 기존 Method를 보존하는 WV3 실험 방향 조정안

**작성일:** 2026-09-19  
**적용 대상:** WV3, s1–s5  
**기존 학습 campaign:** `WV3_FH20R1_20260919_v1` — 유지  
**이번 우선순위 revision:** `FH20R1_PRIORITY_METHOD_PRESERVE_R2_20260919`  
**수치 method:** `FH12_SYNC_FREQ_NATIVE_TEACHER_v1` — 유지  
**상태:** 실행 인계용 MD·우선순위 overlay·검사 목록 작성. 서버 코드·실제 큐·Sheet·학습은 이 작업에서 변경하지 않았다.

> 이번 조정은 **기존 Teacher → consistency → Student hard/soft KD → q-weighted GT edge → q-weighted Student Aligner adjustment를 보존하면서, 어떤 비교를 먼저 끝내고 무엇을 확인할지 바꾸는 것**이다.  
> 진행 중인 FH20R1의 누적시간을 이어 사용한다. **새로 20시간을 추가하거나 기존 20시간 시계를 리셋하는 계획이 아니다.**

---

## 0. 실행자가 먼저 읽을 결론

**중심 후보는 `PLH · W104D122 · BASE`로 올린다.** 단, 한 seed의 성공으로 모든 서버의 Teacher를 F1로 바꾸거나 PL·PH·D121 비교군을 없애지 않는다.

이번 조정의 실제 변경은 다음이다.

| 항목 | 기존 편성의 무게중심 | 이번 조정 |
|---|---|---|
| 입력·구조의 주 가설 | PL×D122 누락 비교와 PH depth 반복 | **PLH×D122의 HQNR 적격 Target 재현**, PL은 낮은 ERGAS의 중요한 비교군 |
| 성능 요약 | 본행 RAW_MAX를 먼저 읽기 쉬움 | **TARGET을 대표 성능으로 읽고**, RAW_MAX·Exact50K·VAL·E_MIN을 분리 |
| W112D122 | 다수 반복 뒤쪽의 예비적 확인 | **시작한 pair/quartet을 완결한 뒤 첫 capacity bridge를 앞당김** |
| Teacher·loss | F1–F5와 기존 KD/edge/adjustment | **변경 없음** |
| A10·N2 | 사전 정의된 조건부 경로 | 현재 선택된 BASE/F2 대체 경로 유지. 원인·가용성만 별도 점검하고 소급 전환하지 않음 |
| 시간 | 서버별 FH20R1 유효작업 ≥20h | **기존 ledger 계승**, CORE 완결·유한 예비 원칙 유지 |

**새 학습 case ID는 0개다.** 기존 145개 정의 중 현재 기록된 branch에 해당하는 **CORE Student 84개 + 시간예비 40개**를 그대로 사용한다. 다른 branch의 A10·N2 정의도 삭제하지 않고 비활성 대안으로 보존한다.

현재 Sheet에서 CORE 12개의 완료가 확인된다. 나머지 72개를 “미시작”으로 간주하지 않는다. 일부는 현재 학습 중이거나 로컬에서 완료됐지만 미업로드일 수 있으므로 **V00 로컬 대조 후** 다음 순서를 적용한다.

---

## 1. 적용 근거와 아직 단정하지 않을 부분

### 1.1 사용한 결과 범위

직전 `PAN_FH20R1_Interim_12Run_Review_2026-09-19.md`와 12run CSV, 기존 FH12 21run 결과, 원 FH20R1 case CSV를 기준으로 구성했다. 작성 중 live Sheet를 다시 export하여 **신규 FH20R1 완료 행 12개와 아래 핵심값이 여전히 일치**함을 확인했다. [R1–R4]

서버의 실제 프로세스, 전체 campaign ledger, A/U 교차 원자료, 상세 calibration 원자료는 이번에 직접 조회하지 않았다. 이를 확인 대상으로 남겼다. 새 결과가 뒤늦게 등록되더라도 이미 정의한 case의 seed·loss·Teacher를 자동 변경하지 않는다.

### 1.2 중요한 두 자산

| 자산 | 선택점 | Step | HQNR(raw) ↑ | ERGAS ↓ | 의미 |
|---|---|---:|---:|---:|---|
| s1 / F1 / PLH W104D122 / S73101 | RAW_MAX | 22220 | 0.961216430090 | 2.090374252705 | 높은 FR peak |
| **동일 run** | **TARGET** | **38380** | **0.958732552970** | **2.056060193721** | 현재 우선순위에 맞는 대표 후보 |
| 동일 run | EXACT50K | 50000 | 0.958166643425 | 2.051061842559 | HQNR 하한 미달 |
| s2 / F2 / PL W104D122 / S72001 | RR_VAL_SELECTED=EXACT50K | 50000 | 0.953174037349 | 2.040436446740 | 낮은 RR 오차, HQNR 미달 |
| 동일 s2 run | E_MIN_DIAG50 | 46460 | 0.953021374178 | 2.039931415906 | **test-aware 개발 진단**, 공동목표 아님 |

s1 TARGET checkpoint SHA256:

```text
3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50
```

s2 E_MIN checkpoint SHA256:

```text
76cb4282e51c92f45aa1d9930d23dfa29a3d689198f1719d94f23a4caab59029
```

s1은 50후보 중 19개가 HQNR 하한을 통과했다. 19개가 연속된 구간인지, 개선이 특정 FR 장면에 집중되는지는 V02에서 확인한다. **최고 H=0.961216과 다른 step의 E=2.051062를 합친 대표값을 만들지 않는다.**

### 1.3 방향 조정의 근거

- FH12와 이번 결과를 합친 **9개 같은-Teacher/같은-seed depth 대응 block에서 D122의 Exact50K ERGAS가 모두 낮아졌다.** HQNR은 2개 개선·7개 악화이므로 D122가 두 목표를 자동으로 해결한다고 해석하지 않는다.
- 완성된 s4/F4/S72001과 s5/F5/S72002의 2×2에서는 **D121일 때 PL의 ERGAS가 낮았지만, D122일 때 PLH가 PL보다 HQNR·ERGAS 모두 좋았다.**
- 신규 12개 Student의 RAW_MAX→Exact50K 구간에서 ERGAS 감소와 D_s 증가가 동반됐다. RAW_MAX의 높은 H 자체에는 선택 정의가 개입하므로, D_s/c/출력의 시간변화를 별도로 확인한다.
- 새 최고 자산은 **BASE·F1**이다. **A10이나 N2의 성능 이득으로 설명할 수 없다.**
- 이는 중간 결과다. **서로 다른 Teacher/서버의 절대 성능을 곧바로 Teacher 입력의 인과효과로 해석하지 않는다.**

### 1.4 현재 성공 판정

\[
\mathcal K_H=\{k:H_{\mathrm{raw},k}\ge0.9585\},\qquad
k^*=\arg\min_{k\in\mathcal K_H}(E_k,-SCC_k,-PSNR_k,k).
\]

공동목표는 \(E_{k^*}<2.040\)이다. 적격점이 없으면 `no_eligible`이고 Target 값은 빈칸이다.

**이번 데이터에서 공동목표가 이미 달성됐다고 판정하지 않는다.** s1은 H 조건을 만족하나 E가 남았고, s2의 E_MIN은 E 기준에 근접하지만 H가 낮다. PAN-Crafter의 WV3 보고값 H=.958/E=2.040과 사용자 개발 하한 .9585는 구분한다. [R5 Table 1]

---

## 2. 변경하지 않는 제안 Method

프로젝트 연구자료 p.8–10의 Stage 1과 p.12–15의 Stage 2가 핵심 구조다. 이 revision은 그 구성요소를 빼지 않는다. 아래 수식의 정규화·detach·빈도·계수는 **현재 FH12/FH20R1 구현 계약**이다. 도식의 축약 표기를 이유로 구현 계수를 임의 변경하지 않는다. [R2 §3; R6 pp.8–15]

### 2.1 Forward: Aligner는 PAN과 MS만 입력받는다

\[
M=U_4(MS),\quad L=U_4(LPAN),\quad
c=A(P_{m4},M_{m4}),
\]
\[
\widetilde P=W(P,c),\quad
\widetilde L=W(L,c),\quad
\widetilde H=\widetilde P-\widetilde L,\quad
Z=M+F(X).
\]

| Layout | Reconstruction U-Net 입력 | 채널 |
|---|---|---:|
| P0 | \([\widetilde P,M]\) | 9 |
| PL | \([\widetilde P,\widetilde L,M]\) | 10 |
| PH | \([\widetilde P,\widetilde H,M]\) | 10 |
| PLH | \([\widetilde P,\widetilde L,\widetilde H,M]\) | 11 |

- A의 입력은 PAN1/MS8, 내부 margin4다. **LP/HP를 A stem에 추가하거나 성분마다 c를 따로 예측하지 않는다.**
- c는 `(dy,dx)` HR-PAN pixel 단위다. LP를 먼저 HR로 올리고 동일한 grid를 적용한다. HP는 signed 차이이며 clip/abs/재정규화를 하지 않는다.
- Warp는 현재 FP32/bicubic/border/align_corners=False 규약을 유지한다. MS base·GT·최종 출력 좌표계는 움직이지 않으며 base를 정확히 한 번 더한다.
- Teacher는 자기 \(c_T\), Student는 자기 \(c_S\)를 쓴다. Student 추론에 Teacher c를 대신 제공하지 않는다.
- **LP/HP 경로의 c-gradient도 보존**한다. q·KD·edge의 U/A 분리와 혼동하지 않는다.
- 입력 LP cache의 Gaussian σ1.98/k41/replicate/offset2, split별 phase와 hash를 유지한다. \(W(P-L,c)=W(P,c)-W(L,c)\)의 선형성을 이용하되 filter/downsample/warp 순서가 교환 가능하다고 가정하지 않는다.

### 2.2 Stage 1: Teacher reconstruction + PAN shift consistency

\[
c_0=A_T(P,M),\quad
P_\epsilon=W(P,\epsilon),\quad c_\epsilon=A_T(P_\epsilon,M),
\]
\[
L_{\mathrm{off}}=\frac1{2B}\sum_i
\left\|c_{\epsilon,i}+\epsilon_i-\operatorname{sg}(c_{0,i})\right\|_1,
\]
\[
L_T(t)=\operatorname{mean}|Z_T-Y|
+\mathbf 1[t\bmod2=1]\,10^{-4}L_{\mathrm{off}}.
\]

`t`는 0-based update다. Native reconstruction은 매 update Teacher A/U를 함께 갱신한다. **Synthetic shifted PAN은 A-only consistency branch에만 들어가고 U-Net의 reconstruction 입력으로 들어가지 않는다.** Native reconstruction의 c0는 detach하지 않는다.

지금 쓰는 F1–F5는 이 절차로 이미 학습된 **exact50K A+U 전체 reference**다. 이를 재사용한다고 Stage 1 또는 consistency를 생략한 방법으로 바뀌는 것은 아니다. Teacher를 매 Student마다 새로 학습하지 않으며, 이미 학습된 A/U와 calibration을 고정해서 사용한다.

N2INIT는 기존에 정의된 **Teacher A 초기화 대조**일 뿐이다. 이를 수행하더라도 reconstruction·consistency 학습을 없애거나 N2 donor를 곧바로 최종 Teacher로 대체하지 않는다.

### 2.3 Stage 2: error-based hard/soft KD

\[
e_T(i,p)=\operatorname{mean}_b|Z_T-Y|,\quad
e_S(i,p)=\operatorname{mean}_b|Z_S-Y|,
\]
\[
d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\qquad
a_T=\operatorname{sg}\frac{[e_S-e_T]_+}{e_S+10^{-6}},
\]
\[
\ell_{H,i}=\operatorname{mean}_p[(1+\alpha d_T)e_S],
\]
\[
\ell_{K,i}=\operatorname{mean}_p\left[
\beta(1-d_T)a_T\,\operatorname{mean}_b|Z_S-Z_T|\right].
\]

**\(\alpha=1.0,\beta=0.1\)**을 유지한다. β는 soft 항에 한 번만 곱한다. Teacher·GT와 d/a 계산은 detach하되 live eS·Student–Teacher discrepancy는 미분 가능하다.

Teacher가 실제로 더 나은 위치에서만 soft가 활성화되는 원리를 보존한다. 따라서 어떤 batch에서 gated soft가 작거나 0이라는 사실과 **KD 항을 코드에서 삭제한 상태**는 다르다.

### 2.4 q-based GT edge와 Student Aligner adjustment

Frozen Teacher A의 AXIS16 진단값을 사용한다.

\[
q_{i,r}=\frac1{2K}\sum_{j=1}^{K}
\left\|c^{T}_{i,r,j}+\epsilon_j-c^{T}_{i,r,0}\right\|_1,\quad K=16,
\]
\[
s_{i,r}=\operatorname{sg}
\frac{q_{\mathrm{ref}}}{q_{\mathrm{ref}}+q_{i,r}},
\]
\[
L_U=\operatorname{mean}_i\left[\ell_{H,i}+\ell_{K,i}
+0.002\,s_i\ell_{E,i}\right],\qquad
L_A=\operatorname{mean}_i[s_i\ell_{H,i}].
\]

q 분자의 2를 추가하거나 batch 평균으로 재정규화하지 않는다. q는 Student가 아닌 **Frozen Teacher A**에서 얻고, augmentation view와 cache index를 정확히 맞춘다.

\(\ell_E\)는 **최종 HRMS 출력과 GT의 signed Scharr x/y 차이**이며 두 방향 0.5 평균, 경계1 pixel 제외의 현재 구현을 유지한다. 입력 HPAN을 GT edge 대신 쓰지 않는다.

| 파라미터 집합 | 갱신에 사용하는 loss |
|---|---|
| Teacher U | Native HRMS reconstruction |
| Teacher A | Native reconstruction + 정해진 consistency |
| Student U | Hard + adaptive soft KD + q-weighted GT edge |
| Student A | **q-weighted GT hard만** |

U/A의 gradient를 각각 미분하고 양쪽 gradient 계산이 끝나기 전에 어느 optimizer도 step하지 않는다. **두 loss를 더한 뒤 전체 모델에 backward하는 구현은 금지한다.**

### 2.5 이번 조정에서 명시적으로 금지하는 방법 변경

**KD 제거·β=0, edge 제거·λE=0, q 제거/항상 0.5 대체, Aligner 제거/NOA를 주 추론으로 채택, Student A freeze, Teacher consistency 제거는 이 revision의 학습 case가 아니다.**

또한 Student A에 soft·edge·offset·직접 c-distillation을 추가하지 않는다. 새 ERGAS/L2 loss, PAN reconstruction auxiliary, attention, 다른 정규화, LP kernel/phase 변화도 넣지 않는다.

이 목록은 영구적인 연구 금지가 아니라 **이번 round에서 제안 method를 훼손하지 않는 범위**다. 원인 진단에서 임시 A/U를 교차하는 것은 별도 평가이며 학습 method나 최종 모델을 대체하지 않는다.

---

## 3. 공통 설정과 reference 보존

| 항목 | 유지할 값 |
|---|---|
| Dataset | WV3, 8 bands, train9714 / val1080 / RR20 / FR20 — 실제 hash 검증 |
| 입력 크기 | train PAN64/MS16, RR PAN256, FR PAN512 |
| 정규화 | max_pixel2047, 기존 \(2DN/2047-1\) |
| Backbone | MS-only, LN, attention OFF, mode modulation OFF |
| 학습량 | 각 run fresh50K, batch48, workers4, FP32/no AMP |
| Optimizer | AdamW, betas(.9,.999), eps1e−8, wd .01 |
| U LR | peak1e−4, 기존 warmup100+cosine |
| A LR | 현재 선택된 BASE에서 peak3e−6, 기존 cosine |
| 초기화 | U fresh, Teacher A의 독립 clone; 기존 named-tensor 정책·추가 L/H kernel=0 |
| Teacher 선택 | 지정된 exact50K의 A/U 전체, frozen |
| q / τR | 기존 train-only reference/calibration 그대로 |
| 추론 | 같은 run·같은 step의 정상 A_ON, 전체 U-Net RR/FR |

### 3.1 F1–F5는 현재 서버별로 그대로 유지

| 서버 | Reference | Teacher 입력 | T seed | exact50K SHA 앞16자리 | τR / q_ref: 감사용 표시 |
|---|---|---|---:|---|---|
| s1 | F1 | P0 | 71001 | `04ef8e756ae8b229` | 0.012118559331 / 0.453260362148 |
| s2 | F2 | PL | 71001 | `d2832f5a2d40536a` | 0.012083493173 / 0.454883515835 |
| s3 | F3 | PH | 71001 | `8a971c857a1f386f` | 0.012089714408 / 0.455190271139 |
| s4 | F4 | PLH | 71001 | `9b6fd8ccc0e3e567` | 0.012065213174 / 0.455461412668 |
| s5 | F5 | PLH | 71002 | `dff2b0c1b4e00422` | 0.012087114155 / 0.456601321697 |

위 숫자를 config에 다시 적는 방식으로 사용하지 않는다. **원 reference manifest·실제 cache와 full SHA를 읽어 검증**한다. Full SHA와 원 경로는 부록 reference JSON을 함께 제공한다.

Student의 input/W/D/seed만 바뀌는 경우 calibration을 재생성하지 않는다. Teacher checkpoint·source data·augmentation·전처리가 실제로 바뀔 때만 새 reference revision이 필요하다. 이번에 그런 변경을 자동 수행하지 않는다.

**F1이 최고 Student를 만들었다고 나머지 서버의 Teacher를 F1로 바꾸지 않는다.** 아직 server/Teacher/seed 효과가 섞여 있다.

---

## 4. 서버별 조정된 실험 순서

### 4.1 우선순위 적용 원칙

1. **이미 admission된 atomic block이 최우선**이다. 현재 학습과 미완성 pair/quartet을 원래 순서대로 마친다.
2. 아래 표는 그 이후 **완전히 미시작·미admission block**의 선호 순서다. 현재 block보다 앞으로 삽입하지 않는다.
3. block 내부의 run 순서는 원 `primary_order/alternative_orders` 그대로 유지한다. 매번 PLH/D122를 먼저 돌리도록 임의 변경하지 않는다.
4. 완료된 run은 원 결과/정체성/업로드를 대조해 재사용한다. **새 run ID·새 seed·새 config 수치를 만들지 않는다.**
5. 이미 로컬이 아래 상태보다 앞서 있다면 로컬 상태를 우선하며 이 MD snapshot 위치로 되돌리지 않는다.

### 4.2 전체 배정표

| 서버 | 당장 완결할 비교 | 다음 우선순위 | 보존하는 대조 | CORE run 수 |
|---|---|---|---|---:|
| **s1** | F1·PLH의 다음 731xx D121/D122 pair | **S73101의 TARGET 재현**, 73102–73107 대응 반복 | D121을 유지해 D122 개선·seed효과 분리 | 14 |
| **s2** | F2·S72001 **PL/PLH D122 pair** | 73201–73204의 같은 입력 pair 반복 | 낮은 RR의 PL과 높은 H 가능성의 PLH | 10 |
| **s3** | F3·S73301 PH D121/D122 pair | 첫 **W104/W112 D122 pair B06**을 앞으로, 나머지 depth·width 반복 | PH 전체 경로를 유지 | 20 |
| **s4** | F4·S73401의 **PL/PLH×D121/D122 quartet** | **B06: PLH W112D122/S72001 bridge**, 다음 quartet, 첫 W112 depth pair | 입력×depth의 4조건 유지 | 20 |
| **s5** | F5·S73501의 같은 quartet | **B06: PLH W112D121/D122 S72002 pair**를 앞으로 | 다른 Teacher 아래 4조건 반복 | 20 |
| **합계** | | | | **84** |

### 4.3 s1: 최고 자산 재현이 중심, A10으로 바꾸지 않는다

현재 실제 분기는 `S1_NO_A_SUPPORT_OR_INCONCLUSIVE`다. 따라서 **PLH W104D121/D122 BASE pair**가 활성 경로다.

- B01–B07: seed73101–73107, 각 D121/D122.
- B01은 Sheet에서 완료 확인. B02 이후는 로컬 상태 확인.
- time reserve B08–B11: seed73108–73111, 동일 pair.
- 우선순위 순서는 원래대로 유지한다.
- s1 TARGET은 비교 자산으로 보존하되, 그 checkpoint에서 fine-tuning하거나 해당 seed만 반복하지 않는다.
- V04를 수행해 A-support의 미지지/불충분을 설명한다. **그 결과를 이용해 이미 선택된 branch를 중간에 바꾸지 않는다.** A10을 재시험할 필요가 생기면 이후 독립 cohort로 명시한다.

### 4.4 s2: F2 PL/PLH D122를 계속하고 donor 문제를 별도로 처리

현재 branch는 `S2_DONOR_OR_REFERENCE_UNAVAILABLE`, reason=`BLOCKED_DONOR`다.

- B01: seed72001, PL/PLH W104D122. PL은 등록 완료, PLH는 로컬 확인 후 완결.
- B02–B05: seed73201–73204, 동일 pair.
- time reserve B06–B09: seed73205–73208, 동일 pair.
- V03은 2.039931의 아주 작은 E threshold 통과를 채택하기 전에 수행한다.
- V07의 donor 가용성 점검은 병행하되 **현재 F2 입력 비교를 기다리게 하지 않는다.**
- donor가 뒤늦게 확보되더라도 진행 중 block의 F2를 N2PL로 바꾸지 않는다. N2INIT는 정의·출처·가용성 결과를 보존한 **별도 후속 Teacher 대조**이며 자동 전환하지 않는다.

### 4.5 s3: PH 대조는 유지하고 첫 width×D122 비교만 앞당긴다

Reference F3를 유지한다.

```text
CORE 선호 순서:
B01 → B02 → B06 → B03 → B04 → B05 → B07 → B08 → B09 → B10
RESERVE:
B11 → B12 → B13 → B14
```

- B01/S72002 완료, B02/S73301은 한쪽만 Sheet에 있으므로 pair를 완결한다.
- B06/S73305: PH D122에서 W104/W112를 비교한다. **W112D122의 효과가 아직 미측정인 공백**을 먼저 채운다.
- B03–B05/S73302–73304: PH W104D121/D122 반복은 취소하지 않는다.
- B07–B10/S73306–73309: width pair는 그 이후 실행한다.
- 시간예비 S73310–73313의 기존 depth pair도 유지한다.

PH 결과를 새 PLH run으로 이름만 바꿔 대체하지 않는다. 이 서버의 역할은 **HPAN 입력 아래 depth/width 효과를 확인하는 비교군**이다.

### 4.6 s4: PLH×D122의 균형과 capacity 결합을 먼저 확인

```text
CORE 선호 순서:
B01 → B02 → B06 → B03 → B07 → B04 → B05
RESERVE:
B08 → B09
```

- B01: PL W104D122 S72001은 완료 확인.
- B02/S73401: PL D121/D122가 등록됐고 PLH D121/D122를 완결해야 한다.
- **B06/S72001: PLH W112D122 한 개.** 기존 FH12 PLH W112D121과 W104D122를 재사용해 한계비용이 작은 capacity 결합 비교를 만든다.
- B03/S73402: 두 번째 신규 quartet 완결.
- **B07/S73405: PLH W112D121/D122 pair.** S72001에서 관측한 depth 효과의 다른 seed 확인.
- B04–B05/S73403–73404: quartet 반복.
- 기존 S73406–73407 reserve quartet은 최소시간 미달일 때만.

**좋은 PLH만 남기고 PL·D121을 버리는 조정이 아니다.** 결과의 우선 해석은 PLH D122에 두되, 비교 설계는 2×2 전체를 보존한다.

### 4.7 s5: 독립 Teacher 아래에서 같은 결론이 나오는지 확인

```text
CORE 선호 순서:
B01 → B02 → B06 → B03 → B04 → B05
RESERVE:
B07 → B08
```

- B01/S72002의 PL W104D121/D122는 완료 확인.
- B02/S73501 quartet은 완결한다.
- **B06/S72002: PLH W112D121/D122 pair**를 앞당겨, 기존 PLH W104D121/D122와 width×depth 2×2를 완성한다.
- B03–B05/S73502–73504: 원 quartet 반복.
- S73505–73506 reserve 유지.

s4/F4와 s5/F5는 Teacher seed와 서버가 다르다. 이 둘을 Student seed만의 반복으로 합치지 않는다.

### 4.8 바뀌지 않는 개수

| 구분 | 개수 | 해석 |
|---|---:|---|
| 원 case 정의 전체 | 145 | 상호 배타적 대안 포함 |
| 현재 branch CORE | 84 | 모든 Student, 이미 등록된 12개 포함 |
| 현재 branch RESERVE | 40 | CORE 후 실제 유효시간 <20h일 때 유한 block 실행 |
| 이번에 새로 만든 학습 ID | **0** | 기존 수치 recipe와 ID 재사용 |
| Sheet에서 확인한 CORE 완료 | 12 | 로컬 미업로드/진행 상태는 별도 |
| 나머지 CORE | 72 | **미시작이 아니라 로컬 대조 대상** |

총 run 수를 늘리는 것이 아니라 **같은 실험 집합에서 중요한 판별 결과를 앞당긴다.**

---

## 5. 안전한 적용과 시간 정책

### 5.1 기존 20시간을 이어 사용

서버별 기존 FH20R1 `effective_seconds`를 보존한다.

\[
T_{\mathrm{remaining,min}}=\max(0,\ 20h-T_{\mathrm{FH20R1,already\ credited}}).
\]

이 값은 최소시간의 잔여량이지, 실행 중인 block을 강제 종료할 하드 deadline이 아니다. **CORE 완료와 20h 충족을 모두 요구**하고, 미완성 admitted block은 끝낸다.

- FH12/MIX20H 과거 시간은 포함하지 않는다.
- 이번 R2에서 필요한 실제 진단은 기존 campaign ledger에 정확히 한 번만 기록한다.
- 완료행 시간 합을 전체 campaign ledger로 대신하지 않는다. 현재 학습·미업로드 결과가 있을 수 있다.
- 대기·sleep·업로드 재시도·쓸모없는 중복 평가로 20h를 채우지 않는다.
- CORE 완료 후 <20h이면 기존 유한 reserve를 block째 실행한다. 예비 소진 후에도 부족하면 사실대로 보고하며 무한 seed를 생성하지 않는다.
- 한 서버가 20h를 채웠다고 다른 서버를 중단시키거나, 반대로 다른 서버를 기다리게 하지 않는다.

### 5.2 수치 code/config를 그대로 두고 큐 우선순위만 적용

제공 JSON/CSV는 **우선순위 명세**다. 현재 runner가 외부 overlay를 읽는 기능을 이미 갖췄다고 가정하지 않는다.

적용 담당자는 다음을 수행한다.

1. runtime 상태·branch·현재 block·완료 결과·기존 ledger를 `runtime_reconciliation.json`에 보존한다.
2. 현재 admitted block은 원래 순서대로 완결한다.
3. 아직 admission되지 않은 block에만 이번 순서를 반영한다. 파일·메모리에 이미 적재된 queue가 어느 것인지 함께 확인한다.
4. trainer/수치 파일·registered config·Teacher manifest를 변경하지 않는다.
5. source identity가 scheduler/runner 코드까지 포함하는 환경에서는 실행 중에 파일을 수정하지 않는다. controller 지원을 추가해야 하면 별도 worktree/원 source 보존 및 다음 안전한 경계에서 배포한다.
6. 새 consumer가 기존 reference를 읽는 검증은 bridge로 보존한다. 과거 artifact의 source SHA를 현재 값으로 덮어쓰지 않는다.
7. resume가 필요한 활성 run은 **원 source/runtime/full-state**에서 이어간다. 새 revision에서 weights만 읽고 exact resume라고 표시하지 않는다.
8. 적용 후 실제 다음 두 run ID·현재 block·priority revision·ledger hour를 readback 보고한다.

**원 model/loss 파일이나 case numeric definition의 차이가 발견되면, 이를 단순 큐 revision으로 숨기지 않는다.** 원인과 영향을 먼저 기록한다.

---

## 6. 확인 작업: 담당자·산출물·완료 기준

아래의 `P0`는 무결성 또는 잘못된 실행 방지 항목, `P1`은 과학적 해석을 위한 항목이다. **P1을 모든 서버의 공통 학습 대기 조건으로 만들지 않는다.**

- 기존 산출물이 있으면 먼저 읽고 재사용한다.
- GPU 진단은 별도 자원이 없으면 해당 서버의 안전한 작업 경계에서 실행한다.
- 로그 수집·hash 비교·곡선 분석은 학습 소스·RNG를 변경하지 않는 별도 작업으로 수행한다.
- 하나의 reference/공통 수치 오류가 발견되면 그 영향을 받는 신규 학습만 보류한다. 성능 미달과 무결성 오류를 혼동하지 않는다.
- 모든 확인 작업의 초기 상태는 **TO_VERIFY**다. 이 문서를 작성했다는 이유로 PASS로 표시하지 않는다.

### V00 · run/step/현재 atomic block/로컬 official 결과/미업로드 결과/ledger/실제 queue revision 대조

**우선순위 / 담당:** P0 / 각 서버  
**실행 시점:** 다음 block 입장 전  
**산출물:** `runtime_reconciliation.json`  
**완료 기준:** 모든 활성 run과 미완성 block이 식별되고 완료 자산은 재실행 대상에서 제외된다.  
**이상·미완료 시:** 영향받은 신규 admission만 보류한다. 현재 정상 학습은 원 소스로 계속한다.  
**상태:** `TO_VERIFY`

### V01 · Teacher consistency와 Student H/K/E/q, U/A gradient routing, A trainable, frozen Teacher 및 동기 warp 확인

**우선순위 / 담당:** P0 / 각 서버 구현 담당  
**실행 시점:** 수치 코드가 바뀌면 신규 run 전, 미변경이면 기존 검증 receipt 재사용  
**산출물:** `method_invariant_receipt.json`  
**완료 기준:** §3의 수식·gradient·초기화·정규화 계약이 동일하다. synthetic 활성 표본에서 각 경로를 검사한다.  
**이상·미완료 시:** 수식/gradient 위반이 있으면 해당 release 신규 학습을 중지한다. 안전검사를 지우거나 loss를 끄지 않는다.  
**상태:** `TO_VERIFY`

### V02 · S73101 RAW_MAX/TARGET/EXACT50K/VAL SHA 보존, 19개 eligible의 step·H/E/Dλ/Ds·장면별 결과 확보

**우선순위 / 담당:** P1 / s1  
**실행 시점:** 다음 새 seed 결과 해석 전; 학습과 독립  
**산출물:** `S73101_selection_audit.json + eligible_curve.csv`  
**완료 기준:** TARGET=38380 및 정상 A/U SHA가 재현되고 19개 후보 목록과 연속 구간 길이가 계산된다.  
**이상·미완료 시:** RAW_MAX 요약만 정리하지 않는다. 기존 결과는 보존하고 차이를 명시한다.  
**상태:** `TO_VERIFY`

### V03 · 실행 revision 0ec113…와 기준 cccedee…의 model/loss/data/warp/metric 파일, 데이터/LP source SHA, 동일 checkpoint 평가 동등성 비교

**우선순위 / 담당:** P0-채택 / s2  
**실행 시점:** 2.039931의 근소한 threshold 통과를 성과로 채택하기 전  
**산출물:** `s2_numerical_parity.json`  
**완료 기준:** 해시 범위를 분리해 수치 차이 유무를 설명하고 E_MIN46460/EXACT50000의 재평가 오차를 보고한다.  
**이상·미완료 시:** bundle SHA 차이만으로 폐기하지 않는다. 실제 수치 불일치가 있으면 해석을 보류하고 affected release를 조사한다.  
**상태:** `TO_VERIFY`

### V04 · branch_record와 기존 P0/PLH A/U 교차 원자료를 읽어 NO_SUPPORT와 INCONCLUSIVE의 원인을 구분

**우선순위 / 담당:** P1 / s1  
**실행 시점:** 이번 캠페인 분석 중; 이미 선택한 branch는 불변  
**산출물:** `s1_branch_evidence_review.md`  
**완료 기준:** 대각 재현·비대각6개·4개 판정 비교의 값, 미계산/오류가 분리된다.  
**이상·미완료 시:** 불충분을 A 무관의 증거로 쓰지 않는다. 현재 BASE depth pair를 계속하며 A10으로 소급 전환하지 않는다.  
**상태:** `TO_VERIFY`

### V05 · F1–F5 calibration의 q 분포·반경별 반응·native c·augmentation indexing·q weight 분포 대조

**우선순위 / 담당:** P1 / 각 서버  
**실행 시점:** 시간 여유가 있는 run 경계; 기존 cache 우선  
**산출물:** `teacher_q_audit.json`  
**완료 기준:** 원 q_ref/τ_R/cache SHA 보존과 분포/기울기 통계가 확보된다. 절대 q와 s 분포를 분리한다.  
**이상·미완료 시:** q≈상수-A 기준만으로 q를 제거/균등화/재보정하지 않는다. 실제 원인 확인을 별도 기록한다.  
**상태:** `TO_VERIFY`

### V06 · 같은 run/layout/W/D 내 A/U{10100,24240,50000}3×3, diagonal 재현 후 원인 분리

**우선순위 / 담당:** P1 / s1·s3·s4·s5  
**실행 시점:** 기존 진단 누락 시에만 동일 GPU의 안전한 작업 경계  
**산출물:** `au_cross_summary.csv + per_scene.json`  
**완료 기준:** 정상3개·교차6개, A/U SHA와 c, 장면별 ΔH/ΔE/ΔDs가 구분돼 저장된다.  
**이상·미완료 시:** 교차성능은 최종 Target이 아니다. late U+early A가 나쁜 것만으로 A 무관이라 하지 않는다.  
**상태:** `TO_VERIFY`

### V07 · BLOCKED_DONOR를 파일부재/이전실패/SHA/step/view mismatch로 분류하고 N2 donor 가용성을 확인

**우선순위 / 담당:** P1 / s2  
**실행 시점:** 로컬 병렬 파일 점검; Student 대기 조건 아님  
**산출물:** `donor_availability_receipt.json`  
**완료 기준:** 알려진 donor A의 전체 SHA·step50000·margin4·config 경로가 검증되거나 정확한 차단 사유가 남는다.  
**이상·미완료 시:** 기존 F2 PL/PLH branch 계속. N2 이름의 fresh 대체나 진행 중 Teacher 교체 금지.  
**상태:** `TO_VERIFY`

### V08 · train/RR/FR LP recipe·phase·canonical split hash, P/L 같은 grid·H=P−L·signed range·고정 M-frame 확인

**우선순위 / 담당:** P1 / 각 서버  
**실행 시점:** 신규 입력 또는 reader 수정 전; 결과 진단은 별도  
**산출물:** `frequency_frontend_receipt.json`  
**완료 기준:** LP/HP c-gradient와 64/256/512 shape/identity가 재현되고 GT가 forward에 들어가지 않는다.  
**이상·미완료 시:** LP phase/kernel을 성능에 맞춰 바꾸지 않는다. 오류 수정은 새 수치 revision·영향표로 분리한다.  
**상태:** `TO_VERIFY`

### V09 · aligner full vs crop64 consensus, 크기별 q·c, z-score/GN/GAP 점검 상태 수집

**우선순위 / 담당:** P1 / s1·s4  
**실행 시점:** 이미 계획한 size 진단 결과가 없을 때 대표 자산만  
**산출물:** `aligner_size_diagnostics_status.md`  
**완료 기준:** 기존 C00–C05별 완료/미완료/미구현과 실제 산출물·장면 수·SHA가 연결된다.  
**이상·미완료 시:** 추론 정책을 자동 교체하지 않는다. 전체 U-Net tiling으로 원인을 섞지 않는다.  
**상태:** `TO_VERIFY`

### V10 · hard/soft/edge의 크기·gradient norm, soft 활성 비율, eS−eT, q 가중 분포를 train 고정 표본에서 읽는다

**우선순위 / 담당:** P1 / 각 서버  
**실행 시점:** 가능한 기존 10K/24240/50K train diagnostics부터  
**산출물:** `loss_activity_diagnostics.csv`  
**완료 기준:** 정상 gated-zero와 β/edge 구현 누락을 구별하고 Teacher/hash 불변·RNG 격리가 확인된다.  
**이상·미완료 시:** loss가 약하다고 즉시 제거/자동 재가중하지 않는다. FR test에 GT를 가정하지 않는다.  
**상태:** `TO_VERIFY`

### V11 · 동일 seed의 공통 tensor/배치·증강 순서, W/D만의 변경, params/FLOPs/inference scope 재측정

**우선순위 / 담당:** P1 / 각 서버  
**실행 시점:** W112D122 또는 반복 pair 결과 업로드 전  
**산출물:** `pair_identity_and_cost.json`  
**완료 기준:** Teacher reference·α/β/λE·A-LR가 같고 capacity 변경 외 numerical diff가 없다.  
**이상·미완료 시:** 다른 seed나 Teacher와 짝짓지 않는다. W/D가 다른 전체 함수가 같은 초기값이라고 주장하지 않는다.  
**상태:** `TO_VERIFY`

### V12 · RAW_MAX/TARGET/EXACT50K/VAL/E_MIN 구분, 원본 precision, 공식50후보, Sheet header readback, source-row/step/SHA

**우선순위 / 담당:** P0-보고 / 업로드 담당·각 서버  
**실행 시점:** 매 postrun과 요약표 갱신  
**산출물:** `selection_upload_receipt.json`  
**완료 기준:** 주 비교값은 TARGET, 미적격은 빈칸, 교차모델/다른 checkpoint 지표 혼합 없음.  
**이상·미완료 시:** 업로드 실패는 재업로드만. 이미 완료 학습을 재시작하거나 legacy 본열을 덮어쓰지 않는다.  
**상태:** `TO_VERIFY`

### V13 · same-Teacher same-seed 대응차, 완료 quartet, Target 적격률·조건부 E·Exact50K·VAL의 동시 집계

**우선순위 / 담당:** P1 / 각 서버  
**실행 시점:** 블록 완료 후  
**산출물:** `block_summary.csv`  
**완료 기준:** 미완성 pair와 공유 baseline의 의존성, test-aware 선택이 명시되고 모든 seed가 포함된다.  
**이상·미완료 시:** 최고 seed만 남겨 평균을 만들거나 no_eligible을 임의 E로 대체하지 않는다.  
**상태:** `TO_VERIFY`

### V14 · 기존 FH20R1 유효시간 ledger의 interval union, 이번 진단 시간의 중복 제외, CORE·reserve 종료 조건

**우선순위 / 담당:** P0-종료 / 각 서버  
**실행 시점:** CORE/예비 경계와 캠페인 종료  
**산출물:** `campaign_time_and_completion.json`  
**완료 기준:** 기존 누적시간+유효 신규작업>=20h이며 admitted block이 완결된다.  
**이상·미완료 시:** 20h 시계 리셋/대기 시간 채우기/무한 seed 생성 금지. 전체 ledger 미확보를 20h 완료라고 쓰지 않는다.  
**상태:** `TO_VERIFY`

### 6.1 V02의 필수 산출물: 높은 HQNR이 언제까지 유지되는가?

s1 S73101에서 고정50후보의 step/H/E/Dλ/Ds/eligible와 장면별 값을 확보한다. 최소 다음을 계산한다.

- eligible 개수, 첫/마지막 eligible step, **연속 eligible 구간**과 가장 긴 구간;
- RAW_MAX·TARGET·Exact50K의 장면별 FR 변화와 밴드별 RR 오류;
- 후보 구간을 임의 보간하여 실제로 평가하지 않은 threshold 교차점을 성과로 쓰지 않는다;
- 원 대표 G23, FH12 W112D121, 새 TARGET은 각자 고유 run·step·SHA로 보존한다.

현재 높은 H의 원인이 P0 Teacher인지, 초기화/seed인지, D122인지 한 결과만으로 결정하지 않는다.

### 6.2 V03의 수치 동등성: s2의 근소한 ERGAS 통과

`evaluator SHA256` 열은 이전 코드에서 전체 source bundle hash를 담았으므로, 이름만 보고 평가 함수의 차이라고 단정하지 않는다.

검사 파일은 최소 `fh12/model.py`, `fh12/losses.py`, 데이터 reader/LP 생성, `pa/aligner.py`, `pa/warp.py`, RR/FR/Q2n 구현이다. raw H5와 split별 LP canonical hash도 확인한다.

같은 weights를 원 환경과 비교 환경에서 평가하고 **출력, c, 장면별 metric, 평균 metric**의 최대 차이를 보고한다. 참고용 구현 검사 기준은 동일 FP32 reader에서 output atol3e−6/rtol2e−5, macro E/H 절대차1e−6이다. 이 수치는 성능 유의성 기준이 아니며, hardware 차이로 초과할 때 tolerance를 조용히 키우지 말고 원 환경 재현 또는 차이 설명을 남긴다.

`E_MIN=2.039931`은 `2.040` 아래 폭이 약6.9e−5이고 validation-selected는2.040436이다. 근소한 crossing 자체보다 다른 seed의 방향과 정상 TARGET을 본다.

### 6.3 V05의 q 진단은 q 제거를 위한 절차가 아니다

상수 응답 Aligner의 AXIS16 기준은
\[
q_{\mathrm{const}}=\tfrac12\operatorname{mean}(0.25,0.5,1,2)=0.46875.
\]

현재 q_ref가 그 값에 가깝다는 이유만으로 native 정합 정확도를 수치화하거나 q가 무용하다고 판정하지 않는다. `q_std/quantiles`, 반경별 `c_shift−c_native`, 기대 방향 `−epsilon`, 축별 cross-response, native c와 s 분포를 함께 본다.

상대적 consistency와 native 정합의 오차는 다른 개념이다. **d_T도 q로 대체하지 않는다.** 연구자료 p.11–15에서 구분한 reconstruction error e와 alignment-consistency error q의 역할을 유지한다. [R6]

### 6.4 V06의 A/U 교차와 V09의 크기 검증

한 run의 A/U{10100,24240,50000}을 사용하며 **다른 run/layout/W/D를 섞지 않는다.** 정상 diagonal 3개가 원 평가를 재현한 뒤 cross 6개를 본다. 모든 경우 P/L/H는 같은 A가 만든 c로 다시 구성한다.

권장 로컬 대상은 원 계획을 유지한다.

| 서버 | 우선 대상 |
|---|---|
| s1 | F1 아래 FH12 P0·PLH D121의 기존 진단, 추가로 새 S73101 PLH D122의 기록 |
| s2 | F2 q·native c·donor receipt; Student A/U 추가 진단은 결과 여유에 따라 |
| s3 | F3 PH D121/D122의 기존 진단 |
| s4 | F4 PLH W104D122·W112D121 |
| s5 | F5 PLH W104D121/D122 |

3×3 cross는 평가용이다. full-state를 확보한 특정 branch로 학습을 fork하는 것은 별도 실험이며 이번 queue overlay에 포함되지 않는다.

size 검증은 U-Net을 full-frame으로 유지한 채 A의 관측 영역만 바꾼다. crop-consensus가 좋더라도 정상 추론으로 자동 채택하거나 train cache를 바꾸지 않는다. z-score/GN/GAP 원인 분리를 구현하지 않았다면 **미구현**으로 남기며 크기별 q 한 숫자로 전체 검증을 완료 처리하지 않는다.

---

## 7. 결과 읽기·보고 방식

### 7.1 동일 checkpoint 규칙과 대표값

| 선택점 | 사용 목적 | 다른 선택점과 결합 가능 여부 |
|---|---|---|
| RAW_MAX | 최고 HQNR과 peak 시점 | 불가 |
| **TARGET** | **주 비교: H 하한 후 E 최소** | 불가 |
| EXACT50K | 같은 학습량에서 입력/capacity 효과 | 불가 |
| RR_VAL_SELECTED | validation 선택에 따른 공식 RR/FR | 불가 |
| E_MIN_DIAG50 | test-aware RR 하한 진단 | 독립 test 우위 주장 불가 |
| A/U cross | 원인 진단 | 정상 Target/benchmark 표에 넣지 않음 |

본 Sheet의 기존 RAW_MAX 열을 유지하고 TARGET을 별도 비교 요약에 사용한다. 요약 탭에는 `selection_id / step / checkpoint SHA / raw HQNR / E`를 같이 기록해 “Ours+LPAN/HPAN”이라는 이름만으로 선택점이 생략되지 않게 한다. 기존 NOA·V64·JQM을 계산하지 않았으면 빈칸이다.

### 7.2 입력×depth와 seed 집계

**주 단위는 같은 Teacher·같은 서버·같은 Student seed의 완결 pair/quartet**이다.

각 조건에 대해 아래를 같이 보고한다.

1. 완료 run 수 / 예정 run 수, Target 적격 run 수와 비율, joint-pass 수;
2. 적격 run에서만 Target E의 median/range와 개별 값; 미적격은 별도 수로 보고;
3. 모든 run의 Exact50K·VAL E/H/Dλ/Ds와 대응차;
4. 후보별 eligible count·late eligible 여부 — 50개 checkpoint를 독립 seed로 세지 않음.

입력×depth 상호작용은 각 quartet에서
\[
I_E=(E_{\mathrm{PLH},122}-E_{\mathrm{PL},122})
-(E_{\mathrm{PLH},121}-E_{\mathrm{PL},121})
\]
로 계산할 수 있다. 음수는 해당 block에서 PLH가 추가 depth의 RR 이득을 더 받았다는 관측이다. 이것을 직접적인 내부 feature 메커니즘의 증명으로 쓰지 않는다.

**중간 우수 결과를 보고 재편한 스케줄이라는 사실도 보존한다.** 새 seed가 시험셋 선택의 독립성을 복구하지 않는다. 최고값만 보고하지 말고 개발 단계의 모든 완료 결과와 validation 선택을 병기한다.

### 7.3 다음 채택에 필요한 근거

- PLH D122는 지금부터 **우선 후보**이지 전 서버 확정 recipe가 아니다.
- s1 다음 두 사전 지정 seed73102/73103의 완결 pair를 우선 읽는다. 그 결과를 기다리며 다른 서버가 멈추지는 않는다.
- s4/s5 각각 최소 다음 한 개의 완결 quartet을 확보해 기존 입력×depth 순위 변화가 재현되는지 확인한다.
- W112D122는 첫 bridge와 대응 pair에서 RR 개선·HQNR 비용·cost를 함께 본다. 단일 peak만으로 W를 일괄 올리지 않는다.
- no_eligible run이 많으면 적격 run의 낮은 E 평균만으로 좋은 설정이라 판단하지 않는다.
- F1/F2의 절대 차이는 Teacher·서버 교락이 있으므로, reference 전환은 **별도 가설의 새로운 대조** 없이 시행하지 않는다.

---

## 8. 확인 결과에 따른 후속 분기 — 현재 Method 유지가 전제

| 확인되는 상황 | 가능한 다음 조치 | 자동으로 하면 안 되는 것 |
|---|---|---|
| PLH D122의 Target 개선이 여러 seed에서 재현 | 같은 method의 대표 입력/구조 후보로 좁힘 | 낮은 E/H의 다른 checkpoint를 조합 |
| W112D122에서 정상 Target/VAL 개선 | 한 축 capacity 후보로 유지하고 동일 seed/다른 seed 대응 | attention·PAN auxiliary까지 함께 추가 |
| A/U cross가 후반 A 변화의 기여를 지지 | **별도 cohort**에서 BASE/A10 같은 nonzero A-LR 대조 검토 | A freeze, NOA 배포, edge/soft를 A에 전달 |
| A/U cross가 미지지/불충분 | 현재 BASE 및 입력/깊이 비교 계속, U/c/edge 진단 보완 | “A는 원인 아님” 또는 “q 제거” 결론 |
| N2 donor 가용성이 회복 | 기존 정의의 N2INIT+consistency Teacher 대조를 후속으로 보존 | 현재 F2 block/기존 branch를 소급 N2로 변경 |
| soft/edge gradient가 작게 관측 | 데이터 분포·advantage gating·구현 누락을 먼저 구별 | KD/edge를 삭제하거나 coefficient를 자동0으로 만들기 |
| 모든 조합에서 후반 D_s 악화 지속 | 고정 train 진단으로 각 항의 기여를 확인한 뒤 **nonzero 계수/학습정책의 단일 축** 별도 검토 | 이번 실행 중 loss와 checkpoint 선택을 동시에 변경 |

위 표는 후속 결정을 위한 범위다. **현재 제공 overlay가 새로운 loss 또는 A10 cohort를 자동 실행하도록 정의하지 않는다.** 기존 구성요소는 그대로 두고, 근거가 생겼을 때도 한 축의 새 실험임을 명확히 표시한다.

---

## 9. 적용 완료 보고 형식

각 서버 담당자는 다음 필드를 채워 전달한다. MD를 받았다는 사실이나 YAML이 있다는 사실만으로 `applied=true`로 표시하지 않는다.

```json
{
  "server": "sX",
  "campaign_id": "WV3_FH20R1_20260919_v1",
  "priority_revision": "FH20R1_PRIORITY_METHOD_PRESERVE_R2_20260919",
  "method_numeric_changed": false,
  "case_numeric_changed": false,
  "old_campaign_ledger_preserved": true,
  "global_performance_lock": false,
  "branch_from_actual_record": "<actual>",
  "current_run_id": "<actual or none>",
  "current_update": "<actual>",
  "current_admitted_block": "<actual or none>",
  "next_two_run_ids": ["<actual>", "<actual>"],
  "teacher_reference_sha256": "<actual>",
  "local_completed_runs": [],
  "sheet_uploaded_runs": [],
  "completed_but_upload_pending": [],
  "effective_hours_from_campaign_ledger": "<actual>",
  "source_runtime_for_active_run_preserved": true,
  "verification_receipts": {},
  "applied": false
}
```

체크리스트는 `PASS / NEEDS_DATA / INCONCLUSIVE / FAIL_IDENTITY / FAIL_NUMERICAL / NOT_IMPLEMENTED` 등으로 구분하고 사유를 쓴다. **무결성 실패를 과학적 실패로, 미구현을 미지지 결과로 변환하지 않는다.**

---

## 10. 제공 파일과 적용 범위

- 본 MD: 유지할 method, 방향 조정, 실제 block 우선순위, 확인 작업, 보고 기준.
- `PAN_FH20R1_R2_QueueOverlay_2026-09-19.csv`: 현재 branch CORE84+RESERVE40의 정확한 기존 case ID와 순서. 12개는 Sheet 완료, 나머지는 로컬 대조가 필요하다고 표기.
- `priority_overlay.json`: 같은 우선순위의 기계 판독 명세. **기존 runner용 실행 config라고 주장하지 않는다.**
- `PAN_FH20R1_R2_VerificationChecklist_2026-09-19.csv`: V00–V14의 담당·산출물·기준·실패처리.
- `reference_assets.json`: 기존 F1–F5 및 N2 donor 출처. 새 reference를 만들지 않음.
- `definition_validation.json`: 기존 선택 branch의 case membership/순서/개수·중복·신규ID0 검사.
- `sources/`: 이번 live 12run 조회 snapshot과 근거 문서/원 case CSV.

검증한 것은 **문서·case membership·숫자·실험 순서의 일관성**이다. 서버의 실제 적용, GPU smoke, gradient 검사, 기존 branch 원자료와 상세 q 진단은 담당자가 수행해야 한다.

---

## 부록 A. 근거 자료

**[R1]** `PAN_FH20R1_Interim_12Run_Review_2026-09-19.md`, 특히 §§2–8. 수치 원본은 동명12run CSV.  
**[R2]** `PAN_FH20R1_AllServers_20Hplus_ExperimentPlan_2026-09-19.md` 및 `PAN_FH20R1_Cases_2026-09-19.csv`. 수치 recipe/기존 ID/branch/20h 정의의 기준.  
**[R3]** `PAN_FH12_Results_Review_and_NextDirection_2026-09-19.md`와 `PAN_FH12_21Run_Results_2026-09-19.csv`.  
**[R4]** Live Sheet `pan-cvpr27`, 2026-09-19 재조회. WV3-s1 95–96, s2 108, s3 145–147, s4 75–77, s5 63–65. 이 파일은 조회 snapshot이며 현재 서버 프로세스 상태가 아니다.  
**[R5]** 첨부 `pancrafter.pdf`, PAN-Crafter Table1(p.6), WV3 H=.958/E=2.040. 이 benchmark를 넘는 것을 모든 최신 연구에 대한 SOTA 확정과 동일시하지 않는다.  
**[R6]** 첨부 `Pansharpening_research.pdf`, pp.8–10 Teacher/consistency, pp.11–15 e/q 구분·hard/soft·edge·Student A adjustment.

Sheet URL:
https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit

수치 source의 상세 기준은 R1에 기록된 `fh20r1/training.py` 및 FH12 import 구조의 감사다. **이번에는 새 code diff 전체를 재감사하지 않았으며, V01/V03이 그 실행 확인 항목이다.**


이번 export SHA256: `0dabfcc8f6df26558cfb13840e7a894d7fca93bf56f767b1a50b19d1e3fb301d`  
원 case CSV SHA256: `08afac3617b376b3bef32861aa1479a1c119f7ae6e1f0437002da0c39362ac52`  
Manifest 작성 UTC: `2026-09-19T08:21:27.356190+00:00`

## 부록 B. 서버별 실제 기존 run ID — 우선순위 overlay

아래 순서는 **현재 admitted block을 먼저 끝낸 뒤** 적용하는 선호 순서다. 같은 block 안의 순서는 원 registry에서 그대로 복사했다. `Sheet 완료`가 아니라고 자동 fresh 실행하지 않는다.

### s1

#### 01. FH20R1_S1_B01 · CORE · 원 순번 1

Snapshot: `REPORTED_COMPLETE` · 완료등록 2/2

```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1
```
- Sheet 완료: `FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1` — WV3-s1 95행. 원 자산 검증 후 재사용.
- Sheet 완료: `FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1` — WV3-s1 96행. 원 자산 검증 후 재사용.

#### 02. FH20R1_S1_B02 · CORE · 원 순번 2

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73102_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73102_BASE_FRESH50_v1
```

#### 03. FH20R1_S1_B03 · CORE · 원 순번 3

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73103_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73103_BASE_FRESH50_v1
```

#### 04. FH20R1_S1_B04 · CORE · 원 순번 4

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73104_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73104_BASE_FRESH50_v1
```

#### 05. FH20R1_S1_B05 · CORE · 원 순번 5

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73105_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73105_BASE_FRESH50_v1
```

#### 06. FH20R1_S1_B06 · CORE · 원 순번 6

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73106_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73106_BASE_FRESH50_v1
```

#### 07. FH20R1_S1_B07 · CORE · 원 순번 7

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73107_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73107_BASE_FRESH50_v1
```

#### 08. FH20R1_S1_B08 · RESERVE · 원 순번 8

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73108_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73108_BASE_FRESH50_v1
```

#### 09. FH20R1_S1_B09 · RESERVE · 원 순번 9

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73109_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73109_BASE_FRESH50_v1
```

#### 10. FH20R1_S1_B10 · RESERVE · 원 순번 10

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73110_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73110_BASE_FRESH50_v1
```

#### 11. FH20R1_S1_B11 · RESERVE · 원 순번 11

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73111_BASE_FRESH50_v1
FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73111_BASE_FRESH50_v1
```

### s2

#### 01. FH20R1_S2_B01 · CORE · 원 순번 1

Snapshot: `PARTIAL_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 1/2

```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS72001_BASE_FRESH50_v1
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS72001_BASE_FRESH50_v1
```
- Sheet 완료: `FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS72001_BASE_FRESH50_v1` — WV3-s2 108행. 원 자산 검증 후 재사용.

#### 02. FH20R1_S2_B02 · CORE · 원 순번 2

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73201_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73201_BASE_FRESH50_v1
```

#### 03. FH20R1_S2_B03 · CORE · 원 순번 3

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73202_BASE_FRESH50_v1
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73202_BASE_FRESH50_v1
```

#### 04. FH20R1_S2_B04 · CORE · 원 순번 4

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73203_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73203_BASE_FRESH50_v1
```

#### 05. FH20R1_S2_B05 · CORE · 원 순번 5

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73204_BASE_FRESH50_v1
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73204_BASE_FRESH50_v1
```

#### 06. FH20R1_S2_B06 · RESERVE · 원 순번 6

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73205_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73205_BASE_FRESH50_v1
```

#### 07. FH20R1_S2_B07 · RESERVE · 원 순번 7

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73206_BASE_FRESH50_v1
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73206_BASE_FRESH50_v1
```

#### 08. FH20R1_S2_B08 · RESERVE · 원 순번 8

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73207_BASE_FRESH50_v1
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73207_BASE_FRESH50_v1
```

#### 09. FH20R1_S2_B09 · RESERVE · 원 순번 9

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS73208_BASE_FRESH50_v1
FH20R1_S2_S_PLH_W104_D122_WV3_TF2_TS71001_SS73208_BASE_FRESH50_v1
```

### s3

#### 01. FH20R1_S3_B01 · CORE · 원 순번 1

Snapshot: `REPORTED_COMPLETE` · 완료등록 2/2

```text
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS72002_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS72002_BASE_FRESH50_v1
```
- Sheet 완료: `FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS72002_BASE_FRESH50_v1` — WV3-s3(5090) 145행. 원 자산 검증 후 재사용.
- Sheet 완료: `FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS72002_BASE_FRESH50_v1` — WV3-s3(5090) 146행. 원 자산 검증 후 재사용.

#### 02. FH20R1_S3_B02 · CORE · 원 순번 2

Snapshot: `PARTIAL_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 1/2

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73301_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73301_BASE_FRESH50_v1
```
- Sheet 완료: `FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73301_BASE_FRESH50_v1` — WV3-s3(5090) 147행. 원 자산 검증 후 재사용.

#### 03. FH20R1_S3_B06 · CORE · 원 순번 6

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W112_D122_WV3_TF3_TS71001_SS73305_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73305_BASE_FRESH50_v1
```

#### 04. FH20R1_S3_B03 · CORE · 원 순번 3

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73302_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73302_BASE_FRESH50_v1
```

#### 05. FH20R1_S3_B04 · CORE · 원 순번 4

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73303_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73303_BASE_FRESH50_v1
```

#### 06. FH20R1_S3_B05 · CORE · 원 순번 5

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73304_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73304_BASE_FRESH50_v1
```

#### 07. FH20R1_S3_B07 · CORE · 원 순번 7

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73306_BASE_FRESH50_v1
FH20R1_S3_S_PH_W112_D122_WV3_TF3_TS71001_SS73306_BASE_FRESH50_v1
```

#### 08. FH20R1_S3_B08 · CORE · 원 순번 8

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W112_D122_WV3_TF3_TS71001_SS73307_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73307_BASE_FRESH50_v1
```

#### 09. FH20R1_S3_B09 · CORE · 원 순번 9

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73308_BASE_FRESH50_v1
FH20R1_S3_S_PH_W112_D122_WV3_TF3_TS71001_SS73308_BASE_FRESH50_v1
```

#### 10. FH20R1_S3_B10 · CORE · 원 순번 10

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W112_D122_WV3_TF3_TS71001_SS73309_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73309_BASE_FRESH50_v1
```

#### 11. FH20R1_S3_B11 · RESERVE · 원 순번 11

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73310_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73310_BASE_FRESH50_v1
```

#### 12. FH20R1_S3_B12 · RESERVE · 원 순번 12

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73311_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73311_BASE_FRESH50_v1
```

#### 13. FH20R1_S3_B13 · RESERVE · 원 순번 13

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73312_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73312_BASE_FRESH50_v1
```

#### 14. FH20R1_S3_B14 · RESERVE · 원 순번 14

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S3_S_PH_W104_D122_WV3_TF3_TS71001_SS73313_BASE_FRESH50_v1
FH20R1_S3_S_PH_W104_D121_WV3_TF3_TS71001_SS73313_BASE_FRESH50_v1
```

### s4

#### 01. FH20R1_S4_B01 · CORE · 원 순번 1

Snapshot: `REPORTED_COMPLETE` · 완료등록 1/1

```text
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS72001_BASE_FRESH50_v1
```
- Sheet 완료: `FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS72001_BASE_FRESH50_v1` — WV3-s4 75행. 원 자산 검증 후 재사용.

#### 02. FH20R1_S4_B02 · CORE · 원 순번 2

Snapshot: `PARTIAL_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 2/4

```text
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73401_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73401_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73401_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73401_BASE_FRESH50_v1
```
- Sheet 완료: `FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73401_BASE_FRESH50_v1` — WV3-s4 76행. 원 자산 검증 후 재사용.
- Sheet 완료: `FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73401_BASE_FRESH50_v1` — WV3-s4 77행. 원 자산 검증 후 재사용.

#### 03. FH20R1_S4_B06 · CORE · 원 순번 6

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/1

```text
FH20R1_S4_S_PLH_W112_D122_WV3_TF4_TS71001_SS72001_BASE_FRESH50_v1
```

#### 04. FH20R1_S4_B03 · CORE · 원 순번 3

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/4

```text
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73402_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73402_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73402_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73402_BASE_FRESH50_v1
```

#### 05. FH20R1_S4_B07 · CORE · 원 순번 7

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S4_S_PLH_W112_D121_WV3_TF4_TS71001_SS73405_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W112_D122_WV3_TF4_TS71001_SS73405_BASE_FRESH50_v1
```

#### 06. FH20R1_S4_B04 · CORE · 원 순번 4

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/4

```text
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73403_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73403_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73403_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73403_BASE_FRESH50_v1
```

#### 07. FH20R1_S4_B05 · CORE · 원 순번 5

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/4

```text
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73404_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73404_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73404_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73404_BASE_FRESH50_v1
```

#### 08. FH20R1_S4_B08 · RESERVE · 원 순번 8

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/4

```text
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73406_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73406_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73406_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73406_BASE_FRESH50_v1
```

#### 09. FH20R1_S4_B09 · RESERVE · 원 순번 9

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/4

```text
FH20R1_S4_S_PLH_W104_D121_WV3_TF4_TS71001_SS73407_BASE_FRESH50_v1
FH20R1_S4_S_PLH_W104_D122_WV3_TF4_TS71001_SS73407_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D122_WV3_TF4_TS71001_SS73407_BASE_FRESH50_v1
FH20R1_S4_S_PL_W104_D121_WV3_TF4_TS71001_SS73407_BASE_FRESH50_v1
```

### s5

#### 01. FH20R1_S5_B01 · CORE · 원 순번 1

Snapshot: `REPORTED_COMPLETE` · 완료등록 2/2

```text
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1
```
- Sheet 완료: `FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1` — WV3-s5 63행. 원 자산 검증 후 재사용.
- Sheet 완료: `FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1` — WV3-s5 64행. 원 자산 검증 후 재사용.

#### 02. FH20R1_S5_B02 · CORE · 원 순번 2

Snapshot: `PARTIAL_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 1/4

```text
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73501_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73501_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73501_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73501_BASE_FRESH50_v1
```
- Sheet 완료: `FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73501_BASE_FRESH50_v1` — WV3-s5 65행. 원 자산 검증 후 재사용.

#### 03. FH20R1_S5_B06 · CORE · 원 순번 6

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/2

```text
FH20R1_S5_S_PLH_W112_D121_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W112_D122_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1
```

#### 04. FH20R1_S5_B03 · CORE · 원 순번 3

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/4

```text
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73502_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73502_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73502_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73502_BASE_FRESH50_v1
```

#### 05. FH20R1_S5_B04 · CORE · 원 순번 4

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/4

```text
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73503_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73503_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73503_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73503_BASE_FRESH50_v1
```

#### 06. FH20R1_S5_B05 · CORE · 원 순번 5

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/4

```text
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73504_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73504_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73504_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73504_BASE_FRESH50_v1
```

#### 07. FH20R1_S5_B07 · RESERVE · 원 순번 7

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/4

```text
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73505_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73505_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73505_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73505_BASE_FRESH50_v1
```

#### 08. FH20R1_S5_B08 · RESERVE · 원 순번 8

Snapshot: `NOT_REPORTED_NEEDS_LOCAL_CHECK` · 완료등록 0/4

```text
FH20R1_S5_S_PLH_W104_D121_WV3_TF5_TS71002_SS73506_BASE_FRESH50_v1
FH20R1_S5_S_PLH_W104_D122_WV3_TF5_TS71002_SS73506_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D122_WV3_TF5_TS71002_SS73506_BASE_FRESH50_v1
FH20R1_S5_S_PL_W104_D121_WV3_TF5_TS71002_SS73506_BASE_FRESH50_v1
```

---
**문서 종료. 실제 적용 전 V00 및 method 보존 점검이 필요하다. 현재 작성물로 서버 큐·학습·Sheet를 수정하지 않았다.**
