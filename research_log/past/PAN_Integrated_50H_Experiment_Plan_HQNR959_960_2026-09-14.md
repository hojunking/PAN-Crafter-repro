# PAN 통합 50시간 실험계획
## L1E4 Aligner 재사용·적응 × Q12/R1 Fitting — Raw HQNR 0.959–0.960

**작성일:** 2026-09-14  
**캠페인 제안 ID:** `PAKD50_W112D123_WV3_20260914_v1`  
**실행 환경:** s1 구현·검사·Git release, s2/s3 동일 release 수신·학습  
**시간 범위:** 캠페인 시작으로부터 **50시간 경과시간**, 세 서버 병렬 사용  
**주 지표:** WV3 FR `.mat` 20장 전체, 원본 PAN 참조 **raw HQNR**  
**목표:** raw HQNR **0.959 이상**, 확장 목표 **0.960 이상**  
**문서 상태:** 실행 설계. 실제 통합 학습·서버 배포·성능 검증을 수행한 기록이 아니다.  
**요약 문서:** [현재 접근 요약](PAN_Integrated_Method_Summary_2026-09-14.md)

> 이번 실험의 중심은 새로운 loss를 계속 추가하는 것이 아니라, **Teacher에서 배운 정합 상태를 Student가 어떤 감독과 시점으로 재사용·적응할 때 최종 복원이 가장 좋아지는가**이다. HQNR 목표 달성과 seed에 견디는 방법 개선을 별도로 판정한다.

---

## 문서 사용 순서

| 역할 | 우선 읽을 부분 |
|---|---|
| 연구 방향·목표 판단 | §1–3, §10–12 |
| s1 구현 | §4–6, §13–14 |
| 실행 큐 구성 | §7–9, §11 |
| 결과 해석·인계 | §10, §12, §15–16 |

본문의 `[A, §x]`와 `[K, §x]`는 마지막 절에 명시한 두 첨부 보고서를 뜻한다. **기존 명세/관측**, **이번 합의**, **신규 실험 제안**, **미확인 실행값**을 구분한다. 새 LR·freeze 시점·tail 길이·탐색 판정값은 기존에 검증된 최적값이 아니다.

# 1. 목표와 실험 범위

## 1.1 정확히 어떤 HQNR을 올리는가

장면별 지표를 먼저 계산하고 평균한다.

\[
H_i=(1-D_{\lambda,i})(1-D_{s,i}),\qquad
H_{\mathrm{raw}}=\frac{1}{20}\sum_{i=1}^{20}H_i.
\]

**원본 PAN/MS 참조, 전체 domain, 기존 evaluator·export 규약**을 유지한다. `(1−mean Dλ)(1−mean Ds)`로 대체하지 않는다. Aligner를 사용하는 모델도 raw 평가는 aligner를 끄는 것이 아니라, 정상 추론한 출력에 대해 **평가용 PAN 참조를 원본으로 유지**하는 것이다. [A, §6]

| 결과 필드 | 역할 | 목표 달성으로 인정하는가? |
|---|---|---|
| `hqnr_raw_original` | 공식 개발·선택 지표 | **예** |
| `hqnr_raw_v64` | 경계 제외 진단 | 아니오 |
| `hqnr_aligned_self_v64` | 자기 정합 참조 진단 | 아니오 |
| 임의 subset / 장면별 최고 모델 조합 | 이번 benchmark와 다른 결과 | 아니오 |
| TTA / ensemble / 출력 사후 이동 | 별도 추론 방법 | 이번 기본 목표에는 미포함 |

`0.959=95.9`, `0.960=96.0`으로 표시하되 내부 저장·판정은 원 정밀도로 한다. **0.95896을 반올림해 95.9라고 표시해도 목표 미달**이다. 0.960을 넘는 결과는 상한 초과로 배제하지 않는다.

## 1.2 현재 기록에서 목표까지의 간격

| 참고점 | 기존 raw HQNR | 0.959까지 | 0.960까지 |
|---|---:|---:|---:|
| L1E4 보고서 3-seed 평균 | 0.95597 | +0.00303 | +0.00403 |
| L1E4 seed2025 단일 best 기록 | 0.95698 | +0.00202 | +0.00302 |

단위 환산상 평균 기준으로 약 **0.303–0.403 percentage point**가 필요하다. [A, §5.3, §8.1에서 산술 계산]

Q12의 no-align s2 평균 이득 +0.00245를 0.95597에 단순히 더하면 0.95842다. **이는 규모를 가늠하는 산술 예시일 뿐 통합 성능 예측이 아니다.** Backbone·Teacher·정합·학습 경로가 달라 단독 이득의 가산성은 확인되지 않았다. 따라서 단순 부착 비교와 함께 routing·학습 시점·추가 fitting을 탐색한다. [K, §8.2]

## 1.3 성공을 세 수준으로 기록한다

| 수준 | 이번 캠페인의 운영 정의 | 허용되는 설명 |
|---|---|---|
| **Asset hit** | 적법한 단일 checkpoint의 재현된 raw HQNR ≥0.959; ≥0.960 별도 표시 | 목표를 달성한 단일 자산을 확보했다. |
| **Repeated gain** | 고정 방법의 3개 대응 block에서 평균 ΔHQNR>0, 2/3 이상 양성이고 중앙값도 양성 | 고정 Teacher 아래 반복 개선의 패턴이 있다. 통계적 유의성 확정은 아님. |
| **Method-level target** | 고정 방법의 3개 block 평균 raw HQNR≥0.959, 중앙값·worst·target-hit 수를 함께 보고 | 해당 고정 Teacher 및 실행 조건에서 평균 목표 달성. 모든 seed 보장은 아님. |

세 block 모두 ≥0.959이면 별도로 **3/3 target hit**를 명시한다. 평균만 달성했으면 그대로 평균 달성으로 쓴다. 좋은 seed 하나를 고르는 작업과 방법의 평균 개선을 혼동하지 않는다.

## 1.4 이번 50시간에 포함하고 제외할 것

**포함:** 기존 L1E4 A/U 자산의 재현·패키징, Q12/R1 이식, Student 초기화·정합 업데이트·loss routing 비교, 필요한 단일축 조정, 제한적인 10K 추가 fitting, seed 대응 분석과 최종 export.

**기본 제외:** N2 전체 재학습, 새 backbone 탐색, dense flow·geometry loss 재도입, GV/GC/SC·feature KD·uncertainty head 추가, online Teacher 업데이트, FR HQNR 직접 역전파, benchmark 참조 변경. 기존 자산이 사용 불가능할 때의 Teacher 재준비 비용은 별도로 재산정한다.

# 2. 두 보고서에서 계승할 사실과 주의점

## 2.1 앞단 기준

L1E4는 N2 aligner를 새 U-Net과 native fitting으로 공동 학습하면서 `λoff=1e-4`의 relative-offset consistency를 격번 유지한 방법이다. **Method fixed와 weights frozen은 다르다.** [A, §1.3, §4]

| 항목 | 계승 내용 |
|---|---|
| 검증 backbone | W112–D123, `depth=[1,2,3]` |
| 입력/출력 | PAN1 + MS8 = 9ch → residual8 + 이동하지 않은 MS base |
| Aligner | 기존 global `(dy,dx)` CNN, 현재 HR pixel 단위 |
| Aligner 입력 view | 공통 margin4, crop 후 band별 z-score |
| Sampler | 기존 bicubic / border / align_corners=False |
| Reconstruction | 모든 update의 native HRMS GT L1 |
| Offset | 홀수 optimizer update의 component-mean L1, native target만 SG |
| Jitter | 반경2 HR pixel, 원판 면적 uniform, PAN 복사본에만 적용 |
| 기본 학습 | U LR1e-4 / A LR1e-5, AdamW WD0.01, warmup100+cosine, batch48, 50K |

L1E4의 보고서 3-seed 평균 raw HQNR은 0.95597이고 report sd는 0.00087이다. Sheet 표시값에서 계산한 표본 sd 0.00090과 동일한 숫자로 취급하지 않는다. 기존 평균 P0 차이도 출처별 집계 차이가 있어 이번 KD 순증분의 분모로 섞지 않는다. [A, §8.1–8.3, §10]

## 2.2 뒷단 기준

| Backend | 기존 관측 | 이번 해석 |
|---|---|---|
| Q12 | no-align s2 N0 대비 +0.00245, 3/3; s3 −0.00034, 1/3 | 주력이나 통합 우위는 미검증 |
| R1 | no-align s3 +0.00297, 2/2; s2 −0.00078, 0/2 | 단순 대안으로 유지 |
| X02 | s2 Q12−X02 +0.00266, 3/3 | 통합 후에도 soft 추가 기여를 다시 확인할 대조 |

기존 뒷단은 W104–D122에서 시험했다. 이번 W112–D123으로 이식되는 것은 **목적함수와 감독 원리**이며, 기존 효과 크기를 그대로 옮기는 것이 아니다. s2/s3의 Teacher hash와 실행 revision도 달랐으므로 과거의 서버 차이를 seed 차이나 GPU 차이 하나로 설명하지 않는다. [K, §1.2, §8]

## 2.3 ERGAS 변동을 감안해야 하는 직접 근거

아래는 [A, §8.3]의 **동일 Sheet export 표에 있는 표시값**이다.

| L1E4 학습 seed | Raw HQNR@selected | ERGAS@selected | 같은 seed L000 ERGAS | L1E4−L000 |
|---:|---:|---:|---:|---:|
| 1234 | 0.9555 | 2.0373 | 2.0411 | −0.0038 |
| 7777 | 0.9554 | 2.1228 | 2.0586 | +0.0642 |
| 2025 | 0.9570 | 2.0694 | 2.0468 | +0.0226 |

L1E4 ERGAS 평균은 2.0765, n−1 표본 sd 약0.0432, 범위0.0855다. **계산된 참고 통계**이며 보편적인 seed noise floor가 아니다. 각 행의 선택 checkpoint가 다를 수 있고, 보고서의 다른 ERGAS 표와 평가값 불일치도 미해결이다. [A, §8.6, §10]

실행자는 한 seed에서 ERGAS가 조금 나빠졌다는 이유만으로 방법을 즉시 탈락시키지 않는다. 동시에 그 악화를 모두 seed 탓으로 처리하지도 않는다. 공통 평가기·checkpoint·late training 상태를 확인하고 paired 반복으로 판단한다.

# 3. 공통 모델·자산·좌표 계약

## 3.1 Teacher package T0

```text
run: PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1
checkpoint kind: best_raw
Teacher A: 같은 checkpoint의 최종 L1E4 aligner
Teacher U: 같은 checkpoint의 최종 L1E4 U-Net
```

**T0 선정은 이번 개발 기준의 고정 결정**이다. 이 seed의 높은 단일 결과를 보고서 전체의 대표 성능으로 쓰지 않는다. 파일이 확인되기 전에는 로컬 경로·selected update·hash를 만들어 채우지 않는다. [A, §5.3]

필수 확인값은 A/U state hash, 전체 checkpoint hash, selected optimizer update, backbone spec, 전처리·sampling·export/evaluator hash다. Teacher는 frozen/eval/no-grad이고 Student와 parameter/buffer 객체를 공유하지 않는다.

## 3.2 기본 Student 초기화

\[
A_S(0)=\operatorname{copy}(A_T^*),\qquad
F_S(0)=F_{\mathrm{init},s},\quad s\in\{1234,777,2026\}.
\]

각 seed의 U-Net 초기 state를 한 번 저장하고 대응 case들이 같은 파일을 로드한다. Teacher 구성·calibration 실행·진단 호출 때문에 초기화 RNG가 바뀌지 않게 한다. **A만 복사하며 Teacher optimizer·scheduler·U-Net weight는 기본 Student로 넘기지 않는다.**

Student stage의 optimizer와 50K schedule은 새로 시작한다. Teacher의 `best_raw` selected step이 몇인지와 무관하게 Student update는 0부터 센다. Teacher-copy/continuation은 별도 protocol ID를 사용한다.

## 3.3 두 forward와 stationary Teacher

\[
c_T=A_T^*(P,M),\quad Z_T=M+F_T^*([\mathcal W(P,c_T),M]),
\]
\[
c_S=A_S(P,M),\quad Z_S=M+F_S([\mathcal W(P,c_S),M]).
\]

Teacher는 동일하게 변환된 native P/M을 받는다. **입력 sample·augmentation이 바뀌어 Teacher 출력이 달라지는 것은 정상**이며, 고정 입력에 대한 Teacher package가 학습 중 바뀌지 않는다는 것이 stationary의 의미다.

F 정책에서는 Teacher/Student aligned tensor를 안전하게 재사용할 수 있지만, J/P/D에서는 Teacher의 frozen 입력 경로와 Student의 live 경로를 분리한다. 별도 Teacher feature나 cT를 Student U-Net 입력 channel로 추가하지 않는다. [A, §11.3–11.4]

## 3.4 절대 바꾸지 않을 인터페이스

PAN sampling은 `W(P,c)[y,x]=P[y+cy,x+cx]`이다. 출력 순서는 `(dy,dx)`, grid는 `(x,y)`다. z-score tensor는 aligner용이며 실제 U-Net의 intensity를 대체하지 않는다. MS/GT/base/output inverse warp, 복원 경로의 extra jitter, 이중 warp·이중 base 합산은 금지한다. [A, §2–3]

# 4. 목적함수와 loss 수신 모듈

## 4.1 N0 / R1 / Q12 / X02

\[
e_T=\operatorname{mean}_c|Z_T-Y|,\quad
e_S=\operatorname{mean}_c|Z_S-Y|,\quad
k=\operatorname{mean}_c|Z_S-\operatorname{sg}(Z_T)|.
\]

\[
d_T=\operatorname{sg}\!\left(\frac{e_T}{e_T+\tau_R}\right),\qquad
a_T=\operatorname{sg}\!\left(\operatorname{clip}_{[0,1]}
\frac{[e_S-e_T]_+}{e_S+10^{-6}}\right).
\]

\[
L_0=\langle e_S\rangle,\quad
L_D=\langle\alpha d_Te_S\rangle,\quad
L_K=\langle\beta(1-d_T)a_Tk\rangle,\quad
L_E^w=\lambda_E L_E.
\]

| Backend | Scalar objective | 기본 계수 |
|---|---|---|
| N0 | `L0` | α=0, β=0, edge OFF |
| R1 | `L0+LD` | α=1, β=0, edge OFF |
| Q12 | `L0+LD+LK+LEw` | α=1, β=0.1, edge ON |
| X02 | `L0+LD+LEw` | α=1, β=0, edge ON |

Loss는 전체 고정 batch·pixel support에 대해 평균한다. β는 LK 안에서 한 번만 적용한다. Q12의 α=0만으로 N0가 되지 않으며, β=0만으로도 N0가 되지 않는다. [K, §4–6]

## 4.2 GT signed-edge

\[
K_x=\frac1{32}
\begin{bmatrix}-3&0&3\\-10&0&10\\-3&0&3\end{bmatrix},\qquad K_y=K_x^\top,
\]
\[
L_E=\tfrac12\operatorname{mean}_{B,c,V_E}|K_x*Z_S-K_x*Y|
+\tfrac12\operatorname{mean}_{B,c,V_E}|K_y*Z_S-K_y*Y|.
\]

**기존 구현**의 reflect1 padding, 상하좌우1px 제외, FP32 계산, 밴드별 signed 비교를 그대로 사용한다. `stat.window=5`를 근거로 variance pooling을 만들지 않는다. Edge에는 dT/aT를 추가로 곱하지 않는다. [K, §5.2]

## 4.3 Student offset auxiliary

\[
P_\epsilon=\mathcal W(P,\epsilon),\quad
c_{S,\epsilon}=A_S(P_\epsilon,M),
\]
\[
L_{O}=\mathbf1[t\bmod2=1]10^{-4}
\operatorname{mean}_{B,2}|c_{S,\epsilon}+\epsilon-\operatorname{sg}(c_S)|.
\]

`t`는 zero-based **성공한 optimizer update**의 index다. Gradient accumulation의 microbatch index가 아니다. AMP overflow로 update가 건너뛰는 경우 counter/RNG/resume 정책은 성공 trainer의 계약을 확인해 고정한다. Native target은 Student의 현재 cS이며 Teacher cT로 바꾸지 않는다. [A, §4.2–4.4]

Frozen 구간에는 offset을 training total에 넣지 않고 진단용으로만 계산한다. 학습 중인 Student A의 native/jittered 호출은 같은 parameter를 공유한다.

## 4.4 Gradient routing의 일반식

Student U-Net에는 선택 backend 전체를 보낸다. Trainable Student aligner에는 다음을 보낸다.

\[
\nabla_{\phi_S}\left(L_0+q_D L_D+q_K L_K+q_E L_E^w+L_O\right).
\]

| 정책 | `(qD,qK,qE)` | 의미 |
|---|---|---|
| J | (1,1,1) | Backend 전체로 정합도 적응 |
| P | (0,0,0) | A는 plain GT+offset만, U는 Q12 전체 |
| JK0 | (1,0,1) | Output soft의 A 전달만 차단 |
| JE0 | (1,1,0) | GT edge의 A 전달만 차단 |
| F | 해당 없음 | Student A 전체 frozen |

R1에는 LK/LEw가 없어 J는 `L0+LD+LO`다. N0에는 LD/LK/LEw가 없어 J와 P가 같은 학습 규약이 된다. **PQ용 별도 P0를 새 학습할 필요는 없고 J0를 재사용**한다. 단, routing 구현의 수치 동치는 검사한다.

P/JK0/JE0는 parameter별로 다른 loss를 받으므로, 로그의 scalar total이 모든 parameter에 대한 단일 backward objective와 동일하다고 설명하지 않는다. `routing_policy`와 실제 수신 gradient를 함께 기록한다.

# 5. Calibration과 실행 선행조건

## 5.1 τR: T0의 오차 scale

\[
\tau_R=\max\left(\operatorname{median}_{\mathcal D_{cal},p}e_T(p),10^{-6}\right).
\]

기존 규약의 train 고정 subset 최대3072 patches, calibration seed1234, batch48, geometric augmentation/crop OFF를 사용한다. Patch ID 목록과 sampling order를 저장한다. 학습 RNG와 분리한다. T0의 A/U·intensity·MS 보간·view·sampler를 포함한 package로 새로 산출한다. [K, §7.1]

기존 no-align 값0.012367을 복사하지 않는다. T0가 고정되므로 Student A가 변해도 τR를 자동 업데이트하지 않는다. T1 package로 교체하는 민감도 실험은 새 τR를 계산한다.

## 5.2 λE: J0 pilot exact50K

**이번 공통 pilot 결정:** `J0 / Student seed1234 / exact50K`.

\[
\lambda_E^{(0)}=0.05\cdot\operatorname{median}_{b\in\mathcal D_{cal}}
\frac{\mathrm{RMS}(\partial L_0/\partial Z)}
{\mathrm{RMS}(\partial L_E/\partial Z)+10^{-30}}.
\]

출력 Z를 detach 후 requires-grad로 만들고 **출력 gradient**를 비교한다. Teacher hard 재가중·offset·parameter gradient를 분자에 넣지 않는다. [K, §7.2]

FQ/JQ/PQ/DQ, seed 반복, Q12↔X02에는 같은 λE package를 사용한다. F 전용으로 재보정한 최적 모델 비교가 아니라 **공통 계수 아래 정책 비교**다. 필요하면 최종 정책을 선택한 뒤 별도 calibration sensitivity로 구분한다.

Edge 강도0.5×/2× 실험은 같은 pilot ratio에서 λE를 곱해 변경한다. 재학습 결과로 다시 calibration해 두 변화가 섞이게 하지 않는다.

## 5.3 Calibration의 의존성과 서버 대기시간

T0 검증 후 τR는 즉시 계산할 수 있다. λE는 J0 seed1234 완주가 필요하므로 **J0-1234를 critical-path run**으로 우선 실행한다. s2/s3는 자기 block의 J0를 수행하고, 먼저 끝나면 λE가 필요 없는 F0 또는 JR을 진행할 수 있다.

시간이 촉박하다고 Q12를 임의 λE로 시작했다가 중간에 새 값으로 바꾸지 않는다. 기존 same-policy J0 exact50K 자산이 별도로 있다면 hash·입력·코드·초기화 조건을 검증한 뒤 재사용할 수 있다. 현재 그런 자산이 존재한다고 가정하지 않는다.

## 5.4 미확인 값은 시작 시 fail-fast

다음은 보고서만으로 확정되지 않았다: 실제 T0 파일·selected step, 완전한 optimizer beta/eps·exclusion, min LR, AMP/scaler·accumulation·clipping, 데이터·MS intensity 처리의 이식 동치, 실제 GPU slot 수·처리시간.

성공 run의 resolved config로 채운다. 라이브러리의 현 기본값을 대신 넣지 않는다. 코드 release가 통과해도 자산 hash·calibration·metric manifest가 미완성이면 정식 결과 run으로 승인하지 않는다.

# 6. 공통 학습·평가 preset

## 6.1 FRESH50: 독립 Student 50K

| 설정 | 기본값 / 처리 |
|---|---|
| Teacher | T0 A/U frozen; 모든 서버 동일 hash |
| Student | A만 T0 복사, U는 seed별 저장 초기 state |
| Backbone | W112–D123, 9ch→8ch residual |
| Updates / batch | 50,000 / 실효48 |
| U-Net LR | peak1e-4 |
| Trainable aligner LR | peak1e-5; 명시한 variant만 변경 |
| Optimizer | 성공 L1E4의 AdamW, WD0.01, 나머지 resolved 필드 계승 |
| Schedule | warmup100 + cosine; min LR·step 순서도 source에서 고정 |
| Offset | λ=1e-4, global odd update, disk radius2 |
| Q12 | α1, β0.1, λE=공통 calibration |
| Teacher/gate/GT | no-grad / detach / no-grad |
| Loss support | reconstruction full domain; signed edge 내부1px |
| Precision | 성공 source의 U-Net 정책; warp/off/edge 및 KD 수치 규약 보존 |
| 기본 추론 | 단일 Student A+U, no TTA, no ensemble, no 사후 보정 |

Warmup·cosine은 **처음부터 50K horizon**이다. 25K screening을 위해 25K cosine으로 학습한 뒤 50K로 연장하지 않는다. 10K/25K는 중간 관측 시점이지 별도 horizon이 아니다.

## 6.2 이번 campaign의 공통 checkpoint grid

**신규 캠페인 결정:** `GRID1K_50K_v1 = {1000,2000,…,50000}`의 50개 candidate를 모든 FRESH50 run에 적용한다. 성공 run의 evaluator와 exact comparator는 유지하고, 후보 update 목록은 이번 비교용으로 명시적으로 통일한다.

이것은 과거 run의 저장 candidate가 같았다고 가정하는 조치가 아니다. 과거 공식 best는 보존하며, 새 통합의 순증분은 이 grid의 **새 paired N0**로 계산한다. 다른 grid를 사용해야 할 실행상 이유가 생기면 정식 run 시작 전에 전체 cohort의 protocol ID를 바꾸고, 일부 후보만 촘촘히 평가하지 않는다. [A, §6.3의 공통 선택 기회 원칙을 적용]

| 저장·집계 | 정의 |
|---|---|
| `best_raw` | 기존 raw HQNR running-max/tie comparator, tie band1e-4 → FR fSCC |
| `last50k` | 정확한 50,000번째 optimizer update 직후 |
| `plateau45_50` | 45K/46K/47K/48K/49K/50K, **6개 고정 시점**의 평균 |
| `hqnr_run_max_value` | 모든 공식 candidate의 raw HQNR 숫자상 최대, 선택 checkpoint와 따로 보존 |
| `best_rr_val` | 진짜 독립 RR validation split이 확인된 경우에만 보조 저장 |

Tie comparator가 최대 HQNR보다 조금 낮은 candidate를 선택할 수 있으므로, **최고 수치와 공식 선택 자산을 분리 저장**한다. Target 판정은 선택 자산에 우선 적용한다. 숫자상 최대만 threshold를 넘으면 `max-only hit`로 별도 표시하고 comparator를 바꾸지 않는다.

선택된 best가 어느 update인지, 그 checkpoint의 ERGAS·SAM·PSNR·Dλ·Ds·fSCC·cS 진단을 함께 묶는다. 여러 checkpoint에서 좋은 열만 뽑아 한 행으로 만들지 않는다.

## 6.3 실행량이 늘어도 seed 대응을 유지하는 방법

기본 block은 `s1:1234`, `s2:777`, `s3:2026`이다. 각 서버에서 J0↔JQ, F0↔FQ, 필요한 변형을 같은 초기 state와 데이터 순서로 비교한다. 다른 seed를 같은 서버에서 실행해도 되지만 `block_id=(server,student_seed,release,teacher_package)`를 명확히 남긴다.

진단·calibration·Teacher forward가 train loader나 augmentation RNG를 소모하지 않게 한다. Frozen case에서 epsilon을 생성하지 않아도 data RNG 순서가 달라지지 않도록 generator를 분리한다. GPU 간 학습의 bitwise 동일성을 약속하는 설계는 아니다.

# 7. 실험 case 상세 — 기본 결합과 정합 적응

> 아래 목록은 **우선순위가 있는 case registry**다. 전 case×3seed를 50시간 안에 모두 수행하라는 뜻이 아니다. §9의 큐·§11의 예산 승인 규칙으로 실행 대상을 고른다. 새로운 조건은 실행 전에 config를 고정하고, 결과를 보고 실행 중 loss/LR를 바꾸지 않는다.

## 7.1 핵심 6개 case

모두 FRESH50, T0, 기본 calibration을 사용한다. A는 완성 Teacher의 aligner에서 시작한다.

| ID | Student A | U-Net 목적함수 | A 목적함수 | 필수 비교 | 우선순위 |
|---|---|---|---|---|---|
| **J0** | 즉시 trainable | L0 | L0+LO | 모든 joint backend의 baseline | 필수 |
| **JQ** | 즉시 trainable | Q12 | Q12+LO | **JQ−J0**, JQ−FQ | **주력·필수** |
| **F0** | 전체 frozen | L0 | 없음 | Frozen baseline | 다음 우선 |
| **FQ** | 전체 frozen | Q12 | 없음 | **FQ−F0**, FQ−JQ | 다음 우선 |
| **JR** | 즉시 trainable | R1 | R1+LO | JR−J0, JQ−JR | 단순 대안 |
| **FR** | 전체 frozen | R1 | 없음 | FR−F0, FQ−FR | Frozen 유력 시 우선 |

### J0 — Teacher-trained aligner를 새 Student와 재적응시키는 기준

**질문:** 최종 L1E4 aligner를 새 U-Net에 연결해 native GT만 학습해도 어느 수준까지 가는가?

J0는 원래 N2 aligner에서 출발한 L1E4 재학습과 다른 초기화다. `Teacher-final-A → fresh-U`라는 이번 Student protocol의 baseline으로 명명한다. Teacher는 bin 진단에만 사용할 수 있으며 학습 loss에는 사용하지 않는다.

J0-1234는 λE pilot을 겸하므로 우선 완주한다. J0의 3개 block은 단일 variant의 성능이 좋거나 나쁜 것과 무관하게 확보하는 것이 원칙이다.

### JQ — 전체 결합의 주력

**가설:** Teacher의 실패 지도·유효한 output soft·GT edge가 복원망뿐 아니라 PAN을 읽는 위치도 유리하게 조정한다.

변경은 J0의 backend를 Q12로 교체하는 것뿐이다. 별도 plain L1을 다시 더하지 않는다. JQ가 J0보다 좋아지면 전체 backend 묶음의 추가 효과이며, output soft 단독 효과는 X02로 확인한다.

**주요 읽을 값:** raw ΔHQNR, Dλ/Ds, ERGAS@last 및 plateau, A의 LD/LK/LE gradient norm·cosine, cS−cT drift. cS가 cT에서 벗어나는 것 자체를 실패로 보지 않는다.

### F0/FQ — 완성 정합을 그대로 재사용하는 기준

**가설:** Stage1에서 얻은 정합 상태가 충분히 좋다면, Student 학습 중 앞단을 바꾸지 않는 편이 안정적일 수 있다.

F0/FQ의 A state는 optimizer update·weight decay·buffer update를 포함해 불변이어야 한다. Off loss는 학습 효과가 없으므로 total에서 제거한다. 동일 aligned PAN을 T/S에 공유해도 되며 기능적으로 같음을 검사한다.

FQ−F0는 frozen 조건의 backend 이득이다. JQ−FQ만으로 KD의 효용을 말하지 않고, **ΔQ_joint=JQ−J0와 ΔQ_frozen=FQ−F0를 함께 비교**한다.

### JR/FR — R1을 실제 경쟁 후보로 둔다

**가설:** 직접 Teacher 모방과 edge가 없어도 실패 지도 재가중만으로 충분한 fitting 이득을 얻을 수 있다.

Edge·soft를 완전히 끈다. R1의 Teacher output은 eT를 만드는 데 쓰이지만, regression target은 항상 GT다. JR/FR이 Q12보다 좋으면 그 결과를 보존하고, 사전에 정한 distillation 명칭 때문에 불리한 Q12를 유지하지 않는다.

JR−JQ 또는 FR−FQ는 **soft와 edge가 동시에 달라진 비교**이므로 두 항 중 무엇이 원인인지 단독 귀속하지 않는다.

## 7.2 PQ / PR — Protected adaptation

| 항목 | PQ | PR: R1 유력 시만 |
|---|---|---|
| 초기화 / horizon | FRESH50 | FRESH50 |
| U 목적함수 | Q12 전체 | R1 |
| A 목적함수 | **L0+LO** | **L0+LO** |
| qD/qK/qE | 0/0/0 | LD→A 차단 |
| 대응 no-KD | **J0 재사용** | **J0 재사용** |
| 핵심 비교 | PQ−JQ, PQ−FQ, PQ−J0 | PR−JR, PR−J0 |

**실행 trigger:** JQ의 복원 U fitting은 개선되지만 raw HQNR이 불리하거나, extra loss의 A gradient가 native reconstruction과 반복 충돌하는 경우. 또는 JQ와 FQ가 비슷해 앞단 업데이트의 역할이 불명확한 경우.

**판독:** PQ>JQ이면 Q12 추가 항의 A 전달을 제한하는 정책이 유리한 패턴이다. 이 비교만으로 soft·edge·hard 재가중 중 특정 항을 원인으로 확정하지 않는다.

P0를 별도 full run으로 중복 생성하지 않는다. PQ의 LD/LK/LE가 A에는 없고 U에는 존재한다는 수신자 test는 필수다. 추가 autograd 비용은 실측해 예산에 반영한다.

## 7.3 JK0 / JE0 — 원인을 하나씩 좁히는 routing

| ID | A 목적함수 | U 목적함수 | JQ 대비 단 하나의 변경 |
|---|---|---|---|
| **JK0** | L0+LD+LEw+LO | Q12 | LK→A 차단 |
| **JE0** | L0+LD+LK+LO | Q12 | LEw→A 차단 |

둘 다 FRESH50, 기본 LR/계수/초기값을 유지하고 J0를 no-KD 기준으로 쓴다. PQ가 유리하거나 gradient 진단에서 명확한 경로 후보가 나타날 때 **둘 중 유력한 하나부터** 시험한다.

JK0가 좋아져도 U에 output soft가 필요 없다는 뜻은 아니다. JE0가 좋아져도 GT edge 자체가 불필요하다는 뜻은 아니다. 차이는 aligner 수신 경로에 한정한다.

## 7.4 D0 / DQ / DR — 첫5K 정합 고정 후 해제

| ID | Update 0–4999 | Update 5000–49999 | U 목적함수 |
|---|---|---|---|
| **D0** | A frozen | A receives L0+LO | L0 |
| **DQ** | A frozen | A receives Q12+LO | 처음부터 Q12 |
| **DR** | A frozen | A receives R1+LO | 처음부터 R1 |

**가설:** 학습된 aligner를 독립 초기화 U-Net에 연결했을 때 초기 부정확한 복원 gradient가 정합 상태를 불필요하게 바꾸는 문제를 줄인다. 5K는 이번에 제안하는 값이며 기존 결과가 아니다.

**정확한 schedule:** U의 50K scheduler는 처음부터 연속 진행한다. A를 해제할 때 U optimizer/scheduler를 reset하지 않는다. A는 global scheduler의 당시 multiplier를 적용한 `1e-5×g(5000)`으로 시작한다. A frozen 기간에는 optimizer moment/weight가 바뀌지 않아야 하며, A의 첫 실제 optimizer update부터 moment step을 센다. Off는 global parity를 유지하므로 해제 후 첫 odd update는5001이다.

**필수 대조:** DQ−D0, DQ−JQ, D0−J0. DQ만 좋아졌다고 KD 이득이라고 하지 않는다. D0/DQ는 **두 개의 신규 run**이며 seed를 늘릴 때 대조도 함께 늘린다.

DR은 R1이 유력할 때 DQ 대신 또는 추가로 실행한다. DQ와 DR 둘 다 처음부터 전 seed로 펼치지 않는다.

## 7.5 AL0 / ALQ — Aligner LR만 0.3배

| 항목 | AL0 | ALQ |
|---|---|---|
| A 정책 | 즉시 trainable | 즉시 trainable, joint |
| Peak A LR | **3e-6** | **3e-6** |
| Peak U LR | 1e-4 | 1e-4 |
| Backend | N0 | Q12 |
| Offset / jitter | 1e-4 / radius2 유지 | 동일 |

**질문:** Aligner가 학습되어야 하지만 업데이트 폭이 너무 큰가? JQ가 후반 drift와 성능 하락을 보일 때 유력하다.

ALQ−JQ는 Q12 아래 LR 정책 비교, ALQ−AL0는 같은 작은 A LR 아래 backend 이득이다. DQ와 ALQ를 처음부터 동시에 바꾸지 않는다. Delay+smallLR 결합은 두 단독 결과를 본 뒤 새 ID로만 허용한다.

## 7.6 LF0 / LFQ — 25K 이후 aligner 고정

| ID | 0–24999 | 25000–49999 | U schedule |
|---|---|---|---|
| **LF0** | J0와 동일 | A frozen, Off OFF | 50K schedule 연속 |
| **LFQ** | JQ와 동일 | A frozen, Off OFF; U는 Q12 계속 | 50K schedule 연속 |

**가설:** 앞부분의 공동 적응은 유용하지만 후반의 A drift가 좋은 raw 균형을 무너뜨린다.

사전 정의한 exact25K parent를 이용해 후반25K만 분기할 수 있다. A/U/optimizer/scheduler/train RNG를 정확히 이어 받고 **A 업데이트만 중단**한다. Late-freeze를 하기 위해 parent를 raw best로 바꾸지 않는다.

LF0−J0, LFQ−JQ 및 LFQ−LF0를 함께 본다. 공통 prefix를 재사용한 두 tail은 별개의 독립 초기화 run이 아니다. 비용은 실제 추가25K로 계산하고 전체 trajectory는 50K로 표시한다.

# 8. 실험 case 상세 — Q12 조정·제거 대조·추가 fitting

## 8.1 Q12 계수 조정: 한 번에 한 축만

정합 정책은 우선 `p*`로 고정한다. p*는 J/F/P/D 등에서 현재 가장 유력한 하나이며, **variant를 생성하기 전에 명시**한다. 아래 값은 모두 이번 신규 제안이다.

| ID suffix | 기본 Q12에서 변경 | 유지할 것 | 실행할 근거 |
|---|---|---|---|
| **QA05** | α:1→0.5 | β0.1, λE, A 정책 | Teacher 고오차 bin 과강조와 반복 RR/분광 비용이 관측됨 |
| **QB005** | β:0.1→0.05 | α1, λE, gate 정의 | Soft gradient 충돌·Teacher bias 모방이 의심됨 |
| **QB02** | β:0.1→0.2 | α1, λE, gate 정의 | Soft가 유효한 방향인데 기여가 약하고 Q12>X02 근거가 있음 |
| **QE025** | λE:λE0→0.5λE0 | α1, β0.1 | Edge 추가에 따른 RR/FR trade-off가 큼 |
| **QE10** | λE:λE0→2λE0 | α1, β0.1 | GT 경계 오차가 크고 edge gradient가 유효하며 ringing 증가가 없음 |
| **QSF** | β는30K까지0.1, 30–45K 선형 감소, 45K 이후0 | α1, λE | 초기 soft 이득은 있으나 후반에는 Teacher 참조가 제약일 가능성 |

실제 run ID에는 정책을 포함한다. 예: `J_QE025`, `P_QB02`. `QE025/QE10`은 공통 pilot 기준 출력 gradient calibration 목표0.025/0.10에 해당하는 **계수 배율**이며, 실제 parameter gradient가 그 비율로 유지된다는 뜻은 아니다.

### β를 키울 때의 제한

후반 soft/hard loss ratio가0.052%였다는 과거 관측만으로 β를 키우지 않는다. 현재 통합 run의 coefficient/loss/parameter-gradient 비율과 방향, Q12↔X02 결과를 먼저 본다. [K, §7.4, §8.5]

`aT=0`인 위치는 β를 키워도 soft0이다. β 변경 시 active set과 reduction은 그대로 두고, soft weight 합으로 나누거나 advantage gate를 동시에 제거하지 않는다.

### QSF의 정의

\[
\beta(t)=
\begin{cases}
0.1,&t<30000,\\
0.1(45000-t)/15000,&30000\le t<45000,\\
0,&t\ge45000.
\end{cases}
\]

QSF는 마지막 구간의 backend가 X02가 되는 schedule이다. GT hard 재가중과 edge는 계속 남는다. 기본 Q12와 같은 parent exact30K에서 분기하면 추가20K만 필요하지만, 전체 경로는 처음부터 정의된 FRESH50 schedule로 기록한다.

**탐색 상한:** 첫 scalar 라운드는 위6개 중 로그로 근거가 있는 **최대2개**만 고른다. 좋은 결과의 계수를 다시 여러 축으로 조합하는 것보다 seed 확인을 우선한다.

## 8.2 X02-p* — Output soft의 순증분

선택한 정합 정책·초기화·LR·calibration·α·edge 계수를 그대로 두고 **β만0**으로 한다. 기본 Q12가 아니라 β0.2나 edge0.5×가 최종 후보라면 그 후보와 정확히 대응하는 X02를 만든다.

| 비교 | 구분하는 효과 |
|---|---|
| Q12-p* − X02-p* | 같은 hard/edge 아래 output imitation 추가 |
| X02-p* − R1-p* | 같은 Teacher-error hard 아래 GT edge 추가 |
| R1-p* − N0-p* | Teacher 실패 지도의 GT 재가중 |

J형에서는 soft 제거가 U와 A 양쪽에 적용되고, P형에서는 원래 A가 soft를 받지 않으므로 U 쪽 차이만 남는다. 정합 경로 차이를 숨기지 않는다.

X02는 최소1개 대응 seed, 최종 output-soft 기여를 주된 주장으로 남길 때는 **3개 대응 block을 우선 확보**한다. 시간이 부족해1seed면 그것을3seed 검증처럼 서술하지 않는다.

## 8.3 TC0 / TCQ / TCR — Teacher A/U를 모두 복사한 10K 추가 fitting

**성능 접근용 별도 protocol:** `TCOPY10`.

| 항목 | 정의 |
|---|---|
| Student 초기 A/U | T0의 같은 best_raw A/U 전체 복사 |
| Teacher | 원래 T0 frozen 유지 |
| 추가 update | **10,000**, total Student tail horizon10K |
| U peak LR | **3e-5** |
| A peak LR | trainable이면 **3e-6**, frozen이면 업데이트 없음 |
| Schedule | 새 AdamW, warmup100 + cosine10K; min/peak 비율은 baseline 규약 계승 |
| A 정책 | 첫 copy 묶음은 J 또는 F 중 하나로 고정; loss별 routing은 해당 정책을 따름 |
| Backend | TC0=N0, TCQ=Q12, TCR=R1 |
| Calibration | T0의 τR, 공통 λE 유지; 복사 초기값으로 몰래 재보정하지 않음 |
| Candidate grid | 추가500,1000,…,10000; t=0은 reference로 따로 기록 |
| Plateau | 추가8K/8.5K/9K/9.5K/10K 평균 |

**가설:** 이미 학습된 A/U의 복원 능력을 유지하면서 Teacher-informed fitting으로 목표 구간에 접근할 수 있다. 10K 및 LR는 이번 제안이며 보장된 향상값이 아니다.

처음 Student와 Teacher가 같은 출력을 내면 aT=0이며 soft0인 것이 정상이다. Hard와 edge는 계속 작동할 수 있다. U-Net 전체를 복사했기 때문에 새 seed는 **data/augmentation/epsilon 실행 seed**이고 독립 weight initialization seed가 아니다. [K, §4.3의 정의에서 유도]

**필수 대조:** TCQ−TC0 또는 TCR−TC0, 같은 parent·A 정책·optimizer reset·tail schedule·batch order. 원 Teacher보다 좋아졌다는 사실만으로 KD 성공이라고 하지 않는다.

처음에는 TC0/TCQ 한 대응 seed를 사용한다. R1이 더 유력하면 TCQ 대신 TCR로 지정한다. Frozen↔joint와 backend를 한 번에 바꾼 두 run을 직접 KD 효과로 비교하지 않는다.

**Target hit 해석:** TCOPY가 목표를 넘으면 “Teacher-copy 추가 fitting protocol의 결과”로 기록한다. Fresh Student의 50K 결과와 한 family의 seed 평균으로 합치지 않는다.

## 8.4 EX0 / EXM — 완주 Student에서 10K continuation

**Protocol:** `CONT10`, parent는 선택한 family의 **exact50K**다. Raw best와 last를 임의로 골라 parent 후보 수를 늘리지 않는다.

| 항목 | 정의 |
|---|---|
| Parent | 같은 Student A/U exact50K, hash 고정 |
| Teacher | 해당 parent 학습에 사용한 T0 frozen |
| 추가 update | 10K |
| U peak LR | **1e-5** |
| A peak LR | trainable policy면 **1e-6**, frozen이면0 |
| Optimizer | 두 tail 모두 새 AdamW; parent moment를 한쪽에만 계승하지 않음 |
| Schedule | 새 warmup100+cosine10K, 두 tail 동일 |
| EX0 | 같은 parent에서 plain GT L1(+trainable이면 offset) |
| EXM | 같은 parent에서 parent의 선택 backend·routing(+offset) 유지 |
| 평가 | TCOPY10과 같은 tail grid, parent t=0 성능 병기 |

EXM은 Q12/R1 및 variant 이름을 resolved config에 명시한다. **Tail에서 backend를 유지하는 것의 순증분**을 EX0와 비교한다. Parent가 이미 KD로 학습됐으면 EX0도 KD 계보를 가진다. EX0를 전체 pipeline의 KD-free baseline이라고 부르지 않는다.

CONT10은 50K cosine을 그대로 더 돌리는 단순 resume가 아니라 **명시적 저학습률 추가 fitting**이다. Fresh50/TCOPY10/CONT10을 구분하고 누적 비용을 기록한다.

## 8.5 T1 / 추가 seed / 서버 교차 — 최종 후보 검증

| 검증 ID | 구성 | 무엇을 검증하는가? |
|---|---|---|
| **T1-N0 / T1-WIN** | L1E4 seed1234의 같은 best_raw A/U package로 교체; A source도 함께 변경 | 특정 T0 package에만 의존하는가? |
| **S3407-N0 / S3407-WIN** | T0 유지, 새 Student seed3407; 같은 초기값·data order 대응 | 기존3seed 선택에 대한 추가 확인 |
| **XSRV-N0 / XSRV-WIN** | 이미 수행한 seed의 대응쌍을 다른 서버에서도 실행 | 환경 차이와 method×환경 민감도 |

T1의 τR는 다시 계산한다. λE는 GT-only 공통 pilot 기반 계수를 그대로 유지해 package 민감도를 보는 것으로 우선 정의한다. T1 전용 pilot 재보정까지 하면 추가 변화이므로 별도 ID와 비용을 부여한다. T1에서 바뀌는 것은 Teacher U뿐 아니라 Teacher/Student 초기 A package이므로 “Teacher U-Net seed 효과만”이라고 하지 않는다.

S3407은 이번 문서에서 미리 정한 추가 확인 seed다. 이미3seed에서 선택한 방법이기 때문에 개발에 쓰지 않은 확인 seed라는 역할은 가능하지만, 같은 FR20을 계속 평가하므로 완전한 독립 benchmark 검증이라고 부르지 않는다.

같은 seed를 다른 서버에서 반복한 결과는 새로운 Student seed가 아니다. 최종 표는 block와 seed ID를 별도로 보관한다.

# 9. 50시간 실행 순서와 서버 배치

## 9.1 50시간의 해석

본 계획은 **wall-clock 50시간 동안 s1/s2/s3를 병렬 사용**하는 것으로 해석한다. 서버당 독립 GPU slot1개라면 gross150 GPU-hour다. GPU 종류·개수·현재 가용 메모리는 보고서만으로 확정하지 않는다.

기본 운영상 0–4h는 구현·동치 검사·배포, 46–50h는 최종 평가·복구·집계에 남긴다. 이 가정에서 4–46h의 학습·정기평가 slot 상한은 **126 GPU-hour**다. 실제로 구현이 더 걸리면 같은50h 종료시각을 유지하면서 case 수를 줄인다.

사용자의 예산이 추후 **세 서버 합산50 GPU-hour**로 확정되면 150으로 확대하지 않는다. §11.5의 축소안을 적용하고 ledger의 합계가50을 넘지 않게 한다.

## 9.2 기본 block와 release 역할

| 서버 | 기본 Student seed | 담당 역할 | 공통 원칙 |
|---|---:|---|---|
| s1 | 1234 | 구현·golden test·release, J0 calibration pilot, 대응 학습 | 실행은 고정 commit worktree |
| s2 | 777 | 동일 package 실행, 후보 반복 | s1 release를 받되 서버 전용 method 변경 금지 |
| s3 | 2026 | 동일 package 실행, 후보 반복 | 같은 Teacher·evaluator·calibration hash |

이 배치는 같은 seed의 N0↔후보를 같은 환경에서 비교하기 위한 것이다. 각 서버에 방법 하나씩만 맡겨 “J=s1, F=s2, R1=s3”로 만드는 배치는 하지 않는다.

정식 hyperparameter/정합 정책은 모든 서버의 공통 config에 들어간다. Local overlay에는 파일 경로·GPU 번호·worker 수 등 실행 환경만 허용하며, 수치에 영향을 줄 수 있는 worker/precision 차이도 manifest에 남긴다.

## 9.3 시간대별 운영 gate

| 경과시간 | 해야 할 일 | 다음 단계 통과 조건 |
|---|---|---|
| **0–4h** | T0 package·기존 지표 재현, F/J 코드·loss ownership·RNG/gradient test, release C0 배포, 속도 측정 | Teacher 고정·KD OFF 동치·원 PAN 평가·source hash 확인 |
| **4–12h** | J0 3block, J0-1234 exact50K 후 λE 생성; JQ 시작 | 공통 calibration checksum, 각 run ETA |
| **12–24h** | JQ 확인, F0/FQ, JR 중심 비교 | 적어도 J0↔JQ의 대응 결과와 frozen/R1 유력성 판단 |
| **24–34h** | PQ 또는 D0/DQ, 필요한 단일축, TCOPY10 pilot | 추가 case는 완료+반복+재평가 예산이 남는 경우에만 승인 |
| **34–36h** | 유력 family1개, 보조 family 최대1개로 설정 고정 | 추가 seed·대조의 remaining ETA가46h 이내 |
| **36–46h** | 누락된 paired 반복, X02, 필요 시10K tail/추가 확인 | 새로운 광범위 방법 탐색 금지, 미완료 core 완주 우선 |
| **46–50h** | selected/last 재평가·동일 checkpoint export·seed 집계·artifact/hash 검증 | 최종 표·미완료/실패·provenance까지 인계 |

시간대는 운영 목표이지, 처리속도를 확인하지 않은 완료 보장이 아니다. J0 pilot이 늦으면 JQ를 임의 calibration으로 시작하지 말고 F0/JR로 가용 slot을 사용한다.

## 9.4 기본 큐 우선순위

**P0 — 반드시 먼저 확보:** 세 block의 J0/JQ, 총6run. 중간 ERGAS 하나만으로 core를 종료하지 않는다.

**P1 — 결합 비교:** F0/FQ 총6run, JR 최대3run. FR 최대3run은 frozen이 유력하거나 예산이 충분하면 수행한다. 여섯 기본 case 전부3block이면18run이다.

**P2 — 목표 접근:** PQ 또는 D0/DQ를 우선한다. 명확한 drift라면 AL0/ALQ나 LF0/LFQ가 대체할 수 있다. Q12 scalar는 진단을 근거로 최대2개. TCOPY10은 목표 접근용 작은 대응 묶음으로 병행 가능하다.

**P3 — 반복·귀속:** 선택한 방법의 빠진 block와 그 대조를 완주하고 X02를 확보한다. 새 candidate 하나를 더 시도하는 것보다 **현재 candidate가 다른 seed에서도 유지되는지**를 우선한다.

**P4 — 추가 확인:** T1 package, seed3407, 같은-seed 서버 교차. 순서는 관측된 위험에 따라 선택하되 이름과 해석을 구분한다.

## 9.5 설명용 실행표 — 모든 FRESH50이4시간일 때

아래는 **가정값을 둔 실행 가능한 예시**다. 실제 측정치가 아니다. PQ도4h, TCOPY10 두 tail 합계2h 이내라고 가정하고, DQ가 유력 후보로 선택되는 분기를 보여 준다. 정기 FR/RR 평가·저장 시간은 각4h에 포함한다. λE 산출·전달의 추가 시간은4–8h pilot slot 또는44–46h 여유에 포함 가능한 경우를 가정하며, 실제 산출이 늦으면 JQ 시작을 그만큼 뒤로 이동한다.

| 시간 | s1 / seed1234 | s2 / seed777 | s3 / seed2026 |
|---|---|---|---|
| 0–4 | 구현·test·release | package·환경 검사 | package·환경 검사 |
| 4–8 | **J0 + λE pilot** | J0 | J0 |
| 8–12 | JQ | JQ | JQ |
| 12–16 | F0 | F0 | F0 |
| 16–20 | FQ | FQ | FQ |
| 20–24 | JR | JR | JR |
| 24–28 | **D0 pilot** | **PQ pilot** | FR |
| 28–32 | **DQ pilot** | FR | TC0/TCQ pilot + 진단 |
| 32–36 | X02-D / seed1234 | **D0 확인** | **D0 확인** |
| 36–40 | FR / seed1234 | **DQ 확인** | **DQ 확인** |
| 40–44 | 재평가·pair 분석·필요한 tail | X02-D / seed777 | X02-D / seed2026 |
| 44–46 | 지연·재시도 여유 | 지연·재시도 여유 | 지연·재시도 여유 |
| 46–50 | 최종 export·집계 | 결과 artifact 검사 | 결과 artifact 검사 |

이 예시는18개 기본 run, D0/DQ6run, PQ1run, X02-D3run 및 TCOPY10 한 대응 묶음으로 구성된다. DQ pilot은 seed1234, PQ pilot은 seed777이므로 **두 단일 pilot의 절대 HQNR로 방법 우위를 확정하지 않는다.** 각자의 paired N0, 진단, 남은 확인 비용으로 우선순위를 정하고 최종 비교는3block에서 한다.

PQ가 선택되면 D 확인 slot을 PQ의 빠진 seed1234/2026 및 X02-P로 바꾼다. JQ/FQ/JR이 그대로 최선이면 불필요한 D 반복을 강제하지 않고 X02·TCOPY 확인에 배분한다.

이 예시의 core 순서는 시간을 활용하기 위해 FR을 뒤로 미뤘다. **18core를 모두 끝낸 다음 새 방법을 시작해야 한다는 제약은 없다.** 반대로 한 seed pilot만 보고 나머지 core를 근거 없이 모두 취소하지도 않는다.

## 9.6 속도가 다를 때의 조정

s1의 J0 pilot이 가장 늦으면 다른 서버는 τR만 필요한 JR/F0를 진행한다. JQ가 가능한 시점에 기존 run을 무리하게 중단하지 않고 다음 slot부터 넣는다.

특정 서버가 느리면 paired 묶음 전체를 다른 slot으로 옮기는 것을 우선한다. 같은 method의 candidate만 빠른 서버로 이동시키고 baseline은 다른 환경 숫자를 그대로 붙이지 않는다. 불가피하게 이관하면 block을 새로 정의하고 최소한 같은-seed baseline 재현을 함께 고려한다.

새 method의 완주 예상시간만 보지 말고, **그 method의 부족한2seed와 대응 control까지46h 전에 끝나는지**로 시작 여부를 결정한다.

# 10. Seed-aware 결과 해석과 ERGAS 정책

## 10.1 네 종류의 변동을 구분한다

| 변동 축 | 이번 대응 | 주의 |
|---|---|---|
| Student 초기화·데이터 순서 | 기본3seed의 paired N0/후보 | 한 seed의 절대 수치로 결론 내리지 않음 |
| Teacher/aligner package | T0 고정, 필요 시 T1 | Student seed 변동과 합치지 않음 |
| 서버·실행 환경 | 같은 release/package, 필요 시 동일-seed 교차 | 3서버=독립3seed라는 등식은 성립하지 않음 |
| Checkpoint 선택·평가기 | 공통 grid, selected/last/plateau, 동일 export | ERGAS@best 차이를 전부 seed로 설명하지 않음 |

FRESH50의 같은 초기 state/데이터 순서 대응은 변동을 줄이려는 통제다. 다른 objective가 학습 궤적을 바꾸므로 같은 seed라고 gradient나 최종 output이 같아야 하는 것은 아니다.

## 10.2 기본 집계 단위와 부호

block b에서:

\[
\Delta H_b=H_{candidate,b}-H_{control,b},
\]
\[
\Delta E_b=ERGAS_{candidate,b}-ERGAS_{control,b},\qquad
\Delta E_b^{rel}=100\frac{\Delta E_b}{ERGAS_{control,b}}.
\]

**ΔH는 양수가 개선**, **ΔE/ΔErel은 음수가 개선**이다. Table header에 부호를 적는다. 통계는 block별 scene 평균을 만든 뒤 block 평균·표본 sd를 계산한다. 방법마다 서로 다른 seed subset을 사용한 평균을 직접 차이로 계산하지 않는다.

필수 요약은 n, 평균±표본sd(n−1), 중앙값, min/max, paired Δ의 평균·sd·중앙값, 양성 block수, target-hit수다. n=1은 sd를0으로 쓰지 말고 `NA`로 기록한다.

## 10.3 ERGAS를 세 시점으로 나눈다

| 필드 | 질문 |
|---|---|
| `ergas_at_raw_selected` | 실제 선택·배포 자산의 RR 품질은 어떤가? |
| `ergas_at_exact50k` | 같은 학습량의 수렴 상태에서 어떤가? |
| `ergas_plateau45_50` | 후반 고정6개 checkpoint에서 반복되는 비용인가? |

`ERGAS@raw-selected`가 나쁘지만 last와 plateau가 비슷하면 선택 시점 효과가 유력한 설명 후보다. Last와 plateau도 여러 seed에서 나쁘면 학습 trade-off 가능성을 남긴다. ERGAS가 run 중 가장 좋은 checkpoint만 골라 공식 selected 자산의 값을 바꾸지 않는다.

같은 checkpoint를 같은 evaluator로 다시 계산했는데 값이 다르면 seed 문제가 아니라 **metric/export 재현 문제를 먼저 점검**한다. Dataset reference·clip·DN 복원·평가 domain·checkpoint hash를 확인한다. [A, §10; K, §8.1]

## 10.4 운영상 경고와 탈락을 구분한다

아래 숫자는 기존 실험에서 도출한 통계 검정선이 아니라 **이번 budget 운용용 제안값**이다. Formal launch 전에 고정하며 결과에 맞춰 바꾸지 않는다.

| 상태 | 제안 조건 | 조치 |
|---|---|---|
| 작은 단일-block ERGAS 후퇴 | 한 seed에서 +2% 이내, H 방향은 유리 | 즉시 탈락하지 않고 다른 seed·last·plateau를 읽음 |
| 반복 RR 비용 경고 | 같은 protocol의 last/plateau ΔERGASrel 평균>+2%이며 최소2/3 block에서 양수 | `RR trade-off` 표식. H 승자를 삭제하지 않지만 균형 후보도 별도 보존 |
| 큰 단일-block 경고 | 25K 이후 공통 평가3회에서 ERGAS가 control보다 +5% 초과 | checkpoint/scale/clip/drift 감사, 후보 추가 확대는 보류 |
| 계산·계약 실패 | NaN/Inf, Teacher 변화, loss 중복, GT frame 변경 등 | 즉시 정식 결과에서 격리, 같은 case명 덮어쓰기 금지 |

+2%와+5%는 **허용 가능한 화질 기준을 증명한 값이 아니다**. 경고는 확인을 요구하는 것이지 악화값을 삭제하는 규칙이 아니다. 한 seed의 악화로 전체 방법을 실패라 하지 않고, 반복 악화를 “seed 때문”으로 면제하지도 않는다.

기존0.0031 HQNR 경험적 seed 판정선은 참고 provenance로만 보관한다. 이번 small KD Δ를 자동으로 무효화하거나 유의하다고 만드는 threshold로 사용하지 않는다. [A, §6.1]

## 10.5 후보 유지·승격 규칙

**Core J0/JQ:** 계약 실패 외에는50K 완주가 원칙이다. 조기 LR 구간의 낮은 HQNR이나 높은 ERGAS만으로 자르지 않는다.

**선택적 신규 variant:** 10K는 구현·학습 안정성 확인, 25K는 남은 시간과 정합 drift·fitting 방향 확인에 사용한다. 25K까지의 공통 최근3평가에서 paired HQNR가 모두−0.002 이하이고 plain GT/edge fitting도 회복 경향이 없으면, 그 variant의 추가 seed를 보류할 수 있다. 현재 run의 중단은 예산 필요성과 함께 기록한다. 이 조건도 **미래 최종 성능을 보장하는 검정은 아니다**.

단일 seed 양성은 `pilot-positive`이고, 0.959 단일 hit는 `asset-hit`다. 다음 우선순위는 계수 재탐색이 아니라 같은 방법의 다른 seed 확인이다. 실제 방법 선택은 가능한 한3개 공통 block의 paired 결과로 한다.

## 10.6 평균과 변동을 함께 읽는 최종 순위

공식 run selector는 여전히 raw HQNR→fSCC이다. 방법 수준에서는 다음을 함께 보관한다.

**Raw leader:** 공통 block 평균 raw HQNR가 가장 높은 방법과 해당 단일 최고 자산.

**Robust/balanced candidate:** raw 평균이 raw leader와0.0005 이내인 후보 중 paired 중앙값·worst block·ERGAS/fSCC 비용이 더 안정적인 후보. 0.0005는 이번 실무적 근접 구간이지 동등성 검정선이 아니다. 다른 후보를 선택하더라도 raw leader의 결과를 숨기지 않는다.

**질문에 대한 최종 답:** “0.959/0.960 달성 여부”와 “GT-only보다 반복적으로 좋아졌는가”를 각각 명시한다. 단일 최고 수치, 평균, worst, RR trade-off가 서로 다른 방향이면 그대로 쓴다.

## 10.7 Scene 분석과 통계의 범위

각 block의 동일20장 장면별 ΔH 분포를 보관하고, 몇 장의 큰 이득에만 의존하는지 확인한다. Scene별 Dλ/Ds 변화, RR per-band 오차를 함께 보되 scene이나 pixel을 학습 seed로 세지 않는다.

n=3에서 bootstrap이나 t-interval을 산출하더라도 작은 표본·고정 Teacher·선택 편향의 한계를 붙인다. 장면 수를 늘려 학습 seed 불확실성이 사라졌다고 주장하지 않는다. 이50시간 캠페인의 primary 판단은 transparent paired summary이며 작은 p-value를 만드는 것이 아니다.

FR20은 현재 checkpoint와 method 개발에 반복 사용되는 세트다. **이번 결과는 이 benchmark에 적응적으로 개발된 성능**으로 명시한다. 독립 held-out/generalization 검증을 완료했다고 쓰지 않는다.

# 11. 실측 처리시간에 따른 예산과 case 승인

## 11.1 최초 측정할 값

서버 이름이나 과거 no-align 시간으로 통합 처리속도를 가정하지 않는다. 처음에는 서버당 독립 학습 slot 1개를 계획 단위로 두고, 실제 GPU 수·가용 메모리·동시 실행 시 처리량을 확인한 뒤 slot 수를 확정한다.

| 측정값 | 측정 범위 | 사용처 |
|---|---|---|
| `step_sec_N0/JQ/FQ/PQ` | 초기 준비 이후 200–500 update의 실제 wall-time, data wait 포함 | case별 학습시간 예측 |
| `eval_sec_FR20/RR` | 기존 전체 evaluator·실제 저장 정책 | 평가 주기 포함 비용 |
| `probe_sec` | 고정 batch의 분리 gradient·offset response | 진단 예산 |
| `save_sec` | A/U·optimizer·RNG 등을 실제 저장 | 체크포인트 비용 |
| `calibration_sec` | τR와 λE 산출·검사 | pilot 뒤의 의존시간 |
| `peak_memory` | Teacher+Student+graph 및 routing을 포함 | slot 수와 batch 유지 가능성 |

GPU 커널 호출 직후의 짧은 CPU 시간만으로 처리량을 산정하지 않는다. 충분히 긴 구간의 완료 시간으로 측정하며, 측정 코드가 학습 RNG나 데이터 순서를 바꾸지 않도록 한다. PQ 등 분리 gradient routing은 JQ보다 비쌀 수 있으므로 별도 계측한다.

실효 batch48을 유지할 수 없으면 accumulation 변경을 명시한 새 실행 설정으로 처리한다. 서버마다 batch를 몰래 바꾸고 같은 case로 합치지 않는다.

## 11.2 Admission 산식

아직 완료하지 않은 작업의 보수적 예상시간은 다음을 출발점으로 삼는다.

\[
T_{job}=1.10\,\frac{T_{init}+N_{remaining}t_{step}
+N_{FR}t_{FR}+N_{RR}t_{RR}+N_{probe}t_{probe}+T_{save}}{3600}.
\]

1.10은 이번 운영용 **10% 시간 여유**이며 실제 성능 특성이 아니다. `t_step`에 정기 평가가 포함된 실측 총시간을 사용한다면 평가 시간을 다시 더하지 않는다.

새 case 승인은 두 조건을 모두 통과해야 한다.

**총량 조건:** 남은 GPU-hour에서 이미 승인한 작업·최종 확인 묶음·복구 여유를 뺀 예산 안에 들어간다.

**의존시간 조건:** pilot calibration, 대응 control, 추가 seed 확인을 포함한 각 서버의 실행 경로가 원칙적으로 **46h 이전에 끝난다**. 총 GPU-hour가 남아 있어도 J0→λE→JQ처럼 직렬 의존성이 긴 작업은 종료시각을 넘을 수 있다.

매 의사결정 때 다음 표를 갱신한다.

```text
현재 경과시간 / 최종 deadline = __ / 50h
신규 학습 완료 목표 = 46h
서버별 가용 slot 및 종료예상시각 = __
이미 승인된 잔여 GPUh = __
후보의 부족한 seed/control 확인에 필요한 GPUh = __
재평가·복구 예약 GPUh = __
신규 case 허용 GPUh = __
```

**36h 이후에는 원칙적으로 새 대규모 family를 시작하지 않는다.** 유력 후보의 빠진 반복·대조·재평가를 먼저 끝낸다. 10K tail도 단독 실험보다 같은 parent의 대조가 함께 종료될 때 승인한다.

## 11.3 Case 묶음의 계산량

| 묶음 | 필요한 학습 | 비용 계산 시 주의 |
|---|---|---|
| J0/JQ 핵심 | 2case×3block=6개 FRESH50 | J0-1234가 λE pilot도 겸함. pilot 학습을 이중 계산하지 않음 |
| 여섯 기본 case 전체 | 6case×3block=18개 FRESH50 | F/J/R1 실제 시간이 다르면 각각 합산 |
| PQ 3block | 3개 FRESH50 | plain control은 J0를 재사용. PQ routing 추가 비용은 반영 |
| D0/DQ 3block | 6개 FRESH50 | 기존 J0를 D0 대조로 대체하지 않음 |
| AL0/ALQ 3block | 6개 FRESH50 | A LR 변화에 맞춘 control 필요 |
| LF0/LFQ | 유효한 공통 parent에서 각각 남은25K | prefix와 parent hash 공유를 명시; 독립50K로 세지 않음 |
| QSF | 같은 Q12 exact30K에서 남은20K 가능 | 기존 Q12의0–30K 의미·optimizer·RNG가 완전히 같을 때만 prefix 재사용 |
| TCOPY10 / CONT10 한 쌍 | 10K+10K | 약0.4개50K의 update 수지만 시간은 실제 forward·평가 비용으로 계산 |
| X02 3block | 일반적으로3개 FRESH50 | 다른 λE나 다른 정책의 X02 재사용 불가 |
| XSRV 한 대응 쌍 | 같은 seed의 N0+후보 | Student seed 수를 늘린 것으로 집계하지 않음 |

기존 N2와 Teacher 준비 비용은 이번50시간에 다시 소비하지 않는 **기존 자산 비용**이다. 논문의 전체 학습 비용을 보고할 때는 그 사전학습을 생략하지 말고, 이번 캠페인 소비량과 별도로 기재한다. [A, §8.7]

## 11.4 처리속도별 축소 규칙

서버당1slot, 준비4h·최종감사4h를 가정한 순운영 pool은 최대126 GPUh다. 아래 최대 run 수는 단순히 `floor(126/T50)`이며 의존성·case별 비용 차이·tail·재시도를 무시한 **이론적 상한**이다.

| 대표 FRESH50 시간 가정 | 단순 최대50K 개수 | 권장 운영 범위 |
|---:|---:|---|
| 2h | 63 | 18core + 제한적인 결합 조정 + 반복·귀속·교차 확인 가능성을 검토 |
| 4h | 31 | §9.5와 같은 약28개50K 수준의 조건부 계획 검토 |
| 6h | 21 | core를12–15개 수준으로 좁히고, 나머지는 유력 정책 반복·X02에 우선 배정 |
| 8h | 15 | J0/JQ6개부터 확보. F0/FQ 또는 한 유력 variant의 대응 묶음 중 선택 |

예측 처리시간이 길수록 **새 hyperparameter 수를 먼저 줄이고 seed 확인과 같은-policy control을 남긴다.** 처리속도 때문에50K를25K로 줄인 run은 완주50K 결과와 같은 칸에 넣지 않는다. 이미50K horizon으로 시작한 학습을 중간에25K cosine으로 재해석하지도 않는다.

## 11.5 50시간이 합산 GPU-hour 예산인 경우의 축소안

본문의 기본 해석은 **세 서버를 병렬 쓰는50h 경과시간**이다. 운영 예산이 실제로는 세 서버의 사용시간을 모두 합한50 GPUh라면 다음처럼 바꾼다.

우선 준비·calibration·최종평가·재시도에 **6 GPUh를 예시 예약**하고 학습에44 GPUh를 남긴다. FRESH50이 평균4h라면 J0/JQ6개에24 GPUh를 쓰고20 GPUh가 남는다. 그 안에서 PQ3개12 GPUh와 TCOPY10 한 쌍의 실측 비용, 필요한 X02 pilot을 선택할 수 있다. D0/DQ3block24 GPUh는 같은 잔여예산에 들어가지 않으므로 다른 묶음을 줄이지 않고 추가할 수 없다.

이는 모든 선택을 동시에 실행하는 계획이 아니다. 핵심은 `J0/JQ 대응 반복 → 한 개의 결합 개선 → 필요한 귀속 대조`를 남기는 것이다. 실제 준비·평가 비용이6 GPUh를 넘으면 학습예산을 즉시 줄인다.

# 12. Raw HQNR 목표에 접근하는 분기별 의사결정

## 12.1 결과별 다음 case

| 첫 관측 | 우선 실행 | 필요한 대조·진단 | 지금 하지 않을 일 |
|---|---|---|---|
| JQ가 J0보다 좋고0.959 이상 단일 hit | JQ의 남은 block, X02-J | 같은 seed·같은 calibration의 paired Δ, raw 재현 | 최고 seed 주변에만 계수 재탐색 |
| FQ가 F0보다 좋고 JQ보다 안정적 | FQ 반복, FR 또는 X02-F | J와 F 각각의 no-KD 순증분 | frozen N2의 과거 결과로 FQ를 기각 |
| JQ의 A drift·추가 감독 gradient 충돌이 크고 FQ는 양성 | PQ 또는 JK0/JE0 중 한 개 | L0와 LD/LK/LE의 A gradient 방향 | offset λ를 먼저 대규모 재탐색 |
| 초반에만 A 상태가 크게 흔들림 | D0/DQ | 초기5K shift·gradient, 해제 전후 동작 | DQ만 실행하고 J0를 control로 사용 |
| 후반 A drift와 HQNR 후퇴가 같이 나타남 | LF0/LFQ 또는 AL0/ALQ | exact25K parent, last·plateau·response | 좋은 best만 남기고 후반 악화를 삭제 |
| JR/FR이 Q12보다 반복적으로 좋음 | 유력 R1의 반복; 같은 정책 X02로 edge 효과 분리 | R1↔X02↔Q12 구성 차이 | R1에 edge를 남겨 놓고 R1-only로 보고 |
| Q12는 양성이지만 목표에 조금 미달, 학습이 안정적 | 진단에 따른 단일 α/β/edge 조정 또는 TCOPY10 | 같은 정책 reference; tail은 같은 parent GT-only | covariance·feature KD를 즉시 재도입 |
| best만 높고 plateau/last는 개선 없음 | 해당 selected 자산 재현, LF/CONT의 제한적 비교 | selected step, raw/RR trade-off | best를 무효화하거나 last 개선까지 주장 |
| 거의 모든 통합이 no-KD보다 나쁨 | Teacher 재현·scale·τR·loss 중복 검사 → R1 또는 T1 대응 | 같은 asset/candidate grid, N0 fitting bin | 더 강한 soft·edge를 무조건 추가 |

표의 관계는 원인을 확정하는 진단이 아니다. 예를 들어 drift와 HQNR 저하의 동반 관측만으로 drift가 유일한 원인이라고 결론내리지 않는다. 대응 개입을 통해 질문을 좁힌다.

## 12.2 HQNR의 두 성분은 원인 진단에 사용한다

Candidate와 control의 같은 장면별 `Dλ`, `Ds`, `HQNR`를 보관한다. 분광 항의 개선과 공간 항의 비용이 상쇄되는지, 특정 장면에서만 raw 공간 항이 크게 변하는지 확인한다.

다만 평균 Dλ/Ds만 보고 HQNR를 다시 만들거나, 특정 수치 감소를 얻으면 반드시0.959가 된다고 역산해 학습 target으로 삼지 않는다. **GT edge 강화→Ds 감소→HQNR 상승**도 보장되는 관계가 아니다. GT edge는 M-frame 감독이고 Ds는 원 PAN 참조 평가라는 차이가 있다. [A, §6, §8.5–8.6; K, §5.2]

Raw와 V64가 크게 갈릴 때는 boundary·sampler·full-image filter 경로를 감사한다. V64가 목표를 넘었다는 사실로 raw 미달을 대체하지 않는다. 경계 손실이나 interpolation을 수정해야 한다면 기존 method의 버그 수정인지 새로운 방법인지 명시하고 대응 재실험한다.

## 12.3 목표 근처에서의 우선순위

**한 seed가0.959를 넘었으면:** 그 자산을 먼저 고정·재현하고, 같은 방법의 확인 block를 끝낸다. 더 높은 수치를 찾아 기존 결과를 덮어쓰지 않는다.

**평균은 개선되지만 모든 seed가0.959에 못 미치면:** 유력 정책을 고정한 뒤 small scalar 또는 추가 fitting 한 축을 선택한다. 서로 다른 seed의 최고 case를 조합한 평균을 만들지 않는다.

**TCOPY/CONT만 목표를 넘으면:** `FRESH50`, `TCOPY10`, `CONT10`의 초기화·학습량·비용을 구분해 보고한다. 추가 fitting protocol 자체를 유력 방법으로 채택할 수 있지만 독립초기화50K와 동등한 조건이라고 쓰지 않는다.

**R1만 안정적으로 목표에 도달하면:** 기여는 Teacher-error-guided GT fitting으로 정리한다. output soft가 없는 방법에 직접 증류 성과를 귀속하지 않는다.

## 12.4 50h 종료 시 목표 미달인 경우

종료 시 재현된 최고 raw 값과0.959/0.960까지의 실제 간격, 가장 안정적인 방법, paired 이득과 ERGAS 비용을 함께 남긴다. 자료가 부족한 후보는 `incomplete_budget`으로 두고 실패로 세지 않는다.

다음 캠페인은 이번에 확인된 병목 하나에서 시작한다. 예를 들어 “초기 A drift를 줄였지만 late spectral 오차가 남음”, “Q12 자체보다 R1이 반복적으로 좋음”, “Teacher package를 바꾸면 부호가 바뀜”처럼 **측정된 결과를 근거로** 범위를 좁힌다. 목표 도달을 전제로 결과 칸을 선기입하지 않는다.

# 13. s1 구현 및 s2/s3 Git 배포

## 13.1 구현 범위와 release 순서

[A]의 **성공 run aligner class·sampler·전처리**를 재사용하고, [K]의 Q12/R1 criterion을 연결한다. 보고서의 layer 표만 보고 비슷한 새 aligner를 만들지 않는다. 이번 문서는 현재 repository의 live source를 조회하거나 통합 코드를 실행한 결과가 아니다.

[K, §12]가 확인했던 경로는 `kdv/losses_rec.py`, `pa/losses.py`, `kdv/forward.py`, `kdv/calibration.py`, `train_kdv.py`다. 이는 구현자가 찾을 **과거 source 단서**이며, 현재 HEAD의 동일 동작을 보장하지 않는다. 실제 성공 run commit과 연결한다.

| release 단계 | s1 구현 내용 | 배포 조건 |
|---|---|---|
| C0: 최소 통합 | T0 binding, J/F, N0/R1/Q12/X02, calibration, RNG·평가·저장 규약 | core unit/smoke test 및 Teacher 재현 통과 |
| C1: 결합 조정 | P/JK0/JE0 routing, D/AL/LF schedule, scalar variant | C0 기본 case가 동일 입력에서 동치, 신규 경로 검사 통과 |
| C2: fitting·보고 | TCOPY/CONT, paired summary, deadline-aware queue | parent·optimizer·schedule·selector 이력 검사 통과 |

기본 기능이 구현돼 있으면 이 단계를 한 release로 묶을 수 있다. 반대로 routing 구현 때문에 core J0/JQ를 불필요하게 지연시키지 않는다. C1/C2 개발은 s1의 별도 개발 checkout에서 하고 C0 학습 소스는 고정한다.

같은 방법을 C0와 C1에서 실행했다면 full SHA를 모두 기록한다. Trainer의 실제 수치 경로가 바뀐 경우에는 단순히 새 기능 flag가 꺼져 있다고 동일 release 취급하지 말고 회귀 검사를 한다. **버그가 확인된 이전 run은 실패 원인을 남기고 새 ID로 다시 실행**한다.

## 13.2 논리적 모듈 책임

| 모듈 | 계약 |
|---|---|
| `predict_shift` adapter | 기존 crop/정규화/CNN을 정확히 한 번 호출, `[B,2]` yx 반환 |
| `warp_pan` adapter | 기존 FP32 sampling 의미 보존, native A graph 유지 |
| `reconstruct` adapter | residual8과 `M+residual` 소유자를 명확히 구분 |
| backend criterion | 최종 Student/Teacher/GT HRMS에서 L0/LD/LK/LEw 생성 |
| aligner auxiliary | Student native SG target, odd-update PAN jitter만 처리 |
| gradient router | A/U별 수신 loss를 지정, 한 optimizer update에 합산 |
| schedule controller | global update와 A freeze/unfreeze, LR, resume 상태 |
| evaluation/export | raw-original FR20 및 별도 진단 view; 같은 checkpoint metadata |

이 이름들은 설계용 adapter 명칭이다. 현재 코드의 실제 API/CLI 이름은 s1 구현자가 확인해 manifest에 매핑한다.

## 13.3 P 및 부분 routing의 구현 의미

P 정책에서 필요한 것은 다음 두 gradient다.

\[
g_A=\nabla_{\phi_S}(L_0+L_O),\qquad
g_U=\nabla_{\theta_S}(L_0+L_D+L_K+L_{Ew}).
\]

J/JK0/JE0도 §4의 A 수신 계수만 달라진다. 손실별 gradient를 구해 해당 parameter group에 누적하는 방식이나, 추가 감독용 U-Net forward에만 detached aligned PAN을 주는 방식을 사용할 수 있다. 후자는 **추가 forward 비용과 stochastic layer·buffer 동작**까지 확인해야 한다. [A, §11.4]

구현의 최소 원칙은 다음이다.

- `P_aligned` 전체를 detach하지 않는다. 그렇게 하면 plain L1의 A 학습도 사라진다.
- 분리 계산한 gradient를 더한 뒤 **A/U optimizer update는 각각 한 번**만 수행한다. Loss마다 optimizer를 별도로 step하지 않는다.
- AMP scale/unscale, gradient accumulation, clipping을 중복 적용하지 않는다. 모든 gradient가 같은 scale인지 검사한다.
- Teacher target과 gate는 detach하고, Student의 실제 오차는 유지한다. Diagnostic gradient 계산을 실제 optimizer update에 다시 더하지 않는다.

특정 PyTorch 코드 몇 줄로 AMP/DDP/resume 동치까지 검증됐다고 취급하지 않는다. 기존 trainer의 실제 설정 아래에서 plain J와 routed J의 gradient·1-step 결과를 먼저 대조한다. 다중 GPU 분산학습을 새로 도입하는 것은 이50h 기본 범위가 아니다.

## 13.4 Git 배포 흐름

**원칙:** s1에서 검증한 full commit SHA를 release 단위로 전달한다. s2/s3는 코드 수정 없이 그 SHA를 사용한다. 학습 중인 작업 디렉터리의 소스를 `git pull`로 변경하지 않는다.

아래는 실행자가 자신의 repository 경로·branch를 설정해 사용하는 **절차 예시**이며, 여기서 수행한 명령이 아니다.

```bash
# s1: 기존 개발 repository에서 구현·검사를 마친 후
# RUN_BRANCH는 새 캠페인 branch. 실제 충돌 여부를 먼저 확인한다.
RUN_BRANCH=exp/pakd50-20260914

git status --short
git diff --check
# 새 branch 준비, 구현, 회귀 검사는 기존 작업 상태를 보존하며 수행한다.
# 검토한 source/config/test/manifest 파일만 명시적으로 git add 한다.
# 데이터, checkpoint, credential, 대용량 결과 전체를 git add . 하지 않는다.
# git commit -m "Add PAKD50 integrated training release"

git push -u origin "HEAD:refs/heads/${RUN_BRANCH}"
RELEASE_SHA=$(git rev-parse HEAD)
printf 'release_sha=%s\n' "$RELEASE_SHA"
```

s2/s3에서는 Git으로 전달된 release를 확인하고 별도 고정 checkout을 만든다.

```bash
# s2/s3: 실행 중인 worktree가 아닌 repository 관리 checkout에서 수행
: "${RELEASE_SHA:?s1에서 확정한 full commit SHA를 설정해야 합니다}"
: "${RUN_WORKTREE:?아직 존재하지 않는 실행 전용 경로를 설정해야 합니다}"
RUN_BRANCH=exp/pakd50-20260914

git fetch origin "${RUN_BRANCH}"
git cat-file -e "${RELEASE_SHA}^{commit}"
# 기존 tracking branch 갱신에 pull을 사용할 때는 clean 관리 checkout에서만
# git pull --ff-only origin "${RUN_BRANCH}"

git worktree add --detach "$RUN_WORKTREE" "$RELEASE_SHA"
test "$(git -C "$RUN_WORKTREE" rev-parse HEAD)" = "$RELEASE_SHA"
test -z "$(git -C "$RUN_WORKTREE" status --porcelain)"
```

`RELEASE_SHA`는 실제 full SHA다. Branch 이름만 저장하면 이후 push로 대상이 바뀔 수 있다. 기존 작업물을 없애는 `reset --hard`, `clean -fd`, 강제 push를 배포 절차에 넣지 않는다. 실행 중인 source를 바꾸지 않고 새 release마다 새 worktree를 만든다.

배포 후 실행 전에 source SHA, resolved method config, 데이터 manifest, Teacher/pilot/calibration hash를 함께 검사한다. Git 코드가 같아도 환경·학습 자산이 같다는 뜻은 아니다.

## 13.5 코드와 학습 자산의 전달을 구분한다

**Git 관리 대상:** source, 작은 config, tests, case registry, release manifest, 방법 문서, 요약 결과표.

**별도 자산 대상:** Teacher checkpoint, Student initial states, calibration 원시값·대용량 캐시, 학습 데이터, run checkpoint·영상 export. 이미 허용된 공유 저장소나 자산 전달 절차를 사용한다. 기존에 Git LFS 등으로 관리하고 있다면 그 계약을 유지하되, 코드 push만으로 binary가 전달됐다고 가정하지 않는다.

세 서버에서 실제 수신 파일의 SHA-256을 계산해 원본 manifest와 비교한다. 파일 경로는 서버마다 달라도 asset hash와 내용은 같아야 한다. Missing asset은 즉시 실패시키고 오래된 no-align Teacher나 임의 λE로 대체하지 않는다.

학습 target 캐시를 쓰려면 Teacher A/U뿐 아니라 **sample ID·augmentation·crop·radiometric scale·MS base·sampling 정책**을 cache key에 포함한다. 학습 augmentation을 무시한 Teacher output 캐시는 다른 감독이므로, 동치 검증 없이 속도 최적화로 도입하지 않는다. [K, §9.5]

## 13.6 실행 설정과 장애 복구

각 run의 시작 시 resolved config를 동결하고 hash를 저장한다. 서버 로컬 경로·GPU 선택 같은 실행값과 LR·loss·selector 같은 method 값을 구분한다. Run이 끝난 뒤 현재 config를 다시 읽어 과거 실행 설정을 재구성하지 않는다.

중단 후 resume에는 A/U state, optimizer, scheduler, scaler, global update, freeze 상태, 데이터 sampler·augmentation·epsilon RNG, selector running state를 복원한다. `best_raw` weights만 로드하는 것은 resume이 아니라 새 parent 학습이다.

성능에 영향을 주는 precision·backend 설정이 다르면 환경 차이로 기록하고 대응 block 내에서 일치시킨다. 다른 서버와 bitwise 동치가 확보되지 않았다는 사실만으로 학습 실패라고 하지는 않지만, 그 차이를 seed 반복과 분리한다.

# 14. 회귀 검사와 최소 진단

## 14.1 Launch gate

아래 검사는 통합 코드에서 수행할 목록이며 **현재 통과했다는 기록이 아니다**.

| 검사 | 통과 조건 |
|---|---|
| T0 재현 | 정확한 같은 A/U checkpoint로 기존 raw 지표 재현, 미일치 원인 기록 |
| Shape/base | `[PAN_aligned,M]` 9ch, residual8, MS base 한 번 합산 |
| Crop/scale | A margin4→z-score 한 번; U 입력 강도는 원래 규약 |
| Warp 의미 | yx/xy·부호·HR 단위·bicubic/border/align_corners 일치 |
| Native 보존 | M/GT 불변, 복원 경로 extra jitter 없음, PAN 중복 warp 없음 |
| SG 범위 | Off target의 cS만 detach; native reconstruction→A 유지 |
| Offset 수신자 | Loff 단독 backward는 A에만 gradient, U에는 없음 |
| Q12 해제 | β=0→X02. α=β=λE=0→N0; Off는 별도 정책 유지 |
| R1 정의 | LK=0, edge 없음. Teacher difficulty에 따른 hard만 사용 |
| Gate | eS≤eT이면 a=0; gate detached, 실제 eS/k live |
| Edge | ZS=Y이면0; signed bandwise Scharr·/32·interior·0.5 reduction 일치 |
| Teacher 고정 | Student와 storage alias 없음; 여러 update 후 A/U/buffer hash 변화 없음 |
| F/J/P routing | 각 수신 모듈이 §4의 그래프·gradient와 일치 |
| Frozen AdamW | frozen A는 grad=None 및 update/WD 적용 제외, state 정확히 유지 |
| RNG | 진단·calibration on/off가 학습 sample·augmentation·epsilon 순서에 영향 없음 |
| Resume | 동일 저장점의 연속 실행과 resume의 상태·출력·다음 update를 대조 |
| Selector/export | 공통 candidate grid·full20·raw reference·same checkpoint 연결 |

같은 source·input의 회귀 허용오차는 CPU/FP32 참조와 실제 실행환경을 기준으로 사전 고정한다. HQNR seed 변동0.0031을 수치 동치 허용오차로 사용하지 않는다. 실측 오차를 본 뒤 유리하게 허용오차를 늘리지 말고, 달라진 연산·precision을 먼저 설명한다.

## 14.2 Freeze 경계와 prefix 재사용 검사

D 정책은 global update4999/5000, LF는24999/25000의 상태를 특히 확인한다. 문서의0-based update index와 로그의 completed update count가 다르면 둘을 별도 필드로 기록한다.

D에서 frozen 구간에 optimizer에 zero gradient를 계속 넣어 A의 weight decay/moment가 변하지 않게 한다. A를 해제할 때 U optimizer·scheduler를 새로 시작하지 않는다. Frozen 상태에서 Off는 학습항으로 계산하지 않으며, 필요 시 독립 진단으로만 계산한다.

LF와 QSF의 prefix 공유는 **해당 시점까지 전체 학습 의미가 같은 경우**에만 허용한다. Weights뿐 아니라 optimizer·scheduler·RNG가 연결돼야 한다. Prefix 공유 후보와 처음부터 독립 학습한 후보를 서로 다른 seed 반복으로 더하지 않는다.

## 14.3 Loss별 parameter 및 이동량 gradient

고정 진단 batch에서 다음을 계산한다.

\[
g_j^A=\nabla_{\phi_S}L_j,\qquad
g_j^U=\nabla_{\theta_S}L_j,
\quad j\in\{0,D,K,Ew,O\}.
\]

F 정책의 frozen A에는 학습 parameter gradient를 요청하지 않고 `frozen/not_applicable`로 기록한다. 필요 시 별도 진단 forward에서 cS를 독립 leaf로 두어 출력의 이동 민감도만 측정하며, 학습 graph나 A 상태는 바꾸지 않는다.

Norm과 `cos(g0,gj)`를 A/U별로 분리한다. Norm이0에 가까우면 cosine을 임의0/1로 채우지 말고 `undefined/near_zero`로 둔다. Raw edge gradient와 λE가 포함된 edge gradient를 구분한다.

Native branch에는 `∂L0/∂cS`, `∂LD/∂cS`, `∂LK/∂cS`, `∂LEw/∂cS`의 y/x 성분도 기록한다. 이는 각 감독이 어떤 sampling 이동을 요구하는지 보여 준다. **Off는 native cS target을 SG하므로 해당 직접 미분이0인 것이 정상**이며, ce 및 A parameter 쪽에서 Off 영향을 확인한다.

모듈 gradient는 전체 norm뿐 아니라 가능하면 U encoder/decoder/head 단위로 요약한다. Soft loss가 작아도 특정 경로의 방향에 영향을 줄 수 있으므로 loss ratio로 gradient 기여를 대신하지 않는다. [K, §7.4, §8.5]

**제안 빈도:** FRESH50의1K/5K/10K/25K/40K/50K에 작은 고정 batch1–2개. Full FR offset response는 선택 checkpoint와 exact50K 중심으로 수행한다. 진단 비용이 커지면 방법 간 같은 schedule로 줄이며, 매 update에 모든 gradient를 저장하지 않는다.

## 14.4 Fitting·gate·정합 로그

| 로그 | 필수 내용 |
|---|---|
| Scalar losses | L0, LD, LH=L0+LD, LK, raw LE, λE LE, raw Loff, weighted Off, display total |
| Gate | eT/eS, d/a 분위수, mean wH/wK, soft active fraction, Teacher/Student 우세 비율 |
| 세 비율 | coefficient ratio / weighted loss ratio / parameter gradient ratio를 다른 필드로 저장 |
| Fitting bins | 공통 Teacher error 기준 Q0–50/Q50–90/Q90–100의 plain Student GT L1 |
| A 상태 | cS/cT yx 평균·중앙값·분위수·norm, cS−cT, 학습 시점별 drift |
| 반응 | signed2×2 B, intercept, component MAE, EPE, 반경별 결과 |
| 영상 평가 | same checkpoint의 raw HQNR/Dλ/Ds/fSCC와 RR ERGAS/SAM/PSNR |

Teacher-error bin 경계는 고정 calibration 자료에서 계산하고, N0와 후보에 동일하게 적용한다. N0도 평가 전용 Teacher 호출로 bin을 기록한다. 높은 Teacher-error bin이 모든 모델에 본질적으로 어려운 영역이라는 뜻은 아니다.

Synthetic response는 상대 이동 관계의 진단이다. `cS−cT` 감소, offset EPE 감소, native correction norm0.55px 접근 중 어느 것도 단독 성공 조건으로 쓰지 않는다. 특정 norm을 target/cap으로 추가하지 않는다. [A, §3.4, §9]

## 14.5 최종 자산 평가

각 최종 후보는 **A/U 같은 checkpoint**를 새 평가 프로세스에서 로드하여 raw FR20과 RR를 다시 산출한다. `raw_original`, `raw_v64`, `aligned_self_v64`를 다른 필드로 저장하고, raw로 선택한 하나의 HRMS 출력에서 보조 view를 계산한다.

FR20 중 실패 장면이 있으면 전체 평균을 적법한 값으로 export하지 않는다. 제외 장면을 숨긴19장 평균이나 scene별 다른 모델 조합은 목표 달성 자산이 아니다.

추론 비용은 Teacher·loss를 제외한 **Student A+sampler+U**로 실제 측정한다. KD 자체의 train-only 비용과, 최종 front-end의 추가 추론 비용을 구분한다. [A, §5.4, §8.7]

# 15. 실행 manifest와 결과 인계 양식

## 15.1 최소 campaign manifest

다음은 **구현용 의미 schema 예시**다. 현재 repository가 그대로 읽는 config라는 뜻이 아니다. `null`은 실제 자산·환경에서 채워야 할 미확인 값이며, 필수값이 null이면 formal run을 시작하지 않는다.

```yaml
campaign:
  id: PAKD50_W112D123_WV3_20260914_v1
  status: planned
  budget_kind: elapsed_wall_hours_parallel_servers
  wall_hours: 50
  training_finish_target_hour: 46
  final_audit_hours: 4
  start_time: null
  deadline: null
  servers: [s1, s2, s3]
  gpu_slots_per_server: null
  measured_case_hours: null
release:
  git_full_sha: null
  source_worktree: null
  resolved_config_sha256: null
  environment_manifest_sha256: null
model:
  width: 112
  depth: [1, 2, 3]
  input_channels: 9
  output_channels: 8
  output_kind: residual_plus_ms_base_once
  frame: M
  aligner_source_class: null
  warp: {mode: bicubic, padding_mode: border, align_corners: false}
  aligner_margin_hr: 4
  crop_before_zscore: true
teacher:
  package_id: T0
  run_id: PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1
  checkpoint_kind: best_raw
  selected_update: null
  checkpoint_path: null
  checkpoint_sha256: null
  aligner_state_sha256: null
  unet_state_sha256: null
  frozen_own_aligner: true
  stationary_given_fixed_input: true
student:
  protocol: FRESH50
  seed: null
  init_policy: copy_teacher_aligner_fresh_unet
  unet_init_state_sha256: null
  parent_checkpoint_sha256: null
  aligner_policy: J
  aligner_loss_recipients: [L0, LD, LK, LEw, LO]
  unet_loss_recipients: [L0, LD, LK, LEw]
  freeze_until_update: 0
  freeze_after_update: null
objective:
  backend: Q12
  alpha: 1.0
  beta: 0.1
  eps: 0.000001
  lambda_off: 0.0001
  off_period: 2
  off_phase_zero_based: 1
  off_target: stop_gradient_student_native_shift
  jitter_radius_hr: 2.0
  jitter_distribution: disk_area_uniform
  tau_R: null
  lambda_E: null
  edge_r_grad: 0.05
  edge_definition: signed_bandwise_scharr32_reflect1_interior1
  edge_teacher_weighting: false
calibration:
  tau_teacher_package: T0
  max_train_patches: 3072
  seed: 1234
  batch: 48
  geometric_augmentation: false
  patch_manifest_sha256: null
  edge_pilot: J0_seed1234_exact50k
  edge_pilot_checkpoint_sha256: null
  resolved_package_sha256: null
training:
  total_updates: 50000
  effective_batch: 48
  optimizer: AdamW
  lr_unet_peak: 0.0001
  lr_aligner_peak: 0.00001
  weight_decay: 0.01
  warmup_updates: 100
  schedule: cosine
  optimizer_betas_eps_from_source: null
  minimum_lr_from_source: null
  precision_amp_clip_accumulation_from_source: null
  data_rng_state_sha256: null
  epsilon_rng_state_sha256: null
evaluation:
  primary: hqnr_raw_original
  fr_scene_set: WV3_paper_mat20
  fr_scene_manifest_sha256: null
  reference_pan: original
  aggregation: mean_of_per_scene_products
  candidate_grid_id: GRID1K_50K_v1
  candidate_updates: {start: 1000, stop: 50000, step: 1000}
  exact_last_update: 50000
  plateau_updates: [45000, 46000, 47000, 48000, 49000, 50000]
  selector: inherit_exact_running_max_hqnr_fscc_comparator
  hqnr_tie_band: 0.0001
  evaluator_sha256: null
  export_contract_sha256: null
  target_lower: 0.959
  target_stretch: 0.960
pairing:
  block_id: null
  control_run_id: null
  discovery_or_confirmation: null
```

이 예시는 JQ다. R1/N0에서 비활성 loss가 config에 남아 있어도 실제 수신 loss 목록과 계수는 맞춰야 한다. TCOPY/CONT/D/F/P는 protocol·초기화·freeze·LR·목적함수·candidate grid를 각 명세에 맞게 override하고 최종 resolved config를 저장한다.

## 15.2 Run 이름과 상태 관리

권장 full run ID 예시:

```text
PAKD50_J0_W112_D123_WV3_T0_S1234_FRESH50_v1
PAKD50_JQ_W112_D123_WV3_T0_S1234_FRESH50_v1
PAKD50_PQ_W112_D123_WV3_T0_S777_FRESH50_v1
PAKD50_DQ_W112_D123_WV3_T0_S2026_FRESH50_v1
PAKD50_TC0_W112_D123_WV3_T0_S2026_TCOPY10_v1
PAKD50_TCQ_W112_D123_WV3_T0_S2026_TCOPY10_v1
```

이 ID는 새 캠페인용 제안명이지 현재 서버에 존재하는 run이 아니다. 긴 method override는 `case_resolved.json`의 hash로 연결하며, 이름만으로 λE·LR·release를 추정하지 않는다.

| 상태 | 의미 |
|---|---|
| `planned` | 명세만 있음 |
| `queued` | 자산·의존성·예산 검사 후 실행 대기 |
| `running` | 실제 프로세스와 출력 경로가 연결됨 |
| `completed` | 계획 update 완료; 최종 평가 상태는 별도 |
| `validated` | 최종 자산 재평가와 metadata 검사 완료 |
| `incomplete_budget` | 시간 제한으로 미완주; 실패 점수로 집계하지 않음 |
| `paused_diagnostic` | 이상 진단 때문에 보류 |
| `failed_contract` | 입력·loss·asset·평가 계약 위반 |
| `failed_runtime` | 런타임 장애; 방법 성능 실패와 구분 |
| `skipped` | 계획했지만 실행하지 않음, 사유 기록 |

미실행·미평가 metric은 `null/NA`다. 0을 넣어 평균을 낮추거나 누락 결과를 모두 성공/실패로 치환하지 않는다. 실패를 재실행했을 때 원 기록도 남긴다.

## 15.3 결과 파일의 최소 구조

아래는 **실행자가 생성할 산출물 목록**이며, 이 문서와 함께 실제 생성됐다는 뜻이 아니다.

```text
campaign_artifacts/
  campaign_manifest.yaml
  release_manifest.json
  teacher_assets.json
  student_initial_states.json
  calibration_resolved.json
  candidate_updates.json
  case_registry.csv
  runtime_budget_ledger.csv
  decisions.md
  runs/<full_run_id>/
    config_resolved.yaml
    source_environment_manifest.json
    lineage_and_rng_manifest.json
    checkpoint_metrics.csv
    per_scene_raw_metrics.csv
    per_scene_rr_metrics.csv
    fitting_bins.csv
    loss_gate_gradient_diagnostics.csv
    aligner_response.csv
    checkpoint_assets.json
    validation_status.json
  final/
    per_run_summary.csv
    paired_block_summary.csv
    method_summary.md
    target_assets.json
    remaining_questions.md
```

최종 체크포인트 binary의 실제 저장 위치는 `checkpoint_assets.json`에 연결한다. Git에 대용량 output 전체를 넣지 않아도 source/config/metric과 정확히 연결할 수 있어야 한다.

## 15.4 필수 결과 열

| 파일/단위 | 반드시 포함할 열 |
|---|---|
| Run summary | run/case/protocol/seed/server/block/Teacher/commit/parent/init hash/status |
| Selected asset | checkpoint hash·selected update·raw HQNR·Dλ·Ds·fSCC·ERGAS·SAM·PSNR |
| Fixed training | exact50K 또는 exact10K 지표, 사전 plateau 평균, run max 별도 필드 |
| Paired block | candidate/control ID, matched protocol, ΔH, ΔERGAS, ΔERGASrel, target hit |
| Method summary | n, 평균±표본sd, 중앙값, min/max, paired 양성수,0.959/0.960 hit 수 |
| Cost | 실제GPUh·wallh·Teacher forward·calibration·평가·재시도 비용 구분 |
| Claims | asset-hit / repeated-gain / method-level-target / RR-trade-off / validation 상태 |

간략 결과표 template:

| Method | Protocol | n | Raw H 평균±sd | 중앙값 / worst | ΔH paired 평균 | 양성 block | ≥.959 / ≥.960 | ERGAS selected / last | 비용·주의 |
|---|---|---:|---|---|---|---|---|---|---|
| J0 | FRESH50 | NA | NA | NA | 기준 | NA | NA | NA | 미실행 |
| JQ | FRESH50 | NA | NA | NA | NA | NA | NA | NA | 미실행 |
| 선택 variant | 명시 | NA | NA | NA | NA | NA | NA | NA | 미확정 |

같은 표에서 TCOPY10의 다른 augmentation seed와 FRESH50의 독립 초기화 seed를 합쳐 하나의 sd를 만들지 않는다. Method별 seed subset이 다르면 교집합 paired 결과와 전체 절대결과를 분리한다.

## 15.5 Decision log template

```text
결정시각 / 경과시간:
근거 run 및 exact checkpoint:
비교 가능한 block 수:
Raw HQNR 변화 / target gap:
ERGAS selected / last / plateau:
Aligner drift 및 gradient 진단:
현재 남은 서버별 slot / GPUh:
선택한 다음 case / 반드시 동반할 control:
이번에 보류한 case와 이유:
유지한 code / Teacher / calibration hash:
새로운 가정 또는 protocol 변경:
```

0.959 hit가 나온 시점 이후에도 selector·case 이름·평가 domain을 바꾸지 않는다. 결과를 본 뒤 추가한 실험은 discovery로 표시하고, 다음 확인 run과 구분한다.

# 16. 근거 registry와 최종 인계

## 16.1 직접 사용한 두 최신 명세

| ID | 파일 | 이번 문서에서 사용한 범위 |
|---|---|---|
| **A** | `PAN_Aligner_L1E4_TechnicalSpec_Evidence_KD_Handoff_2026-09-14.md` | §1–6: 구조·좌표·sampling·학습·자산·평가; §8–10: 수치·seed·한계; §11–13: 통합·검사·metadata 계약 |
| **K** | `PAN_KD_Q12_R1_Spec_and_Aligner_Interface_2026-09-14.md` | §1–7: 선정 backend·수식·edge·calibration; §8: 조건부 결과; §9–12: 통합 계약·unit test·source 단서 |

작성 시 작업 공간에 있는 **첨부 Markdown bytes**의 SHA-256:

```text
A: 42e074a22e6f7c7fa255e8fc59c014b062d680d5a308081ee2512df850a351d2
K: 1677ba984ce2ebfc1b69192dbfa5b574efa085156bfb621ab515a72103982de6
```

이 값은 서버 model weight나 Git commit hash가 아니다. 원문의 수치·절 번호를 추적하기 위한 문서 식별자다. 이번 작성에서 Sheet의 최신 행이나 서버 queue·checkpoint·처리속도를 실시간 조회한 것은 아니다.

## 16.2 기존 사실과 새 결정을 다시 구분한다

**보고서에서 계승:** L1E4의 앞단 정의·검증 backbone·native/offset 경로, Q12/R1의 실제 목적함수·signed-edge·gate·reduction, source의 조건부 성능·평가 한계.

**사용자와 합의한 통합 방향:** Teacher 학습에서 얻은 aligner를 Student에 사용, 파티셔닝에 묶이지 않은 통합 최적화, seed/ERGAS 변동 고려, s1 구현과 Git 기반 s2/s3 전달,50h와 raw HQNR95.9–96.0 목표.

**이번 문서가 제안한 실행값:** T0 선택, J/F/P/D/AL/LF 및 routing case,5K/25K/30–45K 시점,scalar값,10K tail LR·reset protocol,GRID1K,운영 경고선,case 우선순위,50h 시간대와 처리시간 예시. **모두 신규 통합 설계이며 기존 완료 결과가 아니다.**

**실행 전 확정해야 하는 것:** 실제 checkpoint 경로·hash·selected step, 정확한 source config·환경·최소LR·AMP, 실제 GPU slot/처리시간, τR/λE, 배포 commit, metric/export 재현 검사.

## 16.3 시작 시 실행자가 확인할 핵심 항목

- [ ] 50h가 병렬 경과시간인지 합산GPUh인지 운영 ledger에 명시했다.
- [ ] T0의 같은 best_raw A/U와 성공 run의 sampler·전처리를 연결했다.
- [ ] J0 seed1234 pilot과 공통 τR/λE package의 의존성을 큐에 반영했다.
- [ ] Q12/R1/X02/N0의 차이와 A/U gradient 수신자를 검사했다.
- [ ] 세 서버에 같은 full commit·학습 자산·평가 manifest를 전달했다.
- [ ] J0/JQ 대응3block와 마지막 확인·재평가 예산을 우선 예약했다.
- [ ] Raw-original HQNR와 V64 열을 분리하고, seed별 ERGAS selected/last/plateau를 저장한다.

## 16.4 50h 후 남겨야 할 결론

**어느 적법한 단일 자산이 raw0.959/0.960을 넘었는가, 어느 방법이 같은 정합 자산의 GT-only보다 반복적으로 좋았는가, 그 이득의 ERGAS·fSCC 비용은 어느 정도인가**를 각각 답한다.

주력 가설은 **L1E4 정합 자산을 Q12로 공동 적응시키는 JQ**다. FQ, R1, protected/delayed adaptation, 제한적인 추가 fitting은 그 가설을 개선하거나 더 단순한 대안을 찾는 후보로 둔다. 최종 선택은 이름이나 노벨티의 기대가 아니라 같은 조건의 raw HQNR·반복 이득·복원 품질·비용에 근거한다.
