# PAN Aligner — L1E4 근방 미세조정·정합 능력 검증: s1 18시간 실행 계획

**작성일:** 2026-09-13  
**Campaign ID:** `PALSV18_W112D123_N2LAST_R200_v1`  
**기반 학습 의미:** `PALS_W112D123_N2LAST_R200_v1` 계승  
**서버·데이터:** s1 단일 GPU / WV3 / 재구성본(`fixed`)  
**예산:** 이번 후속 실행의 추가 총 **18 GPU-hours 상한**  
**기준 방법:** PALS24 `L1E4`, `lambda_off=1e-4`  
**신규 학습:** 두 λ × 세 seed = **6회**, 각 **50,000 optimizer updates**  
**상태:** 실행 설계 문서. 문서 작성으로 서버 학습·평가·큐 수정이 실행된 것은 아니다.

> **이번 목표는 두 가지다.** ① `1e-4` 주변에서도 HQNR 이득이 유지되는지 확인한다. ② 같은 checkpoint에서 변위 반응, native 위치, 출력 선명도를 함께 검사하여, 이득이 실제 정합과 어떻게 연결되는지 확인한다.
>
> **학습 구조는 바꾸지 않는다.** Native reconstruction은 aligner와 U-Net 모두를 학습시킨다. Jittered PAN은 같은 aligner의 offset consistency에만 사용하고, U-Net에 넣지 않는다. 두 모듈은 첫 update부터 공동 학습한다.
>
> **판정 규약은 유지한다.** 공식 성능은 원 PAN·논문 FR `.mat` 20장에 대한 `best_raw` HQNR → **FR fSCC**다. `aligned_*`, V64, 정확한 50K last는 진단이다. HQNR 우위를 aligned 점수로 대체하거나, best의 영상 성능에 last의 정합 능력을 합쳐 보고하지 않는다.

---

## 목차

1. [18시간 변경 요약과 근거](#basis)
2. [방법·초기화·학습 연결](#method)
3. [학습 case와 재사용 대조군](#cases)
4. [18시간 예산·실행 순서](#budget)
5. [실행 전 gate](#gates)
6. [V0: HQNR·checkpoint·평가 격자](#v0)
7. [V1: signed 2D 변위 반응](#v1)
8. [V2: 모달리티 대응과 shortcut 검증](#v2)
9. [V3: native 위치·선명도·보간 영향](#v3)
10. [V4: correction 치환·복원 stress](#v4)
11. [학습 구현·gradient·재현성](#implementation)
12. [선택·해석·종료 규칙](#decision)
13. [Config·queue template](#config)
14. [산출물·인계·범위 제한](#outputs)
15. [근거 문서](#sources)

---

<a id="basis"></a>
## 1. 18시간 변경 요약과 근거

### 1.1 이전 후속안에서 달라지는 것

| 항목 | 직전 후속안 [S2] | 이번 18시간안 |
|---|---|---|
| 예산 | PALS24 보고서의 미사용 9.80h에 맞춘 안 | **이번 후속 실행을 최대 18h로 재편성** |
| 신규 λ | `3e-5`, `3e-4` | 동일 |
| 기본 신규 seed | 1234·7777, 4회 | **1234·7777·2025, 6회** |
| 현재 기준 | 기존 `1e-4` | 동일, 3-seed asset 재사용 검증 |
| 학습량 | case당 50K | 동일 |
| 정합 검증 | V0–V4 | **전체 3-seed 대응 및 동일 checkpoint 연결을 강화** |
| 새 모듈·loss | 없음 | 없음 |
| L1E4 전체 모델 연장 학습 | 별도 단계로 보류 | 계속 기본 큐에서 제외 |

**예산 해석:** 18h는 이 후속 작업의 상한이다. 이전에 사용한 14.20h를 소급 포함하지 않으며, 미사용 9.80h를 18h 위에 더해 27.80h를 자동 배정하지 않는다. 서버 장부가 기존 프로젝트 예산을 이어 쓰더라도, 이 문서에 따른 후속 작업의 한도는 18h다. 이미 수행한 후속 작업이 있으면 hash와 실행 이력을 확인하고 중복 실행하지 않는다.

이번의 ‘미세조정’은 **N2 사전학습 aligner를 미세조정하며 새 U-Net을 학습하는 기존 방법의 λ 정밀 조정**이다. 완료된 L1E4 전체 모델을 50K 이후로 이어 학습하는 제3단계가 아니다. [S2 §3, §9–10]

### 1.2 보고서에서 가져온 관측

[S1 §2]의 값이다. 다음 표는 새로운 재측정 결과가 아니다.

| Seed | P0 raw HQNR | P2/L000 raw HQNR | L1E4 raw HQNR | L1E4−P0 | L1E4−L000 |
|---|---:|---:|---:|---:|---:|
| 1234 | 0.95316 | 0.95389 | 0.95552 | +0.00236 | +0.00163 |
| 7777 | 0.95081 | 0.95222 | 0.95542 | +0.00461 | +0.00320 |
| 2025 | 0.95068 | 0.95547 | 0.95698 | +0.00630 | +0.00151 |

전체 묶음의 P0 대비 평균 이득은 약 **+0.00442**, consistency 추가분의 L000 대비 평균은 약 **+0.00211**이다. 보고서는 전자를 양성으로 판단하지만, 후자는 판정선 초과가 1/3 seed라 크기가 미확정이라고 구분한다. [S1 §2.3–2.4]

Seed 1234의 별도 정합 진단에서 L000→L1E4는 다음처럼 변했다. [S1 §2.1]

| 항목 | L000 | L1E4 |
|---|---:|---:|
| Native 보정 크기, last | 0.344 px | 0.557 px |
| 보고서의 반응 요약량 `norm(B_resp)` | 0.194 | 0.525 |
| 보고서의 offset EPE | 0.937 px | 0.575 px |

EPE는 약 38.6% 낮아졌지만, **이 값은 원래 PAN–MS의 물리적 정합 GT에 대한 오차가 아니다.** Legacy EPE·반응 요약량의 probe, split, norm 정의, checkpoint ID는 원시 기록으로 확인한다. `0.525`를 정확도 52.5%나 각 축 slope −0.525로 번역하지 않는다. [S2 §2.2]

또한 P0→L1E4의 **raw fSCC는 세 seed 모두 감소**했다.

| Seed | P0 fSCC | L1E4 fSCC | 차이 |
|---|---:|---:|---:|
| 1234 | 0.89980 | 0.87547 | −0.02433 |
| 7777 | 0.89313 | 0.88410 | −0.00903 |
| 2025 | 0.89661 | 0.87856 | −0.01805 |

이 때문에 V3의 위치·선명도 분리를 필수로 둔다. **fSCC 저하가 반드시 blur라는 뜻도, 반드시 올바른 좌표 보정의 대가라는 뜻도 아니다.** [S1 §2.6; S2 §2.3]

### 1.3 근거의 한계와 이번 질문

- [S1]은 HQNR은 best_raw, 보정 크기는 last로 제시한다. 같은 배포 checkpoint가 성능과 정합 반응을 동시에 가졌는지는 이번 V0/V1에서 확인한다.
- 0.45–0.7 px가 이득 구간이라는 내용은 보고서의 해석이다. **0.55 px를 정답, 강제 목표, 보정 cap으로 사용하지 않는다.**
- `off_unet_grad_absent`와 metric 재현 통과는 보고서의 검증 결과다. 새 실행 코드에서도 해당 경로를 재확인한다.
- 3-seed는 같은 데이터셋·같은 donor를 공유하는 downstream 초기화 반복이다. 새로운 센서·새 장면 일반화나 donor 사전학습의 독립 3회 반복은 아니다.
- 보고서에 없는 실제 checkpoint, raw scene CSV, `B_resp` 코드, 그림 asset을 이번 문서에서 검사했다고 가정하지 않는다.

**가설 Hλ:** `3e-5 / 1e-4 / 3e-4`에서 현재 이득이 얼마나 넓게 유지되는가?  
**가설 Hresp:** 같은 checkpoint에서 L1E4 및 근방이 L000보다 올바른 상대 변위 반응을 보존하는가?  
**가설 Hnative:** 그 반응이 원래 입력의 위치 개선 및 RR 경계 복원으로도 연결되는가?  
**가설 Hartifact:** HQNR 이득 또는 fSCC 하락을 위치 변화와 보간·선명도 변화로 어느 정도 구분할 수 있는가?

Hnative와 Hartifact는 아직 미확정인 설명을 검증하는 질문이다. 이번 설계가 이를 미리 참으로 전제하지 않는다.

---

<a id="method"></a>
## 2. 방법·초기화·학습 연결

### 2.1 공통 구조

| 항목 | 실행값 / 원칙 |
|---|---|
| Backbone | **W112, depth=[1,2,3]**, 기존 `fixed` 구현 |
| 입력 / 출력 | PAN 1ch + bicubic MS 8ch = **9ch** / HRMS residual 8ch |
| Task | MS reconstruction만. PAN reconstruction·MARs mode modulation 없음 |
| 기타 | Attention/CM3A, LPAN/HPAN, KD, GT variance, geometry loss 추가 없음 |
| Aligner | 기존 N2/NF16 global CNN, 약 0.105M; 아키텍처·정규화 유지 |
| Aligner 출력 | sample당 `(dy, dx)` 두 숫자, 현재 PAN/HR pixel |
| Aligner view | PAN/MS 모두 margin4, train64→56, FR512→504; 기존 wrapper에서 한 번만 적용 |
| U-Net 영역 | 원래 전체 domain, 추가 crop 없음 |
| Warp | 기존 bicubic, `border`, `align_corners=False` |
| 좌표 | MS condition·residual base·GT·출력 배열은 M-frame 유지 |
| 금지 | Output inverse, GT/MS warp, correction clamp/tanh, λ로 변위 직접 축소 |

MS-only·9ch·W112–D123을 고정한다. 과거 W168 dual/11ch나 W96–D124를 대조군으로 혼합하지 않는다. 전체 2.6589M은 보고서의 backbone 수치이며, aligner 포함 total은 실제 `numel()`로 별도 기록한다. [S1 머리말; S2 §3; S3 §3–5]

### 2.2 초기화: 모든 λ가 같은 출발점을 공유한다

```text
Donor run:
  PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT
Donor checkpoint:
  정확한 optimizer update 50,000의 last
복사:
  aligner state_dict만
복사하지 않음:
  N2 U-Net / N2 optimizer / N2 scheduler / 초기 best_hqnr
```

각 seed의 신규 두 λ는 **그 seed의 기존 PALS24와 동일한 저장 U-Net 초기 tensor**를 사용한다. Aligner는 모든 seed에서 동일 donor hash를 쓴다. N2의 마지막 prediction head를 다시 zero-init하지 않는다. [S3 §3]

Seed 2025가 donor seed와 같아도 donor U-Net을 가져오지 않는다. 각 run은 독립 학습이며, 앞 run의 최종 가중치를 다음 λ에 넘기지 않는다.

**계약 불일치 시:** 기존 init tensor 또는 data/RNG 순서를 복원할 수 없으면 과거 L000/L1E4와의 통제 비교가 깨진다. Gate에서 이를 기록하고 대조군 재학습을 우선한다. 모든 baseline이 같다고 가정한 채 신규 6회를 밀어 넣지 않는다.

### 2.3 Native 복원 경로 — 매 update

\[
M=U_{\mathrm{bicubic}}(S),\qquad
\hat c_0=A_\phi(P,M),
\]
\[
\widetilde P_0=\mathcal W(P,\hat c_0),\qquad
\hat Y_0=M+F_\theta([\widetilde P_0,M]),
\]
\[
L_{\mathrm{rec}}=\operatorname{mean}_{B,C,H,W}|\hat Y_0-Y|.
\]

**Native는 추가 synthetic jitter가 없다는 뜻이지 실제 정합 오차가 0이라는 뜻이 아니다.** Aligner의 예측 보정은 적용한다. `c0`와 `P_aligned`를 reconstruction graph에서 detach하지 않는다.

Sampler 부호는 다음을 계승한다.

\[
\mathcal W(P,c)[y,x]=P[y+c_y,x+c_x].
\]

Grid 마지막 축은 `(x,y)`, 모델 출력은 `(dy,dx)`다. 육안상 영상 내용의 이동 방향과 sampling 좌표 부호를 혼동하지 않는다.

### 2.4 추가 변위 연습 — 두 update 중 한 번

\[
P_\epsilon=\mathcal W(P,\epsilon),\qquad
\hat c_\epsilon=A_\phi(P_\epsilon,M),
\]
\[
L_{\mathrm{off}}=
\frac{1}{2B}\sum_{i,k\in\{y,x\}}
|\hat c_{\epsilon,i,k}+\epsilon_{i,k}-\operatorname{sg}(\hat c_{0,i,k})|.
\]

0-based optimizer update index를 t라고 하면:

\[
m_t=\mathbf1[t\bmod 2=1],\qquad
\boxed{L_t=L_{\mathrm{rec}}+m_t\lambda_{\mathrm{off}}L_{\mathrm{off}}.}
\]

같은 batch의 같은 PAN 복사본을 흔든다. MS는 동일하다. **Jittered PAN이나 그 보정 결과를 U-Net에 넣지 않는다.** `sg()`는 consistency target 경로만 차단한다. 정확히 아는 감독량은 추가한 ε이며, 원래 offset의 GT는 아니다.

\[
\boxed{\nabla_\phi L_t=\nabla_\phi L_{\mathrm{rec}}
+m_t\lambda_{\mathrm{off}}\nabla_\phi L_{\mathrm{off}}},\qquad
\boxed{\nabla_\theta L_t=\nabla_\theta L_{\mathrm{rec}}}.
\]

| Update | Native aligner | U-Net | Jittered aligner | Loss |
|---:|---|---|---|---|
| 0 | gradient 있음 | 1회 복원 | 없음 | Rec |
| 1 | gradient 있음 | 1회 복원 | 같은 가중치로 1회 | Rec + λ·Off |
| 2 | gradient 있음 | 1회 복원 | 없음 | Rec |
| 3 | gradient 있음 | 1회 복원 | 같은 가중치로 1회 | Rec + λ·Off |

두 모듈을 번갈아 따로 학습하는 스케줄이 아니다. 두 loss를 합쳐 backward·optimizer update를 한 번 수행한다. 두 호출이 가중치를 공유하므로 auxiliary update는 다음 native 보정에도 간접 영향을 준다. [S2 §3; S3 §4–5]

### 2.5 Sampling·최적화 값

\[
\rho=b\sqrt{u_1},\quad \vartheta=2\pi u_2,\quad
\epsilon=(\rho\sin\vartheta,\rho\cos\vartheta),\quad b=2.0,
\quad u_1,u_2\sim U(0,1).
\]

| 항목 | 유지값 |
|---|---|
| ε 분포 | 0 중심 **원판 면적 uniform**, 길이 ≤2 현재 PAN px |
| Sampling 단위 | sample당 하나의 전역 벡터; 매 auxiliary update·sample마다 새로 생성 |
| 적용 시점 | 기존 공통 flip/rotation 이후, 원래 PAN에서 생성; 변형 누적 없음 |
| Aligner / U-Net LR | **1e-5 / 1e-4** |
| Optimizer / WD | 실제 PALS24 AdamW 설정 / 기존 0.01 및 parameter exclusion 정책 유지 |
| Scheduler | 기존 warmup100 + cosine, 같은 배율·step 순서, 50K 종료 |
| Batch | 기존 실효 48, accumulation·정밀도 정책 유지 |
| λ schedule | 각 run에서 고정. ramp·decay·adaptive balancing 없음 |
| Loff reduction | batch×2 평균 L1, HR px 단위; EPE loss로 변경하지 않음 |
| 수치 정책 | 기존 native 전처리 유지; warp·offset 계산 FP32 |
| 추론 | synthetic ε=0; 예측한 c0는 사용 |

`b=2`는 기존 FR 입력 분포를 참고한 개발 설정을 계승한다. 이번 실험을 train-only calibration 또는 test-untouched 검증으로 표현하지 않는다. 2 HR px=0.5 LRMS px라는 환산 때문에 sampler에 0.5를 넣지 않는다. [S3 §5]

---

<a id="cases"></a>
## 3. 학습 case와 재사용 대조군

### 3.1 기본 신규 6회

| 순번 | Case ID | λoff | Seed | Update | 예약 GPU-h |
|---:|---|---:|---:|---:|---:|
| 1 | `R-L3E5-S1234` | **0.00003** | 1234 | 50,000 | 1.70 |
| 2 | `R-L3E4-S1234` | **0.0003** | 1234 | 50,000 | 1.70 |
| 3 | `R-L3E4-S7777` | **0.0003** | 7777 | 50,000 | 1.70 |
| 4 | `R-L3E5-S7777` | **0.00003** | 7777 | 50,000 | 1.70 |
| 5 | `R-L3E5-S2025` | **0.00003** | 2025 | 50,000 | 1.70 |
| 6 | `R-L3E4-S2025` | **0.0003** | 2025 | 50,000 | 1.70 |
| | **학습 합계** | | | **300,000** | **10.20** |

Run 명:

```text
PALSV18_L3E5_W112_D123_WV3_S<seed>_N2LAST_R200_v1
PALSV18_L3E4_W112_D123_WV3_S<seed>_N2LAST_R200_v1
```

같은 seed의 두 λ를 한 묶음으로 끝낸다. Seed 7777의 순서는 반대로 두어 실행 순서와 λ가 완전히 결합하지 않게 한다. 순서가 초기화나 데이터 순서를 바꾸어서는 안 된다.

### 3.2 기존 9개 대조군 재사용

| 방법 | Seed 1234 | Seed 7777·2025 | 사용 목적 |
|---|---|---|---|
| P0 / CTRLP0 | NF16 P0 | PALS24 CTRLP0 | 무정합 baseline |
| P2 / L000 | NF16 P2 | PALS24 L000 | Recon-only 공동 학습 |
| L1E4 | PALS24 L1E4 | PALS24 L1E4 | 현재 기준 λ=1e-4 |

추가 진단 기준으로 **N2 exact50K last**를 쓰며, 강한 λ의 대조는 **seed1234 L1E2/NF16 P3**를 재사용한다. 기존 폴더·checkpoint·Sheet 행은 덮어쓰지 않는다.

다음의 세 질문을 별도로 비교한다.

1. 새 λ − **L1E4**: 현재 방법의 정밀 조정 이득이 있는가?
2. 새 λ − **L000**: 보조 offset 감독이 여전히 유용한가?
3. 새 λ − **P0**: 정합 포함 전체 방법이 무정합보다 나은가?

이번 세 seed는 이미 이전 탐색·확인에 사용됐다. 두 신규 λ를 모두 세 seed에서 계산하는 이번 결과는 **국소 개발 grid의 반복 확인**이지, 새로 확보한 독립 최종 test seed가 아니다.

---

<a id="budget"></a>
## 4. 18시간 예산·실행 순서

### 4.1 예산표

| 단계 | 내용 | 예약 GPU-h | 누적 |
|---|---|---:|---:|
| **G0** | Asset·metric contract·선택 격자·gradient smoke·profiling | **0.80** | 0.80 |
| **V-pre** | 기존 best/last V0, 기존 L000/L1E4/N2의 V1·V2 우선 검사 | **2.00** | 2.80 |
| **R1234** | 새 λ 두 개 × 50K | **3.40** | 6.20 |
| **R7777** | 새 λ 두 개 × 50K | **3.40** | 9.60 |
| **R2025** | 새 λ 두 개 × 50K | **3.40** | 13.00 |
| **V-post** | 신규 best/last 연결, V1·V2 보완, V3·V4·stress | **3.50** | 16.50 |
| **Report** | 동일 evaluator 최종 확인·표·산출물 목록·결론 | **0.50** | 17.00 |
| **Buffer** | 실행 변동·재현 재확인 | **1.00** | **18.00** |

**학습 10.20h + 검증 5.50h + gate 0.80h + 집계 0.50h + buffer 1.00h = 18.00h.**

[S1 §6]의 run 실측은 9회 합계 12.93h, 평균 약 1.44h다. 이번 1.70h/run은 정기 평가를 포함한 예약이며 완료 시간 보장이 아니다. 진단을 추가한 실제 속도를 profiling한다. 새 학습의 정기 native 평가는 학습 슬롯에, 별도 stress·정합 분석은 V 슬롯에 기록하여 이중 계산을 피한다.

s1 단일 GPU에 동시 training run을 여러 개 올리지 않는다. CPU 분석은 자원이 허용하면 겹칠 수 있지만, 이를 가정해 필수 GPU 작업 시간을 삭제하지 않는다. **18 GPU-h와 실제 경과시간은 I/O·CPU 작업 때문에 다를 수 있다.** 18시간 경과 제한이 함께 있으면 deadline도 같은 gate에서 검사한다.

### 4.2 실행 큐

```text
G0  보고서/기존 asset 고정 → old pals24 종료 여부 확인 → 새 campaign ledger
G1  3 seed init/controls/metric/hash 검사 → 작은 실제 wrapper smoke
V0  기존 3-seed P0/L000/L1E4 best/last 재평가 및 selection-grid 연결
V1  기존 L000/L1E4/N2의 signed PAN-only + MS-only/common-shift 핵심 검사
R1  seed1234: L3E5 → L3E4
R2  seed7777: L3E4 → L3E5
R3  seed2025: L3E5 → L3E4
V2  신규 6회 best/last V0/V1 → 동일 조건 V2 보완
V3  Native 위치·RR edge/선명도·보간 대조
V4  Correction 치환 + RR/FR synthetic stress
END 공식 raw 표 / 공통 grid 표 / 정합·artifact 표 / manifest / 종료 기록
```

V-pre의 실제 probe별 raw cache는 V-post에서 재사용한다. 같은 hash를 중복 평가하지 않는다. 신규 run 중간의 무거운 진단은 학습을 방해하지 않도록 독립적으로 수행하며, training RNG·buffers를 바꾸지 않는다.

### 4.3 예산 초과 처리

완료하지 못할 run을 시작하지 않는 것이 기본이다. 새 seed 묶음을 시작하기 전에:

\[
T_{used}+\widehat T_{pair}^{safe}
+T_{required\_diagnostics\_remaining}
+T_{report\_remaining}+T_{buffer\_remaining}\le18
\]

를 만족해야 한다. `pair`는 해당 seed의 두 λ다. Safe ETA는 실측 시간에 여유를 붙여 갱신하며, 초기에는 **각 run 1.70h와 실측 기반 10% 여유 ETA 중 큰 값**을 쓴다. 예약 시간과 ETA를 중복 가산하지 않는다.

우선순위:

- **유지:** 올바른 대조군, 완전한 50K, 같은 checkpoint의 V0/V1, fSCC 손실을 설명할 V3, 최종 보고.
- **감축 1:** 세부 kernel/padding 조합·추가 tile/OOD 등 아래에서 선택 사항으로 둔 진단.
- **감축 2:** seed2025의 신규 두 λ를 **한 묶음으로** 보류.
- **감축 3:** 그래도 부족하면 seed7777 묶음을 보류하고 seed1234 pair+기존 3-seed 검증을 끝낸다.

대조군 hash 불일치 등으로 재학습이 필요하면 낮은 우선순위 seed 묶음의 예산을 대체한다. **비교 대상만 30K에 끊거나, 잘 나오는 λ만 나머지 seed에서 실행하지 않는다.** NaN·잘못된 데이터·budget 중단은 실패/미완료로 기록하고 50K 완주 결과에 합치지 않는다.

이익이 없다는 이유만으로 같은 seed의 반대 λ를 중간에 빼지 않는다. 반대로 18h를 채우려고 새 λ·hard cap·KD case를 즉석 추가하지 않는다.

---

<a id="gates"></a>
## 5. 실행 전 gate

### 5.1 필수 검사

| Gate | 검사 | 통과 / 실패 처리 |
|---|---|---|
| G-A asset | N2 exact50K aligner, source prefix·hash·view·초기 tensor | 정확한 asset만 사용. N2 초기 best 또는 L1E4 전체 checkpoint로 대체 금지 |
| G-C control | 기존 P0/L000/L1E4와 신규의 init·data·optimizer·augmentation | 차이가 있으면 원인을 기록하고 통제 비교를 재구성 |
| G-M metric | evaluator·FR20 manifest·raw 참조·ROI·export 수치 정책 | 정의와 원시 재현이 맞아야 학습 시작 |
| G-S selector | `best_raw HQNR → FR fSCC`, candidate update 목록 | RR SCC/JQM를 fSCC 대용으로 사용하지 않음 |
| G-G graph | Rec→A/F, Off→A only, target c0 SG, 두 호출 동일 A | 실제 wrapper에서 검증. 누락/누출은 구현 오류로 중단 |
| G-W warp | `(dy,dx)`와 grid `(x,y)`, 부호·unit·지원영역 | impulse/coordinate-ramp와 알려진 offset으로 검사 |
| G-R RNG | data/augmentation/offset/probe RNG 분리 및 resume parity | 진단 횟수에 따라 training 입력이 바뀌면 수정 |
| G-T time | 실제 batch forward/backward·평가 속도·peak memory | 50K+필수 진단 ETA가 예산에 들어가야 실행 |

`Lrec→aligner`가 연결됐다는 사실과 매 batch의 gradient norm이 크다는 사실은 다르다. 테스트는 일부러 오차가 있는 비퇴화 batch로 graph 연결을 확인하고, 실제 gradient 크기는 로그로 본다. 모든 실제 step에서 특정 norm 하한을 강제하지 않는다.

Smoke 후에는 **저장된 초기 model·optimizer·scheduler·RNG로 복구하여 본 학습을 시작**한다. Smoke의 수백 update를 특정 case에만 추가하지 않는다.

### 5.2 문서에 없는 구현값 처리

실제 PALS24 코드에서 beta/epsilon, gradient clipping, AMP/scaler, minimum LR, normalization, 입출력 범위, 저장 포맷을 계승한다. 아래 template는 실행 의미를 설명하며 repository의 실제 API 이름을 보장하지 않는다. 기존 파라미터가 누락되면 라이브러리 기본값으로 조용히 대체하지 않는다.

독립 native 정합 추정기의 코드가 없는 경우 `not_available`로 기록한다. 다른 일반 공식을 같은 estimator 이름으로 구현해 검증 완료라고 처리하지 않는다. Native 위치 근거가 부족하면 성능 결과는 보고하되 절대 정합 주장을 보류한다.

---

<a id="v0"></a>
## 6. V0 — HQNR·checkpoint·평가 격자

### 6.1 동일 checkpoint 연결

기존 P0/L000/L1E4의 3 seed, 신규 두 λ의 3 seed 모두 **best_raw와 exact50K last**를 검사한다. N2 donor는 exact50K last, 강한 λ L1E2는 seed1234의 best/last를 진단 대조로 둔다. 동일 hash인 best/last는 계산을 공유하되 역할 메타데이터는 둘 다 남긴다.

모든 레코드의 key:

```text
run_id, seed, checkpoint_kind, optimizer_update, checkpoint_sha256,
code_hash, evaluator_hash, dataset_manifest_hash,
split, reference_policy, roi_id, probe_manifest_hash
```

각 checkpoint에서 **HQNR·fSCC·Dλ·Ds·RR·native c·반응·EPE를 연결**한다. 미측정 값은 `not_measured`다. Best의 점수와 last의 반응을 하나의 가중치 결과처럼 묶지 않는다.

### 6.2 주 지표와 보조 view

\[
H_i=(1-D_{\lambda,i})(1-D_{s,i}),\qquad
HQNR=\frac1{20}\sum_{i=1}^{20}H_i.
\]

평균 Dλ·Ds를 곱하지 않는다. FR20은 현재 `fr_mat20`의 명시적 20장 manifest이며 과거 12–19 subset으로 되돌리지 않는다.

| View | PAN reference | 영역 | 용도 |
|---|---|---|---|
| `raw_original` | 원 P | 기존 전체 domain | **공식 성능·best 선택** |
| `raw_v64` | 원 P | 고정 `[64:448,64:448]` | 경계 효과 대조 |
| `aligned_self_v64` | 해당 모델이 사용한 W(P,c0) | 동일 V64 | 자기 정합 참조 진단 |
| `aligned_fixedN2_v64` | frozen N2가 native pair에서 만든 PAN | 동일 V64 | 공통 참조 진단 |

같은 checkpoint의 **동일 HRMS 출력**을 재사용한다. Raw를 계산한다고 aligner를 끄지 않는다. Zero correction 재추론은 V4의 별도 개입이다.

Aligned 평가에서는 저해상도 PAN도 해당 high-resolution reference에서 기존 evaluator 경로로 다시 만든다. 필터·MTF·decimation은 full domain에서 수행한 뒤 정해진 ROI를 집계한다. PAN reference만 달라졌을 때 같은 출력·MS·ROI의 Dλ는 같아야 한다.

FixedN2도 물리적 정합 GT가 아니다. 모든 모델에 같은 reference를 주는 진단으로만 사용한다. `max(raw,aligned)` 또는 가중합으로 새로운 주 점수를 만들지 않는다.

**Mask:** 보간·필터·metric window의 유효 support를 확인한다. 무효 픽셀을 0으로 채워 일반 metric에 넣지 않고, 문제 장면을 조용히 제외하지 않는다. V64까지 무효 support가 침범하면 aggregate eligibility 실패를 기록한다. 학습 recon 영역은 예측값에 따라 줄이지 않는다.

### 6.3 평가 후보 수 불균형의 처리

[S1 §6]에 따르면 재사용 NF16 P0 seed1234는 `eval_epoch=5`라 후보가 더 많았다. 기록된 best는 **0.95316**, 공통 10 격자 재선택은 **0.95275**다. 이것은 우위를 반전시키지 않았지만 선택 예산 차이이므로 남겨야 한다.

1. 기존 PALS24 L1E4의 **실제 optimizer-update 평가 목록**을 `selection_grid.json`으로 고정한다. `eval_epoch=10`을 `매 10 update`로 오해하지 않는다.
2. 신규 6회는 같은 목록을 사용하고 exact50K를 반드시 평가한다. 전환 과정에서 후보 수를 늘리지 않는다.
3. 기존 공식 `best_raw`와 Sheet 값은 보존한다.
4. 대조군과 후보의 공통 평가 update에서 `matched_grid_best_raw`를 별도 산출한다.
5. 메트릭 로그는 있으나 해당 가중치가 없으면 공통 grid 점수 자체는 집계 가능하더라도 그 checkpoint의 정합 진단은 미측정으로 남긴다. 점수를 근거로 가중치를 재현했다고 가정하지 않는다.
6. 신규 native 평가 checkpoint는 공통 grid 재선택을 재현할 수 있도록 보존한다. Disk projection을 gate에서 확인한다.

**두 표가 필요하다:** 과거와 연결되는 원래 selector 결과표와, 선택 기회를 맞춘 matched-grid 결과표. 새 λ간 주요 비교는 후보군 모두 같은 grid를 사용하므로 동일 조건에서 수행한다. 기존 P0의 selector 차이는 Notes에 남긴다.

### 6.4 Tie와 판정선

- Checkpoint: historical maximum HQNR 기준 `1e-4` tie band → **같은 raw 참조 FR fSCC** → 기존의 동일 tie 처리.
- fSCC 자체의 수치 tie tolerance는 현재 코드값을 고정한다. RR SCC로 치환하지 않는다.
- 방법 간 경험적 HQNR 분류선: **0.0031 유지**. 새 보고서의 2σ=0.00279는 일관성 확인용이며 유리한 값을 선택하지 않는다.
- 재현 numerical tolerance는 dtype·저장 포맷을 확인해 별도로 고정한다.
- 0.0031은 통계적 동등성/유의성 증명이나 EPE·fSCC·PSNR 임계값이 아니다.

RR은 같은 checkpoint의 export 경로를 기준으로 재산출하고, 학습 로그 정의도 별도 열로 남긴다. 보고서 RR 숫자와 Sheet export 숫자를 아무 설명 없이 혼합하지 않는다. 초기 best는 삭제하지 않는다.

---

<a id="v1"></a>
## 7. V1 — signed 2D 변위 반응

### 7.1 고정 probe manifest

아래 표본 수는 **이번 실행을 위한 신규 운영 설정**이다. 과거 보고서가 이 probe로 측정됐다고 가정하지 않는다.

| Split | 기본 표본 | 입력 크기 | 처리 |
|---|---:|---:|---|
| Train-size diagnostic | 64 patch | 64² PAN | 기존 고정 non-training diagnostic set 우선; 없으면 train에서 고정 추출하고 `in_sample` 표시 |
| RR | 기존 test 20장 | 256² PAN | scene 전수 |
| FR | 논문 test 20장 | 512² PAN | scene 전수 |
| Calibration | train에서 고정 256 patch | 64² PAN | blur/공통 vector 등 보조 진단의 사전 calibration에만 사용 |

별도 scene holdout이 실제로 있으면 그 정보를 기록한다. 단순 train 재추출을 unseen validation이라고 부르지 않는다. 표본 ID·sampling seed·frame augmentation은 점수를 보기 전에 고정하며, 모든 모델에서 같은 입력을 쓴다.

반경 `r={0.25,0.5,1,2}` HR px, 방향 `θ=kπ/4, k=0,…,7`, 그리고 native zero를 사용한다. 즉 **native 1 + nonzero 32개**다. Zero는 identity/반복 오차용이고 EPE의 주 평균에서는 제외한다. Radius와 방향별 표도 별도 저장한다.

### 7.2 핵심 계산

\[
c_{i,0}=A(P_i,M_i),\quad c_{i,e}=A(W(P_i,e),M_i),
\]
\[
r_{i,e}=c_{i,e}+e-c_{i,0},\qquad
EPE_i=\frac1K\sum_e\|r_{i,e}\|_2,
\]
\[
MAE_{component,i}=\frac1{2K}\sum_e(|r_{i,e,y}|+|r_{i,e,x}|).
\]

각 장면 안에서 response를 적합한다.

\[
c_{i,e}-c_{i,0}=B_i e+a_i,
\qquad B_i\in\mathbb R^{2\times2}.
\]

저장: `B_yy, B_yx, B_xy, B_xx`, `a_y,a_x`, signed 대각, 교차항, singular values, fit residual. 이상적 PAN-only 반응은 **B=−I**다. `norm(B)` 한 숫자로 합격을 판정하지 않는다.

**무반응 대조:** 같은 probe에서 c_e=c_0이면 EPE는 `mean(norm(e))`다. 이번 32개 균등 probe의 이론값은 **0.9375 px**다. 이는 이번 고정 probe의 직접 계산이며, **보고서의 0.937과 정의가 같다고 추정하지 않는다.** Legacy probe를 별도로 재현하고 이름을 구분한다.

L1E4를 무조건 −I에 맞춘 모델로 요구하지 않는다. 판정할 것은 L000보다 올바른 부호·크기의 반응이 강해지고 잔여 오차가 낮은지, 그 경향이 다른 seed·split·반경에서도 유지되는지다.

### 7.3 집계·필수 범위

- **필수:** L000, L1E4, 신규 L3E5/L3E4의 세 seed에서 best_raw·last 모두 V1. N2 last 추가.
- **강한 감독 대조:** 기존 L1E2 seed1234를 같은 probe로 측정.
- P0의 aligner response는 **N/A**다. 가상의 CNN이 0을 예측했다고 구현하지 않는다. EPE의 no-response 참고선은 별도로 둔다.
- 먼저 scene별 probe 평균을 구한 뒤 scene 동등 가중으로 집계한다. 반경·방향 수를 독립 scene 수로 세지 않는다.
- Scalar mean/P50/P90뿐 아니라 scene별 원값, 실패·boundary 비율을 남긴다.
- Calibration jitter는 매번 원본에서 생성한다. 앞 probe의 변형 결과를 다시 변형하지 않는다.

---

<a id="v2"></a>
## 8. V2 — 모달리티 대응과 shortcut 검증

### 8.1 Aligner만 검사하는 세 변환

Mainline의 MS·GT·U-Net 학습을 변경하지 않는 **추론 진단**이다.

| Probe mode | 입력 | 기대 상대 변화 |
|---|---|---|
| `pan_only` | A(W(P,e), M) | `c_e−c_0 ≈ −e` |
| `ms_only` | A(P, W(M,e)) | `c_e−c_0 ≈ +e` |
| `common` | A(W(P,e), W(M,e)) | `c_e−c_0 ≈ 0` |

일반식은 `c(eP,eM)−c(0,0)≈eM−eP`다. MS-only에서는 **upsampled M의 모든 8band에 같은 HR shift**를 적용한다. LRMS 생성 파이프라인이나 residual base를 재정의한 실험이 아니다.

PAN-only는 이미 학습한 과제이고, MS-only/common은 학습하지 않은 변환에 대한 추가 검사다. 실패해도 기존 PAN-only 학습 성공을 지우지 않지만, 일반적인 cross-modal 상대위치 추정이라는 주장의 범위는 좁아진다.

### 8.2 유효 영역·common-shift의 한계

기존 학습 view margin4를 임의로 바꾸지 않는다. Probe는 full input에서 먼저 만들고 그 뒤 같은 aligner view를 취한다. 최대 2 px와 bicubic support를 실제 좌표로 추적한다.

Common shift에서 완전한 0을 강제 합격선으로 쓰지 않는다. 유한 crop·padding·공간 정규화·pooling 때문에 두 영상이 함께 이동해도 CNN이 보는 내용은 달라질 수 있다. 가능한 경우 원본의 충분한 주변 여유에서 생성한 probe를 추가하고, 없는 주변 픽셀을 관측값처럼 만들지 않는다. Interior-only 진단을 추가하면 **모든 비교 모델에 같은 view**를 쓰며 별도 `view_id`로 기록한다.

### 8.3 Shortcut 대조

**핵심 모델:** 기존 L000/L1E4 세 seed 및 신규 L3E5/L3E4 세 seed의 동일 checkpoint. 우선 best_raw에서 전수, last에서 같은 핵심 변환을 확인한다. N2 last를 기준으로 추가한다. 무거운 kernel 조합은 우선 seed1234의 전체 λ에서 수행하고, 양성/이상 징후를 확인 seed에 동일 조건으로 재검증한다.

| 대조 | 방법 | 해석의 한계 |
|---|---|---|
| MS swap | split별 고정 derangement로 다른 장면의 M 사용 | 틀린 pair의 절대 정합 GT는 없음 |
| Constant MS | band별 공간 평균값만 남김 | 구조 제거와 입력분포 변화가 함께 있음 |
| 다른 보간 | Probe 생성만 bilinear로 변경, 학습된 correction warp는 그대로 | Kernel 민감도이지 단독 실패 증명 아님 |
| Padding | border/reflection probe 또는 실제 source margin 비교 | view/support 차이를 별도로 기록 |

서로 다른 장면으로 swap할 때 실제 원본 source ID가 같지 않은지 확인한다. Patch 여러 개가 같은 장면에서 왔다면 가능한 한 source가 다른 patch로 바꾼다.

MS swap/constant 조건의 반응은 정상 pair EPE 표에 합치지 않는다. 반응이 남으면 PAN-only 지름길 가능성을 조사하고, 반응이 줄었다고 정확한 native 정합이 증명됐다고 쓰지 않는다.

---

<a id="v3"></a>
## 9. V3 — native 위치·선명도·보간 영향

**필수 이유:** L1E4는 raw HQNR이 높지만 raw fSCC가 P0보다 모든 seed에서 낮았다. 추가 offset 상쇄만으로 이 문제를 설명할 수 없다. [S1 §2.6]

### 9.1 독립 구조 추정기로 보정 전후 비교

동일 checkpoint에서:

```text
원본 P, M                 → independent_estimator → native residual proxy before
W(P, c0), 같은 M          → 같은 estimator         → native residual proxy after
```

기존 검증된 **Scharr-ZNCC**와 **census 또는 phase-correlation 중 확보된 독립 구현**을 사용한다. Secondary 방법은 출력 점수를 보기 전 calibration에서 부호·단위·identity를 확인한 뒤 고정한다. 유리한 estimator를 scene마다 고르지 않는다.

**실행 계약:** 비교 해상도/blur, moving·reference 순서, sampling phase, grid 범위, subpixel fit, confidence, HR/LR 변환을 `estimator_contract.json`에 저장한다. 실제 모델은 PAN→MS correction을 내므로 audit이 반대 방향을 출력하면 먼저 부호를 canonical PAN→MS로 변환한다. 기존 1.79 px 표와 단순히 부호·크기를 섞지 않는다.

각 방법에 identity·알려진 shift 검사를 수행하고 측정 noise floor와 descriptor간 불일치를 기록한다. 실패한 추정값을 정합 label로 사용하지 않는다. **0.557이나 1.79를 장면별 GT로 대입하지 않는다.**

기록: 장면별 before/after 벡터·길이, confidence, 두 추정기의 일치/불일치, 보정 후 개선 scene 수. 저신뢰 scene도 목록에서 없애지 않고 상태를 남긴다. Confidence 필터를 적용한 보조 분석과 전체 scene의 상태표를 분리한다.

이 검사는 native 위치에 대한 **proxy 근거**다. 센서 ground truth가 없으므로 true registration error를 측정했다고 주장하지 않는다. [S2 §7.1]

### 9.2 RR의 동일 band GT 경계

P0/L000/L1E4/새 λ의 같은 seed·checkpoint에서 GT를 이동하지 않고 비교한다. RR test 20장·8band를 기본으로 하고, 시각 panel은 GT 또는 입력만으로 사전에 고른 위치를 사용한다.

**필수 기록**

- 원래 ERGAS·SAM·PSNR·SSIM·RR SCC/Q8.
- GT가 정한 고정 edge mask에서의 intensity error와 gradient error.
- 같은 band의 경계 위치 차이와 폭, double-edge·overshoot 여부.
- Mask 밖 평탄 영역 오차와 edge 영역 오차를 따로 제시.

기존 `tools/align_after_training_diag.py` Part B와 edge profile 구현이 있으면 정의·hash를 확인해 그대로 쓴다. 이름만 확인하고 같은 지표를 재현했다고 쓰지 않는다.

**기존 edge helper가 없을 때의 신규 진단 규격 `edge_profile_v1`:** 이 경우 구현·calibration 시간을 V3 안에 포함하며, 기존 Part B와 다른 이름으로 저장한다.

1. GT의 band별 Scharr magnitude로 강한 edge 후보를 만들고 공간 cell별로 균등하게 고른다. 각 장면의 상위 30% gradient를 후보로 하며, 16×16 HR cell에서 가장 강한 위치 하나를 선택하는 기본 규칙을 calibration 후 출력 확인 전에 고정한다.
2. GT gradient 방향을 normal로 사용해 중심 ±4 HR px, 간격 0.25의 profile을 GT와 출력에서 동일하게 읽는다. Source support가 없는 후보는 GT 기반 공통 목록에서 제외한다.
3. GT profile이 단순한 한 경계이고 양쪽 plateau contrast가 충분한 경우에만 GT 대비 50% crossing 위치 및 10–90% 폭을 잰다. Contrast/단순 경계의 판정값은 train calibration과 GT만으로 고정한다.
4. Candidate 출력이 crossing을 만들지 못하거나 여러 crossing을 만들면 `missing/multiple_crossing`으로 기록한다. 좋은 profile만 골라 평균하지 않는다.
5. Intensity·gradient error는 전체 고정 edge mask에 계산하므로 복잡한 GT 경계도 성능 표에서 사라지지 않는다. Profile 분석의 선택률·실패율을 함께 보고한다.

이 규격은 새 보조 진단이지 학습 loss나 checkpoint selector가 아니다. GT가 정한 평가 mask를 학습에 역전파하지 않는다.

**해석:** 위치가 GT에 가까워지고 폭이 유지되면 위치 개선 설명을 지지한다. 위치 차이 감소 없이 폭·오차가 커지면 detail 손실을 의심한다. 두 현상이 함께 나타날 수도 있다. fSCC만으로 구분하지 않는다.

### 9.3 보간·선명도 대조

Native warp는 한 번이지만 interpolation 영향은 남는다. 다음은 **추가 학습 없는 사후 진단**이다.

- 실제 P→W(P,c0)의 gradient energy, frequency energy, 범위 초과/overshoot.
- 동일 sampler의 zero-shift identity 및 coordinate-ramp 검사.
- 별도 train calibration에서 energy 변화를 근사한 **zero-phase smoothing** PAN 입력 대조.

Energy-match blur를 구현할 때의 신규 고정 규칙:

```text
calibration: 고정 train 256 patch, common interior
target: 각 checkpoint의 실제 native warp 전후 Scharr squared-energy ratio
sigma candidates (HR px): 0 / 0.1 / 0.2 / 0.3 / 0.4 / 0.5 / 0.75 / 1.0
fit criterion: calibration 평균 energy ratio 차이 최소
fit에 FR HQNR·fSCC·candidate RR 출력 점수를 사용하지 않음
```

Blur kernel은 중심대칭·합 1, support·padding을 기록한다. Bicubic warp가 에너지를 증가시키거나 blur 후보가 target을 맞추지 못하면 **`unmatched_blur_control`**로 남긴다. 단일 에너지 비율을 맞췄다고 warp와 blur가 모든 주파수에서 동등하다는 뜻은 아니다. Mainline 입력·학습은 변경하지 않는다.

PAN에서 얻은 한 에너지 비율만으로 여러 MS band 출력의 선명도 보존을 확정하지 않는다. 반드시 RR GT 경계와 함께 해석한다.

### 9.4 V3의 최소 완료 범위

기존 **P0/L000/L1E4 세 seed의 best/last**에서 native before/after proxy와 RR edge 오류를 완료한다. 신규 양쪽 λ도 동일 집계를 적용한다. 시간이 부족하면 세밀한 추가 주파수 panel·여러 blur 커널을 줄이되, 이 기본 위치·edge 검사를 통째로 생략하지 않는다.

---

<a id="v4"></a>
## 10. V4 — correction 치환·복원 stress

### 10.1 같은 가중치의 correction 치환

U-Net에는 항상 해당 장면의 정상 MS를 주며, 적용할 correction만 바꾼다. 새 학습 없이 아래 개입을 수행한다.

| Intervention | 적용값 | 질문 |
|---|---|---|
| `learned` | c_i | 실제 동작 |
| `zero` | 0 | 이 모델의 보정이 유효한가? |
| `wrong_sign` | −c_i | 방향이 의미 있는가? |
| `scene_shuffle` | 다른 장면 c_j, 고정 derangement | sample별 예측의 추가 기여가 있는가? |
| `constant_calibration` | 별도 calibration에서 얻은 고정 2D 벡터 | 공통 offset으로 근사 가능한가? |

Warp 횟수와 kernel은 원래 native 경로와 동일하게 한 번으로 유지한다. Zero control에서는 graph가 필요 없는 추론이므로 원 PAN 직접 전달과 zero sampler의 수치 동치를 먼저 확인한다.

Constant는 calibration의 원좌표에서 얻은 **벡터 통계**다. Augmentation된 signed vector를 그대로 평균해 0으로 만드는 오류를 피한다. Train64와 FR512의 분포 차이는 남으므로 그 상수가 FR 최적값이라고 가정하지 않는다.

별도 적절한 calibration이 없으면 FR 예측값에서 평균을 얻는 constant를 **test-informed post-hoc 진단**으로만 표시한다. 새 train-derived 방법이나 공식 결과로 사용하지 않는다. 0.55는 길이 하나일 뿐이므로 `(0.55,0.55)`로 치환하지 않는다.

**해석 제한:** 모든 개입은 같은 모델이 학습한 입력조건을 바꾸는 실험이다. 정합 없는 P0 또는 trainable constant를 처음부터 학습한 대조를 대체하지 않는다. Correction 분산이 작으면 scene-shuffle은 약한 개입이므로 작은 차이를 “동적 정합 불필요”로 확정하지 않는다. V1/V2와 함께 본다.

### 10.2 실제 복원 강건성 stress

Aligner response가 좋아도 U-Net의 출력이 안정적인지는 별도로 본다. **학습에서는 계속 native-only reconstruction**, 아래는 frozen inference다.

고정 반경 `r={0.5,1,2}` HR px, 8방향에서:

\[
P_e=W(P,e),\quad c_e=A(P_e,M),\quad
\hat Y_e=M+F([W(P_e,c_e),M]).
\]

RR은 GT Y를 움직이지 않고 같은 고정 유효 영역에서 오차를 계산한다. FR은 **native 원 P**와 **native frozen-N2 reference**를 고정하고 비교한다. ε마다 평가 reference를 움직여 점수를 유리하게 만들지 않는다. `raw_input_ref=P_e`나 self-aligned stress 점수는 별도 열이며 native reference stress와 혼합하지 않는다.

FR stress의 기존 V96을 재검증해 고정하고, 비교할 native도 같은 V96에서 측정한다. Native V64 점수에서 stress V96 점수를 빼지 않는다. RR stress ROI는 ratio4 grid와 두 warp support를 고려한 기존 공통 ROI를 사용하고 manifest에 고정한다.

**보간 오차 기준선:** 알려진 e를 이용해 diagnostic correction `c0−e`를 적용한 출력도 계산한다. 이는 synthetic 변화만 정확히 상쇄하는 좌표 기준선으로, 원래 c0가 참값이라는 의미는 아니다. 두 번 raster sampling의 손상이 남으므로 이를 native output과 동일하다고 가정하지 않는다. 이 기준은 모델 입력에 e를 제공하는 실제 추론 방법으로 쓰지 않는다.

V4는 L000/L1E4와 신규 λ에서 같은 scene·같은 probe를 사용한다. 우선 세 seed의 best_raw를 전수 검사하고 exact50K의 주요 learned/zero 및 stress 집계를 같은 방식으로 확인한다. 후보별 잘 나오는 scene만 고르지 않는다.

---

<a id="implementation"></a>
## 11. 학습 구현·gradient·재현성

### 11.1 학습 코드 변경은 최소화한다

PALS24 trainer의 scalar λ를 config에서 `3e-5` 또는 `3e-4`로 지정한다. Aligner, backbone, warp 위치, loss 종류·reduction, native 입력 경로, LR scheduler를 변경하지 않는다. New diagnostics는 training graph와 RNG를 오염시키지 않게 분리한다.

아래 코드는 **통합 의미를 설명하는 pseudocode**다. Helper 이름은 실제 repository API와 연결해야 하며, wrapper가 이미 MS residual을 더하는지 확인해 중복 합산하지 않는다.

```python
# t: completed optimizer updates, 0-based; not a microbatch index.
optimizer.zero_grad(set_to_none=True)

M = existing_bicubic(S)
c0 = aligner_with_existing_view(P, M)       # Same margin4 and normalization.
P0 = existing_warp(P, c0)                   # Keep gradient to c0.
residual = unet_residual(torch.cat([P0, M], dim=1))
pred = M + residual
Lrec = (pred - Y).abs().mean()
Loff = None
loss = Lrec

if (t % 2 == 1) and lambda_off > 0:
    with torch.no_grad():
        eps = sample_disk_hr(
            P.shape[0], radius=2.0, generator=offset_rng,
            device=P.device, dtype=torch.float32,
        )
        Pe = existing_warp(P, eps)         # Always start from original PAN.
    ce = aligner_with_existing_view(Pe, M)  # Same trainable aligner.
    target = c0.detach() - eps             # Detach only this target path.
    Loff = (ce - target).abs().mean()       # B x 2 component mean, HR px.
    loss = Lrec + lambda_off * Loff

# Optional diagnostics must not modify .grad, model buffers or training RNG.
loss.backward()
existing_clip_policy_if_any()
optimizer.step()
existing_scheduler_step()
t += 1
```

Trainable native warp를 보정량이 우연히 0이라는 이유로 bypass하지 않는다. Grid를 만들면서 `.item()`/NumPy/새 tensor 재포장으로 c0의 graph를 끊지 않는다. Crop한 56² PAN을 먼저 jitter하여 artificial border를 만들지 않고, full PAN을 변형한 뒤 margin4를 취한다.

### 11.2 Gradient 진단

기존 PALS24의 기록 cadence를 계승하고, 비용이 큰 `autograd.grad` 분해는 고정된 diagnostic batch·update에서 수행한다. 새 기본 분해 지점은 **update 1, 1K, 5K, 10K, 25K, 50K 직전의 active update**로 둔다. 실제 active parity에 맞춘 정확한 목록을 저장한다. 기존 코드가 이미 동등한 항목을 기록한다면 중복하지 않는다.

\[
g_r=\nabla_\phi L_{rec},\quad g_o=\nabla_\phi L_{off},\quad
q=\frac{\|\lambda g_o\|_2}{\|g_r\|_2+\eta},\quad
\cos(g_r,g_o)=\frac{\langle g_r,g_o\rangle}{\|g_r\|\|g_o\|}.
\]

기록: raw Lrec/Loff, weighted Loff, `norm(g_r)`, `norm(g_o)`, `norm(lambda*g_o)`, cosine, native c0 변화. 분모가 실질적으로 0이면 cosine/ratio는 N/A와 원 norm을 기록한다. 큰 ratio만으로 실패 처리하지 않는다.

`off_unet_grad_absent=True`, `rec_aligner_graph_connected=True`, `native_target_sg=True`를 실제 wrapper에서 점검한다. Aligner parameter grad의 합이 두 경로의 합과 수치적으로 일치하는지도 smoke에서 검사한다. Gradient norm 비율로 λ를 자동 변경하지 않는다.

### 11.3 RNG·resume·state

Data order, augmentation, offset, probe RNG를 분리한다. 같은 seed·update·sample은 두 신규 λ 및 기존 기준과 동일 ε를 가져야 한다. Probe가 실행됐다는 이유로 training sample이나 ε sequence가 달라지지 않게 한다.

Resume에 포함:

```text
model / optimizer / scheduler / scaler
completed_optimizer_updates / parity / sampler cursor
Python·NumPy·CPU·CUDA RNG / augmentation RNG / offset RNG
lambda_off / radius / frequency / SG policy / model state hash
selector historical_max / candidate metadata / evaluation-grid hash
code·resolved-config·dataset·donor·initial-state hashes
campaign used-time ledger / partial-run status
```

같은 run의 resume는 모든 정책과 hash가 동일할 때만 허용한다. λ 변경·scheduler 재시작·donor 변경은 별도 protocol이다. 기존 `L1E4/last`에서 이어 학습하려고 설정을 바꾸어 이 50K 독립 비교에 섞지 않는다.

### 11.4 후반 전체 모델 연장 학습·cap은 제외

기존 후속안 [S2 §10–11]을 유지한다. 완료 L1E4 전체 모델을 더 학습할 때는 동일 source checkpoint에서 동일 기간 연장한 무변경 대조, optimizer/LR restart 규약이 필요하므로 별도 단계로 둔다.

Prediction 양쪽 norm을 0.55 px 이하로 제한하면 차이는 최대 1.10 px다. 현재 최대 2 px의 relative supervision과 양립하지 않는 표본이 생긴다. 따라서 이번에는 hard cap이나 c_apply/c_pred 분리, FiLM, feature warp를 추가하지 않는다.

---

<a id="decision"></a>
## 12. 선택·해석·종료 규칙

### 12.1 성능과 정합 주장을 분리한다

이번에는 두 λ를 세 seed 모두 계산하고, 결과를 보고 중간에 grid를 늘리지 않는다. 다음 이름을 분리해 기록한다.

- **`performance_leader`**: 동일 selection grid의 각 seed best_raw HQNR을 먼저 얻고, λ별 3-seed 평균을 비교한 수치상 선두. fSCC는 HQNR 동률에서만 사용한다.
- **`working_reference`**: 현재 방법 L1E4. 작은 수치차만으로 이것을 확정 최적값으로 교체하지 않는다.
- **`alignment_evidence`**: 같은 checkpoint의 V1–V4가 뒷받침하는 정합 능력의 주장 범위.

3-seed 평균, 표본표준편차, 각 seed의 대응 차이를 모두 제시한다. 전체 60개 scene×seed 또는 32방향 probe를 서로 독립 seed처럼 세어 신뢰도를 부풀리지 않는다. 같은 donor에 조건부인 결과임을 명시한다.

### 12.2 λ 국소 안정성 판정

| 관찰 | 정리 |
|---|---|
| 새 두 λ도 L1E4 근방의 raw 성능·반응·위치 경향을 유지 | 현재 이득이 단일 λ의 극단적 우연에만 의존한다는 우려가 약해짐 |
| L1E4만 좋고 양옆에서 하락 | 좁은 최적 구간 또는 seed/selection 민감성. 더 많은 λ를 자동 생성하지 않음 |
| 새 λ가 L1E4 대비 raw 평균·대응 차이에서 분명히 개선 | 새 성능 후보로 기록하고 동일 checkpoint 진단 확인 |
| Raw 차이는 경험적 0.0031 내부 | 수치상 순위는 보고하되 우월/동등성 확정은 보류. L1E4를 실무 기준으로 유지 가능 |
| 모든 새 λ가 불리 | 기존 L1E4를 유지하고 이번 local grid 종료 |

0.0031은 기존 규약의 경험적 분류선이다. 개별 seed와 평균에 적용한 결과를 따로 제시하고 통계적 유의성·비열등성 검정이라고 부르지 않는다. Best가 다른 update에서 나왔다는 사실을 결론에 포함한다.

### 12.3 정합 검증의 결론 수준

| 근거 | 허용되는 주장 |
|---|---|
| V1에서 L000 대비 올바른 signed response·낮은 EPE | **추가 PAN 변위 대응을 더 보존** |
| V2의 반대 MS 이동·common shift·shortcut 대조도 일관 | **모달리티 상대위치 정보를 이용한다는 근거 강화** |
| V3의 독립 residual proxy·RR 경계 위치 개선 | **Native 위치 개선을 지지하는 근거**; 센서 GT 정확도는 아님 |
| 위치 개선과 동시에 RR 폭/edge 오차 유지 | 정합과 detail 보존의 양립 근거 |
| V4 learned가 zero/wrong/shuffle보다 유리 | 학습된 보정의 적용이 해당 모델에 유용하다는 근거 |
| HQNR만 개선, response/위치 개선 불명확 | 유효한 성능 방법으로 남기되 정합 기여 설명 제한 |
| EPE 개선과 동시에 native edge·RR detail 악화 | 상대 과제의 성능과 실제 복원의 충돌이 남음 |
| Blur/공통 보정으로 유사한 효과 | 동적 정합만의 기여 주장은 약화; 별도 학습 대조 필요 |

L1E4의 완전한 변위 보상(B=−I)을 성공의 필수조건으로 새로 강제하지 않는다. 반대로 양의 λ를 썼다는 이유만으로 정합 능력이 보존됐다고 쓰지 않는다. Geometry proxy 실패를 raw HQNR 표에서 숨기지 않고, raw 점수 손실을 aligned로 만회 판정하지 않는다.

### 12.4 다음 단계 분기

- 성능·반응·native 위치·detail이 함께 지지되면 현재 구조와 λ를 잠정 고정하고, 이후 전체 모델 후반 fitting 또는 KD 통합의 출발 asset을 선정할 근거가 된다.
- 성능은 좋지만 native 위치가 불명확하면 성능 경로는 보존하고 “물리적으로 정확한 정합”이라는 주장을 보류한다.
- fSCC 감소가 detail 손실과 연결되면 다음 변경은 단순 λ 추가보다 보간/위치정보 활용 방식을 검토한다.
- Hard cap·conditioning·feature warp·새 센서 학습을 이번 18h 안에 무단으로 확대하지 않는다.

**종료 시 남길 답:** 현재 `1e-4`가 주변에서도 안정적인가? 같은 best_raw가 실제 반응을 보존하는가? fSCC 비용은 위치·detail 중 무엇과 연결되는가? 다음 단계에 넘길 checkpoint의 정확한 hash와 trade-off는 무엇인가?

---

<a id="config"></a>
## 13. Config·queue template

아래는 repository의 실제 key를 보장하지 않는 **실행 계약 template**다. 기존 PALS24 config를 복사하고 허용된 λ·run ID·경로·seed만 변경한다. `null` 필수값은 gate에서 채운다.

```yaml
campaign_id: PALSV18_W112D123_N2LAST_R200_v1
training_semantics: PALS_W112D123_N2LAST_R200_v1
server: s1
gpu_count: 1
parallel_training_runs: 1
budget:
  cap_gpu_hours: 18.0
  include_previous_spent_hours: false
  auto_add_previous_unused_9p8_hours: false
  gate_hours: 0.8
  pre_validation_hours: 2.0
  new_training_hours: 10.2
  post_validation_hours: 3.5
  reporting_hours: 0.5
  buffer_hours: 1.0
  reservation_per_full_run_hours: 1.7
  eta_margin_fraction: 0.10
  drop_new_seed_blocks_in_order: [2025, 7777]
  allow_unequal_completed_updates: false
  allow_new_unplanned_lambdas: false

initialization:
  donor_run: PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT
  donor_checkpoint_kind: last
  donor_optimizer_update: 50000
  copy_aligner_only: true
  donor_path: null
  donor_sha256: null
  initial_unet_registry: null
  reuse_registry: null
  resume_from_completed_L1E4: false

model:
  width: 112
  depth: [1, 2, 3]
  input_channels: 9
  output_channels: 8
  aligner_arch: inherit_verified_PALS24
  aligner_trainable: true
  aligner_view_margin_hr: 4
  attention: false
  pan_reconstruction: false
  mode_modulation: false
  output_inverse_warp: false
  ms_base_warp: false
  gt_warp: false
  correction_cap: null

training:
  max_optimizer_updates: 50000
  effective_batch_size: 48
  unet_lr: 0.0001
  aligner_lr: 0.00001
  weight_decay: 0.01
  optimizer_other_settings: inherit_verified_PALS24
  scheduler: inherit_verified_PALS24
  warmup_updates: 100
  precision_policy: inherit_verified_PALS24
  native_reconstruction_every_update: true
  reconstruction_weight: 1.0
  reconstruction_loss: mean_L1
  geometry_weight: 0.0
  kd_weight: 0.0

offset:
  lambda: 0.00003  # overridden only by the fixed run matrix
  weight_schedule: constant
  every_optimizer_updates: 2
  active_remainder: 1
  stop_gradient_native_target: true
  detach_native_reconstruction: false
  feed_jittered_pan_to_unet: false
  reduction: mean_over_batch_and_two_components
  distribution: uniform_disk_area
  radius_hr: 2.0
  unit: current_PAN_pixel
  coordinate_order: dy_dx
  apply_after_common_augmentation: true
  independent_rng: true

warp:
  mode: bicubic
  padding_mode: border
  align_corners: false
  implementation_hash: null

metrics:
  fr_manifest_id: fr_mat20
  fr_scene_count: 20
  primary_view: raw_original
  primary_checkpoint: best_raw
  primary_metric: hqnr_raw_original
  secondary_metric: fscc_raw_original
  secondary_split: FR
  secondary_reference: native_original_PAN
  evaluator_hash: null
  fscc_function_key: null
  selection_grid_path: null
  selection_grid_sha256: null
  aggregate: mean_of_scene_products
  tie_band_hqnr: 0.0001
  tie_anchor: running_maximum
  empirical_hqnr_method_margin: 0.0031
  keep_original_selector_results: true
  export_matched_grid_results: true
  exact_last_update: 50000
  retain_evaluated_native_checkpoints: true
  aligned_for_primary_selection: false
  views: [raw_original, raw_v64, aligned_self_v64, aligned_fixedN2_v64]
  exclude_early_checkpoints: false

validation:
  manifest_path: null
  checkpoint_roles: [best_raw, exact50k_last]
  train_size_probe_count: 64
  calibration_train_count: 256
  rr_scene_count: 20
  fr_scene_count: 20
  response_radii_hr: [0.25, 0.5, 1.0, 2.0]
  directions_per_radius: 8
  zero_probe_for_identity_only: true
  response_modes: [pan_only, ms_only, common]
  report_signed_2x2: true
  legacy_metric_definitions_path: null
  stress_radii_hr: [0.5, 1.0, 2.0]
  stress_reference: native_fixed
  stress_fr_margin_hr: 96
  edge_selection_source: GT_or_input_only
  allow_test_metric_fitting_of_blur_or_constant: false
  allow_silent_scene_exclusion: false

new_runs:
  - {case_id: R-L3E5-S1234, seed: 1234, lambda_off: 0.00003}
  - {case_id: R-L3E4-S1234, seed: 1234, lambda_off: 0.0003}
  - {case_id: R-L3E4-S7777, seed: 7777, lambda_off: 0.0003}
  - {case_id: R-L3E5-S7777, seed: 7777, lambda_off: 0.00003}
  - {case_id: R-L3E5-S2025, seed: 2025, lambda_off: 0.00003}
  - {case_id: R-L3E4-S2025, seed: 2025, lambda_off: 0.0003}
```

이 template의 U-Net initial registry에는 seed별 실제 state_dict path·hash를 넣는다. Donor나 evaluator null을 남긴 채 실행기가 알아서 추정하게 하지 않는다. 해당 code path의 config diff는 허용 목록을 검사하고 저장한다.

---

<a id="outputs"></a>
## 14. 산출물·인계·범위 제한

### 14.1 필수 산출물

```text
work_dir/_palsv18_campaign/
  campaign_manifest.yaml
  campaign_budget_ledger.json
  source_asset_registry.json
  reuse_registry.json
  initial_tensor_registry.json
  metric_contract.json
  estimator_contract.json
  selection_grid.json
  probe_manifest.json
  checkpoint_manifest.csv
  gate_results.json
  official_best_raw_results.csv
  matched_grid_best_raw_results.csv
  paired_seed_differences.csv
  response_signed_2x2.csv
  offset_component_mae_epe.csv
  native_shift_proxy_before_after.csv
  shortcut_controls.csv
  correction_interventions.csv
  synthetic_stress_native_reference.csv
  edge_and_frequency_metrics.csv
  figures/
  alignment_evidence_summary.md
  final_report.md
```

각 신규 run은 기존 PALS24 산출물에 더해 본 campaign ID·사용 시간·code/config/hash·완료 여부를 연결한다. 모든 metric 표에 checkpoint ID와 split·reference·ROI를 명시한다. 그림은 원본 또는 GT 기준의 고정 crop을 포함하고, 사후 선정 예시라면 선정 규칙을 표시한다.

### 14.2 최종 보고의 네 표

**A. 공식 성능**  
λ×seed별 best_raw의 update/HQNR/fSCC/Dλ/Ds/RR와 P0·L000·L1E4 대비 차이. 과거 selector 차이가 있으면 original과 matched-grid를 함께 제시한다.

**B. 같은 checkpoint의 정합 능력**  
A에 있는 정확한 best hash, 그리고 별도 exact50K에서 native correction, signed B, EPE/MAE, V2 반응을 함께 제시한다.

**C. 실제 위치·선명도·보간**  
Native independent proxy, RR edge 위치·폭·오차·실패율, raw/aligned fSCC, energy-match 대조의 성립 여부, V4 개입 결과.

**D. 실행·한계**  
실제 18h 내 사용 시간, 완주/재사용/보류 목록, source hash, definition mismatch, estimator 한계, 고정 donor·기존 test 사용 여부.

### 14.3 운영 정리

[S1 §6]은 과거 `pals24` gate가 남아 있다고 보고했다. **현재 서버 상태를 확인한 뒤**, 종료된 token만 정리하고 새 `palsv18` token으로 관리한다. 이 문서가 실제 서버 파일을 삭제하거나 token을 활성화한 것은 아니다. 중복 queue 실행을 막고 종료 시 이번 token도 닫는다.

문서에 나온 `tools/pals24_report.py`, `tools/best_on_grid.py`, `tools/campaign_gate.py`는 보고서가 언급한 기존 도구다. 경로·CLI를 서버에서 확인한다. 확인되지 않은 새 shell 명령을 실행 가능한 것처럼 복사하지 않는다.

### 14.4 이번 범위 밖

완료 L1E4 전체 모델의 추가 10K/20K 연장, 새 N2 donor 사전학습, 다른 센서 학습, output inverse, MS correction mainline, hard cap, conditioning/feature warp, local alignment, PAN reconstruction, KD/variance 결합은 이번 기본 18h에 포함하지 않는다.

**현재 방법을 설명하는 문장:**

> 사전학습된 작은 PAN aligner가 PAN–MS의 전역 보정량을 예측한다. 원래 PAN을 그 값으로 보정하고 MS와 함께 U-Net에 넣어 HRMS를 복원한다. Native reconstruction은 두 모듈을 함께 학습시키며, 알려진 random PAN 변위의 전후 보정량 관계를 약한 offset consistency로 감독한다. Jittered PAN은 aligner 연습에만 사용한다. 이번에는 현재 가중치의 근방 안정성과, 같은 checkpoint의 상대 반응·native 위치·detail 보존을 검증한다.

---

<a id="sources"></a>
## 15. 근거 문서와 적용 우선순위

| ID | 자료 | 본 문서에서 사용한 부분 |
|---|---|---|
| **S1** | `2026-09-13_s1_pals24-lambda-sweep.md` | 최신 결과, λ 근방 제안, fSCC 비용, checkpoint 시점, 평가 cadence 차이, runtime |
| **S2** | `PAN_L1E4_Refinement_and_AlignmentValidation_2026-09-13.md` | 이번 개정의 직접 원안: V0–V4, 동일 checkpoint 검증, 기존 4회·9.8h 안 |
| **S3** | `PAN_P2_P3_LambdaSweep_MetricAware_24GPUh_Plan_2026-09-12_v2.md` | 모델·초기화·forward·loss·RNG·metric 의미 |
| **S4** | `PAN_Aligner_Directions_2026-09-12.md` | 이전 가설과 inverse warp 대조. S1이 수정한 일반 결론은 그대로 채택하지 않음 |

해석 우선순위는 **현재 사용자 18h 지시 → S1의 최신 관측 → S2의 후속 방향 → S3의 유지되는 구현 계약**이다. S1의 기전 설명은 설명 가설로 보존하며, 관측된 상대 EPE를 절대 정합의 참값으로 바꾸지 않는다.

직접 원안의 SHA-256:

```text
PAN_L1E4_Refinement_and_AlignmentValidation_2026-09-13.md
c6bff20f60ec41c6557efee92e95dc048fc0335806b6275e6a7a0c9ca3f6559d

PAN_P2_P3_LambdaSweep_MetricAware_24GPUh_Plan_2026-09-12_v2.md
70b667003bf7249a2382e56b155ed9dd3cd061b6b1c4bba811c87add89fcc9fc

2026-09-13_s1_pals24-lambda-sweep.md
344a77e7342de5ed84a529e5ba53a8474393124462829c9ab13e6b1bfa700679
```

**신규 설계임을 구분할 항목:** 18h 배정, 기본 6회 편성, 진단 표본 수·우선순위, 고정 probe/edge fallback/blur calibration의 세부 실행값은 이번 계획의 선택이다. 이미 성능이 검증된 최적값으로 제시한 것이 아니다. 실제 서버 source code·checkpoint·원시 CSV의 재평가 결과는 실행 후 보고서에 추가한다.
