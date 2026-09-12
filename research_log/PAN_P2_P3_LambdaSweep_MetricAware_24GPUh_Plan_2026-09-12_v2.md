# PAN aligner: P2 기반 offset-consistency 가중치 비교 — 24 GPU-hour 실행 계획 (v2)

**작성일:** 2026-09-12  
**학습 Protocol:** `PALS_W112D123_N2LAST_R200_v1` — 기존 학습 의미 유지  
**Campaign:** `PALS24_W112D123_N2LAST_R200_v2`  
**대상:** WV3 / s1 / 탐색 seed 1234 / 확인 seed 7777·2025  
**예산:** 이번 캠페인 추가 총 **24 GPU-hours 상한**, s1 단일 GPU 순차 실행 기준  
**개정 범위:** 예산·실행 순서·seed 반복·결과 집계. λ 후보·모델·loss·50K 학습량·평가 규약은 유지  
**문서 상태:** 기존 승인 계획의 24시간 예산 개정본. 원본 v1은 보존한다. v1의 보고서 검토·Sheet 조회 이력을 계승하며, 이번 개정에서 Sheet를 새로 조회하거나 서버 구현·학습·원본 MAT 재평가를 수행하지 않았다.

> **목표:** P2의 native HRMS 복원 적합성을 기준으로, P3의 상대 offset consistency를 약하게 추가했을 때 복원 성능을 보존하면서 입력 변위에 대한 반응을 유지할 수 있는지 확인한다.
>
> **학습 연결:** reconstruction → U-Net + aligner, offset consistency → aligner만. 두 모듈은 첫 update부터 공동 학습한다. 복원 U-Net에는 추가 jitter 없는 PAN을 aligner로 보정하여 넣고, jittered PAN은 같은 aligner의 보조 연습에만 사용한다.
>
> **평가 결정:** 공식 주 판정은 **원본 PAN을 참조한 논문 세트 20장의 best checkpoint HQNR → SCC**다. `aligned_*`는 진단용이다. 정확한 50K `last`의 복원·반응도 반드시 보고하지만, 주 판정을 last나 aligned 점수로 사후 교체하지 않는다. 최신 보고서 §6을 우선한다. [S1, §6]

### v2 변경 요약

- **사용자 확정 변경:** 캠페인 예산을 추가 총 24 GPU-hours로 설정한다. 한 case를 24시간 학습하거나 기존 NF16의 사용 시간을 소급 합산한다는 뜻은 아니다.
- **24시간 배정안:** seed 1234에서 기존 중간 λ 세 개를 비교하고, 사전 규칙으로 정한 하나의 λ*를 seed 7777·2025에서 각각 **P0(no-align), P2(λ=0)**와 대응 비교한다. 이 반복 배치는 이번 개정의 설계 결정이며 완료 결과가 아니다.
- **변경하지 않는 것:** λ = 0 / 0.0001 / 0.001 / 0.003 / 0.01, W112–D123, N2 exact50K aligner, native reconstruction, SG consistency, b=2, loss 빈도·학습률·case당 50K, raw best HQNR 주 판정.
- **시간 사용 원칙:** 탐색값·새 아키텍처를 늘리지 않고 대조군·seed 반복·metric 재검증에 사용한다. 검증 실패나 유효 후보 부재 시 24시간을 채우기 위해 새 방법을 임의로 추가하지 않는다.
- **원본 계획:** `PAN_P2_P3_LambdaSweep_MetricAware_Plan_2026-09-12.md` / SHA-256 `367ddd2041338d3ebe2c3a6c633d47ee542aa087a8a9d4f7388e150368609cfd`.

---

## 목차

1. [보고서 재검토와 HQNR 이슈](#report)
2. [질문·범위·실험 case](#cases)
3. [공통 모델·초기화](#model)
4. [Forward·loss·gradient](#loss)
5. [학습 시점·가중치·학습률](#schedule)
6. [구현 명세와 통합 pseudocode](#implementation)
7. [HQNR 평가와 checkpoint 규약](#metrics)
8. [실행 전 metric 검증 gate](#metric-gate)
9. [반응·gradient·native 위치 진단](#diagnostics)
10. [성공·보류·종료 판정](#decision)
11. [실행 순서·예산·반복](#execution)
12. [Config template](#config)
13. [산출물·재현·보고 형식](#outputs)
14. [미확인 사항과 다음 단계](#limits)
15. [근거 자료](#sources)

---

<a id="report"></a>
## 1. 보고서 재검토: HQNR 이슈를 어떻게 반영하는가

### 1.1 이번에 직접 확인한 범위

**v1 작성 시 확인 기록:** 2026-09-12 업로드된 `PAN_Aligner_Directions_2026-09-12.md` 전문과 기존 NF16 실행 계획서를 읽었고, `pan-cvpr27 / WV3-s1`의 NF16 행을 다시 조회했다. [S1–S3] 이번 v2는 그 자료와 수치를 유지하는 예산 개정이며 최신 Sheet 재조회 결과를 추가한 것이 아니다.

**확인하지 못한 범위:** S1이 링크한 `2026-09-12_s1_alignment-axis-verdict.md`의 원문, 17벌의 원시 scene-level 결과, 실제 서버의 평가 코드·resolved config·checkpoint, 원본 MAT 예측은 이번 작업 공간에서 확보하지 못했다. 따라서 아래의 수치는 **보고서 또는 Sheet에 기록된 결과**이며, 이번에 재실행하여 얻은 값이 아니다.

보고서에 적힌 원인 해석과 이번 실험의 신규 설계를 구별한다. 공개 구현 문서 확인은 gradient 동작의 참고이며, 프로젝트의 metric 코드를 검증한 것으로 대체하지 않는다.

### 1.2 핵심 HQNR 이슈는 ‘참조 관계의 충돌’이다

S1은 다음과 같이 설명한다.

- `D_λ`: 출력과 MS 참조의 관계.
- `D_s`: 출력과 원래 PAN 참조의 관계.
- 두 모달리티의 기준이 어긋나 있으므로, PAN을 MS 쪽으로 크게 이동하여 사용하는 경우 spectral 항의 개선과 원 PAN 기준 spatial 항의 악화가 함께 나타난다.
- 보고된 상관은 `r(|Δ|, D_s)=+0.945`, `r(|Δ|, D_λ)=−0.689`다. [S1, §1–2]

**이 문서의 적용:** 이를 현재 실험군에 대한 보고서의 중심 해석으로 유지한다. 다만 ‘HQNR 구현에 버그가 있다’, ‘모든 네트워크에서 두 항을 동시에 개선하는 것은 수학적으로 불가능하다’, ‘점수 하락은 전부 지표 탓이다’라고 확대하지 않는다. 현재 제공된 자료만으로는 그러한 일반 명제나 코드 오류를 독립적으로 확정할 수 없다.

특히 **reconstruction은 GT에 대한 L1이고, HQNR은 평가·checkpoint 선택 기준**이다. 이 실험에서는 HQNR 또는 `D_s`를 학습 loss로 역전파하지 않는다. P2가 보정량을 줄인 현상을 ‘HQNR gradient가 aligner를 밀었다’고 기술하지 않는다.

### 1.3 Self-aligned 점수의 상승은 원본 기준 성능 개선과 다르다

S1은 `aligned_valid` HQNR 0.951–0.957, fSCC 0.918 대 raw 0.801을 기록하지만, 마지막 판정 규약에서는 **aligned view를 주 판정에서 제외**한다. 자기 예측 변위로 평가 PAN까지 바꾸면 평가 참조가 모델마다 달라지기 때문이다. [S1, §1, §6]

따라서 이번에는 다음을 금지한다.

- `max(raw HQNR, aligned HQNR)`를 하나의 대표 HQNR로 보고하기.
- P2의 raw 점수와 후보의 aligned 점수를 비교하여 개선이라고 쓰기.
- Aligned 점수만으로 λ를 선정하거나 raw 손실을 성공으로 바꾸어 해석하기.
- V64를 aligned metric으로 이름 바꾸기.

**V64는 원본 PAN 참조를 유지한 고정 crop 대조**다. FR 512²에서 `[64:448, 64:448]`를 평가한다. Shift별로 바뀌는 mask가 아니다. [S4, §10.6]

### 1.4 출력 inverse warp는 재도입하지 않는다

보고서가 수행한 P1 출력 재평가 결과는 다음과 같다. [S1, §2.1]

| P1 처리 | HQNR | D_λ | D_s |
|---|---:|---:|---:|
| 원래 출력 | 0.93298 | 0.02157 | 0.04659 |
| 출력 전체를 −Δ로 이동 | 0.89152 | 0.07046 | 0.04097 |

S1은 이 결과를 ‘출력을 단순히 원래 위치로 돌리는 것으로 구조·색의 관계를 함께 복구할 수 없다’고 해석한다. 본 계획은 이 실측 결과를 받아 **출력 inverse warp, GT warp, 평가 시 출력 최적 재정렬을 사용하지 않는다.**

### 1.5 P2의 작은 보정량과 반응 능력을 혼동하지 않는다

S1은 P2의 native 보정 크기가 donor의 1.511 px에서 0.344 px로 감소했다고 보고한다. 원문은 이를 ‘정합을 포기’한 것으로 해석한다. [S1, §2.2]

이번 비교에서는 관측량을 더 분리한다.

- `median ||ĉ₀||`: 원래 입력에서 얼마나 보정하는가.
- `ĉε−ĉ₀`의 반응 기울기: 추가 이동에 얼마나 반응하는가.
- `ĉε+ε−ĉ₀`의 오차: 추가 이동을 얼마나 상쇄하는가.

**0.344라는 크기만으로 P2가 상수 출력으로 돌아갔다고 확정하지 않는다.** S1에는 P2의 해당 반응 기울기·closure 표가 없다. 이를 이번 사전 평가에서 확보한다.

### 1.6 자료 사이의 정의 차이는 확인 대상으로 남긴다

| 항목 | 자료 A | 자료 B | 처리 |
|---|---|---|---|
| PO10 D_λ | S1 §1: 0.0112–0.0116, 무정합 0.0177 | S4 §10.2의 R200 50K: 0.0171–0.0175 | checkpoint·참조·ROI·파일 포맷·evaluator 차이를 원자료에서 확인. 같은 수치로 합치거나 −35%를 이번 목표값으로 복사하지 않음 |
| 주 판정 | S1 §6: best HQNR → SCC | S2의 기존 NF16 설명: last 대응 비교 강조, tie는 fSCC | 최신 보고서의 주 판정을 우선. last는 원인 분석용. SCC/fSCC 실제 코드 key는 gate에서 명시적으로 확인 |
| 후보 방향 | S1 §4: B(no-warp conditioning) → A(feature alignment) | 사용자: P2–P3 사이 λ 조절을 우선 | 사용자 결정에 따른 **제한적인 λ 확인 캠페인**으로 별도 명명. 보고서가 λ sweep을 최우선으로 권고했다고 쓰지 않음 |

**SCC와 fSCC는 자동으로 같은 지표라고 간주하지 않는다.** S1 §6의 ‘SCC’가 RR GT-SCC인지, 원 PAN 기반 FR fSCC의 축약인지 원문만으로는 명확하지 않다. 서버의 실제 selector와 연결해 metric 함수·split·reference를 기록한 후 모든 case와 과거 대조군에 동일하게 적용한다. 확인 없이 기존 fSCC를 SCC로 이름만 바꾸지 않는다.

### 1.7 최신 주 판정 규약

S1 §6에 따라 **HQNR seed 2σ=0.0031**을 이번 연구의 경험적 판정선으로 사용한다. 종전 0.011은 사용하지 않는다.

이 값은 다음과 다르다.

- Checkpoint 동률 허용치 `1e−4`.
- 지표 계산의 수치 오차 허용치.
- 모든 데이터셋·모든 metric에 대한 유의수준 또는 보편적 신뢰구간.
- ‘차이가 0.0031 안이면 반드시 무손실’이라는 통계적 동등성 증명.

원시 반복 결과가 없으므로 2σ 산출 자체를 이번에 재계산했다고 말하지 않는다. `CLAUDE.md` 등 서버 문서의 교체 필요성은 S1의 권고로 남기며, 이번 파일 작성으로 서버 문서를 변경하지 않는다.

---

<a id="cases"></a>
## 2. 연구 질문과 case 구성

### 2.1 이번에 검증할 가설

**Hλ:** `Lrec + λoff Loff`에서 λoff를 기존 0.01보다 낮추면, P2의 native 복원 성능을 유지하면서 P2보다 추가 PAN 변위에 대한 반응을 더 잘 보존할 수 있다.

이를 성공으로 전제하지 않는다. 작은 λ가 단순히 P2와 같은 동작을 만드는 경우, 혹은 모든 양의 λ가 복원에 불리한 경우도 유효한 결론이다. 이 실험은 절대 정합 GT를 새로 만드는 방법이 아니다.

### 2.2 비교 case — λ 외의 변경 없음

| ID | λoff | 의미 | 기본 seed 1234의 실행 |
|---|---:|---|---|
| `CTRL-P0` | 해당 없음 | W112–D123, aligner 없음 | 기존 NF16 P0 재사용 검증 |
| `L000` | 0.0 | **P2와 같은 recon-only 공동 학습** | 기존 NF16 P2 재사용 검증 |
| `L1E4` | **0.0001** | 기존 P3의 1/100 consistency | **신규 50K** |
| `L1E3` | **0.001** | 기존 P3의 1/10 consistency | **신규 50K** |
| `L3E3` | **0.003** | 기존 P3의 0.3배 consistency | **신규 50K** |
| `L1E2` | 0.01 | 기존 P3와 같은 감독 강도 | 기존 NF16 P3 재사용 검증 |

중간값은 **이번 계획의 새 설계값**이다. 이전 대화에서 대안으로 언급한 0.0003은 이번 기본 grid에 넣지 않는다. 탐색 중간에 grid를 계속 추가하지 않는다.

λ는 각 run에서 처음부터 끝까지 고정한다. `Lrec` 계수는 항상 1이다. `(1−λ)Lrec + λLoff`로 변경하지 않는다.

### 2.3 ‘P2 기반’의 정확한 뜻

**완료된 P2 전체 checkpoint에 이어 학습하는 것이 아니라, P2와 같은 초기화·학습 절차를 기준으로 λ를 비교한다.**

모든 λ case는 같은 N2 donor aligner와 같은 seed의 새 U-Net 초기 tensor에서 독립적으로 시작한다. 그래야 기존 P2(λ=0)와 P3(λ=0.01)가 양 끝의 대조가 된다.

완료된 P2를 로드해 이어 학습하는 방식은 이번 기본 큐에 포함하지 않는다. 이를 별도로 채택하면 ‘동일 P2 checkpoint에서 λ=0으로 같은 학습량을 연장한 대조’가 필요하며 protocol을 분리한다.

### 2.4 이번에 바꾸지 않는 것

- W112–D123, MS+PAN 9채널, 기존 dual-stem global aligner.
- PAN-only input warp, 기존 bicubic 경로, MS residual base·GT·최종 출력 좌표.
- 원판 radius `b=2 HR px`, SG, consistency 격번 적용.
- 학습률, optimizer, scheduler, batch, 50K budget, 평가 cadence.
- 출력 edge·geometry·PAN reconstruction·KD·GT variance·attention·FiLM·feature warp는 추가하지 않음.
- 보정량에 tanh/clamp를 추가하거나 λ를 실제 warp displacement에 곱하지 않음.

---

<a id="model"></a>
## 3. 공통 모델·초기화

### 3.1 변수와 좌표

| 기호 | 의미 | Shape / 단위 |
|---|---|---|
| P | 추가 변형 전 PAN | `[B,1,H,W]` |
| S | LRMS | `[B,8,H/4,W/4]` |
| M | 기존 bicubic으로 확대한 S | `[B,8,H,W]` |
| Y | HRMS reconstruction GT | `[B,8,H,W]`, MS 기준 |
| Aφ | 기존 aligner | 입력 PAN+M, 출력 `[B,2]` |
| Fθ | W112–D123 U-Net | 9채널 입력 → 8채널 residual |
| ĉ₀ | native pair의 예측 보정량 | `(dy,dx)`, 현재 PAN/HR px |
| ε | 우리가 만든 추가 변위 | `(dy,dx)`, 같은 단위 |
| ĉε | jittered pair의 예측 보정량 | `(dy,dx)`, 같은 단위 |

**Native는 실제 오차가 0인 ‘정합 GT’라는 뜻이 아니다.** 추가 synthetic ε만 없다는 뜻이다.

Sampler 부호를 다음으로 고정한다.

\[
\mathcal W(P,c)[y,x]=P[y+c_y,x+c_x].
\]

Grid의 마지막 좌표 순서는 `(x,y)`, 모델 출력은 `(dy,dx)`다. 위 부호에서 추가 변위를 상쇄하는 관계는 `ĉε ≈ ĉ₀−ε`다. GT·MS를 함께 shift하지 않는다.

### 3.2 Donor와 초기값

```text
Donor run:
  PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT
Donor checkpoint:
  exact optimizer update 50,000 / last
복사 대상:
  aligner state_dict만
복사하지 않는 것:
  donor U-Net, optimizer state, scheduler state, 초기 best_hqnr
```

각 λ run은 **동일 donor aligner hash**를 로드하고, U-Net은 동일 seed의 **저장된 initial state_dict**를 사용한다. Donor final head를 다시 zero-init하면 안 된다. Aligner 모듈의 shape·normalization·margin4·채널 순서도 그대로 유지한다. [S2, §2]

별도의 teacher/EMA aligner는 만들지 않는다. `ĉ₀` target은 현재 공유 aligner의 예측이지, 동결 donor가 내는 별도 target이 아니다.

### 3.3 초기화 검증

다음은 숫자가 같아 보이는 수준이 아니라 파일/가중치 hash로 남긴다.

- Donor aligner hash, exact update, state_dict prefix.
- Seed별 U-Net initial tensor hash.
- 같은 step/sample의 native batch 및 ε hash.
- 학습·평가 dataset manifest, preprocessing·sampler·evaluator version.

기존 NF16 초기 tensor를 복원할 수 없거나 코드의 의미가 바뀌었다면, 기존 P2/P3는 배경 비교로만 두고 동일 새 block의 λ=0, 0.01 대조를 다시 수행한다. 이전 점수를 새 실험의 통제된 결과처럼 끼워 넣지 않는다.

---

<a id="loss"></a>
## 4. Forward·loss·gradient

### 4.1 매 update: native reconstruction

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

`ĉ₀`와 `P̃₀`는 reconstruction graph에서 detach하지 않는다. 매 update 큰 U-Net은 한 번 호출한다.

U-Net wrapper가 이미 M을 더하는 API라면 위 수식의 residual 합산을 두 번 하지 않는다. 실제 return 의미를 확인한다.

### 4.2 홀수 update: 작은 aligner에만 추가 변위 연습

\[
P_\epsilon=\mathcal W(P,\epsilon),\quad
\hat c_\epsilon=A_\phi(P_\epsilon,M),
\]
\[
\boxed{
L_{\mathrm{off}}=
\frac{1}{2B}\sum_{i=1}^{B}\sum_{k\in\{y,x\}}
\left|\hat c_{\epsilon,i,k}+\epsilon_{i,k}-\operatorname{sg}(\hat c_{0,i,k})\right|.
}
\]

- 같은 batch의 같은 PAN에 ε를 추가한다. 다른 장면을 consistency pair로 쓰지 않는다.
- PAN과 M은 양쪽 aligner 호출에서 동일한 고정 margin4 내부 view를 사용한다.
- **Jittered PAN 또는 그 보정 결과를 U-Net에 넣지 않는다.**
- ε는 input image 생성과 loss 계산에만 쓴다. 모델 입력 채널로 제공하지 않는다.
- Stop-gradient는 **offset target의 `ĉ₀` 경로에만** 적용한다.
- 전체 native aligner 호출을 `no_grad()`로 감싸지 않는다.
- Loff는 component-mean L1, 단위 HR pixel. L2/EPE/Huber 또는 sample별 `||ε||` 정규화로 바꾸지 않는다.

### 4.3 총 loss와 update

0-based optimizer update index를 t라 두면,

\[
m_t=\mathbf 1[t\bmod2=1],\qquad
\boxed{L_t=L_{\mathrm{rec}}+m_t\lambda_{\mathrm{off}}L_{\mathrm{off}}.}
\]

\[
\boxed{\nabla_\phi L_t=\nabla_\phi L_{\mathrm{rec}}
+m_t\lambda_{\mathrm{off}}\nabla_\phi L_{\mathrm{off}}},\qquad
\boxed{\nabla_\theta L_t=\nabla_\theta L_{\mathrm{rec}}}.
\]

Forward 중간에는 optimizer step을 호출하지 않는다. Native와 offset graph를 계산한 뒤 loss를 합산하여 **backward 한 번, optimizer step 한 번**을 실행한다. 두 모듈은 모두 첫 update부터 학습한다.

λ가 scale하는 것은 합산 전 **consistency gradient 성분**이다. 이동량이나 전체 optimizer update가 λ에 비례한다고 보장되지 않는다. AdamW의 상태와 reconstruction gradient가 함께 작용하기 때문이다. [수식에 대한 구현 해석]

### 4.4 상대 loss가 해결하지 못하는 것

\[
\hat c_0=\mu,\qquad \hat c_\epsilon=\mu-\epsilon
\]

이면 μ가 정확한 절대 보정량이 아니어도 Loff 값은 0이다. 따라서 목표는 큰 `||ĉ₀||`를 유지하는 것이 아니라, **native fitting과 추가 변위 반응의 양립**이다.

SG는 이 절대 기준의 미식별성을 해결하지 않는다. 또한 두 호출이 공유 가중치를 쓰므로 jittered 경로의 update가 다음 native 예측도 바꾼다. ‘입력을 분리했으니 두 학습이 완전히 독립’이라고 해석하지 않는다.

추가로, 공통 가산 bias가 두 출력 차이에서 상쇄된다는 **loss 값의 성질**과, target 한쪽을 detach한 **한 update의 미분 경로**도 구분한다. SG에서는 현재 target이 고정되어 jittered 출력의 bias에 gradient가 생길 수 있다. 이것이 실제 NF16 drift의 원인이었다고 확정한 것은 아니다.

---

<a id="schedule"></a>
## 5. 학습 시점과 세부 설정

### 5.1 Update 표

| Update | Native aligner | U-Net native 복원 | Jittered aligner | 학습 loss |
|---:|---|---|---|---|
| 0 | 1회, gradient 있음 | 1회 | 없음 | Lrec |
| 1 | 1회, gradient 있음 | 1회 | 1회, gradient 있음 | Lrec + λLoff |
| 2 | 1회, gradient 있음 | 1회 | 없음 | Lrec |
| 3 | 1회, gradient 있음 | 1회 | 1회, gradient 있음 | Lrec + λLoff |

λ=0 대조는 training에 jittered forward가 필요 없다. 동일한 진단 probe에서는 다른 case와 같은 입력으로 response를 측정한다.

### 5.2 공통 설정

| 항목 | 값 / 처리 |
|---|---|
| Dataset / 서버 | WV3 / s1 |
| 기본 seed | 1234 |
| 확인 반복 seed | **7777·2025**, 같은 s1에서 각 seed의 P0·L000·λ*를 독립 학습 (§11) |
| Backbone | W112, depth=[1,2,3], attention 없음 |
| Aligner | NF16의 약 0.105M global CNN, 기존 class 그대로 |
| 입력·출력 | MS+PAN 9ch / 단일 HRMS residual |
| Aligner view | 고정 margin4, 64²→56², 512²→504² |
| U-Net domain | 원래 전체 영상; extra crop 없음 |
| Optimizer | NF16 P2의 실제 AdamW 설정 계승 |
| U-Net base LR | 0.0001 |
| Aligner base LR | 0.00001 |
| Weight decay | 기존 0.01, parameter exclusion 정책도 계승 |
| LR scheduler | NF16 warmup100 + cosine, 두 parameter group에 같은 배율 |
| 총 학습 | 50,000 optimizer updates |
| 실효 batch | 48; accumulation 변경 시 모든 case에 동일 적용 |
| 정밀도 | 실제 NF16 정책 계승. warp/offset 계산은 FP32 |
| Lrec 계수 | 1.0 |
| Loff 계수 | 각 case의 고정 λ; **이번 sweep에서는 ramp/decay 없음** |
| Loff 빈도 | 2 update 중 1회 |
| 보조 loss | geometry, output edge, PAN loss, KD 모두 0 |

Beta·epsilon·grad clipping·scheduler minimum LR 같은 기존 세부값을 임의로 새로 정하지 않는다. 서버 NF16 P2 resolved config에서 가져와 manifest에 동결한다. 기록이 누락됐으면 기본 라이브러리값으로 조용히 대체하지 않는다.

### 5.3 ε sampling

\[
u_1,u_2\sim U(0,1),\quad
\rho=b\sqrt{u_1},\quad \vartheta=2\pi u_2,
\]
\[
\epsilon=(\rho\sin\vartheta,\rho\cos\vartheta),\qquad b=2.0.
\]

- 원판 **면적 uniform**, `||ε||₂≤2` 현재 PAN pixel.
- 축별 독립 `uniform(-2,2)`인 정사각형으로 바꾸지 않는다.
- 같은 sample의 모든 위치에 동일 vector를 적용한다.
- 공통 flip/rotation 이후 original PAN에서 매번 새로 만든다. 반복 warp로 누적하지 않는다.
- 2 HR px는 같은 입력 pair에서 0.5 LRMS px에 해당하지만 sampler에는 **2.0**을 넣는다.
- 추론에는 추가 ε가 없으며, native correction은 계속 사용한다.
- b=2는 기존 FR 입력 통계를 참고해 정한 개발 설정이다. 이번 결과도 train-only/test-untouched 설정이라고 쓰지 않는다. [S2, §4.3]

---

<a id="implementation"></a>
## 6. 구현 명세

### 6.1 필요한 코드 변경 범위

기존 NF16 P3 trainer의 `0.01 * Loff`를 **config에서 읽는 scalar `lambda_offset`**로 교체한다. 모델 구조, input warp 위치, loss reduction, sampler, augmentation, optimizer schedule은 수정하지 않는다.

실제 코드 key나 module 경로는 서버에서 확인한다. S1의 `pa/model.py`는 보고서에 적힌 경로지만, 아래 helper 이름은 실행 repository의 API를 보장하지 않는다.

### 6.2 통합 pseudocode

```python
# Integration pseudocode: adapt helper names to the existing NF16 trainer.
# P, S, Y are the same native batch after the existing common augmentation.
# global_update counts completed optimizer updates, not microbatches.

optimizer.zero_grad(set_to_none=True)

M = existing_bicubic(S)
c0 = aligner(valid_view(P, 4), valid_view(M, 4))
P_aligned = existing_warp(P, c0)             # DO NOT detach c0.
pred = M + unet(torch.cat([P_aligned, M], dim=1))
Lrec = (pred - Y).abs().mean()
loss = Lrec
Loff = None

active_offset = (global_update % 2 == 1) and (lambda_offset > 0.0)
if active_offset:
    with torch.no_grad():
        eps = sample_uniform_disk(
            batch_size=P.shape[0], radius_hr=2.0,
            generator=offset_rng, device=P.device,
        )
        Pe = existing_warp(P, eps)          # Start from original P every time.
    ce = aligner(valid_view(Pe, 4), valid_view(M, 4))
    target = c0.detach() - eps              # Only this target is detached.
    Loff = (ce - target).abs().mean()        # Mean over batch and 2 components.
    loss = Lrec + lambda_offset * Loff

# Optional autograd.grad diagnostics run here and must not change .grad or RNG.
loss.backward()
existing_clip_policy_if_any()              # No newly introduced clip rule.
optimizer.step()
existing_scheduler_step()                  # Same order/index as NF16.
global_update += 1
```

**λ=0**은 기존 P2의 학습과 동치여야 한다. 진단용 forward를 추가해도 BN buffer·dropout·data RNG 등 학습 상태를 바꾸지 않게 해야 한다. Aligner normalization은 donor와 동일하게 유지하고, view를 crop한 뒤 정규화하는 기존 순서를 지킨다.

### 6.3 Sampler·boundary

기존 `bicubic / border / align_corners=False` warp를 재사용한다. 보정량이 우연히 0이라고 trainable warp를 생략하지 않는다. `.item()`, NumPy, `torch.tensor(existing_tensor)` 재포장, `c0.detach()`로 grid graph를 끊지 않는다. [S2, §5.3; R2]

훈련용 Lrec domain은 기존 전체 영상이다. 예측 shift마다 reconstruction 영역을 줄이지 않는다. 평가용 ROI와 학습 loss domain을 구분한다.

Jitter를 먼저 전체 PAN에 적용한 다음 aligner의 고정 내부 view를 취한다. Crop한 작은 PAN을 먼저 warp하여 artificial border를 재도입하지 않는다. b는 고정 2, view margin도 모든 case에서 고정 4다.

### 6.4 RNG·resume

Data order/augmentation RNG와 ε RNG를 분리한다. λ가 달라도 같은 seed·update·sample에서 같은 ε를 생성한다. Probe RNG도 분리하여 진단 횟수가 훈련 ε를 바꾸지 않게 한다.

Resume에는 아래 상태를 모두 저장·복구한다.

```text
model / optimizer / scheduler / scaler(if used)
completed_optimizer_updates / update_parity
sampler cursor / data RNG / augmentation RNG / offset RNG
lambda_offset / b / offset_every / target_detach policy
selector state / candidate checkpoint metadata
code + config + dataset + donor hashes
```

λ를 바꿔 resume한 결과는 fixed-λ run으로 부르지 않는다. 기존 실험의 directory·checkpoint·Sheet 행을 덮어쓰지 않는다.

---

<a id="metrics"></a>
## 7. HQNR 평가와 checkpoint 규약

### 7.1 주 판정과 진단을 분리한다

**주 판정:** `best_raw`의 raw-original HQNR, 논문 FR `.mat` 20장, 원 PAN 참조. 최신 보고서의 보조 우선순위 표기는 SCC다. [S1, §6]

**진단:** 같은 checkpoint의 raw-V64, aligned-self-V64, fixed-N2-V64, RR 지표, 정합 반응, 그리고 정확한 50K last.

지난 NF16 설명에서 ‘last끼리의 인과 비교’를 강조한 것은 정합 능력·수렴 상태를 분석하기 위한 것이었다. 이번 주 성능 판정을 last로 바꾸는 근거로 사용하지 않는다. **공식 성능 비교와 학습 동작 분석을 서로 다른 표에 둔다.**

### 7.2 동일 출력으로 계산할 view

| metric 열 | PAN reference | 평가 영역 | 용도 |
|---|---|---|---|
| `hqnr_raw_original` | 원 P | 기존 전체 domain | **주 판정, best_raw 선택** |
| `hqnr_raw_v64` | 원 P | 고정 V64 | crop 효과 대조 |
| `hqnr_aligned_self_v64` | 실제 사용한 W(P,c0) | 같은 V64 | 모델 자신의 정합 조건 진단 |
| `hqnr_aligned_fixedN2_v64` | 동결 N2 donor가 native pair에서 만든 PAN | 같은 V64 | 동일 참조에서 출력 차이 진단 |

위 네 값은 **같은 checkpoint의 같은 예측**을 사용한다. Raw evaluation은 aligner를 끄고 다시 추론하는 것이 아니다. `c0=0` 재추론은 별도의 ablation이다.

Fixed-N2 reference도 정합 GT가 아니며 donor 계열에 유리할 수 있다. Aligned 계열은 λ 선택과 우월성 판정에 사용하지 않는다.

### 7.3 장면별 곱의 평균

각 scene i에서:

\[
H_i=(1-D_{\lambda,i})(1-D_{s,i}),\qquad
\boxed{\overline H=\frac1{20}\sum_{i=1}^{20}H_i.}
\]

`(1−mean Dλ)(1−mean Ds)`로 대체하지 않는다. 수식상 두 방식의 차이는 scene 간 `Dλ, Ds`의 모집단형 covariance다. 따라서 반올림된 aggregate 두 열을 곱하여 보고된 HQNR과 다르다는 이유만으로 evaluator 오류라고 판정하지 않는다. 반드시 scene별 원자료에서 확인한다. [S2, §6.2; 집계식의 직접 계산]

### 7.4 Reference를 바꾸는 두 경로

Aligned 평가에서는 높은 해상도의 PAN만 바꾸고 저해상도 PAN을 원본에서 만드는 혼합을 금지한다.

```text
P_ref_hr = chosen PAN reference
P_ref_lr = existing_evaluator_degradation(P_ref_hr)
```

필터·sampling phase·block grid를 기존 evaluator와 동일하게 유지한다. 일반적인 개념 설명만으로 `D_s`를 새로 구현하지 않는다. 프로젝트의 실제 `D_s`, Q 계산, reference 생성 경로를 고정한다.

같은 출력·MS 참조·ROI에서 PAN reference만 바꿨다면 spectral `D_λ` 값은 같아야 한다. 다르면 reference policy에 따라 spectral 경로까지 달라졌는지 조사한다. [S2, §6.2]

### 7.5 ROI·mask

FR512의 V64는 384×384 고정 영역이다. MTF/filtering/저해상도 reference 생성은 원래 full frame에서 수행한 뒤 정해진 위치로 집계한다. 먼저 crop하여 sampling phase를 바꾸지 않는다. HR/LR는 같은 물리 영역으로 연결한다.

Bicubic support, 후속 필터 및 metric window가 영상 밖의 복제 값을 읽는지 검사한다. Invalid pixel을 0으로 채워 일반 metric에 포함하지 않는다. V64까지 무효 support가 들어온 scene은 조용히 제외하지 않고 해당 aggregate의 eligibility 실패로 기록한다.

Raw-original은 기존 benchmark domain 그대로 남긴다. 실패를 숨기기 위해 ROI를 줄이거나 참조를 이동하지 않는다.

### 7.6 Checkpoint 저장

| 이름 | 선택·저장 방식 | 보고 위치 |
|---|---|---|
| `best_raw` | Raw-original HQNR 최고, 확정된 SCC tie-break | **주 표** |
| `last` | exact 50,000 update | 수렴·반응·복원 진단 표 |
| `best_aligned_diagnostic` | 기존 aligned selector를 유지할 경우 별도 저장 | 진단만; 주 판정에 사용 금지 |

평가 cadence는 NF16와 동일하게 고정하고 종료 50K 평가를 추가로 보장한다. 이전 PO10의 1,010 step best 현상을 알고 있다고 이번에 초기 평가를 삭제하지 않는다. Sheet의 `(50K)` 학습량과 `selected_update`는 별도 열로 저장한다. [S4, §10.2, §10.5]

### 7.7 Tie·2σ·수치 허용치: 세 종류를 분리

1. **Checkpoint tie band:** 기존 `1e−4` 유지. 기준은 running maximum. 점진적으로 기준이 내려가는 누적 tie 교체 금지.
2. **방법 간 경험적 판정선:** 최신 보고서의 `0.0031`. Raw HQNR에만 적용.
3. **계산 재현 tolerance:** evaluator 검증용 별도 값. 원본 정밀도에 맞춰 gate에서 고정.

SCC의 정확한 정의는 §8 G-M0에서 연결한다. 만일 기존 NF16 selector의 fSCC와 다른 함수라면, 과거 대조군도 동일한 선택 규약으로 정리해야 한다. 이는 λ와 별개의 protocol 변경이므로 기록한다. 기존 candidate checkpoint가 없어서 재선택이 불가능하면 비교를 ‘selector 차이가 있는 배경 비교’로 제한하거나 대조군을 다시 실행한다.

동률 후보는 `HQNR >= historical_max−1e−4`인 후보 집합 안에서 확정된 SCC로 고른다. 그 안에서도 동률이면 더 늦은 update를 택하는 기존 규칙을 계승한다. SCC 자체의 수치 tie tolerance는 실제 selector 값을 가져오고 기록한다.

---

<a id="metric-gate"></a>
## 8. 실행 전 metric 검증 gate — λ 실험보다 먼저

**이 절은 새 네트워크 학습을 시작하기 전, ‘같은 지표를 보고 있는가’를 확인하는 필수 절차다. 현재 서버에서 통과했다고 주장하지 않는다.**

| Gate | 점검 | 통과 조건 / 실패 시 조치 |
|---|---|---|
| **G-M0 정의 고정** | HQNR 함수·Dλ/Ds 함수·SCC key·split·ROI·정밀도·reference hash | 모호한 SCC/fSCC 이름이 없어야 함. 미확정이면 full run 보류 |
| **G-M1 무변경 재평가** | 기존 P0/P2/P3 저장 예측을 학습 없이 동일 evaluator에 입력 | 기존 기록 재현. 다른 checkpoint나 ROI이면 별도로 표기 |
| **G-M2 scene 집계** | 20 scene ID 누락·중복 여부, scene별 곱의 평균 | 전 scene 동일 manifest, 합의한 집계식 |
| **G-M3 view identity** | P0의 aligned-self-V64 vs raw-V64 | aligner가 없으므로 동일해야 함 |
| **G-M4 fixed reference** | Frozen P1의 self vs fixed-N2 reference·ROI | 동일 donor와 전처리라면 일치해야 함 |
| **G-M5 spectral invariance** | 같은 예측·MS·ROI에서 PAN 참조만 교체 | Dλ 동일; 다르면 spectral 평가 분기 점검 |
| **G-M6 checkpoint 대응** | best_raw 실제 update·export hash, exact50K last | best가 초기여도 보존; last를 best로 이름 바꾸지 않음 |
| **G-M7 support** | 원/보정 PAN의 보간·필터·metric window support | 고정 ROI 적격, scene 제외로 평균 개선하지 않음 |
| **G-M8 provenance 차이** | S1의 Dλ 0.0112–0.0116과 S4/Sheet의 다른 값 | 원자료에서 view/checkpoint 차이를 설명하거나 미해결로 분리. sweep 기준은 재현된 P0/P2/P3 동일 프로토콜로 고정 |

G-M1 참고값(현재 Sheet 재조회, best_raw):

| s1 seed1234 | Raw HQNR | Raw-V64 HQNR | RR ERGAS | RR PSNR | Dλ / Ds |
|---|---:|---:|---:|---:|---|
| NF16 P0 | 0.9532 | 0.9587 | 2.0561 | 37.9465 | 0.0215 / 0.0259 |
| NF16 P2 | 0.9539 | 0.9593 | 2.0411 | 38.0070 | 0.0227 / 0.0240 |
| NF16 P3 | 0.9463 | 0.9521 | 2.5834 | 36.1902 | 0.0238 / 0.0306 |

이 표는 **2026-09-12 Sheet 조회값**이지 50K last 성능표가 아니다. [S3]

Source precision에 따른 기본 검사는 다음처럼 구분한다. 아래 tolerance는 **이번 수치 검사용 제안**이고 방법 우월성 판정선과 다르다.

- 소수 4자리 Sheet 값과 비교: 동일 rounding으로 일치 확인(또는 rounding 폭 5e−5 수준). 미세한 우열에는 원시값 사용.
- 원시 float64 scene 결과와 동일 경로 비교: `atol=1e−6`를 우선 검토하고 실제 기존 evaluator 테스트 tolerance가 있으면 그 값을 모든 case에 고정.
- P1 inverse 실험의 **무변경** 원본 결과 0.93298은 확보된 같은 MAT/코드로 재현될 때만 회귀 검사에 사용. 이번 캠페인은 inverse-warp 학습 또는 추가 성능 sweep을 재실행하지 않음.

**보고서가 말하는 참조 프레임 이슈와, 구현의 산술/phase/포맷 오류를 구분한다.** Gate에서 오류를 발견하면 기존 baseline을 포함해 재평가하고 새 metric version을 만든다. 후보에게만 수정 평가기를 적용하지 않는다.

---

<a id="diagnostics"></a>
## 9. Aligner 반응·gradient·native 위치 진단

### 9.1 반드시 확보할 시점

- 시작 donor, 기존 P2/P3의 `best_raw`와 exact50K `last`.
- 신규 run의 10K/25K/50K 고정 진단과 저장된 `best_raw`.
- Full FR stress 등 고비용 평가는 최소 exact50K와 최종 선택 후보의 best_raw에서 수행.

이는 분석용 일정이다. 공식 HQNR 평가 cadence를 case마다 달리하지 않는다.

### 9.2 상대 반응

\[
r_\epsilon=A_\phi(P_\epsilon,M)+\epsilon-A_\phi(P,M),
\]
\[
\hat c_\epsilon-\hat c_0=B_{\rm resp}\epsilon+a.
\]

기록:

- `B_resp` 전체 2×2, 대각과 교차축 항, intercept.
- `offset_mae_component=mean(abs(rε))`.
- `offset_epe=mean(norm(rε,dim=-1))`.
- 기존 `po10_diag` closure는 정의를 확인하여 별도 이름으로 유지.
- `||ĉ₀||` 중앙값/P90, dy/dx 중앙값/IQR, donor 대비 drift.

64²/256²/512²에서 같은 radius set `{0.5,1,2}`와 8방향의 고정 probe를 사용한다. Sample/scene ID와 ε를 저장한다. HQNR 선택에 이 probe 점수를 사용하지 않는다.

‘정합 학습 성공’은 보정 크기가 커진 것과 다르다. 작은 native 보정이라도 `B≈−I`를 유지할 수 있고, 큰 native 보정이라도 상수일 수 있다. [S2, §8; S4, §10.5]

### 9.3 λ가 실제로 얼마나 전달됐는지

고정된 진단 update의 동일 graph에서:

\[
g_{rec}=\nabla_\phi L_{rec},\qquad
g_{off}=\nabla_\phi L_{off},\qquad
\rho_g=\frac{\lambda\|g_{off}\|_2}{\|g_{rec}\|_2+\varepsilon_{num}},
\]
\[
\cos\psi=\frac{g_{rec}^{\top}(\lambda g_{off})}
{\|g_{rec}\|_2\|\lambda g_{off}\|_2+\varepsilon_{num}}.
\]

Raw/weighted Loff, 두 norm, cosine을 기록한다. `ε_num`은 로그의 수치 안정화용 상수로 `1e−12`를 제안한다. Norm이 0인 cosine은 0이라고 의미를 부여하지 말고 `undefined`로 기록한다.

진단은 `autograd.grad(..., retain_graph=True)`로 수행하되 optimizer update를 추가하지 않고 `.grad`를 오염시키지 않는다. AMP 사용 시 scaled/unscaled gradient를 혼합하지 않는다. 전체 module norm 외에 head와 encoder를 나누어 볼 수 있지만 이번에는 새 적응형 weight 알고리즘으로 자동 변경하지 않는다.

**Gradient가 서로 반대 방향이라는 한 측정만으로 장기적인 목적 충돌이 입증되지는 않는다.** Update별 분포·native 품질·반응 추세를 함께 읽는다.

### 9.4 Aligner-only loss임을 실제로 검사

진단 batch에서 다음을 독립 검증한다.

| 검사 | 기대 |
|---|---|
| Lrec만 미분 | U-Net와 aligner 양쪽 gradient 존재 |
| Loff만 미분 | Aligner gradient 존재, U-Net 직접 gradient 없음 |
| Loff에서 ĉ₀ target 경로 | gradient 없음 |
| Lrec에서 ĉ₀ 경로 | gradient 존재 |
| 합산 loss gradient | `g_rec + λ g_off`와 수치적으로 일치 |
| λ=0 | 같은 초기화의 P2 loss·gradient와 일치 |
| 중간 optimizer step | 0회 |
| U-Net forward 수 | update당 1회 |

이상적인 closure 관계를 상수값 예측기로 대체한 negative control도 둔다. 원판 b에서 무반응 constant predictor의 component-mean L1 기대값은 직접 적분하면 `4b/(3π)`다. b=2면 약 0.8488 px다. 이 값은 sampler/control 점검용이지 실제 정합 오차의 하한이 아니다.

### 9.5 Native 위치와 복원

RR에서는 같은 band의 출력–GT 오차와 경계 profile을 본다. GT를 출력에 맞춰 재정렬하지 않는다. FR에서는 독립적인 입력 구조 비교를 proxy로 사용하며 정확한 정합 GT라고 부르지 않는다.

S1이 권고한 `tools/align_after_training_diag.py` Part B(출력의 구조–색 관계)는 **기존 구현과 정의가 확보될 때 그대로** 수행한다. 현재 원문 코드가 없으므로 임의의 새 공식을 같은 지표 이름으로 구현하지 않는다. [S1, §3-D]

MS swap/상수 MS/다른 interpolation kernel/고정 padding 대조는 N2 last와 최종 후보에서 다시 확인한다. 초기 best의 검사 결과를 last의 검사로 승계하지 않는다. [S2, §8.5]

Synthetic stress의 모델 입력은 Pε지만, 평가 reference는 native P 및 고정 native donor reference로 유지한다. FR stress의 V96과 native V64를 직접 빼지 말고 native도 같은 V96에서 평가하여 비교한다. RR stress의 GT는 동일 Y다. [S2, §8.4]

---

<a id="decision"></a>
## 10. 성공·보류·종료 판정

### 10.1 주 성능 판정

각 seed block에서 raw best HQNR 차이를 두 기준에 대해 계산한다.

\[
\Delta H_{P2}=H_{candidate}-H_{\lambda=0},\qquad
\Delta H_{P0}=H_{candidate}-H_{noalign}.
\]

S1의 `0.0031`을 경험적 분류선으로 사용한다.

| 관찰 | 주 성능 해석 |
|---|---|
| 대응 차이 > +0.0031 | 보고서 판정선을 넘는 개선 후보. 반복에서 방향 확인 필요 |
| 대응 차이 < −0.0031 | 보고서 판정선보다 큰 후퇴 |
| 차이의 절댓값 ≤0.0031 | 판정선 내부. ‘통계적 동등성 입증’이라고 쓰지 않음 |

P2 자체의 현재 +0.0007 대 P0 차이는 이 선 안이다. 따라서 P2를 ‘우월성이 확정된 방법’이라고 쓰지 않고 **가장 유리한 현재 기준점**으로 사용한다. [S3; 계산]

`0.0031`을 Dλ, Ds, PSNR, closure에 복사 적용하지 않는다. 특히 self-aligned의 상승폭이 이 선을 넘었다고 성공 판정하지 않는다.

### 10.2 방법론적 진단 판정

| Raw 품질 / response 관찰 | 허용되는 결론 |
|---|---|
| P2의 raw 성능 범위를 유지하며 P2보다 closure·반응이 개선 | **복원–반응 절충 후보**. Raw SOTA 개선과는 별개 |
| Raw·response 모두 P2와 구별 안 됨 | 이 λ에서는 보조 감독의 실효가 확인되지 않음 |
| Response가 개선되나 raw가 판정선 밖으로 후퇴 | 반응 보존의 비용이 남음. Aligned 점수로 만회 판정하지 않음 |
| Raw만 좋아지고 response는 약해짐 | P2와 유사한 복원 적합/보정 축소 방향. 정합 능력 보존 성공 아님 |
| Native 보정 크기만 커짐 | 반응 또는 native 정합 개선의 증거가 아님 |
| 모든 λ가 P2보다 불리함 | 이 input-warp 구조의 λ 절충은 유효성 미확보. 추가 무한 sweep 금지 |

**RR 악화와 미학습 초기 best 현상은 별도 경고로 반드시 제시한다.** Raw best를 주 판정으로 유지하더라도, 그것만으로 ‘정합·복원 둘 다 성공’이라는 주장을 하지 않는다. Response 성공은 exact50K와 같은 checkpoint의 복원 결과에 연결하여 기술한다.

### 10.3 반복 후보 λ의 사전 선택 규칙

세 신규 λ 중 **raw best HQNR이 가장 높은 것**을 확인 반복 후보 λ*로 둔다. 동률은 확정된 SCC, 이후 더 작은 양의 λ 순서로 정한다. Aligned 점수나 closure가 좋아 보인다는 이유로 선택 순서를 사후 교체하지 않는다.

모든 신규 λ가 P2보다 0.0031 초과 낮으면 추가 반복 대신 원인 분석으로 종료할 수 있다. 그보다 작은 차이 안에서 유력 후보가 있어도 그것은 최종 winner가 아니라 다음 seed 확인 대상이다.

확인 반복에서는 반드시 같은 seed의 λ=0 대조를 함께 확보한다. **24시간 개정안은 seed 7777·2025 각각에서 no-align P0도 함께 확보**하여 P2 대비 보조 감독의 효과와 P0 대비 정합 포함 방법의 효과를 구분한다. Donor는 동일하므로 이 반복이 donor 사전학습 seed 변동까지 측정한 것은 아니다.

**λ*는 seed 1234 탐색을 마친 뒤, 확인 seed를 실행하기 전에 한 번 정하고 고정한다.** seed 7777의 결과를 보고 다른 λ로 바꾸어 seed 2025를 실행하지 않는다. 양쪽 확인에서 같은 λ를 사용하며, 확인 실패도 그대로 보고한다. 탐색 seed와 확인 seed는 별도 표기한다.

---

<a id="execution"></a>
## 11. 24시간 실행 순서·예산·반복

### 11.1 예산의 의미와 우선순위

**이번 캠페인에 허용한 추가 총예산은 24 GPU-hours다.** 기존 N2 사전학습 및 이미 끝난 NF16 학습 비용은 이번 추가 예산에서 다시 차감하지 않지만, 전체 방법의 비용을 보고할 때 별도로 명시한다. 기존 가중치 재평가, 새 smoke, profiling, 새 학습, GPU 진단은 이번 ledger에 모두 포함한다.

기본은 **s1의 단일 GPU에서 순차 실행**이다. 24 GPU-hours는 case당 24시간이나 서버마다 24시간이 아니다. GPU 작업 외 CPU-only 준비·파일 I/O·사람의 검토 대기 시간은 별도 wall-time으로 기록하므로, 전체 작업의 달력상 종료를 정확히 24시간으로 보장하지 않는다. 다른 GPU를 추가로 사용하면 그 사용 시간도 합산하며, 서버 간 절대 지표를 직접 섞지 않는다.

이번 시간 확장의 우선순위:

1. HQNR 정의·재현과 기존 대조군 동치 검증.
2. 기존 중간 λ 세 개의 seed 1234 비교.
3. 같은 λ*와 P0/P2의 seed 7777 대응 확인.
4. 같은 λ*와 P0/P2의 seed 2025 대응 확인.
5. Best와 exact50K의 다중 참조 평가, response·gradient·shortcut 원인 분석.

**모든 신규 run은 동일하게 50,000 optimizer updates다.** λ 또는 case마다 시간을 맞추려고 update 수를 다르게 줄이지 않는다. 추가 시간은 λ grid나 구조를 넓히기보다 결과의 재현성과 metric 검증에 사용한다.

### 11.2 Case × seed 구성

| Case | λoff | s1 / seed 1234 — 탐색 | s1 / seed 7777 — 확인 1 | s1 / seed 2025 — 확인 2 |
|---|---:|---|---|---|
| **CTRL-P0** | 해당 없음 | 기존 NF16 P0 재사용 gate | **신규 50K** | **신규 50K** |
| **L000 / P2** | 0 | 기존 NF16 P2 재사용 gate | **신규 50K** | **신규 50K** |
| L1E4 | 0.0001 | **신규 50K** | λ*로 선택된 경우만 신규 50K | 같은 λ*인 경우만 신규 50K |
| L1E3 | 0.001 | **신규 50K** | λ*로 선택된 경우만 신규 50K | 같은 λ*인 경우만 신규 50K |
| L3E3 | 0.003 | **신규 50K** | λ*로 선택된 경우만 신규 50K | 같은 λ*인 경우만 신규 50K |
| **L1E2 / 기존 P3** | 0.01 | 기존 NF16 P3 재사용 gate | 기존 P3가 검증되면 배경 대조로만 재사용 | 이번 기본 신규 큐에는 없음 |

기본 추가 학습은 **3 + 3 + 3 = 9회**다. 첫 번째 3회는 서로 다른 λ이고, 각 확인 block의 3회는 **P0 + L000 + 고정 λ***다. 이는 모든 λ를 3 seed씩 학습하는 15회 전체 grid가 아니다.

Seed 1234의 기존 P0/P2/P3 재사용이 모두 승인되고, §10.3의 확인 진행 조건을 만족하는 경우의 run 수다. 확인 seed에 정확히 같은 조건의 완료 대조군이 실제 존재하면 §11.3을 통과한 것만 재사용하고 비용을 절약한다. 조회 없이 존재를 가정하지 않는다.

P0에는 aligner가 없지만, 같은 seed의 U-Net initial state_dict와 데이터·평가 설정은 L000/λ*와 일치해야 한다. 모든 aligner 포함 run은 동일한 **N2 donor seed 2025의 exact50K aligner hash**에서 시작하며, U-Net만 각 seed의 새 초기값을 사용한다.

**확인 seed 2025와 donor의 seed 숫자가 같아도 donor U-Net이나 optimizer를 가져오지 않는다.** N2와 별도로 생성·저장한 새 복원 U-Net 초기값을 사용한다. 이번 3-seed 결과는 고정 donor에 조건부인 downstream 학습 변동이며, 전체 사전학습+복원의 독립 3-seed 변동이 아니다.

### 11.3 기존 결과 재사용 조건

기존 NF16 P0/P2/P3를 다시 학습하지 않는 것이 기본이지만, §3.3과 §8 검증을 통과해야 한다. 특히 **새 scalar key로 바꾼 코드의 λ=0.01 경로가 기존 P3와 의미상 동일**해야 한다.

비교가 무효인 경우:

- Donor 또는 U-Net initial tensor가 달라짐.
- LR·seed·augmentation 순서·sampling kernel·active update parity가 달라짐.
- Loff의 mean/sum, pixel/normalized-grid 단위가 달라짐.
- Reconstruction을 jittered PAN으로 수행함.
- Evaluator reference·ROI·selector가 달라졌는데 과거 점수 그대로 사용함.

과거 run의 이름이나 metadata를 새 캠페인으로 덮어쓰지 않는다. `reuse_registry`에 원 run, checkpoint/update/hash, 초기값, metric contract, 승인 여부와 재평가 비용을 기록한다.

재사용 검증에 실패하면 필요한 대조군을 새 matched block에서 재실행한다. **잘못된 대조군을 유지하고 세 번째 seed 수만 늘리지 않는다.** 그 비용은 buffer와 후순위 확인 block의 배정에서 먼저 확보한다 (§11.6).

### 11.4 24 GPU-hour 배정표

아래는 실측 소요 시간이 아니라 **실행 전 예약 예산**이다. v1이 인용한 NF16 학습 시간은 대체로 1.5시간 수준, S1의 1벌 비용 추정은 약 2시간이므로, 신규 run당 **2.0 GPU-h**를 예약한다. 실제 값은 새 코드와 같은 평가 cadence로 profiling한다. [S3; S1, §3; v1 §11.3]

| 작업 | 추가 학습 수 | 예약 GPU-h | 누적 GPU-h |
|---|---:|---:|---:|
| Metric provenance·donor/대조 재평가·구현/gradient gate·profiling | — | **2.0** | 2.0 |
| Seed 1234: L1E3 / L1E4 / L3E3, 각 50K | 3 | **6.0** | 8.0 |
| Seed 7777: P0 / L000 / λ*, 각 50K | 3 | **6.0** | 14.0 |
| Seed 2025: P0 / L000 / 같은 λ*, 각 50K | 3 | **6.0** | 20.0 |
| 최종 best/last 재평가·response/stress·gradient/영상 진단·결과 집계 | — | **3.0** | 23.0 |
| Buffer: 추가 재현 확인·재시작/예측 오차 여유 | — | **1.0** | **24.0** |
| **합계** | **기본 신규 9회** | **24.0** | |

Run당 2시간에는 **기존 cadence의 정규 학습 중 평가**를 포함한다. 마지막 3시간은 공통 best/last 재평가와 큰 stress/shortcut 진단 등 추가 작업이다. GPU에서 중복 계산한 실제 시간은 ledger에 모두 기록하고, 동일 작업의 비용을 예상표에서 이중 산정하지 않는다.

24시간은 상한이다. 재사용 또는 빠른 실행으로 시간이 남아도 새 λ, 새 radius, adaptive λ, conditioning/feature warp, KD를 자동으로 추가하지 않는다. 우선 미완료 대조·다른 seed 확인·기존 진단을 완료하고 남은 예산은 미사용으로 기록한다.

### 11.5 실행 순서와 후보 고정

```text
G0  기존 보고서·metric 정의 확인; SCC/fSCC key 확정
G1  N2 exact50K donor + seed1234 기존 P0/P2/P3의 best/last 재평가
G2  λ=0 / 0.01 동치 및 gradient unit/smoke, GPU 시간 profiling

A1  L1E3 / seed1234 / 50K
A2  L1E4 / seed1234 / 50K
A3  L3E3 / seed1234 / 50K
A4  §10.3의 기존 규칙으로 λ*를 한 번 선택·manifest에 고정

B1  CTRL-P0 / seed7777 / 50K
B2  L000    / seed7777 / 50K
B3  λ*      / seed7777 / 50K

C1  L000    / seed2025 / 50K
C2  λ*      / seed2025 / 50K
C3  CTRL-P0 / seed2025 / 50K

D0  모든 비교의 best_raw와 exact50K last를 같은 evaluator로 집계
D1  raw_original / raw_v64 / aligned_self / fixedN2 진단과 response 비교
D2  탐색 seed와 확인 seed를 구분한 최종 표·실사용 GPU-hour 기록
```

각 run의 forward 안에서는 기존대로 recon과 consistency gradient를 합산한다. 위 큐 순서는 **서로 독립적인 run들의 순서**이며, P0→L000→λ* 가중치를 이어 학습하는 순서가 아니다.

후보는 seed 1234의 **raw best HQNR → 확정 SCC → 더 작은 양의 λ**로 선택한다. 정확한 comparator는 §7·§8의 metric contract를 사용한다. 선택 직후 후보 ID와 source checkpoint/metric hash를 저장한다. Aligned 점수나 확인 seed 결과를 보고 λ*를 바꾸지 않는다.

**확인 종료 조건:** 세 중간 λ가 모두 seed 1234의 P2보다 raw HQNR에서 0.0031 초과 후퇴하면, v1 §10.3의 규칙대로 B/C의 추가 반복을 생략하고 원인 분석으로 종료할 수 있다. Metric·구현 검증에 실패하면 먼저 이를 해결하며, 부적격 결과로 후보를 고르지 않는다. 예산이 충분하다는 이유만으로 실패 후보를 의무 반복하지 않는다.

### 11.6 실측 시간 gate와 예산 초과 처리

각 run/block 시작 전에 실제 ledger와 갱신된 예상시간을 확인한다.

```text
H_spent + H_projected_next + H_reserved_evaluation + H_reserved_safety <= 24.0
```

- `H_spent`: 완료/중단 run, profiling, GPU 진단을 포함한 실제 누적 사용시간.
- `H_projected_next`: 같은 코드의 실측 학습 속도 + 평가 cadence를 이용한 남은 전체 run 또는 비교 block 예상치. 실측 추정치에 10% 시간 여유를 둔다. 이 10%는 예산 운영용 설계값이다.
- `H_reserved_evaluation`: 아직 수행하지 않은 최종 평가·진단 예산. 초기에는 3.0 GPU-h를 보호한다.
- `H_reserved_safety`: 아직 사용하지 않은 buffer. 초기에는 1.0 GPU-h다.

학습 run 단위 gate와 함께 **확인 block의 P0·L000·λ* 세 개가 완결 가능한지**도 확인한다. 이 검사를 통과하지 못하면 일부 case만 먼저 실행해 불완전한 비교를 만들지 않는다.

시간이 부족할 때의 순서:

1. 유효한 과거 대조의 재사용으로 중복 학습을 줄인다.
2. 검증 실패 대조군 재실행 비용을 우선 확보하고, **seed 2025 확인 block 전체를 먼저 보류**한다.
3. 여전히 부족하면 seed 7777 확인 block을 보류한다. 기본 탐색과 그에 필요한 matched 대조·metric 검증을 우선한다.
4. 핵심 탐색/대조 자체가 상한 안에 완결되지 않으면 예상 추가 비용과 미완료 항목을 기록하고 중단한다. 사용자 승인 없이 24시간을 넘기지 않는다.

어떤 경우에도 case별 50K를 20K/30K로 임의 축소하거나, LR·λ·radius를 run 중 바꾸어 시간을 맞추지 않는다. 예상보다 긴 run은 checkpoint를 보존해 미완료로 표시할 수 있지만, 이를 exact50K 결과로 집계하지 않는다. 재시작·NaN 실패에 쓴 시간도 제외하지 않는다.

### 11.7 반복 결과의 집계

λ*와 그 seed의 대조군에 대해 아래 차이를 계산한다.

\[
 d_{s,P2}=H_{\lambda^*,s}^{best\_raw}-H_{L000,s}^{best\_raw},\qquad
 d_{s,P0}=H_{\lambda^*,s}^{best\_raw}-H_{P0,s}^{best\_raw}.
\]

- seed 1234: **탐색/후보 선택에 사용한 결과**로 표시.
- seed 7777·2025: λ를 바꾸지 않고 수행한 **추가 학습 seed 확인 결과**로 표시.
- 각 seed의 원 점수, 대응 차이, 부호 일관성, 모든 seed의 요약 평균·표본 표준편차를 기록하되, 탐색을 포함한 평균과 확인 seed만의 평균을 구분한다.
- 같은 50K last에서도 별도 표로 복원·response를 비교한다. Best 점수에 last의 response를 붙여 한 checkpoint의 성능처럼 쓰지 않는다.
- 학습 seed를 바꿔도 평가 장면은 기존 FR 20장이다. 확인 seed가 **새로운 미사용 test set**을 뜻하지 않으며, 같은 test 입력에 의한 개발 선택 이력은 유지된다.
- HQNR 0.0031은 기존 경험적 판정선이다. 반복이 세 개 생겼다고 자동으로 새 유의수준이나 통계적 무손실 증명이 되는 것은 아니다. 작은 closure를 절대 정합 uncertainty로 바꾸지 않는다.

핵심 결과는 **“P2 대비 복원 품질과 반응 보존의 절충이 같은 λ에서 재현되는가”**다. P0도 함께 두어, 정합 사전학습을 포함한 전체 방법이 무정합 baseline 대비 어떤 비용·이득을 갖는지 분리한다.

---

<a id="config"></a>
## 12. Config 통합 template

아래는 **실행 semantics를 표현하는 template**다. 기존 repository의 실제 config key라고 주장하지 않는다. `null`인 필수 항목은 서버에서 채운 뒤 gate를 통과해야 한다. 로컬 확인 코드가 YAML을 읽는 것만으로 실제 trainer 연동이 검증되는 것은 아니다.

```yaml
protocol_id: PALS_W112D123_N2LAST_R200_v1
campaign_id: PALS24_W112D123_N2LAST_R200_v2
document_revision: 24GPUh_v2
case_id: L1E3
run_id: PALS24_L1E3_W112_D123_WV3_S1234_N2LAST_R200_v1
server: s1
seed: 1234

initialization:
  donor_run: PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT
  donor_kind: last
  donor_update: 50000
  donor_copy: aligner_only
  donor_path: null                 # Required: exact server path.
  donor_sha256: null               # Required: verified asset.
  unet_initial_state_path: null    # Required: matched saved tensor.
  unet_initial_state_sha256: null
  resume_from_finished_P2: false

model:
  width: 112
  depth: [1, 2, 3]
  input_channels: 9
  output_channels: 8
  attention: false
  pan_reconstruction: false
  mode_modulation: false
  aligner_arch: inherit_NF16
  aligner_trainable: true
  aligner_view_margin_hr: 4
  output_inverse_warp: false
  ms_base_warp: false
  gt_warp: false
  delta_output_cap: null

training:
  max_optimizer_updates: 50000
  effective_batch_size: 48
  unet_lr: 0.0001
  aligner_lr: 0.00001
  weight_decay: 0.01
  optimizer_other_settings: inherit_verified_NF16_P2
  scheduler: inherit_verified_NF16_P2
  warmup_updates: 100
  reconstruction_input: native_aligned_pan
  reconstruction_weight: 1.0
  reconstruction_loss: mean_L1
  reconstruction_domain: full_existing_domain
  geometry_weight: 0.0
  output_edge_weight: 0.0
  kd_weight: 0.0

offset:
  lambda: 0.001
  weight_schedule: constant
  every_optimizer_updates: 2
  active_remainder: 1
  stop_gradient_native_target: true
  detach_native_reconstruction: false
  feed_jittered_pan_to_unet: false
  reduction: mean_over_batch_and_two_components
  distribution: uniform_disk_area
  radius_hr: 2.0
  order: dy_dx
  unit: current_PAN_pixel
  apply_after_common_augmentation: true
  independent_rng: true
  radius_provenance: inherited_FR_informed_R200_development_setting

warp:
  source: inherit_verified_NF16
  mode: bicubic
  padding_mode: border
  align_corners: false

metrics:
  fr_manifest_id: fr_mat20
  fr_scene_count: 20
  evaluator_commit: null           # Required before training.
  dataset_manifest_sha256: null
  aggregation: mean_of_scene_HQNR_products
  primary_view: raw_original
  views: [raw_original, raw_v64, aligned_self_v64, aligned_fixedN2_v64]
  fixed_roi_margin_hr: 64
  aligned_reference_lowres: regenerate_from_chosen_highres_pan
  aligned_is_diagnostic_only: true

selection:
  primary_checkpoint: best_raw
  primary_metric: hqnr_raw_original
  report_secondary_name: SCC
  secondary_metric_key: null       # Resolve SCC vs fSCC; do not guess.
  secondary_split: null
  secondary_reference: null
  hqnr_tie_band: 0.0001
  tie_anchor: running_maximum
  second_tie_preference: later_update
  empirical_HQNR_method_margin: 0.0031
  save_exact_last: 50000
  save_best_aligned_diagnostic: true
  exclude_early_checkpoints: false

gates:
  require_metric_definition_resolved: true
  require_reference_reproduction: true
  require_gradient_routes: true
  require_control_initialization_match: true
  allow_silent_scene_exclusion: false
  allow_protocol_change_on_resume: false
```

**λ별 변경 허용:** `case_id`, `run_id`, `offset.lambda`, 출력 경로 및 시간 기록. Seed 반복 때는 seed와 그 seed의 matched initial tensor/RNG만 함께 바뀐다. 모델·loss·평가의 나머지 의미는 동일해야 한다. `CTRL-P0`는 표의 λ sweep 모델이 아니라 **기존 NF16 P0 정의대로 aligner와 sampler를 제거한 별도 no-align control**이다.

### 12.1 Campaign budget template — trainer 설정과 분리

아래는 큐 운영용 개념 template이다. 기존 실행기의 실제 key인지 확인하여 연결한다. 모델 학습의 λ/LR schedule을 바꾸는 설정이 아니다.

```yaml
campaign_id: PALS24_W112D123_N2LAST_R200_v2
budget_gpu_hours: 24.0
default_server: s1
default_gpu_count: 1
parallel_training_runs: 1
training_updates_per_run: 50000
screen_seed: 1234
screen_lambdas_new: [0.0001, 0.001, 0.003]
screen_control_lambdas: [0.0, 0.01]
confirmation_seeds: [7777, 2025]
confirmation_cases: [CTRL-P0, L000, SELECTED_LAMBDA]
selected_lambda: null       # Fix once after the complete seed1234 screen.
selection_rule: inherit_resolved_raw_HQNR_then_SCC_then_smaller_lambda
reselect_after_confirmation: false
donor_seed_fixed: 2025
reuse_requires_verified_equivalence: true
projected_training_hours_per_run: 2.0
empirical_eta_margin_fraction: 0.10
reserved_initial_gate_gpu_hours: 2.0
reserved_final_evaluation_gpu_hours: 3.0
reserved_buffer_gpu_hours: 1.0
optional_blocks_drop_order: [confirmation_seed2025, confirmation_seed7777]
allow_unequal_update_budget: false
allow_automatic_new_lambda_or_architecture: false
```

---

<a id="outputs"></a>
## 13. 산출물·재현·보고 형식

### 13.1 Run별 필수 파일

```text
work_dir/<run_id>/
  config_resolved.yaml
  provenance.json
  metric_contract.json
  source_asset_hashes.json
  train_log.jsonl
  gradient_diagnostics.jsonl
  checkpoint_metrics.csv
  scene_metrics.csv
  delta_predictions.csv
  offset_response_<checkpoint>.csv
  results/po10_diag_<checkpoint>.json
  best_raw / best_raw_meta.json
  last / last_meta.json
  best_aligned_diagnostic / metadata
  evaluation_reproduction.json
  budget_ledger.json
```

File extension은 기존 서버 규칙을 따른다. 이름만 `last`인 파일이 아니라 optimizer update가 정확히 50K인 것을 확인한다.

**24시간 캠페인 추가 기록:** `campaign_manifest.yaml`, `reuse_registry.json`, `selected_lambda.json`, `campaign_budget_ledger.json`, `paired_seed_results.csv`를 캠페인 수준에서 남긴다. 탐색/확인 단계, donor 고정 여부, GPU 개수, 실제 사용·예약·미사용 시간, 보류 사유와 미완료 run을 구분한다.

### 13.2 결과표를 세 개로 분리

**표 A — 공식 품질 판정**

```text
case, seed, lambda, selected_update, checkpoint_hash,
HQNR_raw_original, SCC(확정 key), D_lambda, D_s,
RR_ERGAS, RR_PSNR, RR_SAM, delta_H_vs_P2, delta_H_vs_P0
```

**표 B — 같은 50K에서 학습 동작**

```text
case, seed, exact_update,
raw_original, raw_v64, aligned_self_v64, aligned_fixedN2_v64,
RR_metrics, native_delta_dy_dx, native_delta_norm,
B_resp_yy_yx_xy_xx, component_MAE, EPE, legacy_closure
```

**표 C — 효과 귀속과 구현 검증**

```text
grad_rec_norm, grad_off_norm, weighted_grad_off_norm, cosine,
raw_Loff, weighted_Loff, support_validity, source_hashes,
metric_reproduction_error, interpolation_MSswap_controls
```

학습 로그의 RR metric과 export MAT 평가의 RR metric을 같은 열로 합치지 않는다. 원자료가 다르면 포맷/정밀도/clip·rounding 정책을 별도 표기한다.

Sheet의 공식 HQNR 열은 best_raw raw-original로 유지한다. Aligned 값은 별도 진단 표나 분명히 표시한 열에 넣고 기존 HQNR/V64 헤더를 덮어쓰지 않는다. 이 명세 작성만으로 Sheet를 변경한 것은 아니다.

### 13.3 재현성 관련 필수 주의

- Aligner의 가중치 update와 source PAN copy에 대한 `no_grad()`를 혼동하지 않는다.
- Random seed만 같다고 GPU/platform 간 완전한 수치 동치를 보장하지 않는다.
- CUDA grid sampling backward의 비결정성 여부를 설치 버전과 함께 기록한다. [R2]
- Gradient test는 작은 proxy만으로 끝내지 않고 실제 wrapper에서 수행한다.
- ROI·mask·sampling phase를 변경했다면 baseline을 포함해 새 metric version으로 재평가한다.
- Source path·metadata·각 loss 유무가 summary 문자열과 실제 graph에서 일치하는지 확인한다. 범용 trainer의 ‘KDV’ 같은 표시만 보고 KD가 적용되었다고 판단하지 않는다.

---

<a id="limits"></a>
## 14. 범위 제한과 다음 단계

**이번 계획은 사용자 요청에 따른 P2–P3 사이의 좁은 가중치 검증이다.** S1의 최종 권고는 warp 없는 Δ conditioning(B)을 먼저, PAN feature 정렬(A)을 다음으로 검토하는 것이며, 이번 계획이 그 권고와 동일한 것은 아니다. [S1, §4–5]

세 중간 λ와 대응 대조에서 양립 구간이 없으면:

1. 더 많은 λ·b·스케줄을 무제한 추가하지 않는다.
2. 입력 PAN을 실제로 옮기는 적용 경로가 병목이라는 보고서 해석을 다시 검토한다.
3. 다음 구조 후보는 S1의 B/A 순서로 별도 명세·승인 후 진행한다.
4. 실패의 범위는 이번 구조·분포·loss에서의 유효성 미확보다. 모든 정합 정보나 모든 conditioning 설계의 불가능을 주장하지 않는다.

**원문과 별개로 새로 검증해야 할 것:** P2의 native 보정 감소와 반응 보존 여부, 약한 λ의 실제 gradient 영향, 같은 checkpoint에서 RR/raw/geometry가 함께 좋아지는지, SCC selector 정의, S1의 일부 Dλ 수치 provenance.

**v1에 기록된 완료 사항:** 보고서·기존 명세 검토, Sheet NF16 행 조회, 실행 설계 및 제한적인 CPU proxy 검사.

**이번 v2에서 완료한 것:** 원본 계획의 24 GPU-hour 예산·반복·실행 순서 개정과 문서 형식/예산 합계 검사. v1의 CPU proxy 검사나 서버 metric 재평가를 이번에 다시 수행한 것은 아니다.

**현재 완료하지 않은 것:** 실제 s1 학습, project evaluator의 원본 MAT 재평가, P2/P3 last response 재측정, 2σ 원시 자료 재계산, 신규 λ의 성능 검증. 서버 코드·큐·Sheet는 변경하지 않았다.

---

<a id="sources"></a>
## 15. 근거 자료와 출처 추적

| ID | 자료 | 사용 범위 |
|---|---|---|
| **S0** | `PAN_P2_P3_LambdaSweep_MetricAware_Plan_2026-09-12.md` 및 사용자의 24시간 예산 변경 지시 | 승인된 원본 방법·평가 규약; 이번 개정에서 시간·반복만 확장 |
| **S1** | `PAN_Aligner_Directions_2026-09-12.md` | §1–2 HQNR/프레임 해석, P1 inverse 재평가, P2 크기 감소; §4 대안 우선순위; §6 최신 판정 규약 |
| **S2** | `PAN_N2_NativeFitting_16GPUh_W112_D123_2026-09-11.md` | §2–5 초기화·forward·loss·schedule; §6–8 평가·진단; 나머지는 현재 계획에 맞게 범위 제한 |
| **S3** | Google Sheet `pan-cvpr27`, `WV3-s1`, NF16 28–35행; **v1에서** 2026-09-12 직접 조회한 기록 | 기존 P0/P2/P3 best_raw 수치, case 설정, 학습 시간. 새 last 결과를 확인한 자료가 아님 |
| **S4** | `2026-09-10_pa-a1-a3-s1-results.md` | §10.2 PO10 best/50K 차이, §10.5 response, §10.6 V64 정의 |
| **R1** | PyTorch 공식 `Autograd mechanics` | 공개 구현 참고: 계산 graph와 gradient 차단. 프로젝트 실험 결과의 증거가 아님 |
| **R2** | PyTorch 공식 `torch.nn.functional.grid_sample` | 공개 구현 참고: sampler 좌표·padding·gradient/비결정성 주의. 실제 서버 버전은 별도 확인 |

공개 참고 주소:

```text
R1 https://docs.pytorch.org/docs/2.13/notes/autograd.html
R2 https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.grid_sample.html
```

Sheet 위치는 다음 식별자로 보존한다. Private 자료를 공개 웹 검색으로 대체하지 않았다.

```text
spreadsheet_id: 1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0
sheet: WV3-s1
sheet_id: 994031662
queried_campaign: NF16
read_date: 2026-09-12
```

### Source file hashes

| 자료 | SHA-256 |
|---|---|
| S0 승인 원본 계획 | `367ddd2041338d3ebe2c3a6c633d47ee542aa087a8a9d4f7388e150368609cfd` |
| S1 `PAN_Aligner_Directions_2026-09-12.md` | `268a7eea6142141339ea29746cfc2d8133403342c02cd24511005db29373a331` |
| S2 `PAN_N2_NativeFitting_16GPUh_W112_D123_2026-09-11.md` | `1d895e87e018ac3d17f367a4d330350f38194c07b003f042a211c4e9d5a216f9` |
| S4 `2026-09-10_pa-a1-a3-s1-results.md` | `3fd15614bf37ebecd9a96af0d726302cedc13ae7d4696f30ff2559819e6a539e` |

원문에 링크되어 있으나 확보하지 못한 결과·코드의 내용은 새로 만들어 넣지 않았다. S1의 강한 원인 해석은 해당 보고서의 해석으로, 중간 λ·실행 순서·일부 진단 tolerance는 이번 계획의 제안으로 구분했다.

---

## 실행 요약

**추가 총 24 GPU-hours 안에서, seed 1234 중간 λ 세 개를 탐색하고 고정한 λ*를 P0·P2와 함께 seed 7777·2025에서 확인한다. 기존 대조군 재사용 통과·후보 진행 조건 충족 시 기본 신규 학습은 9회이며, 모두 50K다.**

**같은 N2 last aligner + 같은 새 W112–D123 U-Net에서 출발하여 `λoff = 0, 0.0001, 0.001, 0.003, 0.01`을 비교한다.**

**Native reconstruction은 두 모듈을 매 update 공동 학습시키고, 같은 PAN의 jittered 복사본에 대한 SG offset consistency는 격번으로 aligner만 직접 감독한다.**

**주 성능 판정은 원 PAN·논문 20장·best_raw HQNR → 확정된 SCC, 경험적 판정선은 0.0031이다. V64와 aligned view 및 exact50K last는 필수 진단이지만, 낮은 raw 성능을 구제하는 대체 점수로 사용하지 않는다.**


---

## 부록. v1 작성 시 기록한 제한적 검사 — 이번 v2의 재실행 결과가 아님

**환경:** CPU PyTorch `2.10.0+cpu`. 작은 proxy CNN과 합성 tensor를 사용했다. 실제 W112–D123 U-Net, 프로젝트 aligner, PanCollection 영상, CUDA trainer를 사용한 검사가 아니다.

| 검사 | 결과 |
|---|---|
| Lrec → 두 module, Loff → aligner만 | Proxy 계산 graph에서 확인 |
| Offset target의 c0 detach, native recon의 c0 gradient 유지 | 확인 |
| 5개 λ에서 `g_total = g_rec + λg_off` | 최대 절대 오차 8.7e−19 이하 |
| 한 update에서 복원 forward 1회, optimizer update 1회로 두 module 변경 | Proxy에서 확인 |
| b=2 원판 표본 100,000개 | 축 표준편차 약 0.9995/0.9997, 평균 반경 1.3325, 최대 반경 2 미만 |
| `out[y,x]=src[y+dy,x+dx]` 부호 | 합성 ramp에서 확인 |
| 장면별 HQNR 곱의 평균과 평균들의 곱 구분 | 합성 수치의 covariance 항 관계 확인. 실제 HQNR 평가 아님 |
| Markdown code fence·수식 delimiter·YAML parsing | 확인. 필수 서버 필드는 미해결 상태로 유지 |

이 결과는 **loss 연결의 논리와 문서 내 설정 형식**을 확인한 것이다. §8의 실제 metric 재현 gate, 실제 모델의 gradient 검사, 서버 학습을 대체하지 않는다.


### v2 문서 검사 범위

24시간 배정의 합계(2+6+6+6+3+1), 기본 신규 run 수(3+3+3), Markdown code fence/수식 delimiter, YAML template의 구문과 λ·seed·budget 일치를 로컬에서 확인한다. 실제 서버 학습 시간, GPU gradient/metric gate, 신규 λ 성능은 실행 전 또는 실행 후 확인할 사항으로 남긴다.
