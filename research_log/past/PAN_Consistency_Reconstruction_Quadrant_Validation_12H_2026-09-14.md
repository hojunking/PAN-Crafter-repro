# DCR12 — PAN offset consistency × reconstruction 불일치 검증 실험
## 사분면 진단 → 보정의 실제 이득 → Aligner 적응 → 조건부 KD routing

**작성일:** 2026-09-14  
**문서 성격:** 실행 설계·신규 진단 명세. 학습·실측 분석을 완료한 보고서가 아니다.  
**권장 예산:** 단일 GPU 기준 약 **12–16 GPU-hours**, 12시간에서 강제 종료하지 않는다. 필요한 대응 run을 완주하면 더 초과해도 된다.  
**대상:** 현재 PAKD50의 **WV3 · W112–D123 · 9ch · PAN aligner + 동일 용량 Teacher–Student**.  
**주 성능:** 원본 FR 논문 20장, 기존 evaluator의 **raw-original HQNR**.  
**실행 호스트:** 자산이 준비된 한 서버를 먼저 지정한다. s2/s3/s4 어느 서버든 가능하나, 핵심 대조쌍은 같은 호스트에서 끝낸다.  
**연구 질문:** Offset consistency와 reconstruction의 불일치가, Teacher 참조 사용과 Student aligner 적응을 구분하는 유효한 신호인가?

> **핵심 결정:** 처음부터 네 사분면마다 hard/soft/learning-rate를 다르게 만들지 않는다. 기존 Q12를 보존한 채 **읽기 전용 진단과 B0–B1 대조**부터 수행한다. 사분면이 실제로 추가 정보를 제공할 때만 B2–B3의 soft 조절을 해제한다. 새로운 variance·covariance loss나 uncertainty head는 추가하지 않는다.

---

# 0. 실행자가 먼저 볼 요약

## 0.1 필수 순서

| 순서 | ID | 작업 | 핵심 산출물 / 질문 |
|---:|---|---|---|
| 1 | D00 | 자산·코드·설정·공통 격자·처리량 점검 | 같은 Teacher, 초기값, loss, 평가 조건인가? |
| 2 | D01 | Teacher/Student의 consistency–reconstruction 사분면 | 어떤 불일치가 실제로 존재하는가? |
| 3 | D02 | 같은 backbone에서 correction만 변경 | Learned correction이 native 복원에 이득인가? |
| 4 | D03 | Aligner의 task/offset/soft gradient 분해 | 두 loss가 같은 파라미터에서 충돌하는가? |
| 5 | B0/B1 | **FQ ↔ JK0**, 두 seed의 대응 50K | GT 기반 aligner 적응이 frozen보다 유효한가? |
| 6 | D04 | 제한된 KD micro-update 진단, 조건부 | 사분면이 Teacher soft의 유용성을 추가로 설명하는가? |
| 7 | B2/B3 | 검증된 사분면 soft 조절, 조건부 | 분류 정보와 aligner 적응을 결합할 가치가 있는가? |

**12시간 핵심 묶음은 D00–D03 + B0/B1 두 seed + 보고서다.** D04/B2/B3는 자동 필수 작업이 아니다. 이들이 열리면 12–16시간을 넘는 연장 묶음으로 구분한다.

## 0.2 기존 코드와의 정확한 대응

| 본 문서 | 기존 PAKD50 case | Student backbone | Student aligner | 구현 상태의 근거 |
|---|---|---|---|---|
| **B0** | **FQ** | Q12 전체 | Frozen | 현재 generator에 정의됨 [S2] |
| **B1** | **JK0** | Q12 전체 | GT hard + GT edge + offset; **soft→A 차단** | 현재 generator에 `routing.qK=0` 정의 [S2] |
| B1-ALL | JQ | Q12 전체 | Q12 전체 + offset | Soft→A 경로의 영향 확인용 보조 대조 [S2] |
| B2 | FQ + 사분면 multiplier | Q12의 soft만 수정 | Frozen | **이번 신규 제안, 미구현으로 취급** |
| B3 | JK0 + 같은 multiplier | B2와 동일 | B1과 동일 | **이번 신규 제안, 미구현으로 취급** |

`JK0`의 0은 **Student 전체의 soft를 0으로 만드는 것**이 아니다. **Aligner로 전달되는 soft gradient만 0**이고, U-Net에는 soft가 그대로 간다. `X02/R1+EDGE`와 다르다.

기존 JQ를 B1 대신 쓰면 질문이 달라진다. JQ에는 soft→aligner까지 포함되므로, B1(JK0)로 표시하지 않는다.

## 0.3 완료 자산은 재사용하되, 빈칸을 완료로 가정하지 않는다

현재 저장소 정의를 확인했지만 **실제 로컬 완료·실행 중 상태와 새 사분면 CSV는 이번에 확인하지 않았다.** 운영자는 local status/checkpoint로 확인한다. 정확히 같은 조건의 완료 FQ/JK0가 있으면 재학습하지 않고 진단만 보강한다. 과거 NA104의 W104 no-align Q12는 본 B0의 대조군이 아니다.

---

# 1. 무엇을 검증하는 실험인가?

## 1.1 사전에 구분할 다섯 가설

| 가설 | 질문 | 긍정 결과가 허용하는 범위 |
|---|---|---|
| H1 | 작은 offset consistency가 낮은 native GT L1과 연관되는가? | 해당 데이터·checkpoint에서의 예측 관계 |
| H2 | 같은 지표가 native FR HQNR 또는 learned correction의 이득과 연관되는가? | 해당 평가 조건에서의 복원 관련성 |
| H3 | Native 복원과 offset consistency가 Aligner 파라미터에서 충돌하는가? | 국소적인 최적화 충돌 / 협력 |
| H4 | GT hard·edge·offset으로 A를 적응시키면 frozen A보다 HQNR이 좋아지는가? | 현재 Q12 조건에서의 적응 묶음의 효과 |
| H5 | Consistency가 e_T/a_T만으로는 설명되지 않는 KD 유용성을 제공하는가? | 사분면 routing의 후속 검증 근거 |

H1이 양성이라고 H4/H5가 자동으로 양성이 되는 것은 아니다. H3가 음의 내적을 보였다고 전체 학습을 중단하거나 consistency를 바로 제거하지 않는다.

## 1.2 전제를 잘못 두지 않는다

기존 offset consistency는 알려진 추가 변위의 상대 관계를 감독한다 [S4].

$$
c_0=A(P,M),\qquad c_\epsilon=A(W(P,\epsilon),M),
$$
$$
C=\operatorname{mean}|c_\epsilon+\epsilon-\operatorname{sg}(c_0)|.
$$

`c0=b, cε=b−ε`이면 b가 native 정합의 참값과 달라도 C=0이다. 따라서 **C가 작다 = 절대 정합 성공**이라고 전제하지 않는다. 두 입력에 똑같은 상수 b를 출력하면 C는 mean|ε|로 남는다는 다른 경우와도 구분한다.

두 loss가 같은 방향으로 움직여야 한다고 가정하지 않는다. 반대 방향은 데이터 특성, bias, 복원망의 PAN 이용 정도, 보간·경계, 학습 단계, loss gradient의 차이 등 여러 원인과 양립한다. 이번 실험은 그 원인을 좁히는 진단이다.

## 1.3 자료에서 이미 알려진 것과 아직 모르는 것

기존 NA104 보고서는 W104 no-align에서 Q12가 s2의 세 seed에 대해 양의 HQNR 차이를 보였고, s3에서는 같은 방향이 유지되지 않았다고 기록한다 [S6, S7]. 이 때문에 Q12의 정의를 고정한 대조를 사용한다. **그 자료는 consistency–reconstruction 관계의 증거가 아니다.**

과거 보고서의 계수비·손실비·noise floor는 새 통합 실험의 gradient 비율이나 합격선으로 복사하지 않는다. 보고서 저자의 해석과 실측 표를 구분하며, 현재 두 loss의 관계는 본 실험에서 새로 측정한다.

---

# 2. 기준 파이프라인과 바꾸지 않는 값

## 2.1 현재 PAKD50 계약을 사용한다

| 항목 | 이번 계승값 / 처리 |
|---|---|
| Backbone | **W112, depth=[1,2,3]**, Teacher/Student 동일 용량 |
| 입력 | PAN1 + 기존 bicubic MS8, 총 9ch |
| 출력 | residual8 + 움직이지 않는 MS base M |
| Aligner | 기존 global `(dy,dx)` CNN, sample별 벡터 |
| Aligner view | 기존 margin4, crop 후 bandwise z-score |
| Warp | bicubic, border, align_corners=False; PAN만 sampling |
| 좌표 | M·GT·최종 출력 고정. 역방향 출력 warp 없음 |
| Teacher | 고정 T0의 A/U 쌍, frozen/eval/no-grad |
| Student init | A는 T0의 A 복사, U는 seed별 저장 독립 초기값 |
| 학습 | 50K successful update, effective batch48, AdamW WD0.01 |
| LR | U peak1e-4, trainable A peak1e-5 |
| 스케줄 | warmup100+cosine, **처음부터 50K horizon** |
| Reconstruction | **모든 update에서 native PAN을 복원** |
| Offset 학습 | 홀수 successful update에만, λoff=1e-4, 별도 ramp 없음 |
| Corruption | PAN 복사본만 radius2 HR px의 disk-area-uniform; **offset용 A 호출** |
| Q12 | α=1, β=0.1, ε=1e-6, τ/λE는 현재 공통 calibration asset |
| 제외 | 새 covariance/variance loss, geometry loss, uncertainty head, feature KD, dense flow, Teacher 교체 |

위 값은 현재 통합 계획과 generator/구현 노트에서 확인한 정의다 [S1–S3]. Adam beta/eps, min LR, AMP·clip·gradient accumulation, intensity/export 세부값은 성공한 resolved config에서 상속한다. 라이브러리 기본값을 임의로 넣지 않는다.

**오래된 W96 N2의 native/corrupted U-Net 교대·λoff=.01·radius1을 이식하지 않는다.** 현재 통합에서는 U-Net이 항상 native를 보고, 추가 PAN 이동은 consistency용 Aligner 호출에 사용된다.

## 2.2 T0와 calibration

논리적 T0:

```text
PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1
source checkpoint: 계획의 best_raw = 저장소의 best_hqnr
배포 layout: assets/pakd50/T0_run/
```

원본 A/U는 같은 source checkpoint에서 가져온다. 소스 노트에 selected update 24240이 기록돼 있지만, 실제 파일 메타데이터와 full SHA를 로컬에서 확인한 후 고정한다. 경로 문자열·짧은 hash만으로 동등성을 선언하지 않는다.

- τR: T0+현재 preproc의 train-only Teacher error scale.
- λE: 공통 J0 seed1234 exact50K pilot의 출력 gradient RMS calibration.
- `r_grad=0.05`는 `lambda_E=0.05`가 아니다.
- No-align τ=0.012367을 복사하지 않는다. 새 seed마다 λE를 재보정하지 않는다.
- Calibration 자산이 없으면 정식 Q12 학습은 시작하지 않는다. J0 pilot을 먼저 확보하며 그 비용은 추가한다.
- Teacher 입력은 항상 자기 frozen A를 사용한다. Trainable Student A를 Teacher와 공유하지 않는다.
- B0에서 correction 공유 최적화를 쓰는 경우에도 두 A와 view가 동일한지 확인하고 Teacher 출력 불변을 검증한다.

## 2.3 공통 평가 격자 — 계획과 구현의 차이를 명시한다

원 통합 계획에는 `GRID1K_50K_v1`이 제안됐지만, 실행 노트와 generator는 **`GRID1010_50K_v1`**로 대체했다고 명시한다 [S2, S3]. 본 계획은 실제 실행 규약을 계승한다.

```text
candidates = {1010, 2020, ..., 49490, 50000}  # 50개
plateau = {45450, 46460, 47470, 48480, 49490, 50000}  # 6개
selector = 기존 best_hqnr(raw HQNR, tie band 1e-4 → 기존 fSCC comparator)
```

숫자상 최대 `hqnr_run_max_value`와 공식 선택 자산의 HQNR을 구분한다. **선택 자산의 모든 metric은 같은 checkpoint에서** 계산한다. Full precision으로 paired 차이를 계산하며, 표시값 반올림으로 합격 처리하지 않는다.

RR 진단과 FR HQNR은 서로 다른 split이다. 같은 scene 번호를 실제 동일 지상 장면으로 추정해 연결하지 않는다. FR에는 GT L1을 계산하지 않는다. `M`을 FR GT로 대신 쓰지도 않는다.

---

# 3. 이번에 필요한 loss·수신 gradient 정의

## 3.1 Q12는 그대로 둔다

$$
e_T(p)=\frac1C\sum_c|\hat Y_{T,c}(p)-Y_c(p)|,\quad
e_S(p)=\frac1C\sum_c|\hat Y_{S,c}(p)-Y_c(p)|,
$$
$$
k(p)=\frac1C\sum_c|\hat Y_{S,c}(p)-\operatorname{sg}(\hat Y_{T,c}(p))|,
$$
$$
d_T=\frac{e_T}{e_T+\tau_R},\qquad
 a_T=\frac{[e_S-e_T]_+}{e_S+10^{-6}}.
$$

$$
L_H=\langle(1+d_T)e_S\rangle,\quad
L_K=0.1\langle(1-d_T)a_Tk\rangle,\quad
L_Q=L_H+L_K+\lambda_E L_E.
$$

Teacher/GT/gate는 detach하되 실제 Student reconstruction·edge는 live다. 모든 위치에 GT hard가 남는다. 이번 핵심 비교에서 `(1-d_T)`를 제거하거나 β를 조정하지 않는다.

$$
L_E=\frac12\langle|K_x*\hat Y_S-K_x*Y|\rangle+
\frac12\langle|K_y*\hat Y_S-K_y*Y|\rangle.
$$

기존 signed Scharr/32, reflect1, 내부1px, 밴드별 비교다. GT edge에 사분면 가중치를 새로 곱하지 않는다. 앞단 wrapper에서 같은 reconstruction/edge를 두 번 합산하지 않는다.

## 3.2 B0/B1의 parameter별 목적

Student U-Net θ에는 두 case 모두:

$$g_\theta=\nabla_\theta L_Q.$$

B0(FQ)의 Student A는 완전히 고정한다.

B1(JK0)의 Student A φ에는:

$$
g_\phi=\nabla_\phi\left[L_H+\lambda_E L_E+
\mathbf1[t\bmod 2=1]10^{-4}L_{\rm off}^S\right].
$$

**Soft의 φ gradient만 빼고 θ gradient는 유지한다.** 이는 기존 `qD=1,qK=0,qE=1` 정책이다 [S1, S2]. 구현에서 단일 scalar total만 기록하지 말고 `objective_U`, `objective_A`, `routing_policy`를 각각 기록한다.

`B1−B0`는 GT hard·edge·offset을 이용한 **A 재적응 묶음**의 효과다. 이 대조만으로 off 자체의 효과나 plain reconstruction 하나의 효과를 주장하지 않는다.

B1-ALL(JQ)은 φ에도 LK가 전달된다. `JQ−JK0`가 soft→A 경로의 추가 효과다. 이미 동일 block의 JQ가 있으면 우선 진단에 재사용하고, 새 학습은 핵심 B0/B1 이후 필요한 경우만 수행한다.

---

# 4. D00 — 자산·계측·수치 검사 (필수)

## 4.1 실행 manifest

정식 학습 전에 다음 항목을 machine-readable JSON으로 고정한다.

```text
plan_id, host, GPU/driver/torch version, git commit, dirty diff hash
case_id, existing_case_alias, run_id, seed, parent/init hash
Teacher checkpoint file SHA, A tensor SHA, U tensor SHA, source selected update
Student init U/A tensor SHA, optimizer/scheduler definitions
calibration SHA, tau_R, lambda_E, radius/train sampler/view manifest
data/split IDs, scene-group metadata, intensity/upscaler/export/evaluator hash
candidate grid, comparator, loss support, observed throughput
artifact status: existing / running / pending / blocked
```

Teacher A와 Student A는 복사 초기값일 뿐 객체·buffer를 공유하지 않는다. 같은 T0 asset을 두 서버에서 사용한다면 serialized 파일 SHA뿐 아니라 정규화한 state tensor hash를 비교한다.

## 4.2 필수 smoke

| 검사 | 합격 조건 / 주의 |
|---|---|
| Warp 부호 | `(dy,dx)`와 grid `(x,y)` 순서, known shift 상쇄를 synthetic 비영 패턴으로 확인 |
| Native task→A | B1에서 LH/edge의 A gradient가 연결. 초기 특수값만으로 단정하지 않음 |
| Soft routing | 같은 모델 상태에서 B1의 A gradient는 HK제외 식과 일치, U gradient는 JQ와 일치 |
| Offset→U | U gradient 없음 |
| Frozen asset | B0 A와 T A/U의 weight·buffer hash 불변 |
| Teacher 출력 | 같은 입력을 Student 업데이트 전후 넣어 T 출력 불변 확인 |
| Identity | zero correction도 같은 sampler 사용. 학습 중 zero shortcut 금지 [S5] |
| Probe 0 | C(ε=0)은 numerical floor 수준; 주 consistency 평균에는 넣지 않음 |
| Forward 기준 | 기본 설정에서 기존 FQ/JK0 구현과 동일 출력·loss; 신규 진단은 결과를 바꾸지 않음 |
| RNG 격리 | 진단 ON/OFF가 Student init·data/probe order를 바꾸지 않음 |

Tolerance는 동일 입력 반복과 현재 FP32 연산의 numerical floor를 먼저 기록해 정한다. 수치 smoke를 통과했다고 CUDA 전체 재현이나 성능 검증을 완료한 것으로 쓰지 않는다.

---

# 5. D01 — 네 사분면을 실제 자료에서 정의한다 (필수)

## 5.1 데이터 panel — 신규 진단용 제안값

가능한 기존 scene/group split을 우선 사용한다. 새 panel 기본 목표는 다음과 같다.

| Panel | 기본 크기 | 용도 |
|---|---:|---|
| CAL | train 1024 patches | C/R 중간 기준과 nuisance scale 설정 |
| DISC | train 512 patches | 관계·개입·KD utility 탐색 |
| CONF | train 512 patches | DISC와 분리한 확인; routing 생성에 test 사용 방지 |
| GRAD | DISC의 고정 64 patches | gradient 분해; 가능하면 사분면별 균형 추출 |
| RR-test | 기존 RR 20 scenes | 고정 Teacher 사분면별 외부 기술통계, GT L1 |
| FR-test | 기존 FR 20 scenes | 공식 HQNR; 사분면 gate 학습에는 쓰지 않음 |

CAL/DISC/CONF의 ID를 seed314159로 결과를 보기 전에 고정하고 **서로 겹치지 않게** 한다. 가능하면 원 장면 단위로 분리한다. 장면 metadata가 없으면 `independence=patch_only`로 명시하고 held-out scene 일반화로 주장하지 않는다. 이때 B2/B3의 gate 개방은 보류하고 기본 진단·B0/B1은 계속 수행한다.

모든 panel은 진단 시 augmentation OFF다. Train data에서 뽑은 진단 panel은 B0/B1 기본 학습에서 배제하지 않는다. 이는 모델의 미사용 test가 아니라 **분석 규칙의 개발/확인 분리**다. 진짜 일반화 평가는 기존 RR/FR 또는 별도 미사용 자료로 수행한다.

기존 train calibration 3072 patch 전체를 위 panel로 다시 정의하지 않는다. **KD의 τ/λE calibration과 사분면 기준 calibration은 다른 manifest**다.

## 5.2 고정 probe

새 진단용 probe만 다음과 같이 고정한다. 학습의 radius2 disk-uniform sampler는 바꾸지 않는다.

$$
\mathcal E=\{r(\sin\theta,\cos\theta): r\in\{0.5,1.0\},\theta=k\pi/4,k=0,\dots,7\}.
$$

총 16개, 단위는 현재 PAN pixel이다. θ convention은 dy/dx 순서에 맞춘다. θ를 π/8 회전한 동일 크기의 별도 16개를 probe-confirm 집합으로 둔다. 주 평균은 두 반경별 결과를 먼저 보고하며, 합친 값만으로 크기 의존성을 숨기지 않는다.

ε=0은 identity test용으로만 둔다. Padding을 보지 않도록 기존 margin4 view가 해당 probe의 support를 만족하는지 확인한다. 두 단계 sampling 또는 더 큰 bias 개입에서는 지원영역을 재검사한다.

## 5.3 값과 label

같은 pipeline j∈{Teacher, Student}에 대해:

$$
C_{j,i}=\frac1{2K}\sum_{\epsilon\in\mathcal E}\sum_{q\in\{y,x\}}
|c_{j,i,\epsilon,q}+\epsilon_q-c_{j,i,0,q}|,
$$
$$
R_{j,i}=\operatorname{mean}_{c,p}|\hat Y_{j,i,c}(p)-Y_{i,c}(p)|.
$$

Teacher의 `tC=median_CAL(C_T)`, `tR=median_CAL(R_T)`로 고정한다. `<=`는 low, `>`는 high다. 이 경계는 **좋고 나쁨의 물리적 참값이 아니라 진단상 상대 구간**이다.

| Label | C_T | R_T | 허용하는 해석 |
|---|---|---|---|
| A | low | low | Probe 반응과 native 복원 오차 모두 상대적으로 낮음 |
| B | low | high | 일관된 잘못된 native bias 또는 비기하학적 복원 한계 가능 |
| C | high | low | Native 복원은 양호하지만 probe 반응 불안정 가능 |
| D | high | high | 두 축 모두 상대적으로 큼; 단일 원인 확정 불가 |

C 변동이 반복 수치오차와 구분되지 않거나 median 동률로 일부 셀이 사라지면 `DEGENERATE_METRIC`으로 기록한다. Rank를 임의로 흔들어 강제로 네 cell을 만들지 않는다. Empty/small cell은 그대로 남긴다.

Student는 별도 `(C_S,R_S)` 표를 만든다. 주 비교 bin은 동일 seed B0의 **고정 reference checkpoint**에서 정의해 B1에도 적용한다. 학습마다 새로 구간을 나눠 B1의 쉬운 sample만 특정 cell로 옮기는 선택 편향을 피한다. 각 method 자체의 동적 상태도 별도 표로 기록할 수 있으나 직접 대조와 섞지 않는다.

Teacher는 고정돼 있어 동일 입력의 C_T/R_T/label이 Student step마다 바뀌지 않는다. 변화하면 입력·probe·view·집계 차이를 점검한다.

## 5.4 분석과 불확실성

필수: C–R scatter, Spearman, 보조 Pearson, 각 cell의 n·median/IQR·R·edge-L1·C, correction magnitude, PAN texture/밴드별 대비·포화 정도. 시간 추세는 마지막 항목으로 분리한다.

- 높은 C/high R cell의 평균 R이 높은 것은 label 정의상 자명하다. 그것을 추가정보로 제시하지 않는다.
- `eT/aT`와 C의 정보 차이는 R_T 분위·texture 수준이 비슷한 sample에서 D02의 **보정 이득** 또는 D04의 **KD 실제 효과**가 달라지는지로 확인한다.
- 통계량의 불확실성은 원 장면 group bootstrap 1000회로 제시한다. 픽셀·probe를 독립 표본으로 부풀리지 않는다.
- 보정 모델은 소수 nuisance 변수로 제한하고, 같은 자료에 유연한 모델을 맞춘 성능을 검증 결과로 쓰지 않는다.
- 추세는 같은 sample/probe의 초기·중기·후기 checkpoint에서 본다. 서로 다른 minibatch loss를 짝지어 인과관계로 해석하지 않는다.
- Cell<32 patches 또는 독립 scene group<4이면 `SMALL_CELL` 표시. 이는 새 운영상의 안정성 경고선이며 통계적 보증이 아니다.

FR에는 R_GT가 없다. 따라서 FR 자체의 C×R_GT 사분면을 만들지 않는다. C와 raw HQNR/Dλ/Ds의 관계만 **장면 단위**로 기술한다. RR에서 만든 threshold를 FR 해상도에 그대로 적용한 confidence처럼 사용하지 않는다.

---

# 6. D02 — 같은 backbone에서 correction만 바꾸는 대조 (필수)

## 6.1 개입 정의

Teacher T0 또는 Student의 고정 checkpoint F를 사용한다.

$$Z(c)=M+F([W(P,c),M]).$$

| ID | correction | 목적 |
|---|---|---|
| I0 | c0 | 실제 learned correction |
| IZ | 0 | 같은 sampler를 통한 zero correction |
| IN | −c0 | 부호 반대 대조; 반드시 나빠진다고 가정하지 않음 |
| IB±y | c0±(0.25,0) | native bias에 대한 task 민감도 |
| IB±x | c0±(0,0.25) | 동일 |

0.25 HR px는 **이번 진단의 고정 proposal**이다. 실제 정합 참값·최적 step이 아니다. 결과를 보며 값을 탐색하지 않는다. Primary 결과는 모든 native 출력에서 동일 공식 domain, 추가로 안전한 고정 interior 결과를 함께 기록한다. Safety ROI는 각 변위마다 달라지는 moving mask로 만들지 않는다.

모든 개입은 **원래 PAN에서 같은 sampler를 정확히 한 번 호출**한다. Native correction 개입만으로 student/teacher weight 또는 normalization을 바꾸지 않는다. Clamp·export는 기존 규약 그대로다.

## 6.2 핵심 효과

$$
B_{R,i}=R_i(0)-R_i(c_0),\qquad
B_{Q,i}=Q_i(c_0)-Q_i(0).
$$

양수면 현재 고정 F에서는 learned correction이 유리하다. R와 Q는 각각 RR/FR 자료에서 계산하며 같은 sample metric처럼 결합하지 않는다.

±0.25 개입의 평균·최선·최악 차이를 모두 저장한다. 가장 좋은 bias만 골라 공식 benchmark 출력으로 바꾸지 않는다. GT로 고른 best bias는 oracle diagnostic이지 inference method가 아니다.

## 6.3 반드시 포함할 closure 불변 대조

예측된 두 correction 모두에 같은 b를 더한다.

$$
(c_\epsilon+b)+\epsilon-(c_0+b)=c_\epsilon+\epsilon-c_0.
$$

따라서 forward consistency는 수치오차 안에서 같아야 한다. 그런데 R/Q는 변할 수 있다. 이를 확인하면 **C가 native 품질의 충분한 인증값이 아니라는 반례**가 된다. 반례가 나와도 C에 예측력이 전혀 없다고 결론내리지는 않는다.

이때 b를 넣은 뒤 네트워크 입력까지 다시 바꾸어 별도 cε를 예측한 결과를 위 항등식과 혼동하지 않는다. 위 검사는 **예측 correction에 동일한 additive bias를 붙이는 wrapper 개입**이다.

## 6.4 합성 이동 강건성은 별도

$$
Z^{seq}_\epsilon=M+F([W(W(P,\epsilon),c_\epsilon),M]),
$$
$$
Z^{comp}_\epsilon=M+F([W(P,\epsilon+c_\epsilon),M]).
$$

추정용 Pε는 두 경우 동일하다. Sequential은 두 번 보간, composed는 원 PAN에서 한 번 보간이므로 차이를 따로 기록한다. 연속 translation에서의 합성 관계를 bicubic 두 번의 정확한 수치 항등식으로 가정하지 않는다.

12시간 기본 진단은 DISC128 + RR20에서 수행한다. FR은 I0/IZ/IN을 전체20장에서, ±bias와 synthetic reconstruction은 비용이 크면 결과와 독립적인 seed314159 RNG로 사전 고정한 8장(known scene group이 있으면 층화 추출)에서 수행하고 partial로 표시한다. Subset을 결과에 따라 바꾸지 않는다. 공식 HQNR은 항상 전체20장이다.

---

# 7. D03 — 실제 gradient 충돌·민감도 진단 (필수)

## 7.1 같은 A 파라미터에서 비교한다

$$
g_0=\nabla_\phi L_0,\quad
g_H=\nabla_\phi L_H,\quad
g_E=\nabla_\phi(\lambda_E L_E),\quad
g_K=\nabla_\phi L_K,\quad
g_C=\nabla_\phi L_{off}.
$$

`L0=plain native GT L1`. `gH`와 `g0`는 구분한다. 비교할 실제 task gradient는 `gT=gH+gE`, 실제 offset gradient는 홀수 update에서 `1e-4*gC`다.

필수 로그:

```text
grad_norm: g0, gH, gE, gK, gC_raw, 1e-4*gC, gT
cos(g0,gC), cos(gT,gC), cos(gT,gK)
||1e-4*gC||/(||gT||+eps)
native dL0/dc0, native dLE/dc0
routing별 실제 .grad 와 기대 .grad 의 상대/절대 차이
```

Near-zero gradient에서는 cosine을 0이라고 결론내리지 말고 `NA_ZERO_GRAD`로 표시한다. Threshold는 같은 dtype 반복 수치오차와 raw norm을 함께 기록한다.

## 7.2 부호 해석

$$\cos(g_T,g_C)=\frac{g_T^\top g_C}{\|g_T\|\|g_C\|+\epsilon}.$$

고정된 local loss에서 SGD-like virtual step `δφ=−ηgC`의 1차 근사는:

$$\Delta L_{task}\approx-\eta g_T^\top g_C.$$

음의 내적은 그 local direction에서의 충돌 가능성이다. 실제 Adam update는 momentum·preconditioning을 포함하므로, 내적만으로 다음 step의 실제 metric을 예측했다고 쓰지 않는다. 계수와 norm을 빼고 cosine 부호만으로 위험 정도를 정하지도 않는다.

SG offset의 c0 target을 **해당 virtual step에서 고정한 값**과 step 이후 다시 계산한 실제 closure를 모두 보고한다. 둘은 같은 목적함수 값 변화가 아닐 수 있다.

## 7.3 구현 범위와 비용

- T0 진단이 필요하면 별도 clone에서 A만 requires-grad로 열고, 실제 Teacher asset은 건드리지 않는다.
- GRAD64를 작은 microbatch로 순회한다. 가능하면 sample별 φ gradient, 비용이 크면 같은 cell 내 batch gradient를 쓰되 aggregation을 표시한다.
- 기존 fixed/live gradient diagnostics를 먼저 재사용하고 누락된 task/soft 분해만 보완한다.
- 학습 D03 시점은 저장 가능한 공통 격자의 `5050, 25250, 45450, 50000`으로 고정한다.
- 복제 상태의 1-step 검증은 선택 사항이다. optimizer state와 RNG를 완전히 clone하고 원본 상태를 바꾸지 않는다.
- **Bicubic sampler의 이중 미분은 사용하지 않는다.** 이 진단은 first-order autograd와 고정 correction 개입으로 충분하다.

`g_R`이 거의 0인데 R이 크면 PAN 위치 경로가 현재 task에 둔감할 수 있다. 큰 C/R이나 음의 cosine만으로 데이터 원인을 단정하지 않는다. D02와 함께 읽는다.

---

# 8. B0/B1 — 바로 실행하는 핵심 학습 대조

## 8.1 run 정의

| Case | Aligner | Backbone 수신 loss | Aligner 수신 loss | 변하지 않는 것 |
|---|---|---|---|---|
| B0/FQ | T0 copy frozen | LH+LK+λE LE | 없음 | T0, U init, data, Q12 계수, 50K grid |
| B1/JK0 | T0 copy trainable | LH+LK+λE LE | LH+λE LE+odd·1e-4 Loff | 동일 |

두 case는 Teacher와 Student 간 weight를 공유하지 않는다. B0와 B1이 같은 U 초기값·학습 data order를 쓰게 고정한다. Probe/diagnostic RNG는 별도 generator를 쓴다.

Frozen B0도 고정 probe consistency를 **진단**으로 계산할 수 있다. 진단값을 0으로 채우지 않는다. Aligner는 optimizer에서 제외하고 weight decay도 받지 않는다.

## 8.2 기본 seed·순서

단일 호스트에서:

```text
B0/FQ S1234 → B1/JK0 S1234 → B0/FQ S777 → B1/JK0 S777
```

각 seed block의 기존 exact-matching asset이 있다면 생략한다. 실행 순서에 따른 시스템 상태는 기록하되 동일 seed 반복이 서로 독립 GPU 재현인 것처럼 주장하지 않는다.

두 서버를 동시에 쓴다면:

```text
host A: FQ1234 + JK0 1234 (완전한 대조쌍)
host B: FQ777  + JK0 777  (완전한 대조쌍)
```

이는 빠른 확인 배치지만 seed와 host가 결합돼 있다. 최종 재현 주장은 한 호스트의 두 seed 또는 양쪽에 같은 pair를 복제한 결과로 보강한다. 한 서버에 FQ만, 다른 서버에 JK0만 보내 절대 HQNR을 빼지 않는다.

## 8.3 12시간에서 하지 않는 변경

Backbone·계수·Teacher·training corruption distribution·scheduler horizon을 동시에 바꾸지 않는다. ε 반경을 실험마다 조정하지 않는다. 25K cosine으로 학습한 run을 50K baseline과 비교하지 않는다. R1/variance 전수 sweep을 다시 열지 않는다.

---

# 9. D04 — 사분면이 실제 KD utility를 구분하는지 (조건부)

**D01의 C–R 상관만으로 B2/B3를 실행하지 않는다.** 같은 eT/eS로 정의한 Teacher 우위를 다시 eT/eS로 예측하면 정의상 자명하다. 여기서는 실제 짧은 업데이트 효과를 별도로 측정한다.

## 9.1 짧은 paired micro-update — 신규 진단 제안

저장된 B0의 같은 checkpoint U를 두 복사본으로 만든다. T0/A는 고정한다.

- H-copy: `LH+λE LE`로 미세 업데이트.
- K-copy: `LH+LK+λE LE`로 미세 업데이트.
- 같은 초기 weight, optimizer state, LR, data order를 사용한다.
- 기본 checkpoint: `25250`과 `45450` 두 개. 사전 지정하며 best만 골라 쓰지 않는다.
- DISC에서 support 4 minibatches×batch16, 8 successful updates씩 진행한다. 같은 episode의 query는 support와 분리한다.
- Cell별 target8 episodes를 목표로 하되 group/표본 부족은 그대로 보고한다.
- Query는 동일한 frozen Teacher cell과 유사한 R_T/texture 구간에서 뽑는다.
- CONF에서는 같은 절차를 group-disjoint로 반복한다.

Optimizer state가 저장되지 않았다면 **양쪽 모두 fresh optimizer의 새 진단**으로 표시한다. 기존 Adam trajectory의 정확한 continuation이라고 쓰지 않는다. LR를 sweep하지 않고 해당 checkpoint의 값으로 고정한다.

$$
U_i=R^{query}_{H\text{-copy},i}-R^{query}_{K\text{-copy},i}.
$$

U>0이면 이 짧은 학습에서 soft가 query GT L1에 유리했다. 이는 **full50K나 FR HQNR 이득의 보장**이 아니다. Micro-trained weight를 정식 checkpoint로 업로드하거나 배포하지 않는다.

## 9.2 routing 후보를 여는 조건

다음이 모두 충족될 때만 9.3의 multiplier를 만든다.

1. Teacher C가 수치오차 이상으로 변하고 cell support가 충분하다.
2. R_T·Student 우위·texture가 비슷한 조건에서도 C 구분에 따라 U의 부호/크기 차이가 남는다.
3. DISC에서 soft가 해로운 cell 후보가 나타나며, CONF와 두 checkpoint에서 같은 방향을 보인다.
4. 해당 cell의 CONF group-bootstrap interval이 충분히 일관된 음성 방향을 지지한다. 작고 불안정한 표본에서는 `INCONCLUSIVE`, 자동 gate 없음.
5. 다중 cell을 보고 고른 사실을 기록한다. Test HQNR로 cell/threshold/multiplier를 고르지 않는다.

표본·그룹 정보가 부족하거나 어떤 cell이 해로운지 재현되지 않으면 **m=1 유지**, B2/B3를 `NOT_OPENED_NO_INCREMENTAL_EVIDENCE`로 기록한다. 이때 B0/B1의 결과는 여전히 의미 있다.

## 9.3 최소 사분면 policy

네 개의 자유 계수를 최적화하지 않는다. DISC에서 가장 불리한 cell **한 개 z\***만 후보로 정하고 CONF로 확인한다. 기본 attenuation도 사전에 **0.5**로 고정한다.

$$
m_z=\begin{cases}0.5 & z=z^*\\1 & \text{otherwise.}\end{cases}
$$

hard와 edge는 그대로 두고 soft에만 m을 곱한다. Low/high C의 단순 순서만으로 z*를 정하지 않는다. High C/low R인 cell의 Teacher output도 실제로 유용할 수 있다.

`z_T`는 Teacher의 sample-level label이며 detach한다. Training augmentation을 사용하면 해당 입력의 Teacher R/C를 같은 정의로 계산한다. 미증강 index의 label을 다른 crop/입력에 무조건 재사용하지 않는다. 비용을 줄여 cache하려면 augment identity·probe·Teacher·view를 key로 하고 동치성을 검증한다. 추가16회 A 호출 비용도 전체 budget에 포함한다.

## 9.4 B2/B3와 최소 양 대조

$$L_K^q=0.1\langle m_{z_T(i)}(1-d_T(p))a_T(p)k(p)\rangle.$$

| Case | 기반 | 변경 |
|---|---|---|
| B2 | B0/FQ | U의 soft만 LKq로 변경 |
| B3 | B1/JK0 | U의 soft만 LKq; A 수신식은 B1과 같음 |
| CMASS (후속 필수 대조) | B0 또는 B1의 선택 기반 | 동일 batch 총 soft 계수량이 되도록 단일 scalar만 적용 |

CMASS의 예: 원 soft coefficient v=(1-dT)aT, `mbar=sum(m*v)/sum(v)`를 detach해 모든 위치에 같은 mbar를 곱한다. 분모0이면 해당 batch soft는 이미0이다. **동일 총 coefficient**를 맞춘 것이지 동일 loss/gradient를 맞춘 것은 아니므로 셋을 따로 기록한다.

B2/B3가 좋아져도 CMASS와 구분되지 않으면 “사분면 위치 정보가 유효했다”가 아니라 **soft 총량 감소로 설명 가능**하다고 남긴다. 한정 예산에서는 B2/B3가 비양성일 때 CMASS를 새로 돌리지 않는다.

B2/B3는 우선 seed1234에서 각각 50K로 수행한다. 이 두 run이 개방되면 별도 연장 예산(대략2 run의 실측시간)을 사용한다. 최종 채택은 다른 seed·CMASS까지 확인한 뒤이며, 12시간 안에 완료되지 않았다고 정의나 결과를 축약하지 않는다.

---

# 10. 더 값어치가 큰 진단이 나오면 무엇부터 연장하는가?

| 관측 | 우선 연장 | 바로 하면 안 되는 결론 |
|---|---|---|
| B1−B0가 두 seed 모두 양수 | 같은 호스트 seed2026의 B0/B1 pair | 이미 일반화 성공이라고 확정 |
| B1은 양성, 기존 JQ는 음성 | 같은 block JQ↔JK0 확인 | Teacher output 전체가 나쁨 |
| Offset gradient 충돌이 크고 B1이 손해 | 선택적 **B1-NOOFF** 1개, B1과 직접 비교 | C가 크니 λoff를 무조건 더 키움 |
| C는 synthetic 강건성만 예측 | Native KD gate 종료; robustness 지표로 보존 | Aligner 모듈 자체가 무효 |
| D04의 추가정보가 검증됨 | B2/B3 pair, 양성이면 CMASS | C 작음만 보고 soft 강화 |
| Teacher C가 거의 상수 | B2/B3 종료; B0/B1 효과만 판정 | 가짜 cell 네 개 생성 |

B1-NOOFF는 **새 대조 제안**이다. A에는 LH+λE LE만, U에는 Q12를 유지한다. 학습입력·native task는 동일하고 offset용 forward는 진단으로 유지할 수 있다. I-AEQ/offset0 조합을 기존 resolver가 지원하는지 구현자가 확인한다. 이름만 바꾸고 별도 sampler·전처리를 바꾸지 않는다.

PCGrad/gradient projection, dynamic LR, quadrant별 freeze scheduler는 이번 12시간에 넣지 않는다. Gradient 충돌은 먼저 측정 대상으로 둔다.

---

# 11. 시간 배분과 중단·연장 규칙

## 11.1 기본 예산

| 작업 | 계획용 시간 | 성격 |
|---|---:|---|
| 자산 audit·smoke·처리량 측정 | 0.5–1.0h | 일부 CPU 작업, 로컬 상태에 따라 달라짐 |
| D01–D03 초기 진단 | 1.5–2.5h | panel/probe 크기 고정, 신규 구현 시간은 별도일 수 있음 |
| B0/B1 × 2 seeds | 4×약1.8–2.8h = 7.2–11.2h | **현재 호스트 미측정인 예상 범위** |
| 결과 재평가·집계·판정 | 0.5–1.0h | 학습 중간 진단 포함 여부 기록 |
| **기본 합계** | **약10–16h** | 중심 목표 약12h, 12h hard cap 아님 |

기존 PAKD50 구현 노트의 s1 smoke/시간 추정은 참고만 한다 [S3]. NA104 시간이나 s1 기록을 s4 실제 처리량으로 대체하지 않는다. 새 계측 기능을 구현하는 사람 작업시간과 GPU 점유시간도 따로 적는다.

D04와 B2/B3/CMASS는 선택된 확장이다. 전부 개방되면 기본 예산을 넘는다. 사용자는 12시간 초과를 허용했으므로 **과학적으로 필요한 대응쌍을 완주**하되, 모든 선택 항목을 자동 실행하지 않는다.

## 11.2 처리량으로 재산정

처음 300 update-equivalent smoke(실제 run weight와 분리)에서 warmup을 제외한 step 시간, offset odd/even, 평가1회, checkpoint 저장시간을 잰다.

```text
예상 = 50K×mean(step time) + 50×eval time + 저장/진단 overhead
reserve = 예상의 10–20% (측정 변동에 따라 manifest에 고정)
```

1000/1010 grid를 더 드물게 바꿔 예산을 맞추지 않는다. 지원되지 않는 partial resumption으로 remaining budget을 채우지 않는다.

## 11.3 정상/비정상 중단 구분

- 12시간 도달 또는 중간 HQNR이 약하다는 이유로 같은 paired block 한쪽만 중단하지 않는다.
- NaN/Inf, gradient 경로 위반, Teacher 변동, sampler 부호 오류, 데이터 누출 등은 즉시 해당 run을 중단하고 INVALID로 표시한다.
- Displacement 폭주·boundary support 이탈은 현재 안전 규칙에 따라 경고/중단한다. metric이 나빠졌다는 이유로 유리한 max-shift cap을 사후 추가하지 않는다.
- T0/λE 자산이 없으면 BLOCKED이고 새 Teacher 또는 임의 λE로 메우지 않는다.
- 완료 asset 재사용으로 시간이 남으면 third-seed pair/필수 진단을 먼저 보강한다. 폐기한 variance·A/B/C 전수를 부활시키지 않는다.

---

# 12. 구현·실행 인계

## 12.1 이미 있는 것과 새로 필요한 것

| 구성 | 처리 |
|---|---|
| FQ/JK0/JQ, Q12 loss, routing flags | 현 generator/trainer 재사용 [S2] |
| 기존 fixed/live gradient log | 필드 확인 후 재사용 [S3] |
| 사분면 panel/threshold CSV | 신규 read-only 분석 도구 필요 |
| correction 개입·bias closure 대조 | 기존 warp/forward 재사용 wrapper 필요 |
| sample/cell 단위 gradient 분해 | 기존 진단 확장 필요 |
| D04 micro-update | 선택적 신규 분석 도구 |
| B2/B3/CMASS | Gate 개방 후에만 신규 multiplier 구현·검사 |

**본 문서는 위 신규 도구를 구현·CUDA 검증한 패키지가 아니다.** 기존 FQ/JK0 학습은 이미 있는 정의로 바로 준비할 수 있고, 진단은 별도 worktree에서 추가한다. 계측용 변화가 RNG/학습 경로를 바꾸지 않는지 smoke를 통과시킨다.

## 12.2 확인된 generator CLI를 사용하는 준비 예

아래는 **config 생성만 하는 명령 예**다. `SERVER`는 실제 실행 호스트로 바꾼다. 공통 자산·calibration이 준비돼 있어야 한다.

```bash
export SERVER=s4             # 실제 호스트로 변경
export DCR_DIR=research_runs/DCR12_20260914
mkdir -p "$DCR_DIR/config" "$DCR_DIR/analysis"

python tools/gen_pakd50_configs.py \
  --server "$SERVER" --cases FQ,JK0 --seed 1234 \
  --updates 50000 --eval-epoch 5 --version v1201 \
  --out-dir "$DCR_DIR/config"

python tools/gen_pakd50_configs.py \
  --server "$SERVER" --cases FQ,JK0 --seed 777 \
  --updates 50000 --eval-epoch 5 --version v1201 \
  --out-dir "$DCR_DIR/config"
```

이 CLI 옵션은 현 generator에 존재한다 [S2]. `v1201`은 본 계획에서 제안한 충돌 회피 version이다. 기존에 같은 version/run이 있으면 생성 전에 새로운 숫자 version을 manifest에 고정한다. 다른 run을 덮어쓰지 않는다.

Custom `--out-dir`, 명시적 `--cases/--seed`를 사용해 기존 stage queue 자동 덮어쓰기를 피한다. Generator가 쓰는 원 `campaign_id=PAKD50`는 그대로 남을 수 있으므로, 별도 sidecar/실행 manifest에 `analysis_campaign_id=DCR12_20260914_v1`을 기록한다. registry가 모르는 키를 YAML에 무작정 추가하지 않는다.

생성 후 `routing.qK=0`/frozen·실제 τ/λE·Teacher path·budget ledger를 확인하고, 현 호스트에서 검증된 runner로 **이 네 config만** 실행한다. 이 문서는 trainer 진입점 옵션을 검증하지 않았으므로 새로운 학습 CLI 이름을 만들어 제공하지 않는다.

기존 50시간 campaign gate가 원 큐를 재주입하거나 마감으로 run을 거절할 수 있다. 운영자는 현재 run 종료 경계에서 **DCR12 전용 pending manifest·budget ledger**를 마련한다. 기존 gate의 안전 검사를 무조건 끄거나 이미 동작 중인 코드 checkout을 pull로 변경하지 않는다. 실행 격리 상태를 보고서에 기록한다.

## 12.3 parameter별 gradient의 참조 의사 코드

```python
# 원리 설명용. 실제로는 검증된 kdv.routing 경로를 우선 재사용한다.
# T: frozen, theta: Student U, phi: Student A
L_u = L_h + L_k + lambda_e * L_edge
L_a = L_h + lambda_e * L_edge + lambda_off_this_step * L_off

g_theta = autograd_grad(L_u, theta, retain_graph=True)
g_phi   = autograd_grad(L_a, phi)  # B1에서만; B0는 A optimizer 없음
assign_grads(theta, g_theta)
assign_grads(phi, g_phi)
# 기존 AMP scale/unscale, clipping, overflow의 전체-step 일관성, optimizer/scheduler 수행
```

Teacher target detach만으로 soft→phi가 없어지지는 않는다. 반대로 aligned PAN 전체를 detach하면 GT→phi도 끊긴다. AMP라면 두 parameter group의 동일한 skip/update 정책을 유지하고, 한쪽만 성공한 step을 정상 paired update로 세지 않는다.

---

# 13. 필수 산출물과 보고서 형식

```text
analysis/DCR12_20260914/
  manifest.json
  source_identity.json
  panel_ids.json
  probe_manifest.json
  quadrant_thresholds.json
  sample_metrics.csv
  correction_interventions.csv
  gradient_conflicts.csv
  checkpoint_metrics.csv
  quadrant_summary.csv
  micro_update_utility.csv             # D04 수행 시
  routing_policy.json                 # Gate 통과 시; 없으면 만들지 않음
  paired_results.csv
  run_status.json
  report.md
```

## 13.1 CSV 핵심 필드

- Sample: host/run/case/seed/step/split/group/sample/probe_id, pipeline=T/S, C·axis_C·R_plain·edge_plain·cell_ref, c_y/c_x, texture·saturation.
- Interventions: correction mode·b·sampler count·domain, R 또는 HQNR/Dλ/Ds, Δvs learned/zero, support flags.
- Gradient: checkpoint/sample_group/loss_group, raw norm·weighted norm·cos·nearzero·routecheck·virtual-target-policy.
- Result: full precision selected HQNR·selected step·numerical max·plateau·last, same-checkpoint Dλ/Ds, paired Δ, time·energy/peakmem가 있으면 부가 기록.

## 13.2 보고서 첫 페이지에서 답할 것

1. 실제 네 cell의 개수·특성과 C의 변동성이 충분했는가?
2. C가 native R/HQNR과 관련됐는가, 아니면 probe 강건성에만 관련됐는가?
3. 같은 F의 learned correction이 zero보다 유리했는가? C가 그 이득도 설명했는가?
4. Task/offset gradient 충돌이 언제·어떤 cell에서 나타났는가?
5. B1−B0의 HQNR·plain GT L1·C 변화는 각 seed에서 무엇인가?
6. D04/B2/B3를 열었는가? 미개방이면 데이터 부족/추가정보 없음/시간/미구현 중 사유는 무엇인가?
7. 실제 실행·재사용·실패·미실행을 구분한 case 표.

---

# 14. 판정 규칙

## 14.1 지표의 지위

| 판정 | 요구되는 증거 |
|---|---|
| **ROBUSTNESS_ONLY** | C가 합성 이동 반응/출력 안정성은 설명하지만 native 품질·KD utility의 추가정보는 없음 |
| **ALIGNER_USEFUL_METRIC_UNPROVEN** | D02 또는 B1에서 A의 이득은 보이지만 C 기반 선별은 지지되지 않음 |
| **TASK_ADAPTATION_CANDIDATE** | 동일 조건 B1−B0의 HQNR이 두 seed에서 같은 양의 방향; 제3 seed 확인 필요 |
| **QUADRANT_ROUTING_CANDIDATE** | DISC/CONF 추가정보와 실제 B2/B3 개선이 있으며 단순 mass 대조와 구분 필요/확인 |
| **NO_SUPPORT_AT_TESTED_SETTING** | 유효한 진단·대조에서 해당 주장이 지지되지 않음 |
| **INCONCLUSIVE** | 표본·asset·반복·수치 안정성 부족 또는 seed별 부호 반전 |

상관이 없다는 것과 데이터가 없어 계산하지 못했다는 것을 구분한다. 두 seed 양성은 확인 후보이며 학계 일반성·통계적 우수성 확정이 아니다.

## 14.2 주 비교

- `B1−B0`: GT 기반 A adaptation 묶음.
- `B2−B0`: frozen A에서 사분면 soft 변경.
- `B3−B1`: trainable A에서 같은 사분면 soft 변경.
- `(B3−B1)−(B2−B0)`: 결합 상호작용의 기술적 차이. 적은 seed로 유의성을 확정하지 않음.
- `JQ−JK0`: soft→A 추가 경로.
- `B1−B1-NOOFF`: offset 유지의 조건부 기여.

선택·보고에 사용한 FR20을 미사용 test로 표현하지 않는다. HQNR을 주 기준으로 유지하되, 해당 기준의 상승을 물리적 registration GT의 회복 또는 train fitting 원인의 입증과 동일시하지 않는다.

과거 다른 골격의 seed σ를 합격선으로 복사하지 않는다. 현재 paired 차이의 seed 변동과 scene 조건을 따로 보고하며, scene bootstrap CI만으로 학습 seed 일반성을 주장하지 않는다.

---

# 15. Gspread 코멘트 — 실험 실행자용 메모만

기존 gspread uploader와 server sheet 포맷을 그대로 사용한다. 별도 시트 확인·생성·편집은 본 작업에서 하지 않는다. 향후 업로드 시 설명 또는 지원되는 구분 필드에 다음을 남기는 정도로 충분하다.

```text
analysis_campaign_id=DCR12_20260914_v1
validation_case=B0|B1|B2|B3|Dxx
existing_case=FQ|JK0|JQ|new
protocol=FRESH50; metric=raw_original_HQNR; grid=GRID1010_50K_v1
status=measured|diagnostic_only|not_opened|invalid
```

진단용 bias oracle·micro-update 결과를 정식 benchmark run 행으로 섞지 않는다. 기존 결과·run ID를 덮어쓰지 않는다.

---

# 16. 근거와 신규 제안의 경계

**[S0] 최신 사용자/대화 합의.** Consistency와 reconstruction의 네 상태, KD/aligner 역할 분리, 약12시간 이상의 검증·판정 실험 요청. 본 문서의 D00–D04, panel/probe 선택, B2/B3 routing 개방 규칙, budget 배치는 이번에 구체화한 제안이다.

**[S1] 현재 통합 계획.** `research_log/PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md`, §§2–6. W112D123/L1E4·T0·native task·offset·gradient routing·calibration 계약. 조회 파일 blob `4f8998e3302f2868b4ad9fd4b8e69742c359cdf4`.

**[S2] 현재 generator.** `tools/gen_pakd50_configs.py`, POLICY/CASES/CLI. FQ·JK0·JQ의 실제 정의, qK routing·학습계수·grid·seed/version/out-dir 지원. 조회 blob `3003033c66751404e70789c735087f025bafdbf1`. 파일 첫머리 C0 설명에 일부 stale 주석이 있지만, 실제 POLICY/CASES에는 JK0/JE0/P 등이 정의돼 있음을 구분한다.

**[S3] 현재 구현 노트.** `research_log/2026-09-14_pakd50-implementation.md`, §§1–6. Plan GRID1K→실행 GRID1010 변경, T0 layout·τ·λE·격번 offset과 실행 원칙. 이 노트의 초기 “C1 미구현”은 작성 당시 상태이며, 현재 S2의 routing 정의와 local trainer smoke로 다시 확인한다.

[S1–S3 저장소](https://github.com/hojunking/PAN-Crafter-repro). 조회 검색 결과에 표시된 commit: `4054135aa4a00443edb893433563c1f738c0c785`. 실제 실행 release는 작업자가 별도 full commit/diff hash로 고정한다.

**[S4] Offset 의미의 기존 명세.** `PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md`, §§4.3–4.7. 상대 변위 감독, native target SG, common-bias 반례. **옛 W96·λoff=.01·radius1·입력 교대 설정을 현재 실험의 값으로 승계하지 않는다.**

**[S5] Warp/edge의 기존 명세.** `PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2 (1).md`, §§4–5.2. Sampling 부호·zero shortcut 금지·FP32·signed Scharr·고정 domain. 현재 backbone과 hyperparameter의 근거로 쓰지 않는다.

**[S6] s2 분석본.** `2026-09-14_na104-20h-s2-results.md`, §A.4. Q12의 no-align 3-seed 방향과 soft 제거 대조. Report의 다른 인과·통계 해석을 본 계획의 입증 사실로 일괄 계승하지 않는다.

**[S7] s3 분석본.** `2026-09-14_s3_na104-campaign-analysis.md`, §§7–8. Q12/R1의 서버·seed 조건부 결과. Consistency–reconstruction의 실측 근거는 아니다.

---

# 17. 실행 지시문 — 운영자에게 그대로 전달

> 현재 PAKD50 자산과 release를 기준으로 DCR12 검증을 진행하라. 처음에는 기존 Q12의 수식·계수·Teacher를 바꾸지 않는다. D00 identity와 D01–D03 read-only 진단을 만들고, 같은 호스트에서 FQ와 JK0를 seed1234·777로 각각 50K 대응 비교하라. 이미 동일한 완료 run은 재사용한다. 주 지표는 raw-original HQNR이며 GRID1010과 같은 checkpoint의 Dλ/Ds·plain RR L1·offset을 기록하라. Student A의 soft gradient 차단은 기존 JK0 routing을 재사용해 검증하고, frozen Teacher/A를 공유해 target을 움직이지 마라. Consistency가 eT/aT 대비 추가 KD utility를 준다는 별도 확인 없이 사분면 multiplier를 학습에 넣지 마라. 근거가 있을 때만 B2/B3를 한 policy로 열고, 단순 soft 감소 효과와 구분하라. 약12시간을 목표로 하되 대응쌍과 평가 완주를 위해 초과할 수 있다. 기존 mainline 큐·결과를 덮어쓰지 말고 실행·재사용·미개방·무효를 분리한 report와 CSV를 남겨라.
