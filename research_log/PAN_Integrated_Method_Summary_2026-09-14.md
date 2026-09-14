# PAN 통합 방법 요약 — L1E4 Aligner + Q12/R1 Student Fitting

**작성일:** 2026-09-14  
**문서 상태:** 사용자와 합의한 통합 방향 및 다음 실험의 실행 기준. 통합 성능은 아직 미검증.  
**동반 문서:** [50시간 상세 실험계획](PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md)  
**주 목표:** WV3 FR20 **원본 PAN 기준 raw HQNR 0.959 이상, 가능하면 0.960 이상**. HQNR(V64)가 아니다.

---

## 1. 현재 합의한 방향

> **Teacher를 학습할 때 얻은 PAN aligner를 Student에 전달하고, Teacher의 실패 위치·유효한 복원값·GT 경계 감독을 이용해 동일 용량 Student의 fitting과 최종 복원을 개선한다.**

앞단과 뒷단의 파티셔닝은 각 기술의 효용을 확인하기 위한 실험 장치였다. 이번 통합에서는 그 경계를 고정하지 않는다. Student aligner를 고정하는 방법과, 복원·KD 감독으로 함께 조정하는 방법을 모두 시험한다.

첫 통합 backbone은 **W112–D123, depth=[1,2,3]**으로 한다. 앞단의 실제 검증 자산을 그대로 사용하기 위한 결정이다. 뒷단 독립 실험의 W104–D122나 이전 구상의 W112–D122로 조용히 바꾸지 않는다. [A, §1.3; K, §1.2]

현재 뒷단은 **Q12와 R1**이다. Q12는 GT signed-edge를 사용하지만 GT variance/covariance를 사용하지 않는다. 이번 방법을 variance distillation으로 기술하지 않는다. [K, §1.1, §2.4]

## 2. 두 기술이 확보한 근거와 남은 질문

| 축 | 확보한 근거 | 이번 통합에서 확인할 것 |
|---|---|---|
| L1E4 PAN aligner | 보고서의 3-seed 평균 raw HQNR 0.95597. Native 공동 학습에 작은 relative-offset consistency를 결합한 방법을 채택 | 완성 Teacher의 aligner를 Student에 재사용할 때 고정/적응 중 무엇이 유리한가? |
| Q12 | no-align s2에서 N0 대비 평균 +0.00245, 3/3 양성. X02 대비 +0.00266, 3/3 양성 | 정합을 결합한 뒤에도 output soft와 GT edge가 추가 이득을 만드는가? |
| R1 | no-align s3에서 N0 대비 평균 +0.00297, 2/2 양성 | 직접 모방 없이 Teacher 실패 지도만 활용하는 편이 더 유리한 조건이 있는가? |

Q12는 s3에서 평균 −0.00034, R1은 s2에서 평균 −0.00078이었다. 과거 두 서버의 Teacher hash와 실행 revision도 달랐다. 따라서 서버별 우위를 하드웨어 효과로 단정하거나, 단독 개선량을 L1E4에 그대로 더해 성능을 예측하지 않는다. [A, §8.1; K, §8.2–8.6]

L1E4의 raw fSCC 하락, 일부 RR 손실, 절대 native 정합 GT 부재와 blur 대조 미성립도 계속 기록한다. 정합 반응이 좋다는 것과 최종 복원이 좋다는 것은 별개의 질문이다. [A, §8.6, §9]

## 3. 전체 학습 파이프라인

### Stage 1 — 이미 확보한 Teacher–aligner package

기존 N2 사전학습 → L1E4 native 공동 학습으로 얻은 **같은 checkpoint의 Aligner와 U-Net**을 한 쌍으로 지정한다.

```text
Native PAN P ── Aligner A_T ── c_T ── PAN sampling ──┐
MS base M ──────────────────────────────────────────┼─ Teacher U-Net ─ residual + M ─ Z_T
                                                  ┘
```

Teacher 학습의 기준은 native GT L1과 격번 offset consistency다. `λoff=1e-4`, aligner LR `1e-5`, U-Net LR `1e-4`, 50K update, batch48을 계승한다. Jittered PAN은 aligner 보조 경로에만 들어간다. [A, §4]

첫 Teacher 후보는 다음 자산이다.

`PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1 / best_raw`

이는 보고서에서 높은 단일 결과를 보인 **조건부 개발 자산**이지, 전체 방법의 평균 성능을 대표하는 seed가 아니다. 실제 선택 update·파일·A/U hash를 확인한 뒤 사용한다. 원래 N2 donor나 별도 last aligner를 혼합하지 않는다. [A, §5.3]

### Stage 2 — Student 학습

Teacher의 A/U는 모두 frozen/eval/no-grad로 유지한다. Student는 기본적으로 **Teacher aligner를 복사하고 U-Net은 독립 초기화**한다.

\[
Z_T=M+F_T^*([\mathcal W(P,A_T^*(P,M)),M]),
\]
\[
Z_S=M+F_S([\mathcal W(P,A_S(P,M)),M]).
\]

Student aligner를 학습할 때는 Teacher와 다른 객체를 사용한다. Teacher는 자신의 frozen aligner로 입력을 만들며, live Student aligned PAN을 Teacher에 전달하지 않는다.

MS base·GT·최종 출력의 목표 좌표는 M-frame으로 유지한다. PAN만 한 번 보정하고 MS base를 한 번 더한다. 추론에는 Student aligner와 Student U-Net만 남는다. [A, §2, §11.3–11.4]

## 4. 뒷단 목적함수

밴드 평균 L1로 Teacher–GT 오차 `e_T`, Student–GT 오차 `e_S`, Student–Teacher 오차 `k`를 계산한다.

\[
d_T=\operatorname{sg}\!\left(\frac{e_T}{e_T+\tau_R}\right),\qquad
 a_T=\operatorname{sg}\!\left(\operatorname{clip}_{[0,1]}
 \frac{[e_S-e_T]_+}{e_S+10^{-6}}\right).
\]

\[
L_0=\langle e_S\rangle,\quad
L_D=\langle\alpha d_T e_S\rangle,\quad
L_K=\langle\beta(1-d_T)a_T k\rangle.
\]

| Backend | 목적함수 | Teacher의 역할 |
|---|---|---|
| N0 | `L0` | 학습에서 사용하지 않는 대응 대조 |
| R1 | `L0 + LD` | GT 실패 위치 재가중만 사용 |
| Q12 | `L0 + LD + LK + λE LE` | 실패 지도 + adaptive output soft + GT signed-edge |
| X02 | `L0 + LD + λE LE` | Q12의 soft 제거 대조 |

기본 `α=1`, `β=0.1`이다. `LE`는 최종 8-band HRMS와 GT의 **signed Scharr /32, reflect1, 내부1px 비교**다. Teacher edge나 PAN edge가 target이 아니다. Gate는 detach하지만 실제 Student 오차는 live다. Soft-active pixel 수로 재정규화하지 않는다. [K, §4–6]

`τR`는 새 Teacher package로 train-only 재산출한다. `λE`는 J0 seed1234 exact50K pilot에서 plain GT L1과 edge의 **출력 gradient RMS 비율 0.05**로 보정한다. `λE=0.05`라는 뜻이 아니다. 초기 정책 비교에서는 공통 calibration package를 사용한다. [K, §7; 통합 결정]

## 5. 통합의 핵심 실험축

| 정책 | Student aligner가 받는 감독 | Student U-Net이 받는 감독 | 기본 case |
|---|---|---|---|
| **J: Joint** | 선택 backend 전체 + offset | 선택 backend 전체 | J0 / **JQ** / JR |
| **F: Frozen** | 가중치 고정 | 선택 backend 전체 | F0 / FQ / FR |
| **P: Protected** | plain GT L1 + offset만 | Q12 전체 | PQ, 필요 시 PR |
| **D: Delayed** | 첫5K 고정, 이후 J 정책 | 처음부터 선택 backend | D0 / DQ, 필요 시 DR |

**JQ가 주력**, FQ는 고정 정합 재사용 대조다. PQ는 추가 hard·soft·edge의 aligner 전달이 불리한지 확인하고, DQ는 초기 Student gradient로 인한 정합 상태 변화를 줄이는 후보다.

필요할 때만 soft→aligner 차단, edge→aligner 차단, aligner LR 축소, 후반 freeze, Q12 계수의 작은 단일축 조정을 추가한다. `λoff`·jitter 반경·backbone을 다시 넓게 탐색하지 않는다.

Teacher U-Net까지 복사하는 **TCOPY 10K 추가 fitting**과, 완주 Student에서의 **CONT 10K 추가 fitting**도 조건부 후보로 둔다. 각각 같은 parent·초기값·LR·추가 update의 GT-only 대조가 필요하다. 복사 초기값의 반복은 독립 U-Net 초기화 반복이 아니라 데이터/augmentation seed 반복이다.

## 6. Seed와 ERGAS를 해석하는 원칙

사용자 관측처럼 seed 변동을 결과 해석에 반영한다. 단, 어떤 악화든 seed 때문이라고 면제하지는 않는다.

L1E4의 Sheet 표시값은 다음과 같다. 모두 각 run의 raw-HQNR 선택 checkpoint이며 exact50K 표가 아니다. [A, §8.3]

| Teacher 학습 seed | Raw HQNR | ERGAS@raw-selected |
|---:|---:|---:|
| 1234 | 0.9555 | 2.0373 |
| 7777 | 0.9554 | 2.1228 |
| 2025 | 0.9570 | 2.0694 |

이 세 ERGAS의 표본 sd는 약 0.0432, 범위는 0.0855다. 이는 표시값에서 계산한 참고량이며 **순수 seed 분산의 추정치나 새 실험의 허용오차가 아니다**. 선택 checkpoint 차이와 평가 규약 문제를 먼저 분리한다.

새 Student seed는 **1234 / 777 / 2026**을 기본으로 둔다. 같은 seed의 후보와 N0는 초기 U-Net state·데이터 순서·augmentation·epsilon RNG를 대응시킨다. `777`과 과거 앞단의 `7777`을 혼동하지 않는다.

판정은 seed별 paired ΔHQNR·ΔERGAS, 평균·표본 sd·중앙값·최소/최대·양성 seed 수를 함께 사용한다. ERGAS는 `raw-selected`, `exact50K`, 사전 정의한 후반 plateau를 따로 본다. 단일 seed의 ERGAS 악화만으로 좋은 후보를 즉시 버리지 않지만, 여러 seed에서 반복되는 악화도 숨기지 않는다.

세 서버에 seed를 하나씩 배치한 결과는 우선 **server×seed 대응 block**이다. 최종 후보는 같은 seed의 N0/후보 한 쌍을 다른 서버에서도 반복해 환경 민감도를 분리한다. 20장 scene이나 수백만 pixel을 학습 seed 반복 수로 세지 않는다.

## 7. 50시간 운영과 목표 판정

**시간 해석:** 시작부터 종료까지 50시간의 경과시간 동안 s1/s2/s3를 병렬 활용한다. 서버당 1 GPU slot이면 gross 상한은 150 GPU-hour다. 구현·검사·평가·장애 시간을 제외한 순학습 예산은 더 작다. 실제 GPU 수와 처리속도는 시작 시 측정한다.

s1은 코드 구현·검사·release의 기준이다. s2/s3는 Git push/pull로 같은 release를 받고, 학습은 **고정 commit의 별도 worktree**에서 실행한다. 실행 중인 소스 checkout을 pull로 바꾸지 않는다. 가중치 전달은 코드 Git 동기화와 구분하고, 세 서버에서 같은 asset hash를 검증한다.

우선순위는 다음과 같다.

1. 공통 package 검증 → **J0/JQ 3개 대응 block**.
2. F0/FQ 및 R1 비교 → 유력 결합 정책 선택.
3. PQ·DQ·필요한 단일축 조정·10K fitting으로 목표 수치 접근.
4. 새 case 수를 늘리기보다 유력 후보의 3-block 확인, X02, 재평가를 완료.

모든 후보를 무조건 실행하는 전수 grid는 아니다. 상세 문서의 처리시간 산식과 남은 시간을 사용해 case를 승인하며, 마지막4시간은 결과 재현·export·집계에 남긴다.

**목표의 층위도 분리한다.** 한 run이 0.959를 넘으면 단일 자산 목표 달성이다. 여러 seed의 평균·중앙값과 대응 이득까지 유지되어야 반복 가능한 방법 성능으로 보고한다. 0.95896을 소수점 반올림해 95.9로 표시한 것은 0.959 달성이 아니다.

---

## 8. 근거 문서와 지위

본문의 `[A]`, `[K]`는 아래 첨부 명세를 가리킨다. 수치·기존 구현은 해당 절을 근거로 하며, J/F/P/D 조합·50시간 운영·새 case와 숫자 설정은 이번 **실험 설계**다.

- **[A]** `PAN_Aligner_L1E4_TechnicalSpec_Evidence_KD_Handoff_2026-09-14.md`
- **[K]** `PAN_KD_Q12_R1_Spec_and_Aligner_Interface_2026-09-14.md`

현재 서버의 checkpoint·처리속도·통합 CUDA 실행 결과는 이 문서 작성으로 검증된 것이 아니다. 상세 계획은 미확인 자산을 확정한 뒤 실행하는 기준이며, HQNR 0.959–0.960 달성을 보장하는 결과 보고서가 아니다.
