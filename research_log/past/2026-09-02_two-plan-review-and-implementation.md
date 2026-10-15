# s1 teacher 탐색 · s2 uncertainty KD — 계획 검토와 구현 보고 (2026-09-02)

대상 계획 두 건:
[s1 4-6M teacher 탐색](2026-09-01_s1-teacher-architecture-4-6m-plan.md) ·
[s2 uncertainty distillation + GT-variance](2026-09-01_s2-uncertainty-distillation-gtvar-plan.md).
**상태: 구현·검증 완료, 캠페인 미기동.**

---

# Part 1. 검토

## 1.1 가장 중요한 문제 — 주가설의 축이 판정 지표로 보이지 않는다

s1 계획 §1 의 근거표는 width 를 키우면 좋아진다는 것을 **ERGAS 로만** 읽었다.
같은 실행들을 확정 판정 지표인 HQNR 로 다시 읽으면 방향이 반대다.

| depth [1,2,4] · dual | HQNR↑ | (참고) ERGAS↓ |
|---|---:|---:|
| W96 `R4_w96_d124_noattn` | **0.9561** | 2.1418 |
| W112 `R3_w112_d124_noattn` | 0.9540 | 2.1166 |
| W128 `c6_c4d124` | 0.9536 | 2.0826 |

width 가 커질수록 HQNR 은 **낮아진다**. 폭 0.0025 로 동급 band(0.011) 안이라
"HQNR 로는 구분되지 않는다"가 정확한 표현이지만, 적어도 **계획이 기대하는 방향의
근거는 판정 지표에 없다.**

MS-only 계열에서는 반대로 보인다(`MS1_R4` 0.9512 → `MS1_w128` 0.9539). 그러나
`tools/plateau_report.py` 로 선택 편향을 걷어내면 **부호가 뒤집힌다**:

```
MS1_w128 vs MS1_R4 :  best Δ+0.0027   공통epoch Δ−0.0010   ← 부호 반대
```

즉 **width 축은 HQNR 로 판별되지 않으며, best-checkpoint 비교는 그 부호마저
뒤집는다.** 이 캠페인이 HQNR 로 winner 를 고르려 하면 사실상 잡음을 고르게 된다.

**대응(권고).**
1. §8 의 "최소 조건"은 그대로 두되, **전 case 가 band 안이면 그것이 결론**임을
   미리 못 박는다 — 계획 §8 마지막 줄이 이미 그렇게 쓰여 있다(“capacity 가 커져도
   KD 에 전달할 headroom 부족으로 기록”). 이 문장이 이번 캠페인의 **주된 예상 산출**이다.
2. 모든 비교에 **plateau(공통 eval epoch) HQNR 을 병기**한다 — 저장소 확정 규칙이고
   위 부호 반전이 그 이유다. 도구를 만들어 뒀다(§2.4).

## 1.2 판정 규칙 충돌 — ERGAS 로 winner 를 고르게 되어 있다

s1 §8-2 "동급이면 SCC → ERGAS → SAM/Q2n" 과 "teacher 최소 조건: HQNR 동급이면
ERGAS 최소 1% 개선" 은 확정 지시(판정은 HQNR, 보조 SCC, ERGAS 로 내려가 tie-break
하지 않는다)와 충돌한다. §1.1 을 감안하면 이 사다리는 **실제로 ERGAS 가 winner 를
고르는 결과**가 된다.

**대응(권고).** 두 질문을 분리한다.
- **순위 판정**(어느 구조가 더 좋은가): HQNR → SCC 에서 끝낸다. 동급이면 "구분되지 않는다".
- **KD headroom 존재 확인**(teacher 로 쓸 만한가): 이것은 순위가 아니라 별도 진단이다.
  ERGAS 를 쓰더라도 **"참고 지표 기반 headroom 진단"** 이라고 명시하고, 결론 문장의
  근거로 앞세우지 않는다. HQNR 해석이 필요하면 D_λ·D_s 분해로 설명한다.

## 1.3 control 재사용 조건이 s1 에서 성립하지 않는다

계획 §5 는 "clean MS2 control 이 이미 있으면 재사용" 이라 했지만, s1 에 있는 것은
**MS1**(`mars: ms` 이되 mode modulation 유지)이고 계획이 요구하는 **MS2 plain**
(`mode_modulation: false`)이 아니다. MS2 는 s2 에만 있다.

- 따라서 `S1_C0`·`S1_A0` 두 control 은 **새로 돌려야 한다** → 시간은 §7 의
  "controls 신규" 대역(20–23h)으로 확정된다.
- 더 위험한 것은 **혼동 가능성**이다: MS1 과 MS2 의 params 차이는 0.18% 뿐이라
  기존 `expect_params_m` 검사(허용 0.5%)로는 절대 걸리지 않는다. 실제로 MS1 을
  MS2 로 오인해 anchor 로 쓰면 width scaling 결론 전체가 오염된다.
  → **구조 검사를 코드로 넣었다**(§2.4).

## 1.4 조건부 case 를 열어줄 장치가 없었다

§4.2 중간점, §4.3 NAF fallback, §6 winner dual 은 전부 "결과를 보고 정한다" 인데,
러너의 유일한 조건부 장치인 게이트는 직전 KD 캠페인 사다리였다. 그대로 기동하면
본 큐 6건만 돌고 끝난다. 또 §4.2 의 "서로 다른 지표 방향" 은 그대로는 코드가 될 수
없다. → **게이트를 재작성하고 조작적 정의를 넣었다**(§2.3).

## 1.5 그 밖에 확인한 것

- **params·시간 추정은 정확하다.** 계획의 params 는 실측과 전부 0.13% 이내이고
  `T03`(W176-d122) = **5.9882M** 으로 6M 미만이라 §4.1 의 대체(W168-d122)는 불필요하다.
  시간 추정(T00 2.2–2.5h 등)은 실측(MS-only W96 1.25h·W128 1.48h)에서 보면 오히려
  **보수적**이다 — 고정 오버헤드가 커서 width 제곱만큼 늘지 않는다.
- **§2 의 공통 설정 YAML 은 현재 config 스키마가 아니다**(`task:`, `pan_reconstruction:`
  등은 파서에 없는 키다). 의미를 실제 키로 옮겨 config 를 생성했다:
  `mars: ms` + `model_args.mode_modulation: false` + `attn_locations: []`.
- **matched-budget 짝의 예산 편향**: T00(4.95M) vs T01(4.76M), T03(5.99M) vs T02(5.88M)
  모두 가설이 유리한 wider-shallow 쪽에 파라미터가 2~4% 더 간다. 결과 해석 시 명시할 것.
- **단일 시드**다. band 안 차이로는 어떤 case 간 결론도 주장할 수 없다(확정 규칙).
- s2 계획은 **seed 2회**를 처음부터 요구해 이 문제를 스스로 피한다 — 설계상 더 안전하다.

## 1.6 s2 계획 검토

s1 보다 짜임새가 좋다. seed 2회, plain KD 대조군, 사전 등록된 진단(§9.2·§9.3),
"한 번에 하나만 바꾼다"(§10)가 전부 들어 있다. 실질 지적은 둘이다.

1. **head-only calibration 의 실패 모드.** 계획 §2 의 게이트는 "전역 분산 대비 NLL
   개선"인데, head 를 랜덤 초기화에서 출발시키면 짧은 학습으로는 이 게이트를 통과
   못 한다(실측: 150 iter 에서 Spearman 0.72·단조 통과인데 NLL 은 −0.07 악화 → FAIL).
   → **warm start** 를 넣어 구조적으로 제거했다(§2.2).
2. **λ_U* 선택이 사람 손을 탄다.** §5.3 은 "두 seed 평균으로 λ 를 먼저 고정" 이라
   하는데 무인 큐에서는 그 판단 주체가 없다. → 게이트가 seed 쌍 평균 HQNR→SCC 로
   자동 선택하고, GTVar config 를 λ별로 미리 만들어 두는 방식으로 해결했다(§2.3).

---

# Part 2. 구현

## 2.1 s1 — config 만으로 성립

`mars: ms`(PAN task·배치 복제 제거)와 `model_args.mode_modulation: false`(γβ 제거)가
이미 있어 코드 변경이 필요 없었다. LayerNorm 이라 width 32배수 제약도 없다.

| config | 구성 | 실측 params |
|---|---|---:|
| `S1_C0_W96_D124_MS2` | student 동일 control | 2.1247M |
| `S1_A0_W128_D124_MS2` | c6 골격 anchor | 3.7668M |
| `S1_T00_W160_D122_MS2` | 첫 teacher 후보 | 4.9523M |
| `S1_T01_W144_D124_MS2` | 5M 급 depth 대조 | 4.7630M |
| `S1_T02_W160_D124_MS2` | W160 bottleneck +2 | 5.8758M |
| `S1_T03_W176_D122_MS2` | 6M 한계 width | 5.9882M |
| `S1_T04A_W152_D123_MS2` / `S1_T04B_W168_D123_MS2` | 조건부 중간점 | 4.8880 / 5.9670M |
| `S1_T05_{W160_D122,W144_D124,W160_D124,W176_D122}_DUAL` | winner dual 대조 (게이트가 1벌만) | 4.96~5.99M |

## 2.2 s2 — 신규 구현

| 파일 | 내용 |
|---|---|
| `kd/ops.py` | `local_variance(k)` · `multiscale_variance`(V₃·0.5+V₅·0.3+V₉·0.2) · `squash_variance` |
| `kd/losses.py` | `local_error_map`(A₃ 국소 MSE) · `logvar_nll`(½e^{-s}e+½s) · `uknow_weights_fixed`(고정 분위수·soft 바닥) · `gtvar_loss`(SmoothL1 + V 상관 진단) |
| `kd/features.py` | `UncertaintyHead(out="softplus"\|"logvar")` — **state_dict 키가 동일**해 기존 T1/T2 checkpoint 가 그대로 로드된다. `load_teacher` 는 `uq_norm.json` 의 `head_out` 을 읽어 의미를 틀리지 않는다 |
| `train_kd.py` | **MS-only student 지원**(rep=1·switch=1·PAN 항 소거) · 신규 variant `u_full`/`u_full_gtvar` · teacher 디렉터리의 고정 상수 자동 로드 |
| `tools/calibrate_head.py` | mean 고정 head-only 보정 + q05/q95·κ 산출 + 계획 §2 게이트 3종 검사 |

**warm start (핵심 설계 결정).** head 의 마지막 conv 를 (weight 0, bias log E[e_loc])
로 두어 **전역 상수 분산 해에서 출발**시킨다. 게이트가 "전역 상수보다 나은가" 이므로
이 초기화 뒤의 학습은 원리적으로 개선만 한다. 실측으로 확인:

```
warm start 없음, 150 iter : NLL −2.862 vs 전역 −2.932  →  FAIL
warm start 있음, 150 iter : NLL −3.181 vs 전역 −2.900  →  PASS (+0.281)
```

s2 config 15벌: teacher 1 · plain KD 2(seed) · UKD λ{0.03,0.10,0.30}×2 = 6 ·
GTVar λ별 6(게이트가 승자 2벌만 연다).

## 2.3 게이트 (`tools/campaign_gate.py` 재작성)

두 서버가 같은 파일을 쓰고, 전제 실행이 없는 쪽은 자연히 닫힌다.

- **s1 중간점** — "서로 다른 지표 방향"의 조작적 정의: HQNR 동급(|Δ|≤0.011)이면서
  SCC 와 ERGAS 가 **서로 반대 모델을 가리킬 때**. 두 구간 중 더 모호한(ΔHQNR 작은)
  쪽 **하나만** 연다.
- **s1 winner dual** — 탐색이 전부 끝난 뒤 HQNR→SCC 로 winner 를 뽑아 해당 dual config 하나.
- **s1 NAF 조건** — T00·T01 이 anchor 를 모두 개선 못 하면 로그로 기록(구현본이 없으면 열지 않음).
- **s2** — teacher 완료 → `calibrate_head.py` 자동 실행 → PASS 면 student 8벌 개방 →
  UKD 6벌 완료 후 seed 쌍 평균으로 λ_U\* 선택 → GTVar 2벌.

## 2.4 검증 장치 보강 (검토 §1.1·§1.3 대응)

- `tools/smoke_cases.py`: **mode_modulation 구조 검사**(params 로는 못 잡는 MS1/MS2
  혼동을 차단) + **work_dir 이름 일치 검사**(복사 config 사고 방지). smoke 출력에
  `mars=ms+plain` 처럼 표시된다.
- `tools/plateau_report.py`: best·plateau·**공통 eval epoch** HQNR 을 한 표로 내고
  best 와 plateau 의 **부호 반전**을 표시한다. 결과 보고 시 반드시 병기한다.
- `gspread`: `plain` 서술자와 `mode_modulation=False` Notes — 시트에서 MS1 행과
  MS2 행이 같은 설명으로 남던 문제 해소.

## 2.5 검증 결과

```
unit test            14/14 (신규 4: logvar 최적해 회복 · local_error_map ·
                            고정분위수 weight 배치 불변 · GTVar 텍스처 구분/grad)
config smoke         s1 9벌 + s2 teacher = 10/10
통합 smoke(실학습)   calibration(PASS) → u_full → u_full_gtvar 전 구간 관통
                     MS-only 확인: Loss PAN 0.000 · w_hard 1.00±0.12 · w_soft 1.00±0.41
                     GTVar: V_S 0.372±0.257 vs V_GT 0.424 (r 0.941)
게이트 dry-run       정상 (전제 미완 → 닫힘)
```

## 2.6 기동 절차

```bash
# s1
./tools/campaign_start.sh --queue config/queues/s1_teacher_arch.txt --hours 24
# s2 (pull 후)
./tools/campaign_start.sh --queue config/queues/s2_uncertainty_kd.txt --hours 30
```

s1 큐 6건(T00→C0→A0→T01→T02→T03), 나머지는 게이트. s2 큐는 teacher 1건뿐이고
calibration·student 10건·GTVar 2건을 전부 게이트가 순차 개방한다.

## 2.7 남은 것

- **NAF-U fallback 미구현** — 조건이 성립하면 게이트가 로그만 남긴다. 계획 §4.3 자신이
  "1시간 넘기면 다음 캠페인으로" 라 했고, 기존 인터페이스 계약(residual·forward 시그니처)
  을 지켜야 해 즉흥 구현은 위험하다. 필요해지면 별도로 만든다.
- **§8-4 "RR 20장 중 12장"** — 장면별 지표 산출은 가능하나 임계값 12/20 은 우연히
  통과할 확률이 25% 라 판정 기준으로는 약하다. 쓰려면 임계를 올리거나 별도 검정이 필요하다.
- **§9 의 evaluator version·checkpoint hash** 는 자동 기록되지 않는다.

---

# Part 3. 적대 검증 결과 추기 (2026-09-02)

4관점 검토 + 지적별 반박을 34 agent 로 돌려 **29건 중 11건 확정, 18건 반박**됐다.
반박된 것 대부분은 Part 2 구현에서 이미 막힌 항목이다(게이트 부재·work_dir 검사·
mode_modulation 검증·T05 config 부재·시트 기록 등). 확정분 중 Part 1 에 없던 것만 적는다.

## 3.1 [critical] §8 의 "최소 조건"이 실패가 확정된 teacher 를 전항목 통과시킨다

직전 KD 캠페인이 무효가 된 원인으로 지목된 teacher **c6 를 §8 의 네 조건에 넣으면
전부 통과한다.** 직접 재현했다 (student = W96 계열):

| §8 조건 | c6 실측 | 판정 |
|---|---|---|
| student 보다 HQNR 이 0.011 이상 나쁘지 않음 | Δ −0.0026 (best) / −0.0048 (plateau) | **통과** |
| HQNR 동급이면 ERGAS 1% 개선(2% 권장) | −2.76% | **통과** |
| SCC 비열화 | 0.9908 vs 0.9901 | **통과** |
| RR 20장 중 12장 이상 승 | **20/20** (`tools/scene_compare.py`, 이항 p<0.0001) | **통과** |

즉 이 게이트는 **KD headroom 유무와 무관하게 통과 신호를 낸다.** "조건 미충족이면
headroom 부족으로 기록" 이라는 §8 의 안전장치가 c6 반례로 무력화된다.

**권고.** 조건 ①의 부등호를 뒤집는다 — "student 보다 **HQNR 이 높을 것**", 그리고
best 와 plateau 의 **부호가 일치**할 것. 조건 ②의 ERGAS 는 판정에서 빼고 기록 항목으로
내린다. 이 조건을 만족하는 case 가 하나도 없으면 그것이 "4-6M width 축에 HQNR
headroom 없음" 이라는 확정 결론이다.

## 3.2 [major] best−plateau 격차가 band 를 넘는 실행이 실재한다

완주 43개 실행 전수로 `best_hqnr − plateau(ep200–245) 평균` 을 계산하면
**평균 0.00213 · 최대 0.01297** 이다. 최대값이 동급 band(0.011)를 넘는다 —
best 하나로만 비교하면 band 판정 자체가 뒤집힐 수 있다는 뜻이고, Part 1 §1.1 의
부호 반전 사례와 같은 뿌리다. `tools/plateau_report.py` 병기가 선택이 아니라 필수다.

## 3.3 [major] §8-4 "20장 중 12장" 은 산출 수단이 없었고, 문턱도 약하다

기존 평가 경로는 장면별 값을 즉시 평균으로 접어 반환해 이 조건을 낼 수 없었다.
→ **`tools/scene_compare.py` 를 만들었다** (장면별 ERGAS/SAM/RMSE + 승수 + 이항검정).
다만 **12/20 은 우연히 통과할 확률이 25%** 다. 양측 p<0.05 는 15/20 부터이므로,
쓰려면 문턱을 15/20 으로 올리거나 p 값을 함께 보고해야 한다.

## 3.4 [minor] §8-2 의 tie-break 지표 SCC 에 band 가 없다

계획은 SCC 순서만 정하고 무시 임계를 정의하지 않는다. 이미 포화된 지표(0.990 대)의
4자리 차이로 winner 가 갈릴 수 있다. 게이트는 `SCC_TIE = 0.0005` 를 쓰므로
**보고서도 같은 값을 명시**해 코드와 문서를 맞춘다.

## 3.5 시간 추정 정밀화

평가·체크포인트 오버헤드가 width·MS/dual 과 무관하게 **0.406–0.427h 로 일정**함이
전 실행에서 확인됐다. 순수 학습시간만 width 에 비례하므로, Part 1 §1.5 의
"계획 추정이 보수적" 이라는 판단이 정량으로 확인된다.

---

# Part 4. 실행 전 지적 3건 수정 (2026-09-02)

기동 직전 점검에서 나온 지적 세 건을 전부 재현·수정하고 시뮬레이션으로 검증했다.

## 4.1 [확정 버그] s1 에서 T04 와 T05 가 같은 패스에 열렸다

게이트가 T04 를 emit 한 **직후 같은 호출에서** T05 판정을 했다. T04 의 work_dir 이
아직 없어 "중간점 진행 중" 대기 조건을 통과했고, 러너는 출력 전체를 먼저 수집하므로
**T04 결과를 보지 않은 채 winner 가 정해졌다.**

- 수정: `emit()` 이 이번 호출에서 연 tag 를 `EMITTED` 에 기록하고, T05 게이트가
  `work_dir` 존재뿐 아니라 `EMITTED` 도 확인해 다음 패스로 미룬다.
- 함께 고친 것: **중간점이 winner 가 되면 대응 dual config 가 없었다.**
  `S1_T05_W152_D123_DUAL`(4.8935M)·`S1_T05_W168_D123_DUAL`(5.9731M)을 만들고
  매핑에 추가했다.
- 검증(임시 ROOT 시뮬레이션): 패스 1 = `S1_T04A` 만 개방 → 패스 2 = 중간점 완료 후
  그 winner 의 `S1_T05_W152_D123_DUAL` 개방. T05 는 패스 1 에서 열리지 않는다.

## 4.2 [확정] s2 calibration PASS 가 학습 데이터 재평가였다

validation loader 를 만들어 두고 쓰지 않아, head 학습과 Spearman·5분위·NLL 판정이
**모두 같은 augmented train loader** 에서 이뤄졌다. 일반화를 주장하는 게이트가
외운 것을 통과로 읽는 구조다.

- 수정: 고정 상수(q05/q95·κ)는 계획대로 **train** 에서, PASS 3지표는 **held-out
  validation 전체**에서 계산한다. `uq_norm.json` 에 `diag_split`/`const_split` 기록.
- 검증: 재실행 결과 `진단 split: val · 상수 split: train · val 픽셀 1,105,920`,
  held-out Spearman 0.7097 · 5분위 단조 · NLL +0.266 개선 → PASS.

## 4.3 [확정] GTVar 의 1K gradient audit 미구현 + 한 seed 만 끝나면 복구 불가

계획 §6 이 요구한 `||grad(λ_V·L_GTVar)|| / ||grad(L_hard)||` 측정·조정이 없었고,
GTVar 게이트는 어느 한 seed 만 완료돼도 전체를 닫아 잔여 seed 를 복구하지 못했다.

- `tools/gtvar_audit.py` 신설. 계획이 못 박은 **"두 seed 전에 한 번만 조정하고 양
  seed 에 동일 고정"** 을 지키기 위해 학습 루프가 아니라 **teacher 디렉터리에 결정을
  기록**하는 방식으로 만들었다(`uq_head/gtvar_audit.json`). 두 seed 의 KDTrainer 가
  같은 파일을 읽으므로 λ_V 가 자동으로 일치한다. audit 결과가 없으면
  `u_full_gtvar` 는 아예 시작하지 못하도록 assert 를 걸었다.
- 게이트가 GTVar 를 열기 전에 audit 을 자동 실행하고, 실패하면 열지 않는다.
- seed 복구: 이미 어느 λ 로 시작했으면 **그 λ 를 고정한 채 잔여 seed 만** 연다.
  검증(시뮬레이션): `S2_GTVAR_L010_S2025` 만 완료된 상태에서 `S2_GTVAR_L010_S1234`
  가 열리고 다른 λ 는 열리지 않는다.

### 4.3.1 audit 이 드러낸 추가 문제 — 계획의 이분 규칙이 부족하다

스모크 측정에서 λ_V=0.10 의 gradient 비율이 **1.2~1.4** 로 목표(0.05–0.10)의
12배를 넘었다. 계획의 규칙대로 0.05 로 낮춰도 **0.6~0.7 로 여전히 6배**다.

→ 도구가 이분 규칙을 먼저 적용하되, 그것으로도 대역에 못 들어가면 **대역 중앙
(0.075)을 맞추는 λ_V 를 풀어서** 쓰도록 했다(어느 경로였는지 `rule` 에 기록).
규칙의 문언보다 규칙이 명시한 목표를 지키는 쪽이 맞다고 판단했다.

**단, 이 수치는 확정이 아니다** — 스모크는 40 step·미학습 student 에서 잰 값이라
실제 1K audit 값과 다르다. 실행 시 게이트가 1K 로 다시 재고 그 값을 기록한다.
λ_V 가 0.10 에서 크게 내려가면 "GT-variance 항이 원안 강도로는 hard loss 를
압도한다" 는 것 자체가 기록할 결과다.

## 4.4 수정 후 재검증

```
unit test        14/14
smoke            신규 dual 2벌 포함 통과 (mode_modulation·work_dir 검사 포함)
게이트 시뮬레이션  3/3 (T04→T05 순서 · 중간점 winner dual · GTVar seed 복구)
calibration      held-out val 로 PASS 재확인
gtvar audit      실행 확인 — 목표 대역 해석 경로까지 동작
```
