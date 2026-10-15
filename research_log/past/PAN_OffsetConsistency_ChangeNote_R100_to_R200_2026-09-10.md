# PAN Offset Consistency — 데이터셋 규모 반영 및 WV3 반경 변경 명세

**작성일:** 2026-09-10  
**문서 유형:** 기존 10 GPU-hour 계획에 대한 변경사항만 정리한 별도 문서  
**상위 계획:** `PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md`  
**새 근거:** `2026-09-10_pa-a1-a3-s1-results.md`의 **§7 추기 포함 버전**  
**현재 적용 대상:** WV3의 N1 / N2_SG / 시간 적격 시 N3_NOSG  
**상태:** 사용자 요청에 따른 설정 변경 명세. 실제 서버 config 수정·작업 중단·재기동을 수행한 기록은 아니다.

> **이번 변경의 핵심**  
> WV3의 추가 PAN 변위 반경을 **1.0 → 2.0 현재 PAN/HR pixel**로 변경한다.  
> 반경은 데이터셋의 관측된 변위 규모를 참고해 **run 시작 전에 결정하고 고정**한다. 매 sample마다 무작위로 바뀌는 것은 반경이 아니라 그 안에서 뽑는 **ε**이다.  
> 원판 uniform, native/corrupted 1:1, N1–N3 loss, W96–D124 및 정합 CNN, 평가 방법은 유지한다.  
> 이미 학습 중인 R100 run에 값을 몰래 덮어쓰지 않는다. **주 비교는 R200 새 독립 학습**, 진행분을 이어 쓰는 경우에는 **R100→R200 단계 학습**으로 별도 기록한다.

---

## 1. b의 정확한 지위

| 기호 | 정의 | 언제 정하는가? |
|---|---|---|
| `b_d` | 데이터셋 d에서 사용할 추가 변위의 최대 길이, HR px | 관측 통계를 검토한 뒤 run 시작 전에 고정 |
| `ε_(t,i)` | update t의 sample i에 실제로 추가하는 2차원 PAN sampling shift | corrupted update마다 sample별로 새로 추출 |
| `δ_i` | audit이 입력 영상에서 추정한 기존 정합 차이 | 분석 자료; 이번 consistency의 sample별 label이 아님 |
| `ĉ₀`, `ĉε` | 동일한 aligner가 예측한 원본/추가 변형 입력의 전체 correction | 해당 입력의 forward에서 예측 |
| `μ_d` | 설명에 필요한 경우 사용하는 데이터셋 공통 offset 기호 | b와 혼용하지 않음 |

**b는 데이터셋 의존적인, 관측 통계를 참고한 augmentation 설정값이다.** 네트워크가 역전파로 배우는 parameter도, 영상마다 자동 추정하는 값도 아니다. 같은 dataset·protocol·case 비교에서는 고정한다.

현재 수치는 데이터가 유일하게 결정한 통계량 자체가 아니라, 관측된 규모를 학습 범위로 옮기는 **설계 선택**이다. 따라서 “완전 무하이퍼파라미터”, “학습 중 자동 적응”, “정확한 센서 noise 분포의 재현”이라고 설명하지 않는다. 데이터 버전·전처리·calibration split이 바뀌면 별도 protocol에서 다시 검토한다.

기존 구현의 `corruption.radius_hr`가 본 문서의 b에 해당한다. 코드에서 `b`라는 키를 추가해 기존 `radius_hr`와 서로 다른 값이 동시에 존재하게 만들 필요는 없다.

---

## 2. 변경 근거와 적용 범위

### 2.1 기존값과 새 관측

기존 계획 §3.1은 과거 train audit의 전역 변위 크기 P90 **0.250 LR px**에 배율 4를 적용해 `radius_hr=1.0`을 사용했다. 작은 patch 추정의 불확실성을 인정한 임시 proxy였다. [S1 §3.1]

새 보고서 §7은 다음을 기록한다. 아래 수치는 **각 split의 PAN/HR pixel 단위**이다. [S2 §7]

| Dataset | train 변위 크기 중앙값 / P95 | train 측정 하한 | FR 변위 크기 중앙값 / P95 | 이번 설정의 지위 |
|---|---:|---:|---:|---|
| WV3 | 0.17 / 0.76 | 0.16 | 1.79 / 1.98 | **이번 변경: b=2.0** |
| QB | 0.35 / 1.12 | 0.03 | 2.78 / 2.98 | 후속 적용 후보: b=3.0 |
| GF2 | 0.69 / 1.00 | 0.08 | 1.62 / 1.95 | 후속 적용 후보: b=2.0 |
| WV2 | train 없음 | — | 0.36 / 0.61 | zero-shot용 별도 학습 반경을 정하지 않음 |

WV3의 FR P95=1.98 근방까지 추가 상대 이동을 경험시키도록 **실행값 2.0**을 선택한다. QB/GF2 값은 같은 관점의 후속 후보이며, 이번 10시간 큐에 다른 데이터셋 run을 추가하지 않는다. 일반적인 자동 반올림·최적화 규칙이 검증된 것으로 주장하지 않는다.

**비교 대상의 차이:** FR은 `PAN_b ← bicubic↑MS`, RR/train은 `GT ← PAN`이다. 단위가 HR px라는 이유로 두 분포가 같은 절대 정합량을 측정했다고 간주하지 않는다. 보고서는 train/RR과 FR의 방향도 다를 수 있다고 기록한다. 이번에는 방향을 target으로 옮기지 않고 **크기만 augmentation의 참고**로 사용한다. [S2 §7]

### 2.2 자료 사용의 제한 — 기존 원칙에서 명시적으로 달라지는 부분

새 규모의 근거에는 **FR 논문 평가용 20장 입력의 통계**가 포함된다. 따라서 이번 변경은 `FR-input-statistics-informed development` 설정이다.

- FR의 고해상도 정답을 사용한 것은 아니지만, 평가 입력을 분석한 결과가 설정 결정에 영향을 주었다.
- “train 통계만으로 정했다”, “test를 전혀 보지 않은 설정”으로 보고하지 않는다.
- 기존 계획 §3.1의 **FR 통계를 학습 radius에 사용하지 않는 원칙은 이번 R200 개발 protocol에서 변경**된다. 원래 R100 기록은 그대로 보존한다.
- FR 장면별 shift 벡터를 학습 label로 쓰지 않는 원칙은 유지한다.
- 이후 엄격한 독립 평가를 하려면 평가 장면과 분리된 calibration 자료에서 반경을 정한 별도 protocol이 필요하다. 이번 결과를 사후에 train-only로 재분류하지 않는다.
- b=2가 최적인지, native FR 품질이 실제 좋아지는지는 아직 확인하지 않았다.

---

## 3. 변경/유지 대조표

| 항목 | 이전 R100 | 변경 후 WV3 R200 |
|---|---|---|
| 반경 | **1.0 HR px** | **2.0 HR px** |
| 규모 근거 | 과거 train aggregate P90 | 새 보고서 §7의 FR 규모도 참고 |
| b의 선택 시점 | run 시작 전 | **동일**; run 내부에서 변하지 않음 |
| sampling 분포 | 0 중심 원판 면적 uniform | **동일** |
| 총 이동 길이 | 0 ≤ \|ε\|₂ ≤ 1 | **0 ≤ \|ε\|₂ ≤ 2** |
| 방향 | 모든 방향 | **동일**; 관측된 센서 방향을 평균으로 넣지 않음 |
| native : corrupted | 1:1 update 교대 | **동일** |
| offset loss | 상대 correction 차이의 평균 L1 | **동일** |
| N2/N3 λ | 최대 0.01, 처음 5K update ramp | **동일** |
| b 또는 \|ε\|로 loss 나누기 | 없음 | **추가하지 않음** |
| Aligner margin | 4 HR px | **4 HR px**; 공식 재계산 결과 동일 |
| 64² train aligner view | 56×56 | **동일** |
| 512² FR aligner view | 504×504 | **동일** |
| U-Net / aligner 구조 | W96–D124 / A1 dual-stem | **동일** |
| MS·base·GT·출력 frame | 이동 없음, M-frame | **동일** |
| correction 출력 제한 | b 기반 tanh/clamp 없음 | **동일** |
| corrupted raster 경로 | PAN 추가 warp → 예측 correction warp | **동일**, 두 번 보간 |
| native 평가·best 선정 | 세 view / dual-best / last | **동일** |
| run ID·config hash | 기존 protocol | **새 protocol 및 R200 suffix** |

실질적인 학습 분포 변경은 **추가 변위의 규모**이다. 여기에 규모 근거·metadata·run 이력 관리·진단 범위를 함께 갱신한다. 새로운 loss나 case는 만들지 않는다.

---

## 4. Sampling과 단위 — 기존 코드를 유지할 부분

\[
 u_1,u_2\sim U(0,1),\quad
 \rho=b_d\sqrt{u_1},\quad \theta=2\pi u_2,
\]
\[
 \boxed{\epsilon=(\rho\sin\theta,\rho\cos\theta)}.
\]

WV3에서는 b=2.0이다. 매번 2픽셀 이동하는 것이 아니라 **반지름 2인 원판 안에서 2차원 벡터를 새로 추출**한다.

- 축별 `uniform(-2,2)`로 바꾸지 않는다. 그것은 전체 길이가 최대 2√2인 정사각형 분포이다.
- `rho = b * rand()`로 바꾸지 않는다. 기존 `sqrt(rand())`를 유지한다.
- 별도의 양의 최소 반경을 주지 않는다. audit의 측정 하한을 synthetic ε의 최소값으로 쓰지 않는다.
- 공통 flip/rotation과 B0 전처리 이후 **PAN만** 추가 이동한다.
- ε는 sample별 global translation이며, 모든 공간 위치에 동일하다. band별 변형·회전·local flow가 아니다.
- 항상 원 PAN에서 새 Pε를 만든다. 이전 update의 이동을 누적하지 않는다.
- ε는 생성과 target 계산에만 사용한다. aligner/U-Net 입력으로 주지 않는다.

단위는 **현재 네트워크 입력 PAN 격자**이다.

\[
 2.0\ \text{현재 PAN/HR px}=0.5\ \text{현재 LRMS px}.
\]

실제 sampler에는 **2.0**을 사용한다. 새 보고서의 HR 값을 다시 ×4하거나, 적용 직전에 ÷4하지 않는다. 64²→512² 입력 크기 변화에 따라 correction을 ×8하는 것도 하지 않는다.

분포에서 유도되는 검사값은 다음과 같다. 관측된 센서 통계가 아니라 **sampler 자체의 이론값**이다.

| 양 | R100 | R200 |
|---|---:|---:|
| 축별 평균 | 0 | 0 |
| 축별 표준편차 b/2 | 0.5 | 1.0 |
| 평균 이동 길이 2b/3 | 약 0.667 | 약 1.333 |
| P(\|ε\|₂ ≤ 1), corrupted 표본 중 | 1.0 | 0.25 |

이전과 같은 RNG seed·같은 난수 추출 순서를 사용하면, **원판의 기초 난수는 같고 ε의 크기만 두 배**가 된다. 새 N1/N2/N3끼리는 같은 sample/update에서 동일한 ε를 받아야 한다. Native의 ε=0을 포함한 전체 로그와 corrupted-only 통계를 분리한다.

---

## 5. Loss·역전파는 바꾸지 않는다

Corrupted step에서:

\[
 P_\epsilon=\mathcal W(P,\epsilon),\quad
 \hat c_0=A_\phi^V(P,M),\quad
 \hat c_\epsilon=A_\phi^V(P_\epsilon,M),
\]
\[
 \hat Y_\epsilon=M+F_\theta\big([\mathcal W(P_\epsilon,\hat c_\epsilon),M]\big).
\]

| Case | Loss | clean aligner 참조의 gradient |
|---|---|---|
| N1 | L_rec만 | no-grad, offset 항은 로그에만 사용 |
| N2_SG | L_rec + λ·mean\|ĉε+ε−sg(ĉ₀)\| | stop-gradient |
| N3_NOSG | L_rec + λ·mean\|ĉε+ε−ĉ₀\| | 양쪽 역전파 |

N2의 target은 계속 **`c0.detach() - eps`**이다. `-eps`만 total correction의 target으로 쓰거나, FR 중앙값·audit edge 추정값을 target으로 대체하지 않는다.

Native step에서는 offset loss를 학습에 넣지 않고 **clean aligner와 U-Net에 reconstruction gradient를 전달**한다. 큰 U-Net은 step마다 한 번만 실행한다.

\[
 \lambda(t)=0.01\min\left(1,\frac{t+1}{5000}\right)
\]

위 coefficient는 N2/N3의 corrupted step에만 적용한다. N1과 native step은 0이다. b가 두 배가 되더라도 λ를 자동으로 반으로 줄이거나, offset L1을 b로 나누는 새 정규화를 추가하지 않는다. L1 값의 변화와 parameter gradient의 변화는 같은 양이 아니므로 기존 gradient 진단을 유지한다.

상대 loss는 **추가 변위에 대한 반응**을 학습시키며, 미지의 절대 offset까지 참값으로 고정하지는 않는다. b 확대가 이 한계를 해결했다고 주장하지 않는다.

---

## 6. Crop·보간·평가에서 확인할 연쇄 변경

### 6.1 Aligner crop

기존 규칙을 유지한다.

\[
 m_A=4\left\lceil\frac{b+2}{4}\right\rceil.
\]

b=1과 b=2는 모두 **m_A=4**이다. 따라서 WV3에서는 `P[...,4:-4,4:-4]`, `M[...,4:-4,4:-4]`를 그대로 쓴다. 먼저 full PAN을 warp하고 이후 crop한다. crop 후 정규화한다. sample의 실제 ε 크기에 맞춰 crop을 동적으로 바꾸지 않는다.

이 crop은 **aligner 입력용**이다. U-Net 입력·residual base·GT를 자르지 않으며, reconstruction loss는 기존 full domain을 유지한다.

QB b=3을 후속 적용할 경우에는 같은 공식에서 m_A=8이 된다. 이번 WV3 구현에 그 값을 적용하지 않는다.

### 6.2 보간과 유효 영역

`W(W(P,ε),ĉε)`의 두 단계 보간을 N1/N2/N3에서 동일하게 유지한다. 이번 변경 중 일부 case만 `W(P,ε+ĉε)`로 합성하지 않는다. 알려진 역변위가 맞아도 보간·경계로 남는 영상 오차는 offset 예측 오류와 구분한다.

**b=2는 총 correction의 상한이 아니다.** `ĉε` 또는 native `ĉ₀`를 2로 clamp하거나, 평가 ROI의 안전 범위를 b만으로 자동 선언하지 않는다.

기존 native 평가의 ROI와 scene set은 유지한다. 더 큰 correction으로 공통 ROI의 support가 실패하면 기존 적격성 규칙대로 **Aligned-valid selection을 INVALID**로 남긴다. 실패 장면만 제외하거나 ROI를 줄여 새 best를 만들지 않는다. Raw-original 경로는 별개로 보존한다. 필요한 경우 ROI 재설계는 별도 평가 protocol과 공통 재평가로 수행하며 이번 patch에 묵시적으로 포함하지 않는다.

### 6.3 평가·진단

유지: `fr_mat20`, metric v2 py, 장면별 HQNR 곱의 평균, Raw-original/Raw-valid/Aligned-valid, best_raw/best_aligned/last. Native inference의 추가 ε는 0이며 **학습된 correction은 계속 사용**한다.

진단의 반경 정보만 갱신한다. 이전 `0, ±0.5, ±1.0` 축 probe를 보존하고 `±1.5, ±2.0`을 추가해, **같은 절대 변위 구간에서 이전 run과 비교**할 수 있게 한다. 이는 radius를 다시 튜닝하는 학습 sweep이 아니다. 새 stress manifest/hash를 저장하고 native selector에 사용하지 않는다.

- 주 offset error는 HR px 원단위로 보고한다.
- b로 나눈 상대 error를 함께 기록해도, b=1과 b=2의 정규화값만 직접 비교하지 않는다.
- 0~0.5, 0.5~1, 1~1.5, 1.5~2 구간을 분리하고 sample 수를 기록한다.
- 같은 축 response slope의 이상값은 −1, 교차축은 0이다.
- FR probe에서는 절대 참 정합이 아니라 **알려진 추가 변화의 closure**를 검사한다.
- 구버전의 자동 `±2b` 외삽은 새 b에서 ±4가 된다. 이를 이번 변경의 필수 진단으로 자동 확대하지 않는다. 필요 시 별도 support/ROI 검증과 진단 명시 후 사용한다.

---

## 7. 구현자가 수정할 config

아래는 **기존 계획 §14 template 기준의 patch 명세**이다. 실제 repository key 이름은 구현과 연결해야 하며, 이 파일이 실행 서버의 config를 자동 수정하지는 않는다.

### 7.1 반드시 바꾸는 값

```diff
-protocol_id: pan_offset_rel_10h_v1
+protocol_id: pan_offset_rel_10h_v2_wv3_r200_frstat

 corruption:
-  scale_source: training_audit_appendix_E_WV3_global_norm_P90
-  calibration_quality: train_aggregate_proxy
-  observed_value_lr: 0.250
-  ratio: 4
-  radius_hr: 1.000
+  scale_source: pa_a1_a3_s1_results_20260910_section7_WV3_FR_report_norm_P95
+  calibration_quality: fr_input_statistics_informed_development
+  observed_value_hr: 1.98
+  radius_hr: 2.000
```

`ratio: 4`의 제거는 **옛 calibration 환산 필드만** 뜻한다. 실제 PAN/MS 해상도 배율 4, bicubic upsampling 4는 그대로 유지한다. 새 source가 이미 HR 단위이므로 옛 `0.250 × 4` 계산이 `radius_hr`를 1.0으로 덮어쓰지 않도록 loader를 확인한다.

위 diff에 없는 기존 corruption 필드(`uniform_disk_area`, center, RNG, 적용 순서, 1:1)는 유지한다. active 설정에서는 **`radius_hr=2.0`이 단일 source of truth**이며, 변경된 source는 근거 metadata이다. 매 step/epoch/inference마다 FR를 다시 측정해 반경을 갱신하지 않는다.

### 7.2 Manifest에 추가할 provenance — metadata 예시

```yaml
radius_provenance:
  dataset: WV3
  profile: FRSTAT_R200
  selected_radius_hr: 2.0
  observed_statistic: report_abs_delta_P95
  observed_value_hr: 1.98
  source_split: FR_paper_mat20
  source_scene_count: 20
  report_file: 2026-09-10_pa-a1-a3-s1-results.md
  report_section: 7
  reference_pair: PAN_b_from_bicubic_upsampled_MS
  uses_evaluation_input_statistics: true
  uses_FR_HRMS_ground_truth: false
  selection_note: nearby_execution_scale_chosen_before_new_run
  radius_is_trainable: false
  radius_changes_within_pure_run: false
  runtime_recalibration: false
```

이 metadata는 edge matching의 sample별 label cache가 아니다. 원자료 JSON 경로는 보고서에 적혀 있지만, 이번 문서 작성 시 그 JSON이나 실제 서버 코드를 직접 읽은 것은 아니다. 보고서 스냅샷의 SHA256은 §10에 기록한다.

### 7.3 Run ID

주 비교용 새 독립 run의 예시:

```text
PO10_N1_REC_W96_D124_WV3_S2025_R200_FRSTAT
PO10_N2_OFFSG_W96_D124_WV3_S2025_R200_FRSTAT
PO10_N3_OFFNOSG_W96_D124_WV3_S2025_R200_FRSTAT
```

실제 seed/server는 기존 대응 block을 유지한다. 25K screening을 택하면 모든 해당 case에 기존 규칙대로 `SCREEN25K`를 붙인다. 기존 R100 결과 폴더나 config를 덮어쓰지 않는다.

---

## 8. 이미 실행 중인 R100 실험의 전환

### 8.1 주 비교의 기본 선택: 새 R200 독립 학습

**“처음부터 b=2로 학습한 N1–N3”를 비교하려면 이 방식을 사용한다.**

1. 실행자는 R100의 다음 안전한 저장 시점에서 checkpoint·실제 update 수·optimizer/scheduler/scaler·RNG·resolved config·로그를 보존한다. 중단하면 정상 완료가 아니라 `STOPPED_FOR_PROTOCOL_CHANGE` 등으로 표시한다. 정상 완료된 R100 결과는 그대로 완료 기록으로 둔다.
2. 이미 queue에 올라간 구버전 다음 case가 자동 실행되지 않도록 runner 설정도 함께 확인한다.
3. 새 config·새 run ID로 **동일한 random initialization snapshot**에서 시작한다. 학습된 R100 weight를 새 pure R200의 초기값으로 사용하지 않는다.
4. optimizer·scheduler·scaler·update counter·data/augmentation/corruption RNG를 새 run 시작 상태로 설정한다. smoke에서 소비한 상태를 학습에 이어 쓰지 않는다.
5. N1/N2/N3는 모두 처음부터 같은 b=2, 같은 데이터 순서·기초 난수·budget으로 실행한다. 일부만 b=1이면 loss-only 비교가 아니다.
6. native 평가 selector는 새로 시작한다. 기존 R100 best는 별도 이력으로 보존한다.

**처음부터의 대응 비교에서 바꾸는 것은 반경과 근거 metadata이며, 학습 완료 weight는 공유하지 않는다.**

### 8.2 계산량을 보존해 이어가야 하는 경우: 별도의 단계 학습

진행분을 이어 쓰는 방법도 가능하지만, **pure R200 결과와 같은 것으로 분류하지 않는다.** 이것은 재시작의 대체 경로이며, 추가 case 실행을 요구하는 것이 아니다.

공통 전환 update를 `t_switch`라고 정의한다. `t_switch`는 R100으로 완료한 optimizer update 수이며, 새 update index `t_switch`부터 b=2이다.

```text
0 ≤ update_index < t_switch : b=1
update_index ≥ t_switch     : b=2
```

필수 조건:

- N1/N2/N3가 각자의 대응된 R100 학습 이력에서 **같은 t_switch**로 전환해야 상대적인 비교가 유지된다. 임의의 서로 다른 학습 시점 또는 다른 부모 case를 섞지 않는다.
- 부모 checkpoint와 그 hash를 기록하고 **새 디렉터리로 fork**한다. 예: `..._R100toR200_U10000_FRSTAT`.
- 기존 weights뿐 아니라 optimizer/scheduler/scaler/data/RNG 상태를 이어간다. update counter와 native/corrupted 교대 parity도 유지한다.
- **LR warmup과 5K offset ramp를 다시 시작하지 않는다.** 원래 전체 update index에 따라 이어간다. 그렇지 않으면 반경뿐 아니라 scheduler도 바뀐 새 설계이다.
- 남은 update 수와 R100/R200에서 학습한 update 수를 각각 기록한다. 전체가 50K라면 `t_switch + remaining = 50K`이며 R200-only 50K가 아니다.
- 부모 단계의 best 기록은 보존하고, 새 단계의 `best_raw`/`best_aligned`는 분리한다. 전환 직전 부모 재평가는 참고 baseline으로만 남기고, 변경 후 실제 학습된 checkpoint부터 새 selector 후보로 넣는다.
- 중단 없이 config 파일의 숫자만 바꾸거나, 기존 run의 header를 처음부터 b=2였던 것처럼 수정하지 않는다.

현재 각 run의 실제 진행 update와 잔여 시간은 이 문서에서 확인하지 않았다. 대응 이력이 없는 case를 만들어 맞춘 것처럼 기록하지 않는다.

### 8.3 시간 예산

이번 문서는 **기존 10 GPU-hour 계획의 변경**이며, 새로운 10시간을 추가 승인하거나 GPU 작업을 시작한 것이 아니다. 재시작에 드는 시간은 이미 소비한 시간에 더해 별도 집계한다.

남은 예산에서는 **N1/N2의 같은 학습량과 진단을 우선하고 N3는 먼저 연기**한다. 그래도 full run이 불가능하면 원문 §11.2의 공통 screening 규칙 또는 명시적인 단계 학습 분류를 사용한다. 서로 다른 budget 결과를 숨겨서 비교하지 않는다.

---

## 9. 변경 적용 전 검사와 로그

다음은 신규 case 계획이 아니라, 이번 반경 변경을 안전하게 적용하는 확인 항목이다.

- [ ] resolved config에서 모든 현재 WV3 N-case가 `radius_hr=2.0`이고, legacy calibration 코드가 1.0으로 재계산하지 않는다.
- [ ] sampler는 같은 원판 uniform이며 output shape `[B,2]`, 순서 `(dy,dx)`, 단위 현재 PAN px이다.
- [ ] 충분한 corrupted 표본에서 평균≈0, 축별 SD≈1, 평균 길이≈1.333, 길이 상한≤2가 확인된다. 소형 batch에 이론값과의 정밀 일치를 강제하지 않는다.
- [ ] margin=4, train view56²·FR view504²이며 clean/corrupt/case/inference 모두 같다.
- [ ] `(±2,0)`, `(0,±2)`, 원판 안의 대각선에서 aligner crop의 보간 source support가 실제로 유효하다.
- [ ] known offset을 추가했을 때 올바른 상대 target 부호가 `c0 − eps`이다.
- [ ] N2의 stop-gradient가 native recon까지 막지 않는다. N3의 clean 호출은 `no_grad`가 아니다.
- [ ] learned correction=0에서도 sampler를 건너뛰지 않는다. correction을 b로 clamp하지 않는다.
- [ ] λ=0.01, ramp5K, native/corrupt1:1, sequential sampling, full recon 영역은 유지된다.
- [ ] 동일 N-case끼리 sample/update/ε 이력이 대응하고 RNG 상태를 저장·복구할 수 있다.
- [ ] native ROI·scene set·metric version은 동일하며, support 실패 장면을 선택적으로 버리지 않는다.
- [ ] old run 기록·새 run 기록·전환 이력이 구분되고, restart/staged 여부가 manifest에 있다.

기존 로그에 `protocol_id`, `radius_profile`, `radius_hr`, `source_split`, `uses_evaluation_input_statistics`, `training_regime`, `parent_run_id`, `switch_update`를 추가한다. 재시작이면 parent weight가 없음을 명시하고, 단계 학습이면 기존/새 구간 길이를 남긴다.

기존의 `loss_rec`, `loss_offset`, λ, native/corrupt 구분, ε·ĉ₀·ĉε의 통계, gradient norm·cosine, native 품질과 세 평가 view도 유지한다.

**이번 문서 작성에서 확인한 범위:** 첨부 계획·추가 보고서를 대조하고 설정 diff와 수식을 작성했다. 실제 repository 구현, CUDA warp, GPU 학습, 변경 효과는 확인하지 않았다. 이 파일은 실행자가 적용할 변경 명세이며 적용 완료 보고서가 아니다.

---

## 10. 출처·원문 보존

### 내부 근거

- **[S1]** `PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md`: §3 원래 R100 규모·원판 sampler, §4 loss·교대, §5 margin, §6 warp, §7 이력, §10 평가, §11 budget, §14 template.
- **[S2]** `2026-09-10_pa-a1-a3-s1-results.md`: **§7을 포함한 126행 버전**, split별 측정 정의 및 WV3/QB/GF2/WV2 분포. §7이 없는 이전 동명 문서와 구분한다.
- **[S3]** `PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md`: 정합 CNN·W96–D124·세 평가 view·dual-best의 상위 정의.

작성 시 확인한 로컬 원문 SHA256:

```text
S1 334a13fdae48faa177d839d5b04e938b283083c97edc56b67c430152f70077f3
S2 d81d3ca22fe52364e0e3fa33b35181be0fb6a282dc431ad58eeaf68bbbc6b729
```

원문 두 파일은 수정하지 않는다. 충돌하는 항목 중 **WV3 규모 근거·radius·변경 이력 처리**는 본 변경 명세를 우선하고, 나머지 학습·평가 정의는 S1 및 S3를 따른다.

---

## 실행자 전달용 요약

**WV3 N1/N2/N3의 `corruption.radius_hr`만 학습값 기준으로 1.0→2.0 HR px로 변경한다.** 원판 uniform, 0 중심, native/corrupt1:1, λ=0.01/5K, margin4, 모델·loss·native evaluator는 유지한다. 근거는 새 §7 FR P95=1.98이며 **평가 입력 통계를 참고한 개발 설정**임을 기록한다. 반경은 run 내 고정이고 ε만 sample마다 새로 뽑는다. 현재 R100 진행분은 보존하고 새 R200 대응 학습으로 전환하는 것을 기본으로 한다. 이어서 학습하면 R100→R200 단계 학습과 전환 update를 명시한다. 기존 run에 조용히 덮어쓰지 않는다.
