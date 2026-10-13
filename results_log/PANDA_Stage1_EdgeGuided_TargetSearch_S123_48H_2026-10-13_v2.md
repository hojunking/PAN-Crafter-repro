# PANDA Stage 1 — Edge-Guided Residual Aligner / 48시간 Teacher 목표 탐색

**작성일:** 2026-10-13  
**실험계획:** `EdgeRes-Teacher48H-v2.0`  
**Campaign ID:** `PANDA_S1_EDGERES_TEACHER48H_S123_20261013_v2`  
**상태:** `PLAN_ONLY / IMPLEMENTATION_AND_RUNTIME_UNVERIFIED`  
**역할 분담:** 이 문서는 실험 case·평가·해석·저장·기록 규칙이다. 모델, runner, uploader 구현과 실제 기동은 서버 구현 담당자가 수행한다.

> **목표:** 새 Edge-Guided Residual Stage 1 전체 방법을 유지하여, 같은 Teacher+Residual Aligner 상태에서 각 데이터셋의 **HQNR 하한과 ERGAS 상한을 동시에 만족**한다. 미달 weight와 정성 이미지는 저장하지 않지만, 미달을 포함한 모든 평가 수치·case·진단은 로컬에 먼저 기록하고 `analysis` 시트에 올린다.
>
> **이전 계획의 데이터 차단 조건은 폐기한다.** 이번에는 기존 **RR train만 학습**한다. Native FR 학습 데이터, 원본 acquisition scene, RR patch→source-scene 매핑, scene/global prior cache는 필요 없다. FR test는 추론·평가에만 사용한다. 입력별 ID와 전처리 이력은 재현성을 위해 남기지만, 이것은 부모 scene 매핑 요구가 아니다.

---

## 0. 실행 요약 — 이 표가 이전 48시간 계획보다 우선한다

| 항목 | 이번 규칙 |
|---|---|
| 서버 | **s1=WV3 / s2=QB / s3=GF2**, s4·s5 제외 |
| 모델 | Stage 1 Teacher reconstruction U-Net + EdgeMatch-guided residual PAN Aligner |
| Stage 2 | Student·KD·e/q fitting/calibration 전부 제외 |
| 데이터 | 기존 PanCollection 계열 RR train/validation; 기존 RR/FR test는 평가 전용 |
| 초기 이동 | **입력 쌍마다** Scharr absolute-XY gradient L1의 정수 argmin `b` |
| CNN 입력/출력 | `[W(P,b), L_A] → Δu=(Δdy,Δdx)`; `u=b+Δu` |
| Teacher 입력 | **원본 PAN에서** `W(P,u)` 한 번 + 기존 bicubic MS base |
| 초기화 | residual FC2 zero-init; 초기 `Δu=0`, `u=b` |
| 학습 objective | `L_rec_RR + λ_abs L_abs_RR + λ_eps L_eps_RR` |
| 기본 계수 | `λ_abs=1`, `λ_eps=0.001`; 아래 양수 scalar 튜닝 허용 |
| ε | radius 2 current RR PAN px, uniform disk, 매 update; **shifted PAN의 EdgeMatch 재실행** |
| FR | same pipeline inference only; FR training loss·출력 ×4 없음 |
| 탐색 | loss 16개 기본 + 4개 조건부, optimizer 9개 profile, H50/H100/H150 |
| 시간 | 서버별 독립 48시간; 모든 조합의 완주가 아니라 실제 속도로 실행 수 결정 |
| 반복 | R1 탐색, 최종 선택 recipe의 R2·R3 확인을 우선 예약 |
| 저장 | **STRICT_TARGET_ONLY**. 미달 `last/resume/temp_eval/초기화 weight`도 저장하지 않음 |
| 시트 | 기존 `analysis`에 새 campaign으로 분리; `paper`·`ablations` 대표값 자동 변경 없음 |
| 실행 독립성 | 서버 간 대기·공유 lock 없음. 각 서버 내부 중복 GPU 실행만 방지 |

### 0.1 목표값과 동일-state 판정

| 서버 | Dataset | HQNR ≥ | ERGAS ≤ |
|---|---|---:|---:|
| s1 | WV3 | **0.958** | **2.040** |
| s2 | QB | **0.923** | **3.500** |
| s3 | GF2 | **0.965** | **0.522** |

반올림 전 값으로 `H>=H* AND E<=E*`를 판정한다. RR와 FR은 서로 다른 평가 집합이지만 **A/T weights, step, inference algorithm revision**은 같아야 한다. 입력이 다르므로 `b`·`Δu`의 숫자가 RR/FR에서 같은 것은 아니다.

저장 자격은 **full method + 전체 RR/FR 평가 완료 + finite + 동일 state 검증 + 두 목표 통과**다. 예: HQNR 0.9650인 모델과 ERGAS 0.5219인 다른 step을 합쳐 통과로 만들지 않는다. 입력별 EdgeMatch는 각 evaluation input에서 결정적으로 다시 계산할 수 있으나, FR에서는 다른 weight·보정 배율·test-time optimizer를 쓰지 않는다.

---

## 1. 근거와 변경 범위

### 1.1 우선순위

**[S1, 최우선 방법 명세]** `PANDA_Stage1_EdgeGuided_Residual_Aligner_Final_2026-10-13.md`  
SHA256: `ac9e2cc7ef58defb03bf94b615faa525c911ead64471e9e7f6348a91d63d30d7`

**[S2, 운영 규칙만 상속]** `PANDA_Stage1_Teacher_TargetSearch_S123_48H_2026-10-13_v1.md`  
SHA256: `9bc1cf41080b56ae91a303005f5dc453c2c0fa87e0bf6b035cc701111a2bddad`

**[S3, 과거 결과의 탐색 우선순위 참고]** `TA2_Current_Results_Synthesis_2026-10-13.md`  
SHA256: `d0847f941f34b2b2b4123c1dcc0e3b9ecd2b34980eae6557beb8b369edf2b498`

[S1]의 방법이 [S2]의 scene-conditioned 방법을 대체한다. [S2]에서는 서버/목표/48시간/미달 미저장/전체 숫자 기록/평가 cadence를 상속한다. [S3]는 새 EdgeMatch 모델의 성능 근거가 아니라 시도 순서 참고다. 과거 NCC/NGF structural loss와 이번 raw Scharr L1의 계수는 같은 척도가 아니다.

문서의 **[원문 고정]**, **[이번 실험 제안]**, **[서버에서 확정할 운영값]**을 구분한다. 아래 탐색 범위, EdgeMatch 반경 후보, padding-support 수치, seed 및 시간 배정은 검증된 최적값이 아니다.

### 1.2 삭제·교체할 과거 구성

| 이전 구성 | 이번 구성 / 조치 |
|---|---|
| `b_s` / source-scene direction+magnitude | **삭제**; per-input `b=EdgeMatch(P,L_A)` |
| `b_global` / sensor-wide constant | **삭제** |
| source-scene ID/crop map 필수 | **삭제**; 실제 입력 sample ID만 재현성용으로 기록 |
| FR training assets 필수 / `L_eps_FR` | **삭제**; RR training only |
| `concat(h,b)`·FC66 / `αd+βn` | **삭제**; FC64→32→2 직접 Cartesian residual |
| native/shifted branch에 같은 prior 강제 | **금지**; `b0`, `b_eps` 각각 해당 PAN에서 계산 |
| prior 뒤 residual 순차 warp를 T에 제공 | **금지**; `W(P_original,b+Δu)`로 한 번 |
| FR 크기 보정 ×4 | **없음**; 매 입력 PAN pixel 단위 |
| 외부 SpaceNet/PSGAN 등 다운로드 | 이번 48시간 작업의 선행조건·학습 범위에서 제외 |

실행 config의 `requires_fr_train=false`, `requires_source_scene_map=false`를 확인한다. 오래된 runner의 scene/prior/FR-data 검사 때문에 새 실험을 차단하지 않는다. **실제 RR 데이터·LMS·RGB mapping·loss/warp 정합성 오류만 차단**한다.

---

## 2. 추가 검토 — 새 설계에서 실제로 검정할 문제

### 2.1 실행 가능성은 개선됐지만, 성능 문제까지 해결됐다는 뜻은 아니다

[S1 §0, §8]은 native FR train과 scene mapping 없이 실행할 수 있도록 요구 데이터를 기존 RR로 축소했다. 입력별 EdgeMatch가 각 해상도에서 초기값을 직접 계산하므로, RR에서 배운 절대 correction을 FR에 그대로 외삽하는 부담을 나누려는 설계다.

그러나 [S3]의 사전 정합 실패와 FR 후기 악화는 새 frontend에도 확인해야 할 위험이다. 새 EdgeMatch는 과거의 NCC/subpixel 사전 정합과 달라 결과를 그대로 대입할 수 없고, 반대로 **단순화했으니 더 정확해졌다고 주장할 수도 없다**.

### 2.2 중요한 차이 ① — 큰 total gain이 CNN 학습을 증명하지 않는다

새 구조는 `u=b+Δu`이며 shifted branch의 `b_eps`를 재계산한다. 그러므로

`u_eps-u0 = (b_eps-b0) + (Δu_eps-Δu0)`.

EdgeMatch만으로 known ε의 상당 부분을 상쇄하면, 아직 residual이 0인 초기 모델에서도 total gain이 높을 수 있다. **total gain↑를 learned Aligner의 기여↑로 단독 해석하지 않는다.** 이 문제는 [S1 §7.3]에도 명시되어 있다. §11에서 edge/residual/total 세 성분을 나눠 기록한다.

반대로 residual gain이 작다는 이유만으로 collapse라고 하지 않는다. EdgeMatch가 이미 이동을 설명했다면, CNN의 올바른 역할은 남은 subpixel·잘못된 matching 오차를 처리하는 것이다.

### 2.3 중요한 차이 ② — residual의 ε 정답은 단순히 `-ε`가 아니다

이번 loss가 요구하는 변화는

`Δu_eps - Δu0 ≈ -(ε + b_eps - b0)`.

`b_eps=b0`로 고정했을 때의 이전 식을 그대로 쓰면 방법이 달라진다. `b_eps=b0-ε`를 강제로 주입하면 deterministic estimator의 실제 동작을 우회한다. **두 방식 모두 이번 후보에서 금지**한다.

### 2.4 중요한 차이 ③ — 느린 A가 항상 보수적인 안전책은 아니다

이전 direct Aligner는 초기 correction이 0인 경우가 많았지만, 이번 모델은 **초기 `u=b`**다. 만약 정수 EdgeMatch가 잘못된 위치를 선택했다면 A 학습률을 낮출수록 그 잘못된 보정이 오래 남을 수 있다.

따라서 QB에서 과거에 느린 A를 우선 검토했다고 이번에도 무조건 느린 A만 시도하지 않는다. **`prior-only b`와 `current b+Δu`의 숫자 비교를 먼저 보고**, 잘못된 초기값의 복구가 늦으면 빠른 A(ET02/ET08), residual이 좋은 초기값을 손상시키면 느린 A(ET01/ET00)를 비교한다. 이는 탐색 가설이지 자동 gradient routing 변경이 아니다.

### 2.5 중요한 차이 ④ — Scharr 절댓값도 분광·blur 차이를 제거하지 않는다

EdgeMatch는 `PAN↔LMS RGB 평균`, L_abs는 `corrected PAN↔GT RGB 평균`이다. 두 참조는 위상 외에도 대역폭·edge amplitude가 다르다. 절댓값은 gradient polarity를 제거하지만 magnitude 차이까지 없애지는 않는다. **같은 Scharr를 쓴다고 두 최적 shift가 같아지는 것은 아니다** [S1 §3.1, §6].

이 검토를 이유로 NCC/정규화/blur/learned weighting을 기본 method에 넣지 않는다. 대신 raw edge scale, score 분포, `b`가 GT 구조를 악화시키는 비율, residual이 되돌린 비율을 수치로 기록한다. `L_abs` 감소만으로 물리적 변위 정답을 회복했다고 하지 않는다.

### 2.6 integer prior가 틀렸다면 residual이 0.5px보다 클 수도 있다

subpixel refinement가 주요 동기지만, CNN이 틀린 정수 prior를 되돌리려면 `|Δu|>0.5` 또는 `>1`이 필요할 수 있다. **residual/total에 tanh·hard clamp·L1 shrinkage를 새로 추가하지 않는다.** 큰 residual은 진단 대상이며 자동 실패가 아니다. nonfinite나 실제 loss support 소실은 별도 수치 실패다.

### 2.7 데이터 문제는 줄었지만 계산비용과 초기화 검증이 중요해진다

각 training update에는 native와 shifted 입력의 **EdgeMatch 두 번**이 필요하다. 출력 해상도가 커지는 FR 평가에도 per-input matching 시간이 추가된다. CNN-only 시간만 `Infer(ms)`로 쓰면 안 된다.

또한 FC2와 공통 ResBlock의 마지막 Conv가 zero-init이다. 초기 앞단 gradient가 0인 것은 구조적으로 가능하므로, 최초 step의 body gradient만으로 구현 오류라 하지 않는다. 초기값·FC2·ResBlock·앞단 전달을 순서대로 검사한다 [S1 §4.2].

---

## 3. 모든 성능 후보의 방법 계약

### 3.1 Native branch

```text
P_original + LRMS M
    ├─ L_A = verified dataset LMS / phase-consistent upsampling
    │       └─ RGB 값 평균 → Scharr abs-XY
    ├─ P → Scharr abs-XY
    │       └─ integer raw-L1 EdgeMatch → b0 (no gradient)
    ├─ P_pre = W(P_original,b0) → residual CNN(P_pre,L_A) → Δu0
    └─ u0=b0+Δu0 → W(P_original,u0) → Teacher + bicubic(M) → Y_hat
```

- A는 **C-band L_A 전체**를 입력받는다. RGB 평균은 EdgeMatch/L_abs의 scalar projection용이지, A 입력을 3채널/1채널로 바꾸는 지시가 아니다.
- PAN/LMS stem·joint의 기존 GN, 공통 64ch pre-LN ResBlock, GAP, FC64→32→2를 유지한다.
- Teacher U-Net 입력은 `PAN(1)+B_M(C)`이며 base도 B_M다. width/depth 등은 서버의 승인된 Teacher 구조를 pin하고 바꾸지 않는다. 이전 W112/D[1,2,3]가 현재 선택과 일치하는지는 서버에서 확인한다.
- 추가 LPAN/HPAN, attention, scene head, sensor-conditioned embedding을 넣지 않는다.
- `P_pre`는 A 입력용이고, T에는 **원본에서 한 번 warp한 PAN만** 공급한다.

### 3.2 Shifted branch

`P_eps=sg(W(P,ε))`, `b_eps=EdgeMatch(P_eps,L_A)`, `Δu_eps=A_res(W(P_eps,b_eps),L_A)`, `u_eps=b_eps+Δu_eps`.

`L_eps=mean_{sample,dy/dx}|u_eps+ε-sg(u0)|`.

같은 M/L_A/Y, 같은 학습된 A를 사용한다. shifted PAN을 T에 넣지 않는다. EdgeMatch와 ε 생성은 비학습이고, `sg(u0)`는 ε target 경로에만 적용한다.

### 3.3 Objective와 튜닝 권한

`L = L_rec_RR + λ_abs L_abs_RR + λ_eps L_eps_RR`.

| 항목 | 고정/변경 |
|---|---|
| L_rec | 전 밴드 HRMS L1, 계수 1 고정 |
| L_abs | RGB 평균 후 동일 Scharr absolute-XY L1, loss 종류 고정 |
| L_eps | 총 shift의 RR consistency, 매 update, radius 2 고정 |
| λ_abs, λ_eps | §6의 **양수 scalar** 범위에서 변경 |
| optimizer | §7의 A/U LR 및 WD profile만 변경 |
| branch/architecture | 기본 full-method 후보에서는 제거·추가·freeze 없음 |

L_rec는 A/T, L_abs와 L_eps는 A만 update한다. 원문의 coefficient 1/0.001은 초기값이므로 scalar sweep은 사용자 허용 범위다. Scharr scale·intensity scaling·ROI를 loss 계수와 동시에 바꾸지 않는다.

---

## 4. 원문의 열린 구현 항목 — 이번 실행용 제안과 사전 확정

이 절은 [S1]에서 확정되지 않은 값의 **실행 제안**이다. 서버에 이미 승인된 값이 있으면 먼저 대조하고 `baseline_resolved.json`에 실제 값을 고정한다. 변경이 필요하면 공식 성능 결과를 보기 **전**에 이유를 남긴다. 값이 합의되지 않았다는 이유로 존재하지 않는 scene/FR 자산을 다시 요구하지 않는다.

### 4.1 EdgeMatch 후보 범위

**첫 구현 검증 제안:** square integer grid `dy,dx∈[-4,4]`, 총 **81개**. train/RR/FR 모두 **현재 PAN px의 같은 반경**을 사용한다. 이것은 원문 확정값이나 센서의 물리 misalignment 상한이 아니다. `R_edge=4`와 `R_eps=2`는 별개 변수다.

공식 RR/FR 성능 평가를 시작하기 전에 training-only 입력·known shift·유효영역·실행시간으로 범위가 작다고 확인되면, `R_edge=6` 또는 `8`로 **한 번의 사전 프로토콜 결정**을 할 수 있다. 6/8은 각각 169/289 후보다. 실제 채택값과 근거를 고정하고 모든 scalar 비교군에 적용한다. **FR HQNR를 보고 4/6/8 중 좋은 값을 고르거나 장면별로 바꾸지 않는다.** train-only 근거가 없으면 초기 제안 4를 유지하고 coverage 한계를 보고한다.

실험 도중 범위를 변경하려면 새 `edge_protocol_revision`과 새 campaign/subcampaign으로 나눠 비교한다. 기존 결과를 같은 scalar sweep으로 합치지 않는다. 사전 범위 검사에서 boundary hit가 많다는 사실만으로 정확한 변위를 안다고 주장하지 않는다.

### 4.2 EdgeMatch score의 공통 support

**실행 제안:** `m_edge=R_edge+ceil(R_eps)+2+1=R_edge+5`, `Ω_edge=[m_edge:H-m_edge, m_edge:W-m_edge]`. 2는 기존 bicubic 4-tap support의 보수적 여유, 1은 Scharr radius다. native와 shifted branch 모두 이 동일한 고정 규칙을 사용한다. 모든 후보가 **동일한 영역·pixel 수·두 방향 채널**에서 평균되며, 64²와 R=4이면 46²의 score support다. 정확한 sampler stencil로 이 여유가 충분한지 unit test한다.

Edge map을 먼저 계산한 뒤 integer slicing할 수 있다. 검색 후보별로 유효 crop 크기를 달리하여 평균하거나, padding을 score에 섞거나, candidate별 별도 normalization을 적용하지 않는다. **Scharr reflection 경계뿐 아니라 synthetic P_eps를 만들 때 원본 밖에서 채워진 값까지 score에서 배제**한다. 위 고정 여유는 이 목적의 제안이며, ε별 가변 ROI로 정답 단서를 만들지 않는다. RR/FR 추론에도 확정된 같은 score-support 규칙을 적용한다. 양·음 shift의 index sign은 W 규약과 일치해야 한다.

### 4.3 동점·빈 영역·저텍스처

**실행 제안:** finite score의 최소값을 먼저 선택하고, 정확한 동점이면 `||δ||² 최소 → |dy|+|dx| 최소 → dy → dx`의 고정 순서를 쓴다. 완전히 평탄하고 zero가 tied minimum에 포함되면 zero가 선택된다. 임의 epsilon을 더해 score 순위를 바꾸는 방식은 사용하지 않는다.

비영 방향의 완전 대칭 tie에서는 이 결정 규칙이 flip/rotation에 대해 유일하게 equivariant일 수 없으므로 tie 여부를 로그로 남긴다. equivariance unit test는 unique-minimum 예제와 tie 예제를 분리한다.

- 저텍스처·작은 score gap은 **진단 label**이다. 새 confidence threshold로 b를 바꾸거나 loss를 낮추지 않는다.
- no valid Ω / nonfinite inputs·scores는 `FAILED_INPUT_OR_SUPPORT`, 학습 중단·숫자 기록. 유효하지 않은 sample/평가 장면을 조용히 제외하지 않는다.
- 최적값이 검색 경계에 있으면 `EDGE_BOUNDARY_HIT`를 기록하되, 자동으로 값 축소·zero 대체·test 장면 제외를 하지 않는다.
- 최적 score·zero score·2위와의 gap·score range·두 영상의 edge 평균을 기록한다. 이 값은 물리 confidence 정답이 아니다.

### 4.4 L_abs support

원문은 interior-support의 필요를 명시하지만 정확한 margin은 열린 항목이다. **이번 실행 제안은 RR train에서 고정 `m_abs=16`, 32² 내부에 대한 batch×pixel×2-direction mean**이다. m_edge와 m_abs는 역할이 다르다. 서버의 이미 승인된 고정 support가 있으면 첫 공식 run 전에 그 정의로 확정하고 전체 sweep에서 유지한다.

Scharr neighborhood와 최종 bicubic sampling support가 이 고정 영역에서 원본 안에 있는지 확인한다. **u에 따라 loss 영역을 점점 줄여 큰 이동이 loss를 회피하도록 만들지 않는다.** 영역을 보장하지 못하면 sample 삭제·zero loss·output clamp 대신 수치 실패를 기록한다. L_rec와 공식 평가의 기존 support는 변경하지 않는다.

`FULL_FRAME_LABS` 구현과 이 interior proposal은 다른 objective다. 구현 담당자는 어느 쪽인지 `absolute_support_policy`에 명확히 기록해야 한다. 이번 기본 권장은 interior이며, 이를 원문의 이미 확정된 16px라고 표기하지 않는다.

### 4.5 LMS·augmentation·cache

기존 저장 lms가 수치 검증되면 직접 사용한다. 없다면 데이터셋과 phase가 일치한다고 확인된 연산자로 만든다. **LMS 생성 → P/LMS/GT의 동일 HR flip/rotation/crop** 순서를 유지한다. LRMS M에도 같은 flip/rotation과 4:1 격자에 대응하는 crop을 적용한다. 임의 HR crop은 LR 정수 격자 대응을 보장해야 하며 기본 학습은 주어진64² patch를 유지한다. T용 bicubic base는 별도 경로다. CNN의 znorm은 **P_pre/L_A의 CNN용 복사본에만** 적용하며, 원본 PAN·EdgeMatch edge·최종 warp의 radiometry를 바꾸지 않는다.

원문 band 순서가 현재 파일과 일치할 때 RGB index(0-based)는 WV3 `(4,2,1)`, QB/GF2 `(2,1,0)`다. 실제 H5/loader와 검사 후 확정한다. 데이터 intensity scaling은 기존 방식으로 고정한다.

Native b0를 cache할 수 있는 것은 **실제 들어가는 augmentation view가 정확히 같을 때**다. key는 dataset/input hash, sample index, 모든 flip/rot/crop, LMS operator/phase, shape, edge protocol hash를 포함한다. canonical b를 회전만 해 쓰는 최적화는 tie·boundary까지 동등성이 검증되지 않으면 사용하지 않는다.

**b_eps는 매번 현재 P_eps에서 계산한다.** b0 cache를 복사하거나 b0-ε로 치환하지 않는다. shifted cache가 있다면 정확한 ε·input bytes·protocol이 같은 경우만 허용한다. cache에는 비학습 숫자만 저장하며 모델 weight를 숨겨 넣지 않는다. 공식 cost는 cache warm 여부를 분리한다.

---

## 5. 48시간 예산과 기동 조건

### 5.1 독립 clock

서버별 `t0`를 새 방법의 runnable 구현으로 preflight를 시작할 때 기록한다. 그 뒤 **preflight·학습·평가·진단·통과 모델 저장·정성 export 모두 48시간에 포함**한다. 구현 개발 시간까지 완료됐다고 가정하지 않으며, 이 MD는 코드를 배포하거나 프로세스를 시작하지 않는다.

| 경과 시간 | 단계 | 우선 작업 |
|---|---|---|
| 0–2h | P0 / 검증·실측 | 새 데이터 요구 확인, EdgeMatch/gradient 계약, 300–500 update disposable timing |
| 2–22h | A / 계수 탐색 | EF00 먼저 H50 완료; 서버별 우선 계수 탐색 |
| 22–36h | B / 최적화·수렴 | 상위 최대 2개 recipe의 ET profile 또는 H100 |
| 36–46h | C / 확인 | 우수 recipe 1개의 R2·R3를 우선; 목표 미달이면 가장 근접한 recipe도 반복 대상 |
| 46–48h | 마감 | 신규 학습 금지, 마지막 평가·통과 artifact/qual·로컬/시트 대조 |

단계 시각은 **새 run의 admission 경계**다. 정상 실행 중인 run을 22/36h가 됐다는 이유만으로 버리지 않는다. 반대로 긴 run을 뒤늦게 시작해 마감 검증 시간을 없애지 않는다. 실제 종료 예상이 맞지 않으면 `TIMEBOX_STOP`으로 기록한다.

### 5.2 실제 속도 기반 admission

다음 비용을 **새 pipeline 자체**에서 측정한다: native/shifted EdgeMatch, A/T forward/backward, 전체 RR·FR pair 평가, lean 진단, qualifying save/qual. 과거 TA2 또는 다른 GPU의 시간으로 실행 수를 보장하지 않는다.

`ETA_run = 1.2 × (updates×t_update + N_eval×t_pair_eval + t_numeric_probe + t_expected_qual_reserve)`.

warmup/compile을 별도 기록하고 t_update는 실제 안정 구간의 robust 측정치로 정한다. `remaining_wall >= ETA_run + closeout_reserve`일 때만 시작한다. closeout은 최소 마지막 2h, artifact backlog가 있으면 더 크게 잡는다. EF00 완료 후 t50 실측으로 다시 계산한다.

**우수 설정의 추가 두 번을 위한 시간을 매 run admission 시 예약**한다. 기본 예약은 `2×ETA_selected_recipe`. 아직 selected recipe가 없으면 같은 horizon의 추정 ETA를 사용한다. 36h에 반복을 시작하면 부족할 것으로 예측되면 계수 탐색을 일찍 멈추고 확인 단계로 전환한다.

예산상 3회가 불가능하면 완료된 n만 보고한다. 새로운 seed를 더 뽑아 성공률을 만들지 않는다. 모든 20×9×3 조합의 완주는 계획하지 않는다.

### 5.3 진짜 blocking 항목

- 기존 RR train/validation 및 RR/FR evaluation 파일의 실제 존재, dataset/evaluator identity.
- PAN:MS=4:1, band/RGB/GT 정의, native LMS phase, T base 분리.
- sign·integer matching·공통 support·초기 `Δu=0`·새 ResBlock/head·one-shot 최종 warp.
- ε 재matching과 loss gradient route.
- 미달 weight/image 미저장 및 local-first 수치 기록 경로.

**blocking에서 제외:** native FR **학습** 데이터 부재, 부모 scene 매핑 부재, b_s/b_global cache 부재. full-method flag는 새 정의로 검사한다.

동일 영상 synthetic unique-texture 검사, ε=0 검사, fractional known shift, 64/256/512 shape·좌표 ramp, cache on/off 동일 입력 검사를 수행한다. 현저한 cross-modal matching 실패는 설계 관측이지 즉시 코딩 오류라고 판정하지 않는다. GPU 연산 tolerance는 dtype/실측에 맞춰 기록한다.

---

## 6. EF 계열 — loss 계수 탐색

**공통:** fresh A/T, H50, R1, 동일 baseline optimizer·sample/augmentation/ε stream. λ_rec=1. 모든 후보는 EdgeMatch+residual+RR rec/abs/ε의 **full method**다.

| Case | λ_eps | λ_abs | 목적 |
|---|---:|---:|---|
| **EF00** | 0.001 | 1 | 새 명세의 초기 full-method |
| **EF01** | 0.0003 | 1 | epsilon 0.3배 |
| **EF02** | 0.0001 | 1 | epsilon 0.1배 |
| **EF03** | 0.003 | 1 | epsilon 3배 |
| **EF04** | 0.001 | 0.3 | absolute 구조항 0.3배 |
| **EF05** | 0.001 | 0.1 | absolute 구조항 0.1배 |
| **EF06** | 0.001 | 3 | absolute 구조항 3배 |
| **EF07** | 0.0003 | 0.3 | 약한 epsilon + 약한 absolute |
| **EF08** | 0.0001 | 0.3 | 더 약한 epsilon + 약한 absolute |
| **EF09** | 0.003 | 0.3 | 강한 epsilon + 약한 absolute |
| **EF10** | 0.0003 | 0.1 | 약한 epsilon + 작은 absolute |
| **EF11** | 0.0001 | 0.1 | 두 보조항 모두 0.1배 |
| **EF12** | 0.003 | 0.1 | 강한 epsilon + 작은 absolute |
| **EF13** | 0.0003 | 3 | 약한 epsilon + 강한 absolute |
| **EF14** | 0.0001 | 3 | 더 약한 epsilon + 강한 absolute |
| **EF15** | 0.003 | 3 | 두 보조항 모두 3배 |

### 6.1 약한 신호 범위 — 조건부 확장 4개

새 EdgeMatch가 ε 이동을 이미 대부분 처리할 수 있어 기존 범위보다 약한 ε도 사전에 등록한다. 다만 기본 16개와 모두 곱해서 실행하지 않는다.

| Case | λ_eps | λ_abs | 활성화할 상황 |
|---|---:|---:|---|
| **EF16** | 1e-05 | 0.1 | 조건부: EdgeMatch가 상대 이동을 대부분 처리할 때 |
| **EF17** | 1e-05 | 1 | 조건부: epsilon만 크게 낮추고 absolute 유지 |
| **EF18** | 0.0001 | 0.03 | 조건부: raw-gradient L1 영향 추가 완화 |
| **EF19** | 0.001 | 0.03 | 조건부: epsilon 유지, absolute만 크게 완화 |

EF16/17: total gain이 높지만 대부분 g_b이고 ε gradient와 native 복원 경향의 충돌이 관측될 때 우선 검토. EF18/19: raw/weighted absolute gradient의 영향과 품질 교환이 커 더 약한 구조항 비교가 필요할 때 검토. **관측 조건·채택 이유를 decision record에 남긴다.** zero coefficient, 자동 weight normalization, adaptive gate는 없다.

### 6.2 기본 실행 우선순위

| 서버 | 우선 6개 | 나머지 10개 순서 |
|---|---|---|
| s1 / WV3 | EF00 → EF01 → EF04 → EF07 → EF02 → EF06 | EF08 → EF05 → EF10 → EF11 → EF13 → EF14 → EF03 → EF09 → EF12 → EF15 |
| s2 / QB | EF00 → EF02 → EF08 → EF11 → EF01 → EF07 | EF04 → EF10 → EF05 → EF06 → EF14 → EF13 → EF03 → EF09 → EF12 → EF15 |
| s3 / GF2 | EF00 → EF06 → EF13 → EF01 → EF14 → EF04 | EF07 → EF02 → EF08 → EF05 → EF10 → EF11 → EF03 → EF15 → EF09 → EF12 |

첫 여섯 개도 완료 보장이 아니라 admission 우선순위다. EF00/R1은 수치 실패·시간 제한 외에는 50K를 우선 완주한다. 그 외에는 §10 pruning만 허용한다.

WV3는 아직 과거 반복 근거가 부족하므로 기본·단일축 비교를 먼저 본다. QB는 약한 ε를 빨리 보되, **초기 b가 잘못됐다면 residual recovery 속도도 확인**한다. GF2는 absolute 강도·RR 수렴을 우선 보지만 과거 G05의 NCC 계수를 이번 λ_abs와 동일시하지 않는다.

---

## 7. ET 계열 — 선택 recipe의 optimizer 튜닝

서버가 승인한 `lr_A0, lr_U0, WD0, optimizer, betas/eps, warmup, min_lr, effective_batch, precision`을 먼저 pin한다. 값이 없을 때의 **초기 제안**은 AdamW, lr_U=1e-4, lr_A=1e-5, WD=0.01, betas=(0.9,0.999), eps=1e-8, effective RR batch=48, warmup100, cosine, FP32/AMP off/TF32 off다. 이는 현재 서버의 확인값이 아니라 [S2] 및 과거 recipe를 참고한 운영 제안이다. 선택 후 모든 비교군에서 고정한다.

실제 Teacher 구조/기존 승인 설정과 충돌하면 구현 담당자는 baseline 변경 여부와 이유를 먼저 남긴다. microbatch/accumulation은 메모리에 맞게 바꾸되 effective batch와 loss mean은 고정한다.

Phase A의 **최대 2개 loss recipe**를 anchor로 선택한다. Anchor는 config이며 미달 weight가 아니다. 아래 배율은 baseline lr/WD에 적용하고, 앞 ET case의 배율을 누적하지 않는다.

| Profile | lr_A/lr_A0 | lr_U/lr_U0 | WD/WD0 | 검정 목적 |
|---|---:|---:|---:|---|
| **ET00** | ×0.1 | ×1 | ×1 | 아주 느린 residual A |
| **ET01** | ×0.3 | ×1 | ×1 | 완만한 residual A |
| **ET02** | ×3 | ×1 | ×1 | 잘못된 초기 b의 복구가 느린지 확인 |
| **ET03** | ×1 | ×0.5 | ×1 | U learning rate 감소 |
| **ET04** | ×1 | ×1.5 | ×1 | U learning rate 증가 |
| **ET05** | ×1 | ×1 | ×0.1 | weight decay 감소 |
| **ET06** | ×1 | ×1 | ×0 | weight decay 제거 |
| **ET07** | ×0.3 | ×1.5 | ×1 | 느린 A + 빠른 U |
| **ET08** | ×3 | ×0.5 | ×1 | 빠른 A + 느린 U; 초기 prior 복구 대조 |

`WD0=0`이면 ET05/06는 변경 없는 중복이므로 실행하지 않는다. 결합 profile ET07/08의 결과를 A 하나의 순수 효과로 설명하지 않는다. optimizer family·betas·warmup·batch·backbone은 별도 sweep하지 않는다.

### 7.1 관측에 따른 profile 선택

| 관측 | 우선 profile | 주의 |
|---|---|---|
| `u≈b`가 오래 유지되고 prior-only가 zero보다 나쁘며 current가 복구하지 못함 | **ET02 → ET08** | 빠른 residual 복구 가설; A freeze/작은 LR만 고집하지 않음 |
| prior-only보다 current에서 native/RR·FR가 악화되고 residual norm이 증가 | **ET01 → ET00** | 작은 LR가 좋다는 확정은 아님 |
| HQNR는 목표에 가깝고 RR 오차가 계속 감소, U 수렴이 병목 | **ET04 / ET05 또는 H100** | GF2에서 우선 점검; target 통과 보장 없음 |
| 학습 불안정·loss 급변, finite하지만 진동 | **ET03** | 새 clipping/정규화를 슬쩍 넣지 않음 |
| 특별한 기전 단서가 없음 | s1: ET01→ET03, s2: ET01→ET02, s3: ET04→ET05 | 대비 방향을 하나씩 검정 |

profile 선택은 기록된 숫자에 근거한 **개발 탐색**이다. 모든 scalar 실험 중간에 학습률을 임의로 바꾸지 않는다. 변경 config는 fresh run으로 시작한다.

---

## 8. 학습 길이와 반복

| Schedule | 목표 updates | 시작 방식 | 우선순위 |
|---|---:|---|---|
| **H50** | 50,000 | fresh, cosine horizon50K | 기본 |
| **H100** | 100,000 | fresh, 처음부터 horizon100K | 유망 recipe의 RR 추가 수렴 |
| **H150** | 150,000 | fresh, 처음부터 horizon150K | H100 근거·잔여시간이 모두 있을 때만 |

H50 미달 weight를 저장해 뒀다가 H100으로 이어가는 계획이 아니다. H100/H150은 config를 재사용해 처음부터 학습한다. H100의 중간50K와 H50 최종점은 scheduler 이력이 다르므로 동일 예산 인과 대조로 합치지 않는다.

긴 학습은 **한 지표가 이미 목표권이고 다른 지표가 후기까지 개선되며, 앞 지표가 지속 붕괴하지 않을 때** 우선한다. FR가 일찍 최대이고 이후 계속 악화되면 무조건 H100/H150을 시작하지 않는다. training midpoint A freeze, ε ramp, direct PAN reconstruction 추가, PANMIX는 이번 full-method 탐색에 포함하지 않는다.

### 8.1 Seed

| 서버 / dataset | R1 탐색 | R2 확인 | R3 확인 |
|---|---:|---:|---:|
| s1 / WV3 | 261013101 | 261013102 | 261013103 |
| s2 / QB | 261013201 | 261013202 | 261013203 |
| s3 / GF2 | 261013301 | 261013302 | 261013303 |

이 seed는 이전 48h 운영 계획을 상속한다. 새 campaign/방법 revision이므로 과거 실행을 이번 반복으로 세지 않는다. probe seed는 261013900, 학습과 분리한다.

같은 dataset/replica의 initial T/A는 **독립 RNG로 생성하고 RAM에서 hash만 기록**한다. 초기 weight snapshot을 디스크에 저장하지 않는다. 데이터/augmentation/ε/진단 RNG도 분리하고, 전체 재생 규칙을 기록한다. Python process의 비고정 `hash()`로 seed를 파생하지 않는다.

우수 **recipe 1개**의 R1/R2/R3를 먼저 확보한다. 성공 R1만 있고 R2/3가 미달이어도 그 수치와 성공률 분모를 유지한다. 기술적 재시도는 같은 seed의 새 attempt이지 새 독립 반복이 아니다. 이틀 안에 3회가 끝나지 않으면 n=1/2로 보고한다.

---

## 9. 공식 평가와 목표 통과 snapshot

### 9.1 이전 48시간 계획의 cadence 유지

`E(H) = {1,000} ∪ {2,000,4,000,…,H} ∪ {25,000} ∪ {H}`.

| Horizon | 예정 전체 RR/FR 평가점 |
|---|---:|
| H50 | **27** |
| H100 | **52** |
| H150 | **77** |

동일 step 중복은 한 번 평가한다. pruning/deadline 최종 추가점이 있으면 별도 count로 기록한다. **TA2의 HQNR_MAX50 label을 재사용하지 않는다.** 최대 관측 HQNR에는 실제 candidate 수와 `OBSERVED_EVAL_GRID`를 붙인다. 평가 간격 사이에 잠깐 목표를 넘었더라도 관측하지 못했다면 저장/달성했다고 주장하지 않는다.

성능이 근접했다고 사후 특정 run만 평가 간격을 촘촘히 바꾸지 않는다. 필요하면 별도 사전 등록 evaluation protocol의 새 run이다.

### 9.2 평가 규약

기존 고정 RR/FR 목록 전체를 동일 evaluator revision으로 계산한다. FR는 **original PAN / native LMS reference, 기존 full-frame HQNR**다. RR의 crop·DN scaling·output clipping·밴드 순서를 그대로 pin한다. aligned-PAN reference HQNR나 임의 border masking으로 목표를 통과시키지 않는다.

한 snapshot에서 모든 지표 `HQNR, D_s, D_lambda, JQM, ERGAS, SCC, SAM, PSNR, SSIM, Q4/Q8, RMSE, CC`와 나머지 지원 metric을 기록한다. 미지원 값은 공란/사유이며 0으로 대체하지 않는다. JQM·FLOPs 등의 variant를 보존한다. HQNR와 ERGAS가 다른 input/reference 정책에서 나온 값을 합치지 않는다.

### 9.3 In-memory evaluate → gate → save

1. optimizer update 경계에서 현재 A/T를 고정하고 canonical `model_state_sha256`을 계산한다.
2. mode/RNG를 보존한 뒤 RR 전체와 FR 전체를 평가한다. 두 평가 사이에 update를 하지 않는다.
3. prediction은 metric 계산용 RAM에서만 사용하고 장면별 수치가 끝나면 해제한다. 미달 여부 확인용 weight/이미지 임시 파일은 만들지 않는다.
4. completeness·finite·same-state와 두 threshold를 검증한다. **HQNR만 실패했다고 RR 평가를 생략하지 않으며, 그 반대도 마찬가지**다.
5. 모든 metric과 gate 결과를 로컬 durable record로 먼저 쓴다.
6. **full-method 통과 snapshot만** A/T weight를 atomic 저장·load/hash 확인하고 같은 snapshot으로 정성 결과를 만든다.
7. mode/RNG 복원 후 예정 horizon까지 학습한다. 목표 통과가 곧 학습 종료 조건은 아니다.

일부 장면 실패 또는 평가 중 crash면 `EVAL_INCOMPLETE`, 통과가 아니다. state가 RAM에 그대로 있으면 누락 evaluation을 재시도할 수 있다. state를 잃었고 저장 자격이 없었다면 정확한 snapshot을 나중에 복구할 수 없다.

### 9.4 대표 Teacher

통과한 **정상 native full pipeline** 후보 중 `HQNR↓정렬(큰 값 우선) → ERGAS↑정렬(작은 값 우선) → step 작은 순 → run_id`로 결정한다. 앞 항목의 정확한 동점에만 다음을 적용한다. 통과 후보 중 ERGAS 최소인 예비 Teacher도 따로 표시한다.

모든 통과 snapshot을 보존하고 더 좋은 후보가 생겼다고 자동 삭제하지 않는다. 동일 state/file alias는 중복 저장 없이 참조할 수 있다. 성공이 없으면 **NO_TARGET_QUALIFIED_TEACHER**가 결론이다. 미달의 최고 관측값은 숫자만 남고 재평가용 모델은 없다.

---

## 10. 탐색 선택·조기 중단 — 미달 미저장과 양립하도록

한 snapshot의 H,E에 대해

`v_H=max(0,(H*-H)/H*)`, `v_E=max(0,(E-E*)/E*)`, `D_inf=max(v_H,v_E)`, `D_1=v_H+v_E`.

통과가 없으면 **D_inf → D_1 작은 순 → HQNR 큰 순 → ERGAS 작은 순**으로 다음 recipe를 고른다. Pareto 표도 함께 남겨 한쪽 개선 방향을 놓치지 않는다. 이 점수는 개발 탐색 기준이지 새로운 loss·유의성 검정이 아니다. 다른 checkpoint의 최고 H/최저 E를 합성하지 않는다.

한 sensor에서 한쪽 목표를 충족한 recipe가 여러 개면, trade-off가 다른 최대 두 개 anchor를 선택할 수 있다. 선택 이유, 근거 snapshot, 실제 config, 남은 시간/ETA를 `DECISION` record로 남긴다.

**Pruning:** EF00/R1과 R2/R3 확인 run은 성능상 pruning 없이 endpoint를 우선한다. 다른 탐색 run은 최소25K 관측 후, 같은 dataset/horizon/R1의 두 선행 조건이 공통 평가점 세 개에서 H≥현재·E≤현재로 일관 우세하며 현재 target-gap의 개선도 정체된 경우에만 `PRUNED_DOMINATED`로 중단할 수 있다. 공동 관측점/대조가 없으면 적용하지 않는다. 정체 판정의 수치 tolerance와 최근 window는 preflight에서 고정해 로그에 남긴다. 이 휴리스틱은 후기 회복 불가능의 증명이 아니다.

RAM의 live run을 계속 학습하는 것은 가능하지만, 미달 checkpoint를 저장해 두었다가 되살리는 successive-halving은 하지 않는다. 종료한 미달 anchor를 다시 시도하려면 **fresh replay 비용을 포함**한다.

운영 장애 `FAILED_*`, 시간 제한 `TIMEBOX_STOP`, 과학적 성능 pruning을 분리한다. 중단된 30K를 완료50K로 평균내지 않는다. 미실행 case는 `NOT_STARTED_BUDGET` 또는 `NOT_TRIGGERED`다.

---

## 11. 숫자 진단 — EdgeMatch와 CNN의 역할을 분리

**성능 확보가 주목적이므로 매 checkpoint마다 TA2 전체 진단을 실행하지 않는다.** 그러나 미달 weights를 버리므로 필요한 핵심 숫자는 live model이 있을 때 반드시 남긴다.

### 11.1 실행 수준

| 수준 | 시점/대상 | 기록 |
|---|---|---|
| TRACE | 매 update 로컬 scalar, 100-update 구간 요약 | rec/abs/eps raw·weighted, LR, b/Δu/u norm, support, 시간 |
| LEAN | **10K, 25K, 예정 endpoint, pruning/deadline 마지막 상태** | 고정 소규모 response·geometry·gradient·current/prior/zero |
| QUALIFIED | 통과 snapshot | RR/FR 전체 per-image vector/metric, 동일 모델 정성, cold cost |
| EXTENDED | 시간 여유의 대표 통과 모델 | 20장 전체 response·size·추가 swap; 기본 score와 분리 |

TRACE의 update별 scalar는 로컬 파일에 모으고 시트에는 구간 요약·상태 전이만 올린다. LEAN 제안 표본은 canonical train 8 + validation 8 + RR evaluation 4 + FR evaluation 4이며 고정 ID를 성능과 무관하게 정한다. 없는 validation을 평가 test로 대체해 학습하지 않는다. probe는 모두 입력/gradient 측정용이며 update를 만들지 않는다.

### 11.2 Epsilon response: 세 성분을 같은 ε 묶음에서 계산

`db=b_eps-b0`, `dr=Δu_eps-Δu0`, `du=u_eps-u0=db+dr`.

```text
g_edge     = -Σ(db·ε) / Σ||ε||²
g_residual = -Σ(dr·ε) / Σ||ε||²
g_total    = -Σ(du·ε) / Σ||ε||² = g_edge + g_residual
r_edge     = db + ε
r_total    = du + ε
EPE_edge   = mean ||r_edge||
EPE_total  = mean ||r_total||
MAE_edge / MAE_total = componentwise absolute mean
```

- 위 gain은 **같은 samples·offsets·denominator**에서 계산해야 합이 맞는다. g_residual을 별도로 정규화하여 더하지 않는다.
- `EPE_edge−EPE_total`을 CNN이 coarse matching 이후 남은 이동 반응을 얼마나 줄였는지 보는 보조값으로 기록한다.
- `g_residual=0` 자체는 실패가 아니다. `g_edge≈1`, low residual error이면 추가 반응이 필요 없을 수 있다.
- integer-only와 fractional ε를 분리한다. AXIS16(0.25/0.5/1/2 px ×4축)을 기본 사용하고, DIAG16·held-disk64는 추가 예산에서 계산한다. OOD4/8을 in-range와 합치지 않는다.
- 초기0 / 10K / 25K / endpoint의 edge-only와 total을 비교한다. 초기 EdgeMatch 성능을 CNN이 학습해서 얻은 값으로 보고하지 않는다.
- EF00와 대표 recipe의 LEAN에서는 normal/clean-support 진단을 구분해 synthetic border를 제외해도 반응이 남는지 확인한다. 이 진단에서 ROI를 바꾼 값은 공식 native score나 학습 설정으로 대체하지 않고 domain을 명시한다.

### 11.3 Native 정합·residual 역할

각 probe image에서 `b0`, `Δu0`, `u0`의 dy/dx, norm, split별 평균·SD·p50/p95, `cos(b0,Δu0)`와 zero-vector NA를 기록한다. prior를 되돌리는 residual을 자동 이상치로 제거하지 않는다.

같은 interior에서 다음 두 구조 discrepancy를 **원본 P / prior-only W(P,b) / final W(P,u)**에 대해 각각 계산한다.

- RGB-LMS와의 raw Scharr L1: EdgeMatch 목적과 같은 계열이므로 독립 정확도 증거가 아니다.
- RR GT RGB와의 raw Scharr L1: absolute 지도 목적이며 물리 displacement GT가 아니다.

RGB-LMS/GT 평균 edge amplitude와 PAN amplitude, boundary-hit·tie·flat score 비율, actual warp valid fraction을 남긴다. 독립 proxy가 기존에 검증되어 있으면 평가 전용으로 추가할 수 있지만 새 matching/normalization을 학습에 넣지 않는다.

### 11.4 같은 Teacher에서 correction만 바꾸는 numeric 대조

고정 소규모 RR/FR probe에서 **같은 frozen T**로

1. `CURRENT`: u=b+Δu,
2. `PRIOR_ONLY`: u=b,
3. `ZERO`: u=0,

를 비교한다. optional `SHUFFLED_RESIDUAL`: b_i는 유지하고 Δu만 고정 순환 교환하여 장면 대응성을 검사한다. 현재 full output을 만들기 위한 구조는 변경하지 않는다.

이 결과는 **추론 개입이지 그 조건으로 재학습한 ablation이 아니다**. 미달 full model에서 ZERO나 PRIOR_ONLY가 목표를 통과해도 **그 weight를 정식 full-method 통과로 저장하지 않는다**. 해당 경로는 DIAG_ONLY 수치다. 큰 대조군 학습은 이번 48h 필수 queue에 넣지 않는다.

### 11.5 Gradient·초기화

고정 train effective batch(제안 4 batches)에서 A에 대한 `g_rec`, `λ_abs g_abs`, `λ_eps g_eps`의 norm/ratio/cosine을 whole/fc2/fc1/body/stem으로 나눈다. 실제 optimizer update norm은 training loop에서 별도로 기록하며 gradient norm×LR로 추정해 대체하지 않는다.

초기0/1/10/100 update에서는 가벼운 한 batch trace로 FC2와 zero-init ResBlock의 단계별 전달을 확인한다. raw ε vector loss가 작아져 gradient가 적은 상황과 구현상 graph가 끊긴 상황을 구분한다. native rec/abs에는 stop-gradient를 걸지 않는다.

### 11.6 64→512와 cost

대표 모델의 size probe는 같은 FR input에서 resize 없는64/128/256/512 crop으로 g_edge/g_residual/g_total을 각각 비교한다. resize한 영상은 별도 domain으로 표기한다. 큰 배열을 받는 것과 물리 GSD 일반화는 동일하지 않다.

`Infer(ms)`는 **L_A 준비 + EdgeMatch + P_pre warp + CNN residual + 최종 warp + T**의 end-to-end 측정을 기본으로 한다. 저장된 LMS를 입력으로 쓰면 그 I/O 포함/제외 범위를 명시한다. EdgeMatch-only, A-only, T-only, transfer, warm-cache 비용도 분리한다. RR256/FR512, batch1, 장치·precision·warmup/repeats·CPU threads를 고정한다. global timing은 비동기 연산 종료를 동기화하여 재고, branch 합만으로 total을 만들지 않는다.

Params는 학습 가능한 A+T, EdgeMatch는 parameter 0이다. FLOPs 계산에서 integer search·abs/reduction 등이 누락되면 `partial coverage`로 표기한다. **새 모델의 Params/FLOPs에 옛 Aligner 숫자를 복사하지 않는다.**

---

## 12. Weight·정성 산출물 — STRICT_TARGET_ONLY

| 산출물 | 미달/미완료 snapshot | 정상 full-method 두 목표 통과 |
|---|---|---|
| A/T weight | **저장 금지** | 저장·hash/load 검증 |
| last/resume/임시 평가 weight/초기화 snapshot | **저장 금지** | 해당 통과 시점의 state만 가능 |
| optimizer dump·crash dump의 모델 복구 가능 tensor | **저장 금지** | 같은 통과 시점만 가능 |
| PNG/TIFF/MAT/NPY reconstruction·overlay·heatmap/TensorBoard image | **저장 금지** | 같은 checkpoint로 생성 |
| config/seed/RNG 규칙·dataset/protocol·metric·scalar vector 진단 | **모두 저장** | **모두 저장** |
| 시트 수치 업로드 | **전부 수행** | **전부 수행** |

RAM의 현재 training state는 유지할 수 있다. 하지만 미달 프로세스가 종료되면 **exact resume는 불가능**하며 같은 config/seed의 fresh attempt로 다시 시작한다. 자동 rolling checkpoint 예외는 없다. 수치·seed만 남았다고 특정 GPU 학습 weight가 bitwise 복구된다고 설명하지 않는다.

통과 모델은 A/T와 **EdgeMatch/LMS/warp/RGB/normalization/ROI/architecture 설정 및 실행 source snapshot**을 함께 보존한다. `b`는 입력별로 계산되므로 **scene map·scene prior weight/cache는 필요하지 않다**. 실제 데이터·dependency 위치와 hash를 재평가 가능하게 남긴다.

통과 직후 weight 저장을 우선하고, 정성은 저장된 동일 snapshot에서 RR/FR 전체 evaluation output과 미리 정한 RGB/ROI/scale로 생성한다. output은 all-band 원자료와 display RGB를 구분한다. 같은 state alias는 한 번 export해 공유할 수 있다. backlog가 있으면 `QUAL_PENDING`, 저장/load 실패면 `TARGET_PASS_ARTIFACT_PENDING/FAILED`이며 완료로 표시하지 않는다.

미달 qualitative를 남기지 않는 정책 때문에 실패 영상의 사후 육안 분석이 제한된다. 이를 보완하는 범위는 **§11 numeric 진단**까지이며, 몰래 screenshot/image를 저장하는 예외를 두지 않는다. 기존 TA2·paper·ablation·PAN-Crafter 자료는 소급 삭제하지 않는다.

---

## 13. 로컬 기록 — 시트가 실패해도 분석 가능한 원장

다음은 논리적 디렉터리 예시이며 실제 mount는 서버에서 확정한다.

```text
<work_root>/<campaign>/<server>/<dataset>/
  campaign_manifest.json
  baseline_resolved.json
  method_source/                         # 코드·새 방법 문서, 기존 데이터는 참조
  preflight/                             # 숫자·검증 결과; 모델 weight 없음
  cases/<resolved_case_id>/config.json
  runs/<run_id>/attempt_<n>/
    identity.json
    events.jsonl                         # 상태 전이·중단/실패 이유
    train_scalar.jsonl                   # loss·LR·shift·시간
    evaluations/<step>/metrics.json      # snapshot hash, 전체 평균·per-image
    diagnostics/<step>/*.json            # scalar·vector·통계, 예측 이미지 없음
    decisions.jsonl
    qualified/<step>/                    # 조건 통과 시에만 생성
      teacher_aligner.*
      artifact_manifest.json
      qualitative/
  outbox/pending/                        # 불변 수치 payload
  outbox/verified/                       # 실제 write+readback receipt
  closeout/                              # 예정/완료/미완료 전체 목록
```

### 13.1 필수 identity

campaign/method/edge/evaluation/storage protocol version, dataset, server, role, EF anchor, ET profile, horizon, replica/seed, attempt, actual completed updates, effective batch·optimizer, source/runtime/dataset/LMS/initial-state/stream digest, model_state_sha256, gate H*/E*.

`run_id` 제안: `EGR2_<DATASET>_<SERVER>_<EF>_<ET-or-BASE>_H<50|100|150>_R<1|2|3>_S<seed>`.

runtime config가 다르면 같은 run_id로 이어 쓰지 않는다. 모델이 저장되지 않았으면 `checkpoint_path=null`, `weights_file_sha256=null`, `weights_saved=false`다. **model-state hash가 있다는 것을 파일이 있다는 뜻으로 쓰지 않는다.**

### 13.2 Local-first와 중복 방지

`평가/진단 → local atomic write·flush → immutable outbox → Sheets append → 정확한 readback → receipt` 순서다. 네트워크 실패·quota는 학습의 barrier가 아니다. pending 수와 최고 대기시간은 상태 숫자로 남긴다.

record key는 campaign/run/attempt/step/type/split/policy/probe/sample-id/metric-name로 정의하고 그 immutable payload의 digest를 record_id로 만든다. retry마다 새 timestamp로 새 record_id를 만들지 않는다. receipt·전송시각은 payload와 별도 관리한다. concurrent publisher도 같은 key로 재전송 중복을 검사한다.

시트가 비어 있어도 local config, 모든 per-image metric, gate 판정, b/residual/total 진단, 상태·선정 이력을 통해 해당 실험을 설명할 수 있어야 한다. 작은 scalar summary뿐 아니라 평균의 근거 per-image 수치도 보존한다.

---

## 14. `analysis` 시트 업로드 지시

### 14.1 기존 원장에 새 campaign으로 분리

대상은 기존 `pan-cvpr27` workbook의 **`analysis`**다. known workbook ID는 `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`이며, 실행 시 실제 title/header/쓰기권한을 확인한다. 이 문서 작성 시에는 live sheet를 수정하거나 재조회하지 않았다.

기존 TA2 결과·152열·계획/보고서 행을 덮어쓰지 않는다. 현재 header를 읽어 **열 이름으로 매핑**하고 row 번호를 하드코딩하지 않는다. 새 campaign label은 `Stage 1 EdgeMatch + residual / RR-only / target-qualified storage`로 한다. 기존 `paper`, `ablations`, 대표 model selection은 자동 갱신하지 않는다.

식별 열 다음 metric 순서는 기존 요구를 유지한다.

`HQNR | D_s | D_lambda | JQM | ERGAS | SCC | SAM | PSNR | 나머지 RR | Infer(ms) | Mem(MB) | Params(M) | FLOPs(G) | Train/Eval/Wall(h)`.

Seed는 별도 열에 보존하고 Experiment 문구는 개행 없는 한 줄로 쓴다. 예:

`EF08 | EdgeMatch+residual | λabs=0.3 λeps=0.0001 | A LR×1 U LR×1 | fresh50K`.

### 14.2 올릴 record 종류

| Type | 전송 내용 | 빈도 |
|---|---|---|
| PLAN / DECISION | 계획 template, 실제 선택/미선택 이유, edge protocol 확정 | 생성/변경 시 |
| RUN_STATUS | 시작·완료·실패·pruned·timebox·upload pending | 상태 전이와 제한된 heartbeat |
| **EVAL_SUMMARY** | 전체 RR/FR metric + gate + same-state + artifact 상태 | **모든 공식 평가점, 미달 포함** |
| EVAL_IMAGE | 실제 각 평가 image의 metric | 모든 공식 평가점; 상세 구획 가능 |
| **RESPONSE** | g_edge/g_residual/g_total, EPE/MAE, 반경·split | LEAN/확장 probe |
| GEOMETRY | b/Δu/u 분포, prior/current 구조 discrepancy, boundary/tie/support | LEAN |
| GRAD | 원/가중 norm, cosine, scope, sample 수 | LEAN |
| COUNTERFACTUAL | current/prior/zero의 같은 T 숫자 | LEAN; `DIAG_ONLY` |
| COST / ARTIFACT | 전체 pipeline 비용, 저장·load·qual 상태 | 통과 모델·대표 cost |
| CAMPAIGN_SUMMARY | 실행/미실행·평가 count·목표 성공률·최종 선택 | 마감 |

**JSON 링크만 올리고 핵심 수치를 공란으로 두는 것은 분석 업로드 완료가 아니다.** RESPONSE/GRAD/GEOMETRY는 기존 `Metric_name / Estimate / N_units / Split_probe / Measurement_domain`의 long-form 행을 활용해 열 폭을 불필요하게 늘리지 않는다. `g_edge`, `g_residual`, `g_total`은 각각 구분한 행이며 기존 `Gain`을 조용히 다른 정의로 바꾸지 않는다.

필요한 최소 보존 필드 `Model_state_SHA`, `Target_pass`, `Weights_saved`, `Qual_status`가 기존 schema에 없으면 **기존 열의 뜻을 바꾸지 말고** 오른쪽에 명시적 새 열을 추가하거나 별도 ARTIFACT long-form으로 매핑한다. 실제 매핑표를 로컬에 보존한다. `Checkpoint_SHA`는 저장 파일의 hash 정의를 유지하고 미저장 시 공란으로 둔다.

### 14.3 상태·색상

계획 회색, campaign 제목 청록, target 통과 녹색, 정상 미달 연한 황색, 데이터/수치 오류 빨강, response/gradient 보라, 업로드 pending 주황으로 구분한다. **미달은 기술 실패와 다르다.** 동일 상태를 텍스트와 boolean으로도 남겨 색만으로 판정하지 않는다. 제목/접근 label 행은 result record로 집계하지 않는다.

입력별 ID를 원본 scene ID라고 부르지 않는다. parent acquisition mapping이 없는 이번 데이터에서는 `sample_id`/`evaluation_image_id`가 충분하다. 지리적 독립성까지 확인된 반복이라는 주장을 하지 않는다.

---

## 15. 마지막 2시간과 보고서

46h 이후 새 training run을 시작하지 않는다. 등록한 deadline 전에 현재 state의 마지막 RR/FR 평가와 lean 수치를 완료할 시간을 먼저 확보한다. 예상 평가시간보다 여유가 작으면 training을 더 일찍 멈춘다.

모든 통과 snapshot의 file/load hash, metric identity, qual 상태를 대조한다. 시트 pending이 남아도 로컬 payload를 유지하고 **업로드 미완료 수를 그대로 보고**한다. 48h가 끝났는데 학습·평가를 조용히 연장하거나 새로운 seed를 시작하지 않는다. 미완료 qual/업로드의 추후 처리에는 기존 통과 모델과 outbox를 사용할 수 있지만 이번 48h 완료 작업으로 표시하지 않는다.

최종 report는 다음을 포함한다.

- 데이터셋별 target-qualified teacher 유무, 같은 snapshot의 HQNR·ERGAS·전체 지표.
- 모든 run의 실제 시작/종료/update, 계획 horizon, candidate 평가 수, pruning/실패/미실행 이유.
- 통과 snapshot 수와 통과 **run/seed** 수의 구분. 한 run의 통과27개가 27회 재현은 아니다.
- chosen recipe의 각 seed 원값, 같은 selector의 평균·표본 SD, 성공률. 미달 seed도 포함한다.
- raw EdgeMatch의 기여, learned residual의 기여, quality trade-off. total gain만으로 learned alignment 성공을 선언하지 않는다.
- weight 미저장 snapshot의 재평가 불가능 범위, qual/Sheets pending, 남은 검증 사항.

**연구적 표기:** 기존 benchmark RR/FR metric으로 hyperparameter·checkpoint를 고르는 test-aware 개발 탐색이다. FR test weight update는 하지 않지만 독립 test 선택 실험이라고 표현하지 않는다. 이번 목표 통과는 Teacher 성능 후보 확보이며 Stage 2 또는 물리적 정합 정확도의 입증이 아니다.

---

## 16. 실행 담당자에게 전달할 시작 지시

> 이 캠페인은 최신 `PANDA_Stage1_EdgeGuided_Residual_Aligner_Final_2026-10-13.md`를 구현한 Stage 1만 대상으로 한다. 예전 scene/global prior·FR ε 학습·scene mapping blocking을 제거하고, s1/WV3·s2/QB·s3/GF2에서 독립48시간을 시작한다. 기존 RR train만 사용하며 FR test는 평가에만 사용한다.
>
> 매 입력 Scharr raw-L1 integer EdgeMatch로 b를 구하고, residual CNN은 W(P,b)를 입력으로 받아 Δu를 예측한다. T에는 원본 PAN을 b+Δu로 한 번 warp한 결과를 제공한다. shifted branch도 EdgeMatch를 재실행한다. architecture/edge/loss routing은 바꾸지 않고 등록된 양수 loss계수·A/U LR·WD·학습 길이만 탐색한다.
>
> 먼저 §4의 열린 수치를 공식 성능 결과를 보기 전에 확정하고 §5 preflight를 통과시킨다. EF00 fresh50K를 기준으로 측정한 실제 비용에 따라 §6–8 queue를 실행하며, 우수 recipe의 R2/R3 시간을 예약한다. 미달 weight는 last/resume/평가 임시본도 만들지 않는다. 모든 snapshot의 수치와 진단은 로컬 durable record→outbox→analysis 순으로 전송한다. 두 목표를 같은 full-method A/T 상태에서 동시에 통과했을 때만 weight와 정성 결과를 저장한다.
>
> total ε gain뿐 아니라 edge/residual 분해와 prior-only/current/zero의 numeric 개입을 남긴다. 새 결과가 없거나 목표 미달이어도 그 사실을 기록하며 성공할 때까지 무한 반복하지 않는다. 구현 변경·checkpoint 저장·시트 업로드의 실제 완료 여부를 각각 구분하여 보고한다.

---

## 부록 A. 숫자·범위의 출처 구분

| 항목 | 출처/성격 |
|---|---|
| RR-only, 입력별 b, prewarped CNN input, Cartesian residual, one-shot T warp, shared Scharr, b_eps 재계산 | **[S1] 새 방법 고정** |
| λabs=1, λeps=0.001, R_eps=2, every update | **[S1] 초기 설정**, 최적값 아님 |
| s1/WV3·s2/QB·s3/GF2, HQNR/ERGAS 목표, 48h, 미달 weight·정성 미저장, local+sheet | **사용자 지시/[S2] 상속** |
| EF/ET/H 후보, phase budget, response 분해, 숫자 대조와 계수별 선택 순서 | **이번 실험 설계 제안** |
| R_edge=4 및 train-only 6/8 사전 재결정, tie 정책, m_edge=R+5, m_abs=16, probe 표본 수 | **이번 운영 제안**, 실행 전 명시 확정 |
| 실제 Teacher width/depth·optimizer baseline·장치·처리속도·loader/evaluator·업로드 header | **서버 확인값 필요**, 이 문서에서 확인 완료로 주장하지 않음 |
| 성능 목표 달성·48h 내 실행 수·정합 정확도·GF2/QB 최적계수 | **미검증** |

## 부록 B. 파일 사용 범위

이 MD와 동봉 `EdgeRes48H_Case_Registry.json`은 실행 **명세**다. registry의 EF20개·ET9개·H3개를 전부 곱한 exhaustive queue가 아니다. `UNRESOLVED_SERVER_BASELINE`은 실제 baseline/edge protocol을 실행 전 pin하라는 뜻이지 외부 scene/FR 자료를 구하라는 뜻이 아니다. 검증 스크립트/runner/모델 구현은 동봉하지 않는다.
