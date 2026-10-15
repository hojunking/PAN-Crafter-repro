# s2 · W104–D122 · No-Align: KD 독립 효과 탐색 실험·구현 통합 명세

**작성일:** 2026-09-11  
**버전:** `v1.0-proposal`  
**캠페인 ID:** `S2_W104_D122_NOALIGN_KD_20260911`  
**서버:** s2. 실제 GPU·VRAM·소프트웨어·작업 경로는 실행 manifest에서 확인한다.  
**시간 정책:** **시간 제한 없음.** 10시간·24시간·30시간 종료 조건을 계승하지 않는다. 우선순위와 의존성에 따라 진행하며, 대응 대조군·장기 fitting·반복 실험을 마치는 것이 기준이다.  
**상태:** 기존 계획을 새 사용자 결정에 맞춰 재구성한 문서다. 이 문서 작성 과정에서 s2 접속, W104 모델 통합, 위성 데이터 학습 또는 성능 측정을 실행하지 않았다.

> **이번에 먼저 답할 질문**  
> PAN aligner의 성능과 무관하게, **동일한 W104–D122 U-Net만으로 학습한 Teacher가 같은 구조의 Student에게 유효한 복원 지식을 전달하는가?**  
> 효과가 있다면 그것은 ① Teacher 실패 영역의 GT 재가중, ② Teacher 출력 모방, ③ GT/Teacher 구조 통계, ④ 성분별 삼각관계 중 어디에서 발생하는가?

**핵심 고정:** `width=104`, `depth=[1,2,2]`, WV3 입력 9ch, 단일 HRMS residual 복원, **Teacher·Student 모두 aligner 없음**. PAN은 dataset/native 입력 그대로 쓰며, MS base와 GT의 기존 좌표를 유지한다.

**기존 연구는 폐기하지 않는다.** Aligner 최적화는 별도 캠페인으로 진행한다. 이번 결과를 얻은 뒤 유력 KD를 최종 aligner에 이식하여 상호작용을 검증한다. 이번 no-align 결과를 정합 모듈 개선의 증거로 사용하지 않는다.

---

## 0. 결정의 우선순위와 출처

### 0.1 이번 변경은 이전 폭·깊이·aligner 조건보다 우선한다

| 항목 | 이전 계획 | 이번 고정/처리 |
|---|---|---|
| 복원 backbone | W112–D124, 더 과거에는 W96–D124 | **Teacher·Student 모두 W104–D122** |
| Depth 설정 | `[1,2,4]` | **`[1,2,2]`**. 실제 stage 배치 해석은 현재 모델 코드 확인 |
| Aligner | frozen/미세조정/scratch 비교 | **이번 모든 정식 모델에서 제거** |
| PAN sampling | learned correction 또는 donor correction | **native forward에서 수행하지 않음** |
| 입력 | PAN + bicubic MS | **9ch 그대로 유지** |
| Reconstruction | 연속 가중 hard/soft | **수식 유지**, N0/R0/R1/R2/R3 비교 |
| 출력 통계 | IV/GV/GC/SC, H/T/FIX/WH/AD | **유지 및 비교** |
| TRI-A/B | 픽셀·통계의 GT 방향 KD | **그대로 이행 가능** |
| G 계열 | 이동량·precision·분포 KD | **현재 모델에 적용 불가. 정합 캠페인으로 이관** |
| TRI-C | 정합 covariance 전파 및 민감도 | covariance형은 이관. **입력 민감도형만 별도 선택적 no-align 재정의** |
| 시간 | 약 24h 첫 검토 | **상한 없음** |
| Mutual/EMA | 기본에 없음 | 자동 도입하지 않음 |

**No-align은 frozen aligner와 다르다.** Teacher aligner, 상수 보정, 학습된 shift cache, 사전 정합 PAN을 뒤에 숨겨 쓰지 않는다. 데이터셋 자체의 전처리 이력은 manifest에 보존하되, 이번 캠페인의 새 정합 처리는 없다.

### 0.2 이 문서의 근거와 신규 제안

- **[U0] 최신 사용자 결정:** W104·d122 고정, align module 없이 KD 효과부터 확인, 시간 제한 없이 case 정리.
- **[S0]** `PAN_S2_W112_D124_KD_Variance_Adaptive_Campaign_2026-09-10.md`: §§6–9의 R3·출력 통계, §§13–17의 비교·구현·상태 관리, §§18–19의 평가·fitting 해석.
- **[S1]** `PAN_S2_W112_D124_TGeo_ABC_Addendum_2026-09-10.md`: §§3–5의 TRI-A/B, §6의 C 민감도/공분산 구분, 대응 대조군.
- **[S2]** `PAN_W96_D124_GT_Anchored_KD_Experiment_Plan_2026-09-09.md`: §§3–4의 연속 hard/soft와 GT 방향 성질, §§7·10–13의 대조·초기화·평가.
- **[S3]** `PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2 (1).md`: §5.2의 signed Scharr edge, §10의 raw/정합 기준 구분. **앞단 아키텍처·warp는 이번에 계승하지 않음.**
- **[S4]** `PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md`: 정합 캠페인과 이번 실험의 구분 근거. **N1/N2/N3 입력 교대와 offset loss는 이번에 OFF.**

**계승과 제안의 경계:** 수식·통계 정의는 위 자료를 따른다. W104 no-align Teacher 확보 절차, 새 run ID/우선순위, 선택적 `C-NASENS`, 대조군 재배치는 **이번 실행 설계**다. 알려지지 않은 실제 LR·batch·checkpoint·처리량은 `REQUIRED`로 남긴다. 외부 논문을 새로 재현하거나 최신 문헌을 추가 조사한 문서는 아니다.

---

## 1. 무엇을 이번에 실행하고 무엇을 분리하는가?

| 계열 | 이번 상태 | 이유/실행 범위 |
|---|---|---|
| N0/R0/R1/R2/R3 | **필수** | KD와 GT 재가중의 독립 효과 |
| STAT-IV/GV/GC/SC | **실행** | Student 출력에서 계산하므로 aligner 불필요 |
| 통계 H/T/FIX/WH/AD | **실행** | GT 감독과 Teacher 통계 전달을 구분 |
| Signed output-edge | **대조로 실행** | variance 고유 효과와 일반 경계 감독 구분 |
| TRI-A-SIGN/COS/CAP/BANDADV | **실행** | 영상값의 성분별 soft 제어 |
| TRI-B-SIGN/CAP/COMPADV, GC/SC 확장 | **실행** | 구조·분광 통계 성분별 soft 제어 |
| TRI-AB, GV+SC | **단독 비교 후 실행** | 실제 상보성 검증 |
| Window/std/log/residual/이차 모멘트 | **후속 실행 목록에 유지** | 통계 표현의 민감도·정보 차이 |
| 장기 fitting/Teacher-copy/seed 반복 | **필수 확인 단계** | 추가 학습시간·초기화 효과와 KD 구분 |
| G1–G5/G-GEO/G-EQ/G-STRUCT | **`DEFERRED_ALIGNMENT`** | 이번에는 예측 이동량 μ 자체가 없음 |
| N1–N3/offset consistency/PAN–GT geometry | **`NOT_APPLICABLE_MAIN`** | aligner가 없으며 입력 정합 실험과 분리 |
| C-DIAG/FULL/QSCALAR/EQPROXY | **`DEFERRED_ALIGNMENT_COV`** | 근거 있는 정합 covariance가 없음 |
| C-QISO | **단독 신규 방법으로 집계하지 않음** | 필요하면 일반 quadratic output-KD 대조로 재명명. FULL 없는 방향성 검증은 아님 |
| C-SENS | **`OPTIONAL_REFORMULATION`** | 학습된 보정량 대신 raw PAN 입력의 국소 민감도로 재정의 가능. §9 참조 |
| 새 uncertainty/covariance head, learned gate | **`PARKED_SCOPE`** | 먼저 head 없는 KD와 통계의 효용을 확인 |
| TGeo/WSL 원래 분류 알고리즘의 직접 복사 | **이번 재현 대상 아님** | TRI-A/B는 관계 관점의 회귀용 제안이며 원 논문의 공식 재현이 아님 |

**중요:** G-GEO/G-EQ가 준비될 때까지 기다리지 않는다. 이번 no-align Teacher에는 이동량 자산이 필요 없으며, 원래 donor의 반응 기울기는 이번 output KD의 시작 조건이 아니다.

---

## 2. 고정 아키텍처와 전방 계산

### 2.1 모델 식

LRMS를 `S_LR`, PAN을 `P`, RR HRMS GT를 `Y`라고 둔다.

$$M=U_4(S_{LR}),\qquad X=\operatorname{Concat}(P,M),$$
$$\hat Y_T=M+F_{\theta_T}(X),\qquad\hat Y_S=M+F_{\theta_S}(X).$$

`F_T`와 `F_S`는 **동일 W104–D122 구조, 별도 가중치**다. Teacher는 사전학습 후 고정한다. 입력 순서는 실제 기존 구현과 checkpoint에 맞춰 확정하되 모든 run이 공유한다.

| Tensor | train 예시 | 의미 |
|---|---|---|
| P | `[B,1,64,64]` | 원래 학습 PAN |
| S_LR | `[B,8,16,16]` | 원래 학습 LRMS |
| M | `[B,8,64,64]` | 기존 bicubic MS base |
| X | `[B,9,64,64]` | PAN+MS 결합 |
| residual | `[B,8,64,64]` | U-Net 출력 |
| Yhat, Y | `[B,8,64,64]` | 최종 HRMS와 고정 GT |

위 크기는 기존 캠페인의 예시다. 실제 W104 dataset config가 다르면 기록하고 같은 cohort 전체에 적용한다. `d122`를 입력 해상도나 총 블록 수로 임의 해석하지 않는다.

### 2.2 넣지 않는 구성

- Teacher/Student aligner, 변위 출력, warp wrapper, deformation field.
- 상수 `(+0.23, -0.06)` 보정, old donor cache, zero-shift `grid_sample`.
- Aligner-only crop와 z-score. **기존 데이터의 radiometric normalization은 유지.**
- LPAN/HPAN 입력, PAN reconstruction, dual-MARs, attention/CM3A.
- Native/corrupted step 교대, PAN/MS random jitter, clean/jitter 복원 두 경로.
- GT나 output의 재정렬, inverse warp, 예측에 따라 달라지는 reconstruction mask.

`aligner=None`을 설정한 뒤 내부에서 기존 donor를 호출하는 fallback이 없는지 확인한다. Mainline에서 **PAN은 복원망 입력까지 새로 보간되지 않아야 한다.** MS 확대에 필요한 기존 bicubic은 별개다.

### 2.3 실제 모델 확인 목록

```text
backbone class / repository commit / source path
width=104, depth=[1,2,2]
encoder/bottleneck/decoder별 실제 채널과 반복 수
9ch input, 8ch residual, MS base 합산 위치 1회
normalization 종류, group/channel divisibility
MARs mode-specific parameter/routing 잔재 여부
aligner parameter=0, native PAN warp call=0
params / trainable params / input-size별 latency / peak memory
```

W96/W112의 weights를 W104로 `strict=False` 로드해 pretrained라 부르지 않는다. Width와 depth가 함께 달라졌으므로 shape·key를 strict 검사한다. GN을 쓰는 경우 104가 group 수로 나누어지는지 확인하되, 현재 backbone이 특정 GN 설정이라고 가정하지 않는다.

---

## 3. Teacher 확보와 초기화: KD의 출발점을 분리한다

### 3.1 정식 Teacher

**T00:** W104–D122, no-align, 동일 데이터·전처리로 plain GT L1 학습.

$$L_T=\operatorname{mean}|\hat Y_T-Y|.$$

이미 정확히 같은 조건의 checkpoint가 있으면 재사용한다. 없으면 T00부터 학습한다. Teacher의 준비시간도 전체 비용에 기록한다.

| 보유 자산 | 처리 |
|---|---|
| 검증된 W104–D122 no-align checkpoint | 평가·config/hash 확인 후 T00으로 사용 가능 |
| W112–D124 또는 W96–D124 checkpoint | 정식 동일 구조 Teacher로 사용하지 않음 |
| W104이지만 aligner와 함께 학습된 복원망 | aligner를 떼는 순간 입력 조건이 달라짐. native no-align 재검증/적응 전에는 T00으로 간주하지 않음 |
| 사용할 Teacher 없음 | GT-only baseline/통계-H 구현은 먼저 가능. KD는 T00 준비 후 진행 |

별도 크기의 Teacher나 정합된 Teacher를 쓰는 것은 나중의 다른 연구 축이다. 이번 캠페인에서 조용히 허용하지 않는다.

### 3.2 Teacher 선택 규칙

Teacher checkpoint는 사전에 정한 validation selector로 고정한다. 장기 fitting Teacher를 추가하면 `T00_BESTVAL`, `T00_LAST_N`, `T00_LONG2N`처럼 **별도 Teacher ID**로 관리하고 tau를 다시 계산한다. Train loss가 가장 낮다는 이유만으로 가장 좋은 Teacher라고 가정하지 않는다.

같은 native input에서 Teacher–GT 오차, 경계/통계 오차, late checkpoint의 악화 여부를 확인한다. 일부 위치에서 Student가 Teacher보다 나은 것은 R3가 예정한 상황이며 Teacher 탈락 조건 자체는 아니다.

### 3.3 Student와 baseline

주력은 **독립 Student 초기값**을 저장한 뒤 모든 비교에 재사용한다. Teacher와 Student는 가중치를 공유하지 않는다. Teacher seed와 Student seed도 역할을 분리한다.

- Teacher seed: 기존 유효 checkpoint의 값을 기록. 없으면 `2025`를 시작 제안으로 사용.
- Student paired seed: 기존 s2 대응 block의 seed 우선. 없으면 `1234`를 시작 제안으로 사용.
- 확인 반복: 같은 s2에서 Student seeds `[1234,2025,777]` 같은 사전 고정 집합으로 최소 3회. 이는 제안값이지 이미 실행한 기록이 아니다.

Teacher와 seed가 같아도 초기 tensor·data RNG 경로가 달라질 수 있으므로 hash를 비교한다. 완전히 같은 초기값·순서의 결정론적 GT 학습이 Teacher를 재생하는 상황만으로 KD 일반 효과를 판단하지 않는다.

**N0 baseline은 동일 Student 초기값의 독립 GT 학습이다.** Teacher 숫자 하나를 N0 대용으로 모든 Student에서 빼지 않는다. Teacher-copy Student는 §13의 별도 초기화 대조다.

---

## 4. Reconstruction case: 승인한 hard/soft 유지 [S0 §6, S2 §§3–4]

### 4.1 세 오차와 두 gate

$$e_T(p)=\frac1C\sum_c|\hat Y_{T,pc}-Y_{pc}|,$$
$$e_S(p)=\frac1C\sum_c|\hat Y_{S,pc}-Y_{pc}|,\qquad e_{ST}(p)=\frac1C\sum_c|\hat Y_{S,pc}-\operatorname{sg}(\hat Y_{T,pc})|.$$

$$d_T(p)=\frac{e_T(p)}{e_T(p)+\tau_R},\qquad a_T(p)=\frac{[e_S(p)-e_T(p)]_+}{e_S(p)+\varepsilon_R}.$$

$$w_H=1+\alpha_Rd_T,\qquad w_K=\beta_R(1-d_T)a_T.$$

- `d_T`: Teacher가 남긴 실제 복원 오차의 크기.
- `a_T`: 같은 위치에서 현재 Teacher가 Student보다 나은 정도.
- Gate용 `e_S`만 detach하고 실제 loss의 `e_S/e_ST`는 live.
- Gate는 `[B,1,H,W]`, 8개 band에 공통. Teacher target·GT·모든 gate는 gradient 없음.

### 4.2 다섯 R-case

| Case | Reconstruction loss | 확인하는 것 |
|---|---|---|
| **REC-N0** | `mean(e_S)` | **완전한 no-KD 기준**. 이번에는 alignment KD도 없음 |
| **REC-R0** | `mean(e_S + beta_R e_ST)` | 일반 고정 가중 output KD |
| **REC-R1** | `mean((1+alpha_R d_T)e_S)` | Teacher 실패 지도에 따른 GT 재가중만 |
| **REC-R2** | `mean((1+alpha_R d_T)e_S + beta_R(1-d_T)e_ST)` | Teacher 자체 오차로 hard/soft 배분 |
| **REC-R3** | `mean(w_H e_S + w_K e_ST)` | 위 배분 + Student가 더 나은 영역의 모방 해제 |

**첫 질문은 R0 대 N0, R3 대 N0다.** 이어 R1 대 N0, R3 대 R1, R3 대 R2로 작동 원리를 분해한다. R3가 좋지 않다고 통계-H나 일반 KD까지 자동 폐기하지 않는다.

### 4.3 Hard를 없애지 않는 이유와 제한

둘 다 같은 오답이면 `e_ST=0`이어도 GT loss는 남는다. 둘 다 크게 틀리되 Teacher가 조금 나으면 soft는 작은 양수일 수 있다. **이진 target switch가 아니다.**

`alpha_R>=0`, `0<=beta_R<1`이면 GT와 다른 출력 성분에 대한 reconstruction gradient는 hard가 soft보다 커 GT 방향을 유지한다. 이는 그 loss의 출력 tensor에서의 국소 성질이며, 파라미터 업데이트·통계 loss까지의 전체 gradient·검증 성능 보장이 아니다.

특히 일부 band만 이미 GT와 같고 같은 pixel의 다른 band는 틀린 경우, 평균 gate가 그 정확한 band의 soft를 허용할 수 있다. 이 부분을 TRI-A가 다룬다.

### 4.4 Calibration과 시작 계수

$$\tau_R=\max(\operatorname{median}_{train,p}e_T(p),\varepsilon_{scale}).$$

고정 native train subset 또는 전체 train에서 한 번 계산한다. Subset이면 sample IDs·RNG·전처리·Teacher hash를 저장한다. 같은 cohort의 Student는 동일 tau를 쓴다. Per-image/batch min–max나 test 통계를 사용하지 않는다.

기존 시작 제안은 `alpha_R=1`, `beta_R=0.1`, `epsilon_R=1e-6`([0,1] intensity 기준)이다. 최적값으로 보장된 수치는 아니다. Teacher를 바꾸거나 입력 단위/전처리를 바꾸면 calibration을 새로 한다.

**Reduction:** 전체 고정 픽셀/밴드 수로 평균한다. Soft-active pixel 수나 soft weight 합으로 나누어 KD의 약화를 다시 상쇄하지 않는다.

---

## 5. 통계 표현: aligner 없이 그대로 가능한 variance [S0 §8]

### 5.1 공통 계약

비교하는 대상은 기본적으로 **최종 HRMS** `Y/Yhat_T/Yhat_S`다. 이동량 variance, 입력 표준화 variance, 예측 uncertainty와 혼동하지 않는다.

국소 평균을 `B_k`라 두면:

$$\operatorname{Var}_k(Z)=\mathcal B_k(Z^2)-\mathcal B_k(Z)^2.$$

시작은 `window=5`, `stride=1`, population variance(`correction=0`). Teacher·Student·GT의 대비를 각각 z-score해 분산 차이를 지우지 않는다. FP32 이상에서 계산하고 필요하면 공간 상수를 미리 빼 수치 취소를 줄인다.

Signed Scharr:

$$K_x=\frac1{32}\begin{bmatrix}-3&0&3\\-10&0&10\\-3&0&3\end{bmatrix},\qquad K_y=K_x^\top.$$

Scharr는 valid convolution. k5에서는 공통 통계 margin3, train64라면 중심58×58. IV/SC도 1px interior 후 local pooling을 적용해 중심을 맞춘다. **통계 중심만 제한하며 reconstruction 전체 영역이나 dataset crop은 바꾸지 않는다.**

### 5.2 네 가지 표현

| ID | 수식/표현 | WV3 출력 shape | 쉬운 의미 |
|---|---|---|---|
| **STAT-IV** | `Var_k(Z_c)` | `[B,8,Hv,Wv]` | 국소 밝기·대비 변화 |
| **STAT-GV** | `Var_k(K_y*Z_c), Var_k(K_x*Z_c)` | `[B,16,Hv,Wv]` | 각 방향의 경계 응답 변화 |
| **STAT-GC** | `B_k(ggᵀ)-B_k(g)B_k(g)ᵀ` | `[B,32,Hv,Wv]` | band별 가로·세로 gradient 관계 |
| **STAT-SC** | `B_k(zzᵀ)-B_k(z)B_k(z)ᵀ` | `[B,64,Hv,Wv]` | 국소 영역의 8개 band 공동 변화 |

`g=(g_y,g_x)`, `z=(Z_1,...,Z_8)`. GV 순서는 `[dy 8bands, dx 8bands]`. GC는 band별 `[cyy,cyx,cxy,cxx]`, SC는 8×8 row-major **모든 원소**를 펼친다. 대칭 원소가 두 번 들어가는 기존 all-entry mean L1 reduction을 유지한다.

IV/GV의 미세한 음수 roundoff만 0으로 clamp한다. **공분산 off-diagonal의 실제 음수를 clamp하지 않는다.** 통계 S/T/G에 대해 같은 전처리·창·중심·band 순서를 사용한다.

SC 구현에서 `[B,C,C,H,W,k,k]` 전체 unfold를 만들지 말고 band row/pair 단위로 pooling한다. 처음부터 행렬 inverse/eigenvalue loss로 바꾸지 않는다.

### 5.3 후속 표현: 모두 보존하되 별도 ID

| ID | 변경 | 주의 |
|---|---|---|
| `REP-W3/W7` | 3×3 또는 7×7 창 | 각 설정에 대응 baseline·calibration 필요 |
| `REP-MULTI357` | 3/5/7 창의 별도 loss를 평균 | 단일 창 대비 추가 통계·영역 효과 구분 |
| `REP-STD` | 비음수 variance에 `sqrt(v+eps)` | IV/GV에 우선. 공분산 음수 원소에 elementwise sqrt 금지 |
| `REP-LOGVAR` | 비음수 variance에 `log(v+eps)` | IV/GV에 우선. scale·epsilon 재calibration |
| `REP-RESIDUAL` | `Y-M`, `T-M`, `S-M`에서 통계 | 출력 variance와 같지 않음 |
| `STAT-M2` | `E[ggᵀ]`, 평균을 빼지 않은 구조 이차 모멘트 | **이번 registry의 새 ID**. G-STRUCT 이동량 KD와 다른 loss |
| `REP-GVSC` | GV와 SC loss의 합 | 두 단독 case 후 결합, 계수 중복 증폭 구분 |

픽셀 residual L1은 같은 M을 빼면 원래 L1과 동일하므로 별도 성능 case로 세지 않는다. 반면 residual variance는 `Var(M+R)`의 cross term 때문에 동일하지 않다.

---

## 6. 통계 hard/soft의 다섯 가지 방식 [S0 §9]

선택 표현을 `V(Z)`라 하고:

$$z_G=V(Y),\ z_T=V(\hat Y_T),\ z_S=V(\hat Y_S),$$
$$E_T^V=\operatorname{mean}_j|z_T-z_G|,\ E_S^V=\operatorname{mean}_j|z_S-z_G|,\ E_{ST}^V=\operatorname{mean}_j|z_S-\operatorname{sg}(z_T)|.$$

$$d_T^V=\frac{E_T^V}{E_T^V+\tau_V},\qquad a_T^V=\frac{[E_S^V-E_T^V]_+}{E_S^V+\varepsilon_V}.$$

| Mode | L_V | 해석 |
|---|---|---|
| **H** | `mean(E_S^V)` | GT 통계만 맞춤. 통계 KD 아님 |
| **T** | `mean(E_ST^V)` | Teacher 통계만. 픽셀 GT는 계속 유지 |
| **FIX** | `mean(E_S^V + beta_V E_ST^V)` | 고정 hard/soft |
| **WH** | `mean((1+alpha_V d_T^V)E_S^V)` | Teacher 실패 통계로 GT만 강조 |
| **AD** | `mean((1+alpha_V d_T^V)E_S^V + beta_V(1-d_T^V)a_T^V E_ST^V)` | GT 유지 + 유용한 Teacher 통계 전달 |

전체식은:

$$L=L_{R,chosen}+\lambda_VL_{V,chosen}.$$

- 기본 통계 비교는 reconstruction R3 고정. **N0+통계-H**도 함께 넣어 Teacher 없이 가능한 개선을 분리한다.
- Teacher와 모든 통계 gate는 detach. Student 통계 추출은 live.
- H/T/FIX/WH/AD에서 같은 `V`와 같은 `lambda_V`를 사용한다.
- T의 계수는 원 정의처럼 1이다. FIX/AD soft의 `beta_V`와 강도가 같다고 주장하지 않는다. 필요하면 `T-MATCHBETA`를 별도 대조로 둔다.
- `(z_S-z_T)-(z_G-z_T)`는 `z_S-z_G`와 같으므로 새로운 KD로 집계하지 않는다.

### 6.1 τ_V, λ_V

통계는 intensity² 단위이므로 τ_R나 edge 계수를 그대로 재사용하지 않는다.

```text
e_scale = median(E_T^V on fixed native train calibration)
v_scale = median(abs(V(GT)) over strictly positive finite entries)
tau_V   = max(e_scale, 1e-3*v_scale, 1e-12)
eps_V   = max(1e-3*tau_V, 1e-12)
```

위 하한은 이전 문서의 [0,1] intensity용 시작 제안을 계승한다. Sensor noise의 확률 calibration이 아니다. 고정 Teacher·표현·창·ROI·intensity마다 따로 기록한다.

`lambda_V`는 공통 train pilot Student에서 plain reconstruction과 통계-H의 **출력 gradient RMS**를 측정해 초기 비율 `0.05`를 맞추는 이전 제안을 우선한다. 대응 mode에는 같은 λ를 사용한다. gradient가 0에 가깝다고 거대한 λ를 자동 생성하지 않는다. λ 민감도는 별도 ×0.3/×1/×3 비교다.

`N0+H`는 Teacher가 필요 없다. λ calibration 역시 GT와 공통 Student만으로 가능하다. Teacher-dependent τ는 실제 필요한 mode에서만 요구한다.

### 6.2 직접 edge 대조

$$L_E=\tfrac12\operatorname{mean}|K_x*\hat Y_S-K_x*Y|+\tfrac12\operatorname{mean}|K_y*\hat Y_S-K_y*Y|.$$

밴드별 signed gradient, 고정 1px interior. `N0+EDGE-H`, `R3+EDGE-H`를 GV-H와 비교한다. 이 항은 새 no-align 실험의 **독립적인 대조**이며 예전 A2 Teacher 학습 loss를 자동 계승하는 것이 아니다. λ_E는 선언한 train pilot 절차로 고정한다.

---

## 7. TRI-A: 영상값의 GT 방향을 고려한 KD [S1 §4]

기본 R3의 `w_H,w_K`는 그대로 두고 soft 내부만 바꾼다.

$$u_{pc}=Y_{pc}-\operatorname{sg}(\hat Y_{S,pc}),\qquad v_{pc}=\operatorname{sg}(\hat Y_{T,pc})-\operatorname{sg}(\hat Y_{S,pc}).$$

$$L_R^{A}=\left\langle w_He_S+w_K\operatorname{mean}_c\{m_A|\hat Y_S-\hat Y_T|\}\right\rangle.$$

| Case | m/soft 변경 | 질문 |
|---|---|---|
| **TRI-A-BASE** | `m=1` | R3와 동일. 이미 있으면 재사용 |
| **TRI-A-SIGN** | `1[u*v>0]` | GT 방향과 충돌하는 band의 soft를 빼는가? |
| **TRI-A-CAP** | `SIGN * min(1, abs(u)/(abs(v)+eps_R))` | GT를 지나친 Teacher 방향의 과한 압력을 줄이는가? |
| **TRI-A-COS** | `[dot(u,v)/max(norm(u)norm(v),eps_cos)]_[0,1]` | band 벡터의 평균 방향만으로 충분한가? |
| **TRI-A-BANDADV** | soft의 Teacher 오차·우위를 band별 계산 | 평균 gate가 놓친 좋은 band를 복구하는가? |

COS의 norm0이면 mask0, `eps_cos=eps_R²`를 FP32에서 쓰는 기존 pilot을 계승한다. SIGN은 `u==0` 또는 `v==0`에서 0. CAP은 **loss 계수의 감쇠**이지 실제 optimizer 이동의 clipping이 아니다.

BANDADV는 hard를 그대로 두고:

$$w_{K,c}=\beta_R\frac{\tau_{R,c}}{|T_c-G_c|+\tau_{R,c}}\frac{[|S_c-G_c|-|T_c-G_c|]_+}{|S_c-G_c|+\varepsilon_{R,c}}$$

로 soft만 세분화한다. τ_R,c는 train에서 별도 calibration. 평균 `a_T=0`이어도 좋은 band가 있으면 soft를 줄 수 있으므로 SIGN과 다른 질문이다.

**중복 금지:** scalar 성분에서 Teacher가 GT에 더 가까우면 GT/Teacher 방향 일치가 이미 따른다. BANDADV에 같은 성분 SIGN을 다시 붙인 것을 독립 기여로 세지 않는다. 또한 SIGN은 기존 평균 `a_T=0`이면 좋은 band를 새로 살리지 못한다.

A loss는 기존 R3를 **대체**한다. `R3 + A전체식`으로 hard/soft를 중복 더하지 않는다. 이진값은 soft 성분 mask이며 hard/soft 양자택일 switch가 아니다.

---

## 8. TRI-B: 구조 통계의 성분별 삼각관계 [S1 §5]

기본 STAT-AD의 hard와 pixel reconstruction을 유지하고, 통계 soft의 원소를 고른다.

$$u^V=z_G-\operatorname{sg}(z_S),\qquad v^V=\operatorname{sg}(z_T)-\operatorname{sg}(z_S),$$
$$m_B=\mathbf1[u^Vv^V>0],\qquad m_{Bcap}=m_B\min\left(1,\frac{|u^V|}{|v^V|+\varepsilon_V}\right).$$

$$L_V^{B}=\left\langle w_H^VE_S^V+w_K^V\operatorname{mean}_j\{m_B|z_S-z_T|\}\right\rangle.$$

| Case | 변경 | 쉬운 설명 |
|---|---|---|
| **TRI-B-BASE** | 기존 GV-AD | 통계 전체의 평균 성적으로 판단 |
| **TRI-B-SIGN** | 통계 원소의 방향 mask | 필요한 경계를 더 없애는 Teacher 조언은 제외 |
| **TRI-B-CAP** | SIGN+크기비 감쇠 | 방향이 맞아도 과장된 통계 모방을 감쇠 |
| **TRI-B-COMPADV** | 원소별 Teacher 상대우위 | 전체 평균은 나빠도 잘하는 band/방향은 배움 |
| **TRI-B-GC / SC** | 위 방식을 GC/SC에 적용 | 방향 관계·분광 관계의 일부 지식 전달 |

COMPADV는 hard를 기존 평균 가중치로 유지하고, `j`별 τ_V,j·d_T,j·a_T,j로 soft만 바꾼다. 같은 성분 SIGN을 중복 적용한 별도 case는 만들지 않는다.

예: `V_T<V_S<V_G`이면 Teacher는 Student보다 더 평탄하게 복원했다. SIGN은 그 성분의 모방을 끊는다. `V_S<V_T<V_G`이면 기존 우위 gate가 허용하는 범위에서 Teacher를 참고하되 GT 통계 hard도 남는다. **“분산을 항상 키운다”가 아니다.**

- B는 STAT-AD **통계 항을 대체**한다. AD 전체+B 전체를 중복 더하지 않는다.
- covariance 원소별 gate는 좌표축 의존적인 감독이다. 자동으로 주성분 불변 감독이 된다고 부르지 않는다.
- GT/Teacher 원소를 섞어 새 covariance target 행렬을 조립하지 않는다. 실제 영상에서 계산한 Student 통계와 원래 GT/Teacher 통계를 loss로 비교한다.
- Noise/ringing으로 통계만 맞는지, 정확한 edge가 보존되는지 별도 진단한다.

### 8.1 A/B의 결합

$$L_{AB}=L_R^{A}+\lambda_VL_V^{B}.$$

반드시 같은 baseline을 공유한 네 cell을 비교한다.

| Cell | Pixel | 통계 |
|---|---|---|
| 공통 base | R3 | GV-AD |
| A만 변경 | A-SIGN | GV-AD |
| B만 변경 | R3 | GV-B-SIGN |
| AB | A-SIGN | GV-B-SIGN |

A-only의 통계 OFF 실험과 AB만 비교하면 B의 효과와 통계 추가 효과가 섞인다. 위 2×2 대응을 별도로 남긴다.

---

## 9. C 계열의 no-align 처리: 주 실험과 별도 선택 분기

### 9.1 기본 결정

**주 실험 P0–P3는 Teacher·Student 학습 forward에서 PAN warp가 전혀 없다.** 이동 covariance를 임의의 identity/0으로 채워 C-DIAG/FULL을 실행하지 않는다. 정합 module 연구는 기다리지 않아도 된다.

다만 이전 `C-SENS`의 개념은 learned aligner 없이도 **raw PAN 입력 민감도**로 재정의할 수 있다. 이는 자동 계승이 아니라 이번 문서의 선택적 별도 프로토콜 `NA-TSENS`다. 먼저 R/STAT/A/B의 비교를 완료한다.

### 9.2 선택적 `TRI-C-NASENS`

기본 Teacher 출력은 그대로 `T=M+F_T([P,M])`. 진단 함수만:

$$z_T(\delta)=M+F_T([\mathcal W(P,\delta),M])$$

로 두고 **기준점 δ=0**에서:

$$J_y\approx\frac{z_T(h e_y)-z_T(-h e_y)}{2h},\qquad J_x\approx\frac{z_T(h e_x)-z_T(-h e_x)}{2h},$$
$$q_{pc}=J_{y,pc}^2+J_{x,pc}^2,\qquad r_{pc}=\frac{s_{sens}}{s_{sens}+q_{pc}}.$$

`s_sens`는 고정 native train의 양의 q 중앙값. 양의 값이 전혀 없으면 이 추가 감쇠를 적용하지 않고 원래 soft로 fallback하며 비율을 기록한다.

$$L_R^{CNA}=\left\langle w_He_S+w_K\operatorname{mean}_c\{r_{pc}|\hat Y_S-\hat Y_T|\}\right\rangle.$$

**바뀌는 것:** Teacher probe 계산과 output soft 계수. **바뀌지 않는 것:** 모델 구조, 실제 Student 입력, native Teacher target, GT hard, 추론 모델.

- `h=0.05 HR px`는 이전 수치 pilot의 출발값. h/2·2h로 안정성 확인. augmentation 반경·실제 오정합 표준편차가 아님.
- probe마다 같은 원 PAN을 독립 sampling. native target을 zero-warp 결과로 교체하지 않음.
- Teacher 최종 HRMS를 직접 shift한 값을 대체물로 쓰지 않음.
- Teacher 기본 forward 외에 **복원 U-Net 4회 추가**, 순차 no-grad. 작은 aligner 4회의 비용이 아님.
- finite-difference ROI는 사전에 고정한 공통 내부로 제한한다. 검증되지 않은 경계에서는 r=1로 원래 soft에 복귀하고 hard는 전체 유지한다. ROI를 결과에 따라 줄이지 않는다.
- Teacher의 입력 민감도 대조이지 정합 uncertainty 또는 정합 개선으로 부르지 않음.
- `NA-STRICT` 결과와 별도 cohort로 기록. 실험 요구를 “Teacher probe까지 어떤 warp도 금지”로 운용한다면 **이 case도 제외**하고 R/STAT/A/B만 실행한다.

### 9.3 C와의 후속 결합

CNA가 유효할 때만 `A-SIGN × r_CNA`를 soft에 적용하는 `TRI-ACNA`를 비교한다. 통계 B에 C를 연결하려면 `z_T(delta)=V(M+F_T([W(P,delta),M]))`의 **통계 Jacobian을 직접 다시 계산**한다. Pixel q를 resize해 통계 uncertainty로 부르지 않는다.

`C-DIAG`, `C-FULL`, `C-QSCALAR`, `C-EQPROXY`, 공분산을 쓰는 `BC/ABC`는 모두 최종 aligner·covariance가 확보된 **후속 정합 캠페인**으로 남긴다. 이번 no-align KD 성립 여부의 필수 조건이 아니다.

---

## 10. 실행 우선순위: 시간 제한 대신 질문의 순서

| 단계 | 우선 작업 | 다음 단계의 의미 |
|---|---|---|
| **P0** | W104 no-align graph·Teacher·동일 초기값·평가 검증 | 폭/깊이/앞단 차이를 제거 |
| **P1** | N0/R0/R1/R2/R3 full-N 대응 | 일반 KD·실패 지도·adaptive KD 중 무엇이 유효한가? |
| **P2-a** | GV H/T/FIX/WH/AD + N0-GV-H + edge 대조 | 통계 supervision과 통계 KD 구분 |
| **P2-b** | A-SIGN, B-SIGN, 이어 COS/CAP/COMPADV | 성분별 충돌 제어의 추가 효과 |
| **P3-a** | IV/GC/SC의 다섯 mode + GT-only 순수 대조 | 어느 통계 표현에 효과가 있는가? |
| **P3-b** | A/B 결합, GC/SC의 B, GV+SC | 상보성 및 고유 기여 |
| **P3-c** | mask mass/shuffle, λ/τ, window/표현 변형 | 단순 loss 규모 또는 smoothing인지 분리 |
| **P4 선택** | C-NASENS 및 대응·결합 | aligner 없는 입력 민감도 접근의 가치 |
| **확인 필수** | 장기 fitting, 동일 compute, seed 반복, Teacher-copy/no-KD | 재현성과 fitting 논지 검증 |

**시간 무제한은 loss를 끝없이 임의 변경하라는 뜻이 아니다.** 각 controlled run은 종료 update N과 schedule을 정한다. 전체 캠페인에 wall-clock 상한을 두지 않으며, 새 결과에 따라 queue 우선순위를 바꾸되 원래 case와 변경 이력을 보존한다.

P1에서 R3가 부진해도 R0가 좋거나 특정 통계-H가 좋을 수 있다. 같은 공통 비교는 완료하고, 우수한 reconstruction을 새 parent로 택할 때에는 `REC_PARENT`를 명시한 새 cohort로 분기한다.

---

## 11. 정식 case registry

모든 ID 앞에 `NA104_`를 붙인다. 아래는 **단일 seed·단일 Teacher·단일 budget당 정의**이며, seed/계수/반복 수를 곱한 총 GPU run 수가 아니다. `OFF`는 해당 항을 계산하지 않는 뜻이다.

### 11.1 준비와 핵심 reconstruction

| ID | R | 통계/추가 | 우선 | 대조/질문 |
|---|---|---|---|---|
| **T00** | plain GT | OFF | P0 | 공통 no-align Teacher |
| **Q00** | N0 | OFF | P1 | 독립 Student 기준 |
| **Q01** | R0 | OFF | P1 | Q00 대비 일반 KD |
| **Q02** | R1 | OFF | P1 | Q00 대비 hard 재가중 |
| **Q03** | R2 | OFF | P1 | Q02 대비 Teacher output 추가 |
| **Q04** | R3 | OFF | P1 | Q03 대비 Student 우위 gate, Q02 대비 KD 고유 효과 |

### 11.2 GV·직접 edge·삼각관계

| ID | R | 통계/추가 | 우선 | 대조/질문 |
|---|---|---|---|---|
| **Q05** | N0 | GV-H | P2 | Teacher 없이 GT 통계가 유효한가? |
| **Q06** | R3 | GV-H | P2 | Q04 대비 GT 구조 통계 |
| **Q07** | R3 | GV-WH | P2 | Q06 대비 Teacher 통계 실패 지도 |
| **Q08** | R3 | GV-T | P2 | Q04 대비 Teacher 통계만 |
| **Q09** | R3 | GV-FIX | P2 | Q06 대비 고정 통계 KD |
| **Q10** | R3 | GV-AD | P2 | Q07/Q09 대비 adaptive 통계 KD |
| **Q11** | N0 | EDGE-H | P2 | Q05 대비 variance 고유 효과 |
| **Q12** | R3 | EDGE-H | P2 | Q06 대비 일반 edge 대조 |
| **Q13** | A-SIGN | OFF | P2 | Q04 대비 pixel sign gate |
| **Q14** | A-COS | OFF | P2 | scalar 방향 감쇠 |
| **Q15** | A-CAP | OFF | P2 | Q13 대비 크기비 감쇠 |
| **Q16** | A-BANDADV | OFF | P2 | 평균 gate 대비 band별 우위 |
| **Q17** | R3 | GV-B-SIGN | P2 | Q10 대비 구조 성분 방향 |
| **Q18** | R3 | GV-B-CAP | P2 | Q17 대비 크기비 |
| **Q19** | R3 | GV-B-COMPADV | P2 | Q10/Q17 대비 성분별 우위 |

### 11.3 다른 출력 통계: 범위를 빠뜨리지 않는다

| ID | R | 통계/추가 | 우선 | 대조/질문 |
|---|---|---|---|---|
| **Q20** | R3 | IV-H | P3 | 국소 밝기 분산 GT |
| **Q21** | R3 | IV-WH | P3 | IV 실패 위치 hard |
| **Q22** | R3 | IV-T | P3 | Teacher IV |
| **Q23** | R3 | IV-FIX | P3 | 고정 IV 결합 |
| **Q24** | R3 | IV-AD | P3 | adaptive IV |
| **Q25** | R3 | GC-H | P3 | 방향 공동 변화 GT |
| **Q26** | R3 | GC-WH | P3 | GC 실패 위치 hard |
| **Q27** | R3 | GC-T | P3 | Teacher GC |
| **Q28** | R3 | GC-FIX | P3 | 고정 GC 결합 |
| **Q29** | R3 | GC-AD | P3 | adaptive GC |
| **Q30** | R3 | SC-H | P3 | 분광 공동 변화 GT |
| **Q31** | R3 | SC-WH | P3 | SC 실패 위치 hard |
| **Q32** | R3 | SC-T | P3 | Teacher SC |
| **Q33** | R3 | SC-FIX | P3 | 고정 SC 결합 |
| **Q34** | R3 | SC-AD | P3 | adaptive SC |
| **Q35** | N0 | IV-H | P3 | Teacher 없는 IV supervision |
| **Q36** | N0 | GC-H | P3 | Teacher 없는 GC supervision |
| **Q37** | N0 | SC-H | P3 | Teacher 없는 SC supervision |

**GT-only 대조의 의미:** Q05/Q35/Q36/Q37는 추론뿐 아니라 학습에도 Teacher가 필요 없다. 반면 Q06/Q20/Q25/Q30은 통계가 H이지만 reconstruction R3에 Teacher가 필요하다. 두 그룹을 모두 “KD 없음”으로 묶지 않는다.

### 11.4 성분·표현의 결합

| ID | R | 통계/추가 | 우선 | 대응 |
|---|---|---|---|---|
| **Q38** | R3 | GC-B-SIGN | P3 | Q29 |
| **Q39** | R3 | SC-B-SIGN | P3 | Q34 |
| **Q40** | A-SIGN | GV-AD | P3 | Q10, Q13 |
| **Q41** | A-SIGN | GV-B-SIGN | P3 | Q10/Q17/Q40의 2×2 |
| **Q42** | R3 | GV-H + SC-H | P3 | Q06/Q30 |
| **Q43** | R3 | GV-AD + SC-AD | P3 | Q10/Q34 |
| **Q44** | R3 | GC-B-CAP | P3 | Q38 |
| **Q45** | R3 | GC-B-COMPADV | P3 | Q29/Q38 |
| **Q46** | R3 | SC-B-CAP | P3 | Q39 |
| **Q47** | R3 | SC-B-COMPADV | P3 | Q34/Q39 |

GV+SC는 단독에서 얻은 λ_GV, λ_SC를 사용한다. 합으로 gradient가 커지는 효과를 분리하려면 **합의 전체 배율만 줄인 대조**도 둔다. 결합에만 유리한 재calibration을 하여 단독과 비교하지 않는다.

### 11.5 선택적 C 및 연결

| ID | R/변경 | 통계 | 상태/대응 |
|---|---|---|---|
| **CS00** | R3 유지, Teacher PAN 민감도 진단만 | OFF | `NA-TSENS` 준비; 학습은 Q04와 같음 |
| **CS01** | C-NASENS | OFF | Q04 및 동일 soft 총계수 대조 |
| **CS02** | A-SIGN × C-NASENS | OFF | Q13/CS01 |
| **CS03** | R3 유지 | GV-B-SIGN soft에 통계-J 기반 감쇠 | Q17; pixel J 재사용 금지 |

CS00을 Q04에서 온라인으로 계산하면 단순 진단 run이며 별도의 새 학습 알고리즘으로 집계하지 않는다. CS 계열을 사용하지 않아도 이번 핵심 KD 가설의 검증은 완료할 수 있다.

### 11.6 표현·초기화·장기 확인 registry

| ID family | 정의 | 필수 비교 |
|---|---|---|
| **VX-W3/W7/MULTI** | 유력 GV 또는 다른 표현의 창 변경 | 같은 창의 H/AD 또는 WH/AD |
| **VX-STD/LOGVAR** | IV/GV 통계 비교 함수 변경 | 동일 R·창·GT/Teacher 설정 |
| **VX-RESIDUAL** | 유력 통계를 residual에서 계산 | 최종 HRMS 통계 |
| **VX-M2-H/AD** | 중심화하지 않은 gradient 이차 모멘트 | 같은 reduction의 GC-H/AD |
| **INIT-TCOPY** | Student를 T00 가중치로 초기화 | KD 없는 동일 restart/continuation |
| **INIT-SUPWARM** | 같은 supervised warm checkpoint에서 시작 | 모든 비교가 같은 warm state |
| **LONG-2N/4N** | N0/R1/R3/유력 통계 장기 fitting | 동일 horizon·schedule·노출량 |
| **COST-MATCH** | KD 실제 학습시간만큼 N0 더 학습 | update 대응과 별도 시간 대응 |
| **REPEAT-S3** | 핵심 대응 최소 3 Student seeds | 같은 Teacher 반복임을 명시 |
| **REPEAT-T** | 다른 Teacher seed/선택 checkpoint | cohort별 재calibration |

---

## 12. 효과의 출처를 가르는 필수 대조군

| 대조 ID | 구성 | 판별할 것 |
|---|---|---|
| **CTL-HSCALE** | R1의 공간 w_H 대신 batch 평균 w_H로 전체 GT만 배율 조정 | 어려운 위치 지도 대 단순 loss scale |
| **CTL-RSHUFFLE** | R3의 d_T,a_T를 함께 공간 permutation | Teacher 실패 지도의 위치 정보 |
| **CTL-AMASS** | A mask와 같은 soft 총계수만 가진 scalar 감쇠 | 방향 선택 대 단순 KD 감소 |
| **CTL-BMASS** | B mask의 soft 총계수 대응 | 성분 선택 대 단순 KD 감소 |
| **CTL-ASHUFFLE/BSHUFFLE** | mask 위치만 사전 고정 RNG로 섞음 | 올바른 위치와 sign 정보의 필요성 |
| **CTL-CMASS** | C-NASENS와 soft 총계수 대응 | 민감도 정보 대 KD 감소 |
| **CTL-TAU** | τ_R ×0.5/1/2 | 특정 정규화 값 의존성 |
| **CTL-BETA** | β_R 0.1/0.3/0.5 등 사전 목록 | KD 강도 의존성; 초기 1미만 유지 |
| **CTL-LAMBDA-V** | λ_V ×0.3/1/3 | 통계 loss scale 의존성 |
| **CTL-GTSTATS** | 유력 R+통계와 N0+같은 통계-H | Teacher 없이 얻는 구조 supervision 효과 |

### 12.1 Soft 총계수 대응의 구현

기존 soft weight `w_K`와 추가 mask m에 대해 동일 minibatch에서:

$$\rho=\frac{\sum_{p,j}w_K(p)m(p,j)}{\sum_{p,j}w_K(p)+\varepsilon},\qquad m_{control}(p,j)=\operatorname{sg}(\rho).$$

분모의 합은 **broadcast된 같은 shape**로 계산한다. 분모가 0이면 원 soft도 0으로 둔다. 이 대조에서는 동일 Student 상태에서 shadow mask를 만들어 scalar로 요약한다. 학습 경로가 달라진 뒤 두 독립 run의 실제 gradient까지 같아지는 것은 아니다.

이는 **대조군의 총계수 매칭**이지 mainline loss를 active pixel로 재정규화하는 규칙이 아니다. Loss값·gradient norm도 별도로 기록한다.

### 12.2 Redundancy를 성공으로 세지 않는다

- R3의 β=0은 R1과 동일하다. 별도 신규 방법 아님.
- TRI-A-BASE는 Q04, TRI-B-BASE는 Q10과 동일하다. hash 일치 시 재사용.
- Student=Teacher라도 GT를 틀리면 hard가 남아야 한다.
- 통계 차이에서 Teacher가 대수적으로 소거되면 GT-only로 표기한다.
- GC 대각과 GV는 같은 통계다. all-entry reduction과 off-diagonal 효과를 분리한다.

---

## 13. 무제한 시간 운영과 장기 fitting

### 13.1 N을 먼저 정한다

`N = REQUIRED_FROM_W104_BASELINE`으로 실제 optimizer update budget을 고정한다. 기존 W104 baseline 설정이 없으면 **새 실행 제안으로 N=50,000**을 선언할 수 있으나 과거 다른 구조에서 확인된 값이라고 쓰지 않는다. Batch, optimizer, LR, warmup, weight decay는 실제 W104 config 확인 후 전 case에 공유한다.

- P1의 다섯 R-case는 같은 full-N으로 비교한다.
- 구현 이상을 찾는 smoke는 성능 실험으로 세지 않는다.
- 긴 queue는 같은 full-N schedule의 앞 0.25N을 screening으로 실행할 수 있다. 이는 짧은 cosine을 끝낸 0.25N run과 다르다.
- 짧은 prefix에서 나쁘다는 이유만으로 강한 fitting 가설을 기각하지 않는다. 해석에 필요한 핵심 대조는 N까지 완료한다.
- 시간 부족 상태를 만들지 않는다. 의존성/수치 오류/과학적 중복은 기록하고 고친다. 모든 유한 확장 목록은 구현 가능성과 판별 목적을 확인한 뒤 실행한다.

### 13.2 Overfitting을 확인하는 두 종류의 장기 비교

**HORIZON 비교:** 처음부터 같은 2N 또는 4N schedule로 N0/R1/R3/유력 방법을 학습한다. N 시점의 LR은 N-horizon run과 다르므로 이를 같은 prefix라고 부르지 않는다.

**CONTINUATION 비교:** 공통 N checkpoint에서 기존 supervised continuation과 새 KD/통계 child를 분기한다. 새로운 tail LR/schedule을 양쪽에 동일 적용한다. 이미 cosine LR이 0에 도달했는데 update만 늘려 추가 학습했다고 하지 않는다.

새 tail LR은 실제 baseline 설정에 맞춰 선언한다. 임의 수치를 본 문서에서 현재 s2의 확인된 값으로 채우지 않는다. Optimizer 유지/restart의 효과도 양쪽에 같게 맞춘다.

### 13.3 Teacher-copy는 기본 scratch와 별도

`F_S <- F_T`이면 처음에 `e_S=e_T`, `a_T=0`, `e_ST=0`이다. R3 soft가 처음에 꺼지는 것은 정상이다. R1형 hard의 추가 fitting과 이후 발생하는 KD를 보는 실험이 된다.

같은 가중치에서 **KD 없는 continuation**, **R1 continuation**, **R3 continuation**을 비교해야 한다. Teacher-copy 하나가 독립 Student보다 좋다고 KD의 고유 효과로 해석하지 않는다.

### 13.4 학습시간 공정성

같은 optimizer update 비교와 같은 wall-time 비교를 둘 다 기록한다. Teacher pretraining과 calibration 비용은 `cost_teacher`, Student 온라인 비용은 `cost_student`, 총비용은 `cost_total`로 분리한다. 여러 Student가 Teacher를 공유하면 amortized 비용도 표시하되 최초 준비비를 숨기지 않는다.

s2 한 서버에서 대응 비교를 유지한다. 다른 서버로 옮기면 동일 case들을 같은 server-seed block에 배치하고 서버 효과를 method 효과로 섞지 않는다.

---

## 14. 평가: aligner가 없으므로 지표 경로도 단순화한다

### 14.1 Native raw 평가를 주 비교로 사용

RR은 `Yhat`와 고정 GT, FR은 `Yhat`·원 PAN·원 MS로 기존 evaluator를 실행한다. GT/output을 정렬한 뒤 점수를 계산하지 않는다.

| 저장/뷰 | 이번 처리 |
|---|---|
| **Raw-original** | 기존 raw metric/ROI/정규화/필터 경로 그대로. 주 비교 |
| **Raw-valid** | 과거 영역 대조가 꼭 필요할 때만 동일 고정 ROI로 diagnostic |
| **Aligned-valid** | **별도 view로 의미 없음.** PAN을 보정하지 않으므로 같은 ROI에서는 raw-valid와 같아야 함 |
| **best_aligned** | 새 selector로 만들지 않음 |
| **best_rr_val** | 독립 RR validation이 있으면 주 selector 후보 |
| **best_raw** | 기존 exploratory selector를 유지하는 경우 별도 표기 |
| **last_N / last_2N / last_4N** | 사전 고정 update 결과를 보존 |

이전 계획의 `FR mat20, metric v2 py`는 최신 확인 출발점이다. 실제 W104 평가의 scene list·dataset hash·metric code hash를 연결한다. 12–19나 임의 subset으로 몰래 되돌리지 않는다. HQNR은 **장면별 계산 후 평균**하며 두 왜곡 평균의 곱으로 바꾸지 않는다.

### 14.2 Selector 차이는 공개적으로 처리

이전 문서에는 RR validation ERGAS 우선과 raw HQNR 탐색 selector가 병존한다. 이번 no-align 변경만으로 과거를 같은 규칙이었다고 고쳐 쓰지 않는다.

**이번 정식 확인 제안:** 독립 RR validation ERGAS로 `best_rr_val`을 선택하고 사전 고정 final-N도 함께 평가한다. 기존 FR test를 매번 선택에 사용하면 `test-adaptive exploratory`라고 기록하며 독립 hold-out 검증이라고 부르지 않는다. 실제 selector 변경은 cohort 시작 전에 manifest에 기록한다.

### 14.3 기록할 지표

- RR: ERGAS, SAM, PSNR, SSIM, SCC, Q8(기존 Q2n 구현과 band 설정 확인).
- FR: HQNR, D_lambda, D_s, 기존 fSCC와 고정 scene/crop 시각 비교.
- Fitting: **가중치 없는 train·validation MAE**, GT 통계 오차, signed edge 오차.
- Artifact: noise/ringing, 과도한 smoothing, 특정 band bias, 평탄·어두운 영역의 악화.

Variance loss가 줄거나 HQNR 하나가 올랐다고 전체 복원 성공으로 판정하지 않는다. GT 구조 통계는 위치 배치를 유일하게 결정하지 않는다.

---

## 15. KD가 유효하다고 말하려면 무엇이 보여야 하는가?

### 15.1 비교별 판정

| 결과 | 허용되는 해석 |
|---|---|
| R0가 N0보다 개선 | 일반 output KD가 no-align에서도 유효한 후보 |
| R1이 N0보다 개선, R3는 R1과 비슷 | Teacher 오류 지도를 이용한 GT 재가중이 주효과 |
| R3가 R2보다 개선 | Student 상대우위에 따른 모방 해제가 기여 |
| GV-H가 개선, GV-AD는 추가 이득 없음 | GT 통계 감독의 이득이지 Teacher 통계 KD의 이득은 미입증 |
| AD가 WH·FIX보다 개선 | Teacher 통계 target과 adaptive 분배의 근거 |
| A/B가 mass control보다 개선 | 단순 KD 약화 이상의 성분 선택 근거 |
| 통계 오차만 개선, RR/경계/FR 악화 | 통계 목적을 맞춘 것과 복원 품질 개선을 분리 |
| train만 좋아지고 validation/FR 악화 | 추가 fitting은 관찰됐지만 활용 품질 개선은 미입증 |
| 장기 supervised와 KD가 같은 수준 | 추가 compute의 설명력이 큼 |

### 15.2 Teacher 실패 영역을 고정해서 본다

고정 Teacher의 평가 오차를 기준으로 `0–50% / 50–90% / 상위10%` 같은 구간을 정의해 모든 Student를 같은 영역에서 비교한다. 이 구간은 평가 분석용이며 hard/soft switch threshold가 아니다.

$$\Delta e(q)=\mathbb E[e_{control}-e_{candidate}\mid q],\qquad WinRate(q)=\Pr(e_S<e_T\mid q).$$

Pixel 오차 bin과 통계 오차 bin은 별도로 만든다. 평가/test bin을 train tau나 파라미터 선택으로 되먹이지 않는다. Student마다 바뀌는 active-gate 영역만 평가해 개선을 주장하지 않는다.

### 15.3 불확실성 없는 현 단계의 주장

이번 결과가 좋아지면 **“Teacher가 남긴 복원·구조 오차를 GT 중심으로 더 fitting하는 동일 구조 KD”**라고 설명할 수 있다. “정합 variance를 학습했다”, “센서 오정합이 해결됐다”, “uncertainty가 calibration됐다”는 주장은 이번 실험으로 뒷받침되지 않는다.

---

## 16. 구현 모듈과 학습 연결

### 16.1 기존 코드의 재사용 범위

현재 제공된 `pan_gt_anchored_kd.py`는 N0/R0/R1/R2/R3 loss reference다. `pan_s2_w112_stat_kd_reference.py`는 이전 zip의 출력 통계 reference다. 이 파일명의 W112는 **아키텍처 구현을 포함한다는 뜻이 아니며**, tensor-level 통계의 재사용 가능성을 실제 코드에서 확인한다.

이전 ABC 문서의 A/B 참고 함수도 검토하여 연결할 수 있다. 다만 **이전 문서의 CPU 테스트 기록이 새 W104 모델·trainer 통합의 통과 기록은 아니다.** 이번 문서에는 새 통합 성공 수치를 기재하지 않는다.

### 16.2 권장 분리 — 실제 저장소 경로가 아니라 제안

```text
experiments/noalign_w104_kd/
  model_adapter.py       # W104 D122 native residual U-Net만
  teacher_registry.py    # no-align checkpoint 검사·freeze
  losses_rec.py          # N0/R0/R1/R2/R3
  statistics.py          # IV/GV/GC/SC/M2·valid window
  losses_stat.py         # H/T/FIX/WH/AD
  trilateral.py          # A/B sign/cos/cap/component gates
  optional_pan_sens.py   # NA-TSENS 전용, 기본 import/호출 비활성
  calibration.py         # train-only tau/lambda
  case_registry.py       # Q00–Q47, CS, VX, CTL, LONG
  evaluator_adapter.py   # native raw 평가
  manifest.py            # source/config/checkpoint/RNG lineage
```

### 16.3 Native training step의 계약

```python
# 설명용 adapter 계약이다. 아래 이름을 실제 저장소 API로 간주하지 않는다.
# optimizer에는 Student U-Net만 들어간다.
pan, lrms, gt = get_batch_with_existing_preprocess()
ms_base = existing_bicubic(lrms)
x = concatenate_pan_ms_in_verified_order(pan, ms_base)

# no-align은 warp(P, 0)가 아니라 원 tensor를 직접 전달한다.
if loss_requirements.needs_teacher:
    with no_grad():
        pred_t = ms_base + teacher_residual_unet(x)
else:
    pred_t = None

pred_s = ms_base + student_residual_unet(x)
rec = build_selected_rec_loss(pred_s, pred_t, gt, rec_config)
stat = build_selected_stat_loss(pred_s, pred_t, gt, stat_config)
edge = build_optional_signed_edge(pred_s, gt, edge_config)
loss = rec + lambda_stat * stat + lambda_edge * edge
backward_and_step(loss)
```

Wrapper가 최종 HRMS를 반환하면 M을 다시 더하지 않는다. **Teacher·Student의 forward 의미를 같은 adapter에서 통일**한다. Loss의 geometry/offset/μ 인자를 null dummy로 강제로 채우는 대신 이번 trainer의 계약에서 제거한다.

### 16.4 Teacher 필요 여부를 정확히 결정

| 구성 | Teacher forward 필요 |
|---|---|
| N0, N0+EDGE-H, N0+통계-H | 없음 |
| R1 | 필요: e_T failure map |
| R0/R2/R3, A 계열 | 필요 |
| 통계 T/FIX/WH/AD, B 계열 | 필요 |
| C-NASENS | 기본 Teacher+추가 4회 probe |

R1/WH를 “Teacher 없는 학습”이라고 표기하지 않는다. Teacher output 자체를 target으로 쓰지 않을 뿐, Teacher 기반 어려움 지도를 사용한다.

### 16.5 Gradient 경로

| 값 | Student gradient | Teacher gradient |
|---|---|---|
| e_S/e_ST의 실제 loss | 있음 | target 없음 |
| d/a/sign/cos/cap/component weight | 없음(detach) | 없음 |
| Student IV/GV/GC/SC | 있음 | 해당 없음 |
| GT/Teacher 통계 | 없음 | 없음 |
| C probe/J/q/scale | 없음 | weights gradient 없음 |
| Aligner/μ/covariance | **이번 모델에 존재하지 않음** | **존재하지 않음** |

Teacher parameter `requires_grad=False`, eval 모드. Optimizer parameter와 교집합이 없어야 한다. Student는 train 모드로 유지한다.

---

## 17. s2 실행·수치·메모리

### 17.1 실행 전 확인할 실제값

```text
repository commit / model code hash
GPU model / VRAM / driver / CUDA / framework versions
W104 D122 resolved model config
train/val/RR/FR file manifest / scene IDs / split lineage
optimizer / LR / betas / weight decay / warmup / scheduler / N
microbatch / accumulation / sample exposure definition
AMP / GradScaler / gradient clipping / TF32 / compile policy
metric code hash / intensity scale / clipping / selector
```

현재 문서만으로 s2를 특정 GPU라고 단정하지 않는다. 이전 W96/W112의 파라미터·학습시간·VRAM을 W104 실측처럼 옮기지 않는다.

### 17.2 Memory/OOM

Teacher를 먼저 no-grad 실행하고 필요한 target만 보존한다. Student activation만 backward한다. Loss와 통계는 autocast를 명시적으로 끈 FP32 영역에서 계산한다. 단순 `.float()`만으로 상위 autocast convolution 정책이 바뀌었다고 가정하지 않는다.

OOM이면 순서대로:

1. 불필요한 graph/중간 feature/시각 로그 보관을 제거한다.
2. microbatch를 줄이고 accumulation으로 유효 batch를 보존한다.
3. SC는 row/pair 단위로 계산하고 큰 unfold를 제거한다.
4. 선택적 C probe는 Student graph를 만들기 전에 순차 실행한다.
5. checkpointing/compile 변경이 필요하면 모든 대응 run의 compute policy를 동일하게 기록한다.

가변 microbatch는 sample 수로 loss를 가중하여 전체 batch 평균과 맞춘다. Batch-dependent layer가 있으면 accumulation이 큰 batch와 완전히 같다는 주장은 하지 않고 기록한다.

### 17.3 Teacher cache

첫 실행은 online target을 권한다. Cache를 쓰면 sample/crop/flip/rotation/intensity/bicubic/Teacher hash를 key에 포함한다. 네트워크가 정확히 회전·반사 equivariant라고 가정해 canonical 예측을 임의 변환해 사용하지 않는다. Gate의 Student 오차 부분은 항상 현재 Student에서 계산한다.

### 17.4 통계 수치 위험

- fp16의 제곱·상쇄·작은 sign 판정은 FP32로 승격한다.
- 공분산 음수 off-diagonal을 보존한다.
- Flat 출력에서 variance 미분이 작을 수 있다. plain reconstruction을 남기고 λ 폭증으로 해결하지 않는다.
- 학습 중 출력 clamp는 새로 넣지 않는다. 기존 evaluator export/clipping만 유지한다.
- Std/log는 원소의 정의역과 ε에 따라 의미가 달라지므로 별도 calibration한다.

---

## 18. 실행 전 검사: 결과를 만들어 쓰지 않는다

아래는 **수행할 검사 목록**이다. 이번 문서 작성으로 실제 s2 통과가 확인됐다는 뜻은 아니다.

| Gate | 확인할 것 |
|---|---|
| NA01 | Teacher/Student width104, d122, 9→8, architecture hash 일치 |
| NA02 | aligner parameter=0, native PAN warp/cached correction 호출=0 |
| NA03 | 채널 순서, MS 확대, residual base 정확히 한 번 |
| NA04 | GT가 inference forward로 전달되지 않음 |
| CK01 | 다른 width/depth를 partial load하지 않음; checkpoint strict 검사 |
| KD01 | Teacher/GT/gate에 gradient 없음, Student loss에는 있음 |
| KD02 | S=T가 같은 오답이어도 GT gradient 유지 |
| KD03 | 전체 pixel에서 S=GT이면 R3 soft0; 부분 정확 band는 A mask0 |
| KD04 | R3 β0=R1, A-BASE=R3, B-BASE=STAT-AD |
| ST01 | explicit toy와 population variance 일치, 상수통계0 |
| ST02 | GV와 GC 대각 일치, SC symmetry·음수 off-diagonal 보존 |
| ST03 | k5 공통 중심58²(train64), 통계 ROI가 recon ROI를 바꾸지 않음 |
| ST04 | S=GT의 H loss0, Student 통계 gradient live |
| TRI01 | sign/cap/cos 손계산 및 동일 GT·반대방향 예외 |
| TRI02 | component advantage+동일 sign의 중복성 확인 |
| TRI03 | mass control의 broadcast 분모와 total coefficient 일치 |
| EV01 | native raw evaluator·scene list·clipping·aggregation·selector 일치 |
| RS01 | optimizer/scheduler/scaler/RNG/data sampler resume 일치 |
| PERF01 | 실제 batch에서 forward/backward/step finite, peak memory·step time 기록 |
| CS01 선택 | δ0 기준, ±h finite difference 안정성, border support, frozen target |

모듈/수학 오류는 `IMPLEMENTATION_ERROR`, 성능 비개선은 `NEGATIVE`, 실행 전 의존성 미충족은 `BLOCKED_INPUT` 등으로 구분한다. Gate 오류 때문에 나온 성능 저하를 연구 가설의 기각으로 쓰지 않는다.

---

## 19. 중간 변경과 결과 관리

### 19.1 두 종류의 경로

**CONTROLLED:** 같은 초기값·동일 budget·고정 loss로 시작. 최종 ablation 근거.

**ADAPTIVE-PATH:** 유망한 결과를 따라 loss를 바꾸거나 continuation. 설계 탐색에 사용하되 처음부터 같은 방법으로 학습한 결과처럼 보고하지 않는다.

변경 시 parent model/optimizer/scheduler/scaler/RNG/sampler/update를 저장한다. 동일 parent에서 기존 설정 continuation과 새 child를 남긴다. 시간 제한이 없으므로 대응 continuation을 예산 때문에 생략하는 것을 기본으로 하지 않는다.

Teacher가 바뀌면 새로운 cohort·τ·stat cache를 만든다. Loss weight 변경은 config version으로 기록한다. **이번 경로 중간에 aligner를 붙이지 않는다.** 그것은 이 캠페인이 아니라 향후 integration run이다.

### 19.2 상태

```text
PENDING -> IMPLEMENTED -> UNIT_PASS -> SMOKE_PASS -> SCREENED
        -> FULL_N -> REPEATED -> CONFIRMED / NEGATIVE / REVISE_REQUIRED
별도: NOT_APPLICABLE_MAIN, DEFERRED_ALIGNMENT, OPTIONAL_NOT_SELECTED
```

지정된 finite registry의 진행 상태를 모두 남긴다. 실제로 실행하지 않은 case에 성능값을 채우지 않는다. Failed/partial checkpoint를 정상 full-N last로 이름 붙이지 않는다.

### 19.3 디렉터리 제안

```text
<ACTUAL_S2_WORK_ROOT>/S2_W104_D122_NOALIGN_KD_20260911/
  campaign_manifest.yaml
  environment.json
  architecture_manifest.json
  source_hashes.json
  teacher_registry.json
  queue.csv
  decisions.md
  initial_states/<student_seed>/
  calibration/<teacher_id>/<representation>/
  runs/<run_id>/
    config_resolved.yaml
    parent_and_phase.yaml
    train_log.jsonl
    gradient_diagnostics.jsonl
    checkpoint_metrics.csv
    scene_metrics.csv
    fitting_bins.csv
    resource_usage.json
    best_rr_val.pt
    best_raw_exploratory.pt       # 사용한 경우에만
    last_NFULL.pt
    last.pt
    predictions/
    visualizations/
  reports/
    reconstruction_ablation.md
    statistics_ablation.md
    trilateral_ablation.md
    long_fitting_and_compute.md
    negative_results.md
    final_summary.md
```

`/mnt/data`는 이 문서의 생성 위치이지 실제 s2 작업 경로가 아니다. 저장소 실제 경로를 별도 확인한다.

---

## 20. 구현자용 설정 템플릿

다음은 **새 adapter를 위한 의미 명세 YAML**이다. 현재 저장소 CLI의 검증된 옵션명이 아니며, `REQUIRED` 항목은 실행 전에 실제 manifest로 치환한다.

```yaml
campaign:
  id: S2_W104_D122_NOALIGN_KD_20260911
  server: s2
  wallclock_limit_hours: null
  protocol: NA-STRICT
  run_kind: CONTROLLED

model:
  class_path: REQUIRED_FROM_REPOSITORY
  width: 104
  depth: [1, 2, 2]
  in_channels: 9
  out_channels: 8
  task: hrms_residual
  residual_base: existing_bicubic_ms
  add_base_exactly_once: true
  aligner_enabled: false
  native_pan_warp: false
  cached_pan_correction: false
  attention: false
  pan_reconstruction: false
  pan_derived_input_channels: false

teacher:
  enabled: true                 # N0+GT-only에서는 false 가능
  architecture: same_as_student
  checkpoint: REQUIRED_T104_D122_NATIVE_NOALIGN
  checkpoint_hash: REQUIRED
  training_recipe: gt_l1_noalign
  eval_mode: true
  frozen: true
  native_input_same_as_student: true

student:
  initial_state: REQUIRED_SHARED_RANDOM_INIT
  seed: REQUIRED_FROM_BLOCK
  copy_teacher_weights: false

input:
  pan: dataset_native
  ms_upsampler: REQUIRED_EXISTING_BICUBIC_CONFIG
  channel_order: REQUIRED_VERIFIED_ORDER
  intensity_policy: REQUIRED
  random_pan_shift: false
  random_ms_shift: false
  native_corrupt_alternation: false
  aligner_only_crop: false
  augmentation: REQUIRED_MATCHED_BASELINE

loss:
  rec:
    mode: adaptive               # gt/fixed_kd/hard_only/teacher_error/adaptive
    alpha: 1.0
    beta: 0.1
    epsilon: 1.0e-6
    tau: REQUIRED_TRAIN_CALIBRATION
    detach_routing: true
    reduction: full_domain_mean
    triangle: none               # none/sign/cap/cos/bandadv
  stat:
    enabled: false
    representation: GV           # IV/GV/GC/SC/M2
    mode: AD                     # H/T/FIX/WH/AD/B_SIGN/B_CAP/B_COMPADV
    window: 5
    stride: 1
    correction: 0
    domain: final_hrms
    alpha: 1.0
    beta: 0.1
    tau: REQUIRED_IF_MODE_NEEDS_CALIBRATION
    epsilon: REQUIRED_STAT_SCALE
    lambda: REQUIRED_TRAIN_GRADIENT_CALIBRATION
    common_valid_center: true
    teacher_student_gt_independent_zscore: false
  edge:
    enabled: false
    kernel: signed_scharr_1over32
    lambda: REQUIRED_IF_ENABLED
  geometry: {enabled: false}
  offset: {enabled: false}
  shift_kd: {enabled: false}
  covariance_head: {enabled: false}
  teacher_pan_sensitivity:
    enabled: false               # NA-TSENS 별도 protocol에서만 true

training:
  optimizer: REQUIRED_FROM_W104_BASELINE
  learning_rate: REQUIRED
  weight_decay: REQUIRED
  scheduler: REQUIRED_FULL_HORIZON
  warmup_updates: REQUIRED
  full_updates: REQUIRED_N
  microbatch: REQUIRED_PROFILED
  accumulation: REQUIRED_TO_MATCH_EFFECTIVE_BATCH
  amp: REQUIRED_SHARED_POLICY
  loss_precision: float32
  teacher_forward: no_grad
  student_forward_count: 1
  eval_interval: REQUIRED_COMMON

selection:
  primary: REQUIRED_DECLARED_SELECTOR
  independent_rr_validation: REQUIRED_BOOL
  keep_exact_final_n: true
  aligned_selector: false
  fr_test_used_for_selection: REQUIRED_BOOL

evaluation:
  evaluator_hash: REQUIRED
  data_manifest: REQUIRED
  scene_ids: REQUIRED
  view: raw_original
  posthoc_alignment: false
  predict_warp_gt_or_output: false
  hqnr_aggregate: mean_of_scene_hqnr
```

### Case resolver 규칙

```text
Q00 -> REC-N0, STAT-OFF, TRI-OFF, EDGE-OFF
Q04 -> REC-R3, STAT-OFF, TRI-OFF
Q10 -> REC-R3, STAT-GV-AD
Q13 -> REC-R3의 soft를 TRI-A-SIGN으로 대체, STAT-OFF
Q17 -> REC-R3, STAT-GV-AD의 soft를 TRI-B-SIGN으로 대체
Q41 -> A-SIGN reconstruction + GV-B-SIGN statistics
모든 Q -> model.width104/depth122/noalign 강제 검사
G/offset/aligner flag가 true이면 config validation 실패
C probe true인데 protocol이 NA-STRICT이면 config validation 실패
```

---

## 21. 최종 실행 순서 요약

1. **P0:** 정확한 W104–D122 no-align 모델·native raw evaluator 확인, T00 확보.
2. **P1:** Q00–Q04를 동일 초기값·full-N으로 비교해 **KD 자체의 이득**을 먼저 확인.
3. **P2:** Q05–Q19로 GT 통계/Teacher 통계/방향 선택의 효과를 분해.
4. **P3:** Q20–Q47, 필요한 CTL/VX를 순서대로 진행. 모든 확장은 같은 backbone을 유지.
5. **장기 확인:** 유력 case와 N0/R1·GT-only 통계를 2N/4N 및 같은 compute로 비교, Student seed 반복.
6. **선택적 P4:** 입력 민감도 C-NASENS를 별도 protocol에서 검사. covariance형은 여전히 정합 연구와 분리.
7. **이후 integration:** 최종 aligner가 정해진 뒤 `KD 없음/있음 × aligner 없음/있음`의 2×2로 재검증. 이번 no-align 결과만으로 결합 성능을 예단하지 않음.

**이번 실험의 결론 형식:** “정합 모듈이 없는 동일 W104–D122에서 어떤 KD/통계 감독이 추가 fitting과 복원 품질에 기여했는가?”를 답한다. 정합 donor를 기다리거나 움직이지 않는 aligner에 covariance를 붙여 이 질문을 흐리지 않는다.

---

## 22. 확인 범위와 출처 보존

본 파일은 사용자 최신 결정과 [S0–S4]를 구분하여 작성한 **실행 명세**다. 원문을 덮어쓰지 않으며 W96/W112 과거 결과를 W104 결과로 전용하지 않는다.

- 문서 작성: 완료.
- case 재분류·수식·config 계약: 본문에 명시.
- 새 W104 통합·정확한 params/latency·VRAM: **미측정**.
- 실제 데이터 training·method 성능: **미실행**.
- 과거 reference CPU test: 원문 기록일 뿐 이번 새 통합 검증 결과 아님.
- 문서의 수식/기호/시작값은 가설의 명세이지 신규성 또는 우수성의 보장이 아님.

### 참조 문서 식별 정보

아래 SHA256은 이 문서 작성 시 로컬에서 읽은 backing file을 식별하기 위한 값이다. 원격 s2 저장소의 commit/hash를 대신하지 않는다.

| ID | 파일 | SHA256 |
|---|---|---|
| S0 | `PAN_S2_W112_D124_KD_Variance_Adaptive_Campaign_2026-09-10.md` | `482303d6adbfbdab0171c32755eb3f0333d191bcb1aa77dd1c3d41ad440ea96e` |
| S1 | `PAN_S2_W112_D124_TGeo_ABC_Addendum_2026-09-10.md` | `b7e1378739c220027a91f25d186d2b56024a82a1c684a157f28d43024d95f167` |
| S2 | `PAN_W96_D124_GT_Anchored_KD_Experiment_Plan_2026-09-09.md` | `32d3907fcfa0a89e1fb5424a006c7596d88d8b728dd9120c0d64aaf197a3ac63` |
| S3 | `PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2 (1).md` | `4ea1f157151a1b8a25033afa175f9923e33d4ee385c6643c46c2c86d6d94d64b` |
| S4 | `PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md` | `334a13fdae48faa177d839d5b04e938b283083c97edc56b67c430152f70077f3` |
