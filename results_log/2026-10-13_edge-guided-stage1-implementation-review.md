# Edge-Guided Residual PAN Aligner — Stage1 구현/검증

기준: `research_log/PANDA_Stage1_EdgeGuided_Residual_Aligner_Final_2026-10-13.md` 전체 445줄.
범위: 새 Stage1 method·데이터 연결·optimizer smoke·회귀 검증. Stage2/KD,
실험 case/queue, 본 학습, Sheets 갱신, commit/push는 수행하지 않았다.

## 1. 결론

**새 Stage1을 별도 모듈로 구현했고, 신규 49개를 포함한 총 368개 테스트가 통과했다.**

이전 scene/bGlobal 방식은 보존하되 새 경로와 checkpoint를 분리했다. 기존
Teacher48 명령이 자동으로 새 method를 실행하도록 바꾼 것은 아니다. 향후
case에서 새 builder/engine을 연결해야 한다.

이제 source-scene mapping, train-wide b 추정, frozen global b 자산이 필요 없다.
FR도 자기 입력에서 EdgeMatch를 실행하므로 이전의 b_RR/4b_RR 선택 문제는
남지 않는다. 새 설계는 **current PAN pixels 그대로**, 출력 ×4 없음이다.

미정인 것은 검색반경·동점/저텍스처 정책 및 학습 ROI/경계 설정이다.
Synthetic 테스트에 쓴 숫자를 실데이터 운영 기본값으로 등록하지 않았다.

## 2. 파일과 구현 계약

| 파일 | 구현 |
|---|---|
| `panda_method/edge_match.py` | 공통 Scharr, 입력별 정수 EdgeMatch, 공통 유효영역·실패 진단 |
| `panda_method/edge_guided.py` | Cartesian residual CNN, native/epsilon 경로, RR-only Stage1, RR/FR 추론 |
| `panda_method/edge_data.py` | 기존 H5/source SHA 및 제공 LMS 연결, QB 원 LMS, 증강, split 분리 |
| `teacher48/edge_training.py` | 명시적 model builder, optimizer step, 손실/gradient/shift 로그 반환, 재개 검증 |
| `config/edge_stage1/README.md` | API·설정 계약·현재 지원 범위 |
| `config/edge_stage1/policy.template.json` | 미정값을 null로 남긴 **실행 불가** 설정 초안 |

### Matching / spatial loss

- WV3 RGB `(4,2,1)`, QB/GF2 `(2,1,0)`로 **값 평균 후** Scharr 적용.
- `G=[abs(Sx*I),abs(Sy*I)]`, 고정 Scharr `/32`, reflect padding.
- EdgeMatch와 `L_abs`가 **동일한 Scharr 객체**를 사용한다.
- Z-score/min–max/NCC/census/gradient magnitude/learned descriptor/subpixel fit 없음.
- 명시 반경 r의 정수 **square lattice `[-r,r]²`**를 검색한다.
- signed `(dy,dx)`는 `W(P,u)[y,x]=P[y+dy,x+dx]` convention 그대로.
- PAN/LMS로만 matching한다. GT는 `L_rec/L_abs`에서만 사용한다.
- 유일한 최솟값이 `(0,0)`이면 정상 결과다. 이전 nonzero prior 제한은 없다.

### CNN / reconstruction / gradient

- 기존 GN dual stem/joint + U-Net 공통 pre-LN 64ch ResBlock 유지.
- GAP64 → **FC64→32→2**. b concatenation, alpha/beta, 방향 basis 제거.
- FC2 weight/bias zero-init → 최초 delta=0, u=b.
- CNN 입력만 `P_pre=W(P,b)`. Teacher 입력은 **`W(P_original,b+delta)`**.
- 기존 Teacher P0, BICUBIC_MS base와 backbone을 유지한다.
- Native에만 Teacher forward 1회. Native/epsilon에서 matcher 2회, CNN 2회.
- Epsilon PAN에서 `b_eps`를 다시 추정한다. b0 재사용이나 b0−epsilon 주입 없음.
- `rec + abs_rr + .001*eps_rr`; radius2 uniform-area epsilon, RR update마다 생성.
- Rec→A/T, abs→A, epsilon→shifted A. Native u0의 detach는 epsilon target에서만.
- FR/held-out native prediction은 gradient off. FR optimizer input 거부.

초기 U-Net output projection과 Aligner FC2가 모두 zero-init이므로 첫 update에서
일부 앞단 gradient가 0인 것은 정상이다. 테스트는 projection을 활성화한
후속 상태에서도 rec→A 및 feature/ResBlock gradient가 연결되는지 확인한다.

## 3. 경계 처리에서 특히 검증한 점

**일반 matching:** 모든 후보에 공통 `r+1` interior를 사용한다. `+1`은
Scharr 3×3 footprint이며 후보마다 다른 개수의 pixel을 비교하지 않는다.

**Epsilon matching:** 이미 synthetic warp된 PAN에는 border padding이 있다.
`r+1`만 쓰면 그 padding을 실제 관측 edge로 오인할 수 있어 다음을 구현했다.

1. epsilon warp의 보수적인 bicubic 4-tap valid support 계산.
2. Scharr 3×3 이웃이 모두 valid인 위치만 유지.
3. 모든 후보의 shifted support **교집합**을 구함.
4. 같은 sample의 모든 후보가 정확히 같은 Ω와 분모 `2*|Ω|`를 사용.

유효영역이 없으면 오류다. 패딩 위치에 큰 값(1e6)을 넣어도 masked matching
score/b가 변하지 않는 테스트를 포함했다. 이는 known geometry padding 제외이며
confidence weighting이나 학습된 weighting이 아니다.

`L_abs`는 별도의 **고정 ROI**를 사용한다. 예측 u에 따라 loss mask/분모를
바꾸지 않는다. `report`는 warp/Scharr support 부족을 기록하고,
`require_spatial_support`는 update를 거부한다. CNN prewarp 및 epsilon+prewarp
두 단계 support도 기록한다. **HQNR 평가 마스킹을 변경한 것이 아니다.**

## 4. 데이터·LMS 정책

- 제공 stored LMS를 읽고 기존 `2*DN/peak-1` scaling만 적용한다.
- LMS의 음수 보간 링잉을 clipping하지 않는다.
- 이미지 Hflip→Vflip→rot90 후 EdgeMatch를 실행한다. 고정 b의 증강은 없다.
- QB는 이전 사용자 결정대로 **원본 LMS + 현행 msfix LRMS**를 유지한다.
  원본/보정본의 각 source SHA를 확인하고, 읽는 행의 PAN/GT exact identity를 검사한다.
- 원본 LMS가 `U_phase(msfix MS)`와 정확히 동일하다고 입증된 것은 아니다.
  Hybrid source와 `lms_reproduction_from_active_ms=not_claimed`를 기록하며,
  근거 없는 `phase_verified` metadata는 사용하지 않는다.
- RR256/FR512 whole-image test를 실제 크기의 disposable H5→data adapter→model
  경로로 실행했다. FR에는 GT가 필요 없고 원본 영상당 shift 한 개를 낸다.
- 실제 위성 H5 전체 matching, 품질 평가, GPU 학습은 이번에 하지 않았다.

## 5. 로그·checkpoint 안전성

`EdgeStage1Trainer.step()`은 JSON 직렬화 가능한 다음 정보를 반환한다.

- rec / abs / raw epsilon / weighted epsilon / total loss.
- 실제 합산 gradient의 A/T norm, 명시 cadence의 항별 A/T gradient norm.
- Native/epsilon 각각 sample별 b, delta, u.
- 동점 수, 저텍스처/검색경계 flag, edge score, 공통 support pixel 수.
- Final warp, CNN preliminary warp, epsilon 두 단계 warp support.

반환된 기록을 실제 run 파일/Sheet에 저장하는 것은 **후속 campaign logger의
책임**이다. 이번 method engine이 자동 로그 위치나 실험 ID를 만들지는 않는다.
항별 gradient 진단은 추가 backward 비용이 있지만 실제 update를 바꾸지 않는
것을 로그 cadence가 다른 동일 모델의 bitwise update 비교로 확인했다.

Checkpoint는 method/matcher/source/ROI/Teacher 정의, epsilon RNG, A/T,
optimizer class와 **순서 있는 parameter 이름**, logging cadence, update count를
포함한다. 기존 FC66 scene/global뿐 아니라 모양이 비슷한 legacy GN64 Aligner도
거부한다. 실제 campaign 재개에는 별도 data-order/augmentation RNG와 scheduler
저장이 필요하다. HQNR 우선 best 선정 및 공식 metric 구현은 변경하지 않았다.

## 6. 테스트 결과와 비용 범위

| 검증 | 결과 |
|---|---:|
| 신규 shared Scharr / EdgeMatch | 28 PASS |
| 신규 residual CNN / Stage1 | 13 PASS |
| 신규 H5 / optimizer / logger / restart | 8 PASS |
| 전체 panda_method | 143 PASS |
| 기존 teacher48 전체 | 206 PASS |
| 기존 ta2 model/loss | 19 PASS |
| 전체 합계 | **368 PASS** |

Scharr `/32`는 cv2와도 대조했다. D4 검사는 unique winner에 대한 것이다.
동점 tie-break에 lexicographic 순서를 쓰는 경우 완전한 D4 대칭성은 보장하지
않으며 manifest에 명시했다. Compileall 및 tracked diff whitespace 검사도 확인했다.

CPU-only synthetic 비용 점검: Torch2.4, FP32, single thread, B=1, WV3 C=8,
warm-up 후 5회 median. 아래 반경은 **테스트 설정이며 운영 확정값이 아니다**.

| 입력 | 반경 / 후보 수 | EdgeMatch median |
|---|---|---:|
| 64² RR | r=2 / 25 | 1.30 ms |
| 256² RR | r=2 / 25 | 6.15 ms |
| 512² FR | r=3 / 49 | 36.48 ms |

이는 native EdgeMatch만의 비용으로 CNN/U-Net/epsilon/데이터 읽기/GPU 성능을
포함하지 않는다. 전체 inference 비용이나 실험 시간 예측으로 사용할 수 없다.
Residual Aligner parameters는 C4 **104,882**, C8 **105,458**이다.

## 7. 실제 실행 전 남은 정책

1. **RR/FR 검색반경**: current PAN pixels 단위로 각각 명시. FR metric을 보고
   반경을 선택하지 않는다. 이번 구현 후보 모양은 square lattice다.
2. **동점/저텍스처/검색경계 처리**:
   - tie: `error` 또는 `min_norm_then_lexicographic`.
   - low texture: 명시 threshold와 `error` 또는 기록된 zero fallback.
   - search boundary: `error` 또는 `report`.
   - empty support/nonfinite는 항상 실패.
3. **Loss ROI/CNN view와 support 정책**, 항별 gradient 기록 cadence.
4. **후속 case의 architecture/seeds/optimizer/schedule/evaluation 연결**.

FR b 단위 정책이나 global-prior asset 생성은 더 이상 남은 결정이 아니다.
반대로 단위·부호가 맞는다고 실제 정합 정확도가 검증된 것은 아니다. Raw
cross-modal edge L1은 spectral contrast/phase/국소 변형 때문에 틀린 정수
후보를 고를 수 있다. Integer guidance + residual의 실제 효과는 후속
train-only 설정 결정과 별도 실험으로 평가해야 한다.
