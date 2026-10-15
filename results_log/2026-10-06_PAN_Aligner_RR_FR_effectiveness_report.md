# PAN Aligner 실효성 검증: RR → FR 스케일 전이, 실제 보정량, ε consistency

작성일: 2026-10-06. 대상: WV3·QB·GF2의 기존 학습 완료 모델. **재학습·운영 코드 변경·checkpoint 재선택 없이 수행한 진단**이다. 수치와 그림은 아래에 명시한 checkpoint와 실제 데이터에서 새로 계산했다.

## 1. 먼저 결론

**“FR에서 shift를 코드상 1/4로 줄여 적용한다”는 것은 아니다. 그러나 현재 Aligner가 FR의 기하학적 misalignment를 충분히 복원한다고 보기도 어렵다.** 확인된 문제는 단순한 배율 하나가 아니라 다음 세 가지다.

1. **스케일 보정이 명시되어 있지 않다.** 예측값은 RR에서도 FR에서도 그 입력 PAN의 픽셀 단위 그대로 적용된다. 동일 물리적 변위를 표현하려면 FR에서 RR의 4배 픽셀이 필요한 조건이 있지만, 모델에는 이를 강제하는 장치가 없다. 같은 FR 영상을 4배 축소한 통제 실험에서도 예측 크기는 약 1.04–1.26배만 변했다. 단, 이것을 “실제 오차의 정확히 1/4만 보정”으로 해석할 수는 없다.
2. **알려진 추가 변위에도 매우 약하게 반응한다.** 선택 Student의 FR 변위 응답 gain은 WV3 0.029, QB 0.020, GF2 0.140이었다. 이상적인 gain은 1이다. RR에서도 각각 0.038, 0.042, 0.204에 그친다. 따라서 FR에서만 생기는 문제도 아니다. 실제 donor Teacher에서도 비슷한 양상이다.
3. **현재 Aligner 입력 MS에는 별도의 약 0.5px/축 리사이즈 위상이 들어 있다.** RR에서 `bicubic(MS)`를 GT에 맞추기 위한 sampling shift는 세 데이터셋 모두 약 `(-0.5, -0.5)`였다. 반면 데이터셋의 native LMS는 약 `(0, 0)`이다. **PAN과 bicubic MS 사이의 차이를 모두 센서 misalignment로 간주하면 안 된다.**

실효성은 “PAN이 움직이는가”, “실제 정합이 좋아지는가”, “최종 융합 영상 품질이 좋아지는가”를 분리해야 한다. 현재 모듈은 실제로 PAN을 움직이고 일부 정합 proxy를 개선하지만, **물리적 global registration을 충분히 수행하는 범용 Aligner라는 주장은 이번 결과로 뒷받침되지 않는다.** 단순히 출력 shift에 4를 곱하는 조치 역시 공통 해법으로 확정할 수 없다.

**GF2에서는 현재 Aligner의 최종 품질 기여가 뚜렷하다.** 이동을 끄면 RR ERGAS가 0.560→0.609, FR HQNR이 0.95470→0.93718로 나빠진다. GF2의 FR-only ×4는 추가 검증할 가치도 있다. 반면 WV3는 기여가 작고 QB는 현재 이동이 FR HQNR을 낮춘다. 최종 영상 비교는 §7에 정리했다.

## 2. 어떤 모델·데이터를 조사했는가

### 2.1 선택 모델

| 데이터셋 | 선택 근거 | Student / step | 기존 공식 기록 HQNR / SCC / ERGAS |
|---|---|---|---|
| WV3 | 기존 논문용 선택 기록 | FH20R1, PLH, W104-D122, SS73101 / 38,380 | 0.958733 / 0.987893 / 2.056060 |
| QB | 기존 논문용 선택 기록 | QG40, PLH, W104-D122, C3 donor, SS82001 / 47,470 | 0.925114 / 0.983421 / 3.561287 |
| GF2 | 로컬 G20 s1의 4개 Student `raw_max` 중 최고 HQNR | G20, R0, PLH, W104-D122, SS93002 / 45,450 | 0.954697 / 0.993398 / 0.560561 |

모든 서버·모든 후속 캠페인의 전역 최고 모델이라는 뜻은 아니다. 특히 GF2는 로컬에서 checkpoint와 원본 데이터를 검증할 수 있는 G20 s1 대표 모델이다. 세 모델 모두 실제 Aligner를 포함한다. WV2의 최근 practical-validation 모델은 Aligner가 없는 별도 설정이므로 이번 learned-Aligner 실험에서 제외했다.

전체 run ID와 SHA-256:

| 데이터셋 | run ID | checkpoint SHA-256 |
|---|---|---|
| WV3 | `FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1` | `3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50` |
| QB | `QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1` | `548d0ed2ccdc968d319e850e1470d7068f9321a9f0cc18e1794807f5cfd3399d` |
| GF2 | `G20_GF2_S_S1_R0_TS91001_SS93002_BASE_PLH_W104_D122_FRESH50_v1` | `effd4b28cb2f72cc7feaaee35999a45a70cda4fc2eefbd2f703c17efc1e65fa5` |

さらに各 Student의 resolved config가 지정하는 **정확한 donor Teacher**의 Aligner도 조사했다. 임의의 다른 Teacher로 대체하지 않았다.

- WV3: `FH12_S1_T_P0_W112_D123_WV3_S71001_FRESH50_v1`, 50,000 step.
- QB: `QG40_QB_TB_C3_P0_W112_D123_TS81002_C3_FRESH50_v1`, 50,000 step.
- GF2: R0에 보존된 `QGBASE_GF2_TA_P0_W112_D123_TS91001_FRESH50_v1` donor bytes. `work_dir/_g20/s1/references/R0/imported_bytes/assets/teacher_checkpoint.safetensors`.

Teacher SHA와 경로는 [WV3](2026-10-06_pan-aligner-effectiveness/WV3_teacher.json), [QB](2026-10-06_pan-aligner-effectiveness/QB_teacher.json), [GF2](2026-10-06_pan-aligner-effectiveness/GF2_teacher.json)에 저장했다.

### 2.2 표본·좌표·실행 환경

| 구분 | 데이터셋당 표본 | PAN | 입력 LRMS | GT | Aligner가 보는 내부 영역 |
|---|---:|---|---|---|---|
| Train probe | 전체 학습 데이터에서 균등 간격 20개, augmentation 없음 | 64² | 16² | 64², probe 입력에는 미사용 | 56² |
| 공식 RR test | **20장 전부** | 256² | 64² | 256² | 248² |
| 공식 FR test | **MAT20에서 변환한 20장 전부** | 512² | 128² | 없음 | 504² |
| 통제된 paired 축소 | 위 FR 20장 각각을 동일 FOV로 축소 | 128² | 32² | 없음 | 120² |

- WV3 8밴드/2047 DN, QB 4밴드/2047 DN, GF2 4밴드/1023 DN. QB 학습 probe는 실제 사용된 `msfix` 자료다. RR test는 기존 원본 RR20이다.
- RR·FR H5와 학습 당시 LPAN cache의 SHA를 manifest와 대조하여 **6개 split 모두 일치**를 확인했다.
- RR20과 FR20을 번호가 같은 지리적 대응 장면으로 가정하지 않았다. 따라서 두 집합의 평균을 나눠 물리적 스케일 법칙을 증명하지 않는다.
- CPU FP32로 새로 추론했다. 이번 세션에서는 NVIDIA 드라이버가 GPU에 연결되지 않아 CUDA 측정은 수행하지 않았다. 기존 GPU 결과와 bitwise 일치를 주장하지 않는다.
- 코드 기준 HEAD: `a2705bd1b92ed9a7f0c54a6b4b9ee4d7b82ea1a5`. 기존 dirty worktree는 보존했다. 실제 읽은 주요 코드의 [SHA 목록](2026-10-06_pan-aligner-effectiveness/source_hashes.json)을 별도로 남겼다.
- 공식 candidate 기록과 대조한 Aligner·warp·frontend·backbone·RR/FR evaluator의 해당 소스 파일 SHA도 일치한다. [산출물 검증 기록](2026-10-06_pan-aligner-effectiveness/artifact_validation.json) 참조. 다만 이는 전체 외부 라이브러리와 GPU 실행까지 동일하다는 의미는 아니다.

## 3. “FR에서 1/4만 보정” 가설을 코드로 풀어보면

### 3.1 예측값과 warp 좌표

[pa/aligner.py](../pa/aligner.py)의 출력은 `c=(dy, dx)`이며 **현재 입력 PAN의 HR 픽셀 단위**다. [pa/warp.py](../pa/warp.py)는 다음을 수행한다.

```text
P_aligned[y,x] = P[y + dy, x + dx]
grid_x = 2 * (x + dx + 0.5) / W - 1
grid_y = 2 * (y + dy + 0.5) / H - 1
grid_sample(..., mode="bicubic", padding_mode="border", align_corners=False)
```

`/W`, `/H`는 grid_sample용 정규화일 뿐, 변위를 1/4로 줄이는 연산이 아니다. 역으로 입력 픽셀 좌표로 환산하면 정확히 `x+dx`, `y+dy`가 된다. `dx>0`은 오른쪽 위치에서 읽는 것이므로 영상 내용은 왼쪽으로 이동한다. `dy>0`이면 내용이 위로 이동한다. 이 부호와 normalized-grid 의미는 [PyTorch 공식 문서](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.grid_sample.html)와 일치한다.

64²·256²·512²에서 동일한 정수 shift `(2,-3)`를 적용한 좌표 ramp 검사는 모두 최대 오차 0이었다. 즉, **이미지가 커진다고 동일한 예측값의 픽셀 이동량이 줄어들지는 않는다.** [단위검사 원시 결과](2026-10-06_pan-aligner-effectiveness/unit_tests.json).

### 3.2 사용자 가설이 성립할 수 있는 조건

동일 지상 변위가 유지되고 RR 생성의 리샘플링 위상이 일치한다면:

```text
1 RR PAN pixel = 4 native FR PAN pixels
따라서 d_FR = 4 × d_RR
```

그런데 네트워크가 FR에서도 `c_FR ≈ c_RR`만 출력한다면, 필요한 **물리적 거리 보정의 약 1/4**에 해당한다. 이 조건부 우려는 타당하다. 하지만 코드만으로 `c_FR=c_RR` 또는 `c_RR=d_RR`를 보장할 수 없으므로 “항상 정확히 1/4”라고 단정할 수는 없다.

여기서 **RR 256² → FR 512²의 배열 크기 2배**와 **GSD의 4배 관계**는 별개다. 학습 patch 64²와 FR 512²의 크기 차이 8배 역시 변위 배율 8이라는 뜻이 아니다.

### 3.3 현재 구조에 없는 보장

Aligner는 PAN/MS stem, stride-2 convolution 두 번, residual block, global average pooling, MLP로 구성된다. 입력마다 채널별 공간 평균/분산으로 표준화하며, 외곽 4px를 제외한 **원래 해상도**를 본다. FR을 학습 크기로 resize해서 예측한 후 좌표를 복원하는 경로가 아니다.

이 구조에는 센서 GSD 입력, RR/FR 구분, cross-scale equivariance loss, `×4` 좌표 복원 규칙이 없다. GAP이 입력 크기 자유도를 주기는 하지만 **스케일에 맞는 변위 보정까지 보장하지 않는다**. 다만 이 구조적 가능성과 실제 관측된 저응답의 인과관계를 이번 실험 하나로 확정하지는 않는다.

## 4. 데이터셋별 misalignment는 어느 정도인가

### 4.1 반드시 구분해야 하는 세 기준

1. **RR PAN ↔ GT:** RR 정답 MS와의 구조적 정합. 이번 자료에서 실제 목표 좌표계를 확인하는 가장 직접적인 proxy다. 그래도 PAN과 MS의 분광 응답 차이가 있으므로 물리적 변위 정답은 아니다.
2. **PAN ↔ native LMS:** 원본 데이터셋이 제공한 고해상도 격자의 LMS와 비교한다. FR에는 GT가 없으므로 이 역시 proxy이며, blur·MTF·국소 시차 영향을 받는다.
3. **PAN ↔ 모델의 bicubic MS:** 현재 Aligner가 실제로 보는 입력 간 정합이다. 센서 차이에 더해 모델의 upsampling 위상이 포함된다.

아래 수치는 모두 **장면별 `(dy,dx)`의 L2 크기를 구한 뒤 20장 평균**이다. 평균 벡터의 norm이 아니다. 단위는 각 설정의 현재 PAN px이다.

| 데이터셋 | RR: PAN↔GT | RR: PAN↔native LMS | RR: PAN↔bicubic MS | FR: PAN↔native LMS | FR: PAN↔bicubic MS |
|---|---:|---:|---:|---:|---:|
| WV3 | 0.371 | 0.368 | 0.600 | 1.106 | 1.408 |
| QB | 0.537 | 0.522 | 0.754 | 2.579 | 2.686 |
| GF2 | 0.439 | 0.462 | 0.839 | 1.390 | 1.529 |

**읽는 법:** QB의 FR은 여기서 검사한 세 데이터셋 중 global translation proxy가 가장 크다. WV3도 “물리적 오정렬이 전혀 없다”고 할 수 없다. GF2의 bicubic-input 차이는 RR GT 기준 차이보다 거의 두 배여서 입력 위상 교란을 특히 주의해야 한다.

RR GT 기준 값을 단지 단위만 FR PAN px로 환산하면 WV3 1.485, QB 2.149, GF2 1.756px이다. 이는 **비대응 RR/FR 장면의 실제 변위가 같다는 주장도, FR 실측치도 아니다.** FR native-LMS proxy와 정확히 4배 관계가 아닌 것을 scale bug의 증거로 사용하지 않았다.

![장면별 정합 proxy 분포](2026-10-06_pan-aligner-effectiveness/01_native_registration.png)

그림의 FR 열은 **현재 모델의 bicubic 입력 기준**이다. native LMS 기준 수치는 위 표 및 [민감도 원시 결과](2026-10-06_pan-aligner-effectiveness/reference_sensitivity.json)를 함께 봐야 한다.

### 4.2 측정 방법과 신뢰 한계

- PAN과 MS 각 밴드를 동일 HR 격자에서 비교했다. 입력 MS/LMS 비교는 PAN Gaussian σ=1.98, MS σ=0.7; RR GT 비교는 양쪽 σ=0.8로 평활화했다. 이는 **진단용 bandwidth matching 근사이지 공식 센서 MTF 인증이 아니다**.
- Scharr x/y gradient의 정규화 상관으로 `[-8,+8]`px 정수 이동을 전수 탐색한 뒤, 3×3 이웃의 2차 곡면으로 subpixel peak를 추정했다. 고정 16px 내부 영역을 사용했다.
- 경계 peak가 아닌 밴드 중 상관이 가장 높은 3개를 선택하고 변위의 성분별 중앙값을 사용했다. 이 선택은 learned shift와 독립적이며, 센서별로 수동 최적화하지 않았다. 주 분석의 선택 밴드에는 탐색 경계 hit가 없었다.
- 별도로 intensity-NCC도 계산했다. gradient/intensity 추정 벡터의 평균 차이는 RR GT 기준 WV3 0.129, QB 0.032, GF2 0.068px; FR bicubic 기준 0.267, 0.079, 0.117px였다. 특히 WV3 FR의 소수점 이하 작은 차이는 과해석하면 안 된다.
- 같은 영상에 알려진 이동을 넣은 부호/회복 검사에서 최대 오차 0.038px였다. **이 테스트가 실제 cross-spectral 정합 오차까지 0.038px 이하라고 보증하는 것은 아니다.**
- RR GT proxy 평균의 장면 bootstrap 95% 구간은 WV3 [0.323,0.427], QB [0.503,0.576], GF2 [0.369,0.510]px였다. 이는 해당 20장 내 표본 변동이며 센서 전체의 신뢰구간이 아니다. 지리적으로 독립적인 20개 관측이라고 확인한 것도 아니다.

### 4.3 중요한 발견: bicubic MS의 반 픽셀 위상

PAN을 아예 사용하지 않고, **같은 밴드의 upsampled MS와 GT만 비교**했다.

| 데이터셋 | `bicubic(MS)`를 GT에 맞추는 평균 `(dy,dx)` | native LMS를 GT에 맞추는 평균 `(dy,dx)` |
|---|---|---|
| WV3 | (-0.502, -0.499) | (-0.002, +0.001) |
| QB | (-0.510, -0.495) | (-0.011, +0.005) |
| GF2 | (-0.500, -0.502) | (-0.002, +0.001) |

따라서 현재의 Aligner 입력 참조와 RR GT는 완전히 같은 위상이라고 볼 수 없다. `align_corners=False`의 4배 upsample 중심 매핑과 source decimation phase의 조합으로 해석할 수 있는 결과다. **측정된 위상 차이 자체는 확인됐지만, 모든 upstream 데이터 생성 경로의 원인을 완전히 역추적한 감사는 아니다.**

![동일 밴드 위상 대조](2026-10-06_pan-aligner-effectiveness/08_upsampling_phase.png)

이 발견 때문에 “PAN–입력 MS NCC가 올랐으니 센서 misalignment를 올바르게 고쳤다”는 추론은 성립하지 않는다. 또한 GT가 없는 FR에서 같은 반 픽셀 보정이 정답이라고 곧바로 강제해서도 안 된다.

## 5. Aligner가 실제로 얼마나 움직이고, 무엇을 개선하는가

### 5.1 실제 적용된 이동량

| 데이터셋 | Student RR 평균 shift | Student FR 평균 shift | Teacher RR 평균 shift | Teacher FR 평균 shift |
|---|---:|---:|---:|---:|
| WV3 | 0.039px | 0.081px | 0.035px | 0.075px |
| QB | 0.041px | 0.062px | 0.043px | 0.056px |
| GF2 | 0.296px | 0.344px | 0.273px | 0.296px |

예측은 사용되지 않고 버려지는 값이 아니다. [FH12 frontend](../fh12/model.py)와 [QG40 frontend](../qg40/model.py)에서 PAN과 upsampled LPAN을 **동일 grid로 함께 warp**하고, `HPAN = warped PAN - warped LPAN`을 만든 뒤 U-Net으로 전달한다. MS base/GT를 같이 움직이지 않는다. 출력은 기존 MS base에 residual을 더하는 좌표계다.

![FR 정합 proxy와 실제 적용 벡터](2026-10-06_pan-aligner-effectiveness/03_fr_vectors.png)

주황점은 PAN–bicubic MS proxy, 파란점은 적용된 sampling shift다. 벡터를 임의로 확대하지 않았다.

### 5.2 실제 warp 후 다시 측정한 잔차

벡터를 단순히 빼서 만든 값이 아니라, **실제 bicubic warp된 PAN에 registration estimator를 다시 적용**했다. `α`는 기존 예측값에 곱하는 진단용 배율이다.

| 기준 / 데이터셋 | 이동 없음 α=0 | 현재 α=1 | 4배 α=4 | α=1에서 잔차 감소한 장면 |
|---|---:|---:|---:|---:|
| RR GT / WV3 | 0.371 | 0.360 | 0.373 | 10/20 |
| RR GT / QB | 0.537 | 0.525 | 0.504 | 17/20 |
| RR GT / GF2 | 0.439 | 0.404 | **1.056** | 13/20 |
| FR bicubic / WV3 | 1.408 | 1.279 | 1.088 | 20/20 |
| FR bicubic / QB | 2.686 | 2.603 | 2.392 | 20/20 |
| FR bicubic / GF2 | 1.529 | 1.165 | 0.331 | 19/20 |

FR native LMS를 참조로 바꿔도 현재 이동은 일부 잔차를 줄인다. WV3 `1.106→0.998`, QB `2.579→2.493`, GF2 `1.390→1.107`px이다. 단, 4배 이동 후 GF2 잔차는 native LMS 기준 **0.875px**이고, bicubic 기준 **0.331px**이다. 참조 위상 선택에 따라 해석이 크게 달라진다.

RR GT 기준 현재 이동의 평균 개선은 WV3 0.011, QB 0.012, GF2 0.035px에 불과하다. 장면 bootstrap 잔차 변화 구간은 WV3 [-0.025,+0.002], QB [-0.017,-0.008], GF2 [-0.099,+0.033]px였다. **WV3/GF2의 작은 평균 개선을 확정적 센서 정합 개선으로 주장하기는 어렵다.**

또한 bicubic interpolation이 edge 모양과 blur도 바꾸므로 추정 잔차 감소량이 적용 shift norm과 정확히 같지 않다. 소수점 셋째 자리까지 적었다고 그만큼의 물리적 정확도를 뜻하지 않는다.

![warp 후 정합 잔차](2026-10-06_pan-aligner-effectiveness/04_residual_registration.png)

## 6. ε shifting은 무엇을 학습시키며, 실제로 잘 되는가

### 6.1 이 모델 계열의 정확한 학습 방식

선택한 Teacher 계열의 [FH12 loss](../fh12/losses.py), [QG40 loss](../qg40/losses.py), [G20 loss](../g20/losses.py)는 아래 구조다.

```text
c0 = A(P, bicubic(MS))
Pε = W(P, ε)                         # synthetic warp, no gradient
cε = A(Pε, bicubic(MS))
Loffset = mean(abs(cε + ε - stopgrad(c0)))

LTeacher = Lreconstruction + λ × Loffset
```

- ε는 현재 PAN 픽셀 단위의 반지름 2 원판 안에서 **면적 균등**으로 뽑는다: `r=2√u`, `θ=2πv`, `(dy,dx)=(r sinθ,r cosθ)`.
- 0-based 홀수 update에서만 offset loss를 더한다. native reconstruction은 매 update 수행한다.
- WV3/GF2 donor의 λ는 `1e-4`, QB C3 donor는 `3e-4`다.
- **이 Teacher 학습에서는 synthetic PAN을 U-Net에 넣지 않는다.** ε branch는 Aligner의 offset consistency용이다. 과거 `pa/offset.py`의 PO 학습에는 synthetic reconstruction 경로도 있으므로 두 구현을 혼동하면 안 된다.
- 선택한 Student는 Teacher A를 복제한 뒤 trainable하게 유지한다. 그러나 Student A의 직접 목적함수는 q-weighted hard reconstruction이고, Teacher의 ε-offset loss를 계속 적용하는 구조가 아니다.

예를 들어 ε의 dx=+2이면 PAN 내용이 왼쪽으로 2px 이동한다. 이상적인 Aligner는 기존보다 dx를 -2만큼 바꿔야 한다. 즉:

```text
cε - c0 = -ε
```

이때 추가 이동이 상쇄된다. 실제 이산영상에서는 두 번의 bicubic resampling이 단 한 번의 합성 shift와 정확히 같지는 않으며, blur와 경계 support를 고려해야 한다.

### 6.2 이 loss가 보장하지 않는 것

Offset consistency는 **같은 스케일에서 알려진 추가 이동을 감지하는 관계**를 학습시킨다. 절대 misalignment 정답을 주는 loss가 아니다. 예측에 일정한 bias가 붙어도 차분에서는 상쇄된다. 또한 `c_FR=4c_RR` 관계나 cross-sensor 정확도를 직접 학습시키지 않는다.

따라서 “ε loss를 넣었으니 실제 센서 정합을 완전히 해결했다”도, “ε 반지름이 2이니 출력이 반드시 ±2 이내다”도 아니다. 이 Aligner 출력에는 해당 hard clamp가 없다.

### 6.3 알려진 변위를 넣은 실측

각 장면에 상하좌우 4방향 × 반경 0.25, 0.5, 1, 2px = 16개 변위를 넣었다. 데이터셋당 Train 20 + RR 20 + FR 20장을 각각 검사했다. 아래 gain은 다음의 방향 투영 평균이다.

```text
gain = -Σ[(cε-c0)·ε] / Σ[ε·ε]
```

정확한 보정은 1, 입력 이동에 무반응이면 0이다. **이 gain은 주입한 추가 변위에 대한 반응률이지 native 물리적 오차의 교정률이 아니다.**

| 모델 / 데이터셋 | Train 64² | RR 256² | FR 512² |
|---|---:|---:|---:|
| Student WV3 | 0.043 | 0.038 | **0.029** |
| Teacher WV3 | 0.040 | 0.034 | 0.026 |
| Student QB | 0.043 | 0.042 | **0.020** |
| Teacher QB | 0.042 | 0.042 | 0.019 |
| Student GF2 | 0.199 | 0.204 | **0.140** |
| Teacher GF2 | 0.170 | 0.174 | 0.118 |

Student FR의 offset consistency MAE는 WV3 0.458, QB 0.463, GF2 0.411px다. 동일한 AXIS16 probe에서 **완전히 무반응인 상수 predictor의 MAE가 0.46875px**다. 특히 WV3·QB는 무반응 기준에서 크게 벗어나지 않는다. Teacher에서도 유사하므로 이번 결과만으로 “Student가 KD 과정에서 잊었다”고 결론내릴 수 없다.

![Teacher/Student의 알려진 추가 변위 반응](2026-10-06_pan-aligner-effectiveness/02_offset_response.png)

λ가 작고 reconstruction과의 목적 충돌이 있다는 것은 가능한 설명이다. 그러나 λ 증량·gradient 크기 비교·재학습 실험을 하지 않았으므로 이것을 확정 원인이라고 쓰지는 않는다.

### 6.4 RR/FR 비대응 문제를 피한 동일-FOV 통제

FR PAN과 FR LRMS 각각을 Gaussian σ=1.98, 41tap, replicate padding, `[2::4,2::4]`로 축소했다. PAN 512→128, LRMS 128→32가 되므로 PAN/MS의 내부 4배 비율은 유지된다. **공식 Wald RR를 재생성했다고 주장하는 것이 아니라, 동일 FOV의 scale 반응을 보기 위한 통제 자료**다.

| Student | native FR 평균 shift | paired 축소 평균 shift | 두 크기 비율 FR/reduced | 알려진 변위 gain: FR / reduced |
|---|---:|---:|---:|---:|
| WV3 | 0.081 | 0.065 | 1.244 | 0.029 / 0.053 |
| QB | 0.062 | 0.060 | 1.044 | 0.019 / 0.044 |
| GF2 | 0.344 | 0.273 | 1.260 | 0.138 / 0.273 |

마지막 열은 **FR에 ±2px를 먼저 주입한 뒤 같은 영상을 축소**하여 reduced ±0.5px와 비교했다. 두 스케일 모두 이상적인 gain은 1이다. FR에서는 reduced보다 약하게 반응하며, 둘 다 이상적 보정에 크게 못 미친다.

크기 비율 1.04–1.26은 “자동으로 4배 scaling한다”는 주장을 지지하지 않는다. 다만 native 예측에 위상/bias가 섞이고 Gaussian surrogate가 공식 RR와 같지 않으므로, 비율을 4로 나눈 26–31%를 실제 물리적 보정률로 부르면 안 된다. 원시 Student 진단 JSON에는 ±4px의 범위 밖 보조 probe도 있지만, 본문 판정은 학습 반지름 이내인 **±2px paired probe**에 근거했다.

![같은 FOV의 스케일별 변위 반응](2026-10-06_pan-aligner-effectiveness/06_paired_scale.png)

## 7. 최종 영상 품질: 같은 checkpoint에서 이동만 변경

이 절은 α=0/1/4의 20장 전체 재추론 결과를 사용한다. α=0은 Aligner를 제거해 새로 학습한 모델이 아니라, **공동 학습된 동일 checkpoint에서 sampling shift만 0으로 만든 counterfactual**이다. α=4 역시 재학습하지 않은 진단이며 새 배포 설정으로 권장하는 것이 아니다.

RR은 기존 공식 함수의 `20:-21` support에서 ERGAS·SCC를 계산한다. FR은 기존 `FRMetrics`의 **native PAN/LMS reference, full512, 장면별 HQNR 평균**이다. masking·aligned-reference HQNR로 바꾸지 않았다. 입력/LPAN은 기존 cache, 출력 clip은 `[-1,1]` 후 DN 복원이다.

### 7.1 전체 20장 재계산

| 데이터셋 | 배율 | RR ERGAS ↓ | RR SCC ↑ | FR HQNR ↑ | FR Dλ ↓ | FR Ds ↓ |
|---|---:|---:|---:|---:|---:|---:|
| WV3 | 0 | 2.067932 | 0.987673 | 0.958689 | 0.016017 | 0.025718 |
| WV3 | **1, 현재** | **2.055864** | **0.987894** | **0.958735** | 0.015799 | 0.025884 |
| WV3 | 4 | 2.091507 | 0.987360 | 0.957364 | 0.015377 | 0.027684 |
| QB | 0 | 3.565844 | 0.983354 | **0.928358** | 0.045272 | 0.027778 |
| QB | **1, 현재** | **3.560986** | **0.983423** | 0.925102 | 0.044298 | 0.032126 |
| QB | 4 | 3.602092 | 0.983049 | 0.907663 | 0.041953 | 0.052540 |
| GF2 | 0 | 0.608522 | 0.992539 | 0.937179 | 0.020322 | 0.043357 |
| GF2 | **1, 현재** | **0.560472** | **0.993400** | 0.954702 | 0.019865 | 0.025927 |
| GF2 | 4 | 1.003761 | 0.978515 | **0.958617** | 0.019630 | 0.022223 |

작은 차이를 숨기지 않기 위해 이 표만 소수점 여섯 자리까지 표시했다.

### 7.2 해석

- **WV3:** 현재 이동은 α=0 대비 RR ERGAS를 약 0.58% 개선한다. FR HQNR 차이는 +0.000047로 매우 작다. ×4에서는 RR·FR 모두 나빠진다. “FR 보정량이 부족하니 4를 곱하면 된다”는 설명으로 맞지 않는다.
- **QB:** 현재 이동은 RR ERGAS를 약 0.14% 개선하지만, FR HQNR은 0.928358→0.925102로 **하락**한다. ×4이면 0.907663으로 더 떨어진다. PAN–MS 정합 proxy는 개선되는데도 Dλ 개선보다 Ds 악화가 커진다. 정합 proxy와 최종 HQNR이 동일한 목적함수가 아님을 보여준다.
- **GF2:** 현재 이동은 α=0 대비 RR ERGAS 약 **7.90% 개선**, FR HQNR **+0.017523**을 제공한다. 동일 checkpoint 안에서 실질적인 효과가 있다. FR ×4는 HQNR을 추가로 +0.003915 개선하지만, RR에도 ×4를 적용하면 ERGAS가 약 **79.09% 악화**한다.

장면별 paired bootstrap으로 본 α=1−α=0의 FR HQNR 변화 구간은 WV3 [-0.000312,+0.000428], QB [-0.005334,-0.000939], GF2 [+0.014858,+0.020098]이었다. WV3의 미세한 FR 개선은 확정적인 효과로 보기 어렵다. QB의 RR ERGAS 개선도 20장 중 9장에서만 나타나며, paired 구간이 0을 포함한다. 이 구간 역시 고정 checkpoint·해당 20장 안의 변동만 나타내며 seed 변동이나 독립 지리 표본을 대체하지 않는다.

**중요:** 마지막 결과가 GF2의 **FR에만** ×4를 적용하는 방식을 반증하지는 않는다. RR α=1 / FR α=4로 운영하면 RR 악화는 발생하지 않는다. 이 FR-only calibration은 GF2에서 별도 검증할 가치가 있다. 다만 이번 FR20은 이미 모델 선택에도 쓰인 자료이고, 물리적 변위 정답도 없으므로 독립 검증 없이 “정확한 기하학 보정”이나 새로운 공식 best로 채택하지 않는다. WV3/QB에는 같은 보정법을 일반화할 근거가 없다.

### 7.3 기존 기록과의 수치 일치 수준

α=1 재계산과 기존 공식 기록의 차이는 WV3 ERGAS -0.000196 / HQNR +0.000003, QB ERGAS -0.000302 / HQNR -0.000012, GF2 ERGAS -0.000088 / HQNR +0.000005였다. CPU·라이브러리·부동소수점 연산 순서가 기존 실행과 동일하지 않으며, 이 차이의 원인을 bitwise 감사로 확정하지 않았다. 따라서 §2의 기존 기록과 §7의 새 재계산을 별도로 표기했다. 배율 간 비교는 모두 동일한 이번 실행 조건에서 수행했다.

![동일 checkpoint의 이동 배율별 최종 지표](2026-10-06_pan-aligner-effectiveness/05_frozen_checkpoint_quality.png)

이 비교는 선택 checkpoint의 동작을 설명하지만 Aligner를 추가한 학습의 순수 기여도를 증명하지 않는다. 그 주장은 동일 시드·예산으로 학습한 no-aligner 모델과의 별도 대조가 필요하다. 작은 HQNR 차이에 학습 seed 변동까지 포함한 유의성을 부여하지 않는다.

## 8. 실제 영상 시각화

아래 그림은 생성형 이미지가 아니다. 실제 H5 영상과 실제 warp를 사용했다. 데이터셋별 RR/FR에서 **보정 전 정합 proxy가 중앙 순위인 장면**을 선택하고, 참조 edge 에너지가 가장 높은 80² ROI를 확대했다. 좋아 보이는 결과를 기준으로 장면을 고르지 않았다. 장면 번호·ROI·밴드는 [선택 기록](2026-10-06_pan-aligner-effectiveness/visual_selection.json)에 남겼다.

첫 행은 RR GT, 둘째 행은 FR bicubic MS를 참조한다. 빨강은 참조 edge, 청록은 PAN edge, 흰색은 중첩이다. α별로 동일한 edge contrast를 사용한다. 원래 PAN/MS는 분광 응답과 선명도가 다르므로 색이 완전히 겹치지 않는 것을 전부 이동 오차로 해석하면 안 된다. 제목의 잔차는 ROI만이 아니라 전체 고정 내부 영역에서 추정한 값이다.

### WV3

![WV3 실제 영상](2026-10-06_pan-aligner-effectiveness/07_WV3_real_overlays.png)

### QB

![QB 실제 영상](2026-10-06_pan-aligner-effectiveness/07_QB_real_overlays.png)

### GF2

![GF2 실제 영상](2026-10-06_pan-aligner-effectiveness/07_GF2_real_overlays.png)

GF2는 FR의 흐린 bicubic 참조와는 4배 이동이 더 잘 맞아 보여도, RR GT 기준에서는 과보정이 분명해지는 사례다. “더 겹쳐 보임”을 최종 융합 품질 또는 물리적 정합의 증거 하나로 삼으면 안 된다.

## 9. masking과 이번 해석의 관계

- 경계 마스크는 shift 때문에 원본 영상 바깥을 읽은 부분을 제외하는 장치다. **shift 크기나 RR→FR 변위 scaling을 수정하지 않는다.**
- 현재 warp는 바깥에서 border 값을 복제한다. `warp_support_mask`는 clamp 이전 좌표의 bicubic 4tap support를 검사한다. synthetic warp와 보정 warp를 연달아 수행하면 두 단계 support가 필요하다.
- 이번 registration probe는 공통 고정 16px 내부 영역을 사용했다. 반면 최종 HQNR은 공식 raw-original **full512**다. 두 수치를 같은 평가 프로토콜이라고 혼용하지 않았다.
- PAN reference까지 함께 움직인 HQNR, shift별로 다른 면적을 버린 HQNR, native PAN full-frame HQNR은 별도의 지표 조건이다. 정합 개선 설명을 위해 바꾸더라도 원래 비교표 HQNR의 대체값으로 쓰면 안 된다.

## 10. 연구 주장과 다음 검증의 권고

### 현재 근거로 가능한 표현

> 이 모듈은 end-to-end reconstruction으로 학습된 저차원 PAN translation frontend다. 데이터셋과 checkpoint에 따라 소량의 정합 proxy 및 융합 지표 개선을 제공할 수 있으나, 절대 물리적 정합이나 RR→FR scale-equivariant registration을 보장하지 않는다.

반대로 **“센서 misalignment를 정확하게 제거한다”, “FR에서도 충분히 보정한다”, “4를 곱하면 문제가 해결된다”**는 표현은 현재 근거로 정당화되지 않는다.

우선순위는 다음과 같다. 이번 작업에서는 제안만 하며 운영 코드는 바꾸지 않았다.

1. **참조 좌표계를 먼저 고정한다.** Aligner가 맞춰야 할 것은 bicubic 입력 위상인지, native LMS/GT 위상인지 명시한다. 반 픽셀 보정/보간법을 바꾸는 ablation은 별도 실험으로 하고 기존 checkpoint에 조용히 섞지 않는다.
2. **학습 반응부터 검증한다.** 동일 ε probe에서 train/RR gain이 1에 근접하는지 확인한다. 지금처럼 RR에서도 0.04 수준이면 FR 배율만 수정하는 접근은 불충분하다. reconstruction/consistency의 A-gradient 비율과 학습 추이를 같이 기록한다.
3. **같은 FOV의 multi-scale 제약을 시험한다.** 동일 물리적 변위를 가진 실제 대응 자료 또는 검증된 degradation으로 `c_FR≈4c_RR`를 검사한다. 단순 고정 `×4` 대신 coordinate-aware 추론 경로·multi-scale 학습을 별도 대조한다.
4. **기하학과 재구성 성능을 동시에 본다.** 알려진 변위 회복, native-LMS/GT registration, RR ERGAS·SCC, native-reference FR HQNR을 함께 기록한다. 하나의 개선으로 나머지 개선을 추정하지 않는다.
5. **역할이 다른 no-aligner 재학습 대조를 둔다.** frozen α=0 결과만으로 Aligner 학습 전체의 필요/불필요를 단정하지 않는다. FR 실제 물리적 정합 성능을 주장하려면 HRMS/독립 tie point 등 정답에 가까운 외부 근거가 필요하다.

## 11. 산출물·재현

모든 산출물은 [분석 폴더](2026-10-06_pan-aligner-effectiveness)에 있다.

- `analyze.py`: warp/registration 단위검사, Student 진단, α=0/1/4 전체 영상 재추론.
- `teacher_probe.py`: config에 고정된 정확한 donor Teacher 대조, ±2px 동일-FOV scale probe.
- `reference_sensitivity.py`: native LMS 참조 대조, data/LPAN SHA 확인.
- `plot_results.py`: 측정 JSON에서 표 요약·실제 영상·그래프 생성.
- `verify_artifacts.py`: 공식 기록과 checkpoint/핵심 소스 일치, 360개 영상 평가의 완결성, 그림 10개, 문서 링크 검사.
- `*_diagnostics.json`, `*_teacher.json`, `*_quality.json`: 전체 장면 결과.
- `summary.json`, `per_scene.csv`, `reference_sensitivity.json`, `source_hashes.json`, `unit_tests.json`: 요약·검증 근거.

저장소 루트에서 실행한다. 학습/운영 데이터는 읽기만 하며 출력은 이 분석 폴더에만 쓴다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/knuvi/miniconda3/envs/pancrafter/bin/python results_log/2026-10-06_pan-aligner-effectiveness/analyze.py --mode unit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/knuvi/miniconda3/envs/pancrafter/bin/python results_log/2026-10-06_pan-aligner-effectiveness/analyze.py --mode diagnostics
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/knuvi/miniconda3/envs/pancrafter/bin/python results_log/2026-10-06_pan-aligner-effectiveness/teacher_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/knuvi/miniconda3/envs/pancrafter/bin/python results_log/2026-10-06_pan-aligner-effectiveness/reference_sensitivity.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/knuvi/miniconda3/envs/pancrafter/bin/python results_log/2026-10-06_pan-aligner-effectiveness/analyze.py --mode quality
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/knuvi/miniconda3/envs/pancrafter/bin/python results_log/2026-10-06_pan-aligner-effectiveness/plot_results.py
```

`quality`는 같은 checkpoint SHA의 이미 계산된 항목을 재사용한다. 기존 공식 결과·Sheets·checkpoint·학습 설정은 변경하지 않았다. 이 보고서는 새로운 실험의 학습 결과가 아니라 기존 자산의 사후 진단이다.
