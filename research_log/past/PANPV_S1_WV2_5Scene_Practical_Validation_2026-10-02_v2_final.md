# s1 전용 WV2 5개 씬 팬샤프닝 — 최종 실험 Case 및 실행 지시서

| 항목 | 확정 내용 |
|---|---|
| 작성일 / 버전 | 2026-10-02 / **v2-final** |
| Campaign ID | `PANPV_S1_WV2_5SCENE_20261002_v2` |
| 실행 서버 | **s1만 사용**. donor가 s2에서 학습됐다는 이력과 실행 서버를 구분한다. |
| 대상 | 사용자 화면의 **WV2 H5 5개**. 파일 목록은 §2에 고정한다. |
| 신경망 실험 | **Zero-shot / 1-scene Scratch / 1-scene Fine-tuning** |
| 필수 비교군 | **8밴드 Brovey만 사용**. Bicubic 독립 비교군은 제외한다. |
| 공통 모델 | **PLH 입력, W104 / D122 residual U-Net**, Aligner·Teacher·KD·attention 없음 |
| 결과 시트 | `pan-cvpr27` → **`practical-validation`**, sheetId `202610020` |
| 1차 실행량 | R01, SC 5회 + FT 5회, **각 20K update**. ZS/Brovey는 학습 없음. |
| 현재 상태 | **계획서와 시트 설정 수정 완료. 실제 H5·s1·weight bytes 검증 및 학습 기동은 미수행.** |

> **실행 담당자:** s1의 독립 프로젝트에서 이 문서의 데이터·donor·모델 검증을 먼저 수행한다. 검증 통과 후에만 지정 5개 씬에 세 신경망 방식과 Brovey를 실행한다. 사내 H5의 내부 구조를 파일명이나 스크린샷만 보고 가정하지 않는다. 기존 s2 benchmark 실험, 기존 s1 작업, 원본 자료, donor checkpoint 및 다른 시트는 변경하지 않는다.

## 0. v1에서 변경한 범위와 유지한 범위

| 구분 | v2 확정 사항 |
|---|---|
| 비교군 변경 | `BICUBIC` case·성능행·비교 이미지를 생성하지 않는다. **모델 내부 MS/LPAN의 bicubic 보간은 그대로 유지**한다. |
| 서버 변경 | 미정이던 실행 서버를 **s1**으로 고정한다. 기존 s2 전용 controller를 server 이름만 바꿔 실행하지 않는다. |
| 입력 구체화 | 스크린샷의 H5 5개를 명시적 allowlist로 고정한다. 화면의 `_reg` JSON 2개는 대응 미확인 보조 파일이다. |
| 유지 | 씬별 독립 학습, WV3-only donor, PLH/no-Aligner, RR 공간 holdout, 동일 씬 FR 적용, 품질·총시간 비교, 원본 로그 보존. |
| 학습 설정 | v1의 20K·effective48·SC LR=1e-4·FT LR=1e-5를 **1차 실행 설계값**으로 고정한다. 최적값 또는 실측값이라는 뜻이 아니다. |

본 문서는 v1 [R1]과 기존 모델 계약 [R0]을 기반으로 한다. 아래 파일 목록은 사용자 화면 [R2]에서 확인한 정보이고, H5 내부 key·shape·dtype·좌표·제품 encoding·개별 patch 수는 아직 확인되지 않았다. §4의 schema mapping은 **실행 전에 채울 요구사항이지, 현재 파일에서 읽어낸 schema가 아니다.**

## 1. 이번 실험이 답하는 질문

> **PAN+MS 한 씬을 입력받았을 때, WV3 모델을 그대로 쓰는 방법과 해당 씬만 학습하는 방법이 Brovey 대비 어떤 품질·처리시간을 보이는가?**

학습에 사용한 씬을 최종 처리하는 **입력 씬 적응형 실험**이다. 씬 내부 일부 영역으로 RR 학습·validation을 구성하고, 같은 씬의 native PAN+MS에 적용한다. 이를 미학습 촬영 장면에 대한 독립 일반화로 표현하지 않는다.

이번 donor는 **WV3-only Single 모델**이다. 이번 세 방식만으로 Shared 사전학습의 우수성, 여러 센서로의 일반화, 신규 센서 학습 후 기존 센서 성능 유지까지 입증했다고 해석하지 않는다. 그 질문은 앞선 unseen-sensor 실험안과 별개다.

## 2. 입력 파일 allowlist — 정확히 5개

### 2.1 학습·평가 대상 H5

| 별칭 | 파일 기반 Scene_ID | 화면에 표시된 H5 파일 | 적용할 방법 |
|---|---|---|---|
| V01 | `200014103525_01` | `200014103525_01.h5` | ZS / SC / FT / BROVEY8 |
| V02 | `200014103539_01` | `200014103539_01.h5` | ZS / SC / FT / BROVEY8 |
| V03 | `200014103658_01` | `200014103658_01.h5` | ZS / SC / FT / BROVEY8 |
| V04 | `200014103676_01` | `200014103676_01.h5` | ZS / SC / FT / BROVEY8 |
| V05 | `200014103802_01` | `200014103802_01.h5` | ZS / SC / FT / BROVEY8 |

- Scene_ID는 문자열로 저장하며 Sheet A열에는 위 실제 ID를 사용한다. V01~V05는 실행 순서용 별칭이다.
- 원 촬영 ID, acquisition ID, 원본 footprint 및 여러 H5 사이의 관계는 별도 metadata로 확인한다. **다섯 파일이 다섯 개의 독립 촬영이라는 주장은 하지 않는다.**
- 파일 크기·수정 시각은 sample 수나 품질을 대신하지 않는다. 화면에서 일부 크기는 잘려 있으므로 정확한 byte 수는 s1에서 읽는다.
- 디렉터리의 모든 `.h5`를 자동 glob하여 학습하지 않는다. 위 다섯 파일만 등록하고 누락·중복·잘못된 내용은 오류로 남긴다.
- CAS500, 나머지 WV2 씬, 공개 benchmark patch를 이번 scene adaptation에 섞지 않는다. WV3 정보는 동결한 donor weight로만 사용한다.

### 2.2 화면의 `_reg` JSON 두 개

| 화면 표시 이름 | 확인된 범위 | 이번 처리 규칙 |
|---|---|---|
| `200014103682_01_reg` | 파일 유형이 JSON으로 표시됨. 실제 확장자·내용은 미열람 | 목록에 보존하되 선택 H5와의 명시적 대응이 확인되기 전에는 적용하지 않음 |
| `200014103703_01_reg` | 동일 | 동일 |

이 두 이름의 ID는 선택한 H5 5개의 ID와 다르다. 이를 가까운 정렬 순서의 H5에 연결하거나, 6·7번째 씬으로 등록하거나, 정합이 완료됐다는 증거로 취급하지 않는다. 실제 JSON이 여러 씬을 참조할 가능성도 있으므로 **내용의 source ID/path/hash를 확인한 뒤 필요한 경우만 mapping**한다. 특정 sidecar가 있다는 이유로 runtime warp를 추가하지 않는다.

### 2.3 서버 전달 원칙

`<S1_INPUT_ROOT>`는 실제 전송 완료 후 확정한다. 화면에는 상위 폴더가 없으므로 Windows 원본 경로나 s1 절대경로를 추측하지 않는다.

전송 전후 동일 H5의 byte 수와 SHA256을 대조하고, metadata·tile index·생성 recipe가 외부 파일에 있으면 함께 전달한다. 전송·검사 단계에서 원본 H5를 재압축하거나 덮어쓰지 않는다. H5 외부 링크 또는 외부 dataset이 있으면 참조 자료도 allowlist 안에서 검증한다.

## 3. 실험 Case와 모델 초기화

### 3.1 정식 비교 방법

| Method | 초기화 | 해당 씬 추가 학습 | 학습 범위 | 5개 씬 처리 |
|---|---|---|---|---|
| `ZS_WV3` | 고정 WV3 donor | 없음 | 없음 | 같은 donor를 각 씬에 적용 |
| `SC_SCENE` | 동일 구조의 랜덤 초기화 | 해당 씬의 RR-train만 | stem + trunk + head 전체 | 씬마다 독립 모델 1개 |
| `FT_WV3_SCENE` | 매 씬 동일 donor를 다시 복사 | 해당 씬의 RR-train만 | stem + trunk + head 전체 | 씬마다 독립 모델 1개 |
| `BROVEY8` | 동결한 8밴드 비학습 구현 | 없음 | 없음 | 동일 입력·영역에 적용 |

**학습 방식은 세 가지이며 Brovey는 비학습 비교군이다.** R01 기준 SC 5회 + FT 5회 = **10개 학습 run**이다. EXACT20K를 기준으로 ZS 5개 + SC 5개 + FT 5개 + Brovey 5개 = **기본 FR 결과 20개**를 얻는다. BEST가 EXACT와 다르면 해당 SC/FT 출력이 추가된다. 중간 checkpoint와 selector alias를 독립 반복으로 세지 않는다.

동일 촬영의 벤더 결과가 추후 실제 제공되면 `VENDOR_PS_REF`로 참고 비교할 수 있으나, 현재 필수 실행 큐에는 넣지 않는다. 벤더 결과를 학습 target 또는 checkpoint 선택 기준으로 사용하지 않는다.

### 3.2 씬별 독립 초기화

```text
고정 WV3 donor ─→ FT(V01) ─→ V01 팬샤프닝
고정 WV3 donor ─→ FT(V02) ─→ V02 팬샤프닝
                         ...
고정 WV3 donor ─→ FT(V05) ─→ V05 팬샤프닝
```

SC와 FT 모두 씬이 바뀔 때 모델·optimizer·scheduler·sample cursor를 독립 초기화한다. `FT(V01) → FT(V02)` 식의 순차 전이는 금지한다. FT는 donor weight만 읽고 WV3 학습 당시 optimizer moment는 가져오지 않는다. 같은 씬·repeat의 SC/FT는 split, sampling, augmentation stream을 대응시킨다.

### 3.3 공통 PLH U-Net 계약

```text
PAN : [B, 1, 4h, 4w]
MS  : [B, 8, h, w]
LPAN: [B, 1, h, w]

P    = normalized PAN
L    = bicubic(normalized LPAN, ×4, align_corners=False)
H    = P - L
M_up = bicubic(normalized MS, ×4, align_corners=False)

X = concat([P, L, H, M_up], dim=1)  # 11 channels
Y = M_up + Head(Trunk(Stem(X)))      # 8 channels, base를 한 번만 더함
```

구조는 [R0]의 attention-free `W104 / depth [1,2,2]` 계약을 계승한다. scale 채널은 `[104,104,104]`, ResBlock 수는 encoder1/encoder2/middle/decoder2/decoder1 = `1/2/2/2/1`이다. LayerNorm·SiLU·Down/UpConv·padding·residual 위치를 일반 U-Net 구현으로 임의 대체하지 않는다.

**Aligner·flow·grid_sample·Teacher·KD·attention·mode modulation은 생성하지 않는다.** 이미 수행된 상위 단계의 기하 처리 이력은 입력 provenance로 기록하되 모델 안에서 다시 적용하지 않는다.

SC는 기존 zero residual head 및 L/H kernel 초기화 규약을 유지한다. 입력 L/H tensor 자체를 0으로 만들지는 않는다. FT에서는 head나 L/H kernel을 다시 초기화하지 않는다. SC step0이 MS_up과 같은지 확인하는 것은 **구현 단위검사**이며, 삭제한 Bicubic 비교군을 다시 생성하는 것이 아니다.

### 3.4 동결할 WV3 donor

| 항목 | v1에서 확인하여 계승하는 값 |
|---|---|
| 원 run | `PDSP_S2_R01_S01_W104_D122_F100_PLH_SE271001_A01` |
| 원 학습 센서 / 구조 | WV3-only / W104 D122 / PLH / no Aligner |
| 원 step | 50,000 |
| 원 selector | `BEST_SENSOR_VAL_TO_B0001@50000`, 당시 EXACT50K와 같은 SHA |
| Weight SHA256 | `6ad7cff8fc60542ab687a405f7b55430926942f008a65345de99fa69fa8b3846` |
| 근거 | v1 §3.2의 `배포용 모델` 조회 기록 [R1]. 이번 작성에서 실제 weight 파일은 미열람. |

s1로 model file·model config·preprocessing contract·source snapshot·license/provenance를 전달한 뒤 SHA256과 strict load를 검증한다. 원 학습 서버가 s2여도 **이번 실행·비용 기록의 Server는 s1**이다. donor를 바꾸어야 하면 사내 결과를 보기 전에 별도 revision으로 고정한다. 이름이 비슷한 Aligner 포함 checkpoint로 fallback하지 않는다.

**필수 수치 검사:** 같은 입력·precision·eval mode에서 FT step0 = ZS여야 한다. SC/FT가 같은 구조를 사용하고, 한 번의 update 후 gradient·parameter가 정상 갱신되는지 별도 smoke test로 확인한다. smoke weight는 정식 run에 재사용하지 않는다.

## 4. H5 입력 계약 — 실제 내용을 검사한 뒤 확정

### 4.1 파일 목록이 알려 주지 않는 것

확인된 것은 **H5 파일명 5개와 별도 JSON 표시 이름 2개**다. 아래 항목은 `h5py` 등으로 실제 파일을 읽고 확인해야 한다.

| 검사 | manifest에 기록할 것 |
|---|---|
| 구조 | group/dataset 전체 경로, shape, axis 순서, dtype, chunk/compression, attrs |
| 값 | 단위, encoding, scale/offset, min/max·분위수·nonfinite·zero·포화 통계 |
| 의미 | native PAN/MS인지, RR PAN/MS/target인지, `lms`·`lpan`·`gt`의 실제 생성식 |
| 공간 | scene 원 크기, 각 tile의 원본 row/col와 extent, PAN↔MS 대응, crop·scale·phase |
| 밴드 | 실제 8밴드 순서와 모델 convention으로의 mapping |
| 유효영역 | NoData 정의, scene/tile mask, 외곽·빈 타일·필터 support |
| 제품 | 사용자가 제공한 WV2/L1B 식별, ACOMP/DRA·방사·기하 처리의 확인된 metadata |
| 이력 | 생성 코드/config hash, 입력 파일 SHA, 상위 registration/resampling 적용 여부 |

기존 ‘4종 배열’이라는 설명만으로 key를 `pan/ms/lms/gt`라고 확정하지 않는다. 자동 대소문자·shape heuristic은 후보 탐색에만 쓰고, **semantic mapping을 동결하기 전에는 학습하지 않는다.** `gt`라는 이름만으로 native HRMS 정답이라고 판단하지 않는다.

### 4.2 Canonical mapping 양식 — 물리 key를 가정하지 않음

```yaml
scene_id: "200014103525_01"
file: "<S1_INPUT_ROOT>/200014103525_01.h5"
file_sha256: null                 # s1 실측값으로 채움
source_acquisition_id: null       # 확인된 metadata로 채움
native_pair:
  pan_dataset_path: null          # 실제 H5 경로 또는 검증한 별도 원본 참조
  ms_dataset_path: null
  axis_order: null
  value_encoding: null
  band_mapping: null
tile_index:
  source: null                    # H5 dataset/attrs 또는 별도 index 파일
  row_col_grid: null              # PAN grid인지 MS grid인지 명시
  source_scene_shape: null
  reconstruction_available: null
rr_arrays:
  already_generated: null
  pan_path: null
  ms_path: null
  target_path: null
  generation_profile_sha256: null
sidecars: []                      # 실제 관계를 검증한 파일만 추가
```

이 양식의 null은 기본값이 아니다. 필수 semantic·geometry·normalization 정보가 미확정이면 해당 gate를 통과시키지 않는다. H5 dataset 이름을 바꾸기 위해 원본 파일을 수정할 필요는 없고, loader mapping으로 연결한다.

### 4.3 64px의 기준과 native FR 자료

| 실제 확인되는 배열 | 처리 |
|---|---|
| native PAN256 + native MS64 | RR 생성 후 PAN64 / MS16 / target64 구성 가능. 실제 crop 좌표·필터 support 검증 필요. |
| native PAN64 + native MS16 | 그 자체는 기존 RR target64 학습 tuple과 다르다. 이웃/원본으로 충분한 문맥을 복구한 뒤 RR 생성한다. |
| 이미 RR PAN64 / MS16 / target64 | 생성 recipe와 공간 provenance가 검증되면 재사용한다. 다시 4배 축소하지 않는다. |
| RR tuple만 있고 native P/MS가 없음 | `BLOCKED_NATIVE_FR`. RR 출력을 원해상도 결과로 위장하지 않는다. |
| native tile은 있으나 배치 index가 없음 | full-scene 재조립·공간 holdout을 완료했다고 주장하지 않는다. 생성 index/원본을 확보한 뒤 진행한다. |

전체 WV2 15씬 약 22만 patch 및 두 데이터셋 합계 25GB는 사용자 측 전체 집계다. 이번 5개 파일의 patch 수·독립 촬영 수·메모리 요구량은 별도로 실측한다. 압축 파일 용량을 GPU 메모리 예상치로 사용하지 않는다.

### 4.4 정규화·밴드·기하

- 모델 밴드 convention은 `[Coastal, B, G, Y, R, RE, NIR1, NIR2]`이다 [R0]. 실제 WV2 파일의 순서는 metadata로 검증한다. 8비트와 8밴드는 구분한다.
- `WV2 → /2047`, `uint16 → /65535`, `이전 샘플이 8비트 → /255`를 자동 적용하지 않는다. 실제 encoding을 해석한 뒤 donor 입력 표현으로 변환한다.
- `N(x)=2*x/Dmax-1`을 사용할 경우 Dmax는 납품 encoding을 확인한 값이며, sample의 관측 최대값으로 씬마다 바꾸지 않는다. affine offset 또는 별도 scale이 있으면 명시적으로 처리한다.
- 이미 정규화된 배열에는 같은 변환을 두 번 적용하지 않는다. finite zero를 자동 NoData로 간주하지 않는다.
- PAN과 LPAN은 같은 수치 표현을 사용한다. HPAN은 float에서 `P-L`로 계산하고 절댓값·재정규화·독립 clipping을 하지 않는다.
- Brovey는 확인된 비음수 제품 표현에서 계산한다. U-Net용 `[-1,1]` 값을 그대로 비율식에 넣지 않는다. 출력은 모든 방법에서 같은 분석 단위로 되돌린다.
- PAN/MS의 shape ratio뿐 아니라 동일 위치 대응을 확인한다. `_reg`라는 이름만 보고 offset을 적용하지 않는다. 추가 기하 보정이 필요하면 자동 추가하지 말고 입력 revision으로 고정하여 모든 방법에 공통 적용한다.
- 출력은 확인한 입력 PAN 대응 격자를 유지한다. WV3 0.31m로 강제 확대하거나 sensor-space 배열에 임의 정사보정 좌표를 붙이지 않는다.

## 5. 1-scene RR 학습과 native FR 처리

### 5.1 학습 자료 생성

공통 입력 전처리가 끝난 씬 `P_s, M_s`에 대해:

```text
P_RR      = D_P(P_s)             # M_s와 대응하는 공간 격자
M_RR      = D_M(M_s)             # M_s보다 선형 4배 낮은 해상도
target_RR = M_s
LPAN_RR   = LP_recipe(P_RR)

학습: model(P_RR, M_RR, LPAN_RR) → target_RR
실제 적용: model(P_s, M_s, LP_recipe(P_s)) → 동일 씬 native HRMS
```

`D_P/D_M`는 **실제 RR 배열 생성 recipe 또는 새로 동결한 degradation profile**로 등록한다. kernel, 밴드별 계수, padding, ratio, decimation origin, dtype를 기록한다. 확인할 수 없는 계수를 임의의 WV2 MTF라고 이름 붙이지 않는다. 근사 profile을 사용할 때는 근사라는 사실을 기록하고 결과를 본 후 변경하지 않는다.

모델 내부 LP recipe는 donor 계약 [R0]을 계승한다: PAN에서 Gaussian σ=1.98 / kernel41 / replicate / phase `[2::4,2::4]`, 계산 float64·cache float32, 그 뒤 normalization과 upsampling. **이 LP recipe와 RR 생성·평가 MTF는 별개**다. 상위 tile에서 계산한 경우 전역 원점에 맞춰 local phase를 환산한다.

저장된 RR/LPAN을 재사용하면 위 계약과의 수치 parity를 검사한다. 잘못된 재생성·두 번의 degradation을 막는다. target은 해당 축척의 MS이며 native HRMS GT가 아니다.

### 5.2 씬 내부 split

v1의 **약 80% train / 10% validation / 10% RR audit** 공간 block 분할을 유지한다. §2 순서나 파일 크기로 split을 바꾸지 않는다.

원본 좌표에서 coarse spatial block을 나누고 필터·학습 문맥이 split 경계를 넘어가지 않도록 buffer를 제외한다. block 크기·분배 seed·실제 tile ID를 첫 추론 전 동결한다. 실제 유효 면적/개수와 split hash를 기록한다. SC/FT/ZS/Brovey의 RR audit support는 같다.

이미 잘린 배열은 좌표 index로 공간 관계를 복원한다. 인접 patch를 무작위로 나누고 독립 촬영 test라고 부르지 않는다. 기존 RR 생성의 필터가 경계를 넘었으면 영향 영역을 제외하거나 검증된 원본에서 재생성한다. 데이터가 작아 유효 split이 성립하지 않으면 다른 씬을 끌어와 채우지 않는다.

### 5.3 결과 범위

| Scope | 처리·평가 영역 | 해석 |
|---|---|---|
| `RR_SPATIAL_HOLDOUT` | 같은 씬 내부의 미학습 audit block | 축소 문제의 복원 진단 |
| `FR_ALL_SAME_SCENE` | 학습 영역을 포함한 동일 입력 씬 전체 | 입력 씬 적응을 포함한 실제 처리 결과 |

audit나 vendor PS는 gradient·LR·checkpoint 선택에 사용하지 않는다. BEST를 고른 뒤 audit 영역까지 다시 학습하여 이전 holdout 점수와 섞지 않는다. 학습 영역을 포함하는 FR 결과는 `Input_adaptation=TRUE`로 명확히 표시한다.

## 6. 학습 설정·checkpoint·원본 로그

### 6.1 1차 실행 설정

| 항목 | SC_SCENE | FT_WV3_SCENE |
|---|---|---|
| 구조 | W104 / D122 / 11→8 PLH | 동일 |
| 초기화 | 기존 규약의 랜덤/zero residual 초기화 | 동결 donor의 모든 weight |
| 업데이트 범위 | 전체 네트워크 | 전체 네트워크 |
| Loss | valid-mask 적용 unclipped normalized mean L1 | 동일 |
| Optimizer | 새 AdamW | 새 AdamW |
| LR | **1e-4, constant** | **1e-5, constant** |
| Betas / eps / weight decay | (0.9,0.999) / 1e-8 / 0.01 | 동일 |
| Effective batch | **48** | **48** |
| Microbatch | s1 preflight에서 48, 24×2, 12×4 중 하나를 고정 | 동일 |
| Precision | FP32, AMP/TF32 OFF | 동일 |
| 보조 기능 | EMA·KD·FR loss·edge/SAM loss·자동 LR 탐색 없음 | 동일 |
| R01 master seed | **261002** | **261002** |
| 1차 endpoint | **20,000 update** | **20,000 update** |
| Wall-clock 상한 | 별도 시간 제한 없음. 사용자 중단·안전 정지 가능 | 동일 |

이 수치는 v1에서 제안한 **실용적 학습 절차 비교용 설계값**이다. SC/FT의 LR가 다르므로 초기화만의 순수 인과 비교라고 주장하지 않는다. 이번 1차에서 LR 탐색이나 반복 seed를 자동 추가하지 않는다. 20K 이후 연장·R02/R03은 별도 결정으로 기존 기록을 보존하며 수행한다. 기존 s2 campaign의 무기한 실행 규약을 이번 캠페인에 자동 적용하지 않는다.

정규화 출력은 clamp 전에 loss를 계산한다. mask의 유효 원소 수를 분모로 사용하고 microbatch 누적에서도 동일한 logical batch48 loss가 되도록 합산한다. 유효 pixel이 없는 sample은 사전 제외하고 실제 분모를 기록한다. sample이 48개보다 적으면 고정 shuffle stream을 순환하여 채우고 반복 노출을 기록한다. batch를 몰래 줄이지 않는다.

증강은 [R0]의 동기화 four-view 규약을 계승한다. PAN/MS/LPAN/target/mask에 같은 view를 적용하고 raw patch ID와 view ID를 저장한다. 추가 synthetic shift·radiometric augmentation은 넣지 않는다.

### 6.2 관측과 선택

| 주기/시점 | 작업 |
|---|---|
| 매 update | finite 검사, committed step/sample count 갱신 |
| 100 update | 최근 100-update loss 합·count·평균, LR, gradient norm, 시간·메모리 기록 |
| 500 update | 동결 train/val probe 평가. 각 최대128개, 부족하면 전수 |
| 2K update 및 20K | 전체 RR validation, resume fullstate 저장 |
| 0 / 500 / 1K / 2K / 5K / 10K / 20K | RR audit 및 고정 FR ROI 진단. 학습 제어·선택에는 사용하지 않음 |
| ZS/Brovey 최초 | native full-scene 추론·저장·평가 |
| SC/FT 20K 종료 | **EXACT20K** 및 **BEST_RRVAL_TO_20K**의 full-scene 추론·저장·평가 |

**Primary는 EXACT20K**다. Secondary BEST는 2K,4K,…,20K 전체 validation의 L1 최소이며, 동률이면 이전 step을 선택한다. step0은 초기 상태 검사·ZS 비교용이고 SC/FT BEST 후보에는 넣지 않는다. FT 결과가 ZS보다 나빠지면 그대로 보고하고 사후에 ZS로 바꾸어 FT 성능이라고 올리지 않는다.

SC step0의 MS-up과 같은 출력은 단위검사로만 기록하며 Bicubic 비교행·이미지를 생성하지 않는다. FT step0=ZS는 ZS 결과를 참조하여 중복 추론을 줄일 수 있다. 동일 SHA의 EXACT/BEST 결과도 검증한 inference cache를 재사용한다.

full-scene 결과가 없는 중간 FR ROI 지표를 Sheet의 `FR_ALL_SAME_SCENE` 값으로 입력하지 않는다. 중간 곡선은 로그에 보존하고 Sheet에는 완료된 full-scene 관측을 올린다.

### 6.3 로그의 분석 가능성

run마다 아래 파일을 보존한다.

```text
run_manifest.json / config.resolved.json / architecture_manifest.json
train_steps.jsonl / probes.jsonl / validation.jsonl
rr_audit.jsonl / fr_roi.jsonl / fullscene_metrics.json
checkpoint_index.jsonl / selection_history.jsonl / resume_events.jsonl
cost_breakdown.json / errors.jsonl
```

`train_steps.jsonl`에는 시작부터의 누적 평균만 남기지 말고 **최근 구간 sum/count/mean**을 반드시 저장한다. 학습곡선의 x축은 update·target samples_seen·effective_epochs·adapt_seconds를 함께 제공한다. effective epoch는 `committed samples / 해당 씬의 실제 RR-train 수`이다.

로그에는 campaign, run, method, scene, seed, parent SHA, source/schema/split/protocol SHA를 넣는다. 공통 장면별 지표와 train/validation gap을 보존하여 추가 학습이 유효했는지 확인할 수 있게 한다. JSON NaN/Infinity 대신 null과 상태/원인을 기록한다.

## 7. Brovey 기준과 품질·비용 평가

### 7.1 Brovey8 설정

v1 §7.1 [R1]을 계승한다. 사내에서 비교 대상으로 사용하는 Brovey 구현이 있으면 source/version/config를 고정한다. 없으면 그 계획에서 지정한 **GDAL WeightedBrovey 8밴드 구현**을 별도로 검증하여 사용한다. 어느 구현을 선택했는지 첫 결과 전에 manifest에 확정하며, 중간에 바꾸지 않는다.

밴드별 가중치, resampling, 출력 dtype·범위, NoData, 처리 격자, software version을 기록한다. 확인된 운영 가중치가 없어서 `1/8` 등가중치를 쓰면 **equal-weight 기준선**으로 명명하고 최적 WV2 가중치 또는 벤더 알고리즘으로 표현하지 않는다. target에서 가중치를 추정하는 기능을 사용하면 추정 영역·방법·시간을 기록하고 audit target을 쓰지 않는다.

Brovey에는 NN과 같은 native P/MS와 유효영역을 주되 각 알고리즘의 수치 표현을 지킨다. RGB 전용 결과를 8밴드 ours와 전 밴드 점수로 직접 비교하지 않는다. 이번 기본 비교는 8밴드 전체다.

### 7.2 평가 protocol 동결

| 범위 | 지표/산출물 | 기록 경계 |
|---|---|---|
| RR holdout | ERGAS, SAM, PSNR, SCC, SSIM, Q8, RMSE, CC | 실제 RR target에 대해서만 계산 |
| Native FR | WV2용으로 검증한 HQNR/Ds/Dlambda, MS 재투영 SAM, PAN 구조 상관 | native HRMS GT가 있다고 가정하지 않음 |
| 고정 ROI | 도로·건물·식생·수면 등 실제 포함된 영역의 edge·색 번짐·halo·blur | 모든 방법에 동일 ROI·band 조합·display stretch |
| 전체 출력 | tile seam, NoData/외곽, 출력 범위·포화율, shape·밴드·좌표 | 큰 씬과 작은 씬 모두 보고 |
| JQM | 구현·variant가 WV2용으로 검증된 경우만 | 미검증이면 빈칸 |

원본 영상의 tile 배치를 복원한 전역 grid에 metric window를 고정한다. crop, valid mask와 window 처리, aggregation, PSNR data range, clipping, MTF/재투영/보간과 phase를 `protocol.json`에 저장한다. 불완전한 window는 사전 규칙대로 제외하고 유효 면적을 보고한다. 타일마다 임의 계산한 평균을 global metric이라고 부르지 않는다. `Q8=32×32 window` 같은 기존 규약을 계승하려면 실제 support와 구현 parity를 확인한다.

native FR에 GT-PSNR을 만들지 않는다. FR 재투영 SAM은 consistency 진단으로 별도 열에 기록한다. 기존 benchmark의 RR20/FR20 고정 shape·crop을 임의의 사내 씬에 강제로 적용하지 않는다. 미검증 지표는 blank/null과 사유를 남긴다.

출력은 원제품에 대응하는 분석 단위의 float 결과를 보관한다. 물리/encoding 유효범위 clipping이 필요한 경우 모든 방법에 사전 고정한 규칙을 적용하고 unclipped 통계도 남긴다. 표시용 stretch나 정수 납품 변환은 별도 산출물로 취급한다.

### 7.3 비교·비용 집계

각 씬에서 ours−Brovey, FT−ZS, FT−SC의 차이를 기록한다. 5씬 요약은 **씬별 동일 가중 평균·중앙값·범위·실패 수**로 작성하고, 넓은 씬의 patch 수가 전체 결론을 지배하지 않게 한다. 서로 같은 원 촬영의 crop이면 그 관계와 독립 촬영 분모를 명시한다.

```text
서비스 비용 = read/preprocess + scene adaptation + inference + stitch/write
실험 부가 비용 = 반복 probe/validation/audit/평가 + 로그 업로드 등
```

두 비용을 분리하고 실제 측정 scope를 기록한다. FT의 WV3 사전학습 비용은 이미 존재하는 donor의 일회성 비용으로 별도 표기한다. SC/FT에는 adaptation 비용을 반드시 포함한다. 캐시 사용 여부·최초 생성 비용·warm/cold 상태를 고정하며, 여러 checkpoint를 시험한 총 실험시간과 최종 결과 하나의 처리시간을 혼합하지 않는다. BEST가 이른 step이어도 그 BEST를 찾기 위해 수행한 20K 탐색 비용을 숨기지 않는다.

Brovey 대비 우위만으로 실제 벤더 산출물 대비 우위 또는 모든 전통 알고리즘 대비 우위로 확대하지 않는다.

## 8. 비중첩 저장·추론·재조립

```text
비중첩 저장 tile + 원본 좌표 index
                  ↓
이웃/원본에서 문맥 확보 → LP/degradation/추론의 전역 phase 유지
                  ↓
유효 중심부 결합 또는 검증된 overlap blending
                  ↓
해당 씬 전체 FR 산출물 + mask + provenance
```

저장 방식은 유지한다. 다만 저장 tile을 경계 문맥 없이 각각 독립 추론하는 방식으로 자동 전환하지 않는다. LPAN·RR 생성도 작은 tile마다 새로운 경계 padding을 만들지 않도록 문맥을 확보한다.

halo·tile 크기는 실제 모델·필터의 full/tile parity로 확정하고 수치 오차를 기록한다. 비정사각형 영역, tile 경계, scene 외곽, mask 주변을 검사한다. 모델 입력 크기 배수 제약과 전체 좌표의 phase를 함께 유지한다.

행/열·scene shape·유효 geo metadata가 없어 완전한 재조립이 안 되면 `FR_ALL_SAME_SCENE` 완료로 올리지 않는다. patch-only 사전 진단은 로컬에서 별도 표시할 수 있지만 필수 full-scene 산출물을 대신하지 않는다. 없는 영역을 임의 보간하여 원본이 존재했던 것처럼 채우지 않는다.

## 9. s1 구현·운영 규약

### 9.1 독립 실행

신규 프로젝트 예시 이름은 `pan_practical_validation`이며, **실제 경로는 s1 inventory 후 확정**한다. 기존 `PANDEP_S2_*` 프로젝트의 server guard·queue·source catalog·uploader를 수정해서 이 실험을 끼워 넣지 않는다. 기존의 서버별 센서 lane 규칙을 재사용하지 않고 `execution_server=s1`, `target_sensor=WV2`, `donor_sensor=WV3`를 분리한다.

- s1 hostname/server mapping, GPU UUID·기존 process, RAM·VRAM·디스크·환경을 읽기 전용 검사한다. GPU 종류는 추측하지 않는다.
- s1에 다른 학습이 실행 중이면 자동 kill/stop하지 않고 `WAIT_RESOURCE`로 둔다. s2 실험도 건드리지 않는다.
- 원본 입력·donor는 읽기 전용이고 신규 cache·결과는 별도 work root 아래에만 쓴다.
- 기존 환경에 전역 package update·CUDA/driver 변경을 하지 않는다. 검증한 격리 환경과 dependency/source hash를 사용한다.
- s1 실행기·학습 프로세스는 하나다. 고유 controller lock, run registry, 시트 outbox를 사용한다.
- 내부 연구 실행과 상업 배포 승인은 별개로 관리한다. 코드·weight·사내 영상의 사용권 증빙을 보존하고 결과를 자동 공개하지 않는다.

### 9.2 실행 순서

| 순서 | 작업 |
|---|---|
| 1 | 입력 전송 무결성, 5개 allowlist, 실제 H5 schema·native pair·tile index·제품 metadata 검사 |
| 2 | donor strict load·PLH/no-Aligner·FT0=ZS·RR/LP/mask·tile parity·resume·시트 gate |
| 3 | 모든 씬의 split·ROI·normalization·degradation·Brovey config·seed 동결 |
| 4 | V01→V05 각각 BROVEY8 및 ZS의 full-scene 결과 생성·저장 |
| 5 | V01 SC→FT, V02 SC→FT, …, V05 SC→FT를 각각 첫 **2K**까지 수행 |
| 6 | 같은 순서로 **2K block**씩 exact-resume하여 모든 SC/FT run을 20K까지 진행 |
| 7 | 각 run의 EXACT20K 및 BEST_RRVAL_TO_20K 전체 씬 결과와 시트 기록 |
| 8 | 5개 씬 paired 결과, 품질–학습량/시간 곡선, 실패·누락을 포함한 최종 보고 |

2K block은 자원 양보·저장 단위이며 초기화를 다시 하는 시점이 아니다. 이 순환 순서는 v2의 실행 설계로, 한 씬만 오래 독점하지 않고 모든 씬의 초기 적응 로그를 먼저 확보하기 위한 것이다. 사용자 중단 시 완료한 범위만 보고한다.

### 9.3 Gate와 안전 정지

| Gate | 통과 조건 |
|---|---|
| G0 서버·분리 | 실제 s1, 기존 작업·원본·시트 보호, 신규 output 경로, 충분한 저장 공간 |
| G1 입력 | 5개 파일 identity, semantic schema, native FR 가능, 좌표·band·normalization·mask 확인 |
| G2 모델·donor | 지정 SHA strict load, no-Aligner/no-KD, PLH 11채널, residual 한 번, FT0=ZS |
| G3 학습 | RR provenance, 공간 split/buffer, masked loss·microbatch parity, SC/FT 단독 씬 범위 |
| G4 추론·평가 | full/tile·전역 phase·8밴드 Brovey·고정 metric support·output 재조립 검증 |
| G5 재개·기록 | fullstate exact-resume, JSONL/outbox, Sheet header/ID/readback, 로그 export 검사 |

검증되지 않은 데이터·모델 정보를 임의 기본값으로 채워 통과시키지 않는다. 필수 자료 미확정은 `BLOCKED_INPUT_SCHEMA`, `BLOCKED_NATIVE_FR`, `BLOCKED_GEOMETRY`, `BLOCKED_NORMALIZATION`, `BLOCKED_DONOR` 등 세부 원인으로 기록한다. 해당 씬을 목록에서 조용히 빼거나 다른 파일로 대체하지 않는다.

OOM·nonfinite·identity 변경·디스크 부족은 안전 정지한다. batch/LR/precision을 자동 변경하거나 파일을 무단 삭제하지 않는다. Sheet 장애만 발생하면 로컬 outbox를 유지하며 학습을 계속할 수 있다. 사용자 stop/pause는 안전 optimizer 경계에서 저장하고, resume 요청 전에는 자동 재개하지 않는다.

resume에는 model, optimizer, scheduler 위치, RNG, sampler·augmentation cursor, committed sample 수, 비용·이벤트 상태를 포함한다. fullstate가 없으면 weights-only 재개를 동일 run continuation이라고 기록하지 않는다.

## 10. `practical-validation` 시트와 로그 열람 통로

### 10.1 시트 식별·수정 범위

```text
Workbook: pan-cvpr27
Spreadsheet_ID: 1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0
Sheet: practical-validation
sheetId: 202610020
Header: row 5, A:BL (64 columns)
Data: row 6 onward
```

이번 작성에서 기존 탭을 읽은 후 **A1:A3 설명을 s1/v2로 수정하고, Method 드롭다운에서 BICUBIC을 제거하고, Server를 s1로 제한**했다. 관측행 수식과 기존 64열 이름·순서, 다른 탭은 유지했다. `VENDOR_PS_REF`는 제공될 경우의 참고용 선택지이며 필수 큐에는 없다. 데이터행은 작성 시 조회에서 비어 있었고 가상 성능값을 넣지 않았다. 실행 직전 상태는 다시 읽는다. [R3]

### 10.2 기록 규칙

| 열 | 의미 / 이번 값 |
|---|---|
| A:C | 실제 Scene_ID / Method / EXACT 또는 BEST selector |
| D:G | full-scene FR HQNR / Ds / Dlambda / 검증된 JQM |
| H:O | 동일 checkpoint의 RR holdout ERGAS / SCC / SAM / PSNR / SSIM / Q8 / RMSE / CC |
| P:T | Infer_s / Adapt_s / EndToEnd_s / PeakMem_MB / Params_M |
| U:W | Updates / Samples_seen / Effective_epochs |
| X:AB | Fit_scene_ID / Eval_scene_ID / RR_scope / FR_scope / Input_adaptation |
| AC:AI | Seed / Repeat / Run_ID / Campaign_ID / Parent_run / Parent_checkpoint_SHA / Checkpoint_SHA |
| AJ:AR | **Server=s1** / Hardware / Precision / Train·Val·Test patches / Train·Val L1 / LR |
| AS:AX | Product / Bands=8 / Norm·Degradation profile / Pair_geometry_ID / Valid_mask_SHA |
| AY:AZ | Status / Readback |
| BA:BC | Source_manifest_SHA / Tile_layout_SHA / Evaluator_SHA |
| BD:BF | **Log_URL / Output_URL / Local_artifact_root** |
| BG:BH | Result_ID / Recorded_at |
| BI:BJ | FR_Reproj_SAM / FR_Spatial_corr |
| BK:BL | Review / Notes |

SC/FT는 `Fit_scene_ID=Eval_scene_ID`, `Input_adaptation=TRUE`다. ZS는 target update=0, fit scene은 빈칸이다. Brovey의 target optimization은 없고 ratio 계산 자체를 신경망 학습으로 세지 않는다. 단 별도 통계 fitting을 했다면 Notes에 명시한다. 비학습 방법에 없는 weight SHA·LR·epoch는 null이며, 알고리즘 config hash를 Notes에 보존한다.

한 행은 `campaign × scene × method × repeat × checkpoint/config SHA × selector × input/split/protocol`의 관측이다. `Result_ID`는 이 key의 canonical JSON SHA256이다. BEST의 Samples_seen·Updates는 **선택된 checkpoint 시점 값**을 적고, 20K까지 후보를 탐색한 총 비용과 범위는 Notes에 추가한다. 같은 모델의 selector alias는 독립 실험으로 세지 않는다.

로컬 원장을 source of truth로 두고, 신규 탭 전용 writer만 사용한다. 기존 `배포용 모델`, `WV3-main`, `_records` 및 기존 uploader에는 쓰지 않는다. 수기 Review를 덮지 않는다. 미측정 값은 blank/null로 둔다. 데이터행에 가짜 결과를 넣어 연결을 시험하지 않는다.

기존 status dropdown은 `PENDING_INPUT / RUNNING / EVAL_PENDING / EVAL_COMPLETE / BLOCKED / FAILED`다. 세부 BLOCKED 원인은 Notes·원본 로그에 넣는다. `EVAL_COMPLETE/VERIFIED`는 필수 산출물 확인과 실제 Sheet readback을 통과한 관측에만 사용한다.

### 10.3 실제 열람 가능한 로그

```text
<NEW_PROJECT_ROOT>/work_dir/PANPV_S1_WV2_5SCENE_20261002_v2/s1/
  campaign_manifest.json / source_manifest.json / donor_manifest.json
  scene_manifest.json / schema_mapping.json / protocol.json
  cases.json / queue.json / control.json
  preflight/
  scenes/<Scene_ID>/split_manifest.json / tile_index.json / roi_manifest.json
  runs/<Run_ID>/ ... §6.3 logs, checkpoints, metrics, outputs ...
  reports/scene_summary.csv / learning_curves.csv / final_report.md
  outbox/ / log_exports/
```

적어도 각 2K block 종료 때 **가벼운 로그 묶음**을 원자적으로 export한다. 대상은 run config·hash·split 요약·train/probe/validation JSONL·metric·cost·현재 상태이며, 원본 H5·원영상·credential·대용량 optimizer state는 기본 묶음에서 제외한다. append 중인 로그는 완료된 JSONL line까지 snapshot하고 export 시각·최신 committed step을 기록한다.

승인된 비공개 Drive/실험 저장소가 연결되면 export를 동기화하고 실제 접근 가능한 주소를 `Log_URL`에 쓴다. 연결 전에는 `Log_URL`을 비우고 `Local_artifact_root`, export 경로 및 `log_access=LOCAL_EXPORT_ONLY`를 기록한다. **서버 경로를 인터넷 URL로 꾸미거나 Sheet가 존재한다는 이유로 원격 로그가 연결됐다고 표시하지 않는다.**

로그 업로드 위치·권한은 실행 전에 확정한다. 사용자에게 전달 가능한 zip과 readback을 확인하여 이전처럼 최종 점수만 있고 학습곡선을 열 수 없는 상태를 구분한다. 사내 영상·민감한 위치·서비스 계정 키를 공개 GitHub에 올리지 않는다.

## 11. 최소 registry와 실행 담당자 제출물

아래는 **구현 입력용 설계**다. 현재 구현된 CLI나 실행 가능한 서버 경로가 있다는 뜻은 아니다. 실제 launcher의 `--help`, gate, source hash 확인 후 명령을 README에 기록한다.

```yaml
schema: PANPV_PRACTICAL_PLAN_v2
campaign_id: PANPV_S1_WV2_5SCENE_20261002_v2
server_allowlist: [s1]
training_processes: 1
input_root: null                  # 전송 후 실제 s1 경로로 확정
work_root: null                   # 기존 실험과 겹치지 않는 실제 경로
scene_files:
  - 200014103525_01.h5
  - 200014103539_01.h5
  - 200014103658_01.h5
  - 200014103676_01.h5
  - 200014103802_01.h5
methods: [BROVEY8, ZS_WV3, SC_SCENE, FT_WV3_SCENE]
bicubic_baseline: false
internal_bicubic_upsampling: true
model:
  width: 104
  depth: [1, 2, 2]
  input_layout: [PAN, LPAN_UP, PAN_MINUS_LPAN_UP, MS_UP]
  ms_bands: 8
  aligner: false
  warp: false
  teacher: false
  kd: false
  attention: false
  mode_modulation: false
parent:
  run_id: PDSP_S2_R01_S01_W104_D122_F100_PLH_SE271001_A01
  checkpoint_sha256: 6ad7cff8fc60542ab687a405f7b55430926942f008a65345de99fa69fa8b3846
  local_weight_path: null         # s1 전달·검증 후 확정
adaptation:
  repeats: {R01: 261002}
  reset_per_scene: true
  training_data: CURRENT_SCENE_RR_TRAIN_ONLY
  effective_batch: 48
  microbatch: null                # preflight에서 고정
  precision: fp32
  tf32: false
  scratch_lr: 0.0001
  finetune_lr: 0.00001
  lr_schedule: constant
  endpoint_updates: 20000
  yield_block_updates: 2000
  early_stopping: false
  auto_extension: false
  auto_add_repeats: false
  wallclock_limit_seconds: null
  diagnostic_steps: [0, 500, 1000, 2000, 5000, 10000, 20000]
selection:
  primary: EXACT_20000
  secondary: MIN_RR_VALIDATION_L1_TO_20000
  best_candidates: EVERY_2000_NONZERO
  tie_break: EARLIER_STEP
  use_fr_for_selection: false
sheet:
  spreadsheet_id: 1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0
  title: practical-validation
  sheet_id: 202610020
  header_row: 5
  data_start_row: 6
  overwrite_other_tabs: false
```

Run ID 예시: `PV_S1_V01_SC_R01_SE261002_A01`, `PV_S1_V01_FT_R01_SE261002_A01`. 실제 Scene_ID와 Method를 각 run manifest에 모두 넣는다. ZS/Brovey도 고유 inference record ID를 사용하지만 학습 run으로 세지 않는다.

### 실행 시작 보고

s1 실제 hostname/GPU·경로·환경, H5 5개 실측 key/shape·native/RR 구분·patch 수·좌표 검증, 입력/도너/프로토콜 hash, G0–G5 receipt, 실제 10개 학습 run, 큐·로그 접근 경로, 실행 명령·PID를 제출한다. 계획서 작성·파일 전송·preflight 완료·실제 학습 시작을 분리해 보고한다.

### 완료 또는 중단 보고

완료한 방법×씬과 누락 원인, EXACT/BEST 각각의 checkpoint·출력·지표, Brovey 대비 품질/시간, 학습곡선과 씬별 실패를 보고한다. 중단 시 마지막 정상 fullstate, 다음 queue 항목, 미완료 평가·outbox, 재개 명령을 남긴다. 성능이 좋은 씬만 남기지 않는다.

## 12. 근거 및 확인 범위

| ID | 근거 | 적용 범위 |
|---|---|---|
| R0 | 첨부 `PANDEP_S2_Shared_PLH_UNet_Experiment_Cases_2026-10-01_v4.md` §3·4·8·9·14 | 모델·PLH·LP·증강·로그/타일 수치 계약. s2 전용 운영과 사내 데이터 제외 규약은 이번 독립 campaign에 적용하지 않음. |
| R1 | 첨부 `PANPV_WV2_5Scene_Practical_Validation_2026-10-02_v1.md` | 세 방법, donor 조회값, RR/FR 분리, 20K/학습률/분할 설정, Brovey·Sheet·로그 설계 계승. |
| R2 | 이번 사용자 메시지와 파일 목록 화면 | s1 지정, Bicubic 비교군 제외, H5 파일명 5개, `_reg` JSON 표시 2개. H5 내용이나 JSON 의미의 증거는 아님. |
| R3 | `practical-validation` live read → scoped update → readback, 2026-10-02 | sheetId/64열/드롭다운·상단 설명·관측행 상태. s1에서 실행했다는 증거는 아님. |

근거 파일 무결성(실행 데이터·weight 검증과 별개):

```text
v1_plan_sha256: 6dbb6bc3c62c69b570c5adb69ff0e938e3c6450be54837fbead80b43591f07f0
v4_model_plan_sha256: eccb999c0a1c2f5ba72a238b7f44904a1bc9fea87985773484382a6c1bdd72ee
input_list_screenshot_sha256: db6d835c4977da8dc88ebc1a5934e297e9373e69105bf6b4ac64ddde7f63e86a
```

**최종 지시:** s1에서 지정 H5 5개의 실물 구조와 native FR 처리 가능성을 확인한 뒤, 동일 WV3 donor를 사용하는 Zero-shot·씬별 Scratch·씬별 Fine-tuning을 독립 실행하고 8밴드 Brovey와 비교한다. Bicubic 독립 비교군은 만들지 않으며 PLH 내부 보간은 유지한다. 결과는 `practical-validation`에만 기록하고 원본 로그·실제 접근 경로·checkpoint·전체 씬 출력을 함께 보존한다. 이번 문서는 실제 서버 학습을 이미 시작했다는 보고가 아니다.
