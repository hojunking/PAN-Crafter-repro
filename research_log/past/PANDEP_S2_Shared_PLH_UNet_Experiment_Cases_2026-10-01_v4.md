# s2 전용 배포용 Shared PLH U-Net — 실험 Case 및 구현·운영 지시서

- 작성일: **2026-10-01**
- 문서 버전: **v4 — 기존 배포 시트의 C00/C11/C01 의미 유지 + 단독학습 대조군·노출량 비교·무기한 학습 규약 구체화**
- Campaign ID: `PANDEP_S2_SHARED_PLH_20261001_v4`
- 실행 서버: **s2만 사용**
- 학습 데이터: **WV3 / GF2 / QB의 지정된 benchmark train split만 사용**
- 결과 시트: **`pan-cvpr27` → `배포용 모델`**
- 상태: **실험 설계·구현 지시서. 본 문서 작성 과정에서는 서버 구현·학습 기동·시트 쓰기를 수행하지 않았다.**

> **실행 담당자에게:** 기존 연구 method·trainer·controller·uploader를 수정하거나 이어서 실행하지 말고, 별도의 프로젝트·환경·작업 디렉터리에서 본 실험을 구현한다. 모든 case는 **PAN + LPAN-up + HPAN + MS-up**을 사용한다. **Aligner, PAN warp, Teacher, KD는 생성하거나 실행하지 않는다.** 실행 전 검증을 통과하면 s2에서 무기한 controller를 시작하고, 사용자의 중단 또는 안전 정지 조건 전까지 실험을 계속한다. 50K/150K는 비교·저장 지점이지 전체 실험 종료 시한이 아니다.

---

## 0. 이번 문서에서 확정하는 범위

### 0.1 사용자의 결정

1. s2에서 WV3·GF2·QB만으로 shared-backbone 실험을 한다.
2. 기본 구조는 기존 **C01 방식의 PLH 입력을 사용하는 단순 residual U-Net**이다.
3. 센서별 입력층·출력층만 분리하고 encoder·bottleneck·decoder는 공유한다.
4. 학습·운영의 wall-clock 상한은 두지 않는다. 사용자가 중간에 끊을 수 있어야 한다.
5. 기록은 `배포용 모델` 탭에 저장한다.
6. 구현과 실행은 기존 연구 코드와 분리한다. 기존 실험·가중치·원본 데이터·시트 원장은 훼손하지 않는다.
7. 사내 데이터셋은 **학습, 검증, 정규화 통계, subset 선택, 모델 선택에서 모두 제외**한다.

### 0.2 현재 시트와의 연결 — 이름 혼동 금지

2026-10-01 live 조회에서 `배포용 모델` 탭은 이미 존재했다. 헤더·메모에는 아래 세 배포 case가 정의되어 있었다.[S01]

| 본 문서 표기 | 시트 `Case_ID` | 현재 배포 시트에서의 의미 |
|---|---|---|
| **DEP:C00** | `C00` | Shared PLH U-Net, W104/D122, train 100% |
| **DEP:C11** | `C11` | Shared PLH U-Net, W128/D122, train 100% |
| **DEP:C01** | `C01` | Shared PLH U-Net, W104/D122, train 50% |

**이 배포 case 번호는 과거 ABLR2의 C00/C01/C11과 다르다.**

- **ABLR2:C01**은 이 문서가 계승하는 `PLH ON / Aligner OFF / Teacher OFF` 방식이다.[S03]
- **DEP:C00도 PLH를 사용한다.** 과거 ABLR2:C00처럼 L/H를 0으로 마스킹하지 않는다.
- **DEP:C01은 데이터 50% 실험**이다. 입력 구성 변경 실험이 아니다.
- **DEP:C11은 width 128 실험**이다. 과거 연구의 같은 번호 case로 해석하지 않는다.
- 식별은 반드시 `(Campaign_ID, Case_ID, Repeat, Attempt)`로 한다.

위 세 case의 의미는 유지한다. 직전 합의의 **단독 3×50K 대 공유 150K 비교**를 위해 `S01/S02/S03` 단독학습 대조군을 추가한다. 이하 seed·LR·queue·평가 주기·안전 조건의 구체적 수치는 **이번 계획의 설계값**이며 과거 학습 설정이 이미 이와 같았다는 뜻이 아니다.

### 0.3 이번에 하지 않는 것

Aligner/Teacher/KD, PAN 정합·flow·`grid_sample`, attention/Swin/SE/FiLM/센서 embedding, FR 자기지도 loss, SAM/edge 보조 loss, 4-band zero-padding을 통한 가짜 8-band 공통화, 사내 데이터, WV2 zero-shot, 자동 architecture search는 이번 case 집합에 없다. 새 기능을 추가하려면 별도 계획 버전을 만든다.

---

## 1. 목적과 판정 질문

| 질문 | 비교 | 해석 범위 |
|---|---|---|
| 공유해도 단독 모델의 복원 성능을 유지하는가? | DEP:C00 대 S01/S02/S03 | 같은 센서별 sample 노출량에서 비교 |
| 추가 학습이 필요한가? | 같은 run의 노출량 단계별 checkpoint | 같은 가중치의 연속 학습곡선; 독립 반복으로 세지 않음 |
| 공통 복원망의 용량이 부족한가? | DEP:C11 대 DEP:C00 | 데이터·loss·sampling을 고정한 width 증가 비교 |
| 고유 학습 데이터 수가 중요한가? | DEP:C01 대 DEP:C00 | 같은 sample 노출량에서 train subset 크기만 변경 |
| 배포 비용에 맞는가? | 센서별 활성 경로의 추론·메모리·연산량 | 서버 비용이며 laptop 실측을 대신하지 않음 |

성능 개선이나 단조 수렴을 보장하지 않는다. 작은 데이터에서 overfitting, shared trunk의 센서 간 간섭, 용량 제한은 서로 다른 원인이므로 train/validation 곡선과 matched-budget 대조군을 함께 본다.

---

## 2. 구현·실행 완전 분리

### 2.1 별도 프로젝트와 경로

기존 연구 repo와 **다른 물리 디렉터리**에 독립 프로젝트를 만든다. 아래는 논리 경로이며, 실제 절대경로는 s2 read-only inventory 후 manifest에 확정한다.

```text
<NEW_PROJECT_ROOT>/                 # 예: 기존 repo의 sibling pan_deploy_shared/
  pyproject.toml
  requirements.lock
  README.md
  src/pan_shared/
    model.py                       # SharedPLHUNet / SingleSensorPLHUNet
    blocks.py                      # 고정된 U-Net 수치 블록
    frontend.py                    # normalization / LPAN / L / H / MS-up
    data.py                        # server와 sensor를 분리한 데이터 로더
    sampling.py                    # 센서 교대·sample·augmentation stream
    losses.py
    train.py
    evaluate.py
    metrics/                       # 검증된 순수 metric 코드의 고정 복사본
    checkpoints.py
    controller.py
    safety.py
    logging.py
    sheets.py                      # 신규 탭 전용 uploader
    export.py
    cli.py
  configs/<Campaign_ID>/
  tests/
  vendor_reference/                # 출처·라이선스·hash를 보존한 검증용 코드
  work_dir/<Campaign_ID>/s2/
    campaign_manifest.json
    architecture_manifest.json
    source_manifest.json
    dataset_manifest.json
    subset_manifest.json
    seed_registry.json
    cases.json
    queue.json
    control.json
    controller.lock
    preflight/
    runs/<Run_ID>/
    outbox/
    reports/
    exports/
```

- 학습 cache·결과·임시 파일은 신규 campaign 아래에만 쓴다.
- 원본 benchmark는 검증된 읽기 전용 경로로 참조한다. 동일 raw bytes를 새 위치에 복사한 경우 원 hash와 대조한다.
- `realpath` 기준으로 신규 결과 경로가 기존 run/campaign 디렉터리와 겹치지 않는지 검사한다. symlink를 통한 우회도 금지한다.
- 신규 Python 환경을 만든다. 기존 conda 환경의 package upgrade, 전역 `pip install`, CUDA driver 변경을 하지 않는다.
- 기존 repo의 branch switch, `git reset/clean`, working tree 수정, 실행 중 trainer 교체를 하지 않는다.
- 새로운 controller·uploader는 고유 PID/lock을 사용한다. 기존 daemon을 재사용하거나 monkey-patch하지 않는다.

### 2.2 기존 코드에서 계승할 것과 금지할 것

**계승:** 고정 source의 U-Net 블록 연산, PLH 전처리, 데이터·metric 수치 규약. source file hash·원 module·라인 또는 symbol mapping·라이선스를 기록한다.[S02–S06]

**금지:** 기존 `qg40.build_model()`, `fh12.build_model()`, ABLR2 trainer/controller를 새 campaign의 최상위 실행기로 사용하는 것. 기존 `ablr2`는 `s1=WV3 / s2=QB / s3=GF2` lane과 기존 시트에 묶여 있으므로, lane 검사를 해제해 shared 학습을 끼워 넣지 않는다.[S04]

`PANCrafterPaper`의 **attention-free, mode-modulation-free 수치 구조를 별도 namespace에 분리 구현**한다. 원본의 사용하지 않는 attention 관련 import까지 runtime 의존성으로 끌고 오지 않는다. validation용 원본 forward는 별도 테스트 프로세스에서만 비교한다. 구조를 단순화한다는 이유로 LayerNorm 종류, Down/UpConv, residual 위치, padding을 바꾸지 않는다.

### 2.3 실제 s2 가동 전

s2의 실제 hostname/server mapping, GPU UUID·활성 process, 기존 controller/queue, RAM·VRAM·디스크·data mount를 읽기 전용으로 기록한다. **서버 별칭이나 GPU 모델은 추측하지 않는다.**

기존 GPU 학습이 활성 상태이면 새 학습을 겹쳐 띄우거나 기존 작업을 자동 중단하지 않는다. `WAIT_RESOURCE`로 남기고 자원이 비거나 사용자가 해당 기존 작업의 중단을 승인했을 때 시작한다. 다른 서버의 결과·공유 lock은 요구하지 않는다.

---

## 3. 고정 모델 계약

### 3.1 공통 입력

센서 `s ∈ {WV3, GF2, QB}`, 밴드 수 `C_s`에 대해:

```text
PAN   : [B, 1, 4h, 4w]
MS    : [B, C_s, h, w]
LPAN  : [B, 1, h, w]
MS_up : bicubic(MS, ×4, align_corners=False)
L     : bicubic(LPAN, ×4, align_corners=False)
H     : PAN - L
X_s   : concat([PAN, L, H, MS_up], channel axis)
```

| 센서 | C_s | 실제 입력 채널 순서 | stem | head |
|---|---:|---|---|---|
| WV3 | 8 | `[P, L, H, Coastal, B, G, Y, R, RE, NIR1, NIR2]` | 3×3 Conv, 11→W | 3×3 Conv, W→8 |
| GF2 | 4 | `[P, L, H, B, G, R, NIR]` | 3×3 Conv, 7→W | 3×3 Conv, W→4 |
| QB | 4 | `[P, L, H, B, G, R, NIR]` | 3×3 Conv, 7→W | 3×3 Conv, W→4 |

GF2/QB의 밴드 이름·개수가 같아도 별도 stem/head를 둔다. 파장 응답이 같다고 가정하지 않는다. 밴드 순서는 현재 repository의 benchmark convention이며, 임의 GeoTIFF의 밴드 metadata를 확인했다는 뜻이 아니다.[S05]

### 3.2 Shared 모델

```text
WV3 X → stem[WV3] ─┐                       ┌→ head[WV3] → + MS_up[WV3]
GF2 X → stem[GF2] ─┼→ 단 하나의 공유 trunk ─┼→ head[GF2] → + MS_up[GF2]
 QB X → stem[QB]  ─┘                       └→ head[QB]  → + MS_up[QB]
```

```text
shared trunk:
  encoder1 → down1 → encoder2 → down2 → middle
  → up2 → decoder2 → up1 → decoder1 → shared output LayerNorm → SiLU
```

- W는 모든 scale에서 기존 코드와 같은 방식으로 사용한다. 일반적인 U-Net처럼 width를 임의로 2배씩 늘리지 않는다.
- `depth=[1,2,2]`: encoder1=1, encoder2=2, middle=2; decoder1=1, decoder2=2로 기존 mirror 규칙을 따른다.
- encoder·bottleneck·decoder·skip 경로와 출력 직전 LayerNorm/SiLU는 공유한다.
- 센서별인 것은 입력 Conv와 마지막 residual 출력 Conv뿐이다.
- 호출은 `forward(sensor_id, pan, ms, lpan)` 형태로 하고 선택한 센서의 stem/head만 실행한다.
- 최종 출력은 **`MS_up + residual` 한 번**이다. head에 base를 넣고 wrapper에서 다시 더하지 않는다.
- Aligner module/parameter는 0개다. `alignment=off`인 A+U 모델을 우회 실행하는 방식이 아니라 **처음부터 A를 만들지 않는 그래프**다.
- PAN은 수치상 원본 정규화 PAN 그대로 사용한다. delta=0으로 `grid_sample`을 호출하는 방식도 금지한다.

수식:

\[
\hat Y_s=M_s^\uparrow+O_s\big(U_\theta(E_s([P_s,L_s,H_s,M_s^\uparrow]))\big).
\]

### 3.3 Single 대조군

S01/S02/S03은 위와 같은 전처리·W104/D122·블록 연산을 사용하되, 해당 센서의 stem/head와 **독립 trunk** 하나만 갖는다. Shared의 해당 센서 초기 경로와 tensor가 대응되도록 초기화한다. 과거 Sheet의 다른 학습 run을 신규 대조군으로 복사해 등록하지 않는다.

### 3.4 초기화

모든 정식 case는 **from scratch**다. 앞서 다운로드 대상으로 선정한 s1 WV3 C01 checkpoint는 선택적인 구조 parity fixture일 뿐, 본 학습의 donor나 필수 의존성이 아니다.

- W104/D122 case 사이에는 같은 repeat의 canonical parameter 이름·shape·seed를 기준으로 동일 초기값을 생성한다.
- canonical key에는 case 번호·subset fraction·실행 순서를 넣지 않는다.
- `stem/WV3`, `stem/GF2`, `stem/QB`는 별도 key를 사용한다. GF2와 QB의 가중치를 동일 객체로 공유하지 않는다.
- 기존 C01 초기화처럼 새 stem의 **L/H 입력 kernel은 0으로 시작하되 학습 가능**하게 둔다. 입력 L/H tensor 자체를 0으로 만들지는 않는다.
- residual block의 zero projection과 최종 zero residual head 초기화를 보존한다.
- W128은 shape가 다르므로 W104와 tensor 동일 초기화를 주장하지 않는다. seed와 데이터 stream만 대응시킨다.
- 첫 zero-head 상태에서 `Y=MS_up`이고 일부 trunk gradient가 0인 것은 정상이다. 이 초기 현상을 그래프 단절로 오판하지 말고 후속 step과 별도 수치 fixture로 검증한다.

---

## 4. 데이터 계약

### 4.1 지정 source

기준은 고정 commit의 `ablr2/sensor_sources.json`이다.[S05] 실제 절대경로는 s2에서 바인딩하되, 원본 byte SHA를 유지한다.

| 센서 | train | validation | train 기대 개수 | validation 기대 개수 |
|---|---|---|---:|---:|
| WV3 | `data/PanCollection/WV3/train_wv3.h5` | `data/PanCollection/WV3/valid_wv3.h5` | 9,714 | 1,080 |
| GF2 | `data/PanCollection/GF2/train_gf2.h5` | `data/PanCollection/GF2/valid_gf2.h5` | 19,809 | 2,201 |
| QB | `data/PanCollection/QB/train_qb_msfix.h5` | `data/PanCollection/QB/valid_qb_msfix.h5` | 17,139 | provenance상 1,905; 실제 shape 확인 |

QB는 **raw `train_qb.h5`/`valid_qb.h5`로 대체하지 않는다.** 현재 catalog의 `msfix`와 그 원본 GT/PAN 보존 검증을 계승한다. source 규약에서는 QB validation 개수 검사가 일부 선택적이므로, 기록된 1,905와 실제 shape를 확인하고 불일치를 조용히 강제 수정하지 않는다.[S04–S06]

RR: 센서별 `reduced_examples_h5/test_<sensor>_multiExm1.h5`, 각 20장.

FR: 센서별 `full_examples_mat20/test_<sensor>_OrigScale_mat20.h5`, 각 20장. 구형 mixed FR H5나 임의 repaired 대체 데이터로 fallback하지 않는다.

### 4.2 Byte hash 기준

| 센서 | split | SHA256 |
|---|---|---|
| WV3 | train | `f9e7ec381390ff36ddffd0714df7af07a3c5f456078b219a25f3e18372228f82` |
| WV3 | val | `ba883e57408d1f3c2263cc2818a9fbbf2dbe8e62554c531288f9e8bf9f59dc87` |
| WV3 | rr | `00db0d62f63693410935208e287aea21a736d11840d92eedd2d093c99be5314a` |
| WV3 | fr | `8fe579a59594ddf42bce19743a5b65361feab22cbd0c8f433d3173f29b87a4d6` |
| GF2 | train | `243a0bc8a4cc0a2740ed24409f9e87afe531d87e9d57524c6bcbec2b5f937621` |
| GF2 | val | `ae15b19ccc6ece799ebfeb3fb374b1a333238226397441e3d523b0195dff4f43` |
| GF2 | rr | `709a9a53b2e0f29d3dcd5c6ca4410c2913b62cc03c104fed6c049911dbe1c8ea` |
| GF2 | fr | `e52d151f262a74f96e59f03124f86a27d80841073ce60376856ae679020d00dc` |
| QB | train | `c7fb3ef0744f3fd21330925003f6bd6eb110f8a181c23ea6d9ce086c81710ba9` |
| QB | val | `79f5bb48905876e8b278868a4ca9b266ef52582bd26b9c31b60462b708a411fa` |
| QB | rr | `9842a1232ad2d5b9a61c9fd7fb353e8fc6214eeac6e947d888f221ad6ecdda35` |
| QB | fr | `bf6ef5df83d2134d5175c00b38d22a2390345d246d00140d758c796761a5aa24` |

hash 불일치는 `BLOCKED_DATA_IDENTITY`로 처리한다. H5 내용이 같아 보인다는 이유만으로 바뀐 bytes를 같은 dataset version으로 취급하지 않는다. 새 버전을 쓸 경우 사용자 확인 후 새 campaign/config hash로 분리한다.

### 4.3 공간·수치 규약

- Train/val: PAN `[1,64,64]`, MS `[C,16,16]`, GT `[C,64,64]`.
- RR: PAN 256×256; FR: PAN 512×512. 실제 H5 geometry를 전수 검사한다.
- Normalization: `2 * DN / max_dn - 1`; WV3/QB=2047, GF2=1023.
- GT와 MS/PAN은 해당 센서의 동일한 affine normalization을 적용한다.
- LPAN은 **해당 sample의 native PAN**에서 생성한다. Gaussian σ=1.98, kernel 41, replicate padding, `[2::4,2::4]`, 계산 float64/cache float32.[S07]
- Native DN PAN에서 LP cache를 만든 후 normalization·동기화 증강을 한다. legacy recipe를 임의의 resize 또는 센서 MTF filter로 교체하지 않는다.
- `L=U4(LPAN_norm)`, `H=PAN_norm-L`. HPAN을 다시 `[-1,1]`로 정규화하거나 절댓값/clip 처리하지 않는다.
- `MS_up=U4(MS_norm)`; 데이터셋에 저장된 `lms`를 모델의 base로 조용히 대체하지 않는다. 공식 evaluator가 필요로 하는 original LMS는 별도로 유지한다.
- 모든 finite zero는 원칙적으로 정상 값으로 보존한다. 0을 nodata로 자동 판단하지 않는다.
- PAN/GT nominal range는 source 규약대로 검사한다. finite MS/LMS의 보간 ringing은 기록하고 보존한다. 일괄 clipping 금지.[S05–S06]
- 모든 split·tensor key의 nonfinite, dtype, shape, min/max, zero/saturation count, 밴드 순서, raw/LP hash를 manifest에 남긴다.

### 4.4 증강

기존 fixed H-flip → fixed V-flip → `rot90(k), k∈{0,1,2,3}` four-view 계약을 계승한다.[S07] PAN/MS/GT/LPAN에 동일한 view를 적용한다. 이 규칙을 independent random H/V flip 8-view로 바꾸지 않는다. crop, radiometric jitter, synthetic shift는 이번에 추가하지 않는다.

검증·평가는 augmentation 없이 원 sample 순서로 진행한다. augmentation은 `(base_sample_id, rot_id)`로 명시적으로 기록·재현한다.

### 4.5 50% subset — DEP:C01만

센서마다 `floor(0.5*N_train)`개의 원본 sample ID를 뽑는다.

| 센서 | full | 50% subset |
|---|---:|---:|
| WV3 | 9,714 | 4,857 |
| GF2 | 19,809 | 9,904 |
| QB | 17,139 | 8,569 |

선택 규약:

```text
subset_seed = 20261001
rank_key(i) = SHA256(UTF8("PANDEP_SUBSET_v1|20261001|" + sensor + "|" + source_sha256 + "|" + decimal_sample_id))
각 sensor에서 (rank_key, sample_id) 오름차순으로 정렬하고 앞 floor(N/2)개 선택
실제 ID 목록과 정렬·저장 규약의 SHA256을 manifest로 동결
```

subset은 **세 repeat에서 동일하게 고정**한다. 반복 실험은 초기화·학습 순서에 대한 변동을 보며, subset 추출 자체의 통계적 변동까지 추정한다고 주장하지 않는다. 증강 view를 기준으로 50%를 뽑거나 validation/test에서 subset을 고르지 않는다. validation/RR/FR은 모든 case에서 **100% 동일**하게 사용한다.

---

## 5. 실험 Case 표 — 6개 설계, 3개 독립 반복

### 5.1 정식 case

| Case_ID | 표시 이름 | Mode | 데이터 | W / depth | train 비율 | 최초 비교 지점 | 주요 대조 |
|---|---|---|---|---|---:|---:|---|
| **C00** | Shared-Base | SHARED | WV3+GF2+QB | 104 / [1,2,2] | 각 100% | global 150K | S01/S02/S03 |
| **C11** | Shared-Wide | SHARED | WV3+GF2+QB | 128 / [1,2,2] | 각 100% | global 150K | C00 |
| **C01** | Shared-HalfData | SHARED | WV3+GF2+QB | 104 / [1,2,2] | 각 50% | global 150K | C00 |
| **S01** | Single-WV3 | SINGLE | WV3 | 104 / [1,2,2] | 100% | local 50K | C00의 WV3 결과 |
| **S02** | Single-GF2 | SINGLE | GF2 | 104 / [1,2,2] | 100% | local 50K | C00의 GF2 결과 |
| **S03** | Single-QB | SINGLE | QB | 104 / [1,2,2] | 100% | local 50K | C00의 QB 결과 |

모든 case는 **PLH ON / Aligner OFF / Teacher OFF / mean-L1 / FP32 / from scratch**다. 같은 번호의 과거 연구 run에서 resume하거나 warm-start하지 않는다.

### 5.2 반복과 이름

| Repeat | model/data master seed |
|---|---:|
| R01 | 271001 |
| R02 | 271002 |
| R03 | 271003 |

최초 등록은 **6 cases × 3 repeats = 18개 logical run**이다. 같은 logical run의 block·checkpoint·pause/resume를 새로운 독립 반복으로 세지 않는다.

```text
Run_ID 예시:
PDSP_S2_R01_C00_W104_D122_F100_PLH_SE271001_A01
PDSP_S2_R01_C11_W128_D122_F100_PLH_SE271001_A01
PDSP_S2_R01_C01_W104_D122_F050_PLH_SE271001_A01
PDSP_S2_R01_S01_W104_D122_F100_PLH_SE271001_A01
```

RNG stream은 `model-init / sensor-order / sample-order/<sensor> / augmentation/<sensor> / probes`로 분리한다. 정확한 seed derivation과 생성된 ID/order hash를 등록한다. case마다 seed를 성능에 따라 새로 뽑거나 실패한 seed를 말없이 교체하지 않는다.

`A01`의 단순 exact-resume는 attempt를 늘리지 않는다. 수치 설정·초기화가 달라진 재시작은 새 attempt 또는 새 plan revision으로 기록하고 이전 실패를 보존한다.

### 5.3 유효성 검사용 Q00

정식 run 전에 **Q00 shared-W104 / 3센서 × 2회 방문 = 6 optimizer step** smoke test를 별도 scratch 경로에서 수행한다. 목적은 그래프·loader·loss·checkpoint·resume·uploader 검증이다. 정식 case의 가중치·optimizer·sample cursor로 재사용하지 않고, 연구 성능 관측행으로 올리지 않는다. 추가 resume equivalence 테스트는 별도 결정론적 fixture에서 진행한다.

---

## 6. 학습 알고리즘과 노출량

### 6.1 Batch 구성

- **한 optimizer step의 batch는 한 센서만 포함**한다.
- effective batch size=48. 원본 sample이 epoch 끝에서 부족하면 다음 shuffle epoch에서 이어서 채운다. tail을 매 epoch 버리는 방식으로 노출을 누락하지 않는다.
- shared는 세 step마다 WV3/GF2/QB를 각각 한 번 선택한다. 세 센서의 순서는 triplet마다 독립 RNG로 permutation한다.
- 따라서 3의 배수 global step에서 센서별 update 수는 정확히 같다. **50,000은 3의 배수가 아니므로 50K block 경계의 센서별 횟수를 50K/3이라고 가정하지 않는다.** 미완성 triplet의 순서와 위치를 저장하고 다음 block에서 이어간다.
- 각 센서 sample/augmentation stream은 서로 독립적이다. 데이터셋 소진은 해당 센서만 재순회한다.
- C00과 대응 Single은 같은 repeat·센서에서 동일한 sample/augmentation stream을 쓴다. C11도 동일 full-data stream을 사용한다. C01은 동결된 subset 안에서 동일 규약을 따른다.

메모리 부족 시 **사전 검증 단계에서** microbatch 24×2 또는 12×4로 effective batch 48을 유지하는 구성을 정한다. 각 microbatch의 mean loss를 그 sample 비중으로 가중 합산한다. 비교군의 논리 batch·sample 순서·optimizer step 수는 유지한다. 정식 실행 도중 batch·precision을 자동 변경하지 않는다.

### 6.2 Loss

\[
L_s=\frac{1}{B C_s H W}\sum|\hat Y_s-Y_s|.
\]

정규화된 출력과 GT의 **전체 원소 mean L1**만 사용한다. clamp 이전 출력으로 계산한다. 센서별 batch가 균형이므로 shared의 목표는 세 센서 기대 loss의 동일 가중 평균이다. 선택된 센서의 step loss를 다시 1/3로 나누지 않는다.

```text
optimizer.zero_grad(set_to_none=True)
y = model(sensor, pan, ms, lpan)
loss = mean(abs(y - gt))
loss.backward()
optimizer.step()
```

선택되지 않은 stem/head는 grad=None이어야 하고 parameter 및 해당 optimizer moment/step도 변하지 않아야 한다. weight decay가 비활성 branch를 갱신하지 않는지 실제 unit test로 검사한다. trunk parameter는 optimizer에 **한 번만 등록**한다.

### 6.3 Optimizer 및 무기한 LR

```text
AdamW
betas = [0.9, 0.999]
eps = 1e-8
weight_decay = 0.01  # 이번 계획에서는 trainable parameter 전체에 동일 적용
peak_lr = 1e-4
initial_lr = 1e-6
floor_lr = 1e-5
AMP / TF32 = OFF (reference FP32)
gradient_clipping = OFF (관측만 기록; nonfinite는 안전 정지)
EMA = OFF
```

`a=3`(shared), `a=1`(single), 다음 optimizer update의 1-based 번호를 `t`라 한다.

```text
warmup = 100 * a
cosine_end = 50000 * a

1 <= t <= warmup:
  lr = initial_lr + (peak_lr-initial_lr) * t/warmup
warmup < t <= cosine_end:
  p = (t-warmup)/(cosine_end-warmup)
  lr = floor_lr + 0.5*(peak_lr-floor_lr)*(1+cos(pi*p))
t > cosine_end:
  lr = floor_lr
```

- step에 사용한 LR과 다음 step의 LR을 명확히 구분해 로그에 남긴다.
- shared 150K와 single 50K까지 센서별 노출량에 대응하는 schedule을 맞춘다. triplet 내 LR의 작은 차이는 기록된 정확한 step 규약을 따른다.
- 그 이후에도 **LR=1e-5로 계속 학습**한다. cosine 종료를 학습 종료로 취급하지 않는다.
- 50K block이 끝나도 LR warmup, optimizer, model, RNG를 재시작하지 않는다.
- floor LR의 정체·진동이 관찰되어도 자동 tuning하지 않는다. 바꿀 경우 별도 recipe/parent 관계로 기록한다.

### 6.4 공정한 비교 축

| 노출량 단계 k | Single별 누적 step | Shared별 global step | 각 센서 sample 노출(48 batch) |
|---:|---:|---:|---:|
| 1 | 50,000 | 150,000 | 2,400,000 |
| 2 | 100,000 | 300,000 | 4,800,000 |
| 3 | 150,000 | 450,000 | 7,200,000 |
| k | 50,000×k | 150,000×k | 2,400,000×k |

주 비교는 같은 `k`·repeat의 **EXACT checkpoint**다. shared 50K는 진행 확인용이지 single 50K와 같은 센서별 노출량이 아니다. C11은 C00과 노출량이 같지만 연산량·시간은 같지 않다.

C01과 C00은 같은 sample 노출량에서 비교한다. C01은 고유 sample 수가 절반이므로 유효 epoch가 약 두 배다. 이를 함께 표시해야 한다. 보조적으로 shared 75K checkpoint를 모든 shared case에 저장하여 C01@75K와 C00@150K의 **근사 동일 유효 epoch** 비교를 제공한다. 홀수 sample 개수의 floor 때문에 정확히 같지 않은 값은 실제 분모로 계산한다. 보조 비교는 동일 update/compute 비교가 아니다.

---

## 7. 무기한 controller와 실행 순서

### 7.1 기본 규칙

```text
wallclock_limit = null
campaign_deadline = null
max_global_steps = null
early_stopping = false
until_operator_stop = true
s2_training_processes = 1
shared_yield_block = 50000 completed optimizer updates
single_yield_block = 50000 completed optimizer updates
```

개별 학습 프로세스는 block 끝에서 상태를 저장하고 다음 case에 자원을 양보할 수 있다. **프로세스 종료/전환과 logical run 완료는 다르다.** controller는 계속 살아서 다음 block을 실행한다. 18개 run은 최초 비교 단계 이후에도 같은 상태에서 계속 학습한다.

### 7.2 실행 우선순위

1. Preflight, Q00, architecture/수치/resume/Sheet dry-run gate를 완료한다.
2. R01의 **C00을 첫 50K**까지 실행한다. 가장 먼저 shared 결과를 확인할 수 있게 한다.
3. R01의 S01 → S02 → S03을 각각 50K까지 실행한다.
4. R01의 C00을 100K, 150K까지 이어서 실행한다.
5. R01의 C11, C01을 각각 150K까지 실행한다. 50K마다 fullstate와 결과를 저장한다.
6. R02, R03의 모든 case를 같은 최초 비교 지점(shared150K/single50K)까지 실행한다. 정식 시작 전에 그 순서를 `queue.json`에 동결한다.
7. 그다음 `k=2,3,...`로 계속 확장한다. 각 k에서 모든 repeat의 Single은 `50K*k`, Shared는 `150K*k`까지 이어서 학습한다.

무기한 확장 단계에서는 repeat별로 C00/S01/S02/S03/C11/C01을 순회하되, 아직 목표에 못 미친 각 case에 **50K block 하나씩** 배정한다. Shared가 세 block을 받아야 matched-exposure 단계가 완성된다. 모든 case가 해당 k에 도달하면 다음 k로 간다.

**한 case를 영원히 독점 실행하지 않는다.** 사용자가 중단하기 전에는 다른 case도 정해진 순서로 진행한다. 성능이 나쁘다는 이유로 case를 삭제하거나 repeat를 바꾸지 않는다.

### 7.3 별도 재학습 없이 이어가기

- Resume는 `last`의 모델·모든 optimizer state·scheduler·RNG·sensor triplet cursor·3개 sample cursor·augmentation state·노출 count·비용 누적값을 모두 복구한다.
- DataLoader prefetch로 미리 읽었지만 optimizer에 쓰지 않은 sample은 노출량에 포함하지 않는다. committed batch ledger를 기준으로 재개한다.
- `k` 증가 시 기존 checkpoint에서 계속 학습한다. 추가 학습을 fresh run이라고 표시하지 않는다.
- 초기18개 외에 새 seed를 무한 생성하지 않는다. 이번 무제한의 기본 의미는 등록한 세 repeat의 장기 학습이다.

---

## 8. 평가, checkpoint 선택, test 사용 경계

### 8.1 평가 주기

기존 시트 메모의 100/2K/10K/25K/50K cadence를 계승한다.[S01]

| 주기 | 작업 |
|---|---|
| 매 step | finite 검사·committed sample/step count 갱신 |
| 100 global/local step마다 | 센서별 train loss·grad norm·LR·성능/자원 요약 로그 |
| 2,000 step마다 | 동결한 train-probe 및 val-probe 평가; 검증된 `last_A/last_B` resume checkpoint 교대 저장 |
| 10,000 step마다 | 해당 run의 전체 validation 평가 |
| 각 50K block의 +25K, +50K | 전체 validation 추가 실행; 중복 step은 한 번만 |
| 각 50K block 끝 | EXACT checkpoint의 해당 센서 RR20/FR20 평가·시트 기록 |
| shared 75K | 위 보조 비교용 EXACT checkpoint 저장 및 RR20/FR20 추가 평가 |
| 모델/코드 버전별 최초1회 | 고정 입력의 추론 시간·메모리·parameter/FLOPs profile |

Shared validation은 세 센서를 모두 평가하고, **한 checkpoint·동일 model bytes**에 연결한다. Single은 해당 센서만 평가한다. train/val probe는 센서당128개 원 sample을 동결한다. train probe는 50% subset 내에서 공통으로 뽑아 모든 case에서 같은 ID를 쓴다. 선택은 hash 기반의 별도 `probe` key로 하며 test를 사용하지 않는다.

probe·validation·test는 학습 sample 순서, 모델 RNG 또는 optimizer를 변경하지 않아야 한다. mode/RNG 복구를 테스트한다. 지표 계산은 같은 sensor별 공식 수치 규약을 사용한다.

### 8.2 Primary: EXACT matched exposure

- 핵심 비교: `EXACT@150K*k`(Shared) 대 `EXACT@50K*k`(Single).
- 성능에 따라 step을 바꾸지 않는 **주 분석**이다.
- C00/C11/C01은 동일 global step에서 비교한다.
- R01/R02/R03의 mean·sample std와 paired difference를 보고한다. 미완료·실패 repeat는 누락 이유와 분모를 표시한다.

### 8.3 Secondary: validation-selected 배포 후보

Shared:

\[
V_{joint}(t)=\frac{1}{3}\sum_{s\in\{WV3,GF2,QB\}}\operatorname{MeanL1}_{val,s}(t).
\]

- 저장 공간으로 변환하거나 clipping하기 전 정규화된 예측의 mean L1이다.
- sample별 전체 원소 평균 후, 해당 sensor validation sample 전체를 같은 가중치로 평균한다. 마지막 작은 batch를 full batch와 동등하게 평균하지 않는다.
- 각 센서 평균을 다시 1/3로 평균한다. validation 장수가 많은 센서나 8밴드 센서에 자동으로 더 큰 가중치를 주지 않는다.
- `V_joint` 최소 checkpoint **전체**를 `BEST_JOINT_VAL`로 선택한다.
- 비교값은 계산된 full-precision 값, 완전 동률이면 낮은 step을 선택한다. NaN/부분 validation은 선택 후보가 아니다.
- Single은 해당 센서 validation mean L1 최소를 `BEST_SENSOR_VAL`로 선택한다.

BEST 후보는 최초 step0을 제외한 **전체 validation 완료 step** 중 선택한다. 후보 수·검색 범위가 shared/single에서 다를 수 있으므로 이 secondary 결과를 primary exposure-matched 결과와 혼동하지 않는다. `selected_through_global_step`, `candidate_count`, selector policy hash를 함께 저장한다.

50K block 끝에서 현재 BEST 후보가 endpoint와 다른 경우 그 후보도 RR/FR 평가한다. 동일 checkpoint가 이미 같은 evaluator로 평가되었으면 검증된 cache를 사용한다. Endpoint와 BEST가 같은 SHA이면 inference를 중복 실행하지 않고 selector 연결만 추가한다.

### 8.4 반드시 같은 shared 가중치로 보고

WV3 최고 step, GF2 최고 step, QB 최고 step을 따로 골라 하나의 모델 결과로 조합하지 않는다. Shared checkpoint SHA는 세 센서 결과행에서 같아야 한다. 센서별 최적 step은 로컬 진단용으로만 허용하며 배포 후보로 자동 조합하지 않는다.

### 8.5 Test 경계

- RR20/FR20은 고정 간격의 모니터링·보고에만 사용한다. loss, LR, seed, subset, queue, checkpoint selector를 바꾸는 입력으로 사용하지 않는다.
- HQNR/ERGAS 목표 달성으로 자동 early-stop하지 않는다.
- FR 지표 계산의 sensor MTF·crop·prediction clipping·LMS/PAN 사용 규칙을 기존 검증 evaluator와 일치시킨다. 알고리즘 입력 LPAN의 Gaussian recipe를 평가 MTF와 혼동하지 않는다.
- source benchmark의 train/test 지리적 독립성은 확인됐다고 주장하지 않는다. 이번 selector가 validation-only라는 사실과 데이터셋이 독립 test라는 주장은 별개다.[S05]
- 정식 GT가 없는 FR에 PSNR/GT 기반 지표를 만들어 넣지 않는다.

---

## 9. 학습 로그 — 추가 학습·데이터·용량 판단용

### 9.1 원본 기록

각 run에 다음 파일을 둔다.

```text
run_manifest.json
config.resolved.json
architecture_manifest.json
train_steps.jsonl
probes.jsonl
validation.jsonl
benchmark.jsonl
cost_profile.json
checkpoint_index.jsonl
exposure_counts/<sensor>.npz
resume_events.jsonl
errors.jsonl
```

로그는 UTF-8이며 schema version·Run_ID·Campaign_ID를 포함한다. JSON의 NaN/Infinity는 쓰지 말고 오류 상태와 null로 기록한다.

### 9.2 필수 필드

| 분류 | 기록할 값 |
|---|---|
| 식별 | campaign/case/repeat/seed/attempt/source/config/data/subset hash |
| 학습량 | global step, sensor별 optimizer visits, 누적 samples, distinct IDs seen, 유효 epoch |
| Loss | 현재 센서 batch L1, 센서별 최근 누적 mean, 고정 train-probe와 val-probe L1 |
| Validation | 센서별 전체 L1, ERGAS/SAM/PSNR, joint L1, split sample count |
| Gradient | 공유 trunk와 활성 stem/head의 norm, nonfinite 여부; 비활성 parameter 변경 검사 |
| Update | 실제 LR, optimizer moment step, 필요 시 norm(Δθ)/(norm(θ)+ε) 진단 |
| 자원 | optimizer/training, data-wait, LP/cache, validation, RR/FR, 저장/업로드 시간; GPU peak/RAM/disk |
| 결과 | RR/FR full precision, selector, 실제 model file SHA256, 평가 protocol hash |

유효 epoch는 센서별 `누적 committed samples / 실제 사용 train subset 개수`로 계산한다. 실제 JSON 필드명은 `effective_epochs`처럼 ASCII로 고정한다. macro-step의 단순 `/3` 추정치 대신 실제 exposure counter를 기준으로 한다.

### 9.3 판단 보고서

노출량 단계 k가 완성될 때 `reports/exposure_kXXXX.md`를 만든다.

- 동일 k에서 Shared-Base 대 세 Single의 차이와 seed별 분포.
- Wide 대 Base의 train/val 차이, 개선량 대비 활성 추론 시간·메모리 변화.
- HalfData 대 FullData의 동일 sample 노출 비교, 유효 epoch 차이 및 75K 보조 비교.
- 추가 노출에 따라 validation이 계속 개선되는지, train만 개선되는지, 특정 센서가 악화되는지.
- 완료하지 못한 run·미측정 비용·protocol mismatch는 명시한다.

원인 판정은 `UNDERTRAINING_CANDIDATE / CAPACITY_OR_OPTIMIZATION_CANDIDATE / OVERFIT_CANDIDATE / CROSS_SENSOR_INTERFERENCE_CANDIDATE`처럼 **후보 해석**으로 남긴다. 단일 지표만으로 원인을 단정하거나 자동 실험을 변경하지 않는다.

---

## 10. `배포용 모델` 시트 기록 규약

### 10.1 이미 존재하는 탭을 사용

```text
Workbook: pan-cvpr27
Spreadsheet ID: 1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0
Target tab: 배포용 모델
Observed sheetId: 1198707876
Template: WV3-main 형태, A:BL = 64 columns
Header row: 5
Data start: 6
Observed grid: 3000 rows × 64 columns
Frozen: 5 rows / 3 columns
```

2026-10-01 조회에서 A6:C3000은 비어 있었고, A1:A3에는 기존 계획 설명·메모, A2에는 관측행 수식, A5에는 `_records!A1:BL1`을 읽는 header 수식이 있었다.[S01] 이것은 당시 조회값이지 실행 시점까지 비어 있음을 보장하지 않는다.

**신규 탭을 중복 생성하거나 기존 탭을 clear/duplicate하지 않는다.** 실행 직전에 metadata와 헤더·본문 수식·현재 row key를 다시 읽는다. 쓰기는 확인한 target sheetId의 데이터행에만 한다. `paper`, `ablations`, `WV3-main`, 숨김 source 탭, `_records`, 기존 uploader는 수정하지 않는다. 헤더 A1:BL5도 기본 보존한다.

### 10.2 한 행의 의미

**한 관측행 = 한 Run_ID × 한 checkpoint SHA × 한 평가 센서 × 한 selector scope × 한 evaluator version.**

Shared EXACT checkpoint는 WV3/GF2/QB **3개 행**을 만든다. Single EXACT checkpoint는 해당 센서 1행이다. BEST 선택행을 추가해도 독립 seed 수가 늘어난 것은 아니다.

`Result_ID = SHA256(canonical JSON([campaign, run, checkpoint_sha, eval_sensor, selector_scope, evaluator_sha]))`

예를 들어 `BEST_JOINT_VAL_TO_B0003`은 block3까지의 후보 중 선택했음을 뜻한다. 이후 범위가 달라지면 기존 행을 덮지 않고 새 selector scope로 남긴다. 같은 Result_ID 재시도는 idempotent upsert한다.

### 10.3 기존 64열에 매핑

| 열 | 의미 / 신규 기록 규칙 |
|---|---|
| A | Experiment: 예 `C00 · Shared-Base · W104 D122 · F100 · R01 · WV3` |
| B | Server=`s2` |
| C | Selection@Step: `EXACT@150000`, `BEST_JOINT_VAL_TO_B0003@...` |
| D:G | 해당 센서 FR HQNR / D_s / D_lambda / JQM |
| H:O | 해당 센서 RR ERGAS / SCC / SAM / PSNR / SSIM / Q4·Q8 / RMSE / CC |
| P:Q | 해당 센서 활성 경로 Infer(ms), Mem(MB) |
| R | Params(M): **해당 checkpoint의 전체 저장 모델 파라미터**. active/shared/stem/head 분해는 notes·manifest |
| S | FLOPs(G): 해당 센서 활성 경로, 측정 규약이 확인된 경우만 |
| T:V | Run 누적 Train(h)/Eval(h)/Wall(h). shared 세 센서행에 반복될 수 있으므로 합산 시 Run_ID로 deduplicate |
| W:Y | HQNR(V64), Signed D_s, Positive fraction: 실제 해당 protocol로 계산한 경우만, 아니면 blank |
| Z | Dataset=`WV3` / `GF2` / `QB`: **평가 센서** |
| AA | Bucket=`Deployment` |
| AB | Approach=`Shared PLH U-Net` 또는 `Single PLH U-Net` |
| AC | Campaign_ID=`PANDEP_S2_SHARED_PLH_20261001_v4` |
| AD | Run_ID |
| AE | Case_ID=`C00/C11/C01/S01/S02/S03` |
| AF:AH | Repeat / Seed / Attempt |
| AI | Role=`SHARED` 또는 `SINGLE`; Teacher/Student 역할로 오해시키지 않음 |
| AJ | Model / inputs: `PLH; W104 D122; train=WV3+GF2+QB; fraction=1.0; noA` 등 |
| AK | Teacher / reference=`NONE` |
| AL | Parent_run: fresh는 blank. 같은 run resume는 parent를 새로 만들지 않음 |
| AM | Updates=현재 run의 committed global/local step |
| AN | Lifetime_updates=동일 run 누적 step; 이번 계획에서는 AM과 동일 |
| AO | Selected_step=해당 결과 checkpoint의 실제 step |
| AP | Checkpoint_SHA=**실제 모델 weight 파일 bytes SHA256** |
| AQ | Selector=EXACT 또는 BEST의 검색 범위를 포함한 ID |
| AR:AS | q_ref/tau_R=blank |
| AT:AV | alpha/beta/lambda_E=0 |
| AW:AX | U_peak_lr=1e-4 / A_peak_lr=0 |
| AY | Status: 평가 전 RUNNING/EVAL_PENDING, 완료 EVAL_COMPLETE 등 실제 상태 |
| AZ | Readback: PENDING → 실제 재조회 일치 후 VERIFIED |
| BA | Eval_scope: `RR20+FR20; protocol=<id>; sensor=<...>` |
| BB | Test_aware: 이 campaign의 선택·적응이 test를 쓰지 않으면 FALSE; 독립 test 인증 의미는 아님 |
| BC | JQM_variant: 검증한 variant 명칭; 미계산이면 blank |
| BD | Cost_scope: GPU/CPU, batch1, PAN256/MS64, FP32, time 범위, FLOPs 정의 |
| BE:BF | Source_sheet / Source_row: 로컬 원본이면 blank; 구형 행을 위조해 연결하지 않음 |
| BG | Source_URL: 실제 접근 가능한 결과 URL이 있을 때만. server path는 notes에 기록 |
| BH | Result_ID |
| BI | Original notes: train_sensors, subset ratio/hash, sensor별 visits/samples/epoch, joint val, block, active params, artifact 경로 |
| BJ | Review: 사용자·검토용 필드. uploader가 기존 수기 검토값을 덮지 않음 |
| BK | Sort_case: C00=0, C11=11, C01=1, S01=101, S02=102, S03=103; 행을 자동 재정렬하지 않음 |
| BL | Date=ISO8601 timezone 포함 기록 시각 |

미측정 값은 **0이 아니라 blank/null**로 기록한다. full precision을 저장하고 표시 형식만 기존 표와 맞춘다. input shape가 다르거나 GPU가 다른 비용을 동일 조건인 것처럼 순위화하지 않는다.

### 10.4 안전한 uploader

- 로컬 JSONL/manifest를 source of truth로 두고, Sheet는 표시용 복제본으로 취급한다.
- 임시 네트워크 실패가 학습을 중단시키지 않도록 영속 outbox와 backoff를 둔다.
- 한 target 탭에는 **하나의 writer만** 쓴다. 기존 배포 uploader가 있다면 확인하고 본 writer와 동시 기동하지 않는다. 기존 연구 uploader를 중단시키지는 않는다.
- 매 write 전 header 이름·순서, target sheetId, 목적 row의 기존 Result_ID와 수식 유무를 검사한다.
- unknown row나 수식 spill 영역을 덮지 않는다. 충돌은 `BLOCKED_SHEET_SCHEMA/WRITER_CONFLICT`로 남기고 로컬 기록을 지속한다.
- 데이터 쓰기 후 재조회로 주요 값·hash·Result_ID를 대조하고, 실제 성공한 후에만 VERIFIED를 기록한다.
- 3000행을 넘기면 필요한 row만 target 탭에서 확장한다. 다른 탭의 grid·수식·filter를 바꾸지 않는다.
- 매100step 로그를 모두 Sheet에 올리지 않는다. Sheet에는 block 및 selector 결과행만 올리고 세부 곡선은 로컬 파일로 보존한다.
- 헤더/기존 결과 보존 검증 및 duplicate retry 테스트가 uploader의 필수 gate다.

---

## 11. Checkpoint·재개·보존 정책

### 11.1 파일

```text
runs/<Run_ID>/
  checkpoints/
    step_000050000/
      model.safetensors
      identity.json
    step_000150000/
      model.safetensors
      identity.json
  resume/
    last_A.pt
    last_B.pt
    last.json
  best/
    <checkpoint_sha>/model.safetensors
    selection_history.jsonl
```

- checkpoint에는 shared의 **모든 stem/head와 단 하나의 trunk**를 함께 저장한다.
- `identity.json`: run/case/repeat/attempt, completed step, exposure counters, input order, width/depth, source/config/data/subset/protocol hash, actual weight SHA, tensor-state hash를 분리 기록한다.
- `state_hash`와 파일 SHA256을 같은 필드로 취급하지 않는다.
- full resume에는 optimizer·scheduler·Python/NumPy/Torch CPU/CUDA RNG·sampler·triplet cursor·증강 stream·각 센서 counts·학습/평가 누적 시간·controller lease가 아닌 실행 상태를 포함한다.
- checkpoint 저장은 임시 파일 작성→flush→검증→동일 filesystem atomic publication으로 한다. 일부만 저장된 checkpoint를 완료로 노출하지 않는다.

### 11.2 보존

- 모든 50K block EXACT weight, shared75K, 모든 BEST로 선정됐던 weight, matching index/metrics/provenance를 보존한다.
- 10K/25K validation 후보는 BEST가 될 경우 즉시 weight를 보존하고, 선택 근거를 남긴다. 사용자가 승인하지 않은 candidate 전체 삭제는 하지 않는다.
- 빈번한 resume용 `last_A/last_B` 두 slot은 **신규 campaign 소유 파일에 한해** 교대 교체한다. 앞 slot의 검증 전 마지막 정상 slot을 덮지 않는다.
- 50K block마다 full resume도 별도로 남긴다. 디스크 부족 시 무단 prune 대신 안전 정지한다.
- raw 데이터·기존 연구 checkpoint·legacy cache는 이동/삭제하지 않는다.

### 11.3 resume gate

중단 전/후 동일 fixture의 연속 N-step 결과와 split-resume N-step 결과를 비교한다. parameter, optimizer moments, 다음 batch ID/rotation, 센서 순서, exposure counts, LR가 일치해야 한다. hardware 수치 비결정성을 숨기지 않고 exact-resume test는 결정론적 검증 환경에서 수행한다. 실제 s2의 수치 정책과 재현 오차는 별도 receipt에 남긴다.

---

## 12. 안전 정지와 자원 보호

시간 상한은 없지만 아래는 정상적인 안전 정지 사유다.

| 조건 | 동작 |
|---|---|
| 사용자 stop 요청 | 다음 안전 optimizer 경계에서 fullstate 저장 후 STOPPED_BY_OPERATOR |
| 사용자 pause 요청 | fullstate 저장 후 PAUSED_SAFE; 명시적 resume 전 자동 재개 금지 |
| SIGTERM/SIGINT | 긴 연산 반환 후 가장 가까운 안전 경계에서 저장; 강제 종료는 마지막 정상 checkpoint에서 복구 |
| loss/output/gradient nonfinite | DIVERGED; 증거·현재 상태 보존, 자동 LR/seed 변경 금지 |
| GPU OOM | 메모리 상황 기록 후 안전 정지. batch/AMP를 몰래 바꾸지 않음 |
| source/config/data hash 변경 | BLOCKED_IDENTITY; 같은 run으로 재개 금지 |
| 새 학습 시작 전 다른 GPU 작업 발견 | WAIT_RESOURCE; 기존 process에 signal을 보내지 않음 |
| disk free가 기준 미만 | PAUSED_DISK; 기존/신규 결과 무단 삭제 금지 |
| Sheet 불가 | 로컬 outbox에 보존; 디스크·원본 로그가 정상인 동안 학습 가능 |

Disk 기준(계획값): `max(20 GiB, 다음 block 예상 신규 저장량×2 + checkpoint atomic 저장 reserve)`. LP cache 초기 생성 전에는 별도 전체 필요량과 여유공간을 계산한다. 매 block admission 및 주기적 telemetry에서 검사한다. 수치는 저장량을 계측해 manifest에 확정하되 안전 여유를 조용히 낮추지 않는다.

`control.json` 예시:

```json
{
  "schema": "PANDEP_CONTROL_v1",
  "campaign_id": "PANDEP_S2_SHARED_PLH_20261001_v4",
  "action": "RUN",
  "reason": null
}
```

`RUN / PAUSE / STOP_AFTER_BLOCK / STOP`을 구현한다. STOP 후 controller가 crash-restart를 이유로 학습을 다시 시작해서는 안 된다. 본 campaign의 종료 flag와 승인된 resume 동작을 구분한다.

---

## 13. 구현 검증 Gate — 정식 학습 전 필수

### G0. 경로·운영 독립성

별도 프로젝트/환경/run root인지, 기존 소스 diff가 0인지, 원본 H5가 read-only인지 확인한다. preflight import만으로 legacy controller·uploader·학습 process가 시작되지 않는지 검사한다.

### G1. 아키텍처

- Shared trunk 객체/parameter storage가 하나이며 optimizer 등록에 중복이 없다.
- WV3 stem 11ch, GF2/QB stem 7ch, head 8/4/4ch다.
- L/H 실제 tensor가 각각 U4(LPAN), P−L과 일치한다. L/H **초기 kernel=0**와 **입력 tensor=0**를 혼동하지 않는다.
- `PANGlobalAligner`, Teacher, 학습형 정합 module/parameter가 없다.
- forward에서 `grid_sample`, `affine_grid` 호출이 없다. runtime과 격리된 테스트에서 호출 감시를 한다.
- Attention/Swin/SE/modulation이 없다.
- zero residual fixture에서 출력은 MS_up과 같고 base가 정확히 한 번 더해진다.
- 선택되지 않은 센서 stem/head·optimizer state는 한 step 후 byte/tensor 단위로 변하지 않는다.

### G2. 원본 수치 parity

고정 source의 C01 reference와 새 single 모델, 새 shared의 해당 센서 경로에 동일 parameter를 매핑하여 FP32 출력을 비교한다. source symbol·parameter-name mapping을 보존한다. train patch64/RR256/FR512와 비정사각형(4의 배수) fixture를 포함한다.

CPU 결정론적 fixture에서 동등 연산은 bitwise 일치를 우선 목표로 한다. backend 차이가 있는 비교에는 사전 등록된 tolerance를 사용한다. 초기 제안 gate는 정규화 출력 `max_abs<=1e-5, mean_abs<=1e-6`이며, 실패하면 먼저 연산·padding·interpolation·weight mapping을 점검한다. 결과를 통과시키려고 실행 후 tolerance를 확대하지 않는다. 이 tolerance는 과거에 이미 승인된 release 기준이라는 뜻이 아니다.

기존 학습 weight가 없어도 동결 synthetic state를 동일하게 넣은 구조 parity는 가능하다. 다만 이를 실제 학습 checkpoint parity 완료로 표현하지 않는다.

### G3. 데이터·sampler·loss

12개 source hash, QB msfix, native LP cache를 검증한다. 다른 band 수·DNmax·split·비정상 sample은 거부한다. full/half ID, 3-step balanced sampling, 미완 triplet resume, mean-L1의 band-count normalization, effective batch48을 검사한다.

### G4. Checkpoint·resume

중간 종료·저장 중 interruption·중복 controller 기동·잘못된 source resume를 테스트한다. sample/augmentation cursor의 prefetched 항목과 committed 항목을 구분한다.

### G5. Metric·Sheet

고정 reference prediction에 대해 기존 sensor별 RR/FR metric과 parity를 확인한다. 새 schema의 임시 로컬 row로 mapping을 검증한 후 target 탭의 허용행만 쓴다. duplicate retry, readback 실패, unknown existing row, header formula 보존을 테스트한다. 가짜 성능값을 실제 결과행에 쓰지 않는다.

G0–G5를 통과한 receipt를 저장한 뒤 정식 run을 시작한다. source parity가 필요한 검증과 사용자의 production release 승인은 별개다.

---

## 14. 비용·배포 관점의 검사

### 14.1 센서별 비용

서버 profile의 고정 조건:

```text
batch=1
PAN=[1,1,256,256]
MS=[1,C,64,64]
LPAN=[1,1,64,64]
FP32 / eval / no_grad 또는 inference_mode
warmup=20, measured repeats=100
CUDA면 구간 전후 synchronize
```

모델 호출 latency는 **정규화된 입력과 LPAN cache가 준비된 상태**에서 stem+trunk+head+MS/L 보간+residual을 측정한다. raw file read, LP 생성, host-device transfer는 별도 end-to-end 항목으로 기록한다. CPU/GPU의 median·p95·peak memory와 하드웨어 정보를 보존한다.

전체 저장 parameter, 센서별 active parameter, shared/각 stem/head의 parameter를 분리한다. FLOPs는 지원되지 않는 연산·계수 규약을 명시하고 `2×MACs`인 경우 그렇게 표시한다. 계산하지 못한 항목을 0으로 쓰지 않는다.

### 14.2 export

최초 matched-exposure 단계가 끝나면 C00 R01의 EXACT150K 및 BEST_JOINT_VAL을 **배포 PoC candidate**로 export한다. 성능 최종 승인 모델로 표시하지 않는다.

```text
exports/<release_id>/
  model.safetensors                 # shared trunk + 3 sensor stem/head
  model_config.json
  sensor_profiles.json
  preprocessing.json
  release_manifest.json            # status=candidate
  source_snapshot/
  reference_inputs/
  reference_outputs/
  SHA256SUMS.txt
  LICENSES/
  README.md
```

Teacher·optimizer·학습 데이터 전체는 inference bundle에 넣지 않는다. 모델 버전과 앱 버전은 분리한다. source 라이선스·가중치 배포 권한을 보존·확인하고 공개/상업 배포를 자동 승인하지 않는다.

노트북용 tile inference는 후속 검증이다. 이 실험에서 Aligner가 없더라도 LPAN filter·decimation phase·bicubic·U-Net 경계 처리는 일치해야 한다. halo를 실제 receptive field 및 수치 test로 결정하고, tile-origin/stride의 global phase를 보존한다. full/tile parity를 확인하기 전 대형 raster 배포 완료로 표시하지 않는다.

---

## 15. 구현해야 할 CLI 계약

**아래 명령은 신규 구현의 인터페이스 요구사항이다. 현재 이미 존재하는 실행 명령이라는 뜻이 아니다.** 실제 구현 후 `--help`와 smoke test를 검증하고 README에 확정한다. `<...>`는 inventory에서 확인한 경로로 채운다.

```bash
# 독립 프로젝트 환경에서만 실행
python -m pan_shared.cli preflight \
  --server s2 --campaign PANDEP_S2_SHARED_PLH_20261001_v4 \
  --data-catalog <FROZEN_SENSOR_SOURCES_JSON> \
  --work-root <NEW_CAMPAIGN_WORK_ROOT>

python -m pan_shared.cli verify --all-gates \
  --work-root <NEW_CAMPAIGN_WORK_ROOT>

python -m pan_shared.cli register \
  --plan <CASE_REGISTRY_JSON> --work-root <NEW_CAMPAIGN_WORK_ROOT>

python -m pan_shared.cli run \
  --server s2 --until-operator-stop \
  --work-root <NEW_CAMPAIGN_WORK_ROOT>

python -m pan_shared.cli status --work-root <NEW_CAMPAIGN_WORK_ROOT>
python -m pan_shared.cli pause  --work-root <NEW_CAMPAIGN_WORK_ROOT>
python -m pan_shared.cli resume --work-root <NEW_CAMPAIGN_WORK_ROOT>
python -m pan_shared.cli stop --after-current-update --work-root <NEW_CAMPAIGN_WORK_ROOT>

python -m pan_shared.cli sync-sheet \
  --sheet-id 1198707876 --work-root <NEW_CAMPAIGN_WORK_ROOT>
```

`register`는 동일 registry hash이면 idempotent, 다른 정의로 같은 Run_ID를 덮으려 하면 실패해야 한다. 실행은 `--until-operator-stop`을 명시해야 무기한 모드로 허용한다. 기존 campaign의 lease·기본 deadline·서버 lane mapping을 가져오지 않는다.

---

## 16. 실행 담당자 제출물

### 시작 보고

실제 s2 hostname/GPU UUID·process·disk inventory, 신규 절대경로·환경, 원본 소스 불변 검증, source/config/data/subset/architecture hash, G0–G5 결과, 실제18개 Run_ID와 queue 순서, target sheetId·헤더 검증, 실행한 명령·PID를 제출한다. **계획만 생성한 것과 실제 기동을 분리해 적는다.**

### 각 block 보고

각 run의 누적 global/local step과 센서별 exposure, 최신 train/val 곡선, EXACT 및 BEST 결과, weight/fullstate 경로·hash, Sheet readback, 비용, 오류·대기 상태를 제출한다.

### 사용자 중단 시

마지막 정상 fullstate, 멈춘 logical run·step, 다음 센서/batch cursor, queue의 다음 case, 미완료 평가·outbox 항목, 재개 명령, 보호해야 할 export/checkpoint 목록을 남긴다.

---

## 17. 최소 registry 예시

아래는 구현자가 독립 registry를 만들 때 사용할 계획값이다. source/data/환경 및 전체18개 run 정의가 해시로 동결되기 전에는 training을 시작하지 않는다.

```json
{
  "schema": "PANDEP_SHARED_PLAN_v4",
  "campaign_id": "PANDEP_S2_SHARED_PLH_20261001_v4",
  "server_allowlist": ["s2"],
  "train_sensors": ["WV3", "GF2", "QB"],
  "private_data_enabled": false,
  "input_layout": ["PAN", "LPAN_UP", "PAN_MINUS_LPAN_UP", "MS_UP"],
  "aligner_enabled": false,
  "warp_enabled": false,
  "teacher_enabled": false,
  "attention_enabled": false,
  "mode_modulation": false,
  "repeat_seeds": {"R01": 271001, "R02": 271002, "R03": 271003},
  "subset_seed": 20261001,
  "effective_batch_size": 48,
  "precision": "fp32",
  "loss": "mean_l1_normalized_unclipped",
  "sensor_sampling": "balanced_shuffled_triplets",
  "wallclock_limit_seconds": null,
  "max_global_steps": null,
  "until_operator_stop": true,
  "early_stopping": false,
  "yield_block_updates": 50000,
  "cases": [
    {"case_id": "C00", "mode": "SHARED", "sensors": ["WV3","GF2","QB"], "width": 104, "depth": [1,2,2], "train_fraction": 1.0, "exposure_stage_updates": 150000},
    {"case_id": "C11", "mode": "SHARED", "sensors": ["WV3","GF2","QB"], "width": 128, "depth": [1,2,2], "train_fraction": 1.0, "exposure_stage_updates": 150000},
    {"case_id": "C01", "mode": "SHARED", "sensors": ["WV3","GF2","QB"], "width": 104, "depth": [1,2,2], "train_fraction": 0.5, "exposure_stage_updates": 150000},
    {"case_id": "S01", "mode": "SINGLE", "sensors": ["WV3"], "width": 104, "depth": [1,2,2], "train_fraction": 1.0, "exposure_stage_updates": 50000},
    {"case_id": "S02", "mode": "SINGLE", "sensors": ["GF2"], "width": 104, "depth": [1,2,2], "train_fraction": 1.0, "exposure_stage_updates": 50000},
    {"case_id": "S03", "mode": "SINGLE", "sensors": ["QB"], "width": 104, "depth": [1,2,2], "train_fraction": 1.0, "exposure_stage_updates": 50000}
  ],
  "primary_selector": "EXACT_MATCHED_SENSOR_EXPOSURE",
  "secondary_selector_shared": "MIN_MACRO_SENSOR_VALIDATION_L1_THEN_EARLIER_STEP",
  "secondary_selector_single": "MIN_SENSOR_VALIDATION_L1_THEN_EARLIER_STEP",
  "spreadsheet_id": "1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0",
  "sheet_title": "배포용 모델",
  "observed_sheet_id": 1198707876,
  "sheet_header_row": 5,
  "sheet_data_start_row": 6,
  "legacy_source_commit": "ea71b68be637a1d1f4d61bf89cb4e023cbf00385"
}
```

---

## 18. 출처·확인 범위

본 계획의 새로운 실험값은 사용자 요청과 위 설계로 정한 것이다. 다음 자료는 **기존 모델·데이터·시트의 수치 및 운영 경계를 확인한 근거**다. 기존 파일에서 본 계획 전체가 이미 구현됐다고 해석하지 않는다.

### [S01] Live Google Sheet, 2026-10-01 조회

- Workbook: `pan-cvpr27`
- URL: <https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit#gid=1198707876>
- 조회: metadata, `'배포용 모델'!A1:BL35`, A1:C6·Z6:BL10의 userEnteredValue/note, A6:C3000.
- 확인: 탭 존재, sheetId 1198707876, 64열 헤더, C00/C11/C01의 기존 배포 의미, Aligner/KD OFF·PLH ON, 50K block·로그 주기 메모.
- 범위 한계: 당시 A6:C3000의 값은 없었다. 실제 서버 process/모델/가중치는 이 조회로 확인하지 않았다.

### [S02] U-Net 구조

- `model/pancrafter_paper.py`
- <https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/model/pancrafter_paper.py>
- 계승 대상: ChannelLayerNorm, ResBlock, 고정 width의 3-scale encoder/decoder, attention/modulation OFF, 마지막 residual projection.
- DownConv/UpConv의 정확한 수치 구현은 같은 snapshot의 `model/pancrafter.py`에서 분리해 검증한다. 본 문서는 미확인 구현을 임의의 일반 U-Net으로 대체하도록 허용하지 않는다.

### [S03] C01 component 및 frontend

- `ablr2/model.py`
- <https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/ablr2/model.py>
- Component catalog:
  <https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/research_log/PANDA_ABLR2X_S123_Continuous_C17_GF2_Bundle_2026-09-23/registries/ABLR2X_ComponentCatalog_18.csv>
- 확인: ABLR2 C01 = PLH, mask_L/H=1, IDENTITY, Teacher NONE. Identity 경로는 PAN sampling을 정확히 bypass한다.

### [S04] 기존 lane 및 sensor specification

- `ablr2/plan.py`
- <https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/ablr2/plan.py>
- 확인: 기존 s2=QB lane, source/run/기존 uploader 계약. 이것을 해제하지 않고 신규 server/sensor registry를 구현한다.

### [S05] Source identity·band order·QB msfix

- `ablr2/sensor_sources.json`
- <https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/ablr2/sensor_sources.json>
- 확인: 12개 source path/hash, band convention, QB msfix, finite ringing 보존, 지리적 test 독립성 미인증.

### [S06] Dataset validation

- `ablr2/data.py`
- <https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/ablr2/data.py>
- 확인: sensor별 shape·range·LP·manifest 검사. 검증 규약을 재사용하되 기존 lane-bound loader 자체를 수정하지 않는다.

### [S07] LP recipe·증강·배포 인계

- `fh12/data.py`, `tools/repair_lpan.py`
- <https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/fh12/data.py>
- <https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/tools/repair_lpan.py>
- 첨부 `PAN_Deployment_Session_Handoff_2026-10-01.md` §2/§7/§10/§12.
- 과거 인계서의 A+U 배포 구성은 **이번 no-Aligner 실험에 적용하지 않는다.** 이번 사용자의 최신 결정과 live 배포 탭의 PLH/noA 정의를 우선한다.

---

## 최종 실행 지시

**별도 프로젝트에서 PLH/no-Aligner shared U-Net을 구현·검증하고, s2에서 DEP:C00/C11/C01 및 S01/S02/S03을 세 seed로 실행한다. 처음에는 shared150K 대 single50K의 노출량 비교를 만들고, 이후 같은 run을 50K block 단위로 계속 이어간다. 전체 시간 상한·성능 기반 early-stop은 없다. 결과는 기존 `배포용 모델` 탭의 64열 규약에 맞춰 checkpoint×센서별로 기록한다. 원본 로그·hash·resume 상태를 보존하고, 사용자 중단 또는 안전 정지 시 복구 가능한 상태에서 멈춘다.**
