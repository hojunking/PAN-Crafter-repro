# PAN-Crafter 재현 case 설계 — s3·s4·s5 독립 무기한 cycle

작성일: 2026-09-22  
Campaign: `PANCRAFTER_REPRO_S345_UNLIMITED_20260922_v1`  
단일 recipe: `PC_PAPER_EQ11_C128_D224_LN_MARS_v1`  
열람 기준 저장소: `hojunking/PAN-Crafter-repro`, commit `762e314444aa1c6dbef0b06c7203bd0fc5505c0a`

## 1. 이번 실험의 범위

s3·s4·s5에서 진행하던 **PANDA/GF2 탐색 campaign을 종료하는 방향으로 전환**한다. GF2 데이터셋을 제외한다는 의미가 아니다. 이제 GF2도 WV3·QB와 함께 **PAN-Crafter 자체를 재현하는 학습 대상**이다. s1·s2의 현재 작업은 변경하지 않는다.

세 서버에 서로 다른 모델이나 LPAN 변형을 배정하지 않는다. **모든 서버가 동일한 모델·학습 recipe를 사용**하며, 각자 네 데이터셋을 포함하는 cycle을 독립적으로 반복한다. 전체 wall-clock 기한, 최대 cycle 수, 목표 HQNR 도달에 따른 종료, 후속 seed 취소 규칙은 없다. 한 학습 run은 논문처럼 정확히 **50,000 optimizer update**에서 종료하고, 다음 학습 run은 새 가중치·optimizer로 시작한다. 동일 모델을 100K·150K로 계속 이어 학습하는 실험이 아니다.

**WV2는 예외를 숨기지 않고 논문 설정을 그대로 따른다.** 논문 §4.1 및 Table9에서 WV2는 WV3 학습 모델의 zero-shot 평가 대상이다. 따라서 한 cycle은 **WV3·QB·GF2 fresh50K 학습 3개 + 해당 cycle의 WV3 모델로 WV2 평가 1개**다. WV2를 따로 학습하거나 fine-tuning하지 않는다. 네 데이터셋 모두에 결과가 남지만, 네 개의 별도 학습이 수행되는 것은 아니다. [S1]

이 문서는 실행 담당자에게 넘길 **설계와 case 정의**다. 실제 서버 작업 종료·배포·GPU 학습은 수행하지 않았다. 동봉 Python은 case 생성기이며 training runner가 아니다.

## 2. WV3-s1에서 찾은 기준 실험

2026-09-22 live Sheet의 `WV3-s1!B5:W14`를 읽었다. 아래 HQNR은 **N열**, 즉 paper FR 세트 결과이며 O열 `HQNR(V64)`가 아니다. 수치는 과거 run의 기록이다. [S2]

| 행 | Run | Params(M) | ERGAS | HQNR | 판단 |
|---:|---|---:|---:|---:|---|
| 5 | Paper reported | 7.1700 | 2.0400 | 0.9580 | 논문 보고값; 실행 결과 아님 |
| 6 | paper_ln (50K) | 7.1707 | 2.1205 | 0.9406 | LN·D224·MLP4·입력9ch. 구조 참고 기준 |
| 7 | s1_A0 (50K) | 7.1707 | 2.0436 | 0.9509 | 같은 골격, crop=False. native patch 운용 참고 기준 |
| 8 | paper_wv3 (50K) | 7.1707 | 2.1494 | 0.9467 | GroupNorm이므로 이번 기준에서 제외 |
| 9 | wv3_fixed_valsel (50K) | 9.9688 | 2.1804 | 0.9558 | 배포 코드의 4-scale 구조. 제외 |
| 11 | paper_ln_mlp1 (50K) | 7.1708 | 2.1229 | 0.9445 | D225·MLP1 변형. 파라미터가 비슷해도 제외 |
| 13 | s1_A1 (50K) | 7.1730 | 2.0351 | 0.9539 | LPAN·PAN−LPAN 추가 입력11ch 변형. 제외 |

찾아야 했던 계열은 **`config/paper_ln.yaml` / `config/s1_A0.yaml`의 `model.pancrafter_paper.PANCrafterPaper`**다. 이름에 `fixed`가 붙었다고 논문 구조인 것은 아니며, 7.17M에 가깝다는 것만으로도 동일성을 판정할 수 없다. [S3, S4]

### 2.1 현재 재구성본에 남은 차이

현재 commit의 `PANCrafterPaper`는 첫 입력을 PAN+MS로 제한하지만, attention을 기존 `CMAAA`에 위임한다. 그 내부는 여전히 다음과 같다. [S5, S6]

- PAN-mode query의 modality prior: `lpan.repeat(Cms)`.
- PAN key: `Conv([x, lpan])`.
- PAN value: `Conv([x, lpan, pan, pan-lpan])`.

논문 Eq.(11)/(12)의 **raw PAN을 band 수만큼 반복한 표현에 기반한 PAN K/V와 PAN query**와 다르다. `in_mode: paper`만 켜서 이를 해결했다고 해석하면 안 된다. 이전 `results_log/2026-08-25_divergences-and-tuning-review.md`도 이 차이가 미반영이라고 기록하고 있다. [S8]

이번 단일 recipe는 기존 골격을 출발점으로 삼되 이 차이를 **구현 전 필수 보완**으로 지정한다. `s1_A0` 과거 수치를 새 recipe의 실행 결과로 재사용하지 않는다. 수식에 맞추려다 파라미터가 조금 늘어나는 경우 그 수를 그대로 보고한다. 과거 audit의 Eq.(11) 수정 예상치는 약 **7.2122M**이었으나, 이는 이번 버전의 실측값이 아니다. 새 코드의 total/trainable parameter 수는 preflight에서 측정하며 **7.1700M에 맞추려고 depth나 FFN을 변경하지 않는다**. [S8]

## 3. 고정 모델 설정

| 항목 | 이번 recipe | 근거의 성격 |
|---|---|---|
| 구조 | 3-scale U-Net, Down2/Up2 | 논문 Figure3 및 기존 재구성본 |
| feature width | 전 scale C=128 | 논문 §4.2 |
| ResBlock depth | encoder [2,2], bottleneck4, decoder [2,2], 총12 | 기존 재구성본 선택; 논문에 배분 미기재 |
| 정규화 | LayerNorm | 논문 Eq.(5); 채널축 구현은 기존 선택 |
| CM3A | H/2 encoder, H/4 bottleneck, H/2 decoder에 3개 | Figure3 및 기존 재구성본 |
| attention | heads8, local k=3, FFN expansion4 | k는 논문; heads/expansion은 기존 선택 |
| modulation | mode별 γ·β 및 α 직접 학습 | 논문 Eq.(6)/(8) |
| dropout | 0 | 기존 재구성본 선택 |
| 최초 입력 | native PAN + upsampled MS만 | 논문 §3.2 |
| band 수 | WV3/WV2 8, QB/GF2 4 | paper/data contract |
| 최초 conv 입력 | WV3/WV2 9ch, QB/GF2 5ch | 위 입력 정의에서 결정 |
| auxiliary task | PAN reconstruction 유지 | 논문 MARs |
| 제외 | PAN Aligner, KD, q/e, PANMIX, synthetic shift, LPAN/HPAN feature 입력, Swin, SE | PANDA/변형과 분리 |

세부 정의는 `recipe.json` 하나로 고정한다. 서버별 YAML에 다른 defaults가 숨어들지 않도록 resolved config와 source hash를 저장한다. 현재 존재하지 않는 strict model symbol이나 실제 배포 commit은 추정하지 않고 null로 둔다.

### 3.1 PAN K/V와 query의 최소 수정

입력 PAN을 P, PAN 해상도로 올린 MS를 M, 현재 scale의 feature를 xℓ, 해당 scale로의 resize를 Rℓ라 한다. P^rep는 P를 Cms번 반복한 것이다.

```text
Q_MS  = Conv([Rℓ(M), xℓ])
Q_PAN = Conv([Rℓ(P^rep), xℓ])
[K_MS  | V_MS ] = Conv([Rℓ(M), xℓ])
[K_PAN | V_PAN] = Conv([Rℓ(P^rep), xℓ])
```

PAN K/V는 동일한 raw-PAN 조건 입력의 결합 projection이다. LPAN을 key에 넣고 PAN−LPAN을 value에 추가하는 경로는 사용하지 않는다. 모드별 query 선택과 MS/PAN 양쪽 local attention, mode별 α 결합은 유지한다. local gather는 정확한 3×3 이웃 추출로 고정하며, 학습 가능한 gather bias가 남아서는 안 된다. [S1, S6]

### 3.2 제거하면 안 되는 LPAN 사용

**LPAN feature 입력 제거와 PAN-mode residual base 제거는 다르다.** 논문 Eq.(3)의 PAN residual base는 유지한다.

```text
HRMS_prediction = M + F(P, M; MS mode)
PAN_prediction  = repeat(U4(D4(P)), Cms) + F(P, M; PAN mode)
```

여기서 U4(D4(P))는 **현재 P 자체**에서 얻은 저해상도 PAN의 upsampling이며, PAN-mode 출력에만 더한다. 다른 장면의 외부 `*_pan.h5`를 읽어 feature나 residual base에 사용하는 것은 금지한다. MS-only inference에는 이 PAN residual base가 필요하지 않다. [S1]

저해상도 PAN 생성 필터·위상과 MS interpolation은 논문에서 완전히 특정되지 않는다. 이번 설계는 기존 로컬 구현에서 사용하던 Gaussian σ1.98/kernel41/replicate/[2::4]와 bicubic upsampling을 **명시적 구현 선택**으로 고정한다. 물리적 센서 MTF나 저자가 사용한 정확한 필터를 확인했다는 뜻이 아니다. `recipe.json`에 dtype, interpolation flag와 residual-only 사용 범위를 적었다. [S10]

## 4. 고정 학습 설정

| 항목 | 값 |
|---|---|
| 한 training run | fresh initialization, 정확히50,000 optimizer update |
| optimizer | AdamW, LR1e-4, weight decay0.01 |
| schedule | warmup100 update + cosine decay |
| batch | 원본 PAN–MS pair48, MARs 복제 후96 |
| loss | mean L1(HRMS,GT) + 1.0×mean L1(PAN,repeated PAN) |
| 입력 patch | PAN64×64, MS16×16 |
| precision | FP32, AMP off, TF32 off; 이번 runtime 고정 선택 |
| augmentation | paired random H/V flip, 0/90/180/270° rotation, resize 없는 고정 크기 crop |
| normalization | 기존 코드의 2×DN/max_pixel−1; DN1023 GF2, DN2047 WV3/QB/WV2 |
| inference | MS mode only |

50K·warmup100·AdamW·LR·weight decay·batch·λ·C·k는 논문 명시값이다. AdamW β=(0.9,0.999), eps1e-8, worker4, drop_last=True, FP32 및 TF32 정책 등은 **명시되지 않은 구현 선택**으로 기록한다. 논문 GPU는 RTX3090이며, s3–s5의 실제 GPU/runtime은 현장 readback으로 기록한다. RTX5090 사용 결과를 RTX3090 측정처럼 보고하지 않는다. [S1, S3, S11]

### 4.1 Crop과 flip을 기존 feeder 그대로 사용하지 않는 이유

현재 `PanFeeder`는 `hflip=True`와 `vflip=True`이면 매 sample을 항상 뒤집는다. `crop=True`는 작은 영역을 자른 뒤 다시64/16으로 늘리므로 scale jitter다. 생성자 내부의 `random.seed(2025)`도 run별 seed 관리와 충돌할 수 있다. [S7]

이번에는 H/V 각각 probability0.5, rotation uniform4를 **run-owned RNG**로 결정한다. crop은 서로 대응하는 위치에서 최종 크기64/16을 자르는 연산만 허용한다. 이미64/16인 배포 patch에서는 crop이 no-op이다. 더 큰 원본 patch가 없으면 논문의 비자명한 random-crop 분포를 완전히 복원했다고 주장할 수 없다. 이를 숨기고 crop-resize를 사용하거나 데이터를 확대해 새로운 학습분포를 만들지 않는다. 더 큰 원본이 나중에 확보되면 새 data revision으로 분리한다.

MARs duplication은 augmentation **후** 수행하여 두 모드가 같은 pair를 보게 한다. 두48-sample mode의 L1을 각각 평균한 뒤 더한다. 합친96 전체에 대한 단일 평균만 사용하면 λ=1 식과 전체 loss scale이 달라질 수 있으므로 unit test로 확인한다.

## 5. 데이터셋과 평가 binding

| case | train | validation | 최종 평가 | optimizer update |
|---|---|---|---|---:|
| WV3_TRAIN | WV3 train | WV3 native validation | WV3 RR/FR | 50,000 |
| QB_TRAIN | QB original train | QB original validation | QB RR/FR | 50,000 |
| GF2_TRAIN | GF2 train | GF2 native validation | GF2 RR/FR | 50,000 |
| WV2_ZERO_SHOT | 없음 | 없음 | WV2 RR/FR | 0 |

WV2의 source는 **같은 서버·같은 cycle의 WV3 run**이다. WV2 평가 결과로 checkpoint를 선택하지 않는다. WV3 학습 실패 시 이전 cycle이나 다른 서버의 좋은 WV3 checkpoint로 대체하지 않고 `BLOCKED_DEPENDENCY`로 남긴다. 다른 데이터셋은 계속 진행할 수 있다.

실제 파일 경로는 `bindings.example.json`의 local manifest에 바인딩한다. 역사적 s1의 `/home/knuvi/...` 절대 경로를 s3–s5에서 존재한다고 가정하지 않는다. train/validation/test split, scene ID, shape, band order, DN scale, 원본·파생 여부, SHA256을 고정한다. 세 서버에서 같은 이름의 데이터가 실제로 같은지 tensor/file manifest로 확인한다.

**QB의 `*_msfix.h5`를 자동 상속하지 않는다.** 저장소에는 QB 원 데이터 결함과 보정 파일 기록이 있지만, 보정된 데이터는 논문 원본 설정과 구분해야 한다. 이번은 원본 PanCollection train/validation을 명시적으로 바인딩한다. 원본이 없으면 QB case를 block하며 `msfix`로 조용히 대체하지 않는다. 수정 데이터 재현을 추가할 때는 별도 사용자 방향에 따라 data revision을 만든다. [S9]

FR은 기존 프로젝트가 구분한 **paper `.mat` scene 세트**를 기준으로 한다. H5와 `.mat`의 scene ID/tensor가 동일하다는 검증 없이 서로 바꾸지 않는다. 특히 과거 WV3/QB H5와 `.mat` FR 차이를 반복하지 않는다. WV2도 available H5가 paper scene set과 동일한지 확인하고, 미확인 입력의 결과는 `PAPERSET_IDENTITY_UNVERIFIED`로 분리한다. [S9]

## 6. 서버별 독립 cycle

| 서버 | 첫 번째 | 두 번째 | 세 번째 | 네 번째 | 이후 |
|---|---|---|---|---|---|
| s3 | WV3 fresh50K | 같은 WV3→WV2 zero-shot | QB fresh50K | GF2 fresh50K | 다음 cycle |
| s4 | QB fresh50K | GF2 fresh50K | WV3 fresh50K | 같은 WV3→WV2 zero-shot | 다음 cycle |
| s5 | GF2 fresh50K | WV3 fresh50K | 같은 WV3→WV2 zero-shot | QB fresh50K | 다음 cycle |

각 서버는 자신의 순서만 따른다. s4가 느리거나 중단돼도 s3·s5는 진행한다. 공유 Teacher·MSTAR·공통 t0·40h deadline·cross-server barrier는 없다. 로컬의 중복 GPU 사용을 방지하는 process lock은 유지한다.

한 cycle에서 서버당 학습3개×50K=150K update, 서버3개 합계는450K update다. 네 번째 zero-shot case의 실행 비용은 평가 비용이지 추가 학습 비용이 아니다. 서버별 cycle 번호가 같아도 완료 시각이 같을 필요는 없다.

### 6.1 Seed 및 반복 정의

첫 cycle index0은 세 서버 모두 **paper seed2025**로 네 데이터셋 절차를 수행한다. 이는 동일 seed의 서버 간 실행 반복이다. 독립 seed3개의 결과로 세지 않는다.

그다음부터 결과와 무관하게 다음 식으로 고정한다.

```text
cycle >= 1:
seed = 1,000,000 + 3*(cycle-1) + (server_number-3)
```

| cycle index | s3 | s4 | s5 |
|---:|---:|---:|---:|
| 0 | 2025 | 2025 | 2025 |
| 1 | 1000000 | 1000001 | 1000002 |
| 2 | 1000003 | 1000004 | 1000005 |

동일 서버/cycle에서는 WV3·QB·GF2에 같은 seed 숫자를 사용하지만, 각 run의 가중치는 독립적으로 새로 생성한다. WV2 case는 source WV3의 seed를 기록한다. 이후 seed 반복은 **논문 seed2025 한 점의 재현과 구분되는 반복 검증**이다. 나쁜 결과의 seed만 바꾸거나, 좋은 seed가 나오면 다음 seed를 취소하지 않는다. 32-bit seed 범위를 소진하면 wrap하지 않고 설정 오류로 중단한다.

## 7. Run 이름과 Sheet 기록

새 전용 탭 `PC-Repro-s3`, `PC-Repro-s4`, `PC-Repro-s5`에 append한다. 기존 WV3-s1 reference, GF2-P40/B20, PANDA 결과와 헤더는 변경하지 않는다. `Run` cell에는 모델 설정만 쓴다.

```text
PAN-Crafter | C128 D224 LN | CM3A3 k3 | MARs lambda1 | PAN+MS in9 | 50K
```

4-band는 `in5`다. heads8, FFN4, optimizer 등의 전체 세팅은 별도 recipe/config 열 및 manifest에서 확인하도록 한다. seed·server·cycle·dataset·source는 별도 열이며, 실험 식별용 `run_id`는 다음처럼 고유하다.

```text
PCREPRO_WV3_s3_C000000_S2025_F50K_v1
PCREPRO_WV3toWV2_s3_C000000_S2025_EVAL_v1
```

필수 식별 열은 `run_id, recipe_id, dataset, train_dataset, stage, server, cycle, seed, start/end time, actual_updates, status, source_commit, data_sha, checkpoint_sha, selection`이다. WV2의 actual_updates는0이며, source_run_id·source_checkpoint_sha를 추가한다. Run cell이 같아도 unique run_id가 다르면 다른 반복이다. 동일 run의 primary와 secondary 선택점을 독립 실험으로 중복 집계하지 않는다.

## 8. Checkpoint 선택과 결과 보고

논문은 best-checkpoint 선택 규칙을 충분히 특정하지 않는다. 따라서 **EXACT_50000를 주 결과**, 동일 학습 중 native validation ERGAS가 가장 낮았던 **RR_VAL_ERGAS_MIN을 보조 결과**로 미리 고정한다. 이 선택은 이번 재현의 보고 정책이지 논문 명시 사실이 아니다. legacy `s1_A0/paper_ln`과의 해석에는 val-selected 보조 결과를 사용한다. [S1, S3, S4]

validation은1000 update마다와 마지막50K에서 수행하며, 동률이면 먼저 나온 checkpoint를 유지한다. test RR/FR은 학습 종료 후 두 선택점에 대해 평가한다. 두 선택점이 같은 SHA이면 한 번 평가하고 별칭만 기록한다. WV2도 이 두 source 선택점을 평가하되 WV2 점수로 재선택하지 않는다.

RR: ERGAS, SAM, PSNR, SSIM, SCC, Q4/Q8. FR: Dλ, Ds, HQNR. RMSE·CC나 기존 추가 지표는 보조 열에 둘 수 있다. `HQNR(V64)`·aligned/valid crop·proxy QNR은 주 HQNR와 섞지 않는다. JQM을 추가하면 기존 SRF-substitute라는 정의를 명시하고 논문 공식 지표로 표기하지 않는다.

모든 지표는 동일 checkpoint·동일 평가 manifest에서 얻는다. 높은 H와 다른 checkpoint의 낮은 E를 결합하지 않는다. HQNR은 각 scene에서 계산한 값을 평균하며, 평균 Dλ와 평균 Ds의 곱으로 대체하지 않는다. 원값은 full precision, 표시는 소수4자리다.

### 8.1 논문 참고값 — 자동 종료 조건 아님

| 데이터셋 | HQNR | ERGAS | PSNR | SCC | 출처 |
|---|---:|---:|---:|---:|---|
| WV3 | .958 | 2.040 | 37.956 | .988 | Table6 |
| QB | .920 | 3.570 | 38.195 | .984 | Table8 |
| GF2 | .964 | .552 | 45.076 | .994 | Table7/12 |
| WV2 zero-shot | .942 | 4.169 | 29.276 | .924 | Table9 |

GF2 본문 Table2는 ERGAS **.522**, Table7/12는 **.552**다. 이 불일치는 해결되지 않았으며 어느 한쪽을 임의로 오타로 확정하지 않는다. 두 비교를 구분해서 적는다. 목표를 넘거나 못 넘는 것은 다음 cycle의 실행 조건이 아니다. [S1]

## 9. 기존 GF2 campaign에서 전환

사용자 방향 전환을 현재 우선 지시로 기록한다. 기존 P40의 모든 pair나20K/60K/100K를 끝낼 때까지 기다릴 필요는 없다. 다만 저장 중인 파일을 깨뜨리지 않도록 안전 경계에서 중단한다.

실행 담당자는 각 서버에서 controller·trainer·cache builder·postrun·watchdog의 실제 PID/소유 run을 먼저 확인한다. 기존 controller의 새 case admission 및 자동 재기동을 막고, 현재 optimizer update를 마친 뒤 full-state를 atomic save한다. checkpoint·optimizer·scheduler·RNG·sampler·actual update·pending evaluation 상태를 보존하고 `STOPPED_BY_USER_DIRECTION`을 남긴다. 미시작 old cases는 `CANCELLED_BY_USER_DIRECTION`으로 남긴다.

controller 종료만으로 watcher가 다시 old run을 켜지 않는지 확인한다. 새 PAN-Crafter runner와 이전 trainer를 같은 GPU에 겹쳐 실행하지 않는다. 범용 `pkill python`, `kill -9` 또는 과거 디렉터리 일괄 삭제는 사용하지 않는다. 이미 존재하는 결과·Sheet·checkpoint는 보존한다. 이번 전환의 과거 실험 결과를 완료된 pair처럼 새로 해석하지 않는다.

## 10. 무기한 실행의 제어와 장애 처리

실행 controller는 **모델을 바꾸지 않는 반복 관리**만 담당한다. 시간 제한이 없다는 것은 안전·저장 무결성 검사를 없앤다는 뜻이 아니다.

```text
load immutable recipe + local bindings + last cursor
verify old worker/watchdog stopped and local preflight passed
while not user_stop:
    generate current local cycle if absent
    for scheduled case in this local cycle:
        honor safe-pause/stop before admission
        check only local dependencies
        run fresh50K, or resume the same interrupted run, or evaluate WV2
        atomically save metrics/status; spool upload failures locally
    preserve cycle report
    advance local cursor to the next cycle
```

이는 controller 구현 계약이며 동봉 case 생성기가 학습을 수행한다는 뜻이 아니다. 상태 파일은 서버별 독립 namespace에 두고, 재기동 시 새로운 cycle을 중복 시작하지 않는다.

운영 제어는 `STOP_NOW_SAFE`, `STOP_AFTER_CURRENT_RUN`, `STOP_AFTER_CYCLE`, `CONTINUE`를 지원하도록 구현한다. 제어 문자열은 **신규 runner에 대한 요구사항**이며 기존 저장소에서 이미 지원되는 CLI라고 가정하지 않는다. immediate safe-stop 요청이 평가 중이면 평가 cursor를 저장하고 다음 checkpoint 평가부터 재개한다.

일시적 I/O 오류는 같은 run full-state로 최대2회 재시도한다. NaN/OOM에서는 상태를 보존하고 실패를 기록하며 batch·precision·patch size를 자동으로 바꾸지 않는다. source/data mismatch나 반복되는 deterministic 오류는 관련 dataset 또는 서버를 block한다. WV3 실패는 해당 WV2만 막으며 QB/GF2는 가능하면 진행한다. runnable case가 하나도 없으면 busy loop로 실패를 반복하지 않고 local pause한다.

Sheet 장애는 학습 실패가 아니다. 로컬 JSONL/CSV와 upload spool을 보존하고 원값 readback이 확인될 때 업로드 완료로 표시한다. 성능 미달은 장애가 아니며 다음 사전 지정 case를 계속 수행한다.

## 11. 저장·프로파일링·장기 반복 비용

무기한 반복이므로 중간 checkpoint를 전부 영구 복제하지 않는다. 마지막 두 resume state, best-validation model, exact50K model/full-state, 모든 config·hash·metric·status를 보존한다. 이미 완료된 run의 최종 결과를 자동 삭제하거나 새 cycle로 덮어쓰지 않는다. 부족한 디스크를 숨겨가며 계속 학습하지 않는다.

초기 free-space guard20GiB는 운영상 시작값이며, 실제 next-run 예상 비용과 checkpoint 크기를 측정해 더 높은 여유가 필요하면 조정한다. 이 threshold는 논문 세팅이 아니다. 한도에 닿으면 안전 저장 후 pause하고 남은 공간·보존 비용을 보고한다.

total/trainable params, inference latency, peak memory, MACs/FLOPs는 새 모델·센서 입력 크기·실제 GPU에서 측정한다. 논문7.17M/79.03G나 이전 `s1_A0`의 비용을 새 run 실측칸에 복사하지 않는다. MAC×2 여부, 포함 연산, 입력256/512, batch1, warmup과 synchronization을 기록한다. 논문 RTX3090의9ms와 현재 서버 latency는 같은 환경 비교가 아니다. [S1]

training time, validation, test evaluation, profile, preprocessing, I/O를 분리한다. WV2는 source WV3 학습 비용을 다시 신규 학습 비용으로 중복 합산하지 않는다. seed0 동일 seed3회와 이후 서로 다른 seed 반복의 summary는 분리한다.

## 12. 구현 전·후 검증

`acceptance_checks.csv`에20개 요구사항을 정의했다. 구현 전에 완료라고 바꾸지 않는다. 특히 다음은 장기 반복 전 필수다.

1. 9/5ch 입력, raw-PAN joint K/V와 mode query, fixed local gather 및 두 MARs loss의 gradient를 검사한다. LPAN 불변성 검사는 zero-init attention/output 때문에 거짓으로 통과하지 않도록 non-zero gate/비퇴화 fixture로 수행한다.
2. 무작위 flip 및 paired geometry, seed별 stream 차이와 재개 다음 batch ID, exact50K 종료를 확인한다.
3. 원본 data manifest와 native RR/paper FR 평가 정체성을 확인한다. WV2의 source SHA 불변·update0을 확인한다.
4. 각 서버의 작은 dry-run에서 stop/resume·원자적 저장·upload spool·중복 runner 금지를 확인한 뒤 전체50K cycle에 들어간다. smoke 결과를50K 재현 성능으로 표기하지 않는다.

## 13. 동봉 case와 구현 상태

`case_templates.csv`는 **4종류×3서버=12 template**이다. `examples/cases_cycle000000_000001.csv`에는 최초 두 cycle의24개 구체 case를 넣었다. 이후는 `generate_cases.py`가 같은 고정 정책으로 생성한다. 이 유한 export가 campaign의 최대 case 수는 아니다.

`recipe.json` 및 `bindings.example.json`의 실제 구현 symbol, deployed commit, data/evaluator manifest, GPU 및 local path는 검증 전 null이다. 원격 checkpoint나 runtime을 확인한 것처럼 채우지 않는다.

**완료된 범위:** 첨부 논문 및 live Sheet/코드 대조, 이 설계/registry/예시case 작성, case 생성기16개 단위 테스트.  
**미수행 범위:** corrected model/trainer 구현, remote stop·배포·GPU 학습·실측 profiling·live Sheet 쓰기.

## 14. 근거 목록

원본 식별과 URL은 `sources/source_index.json`에 저장했다. 논문·코드가 명시한 사실과 이번 운영 선택을 구분한다.

- **S1** 사용자 첨부 `pancrafter.pdf`: Eq.(1)–(12), Figure3/4, §4.1/4.2, Table6–10. Figure·표의 렌더링도 확인.
- **S2** live `pan-cvpr27 / WV3-s1 / B5:W14`, 2026-09-22 열람. 선택 열 전사본 `sources/WV3_s1_reference_rows.csv`.
- **S3/S4** pinned `config/paper_ln.yaml`, `config/s1_A0.yaml`.
- **S5/S6** pinned `model/pancrafter_paper.py`, `model/pancrafter.py`: 실제 내부 입력·모듈 경로.
- **S7** pinned `feeders/feeder.py`: seed 초기화, fixed flip, crop-resize 구현.
- **S8** pinned `results_log/2026-08-25_divergences-and-tuning-review.md`: 남은 차이, 미기재 선택, 파라미터 수정 예상.
- **S9** pinned `KNOWN_ISSUES.md`: LPAN sidecar/FR 세트/QB msfix 및 지표 구분. 초기 해석과 후속 감사가 다르면 현재 코드와 논문을 우선하여 별도 표시.
- **S10** pinned `tools/setup_wv2.py`: WV2 zero-shot 및 로컬 PAN 저해상도 생성 가정.
- **S11** pinned `train.py`: AdamW/scheduler와 frozen parameter 재확인 지점.
