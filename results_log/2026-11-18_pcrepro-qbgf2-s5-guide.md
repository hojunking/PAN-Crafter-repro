# s5: QB·GF2 PAN-Crafter 논문 기준 재현 실행 인계

**s5 에이전트의 실행 지시는 [pcrepro/S5_RUN.md](../pcrepro/S5_RUN.md)가 기준이다.** 사용자의 추가 지시에 따라 ZIP 이관 대신 **Git pull → 고정 commit 자동 배포 → 실험 실행**으로 정리했다. 이 문서는 모델·데이터·평가를 확인한 근거를 담는다.

## 확정된 범위와 현재 상태

사용자 확정: **QB와 GF2 각각 seed 2025로 fresh 50,000 updates 1회**. 순서는 QB → GF2이며, 두 학습과 후처리가 끝나면 종료한다. 준비 서버는 s1이고 실제 실행 서버는 s5다. s1에서는 GPU 학습, 원격 접속, 기존 작업 중단, 시트 쓰기를 실행하지 않았다.

기준 구현은 `pcrepro/model.py::CorrectedPANCrafter`, recipe는 `PC_PAPER_EQ11_C128_D224_LN_MARS_v1`이다. 여기서 **Corrected는 attention 구현의 수정**을 뜻한다. PAN Aligner 또는 corrected PAN 입력을 사용하는 PAN-Weave 모델이 아니다.

| 후보 | 구분 | 이번 사용 |
|---|---|---|
| `model/pancrafter.py` | 배포본 및 fix 플래그 변형, 4-scale 골격 | 사용하지 않음 |
| `model/pancrafter_paper.py` | 3-scale 논문 골격이나 기존 PAN attention 입력 경로가 남아 있음 | 부모 골격으로만 재사용 |
| `pcrepro/model.py` | 위 골격에 Eq.(11)의 raw-PAN query/joint K/V와 정확한 3×3 gather를 반영한 당시 PCREPRO | **이번 학습 모델** |

기존 `tools/pcrepro_start.sh`는 과거 GF2 작업 전환과 WV3/WV2까지 포함한 무기한 cycle 실행기다. 이번 운영 진입점은 **`tools/pcrepro_qbgf2_s5.py`**이며, 내부에서 유한 실행기 `tools/pcrepro_qbgf2.py`를 호출한다. 별도 ID·제어 폴더를 쓰며 기존 작업을 종료하거나 기존 큐를 수정하지 않는다.

## 모델·학습 설정

로컬 원문 `2505.23367v2.pdf`의 식(9)–(12), §4.1–4.2와 기존 recipe를 대조했다. 이번 변경은 실행 범위·데이터 연결·비용 측정 경로에 한정하며 `pcrepro/model.py`, `pcrepro/recipe.json`의 모델·학습 수학은 유지했다.

| 항목 | 고정 값 |
|---|---|
| 모델 | 3-scale, C128, encoder depth 2/2, bottleneck 4, decoder 2/2, channel LN |
| Attention | CM3A 3곳, 8 heads, 3×3 `unfold`, raw PAN을 C4로 반복한 joint K/V, mode별 query |
| MS 추론 | raw PAN + 내부 bicubic MS → backbone → MS residual 덧셈 |
| 채널·파라미터 | QB/GF2 모두 MS 4 bands, 입력 5 channels, **7,129,476 total/trainable parameters** |
| 학습량·seed | 데이터셋별 fresh 50,000 optimizer updates, seed 2025 |
| Batch·patch | 48 PAN/MS 쌍 → MARs 복제 후 96, PAN/GT 64×64, MS 16×16 |
| Optimizer | AdamW, LR 1e-4, weight decay 0.01, betas 0.9/0.999, eps 1e-8 |
| Scheduler | warmup 100 updates → cosine, 50K에서 종료 |
| Loss | L1 MS + L1 PAN, PAN coefficient 1 |
| 정밀도 | FP32, AMP 없음, TF32 off, 기존 deterministic/cuDNN 정책 유지 |
| PAN-mode residual base | 같은 augmented PAN의 Gaussian sigma 1.98, k41, replicate, phase2; 외부 LPAN 파일 불필요 |
| 데이터셋 차이 | QB maxDN 2047, GF2 maxDN 1023; 동일 구조·LR·학습량 |
| 선택 | primary **EXACT_50000**, secondary **RR_VAL_ERGAS_MIN** |

검증은 1,000 updates마다 전체 native64 validation set에서 수행한다. Secondary는 이 50개 지점의 validation ERGAS 최솟값이며 동률이면 앞선 지점이다. Test RR/FR 점수로 checkpoint를 고르지 않는다.

논문에 명시되지 않은 depth 배분, LN 축, heads/FFN, Gaussian residual 세부사항, interpolation flags, precision, checkpoint 선택은 당시 PCREPRO의 명시된 구현 선택을 유지했다. 따라서 이 준비가 논문 표의 점수 일치까지 보증하는 것은 아니다. 7.17M에 맞추기 위한 임의 구조 변경도 하지 않는다.

## 데이터와 평가

| 데이터셋 | Train | Validation | RR | FR |
|---|---:|---:|---:|---:|
| QB | 17,139 | 1,905 | 20 | 20 |
| GF2 | 19,809 | 2,201 | 20 | 20 |

**QB train/val은 원본 `train_qb.h5`, `valid_qb.h5`다.** `ablr2/sensor_sources.json`의 기본 train/val 항목에는 `msfix`가 있으므로, 새 binding 코드는 `source_provenance.raw_train_path/raw_val_path`와 해당 SHA를 명시적으로 사용한다. 이는 당시 PCREPRO 재현 규약을 보존하기 위한 선택이다. GF2는 기존 원본 H5를 사용한다.

- RR: `reduced_examples_h5/test_{qb|gf2}_multiExm1.h5`, 20장, native PAN/GT 256·MS 64. 지표 support는 기존과 같은 `[20:-21,20:-21]`.
- FR: `full_examples_mat20/test_{qb|gf2}_OrigScale_mat20.h5`, 20장, native PAN/LMS 512·MS 128. 구형 `full_examples_h5`로 대체하지 않는다.
- FR 순서는 원본 `full_examples_mat/Test(HxWxC)_{qb|gf2}_data_fr1.mat`부터 `fr20.mat`까지 숫자 순서다. `provenance.json`의 SHA 및 원본 MAT MS/LMS/PAN과 H5 tensor가 정확히 일치해야 한다.
- 입력과 FR 평가 참조 모두 **original PAN**이다. QB/GF2용 Aligner가 없으므로 corrected PAN 결과는 만들지 않는다.
- RR: ERGAS, SAM, SCC, PSNR, SSIM, Q4, RMSE, CC. FR: HQNR, Dλ, Ds, JQM. JQM은 현재 사용 중인 NNLS SRF-substitute 정의다.
- 각 지표는 scene별 계산 후 20장 평균. RR20과 FR20 사이의 일대일 scene 대응은 가정하지 않는다.

`pcrepro/s5_assets.json`에는 필요한 50개 입력 파일의 상대 경로·SHA·크기가 있다. **데이터는 Git에 넣지 않는다.** s5에 같은 파일이 있으면 재사용한다. 파일이 없거나 hash가 다르면 해당 파일을 s1의 검증된 자산으로 옮기거나 서버의 같은 자산 위치에 연결한다. 검증 우회를 위해 SHA를 바꾸지 않는다.

현재 RR20은 원본 H5 행 순서를 고정한 비교 실험 데이터다. 논문 저자가 사용한 RR 목록 자체를 별도로 대조한 것은 아니므로, 완료된 summary가 `PAPERSET_IDENTITY_UNVERIFIED`를 표기할 수 있다. `complete: true`와 해당 문구는 양립하며, 논문 표와의 identity를 임의로 VERIFIED로 바꾸지 않는다.

## s5에서 실행

기존 PAN-Crafter 저장소에서 pull하고, s5의 기존 학습 Python 환경으로 다음 명령을 실행한다. `gspread/server.txt`는 로컬 서버 marker로 `s5`여야 한다. 상세 지시는 [S5_RUN.md](../pcrepro/S5_RUN.md)를 따른다.

```bash
git pull --ff-only origin main
python tools/pcrepro_qbgf2_s5.py start
```

`start`는 commit을 별도 detached worktree에 고정하고 `data`와 `work_dir`를 연결한다. 준비 단계에서 두 데이터셋 전체의 hash/finite/shape/FR MAT identity를 확인하고, s5 소스·패키지·CUDA 버전으로 불변 registration 및 설정을 만든다. s1의 registration을 복사하지 않는다. 준비가 성공하면 연결 종료 후에도 유지되는 별도 세션으로 runner를 실행한다. 원 저장소의 후속 pull은 고정 worktree를 바꾸지 않는다.

내부 runner는 다음 순서로 **총 2건만** 진행한다.

1. QB: 데이터·평가 모듈·여유 디스크·CUDA 점검 → 실제 모델 batch48/MARs96 forward/backward → fresh50K → 두 checkpoint의 RR20/FR20 평가 및 비용 측정.
2. GF2: 같은 순서로 1회 진행.
3. `additional_models_rows.json` 저장 후 COMPLETE로 종료.

기존 GPU compute PID가 있거나 상태를 조회할 수 없으면 실행을 중단한다. 다른 프로세스에 신호를 보내지 않는다. 비용 측정 중에도 다른 compute PID 유무를 반복 확인한다. 이 검사는 일반 그래픽 작업까지 GPU를 완전히 독점한다는 보장은 아니다.

### 환경 확인 범위

로컬 검증 환경은 Python 3.9.19 / PyTorch 2.4.0+cu118이다. 이 버전을 s5에 강제로 설치하라는 의미는 아니다. **s5 GPU와 드라이버는 이번 세션에서 조회하지 않았다.** s5 GPU를 지원하는 기존 PyTorch/CUDA 환경에서 실행하고 실제 GPU 이름·compute capability·PyTorch/CUDA를 preflight에 기록한다.

필요 모듈은 PyTorch 및 호환 torchvision, timm, safetensors, numpy, scipy, scikit-image, h5py, PyYAML, gspread이며 기존 LP residual 검증 테스트에 cv2가 필요하다. gspread는 공통 provenance 의존성이며 이번 실행은 인증정보를 읽거나 API를 호출하지 않는다. 이 모델에는 별도 CUDA extension 빌드 단계가 없다.

Full-batch preflight는 실제 C128/D224 모델의 **48쌍/MARs96 forward/backward**와 batch1 operator counter 실행을 검사한다. optimizer update는 수행하지 않으며 AdamW state와 checkpoint I/O 메모리까지 포함한 완전한 학습 벤치마크는 아니다. OOM·미지원 연산·NaN이 발생하면 로그를 남기고 멈춘다. batch/precision/depth를 자동으로 낮추지 않는다. 여유 디스크는 기존 기준인 최소 20GiB를 검사한다.

### 상태·정지·재개

다른 셸에서도 같은 환경과 `PANCRAFTER_DLPAN`을 설정한 뒤 실행한다.

```bash
python tools/pcrepro_qbgf2_s5.py status
python tools/pcrepro_qbgf2_s5.py pause

# 이전 runner 종료와 GPU 상태 확인 후
python tools/pcrepro_qbgf2_s5.py resume
```

Pause는 optimizer update 경계에서 full state를 저장한다. 재개는 동일 run의 optimizer/scheduler/RNG/sampler 상태를 복원한다. 완료된 학습은 반복하지 않고 미완료 후처리를 재시도한다. 이미 완료한 첫 데이터셋은 검증 후 건너뛴다. source/data/runtime mismatch는 해당 원인을 해결해야 하며 `continue`로 무시할 수 없다.

## 결과 위치와 s1 이관

Run IDs:

```text
PCPAPER_QB_s5_S2025_F50K_20261118_v1
PCPAPER_GF2_s5_S2025_F50K_20261118_v1
```

운영 경로는 `work_dir/_pcrepro_qbgf2_s5_20261118/`이다. `logs/<run_id>.{preflight,train,postrun}.log`, `registration.json`, `status.json`, `preflight/{QB,GF2}.json`을 확인한다.

각 run의 `work_dir/<run_id>/`에 다음이 저장된다.

- `meta/config.resolved.yaml`: resolved 설정과 소스·데이터 identity.
- `checkpoints/exact_50000.json`: primary checkpoint 디렉터리를 가리키는 pointer. 실제 weight는 해당 디렉터리의 `model.safetensors`, identity는 `identity.json`이다.
- validation 최소 checkpoint와 마지막 2개 resume state: 기존 저장 정책 유지.
- `official/summary.json`: 두 선택점의 RR/FR 결과와 provenance.
- `official/profile.json`: primary의 RR/FR time·memory·FLOPs·Param 및 측정 원자료.
- `official/evaluations/<checkpoint_sha>/`: 평가 cursor·원본 지표·예측 자료.

### 비용 측정 정의

| 항목 | 정의 |
|---|---|
| 입력 | batch1, FP32/TF32 off, RR PAN256/MS64 또는 FR PAN512/MS128 |
| 측정 범위 | **MS-mode inference: 내부 bicubic MS resize → backbone → residual add** |
| 제외 | 파일 I/O, H2D, 외부 DN 정규화, prepared model 밖의 검증, metric, training PAN-mode 처리 |
| Time | 고정 scene 0/7/14, 각각 warmup20 + 동기화 wall time100 + 별도 CUDA event100. 대표값은 300개 wall time의 median |
| Memory | 모델+resident input+forward 중간값을 포함한 peak CUDA allocated memory; MiB. Reserved/nvidia-smi 전체 점유와 다름 |
| FLOPs | `tools/additional_models_flops.py`, 1 MAC=2 FLOPs, G=1e9. Conv/Linear, local attention의 scalar multiply/reduction, norm/activation/interpolation 등 포함 |
| Param | total/trainable 모두 7,129,476 = 7.129476M |

RAMSF·ADWM에 사용한 arithmetic counter와 같은 규칙을 사용한다. 이번 PAN-Crafter 경로에 필요한 GELU, transpose-convolution, unfold/chunk 연산 처리를 추가했다. GELU는 erf형 5·tanh형 9 primitive operations/output, bicubic은 32 operations/output으로 추정한다. 인덱싱·메모리 이동·비교는 0으로 별도 기록하며, 미지원 floating 연산은 측정 실패로 처리한다. 하드웨어 instruction 수를 정확히 센 수치는 아니다.

Prepared MS path와 일반 `model(pan, ms)` 출력의 bit equality를 각 측정 scene에서 확인한다. 모델 내부의 mode 검사가 남아 있는 구간은 그대로 측정한다. PAN-Crafter에는 MS resize가 원래 모델 내부에 있으므로 이를 포함한다. 외부에서 이미 만든 MS base를 넣는 PAN-Weave의 시간 범위와 이 차이를 표 주석에 남겨야 한다.

Time/memory는 **측정한 GPU의 수치**다. 기존 비교표가 s1 GPU에서 측정됐다면 s5 시간을 같은 하드웨어 측정처럼 섞지 말고 GPU를 표기하거나 weight를 s1로 옮겨 재측정한다. 이번에는 s5 측정을 실제 실행하지 않았으므로 새 latency/memory 값을 미리 제시하지 않는다.

### addtional-models 업로드용 결과

`work_dir/_pcrepro_qbgf2_s5_20261118/additional_models_rows.json`은 두 primary의 metric·cost·checkpoint SHA·경로를 담은 **이관용 의미 필드 JSON**이다. 현재 Google Sheet의 A:BL 열에 직접 넣을 행 배열은 아니다.

대상: spreadsheet `pan-cvpr27-new` / ID `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE` / tab **`addtional-models`**. 과거 `PC-Repro-s5`로 업로드하지 않는다. 이번 실행기는 자동 시트 쓰기와 인증정보 이관을 하지 않는다.

s1에서 다음 작업을 진행할 Codex는 두 run의 결과·checkpoint 및 이 JSON을 받고, 현재 시트 열을 조회해 primary QB/GF2 original-PAN 결과를 추가한 뒤 readback을 확인한다. 필요하면 s1 환경에서 metric/cost를 재측정한다. s5의 절대 경로와 runtime에 묶인 registration을 s1에서 그대로 실행·수정하지 않는다. 이관 JSON에는 secondary checkpoint 위치도 남지만 두 checkpoint를 두 개의 독립 실험처럼 중복 업로드하지 않는다.

## 로컬 검증 근거

- 변경 전 기존 128 tests, finite 실행/측정 11개 및 Git 배포/중복 방지 7개를 포함한 **146 tests 통과**.
- QB/GF2 train/val 전체 파일 hash·유한값·shape 검사, RR20/FR20 및 40개 원본 FR MAT 대조 통과.
- 두 센서 모두 실제 C4/C128/D224 모델로 CPU native64 MARs 1쌍→2 forward/backward 유한값 확인; optimizer update 0회.
- 출력 head를 0이 아닌 값으로 바꾼 검사에서도 prepared/native 출력 bit equality 확인.
- 실제 데이터 scene0에 동일한 bounded DN 예측을 넣었을 때 `pcrepro.evaluation`과 현재 `ta2.evaluation`의 RR 8개·FR 4개 지표 차이 모두 **0**. 이는 평가 코드 일치 검사이며 새 모델 성능 수치가 아니다.
- 전체 모델 native64 operator audit에서 미계산 floating 연산 0개. s5 실제 CUDA batch48/MARs96, 50K 학습, latency/memory 측정은 아직 수행하지 않았다.

근거 파일: `results_log/assets/2026-11-18-pcrepro-s5-review/`의 `tests_before.log`, `tests_after.log`, `audit_local.py`, `local_audit.json`, 센서별 operator audit. s5 배포 코드는 임시 Git 저장소에서 worktree 생성·원 저장소 후속 commit에도 실행 코드 유지·중복 실행 방지·실패 시 학습 미시작을 검증했다. 데이터 전체 manifest와 이전 ZIP은 로컬 검토 산출물이며 pull 실행에 필요하지 않아 commit 대상에서 제외한다.
