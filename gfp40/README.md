# GFP40 — GF2 PANMIX s3/s4/s5 40h

원본 입력은 `research_log/PANDA_GF2_P40_S345_40H_PANMIX_ExperimentPlan_2026-09-22.md`와 `research_log/GFP40_Cases_All83.csv`다. 이 모듈은 B20 clock·queue를 변경하지 않는 별도 campaign이다. 원본 문서·CSV와 SHA는 보존하며, 이후 사용자의 서버 간 대기 제거 요청을 실행 registry에 명시적으로 반영한다.

## 정의·선택

- Campaign `PANDA_GF2_P40_S345_20260922_v1`, s3 19 / s4 31 / s5 33 = 최대83 학습 구간.
- PRIMARY74 + OPTIONAL9. PRIMARY는 fresh100K10개, FT20K60개, FT60K4개이며 추가2,440,000 update. 전량 완료 보장이 아니다.
- Teacher는 기존 로컬 P0/W112/D123, Student PLH/W104/D122, α1/β.1/edge.002, GF2 C4/DN1023/batch48/FP32 고정이다. 새 Teacher, backbone 변경, s1/WV3·s2/QB 변경은 없다.
- Confirmation은 세 서버 모두 **사전 고정 G025**다. s4 family screen은 탐색 보고만 하며 confirmation 분포를 변경하거나 다른 서버의 진행을 막지 않는다. 실행 phase는 `FIXED_CONFIRM`, 관련 run ID의 `MSTAR` 부분은 `G025_FIXED`로 구분하고 `MSTAR_LOCK` 의존성을 제거한다.
- FT는 자기 부모 U/A만 초기화하고 U/A optimizer·scheduler를 reset한다. fresh는 새 U와 로컬 Teacher A의 독립 clone이다. 같은 run의 재개만 full-state를 사용한다.
- 공식 primary EXACT_FINAL, secondary native val-ERGAS 최소 RR_VAL_SELECTED, auxiliary RAW_AUX(HQNR→SCC→ERGAS). 기존 캠페인의 선택 규칙 변경은 아니다.
- FT20K는1000 간격20후보, FT60K는3000 간격20후보, fresh100K는 B20에서 검증한2020 간격+100000의50후보다. FT60K20K/40K curve와 fresh50K는 진단용이며 공식 선택 후보가 아니다. fresh50500은 공식 후보다.

`GFP40_ParentAssets.json`/author Registry·Queue JSON은 제공되지 않았다. 본 registry는 **제공된 MD+CSV의 파생 구현**이다. 여섯 부모의 원 run/SHA/anchor는 Git `5d9f4d6`의 B20 ParentAssets와 현재 archive bytes가 일치함을 확인해 `parent_registry.json`에 출처와 함께 담았다. 이를 원격 checkpoint bytes 검증 완료로 해석하지 않는다.

서버별 Teacher는 이미 R3_100/R4_100/R5_100으로 분리되어 있다. 기존 MSTAR lock은 Teacher 공유가 아니라 분포 선택 전달이었다. 신규 Teacher를 재학습하면 기존100K 부모가 사용한 reference·q/τ/qref 계보와40h 예산이 달라지므로 이번 독립 실행 수정에서 자동 추가하거나 원 Teacher를 바꿔 끼우지 않는다. 신규 Teacher를 포함하려면 별도 실험 정의와 자산·예산 조정이 필요하다.

## 분포·cache

G025/G0125/G050/P075/P025/LOW/HIGH의 gain과 정수 token multiplicity를 고정한다. Native arm도 동일 uniform53 draw를 소비하며 family CDF에 따라 drawn gamma를 기록한다. 비교 스트림은 source·geometry·uniform 기준으로 맞춘다. family 간 effective gamma의 일치는 요구하지 않는다.

gamma1은 원 PAN/LP tensor를 직접 반환한다. 나머지는 float64 DN의 σ1/k7 reflect Gaussian gain→기존 σ1.98/k41 replicate LP→fixedHV/ROT4 순서다. MS/GT 및 val/RR/FR 입력은 변경하지 않고 clamp·반올림·mask도 추가하지 않는다.

PAN/LP와 Teacher q/error는 gamma별 append-only shard다. q는 전체19809×ROT4×해당 gamma, batch16/FP32를 요구한다. calibration3072의 pooled-pixel exact weighted median 및 qref median을 사용하며 짝수 중앙값은 두 값 평균이다. 토큰을 늘린 작은 fixture와 실제 subset 대조도 수행한다. P025/P075/LOW/HIGH는 가능한 q slice를 공유하고 scale만 별도로 고정한다.

B20 cache 재사용은 실제 B20 source archive, Teacher/data/runtime, 전체 native parity 증거, PAN/LP 전 화소 및 추가 augmented q parity 검증이 필요하다. γ별 error population은 현재 Teacher로 재측정한다. 재사용 자료가 없으면 전체 새 cache를 만들며 비용을 예약한다. 다른 Teacher의 q/τ/qref를 가져오지 않는다.

## 배포 전제

먼저 이번 관련 코드·테스트·MD·CSV를 커밋하고 각 서버에 배포해야 한다. 관련 numerical source의 미커밋 변경이 있으면 launcher는 거부한다. 다른 작업의 변경은 stash·삭제하지 않는다.

원본 자료가 `research_log/past`로 이동해 공통 import가 깨지는 문제를 위해 QG40/G20/L100의 plan loader만 최소 수정했다. `fh12/source_documents.py`는 원 경로가 없을 때만 같은 상대 archive 경로를 읽고 **기존 SHA를 그대로 검증**한다. 원 경로가 존재하지만 내용이 다르면 archive로 우회하지 않는다. 문서 이동·삭제는 되돌리지 않았으며 학습·평가 수식도 바꾸지 않았다. frozen checkout에 이 종속 원본 문서가 실제 존재하는지도 확인한다.

기존 부모를 생성한 RTX5090 환경의 Python을 사용한다. launcher는 Docker 이미지를 자동 생성하거나 패키지를 바꾸지 않는다. CPU/4090 fallback도 없다. 원래의 B20 frozen checkout 또는 검증 가능한 source archive가 필요하다. `060ba8...`와 `5d9f4d...`를 같은 source로 간주하지 않는다.

```bash
python tools/gfp40_runner.py inspect-assets --server s3
```

위 명령은 등록 run 경로 및 기존 B20 C07–C10 상태의 읽기 전용 발견이다. 학습·clock·자산 SHA 검증 완료를 뜻하지 않는다.

## 서버별 독립 시작

각 서버에서 자신의 명령만 실행한다. 공유 디렉터리, 공통 시작 시각, s4 완료 통보 또는 lock 파일 전달이 필요하지 않다.

최초 `start`는 **해당 서버의 실제 시작 시각**을 자동 기록하고 그 시각부터40h를 계산한다. 서버별 시작 시각은 달라도 된다. `build`, `inspect-assets`, `status`는 clock을 시작하지 않는다.

```bash
bash tools/gfp40_start.sh --server s3
bash tools/gfp40_start.sh --server s4
bash tools/gfp40_start.sh --server s5
```

`PYTHON=/absolute/path/to/python`으로 interpreter 지정 가능. 기본 background, `--foreground` 지원. commit을 고정한 sibling worktree에서 실행하며 `data`, `work_dir`, credential만 연결한다. 로컬 clock은 `work_dir/_gfp40/<server>/campaign_window.json`이다. 필요한 경우 최초 시작에만 자기 서버의 timezone-aware 실제 시각을 `--t0`로 명시할 수 있다. 미래 t0 이전의 준비 실행이나 재개 시 clock 초기화는 허용하지 않는다.

원본 계획의 공통 clock·22h MSTAR 확정·서버 간 선택 전달은 이번 사용자 요청으로 대체됐다. s3/s5는 s4 screen 완료 여부와 관계없이 자기 부모·cache가 준비되면 고정 G025 confirmation을 진행한다. s4 screen 결과가 좋아도 이번 confirmation을 사후 변경하지 않는다.

삭제한 것은 **서버 간 선택 대기**다. 같은 서버에서 두 runner가 동시에 GPU를 쓰거나 같은 파일을 덮어쓰는 것을 막는 process/file lock과 checkpoint·cache 무결성 검사는 유지한다. SSH 대상이나 인증을 추측하지 않으며 다른 서버를 자동 실행하지 않는다.

## 자산의 명시적 경로 지정

자동 발견은 정확한 등록 run ID만 사용한다. 필요하면 시작 시 `--bindings /absolute/bindings.json`을 전달한다. 예시의 실제 경로로 교체한다.

```json
{
  "parents": {
    "P3HI": {
      "origin_root": "/absolute/original/repository",
      "parent_config": "/absolute/run/meta/config.resolved.yaml",
      "parent_checkpoint": "/absolute/run/candidates/100000/model.safetensors",
      "parent_identity": "/absolute/run/candidates/100000/identity.json",
      "parent_fullstate": "/absolute/run/candidates/100000/training_state.pt",
      "reference_manifest": "/absolute/original/reference_manifest.json"
    }
  },
  "b20_source_archive": {
    "root": "/absolute/actual/B20/frozen-checkout",
    "source_identity_path": "/absolute/B20/preflight.json"
  },
  "b20_reference": "/absolute/verified/B20/mixed_reference_manifest.json"
}
```

보통 `_gfb20/<server>/runtime_release.json`, `preflight.json`, s5 mixed reference에서 자동 발견한다. 없는 optional b20_reference는 생략한다. 원 부모·Teacher·data/LP·q·calibration·runtime·source archive는 실행 전 검증한다. 부모 native 재평가는 기존 B20 `atol=1e-6,rtol=0`을 고정 사용한다. 이는 사전 engineering equality guard이며 이번 서버 반복 측정으로 산출한 범위가 아니다. 실패 후 자동 완화하거나 더 좋은 부모로 교체하지 않는다.

## 시간·안전·재개

- 같은 서버의 B20 및 다른 worker를 kill하거나 큐를 변경하지 않고 안전 종료를 기다린다. 기존 큐를 자동 drain하는 기능은 없으므로 인계 시점을 조율한다. 자기 서버 t0 이후의 대기도40h에 포함된다. 다른 서버의 실험 완료는 기다리지 않는다.
- 34h 이후 새 block 금지,37h 이후 optimizer 금지,37–40h 평가·보존·보고. 이미 입장한 block은 예측 비용 증가만으로 취소하지 않고37h까지 보존 가능한 범위에서 계속한다.
- CTRL/MIX pair, 두 부모 family4arm, schedule2trunk+4tail, confirmation trunk+2tail의 전체 비용을 예약한다. native trunk 선행 시에도 미완 고정 G025 tail과 cache 시간·디스크를 예약한다. 미정 MSTAR 후보를 위한 별도 cache 여유는 요구하지 않는다.
- train/eval/I/O/diagnostics를 포함한 같은 서버·길이·view 실측×1.15와 초기 학습 floor 중 큰 값을 쓴다. 누락 gamma당.75h+미완 family pooling당.20h를 보수적 cache 예약 floor로 추가했다. 이는 실측 주장이나 학습 recipe 변경이 아니며, 완료된 calibration 실측이 더 크면 올린다. s5의2h 추정은 cache 재사용이 검증되지 않았을 때 적용하지 않는다.
- 시간 부족 시 **미입장 block만** 계획서 우선순위로 제외한다. 높은 HQNR이나 첫 arm의 낮은 점수 때문에 예약한 나머지 arm/seed를 취소하지 않는다.
- transient I/O train 재시도는 동일 full-state로 최대2회. upload 재시도는 재학습하지 않는다. FT val-ERGAS≥1.10×parent가 연속2후보면 안전 pause 후 검토한다. 같은 상태 재개만으로 발산 판정을 지우지 않는다.

```bash
python tools/gfp40_runner.py status --server s3
python tools/gfp40_runner.py stop --server s3 --command STOP_NOW_SAFE
python tools/gfp40_runner.py stop --server s3 --command STOP_AFTER_BLOCK
```

STOP_NOW_SAFE는 trainer의 optimizer 경계 full-state 보존, 준비·평가에서는 소유 subprocess에 cooperative signal을 보낸다. 다른 worker나 저장 중 프로세스를 강제 kill하지 않는다.

```bash
python tools/gfp40_runner.py stop --server s3 --command CONTINUE
bash tools/gfp40_start.sh --server s3
```

재개에 새 t0를 주지 않는다. 같은 frozen release와 **해당 서버의 기존 clock**을 사용한다. 기존 명시적 bindings가 필요하면 동일 파일을 다시 전달한다. 다른 서버의 시작·재개는 이 clock을 바꾸지 않는다.

## 산출물·평가·Sheet

`work_dir/<run_id>/`의 native RR20/FR20 결과를 보존한다. HQNR/SCC/ERGAS를 출력하며 Q4/RMSE/CC/JQM 등 기존 모든 metric을 유지한다. JQM은 SRF-substitute 변형이다. RR20:-21, FR full512/no-mask 규약을 바꾸지 않는다. 표시는 metric 소수4자리, 저장·업로드 원값은 full precision이다.

Sheet는 `GF2-P40-s3/s4/s5` 전용 탭, 실제 시작 UTC/KST·parent/additional/lifetime update·Teacher/parent/new 비용·source/cache/checkpoint·selection별 동일 checkpoint 지표·readback을 기록한다. 기존 B20/L100 탭과 헤더는 덮지 않는다. `--no-upload`로 live 업로드를 끌 수 있다.

```bash
python tools/gfp40_runner.py report --server s3
python tools/gfp40_runner.py retry-uploads --server s3
```

`work_dir/_gfp40/<server>/`에 completed/failed-or-deferred/paired/schedule CSV 및 campaign JSON을 쓴다. 양의 결과만 집계하지 않는다. 대표 G025 pair A02/A03, B03/B04, C01/C02는 CC/CM/MC/MM A/U crossing replay를 별도로 평가한다. 고정 train128/gradient24, native val/RR 품질·c drift·gamma perturbation 및 고정5장×3crop 시각자료를 남긴다. 모든 RR/FR20 scene metric과 leave-one-scene-out은 해석용이다. crossing 출력은 공식 선택 후보·새 부모가 아니다. replay 재시도는 최대3회이고 미완은 `DIAGNOSTICS_PENDING/FAILED`로 남긴다. 독립 정합 추정기는 `ESTIMATOR_NOT_AVAILABLE`이며 물리적 정합 정확도 측정으로 주장하지 않는다.

각 서버의 `confirmation_evidence.json`을 모은 후 다음처럼 로컬6-pair 집계를 수행한다. 이 파일 수집은 **학습 완료 후 분석**이며 학습 진행의 전제 조건이 아니다. 서버별로 다른 clock을 허용하고 각각 보존하되, 동일 고정 G025 정책·registry/method revision과 각 서버 case·Teacher/parent 계보를 검증한다. 서버 간 절대 점수를 같은 Teacher의 반복 실험처럼 섞지 않고 서버 내부 paired effect를 집계한다.

```bash
python tools/gfp40_runner.py aggregate --reports /absolute/s3.json /absolute/s4.json /absolute/s5.json --output /absolute/p40_confirmation.json
```

6개 PRIMARY pair가 모두 없으면6-pair 성공으로 표시하지 않는다. 새 seed라도 전체는 test-aware development이며 Teacher block을 독립·동일분포 표본으로 간주하지 않는다. 공동목표는 같은 EXACT_FINAL H>.964/E<.552, strong goal E<.522를 별도로 기록한다.

## 검증 명령

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest discover -s gfp40 -t .
PYTHONDONTWRITEBYTECODE=1 python -m unittest qg40.test_plan g20.test_plan l100.test_plan
bash -n tools/gfp40_start.sh
```

CPU fixture·mock 결과는 실제 s3–s5 자산/GPU/cache/Sheet 검증이 아니다. 이번 구현 작업에서는 새 clock·학습·Sheet 쓰기를 실행하지 않았다.
