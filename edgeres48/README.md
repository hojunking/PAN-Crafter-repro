# EdgeRes48H-v2 — RR-only Stage 1

사용자 확정: 새 workbook `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE`의
`WV3-ablations / QB-ablations / GF2-ablations`, 대표 순서 **HQNR 내림차순 → ERGAS 오름차순**.
계획서 §14의 구형 workbook/analysis는 사용하지 않는다. SCC는 필수 측정·출력하지만 tie-break가 아니다.

## 바로 실행

각 서버에서 pull 후 해당 명령 하나를 실행한다. pull 자체가 학습을 시작하지 않는다.

```bash
bash scripts/run_edgeres48_docker.sh s1  # WV3 (여기)
bash scripts/run_edgeres48_docker.sh s2  # QB
bash scripts/run_edgeres48_docker.sh s3  # GF2
```

기존 `hojunqueen/pancrafter-env:torch2.4.0-cu118` 환경 이미지를 사용한다. 이미지를 자동 pull하거나
서버 환경을 수정하지 않는다. 기본 registry에 등록된 train/val/RR20/FR20 데이터와
`gspread/account.json`은 각 서버의 기존 로컬 자산이어야 한다. QB는 원배포
`train_qb.h5 / valid_qb.h5`의 LMS도 필요하다. 해당 파일 내용·행 대응은 검증하며 다른 LMS로 대체하지 않는다.

Docker 없이 기존 환경에서 실행하려면:

```bash
conda activate pancrafter
bash scripts/run_edgeres48.sh s1
```

런처는 `MKL_THREADING_LAYER=GNU`를 지정한다. PyTorch 부모 프로세스에서
별도 gspread publisher를 시작할 때 Intel MKL/libgomp 충돌로 업로더만 종료되는 문제를 방지한다.
학습 종료 뒤에도 outbox를 비울 publisher가 끝날 때까지 컨테이너를 유지한다.
48h 이후 이 과정은 업로드 재시도일 뿐 학습·평가 시간을 늘리지 않는다.

서버 간 대기/Teacher 전달/lock은 없다. 같은 서버·캠페인의 중복 실행만 즉시 거부한다.
원래 48h 시계는 재실행해도 유지한다. 죽은 미달 모델은 저장하지 않았으므로 새 attempt로 처음부터 시작한다.

## 계약

- `EdgeMatch(P, provided LMS) + residual CNN`, 원 PAN을 최종 한 번 warp하여 기존 P0/BICUBIC Teacher에 전달.
- RR64 학습만 사용. FR512는 평가 전용이며 FR loss·FR training source·scene map·global prior를 요구하지 않는다.
- 기본 radius4, 모든 native/ε/RR/FR에서 fixed `R+5` score 영역. absolute loss margin16.
  이들은 v2의 운영 제안을 명시적으로 채택한 값이지 측정된 최적값이라는 주장이 아니다.
- 원배포 QB LMS + 기존 msfix reconstruction MS 조합을 명시적으로 보존. 두 영상의 생성 위상 동일성은 주장하지 않는다.
- AdamW, effective48 = micro12 × accumulation4, FP32·AMP/TF32 off. independent seeded A/T/data/augmentation/ε.
- EF/ET/H 설정, 원문 SHA, source SHA, evaluator 구현·라이브러리, 코드 SHA를 실행 전에 고정.
- disposable 300 updates와 실제 full20 RR/FR·진단 시간을 사용하여 ETA×1.2, 후속 R2/R3와 2h 마감을 예약.
- 공식 RR는 기존 `ta2.evaluation`의 all20 / dim_cut21 관례, FR는 원 PAN·native LMS full512 unmasked.
  구조 loss support 또는 aligned PAN을 HQNR reference에 쓰지 않는다.
- 모든 필수 지표: HQNR, D_s, D_lambda, JQM, ERGAS, SCC, SAM, PSNR, SSIM, Q4/Q8, RMSE, CC.
  JQM은 기존 **SRF-substitute** 변형이며 SIPSA 동등 구현이라고 표시하지 않는다.
- 동일 A/T snapshot에서 HQNR·ERGAS를 동시에 통과한 모델만 가중치·정성 이미지 저장.
  미달 모델은 전체 aggregate/per-image 수치와 logical model SHA만 저장. 임시/latest/best-miss weights 없음.
- θ 상태의 SHA와 파일 SHA를 구분. `Checkpoint_SHA`는 실제 저장된 qual 파일에만 존재한다.
- 진단 `CURRENT/PRIOR_ONLY/ZERO` 결과는 DIAG_ONLY. 대조 경로가 목표에 도달해도 full-method qual로 승격하지 않는다.

## 조건부 탐색

16개 기본 EF를 서버별 우선순위로 시도하되, 48h 안에 전부 수행한다는 보장은 없다.
실측 ETA와 반복 예약으로 새 run 입장을 결정한다. 다음 단계는 상위 최대 2개 EF anchor의
서버별 기본 ET 순서, late-convergence 근거가 있는 fresh H100/H150, 선택 recipe의 R2/R3다.
H100/H150은 저장된 미달 H50 가중치를 이어서 학습하지 않는다.

EF16–19와 추가 ET 선택에는 로컬 평가/진단 record ID에 연결된 명시적 근거가 필요하다.
추가 결정을 안 주어도 기본 큐는 계속 진행하며 조건 미충족 항목은 `NOT_TRIGGERED`로 남는다.
필요한 경우 캠페인 디렉터리 `decisions.json`에 `EDGERES48_DECISIONS_V1`의 append-only
decision 목록을 제공한다(`--decisions /path/to/decisions.json`도 가능).
기존 decision ID의 내용 변경/삭제, 다른 캠페인의 근거 또는 미완료 H50에서 임의 H150 시작은 거부한다.

## 저장·업로드

`work_dir/edgeres48/PANDA_S1_EDGERES_TEACHER48H_S123_20261013_v2/<server>/<dataset>/`

- admission/config/catalog, 매 update JSONL, 100-update 요약, full20 수치, LEAN 진단, 비용·실패·마감 기록.
- `records.jsonl` + immutable `outbox/`가 시트와 동일한 64열 payload를 보관한다.
- 별도 publisher가 새 데이터셋별 ablations에 RAW 수치를 append하고 readback 검증한다.
  지표 표시만 소수 4자리이며 로컬/시트 저장값의 정밀도는 보존한다.
- 고정 64열에 없는 진단은 BI에 domain/name/실제 수치를 inline JSON으로 저장한다.
  다른 지표 열을 재해석하거나 새 열을 만들지 않는다. 큰 진단은 여러 관측행으로 분할한다.
- 업로드 장애는 학습과 분리된다. pending은 지워지지 않으며 재시도한다.

```bash
python -m edgeres48 publish --server s1 --watch
python -m edgeres48 retry-artifacts --server s1  # 저장된 통과 snapshot의 정성 export 재시도만
docker logs -f panda-edgeres48-s1
```

실제 production 시트에는 테스트/가짜 지표를 쓰지 않는다. 다음 smoke는 disposable이며 업로드하지 않는다.

```bash
python -m edgeres48.smoke --server s1 --full-eval --output /tmp/edgeres48-smoke
python -m unittest discover -s edgeres48 -t . -q
python -m unittest discover -s panda_method -t . -q
python -m unittest teacher48.test_edge_training -q
```
