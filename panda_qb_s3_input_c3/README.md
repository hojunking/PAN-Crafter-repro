# s3 QB: 기존 Stage 1 이후 C를 3회 반복

신규 Stage 1 학습은 **0회**, 신규 Stage 2는 **3회**다.
s3에서 이미 50K까지 학습 완료한 QB Aligner/Teacher의 기존 선정34K를 고정하여
시도 C를 R1→R2→R3 순서로 새로 학습한다. s2의 weight를 가져오지 않는다.

## 고정 donor

- run: `QBS3_QB_S3_S1_RGBN_FULL_R1_S261016303`
- 선택 checkpoint: **34,000** (기존 s3 PARALLEL P1 등이 사용한 동일 모델)
- 원본 학습 완료 조건: 해당 Stage1 run이 **50,000 / COMPLETED**
- source root: `work_dir/panda_qb_s3_parallel/PANDA_QB_S3_PARALLEL_20261114_v1/s3/QB`
- SHA와 선택 receipt: `config/panda_qb_s3_input_c3/donor.json`

최신 목표로 Stage1을 다시 고르지 않는다. 원래 선택된 weight/평가/receipt의 SHA와
RGBN·Lanczos4·W128/UBIL·공통 유효영역 설정을 검사한다. 기존 calibration이 있으면
같은 binding의 캐시를 읽고, 없으면 새 캠페인 폴더에서 동일 full-train 절차로 계산한다.
기존 donor weight/평가/선정 receipt는 읽기 전용이다.

## 실행 case

| 순서 | Case | Student seed | Step | LR | α / β / γ |
|---|---|---:|---:|---:|---|
|1|C R1|261018101|30K|1e-4|0 / 0 / .25|
|2|C R2|261018102|30K|1e-4|0 / 0 / .25|
|3|C R3|261018103|30K|1e-4|0 / 0 / .25|

Student 입력은 `[corrected PAN, corrected PAN−원본 PAN, interp23 MS]`,
QB 기준6채널이며 출력은4밴드다. W128/D123/UBIL, Aligner scale1,
RGBN `[2,1,0,3]`, Lanczos4, 공통 유효영역을 유지한다.
손실은 GT Hard에 `w=1+.25*d_e*r_q`, q=0이면 r_q=0.
batch48/micro12, AdamW WD.01, FP32, warmup100 + cosine30K→LR0.
새 Student 초기화/데이터 RNG만 반복별로 변경하며 Aligner/Teacher는 동결한다.
매1K RR20/FR20 평가와 weight 보관, 성능 하한에 따른 조기 중단은 없다.

## s3 실행

사용자가 push한 뒤 s3에서 전체 commit을 pull하고 실행한다.

```bash
git pull --ff-only
python3 -m panda_qb_s3_input_c3.enqueue --check-only
python3 -m panda_qb_s3_input_c3.enqueue
```

중단된 신규 C3 컨테이너 재개는 `--resume`. 같은 명령을 반복해도 실행을 중복 생성하지 않는다.
GF2 INPUT3 R2가 남아 있으면 완료와 최종 게시를 기다린다. 해당 큐가 중단돼 있으면
자기 launcher로 먼저 재개한다. 완료 후 새 QB C3 큐가 시작된다.
기존 GF2 결과와 보류 중 QB sweep은 그대로 보존하고, 오래된 QB P5/donorgrid는 재개하지 않는다.

실행 전에 로컬 QB 데이터/고정 donor/환경/Sheet를 검사하고, 실제 s3 CUDA preflight를 통과해야
학습을 시작한다. 컨테이너의 데이터 마운트 선택에 쓰는 `s2`는 QB 데이터 registry 키일 뿐이며,
server marker/결과/원본 weight provenance는 전부 **s3**다.

- 컨테이너: `panda-qb-s3-input-c3-v1`
- 캠페인: `PANDA_QB_S3_INPUT_C3_20261116_v1`
- 결과: `work_dir/panda_qb_s3_input_c3/<campaign>/s3/QB/`
- 상태: `campaign_state.json`, `preflight/verified.json`, `publication/original_pan/last_poll.json`

## Sheet

`QB-ablations`에 `QB-INPUT3-C30-C-CORRECTED-DIFF s3 R1/R2/R3 (RGBN)`으로 별도 게시한다.
**원본 PAN HQNR≥.9315, ERGAS≤3.500**이 동시 달성 목표이며 QB에는 PSNR 하한이 없다.
대표는 HQNR 하한을 충족한 후보 중 ERGAS 최소(동률: PSNR↑, HQNR↑, 이른step).
대표의 ERGAS 미달은 표시하고, 그 외 상세 행은 두 목표 동시 통과 시에만 게시한다.
HQNR 하한을 통과하지 못하면 수치 없는 상태 행을 유지한다.
원본 FR은 D:F, corrected PAN 보조 FR은 H:J에 같은 checkpoint로 기록한다.
표시4자리, 판정은 반올림 전 값. 기존 결과 행을 덮어쓰지 않는다.

세 Student seed 결과는 모두 보관한다. s2와 비교할 때는 donor와 calibration이 다르므로
서버별 결과를 구분하며, 동일 donor의 Student 반복수로 합산하지 않는다.
