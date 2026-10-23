# NEXT30 s3 인계 — 다음 미시작 case부터 Lanczos4

사용자 결정: warp는 **Lanczos4 하나**로 줄이고 다음 미시작 실험부터 학습·추론에 적용한다. Full-u 차영상 Student를 추가한다. 밴드 가중 loss·rotation loss는 제외한다. 이전 Sampling30의 bicubic 구조 큐는 이 인계로 대체한다.

새 캠페인은 `PANDA_S13_NEXT_HARDONLY_20261023_v1`, prefix `PNX30`이다. [현재·후속 전체표](2026-10-23_NEXT30_current_and_queue.md), [방법/실행 README](../panda_next30/README.md), [s3 CSV](../panda_next30/cases_s3.csv)를 함께 본다.

**구현 commit: `fa1e85b` — Apply Lanczos4 to next queued runs and add full-shift PAN difference inputs.** 이 commit을 포함하는 본 인계 문서의 최신 commit까지 받는다. 선행 schedule/sampling 구현도 이 commit의 조상에 포함된다.

## 실행

이 문서의 Git commit을 받아 기존 미커밋 파일·credential·work_dir를 보존한다. 현재 trainer의 pin 대상 소스를 수정하지 않는다. root가 2026-10-23 08:43:41 UTC에 읽은 Sheet에는 S020C30 R1 대표16K(게시 시각 08:35:27 UTC)가 있었으며 실제 진행 상태는 s3 현지 파일로 확인해야 한다.

```bash
python -m panda_next30 catalog --server s3 --csv
python -m panda_next30 smoke
bash scripts/start_panda_next30_transition.sh s3
```

기존 torch/google-auth/requests Python을 사용한다. 필요하면 `PAN_HOST_PYTHON`, `PAN_NEXT30_IMAGE`, `PANCRAFTER_DLPAN`을 지정한다. `gspread/server.txt`는s3이어야 한다. 두 서버 모두WV3이고 historical mount helper의 s1=WV3는 물리 서버와 별개다.

이 명령은 현재 학습을 중단하지 않는다. 정확히 확인된 Sampling30 대기 watcher만 retire하고 미시작 Schedule30 group을 명시적으로 새 큐에 넘긴다. 진행 중 case를 완료한 뒤 GPU idle에서 비용 측정·새 calibration·CUDA 검증을 거쳐 NEXT30을 실행한다. Sampling30 trainer/preflight가 이미 시작된 상태이면 자동으로 덮어쓰지 않는다. Schedule30 자체가 없으면 시작 대기 상태이므로 현지 이력을 먼저 확인한다.

## 조건과 순서

등록 상한17개, 정확한 순서는 CSV/전체표를 따른다. 첫 세 case는 `S020C30 R1 → S100C30 R1 → S100DIFFC30 R1`이다. 기존 started/completed schedule과 대응하는 새 case는 증거를 남기고 중복을 생략한다. 단 S020C30 R1/R2는 새L4 구조 대조군으로 항상 실행한다. 기존Sampling30의6개bicubic구조case는 중복실행하지 않는다.

모든 새 Student는 fresh다. Frozen Teacher는 같은서버 matureRRE donor와원래bicubic경로를 유지한다. Student L4에 맞춰 qcalibration을 다시 계산한다. 기본 scale=.2, 30Kcosine/warmup100/LR1e-4→0, Hardα=.25/γ=.5/β=0이다. H050는α=.125/γ=.25, LR150은1.5e-4, BASE/H050후속50K는CSV에명시돼있다.

`S100DIFFC30`은 scale=1, `[PANc,PANc-PAN,MS base8]`, 10채널, +1,008개가중치, 추가열0초기화, 8밴드출력, 기존GT Hard다. `S100C30`은 동일full-u9채널대조군이다. s1전체train의8tap support검증은통과했지만 s3의실제calibration/preflight도별도로통과해야한다. support위반시shift를축소하거나fallback하지않는다.

## 게시·확인

Approach/Model에 L4,scale,sampler,9/10ch,Hard계수,LR,horizon을 표시한다. 기존 HQNR≥.9585·ERGAS≤2.040통과전체+대표/fallback규칙을 유지한다. 기존결과와Stage1RRE행을보존한다.

전환기록은 `work_dir/panda_next30_launchers/s3/transition_receipt.json`이다. 실행후 campaign state의 `migration_effective_queue`, `preflight/passed.json`, 첫trainupdate, Sheet`readback_verified`를 확인해 보고한다. 사용commit과실제미시작/완료case목록도남긴다. 중단·source/donor/state불일치는 증거를덮어쓰지않고원인을확인한다.

## 예상 시간과 s1 검증 상태

s3 전체 17개는 535,000 update, 공식 평가 511회다. 기존 s3 실측 약 0.152초/update를 바탕으로 새 warp 비용을 소폭 더하면 기준 약 24.6시간이며, 준비·I/O·구조별 속도 차이를 포함해 **25–30시간**을 계획한다. 현재 bicubic case 마무리와 인계 지연은 별도다. 기존 대응 case 진행에 따른 중복 생략이 있으면 실제 새 큐는 더 짧아질 수 있다.

s1은 2026-10-23 17:28 KST에 실제 CUDA preflight를 통과하고 NEXT30을 시작했다. 17:50 KST 전후 상태 파일은 첫 L4 S020C30 R1의 4K checkpoint를 기록했다. s1 18개 전체는 약 43–50시간이며 두 서버를 병행하면 전체 약 2일이다. s1 검증이 s3 현지 검증을 대신하지 않는다. [계산·측정 근거](2026-10-23_NEXT30_runtime_estimate.md)를 참조한다.
