# 현재 실행과 후속 전체 큐 — NEXT30

기록 시점: 2026-10-23 17:20 KST. s1은 현지 campaign state, s3는 Sheet 게시 기록으로 구분해 적었다.

**17:50 KST 전후 업데이트:** 아래 현재 run 표는 17:20 기록이다. 이후 s1 S010F25C35도 35K 완료됐고, 17:28에 CUDA preflight를 통과해 NEXT30을 시작했다. 첫 Lanczos4 S020C30 R1은 상태 파일 4K까지 진행됐다. s3는 17:43 Sheet 읽기에서 기존 S020C30 R1 대표16K(17:35 게시)가 확인됐다. s3 live 진행 상황은 현지 확인이 필요하다. 전체 새 큐 예상은 **s1 43–50시간 / s3 25–30시간**, 병행 약 2일이다. [시간 계산과 근거](2026-10-23_NEXT30_runtime_estimate.md).

**결정:** 다음 미시작 실험부터 Lanczos-4 하나를 사용한다. 현재 bicubic run은 완료하고, 미시작 schedule 및 구조 실험은 NEXT30으로 이관한다. 기존 학습 checkpoint에 추론 warp만 교체하지 않는다. 새 full-u 차영상 입력을 추가하고, 밴드 가중 loss·rotation loss는 제외한다.

## 현재/이미 시작한 run

| 서버 | Case | 확인 상태 | warp |
|---|---|---|---|
|s1|S020F25C35 R1|COMPLETED, 상태 파일 35,000 update|bicubic|
|s1|S010F25C35 R1|RUNNING, 상태 파일 33,000 update|bicubic|
|s3|S020F25C35 R1|Sheet에서 COMPLETED 확인, 게시 대표29K|bicubic|
|s3|S020C30 R1|08:03:56 UTC Sheet 최신 관측 당시 대표5K 게시; live step은 현지 확인 필요|bicubic|

s3 위 상태는 [WV3-ablations](https://docs.google.com/spreadsheets/d/1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE/edit#gid=261001010) rows181/183의 게시 관측이다. 대표 checkpoint step은 실제 진행 step과 다를 수 있다. s3 실행 코드는 Git으로 전달하며 현지 에이전트가 전환 명령을 실행해야 적용된다.

s1에서는 현재 case 뒤의 미시작 schedule8개를 이관하도록 pending_control을 등록했고, 기존 Sampling30 대기 watcher만 retire했다. trainer에는 종료 신호를 보내지 않았다. 기존 Stage1 donor와 이전 결과·checkpoint를 유지한다.

## 새 case 전체

공통: Student warp=Lanczos4, frozen Teacher warp=원래 bicubic, fresh Student, batch48, W112D123, interp23, AdamW, warmup100, cosine floor0, β=0. 표에서 R1/R2는 각각 seed261018101/261018102다. “—”는 해당 서버에 등록하지 않은 case다.

| Case | scale | α / γ | peak LR | horizon | 구조·입력 | s1 | s3 |
|---|---:|---|---:|---:|---|---|---|
|S010C30|0.1|0.25 / 0.5|0.0001|30K|BASE / PANc+MS9ch|R1/R2|—|
|S020C30|0.2|0.25 / 0.5|0.0001|30K|BASE / PANc+MS9ch|R1/R2|R1/R2|
|S020C35|0.2|0.25 / 0.5|0.0001|35K|BASE / PANc+MS9ch|R1|—|
|S020C30LR150|0.2|0.25 / 0.5|0.00015|30K|BASE / PANc+MS9ch|R1|R1|
|S020C30H050|0.2|0.125 / 0.25|0.0001|30K|BASE / PANc+MS9ch|—|R1/R2|
|S020C35H050|0.2|0.125 / 0.25|0.0001|35K|BASE / PANc+MS9ch|—|R1|
|S010BASE|0.1|0.25 / 0.5|0.0001|50K|BASE / PANc+MS9ch|R2|—|
|S020BASE|0.2|0.25 / 0.5|0.0001|50K|BASE / PANc+MS9ch|R2|—|
|S020H050|0.2|0.125 / 0.25|0.0001|50K|BASE / PANc+MS9ch|—|R2|
|DAVG|0.2|0.25 / 0.5|0.0001|30K|DAVG / PANc+MS9ch|R1/R2|R1/R2|
|UBIL|0.2|0.25 / 0.5|0.0001|30K|UBIL / PANc+MS9ch|R1/R2|R1/R2|
|DAVG_UBIL|0.2|0.25 / 0.5|0.0001|30K|DAVG_UBIL / PANc+MS9ch|R1/R2|R1/R2|
|S100C30|1|0.25 / 0.5|0.0001|30K|BASE / PANc+MS9ch|R1/R2|R1/R2|
|S100DIFFC30|1|0.25 / 0.5|0.0001|30K|BASE / PANc+delta+MS10ch|R1/R2|R1/R2|

DAVG는 두 down을 평균 pooling2+conv3로, UBIL은 두 up을 bilinear×2+conv3로 바꾼다. DAVG_UBIL은 둘 다 바꾼다. 네트워크 전체의 회전 등변성을 보장하는 구조는 아니다.

`S100DIFFC30`은 예측 shift 전체를 적용하는 scale=1.0이다. 입력 `[PANc, PANc-PAN, MS base8]`, 출력8밴드, 기존GT Hard다. 추가1,008개 첫 Conv 가중치는0초기화하고 나머지는 동일 seed의9채널 모델과 같게 초기화한다. `S100C30`은 차영상만 없는 full-u 대조군이다. 따라서 scale 효과와 차영상 효과를 분리할 수 있다.

## 실행 순서

| 순서 | s1 | s3 |
|---:|---|---|
|1|S020C30 R1|S020C30 R1|
|2|S100C30 R1|S100C30 R1|
|3|S100DIFFC30 R1|S100DIFFC30 R1|
|4|S010C30 R1|S020C30H050 R1|
|5|S020C35 R1|S020C35H050 R1|
|6|S020C30LR150 R1|S020C30LR150 R1|
|7|DAVG R1|DAVG R1|
|8|UBIL R1|UBIL R1|
|9|DAVG_UBIL R1|DAVG_UBIL R1|
|10|S020C30 R2|S020C30 R2|
|11|S010C30 R2|S020C30H050 R2|
|12|S020BASE R2|S020H050 R2|
|13|S010BASE R2|S100C30 R2|
|14|S100C30 R2|S100DIFFC30 R2|
|15|S100DIFFC30 R2|DAVG R2|
|16|DAVG R2|UBIL R2|
|17|UBIL R2|DAVG_UBIL R2|
|18|DAVG_UBIL R2|—|

**등록 상한 s1 18개 / s3 17개, 총 시간 제한 없음.** 기존 schedule에서 이미 시작한 fresh case는 마친 뒤 대응하는 새 case를 중복 생략한다. 예외로 S020C30 R1/R2는 새 Lanczos4 구조 대조군이므로 항상 실행한다. s3는 Git 적용 시점에 더 진행됐을 수 있으므로 실제 큐는 전환 receipt와 campaign state의 `migration_effective_queue`로 확정된다.

기존 Sampling30의 서버별6개 bicubic 구조 case는 NEXT30 DAVG/UBIL/DAVG_UBIL R1/R2로 대체하며 두 번 실행하지 않는다. 별도 FFT/Lanczos6/Kaiser warp sweep도 등록하지 않는다. 기존 fork25K→35K run은 bicubic 계보로 마무리한다.

## 검증과 게시

s1 실제 donor로 전체 RR-train9,714개를 CPU 검사한 결과 full-u Lanczos4 support 위반은0개였다. raw dy[-1.2754,1.4048], dx[-1.0367,.8842]px였고,64표본×4기존 C4 view도 통과했다. [검증 JSON](assets/2026-10-23_next30_support_cpu.json). 새 CUDA calibration/update/resume/native inference 검증도 17:28에 통과했다. 첫 L4 학습의 초반 평가가 생성되고 있으며 완주 결과는 아직 없다.

새 Sheet Approach에는 `L4 | scale/full-u | BASE/DAVG/UBIL | 9/10ch | Hard α/γ | LR | cosine horizon`을 표시하고 Model/inputs에도 입력 구성을 적는다. 원래 bicubic 행을 L4로 바꾸어 표시하지 않는다. 실제 검증된 배포 비용도 입력·구조별로 분리한다.

게시 조건은 full precision 동일 상태 HQNR≥.9585 AND ERGAS≤2.040 통과 전체 + 대표 하나다. 대표는 HQNR≥.9585 중 최소 ERGAS, 없으면 최고 HQNR이다. 같은 checkpoint는 중복하지 않고 Stage1 RRE 행과 기존 결과를 유지한다.

[구현·실행 설명](../panda_next30/README.md) · [s3 인계](2026-10-23_NEXT30_s3_handoff.md) · [s1 CSV](../panda_next30/cases_s1.csv) · [s3 CSV](../panda_next30/cases_s3.csv)
