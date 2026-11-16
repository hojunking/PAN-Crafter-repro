# s2 QB: 고정 구조·α=0에서 donor 학습 시점 비교

사용자 승인 범위: s2에서 QB 대표 게시를 먼저 복구하고 신규 실험 5개를 실행한다.
Teacher 구조와 α를 늘리지 않는다. 기존 s2 에이전트가 이 패키지를 받아 실행한다.

## 실험 계약

같은 RGBN Stage1 `UNIFIED_QB_S2_S1_RGBN_FULL_R1_S261016301`의
40K/48K/50K Aligner+Teacher checkpoint를 각각 동결한다.
현재 C30의 donor32K를 비교 기준으로 두고, Student는 모두 새로 초기화해 30K 학습한다.
추가 Stage1 학습은 0개, Stage2 학습은 정확히 5개다.

| 순서 | donor | Student seed | 목적 |
|---|---:|---:|---|
| 1 | 48K | 261018101 | 사전 지정 중심 후보 |
| 2 | 40K | 261018101 | 48K보다 이른 donor와 짝비교 |
| 3 | 50K | 261018101 | 후기 donor의 개선·퇴행 확인 |
| 4 | 48K | 261018102 | 중심 후보 독립 반복 |
| 5 | 48K | 261018103 | 중심 후보 독립 반복 |

48K 반복은 결과를 보기 전에 정했다. 40K/50K 중 test 결과가 좋다는 이유로
반복 대상을 자동 교체하지 않는다. 기존 donor32K C30 R1/R2는 별도 비교군이다.

- Teacher/Student: W128, D123, UBIL. 기존 Stage1 λ_abs=4, λ_epsilon=.03 유지.
- Stage2: α=β=0, γ=.25, LR=1e-4, warmup100, fresh cosine30K, batch48/micro12.
- 입력 C: `[Pc, Pc−P, interp23 MS]`, QB 6채널. 학습·추론 모두 같은 정책.
- RGBN, Lanczos4, scale1, 공통 유효영역, q=0이면 rank0 유지.
- 동일 R1 초기 Student와 데이터·증강 stream. donor별 calibration을 다시 고정한다.
- 매 1K RR20 및 FR20 dual-reference 평가, 모든 평가 checkpoint 유지.

donor 교체는 Aligner와 Teacher의 가중치, corrected PAN, guidance/calibration을 함께 바꾼다.
따라서 이 실험의 인과 단위는 **같은 Stage1의 donor checkpoint**다.
결과를 Aligner 단독 효과나 Teacher 단독 효과로 부르지 않는다.
더 오래 학습한 donor가 더 낫다는 것은 검증할 가설이다.

## 판정·시트 게시

이번 s2 C 실험의 주평가는 **일반 PAN**이다: HQNR≥.9315 AND ERGAS≤3.500.
`QB-ablations` D:F에 일반 PAN FR, H:J에 같은 출력의 corrected PAN FR을 쓴다.
판정은 반올림 전 값, 표시는 네 자리다. donor가 달라지면 corrected 참조도 달라지므로
corrected HQNR만으로 donor 우열을 결론짓지 않는다.

평가가 있으면 조건 미달이어도 대표 1행을 게시한다.

1. HQNR 통과 후보가 있으면 ERGAS 최소 후보.
2. 없으면 일반 PAN HQNR 최고 후보를 `BEST_HQNR_FALLBACK`으로 게시.
3. 두 목표를 모두 통과한 checkpoint는 별도 상세 행도 게시.
4. 평가 자체가 없으면 상태만 표시한다. 빈 수치를 실패 결과로 해석하지 않는다.

기존 original/C30/R2 캠페인의 같은 Result_ID를 갱신한다. 원본 config·평가·weight는
읽기 전용이며, 복구 기록과 전체 trajectory CSV만 각 `publication/`에 쓴다.
모든 checkpoint의 원본·보정 지표는
`publication/qb_representative/all_evaluations.csv`에 남긴다.

s1의 `[Pc, MS]` corrected-only 학습·추론 및 corrected 주평가는 별도 진행한다.
이 패키지는 s2 C 입력 실험이다.

## s2 실행

신규 파일이 전달된 s2 저장소에서 실행한다. `gspread/server.txt`가 s2여야 한다.

```bash
python3 -m panda_qb_s2_donor_c.enqueue --check-only
python3 -m panda_qb_s2_donor_c.enqueue
docker logs --tail 80 panda-qb-s2-donor-c-v1
```

실행기는 기존 s2 INPUT3 original/C30/R2 종료와 runner lock 해제를 확인한다.
source Stage1 50K 완료, 40K/48K/50K weight·sidecar·평가 SHA,
데이터·evaluator, Sheet dry-run을 검증하고 **대표값 복구를 먼저 적용**한다.
그 다음 독립 컨테이너를 시작해 GPU가 비기를 기다리고 실제 CUDA preflight 후 5개를 순차 학습한다.
사전 검사 실패 시 원인을 수정하며 donor를 다른 checkpoint로 대체하지 않는다.

CUDA preflight는 donor별 C 입력/유한 loss/동일 출력 dual-FR,
R1 초기화 일치, model·optimizer·RNG·stream 저장 경계의 정확한 복원,
다음 CUDA update의 기존 UBIL 허용 오차(rtol1e-6/atol1e-8)를 검증한다.
평가 weight 전체 저장에 필요한 디스크 공간도 확인한다.

출력:
`work_dir/panda_qb_s2_donor_c/PANDA_QB_S2_DONOR_C_20261116_v1/s2/QB/`.
중단 컨테이너를 점검한 뒤 이어갈 때는 `--resume`을 사용한다.
이미 실행 중이면 같은 컨테이너를 재사용하고 중복 실행하지 않는다.

보류한 `panda_qb_donorgrid`/`panda_qb_lateval` 실행기를 재개하지 않는다.
기존 source의 읽기 전용 검증 함수를 재사용할 뿐이며, 새 큐 ID는 독립적이다.

## 결과를 받을 때

기존 donor32K C30 R1/R2와 신규 donor48K R1/R2를 seed별로 비교하고,
R1의 32K/40K/48K/50K 차이를 같은 30K endpoint 및 같은 대표 선택 규칙으로 정리한다.
48K의 3 seed는 통과 수와 평균·표준편차를 제시한다. 32K R3가 없으므로
3 seed 전체의 짝비교라고 표시하지 않는다.
대표, 30K endpoint, 전체 trajectory를 함께 보존한다.

공유 runtime의 기존 평가 JSON에는 legacy corrected-selection 설명이 남을 수 있다.
새 cfg의 `official_selection_reference=ORIGINAL_PAN`과 V5 publication metadata가
이 캠페인의 실제 게시·선정 규칙이다. 평가 수치/출력 자체는 수정하지 않는다.

production 이미지의 CPU 통합 테스트:

```bash
docker run --rm --network none --cpus 4 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  --workdir /workspace hojunqueen/pancrafter-env:torch2.4.0-cu118 \
  python -m unittest panda_qb_repfix.test_reporting panda_qb_s2_donor_c.test_campaign -v
```

s1에서 CPU 10개 검증을 통과했다. s2 실물 donor·GPU·live Sheet readback 검증은
s2 에이전트의 실행에서 완료해야 한다.
