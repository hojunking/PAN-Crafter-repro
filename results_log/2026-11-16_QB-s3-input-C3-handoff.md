# s3 QB Stage1 → input C ×3 실행 인계

사용자 확인: s3의 QB Stage1 이후 시도 C를3회 진행한다.
이는 **기존 완료 Stage1 재사용 + 신규 Student3회**이며 Stage1을 다시 학습하지 않는다.

1. s3 `QBS3_QB_S3_S1_RGBN_FULL_R1_S261016303`의50K 학습 완료를 확인한다.
2. 이전 s3 실험에 사용했던34K weight/평가/선정 receipt의 SHA를 확인한다.
3. 동일 Aligner/Teacher를 동결하고 C R1→R2→R3를 각각 새30K 학습한다.

입력C = corrected PAN + (corrected PAN−원본 PAN) + interp23 MS.
W128/D123/UBIL, RGBN, Lanczos4, 공통 유효영역, scale1,
α=β=0/γ=.25, q=0 예외처리, LR1e-4, cosine30K, batch48/micro12 유지.
Student seed261018101/261018102/261018103. s2 donor 이동 없음.

사용자 push 후 s3 에이전트 실행:

```bash
git pull --ff-only
python3 -m panda_qb_s3_input_c3.enqueue --check-only
python3 -m panda_qb_s3_input_c3.enqueue
```

현재 GF2 INPUT3 R2가 완료됐다면 바로 시작한다. 남아 있다면 완료 후 이어지며,
기존 GF2 결과와 보류된 QB 큐는 보존한다. 기존 `panda_qb_s3_parallel.enqueue`를
재실행하면 오래된 큐가 재개될 수 있으므로 이번에는 위 신규 명령을 사용한다.

Sheet는 `QB-ablations`, 목표는 원본 PAN HQNR≥.9315 및 ERGAS≤3.500.
기존 PAN HQNR 하한 통과 중 ERGAS 최소 대표 + 전체 목표 통과 checkpoint 행을 게시한다.
corrected PAN FR도 같은 행H:J에 기록한다. 표시4자리, 판정은 원시값.

설정은 `config/panda_qb_s3_input_c3/campaign.json`, 실행 목록은 같은 폴더의
`cases_s3.csv`, donor 고정값은 `donor.json`에 있다.
상세 설명: [패키지 README](../panda_qb_s3_input_c3/README.md).

검증: production Docker CPU에서6개 테스트 통과. s3 형식의 합성 Stage1 checkpoint
저장/로드, C 입력6채널, 실제 Student3개 seed 학습, A/T 동결, 재개 일치,
동일 guidance/mask와 서로 다른 초기화, 잘못된 donor/SHA 거부, QB 데이터 마운트,
QB Sheet/원본선정/동일checkpoint 보정FR, GF2 종료 대기를 확인했다.

실제 s3 donor 파일과 GPU는 이 s1 세션에서 접근할 수 없다. 해당 서버의 첫 실행에서
고정 파일 SHA·데이터·Sheet 및 실제 CUDA preflight를 확인한 뒤 학습을 시작한다.
