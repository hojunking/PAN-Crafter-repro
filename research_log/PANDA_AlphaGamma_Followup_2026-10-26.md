# α·γ 후속 실험 인계

사용자 승인: **기존 큐 유지 + 후속 실험 시간 추가**. β=0, Soft loss 제외.
기존 TUNE30 26run을 보존하고 HARD36으로 24run을 추가한다.
합계 등록 수는 s1 23run, s3 27run이며 실제 완료·budget skip은 별도로 집계한다.

실행 명세와 계수 표, 대조군, 시간 근거는 [HARD36 README](../panda_hard36/README.md)에 있다.
새 계수는 `(α,γ)=(0,0),(.25,0),(0,.5),(.125,.25),(.5,.5),(.25,1)`이다.
각 서버의 같은 R1/R2로 scale1 DIFF+UBIL W112 C30 LR1e-4 조건에서 비교한다.
기존 A0125/G025와 중심점은 중복 학습하지 않는다.

예상 추가 시간은 s1 약29.5h, s3 약19.2h이며 준비·변동 여유를 둔 상한은 각각36h다.
기존 30h 캠페인의 시계를 늘리거나 실행 중 소스를 수정하지 않는다.

s3 에이전트는 Git 반영 후 기존 Python 환경에서 다음을 실행한다.

```bash
python -m panda_hard36 smoke
python -m panda_hard36.queue --server s3 --enqueue
```

TUNE30 종료 전에는 대기 상태이며 GPU를 사용하지 않는다. 대기 상태 파일은
`work_dir/panda_hard36_launchers/s3/status.json`이다.
기존 실패나 중단은 자동 무시하지 않는다. 정상 완료 또는 검증된 budget skip 이후 시작한다.
기존 Sheet의 공동 통과·대표 선택·설정 표시 규칙은 유지한다.
