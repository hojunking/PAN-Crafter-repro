# PANDA M12 실행 지시 자료

먼저 `EXPERIMENT_CASES_KR.md`를 읽는다. 서버별 전달문은 `server_handoffs/`에 있다.

- 이번 새 학습: WV3 10 case × s1/s3/s5 각4회 = **120 fresh Student**.
- 두 step: q cue/routing(72) → hard/soft 분해(48). baseline은 단계 간 공유한다.
- 이전 B01의 두 반복은 별도 보존한다. M12 primary는 새12 seed다.
- 실제 학습 runner/모델 weight/결과는 이 묶음에 없다. registry 생성기만 포함한다.
- `build_plan.py`는 파일 명세만 재생성한다. 서버·GPU·Google Sheets를 호출하지 않는다.
- `verification/test_plan.py`는 명세와 대수 계약24개를 검사한다. GPU/실제 모델 테스트가 아니다.
- Sheet의 자동 표시 대상은 `WV3-ablations`이며 논문용 `ablations`는 쓰기 금지다.
- `archive/`는 과거 문서 보관본이다. 반복 수/실행 범위는 현재 주 문서가 우선한다.
