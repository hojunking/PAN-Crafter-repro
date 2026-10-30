# GF2 목표 달성 종료와 모델 보존

이 패키지는 기존 학습 source를 수정하지 않는 s3 전용 제어 도구다.
기준 모델은 `GF2P_C60LR200 R2 @47K`이며 HQNR≥.9645, ERGAS≤.522, PSNR≥45.08을 같은 상태에서 달성한다.
43K H/E 대표는 PSNR 목표 미달이므로 최종 모델로 대체하지 않는다.

사용자가 GF2 종료와 s3 QB 전환을 요청했으므로, s3 에이전트는 pull 후 아래를 실행한다.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s panda_gf2_closeout/tests -v
PYTHONDONTWRITEBYTECODE=1 python -m panda_gf2_closeout plan
PYTHONDONTWRITEBYTECODE=1 python -m panda_gf2_closeout apply --wait
PYTHONDONTWRITEBYTECODE=1 python -m panda_gf2_closeout verify
```

`plan`은 읽기 전용이다. `apply`는 다음을 수행한다.

1. 알려진 47K 파일 SHA, 평가/sidecar/config, 정확한 세 지표와 같은 모델 상태를 검증한다.
2. **로컬 모든 세 목표 달성 checkpoint**와 관련 sidecar/evaluation/config, frozen donor, 선택 receipt, calibration NPZ를 별도 archive에 복사·hash 검증한다. 원본은 삭제하지 않는다.
3. PID·generation·명령·cwd가 일치하는 GF2 enqueue를 먼저 SIGTERM으로 종료한다. 재시작 방지 controller lock을 획득한 뒤 정확한 Docker ID/image/command/output mount의 GF2 trainer에 SIGTERM을 보낸다.
4. 15초 간격으로 안전 checkpoint 저장·종료를 기다린다. SIGKILL은 사용하지 않는다. 중간에 도구가 종료돼도 같은 `apply --wait`로 durable intent부터 이어간다. 소스나 identity가 바뀌면 멈춰 원인을 조사한다.
5. 종료 후 새로 생긴 목표 달성 모델까지 추가 보관한다. 완료 / 안전 중단 / 미시작 제외를 별도로 기록한다. 기존 campaign를 가짜 `COMPLETED`로 바꾸지 않는다.
6. 기존 GF2 publisher로 최종 Sheet readback을 검증하고 `retirement.json`을 생성한다. 통신 실패 시 QB 전환 gate는 닫힌 상태로 남는다.

산출물: `work_dir/panda_gf2_closeout/s3/`.
재개/최종 모델 원본과 archive 모두 유지한다. 최종 종료 후 이전 GF2 enqueue나 trainer를 다시 실행하지 않는다.
기존 queue의 등록 8개를 전부 완료한 것으로 표시하지 않는다.

후속 QB의 admission API는 `panda_gf2_closeout.verify_retirement(root)`이다.
호스트 launcher는 기본 `check_runtime=True`로 Docker가 계속 정지해 있는지 확인한다.
Docker 내부 읽기 전용 검사에서는 `check_runtime=False`로 이미 인증된 종료 receipt·archive·원본 campaign 상태·Sheet readback hash를 검증한다.
두 모드가 반환하는 receipt는 동일하며 시간이나 새 PID를 추가하지 않는다.

실제 s3 파일/프로세스 확인은 s3 실행 시 이루어진다. s1에서는 server marker 검사로 실행을 거부한다.
이전 focus/GF2/WV3 controller의 완료 근거와 이전 queue watcher 부재도 확인한다. 이전 watcher가 살아 있으면 무작정 종료하지 않고 원인을 조사하도록 거부한다.
학습 환경의 Python을 사용한다. 경로·권한·출처 검사가 실패하면 파일을 지우거나 receipt를 덮어써 우회하지 않는다.
