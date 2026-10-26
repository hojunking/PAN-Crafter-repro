# QB / GF2 Stage 1 + 5 case 실행 인계

2026-10-26 사용자 승인: s1은 QB, s3는 GF2를 담당한다. **센서별 fresh Stage 1 50K + Student 다섯 case**를 실행하고, 완료 즉시 해당 서버의 남은 WV3 TUNE30과 HARD36을 순서대로 재개한다. 현재 WV3 run만 먼저 완료한다.

전체 설정, 데이터 경로 정책, 게시 규칙과 운영 절차는 [SENSOR5 README](../panda_sensor5/README.md)에 있다.

| 서버 | 공동 목표 | Stage 2 case |
|---|---|---|
| s1 / QB | HQNR ≥ .9315, ERGAS ≤ 3.5 | BASE, LR050, LR200, LONG 50K, HARD4 |
| s3 / GF2 | HQNR ≥ .9645, ERGAS ≤ .548 | BASE, LR050, LR200, LONG 60K, HARD4 |

기준은 WV3 유력 설정 scale1 DIFF+UBIL W112/D123 C30 LR1e-4 α.25/β0/γ.5다. LR을 .5×/2×, 추가 Hard 계수를 함께 4×로 넓혀 확인한다. 모든 Student는 같은 seed의 fresh 학습이며 센서별 validation RRE로 선택한 동일 donor를 공유한다. 4밴드 mean-GT Stage 1, 6채널 Student, 센서별 정규화·공식 Q4/MTF 평가를 적용한다.

새 센서 큐는 전부 실행하며 시간 상한을 두지 않는다. WV3의 남은 등록 case도 원래 예산 때문에 다시 skip하지 않도록 명시적 continuation override를 기록한다. 기존 원본 설정·시작 시각·예산·체크포인트는 보존한다.

s3 에이전트 실행:

```bash
# 이 commit을 반영한 뒤 기존 학습 Python 환경에서 실행
export PYTHONDONTWRITEBYTECODE=1
python -m panda_sensor5 check --server s3
python -m panda_sensor5 smoke
python -m panda_sensor5.queue --server s3 --enqueue
```

이 명령은 현재 run 저장·복원·완료, GF2 Stage 1, GF2 다섯 Student, 남은 WV3 재개까지 연결한다. 새 GF2 학습을 먼저 별도로 실행하거나 기존 HARD36 watcher를 다시 enqueue하지 않는다. 새 controller가 순서를 소유한다. `gspread/server.txt=s3`, 기존 자격정보·데이터·Docker 이미지·WV3 work_dir가 필요하며 Git은 이를 운반하지 않는다.

실행 직전에 기존 HARD36이 본 학습 컨테이너 생성 전의 cost/preflight 준비 단계인지 확인한다. 준비 subprocess만 실행 중인 경우에는 이를 중복 실행하거나 강제로 종료하지 말고 준비가 끝난 뒤 enqueue한다. 중간 실패한 준비 상태는 해당 로그부터 확인한다. 일반적인 TUNE30/HARD36 본 학습 또는 정상 종료 상태는 새 큐가 직접 처리한다.

확인할 경로:

- 전체 순서: `work_dir/panda_sensor5_launchers/s3/status.json`
- 학습: `work_dir/panda_sensor5/PANDA_S13_SENSOR5_STAGE12_20261026_v1/s3/GF2/campaign_state.json`
- 컨테이너: `panda-sensor5-stage12-gf2-s3-v1`
- 게시: 같은 캠페인 디렉터리의 `sheet_publish_panda_sensor5/last_poll.json`

Sheet는 센서 목표의 공동 통과 step 전부와 run별 대표를 유지한다. 대표는 HQNR 목표 충족 중 최소 ERGAS, 없으면 최대 HQNR다. Stage 1은 validation RRE 선택 모델을 게시한다. run 이름과 메모에 새 접근·하이퍼파라미터를 함께 남긴다.
