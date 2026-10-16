# PANDA s1/s3 40h

독립 실행 namespace. 기존 donor24 실행 코드와 시트 행을 바꾸지 않는다.

- s1: Direct/S2D Aligner, A1/A3/A5 조건별 paired R1/R2; S2D half-LR 예비.
- s3: 하나의 기존 Direct/MASK2 donor, G→E→ES→Q→Q_FULL 각 R1/R2; Q_WEAK 예비.
- 각 서버 40시간, H50 완주 예상시간으로 그룹 admission, 30분 closeout.
- 학습 MASK2, 공식 RR20/FR20 평가는 원본 reference·비마스킹.
- checkpoint: HQNR≥.959이면 ERGAS 최소. SCC는 모니터링만.
- sheet: 기존 .960/2.045 qualified steps 또는 best-HQNR fallback.
- Stage2 W112D123/P0: LPAN/HPAN 없음. A/T 동결, fresh Student만 업데이트.

계획: [40h case 표](../research_log/PANDA_S13_40H_S2D_Stage2_2026-10-16.md).

```bash
# push/pull은 실행을 시작하지 않는다. 각 서버에서 한 번 명시 실행한다.
bash scripts/run_panda40_docker.sh s1 --replace-donor24
bash scripts/run_panda40_docker.sh s3 --replace-donor24
```

launcher는 기존 업로더용 Python을 우선 사용하고, 필요하면 현재 conda 설치의
`envs/pancrafter/bin/python`을 자동 확인한다. 다른 설치면 `PAN_HOST_PYTHON`으로
지정한다. 새 패키지 설치나 자격증명 복사는 하지 않는다. 기존 donor24의 정확한
서버별 컨테이너만 종료하며 삭제하지 않는다. 다른 GPU 작업이 있으면 시작을 거부한다.

```bash
python -m panda40 catalog --server s1 --csv
python -m panda40 check --server s3  # read-only; training Python
python -m panda40 smoke             # CPU synthetic tests, no sheet writes
docker logs -f panda40-wv3-s1
```

출력: `work_dir/panda40/PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1/<server>/WV3/`.
`campaign_state.json`, `runs/*/attempt_*/models/selection.json`, 전체 평가 JSON,
immutable outbox와 `sheet_publish_panda40/last_poll.json`으로 실행/업로드를 확인한다.

실행 중 코드 변경은 fail-closed한다. 중단 후 같은 캠페인을 다시 실행해도
40시간 시계를 초기화하지 않는다. 완료된 run은 건너뛰며 미완료 run은 새 attempt로
fresh 재시작한다. 이전 모델/optimizer/RNG/stream 전체 파일은 보존하고, 저수준의
명시적 전체-state 복원 API도 제공한다. 서로 다른 코드·donor·mask의 체크포인트를
자동 이어붙이지 않는다. 비용·평가·업로드 실패를 완료 성공으로 표시하지 않는다.
