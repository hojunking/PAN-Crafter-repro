# PANDA rebuttal 단계별 대비 계획 / s1·s3·s5

기준: 제출본 고정. 새 추가 통제실험은 각 case를 세 서버에서 각각 두 번 수행한다.

- `MASTER_PLAN_KR.md`: 변경된 운영 원칙과 전체 로드맵.
- `B01_RB01_RB02_DETAILED_KR.md`: 첫 두 step의 상세 정의.
- `planning/experiment_registry.json`: 24개 학습 run과48개 stress curve record 명세.
- `planning/training_runs_24.csv`, `stress_curves_48.csv`, `server_schedule.csv`: 실제 ID·seed·순서.
- `planning/shift_grid_v1.json`: 고정49점 shift 정의.
- `server_handoffs/`: s1/s3/s5 담당자용 짧은 인계.
- `archive/`: 이전 감사 원본을 수정 없이 보존. 기존 문서의 표 수정 권고를 현재 실행 명령으로 읽지 않는다.

**이번 bundle은 계획·보존본이며 학습 runner, checkpoint, 예측 결과, 원격 실행 완료 자료를 포함하지 않는다.**

원 PDF/XLSX는 archive manifest에 hash로 연결하고 중복 포함하지 않았다. 본 저장은 대화의 파일 산출물이며 원격 Drive/Project/GitHub 업로드는 수행하지 않았다.
