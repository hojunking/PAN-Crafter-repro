# PANDEP s2: Git pull 후 실행

이번 디렉터리는 `PANDEP_S2_SHARED_PLH_20261001_v4`의 **전달용 소스 스냅샷**입니다.
실제 학습은 기존 저장소 밖의 형제 디렉터리 `../pan_deploy_shared`에서 실행합니다.
기존 trainer/controller/uploader, 서버별 데이터, 서비스 계정 키는 수정하거나 복사하지 않습니다.

## s2 시작

기존 PAN-Crafter 저장소에서 실행합니다.

```bash
git pull --ff-only
python3 tools/pandep_s2.py start
```

두 번째 명령이 배포 파일의 SHA를 검증하고 독립 프로젝트를 자동 준비한 뒤,
기존 `gspread/account.json`을 읽기 전용으로 참조하여 독립 런처를 호출합니다.
`preflight → G0–G5/Q00 → 등록 → 학습`은 자동 진행됩니다. 수동 압축파일 전달은 필요 없습니다.
`git pull` 자체는 학습을 자동 시작하지 않습니다.

실제 `gspread/server.txt=s2`, 계획에 고정된 데이터와 Docker 이미지가 준비되어 있어야 합니다.
이미지 ID는 `sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887`입니다.
기존 GPU 작업은 종료하지 않으며, 자원이 사용 중이면 대기합니다. 검증 실패는 우회하지 않습니다.
컨테이너 ID 출력만으로 학습 성공을 판단하지 말고 상태와 로그를 확인하십시오.

## 제어

```bash
python3 tools/pandep_s2.py status
python3 tools/pandep_s2.py pause
python3 tools/pandep_s2.py resume
python3 tools/pandep_s2.py stop
python3 tools/pandep_s2.py sync-sheet
```

기존 독립 프로젝트가 있으면 등록된 소스가 정확히 같은 경우에만 재사용합니다.
다른 코드나 사용자 수정이 있으면 덮어쓰지 않고 중단하며, 기존 `work_dir`·checkpoint를 보존합니다.
중단된 실험은 `start`가 아닌 `resume`으로 이어갑니다. 재개 시 기존 microbatch를 유지합니다.

시트 대상은 기존 `배포용 모델`(gid1198707876)뿐입니다. 시트 장애는 로컬 outbox에 남깁니다.
원본 프로젝트의 상세 규약·라이선스는 `project/README.md`,
130개 CPU 테스트 증거는 `project/verification/local_cpu_tests.json`에 있습니다.
이 영수증은 실제 s2 GPU gate 통과나 학습 기동을 의미하지 않습니다.

`project/PLAN_PROVENANCE.json`은 앞선 독립 구현 당시의 이력입니다.
이번 Git 전달 변경은 사용자 요청에 따라 이 디렉터리와 전용 bootstrap만 추가한 것으로,
기존 실험 코드를 수정하거나 저장소 내부에서 새 학습을 실행하도록 바꾸지 않습니다.
