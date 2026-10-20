# s1/s3 Mature30 구현·전환 검증

## 결과

새 `panda_mature30` 패키지와 서버별 30시간 큐를 추가했다. 기존 실행 코드와 공유 metric 구현은 수정하지 않았다. 기존 s1 admission의 **코드 pin 전체와 현재 코드가 동일**함을 확인했다.

계획: `research_log/PANDA_S13_MatureAlign_Schedule_30H_2026-10-20_v1.md`.
Case 목록: 같은 이름의 `_Cases.csv`.

## 검증

| 검사 | 결과 |
|---|---|
| 호스트 CPU 전체 회귀 | 55/55 통과 |
| 실제 학습 Docker 이미지의 CPU 전체 회귀 | 55/55 통과 |
| 마지막 상태명 변경 후 runtime·publication 회귀 | 23/23 통과 |
| 호스트/실제 Docker 읽기 전용 data·donor·evaluator check | 양쪽 통과 |
| 기존 G8 full50K, 35개 checkpoint·원본 평가·SHA | 확인 |
| WV3 학습/validation 등록 | train 9,714 / val 1,080 |
| shell 구문·Python AST | 통과 |
| 기존 live code pin | 변경 없음 |

주요 회귀 범위:

- 새 G8 Stage1 초기 가중치·첫 optimizer update가 기존 G8과 정확히 일치.
- 고정 HALF의 입력·mask·conditional-q·loss가 기존 구현과 수치 일치.
- DOWN의 10K/30K 경계에서 완료 step 평가와 다음 학습 배율 구분.
- optimizer/LR/RNG/data stream/phase counter 완전 resume.
- phase별 exact calibration, 잘못된/누락된/변조된 bank 거부.
- cap 방향 보존, Teacher는 full-u 고정, MS/GT/output은 원본 좌표.
- Stage1 RRE의 고정 GT eligibility, 실패 벌점, receipt 집계 재검증.
- 같은 model state로 RR20+FR20 평가, FR GT 전달 금지, metric injection 결과의 실제 게시 금지.
- Sheet Stage1 RRE-selected, Stage2 통과/대표/FINAL50K 선택·중복 제거.
- Docker `/workspace` evidence를 호스트 publisher에서 안전하게 해석.
- 서버·캠페인 소유 행 외 변경 거부, 로컬 원자료 보존.
- 30h deadline 재시작 유지, R2 두 벌 전체 admission, 시작된 group skip 거부.

실제 GPU의 신규 학습 smoke는 기존 case와 GPU를 공유하지 않기 위해 실행하지 않았다. 이번 55개 테스트의 모델·optimizer·공식 평가 경로 검사는 CPU이며, full40 평가의 metric 함수는 synthetic TEST injection이다. 공식 evaluator 의존성·실제 데이터 등록은 별도 읽기 전용 check를 통과했다. 따라서 새 모델의 실제 공식 성능이나 학습 성공을 미리 주장하지 않는다.

단일 실제 G8 50K checkpoint의 CPU validation smoke는 RRE50≈0.094862 px, RRE90≈0.241318 px였다. 이는 전체35 후보를 비교한 최종 donor 선정값이 아니다.

## 발견·보완 사항

1. 호스트 publisher가 컨테이너 절대 경로를 그대로 여는 문제를 발견했다. 원본 문자열·SHA는 보존하고 I/O만 제한된 `/workspace → repo`로 변환하도록 수정했다. Stage1/Stage2 경로 회귀 통과.
2. 평가 중 phase counter 변조를 A/T/S tensor hash만으로 잡지 못할 수 있어 deployment manifest 불변성 검사도 추가했다.
3. s3의 donor budget admission 실패 시 후속 Student가 잘못 시작하지 않도록 dependency skip을 넣었다.
4. 새 Stage1 모델 목록의 HQNR-best가 실제 donor로 오해되지 않도록 `best_selected=null`, HQNR은 monitor로만 유지했다. 실제 best pointer는 `donors/*_best_val_rre.json`이다.
5. budget/명시적 skip이 있으면 전체 queue 완료로 표시하지 않도록 terminal 상태를 구분했다.

## s1 운영 상태

- 사용자 지시: **현재 G8_S1_BIC case 완료 후 전환**.
- 기존 trainer PID: `2637036`; 전환 watcher PID: `2002180` (등록 시점).
- 기존 run의 50K COMPLETED 상태, 최종 checkpoint·sidecar·공식 평가 저장을 확인한 뒤 해당 trainer에 SIGTERM으로 정상 종료를 요청한다. SIGKILL·container 제거는 하지 않는다.
- 신규 readiness receipt: `work_dir/panda_mature30/s1_validated_ready.json`.
- 검증 코드 225개 파일 SHA를 고정했다. 코드가 바뀌면 새 실행을 거부한다.
- 전환 로그: `work_dir/panda_mature30/transition-s1.log`.
- 신규 실행 대기 상태이다. **이미 새 학습이 시작됐다는 의미가 아니다.**

## s3 전달

신규 코드와 기존 `panda_band30` 공용 구현이 필요하다. 현재 저장소에서 두 패키지가 untracked이므로 새 패키지만 전달해서는 안 된다. 기반 Git HEAD는 `641032806efc051f4405a981520a31bf66201d07`이다.

해당 서버에 코드 전달 후 기존 GPU 작업이 끝나면:

```bash
bash scripts/run_panda_mature30_docker.sh s3
```

실제 s3 원격 실행·신규 Sheet 행 업로드·Git commit/push는 이번 검증 시점에 수행하지 않았다. s1 publisher는 새 캠페인 시작 시 함께 실행된다.
