# GF2 결과 시트 업로더 보완 — 2026-09-21

대상: `pan-cvpr27`의 GF2 서버 탭. WV3-s1을 열 배치·표시의 기준으로
읽어 비교했다. 평가 수식이나 checkpoint 선택을 변경하는 작업은 아니다.

## 확인된 원인

- G20/QG40 row payload에 `Date`가 없었다. 학습 시각은 로컬에 있지만
  시트에는 전달되지 않았다.
- 비용·시간 수치에 표시 형식이 없어 긴 소수점이 그대로 노출됐다.
- GF2-s4 생성은 `Run` 헤더 하나만 만들고 나머지를 payload 순서로 붙여,
  실제 metric보다 source/calibration metadata가 앞에 오는 구조였다.
- 재업로드 시 Notes를 자동 생성문으로 덮어써 수동 메모를 잃을 수 있었다.

## 변경

- 실제 완료 시각 우선의 KST Date, seed/architecture/input/selection,
  UTC 시작·완료 기록 추가. 업로드일이나 run 이름을 날짜로 추정하지 않는다.
- Train(h)는 optimizer 구간 합계로 유지. Wall(h)는 시작부터 trainer 완료
  기록까지의 별도 경과 시간이다. 복구 시각을 학습 종료 시각으로 쓰지 않는다.
- 숫자 정밀도는 유지하고 표시만 제한한다. metric/Params 4자리,
  FLOPs/Mem 1자리, Infer/시간 2자리, seed/step 정수.
- 기존 Notes, R0 예외 기록, 수식·보호·검증 셀을 보존한다.
- 신규 GF2-s4에는 WV3와 유사한 공통 헤더를 생성한다. 기존 탭 정렬은
  자기 서버의 uploader 잠금 아래 별도 명시적 기능으로 처리한다.
- 진행 중 고정 runtime용으로 학습·평가를 import하지 않는 metadata-only
  sidecar를 추가했다. 기존 공식 receipt와 metric JSON은 수정하지 않는다.

G20 본표 `Exact50K`, QG40 본표 `RAW_MAX` 및 모든 공식 선택 규칙은 그대로다.
GF2 Q4를 WV3 Q8로 바꾸지 않는다.

## 실제 적용·검증

- GF2-s1 16–19행, 완료·원본 업로드 검증된 **4개 실험**의 누락 정보를
  실제 업로드하고 readback 확인. Date 모두 `2026-09-21`.
- GF2-s1 앞쪽 열을 `Run / 캠페인 / RR / FR / 비용 / Date / Notes / 설정`으로
  정렬했다. 과거 행도 열과 함께 이동했으며 모든 기존 값·메모·출처를
  전후 대조했다. 상세 provenance 열은 삭제하지 않았다.
- Google Sheets CellData로 저장 정밀도와 표시 값이 분리됨을 확인했다.
  예: Train(h) 원값 `1.2937689420624843`, 표시 `1.29`.
- 관련 focused CPU 테스트 **135개 통과**. G20 전체 213개 및 QG40 전체
  165개도 통과했다(집합 중복이 있으므로 합산하지 않음).
- 운영 runtime의 봉인된 로컬 소스 **85개 SHA 전부 일치**. 학습 중단·재시작,
  환경 변경, metric 재계산은 하지 않았다.
- s1 보고 전용 watch 프로세스 실행: PID `213953`, 60초 주기.
  로그 `work_dir/_sensor_sheet/s1/watch.log`. 기존 업로더가 잠금을 점유하면
  `REPORTING_PENDING`으로 넘어가 다음 주기에 재시도한다. cron은 추가하지 않았다.
- 다른 서버의 live 탭은 이 호스트에서 쓰지 않았다. 실제 학습 날짜 보완과
  열 정렬은 각 서버의 로컬 기록·잠금으로 실행해야 한다. commit/push는 안 했다.

백업·보완 receipt: `work_dir/_sensor_sheet/s1/<run_id>/`,
정렬 snapshot: `work_dir/_sensor_sheet/s1/layout/`.

서버별 명령 및 배포 범위는
[GF2_SHEETS.md](../reporting_extra/GF2_SHEETS.md)를 따른다.

## 후속 요청: 기존 GF2-s4 즉시 서식 보완

사용자의 기존 시트 수정 요청에 따라 GF2-s4 `A3:HA13`(현재 결과 10행)에
서식을 직접 적용했다. 원격 서버의 접속 주소/포트는 확보되지 않아, 서버
잠금이 필요한 열 이동과 로컬 학습 날짜 추정은 하지 않았다.

- 헤더 색상·굵기, RR/FR/비용 그룹별 헤더 색 구분.
- 상단 3행 및 Run/캠페인 2열 고정.
- 주요 지표·비용 열 너비, Run 420px/Notes 550px, 행 높이와 긴 텍스트 CLIP.
- 본표·선택별 지표 4자리, 비용/시간과 seed/step에 정해진 표시 형식 적용.
- Google Sheets API readback에서 전 범위의 `userEnteredValue` 차이 **0개**.
  예: Params 원값 2.19807 → 표시 2.1981, Train(h) 표시 1.01.

이 직접 작업은 **서식만 변경**했다. s4의 주요 지표 열은 여전히 57번째부터
존재한다. WV3 같은 열 순서 재배치와 실제 Date 보완은 s4에서 보고 도구를
실행해야 하며, 모든 서버 배포가 끝난 것으로 해석하면 안 된다.

후속 처리 중 s1의 R0/SS93002가 새로 완료됐다. 보고 전용 watch가
`2026-09-21T01:14:29Z`에 20행의 메타데이터 12셀을 자동 보완하고 readback을
통과했다. 따라서 s1의 보완 완료 결과는 **5건**이며, 고정 runtime의 기존
업로더와 신규 보고 도구가 함께 동작하는 실제 후속 완료 경로도 확인했다.
