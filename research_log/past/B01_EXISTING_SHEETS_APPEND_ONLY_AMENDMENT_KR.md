# B01 Sheets 표시·누적 규칙 변경 — 기존 시트 하단만 사용

작성일: 2026-10-01  
문서 버전: `B01_EXISTING_SHEETS_APPEND_ONLY_v2`  
대상: `pan-cvpr27` / `PANDA_REBUTTAL_B01_WV3_S135_20260928_v1`  
상태: **Live Sheet와 이전 지시서를 대조해 작성한 변경 지시서. 실제 코드·Sheet 변경은 수행하지 않음.**

## 1. 이번 변경의 결론

**앞으로 공개·숨김을 불문하고 새 worksheet를 생성하지 않는다.**

사용자가 읽는 추가 ablation 표는 **기존 `ablations`의 마지막 사용 행 아래**에 누적한다. Native 개별 결과는 이미 연결된 **`WV3-main`의 B01 하단 그룹**을 유지한다. 제출 표와 과거 실험 표의 숫자·의미·배치는 바꾸지 않는다.

이미 생성되어 데이터와 수식이 연결된 숨김 원본 5개는 재사용한다. 이는 추가 탭 생성 지시가 아니다. 지금 존재하는 공개 `RB-B01`, `RB02-curves`는 먼저 보존하고, 신규 표시 갱신의 목적지를 `ablations` 하단으로 전환한다. 기존 탭의 삭제·이름 변경·숨김 변경은 이번 기본 동작에 포함하지 않는다.

## 2. 이전 지시서와의 관계

대체 대상 문서:
`B01_GSPREAD_NORMALIZED_UPLOAD_IMPLEMENTATION_KR_2026-09-30.md`

이전 문서는 다음을 지시했다.
- 3절: 신규 숨김 `_rb01_s1`, `_rb01_s3`, `_rb01_s5`, `_rb02_points`, `_rb_b01_status` 생성.
- 3절 및 9.2절: 신규 공개 `RB02-curves`, `RB-B01` 생성.
- 1절 및 2.1절: `ablations`를 수정 금지 대상으로 지정.

따라서 **이전 문서에는 ‘새 탭 없이 기존 ablations 하단에 추가’ 규칙이 없었다.** 공개 탭을 ‘뒤쪽에 추가’한다는 문장은 탭 순서에 관한 것이지 기존 시트의 행 아래에 붙이라는 뜻이 아니다.

충돌 시 이 v2 변경 지시서를 우선한다.
- 신규 탭 생성 및 전용 공개 탭 자동 생성 규칙: 폐기.
- `ablations` 전체 쓰기 금지: **기존 사용 구역은 보호하되 새 하단 블록만 쓰기 허용**으로 변경.
- 원 증거 검증, 단일 writer, 수치 원정밀도, idempotency, alias 구분, frozen runtime 보호: 유지.

## 3. 2026-10-01 live 확인 상태

Spreadsheet ID:
`1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`

### 3.1 실제 배치

| 위치 | 관측한 상태 |
|---|---|
| 전체 workbook | 탭 53개. 이전 46개에서 B01 관련 7개가 생성되어 있음 |
| `WV3-main!A795` | `06 | Rebuttal B01 | RB01 q reliability` 그룹 제목 |
| `WV3-main` 796–811행 | s1의 8개 Student, EXACT/VAL 16개 관측행 |
| `_rb01_s1` | s1 native 원본 저장 |
| `_rb01_s3`, `_rb01_s5` | 요약과 대조했을 때 등록된 native 결과 없음 |
| `_rb02_points` | RB01 run ID가 있는 shift-point 784행 |
| `RB-B01!A3:J20` | s1 Student 8/8, native 관측16/16, curve16/16; s3/s5의 수집 수는 0 |
| `ablations!A1:AE1020` | `RB0` 식별자 검색 일치 0행 |

시트상 수집 범위는 **s1만 case당 2개 Student**, 전체 목표 대비 **2/6**이다. s3/s5는 `REMOTE_EVIDENCE_NOT_SUPPLIED` / `FILES_NOT_PROVIDED` 성격의 상태다. **미수집을 미실행·미완료로 바꿔 적지 않는다.** 상태 행의 `READBACK_VERIFIED`는 그 상태 행이 업로드되어 재확인되었다는 뜻이며, 측정 결과가 존재한다는 뜻이 아니다.

s1 stress 요약에는 numerical failure 0과 invalid geometry 640이 별도로 표시된다. **16개 곡선을 수집한 것과 모든 관측이 geometry 검사를 통과한 것은 다르다.** 이 플래그를 유지하고, 이번 레이아웃 변경을 이유로 결과를 다시 평가하거나 실패를 제거하지 않는다.

위 행 번호와 수량은 조회 시점의 관측값이다. 실제 실행 직전에 다시 읽고, 고정 행 번호를 append 좌표로 사용하지 않는다.

### 3.2 관련 기존 sheet ID

| 탭 | ID | v2 처리 |
|---|---:|---|
| `paper` | 1923997092 | 전체 보호 |
| `ablations` | 1697831403 | 기존 내용 보호, 소유한 하단 추가 블록만 갱신 |
| `유의미한결과` | 410652761 | 전체 보호 |
| `WV3-main` | 260928010 | 기존 native 자동 표시 유지. 직접 값 append 금지 |
| `_records` | 260928001 | 기존 통합 수식 유지. 직접 값 append 금지 |
| `_rb01_s1` | 269300000 | 기존 source 재사용 |
| `_rb01_s3` | 269300001 | 기존 source 재사용 |
| `_rb01_s5` | 269300002 | 기존 source 재사용 |
| `_rb02_points` | 269300003 | 기존 stress source 재사용 |
| `_rb_b01_status` | 269300004 | 기존 상태 source 재사용 |
| `RB02-curves` | 269300005 | 기존 자료 보존. 새 표시의 필수 목적지에서는 제외 |
| `RB-B01` | 269300006 | 기존 자료 보존. 새 표시의 필수 목적지에서는 제외 |

제목과 ID, header/schema/campaign을 다시 대조한다. 없는 탭을 자동으로 생성하거나 유사 이름의 다른 탭에 쓰지 말고 `TARGET_MISSING` 또는 `SCHEMA_CONFLICT`로 중단한다.

## 4. 목적 데이터 흐름

```text
검증된 서버 결과 / 기존 업로드 원본
    ↓ 기존 bridge의 수집·검증·중복 방지 유지
이미 존재하는 _rb01_s1 / _rb01_s3 / _rb01_s5
    ├─ 현재 _records union → WV3-main의 B01 하단 native 결과
    └─ 통계·paired 집계 → 기존 ablations 맨 아래의 B01 블록

이미 존재하는 _rb02_points / _rb_b01_status
    └─ 곡선 요약·상태 → 같은 ablations B01 블록의 별도 섹션
```

Native와 stress는 같은 탭에서 볼 수 있지만 **같은 지표 표에 섞지는 않는다.**

## 5. `ablations` 하단 블록 구성

블록 제목: `Rebuttal B01 — RB01 q reliability / RB02 controlled shifts`

순서는 다음으로 둔다.

| 섹션 | 내용 |
|---|---|
| 상태 | 서버별 Student·native 관측·curve 수, 기대 수, missing·failure·업로드 상태 |
| RB01 primary | EXACT50K case별 평균·sample SD·n, QFULL/QMEAN/QSHUF/QESUR |
| RB01 paired | 같은 server/repeat/seed의 case−QFULL, paired n·평균·SD·개선 수 |
| RB01 secondary | RR_VAL_ERGAS_MIN. Primary와 독립 구역, alias 명시 |
| RB01 개별 run | 최대24 Student/48 selector 관측의 읽기 쉬운 표; run ID·SHA·source 연결 |
| RB02 radius summary | case/mode/radius별 seed 평균·SD·n, numerical/geometry flag 수 |
| RB02 paired | A_ON−A_ZERO_INFERENCE_ONLY; 같은 모델의 paired 결과 |
| 출처·한계 | source 범위, cohort, 갱신 시각, missing/실패와 상세 원자료 링크 |

읽기 쉬운 wide table을 사용한다. 긴 JSON을 화면의 주 요약 표에 반복하지 말고 셀 note나 출처 열로 분리한다. native 표에는 HQNR/D_s/D_lambda/ERGAS/PSNR/SAM 등 실제 확보한 지표를 사용한다. Mean과 SD를 계산하는 원 숫자는 분리해 보존하고, 문자열 `mean ± SD`만 유일한 수치 원본으로 삼지 않는다.

RB02의 2,352개 전체 point 예상량을 요약 위에 모두 펼치지 않는다. 전체 point는 **이미 존재하는 `_rb02_points`**를 원본으로 유지하고, `ablations`에는 먼저 반경별 요약과 paired 비교를 표시한다. 상세 point가 추가로 필요하면 같은 블록 맨 아래에 출력할 수 있으며 새 탭을 만들지 않는다.

## 6. 하단 누적과 중복 갱신 규칙

1. `ablations` 전체 기존 열 범위를 읽어 실제 사용된 마지막 행을 찾는다. `rowCount=1020`은 내용 끝이 아니다. A열만 검사하지 않는다.
2. 값·수식·note·병합 범위·다른 배열 수식의 전개 범위와 예약된 사용자 구역을 확인한다. 기존 콘텐츠를 침범하지 않도록 마지막 사용 구역 아래에 2개 빈 행을 두고 첫 B01 블록을 배치한다.
3. 시작 셀 note와 로컬 배치 manifest에 `owner=B01_EXISTING_SHEETS_APPEND_ONLY_v2`, campaign ID, section ID, 범위, schema, source hash를 저장한다. 같은 owner가 있으면 두 번째 블록을 만들지 않는다.
4. 같은 B01의 s3/s5 결과가 뒤늦게 도착하면 이미 소유한 B01 블록 내 빈 슬롯·통계만 갱신한다. **2/6→6/6은 같은 실험의 누적 수집이지 새 독립 실험이 아니다.**
5. 최초 배치에서 B01의 최대24 Student/48 selector 규모를 수용할 수 있게 공간을 정한다. 실제 값이 없는 슬롯은 빈값/미수집으로 두고 0 또는 계획값을 측정치처럼 채우지 않는다.
6. 후속 B02 등 다른 실험 묶음은 현재 소유 블록의 끝과 기존 내용 모두보다 아래에 새 섹션을 추가한다. 기존 B01 표 아래에 다른 사용자 내용이 생겼으면 그 내용 위에 삽입·덮어쓰지 않는다.
7. 이미 소유한 범위가 부족하면 충돌을 검사하고 신규 가용 하단의 명시적 continuation 구역으로 확장한다. 다른 사용자 행을 자동으로 이동시키지 않는다.
8. 수동 입력/placeholder/text는 RAW로 쓰고, 집계·참조 수식이 필요한 셀에만 수식을 쓴다. `=`, `+`로 시작하는 설명을 의도치 않은 수식으로 만들지 않는다.
9. 관측의 upsert 키는 기존 source의 Result_ID/Record_ID를 유지한다. 새로운 표시 행이 만들어져도 별도 측정·별도 seed로 세지 않는다.

`WV3-main`의 native B01 그룹은 이미 맨 아래에 있으므로 불필요하게 재작성하지 않는다. 향후 그룹 추가 때문에 B01이 중간으로 들어가는 경우에는 **표시 순서만** 조정할 수 있다. 수식에서 기존 그룹을 먼저, rebuttal 그룹을 마지막에 붙이고 기존 행의 값·상대순서는 유지한다. 배열 수식 전개 영역에 수동 결과를 직접 추가하는 방식은 금지한다.

## 7. 코드 변경 요구

기존 독립 bridge의 파일·CLI가 현재 어디까지 구현되었는지 먼저 확인한 후 최소 수정한다. 이전 문서의 제안 이름을 확인 없이 실행 가능한 명령으로 제시하지 않는다.

필수 변경:
- 기본 출력 정책을 `existing_sheets_append_only`로 변경.
- `create_sheet`, `add_worksheet`, `addSheet`, copy/duplicate worksheet 등 신규 탭 생성 경로를 금지. 누락 대상은 오류 반환.
- `RB-B01`/`RB02-curves` 생성·갱신 대신 `ablations` 내 소유한 B01 block writer로 연결.
- 정상 source upsert 후 `ablations` 블록을 자동 갱신. Native만 성공하면 native를 표시하고 stress 미수집 상태를 그대로 표시.
- 기존 `_records` source 연결은 이미 있으므로 다시 union을 추가하지 않는다.
- 단일 조정 writer로 공유 통계를 갱신. s3/s5가 동시에 레이아웃을 다시 만들지 않는다.
- before/after snapshot, source payload hash, actual row coordinates, written/no-op/conflict counts, view readback receipt 유지.
- 기존 수치 runtime, `panda_rb` 학습·평가 코드, model weight, 원 metrics JSON/CSV/NPY를 수정하지 않는다.

새 명령을 구현할 경우 dry-run 출력에는 `new_sheet_count=0`, 실제 기존 tab ID, 추가 범위, 갱신 범위, 보호 범위, source 식별자를 반드시 출력한다. 현재 시점 기준 탭 수 53개를 감소시키는 작업도 기본 migration에는 포함하지 않는다.

## 8. 통계와 완료 상태

- B01 목표: Student24, native selector 관측48, curve48, point2352.
- 현재 조회: Student8, native16, curve16, point784. 모두 s1 자료. case당 독립 Student n=2/6.
- EXACT/VAL이 같은 SHA라도 independent n은 늘지 않는다.
- 집계 단위는 Student seed. Scene20개·방향8개·mode2개를 반복 수로 세지 않는다.
- 표준편차는 sample SD(ddof=1). n<2에서 SD를 0으로 만들지 않는다.
- Paired 비교는 같은 server/repeat/seed, 같은 cohort의 차이를 사용한다.
- Radius summary는 각 Student 안에서 방향들을 먼저 집계하고 그 후 seed 통계를 계산한다.
- numerical failure, invalid geometry, evidence missing, delivery/readback failure를 서로 구별한다.
- source에 invalid geometry가 남아 있으면 새 표시에도 건수와 flag를 유지한다. 정상 결과만 임의로 선택해 n을 부풀리지 않는다.
- 현재 s3/s5 상태는 자료 미제공이지 실험 실패 판정이 아니다.

## 9. 변경 전후 확인 및 반환

변경 전에는 live 기존 표와 수식·note·서식을 파일로 백업한다. 새 백업 worksheet는 만들지 않는다.

완료 조건:
1. 새 탭 생성 0개.
2. `ablations`의 과거 사용 구역 불변; 새 하단 B01 블록만 변경.
3. `paper`와 `유의미한결과` 불변. Table 3 교정·대표 모델 변경 없음.
4. `WV3-main` B01 native 값·Result_ID가 source와 일치하고 중복이 없음.
5. 새 `ablations` 요약의 n·mean·SD·paired 값이 source와 일치.
6. 현재 부분 수집을 24/48 완료 또는 case당6개로 표시하지 않음.
7. 재실행 시 중복 블록/관측 0개. 자신의 블록 밖 값·note·수식 변경 0개.
8. 새 formula error나 spill 충돌 없음.
9. 서버 학습·추론 재실행 0회.

반환 보고에는 `ablations`의 실제 시작/끝 범위, `WV3-main` B01 범위, source별 등록 수, missing 서버, numerical/geometry 플래그, 새 탭 생성 수, 동일 payload 재시도 결과를 적는다.

**이 문서는 지시사항의 변경본이다. 현재 live Sheet의 행 이동·탭 숨김·코드 수정·업로드를 이미 실행했다는 보고가 아니다.**

## 10. 확인 근거

- 이전 MD 3절 및 9.2절: 신규 공개/숨김 탭 생성 설계.
- 이전 MD 1절 및 2.1절: ablations 쓰기 금지.
- Live `pan-cvpr27` metadata: 53개 탭과 B01 신규7개 존재.
- Live `WV3-main!AC1:AH3000`의 RB01 run 검색: 796–811행, 16개 selection 관측.
- Live `WV3-main!A791:D814`: B01 그룹 제목795행과 개별 native 결과 확인.
- Live `RB-B01!A1:J180`: s1만 등록된 상태와 case별 n=2/6 확인.
- Live `_rb_b01_status!D10:Y11`: s3 자료 미제공/실행상태 UNKNOWN 확인.
- Live `_rb02_points!D2:F2400`: RB01 run 식별자 포함784개 point행.
- Live `ablations!A1:AE1020`의 RB0 검색: 일치0행.
