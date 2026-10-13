# 새 실험 스프레드시트 초기화 및 고정 컬럼 계약

작성일: 2026-10-13  
대상: **pan-cvpr27-new**  
Schema: **PAN_FIXED64_V1**

## 1. 실제 처리한 내용

- 새 파일 ID: `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE`
- 기존 파일 ID `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`에는 쓰기 요청을 보내지 않았다.
- 56개 탭에서 과거 `analysis` 및 숨김 원본/집계/캠페인 탭 46개를 삭제했다.
- 기존 WV2-ablations와 쌍을 맞추기 위해 WV2-main 1개를 추가했다. 최종 11개 탭이다.
- paper를 제외한 10개 탭의 과거 값·셀 수식·메모·셀 병합/기본서식을 초기화하고 고정 템플릿을 적용했다.
- 10개 탭의 6행 이후에는 현재 결과가 없다. 제목·안내·구획·고정 헤더만 남아 있다.
- 원 main의 64열 이름/순서를 유지했다. 5행 고정 헤더, 6행부터 데이터, 상단 5행/왼쪽 3열 고정이다.
- 헤더 1–5행에는 편집 제한을 적용했다. 문서 소유자는 관리할 수 있으므로 서버의 schema 검증도 필수다.
- 기존 조건부 서식 일부는 그대로 유지되어 있다. 결과를 가져오는 셀 수식이나 과거 결과 값은 남기지 않았다.

## 2. paper 보존의 정확한 의미

paper의 표시값·셀 서식·병합·셀 메모를 보존했다. 삭제 대상 PC-Repro 원본을 참조하던
`D85:D113`의 29개 CHOOSECOLS 배열 수식은 원본 삭제 전에
`D85:L113` 범위의 현재 계산값으로 고정했다. 따라서 참조 탭 삭제 후 #REF!가 발생하지 않는다.

원 수식·원본 값은 `pan-cvpr27-new_before_reset.xlsx`와
`paper_before_formulas.json`에 보관했다. 이 작업은 수치 교정이나 새 모델 결과의 등록이 아니다.
신규 paper 결과를 자동으로 채우지 않는다.

## 3. 탭 목록과 역할

| 탭 | sheetId |
|---|---:|
| `paper` | `1923997092` |
| `ablations` | `1697831403` |
| `유의미한결과` | `410652761` |
| `WV3-main` | `260928010` |
| `WV3-ablations` | `261001010` |
| `QB-main` | `260928011` |
| `QB-ablations` | `261001011` |
| `GF2-main` | `260928012` |
| `GF2-ablations` | `261001012` |
| `WV2-main` | `261013013` |
| `WV2-ablations` | `261001013` |

- `*-main`: 해당 데이터셋의 신규 main 결과를 직접 저장.
- `*-ablations`: 해당 데이터셋의 신규 ablation / sensitivity / 진단 결과를 직접 저장.
- `ablations`: 사용자가 검토할 논문 정리용. 서버 자동 업로드 금지.
- `유의미한결과`: 선별·집계 검토용. 서버 자동 업로드 금지.
- `paper`: 기존 내용 보존. 서버 자동 업로드 금지.
- 아카이브와 이전 실험은 기존 스프레드시트를 참고한다. 새 파일에 자동 복구하지 않는다.

## 4. 고정 저장 형식

1. 모든 비-paper 탭은 A:BL **정확히 64열**이다. 이름·순서·단위를 변경하지 않는다.
2. 헤더는 **A5:BL5의 literal 값**이다. `_records`에서 가져오는 ARRAYFORMULA를 사용하지 않는다.
3. 데이터는 6행부터 직접 기록한다. 기존 raw source의 row1/row2 규칙을 적용하지 않는다.
4. 한 행은 한 checkpoint와 한 selector/evaluation-scope의 관측이다.
5. 같은 run의 여러 selector는 다른 관측행이며, 독립 seed 수로 중복 집계하지 않는다.
6. 미측정 값은 빈 셀(null→빈 셀)로 둔다. 0이나 이전 방법의 값을 채우지 않는다.
7. 모델의 method가 바뀌어도 컬럼을 추가하지 않는다. 부가 설정·진단·통계는 BI의 JSON 또는 BG의 artifact/manifest URL로 기록한다.
8. 기존 파라미터 열 q_ref/tau_R/alpha/beta/lambda_E가 새 방법에 해당하지 않으면 공란으로 둔다. 다른 의미의 파라미터를 이 열에 재사용하지 않는다.
9. 스코프가 다른 stress 평가를 native RR/FR과 혼합하지 않는다. `Eval_scope`와 BI에 별도 프로토콜을 적고 없는 지표는 공란으로 둔다.
10. 그룹 제목·subtotal·분석 문장을 원본 실험 행 사이에 삽입하지 않는다. Approach/Case_ID/Repeat로 필터링한다.

### 컬럼 목록

| 열 | 번호(1-based) | 정확한 헤더 | 저장 자료형 |
|---|---:|---|---|
| A | 1 | Experiment | string |
| B | 2 | Server | string |
| C | 3 | Selection@Step | string |
| D | 4 | HQNR↑ | number |
| E | 5 | D_s↓ | number |
| F | 6 | D_lambda↓ | number |
| G | 7 | JQM↑ | number |
| H | 8 | ERGAS↓ | number |
| I | 9 | SCC↑ | number |
| J | 10 | SAM↓ | number |
| K | 11 | PSNR↑ | number |
| L | 12 | SSIM↑ | number |
| M | 13 | Q4/Q8↑ | number |
| N | 14 | RMSE↓ | number |
| O | 15 | CC↑ | number |
| P | 16 | Infer(ms) | number |
| Q | 17 | Mem(MB) | number |
| R | 18 | Params(M) | number |
| S | 19 | FLOPs(G) | number |
| T | 20 | Train(h) | number |
| U | 21 | Eval(h) | number |
| V | 22 | Wall(h) | number |
| W | 23 | HQNR(V64)↑ | number |
| X | 24 | Signed D_s | number |
| Y | 25 | Positive D_s fraction | number |
| Z | 26 | Dataset | string |
| AA | 27 | Bucket | string |
| AB | 28 | Approach | string |
| AC | 29 | Campaign_ID | string |
| AD | 30 | Run_ID | string |
| AE | 31 | Case_ID | string |
| AF | 32 | Repeat | number |
| AG | 33 | Seed | number |
| AH | 34 | Attempt | number |
| AI | 35 | Role | string |
| AJ | 36 | Model / inputs | string |
| AK | 37 | Teacher / reference | string |
| AL | 38 | Parent_run | string |
| AM | 39 | Updates | number |
| AN | 40 | Lifetime_updates | number |
| AO | 41 | Selected_step | number |
| AP | 42 | Checkpoint_SHA | string |
| AQ | 43 | Selector | string |
| AR | 44 | q_ref | number |
| AS | 45 | tau_R | number |
| AT | 46 | alpha | number |
| AU | 47 | beta | number |
| AV | 48 | lambda_E | number |
| AW | 49 | U_peak_lr | number |
| AX | 50 | A_peak_lr | number |
| AY | 51 | Status | string |
| AZ | 52 | Readback | string |
| BA | 53 | Eval_scope | string |
| BB | 54 | Test_aware | boolean |
| BC | 55 | JQM_variant | string |
| BD | 56 | Cost_scope | string |
| BE | 57 | Source_sheet | string |
| BF | 58 | Source_row | number |
| BG | 59 | Source_URL | string |
| BH | 60 | Result_ID | string |
| BI | 61 | Original description / notes | string |
| BJ | 62 | Review | string |
| BK | 63 | Sort_case | number |
| BL | 64 | Date | string |

모든 metric/시간/횟수는 원정밀도로 저장하고, 표시 반올림은 셀 서식에만 맡긴다.
SHA·Run_ID·URI는 문자열로 저장한다. Test_aware는 실제 boolean으로 저장한다.
Date는 timezone이 포함된 ISO 8601 문자열로 기록한다.
컬럼 M은 Q4/Q8의 공통 슬롯이며 Dataset와 평가 manifest로 실제 밴드 수를 식별한다.
Cost_scope에는 입력 크기, batch, precision, GPU, 시간 집계 범위, MAC/FLOPs 규약을 남긴다.

## 5. 서버 gspread 코드 변경 지시

### 5.1 대상 오인 및 구형 백필 방지

새 결과 전용 writer는 위 **새 spreadsheet ID**를 명시적으로 사용한다.
파일명만으로 검색하여 첫 번째 결과를 선택하지 않는다.
기존 파일 ID, 다른 복제본 ID, 예상하지 못한 sheetId는 쓰기 전에 거부한다.

기존 업로더의 다음 동작을 새 파일에서는 금지한다.

- `_records`, `_rb_*`, `analysis`, `<dataset>-s<server>`, archive를 다시 생성하는 동작.
- 완료된 과거 B01/M12/ABLR2/PAN-Crafter 결과를 새 Sheet에 backfill하는 동작.
- 헤더 자동 추론, schema auto-repair, 컬럼 자동 확장, 캠페인별 열 삽입.
- 기존 ARRAYFORMULA/FILTER/REDUCE 통합 수식을 재설치하는 동작.
- `paper`/`ablations`/`유의미한결과`를 자동 업데이트하는 동작.

이 문서는 새 방법의 모델 구조나 loss/config를 정하지 않는다.
새 method_revision·새 campaign allowlist를 실행 설정에서 명시하고, BI에도 남긴다.
이전의 F1/PLH/q 설정을 자동으로 승계하지 않는다.

### 5.2 쓰기 전 schema 검사

- spreadsheet ID와 sheetId가 계약과 일치하는지 검사.
- 실제 columnCount=64인지 검사.
- A5:BL5를 읽어 `PAN_FIXED64_V1.schema.json`의 header 배열과 정확히 비교.
- A2 note의 schema_version/reset_epoch를 확인.
- header SHA256: `bfc7f73fd406f447edb9721823e2c1df85540f7068e15647947cd0d2e83a11b1`
- 불일치 시 **쓰기만 중단**하고 오류를 반환. 원격 헤더를 자동으로 고치지 않는다.
- dataset/bucket으로 허용된 8개 dataset 탭 중 하나를 결정한다.
- 같은 config라도 schema 검증/대상 allowlist를 건너뛰지 않는다.

### 5.3 행 업로드

payload를 키→값 매핑에서 고정 64열 배열로 직렬화한다.
알 수 없는 최상위 field는 자동 새 컬럼으로 만들지 말고 BI의 명시적 JSON schema로 전송하거나 검증 오류를 반환한다.
예시 BI 내용은 다음과 같다. 이것은 실제 실험 결과가 아니라 필드 구조의 예다.

```json
{
  "schema_version": "PAN_FIXED64_V1",
  "reset_epoch": "2026-10-13",
  "method_revision": "<새 방법의 revision>",
  "config_sha256": "<resolved config SHA>",
  "code_commit": "<실제 실행 commit>",
  "dataset_manifest_sha256": "<실제 데이터 manifest SHA>",
  "evaluator_revision": "<평가기 revision>",
  "extra_parameters": {},
  "artifact_manifest_url": "<실제 접근 가능한 증거 링크>"
}
```

`Result_ID`는 새 namespace·dataset·run·selector·checkpoint·eval protocol을 사용해 결정적으로 만든다.
같은 ID/같은 payload의 재시도는 no-op이다. 같은 ID에 서로 다른 값이 오면 충돌로 보고하고 조용히 덮어쓰지 않는다.
단일 writer를 지정해 여러 서버의 append/upsert 경쟁을 방지한다. 학습 서버는 로컬 결과와 spool을 보존한다.
업로드 실패로 학습이나 기존 산출물을 지우지 않는다.

### 5.4 Readback

쓰기 후 반환된 실제 범위를 다시 읽어 Run_ID/Result_ID/selector/SHA와 수치·열 수를 대조한다.
`Readback=VERIFIED`는 이 검사가 성공한 뒤에만 기록한다.
서버의 로컬 파일 해시 검증과 Sheet readback은 구분하여 기록한다.

행이 부족하면 **행 수만** 늘릴 수 있다. 열 수는 늘리지 않는다.
현재는 각 비-paper 탭 3,000행이며, 결과 영역은 A6:BL3000이다.
행 확대 시 필터·서식을 확장하되 헤더의 값/순서/개수는 바꾸지 않는다.

## 6. 고정 컬럼 회귀 테스트

필수: 64개 헤더 비교, 다른 workbook ID 거부, 허용하지 않은 탭 거부,
모르는 field 거부, 과거 campaign backfill 거부, 동일 ID 중복 방지,
누락값을 0으로 채우지 않기, 선택점별 checkpoint 혼합 방지,
서버 동시 요청 직렬화, RAW 자료형·readback 확인.
실제 프로덕션 시트에 테스트 수치를 남기지 않는다.

## 7. 인계 상태와 검증

Sheet 초기화와 헤더 편집 제한은 실제 적용했다.
**서버의 gspread 코드·환경변수·기존 worker는 이번 작업에서 변경하지 않았다.**
따라서 새 파일에 실험을 올리기 전에 위 writer 계약을 적용해야 한다.

검증 파일: `verification.json`  
헤더 계약: `PAN_FIXED64_V1.schema.json` / `PAN_FIXED64_V1.columns.csv`  
초기화 전 백업: `pan-cvpr27-new_before_reset.xlsx`  
초기화 후 export: `pan-cvpr27-new_clean.xlsx`

시트 기반 수치/구조 검증은 완료했다.
전체 export의 artifact renderer는 import 중 RPC 종료로 실패하여, 브라우저 픽셀 수준 렌더 검증은 완료하지 못했다.
