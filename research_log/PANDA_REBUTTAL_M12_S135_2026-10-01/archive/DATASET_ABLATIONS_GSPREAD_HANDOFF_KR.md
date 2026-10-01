# 데이터셋별 Ablations 시트 전환 — 서버 gspread 호환 코드 변경 지시서

작성일: **2026-10-01**  
정책 ID: **`PAN_DATASET_ABLATIONS_v1`**  
대상 Spreadsheet: **`pan-cvpr27`**  
Spreadsheet ID: `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`  
대상 저장소: `hojunking/PAN-Crafter-repro`

**상태: Google Sheets의 데이터셋별 분리·이동은 이미 실행했고, 전후 export로 검증했다. 이 문서의 남은 실행 대상은 서버 reporting/gspread 코드의 호환 수정과 readback이다. 시트를 다시 만들거나 이전 수식으로 되돌리는 작업이 아니다.**

---

## 1. 최종 사용자 방침과 이전 문서의 대체 범위

1. 이름이 정확히 **`ablations`인 기존 탭은 논문용 수동 정리 탭**이다. 자동 업로더의 쓰기 대상이 아니다. 마지막 행 아래에 추가하는 것도 금지한다.
2. `paper`, `유의미한결과`, `배포용 모델`도 이번 작업에서는 변경하지 않는다.
3. 측정된 ablation/sensitivity/비교 실험의 개별 관측은 **`WV3-ablations`, `QB-ablations`, `GF2-ablations`, `WV2-ablations`**에서 확인한다.
4. 각 데이터셋의 `*-main`과 동일한 **64열 양식, 5행 헤더, 6행 본문 시작, FR/RR/Cost 구획**을 사용한다.
5. 원본 숨김 탭과 `_records` 통합 구조는 보존한다. “이동”은 원본을 지우는 cut/paste가 아니라 **통합 결과의 Bucket 변경에 따른 배타적인 표시 위치 변경**이다.
6. 실험 수치, checkpoint/selection, run/seed/반복, source/hash/Result_ID를 바꾸지 않는다. 재학습·재추론·지표 재계산은 필요하지 않다.

이 정책은 다음 이전 지시의 **표시 목적지·분류 규칙**을 대체한다.

- `B01_GSPREAD_NORMALIZED_UPLOAD_IMPLEMENTATION_KR_2026-09-30.md`의 “B01 native를 WV3-main에 표시” 규칙.
- `B01_EXISTING_SHEETS_APPEND_ONLY_AMENDMENT_KR.md`의 “ablations 하단에 누적” 규칙.

기존 문서의 증거 검증, 멱등성, 통계 단위, 동일 source/runtime, 실패 보존, frozen numerical code 불변 규칙은 유지한다. 이전 문서를 삭제하거나 과거 실행 이력을 다시 쓰지 않는다.

## 2. 이미 완료한 실제 시트 변경

### 2.1 생성된 데이터셋별 표시 탭

| 탭 | sheetId | 현재 관측행 | 그룹 수 |
|---|---:|---:|---:|
| WV3-ablations | 261001010 | 766 | 4 |
| QB-ablations | 261001011 | 455 | 1 |
| GF2-ablations | 261001012 | 668 | 3 |
| WV2-ablations | 261001013 | 0 | 0 |

위 행 수는 **2026-10-01 이전·이후 snapshot 검증 시점**의 값이다. 향후 정상적인 신규 업로드로 늘어날 수 있으므로 업로더의 고정 제한값으로 사용하지 않는다.

네 탭은 WV3-main을 복제해 만들었다. 3000행×64열, 상단 5행·왼쪽 3열 고정, gridline 숨김, 우측 metadata 열, 원래 수치 서식과 조건부 색상 규칙을 유지했다. 신규 탭의 `S/FLOPs(G)` 열은 읽을 수 있도록 96px로 조정했다.

WV2에는 현재 해당 분류의 관측이 없어 헤더와 0건 상태만 존재한다. 가상의 결과나 0점 행은 만들지 않았다.

### 2.2 이동한 범위와 잔여 Main

| 원래 위치 | 이동한 관측 | 이동 후 원래 탭의 관측 |
|---|---:|---:|
| WV3-main | 677 | 123 |
| WV3-archive | 89 | 1226 |
| QB-main | 455 | 177 |
| GF2-main | 668 | 410 |
| 합계 | **1889** | — |

WV3-ablations의 766행 = Component 465 + MAIN-A 196 + RB01 16 + legacy G23 sensitivity 89.

GF2-ablations의 668행 = Component 500 + GFB20 90 + GFP40 78.

QB-archive 16, GF2-archive 20, WV2-archive 9, PC-Repro 38 관측은 그대로다. `01 | FULL method / tuning`, 독립 `05 | Teacher / reference`, 분류가 다른 과거 개발·snapshot은 이번 자동 이동 대상이 아니다. 다만 C00–C17 대조군과 함께 시행한 ABLR2의 TPLUS/TZERO는 Component 그룹 안에 같이 이동했다.

GFP40 B09처럼 paper에 사용된 선택 결과도 해당 비교 campaign의 다른 대조군과 함께 `GF2-ablations`에 위치한다. 이는 목적별 분류이며, 제출본 또는 paper 수치를 바꾸거나 낮은 결과를 제외한 것이 아니다.

### 2.3 보존 확인

- `_records` 전체 관측 **3908 → 3908**.
- Result_ID 집합과 각 ID의 **Bucket 이외 63열이 모두 정확히 동일**하다.
- AA/Bucket만 1889관측에서 `Main/Archive → Ablations`로 변경됐다.
- 기존 숨김 source 탭 39개의 값·수식·note는 동일하다.
- `paper`, `ablations`, `유의미한결과`, `배포용 모델`의 값·수식·note·해석된 셀 서식·layout을 보존했다.
- 기존 Main/Archive/PC-Repro의 A2/A5/A6 수식은 그대로이며, 바뀐 Bucket에 따라 표시가 재계산됐다.
- 새 탭 및 기존 데이터셋 표시 탭의 오류 셀을 확인했으며 오류가 없었다.
- 신규 관측, seed, training run을 만든 작업이 아니다.

검증 파일은 `migration_verification.json`, 실제 이동한 ID 목록은 `migration_observations.csv`를 사용한다.

## 3. 이제 적용할 구조

```text
기존 숨김 source / _rb01_s1 / _rb01_s3 / _rb01_s5
                       ↓
                  _records!A2
            기존 parser와 B01 union 유지
            최종 출력에서 AA/Bucket만 정규화
                       ↓
          Dataset(Z) + Bucket(AA)별 표시
     ┌─────────────────┼───────────────────────┐
   *-main         *-ablations              *-archive
  Main만           Ablations만             Archive만

논문용 ablations / paper / 유의미한결과: 이 자동 경로와 별개, 읽기 전용
배포용 모델: 이번 작업 범위 밖, 변경 금지
```

**직접 append 금지 영역**:

- `_records!A2:BL` — A2 배열 수식의 전개 영역.
- 데이터셋별 `*-main`, `*-ablations`, `*-archive`의 A5:BL 및 A6 본문.
- 기존 논문용 `ablations` 전체.

서버는 등록된 원본 숨김 source의 자기 소유 slot을 upsert하고, 표시 결과는 수식이 갱신하도록 한다. 원래 source 탭명으로 “이동”을 해석하여 `WV3-s1`을 rename/delete하거나, 계산된 view에서 `append_row()`를 호출하지 않는다.

## 4. 분류 계약 — 실제 적용된 규칙

현재 분류에 사용되는 것은 **AB/Approach의 명시된 접두사**다. 성능·서버·파일명 일부를 보고 추측하지 않는다.

| Approach 접두사 | 포함 목적 | normalized Bucket |
|---|---|---|
| `02 | ` | Component ablation, C00–C17 + paired Teacher | Ablations |
| `03 | ` | MAIN-A alpha/beta/edge sensitivity | Ablations |
| `04 | ` | GFB20/GFP40 fitting·PAN-frequency 비교 | Ablations |
| `06 | ` | Rebuttal B01 RB01 q reliability | Ablations |
| `10 | ` | Legacy G23 sensitivity | Ablations |
| 그 외 | 기존 분류 유지 | 원 Bucket |

정확한 정규식:

```text
^(02|03|04|06|10) [|] 
```

끝의 공백까지 포함한다. Legacy G23은 새 recipe와 통합 평균을 내지 않고 기존 Approach 그룹을 유지한다.

새 단계/새 campaign은 의미를 확인한 명시적인 registry로 목적을 지정한다. 알 수 없는 이름을 모두 ablation으로 보내거나, `ablation`이라는 단어가 note에 한 번 나왔다는 이유만으로 기존 개발 기록을 재분류하지 않는다.

### 4.1 기존 원본 Bucket 호환성 — 중요

기존 B01 원본 `_rb01_s1` 등에는 `Bucket=Main`이 남아 있을 수 있다. **정상이다.** 원본과 transport payload/receipt를 훼손하지 않기 위해 이미 업로드된 source row는 그대로 두었다. `_records`의 최종 정규화가 `Ablations`로 바꾼다.

이번 서버 수정의 권장 방식:

- 기존 `PANDA_RB_B01_SHEETS_BRIDGE_v1` source payload, 64열 헤더, 고정 slot, Result_ID, source receipt hash는 그대로 유지한다.
- **source readback**은 원 payload와 비교한다.
- **normalized/view readback**은 AA만 정규화한 기대값과 비교한다.
- “원본 Main과 view Ablations가 다르다”는 이유로 원본 전 행을 재업로드하거나 evidence conflict를 우회하지 않는다.
- 앞으로 별도의 새 source schema가 Bucket=Ablations를 직접 기록하는 것은 가능하지만, 기존 봉인된 payload의 소급 변경과 섞지 않는다.

부작용 없이 사용할 수 있는 순수 기대값 변환 예시:

```python
import re

ABLATION_APPROACH = re.compile(r"^(02|03|04|06|10) [|] ")

def normalized_view_row(source_row: list) -> list:
    if len(source_row) != 64:
        raise ValueError("Expected canonical 64-column native observation")
    result = list(source_row)
    if ABLATION_APPROACH.match(str(result[27] or "")):
        result[26] = "Ablations"   # AA만 변경
    return result
```

이는 구현 참고 코드다. 훈련 evidence를 바꾸는 함수로 사용하지 않는다.

## 5. 실제 시트 수식과 소유권

### 5.1 `_records!A2` 최소 변경

기존 parser, `allrows`, `rbload`, 세 서버 union을 보존한 채 마지막

```text
FILTER(joined,CHOOSECOLS(joined,1)<>"")
```

부분 한 곳만 다음으로 확장했다.

```text
LET(
  kept,FILTER(joined,CHOOSECOLS(joined,1)<>""),
  routebucket,IF(
    REGEXMATCH(TO_TEXT(CHOOSECOLS(kept,28)),"^(02|03|04|06|10) [|] "),
    "Ablations",
    CHOOSECOLS(kept,27)
  ),
  HSTACK(
    CHOOSECOLS(kept,SEQUENCE(1,26)),
    routebucket,
    CHOOSECOLS(kept,SEQUENCE(1,37,28))
  )
)
```

수식 전체는 함께 제공한 `formula_anchors_after.json`의 `_records.A2`에 있다. 줄바꿈 없는 실제 수식 SHA256:

```text
9b3765c74a229e8bdec2cf3696fdd6b159c47f76ac7bab15f75078ef3db7da27
```

이 SHA는 **이번 migration 당시 수식 identity**다. 이후 다른 승인된 source union이 추가되면 달라질 수 있다. SHA가 달라졌다고 오래된 전체 수식으로 덮어쓰지 말고 구조를 검토한다.

DeveloperMetadata도 추가했다.

```text
location: sheetId 260928001 (_records)
metadataKey: PAN_DATASET_ABLATIONS_ROUTING
metadataValue: v1;2026-10-01;Approach=02|03|04|06|10;Bucket=Ablations;only-AA
metadataId: 1466780012
visibility: DOCUMENT
```

### 5.2 새 표시 탭

A5는 기존과 동일하다.

```gs
=ARRAYFORMULA('_records'!$A$1:$BL$1)
```

A6는 해당 Dataset과 Bucket=Ablations를 filter한다. 0건은 빈 본문으로 처리하며, 그룹 제목 다음에 각 관측을 아래 순서로 출력한다.

```text
Approach 그룹 → Sort_case(63) → Repeat(32) → Server(2)
             → Run_ID(30) → Selector(43)
```

HQNR 최고값으로 ablation run을 선별하지 않는다. 행 수를 세면서 `▌` 그룹 제목이나 선택점 alias를 독립 run으로 세지 않는다.

네 탭의 정확한 A2/A5/A6도 `formula_anchors_after.json`에 포함했다. 현재 정상 동작하는 표시 수식은 재설치할 필요가 없다.

## 6. 코드 수정 대상과 실제 발견한 호환 문제

확인한 기준 commit: `ea71b68be637a1d1f4d61bf89cb4e023cbf00385`.

수정 범위는 독립 reporting 코드다.

```text
reporting_bridge/rb_b01_sheets.py
reporting_bridge/rb_b01_contract.py   # 필요한 순수 view helper만; 기존 evidence contract 유지
reporting_bridge/tests/
reporting_bridge/README.md
tools/rb_b01_sheets.py               # help/검증 목적지 안내 등 필요한 부분만
```

기존 `panda_rb/`, `tools/panda_rb_*.py`, 모델·loss·evaluator, frozen release는 수정하지 않는다. 기존 numerical source-identity에 영향을 주거나, 새 코드에서 원 source 검증을 꺼서 통과시키지 않는다.

### 수정 A — `patch_formula()`가 새 wrapper를 인식해야 한다

기존 `rb_b01_sheets.py`는 `NEW_RETURN`의 정확한 문자열을 찾아 기존 B01 union을 인식한다. 지금처럼 반환 부분에 routing LET가 감싸져 있으면, 기존 문자열과 다르므로 **`FORMULA_REVIEW_REQUIRED: unrecognized existing bridge`**를 내는 경로가 있다.

해결:

1. 세 rbload source가 각각 한 번만 들어 있는지 검사한다.
2. 이미 적용된 routing wrapper가 정상인지 검사한다.
3. 정상 v1이면 **원 수식을 그대로 반환하는 no-op**로 처리한다.
4. `setup`이 구형 `NEW_RETURN`을 덮어쓰거나 중복 rbload/routing을 추가하지 않도록 한다.
5. 알려진 구형 수식의 지원이 필요하면 정확한 단일 terminal expression만 patch한다. 알 수 없는 수식은 중단하고 검토한다.
6. 현재 운영 sheet는 migration이 완료돼 있으므로 정상 dry-run setup diff는 **0**이 되어야 한다.

### 수정 B — `verify()`의 목적지를 바꾼다

기존 `verify()`는 `_records`와 **`WV3-main`**에서 B01 source row 전체를 동일하게 찾는다. 이제 그 기대가 맞지 않는다.

새 검증 경로:

```text
원본 _rb01_<server> : 원 RAW payload와 64열 비교
_records           : normalized_view_row(payload)와 64열 비교
WV3-ablations      : 같은 normalized row + Result_ID 정확히 1회
WV3-main           : 해당 B01 Result_ID가 0회
WV3-archive        : 해당 B01 Result_ID가 0회
```

이 검증을 Dataset→sheetId mapping으로 구현하여 QB/GF2/WV2 확장도 일관되게 한다. 비교할 때 Bucket을 통째로 무시하는 느슨한 검사는 금지한다. **정확한 기대 Bucket=Ablations를 검사**하고, 다른 63열은 기존처럼 검사한다.

기존 source payload hash와 새 view-policy hash는 구분해서 receipt에 보존한다. `Result_ID`에 탭명/Bucket을 추가해서 새 관측으로 만들지 않는다.

### 수정 C — `VIEW_NAMES`, 고정 ID, 보호 범위, snapshot 갱신

- `VIEW_NAMES`에 네 `*-ablations`를 추가한다.
- 각 데이터셋의 목적지 ID는 2.1절을 사용하며 실행 때 이름과 ID를 동시에 검증한다.
- 새 탭은 **view registry**에 둔다. 기존 `_schemas()`의 “1행 raw header를 가진 source 소유 탭”과 혼동하지 않는다.
- 새 view의 canonical header는 1행이 아니라 **5행**이다. B01 raw source와 같은 `_owned()` 검사를 그대로 적용하지 않는다.
- 보호 탭은 기존 세 탭 외에 `배포용 모델`도 포함한다.
- 기존 `ablations` 전체 보호를 유지한다. 이전 append-only 수정 지시를 구현하거나 실행하지 않는다.
- 작업 snapshot은 **이번 migration 이후 상태를 새로 읽어** 만든다. 9월 30일 snapshot으로 정상적인 1889개 Bucket 변경을 “손상”으로 판정하거나 rollback하지 않는다.
- 이후 업로드 중 다른 legitimate append가 있으면 오래된 전체 snapshot을 복구하지 말고 ID별로 구분한다.

### 수정 D — 향후 다른 업로더도 같은 계약 사용

ABLR2, MAIN-A, GFB20/GFP40, legacy sensitivity uploader가 source에 새 결과를 추가하는 기존 방식은 유지할 수 있다. 현 parser가 해당 source header를 알고 있고 Approach가 위 계약으로 생성되면 자동 반영된다.

필요한 수정은 **화면 확인·receipt·setup·목적지 안내**다. main에 데이터가 나타나지 않는다고 source 업로드 실패로 처리하거나 반복 append하지 않는다.

새 schema/source를 추가하는 별도 작업에서는 Dataset/Bucket/Approach와 parser/union 계약을 명시한다. source 등록 없이 보이는 탭에 직접 쓰는 우회 경로는 금지한다.

## 7. B01 raw/stress/요약의 처리 범위

이 전환에서 아래 기존 탭은 삭제·rename·원본 이동하지 않았다.

```text
_rb01_s1 / _rb01_s3 / _rb01_s5
_rb02_points / _rb_b01_status
RB02-curves / RB-B01
```

RB01 native 개별 관측은 `WV3-ablations`로 이동했다.

RB02는 43열의 **fixed192 ROI shift-point schema**이므로 이번 64열 native 관측표에 억지로 합치지 않았다. 기존 `RB02-curves`에서 계속 비교한다. `RB-B01`도 완료/누락/paired 통계의 진단용 기존 표시로 보존한다. 서로 다른 ROI의 ERGAS나 A_ZERO를 native HQNR/ERGAS와 같은 정의인 것처럼 저장하지 않는다.

향후 요약/curve를 데이터셋별 ablation 화면에 통합해야 한다면 **별도 이름 붙인 section 또는 명시적인 schema 확장**으로 설계해야 한다. 64열 본문 아래에 수동 데이터를 붙여 spill을 막거나, native 빈값을 0으로 채우지 않는다. 이번 서버 수정에서는 이러한 추가 설계를 시행하지 않는다.

이번 snapshot의 B01 native는 s1의 8 Student × 2 selector = 16행뿐이다. s3/s5 원본 탭에는 측정값이 아직 없다. 이는 서버 실험 미실행을 뜻하지 않는다. 자료 회수 시 기존 s1 package를 포함한 누적 evidence로 반영한다.

## 8. 실행 순서 — 서버 코드 담당자

### 8.1 준비

- 현재 reporting code를 별도 작업 branch/checkout에서 수정한다. 실행 중 frozen 컨테이너 안에 pull하지 않는다.
- 현재 live 탭 ID, `_records` A1/A2, 대상 view A1:A6와 metadata를 읽는다.
- 동시 구조 변경 writer가 없는지 확인한다.
- 위 세 가지 호환 문제(A/B/C)를 먼저 수정하고 로컬 테스트를 통과시킨다.
- Credential은 기존 `gspread/account.json`만 사용한다. MD/ZIP/Git/Sheet에 복사하지 않는다.
- Spreadsheet는 제목 검색 대신 pinned ID로 연다.

### 8.2 기존 CLI를 사용한 새 readback

현재 CLI는 `tools/rb_b01_sheets.py`다. 아래 예시는 **코드 호환 수정·테스트 후** 실행한다. Python 경로는 그 서버의 기존 환경을 사용하며 실험 환경 upgrade는 하지 않는다.

```bash
BRIDGE_PYTHON=/path/to/existing/python
OUT=work_dir/_rb_sheet_upload/B01/dataset_ablations_20261001

"$BRIDGE_PYTHON" -B -m unittest discover -s reporting_bridge/tests -v

"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py inspect \
  --output-dir "$OUT/inspect"

"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py prepare \
  --evidence-root work_dir/_rb_sheet_upload/B01/incoming \
  --output-dir "$OUT/prepared"

"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py plan \
  --prepared "$OUT/prepared" --snapshot "$OUT/inspect" \
  --output-dir "$OUT/plan"

"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py upload \
  --prepared "$OUT/prepared" --output-dir "$OUT/upload_dry_run"

"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py verify \
  --prepared "$OUT/prepared" --snapshot "$OUT/inspect" \
  --output-dir "$OUT/verification"
```

**현재 이미 있는 s1 자료만 재검증한다면 새 관측은 0이어야 한다.** 정상화된 view 검증이 실패했다는 이유만으로 `setup --apply`나 재업로드를 반복하지 않는다.

s3/s5의 검증된 새 evidence가 추가됐다면:
1. 기존 s1을 포함해 `prepare`를 새로 실행한다.
2. dry-run에서 기존 source 값 변경 없이 신규 slot만 추가되는지 확인한다.
3. 단일 조정 writer **s1**에서만 `upload --apply`를 실행한다.
4. 다시 `verify`하여 source → `_records` → `WV3-ablations` 및 main/archival 부재를 확인한다.

```bash
# 새 증거가 있고 dry-run이 통과한 경우에만
"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py upload \
  --prepared "$OUT/prepared" --output-dir "$OUT/receipts" --apply
```

이미 정상 설치된 탭을 재생성하는 `addSheet/duplicateSheet`는 이 코드 수정 실행 단계에서 필요하지 않다. 테스트 합성 결과는 실제 Sheet에 올리지 않는다.

## 9. 필수 테스트와 완료 판정

| 검사 | 반드시 통과할 조건 |
|---|---|
| 새 routing 수식 인식 | 현재 `_records!A2`에 `patch_formula`를 적용하면 동일 수식 반환 |
| 재설치 방지 | 이미 있는 네 탭·기존 B01 union·routing의 setup 변경 0 |
| 목적별 분류 | 02/03/04/06/10은 Ablations, 나머지는 원 Bucket 유지 |
| RAW/view 분리 | source Main payload 유지하면서 normalized/view는 Ablations로 정확히 검증 |
| 열 보존 | Bucket 외 63열, header64, Result_ID 불변 |
| B01 visible 검증 | source와 `_records`·WV3-ablations에서 ID당 1회, WV3-main에는 0회 |
| 표시 형식 | A5 header/A6 spill, 고정행열, 숨김 metadata, 0건 처리 |
| 기존 표 보호 | paper/ablations/유의미한결과/배포용 모델에 write 0 |
| Legacy 구분 | G23/MAIN-A/ABLR2/확장 fitting을 하나의 통계 집단으로 합치지 않음 |
| Stress 분리 | RB02 fixed192 및 A_ZERO를 native 64열 관측으로 변환하지 않음 |
| 반복 통계 | selector alias, scene, direction을 seed 수로 세지 않음 |
| 재시도 | 동일 evidence 재전달 신규 관측 0; timeout 후 ID 조회로 확인 |
| 누락 상태 | s3/s5 package 없음은 not collected; failed/not-run으로 조작하지 않음 |
| Frozen code | numerical source/checkpoint/data/metric hash 변경 0 |

완료 보고에는 다음을 포함한다.

```text
코드 commit와 수정 파일
새 view-policy 버전 및 현재 formula SHA
실제 source/native/unique Student/curve/point 수
데이터셋별 main/ablations/archive 행 수
신규 source write / 기존 no-op / conflict 수
source와 normalized/view 각각의 readback 결과
B01 ID의 WV3-main 부재 확인
보호 탭 회귀검사
미회수 server evidence와 pending 목록
```

코드 수정 완료, 기존 s1 표시 재검증 완료, s3/s5 백필 완료, 6-seed 전체 확보는 서로 다른 완료 항목으로 보고한다.

## 10. 근거·동봉 자료

직접 확인한 저장소 코드:

- `reporting_bridge/rb_b01_sheets.py`: `patch_formula`, `VIEW_NAMES`, `_schemas`, `upload`, `verify`.
- `reporting_bridge/rb_b01_contract.py`: canonical header, fixed slots, Result_ID와 cohort identity.
- `reporting_bridge/README.md`: 기존 CLI, 단일 s1 writer, cumulative evidence, credentials/frozen source 규칙.

이 문서와 함께 보관할 파일:

- `routing_contract.json`: 실적용 분류·ID·보호 규약.
- `formula_anchors_before.json` / `formula_anchors_after.json`: 복구·대조용 수식. 현재 코드가 무조건 before를 복구하는 데 사용하지 않는다.
- `migration_verification.json`: 3908관측과 보호 탭 전후 검증.
- `migration_observations.csv`: 변경한 1889개 ID별 원 Bucket·새 Bucket·목적지.
- `records_header.json`: 기존 64열 전체 헤더.

전체 Google Sheets export의 작업 전·후 사본은 별도 보관했다. XLSX에서는 Google 전용 수식이 호환성 wrapper로 저장될 수 있으므로, 수식 배포의 기준은 **live Google formula 및 after JSON**이다. Excel에서 저장한 파일을 다시 import하여 운영 Sheet를 통째로 덮어쓰지 않는다.

**최종 요약: 시트 이전은 완료됐다. 서버에서는 기존 source upsert를 유지하면서 새 Ablations 라우팅을 이해하고, 검증 목적지를 dataset-ablations로 바꾸면 된다. 논문용 `ablations`에는 아무것도 쓰지 않는다.**
