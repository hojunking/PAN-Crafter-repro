# PAN 새 시트 — 조건 충족 step 전부 + run별 HQNR 대표값 게시 지시
작성일: 2026-10-14  
게시 정책: `PAN_PASSED_STEPS_OR_BEST_HQNR_V1`  
컬럼 계약: `PAN_FIXED64_V1` / A:BL 64열 / 헤더 5행 / 결과 6행부터

## 0. 이번 변경의 목적과 범위
현재 학습은 중단하지 않는다. 학습·평가·checkpoint 저장 정책이 아니라 **Google Sheets 게시 정책**을 변경한다.
기존 `REPRESENTATIVE_ONLY_UPLOAD_AMENDMENT_KR.md`의 “run당 무조건 1행” 제안은 이 문서로 대체한다.

- 같은 run의 정식 전체 평가 중 HQNR 하한과 ERGAS 상한을 동시에 만족하는 **모든 평가 step**을 게시한다.
- 그 run에 충족 step이 전혀 없으면, 정식 전체 평가의 **평균 HQNR가 가장 높은 step 한 개**만 게시한다.
- 각 게시행의 FR 및 RR은 동일 step·동일 A/T 모델 상태의 전체 평가값이어야 한다.
- 장면별 결과, 학습 loss/gradient trace, preflight, diagnostic, 상태 이벤트를 결과표에 append하지 않는다.
- 모든 원시 로그·미달 결과는 서버 JSON/JSONL/outbox 및 정리 전 백업에 보존한다. 데이터 삭제나 원하는 결과만 남긴 통계로 해석하지 않는다.
- 현재 64열 헤더/순서/명칭, tab ID는 변경하지 않는다. 숨김 원본 탭이나 새 분석 탭을 자동 생성하지 않는다.

## 1. 목적지와 임계값
Spreadsheet ID: `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE`  
https://docs.google.com/spreadsheets/d/1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE/edit

| 서버 | Dataset | 게시 대상 | sheetId | HQNR 최소 | ERGAS 최대 |
|---|---|---|---:|---:|---:|
| s1 | WV3 | WV3-ablations | 261001010 | 0.960 | 2.045 |
| s2 | QB | QB-ablations | 261001011 | 0.923 | 3.500 |
| s3 | GF2 | GF2-ablations | 261001012 | 0.965 | 0.522 |

`h >= h_min AND e <= e_max`를 원 수치에 적용한다. 반올림 후 판정하거나 OR 조건으로 바꾸지 않는다.
WV2는 이번 임계값이 지정되지 않았다. WV3 기준을 추정해서 쓰지 않는다.
현재 EdgeRes 실행 경로의 ablations 목적지는 그대로 유지한다. 합격이라는 이유만으로 main에도 중복 게시하지 않는다.
후속 main 캠페인이 별도로 지정될 때에도 같은 컬럼/게시 원리를 적용할 수 있지만, 자동 목적지 변경은 하지 않는다.
`paper`, 논문용 `ablations`, `유의미한결과`는 자동 쓰기 금지다.

**중요:** WV3의 기존 학습·artifact qualification은 HQNR≥0.958, ERGAS≤2.040이었다.
이번 게시 기준은 HQNR≥0.960, ERGAS≤2.045로 서로 다르다.
이번 요청만으로 live training config, pruning, qualify_evaluation, weight retention 기준을 수정하지 않는다.
새 게시 판정은 원본 `joint_pass`를 재사용하지 말고 별도로 계산한다.
게시 조건 통과와 checkpoint 파일 존재 여부는 독립된 정보다.

## 2. 무엇을 평균내고, 무엇을 선택하는가?
현재 `edgeres48/evaluation.py::official_evaluation()`은 같은 모델을 고정하여 다음을 수행한다.

1. RR 20장 각각에서 지표를 계산한다.
2. FR 20장 각각에서 지표를 계산한다.
3. 각 split·각 metric별 20개 값을 산술평균하여 `EVAL_SUMMARY`를 만든다.
4. 평가 전/후와 RR/FR 평가 후의 모델 상태가 동일함을 검증한다.

즉 HQNR(t)=FR 20장 HQNR의 평균, ERGAS(t)=RR 20장 ERGAS의 평균이다.
RR와 FR의 40장 숫자를 한 평균으로 합치지 않는다.
PSNR 등도 원 evaluator가 이미 집계한 결과를 그대로 사용한다. 평균 RMSE로 PSNR를 다시 계산하지 않는다.
HQNR도 평균 D_s·D_lambda로 재구성하지 말고 원본 HQNR 평균을 사용한다.

**게시 선택 순서: 장면별 평가 → step별 전체 평균 → run별 조건 판정/대표 선택.**
기존 시트에 섞여 있는 요약 1행+장면 40행을 통째로 다시 평균내면 안 된다.
장면마다 가장 좋은 step을 골라 평균내거나, HQNR 최고와 ERGAS 최저를 별도 step에서 가져오면 안 된다.
여러 seed/run을 평균내는 통계는 별도 작업이며 이번 게시행의 뜻이 아니다.

## 3. 유효 후보의 최소 검증
현재 원본 형식에서는 `Eval_scope == "EVAL_SUMMARY"`만 후보로 삼는다.
`PREFLIGHT_EVAL`, `TRACE_100_UPDATE_SUMMARY`, `RR_SCENE:*`, `FR_SCENE:*`,
`DIAGNOSTIC_LEAN`, `STATUS`, `CAMPAIGN_AGGREGATE`, `QUAL_ARTIFACT`는 지표 후보에서 제외한다.

각 후보는 다음을 확인한다.
- HQNR, D_s, D_lambda, JQM 및 RR 8개 지표의 numeric/finite 여부.
- HQNR∈[0,1], ERGAS≥0. 빈 값이나 문자열을 0으로 채우지 않는다.
- `same_state_verified`, `metrics_complete`, `metrics_finite`, `full_method_eligible`가 모두 true.
- RR/FR scene completeness가 각각 expected=20, completed=20이며 missing/duplicate/unexpected가 없다.
- Run_ID, seed, attempt, selected_step가 evaluation identity 및 resolved config와 일치한다.
- model_state_sha256 및 evaluator protocol identity가 유효하다.
- 같은 run/attempt에서 config 또는 평가 protocol이 달라지면 충돌로 보고한다. 조용히 합치지 않는다.
- 공식 전체 평가가 없는 run은 metric 표에 빈 행을 만들지 않는다. 미평가/실패 상태는 로컬 report 및 상단 요약에 기록한다.

장면별 행으로 요약을 복원해야 하는 예외 상황에서는 같은 run/attempt/step/model-state/protocol의
정확한 20+20개 고유 scene만 모아 원 evaluator와 동일하게 집계한다.
이번 정리에는 이미 존재하는 정식 summary를 사용했으므로 그런 복원은 필요 없었다.

## 4. run별 선택과 중복 방지
run key는 최소한 `(Campaign_ID, Dataset, Server, Run_ID, Attempt)`다.
Case명이 같아도 seed/반복/run이 다르면 별개의 실험이다.

```python
valid = validated_whole_dataset_summaries_for_one_run
passed = [x for x in valid
          if x.hqnr >= h_min and x.ergas <= e_max]
if passed:
    publish = sorted(deduplicate_same_state_step(passed), key=lambda x: x.step)
    selection = "PASS_ALL_STEPS"
else:
    publish = [] if not valid else [
        min(valid, key=lambda x: (-x.hqnr, x.ergas, x.step, x.source_result_id))
    ]
    selection = "BEST_HQNR_FALLBACK"
```

동률은 HQNR 원값 동일 → ERGAS 낮음 → 이른 step → source Result_ID 순서로 결정한다.
같은 run/step에 다른 모델 상태가 있거나 동일 모델에서 metric이 서로 다르면 충돌로 보고한다.

- 충족행: run key + eval protocol + step을 안정적인 표시 ID에 포함한다.
- 대체 대표행: run key + eval protocol + `"FALLBACK"`을 사용한다. 선택 step/state는 표시 ID에 넣지 않는다.
- 더 높은 HQNR가 나오면 **같은 fallback 행을 갱신**한다.
- 첫 충족 step이 나오면 fallback 행을 제거하고 충족 step 행으로 교체한다.
- 충족 step이 생긴 run에는 미달 best를 추가로 남기지 않는다.
- 목표 통과 run에서 나중 step이 미달이어도 과거 충족 step은 유지한다.
- 재시도는 같은 source/표시 ID를 확인한 후 처리한다. 맹목 append 금지.
- 원시 EGR2 record ID는 변경하지 않는다. 표시행에 별도 `VIEW-...` ID를 사용하고 source ID를 BI에 보존한다.
- 정리된 표를 다시 후보 원본으로 읽지 않는다. **불변 로컬 전체 원시 기록**으로 매번 대상 집합을 재구성한다.

## 5. 읽을 수 있는 표시 — 컬럼 추가 금지
A열 Experiment는 긴 Run_ID 대신 짧은 설정 요약을 쓴다.

예:
`EF01 · abs=1 ε=3e-4 · R1 | U/A=1e-4/1e-5 · 50K`

- EF01: case 구분.
- abs: 절대 정합 loss의 λ_abs.
- ε: 이동 일관성 loss의 **λ_epsilon**. 합성 이동의 반경 ε와 혼동하지 않도록 헤더/메모에 명시한다.
- U/A: 재구성 네트워크/Aligner의 peak learning rate.
- R1: 반복.
- 50K: 계획 학습량. 현재 선택 step이나 실제 완료 update와 같다는 뜻이 아니다.
- ET profile, edge radius, optimizer, batch, width/depth가 달라지면 실제 변경값을 A열 또는 A셀 메모에 표시한다.
  짧게 만들기 위해 값의 차이를 누락하지 않는다. 표시명은 row ID로 쓰지 않는다.

C열 Selection@Step:
- `충족 · 28,000`
- `대표(미달) · 24,000`

`OBSERVED_000...`, `TRAIN_TRACE_000...` 같은 이벤트 번호는 사용자 표시에서 없앤다.
정수 selected_step은 AO, 원본 selector/source ID는 BI에 보존한다.
AQ에는 `PASS_ALL_STEPS` / `BEST_HQNR_FALLBACK`을 쓴다.
D:O의 모든 metric은 선택된 **동일 원본 summary**에서 복사한다.
AD Run_ID, AF Repeat, AG Seed, AH Attempt, AP Checkpoint_SHA와 provenance 열은 유지한다.
없는 cost 수치는 공란으로 유지한다. 다른 step/다른 GPU의 비용을 임의로 가져오지 않는다.

A셀 메모와 BI JSON에는 최소한 다음을 기록한다.
- 실제 resolved hyperparameters, arch, planned horizon, optimizer/scheduler, batch, precision.
- 원 raw source record ID, evaluation step, logical model-state SHA, evaluator ID.
- 새 게시 임계값과 선택 이유, 전체 후보 수, 충족 수.
- `weights_saved`와 checkpoint 파일 SHA/경로의 실제 증거.
- 로그 누적의 마지막 관측 시각과 run 상태. 현재 실행 중인지는 서버에서 확인한 경우에만 단정.
- 별도 정성 결과나 파일이 없는 경우 이를 명시.

같은 상태의 artifact metadata를 QUAL_ARTIFACT 등에서 결합할 때에는 model-state 및 파일 SHA를 검증한다.
다른 step의 checkpoint를 선택된 행에 붙이지 않는다.
기존 fixed64의 옛 방법 전용 q_ref/tau_R/alpha/beta/lambda_E 열은 새 방법에 맞게 임의 재해석하지 말고 공란으로 둔다.

## 6. 현재 Sheet 정리 내역과 인계 시 주의
2026-10-14 00:54:17 UTC에 별도 백업을 만들었다.
https://docs.google.com/spreadsheets/d/13wAyW6CyEqUkEchSxicKQiDFJuW934q3uCd_gR46aT4/edit

백업 snapshot:
- WV3-ablations 9,369개 이벤트 / 4개 run / 정식 summary 87개.
- GF2-ablations 14,462개 이벤트 / 6개 run / 정식 summary 140개.
- 총 summary 227개. 새 기준 충족 step은 0개.
- 따라서 WV3 A6:BL9의 4개, GF2 A6:BL11의 6개 대표행을 만들었다.
- QB의 등록 결과는 없었다. 수치를 임의로 채우지 않았다.
- 원본 227개 summary의 2,724개 지표 평균을 per-scene 원기록과 대조했다.
  각 split 20장이고, 최대 절대 오차 1.43e-14 이내로 일치했다.
- 원시 기록은 백업·XLSX에 보존했다. paper, 논문용 ablations, 유의미한결과 및 빈 main의 수치는 건드리지 않았다.

**중요: 기존 publisher가 계속 쓰고 있었다.**
백업 이후 도착한 이벤트는 데이터 유실 방지를 위해 계속 지우지 않았고, AQ 기준 임시 basic filter로 가렸다.
따라서 현재 보이는 10행은 검증된 snapshot이며 새 평가를 자동 선택하는 코드가 이미 배포된 것이 아니다.
필터 해제나 원본 API 조회에서는 남아 있는 후속 raw 이벤트가 보일 수 있다.
구 publisher 종료·새 projector 전환 후 마지막 raw tail을 보존하고 전체 로컬 evidence로 다시 계산해 물리적 빈 행을 정리해야 한다.

임시 필터:
`=OR($AQ6="PASS_ALL_STEPS",$AQ6="BEST_HQNR_FALLBACK")`
범위 A5:BL, 종료 행 미지정.
새 projector가 위 AQ 값을 사용해야 숨겨지지 않는다.
헤더 보호는 유지한다. 표의 64열 header digest:
`bfc7f73fd406f447edb9721823e2c1df85540f7068e15647947cd0d2e83a11b1`.

## 7. 돌아가는 학습을 보존한 배포
현재 확인한 코드:
- `edgeres48/__main__.py`: training과 publisher가 별도 프로세스이며 `publisher_process.json`에 pid/command 기록.
- `edgeres48/runtime.py::code_pin()`: `edgeres48/*.py`, 모델/warp/평가 파일 등을 hash로 감시.
- `verify_code()`가 live 코드 변경을 감지하면 실행을 중단한다.

따라서 **지금 frozen runtime의 reporting.py/runtime.py/plan.py를 직접 수정하지 않는다.**
새 projector는 별도 디렉터리, 예를 들어 `sheet_publish_v2/`에 구현하고 model/runtime을 import하지 않는다.
`git pull`로 active checkout의 감시 파일을 통째로 변경하는 것도 금지한다.
학습 objective·seed·stream·optimizer·checkpoint·경과 시간은 그대로 둔다.

실행 순서:
1. 해당 서버의 campaign output과 `publisher_process.json`을 읽고 실제 pid의 전체 command를 대조한다.
2. **구 publisher 프로세스만** 정상 종료한다. training PID, process group, Docker 전체 종료 금지.
   PID가 재사용되었거나 command가 다르면 종료하지 않는다. `pkill -f edgeres48` 금지.
3. 외부 supervisor가 publisher를 재기동하는지도 확인하여 해당 publisher 서비스만 교체한다.
4. 로컬 outbox/JSONL/ack와 현재 시트 tail을 백업하고 source ID/개수를 기록한다.
5. 이미 ack된 원시 파일까지 포함하여 전체 EVAL_SUMMARY를 읽는다.
   `pending()`만 읽으면 과거 best 후보를 잃으므로 금지한다.
6. 새 projector를 dry-run하여 후보수/제외사유/선정 step/표시행을 출력한다.
7. 기존 표시용 VIEW 행과 raw tail의 경계를 확인한다. raw는 백업한 후 삭제/압축한다.
   같은 캠페인 외의 데이터나 알 수 없는 외부 행은 지우지 않는다.
8. 64열 헤더를 readback 검증한 뒤, 해당 서버 소유 tab의 6행부터 결과를 빈 행 없이 기록한다.
   `values.update(..., value_input_option="RAW")` 또는 typed `updateCells.numberValue`를 사용한다.
   plain paste/USER_ENTERED로 숫자·날짜·JSON의 뜻이 달라지지 않게 한다.
9. readback으로 12개 metric·selected_step·SHA/protocol·ID·표시명과 행 수를 검증한다.
10. 새 projector polling 시작. 동일 view hash이면 write를 생략하고 API 오류는 bounded retry한다.
11. 학습 update가 계속 진행 중이며 code_pin이 불변인지 확인한다.
12. 마지막 전체 raw를 별도 보존한 후 빈 꼬리 행을 줄인다. **게시 중에 rowCount를 축소하지 않는다.**
    이후 필요한 경우 행만 확장한다. 컬럼/헤더는 변경하지 않는다.

실제 명령/서비스명은 서버의 출력 경로와 프로세스를 확인해 구현한다.
이 문서는 SSH로 서버를 변경하거나 새 publisher를 이미 기동했다는 완료 기록이 아니다.

## 8. 테스트와 완료 조건
다음 테스트를 통과해야 한다.
- 세 센서 모두 임계값 경계의 >= / <= 포함.
- 둘 중 하나만 통과하면 미달 처리.
- 충족 step 3개이면 3행, best 1개만 남기지 않음.
- 충족 없음 → 최대 HQNR 1개; ERGAS 최저점과 혼동 금지.
- fallback step 갱신 시 행 수 불변; 첫 충족 이후 fallback 제거.
- 서로 다른 run/seed의 결과 보존; 실패 run을 성공 run으로 치환 금지.
- 장면·trace·preflight·partial 평가 제외; 누락 지표 0 채우기 금지.
- 같은-state 20+20개 검증; 서로 다른 step의 FR/RR 결합 차단.
- 동일 재전송 멱등; 충돌 데이터 조용히 덮어쓰기 금지.
- 새 threshold가 원본 `joint_pass`와 달라도 새 표시 판정 정확.
- checkpoint 미저장 결과도 측정값으로 표시 가능하되 파일 존재를 허위 표기하지 않음.
- 64열 헤더/순서와 다른 tab의 values/formulas 불변.
- polling 두 번 이상에서 raw/trace/장면행이 더 이상 게시되지 않음.
- 구 publisher 종료 후 training update 지속 확인.

반환할 보고서:
`publisher_pid/command`, source digest/count, run별 candidate/pass/kept 수,
선택된 `(run,step,HQNR,ERGAS,state SHA)`, 제외사유, destination ranges,
readback 결과, 새 projector 코드 revision, code_pin 불변/학습 update 진행 증거.

## 9. 동봉 파일
- `select_qualified_steps.py`: 네트워크/학습 런타임과 분리된 순수 선택 로직. 현재 EGR2 원시 summary 형식용.
- `validated_eval_summaries.jsonl`: 정리 snapshot의 227개 원본 전체 평가행.
- `display_rows.json`: 생성한 표시행. 향후 원시 입력으로 쓰지 않는다.
- `selection_audit.json`, `per_scene_mean_check.json`, `selector_test_results.json`: 검증 근거.
- `pan-cvpr27-new_before_cleanup.xlsx`: 정리 전 백업. 전체 ZIP에는 용량을 줄이기 위해 별도 보관할 수 있다.

## 10. 근거 코드
https://github.com/hojunking/PAN-Crafter-repro/blob/main/edgeres48/evaluation.py
https://github.com/hojunking/PAN-Crafter-repro/blob/main/edgeres48/runtime.py
https://github.com/hojunking/PAN-Crafter-repro/blob/main/edgeres48/reporting.py
https://github.com/hojunking/PAN-Crafter-repro/blob/main/edgeres48/__main__.py

2026-10-14에 연결 저장소와 실제 Sheet를 읽어 확인했다.
서버 배포본의 revision은 담당자가 실제 runtime에서 다시 확인한다.
