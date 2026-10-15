# Teacher48H 구현 검토 및 실행 준비 상태

작성일: 2026-10-13 (Asia/Seoul)  
대상: `PANDA_S1_M12_TEACHER48H_S123_20261013_v1`  
기준 계획: [Stage 1 Teacher Target Search](../research_log/PANDA_Stage1_Teacher_TargetSearch_S123_48H_2026-10-13_v1.md)  
실행 문서: [teacher48/README.md](../teacher48/README.md)

## 판정

새 Stage 1의 실험 case·trainer·same-state 평가·목표 통과 모델 보존·새 fixed64 업로드·동일 로컬 기록을 `teacher48/`에 추가했다. **CPU 합성 검증은 통과했으나, 현재 확보된 등록 데이터만으로는 계획의 full-method 실험을 곧장 시작할 수 없다.**

본 작업에서 실제 campaign/GPU 학습을 시작하지 않았고, production Google Sheet에 쓰기 요청을 하지 않았으며, commit도 실행하지 않았다. 아래 테스트의 synthetic optimizer update는 실제 실험 결과가 아니다.

## 1. 구현 범위와 계획 정합성

| 항목 | 구현/검증 내용 |
|---|---|
| 배정 | s1/WV3, s2/QB, s3/GF2만 허용. s4/s5 대상 아님 |
| Case | F00–F15 정확한 양수 계수, 센서별 우선순위, T00–T08 baseline-relative 배율 |
| Fresh | 모든 A/T fresh 초기화; F/T/H 간 weight 전달·old A 묵시적 이식 없음 |
| Scheduler | H50/H100/H150 별도 horizon, 1-index warmup/cosine convention 명시 |
| ε | RR/FR 독립 stream, radius 2 면적 균등 원판, 매 update, RR mean + FR mean |
| 평가 | RR20/FR20 전체, 동일 A/T state·evaluator·inference/prior 규칙 검증 |
| 목표 | unrounded HQNR≥하한 AND ERGAS≤상한; 27/52/77 cadence |
| 보존 | gate 통과 전 weight/qual/임시 evaluation weight를 디스크에 쓰지 않음 |
| 순위 | qualified HQNR 내림차순 → ERGAS 오름차순 → step 오름차순 → run ID; RR-min 예비/Pareto/gap 별도 |
| Budget | 독립48h, 46h 이후 새 시작 금지, 실제 latency·eval·artifact·probe와 20% 여유 |
| 진단 | 고정 AXIS16, 구조 discrepancy, frozen-T 추론 개입, train-only gradient probe |
| 로컬 기록 | 전체 평가·각 scene·실패 부분 기록·상태·원정밀도 sheet payload·receipt |
| 원격 | 새 workbook의 센서별 `*-ablations` fixed64 전용; 이전 `analysis` writer 미사용 |

`panda_method/`의 method 본체를 사용하며, Stage 2/KD/e/q calibration은 이번 runner에 넣지 않았다. HQNR만 높은 미달 state, ERGAS만 좋은 다른 step을 혼합해 저장하는 경로를 차단한다. FR 공식 metric에는 original PAN/native LMS를 쓰며 training support mask를 이월하지 않는다.

## 2. 코드 검토 중 확인하여 보완한 경계

- 공식 평가, trainer, 진단에서 같은 모델의 logical state SHA를 일관되게 사용하도록 연결했다.
- metric aggregate와 per-scene 평균을 함께 검증한다. RMSE/CC/JQM과 Q4/Q8도 required metric이다.
- 평가 중 오류가 나더라도 완료된 scene과 실패한 step/selector를 먼저 기록한다. 일부 scene의 결과를 완결 평가로 승격하지 않는다.
- 진단 JSON은 큰 BI 셀을 넘지 않도록 숫자를 버리지 않고 scope별로 분할한다.
- 고정 phase 예산 때문에 F00 baseline 자체가 불필요하게 배제되지 않도록 보호한다.
- 실제 run의 update P90, 평가/정성/probe 시간을 다음 admission·마감 reserve에 반영한다.
- 같은 official scene을 다른 ID로 반복하는 file-row/내용 중복, train/eval crop 겹침, source 파일 사후 변경을 점검한다.
- prepared patch의 증강을 prior에 반영하되 이미 저장된 영상을 두 번 증강하지 않는다.
- summary의 numeric qualification과 실제 artifact availability/QUAL_COMPLETE를 구분한다.
- seed 통계는 성공 seed만 남기는 방식이 아니다. 완결 seed의 미달 결과도 평균에 포함하고 실패·중단 seed를 별도로 표시한다. selector 개수나 technical retry를 독립 seed로 세지 않는다.

수치상 통과했어도 저장 실패 또는 `QUAL_PENDING`이면 사용 가능한 완성 Teacher라고 표시하지 않는다. 실패 모델의 exact disk resume를 지원한다고 주장하지 않는다.

`--resume-campaign`은 기존 48h clock과 로컬 장부를 복구하는 명시적 경로다. 미달 weight를 재개하는 대신 같은 seed의 새 attempt로 fresh 재시작하며, 만료된 budget을 초기화하지 않는다.

## 3. 새 Sheet 계약 적용

계획 원문의 구 `pan-cvpr27/analysis` 요구보다 이후 사용자 지시와 [gspread_rules](../research_log/gspread_rules/FIXED_COLUMNS_AND_GSPREAD_HANDOFF_KR.md)를 우선했다.

- 새 workbook: `pan-cvpr27-new` / `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE`.
- s1 → `WV3-ablations` (`261001010`).
- s2 → `QB-ablations` (`261001011`).
- s3 → `GF2-ablations` (`261001012`).
- 공통 `ablations`, `paper`, `유의미한결과`, `*-main`에는 자동 업로드하지 않는다.
- `PAN_FIXED64_V1`, reset epoch `2026-10-13`, A5:BL5 literal header, 64열, sheet ID를 검사한다. 불일치하면 쓰기만 실패하며 원격 헤더를 수리하지 않는다.

공식 평가마다 전체 평균과 RR/FR 각 scene, 목표 미달·중단·실패 상태를 빠뜨리지 않고 outbox에 담는다. 학습 매 update 로그는 로컬에 남기고 약 1시간 요약만 전송한다. RMSE·CC·JQM은 전용 metric 열에 들어간다. 나머지 method-specific 기록은 BI JSON에 보관하고 기존 q_ref/tau_R/alpha/beta/lambda_E를 재사용하지 않는다.

원정밀도 숫자를 `RAW`로 저장하고 화면 표시만 metric/시간 네 자리로 적용한다. null은 공란이지 0이 아니다. 같은 `Result_ID`+payload는 재시도해도 중복 append하지 않으며, 다른 payload는 conflict다. readback 후에만 `VERIFIED`를 기록한다.

`records.jsonl` → durable `outbox` → 별도 publisher → 원격 readback → `receipts` 순서다. 모든 원본 payload를 로컬에 보존한다. 따라서 네트워크 오류가 있어도 학습이 원격 응답을 기다리지 않는다. **이 구현 검증에서 production workbook/header/권한/readback을 실제 확인한 것은 아니다.** 관련 테스트는 fake transport를 사용했다.

## 4. 기존 자산을 직접 읽어 확인한 결과

`teacher48.asset_audit.discover_existing_assets()`로 기존 `ablr2/sensor_sources.json`과 실제 등록 H5 header, 로컬 TA2 config를 read-only 조사했다. 전체 tensor나 파일 전체 SHA를 다시 계산한 작업은 아니며, 전체 머신 검색에 의한 자산 부재 증명도 아니다.

| 센서 | 학습 수 | 학습 PAN/MS | RR 평가 | FR 평가 |
|---|---:|---|---|---|
| WV3 | 9,714 | 64² / 16², 8 bands | 20장, PAN256²/MS64²/GT256² | 20장, PAN512²/MS128², GT 없음 |
| QB | 17,139 | 64² / 16², 4 bands | 20장, PAN256²/MS64²/GT256² | 20장, PAN512²/MS128², GT 없음 |
| GF2 | 19,809 | 64² / 16², 4 bands | 20장, PAN256²/MS64²/GT256² | 20장, PAN512²/MS128², GT 없음 |

세 센서의 기존 등록 catalog에는 `fr_train`이 없고, RR train H5에는 실제 source-scene/crop/acquisition 연결이 내장되어 있지 않다. 승인된 source-scene prior cache와 train/test 근접 crop 분리 증거도 새 manifest로 등록되어 있지 않다. 기존 protocol은 이 지리적 분리 근거의 한계를 이미 명시하고 있다.

따라서 지금 상태는 세 lane 모두 **`BLOCKED_DATA_PROTOCOL`**이다. 이는 code syntax/test 실패라는 뜻이 아니라 새 방법의 실데이터 전제 조건이 아직 충족되지 않았다는 뜻이다.

금지되는 대체:

1. benchmark FR20을 ε 학습용 FR로 재사용.
2. RR64/MS16을 확대해 native FR training이라고 표기.
3. H5 patch index를 실제 source scene ID로 재명명.
4. 증거 없이 source crop/phase/prior/near-duplicate 검증 JSON을 채움.

상세 조사 결과: [2026-10-13_teacher48_existing_assets.json](2026-10-13_teacher48_existing_assets.json).

## 5. baseline에서 회수한 값과 아직 결정하지 않은 값

이 서버의 실제 WV3 TA2 config:

`work_dir/aligner_teacher_lms_v2/s1/runs/ATL2_WV3_S1_R1_SS261006101_M00_F50K_v2/config.json`

확인된 recipe는 W112/D123/P0, bicubic MS base, AdamW, betas `(0.9,0.999)`, eps `1e-8`, U LR `1e-4`, A LR `1e-5`, WD `0.01`, warmup 100, RR effective batch 48/microbatch12/accumulation4, FP32다. QB/GF2에 대해서는 이 서버에서 같은 수준의 실제 TA2 run config를 발견하지 못했으므로 과거 공통 계획의 값과 실측 실행 증거를 구분한다.

이전 모델 weight를 이식하지 않는다. 회수한 것은 architecture/recipe metadata이며 새 scene-conditioned A와 T는 fresh 초기화한다.

새 FR native support·batch, absolute-gradient ROI, low-confidence prior 수용, output display/probe ID·시간 allowance는 아직 실데이터 근거로 pin해야 한다. 기존 NCC ROI=16이나 A margin=0을 새 loss의 승인값으로 자동 승계하지 않는다.

새 warmup/cosine은 **1-index update의 warmup endpoint=peak, 최종 horizon=min LR** convention이다. 과거 zero-based scheduler와의 차이를 숨기지 않고 `scheduler_indexing`으로 명시적으로 승인하도록 했다. draft baseline은 실행 완료 증거가 아니다.

## 6. 실행한 검증과 미검증 범위

본 보고서 작성 시 실행한 명령:

```bash
/home/knuvi/miniconda3/envs/pancrafter/bin/python -m unittest discover -s teacher48 -t . -q
```

결과: **120 tests, OK**. campaign 복구 및 재시작 시 status Result_ID 충돌 방지까지 포함한 최종 통합 테스트를 통과했다.

별도 method 회귀 검증 `panda_method` **56개**, 기존 `ta2.test_model_loss` **19개**도 통과했다. 합계 **195개 PASS**이며 compileall과 세 서버의 정적 catalog/코드 일치도 확인했다. 설치된 공식 evaluator identity를 읽는 명령도 정상 완료했다. 실제 dataset 수치를 재평가한 것은 아니다.

포함하는 검증은 registry/seed/cadence, strict target/동일 state/전체 metric, 미달 weight 미저장, qualified round-trip, microbatch mean 동등성/gradient isolation, RNG 보존, actual source manifest 계약, augmentation/prior, full-precision metric source, fixed64/no-op/conflict/readback 모의 실행, runtime local-first·부분 평가 실패·numeric selector 기록이다.

아직 수행하지 않은 검증:

- 실제 source-scene/native FR/prior 자산으로 admission 통과.
- 실제 GPU 300-update timing 및 48h 캠페인 실행.
- 실제 모델을 사용한 RR20/FR20 공식 수치 산출·목표 달성.
- 실제 production Sheet의 권한/schema 검증 및 upload/readback.
- 실행 서버별 최종 environment/source/evaluator provenance pin.

따라서 “실험을 돌려 성능을 확인했다”, “세 서버의 Sheet에 업로드됐다”, “목표 Teacher를 확보했다”고 결론내릴 단계는 아니다.

## 7. 다음에 필요한 입력

1. 각 서버가 접근 가능한 실제 training-only native FR observation 자산.
2. RR train과 평가 목록의 source scene/acquisition/crop·증강 provenance.
3. accepted prior cache, estimator/sign/unit/phase·RR/FR grid 변환과 confidence 정책 증거.
4. 실제 sample 목록에 결합된 train/eval 겹침·근접 crop audit.
5. 위 자산을 바탕으로 FR shape/batch·ROI/probe·scheduler 결정을 채운 baseline 및 `TEACHER48_SOURCES_V1` manifest.

이 입력이 갖춰지면 `python -m teacher48 run ...`이 admission과 독립48h campaign, 별도 publisher를 시작한다. `admit`를 따로 실행하는 중간 절차는 필수가 아니다. 단, 데이터 전제 조건을 제거하면 계획과 다른 실험이 되므로 이를 우회하지 않는다.
