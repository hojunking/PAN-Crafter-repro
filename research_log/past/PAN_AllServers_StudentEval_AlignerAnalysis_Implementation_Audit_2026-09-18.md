# AllServers StudentEval / AlignerAnalysis 구현 감사 — 2026-09-18

**판정: 추론 모드의 핵심 수학은 통과했다. 전수 평가 → 업로드 확인 → s1 분석 → 승인된 본 실험 복귀의 전체 구현은 계획서 기준 미완료다. 현재 상태로 전체 실험을 실행하고 완료 판정하는 것은 권하지 않는다.**

대상 계획은 `PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md`, protocol은 `PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918`이다. 기준 commit은 `0333bcde47e01bed33cf9f85a1d33ab3e47edccc`. 감사 시작 2026-09-18 01:57 KST의 로컬 checkout과 저장된 s1 산출물을 검토했다. 원격 s2–s5 설치본이나 실제 Google Sheet를 검증했다는 의미는 아니다.

감사 도중 02:11경 별도 작업으로 `tools/noa_eval_switch.sh`가 추가되고 `CLAUDE.md` 및 구현 노트가 수정됐다. 이 추가분도 별도 snapshot으로 보존해 검토했다. 아래 핵심 Python 구현은 처음 고정한 source와 같았다. 195개 회귀검사 결과는 최초 snapshot 기준이고, 새 wrapper는 별도 격리 재현으로 검사했다.

02:14경 K51에도 wrapper 파일 존재·shell syntax·문서 문자열 검사가 추가됐다. 추가 조건은 별도로 모두 통과시켰다(`late_unit_addition_results.json`). 운영 동작을 확인하는 검사는 추가되지 않아 아래 판정은 그대로다.

운영 코드·config·checkpoint·큐·hold·Sheet를 변경하지 않았다. 새로 만든 것은 이 보고서와 [감사 증거 폴더](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18)뿐이다. 오류 재현은 `/tmp` 파일과 모의 inference/Sheet를 사용했고, 실제 checkpoint의 추가 추론은 CPU에서 RR/FR 각 첫 장면만 실행했다.

**확인된 정상 구현**

실제 `S1_G23 / seed1234 / step31310`, checkpoint SHA `b3bb5d615b15c3854ccd09889227fdfaaee4e3554f8a5816ec1192ba595ab09f`를 full strict load하여 확인했다.

| 검증 | RR 256 | FR 512 |
|---|---:|---:|
| A_ON: aligner / warp / grid_sample | 1 / 1 / 1 | 1 / 1 / 1 |
| NOA: aligner / warp / grid_sample | 0 / 0 / 0 | 0 / 0 / 0 |
| ZERO: aligner / warp / grid_sample | 0 / 1 / 1 | 0 / 1 / 1 |
| CROP64_MED: aligner / warp / grid_sample | 16 / 1 / 1 | 64 / 1 / 1 |
| NOA와 독립 계산 `M + backbone(raw PAN, lpan, LRMS, 1)`의 최대 절대 오차 | 0 | 0 |
| crop offset의 NumPy 성분별 median과 구현 결과의 최대 절대 오차 | 0 | 0 |

U-Net은 전체 프레임과 원 LRMS를 받았다. aligner view는 ON에서 248²/504², 64 crop에서 56²로 margin4가 한 번 적용됐다. 파라미터·버퍼 state hash 불변, gradient 없음, sampler 복원도 확인했다. ZERO−NOA가 이번 입력에서는 0이었지만 일반적으로 bitwise 동일하다는 주장은 하지 않는다.

AXIS16의 반경·축과 q_size의 `1/(2N)`도 맞다. 상수 aligner를 넣은 독립 대조에서 모든 crop의 q_size가 이론값 `0.46875`였고, panel 수는 RR `16+4+1`, FR `64+16+4+1`이었다. 기존 q 가중치 `qref/(qref+q)` 및 U/A 분리 gradient 경로는 현재 코드·기존 회귀검사와 연결된다. 이번 commit이 학습 loss/trainer를 변경하지 않은 것도 확인했다.

기존 회귀검사는 격리 사본에서 **195/195 통과**했다. 다만 아래 운영 오류도 별도로 재현되므로 이 통과를 V01–V18 전체 충족으로 해석하면 안 된다.

**F01 · P1 — 부분 재평가가 서로 다른 정의의 RR/FR를 정상 cache로 결합한다**

근거: [noa_eval.py:126](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:126), [noa_eval.py:207](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:207), [noa_eval.py:240](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:240). 계획 §4.5, §6.4, V05/V07/V12.

재현 순서는 다음과 같다.

1. config v1로 ON/NOA의 RR·FR를 계산한다.
2. 같은 checkpoint에서 config를 v2로 바꾼 뒤 `splits=('rr',)`만 재평가한다.
3. 기존 FR 파일은 남고 `identity.json`은 v2로 갱신된다.
4. 다음 전체 평가가 inference 0회로 캐시를 승인한다. RR는 v2, FR는 v1이다.

RR 데이터의 bytes를 같은 경로에서 변경하거나 device를 CPU→CUDA로 바꿔도 캐시를 재사용했다. RR input/GT hash는 identity에 없고, device는 저장하지만 비교하지 않는다. precision/environment, 실제 backbone·warp·feeder·빌더, `noa_eval.py`의 계산 코드 등도 충분히 묶이지 않는다.

**수정 조건:** split별 identity와 완료 상태를 저장하고, 해당 split이 실제 완료됐을 때만 그 identity를 갱신해야 한다. config/data/evaluator/forward/environment가 달라진 기존 산출물을 새 identity로 덮어 인증하지 말아야 한다. RR 성공 후 FR 실패 시에도 성공한 RR를 검증해 재사용할 수 있어야 한다.

**F02 · P1 — cohort의 checkpoint/source 고정이 실제 평가와 Sheet 기록에 이어지지 않는다**

근거: [noa_eval.py:68](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:68), [noa_eval.py:107](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:107), [noa_eval.py:198](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:198), [noa_upload.py:119](/home/knuvi/Desktop/song/PAN-Crafter/gspread/noa_upload.py:119). 계획 §3.1/§3.3/§3.4/§6.3.

`sheet_checkpoint()`는 실제 Sheet 본 행이나 업로드 manifest와 연결하지 않고 로컬 `best_hqnr → best_val → best_reduced` 순으로 고른다. cohort에는 hash를 저장하지만 `evaluate_run()`은 이를 사용하지 않고 현재 checkpoint를 다시 선택한다. 격리 재현에서 cohort에 A의 SHA가 고정된 뒤 파일을 B로 바꿔도 B의 평가가 정상 진행됐다. status 또한 SHA/step 대조 없이 run ID만으로 완료를 센다.

source server도 training-start manifest가 아닌 현재 `server.txt`로 채운다. 이후 평가 record에는 그 필드 자체가 전달되지 않아 uploader의 `if rec.get('source_train_server') ...` 검사가 사실상 생략된다. 다른 서버에서 복사한 자산이나 source가 미확정인 자산을 거부하는 계약이 성립하지 않는다.

Uploader는 같은 run ID의 모든 행에 같은 값을 쓴다. 같은 run의 step100/step200을 나타내는 두 모의 행에 step100 값이 모두 쓰이는 것을 확인했다. 이 검사는 Google API·행 파서만 모의화했으며 실제 row 선택·쓰기 범위 생성 코드는 그대로 실행했다. 현재 s1 Sheet에 그런 오기록이 이미 발생했다는 뜻은 아니다.

**수정 조건:** source execution manifest와 Sheet main-row checkpoint를 대조한 immutable key를 평가 입력으로 전달해야 한다. source/selector/step/SHA가 미확정이면 `source_unresolved`로 남기고 기록을 막는다. alias도 동일 key일 때만 허용한다. legacy ON 대조에 쓰는 `qrecon24_target_selection.json`의 target ERGAS 역시 legacy best와 step이 같다는 보장이 없으므로 checkpoint를 결합해 비교해야 한다.

**F03 · P1 — 당시 진행 중 run은 완료 후에도 전수 평가에서 빠진다**

근거: [noa_eval.py:98](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:98), [noa_eval.py:368](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:368). 계획 §3.2.

`pending_current_run_id`를 기록하는 코드만 있고, 정상 완료 후 `records`에 한 번 편입하는 코드가 없다. `--all`은 기존 `records`만 순회한다. 현재 run의 두 완료 MAT가 나중에 생긴 fixture에서도 `--all`은 그 run을 실행하지 않았다.

실제 s1의 `cohort.json`도 13개 record 외에 `PAKD50_QRC24_S1_G21_W104_D121_WV3_T0_S3407_FRESH50_v1`을 pending으로 기록하고 있다. 이 run은 정상 완료 시 합류해야 할 실제 누락 위험 대상이다. 감사 시점의 프로세스 상태를 이 파일만으로 단정하지 않는다.

**수정 조건:** 고정 snapshot의 pending run만 completion evidence와 함께 정확히 한 번 편입하고 건수를 갱신해야 한다. `--force-cohort` 전체 재캡처는 이후 새 학습까지 섞을 수 있어 같은 해결책이 아니다. checkpoint 유실 등은 출력 메시지뿐 아니라 persistent blocked/missing 상태로 저장해야 한다.

**F04 · P1 — 완료 조건 없는 hold 해제, dry-run의 실제 해제, 단계 순서 미보장**

근거: [eval_phase.py:119](/home/knuvi/Desktop/song/PAN-Crafter/tools/eval_phase.py:119), [eval_phase.py:131](/home/knuvi/Desktop/song/PAN-Crafter/tools/eval_phase.py:131), [s1_aligner_analysis.py:150](/home/knuvi/Desktop/song/PAN-Crafter/tools/s1_aligner_analysis.py:150). 계획 §2.1–§2.4, V13/V14.

`release`는 required RR/FR, identity, 업로드 확인, s1 분석 완료를 검사하지 않고 hold를 제거한다. 빈 평가 상태에서도 해제됨을 재현했다. `resume --dry-run`도 먼저 `cmd_release()`를 호출하므로 실제 hold가 사라진다. 이후 watcher/waiter가 학습을 시작할 수 있어 dry-run 의미를 위반한다.

단계 문자열은 있지만 전이를 검증하지 않는다. `hold` 기본 phase는 학습 중 여부와 무관하게 `LOCAL_STUDENT_EVAL`이고, 평가 CLI는 현재 case 후처리 완료를 기다리지 않는다. s1 분석 CLI에도 자기 Student 전수 평가·업로드 완료 선행 검사가 없다.

추가된 [noa_eval_switch.sh](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval_switch.sh)는 `main.py --config` 프로세스가 보이면 평가를 거부하는 검사를 제공한다. 그러나 hold의 존재와 현재 case의 후처리 완료를 검증하지 않고, `--release`는 상태 출력 후 무조건 해제한다. `--release --dry-run`도 dry-run 분기보다 release 분기가 먼저여서 실제 hold가 제거됨을 /tmp에서 재현했다. 새 진입점만으로 F04는 해결되지 않았다.

**수정 조건:** dry-run은 모든 mutation 전에 분기한다. phase 전이는 로컬 cohort와 검증된 receipt를 근거로 결정해야 하며, s1만 분석 완료를 추가 조건으로 둔다. 명시적 waiver는 정상 완료와 별도 기록한다. hold와 실제 launch의 case boundary는 공유 잠금으로 보호해야 한다. 다른 서버 완료를 조건으로 추가할 필요는 없다.

**F05 · P1 — 복귀가 저장한 승인 큐 복원이 아니며, 현재 main 연결은 R2와 다르다**

근거: [eval_phase.py:88](/home/knuvi/Desktop/song/PAN-Crafter/tools/eval_phase.py:88), [eval_phase.py:137](/home/knuvi/Desktop/song/PAN-Crafter/tools/eval_phase.py:137), [qrecon24_switch.sh:105](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_switch.sh:105), [gen_pakd50_configs.py:130](/home/knuvi/Desktop/song/PAN-Crafter/tools/gen_pakd50_configs.py:130), [qrecon24_select.py:20](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_select.py:20). 계획 §2.2/§10.1–§10.4/V15/V17.

큐 bytes/hash를 저장하는 부분은 있다. 그러나 resume에서 읽는 것은 사실상 `next_case`뿐이고, 저장한 큐/revision/hash/학습 release를 복원하거나 대조하지 않는다. chain이 없으면 현재 checkout의 `qrecon24_switch.sh`를 실행한다. 이 스크립트는 현재 generator를 기준으로 mandatory/reservation/effective/active/persistent 큐를 다시 쓴다. 저장된 승인 큐가 그 generator와 다르면 다른 편성으로 복귀한다.

현재 generator revision은 `QRC24_ADJ_R1_20260917`이고, 계획 §10의 Narrow R2·공통 recipe lock·지정 410xx seed 연결은 찾지 못했다. `main_recipe_lock_id`는 항상 None으로 저장된다. `MAIN_WAIT_RECIPE_LOCK` 처리도 없다.

selector도 `HQNR9585_RR_v1`, 순서는 **SCC → ERGAS**다. H 하한을 모두 통과한 두 후보에서 E=2.039/SCC=.988보다 E=2.070/SCC=.989를 선택하는 반례를 실제 `rank()`로 확인했다. 계획 §10.2의 ERGAS 우선과 다르다.

이는 NOA commit이 새로 만든 loss 오류가 아니라 **통합 계획 전체 실행에 남은 기존 main 연결의 미구현**이다. 계획 자체도 원격 selector 구현을 인증하지 않는다고 적고 있으므로, 선행 NOA 수치 계산의 수학적 실패와 구분한다.

**수정 조건:** 승인 snapshot/revision/lock과 실제 복귀 queue를 대조하고, 변경 필요 시 명시적인 migration 근거를 남긴다. 현재 switch 재호출만으로 큐 보존을 주장하지 않는다. ERGAS 우선 selector와 완료 판정 ID, R2 pre-lock/lock 후 seed 배정을 함께 연결해야 전체 workflow를 완료로 볼 수 있다.

**F06 · P1 — Sheet 공동 잠금과 업로드 확인이 구현 주장과 다르다**

근거: [noa_upload.py:105](/home/knuvi/Desktop/song/PAN-Crafter/gspread/noa_upload.py:105), [noa_upload.py:135](/home/knuvi/Desktop/song/PAN-Crafter/gspread/noa_upload.py:135), [gspread_upload.py:999](/home/knuvi/Desktop/song/PAN-Crafter/gspread/gspread_upload.py:999), [tools/_upload.sh](/home/knuvi/Desktop/song/PAN-Crafter/tools/_upload.sh). 계획 §6.3/V10–V12.

`.gspread_write.lock`을 잡는 것은 NOA uploader뿐이다. 학습 uploader와 그 shell wrapper에는 이 잠금이 없다. NOA 내부에서도 header 확인·쓰기·row lookup은 잠금 밖에서 수행하고, 마지막 값 쓰기만 잠근다. 따라서 같은 탭의 공동 writer 직렬화나 쓰기 직전 row identity 검증이 보장되지 않는다.

업로드 후에는 API 호출이 예외 없이 반환한 것만으로 `uploaded` timestamp를 기록한다. row/checkpoint/값 readback이나 identity와 결합된 receipt가 없다. 저장을 수행하지 않는 모의 backend에서도 uploaded가 생겼고, 읽기 요청은 header 두 행과 B열 run 이름뿐이었다. 실제 Sheets API가 성공 응답 후 저장하지 않는다는 주장이 아니라, 계획이 요구한 검증 단계 부재의 재현이다.

**수정 조건:** 양 uploader가 같은 잠금을 사용하고, header schema 검증·row 재확인·write·확인까지 계약을 공유해야 한다. checkpoint가 다른 행을 거부하고, 성공 receipt에 cohort/key/탭/행/컬럼/schema와 확인 결과를 결합한다. 새 NOA 범위만 쓰는 방향 자체는 맞으며, 기존 B:W와 X를 의도적으로 덮는 코드는 아니다.

**F07 · P1 — s1 기본 실행에서 필수 T0와 전체 분석 coverage가 빠진다**

근거: [s1_aligner_analysis.py:38](/home/knuvi/Desktop/song/PAN-Crafter/tools/s1_aligner_analysis.py:38), [s1_aligner_analysis.py:152](/home/knuvi/Desktop/song/PAN-Crafter/tools/s1_aligner_analysis.py:152). 계획 §7.1/§7.5, V08/V13.

`--assets` 목록은 Student 3개뿐이다. 주석에는 T0는 `--teacher`로 별도 실행한다고 되어 있지만 parser에 해당 옵션이 없으며 실제로 exit 2다. `--assets` 기본 실행은 Student 3개 × RR scene0–4만 수행하고 FR, 나머지 장면, 네 모드 비교는 실행하지 않는다. 자산이 없으면 skip 후 exit 0이다.

`--run <T0 run>`과 명시적 `--split`, `--scenes 0-19`, `--modes`를 사용하면 일부 필요한 계산을 개별 수행할 여지는 있다. 따라서 underlying model이 T0를 절대로 로드할 수 없다는 지적은 아니다. **문서에 연결된 전체 실행 및 완료 계약이 4개 자산의 필수 panel을 보장하지 않는 것**이 문제다.

**수정 조건:** T0 selected A/U를 포함한 4자산 × RR/FR × 전체 scene × size/probe/네 mode의 요구 목록을 manifest로 만들고 실제 산출물로 완료를 검증한다. 진단용 subset 옵션은 명시적 partial 상태여야 한다. s1의 평가/업로드 이후에만 전체 분석으로 전환한다.

**F08 · P2 — s1 통계가 scene 간 차이를 crop 내 변동으로 섞고 probe 응답 원본을 버린다**

근거: [s1_aligner_analysis.py:69](/home/knuvi/Desktop/song/PAN-Crafter/tools/s1_aligner_analysis.py:69), [s1_aligner_analysis.py:118](/home/knuvi/Desktop/song/PAN-Crafter/tools/s1_aligner_analysis.py:118). 계획 §7.2/§7.3/§7.5.

`summarise()`는 같은 크기의 모든 scene/crop을 합쳐 std/median/p90을 계산한다. 계획은 scene 안에서 요약한 뒤 scene 간 집계다. 각 scene 내부 offset이 완전히 일정하되 두 scene의 dy가 각각 0, 100인 반례에서, 실제 scene 내 crop std=0인데 구현의 표시값은 `57.7350269`였다. 이를 지역별 불안정성으로 읽으면 잘못된 해석이 된다.

probe별 `c_j`와 residual은 합산 후 버리고 최종 q_size만 저장한다. 따라서 반경/방향별 응답, support/boundary 효과를 재분석할 수 없다. scene별 요약·full-consensus 차이도 별도 보존하지 않고 일부 전체 평균만 반환한다. 분석 해석 MD 산출 경로도 없다.

**수정 조건:** raw probe response를 scene/crop/radius/axis key로 보존하고 scene별 통계와 scene 간 집계를 분리한다. 예상 RR/FR shape, margin, support 정책을 identity에 명시하며 실제 shape 불일치는 silent panel 변경 대신 차단한다.

**F09 · P1 — 완료 집계와 재시작 기록이 실제 완료를 나타내지 않는다**

근거: [noa_eval.py:210](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:210), [noa_eval.py:247](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:247), [noa_eval.py:327](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:327). 계획 §2.4/§6.4/§11.3/§12.2.

RR 두 mode만 있는 record에서도 `delta_rr_ergas`가 있으면 `N_pair_verified=1`, 행 상태 `완료`, `ΔHQNR=nan`으로 출력됐다. 같은 cohort checkpoint인지, FR 20장이 있는지, source/evaluator가 검증됐는지 확인하지 않는다.

완료 캐시를 재사용해 재실행하면 `records/<run>.json` 전체를 새로 써서 기존 `uploaded`를 삭제한다. cache branch에는 `seconds`도 전달하지 않아 기존 Eval(h)가 빈 값이 된다. 재업로드 시 이미 있던 provenance 값을 지울 수 있다. `run_modes()`도 같은 record를 쓰므로 s1 분석 단계에서 이미 끝낸 업로드 상태를 잃는다.

**수정 조건:** split 완료, pair 검증, upload 확인을 서로 다른 immutable identity별 상태로 저장·병합한다. 기록 손실 없는 재시작, 누락/실패/waiver 구분 및 실제 required 수에 기반한 완료 집계가 필요하다.

**F10 · P2 — 필수 RR per-scene와 비용·일부 보조 지표 산출물이 누락됐다**

근거: [noa_eval.py:161](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:161), [noa_eval.py:183](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:183), [noa_eval.py:221](/home/knuvi/Desktop/song/PAN-Crafter/tools/noa_eval.py:221), [s1_aligner_analysis.py:109](/home/knuvi/Desktop/song/PAN-Crafter/tools/s1_aligner_analysis.py:109). 계획 §2.4/§6.4/§11.3/§12.1.

RR는 aggregate `_rr`만 저장하며 `per_scene_rr.csv`를 생성하지 않는다. 실제 저장된 s1 ON/NOA 둘 다 파일이 없었다. 따라서 scene별 paired RR 차이를 바로 검증할 수 없다. RR `n=20`도 실제 개수를 검증한 값이 아니라 상수다. FR CSV는 6자리로 반올림하지만 원 metric array는 JSON에 남으므로 그 부분은 복구 가능하다.

기존 FR 보조 `raw_views()`는 scene 단위 HWC/HW를 받는데 전체 scene list/배열을 넘기고, 여섯 번째 positional argument에도 ratio 대신 2047을 넘긴다. 실제 두 mode의 `aux_note`에 `ValueError('too many values to unpack (expected 2)')`가 남았다. 주 HQNR 계산에는 영향을 주지 않는다.

NOA 기본 계산은 diag ledger로 분리하지만 mode별 RR+FR 합계만 재고, 계획의 RR/FR/inference/export/metric/upload/cache 비용 분리는 없다. s1 size/probe 분석은 seconds를 JSON에 적지만 diag ledger에 반영하지 않는다. local summary/analysis MD와 verified upload receipt 산출도 미구현이다.

**기존 195개 검사가 이 문제를 놓치는 이유**

[pakd50_unit_tests.py:1381](/home/knuvi/Desktop/song/PAN-Crafter/tools/pakd50_unit_tests.py:1381) 이후 K51의 운영 검사는 상당 부분 소스 문자열·상수 존재 여부를 본다. 예를 들어 uploader에 `fcntl.flock` 문자열이 있는지만 확인하며 학습 uploader가 같은 lock을 쓰는지는 검사하지 않는다. V13/V14도 금지 문자열이 없다는 검사로 대체했고 실제 runner 상태 전이는 실행하지 않는다. 네 자산이라는 설명과 달리 `len(S1_ASSETS)==3`을 통과 조건으로 둔다.

최초 sandbox 실행은 DataLoader의 로컬 IPC 권한 오류로 완료하지 못했다. IPC가 가능한 격리 CPU 실행의 첫 시도는 임시 사본에서 이전 K50 구현 노트 1개를 빠뜨려 194 pass/1 fail이었다. 그 fixture 문서만 보완한 최종 실행이 195 pass/0 fail이다. 이 환경/fixture 실패를 제품 결함으로 세지 않았다.

| 계획 검사 | 감사 판정 |
|---|---|
| V01–V04, V06, V16, V18 | 실제 Student RR/FR 한 장면씩과 호출 hook/독립 계산에서 통과 |
| V05 | 단일 정상 실행은 같은 ckpt를 쓰지만 cohort/Sheet/cache 연결은 실패: F01/F02 |
| V07 | mode 디렉터리 분리는 통과, 데이터/환경 및 부분 갱신 검증은 실패: F01 |
| V08 | Student 기본 경로는 dataset 전체 순회; s1 기본 coverage와 수량 검증은 미충족: F07/F10 |
| V09 | Δ 부호, 같은 mode의 raw H≥.9585 및 E<2.040 판정 통과 |
| V10 | 자기 탭 선택은 있으나 source 확인·전달 누락: F02 |
| V11 | NOA 새 열 범위 갱신은 있으나 공동 lock/identity/재업로드 보존 계약 미충족: F02/F06/F09 |
| V12 | split cache, upload 상태, 시간의 재시작 보존 실패: F01/F09 |
| V13 | s1 전수 평가/업로드 후 분석 순서 미보장: F04/F07 |
| V14 | 다른 서버 의존 문자열은 없지만 필수 runner 모의 전이·lock 대기 경로 미구현/미검증: F04/F05 |
| V15 | snapshot은 저장하지만 복귀 시 검증·복원 없음: F05 |
| V17 | 평가 코드가 lock을 생성하지는 않으나 기존 lock 식별·보존·복귀 연결 미구현: F05 |

**로컬 저장 산출물의 현재 상태**

저장 cohort는 `s1_20260918-013809`, 완료 대상 13개 + pending 1개다. 감사 snapshot에서 paired record는 G23·1234 한 개, uploaded marker는 없었다. 실제 Sheet 업로드 여부는 별도 확인하지 않았다. 이 checkpoint의 저장 metric은 다음과 같으며 FR 20개 scene product의 평균과 aggregate HQNR는 최대 약 `2.2e-16` 차이였다.

| mode | RR ERGAS | FR HQNR |
|---|---:|---:|
| A_ON | 2.0712336831455342 | 0.9591475525152621 |
| NOA | 2.143477163897772 | 0.9541956405517693 |

이는 기존 저장값의 산술·연결 확인이다. 이번 감사에서 20장 전체를 GPU 재추론한 결과가 아니다. 한 checkpoint 결과로 전체 Student나 T0의 NOA 효용을 일반화하지 않는다.

**수정 후 재검증 순서**

1. F01/F02/F03: 유한 cohort, source·checkpoint 고정, split별 cache identity를 먼저 닫는다.
2. F04/F05/F06/F09: 불완전 평가의 resume 거부, dry-run 무변경, 공동 Sheet lock/readback, 업로드 상태 보존 및 승인 큐 복귀를 runner fixture로 검증한다. s1 분석 중에도 완료한 s2는 자기 다음 case 또는 recipe-lock 대기로 갈 수 있어야 한다.
3. F07/F08/F10: T0 포함 전체 분석 coverage, scene별 통계/원본 probe·RR 결과, 진단 비용과 보고 산출물을 채운다.
4. 먼저 자체 대표 checkpoint 1개로 ON/NOA RR20+FR20 및 업로드 확인까지 검증한 뒤 고정 cohort 전체로 확대한다. 기존 수치 중 identity를 독립 확인할 수 있는 것은 보존하고, 확인할 수 없는 비교만 재계산한다.

**재현 자료**

- [고정 source SHA와 commit](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18/snapshot.json)
- [운영 계약 재현 코드](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18/reproduce_contracts.py), [결과 JSON](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18/contract_results.json)
- [실제 checkpoint·독립 수식 검증 코드](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18/verify_math.py), [결과 JSON](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18/math_results.json)
- [최종 195개 회귀검사 결과](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18/unit_results_complete.json), [실행 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18/unit_run_complete.log)
- [기존 저장 metric 확인](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18/saved_results_check.json)
- [감사 도중 추가된 wrapper·문서의 SHA](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18/late_snapshot.json), [wrapper dry-run 재현 결과](/home/knuvi/Desktop/song/PAN-Crafter/research_log/NOA_Implementation_Audit_2026-09-18/late_wrapper_results.json)
