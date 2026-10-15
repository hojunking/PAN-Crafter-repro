# NOA/A_ON 전 서버 Student 평가 — 구현 노트 (2026-09-18)

계획: `research_log/PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md`
(protocol `PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918`). 이 노트는 **계획을 실행 가능하게 만든 코드·검사·절차**를 적는다. 결과·판정은 여기에 쓰지 않는다.

## 0. 요약

- 학습 정의는 건드리지 않았다 — `kdv/`(qrecon·losses·trainer)·`train_kdv.py`·Teacher·q cache·기존 checkpoint 무변경. 추가한 것은 **추론 모드 분기·평가 실행기·phase 제어·시트 열·s1 분석**이다.
- NOA(`A_BYPASS_RAW`) 는 **aligner 0회 · warp 0회**로 구현했고, 그 0회를 매 평가에서 실제로 세어 검증한다. 영점 warp(`A_ZERO_WARP`)와 다른 모드로 분리했다.
- 첫 실제 검증에서 **paired A_ON 이 기존 공식 저장값과 오차 0**으로 재현됐다(G23 S1234: RR ERGAS 2.0712336831455342, FR HQNR 0.9591475525152621). 평가 경로가 기존 공식 경로와 같다는 증거다(§4.5 `legacy_on_check = match`).
- s1 cohort 는 완료 13 run 으로 고정했다. 한 run 당 두 모드 RR+FR 이 **약 47 초**다(캐시 재사용 시 즉시).

## 1. 추가한 것

| 파일 | 역할 |
|---|---|
| `kdv/eval_modes.py` | 네 모드(`A_ON`/`A_BYPASS_RAW`/`A_ZERO_WARP`/`A_CROP64_MED`) · `CallCounter`(aligner·warp 실제 호출 수) · `sampler_off`(복원 보장) · `consensus_delta`(non-overlap 64 crop, 성분별 중앙값, 짝수는 가운데 두 값 평균) · `state_hash` |
| `tools/noa_eval.py` | cohort 고정 · 같은 checkpoint 의 A_ON/NOA RR·FR 쌍 평가 · identity/캐시 · Δ(NOA−ON)·joint_pass · `legacy_on_check` · ledger 기록 · `--sanity`(§4.4) · `--status`(N_* 보고표) |
| `tools/eval_phase.py` | `hold`/`release`/`resume`/`status` — 진행 중 학습은 두고 **새 학습만** 멈춘다. 복귀 manifest(큐 파일 내용·sha256·revision·다음 미완 case·mandatory/held/reservation) |
| `gspread/noa_upload.py` | 자기 탭의 **NOA 26 열만** 갱신(계획 §6.2 키 그대로). `batch_clear`/`--replace` 없음, run id 매칭, 중복 행 alias 갱신, 학습 uploader 와 같은 flock |
| `tools/s1_aligner_analysis.py` | §7 크기 패널(64/128/256/full) · AXIS16 probe 반응 q_size · 네 모드 비교 |
| `tools/pakd50_unit_tests.py` K51 | V01–V18 회귀 검사 6 건 |

## 2. 모드 정의와 호출 계약

`pa/model.py` 의 `forward(pan, ms, lpan, aligner_enabled, delta_override)` 를 **인자로만** 조종한다. 모델 파일·checkpoint 를 고치지 않는다.

| 모드 | 인자 | aligner 호출 | warp 호출 |
|---|---|---:|---:|
| `A_ON` | 기본 | 1 | 1 |
| `A_BYPASS_RAW` (NOA) | `aligner_enabled=False` + `sampler` 일시 False | **0** | **0** |
| `A_ZERO_WARP` | `aligner_enabled=False` | 0 | 1 |
| `A_CROP64_MED` | `delta_override=median(crop Δ)` | crop 수 | 1 |

`aligner_enabled=False` 만으로는 NOA 가 되지 않는다(그때도 sampler 가 켜져 있으면 warp 를 탄다). `sampler` 는 컨텍스트로 잠시 끄고 **반드시 복원**하며, 복원 여부와 호출 0회를 매번 검사한다. 실제 checkpoint 사전 검사(§4.4) 결과는 `work_dir/_eval_phase/sanity_<run>.json` 에 남는다 — s1 G23 S1234 에서 네 모드 전부 기대 호출 수와 일치, state hash 불변, grad 없음, `ZERO−NOA` 최대 0.0, `ON−NOA` 최대 0.107([-1,1] 스케일).

## 3. 지표·identity

RR 은 기존 공식 경로 그대로다 — `gspread_upload._rr`(= `tools/eval_dlpan` 경로: crop `20:-21`, peak 2047, SCC=SCC.m, SSIM=Gaussian 11×11, RMSE/CC 포함). FR 은 `tools/metrics/eval_fr` 의 `d_lambda_k`·`d_s` 로 논문 `.mat20` 20 장, HQNR 은 **장면별 곱의 평균**이다. 모드별로 crop·reference·clamp 를 바꾸지 않는다.

캐시 identity 는 checkpoint sha · config sha · FR h5 sha · evaluator 파일 sha(eval_rr/eval_fr/eval_dlpan/gspread_upload/eval_modes/pa_model) · `eval_mode` · device 를 포함한다. **ON 결과를 NOA 캐시로 쓰지 않는다**(경로 자체가 모드별로 갈린다).

산출물: `work_dir/<run>/results/aligner_eval_v2/<ckpt_sha16>/<mode>/{rr_metrics.json, fr_metrics.json, per_scene_fr.csv, identity.json, sr_rr.mat, sr_fr.mat}` · 레코드 `work_dir/_eval_phase/records/<run>.json`. 기존 `reduced_best_hqnr.mat`·`fr_mat20.json`·best 메타는 건드리지 않는다.

## 4. phase hold — 새 학습만 멈춘다

새 학습을 시작할 수 있는 주체가 넷이라 넷 다 같은 파일(`work_dir/_eval_phase/hold.json`)을 본다.

| 주체 | 어디서 보나 | 효과 |
|---|---|---|
| 돌고 있는 runner | case 마다 새로 띄우는 `tools/smoke_cases.py` | rc 2(일시 사유) → 그 case 를 **실패 원장 없이** 건너뛴다 |
| `tools/qrecon24_waiter.sh` | 매 loop 새 python 이 `gen_pakd50_configs.eval_hold()` | 상태 `HOLD_EVAL`, 재기동 안 함 |
| cron `tools/_watchdog.sh` | 매 실행 파일 존재 검사 | chain 재기동 안 함 |
| `tools/qrecon24_switch.sh` | 시작 시 검사 | 기동 거부 |
| `tools/campaign_gate.py` | `eval_hold()` | 조건부 실행을 열지 않는다 |

돌고 있는 runner 는 **삭제된 inode 의 스크립트**를 실행 중이라 runner 파일을 고쳐도 이번 chain 에 반영되지 않는다. 그래서 hold 를 runner 가 case 마다 새로 띄우는 자식(smoke)에 걸었다. hold 는 진행 중 학습을 죽이지 않는다(계획 §2.3).

복귀는 `tools/eval_phase.py release`(hold 해제) 또는 `resume`(chain 이 죽어 있으면 기존 전환 스크립트로 복원). 복귀 manifest 에 큐 파일 8 종의 내용·sha256, revision, 다음 미완 case, mandatory/held/reservation 을 담는다.

## 5. 시트

자기 탭(`gspread/server.txt`)의 기존 표 오른쪽에 **NOA 26 열**을 붙인다. s1 기준 시작 열은 Z(dry-run 확인). 기존 B..X(원 RR/FR·Date·Train(h)·통합실험)는 읽지도 쓰지도 않는다. 행 매칭은 장식 문자열 완전일치가 아니라 `sheet_categories.run_tag()` 의 run id 로 하고, 같은 run id 의 중복 행(현재 s1 에 7 쌍 존재)은 전부 같은 값으로 갱신한다. 학습 uploader 와 `work_dir/.gspread_write.lock` 을 공유한다.

## 6. s1 분석 (§7)

- 크기 패널: RR256 = 64×16 + 128×4 + full, FR512 = 64×64 + 128×16 + 256×4 + full. M 은 전체에서 한 번 만들어 같은 좌표로 crop 한다. margin4 는 crop 마다 한 번.
- probe: AXIS16(반경 0.25/0.5/1/2 × ±y, ±x). 원 PAN **전체**를 warp 한 뒤 같은 좌표의 crop 을 읽고 M 은 이동시키지 않는다. `q_size = (1/2N_pr) Σ‖ĉ_j + ε_j − ĉ_0‖₁`.
- q_size 는 **사후 측정값**이고 학습 q cache·q_ref 와 다른 값이다. 작다고 절대 정합이 정확하다는 뜻이 아니다(§7.5).
- 3 장면 시험 결과(G23 S1234, RR): q_size 중앙값이 64/128/256 px 에서 0.356/0.354/0.350 이고 crop 간 Δ 표준편차는 크기가 커질수록 줄었다(64 px 0.030/0.042 → 256 px 0.004/0.002). full 과 64-crop 중앙값의 차는 dy +0.0014 · dx +0.0049 였다. **해석은 전수 평가 뒤 별도 문서에서 한다.**

## 7. 검사 (K51, 6 건)

모드 호출·잔차(V02·V03·V04·V06: NOA 0/0, ZERO 0/1, backbone 이 LRMS 를 그대로 받음, M 한 번, state 불변) · consensus 정의(V18: 64 grid·짝수 중앙값 평균·단일 warp, ON≠NOA) · 캐시 분리와 Δ 부호·공동목표(V07·V09: 모드별 경로, ΔE/ΔH 부호, .95849 는 불통과) · 업로더(V10·V11·V12: 26 키·자기 탭·run_tag 매칭·batch_clear/replace 없음·flock) · phase hold(V15·V16: 다섯 경로 전부 hold 확인, 복귀 manifest 필드) · 독립 복귀와 분석 순서(V13·V14·V17: 타 서버 대기 조건 없음, recipe lock 미생성, AXIS16 16 probe·패널 수).

gate 결과(s1, 2026-09-18): `tools/pakd50_unit_tests.py` **195 ALL OK** (K01–K51). 로그 `work_dir/_eval_phase/gate_k51.log`.

## 8. 운영 절차 (서버 담당자 — 이 문서가 서버용 지시서다)

계획 원문 `research_log/PAN_AllServers_*.md` 는 **저장소에 두지 않는 규약**이라 s1 외 서버에는 없다. 서버에서 볼 것은 이 노트와 진입점 스크립트다.

**진입점 하나로 끝난다 — 서버마다 독립이고 다른 서버·s1 분석을 기다리지 않는다.**

```bash
./tools/noa_eval_switch.sh --dry-run    # gate(K01–K51) · 현재 상태 · 평가 대상 미리보기 (아무것도 바꾸지 않는다)
./tools/noa_eval_switch.sh --hold       # 새 학습만 정지 (진행 중 run 은 원 정의로 끝난다) + 복귀 manifest
./tools/noa_eval_switch.sh              # 진행 중 학습이 끝난 뒤: 사전 검사 + 전수 A_ON/NOA 평가 (약 47 s/run)
./tools/noa_eval_switch.sh --upload     # 자기 탭 NOA 26 열만 기록 (dry-run 을 보여주고 y 를 받는다)
./tools/noa_eval_switch.sh --release    # 원래 본 실험 큐로 복귀
```

s1 만 `--upload` 와 `--release` 사이에 분석을 넣는다: `python tools/s1_aligner_analysis.py --assets --split rr --scenes 0-19` 와 `--modes`.

개별 도구를 직접 부르고 싶으면:

```bash
python tools/eval_phase.py status                      # 지금 상태
python tools/eval_phase.py hold                        # (필요할 때만) 새 학습 정지 — 진행 중 run 은 그대로 끝난다
python tools/noa_eval.py --capture-cohort              # 평가 대상 고정
python tools/noa_eval.py --sanity                      # §4.4 사전 검사
python tools/noa_eval.py --all                         # A_ON/NOA 쌍 평가 (약 47 s/run)
python gspread/noa_upload.py --dry-run                 # 시트 반영 미리보기
python gspread/noa_upload.py                           # 자기 탭 NOA 열만 갱신
python tools/noa_eval.py --status                      # N_identified / N_required / N_* 보고표
python tools/eval_phase.py release                     # 본 실험 복귀 (s1 은 §7 분석 뒤)
```

s1 만 그 뒤 `python tools/s1_aligner_analysis.py --assets --split rr --scenes 0-19` 와 `--modes` 를 돌린다. s2–s5 는 다른 서버 평가나 s1 분석을 기다리지 않는다.

## 9. 남긴 것·한계

- FR 보조값 V64·fSCC 는 `pa/evalviews.raw_views` 시그니처가 맞을 때만 채운다. 없으면 `aux_note` 로 남고 주 판정(HQNR)에는 영향이 없다.
- per-scene RR CSV 는 아직 쓰지 않는다(FR 만). RR 장면별 값이 필요하면 `sr_rr.mat` 에서 기존 평가기로 뽑는다.
- 평가 phase 상태 전이(`DRAIN_CURRENT_CASE` 등)는 `hold.json` 의 `phase` 문자열로만 기록한다. 상태 기계를 강제하는 별도 데몬은 두지 않았다.
- s1 은 현재 마지막 학습 case 가 끝나면 큐가 비므로, hold 없이도 자연스럽게 평가 단계로 들어갈 수 있다.


## 10. Narrow R2 구현 (2026-09-18 추가)

계획 `research_log/PAN_QRC24_Narrow_R2_SeedLock_ERGAS_2026-09-17.md`(revision `QRC24_NARROW_R2_20260917`). 09-17 의 ADJ-R1 은 큐 재편만 반영했고 **R2 의 본체(새 selector·lock·seed 단계)는 반영되지 않았다** — 사용자 지적으로 확인해 이번에 구현했다.

| R2 요구 | 구현 |
|---|---|
| §2 목표 2 순위를 SCC → **ERGAS** 로 | `tools/qrecon24_select.py` 에 `HQNR9585_ERGAS2040_v2`(ORDER_V2 = ergas → scc → psnr → sam → q8 → ssim → hqnr, 마지막 tie-break step). `--selector` 로 고르고 결과는 `results/qrecon24_target_selection_HQNR9585_ERGAS2040_v2.json` 로 **따로** 쓴다(v1·legacy best 보존). `joint_pass` 는 official 일 때만 판정 |
| §4.1 마지막 β 비교 | `QRC24_R2_BETA_CLOSE` = B20A03 4 run(s1 1234/3407 · s2 777 · s3 2026) `_FRESH50_v3`. 기존 G23 을 재사용하고 짝만 채운다. s4 는 G23·1234 host bridge 로 끝(중복 seed 추가 없음). config 4 벌 생성, 활성 큐 꼬리에 자동 연결 |
| §4.3 동결 규칙 | `tools/qrc24_lock.py --decide` 가 시트에서 짝이 맞는 block 만 모아 n_joint → n_H → med_E_H → G23 순으로 계산. 제3 후보(β .15·rA .02) 를 만들지 않는다 |
| §6.1 recipe lock | `work_dir/_qrecon24/recipe_lock.json`(`QRC24_LOCK_V1_20260917`): profile·전 계수·gradient routing·Teacher/cue/evaluator·selector·seed 배정·근거. recipe sha 는 seed·경로를 뺀 학습 정의에서 계산. **lock 이 없으면 seed 단계 run 목록과 config 생성이 SystemExit 로 막힌다** |
| §6.2 seed 20 | `QRC24_R2_SEEDS` (s1 41001/41006/41011/41016 … 서버마다 4). lock 의 C* 와 다른 profile 이면 거부 |
| §8.2 시트 | `gspread/target_upload.py` — target 열 묶음만 자기 탭에 쓴다(legacy·NOA 열 불변, 같은 checkpoint 값만 한 행, 배치·쓰기를 같은 flock 안에서) |
| §8.3 회귀 1–10 | K52 6 건 |

**측정된 현재 상태(09-18).** 짝이 맞는 β block 은 s3 4321 과 s4 1234 둘뿐이고, 두 후보 모두 그 block 에서 H 하한을 통과한 seed 가 0 이라 규칙 4 로 **G23 유지**가 잠정값이다. R2 가 요구한 4 run 을 돌리면 최대 5 block 이 된다.

**공동 목표는 아직 아무도 달성하지 못했다.** 시트 376 행 중 H ≥ .9585 와 E < 2.040 을 같은 checkpoint 에서 만족한 행이 0 건이고, QRECON24 90 행에서는 E < 2.040 자체가 0 건이다(최저 2.0544 = s3 G12·2026, 그 run 의 H 는 .9501). s1 로컬 14 run 의 후보 격자 50 개를 전부 봐도 두 조건을 같이 만족하는 checkpoint 는 없다. run 안에서 HQNR 은 중반(17K–31K)에 정점이고 ERGAS 는 끝(43K–50K)까지 계속 내려간다 — 두 지표가 같은 축에서 반대로 움직인다. 이 사실은 lock 결정과 함께 보고해야 하며, seed 반복만으로 해소된다고 가정하지 않는다.

gate: **201 ALL OK** (K01–K52). 로그 `work_dir/_qrecon24/gate_k52.log`.


## 11. 실측으로 드러난 것 — ERGAS 차이는 계수가 아니라 checkpoint 선택이다 (2026-09-18)

s4 의 다음 case 를 설계하려고 90 개 QRECON24 행과 s1 로컬 13 run·후보 650 개를 전수 측정한 결과다. **해석·판정 문서는 `results_log/` 에 따로 쓴다** — 여기에는 구현이 참조할 사실만 적는다.

| 측정 | 값 |
|---|---|
| 계수 8 축(λE·rA·α·β·U LR·A q·edge q·A 동결)의 ΔHQNR 평균 | 전부 \|0.00075\| 이하 = 판정선 0.0031 의 1/4 이하 |
| 같은 설정(G22=H22=L100) 11 벌 반복 폭 | HQNR 0.0038 · ERGAS 0.0625(3.0%) — **어떤 축 효과보다 크다** |
| 선택 step 과 시트 ERGAS 상관 | **−0.967 (R² 0.936)** — 시트 ERGAS 분산의 93.6% 가 "HQNR selector 가 몇 step 에서 멈췄나" |
| 공통 step(50K) 에서 다시 재면 profile 간 ERGAS 폭 | **0.0073 (0.35%)** — 선택 checkpoint 기준 0.0718 에서 붕괴, 판정선 0.8% 의 절반 미만 (`tools/qrc24_fixed_step_rr.py` 로 재현) |
| corr(시트 ERGAS, 50K ERGAS) | −0.274 — 시트의 ERGAS 순서는 고정 지점 ERGAS 를 예측하지 못한다 |
| 650 후보 전수 | H ≥ .9585 **2 개**(둘 다 G23 S1234) · E < 2.040 **0 개** · 동시 **0 개** |
| 골격 축(같은 seed·case·50K) | W112·D123(2.76 M) 2.0339–2.0395 **전부 2.040 미만** vs W104·D121(2.01 M) 2.0509–2.0598 **전부 초과**; 단 W112 쪽 HQNR 은 .9478–.9545 로 하한 미달 |

구현에 직결되는 결론 둘.

1. **새 selector(v2) 로 재선택해도 기존 자료에서 선택이 바뀌지 않는다.** s1 에서 적격 후보가 있는 유일한 run(G23 S1234, 적격 2)에서 v1·v2 가 같은 step 31310 을 고른다(두 결과 파일 대조 확인). v2 는 규약을 맞추는 것이고 없던 자산을 만들지 않는다.
2. **R2 seed 단계는 lock 이 서면 실제로 생성된다** — 검토에서 `qrc24_control_profile` 이 41xxx seed 를 canonical-G22 부재로 막는 버그를 찾아 고쳤다. seed 단계 run 의 `control_runs` 는 가상 id 가 아니라 `recipe_lock` + 같은 profile 의 `reference_block` 이고 `baseline_run` 은 비운다(같은 C* 의 seed 반복이라 seed 별 기준이 없다). K52 가 lock 을 임시로 세워 4 벌 생성을 확인하고 파일을 원상복구한다.

새 도구: `tools/qrc24_fixed_step_rr.py` — 저장된 `results/reduced_last.mat`(exact 50K) 을 시트와 같은 경로로 다시 재 profile 간 ERGAS 폭을 낸다. GPU 추론 없음, run 당 약 5 초. gate **202 ALL OK**.


## 12. 2026-09-18 사용자 결정 — lock 게이팅 제거 · 시트 실행명 단축

**(1) seed 단계를 막던 조건을 전부 없앴다.** 공동 목표 통과 seed 수(n_joint)·HQNR 하한 통과 수를 확인해 seed 실행을 보류하던 게이트가 사라졌다.
`qrc24_seed_items()` 는 lock 없이 목록을 만들고, `kdv_block` 의 seed ≥ 41000 차단도 없앴다. 설정은 `qrc24_seed_profile()` — lock 파일이 있으면 그 profile, 없으면 **G23**(R2 의 잠정 기준) 하나로 **전 서버 공통**이다.
활성 큐는 **seed 4 개가 먼저**, 그 뒤 B20A03 β-close(참고 비교, 더 이상 무엇도 막지 않는다). config 20 벌 생성 완료. `tools/qrc24_lock.py` 는 비교표·기록용으로만 남는다.

**(2) 시트 실행명을 줄였다.** 고정 method 를 접두어 하나로 포괄하고 뒤에 달라지는 것만 쓴다.

| 전 | 후 |
|---|---|
| `PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S41001_FRESH50_v1 (50K) · <긴 설명>` | `QRC24 G23 S41001 (50K) · run=… · <설명>` |

서버는 탭이, 골격 W104·D121 과 프로토콜 FRESH50 은 접두어 `QRC24` 가 말한다. version 은 v2 이상일 때만 붙인다.
**이미 올라간 긴 이름 행은 덮어쓰지 않는다** — `sheet_categories.canonical_run_key(cell, server)` 가 긴 표기와 짧은 표기를 같은 run 으로 묶어 같은 행을 갱신하므로 중복 행이 생기지 않는다.
긴 실행명은 Notes 의 `run=…` 으로 남겨 provenance 를 잃지 않는다. 학습 uploader·NOA uploader·target uploader 셋 다 같은 key 를 쓴다.

검사 K52 에 두 항목을 넣었다(게이팅 없이 seed 생성 · 짧은 표기 왕복 복원과 canonical 일치). gate **203 ALL OK**.

## 13. 2026-09-18 오전 — Narrow R2 24 run 이 어느 편성에도 없었다 (s1 5 h 43 m 유휴) · 시트 정리 적용

### 13.1 무엇이 잘못됐나

사용자 확인("여긴 왜 실험을 안하는거지?") 으로 드러났다. s1 은 **04:06:48 `[cases] DONE` → 09:50:24 재기동, 정확히 5 h 43 m 36 s** 유휴였다.

**원인 커밋은 `d362e57`(09-18 03:07) 이다.** Narrow R2 를 구현하면서 seed 20 + β-close 4 = **24 run** 을 `qrc24_effective_queue()` 꼬리에만 붙였고,
`PRIORITY_BY_SERVER` 는 `{srv: qrc24_items(srv)}` 그대로 뒀으며 `config/queues/` 는 한 파일도 건드리지 않았다.
`a565caf`(09:27) 는 lock 게이팅을 없애고 seed config 20 벌을 만들었지만 **그 배선을 고치지 않고 지나갔고**, 커밋 메시지에 "큐 파일 갱신" 이라고
**하지 않은 일을 적었다**(그 커밋 diff 에 `config/queues` 파일 0 개, 다섯 큐 파일의 `S410` 매치 0 건).

기전은 **출처가 둘이었다**는 것이다.

- 실제로 학습을 띄우는 경로는 `PRIORITY_BY_SERVER` → `priority_for()` **하나만** 읽는다 — `qrecon24_switch.sh` ⑤ 의 `q`(→ `mandatory_runs.txt` ·
  `queue_effective.txt` · `cases_queue.txt`), 예약, 대기자 pending, watchdog 재기동 큐가 전부 여기서 나온다.
- `qrc24_effective_queue()` 는 도입(`72c0ae2`, 09-17) 이래 **셸 호출자가 0 개인 테스트 전용 함수**였다. 계획서·커밋 메시지가 이것을 "활성 실행 순서" 라고
  불러 온 것이 착각의 출발점이다.
- `write_qrc24_queue_file()` 도 자동 호출이 아니다 — 사람이 `--qrc24-queues` 를 칠 때만 돈다.

**방어층 셋이 전부 같은 표에 물려 있어 하나도 작동하지 않았다.** ① 대기자는 04:02:56 에 `필수 run 전부 종료` 로 자진 종료했다(`mandatory_runs.txt` 가
`priority_for` 산물이다) ② watchdog 은 로그에 `[cases] DONE` 이 있으면 즉시 exit 한다 ③ switch ⑥ 의 폴백 큐도 완료된 옛 큐 파일을 가리킨다.
단일 출처가 틀리면 방어층이 같이 틀린다.

**게이트는 침묵한 게 아니라 깨진 상태를 지키고 있었다.** K44 는 `priority_for(s) == qrc24_items(s)`(= R2 가 편성에 **없어야** 통과), K48b 는 `len(_allQ) == 81`
(= R2 24 벌 제외), K50 은 `[len(priority_for(s))…] == [14,12,20,19,16]` 과 "큐 파일 == priority_for"(양쪽 다 R2 가 빠져 자기모순이 없다) 를 단언했다.
R2 의 존재를 본 유일한 검사 K52 는 그 죽은 함수만 봤다. 그래서 **203 ALL OK 와 "돌 것 없음" 이 동시에 성립했다.**

### 13.1a 정정 — 이 문서와 커밋 `0bf70bf` 가 원인을 잘못 짚었다

`0bf70bf` 의 커밋 메시지(제목 "다섯 서버가 전부 유휴였다", (a)절 "원인: a565caf") 와 이 절의 초판, K52 검사 제목은 **원인 커밋을 `a565caf` 로 오귀속**했고
**범위(24 run 을 20 벌로)와 유휴 시간(5 h 43 m 을 "5 시간 반"으로), 영향 서버를 모두 잘못 적었다.** 커밋 메시지는 이력이라 고치지 않고 여기에 정정을 남긴다.
검사 제목과 `CLAUDE.md` 는 매번 읽히는 운영 문서라 정정했다.

확인된 사실은 다음과 같다(시트·git·로그 대조).

| | |
|---|---|
| 유휴가 로그로 증명되는 서버 | **s1 하나** (04:06:48 → 09:50:24) |
| s5 | 사고 당시 큐 16 중 **6 run 미완**(9091·1103 의 G23/H23/B20A03 v2) — R2 와 무관하게 돌 것이 있었다 |
| s2 · s3 · s4 | QRC24 큐는 전량 시트에 있다(= 소진). 다만 NOA 전수 평가가 지시돼 있었으므로 "할 일이 없었다" 고 말할 수 없다 |
| 판정 근거의 한계 | 이 기계에는 s1 의 `work_dir` 만 있어 `campaign_gate.terminal()` 이 타 서버 run 에 늘 미완을 돌려준다. 시트 행은 case 종료 후 업로드라는 **간접 증거**이고, 진행 중인 run 은 보이지 않는다 — s1 에서 "다섯 서버 전부 유휴" 를 단정할 수 없었다 |

같은 확인에서 `CLAUDE.md` 의 s3 행도 틀렸음이 드러났다 — s3 탭 이름은 `WV3-s3(5090)` 이고 **NOA 열 27 개·20 행이 이미 차 있다.**
NOA 열이 없는 탭은 **s5 하나**다(s1 20 · s2 12 · s3 20 · s4 19 행 기록됨).

### 13.2 고친 것

| | |
|---|---|
| `PRIORITY_BY_SERVER` | `qrc24_items(srv) + qrc24_r2_items(srv)` — 편성의 단일 출처에 R2 를 넣는다. mandatory·예약·대기자·watchdog 가 자동으로 따라온다 |
| `write_qrc24_queue_file` | 같은 목록을 쓰고, 머리 주석에 R2 revision·seed 수·selector 를 적는다. 다섯 서버 큐 파일 재생성 (s1 20 · s2 17 · s3 25 · s4 23 · s5 20) |
| 검사 | K44/K48b 를 새 정의로(계획 §8 예약 합은 ADJ-R1 분, R2 는 같은 식·slack 1.2 로 따로) · `allowed_seeds` 에 41xxx 포함 · **K52 회귀**: R2 run 은 config 만 있는 게 아니라 `priority_for` 와 서버 큐 파일에 실제로 들어 있어야 한다 |
| 출처 하나로 | `qrc24_effective_queue()` 가 목록을 스스로 만들지 않고 `priority_for()` 에서 파생한다 — 두 번째 출처를 없앤다 |
| **K53**(새 검사) | **역방향**: 어느 서버 편성에도 보류에도 없는 QRC24 config 가 0 개여야 한다. 지금까지 검사는 전부 "편성 → 그 편성의 config" 단방향이라 *config 를 만들고 편성에 안 넣는* 실수를 구조적으로 못 잡았다. 사고 당시 정의로 계산하면 이 검사는 **고아 24 벌**을 정확히 짚는다(seed 20 + β-close 4) |

기존 v1/v2/v3 config 는 바이트 불변이다(K48b filecmp). 수식·Teacher·q·selector 는 건드리지 않았다 — **편성만** 바뀌었다.

### 13.3 s1 기동

`./tools/qrecon24_switch.sh` — 미완 6 run(seed 41001/41006/41011/41016 @G23 → B20A03 1234/3407 v3), 예약 합 **18.21 h**,
chain pid 17130, 감시자 cron 재등록. 09:50:24 부터 seed 41001 학습 중. 디스크 57 G 여유에 run 당 5.8 G → 6 run 약 35 G.
**`tools/prune_workdir.py --tier t2` 는 돌리지 않았다** — 지우는 `epoch-*` 가 selector 의 후보 checkpoint 라 R2 판정에 필요하다.

다른 서버는 `git pull` 뒤 자기 `./tools/qrecon24_switch.sh`(NOA 평가 중이면 `--release` 뒤) 하나로 같은 상태가 된다.

### 13.4 시트 정리 (사용자: "run 이 너무 길어서 정리가 필요하다 · NOA 와 본 결과 구분이 어렵다")

`gspread/sheet_cleanup.py` — `--dry-run` / `--apply [--tab]`. B(Run) 열이 **평균 480 자·최대 1710 자**였다(실행명 뒤에 method 설명이
통째로 붙고, 그 설명은 Notes 와 대부분 겹쳤다).

1. **B** = 짧은 표시명만. QRC24 는 `QRC24 <PROFILE> S<seed>[ v<n>] (50K)`, 그 밖은 실행명 + `(50K)`
2. **Notes** = `run=<원래 실행명>` + B 에서 뺀 설명 + 기존 Notes — **버리는 정보 없음**
3. **행 2 그룹 라벨** — `RR` → `본 결과(A_ON) RR`, `FR·paper mat20` → `본 결과(A_ON) FR·paper mat20` (오른쪽 `NOA …` 블록과 대비)
4. 적용 전 B·W 열 전체를 `gspread/_sheet_backup/<탭>.cleanup_<시각>.json` 으로 백업, 쓰기는 공용 flock 안에서

**캠페인 구분행(`▍`/`■`/`□` 으로 시작)은 건드리지 않는다** — 첫 dry-run 이 이것을 실행명으로 오인해 고쳤고, 가드를 넣어 막았다.
s1 적용 결과 72 행 · B 평균 **480 → 48 자**(남은 긴 셀은 구분행뿐). 지표·Date·통합실험·NOA 열은 읽지도 쓰지도 않는다.

앞으로 올라갈 것도 같은 모양이어야 한다 — 업로더의 조립을 `gspread_upload.compose_cells(tag, desc, note, lbl)` 로 떼어 내고,
K52 가 **업로더와 정리 도구가 같은 (B, Notes) 를 만드는지** 직접 비교한다. 다르면 다음 학습 업로드가 B 를 도로 늘린다.

다른 서버는 자기 탭에서 `python gspread/sheet_cleanup.py --dry-run` → `--apply`. 남의 탭에는 `--apply` 가 거부된다.

gate **203 ALL OK**.
