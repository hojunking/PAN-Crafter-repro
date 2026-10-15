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

## 8. 운영 절차 (서버 담당자)

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
