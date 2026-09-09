# 2026-09-09 — A1–A3 (PAN 앞단 전역 정합) 명세 v2 검토와 구현 노트

명세: [`PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md`](PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md).
이 문서는 **명세를 저장소 코드로 옮기며 확인한 것·정한 것·미룬 것**의 기록이다. 결과는 `results_log/` 에 쓴다.

## 1. 검토 — 명세가 전제한 것과 저장소의 실제

| 명세 항목 | 저장소 실제 | 처리 |
|---|---|---|
| B0 "완료" (§0.1, §7.4, §16-1) | B0 = `BASE_W96_D124_MSPAN_WV3_S{2025,1234,7777}`, s1 에서 09-09 19:58 기동, 3벌 순차(≈1 h/run). 이 문서 작성 시점 seed 2025 진행 중 | A1–A3 는 B0 와 **같은 config 에서 trainer 만 바꾼 것**이라 B0 완료를 기다릴 필요가 없다. s1 은 B0 체인이 끝나면 자동 기동(`tools/_chain_after.sh`) |
| block seed s3 = 777 (§8.1) | B0 seed 는 2025·1234·7777 | 명세의 대체 규칙대로 **s3 = 7777**. s2/s3 큐는 자기 block 의 B0 를 먼저 돈다(§8.3 같은 서버 대조군) |
| B0 evaluator 의 원래 영역 Ω0 (§10.3) | `tools/metrics/eval_fr.py` 는 **crop 없이 512² 전체**. 논문 anchor 가 이 조건에서 일치 | raw_original = 전체 프레임 |
| PAN 저해상도 reference 생성 경로 (§10.4) | `d_s`: `imresize_matlab(1/4)` → `interp23tap` (support 8 + 44 HR px) | `pa/evalviews.pan_low_reference` 가 같은 두 함수를 P̃ 에 적용. identity gate = E02 통과 |
| 고정 V 의 margin 숫자 (§10.6) | evaluator support 로 계산: warp 2 + imresize 8 + interp23 44 = 54 → 블록(32) 배수 **64 px** ring. V = [64:448]² (12×12 블록) | `selection_roi_manifest.json` 에 규칙·hash. 적격 |Δ| ≤ **8 px**(64−54−2) |
| selection/report scene set (§10.1) | 둘 다 논문 세트 `.mat` 20장(2026-09-09 규약, 12-19 없음) | 동일 집합, 별도 split ID 불필요 |
| DN 변환 (§10.1 intensity/clipping) | 시트 evaluator(`eval_fr_paperset`)는 clip(−1,1)→(x+1)/2·max_pixel, 반올림 없음; `train.py` 옛 경로는 반올림 | **반올림 없음**으로 통일(시트 값과 같은 수) |
| fSCC (§10.4) | `utils.SCC_full_numpy` (Sobel, 옛 정의·상대 비교용) | view 별 PAN reference·영역으로 계산 |
| RR-valid (§10.7) | RR 테스트 PAN 은 256²라 같은 64 px ring 이면 128² 만 남는다 | **RR 은 original 영역만** 기록(GT 기반이라 PAN reference 와 무관, §10.7 규칙대로 두 valid view 에 같은 값). RR-valid 는 미구현(아래 §4) |
| tiling inference (§10.2) | 없음 — FR 512² 를 한 번에 추론 | 해당 없음 |
| U-Net 초기값이 B0 와 대응 (§7.3) | `main.py` 가 `init_seed(seed)` 뒤 모델을 만든다 — B0 와 같은 경로 | 그 tensor 를 `work_dir/_pa_init/init_unet_seed<seed>.pt` 로 저장·hash. **B0 는 초기 가중치를 저장하지 않았으므로 "같은 경로" 로 대응할 뿐 tensor 동치는 검증 불가** |
| aligner 초기값·data RNG 분리 (§7.3) | aligner 는 `torch.random.fork_rng` 안에서 seed+1000 으로 만든다 — U-Net·DataLoader RNG 소비 없음 | `init_aligner_seed<seed>.pt` 저장, 같은 seed 의 다른 case 는 로드해 맞춘다 (T19) |

**검토 의견 (명세 자체).**

- §5.3 L_geo 의 `V_g` margin 11 은 64 patch 에서 42² 만 남긴다. 명세가 인정한 operational constraint 다. 구현은 그대로 두되 **guard 는 모든 case 에 같은 조건**으로 걸었다(|Δ| 이 margin support 를 깨면 `SUPPORT_FAIL`, exit 4).
- `results_log/2026-09-09_alignment-after-training-analysis.md` §8: B0 골격은 학습 규모의 GT–PAN 어긋남(0.37 px)을 **이미 암묵적으로 보정**한다. 따라서 A1 의 Δ̂ 는 "총 어긋남" 이 아니라 backbone 과 나눠 가진 몫이고, FR(1.79 px)에서 필요한 양의 1/4 규모에서 학습된다. 명세 §12.3 의 "Δ≈상수/bias" 해석과 §11.2 known-shift 반응 진단이 이 점을 가른다. 결과 해석 시 이 사실을 전제로 둔다.
- 명세의 stand-in 검사(아래 §3)에서 보이듯 **PAN 을 M-frame 쪽으로 옮기면 raw HQNR 은 오르고 fSCC 는 내린다**(D_s 기전, 같은 문서 §3). best_raw 는 raw_original HQNR 로 고르므로 이 방향을 선호한다. 명세가 best_raw 를 주 비교로 둔 것은 그대로 따르되, §11.1 대조(learned/zero/wrong-sign)와 fSCC 를 반드시 같이 본다.

## 2. 명세 → 코드

| 명세 | 코드 |
|---|---|
| §3 aligner (dual-stem, 105,330 params, zero head, Z-norm) | `pa/aligner.py` — params **105,330** 일치 |
| §4 warp (bicubic·border·align_corners=False, 0 우회 금지, FP32) | `pa/warp.py warp_pan` + `warp_support_mask`(clamp 전 좌표·4-tap) + `support_margin_ok` |
| §5 loss (L_rec 전 영역 · L_edge Scharr signed 1px · L_geo NGF-outer-product, σ=2, margin 11, ramp 5K) | `pa/losses.py`; λ 는 `train_pa.py` (A1 0/0 · A2 0.1/0 · A3 0/0.01) |
| §2 forward Ŷ = M + Fθ(cat(P̃, M)) | `pa/model.py PAModel` (backbone 의 `in_mode paper` 9ch 경로, lpan 미사용) |
| §7 학습 조건 = B0, aligner 같은 LR param group, from-scratch, init 대응 | `train_pa.py PATrainer` (B0 config 위에 `trainer: pa` + `pa:` 블록만 추가, `tools/gen_pa_configs.py`) |
| §6 support guard · §13 step 로그 | `train_pa.py` — `train_log.jsonl`, `gradient_diagnostics.jsonl`(aligner grad from rec/aux, 500 step 마다), Δ 통계·valid fraction·PAN gradient energy before/after |
| §10.3–10.5 세 view · mask · 전체 프레임 필터 후 crop | `pa/evalviews.py scene_views` |
| §10.6 고정 V·적격성 | `pa/evalviews.fixed_roi / MAX_ELIGIBLE_SHIFT / roi_manifest` → `selection_roi_manifest.json` |
| §10.8 best_raw / best_aligned / last, running max·tie 1e-4·fSCC tie·later step, 후보 보존 | `pa/selector.py BestSelector`; `train_pa._select` (후보 `candidates/step-N/` 는 두 stream 의 band 합집합만 보존) ; `best_hqnr/` = **best_raw 의 alias**(체인·시트·`eval_fr_paperset` 호환), `best_aligned/`, `last/` |
| §10.10 교차 평가 · §11 진단 | `tools/pa_diag.py` (→ `results/pa_diag.json`; `_upload.sh` 가 PA_ run 에 자동 실행): 교차표, learned/zero/wrong-sign, known-shift 반응(e∈{−1,−.5,0,.5,1}²), tile 128 vs full |
| §12.2 long-format 로그 | `scene_metrics.csv`(step·scene·view·roi_scope·pan_reference·D_λ·D_s·HQNR·fSCC·Δ·eligible·roi_hash), `checkpoint_metrics.csv`, `delta_predictions.csv` |
| §13 산출물 | `initialization_hashes.json`, `pa_config_resolved.json`, `selector_state_{raw,aligned}.json`, `{best_raw,best_hqnr,best_aligned,last}_meta.json`, `results/{reduced,full}_{best_hqnr,best_aligned,last}.mat`(sr·pan_aligned·delta 포함) |
| §14 run ID | `PA_A1_REC_W96_D124_9CH_S2025` 등 9벌 (서버 접미사는 시트가 `server.txt` 로 붙인다) |
| §8 배치 | `config/queues/pa_s1.txt`(2025: A1→A2→A3) · `pa_s2.txt`(B0 S1234 → A2→A3→A1) · `pa_s3.txt`(B0 S7777 → A3→A1→A2) |
| 시트 | `main.py` 가 PA 선택을 trainer 선택기에 위임; 시트 FR·paper = best_raw 의 raw_original(=`eval_fr_paperset` 값, E02). 범주 ⑱ PA |

## 3. Gate 결과 (s1, 2026-09-09)

- `tools/pa_unit_tests.py`: T01–T05·T07–T20 해당 항목, §9.1(Δ=0 ↔ B0 forward 최대차 0), §6.2 guard, E02(raw_original == `d_lambda_k`/`d_s` 소수점 이하 동일)·E03·E06·E07·E11·E13·E15·E16·E20 **전부 통과**. T17 synthetic: aligned 0.00004 < uncorrected 0.095 < wrong-sign 0.293.
- `tools/smoke_cases.py` A1/A2/A3: 통과, 학습 peak 2.45 GB, step ≈ 0.12 s.
- 40-iteration end-to-end dry run(A3, 임시 config): 학습 로그·support guard·세 view·두 선택기·후보 보존·`best_hqnr/best_aligned/last` 내보내기 전부 동작. Δ=0 근방에서 valid fraction 0.9084 = (61/64)² (보수적 mask).
- `tools/pa_diag.py` stand-in(B0 checkpoint-10000 + Δ=(−0.4,+0.2) 고정 aligner): 교차표·세 대조·known-shift·tile 전부 실행. 고정 aligner 라 known-shift 반응 기울기 0(정의상)·tile 분산 0.
- §9.4 합성 학습 가능성(`tools/pa_synthetic_check.py`, RR GT 로 만든 통제 쌍, e~U(−2,2)² HR px, aligner 만 학습): 400 step 에서 corr 0.97/0.95·중앙 오차 0.31 px(기준 0.25 미달), **1500 step 에서 corr 0.996/0.997·중앙 오차 0.079 px·p95 0.24 → PASS**. 이 checkpoint 는 버린다.
- 미실행 gate: T06(LR→HR 단위 변환 — 이 구현은 HR 단위만 쓴다, 해당 없음), E01/E04/E05/E08–E10/E12/E14/E17–E19 는 코드 구조로 보장(같은 forward 의 SR·Δ·P̃ 를 세 view 가 공유, P̃ 저해상도는 매번 재생성·캐시 없음, 평가 mask 는 loss 에 연결되지 않음)하되 **별도 테스트 코드로 닫지는 않았다**.

## 4. 명세 중 이번 구현이 미룬 것

- ~~RR-valid~~ → §6-7 에서 구현(같은 64 px 규칙, 256² → 128²; 학습 로그·`checkpoint_metrics.csv`·`pa_diag`).
- **report_common 교집합 ROI**(§10.6d): 고정 V 로 충분한 동안 만들지 않는다.
- ~~`dataset_hashes.json` · B0 manifest~~ → §6-7 에서 구현(`dataset_hashes.json`, `baseline_manifest.json`). `environment.json`/`code_commit.txt` 는 `tools/run.sh` 의 `meta/` 스냅샷으로 대신한다.
- **`predictions/<ckpt>/<split>/<scene>/` 디렉터리 구조·overlays**: `results/full_<tag>.mat` 에 sr·pan_aligned·delta, `results/controls_*_best_hqnr.mat` 에 zero/wrong-sign SR 을 넣는 것으로 대신했다. overlay 그림은 없다.
- **A2 대조군 `B0 + edge loss, aligner 없음`**(§11.5): 승인된 9벌에 포함되지 않음 — 결과 후 후속.

## 5. 실행

```bash
# 각 서버 (gspread/server.txt = s1/s2/s3 에 맞는 큐가 자동 선택된다)
export PANCRAFTER_DLPAN=/path/to/DLPan-Toolbox
./tools/pa_prepare.sh            # 점검(데이터·지표·gate·smoke) → 기동 (30h)
./tools/pa_prepare.sh --after    # 다른 체인이 돌고 있으면 끝난 뒤 자동 기동
```

s1: B0 3벌 체인(19:58 기동) 뒤 `pa_s1.txt` 자동 기동 예약(`work_dir/chain_after.log`). run 당 ≈ 1 h + 평가(세 view 20장 ≈ 1 min/epoch 평가).

## 6. 2차 검토(사용자, 8건) 반영 — 2026-09-09 밤

| # | 지적 | 판단 | 반영 |
|---|---|---|---|
| 1 | L_geo guard 가 warp support 를 [11:53] 만 검사 — Gaussian 6 + Scharr 1 만큼 바깥 P̃ 도 참조 | **맞음** | `pa/losses.geometry_support_margin(σ, margin)` = 11 − (6+1) = **4** → guard 는 [4:60]² 의 sampling 이웃을 검사. 64² 에서 2.9 px 통과 / 4.5·9 px 중단(테스트 추가). 모든 case 동일 |
| 2 | `pa_diag` 가 CSV 의 `True/False` 를 float 로 바꾸다 죽고, `_upload.sh` 파이프가 실패를 숨김 | **맞음** | `cross_table` 이 열별로 파싱(bool 은 bool); 실패는 rc 로 잡아 `!!` 로 표시하고 로그 파일에 남김. 교차표는 CSV 에 의존하지 않고 **세 checkpoint 를 재추론**해 만든다 |
| 3 | B0 선택(반올림) · PA 선택(반올림 없음 + lms clip) · 시트(원본 참조) 가 같은 경로가 아님 | **맞음** | PA 평가 참조를 **h5 원본 float64** 로 바꿈(feeder 왕복·clip 없음), P̃_eval = W(P_raw, Δ̂) 를 float64 로 계산 → **시트 evaluator 와 배열 단위로 동일**(E02 확장 테스트: 참조 배열 동일, Δ=0 warp exact). B0 의 **선택** 은 `train.py` 옛 경로(반올림)라 ~1e-6 차이가 남는다 — 시트 보고값은 두 계열 모두 `eval_fr_paperset` 경로라 같다. 구현 노트 표현을 "선택은 B0 와 동일" 에서 "보고 경로와 동일, B0 선택과는 반올림 ~1e-6 차이" 로 정정 |
| 4 | valid fSCC 가 crop 후 Sobel — 규칙 위반, 6e-5 차이 | **맞음** | `sobel_maps` 를 전체 프레임에서 만든 뒤 자른다(`fscc_from_maps`). 전체 프레임은 `SCC_full_numpy` 와 1e-12 이내 동일(테스트) |
| 5 | resume 안전장치·best 메타·NaN | **맞음** | selector 복원은 `--resume` 일 때만, protocol·evaluator hash·FR h5 sha·scene 수 전부 검사(불일치 → 예외). resume 없이 같은 work_dir 면 옛 선택 상태·후보를 `_stale_selection_<time>/` 로 치움. `best_state.json` 의 SCC/ERGAS 는 **선택된 step 의 기록**(`step_records`). eligibility 가 세 view metric 유한성까지 보고 사유(`shift>8px` / `nan_metric:<view>`)를 CSV 에 남김 |
| 6 | 시트 비용에 aligner 누락 | **맞음** | `gspread_upload._cost_model` — trainer pa 면 PAModel(aligner+warp) 로 params/FLOPs/추론시간/메모리 측정. PA **2.228282 M** (backbone 2.122952 + aligner 0.105330) |
| 7 | 미구현 평가·검증 | **부분 수용** | RR-valid(같은 64 px 규칙, 256² → 128²) 학습 중·pa_diag 모두 기록. B0 를 같은 view·공통 V 로 재평가하는 경로(`pa_diag.py --run BASE_…`, Δ=0). zero/wrong-sign SR 저장(`results/controls_*_best_hqnr.mat`). 독립 구조 진단(audit 추정기로 PAN_b←up(MS) vs P̃_b←up(MS)) + RR band 별 Scharr edge error. §9.4 합성 학습 가능성 검사 `tools/pa_synthetic_check.py`. `dataset_hashes.json`(train/valid/RR/FR h5 sha256) · `baseline_manifest.json`(B0 run·config·best 가중치 sha·init hash·scene set). **미반영**: overlay 그림, `report_common` 교집합 ROI(고정 V 로 충분한 동안 보류) |
| 8 | `--after` 도 GPU smoke 를 즉시 돌림 | **맞음** | 예약 모드면 gate 는 CPU(`CUDA_VISIBLE_DEVICES=""`), smoke 는 체인이 case 시작 직전에 스스로 돈다(`_run_cases.sh`) |

재검증: `tools/pa_unit_tests.py` 전부 통과(추가 항목 포함), 40-iter dry run + `pa_diag`(재추론 교차표·대조·shift 반응·tile·독립 구조) 정상, B0 재평가 경로 정상.
