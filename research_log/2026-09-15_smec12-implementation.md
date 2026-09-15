# SMEC12 (Recon–Consistency 불일치의 원인·sample 특성 검증, 다중 데이터셋) — s2 실행판 검토·구현 (2026-09-15)

계획: `research_log/PAN_SMEC12_MultiDataset_SampleMechanism_ExperimentPlan_2026-09-15.md` (이하 §). 사용자 지시 11:0x: "이건 s2 에서 진행할 것이다. 구현/검토 진행해라". 작성 s1.

## 1. 검토 — 이 계획의 실제 크기와 이번 release 의 범위

계획은 (a) **센서별 모델 세트 준비 학습**(§2: P0·DON-N2·L000·L1E4·L1E4-REP × QB·GF2 = 50K × 10) 과 (b) **13 개 분석 stage + 조건부 R60**(§5–§18) 으로 구성된다. 저장소 현황:

| 항목 | 상태 |
|---|---|
| WV3 모델 세트 | 전부 기존 자산: P0 = NF16 P0(S1234)·PALS24 CTRLP0(S2025/S7777), DON-N2 = PO10 N2 R200(S2025), L000 = NF16 P2(S1234)·PALS24 L000(S2025/S7777), L1E4 = PALS24 L1E4(S2025; REP S1234/S7777). W112·D123, 같은 donor, 같은 U 초기값 규약 — §2.3 규칙 2·9 대로 **재학습하지 않는다** |
| QB / GF2 모델 | **없다**(W168 DUAL 아키텍처 run 만 있고 model_class·band 가 다르다) → §2.3 규칙 4–7: 준비 학습 10 run 이 training dependency |
| WV2 | 학습셋 없음 → WV3 모델 zero-shot(RR/FR 20 scene × 16 tile) 만 (§3.1) |
| 데이터 | QB `train_qb_msfix.h5`(+_pan), GF2 `train_gf2.h5`(+_pan, max_pixel 1023), 세 센서 mat20 FR, WV2 RR/FR + mat20 — s1 에 전부 있음. s2 는 `tools/smec12_prepare.sh` ① 이 검사한다(없으면 blocker; 만들어 넣지 않는다) |
| source id | PanCollection 에 scene/strip id 없음 → 32-index block proxy(`source_group_unknown`), CI 는 descriptive (§3.3·§4.5). FIT70 source-holdout cohort 는 기존 full-train 자산과 섞을 수 없어 이번 release 에 없음 (§3.3 의 구분 표시만) |
| 시간 | 준비 학습 10 × ≈2.6 h(s2 W112 50K 실측 2.3 h 대용) ≈ 26 h 순차 — 계획 §19.2 가 명시한 대로 12 h 안에 끝나지 않는다. s2 는 현재 PAKD50 재배정 큐(≈11 h) 를 돌고 있으므로 그 뒤에 기동한다 |

**이번 release 에 구현·검증한 것**: (a) 전부(config·큐·기동 절차·smoke), (b) 의 backbone — A00 · D10(+D11-A/B 일부) · I20(A/B/C) · I23-B · I24-A · I25-A · X40 · REPORT — 를 센서 일반형(4/8 band, max_pixel, zero-shot tile) 으로 만들고 WV3 자산으로 검증. **미구현(pending_compute)**: D11-C edge profile, D12 descriptor bank·blind gallery, I21, I22, I23-A/C/D, I24-B/C, I25-B, S30, C50, R60. 보고서는 이 stage 를 `pending_compute` 로 적고 complete 로 쓰지 않는다(§19.3·§24).

## 2. 준비 학습 (a): `tools/gen_smec12_bootstrap.py` + `config/queues/smec12_s2.txt` + `tools/smec12_prepare.sh`

| role | template(검증된 성공 run) | 센서 치환 | run id |
|---|---|---|---|
| P0 | NF16 P0 (kdv A-ID/NOALIGN/I-A) — `eval.reference_donor` 제거(donor 아직 없음), `aligner_lr` 제거 | dataroot 4개 · num_bands/out_channels · max_pixel · expect_params_m 2.6508 · eval_epoch(≈1000 update 격자: QB 3, GF2 2) · save_iter 25000(exact25K/50K 전체 state) · `kdv.init_dir work_dir/_kdv_init_w112_d123_<s>`(같은 센서 P0/L000/L1E4 가 같은 seed 초기값 공유 §2.4) | `B01_<S>_P0_W112_D123_S2025_BOOTSCRATCH` |
| DON-N2 | PO10 N2_OFFSG R200 (po: native/corrupt 1:1, SG offset .01 ramp 5K, b=2, aligner head zero-init) | 위 + `po.init_dir _pa_init_w112_d123_<s>` + radius_provenance 에 "WV3 공통 이식(센서 최적값 아님)" | `B02_<S>_DONN2_W112_D123_R200_S2025_BOOTSCRATCH` |
| L000 / L1E4 | PALS24 L000 / L1E4 (kdv A-FT; I-NATIVE-TRANSFER offset 0 / I-AEQ λ 1e-4 b 2) | 위 + donor = `work_dir/B02_…/last`(exact50K; sha 는 학습 뒤 A00 이 기록) · `eval.fixed_reference_from_donor` | `B03_<S>_{L000,L1E4}_W112_D123_S2025_DONB02` |
| L1E4-REP | 같은 donor, seed 1234 | | `B04_<S>_L1E4_W112_D123_S1234_DONB02` |

- 4-band backbone 2.6508 M(W112·D123, out 4) · aligner 0.1048 M. smoke 4 벌(QB/GF2 P0·DONN2) 통과 — kdv A-ID 와 po trainer 가 4 band 에서 동작. B03/B04 는 donor 가 생겨야 smoke 가 통과하므로 큐 순서가 의존성이다(chain 이 run 직전에 smoke).
- 큐는 lane 교차(QB P0, GF2 P0, QB DONN2, GF2 DONN2, …): 두 센서의 primary L1E4 가 8 번째 run 뒤에 모두 나온다(§19.3 "아직 분석하지 않은 sensor 우선").
- 예산 ledger `work_dir/_smec12_budget/ledger.json`(40 h, required=true — 준비 학습은 시간 때문에 취소하지 않는다 §2.3 7).
- `tools/smec12_prepare.sh`(s2): 데이터 검사 → unit gate → config == 생성기 → ledger → **현재 chain 이 끝날 때까지 기다린 뒤** gate 파일(`pakd50`) 을 비우고 `campaign_start --queue smec12_s2.txt` (detached waiter). `_upload.sh` 가 B01/B03/B04 에 pa_diag 를 돌리고, 시트 범주 ㉖ SMEC12 로 올린다(B02 po run 은 표준 평가·업로드만; PO10 진단은 WV3 donor 참조라 제외).
- 기록 규약: 새 run 은 새 id/hash — WV3 수치를 재현했다고 덮어쓰지 않는다(§2.3 9). b=2·λ=1e-4·.01 ramp 는 공통 이식(§2.1).

## 3. 분석 backbone (b): `tools/smec12/` (CLI `tools/smec12.py`, runner `tools/smec12_run.sh`, gate `tools/smec12_unit_tests.py` S01–S14)

| stage | 구현 | 산출물 (§21 이름) |
|---|---|---|
| A00 | asset registry(센서 × role × seed → run·status·hash), data registry(파일·shape·sha), bootstrap queue, coverage, reproduction(선택 metric vs fr_mat20), discrepancies, unit(warp 부호·identity·반복 noise·RNG 격리·zero-correction 경로) | `manifests/*.json`, `coverage.csv`, `reproduction.csv`, `analysis/discrepancies.md` |
| D10/D11 | in-domain CAL/DISC/CONF(≤4,096, 32-block, seed 271828) · WV2 RR/FR tile: e_full/e_roi/edge/band, q_A/q_B/EPE(반경별), c0, label(CAL median 고정, boundary_near ±10 % IQR, q_unresolved), e_base/g_restore(D11-B), contrast-normalized e(D11-A, floor = CAL P5), texture·saturation; Spearman(+source bootstrap), LOSO, 반경별 | `raw/sample_metrics.csv`, `manifests/thresholds.json`, `analysis/d10_stats.json` |
| I20 | bank S 32 probe δ 를 c0 에 더해 원본 P 에서 한 번 sampling(A 재호출 없음): s_out(r), signed d_GT; Jacobian h=.125/.25(Jᵀ J/(CHW) 고유값·방향·안정성); bank B 실제 residual vs 90° 회전 residual, 선형 예측 오차 | `raw/fixed_residual_response.csv`, `raw/position_jacobian.csv`, `raw/residual_direction.csv`, `analysis/i20_stats.json` |
| I23-B | I-L/I-Z/I-W/I-C(CAL median)/I-S(10 derangement) 치환, g_corr(coupling 표시), 집단별 | `raw/correction_interventions.csv`, `analysis/i23_stats.json` |
| I24-A | Y_seq/Y_single/Y_0 분해(bank B r .5/1/2 + 축 r1), known-inverse seq vs native | `raw/interpolation_controls.csv`, `analysis/i24_stats.json` |
| I25-A | g_r/g_o(ε = 축 r1 4 개 고정 SG), stem/joint/head norm, weighted ratio, cosine(near-zero NA) | `raw/gradient_interventions.csv`, `analysis/i25_stats.json` |
| X40 | 관계 × 센서 행렬(measured/zero_shot/pending_dependency/not_implemented) + matched 대조(EuCd↔EuCu on e_base·contrast, EuCd↔EdCd on q; caliper .2 CAL SD, 1:1 NN) | `analysis/cross_sensor_results.csv`, `analysis/matched_sample_pairs.csv` |
| REPORT | stage 상태표(complete/pending_dependency/pending_compute/not_implemented), coverage, M1–M7 판정(기본 insufficient; M1·M3·M6 만 자동 근거) | `report_SMEC12.md`, `analysis/hypothesis_verdicts.csv` |

상세 subset: primary L1E4(+REP) 의 네 집단당 ≤32 × DISC/CONF(source 라운드로빈, seed 고정) — `manifests/detail_ids_*.json`. stage 는 증분이라 QB/GF2 모델이 생길 때마다 `tools/smec12_run.sh` 를 다시 돌리면 덧붙는다.

## 4. 검증 (s1)

- `tools/smec12_unit_tests.py` 13 검사 ALL OK (S01–S07 생성기: 이름·센서 항목·init_dir 분리/공유·registry 계약·donor/exact50K·큐 의존성·시트 범주; S10–S14 backbone: probe 정의·q_const·label·warp 부호·asset registry·fixed residual injection 의 "원본 P 한 번 sampling").
- smoke: B01/B02 × QB/GF2 통과.
- A00: WV3 자산 10 개 available, QB/GF2 10 pending, unit 검사 전부 통과.
- D10 → I20 → I23 → I24 → I25 → X40 → REPORT 를 WV3(+WV2 zero-shot) 로 실행한 결과는 §5 (아래) 에 적는다.

## 5. WV3 backbone 실행 결과 (검증용; 결과 문서가 아니다)

(실행 뒤 채움)

## 6. s2 절차

```bash
git pull
./tools/smec12_prepare.sh --dry-run     # 데이터·gate·config·ledger 검사
./tools/smec12_prepare.sh               # 현재 PAKD50 재배정 chain 이 끝나면 자동 기동 (waiter; work_dir/_smec12_s2_campaign/launch.log)
# 준비 학습 중/뒤 (증분):
setsid nohup ./tools/smec12_run.sh >> work_dir/_smec12_s2_campaign/run.log 2>&1 < /dev/null &
```

- s2 의 8-band lane(WV3 자산) 은 s2 에 PALS24/NF16 run 이 없으면 A00 이 pending 으로 표시한다 — WV3 분석은 s1(자산 보유) 의 결과를 coordinator 로 쓴다(§3.2 reference packet).
- 준비 학습이 끝나면 B02 의 exact50K sha 를 B03/B04 config 의 `donor.expected_sha256` 에 넣어 고정할 수 있다(A00 이 registry 에 기록; 현재 null = 검사 생략).

## 7. 남긴 것 (pending_compute)

D11-C, D12, I21, I22, I23-A/C/D, I24-B/C, I25-B, S30, C50, R60. 계획 §19.3 우선순위(전 센서 sampling·primary atlas → primary 상세 개입 → 확인 seed) 대로 backbone 이 먼저이고, 위 항목은 같은 raw 표(join key: sample_id·source_group·model·part) 위에 추가한다.
