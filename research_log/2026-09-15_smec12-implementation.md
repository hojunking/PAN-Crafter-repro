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

## 2. 준비 학습 (a): `tools/gen_smec12_bootstrap.py` + `config/queues/smec12_<server>.txt` + `tools/smec12_prepare.sh`

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

s1, 2026-09-15 10:30–11:30, `work_dir/_smec12_s1_campaign/report_SMEC12.md`. 자산: WV3 P0/DON-N2/L000×3 seed/L1E4 S2025/L1E4-REP×2 seed(available) 중 분석 model 6 (L000 S1234/S2025/S7777 · L1E4 S2025 · L1E4-REP S1234/S7777) × best_hqnr/last. QB/GF2 는 pending_dependency(준비 학습 전). 관측 단위·panel·probe 는 §3. stage 시간: D10 5.2 min(첫 실행)/2.7 min · I20 0.5 · I23 0.3 · I24 0.3 · I25 0.8 · X40/REPORT < 0.5 → WV3 backbone 한 바퀴 ≈ 10 min(학습과 GPU 공유).

| 항목 | primary (WV3 L1E4 S2025 best_hqnr, DISC n 1536 / 48 source) | 다른 model·panel |
|---|---|---|
| D10 raw ρ(q, e) | **−0.169** CI [−0.223, −0.117] · r .25/.5/1/2: −0.068/−0.085/−0.128/−0.190 · LOSO [−0.179, −0.158] | 12 행 전부 음(−0.10 … −0.24, CI 상한 < 0); last 가 best_hqnr 보다 약간 덜 음 |
| D11 대비 정규화 ρ(q, e/contrast) | **+0.498** CI [0.452, 0.542] · ρ(q, e_base) −0.548 · ρ(q, texture) −0.575 | 전부 양(+0.50 … +0.60) → 부호가 뒤집힌다 |
| 네 집단 중앙값 (e / q / contrast) | EdCd .0129/.312/.384 · EdCu .0093/.330/.212 · EuCd .0223/.312/.636 · EuCu .0247/.332/.411 · boundary_near 0.21 | 같은 순서 |
| I20 s_out(.25 px) | EdCd .0077 · EdCu .0057 · **EuCd .0176** · EuCu .0129 (n 32/집단) | REP S1234/S7777 같은 순서 |
| I20 EuCd − EuCu | DISC **+0.00463** CI [−0.00107, +0.00943] · CONF +0.00571 CI [+0.00294, +0.00860] | REP S1234 +0.00388 · S7777 +0.00351 |
| I23-B g_corr = e(0) − e(c0) | +0.00086 CI [0.00071, 0.00101] (양 0.98) · wrong−learned +0.0036 · CAL-median−learned +0.0012 · shuffle−learned +0.0015 · by quadrant EuCd 최대 +0.0017 | REP +0.00085 … +0.00124 (양 0.95–0.98) |
| I24-A r=1 | seq−single 0.0065 · single−0 0.0364 → interp share 0.18 · known-inverse seq vs native 7.1e-3 | CONF 동일 |
| I25-A cos(g_r, g_o) | −0.081 DISC / −0.062 CONF (음 비율 0.60) · 1e-4‖g_o‖/‖g_r‖ 0.016–0.020 | REP S1234 −0.02/−0.06 · REP S7777 **+0.03/+0.07** (부호 seed 의존) |
| WV2 zero-shot (RR 20 scene × 16 tile) | raw ρ −0.366 CI [−0.543, −0.188] · 대비 정규화 −0.087 CI [−0.188, +0.024] | 3 model 전부 raw −0.37 … −0.41, 정규화 −0.06 … −0.14 (WV3 와 달리 부호가 뒤집히지 않음) |
| X40 matched pair | EuCd~EuCu 169 · EuCd~EdCd 339 (Δcontrast +0.23, Δe_base +0.040) | WV2 52 / 63 |

판정(`analysis/hypothesis_verdicts.csv`): **M1 supported_within_condition**(정규화만으로 원인 확정 아님) · **M3 insufficient**(규칙: DISC CI 가 0 을 포함; CONF·REP 두 model 은 양) · M2/M4/M5/M7 insufficient(pending_compute) · M6 insufficient(QB/GF2 pending). 열린 문제: X40 의 "동일 residual 출력 민감도" matched 행이 `measured_unmatched` 다 — I20 상세 subset(집단당 ≤32) 안에 matched pair 가 1 개뿐이라, matched pair 위에서 I20 을 따로 돌려야 한다(다음 release).

검증 중 고친 것: D10 이 `PanFeeder(max_pixel=…)` 로 죽었다(feeder 에 그 인자가 없다 — 인자 없이 만들고 `fr.max_pixel` 을 확인) → model 단위로 즉시 기록하도록 바꿈(첫 실행 30 분 손실) · REPORT 의 M3 가 알파벳순으로 REP model 을 primary 로 집었다 → `|L1E4|S2025|` 로 고정하고 REP 는 따로 · X40 matched s_out 이 model 을 섞어 pooling 했다 → primary 만, WV2 는 CAL 이 없어 RRTILE 을 대신 쓴다.

## 6. 실행 서버 절차 (s1/s2 공용 — 큐 `config/queues/smec12_<server>.txt`)

```bash
git pull
./tools/smec12_prepare.sh --dry-run     # 데이터·gate·config·ledger 검사 (s1 12:00 통과: QB 17139 · GF2 19809 patch, S01–S14 ALL OK)
./tools/smec12_prepare.sh               # 현재 chain 이 끝나면 자동 기동 (waiter; work_dir/_smec12_<server>_campaign/launch.log)
# 준비 학습 중/뒤 (증분):
setsid nohup ./tools/smec12_run.sh >> work_dir/_smec12_<server>_campaign/run.log 2>&1 < /dev/null &
```

- WV3 lane(8-band 자산: PALS24/NF16/PO10 run) 은 **s1 에만 있다** — s1 에서 돌리면 A00 이 available, 다른 서버에서는 pending 으로 표시되고 WV3 분석은 s1 결과를 coordinator 로 쓴다(§3.2 reference packet). 이것이 §8 의 s1 이관 근거다.
- 준비 학습이 끝나면 B02 의 exact50K sha 를 B03/B04 config 의 `donor.expected_sha256` 에 넣어 고정할 수 있다(A00 이 registry 에 기록; 현재 null = 검사 생략).

## 7. 남긴 것 (pending_compute)

D11-C, D12, I21, I22, I23-A/C/D, I24-B/C, I25-B, S30, C50, R60. 계획 §19.3 우선순위(전 센서 sampling·primary atlas → primary 상세 개입 → 확인 seed) 대로 backbone 이 먼저이고, 위 항목은 같은 raw 표(join key: sample_id·source_group·model·part) 위에 추가한다.

## 8. 서버 교체 (2026-09-15 12:00 준비 → 12:34 사용자 결정으로 s1 기동)

사용자가 "SMEC12 는 WV3 자산이 있는 s1 에서, 지금 s1 에서 도는 DCR12 는 s2 에서" 를 제안했다. 검토 결과 맞는 방향이라 **기동만 남기고 준비**했다 (DCR12 쪽은 `research_log/2026-09-15_dcr12-implementation.md` §5).

- s1 의 DCR12 학습 chain 은 11:46 에 `work_dir/cases_deadline.txt` 를 과거로 돌려 **JK0 S1234(≈12:10 완료) 뒤 FQ/JK0 S777 을 시작하지 않게** 잡아 두었다(체크포인트·완료본 손실 없음; 되돌리기 = `./tools/campaign_start.sh --queue config/queues/dcr12_s1.txt --hours 20` + runner 재기동).
- s1 용 큐 `config/queues/smec12_s1.txt`(생성기 `--server s1`; config 10 벌은 s2 와 같은 파일) · `smec12_prepare.sh --dry-run` s1 통과 · 40 h ledger 생성. **기동(`./tools/smec12_prepare.sh`) 은 사용자 확인 뒤** — waiter 가 JK0 S1234 종료 뒤 자동으로 chain 을 연다. 예상: 10 run × ≈2.3 h ≈ 23–26 h(s1 W112 50K 실측 1.9–2.5 h) → 09-16 저녁, 그 뒤 분석 backbone(WV3+QB+GF2+WV2) ≈ 1 h.
- 교체하지 않으면(DCR12 를 s1 에서 마저) SMEC12 는 s1 에서 DCR12 뒤(≈20:00) 시작하거나 s2 에서 돌리되 WV3 lane 없이 간다.
- **12:34 기동**: `./tools/smec12_prepare.sh`(s1) → chain 즉시 시작(B01_QB_P0 부터, 큐 10 run, 마감 09-17 04:34, 감시자 cron). 분석 runner 는 `work_dir/_smec12_s1_campaign/analyze_when_done.sh`(detached) 가 chain DONE 뒤 `smec12_run.sh` 를 전 센서로 한 번 돌린다(그 전에 손으로 돌려도 증분). s2 는 SMEC12 를 돌리지 않는다.
