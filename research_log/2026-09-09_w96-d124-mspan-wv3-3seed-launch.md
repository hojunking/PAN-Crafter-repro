# 2026-09-09 — 새 baseline(W96·D124 · MS+PAN · 단일 task) WV3 3-seed 실행 준비 (s3)

기준 문서: [`PAN_research_baseline_W96_D124_2026-09-09.md`](PAN_research_baseline_W96_D124_2026-09-09.md) §1·§3.
이 문서는 그 고정 기준을 **실행 config 로 옮긴 것**이다. 결과·해석은 s3 에서 기동한 뒤 `results_log/` WIP 문서에 쓴다(CONVENTION §2).

## 1. config 가 기준 문서를 어떻게 구현하는가

| 기준 문서 항목 | config 키 | 값 | 비고 |
|---|---|---|---|
| U-Net width 96 · depth [1,2,4] | `model_args.hidden_size` / `depth` | 96 / [1,2,4] | 골격은 `model.pancrafter_paper.PANCrafterPaper`(attention 없음이면 일반 U-Net) |
| 입력 MS+PAN 만 (WV3 9ch) | `model_args.in_mode` | `paper` | 첫 conv 입력 = concat(PAN, bicubic↑MS). LPAN·PAN−LPAN 채널 없음 |
| PAN reconstruction task·loss 제거 | `mars` | `ms` | `train.py` single_mode: batch 복제·PAN mode forward·PAN loss 자체가 없다(계수 0 이 아님) |
| dual MARs 제거 → mode γ/β | `model_args.mode_modulation` | **`false`** | 단일 mode 에선 상수 affine 이라 LN affine 과 중복. 기준 문서 §3.3 이 "코드 확인 후 기록" 으로 남긴 항목 — 여기서 **제거** 로 정했다. 되돌리려면 생성기 `MODE_MOD` 한 줄 |
| attention/CM3A 미사용 | `attn_locations` | `[]` | |
| nocrop | `train_feeder_args.crop` | `False` | |
| bicubic 잔차 base · M-frame | `res` / `residual_base` | `True` / 기본(bicubic) | 출력 = bicubic↑MS + 잔차 |
| 실행 조건(기존 유지) | optimizer 등 | AdamW 1e-4 · wd 0.01 · cosine · warmup 100 · 50K · batch 48 | 기준 문서 §6: 확정사항 아님 → 기존 값 유지로 기록 |
| best 선택·보고 | `select_on` / `fr_select_indices` / FR dataroot | `hqnr` / `"0-19"` / `full_examples_mat20` | 논문 세트 20장 전체. 12-19 부분집합 없음 |
| seed | `seed` | 2025 · 1234 · 7777 | |

params 실측 **2.123 M** (mode_modulation 을 켠 같은 골격 `MS1_w96_9ch_msonly` 는 2.1268 M).

실행명은 `BASE_W96_D124_MSPAN_WV3_S<seed>` (서술형; 시트 범주 ⑰ BASE96, `gspread/sheet_categories.py`).
생성기 `tools/gen_w96_d124_mspan_configs.py`, 큐 `config/queues/base_w96_d124_mspan_wv3_3seed.txt`.
smoke(`tools/smoke_cases.py`)는 s1 에서 3벌 모두 통과했다(build·params·forward/backward·FR 512² forward·실배치 step).

## 2. s3 에서 할 일

```bash
git pull
cat gspread/server.txt                      # s3 여야 한다 (시트 탭 WV3-s3)
export PANCRAFTER_DLPAN=/path/to/DLPan-Toolbox
./tools/metric_v2_prepare.sh                # 아직이면 — 지표 v2 + 논문 세트 h5(full_examples_mat20). 이미 했으면 생략
python tools/smoke_cases.py BASE_W96_D124_MSPAN_WV3_S2025 BASE_W96_D124_MSPAN_WV3_S1234 BASE_W96_D124_MSPAN_WV3_S7777
./tools/campaign_start.sh --queue config/queues/base_w96_d124_mspan_wv3_3seed.txt --hours 24 --label base-w96-mspan
```

- s3 에 arch-multiset 체인이 아직 돌고 있으면 그것이 `[cases] DONE` 을 찍은 뒤 기동한다(체인은 하나씩).
- run 이 끝날 때마다 `tools/_upload.sh` 가 논문 세트 평가(`fr_mat20.json`) → 시트 `WV3-s3` 업로드를 한다. WV2 zero-shot 은 이 계열에 만들지 않는다.
- 소요: W96 단일 mode 는 W168 dual(≈3.2 h/run, s1)의 절반 이하 연산이다. s1 기준 run 당 1~1.5 h 추정, 3벌 ≈ 4~5 h. s3(5090)은 더 빠르다. 기동 시 `results_log/` 에 WIP 문서를 만든다.

## 3. 해석 원칙 (기준 문서 §7)

- 이 3벌은 새 기준의 **일반 HRMS 복원 baseline** 이다. 이후 정합/fitting 후보는 이 3벌과 비교한다.
- 과거 W168·d123 dual(`S1_T05_W168_D123_DUAL`, ARCH 캠페인) 결과를 새 방법의 직접 대조군으로 쓰지 않는다.
- 판정은 HQNR → SCC. 정렬 축의 실험은 `results_log/2026-09-09_alignment-after-training-analysis.md` §3 에 따라 fSCC·δ_out 을 병기한다.
