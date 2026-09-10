# NA104 — W104·D122 no-align KD 구현 노트 (2026-09-11)

명세: `research_log/PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md` (s2·s3 두 서버 실행).
**정합 모듈이 없는 동일 골격에서 어떤 KD/통계 감독이 추가 fitting과 복원 품질에 기여했는가**만 묻는다. 정합 연구(PA·PO10·NF16·KDV)의 결과를 이 골격으로 전용하지 않으며, 그 반대도 아니다.

## 1. 계획 ↔ 저장소 대조

| 계획 | 저장소 | 비고 |
|---|---|---|
| W104 · depth [1,2,2] · 9→8ch · residual base 1회 | `model_args.hidden_size 104`, `depth [1,2,2]`, `in_mode paper`, `mars: ms` | 실측 **2.0989 M** (W96·D124 2.23 M, W112·D123 2.66 M 과 다른 골격 — 과거 수치를 옮기지 않는다) |
| aligner 없음 · PAN warp 없음 | `kdv.aligner_policy: A-ID` → `PAModel(backbone, aligner=None, sampler=False)` | Δ=0 이고 U-Net 입력이 원 PAN 과 **비트 동일**(gate NA02) |
| 학습 forward 에 어떤 PAN warp 도 금지 | 새 키 `kdv.na_protocol: NA-STRICT` | resolver 가 corruption·offset·geometry·이동량 KD·C probe 를 **전부 오류로 막는다** (조용히 0 으로 두지 않는다) |
| Teacher T00 (같은 골격, plain GT L1) | `NA104_T00_…_S2025_v1`, teacher id `T104_v1` | Student(seed 1234) 와 seed 분리. Teacher checkpoint 는 **사전 선언한 selector `best_rr_val`** 로 고정 |
| REC-N0/R0/R1/R2/R3 | `kdv.rec.case` (기존 `GTAnchoredReconstructionKD`) | τ_R = median e_T (train-only, cohort 공유), α=1·β=0.1·ε=1e-6 시작값 |
| STAT-IV/GV/GC/SC + **STAT-M2** × H/T/FIX/WH/AD | `kdv.stat.kind/mode` — `M2`(비중심 E[ggᵀ]) 를 새로 구현 | M2 는 중심화하지 않는 것이 정의라 공간 상수 제거도 하지 않는다 (gate ST02: M2 − GC = μμᵀ) |
| REP-W3/W7/MULTI357 · STD/LOGVAR · RESIDUAL · GVSC | `stat.window` / `stat.windows` / `stat.transform` / `stat.domain` / `stat.extra[]` | 창마다 τ_V 를 따로 재고, 표현이 바뀌면 λ_V 도 다시 잰다. 공분산 원소에 sqrt/log 는 resolver 가 막는다 |
| TRI-A(SIGN/COS/CAP/BANDADV) · TRI-B(SIGN/CAP/COMPADV, GC/SC 포함) | 기존 `kdv/tri.py` 그대로 | A 는 R3 의 soft 를 **대체**, B 는 통계 soft 를 대체 (중복 합산 없음, gate KD04) |
| C 계열은 no-align 에서 재정의 (TRI-C-NASENS) | `kdv.na_protocol: NA-TSENS` + `tri.c.mode: sens` | Teacher **probe 에서만** PAN 을 ±h 옮긴다. 학습 입력·target·모델·추론은 그대로. δ=0 기준, ROI 경계 margin 4 밖은 감쇠 없음(r=1) |
| C-DIAG/FULL/QSCALAR/EQPROXY | resolver 가 A-ID 에서 **거부** | 이동량 covariance 가 없다 — 임의 identity/0 으로 채우지 않는다 (정합 캠페인으로 유보) |
| 필수 대조군 CTL-* | `rec.control`(hscale/rshuffle) · `tri.a/b.mode`(mass/shuffle) · `tri.c.control: mass` · `rec.tau_scale` · `rec.kd_weight` · `stat.lambda_scale` | 총계수 대응 ρ = Σ b m / Σ b 는 broadcast 된 같은 shape 에서 (gate TRI03) |
| 주 selector = best_rr_val + 고정 final-N | `kdv.select.primary: best_rr_val`, `aligned_selector: false` | best_hqnr(=best_raw) 는 **FR test 로 고른 test-adaptive exploratory** 로 명시 기록. aligned view/selector 는 만들지 않는다 |
| cost_teacher / cost_student / cost_total | `cost_manifest.json` (run 마다) | Teacher 공유 시 amortized 도 적되 최초 준비비를 총계에서 숨기지 않는다 |
| 상태 enum (§19.2) | `work_dir/_na104_campaign/…/case_status.csv` | 실행하지 않은 case 에 성능값을 채우지 않는다 |

## 2. 구현 지도

| 파일 | 이번에 더한 것 |
|---|---|
| `kdv/losses_stat.py` | `grad_moment2`(STAT-M2, 비중심) · `stat_transform`(std/logvar) · `stat_maps`(변환 포함 S/T/G 추출) · `stat_term(transform=…)` |
| `kdv/registry.py` | `M2` kind · `REC_CONTROLS`(hscale/rshuffle) · `STAT_TRANSFORMS`/`STAT_DOMAINS`/다중 창 · `stat.extra[]`(두 번째 통계 항) · `tri.c.control/roi_margin/apply` · `NA_PROTOCOLS`(NA-STRICT/NA-TSENS) · `select.primary/aligned_selector` · `teacher.eval_only` · `baseline_run` · `rec_tag`/`stat_tag` 토큰 확장 |
| `kdv/tri.py` | `roi_gate` — 유한차분이 검증되지 않은 경계 margin 에서는 감쇠 없이(r=1) 원래 soft 로 복귀 |
| `kdv/calibration.py` | τ_V·λ_V·성분 τ 를 **표현(창·변환·도메인)마다** 재도록 인자 확장, `stat_view`(residual 도메인) |
| `train_kdv.py` | rec 대조군(hscale/rshuffle) · 통계 다중 창/변환/residual · 보조 통계 항 · C 진단 전용(`apply:false`)·ROI·총계수 대조 · aligned selector 끄기 · 주 selector 선언 manifest · `cost_manifest.json` · 검증셋 fitting 지표(MAE·edge·통계 오차) |
| `tools/gen_na104_configs.py` | 87 case 생성기 (T00·Q00–Q47·CS·CTL·VX·TCOPY·LONG·CONT·COSTMATCH) + 서버 block 큐 |
| `tools/na104_unit_tests.py` | 계획 §18 gate 전부 (NA01–NA04·CK01·KD01–KD04·ST01–ST04·TRI01–TRI03·EV01·RS01·PERF01·CS01) |
| `tools/na104_prepare.sh` | §17.1 실행 전 확인값(모델·GPU·latency) → gate → config → 캠페인 manifest·case 상태 → 기동 |
| `gspread/sheet_categories.py`, `refile_sheet.py`, `tools/_upload.sh` | 시트 범주 ㉒ `NA104` |

기존 캠페인(S2W112·NF16) 47벌은 이름·resolve 결과가 **바뀌지 않는다**(전부 기본값). 새 축은 값이 기본이 아닐 때만 이름 토큰이 붙는다.

## 3. 약명 → 세팅 (규약: 보고서·시트·config 에 약명 단독 금지)

실행명 자체가 `NA104_<id>_W104_D122_WV3_<rec>_<stat>[_TRI_A*_B*_C*]_S<seed>_v1` 로 설정을 담는다. id 의 뜻은 아래 표에서만 읽는다.

| id | rec | 통계 | 방향 gate | Teacher | N | seed | 서버 | 무엇을 묻는가 |
|---|---|---|---|---|---:|---:|---|---|
| `CONTGVAD` | R3 | GV-AD | — | 학습 | 25000 | 1234 | s2 | 장기 CONTINUATION: 공통 N checkpoint(Q00/last) 에서 같은 tail schedule 로 |
| `CONTN0` | N0 | OFF | — | 평가 bin 전용 | 25000 | 1234 | s2 | 장기 CONTINUATION: 공통 N checkpoint(Q00/last) 에서 같은 tail schedule 로 |
| `CONTR3` | R3 | OFF | — | 학습 | 25000 | 1234 | s2 | 장기 CONTINUATION: 공통 N checkpoint(Q00/last) 에서 같은 tail schedule 로 |
| `CS00` | R3 | OFF | C:sens(진단만) | 학습 | 50000 | 1234 | s2 | Q04 와 같은 학습 + Teacher PAN 민감도 진단만 (감쇠 미적용) — 새 학습 알고리즘이 아니다 |
| `CS01` | R3 | OFF | C:sens | 학습 | 50000 | 1234 | s2 | C-NASENS: Teacher 출력의 입력 민감도 r = s/(s+|J|²) 로 rec soft 감쇠 |
| `CS02` | R3 | OFF | A:sign C:sens | 학습 | 50000 | 1234 | s2 | A-SIGN × C-NASENS |
| `CS03` | R3 | GV-AD | B:sign C:sens(통계 J) | 학습 | 50000 | 1234 | s2 | GV-B-SIGN 통계 soft 에 통계-J 기반 감쇠 (pixel J 재사용 금지 — 통계 Jacobian 을 다 |
| `CTLAMASS` | R3 | OFF | A:mass | 학습 | 50000 | 1234 | s2 | 대조: A-SIGN 과 soft 총계수량만 맞춘 스칼라 감쇠 (방향 선택 vs 단순 KD 감소) |
| `CTLASHUF` | R3 | OFF | A:shuffle | 학습 | 50000 | 1234 | s2 | 대조: A mask 의 공간 위치만 섞음 (올바른 위치·부호 정보의 필요성) |
| `CTLBETA03` | R3 β=0.3 | OFF | — | 학습 | 50000 | 1234 | s2 | 의존성: β_R 0.3 (KD 강도, 1 미만 유지) |
| `CTLBETA05` | R3 β=0.5 | OFF | — | 학습 | 50000 | 1234 | s2 | 의존성: β_R 0.5 |
| `CTLBMASS` | R3 | GV-AD | B:mass | 학습 | 50000 | 1234 | s2 | 대조: B mask 의 soft 총계수 대응 |
| `CTLBSHUF` | R3 | GV-AD | B:shuffle | 학습 | 50000 | 1234 | s2 | 대조: B mask 위치 섞음 |
| `CTLCMASS` | R3 | OFF | C:sens(총계수 대조) | 학습 | 50000 | 1234 | s2 | 대조: C-NASENS 와 soft 총계수 대응 (민감도 정보 vs KD 감소) |
| `CTLHSCALE` | R1 [hscale] | OFF | — | 학습 | 50000 | 1234 | s2 | 대조: R1 의 공간 w_H 를 batch 평균 스칼라로 (어려운 위치 지도 vs 단순 loss scale) |
| `CTLLAMV03` | R3 | GV-AD λ×0.3 | — | 학습 | 50000 | 1234 | s2 | 의존성: λ_V ×0.3 (통계 loss scale) |
| `CTLLAMV30` | R3 | GV-AD λ×3.0 | — | 학습 | 50000 | 1234 | s2 | 의존성: λ_V ×3 |
| `CTLRSHUF` | R3 [rshuffle] | OFF | — | 학습 | 50000 | 1234 | s2 | 대조: R3 의 d_T·a_T 를 함께 공간 permutation (실패 지도의 위치 정보) |
| `CTLTAU05` | R3 τ×0.5 | OFF | — | 학습 | 50000 | 1234 | s2 | 의존성: τ_R ×0.5 |
| `CTLTAU20` | R3 τ×2.0 | OFF | — | 학습 | 50000 | 1234 | s2 | 의존성: τ_R ×2 |
| `LONG2NGVAD` | R3 | GV-AD | — | 학습 | 100000 | 1234 | s2 | 장기 HORIZON: 처음부터 2N schedule 로 R3+GV-AD (N 시점 LR 이 다르므로 N-horizo |
| `LONG2NN0` | N0 | OFF | — | 평가 bin 전용 | 100000 | 1234 | s2 | 장기 HORIZON: 처음부터 2N schedule 로 N0 (N 시점 LR 이 다르므로 N-horizon 의 pr |
| `LONG2NR1` | R1 | OFF | — | 학습 | 100000 | 1234 | s2 | 장기 HORIZON: 처음부터 2N schedule 로 R1 (N 시점 LR 이 다르므로 N-horizon 의 pr |
| `LONG2NR3` | R3 | OFF | — | 학습 | 100000 | 1234 | s2 | 장기 HORIZON: 처음부터 2N schedule 로 R3 (N 시점 LR 이 다르므로 N-horizon 의 pr |
| `Q00` | N0 | OFF | — | 평가 bin 전용 | 50000 | 1234 | s2/s3 | 독립 Student 기준: KD 없음·통계 없음 (N0). Teacher 숫자 하나로 대신하지 않는다 (§3.3) |
| `Q01` | R0 | OFF | — | 학습 | 50000 | 1234 | s2 | 일반 고정가중 output KD (R0) — Q00 대비 KD 자체의 이득 |
| `Q02` | R1 | OFF | — | 학습 | 50000 | 1234 | s2 | Teacher 실패 지도로 GT hard 만 재가중 (R1) — Teacher output 을 target 으로 쓰 |
| `Q03` | R2 | OFF | — | 학습 | 50000 | 1234 | s2 | Teacher 오차로 hard/soft 배분 (R2) — Q02 대비 Teacher output 추가 |
| `Q04` | R3 | OFF | — | 학습 | 50000 | 1234 | s2/s3 | adaptive: Student 우위 영역의 모방 해제까지 (R3) — Q03 대비 a_T gate, Q02 대비  |
| `Q05` | N0 | GV-H | — | 평가 bin 전용 | 50000 | 1234 | s2 | Teacher 없이 GT gradient-variance 통계만 (N0+GV-H) — 통계 supervision 과 |
| `Q06` | R3 | GV-H | — | 학습 | 50000 | 1234 | s2 | R3 + GT 구조 통계 (GV-H) |
| `Q07` | R3 | GV-WH | — | 학습 | 50000 | 1234 | s2 | R3 + Teacher 통계 실패 지도로 GT 통계 hard 재가중 (GV-WH) |
| `Q08` | R3 | GV-T | — | 학습 | 50000 | 1234 | s2 | R3 + Teacher 통계만 (GV-T) — 픽셀 GT hard 는 유지 |
| `Q09` | R3 | GV-FIX | — | 학습 | 50000 | 1234 | s2 | R3 + 고정 hard/soft 통계 KD (GV-FIX) |
| `Q10` | R3 | GV-AD | — | 학습 | 50000 | 1234 | s2/s3 | R3 + adaptive 통계 KD (GV-AD) — 통계 축의 주력 후보 |
| `Q11` | N0 | EDGE-H | — | 평가 bin 전용 | 50000 | 1234 | s2 | Teacher 없이 signed Scharr edge L1 (N0+EDGE-H) — variance 고유 효과와 가 |
| `Q12` | R3 | EDGE-H | — | 학습 | 50000 | 1234 | s2 | R3 + signed edge (EDGE-H) — Q06 대비 일반 edge 대조 |
| `Q13` | R3 | OFF | A:sign | 학습 | 50000 | 1234 | s2 | R3 soft 에 band 별 GT-방향 gate (TRI-A-SIGN) — GT 와 충돌하는 band 의 모방 해 |
| `Q14` | R3 | OFF | A:cosine | 학습 | 50000 | 1234 | s2 | R3 soft 에 pixel cosine 감쇠 (TRI-A-COS) — scalar 방향 대조 |
| `Q15` | R3 | OFF | A:sign_cap | 학습 | 50000 | 1234 | s2 | R3 soft 에 sign + 크기비 cap (TRI-A-CAP) |
| `Q16` | R3 | OFF | A:band_adv | 학습 | 50000 | 1234 | s2 | R3 soft 를 band 별 (1−d_c)a_c 로 세분 (TRI-A-BANDADV) — 평균 gate 가 놓친  |
| `Q17` | R3 | GV-AD | B:sign | 학습 | 50000 | 1234 | s2 | GV-AD 통계 soft 에 성분별 방향 gate (TRI-B-SIGN) |
| `Q18` | R3 | GV-AD | B:sign_cap | 학습 | 50000 | 1234 | s2 | GV-AD 통계 soft 에 sign + 크기비 (TRI-B-CAP) |
| `Q19` | R3 | GV-AD | B:comp_adv | 학습 | 50000 | 1234 | s2 | GV-AD 통계 soft 를 성분별 τ_V,j·a_j 로 (TRI-B-COMPADV) |
| `Q20` | R3 | IV-H | — | 학습 | 50000 | 1234 | s3 | R3 + 국소 밝기 분산 통계 (IV-H) |
| `Q21` | R3 | IV-WH | — | 학습 | 50000 | 1234 | s3 | R3 + 국소 밝기 분산 통계 (IV-WH) |
| `Q22` | R3 | IV-T | — | 학습 | 50000 | 1234 | s3 | R3 + 국소 밝기 분산 통계 (IV-T) |
| `Q23` | R3 | IV-FIX | — | 학습 | 50000 | 1234 | s3 | R3 + 국소 밝기 분산 통계 (IV-FIX) |
| `Q24` | R3 | IV-AD | — | 학습 | 50000 | 1234 | s3 | R3 + 국소 밝기 분산 통계 (IV-AD) |
| `Q25` | R3 | GC-H | — | 학습 | 50000 | 1234 | s3 | R3 + 방향 gradient 공동 변화 통계 (GC-H) |
| `Q26` | R3 | GC-WH | — | 학습 | 50000 | 1234 | s3 | R3 + 방향 gradient 공동 변화 통계 (GC-WH) |
| `Q27` | R3 | GC-T | — | 학습 | 50000 | 1234 | s3 | R3 + 방향 gradient 공동 변화 통계 (GC-T) |
| `Q28` | R3 | GC-FIX | — | 학습 | 50000 | 1234 | s3 | R3 + 방향 gradient 공동 변화 통계 (GC-FIX) |
| `Q29` | R3 | GC-AD | — | 학습 | 50000 | 1234 | s3 | R3 + 방향 gradient 공동 변화 통계 (GC-AD) |
| `Q30` | R3 | SC-H | — | 학습 | 50000 | 1234 | s3 | R3 + 8밴드 분광 공동 변화 통계 (SC-H) |
| `Q31` | R3 | SC-WH | — | 학습 | 50000 | 1234 | s3 | R3 + 8밴드 분광 공동 변화 통계 (SC-WH) |
| `Q32` | R3 | SC-T | — | 학습 | 50000 | 1234 | s3 | R3 + 8밴드 분광 공동 변화 통계 (SC-T) |
| `Q33` | R3 | SC-FIX | — | 학습 | 50000 | 1234 | s3 | R3 + 8밴드 분광 공동 변화 통계 (SC-FIX) |
| `Q34` | R3 | SC-AD | — | 학습 | 50000 | 1234 | s3 | R3 + 8밴드 분광 공동 변화 통계 (SC-AD) |
| `Q35` | N0 | IV-H | — | 평가 bin 전용 | 50000 | 1234 | s3 | Teacher 없이 IV-H — 학습에도 Teacher 가 필요 없는 GT-only 대조 |
| `Q36` | N0 | GC-H | — | 평가 bin 전용 | 50000 | 1234 | s3 | Teacher 없이 GC-H — GT-only 대조 |
| `Q37` | N0 | SC-H | — | 평가 bin 전용 | 50000 | 1234 | s3 | Teacher 없이 SC-H — GT-only 대조 |
| `Q38` | R3 | GC-AD | B:sign | 학습 | 50000 | 1234 | s3 | GC-AD 통계 soft 에 성분별 방향 gate (GC-B-SIGN) |
| `Q39` | R3 | SC-AD | B:sign | 학습 | 50000 | 1234 | s3 | SC-AD 통계 soft 에 성분별 방향 gate (SC-B-SIGN) |
| `Q40` | R3 | GV-AD | A:sign | 학습 | 50000 | 1234 | s3 | A-SIGN(픽셀) + GV-AD(통계) — 2×2 대응의 A 셀 |
| `Q41` | R3 | GV-AD | A:sign B:sign | 학습 | 50000 | 1234 | s3 | A-SIGN + GV-B-SIGN — 2×2 대응의 AB 셀 |
| `Q42` | R3 | GV-H + SC-H | — | 학습 | 50000 | 1234 | s3 | R3 + GV-H + SC-H (두 GT 통계 항의 합, λ 는 각 표현에서 따로) |
| `Q43` | R3 | GV-AD + SC-AD | — | 학습 | 50000 | 1234 | s3 | R3 + GV-AD + SC-AD (두 adaptive 통계 항의 합) |
| `Q44` | R3 | GC-AD | B:sign_cap | 학습 | 50000 | 1234 | s3 | GC-AD + B-CAP |
| `Q45` | R3 | GC-AD | B:comp_adv | 학습 | 50000 | 1234 | s3 | GC-AD + B-COMPADV |
| `Q46` | R3 | SC-AD | B:sign_cap | 학습 | 50000 | 1234 | s3 | SC-AD + B-CAP |
| `Q47` | R3 | SC-AD | B:comp_adv | 학습 | 50000 | 1234 | s3 | SC-AD + B-COMPADV |
| `T00` | N0 | OFF | — | — | 50000 | 2025 | s2/s3 | 정식 Teacher: 같은 W104·D122 no-align 을 plain GT L1 로 학습 (§3.1), see |
| `TCOPYN0` | N0 | OFF | — | 평가 bin 전용 | 50000 | 1234 | s2 | 초기화: Student 를 T00 가중치로 시작한 KD 없는 continuation |
| `TCOPYR1` | R1 | OFF | — | 학습 | 50000 | 1234 | s2 | 초기화: T00 가중치에서 R1 continuation |
| `TCOPYR3` | R3 | OFF | — | 학습 | 50000 | 1234 | s2 | 초기화: T00 가중치에서 R3 continuation (초기 e_S=e_T → soft 0 에서 시작) |
| `VXLOG` | R3 | GV-AD logvar | — | 학습 | 50000 | 1234 | s3 | 표현: GV-AD 를 log(v+eps) 로 비교 (REP-LOGVAR) |
| `VXM2AD` | R3 | M2-AD | — | 학습 | 50000 | 1234 | s3 | 표현: STAT-M2-AD — GC-AD 와 비교 |
| `VXM2H` | R3 | M2-H | — | 학습 | 50000 | 1234 | s3 | 표현: 비중심 gradient 이차 모멘트 GT 대조 (STAT-M2-H) — 같은 reduction 의 GC-H  |
| `VXM357AD` | R3 | GV-AD w3/5/7 | — | 학습 | 50000 | 1234 | s3 | 표현: GV-AD 창 3/5/7 별도 loss 의 평균 |
| `VXM357H` | R3 | GV-H w3/5/7 | — | 학습 | 50000 | 1234 | s3 | 표현: GV-H 창 3/5/7 평균 |
| `VXRES` | R3 | GV-AD residual | — | 학습 | 50000 | 1234 | s3 | 표현: GV-AD 를 residual(Ŷ−M) 에서 (REP-RESIDUAL — 출력 variance 와 같지 않다 |
| `VXSTD` | R3 | GV-AD std | — | 학습 | 50000 | 1234 | s3 | 표현: GV-AD 를 sqrt(v+eps) 로 비교 (REP-STD, τ_V·λ_V 재calibration) |
| `VXW3AD` | R3 | GV-AD w3 | — | 학습 | 50000 | 1234 | s3 | 표현: GV-AD 창 3×3 |
| `VXW3H` | R3 | GV-H w3 | — | 학습 | 50000 | 1234 | s3 | 표현: GV-H 창 3×3 (같은 창의 H/AD 대응) |
| `VXW7AD` | R3 | GV-AD w7 | — | 학습 | 50000 | 1234 | s3 | 표현: GV-AD 창 7×7 |
| `VXW7H` | R3 | GV-H w7 | — | 학습 | 50000 | 1234 | s3 | 표현: GV-H 창 7×7 |

총 87 case

**같은 것을 두 번 세지 않는다** (§12.2): `TRI-A-BASE` = Q04, `TRI-B-BASE` = Q10, `R3(β=0)` = Q02(R1) — 별도 case 로 만들지 않았고 gate KD04 가 수치로 확인한다. `GC` 대각은 `GV` 와 같은 통계이므로 GC 의 기여는 **off-diagonal + all-entry reduction** 으로만 해석한다.

## 4. Gate 결과 (s1 에서 CPU·소형 GPU 확인, 2026-09-11)

| Gate | 확인한 것 | 결과 |
|---|---|---|
| NA01 | Teacher·Student 가 같은 W104·D122·9→8ch (2.0989 M), 가중치는 다름. **zero_module 로 0 초기화된 출력층을 난수화한 뒤** 검사 (초기 출력 0 트랩) | OK |
| NA02 | aligner parameter 0, 학습·평가 forward 의 **PAN warp 호출 0**, Δ=0, U-Net 입력이 원 PAN 과 비트 동일 | OK |
| NA03 | MS base = bicubic ×4, `y − residual == M` (정확히 한 번 합산) | OK |
| NA04 | 추론 forward 가 GT 를 받지 않음, 같은 입력에 결정론 | OK |
| CK01 | W112·D123 state_dict 를 W104·D122 에 strict 로드 → RuntimeError (silent partial load 없음) | OK |
| KD01–KD04 | Teacher·GT·gate 무-gradient / S=T 오답에도 GT gradient 유지 / S=GT 면 soft 0·정확한 band 의 A mask 0 / R3(β=0)=R1, A-BASE=R3, B-BASE=STAT-AD | OK |
| ST01–ST04 | population variance 명시 계산 일치·상수 영상 0 / GC 대각=GV·SC 대칭·PSD·off-diagonal 음수 보존·**M2−GC=μμᵀ** / k5 공통 중심 58²·통계 ROI 가 rec 영역을 바꾸지 않음 / S=GT 면 H loss 0·Student 통계 gradient live | OK |
| TRI01–TRI03 | SIGN/CAP/COS 손계산과 예외(u=0, v=0) / BANDADV 가 평균 gate 0 인 곳을 살리고 같은-부호 중복을 만들지 않음 / MASS 의 broadcast 분모·총계수 일치, SHUFFLE 총량 보존, RSHUFFLE 이 d·a 를 같은 순열로, HSCALE 이 총량 보존·분산 0 | OK |
| EV01 | 87 config 전부 resolve·W104/D122/A-ID/G0/offset0/geometry0 강제·주 selector best_rr_val·aligned selector 없음·FR 논문 세트 20장 공통·NA-STRICT 가 기본이고 C 계열 5벌만 NA-TSENS | OK |
| RS01 | corruption RNG checkpoint 등록 경로 유지, warm start 가 parent step 만큼 scheduler 진행(CONT/TCOPY 는 step 0 = 새 tail) | OK |
| PERF01 | 실배치에서 `rec + λ_stat·stat + λ_edge·edge` 세 항의 forward/backward/step 유한, Teacher 무-gradient | OK |
| CS01 | δ=0 기준·중심 유한차분 J 유한·비영·Teacher hash 불변, h/2 안정(상대 변화 0.00), r∈(0,1]·양의 q 없으면 r=1 fallback, 경계 margin 4 밖 감쇠 없음 | OK |

`tools/na104_unit_tests.py` → **ALL OK (0 failed)**. 이전 W96/W112 gate 통과 기록을 이 골격의 통과로 쓰지 않는다(§16.1) — 위는 전부 W104·D122 에서 다시 잰 것이다.

### 4.1 dry 체인 (s1, 200 updates, 2026-09-11 02:16–02:37)

11 case 를 `T00 → Q00 → Q04 → Q05 → Q10 → Q17 → Q42 → CS01 → CTLHSCALE → VXRES → VXM357H` 순으로 실제 데이터에 돌려 **경로만** 확인했다. 전부 `rc=0`, 36–38 산출물.
**이 수치는 결과가 아니다** (200 update = 계획 N 의 0.4%). 확인한 것은 다음뿐이다.

| 확인 | 결과 |
|---|---|
| aligner 없는 그래프 | 모든 run 에서 `Δ = 0`, `aligned_valid.hqnr == raw_valid.hqnr` (같은 view — 계획 §14.1 의 주장이 실제로 성립) |
| 산출 tag | `best_hqnr` · `best_rr_val` · `last` 셋 (`best_aligned` 없음), `selector_state_aligned.json` = `not_applicable` |
| 선택 로그 | `[select] (주 selector best_rr_val) … | best_aligned: N/A(aligner 없음) | best_rr_val step … (ERGAS_val …)` |
| Teacher 경로 | Q04 가 `T00/best_rr_val` 을 strict 로드, τ_R = 0.021348 (Q00 이 평가 전용으로 이미 계산해 둔 것을 **cache 적중**으로 재사용) |
| GT-only arm | Q00(N0) 이 Teacher 를 학습에 쓰지 않으면서도 `fitting_bins.csv` 4행(고정 Teacher 오차 구간별 Δe·WinRate) 을 남긴다 (§15.2) |
| 통계 항 두 개 | Q42 의 GV-H λ 0.96271 · SC-H λ 0.43474 — **항마다 따로** calibration (결합 전용 재calibration 없음) |
| 다중 창·residual | VXM357H(창 3/5/7 평균)·VXRES(Ŷ−M 도메인) 가 각자 τ_V 를 따로 재고 정상 학습 |
| 방향 gate | Q17 의 `routing_components.jsonl`: same_sign 0.978 · mask_mean 0.978 · soft_mass_ratio 0.984 · `soft_zero_due_to_aT` 0.017 |
| C-NASENS | CS01: s_sens 1.09e-3, FD 안정성 rel(h/2) 0.039 · rel(2h) 0.035 · 선형화 잔차 0.083, 학습 중 J_rms 0.077 · probe 63 ms/step · valid 1.00 · ROI margin 4 · r 평균 0.619 (soft 를 62% 로 감쇠) |
| 비용 분해 | `cost_manifest.json`: cost_student 0.0275 h · cost_teacher 0.0050 h · cost_total 0.0325 h · amortized 기록 |
| fitting 지표 | 로그·CSV 에 `rr_val ERGAS / MAE / edge / stat` 이 매 평가 기록 (§14.3) |
| artifact 진단 | `tools/na104_diag.py` 동작 확인 (200-update 모델: grad_ratio 0.79 · hf_ratio 0.50 → 예상대로 과도한 smoothing 상태) |

dry config·work_dir 은 확인 뒤 지웠다. 실제 캠페인은 같은 코드로 `--updates 50000` 에서 돈다.

## 5. 실행

```bash
./tools/na104_prepare.sh --no-start          # 확인값·gate·config·smoke·캠페인 manifest (서버는 gspread/server.txt)
./tools/na104_prepare.sh                     # + 체인 기동
./tools/gen_na104_configs.py --all           # 전 87벌 재생성 (config 는 손으로 고치지 않는다)
```

순서 의존: **T00**(Teacher) → **Q00**(독립 baseline, 모든 통계 case 의 λ_V pilot `<Q00>/last`) → 나머지. `CONT*` 는 `Q00/last`, `TCOPY*` 는 `T00/best_rr_val` 에서 분기한다.
진단: `tools/na104_diag.py --run <run> --ckpt best_rr_val` (계획 §14.3 artifact — 과도한 smoothing·링잉·band bias·평탄/어두운 영역 악화). `_upload.sh` 가 NA104 run 마다 `best_rr_val`·`last` 에서 자동 실행한다. **선택·판정에는 쓰지 않는다.**
산출물: run 폴더의 `checkpoint_metrics.csv`(raw_original·raw_valid + RR + rr_val ERGAS/MAE/edge/통계 오차), `fitting_bins.csv`(Teacher 오차 10/50/90 구간의 Δe·WinRate), `cost_manifest.json`, `calibration_resolved.json`, `kdv_config_resolved.json`(selection 선언 포함), 시트 범주 ㉒ NA104.

## 6. 계획 감사(다중 agent)와 반영

계획 §1–§20 을 여섯 구간으로 나눠 저장소와 대조한 감사에서 437개 요구사항을 뽑았고(EXISTS 270 / PARTIAL 127 / MISSING 34), 코드로 고쳐야 할 것은 다음이었다.

| 감사가 지적한 것 | 반영 |
|---|---|
| `A-ID + aux.offset_weight > 0` 이 **조용히 0 으로 비활성화**되고 통과했다 | resolver 오류로 승격. `A-ID + I-N/I-AEQ`(corrupted 교대) 도 오류 |
| `A-ID + TRI-C-DIAG/FULL/QSCALAR` 가 통과한 뒤 calibration 에서 `teacher.aligner=None` 로 죽었다 | resolver 가 **설정 단계에서** 거부 (이동량 covariance 부재) |
| C 계열에 유한차분 ROI 제한이 없었다 (§9.2) | `roi_gate(risk, margin=4)` — 경계에서는 감쇠 없이 원래 soft, hard 는 전 영역 유지 |
| CS00(진단만) 을 표현할 방법이 없었다 | `tri.c.apply: false` — probe·진단은 하되 soft 계수는 parent 그대로, 이름 토큰 `CSENSDIAG` |
| GV+SC 처럼 **두 통계 항의 합**을 표현할 수 없었다 (Q42/Q43) | `stat.extra[]` — 항마다 자기 τ_V·λ_V 를 따로 calibration (결합 전용 재calibration 금지) |
| GT-only arm 은 Teacher 가 없어 §15.2 의 **고정 Teacher 오차 bin** 비교에서 빠졌다 | `teacher.eval_only: true` — 학습 loss·forward 에는 전혀 쓰지 않고 평가 bin 에만 (resolver 가 모순을 막고, 서술에도 "평가 bin 전용" 으로 적힌다) |
| §14.3 의 fitting 지표(가중치 없는 MAE·GT 통계 오차·signed edge 오차) 가 없었다 | 검증셋 1회 통과에 `rr_val_mae` · `rr_val_edge_l1` · `rr_val_stat_l1` 을 추가, 로그·`checkpoint_metrics.csv` 에 기록 |
| §13.4 비용 분해가 없었다 | `cost_manifest.json`(cost_teacher/cost_student/cost_total/amortized/calibration cache 적중) |

남은 것(코드가 아니라 **운영 규약**으로 지킨다): 짧은 prefix 결과로 강한 fitting 가설을 기각하지 않기, 2N run 의 N 시점을 N-horizon 의 prefix 라고 부르지 않기, cosine LR 이 0 에 닿은 뒤 update 만 늘린 것을 추가 학습이라 하지 않기, 통계만 좋아졌을 때 복원 품질 개선으로 쓰지 않기.

### 6.1 구현 반증 검토 (6 축 × 반증 검증, 2026-09-11)

구현을 계획서 대비로 **반증하도록** 여섯 축(rec 대조군 / 통계 표현·calibration / no-align 의 C 계열 / 평가·선택·비용 / case 망라 / resolver 적대적 통과)에서 검토하고,
각 지적을 다시 독립 agent 가 반박하게 했다. 43건 중 **9건이 반박을 견뎠고 전부 반영했다**.

| 확인된 결함 | 왜 문제인가 | 반영 |
|---|---|---|
| **[P1] 재개(`--resume`) 가 모든 NA104 run 에서 깨진다** | aligned selector 를 쓰지 않는 run 이 `selector_state_aligned.json` 에 표식(`status: not_applicable`)을 썼는데, 부모 trainer 의 재개 로더가 그 파일을 `BestSelector` 로 읽어 `KeyError`. 감시자(watchdog) 가 자동 재개하므로 실제로 학습이 죽는 경로였다 | 표식을 **다른 파일명**(`aligned_selector_not_applicable.json`)으로 옮기고, 로더에도 방어(선택기 형식이 아니면 건너뛴다)를 넣었다 |
| [P2] §8.1 A/B 2×2 네 셀이 두 서버로 갈라졌다 | Q10(base)·Q40(A)·Q41(AB)은 s3, Q17(B)은 s2 — 서버마다 Teacher 가 달라 2×2 대응이 깨진다 | s3 block 에 Q06·Q13·Q17 을 넣어 **네 셀이 한 서버 안에서** 끝나게 했다 |
| [P2] Q42/Q43(GV+SC) 에 "합의 전체 배율만 줄인 대조" 가 없다 | 두 항이 각각 r_grad 0.05 로 맞춰져 통계 gradient 가 ~2배 — 결합 이득인지 단순 증폭인지 못 가른다 (§11.4 가 요구) | `CTLGVSCHALF`(같은 두 항, 각 λ ×0.5) 추가 |
| [P2] EV01 gate 가 NA-TSENS 개수를 상수로 박아둬, dry config 가 있으면 gate 가 실패해 prepare 가 중단된다 | 개수는 case 를 늘리면 바뀐다 | 개수 대신 **술어**로 판정(NA-STRICT ⇒ C off, NA-TSENS ⇒ C on)하고 `_dry` config 는 세지 않는다. 2×2 셀·큐 순서 의존도 gate 로 추가 |
| [P3] CTL-BETA 두 벌이 시트 서술에서 Q04 와 **글자까지 같다** | 시트에서 약명만으로 구분되면 규약 위반 | `describe()` 가 β_R·α_R 이 기본값이 아니면 표시한다 |
| [P3] Teacher run 의 `cost_total_hours` 가 null | 가장 큰 비용 항목에 총계가 없었다 | Teacher 가 없으면 총계 = student 비용 |
| Teacher 비용을 학습 중 기록에서 읽어 최종 평가·export 가 빠졌다 | 준비비 과소 계상 | Teacher run 의 `cost_manifest.json` 총계를 우선 사용 |
| GT-only arm 이 KD 준비비를 총계에 달고 amortization 을 부풀린다 | 학습에 Teacher 를 쓰지 않는 arm 이다 | `cost_teacher_role: eval_only` 로 적고 총계에서 분리(시간은 그대로 보인다) |
| 성분별 gate(TRI-B-COMPADV) 가 ε_V,j 를 **평균**으로 뭉갰다 | 계획은 성분별 τ_V,j·ε_V,j | `component_weights` 가 성분별 ε tensor 를 받는다 |

같이 고친 것(반박 검증 전에 직접 확인): SHUFFLE/MASS 대조군의 RNG 를 checkpoint 에 등록(재개 후 같은 열), `stat.window`·`stat.windows` 불일치를 오류로,
`stat.extra` 의 λ 배율·창 검증, CTL-TAU 를 성분별 τ_R,c 에도 적용, TRI-C(stat) 과 residual/변환 표현의 혼용 금지, **양의 q 가 없으면 중단이 아니라 감쇠 없이 진행**(§9.2 의 fallback),
`kdv.expect_arch` 로 골격(폭 104·depth [1,2,2]·no-align)을 trainer 가 강제.

기각된 34건은 대체로 (a) 이 캠페인이 생성하지 않는 config 에서만 성립하거나, (b) 계획을 오독했거나, (c) 이미 다른 경로로 막혀 있었다.

## 7. 판정·기록 규약

- **주 selector 는 `best_rr_val`**(검증셋 `valid_wv3.h5` plain ERGAS) + 고정 final-N(`last`). `best_hqnr`(=best_raw) 는 FR test 로 매 평가 고르는 **test-adaptive exploratory** 라고 manifest·문서에 적는다 — 독립 hold-out 이 아니다.
- 판정 지표 우선순위는 저장소 규약대로 **HQNR → SCC**, ERGAS·SAM 은 참고. 단 이 캠페인의 selector 는 위와 같이 별도로 선언한다.
- **aligned view 는 만들지 않는다.** aligner 가 없으면 같은 ROI 에서 `aligned_valid == raw_valid` 다. 시트에는 `HQNR↑`(전체 프레임)·`HQNR(V64)↑` 두 열만.
- Teacher 오차 bin(0–50 / 50–90 / 상위 10%)은 **고정 Teacher** 로 정의하고 모든 Student(GT-only 포함)를 같은 구간에서 비교한다.
- 비용은 `cost_teacher`·`cost_student`·`cost_total` 로 나눠 적고, 같은 update 비교와 같은 wall-time 비교를 **둘 다** 남긴다.

## 8. 두 서버 배치 (§13.4)

대응 비교는 **한 서버 안에서 끝난다**. s2 는 P1/P2 핵심 축(Q00–Q19)과 필수 대조군·C 계열·장기/초기화 갈래를, s3 는 P3 표현 확장(Q20–Q47·VX)을 맡되 그 비교에 필요한 anchor(Q00·Q04·Q10)와 Teacher(T00)를 같은 서버에서 함께 돈다. 서버마다 자기 T00 을 학습하므로 **cohort 도 서버별로 분리**된다 — 서버 간 수치를 한 표에서 섞지 않는다(시트는 서버 식별자로 구분).
