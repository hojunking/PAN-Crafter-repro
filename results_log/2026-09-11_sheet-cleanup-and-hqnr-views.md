# 2026-09-11 — 시트 정리(현 접근 집중) + HQNR 두 가지 측정 방식

## 요지
- **구글시트 WV3 본 탭(WV3-s1 / WV3-s2 / WV3-s3(5090))에는 현 접근만 남겼다**: ① 기준·참조(REF), ⑰ 새 baseline(`BASE_W*_D*_MSPAN_*`), ⑱ PA(A1–A3), ⑲ PO10, ⑳ KDV(s2 예정). 나머지 범주(서브모듈·아키텍처 탐색·attention·KD/mutual·SE·MS-only·GA·SR·UVS·S1 격자·W168 다중 데이터셋·25K 스크리닝·옛 시드 측정·기타)는 각 서버의 **`WV3-<server>_v1` 탭 맨 아래**로 옮겼다(`gspread/archive_to_v1.py`). 옮긴 행의 값은 지표 v2(FR·paper mat20, evaluator 2026-09-10.5) 그대로이고, v1 탭 위쪽의 옛 행(지표 v1, 19열)과는 열이 달라 아래 헤더를 따른다.
- 이동 수: s1 137행(남김 20), s2 50행(남김 9), s3 11행(남김 15). 백업 `gspread/_sheet_backup/*.before_archive_2026-09-11.json`, 반영 뒤 셀 단위 검증 통과. `gspread_upload.py --all` 은 이후 이 범주를 본 탭에 다시 올리지 않는다(`--include-archived` 로만). 범주 정의는 `gspread/sheet_categories.py` 의 `KEEP`/`ARCHIVED`.
- s3 탭의 `BASE_W112_D124 / W96_D123 / W112_D123_MSPAN` 변형은 새 baseline 범주로 재분류해 남겼다(그 전에는 ⑪ 기타로 잡혔다). s1 의 `_po10_gate_N3`(gate 시험 산출물)은 옮겨졌다.

## HQNR 두 가지 — 시트의 HQNR↑ 와 HQNR(V64)↑
둘 다 같은 출력(SR), 같은 원 PAN·LRMS 참조, 같은 DLPan 프로토콜(`py` 재구현, `tools/eval_fr_paperset.py` → `pa/evalviews.raw_views`)이다. 차이는 **어느 영역에서 지표를 모으느냐** 뿐이다.

| | HQNR↑ | HQNR(V64)↑ |
|---|---|---|
| 영역 | 전체 프레임 512² (논문 프로토콜, 자르기 없음) | 고정 내부 영역 V = [64:H−64]² = 384² (가장자리 64 px 제외, 32 px 블록 정렬) |
| D_λ | SR 을 genMTF 로 MTF 필터 → q2n(LRMS, 필터한 SR) 32×32 블록 | 같은 필터를 **전체 프레임에서** 한 뒤 V 로 잘라 q2n |
| D_s | SR 을 imresize↓4·interp23tap 으로 저해상도 PAN 과 짝지어 블록 UQI (S=32) | 같은 저해상도 참조를 전체 프레임에서 만든 뒤 V 로 잘라 UQI |
| HQNR | 장면별 (1−D_λ)(1−D_s) 의 20장 평균 | 같음, V 안에서 |
| 용도 | 논문 표와 비교 가능한 기준값. 시트 비교·best 선택 기준 | 보조. 정합 방법이 PAN sampling 으로 만드는 복제 테두리를 뺀 **동일 영역 비교** |

- V64 는 **shift 나 예측값에 따라 바뀌는 마스킹이 아니다.** 어느 run 이든 같은 V 를 자르므로 aligner 가 없는 run 에도 정의되고 전 행에 채워져 있다. 필터·저해상도 참조를 전체 프레임에서 만든 뒤 자르므로 위상·블록 타일링이 전체 프레임 계산과 같다.
- 정합 방법에서 **"warp 한 PAN 을 참조로 쓰는" aligned_valid** 는 세 번째 view 이며 시트에 올리지 않는다 — 참조 자체가 바뀌어 논문 프로토콜과 비교할 수 없기 때문. run 폴더 `checkpoint_metrics.csv`·`results/pa_diag.json` 에만 있다. PO10 결과(2026-09-10 §10)에서 논문 HQNR 이 내려간 것은 출력이 MS 프레임으로 옮겨간 탓이며, V64 가 아니라 이 aligned_valid 가 그 상황을 보여준다.

## 3. 추기 [WIP] — NF16: N2 정합 능력 보존 + native HRMS fitting (s1, 16 GPU-h, 구현 완료·기동 대기)

명세 `research_log/PAN_N2_NativeFitting_16GPUh_W112_D123_2026-09-11.md` 를 현 workspace(KDV trainer) 위에 구현했다 — 검토표·구현 지도·gate 는 [구현 노트](../research_log/2026-09-11_nf16-implementation.md).
donor = `PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT` 의 정확한 50K `last` aligner(내부 view 4 px; 어제 §10.5 에서 반응 B ≈ −I, closure 0.06 px 확인). U-Net 은 seed 별 저장 초기값에서 새로.

| run | case | aligner | 복원 입력 | loss |
|---|---|---|---|---|
| `NF16_P0_W112_D123_WV3_S1234_N2LAST_v1` | NOALIGN | 없음 | 원 PAN | L_rec |
| `NF16_P1_…_S1234_…` | FROZEN | N2 last 고정 | native 보정 PAN | L_rec |
| `NF16_P2_…_S1234_…` | FT-REC | 학습 (LR 1e-5) | native | L_rec |
| `NF16_P3_…_S1234_…` | FT-EQ | 학습 | native (홀수 update 에 P_ε 는 aligner 에만) | L_rec + 0.01·\|ĉε+ε−sg ĉ0\| |
| `NF16_P4_…_S1234_…` | FT-EQ-GEO | 학습 | native | + λ_geo(t)·L_geo(A3, aligner 에만) |
| `NF16_P3/P4_…_S7777_…` | 반복 pair | | | 예산 gate 통과 시 |

핵심 비교(명세): P1−P0 재사용 / P2−P1 공동 미세조정·망각 / P3−P2 반응 보존 / P4−P3 GT 구조 감독. 인과 비교는 **last(정확한 50K)** 끼리, 시트 best_raw 는 기록용. 네 view(raw_original · raw_valid V64 · aligned_valid(self) · **aligned_fixed_v64(고정 donor 참조)**) 와 last 의 반응·closure·native 참조 stress(`results/po10_diag_last.json`)를 run 마다 남긴다. 기동: s1 `./tools/nf16_prepare.sh`. 예산 ledger `work_dir/_nf16_budget/ledger.json`.
