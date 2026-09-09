# 아키텍처 고정 다중 데이터셋 3-seed 계획 (2026-09-08)

**지시**: `S1_T05_W168_D123_DUAL` 구조(W168 · depth [1,2,3] · dual MARs · 11ch · crop=False · attention 없음 · 50K)만 놓고,
PAN-Crafter / U-Know-DiffPAN 이 쓴 PanCollection 데이터셋 전부에서 3 seed 결과를 얻는다. 각 서버(s1/s2/s3)가 같은 것을 돌린다.

## 대상

| 데이터셋 | 밴드 | max_pixel | 학습 | seed | 비고 |
|---|---:|---:|---|---|---|
| WV3 | 8 | 2047 | O | 2025 · 1234 · 7777 (새 프로토콜로 전부 새로 학습; S1_T05_W168_D123_DUAL 은 12-19 선택이라 별도) | FR = 논문 세트 |
| QB | 4 | 2047 | O | 2025 · 1234 · 7777 | FR lpan 손상(F-1) → `repair_lpan.py --sensor qb` |
| GF2 | 4 | 1023 | O | 2025 · 1234 · 7777 | 배포 lpan 정상 |
| WV2 | 8 | 2047 | X (zero-shot) | WV3 3 seed 의 checkpoint | 논문 Table 3 절차. lpan 은 레시피 생성(검증 불가) |

config: `config/ARCH_W168_D123_DUAL_<DS>_S<seed>.yaml` 8벌 (`tools/gen_arch_multiset_configs.py` 가 템플릿에서 생성 — 손으로 고치지 않는다).
큐: `config/queues/arch_w168_multiset_3seed.txt` (GF2×3 → QB×3(ms 복구본) → WV3×3, seed 2025 포함). params 5.9731M(8밴드) / 5.9610M(4밴드), smoke 통과(peak 8.6GB).

## 평가 (지표 v2)

- best 선택: **논문 세트(.mat 20장 전체) HQNR** (2026-09-09 결정, 12-19 부분집합 폐기; FR feeder = `full_examples_mat20`, `fr_select_indices: 0-19`). 보고도 같은 세트(`results/fr_mat20.json`) — 시트 FR 열.
- 논문 세트 대조(2026-09-08): **QB FR 도 H5 와 다르다**(겹침 4/20), GF2 는 받은 16장이 전부 H5 와 동일(나머지 4장·WV2·RR 세트는 Drive 속도제한으로 재시도 중).
- 센서별 MTF: WV3/WV2/QB 는 genMTF 표, GF2 는 otherwise 0.3 (MATLAB 과 동일).
- WV2 zero-shot: `tools/make_zeroshot_run.py --run <WV3 run> --sensor wv2` → `work_dir/<run>_zs_wv2/` (RR mat + fr_mat20.json). `tools/_upload.sh` 가 WV3 run 완료 시 자동으로 만든다.

## 서버별 절차

```bash
git pull
./tools/metric_v2_prepare.sh          # 아직이면 (지표 v2 재측정)
./tools/arch_multiset_prepare.sh      # QB lpan 복구 · WV2 데이터 · 논문 세트 h5 · 검사 · smoke · 체인 기동(48h)
```

Drive 다운로드가 막히면(속도제한) `.mat` 를 브라우저로 받아 `data/PanCollection/<DS>/full_examples_mat/` 에 두고 다시 돌린다.
WV3 seed 2025 가 없는 서버는 큐 맨 앞에 `S1_T05_W168_D123_DUAL` 을 추가한다.

## 판정

HQNR(논문 세트) 3 seed 평균 ± 표준편차(N−1). seed 간 폭이 방법 간 차이의 척도. 20장 표본 표준오차 ≈ 0.002 이므로 그 안쪽은 구분하지 않는다.
결과 문서: `results_log/2026-09-08_WIP_arch-w168-multiset-3seed.md` → 완료 시 확정 문서로 대체.
