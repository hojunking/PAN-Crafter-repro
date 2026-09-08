# WIP — 아키텍처 고정(W168 d123 dual) 다중 데이터셋 3-seed (2026-09-08 기동)

계획: [research_log/2026-09-08_arch-w168-multiset-3seed-plan.md](../research_log/2026-09-08_arch-w168-multiset-3seed-plan.md).
구조는 `S1_T05_W168_D123_DUAL` 그대로, 데이터셋만 WV3 / QB / GF2 × seed 2025·1234·7777, WV3 checkpoint 의 WV2 zero-shot.
지표 v2(논문 세트 .mat 20장, `results/fr_mat20.json`). 각 서버가 같은 큐를 돈다 — 시트 탭 `QB-<서버>`·`GF2-<서버>`·`WV2-<서버>`·`WV3-<서버>`.

| 서버 | 기동 | 큐 | 상태 |
|---|---|---|---|
| s1 | 2026-09-08 | QB×3 → GF2×3 → WV3 1234·7777 (8 run, ≈4h/run) | 진행 중 |
| s2 / s3 | pull 후 `./tools/arch_multiset_prepare.sh` | 동일 | 대기 |

기동 시점에 없는 것: GF2 논문 세트 4장·WV2 논문 세트·QB/GF2/WV2 RR .mat(Drive 속도제한, 재시도 중). GF2·WV2 의 FR 열은
세트가 만들어진 뒤 `python tools/eval_fr_paperset.py --all` + `gspread_upload.py --all --replace` 로 채운다.

결과는 요청 시 채운다(상시 tracking 안 함).
