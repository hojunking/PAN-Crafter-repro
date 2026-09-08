# WIP — 아키텍처 고정(W168 d123 dual) 다중 데이터셋 3-seed (2026-09-08 기동)

계획: [research_log/2026-09-08_arch-w168-multiset-3seed-plan.md](../research_log/2026-09-08_arch-w168-multiset-3seed-plan.md).
구조는 `S1_T05_W168_D123_DUAL` 그대로, 데이터셋만 WV3 / QB / GF2 × seed 2025·1234·7777, WV3 checkpoint 의 WV2 zero-shot.
지표 v2(논문 세트 .mat 20장, `results/fr_mat20.json`). 각 서버가 같은 큐를 돈다 — 시트 탭 `QB-<서버>`·`GF2-<서버>`·`WV2-<서버>`·`WV3-<서버>`.

| 서버 | 기동 | 큐 | 상태 |
|---|---|---|---|
| s1 | 2026-09-08 10:41 → 18:5x 재기동 | GF2×3 → QB×3(ms 복구본) → WV3 1234·7777 (8 run, ≈3.2h/run) | 진행 중 |
| s2 / s3 | pull 후 `./tools/arch_multiset_prepare.sh` | 동일 | 대기 |

기동 시점에 없는 것: GF2 논문 세트 4장·WV2 논문 세트·QB/GF2/WV2 RR .mat(Drive 속도제한, 재시도 중). GF2·WV2 의 FR 열은
세트가 만들어진 뒤 `python tools/eval_fr_paperset.py --all` + `gspread_upload.py --all --replace` 로 채운다.

결과는 요청 시 채운다(상시 tracking 안 함).

## 2026-09-08 18:xx — QB 학습셋 결함 발견, QB 재시작 (KNOWN_ISSUES F-3)

QB 만 학습이 불안정했다: RR ERGAS 가 eval 마다 3.9~6.5 를 오가고(변동계수 12~14%, WV3 는 4%), 같은 config·seed 인데
s1 4.79 / s2 6.83 / s3 5.28. 원인은 데이터 — `train_qb.h5`/`valid_qb.h5` 의 `ms` 가 패치의 67~70% 에서 `gt` 대비
LR 1픽셀(HR 4px) 어긋나 있다(데시메이션 위상 (1,2)/(2,1)). WV3·GF2 학습셋과 QB 테스트셋은 정상(MAD 0.00).
`tools/repair_qb_ms.py` 로 Wald 재생성한 `*_msfix.h5` 로 QB 세 seed 를 다시 돌린다. 배포 ms 로 학습한 QB run 은
`*_msbug` 로 치웠고 시트·배치에서 제외된다 — **s2·s3 의 QB 행(ARCH_..._QB_*)도 무효**, pull 후 `arch_multiset_prepare.sh`
가 자동으로 격리·복구한다. 첫 QB run(배포 ms) 참고값: 논문 세트 HQNR 0.902 / RR ERGAS 4.79 (논문 0.920 / 3.570).
