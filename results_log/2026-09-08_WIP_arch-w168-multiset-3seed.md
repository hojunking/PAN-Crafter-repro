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

## 2026-09-08 20:xx — 센서별 평가가 논문과 같은지 확인 (KNOWN_ISSUES F-2 표)

네 센서 `.mat` 테스트셋을 전부 받아 H5 와 대조: RR 은 네 센서 모두 동일, FR 은 WV3·QB 만 다르고(논문 세트로 교체) GF2·WV2 는
동일. 어느 센서도 FR 부분집합을 쓰지 않는다(20장 전체). CANConv 배포 가중치를 우리 평가기로 재서 두 논문의 CANConv 행과
대조(`work_dir/_ref_cannet_{qb,gf2,wv2}`, 시트 각 탭의 □ 행):

| | HQNR (우리 / 논문) | D_λ | D_s | PSNR | SSIM | SCC | Q2n | SAM | ERGAS |
|---|---|---|---|---|---|---|---|---|---|
| QB | 0.8941 / 0.893 | 0.0393 / 0.039 | 0.0691 / 0.070 | +0.41% | +0.04% | −0.01% | +0.01% | −1.10% | −0.54% |
| GF2 | 0.9189 / 0.919 | 0.0194 / 0.019 | 0.0629 / 0.063 | +0.39% | +0.01% | +0.06% | −0.06% | −1.27% | −2.57% |
| WV2 (wv3 가중치 zero-shot) | 0.8770 / 0.876 | 0.0690 / 0.068 | 0.0576 / 0.060 | +0.38% | +0.19% | −0.04% | −0.26% | +0.01% | +0.12% |

FR 세 지표와 RR 의 PSNR·SSIM·SCC·Q2n 이 논문 자릿수에서 맞는다. SAM·ERGAS 의 −1~−2.6% 는 WV3 때와 같이 "배포 가중치 ≠
논문 표의 재학습 출력" 차이다(평가기가 아니라 모델 출력). 논문 Table 1/2/3 의 데이터셋별 결과와 같은 데이터·같은 프로토콜로
비교할 수 있다.

## 2026-09-08 21:xx — JQM 추가 (Palubinskas 2015, `tools/metrics/jqm.py`)

시트 FR·paper 에 JQM↑ 열을 넣었다(논문 세트 mat 에서 계산, 재추론 없음). 정의는 논문 Eq.4/6/8/9/11 그대로이고 논문이 정하지 않은
선택 — 전역 CMSC 통계 · lpf = genMTF(센서) + (2,2) 데시메이션 · 분광 가중 w = NNLS(MTF_PAN↓PAN ~ MS) · R = 2^L−1 — 은 모듈 docstring
에 적었다. 두 논문은 JQM 을 보고하지 않으므로 대조값은 없고, **판정 기준은 그대로 HQNR → SCC** 다.

실측 성질: d1·d2 항은 ~1e-4 이하라 CMSC ≈ ρ⁺ 이고, QHR 은 사실상 PAN–intensity 상관이다. 그래서 PAN 구조를 강하게 주입한 출력이
높게 나온다 — WV3 139 run 에서 **Spearman(HQNR, JQM) = −0.56**, LR-Fuse(HQNR 0.907, 최하위)가 JQM 0.988 로 최상위. QLR 만 보면
LR-Fuse 가 최하(0.9878)로 상식과 맞는다. 즉 JQM 은 HQNR 과 다른 것을 재며(공간 충실도 편향), 단독 순위 근거로 쓰면 안 된다.
