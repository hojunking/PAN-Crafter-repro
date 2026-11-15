# QB s2/s3 인계: 기존 Stage 1 3개 checkpoint × Stage 2 3개 스케줄

최종 사용자 확정: **Stage 1 새 학습 없음. 서버별 Stage 2 9회, 총18회.**
직전 `77690e68`의 LATEVAL 단일 donor 선택안을 대체한다.
대화 중 검토한 새 Stage 1 1회 학습안은 취소했으며 실행하지 않았다.

각 서버에 이미 있는 RGBN·Lanczos-4·공통 유효영역·W128/UBIL Stage 1의
40K/48K/50K를 모두 고정 donor로 사용한다. 세 독립 seed가 아닌 동일
학습의 세 checkpoint이다. 서버 간 weight 이동이나 원격 자동 학습은 없다.

| 서버 | 기존 Stage 1 source |
|---|---|
| s2 | `UNIFIED_QB_S2_S1_RGBN_FULL_R1_S261016301` |
| s3 | `QBS3_QB_S3_S1_RGBN_FULL_R1_S261016303` |

| 실행 순서(각 서버) | donor step | Stage 2 |
|---|---|---|
| 1, 2, 3 | 40K, 48K, 50K | C50: cosine50K→0 |
| 4, 5, 6 | 40K, 48K, 50K | C15_HOLD: cosine15K→1e-6, 이후50K까지 유지 |
| 7, 8, 9 | 40K, 48K, 50K | C25_HOLD: cosine25K→1e-6, 이후50K까지 유지 |

전부 총50K, warmup100, peak LR1e-4, WD.01, batch48/micro12,
Student seed261018101, α=β=0, γ=.25. 각 donor의 calibration은 별도로
계산·보관하고 세 스케줄 사이에서는 같은 값으로 유지한다.
LR 조기 감쇠의 효과는 검증 전 가설이며, 목표 달성을 보장하지 않는다.

RGBN은 L_spa/q/g/RRE/calibration 전체에 적용한다. Teacher/Student 모두
W128/D123/UBIL, PAN warp 전 경로 Lanczos-4, 공통 support, scale1 유지.
Student 입력은 corrected PAN + interp23 I_lrms, correction residual 없음.
가중치는 `1+0.25*d_e*r_q`, q=0일 때 r_q=0을 유지한다.

목표는 같은 checkpoint의 **corrected HQNR≥.9315 및 ERGAS≤3.5**.
PSNR 하한 및 HQNR 조기 중단은 없다. 고정50K까지 전부 학습한다.
원본/보정 PAN FR과 RR을 매1K 평가하고 모델을 저장한다.
Sheet는 `QB-ablations`, main, 소수4자리, 원본FR D:F/보정FR H:J를
동일 행에 기록한다. donor step과 스케줄을 Experiment/Approach로 구분한다.
각 donor Teacher의 FR 비교 행도 게시한다.

사용자가 push한 뒤 각 서버에서:

```bash
git pull --ff-only
python3 -m panda_qb_donorgrid.release
python3 -m panda_qb_donorgrid.enqueue --server s2
# s3에서는 마지막 인자를 s3로 변경
```

사전 점검만 하려면 `--check-only`, 중단된 새 컨테이너 재개는 `--resume`.
실행기는 해당 서버의 기존 QB MAIN50/PARALLEL 및 LATEVAL 큐를 안전하게
정지·대체하며 기존 결과는 보존한다. 새 Stage 1 학습 fallback은 없다.
필요한 40K/48K/50K 파일이 없거나 방법/해시가 다르면 전환 전에 실패한다.

설정 전체: `config/panda_qb_donorgrid/campaign.json`, 서버별 CSV.
실행 설명: `panda_qb_donorgrid/README.md`.
이 s1 세션에서 s2/s3 원격 실행은 하지 않았으며, 해당 서버에서 위 명령으로 시작한다.

검증: production Docker 이미지의 CPU 통합검증 9개 통과(11.680초).
기존 donor 직렬화/로딩, 기존 메서드 forward·loss 일치, A/T 동결,
학습 재개와 cosine→hold 경계 재개, q=0 예외, donor별 calibration 경로,
9회 큐 및 FR67 동일 행의 4자리 기록, 이전 큐 보존을 확인했다.
환경 pin과 FR67 adapter 설치도 확인했다. CUDA preflight와 원격 실데이터
세 checkpoint 검증은 각 서버 실행 시 수행한다.
Release SHA-256: `0a19051daa9af356823346fba414da0bcdd147ce917bf8fcdfe365464f032273` (332 source/config files).
