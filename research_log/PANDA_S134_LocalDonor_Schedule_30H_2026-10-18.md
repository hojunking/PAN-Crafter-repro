# PANDA: s1 / s3 / s4 각각 30시간 — 같은 메서드의 donor·학습 일정 분해

2026-10-18. 구현: `panda30r/`. Campaign: `PANDA_S134_DONOR_SCHEDULE_30H_20261018_v1`.
기존 캠페인은 변경하거나 중지하지 않는다. 이 문서의 커밋/pull만으로 학습이 시작되지는 않는다.

## 1. 이번에 답할 질문

현재 증거는 **“아키텍처의 불가능한 하한이 확정됐다”**가 아니라, **“초기 donor를 고정하고 현재 강도로 학습한 50K Student가 비슷한 trade-off에 모였다”**이다.
기존 Direct Aligner / W112-D123 / P0 / MASK2 / conditional-q를 유지하고, 구성요소의 사용 시점과 강도를 검증한다.
출력 앙상블, weight soup, 새 네트워크, 새 loss, 추론 shift 배율, 후처리 blur는 사용하지 않는다.

근거의 구분:

- s3 보고서 `results_log/2026-10-18_s3_panda40-hqnr-ergas-tradeoff.md`: 12개 50K run의 ERGAS 2.048–2.058,
  최고 HQNR 단일 Student는 G/R2@18K .957381 / ERGAS 2.124065. ERGAS 최소는 Q_FULL/R2@48K 약2.0482.
  HQNR 정점 7–24K와 ERGAS 최저 42–50K가 분리된다. 따라서 최고-HQNR 행만 비교해서는 학습 후반을 설명하지 못한다.
- 같은 보고서의 train gradient 진단에서는 hard와 GT 방향 cosine ≥.998, β=.05 soft의 상대 크기가 약0.6–1.5%였다.
  **현재 q가 무의미하거나 β를 키우면 반드시 좋아진다는 뜻은 아니다.** Teacher가 덜 학습됐고, Student가 Teacher를 추월하는 시점도 함께 고려한다.
- s1에서 직접 확인한 동일 A1/R1 lineage: early7K HQNR .960203446 / ERGAS 2.376434176;
  final50K HQNR .915865347 / ERGAS 2.073583715. 실제 체크포인트 A/T tensor hash와 공식 full40 평가 identity가 일치한다.
  early SHA `0abd6238337f34c2af6682d4d05236f4d42a567ba4faacd24284f5e62a7f7764`,
  late SHA `7a424c334da94a4f1368bc8ea7a5ab8e4f94a9b555b8edc66a6d8e7b220aa1f4`.
  **후기 Teacher를 쓴다는 선택은 더 낮은 ERGAS를 뜻한다.** ERGAS는 낮을수록 좋다.
- s4의 α=.25 후보가 유망했던 관찰은 단일-seed screening이다. 이번 Q 공통 출발점으로 쓰되, 우월성이 확정됐다고 해석하지 않는다.
- s3의 출력 앙상블은 개선 가능성에 대한 진단이지 이번 메서드가 아니다. test-aware 혼합비, blur 결과를 공식 성능으로 올리지 않는다.
  “run 편차와 PAN 상관≈0”는 평균에서 편차 합이0인 성질의 영향도 배제하지 못한다. 이를 인과적으로 확정한 근거로 사용하지 않는다.
- s3 상세 진단 도구/이미지 원본은 이 checkout에 없으므로 보고서 결과를 모두 독립 재계산했다고 주장하지 않는다.

## 2. 공통 고정 조건과 donor 세 가지

세 서버 모두 **WV3**, original Direct PANAligner, Teacher/Student **W112·D123·P0(PAN+MS, LPAN/HPAN 없음)**,
bicubic-MS base, batch48=12×4, FP32/noAMP/noTF32, AdamW wd.01, 기존 원본 데이터·평가 함수 유지.

각 서버에서 **하나의 local A1/R1 학습 lineage**만 사용한다. 서버끼리 가중치를 전달하거나 절대 지표로 donor를 순위 매기지 않는다.
s1/s3는 donor24 A1/R1, s4는 pandas4 D_A1/R1의 완료된50K run을 명시적으로 찾는다.
해당 완료 lineage 자체가 없으면 동일 A1(abs2, epsilon.003, lrA1e-5) 한 벌을 새로 학습한다.
존재하는 파일이 손상됐거나 출처가 다르면 다른 donor로 조용히 대체하지 않고 오류를 남긴다.

| 기호 | Aligner | Teacher | 목적 |
|---|---|---|---|
| E | 초기 선택 checkpoint | 같은 초기 checkpoint | 기존 선택 방식 reference |
| L | 같은 run의 final50K | 같은 final50K | 실제 후기 donor, 정합+성숙도 함께 변함 |
| M | **E의 A 고정** | E의 T만 누적50K까지 연속 학습 | A 변경 없이 Teacher 성숙도 분리 |

E는 HQNR≥.959인 저장 checkpoint 중 ERGAS 최소, 없다면 best-HQNR 탐색용 donor다.
L/M은 명시적인 **final50K 처치**이며, 각 학습에서 저장한 best-HQNR로 다시 바꾸지 않는다.
M은 T의 Adam moments, 원래50K cosine 위치, train sampler/augmentation/global RNG를 이어간다.
M에서 A/Scharr/epsilon을 학습하지 않고 MASK2 reconstruction만 사용한다. 새로운 학습률 재시작이나 T 초기화가 아니다.
M 자체의 HQNR은 내려갈 수 있다. 이 경우도 donor 처치로 유지하고 exploratory를 표시한다.

E/L/M 각각 실제 A/T를 strict load하고 full RR20+FR20를 재평가한다. 기존 평가의 tensor identity는 정확 일치,
scene metric은 `rtol=1e-6, atol=1e-7` 이내 일치해야 한다. replay 차이도 기록하며 imported 결과를 새 학습 행으로 위장하지 않는다.
M과 E의 A는 같지만 T가 달라지므로 **calibration도 E/L/M별로 따로** 만든다(train9714 전체, MASK2, exact conditional-q K5).

## 3. Student 공통 baseline

각 case는 R1=`261018101`, R2=`261018102` 순서로 두 seed를 먼저 완료한다.
동일 replica는 S 초기화·train 순서·증강 RNG를 공유한다. Teacher 복사 없이 Student는 fresh다.
기존 캠페인 seed와는 다르므로 이전 수치와 정확한 재현 비교가 아니라, 이번 로컬 paired control을 기준으로 본다.

- G: MASK2 GT L1, α=β=γ=0, Teacher forward/calibration 없음.
- Q: 기존 conditional-q 수식, **α=.25, β=.05, γ=.5**, warmup100 + cosine50K, lrS1e-4.
- 기준 증강: 기존 코드 그대로 H/V flip 확률 각각1 + rot90(실질적으로 rotation4종).
- flip 대조만 H/V 각각.5 + rot90으로 변경한다. 기존 구현의 의미를 조용히 수정하지 않는다.
- 유효 warp support는 현재 shift로 다시 계산하고 detach한 뒤 sample별 평균한다. 학습 mask를 평가에 적용하지 않는다.

## 4. s1 — Teacher 성숙도 vs 후기 정합 (core8 + optional2)

| 순서 | case, 각2seed | 비교의 의미 |
|---|---|---|
| 1 | E_Q | 초기 donor reference |
| 2 | L_Q | same-run final50K donor: E_Q 대비 전체 후기 처치 |
| 3 | M_Q | E_Q 대비 A 고정 상태에서 T 성숙도의 효과 |
| 4 | M_G | 같은 A의 GT-only; M_Q KD가 실제 추가 이득인지 |
| 5, 예비 | M_B2 | M_Q에서 β만 .05→.2; s4와 독립 로컬 재확인 |

L_Q가 악화해도 “성숙한 T는 나쁘다”고 결론 내리지 않는다. L은 A도 바뀌므로 M_Q가 반드시 필요하다.
M_Q가 E_Q보다 낫지만 M_G와 같다면 개선을 KD 고유 효과라고 주장하지 않는다.

## 5. s3 — Stage2 후반, 기존 Aligner의 작은 공동 조정 (core10 + optional8)

먼저 M_G 2seed → M_Q 2seed를50K까지 학습한다. **M_Q의20K/40K/50K는 선택 점수와 관계없이 보존**한다.
각 fork는 동일 seed, 동일 parent checkpoint, 동일 S Adam moments/stream/RNG에서 출발한다.

| 순서 | case, 각2seed | 구간 | 처치 |
|---|---|---|---|
| 1–2 | M_G → M_Q | fresh0→50K | 공통 controls |
| 3 | T40_G | M_Q40K→50K | A 고정, GT-only, 낮은 LR tail 대조 |
| 4 | T40_EPS | 같은40K→50K | 첫5K만 A.fc2+S; rec+.003epsilon; 다음5K A 재동결 |
| 5 | T40_ABS | 같은40K→50K | EPS와 같고 첫5K에만 .2Scharr 추가 |
| 6–8, 예비 | T20_G → T20_EPS → T20_ABS | 같은20K→50K | 같은 비교를 더 이른 시점에 적용; A는 첫5K만 |
| 9, 예비 | M_B2 | fresh0→50K | 시간이 남으면 s4 β=.2 축의 로컬 재확인 |

모든 tail은 lrS1e-5에서 local horizon에 걸친 cosine. A.fc2만 lrA1e-6→0(5K cosine), A body는 동결.
fc2 동결 시 grad=None으로 두어 AdamW decay로라도 움직이지 않도록 한다.
40K와20K는 tail 길이/LR곡선이 다르므로 각 시작점 안에서 G/EPS/ABS를 비교한다.
M_Q50K와 T40_G의 차이는 **KD 제거+LR 재시작**의 공동 효과다. 이를 Aligner 조정 효과라고 잘못 해석하지 않는다.

중요: **A가 움직이는 tail에서 원래 calibration/q/e/KD를 재사용하지 않는다.** 동일 조건의 T40_G/T20_G에서도 전체 tail α=β=γ=0.
따라서 EPS−G가 제한적 A 조정 처치, ABS−EPS가 약한 절대 정합 제약의 증분이다.
epsilon은 기존 R3 uniform-area disk, native target detach, 입력 ROI8 support 검사 유지.
기존 epsilon은 응답성을 제약하지만 절대 shift를 E에 묶는 anchor는 아니다. abs0/.2 비교 및 shift/지원영역 로그로 이 점을 본다.
추론은 원래 A+warp+S 한 번이며, 추론 배율/평균 모델은 없다.

## 6. s4 — 학습 강도·경로 (core10 + optional2)

| 순서 | case, 각2seed | M_Q 대비 하나의 변경 |
|---|---|---|
| 1 | M_G | GT-only control |
| 2 | M_Q | mature donor Q reference |
| 3 | M_B2 | soft β=.2 (4배) |
| 4 | M_LR5 | peak lrS5e-5 (절반) |
| 5 | M_FLIP | H/V 확률 각각.5 (mirror 포함) |
| 6, 예비 | M_B5 | soft β=.5 (10배), 부정 결과도 가능한 stress test |

β 증가가 T 편향을 보존해 ERGAS를 악화시킬 수 있다. 따라서 처음부터 큰 β만 탐색하지 않고 같은 mature T의 Q/G를 둔다.
α/γ까지 동시에 크게 바꾸지 않는다. 이번에는 “너무 약한 soft”, “학습 경로”, “증강 다양성”을 먼저 분리한다.

## 7. 30시간 예산과 자동 진행

각 서버 독립 **wall30h**, 마지막1800초 closeout, admission safety×1.2. 재시작/정지 중 시간도 포함한다.
초기 파일검증, donor replay, 새 donor fallback, M 학습, train 전체 calibration, 프로파일, pilot, 평가/저장 모두 예산 안이다.
30시간을 모든 case 완료 보장 또는 순수 학습 GPU시간으로 쓰지 않는다.

| 서버 | 과거 s3 속도에 기반한 대략적인 train+평가 슬롯 | 예비 |
|---|---|---|
| s1 | core8 Student 약15.8h + M/준비 약2–4h | β=.2 약4.2h |
| s3 | G/Q4벌 약7.4h + 40K forks 약2–4h + M/준비 | 20K forks 약6–9h, β=.2 약4.2h |
| s4 | core10 Student 약20h + M/준비 약2–4h | β=.5 약4.2h |

새 A1 fallback은 대략2.6h 추가이며, 위 숫자는 **현재 각 GPU 실측이 아닌 prior**다.
각 엔진의 disposable100-update + full40 + checkpoint-load pilot로 시간을 다시 재고, 남은 core seed pairs를 예약한다.
예비는 **pair 전체가 들어갈 때만** 시작하고 시간 부족이면 pair 통째로 스킵한다.
s3의 T20_G/EPS/ABS는 비교 대조군을 잃지 않도록 **6-run block 전체를 먼저 예약**한다.
진행 중 느려지거나 오류로1seed만 남으면 완료된2seed로 표시하지 않는다.
완료 가능한 core가 없으면 안전 종료하며 시간 제한을 연장하거나 설정을 자동 축소하지 않는다.
checkpoints는 모든 공식 평가 상태를 저장하므로 통상 수십 GB의 여유가 필요하다. 실제 checkpoint 크기로 저장 공간도 admission한다.
실행 중2GiB 잔여 시 안전 종료. 부족한 공간을 만들기 위해 기존 결과를 자동 삭제하지 않는다.

## 8. 선택·평가·시트 규칙 — 서로 섞지 않기

| 용도 | 규칙 |
|---|---|
| 연구 checkpoint 선택 | HQNR≥.959 달성 시 ERGAS 최소. 미달이면 best-HQNR. SCC는 선택 우선순위 아님 |
| 엄격 공동 목표 | 현 evaluator 그대로 HQNR≥.959 & ERGAS≤2.040 |
| 중간 도달선 | .959 & 2.045도 별도 표시; 엄격 목표 달성과 혼동하지 않음 |
| 시트 기존 게시 규칙 | HQNR≥.960 & ERGAS≤2.045 모든 qualified step, 없으면 run의 best-HQNR 대표행 |
| donor M/L | 실험 처치상 final50K 고정. best selector와 별도 |

RR20 all-scenes dim_cut21 / FR20 full512 original PAN/native LMS, **unmasked**를 그대로 유지한다.
HQNR·SCC·ERGAS는 평가마다 출력한다. Dλ, Ds, SAM, Q8, PSNR, SSIM, JQM 등 기존 공식 지표도 그대로 기록한다.
all40 동일 paused A/T 또는 A/S state, 평가 도중 optimizer/weights/flags/RNG 변동 검사 유지.
평가 checksum·checkpoint SHA·source manifest·부모 step·local/lifetime update를 묶어 보존한다.

Run 표시에는 donor E/L/M, A 동결/일부조정, α/β/γ, lr, flip, parent 시작점을 짧게 넣는다.
상세 config·calibration·loss/gradient 진단은 로컬 파일에 보존한다.
배포 cost는 **A+T 또는 A+S만** 실제 GPU에서 측정하며 Stage2 학습 전용 Teacher를 포함하지 않는다.
Train(h)/Eval(h)는 해당 run의 **local 실제 측정 시간**이고 부모 학습을 포함하지 않는다. 전체 비용은 campaign wall/prep 로그도 함께 본다.

각 평가 checkpoint는 optimizer/scheduler/sampler/global RNG/epsilon RNG까지 보존한다. resume 때 original config/code/data/evaluator pin이 달라지면 거부한다.
기록 중 중단된 체크포인트/evaluation/outbox는 동일 state와 최초 timestamp를 사용하여 복구한다.
마지막 checkpoint 뒤의 미저장 학습 trace는 감사용 백업에 보존한 뒤 동일 stream에서 재진행한다. 이 낭비 시간도 wall 예산에서 빠진다.
Sheet 인증/네트워크 장애는 outbox에 남겨 재시도하며 결과를 업로드 완료로 위장하지 않는다. pilot/import replay는 공식 행이 아니다.

## 9. 사전에 정한 분석 방법

1. 각 replica의 전체 trajectory, selected, best-HQNR, min-ERGAS, final50K를 **같은 checkpoint의 RR/FR 묶음**으로 보고한다.
2. HQNR gate 도달 여부와 gate 내 최소ERGAS를 먼저 비교한다. H/E 최솟값을 서로 다른 step에서 조합하지 않는다.
3. G/Q, E/M, G-tail/EPS/ABS는 같은 seed의 차이를 우선 보고한다. 두 seed의 방향 일치와 scene별 개선 범위를 함께 기록한다.
4. RR/FR shift dy/dx 분포, 유효 mask 비율, Dλ/Ds를 함께 본다. 작은 Ds 하나만으로 물리적 정합이 개선됐다고 결론 내리지 않는다.
5. mature T의 error/τ_e, Teacher-advantage 활성 비율, soft/hard gradient 비율·cosine을 본다. β를 키웠는데 실제 영향이 없거나 T 편향만 옮기는지 구분한다.
6. 공통 .959 gate를 못 넘은 경우 최저ERGAS만으로 성공이라고 하지 않는다. 두 seed만으로 통계적 유의성이나 전체 서버 우월성을 주장하지 않는다.
7. 같은20-scene 테스트를 여러 차례 선택에 사용하므로 **test-aware 개발 실험**이다. 최종 논문에는 독립 검증/추가 seed와 선택 과정을 분리해 명시해야 한다.

## 10. 실행/동기화

사용자가 commit을 push한 후 각 서버에서 변경을 보존하며 pull한다. 기존 실험/GPU가 사용 중이면 launcher는 자동으로 죽이지 않는다.

```bash
# 해당 서버에서 한 줄만 실행 (기존 데이터/평가환경/업로드 계정 필요)
bash scripts/run_panda30r_docker.sh s1
bash scripts/run_panda30r_docker.sh s3
bash scripts/run_panda30r_docker.sh s4
```

자기 서버의 한 줄만 사용한다. container는 `panda30r-wv3-s1|s3|s4`, 결과는
`work_dir/panda30r/PANDA_S134_DONOR_SCHEDULE_30H_20261018_v1/<server>/WV3`.
한 번에 학습과 host sheet watcher를 시작한다. Docker/data/evaluator/GPU 점검은 같은 명령 안에서 자동 수행한다.
기존 동명 container가 있으면 `docker logs`/상태를 먼저 확인한다. 삭제·초기화·기존 학습 중지는 자동 수행하지 않는다.

```bash
python -m panda30r catalog --server s3 --csv
python -m panda30r smoke
docker logs -f panda30r-wv3-s3
```

native resume는 같은 output과 이미지/코드로 기존 container를 재시작하거나 native runner를 다시 실행한다. 최초30h 시각은 리셋되지 않는다.
코드/MD만 공유하고 checkpoint, 데이터, calibration, outbox, 업로드 credential은 Git에 포함하지 않는다.
