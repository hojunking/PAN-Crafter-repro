# TUNE30: scale 1 DIFF+UBIL 집중 튜닝, 서버별 30시간

2026-10-25 사용자 승인에 따른 새 캠페인 `PANDA_S13_TUNE_HARDONLY_30H_20261025_v1`.
**s1 11 run·320K, s3 15 run·450K**를 고정 순서로 등록한다. 이번 작업은 실행 준비와 Git 인계까지이며,
기존 FOLLOW30를 중단하거나 새 학습을 시작하지 않는다. 각 서버의 FOLLOW30가 전부 정상 완료된 뒤 실행한다.

현재 근거는 게시 복구 후 확인한 FOLLOW30 결과다.
s3 DIFF+UBIL R2 대표는 HQNR .958817 / ERGAS 2.041617로 목표에 근접했다.
s1의 두 seed에서도 UBIL은 동일 30K에서 ERGAS를 .0145–.0180 개선했지만 HQNR 비용이 남았다.
따라서 새 입력·loss보다 **현재 구조의 LR·cosine horizon·Hard 두 계수**를 먼저 좁게 탐색한다.

## 공통 설정과 제외 범위

- WV3, **Aligner scale 1.0**: 예측 shift 전량을 PAN에 적용한다. MS·출력·평가 참조는 옮기지 않는다.
- 같은 서버의 Mature30 `BEST_VAL_RRE` frozen Aligner/Teacher를 유지한다. 서버 간 donor가 달라 결과를 단순 seed 반복으로 합치지 않는다.
- Student warp는 Lanczos4, frozen Teacher는 기존 bicubic이다. Stage1 재학습은 없다.
- 모든 입력은 10ch `[PANc, PANc−PAN, I_lrms]`. `I_lrms`는 interp23으로 확대한 LRMS 8밴드다.
  raw PAN은 별도 채널로 추가하지 않지만 두 PAN 채널의 차로 복원 가능하다. 출력은 기존 HRMS 8밴드다.
- 모든 구조는 UBIL: feature upsample을 bilinear×2 + 3×3 conv로 한다. LRMS 확대 연산과는 별개다.
  기본 W112·D123을 유지하며 s3 마지막 대조만 W128로 바꾼다.
- 전 run **fresh Student**. warmup 100 update 후 cosine LR→0, AdamW WD=.01, batch48, FP32.
- 기존 GT Hard만 사용한다. 기본 α=.25, γ=.5, β=0. Aligner/Teacher는 동결한다.
- rotation loss·SWA·EMA·model soup·TTA·band-weighted loss는 없다. 기존 데이터 변환은 유지한다.
- 공식 unshifted/unmasked RR20·FR20을 같은 checkpoint에서 매 1K 평가한다.
- R1/R2/R3 seed는 261018101 / 261018102 / 261018103이다.

명목 scale 1 적용과 실질적인 Aligner 활용은 별도 주장이다. DIFF가 원 PAN 정보를 보유하므로,
유망 최종 후보의 보정 정보 의존성은 기존 입력 개입 진단으로 별도 확인하며 공식 점수와 섞지 않는다.

## 등록 설정과 대응 비교

| case | horizon | peak LR | α / γ | width | s1 | s3 | 질문 |
|---|---:|---:|---|---:|---|---|---|
| S100DIFFUBILC30 | 30K | 1e-4 | .25 / .50 | 112 | R3 | R3 | 기존 BASE R3에 대응하는 UBIL 세 번째 seed |
| S100DIFFUBILC25 | 25K | 1e-4 | .25 / .50 | 112 | R1·R2 | R1·R2 | LR 감소를 앞당겨 두 지표의 좋은 구간을 겹칠 수 있는가 |
| S100DIFFUBILC30LR080 | 30K | 8e-5 | .25 / .50 | 112 | R1·R2 | R1·R2 | 전체 LR 감소가 HQNR 손해를 줄이는가 |
| S100DIFFUBILC30LR120 | 30K | 1.2e-4 | .25 / .50 | 112 | R1·R2 | R1·R2 | 짧은 일정에서 ERGAS 수렴을 앞당기는가 |
| S100DIFFUBILC30G025 | 30K | 1e-4 | .25 / .25 | 112 | R1·R2 | R1·R2 | conditional-q 추가 강조만 줄인 효과 |
| S100DIFFUBILC30A0125 | 30K | 1e-4 | .125 / .50 | 112 | R1·R2 | R1·R2 | Teacher difficulty 기본 강조만 줄인 효과 |
| S100DIFFUBILC35 | 35K | 1e-4 | .25 / .50 | 112 | — | R1·R2 | 조금 더 긴 수렴의 이득·HQNR 비용 |
| S100DIFFUBILC30W128 | 30K | 1e-4 | .25 / .50 | 128 | — | R1·R2 | 기존 U-Net 폭만 늘려 초기 RR 수렴을 당길 수 있는가 |

실행 원본은 [plan.py](plan.py), 정확한 순서는 [s1 CSV](cases_s1.csv)·[s3 CSV](cases_s3.csv)다.
동일 서버·동일 seed의 기존 FOLLOW30 `S100DIFFUBILC30 R1/R2`를 중심점으로 재사용한다.
새 R3는 기존 FOLLOW30 `S100DIFFC30 R3`와 대응한다. 과거 중심점을 새로 학습한 것으로 집계하지 않는다.

**Hard 두 축은 C30·LR1e-4 중심점에서 단독으로 비교한다.** 이전 분석은 LR/일정 선별 후 Hard를 바꾸는
순차 탐색을 제안했지만, 이번에는 서버가 독립 실행할 고정 30시간 큐가 필요하므로 이렇게 등록했다.
중간 성능을 보고 LR·horizon·Hard를 자동 변경하지 않는다. 좋은 두 축이 나와도 결합 효과는 후속 검증 사항이다.

C25는 기존 C30의 25K checkpoint와 다르다. 처음부터 cosine 경로가 달라진다.
공통 step 비교와 각자 종료점 비교를 나눠 해석하며, 종료점 차이를 순수 학습량 효과라고 부르지 않는다.
W128 Student parameter는 3,633,800개로 W112의 2,785,336개보다 30.5% 많다.
이는 CPU 모델 생성으로 센 값이며 GPU 시간·FLOPs·메모리 실측을 대신하지 않는다.

## 30시간 예산과 예상 소요

**서버별 최대 108,000초**, safety factor 1.08, closeout 1,800초를 둔다.
새 launcher가 FOLLOW30 완료를 확인한 뒤 TUNE30 준비를 시작할 때 예산을 시작한다.
cost profiling·CUDA preflight·calibration·학습·checkpoint·평가와 종료 게시 여유를 포함하며,
기존 FOLLOW30 잔여 시간은 새 캠페인의 30시간에 넣지 않는다. 재시작해도 시계를 새로 시작하지 않는다.
네트워크 장애로 인한 Sheet 재시도는 GPU 학습 종료 후에도 계속될 수 있다.

| 서버 | 등록 run / 새 update | 학습·평가 wall 예상 | 8% 변동 + 30분 종료 여유 적용 |
|---|---:|---:|---:|
| s1 | 11 / 320K | **26.20 h** | **28.80 h** |
| s3 | 15 / 450K | **24.95 h** | **27.45 h** |

근거는 [budget_evidence.json](budget_evidence.json)이다. s1 UBIL C30 R1/R2의 실제 시작→완료는
2.4458/2.4673시간이고, s3 게시의 연속 완료 간격은 1.5966/1.6020시간이다.
W112 all-in 평균은 각각 .294789/.191920초 per update다. 평가·저장 시간을 포함한 관측이다.
**W128 시간 1.30배는 계획 가정이며 실측이 아니다.** 실행 시 admission 초기값은 보수적으로
s1/s3 .30/.20초 per update와 W128 1.31배를 사용한다. 새 준비 비용과 실행 속도 변동에 따라 총시간이 달라질 수 있다.
위 여유를 모두 쓰는 학습량을 추가 등록하지 않는다.

실행 시 남은 시간과 실제 관측 속도로 admission을 판단한다. 특히 뒤쪽 W128 seed 쌍은
예상 여유가 부족하면 시작하지 않고 명시적인 budget skip으로 남긴다. 이는 게시 누락이나 학습 성공이 아니다.
중간 성능에 따른 조기 종료는 하지 않는다. 예상보다 느려져 deadline에 도달하면 완료와 예산 중단 상태를 구분한다.

## Sheet 게시와 분석 기준

- 목표는 **동일 checkpoint HQNR ≥ .9585 AND ERGAS ≤ 2.040**이다.
- 공동 통과 checkpoint 전부 + run별 대표 하나를 게시한다. 대표는 HQNR 충족 중 최소 ERGAS,
  충족 checkpoint가 없으면 최고 HQNR이다. 대표 step은 최종 학습 step과 구분한다.
- Approach/설명에서 scale1, L4, DIFF, UBIL, W112/W128, horizon, LR, α·γ를 알 수 있게 한다.
- 기존 fixed64, 같은 상태의 RR/FR, checkpoint SHA, 실제 측정 비용 규칙을 유지한다.
  profiling 전 추정 비용을 실측 Sheet 값으로 쓰지 않는다. budget skip의 가짜 성능 행도 만들지 않는다.
- 비교 시 15/20/25/30K·최종과 HQNR 유지 구간, 가능하면 장면별 변화도 확인한다.
  ERGAS 2.039/2.041 같은 작은 seed 차이만으로 우열을 확정하지 않는다.
- 같은 run의 여러 checkpoint를 독립 seed 반복으로 세지 않는다. 기존 test 지표를 이용한 탐색임을 명시한다.

## 실행 준비와 s3 인계

이 변경을 포함한 Git revision을 받은 뒤 서버의 기존 학습/uploader 환경을 사용한다.
data·checkpoint·credentials는 Git에 포함하지 않는다. `gspread/server.txt`가 실제 서버와 일치해야 한다.

읽기 전용 확인과 CPU 검사:

```bash
python -m panda_tune30 catalog --server s3 --csv
python -m panda_tune30 check --server s3
python -m panda_tune30 smoke
```

host의 `check`는 데이터와 donor 원본 SHA를 검사하며 `/workspace` receipt를 변경하지 않는다.
donor tensor/config 전체 검사는 launcher가 동일 `/workspace` 경로의 컨테이너에서 반드시 수행한다.

실행을 시작할 때:

```bash
bash scripts/start_panda_tune30_costed.sh s3
```

s1은 서버 인자를 `s1`로 바꾼다. launcher는 해당 서버 FOLLOW30 등록 run 전부의 정상 완료 증거를
읽기 전용으로 확인한다(**s1 10개, s3 12개**). 현재 trainer를 정지하거나 predecessor 결과를 수정하지 않는다.
미완료·실패·불명확한 상태면 기존 학습을 유지한 채 새 실행을 거부한다. 기존 GPU 프로세스를 강제 종료하지 않는다.
기존 코드·work_dir를 읽기 전용으로 두고 TUNE30 결과만 새 경로에 기록한다.

결과 위치: `work_dir/panda_tune30/PANDA_S13_TUNE_HARDONLY_30H_20261025_v1/<server>/WV3`.
Git commit만으로 s3 프로세스는 시작하지 않는다. 실제 launch는 인계받은 서버 에이전트가 수행한다.
상세 절차·완료 확인은 [인계 문서](../research_log/PANDA_Tune30_Handoff_2026-10-25.md)를 따른다.

이 README 작성 시 새 GPU 학습·목표 달성·W128 GPU 비용은 검증하지 않았다.
CPU smoke와 실제 데이터 CUDA preflight는 역할이 다르며, 후자는 실행 준비 단계에서 필요하다.
2026-10-25 CPU 검사 44개와 s1 실제 데이터·donor SHA 확인을 통과했다.
기존 실행이 사용하는 350개 파일의 SHA는 모두 유지되었고, 새 구현은 별도 `panda_tune30`에 둔다.
