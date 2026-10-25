# TUNE30 서버별 30시간 집중 튜닝·s3 인계

2026-10-25 사용자 요청: 최신 결과를 바탕으로 s1·s3의 30시간 실험 case를 구성하고 Git으로 전달한다.
이번 작업 범위는 **새 고정 큐 구현·검증·commit**이다. 기존 FOLLOW30 학습은 중단하지 않으며,
새 GPU 캠페인은 아직 시작하지 않는다. Git 반영과 서버에서의 실행은 별도다.

실행 명세는 [TUNE30 README](../panda_tune30/README.md),
[s1 CSV](../panda_tune30/cases_s1.csv), [s3 CSV](../panda_tune30/cases_s3.csv)다.
캠페인 ID는 `PANDA_S13_TUNE_HARDONLY_30H_20261025_v1`이다.

## 새 실험 범위

모든 run은 **scale1 DIFF+UBIL**, fresh Student, single checkpoint 공식 평가다.
입력 `[PANc, PANc−PAN, I_lrms]` 10ch에서 `I_lrms`는 interp23으로 확대한 LRMS 8밴드다.
Student Lanczos4·frozen Teacher bicubic·서버별 기존 frozen donor·GT Hard를 유지한다.
rotation loss·SWA·EMA·soup·TTA·band-weighted loss·Stage1 재학습은 없다.

기본 W112·D123, C30, peak LR1e-4, warmup100→cosine0, α=.25/γ=.5/β=0,
AdamW WD=.01, batch48, FP32. 아래 표의 항목만 단독으로 바꾼다.

| 변경 축 | s1 | s3 | 고정 중심점 대비 변경 |
|---|---|---|---|
| UBIL R3 | R3 | R3 | 세 번째 seed; 기존 BASE R3와 대응 |
| C25 | R1·R2 | R1·R2 | horizon 25K |
| LR080 | R1·R2 | R1·R2 | peak LR8e-5 |
| LR120 | R1·R2 | R1·R2 | peak LR1.2e-4 |
| G025 | R1·R2 | R1·R2 | γ만 .25로 감소 |
| A0125 | R1·R2 | R1·R2 | α만 .125로 감소 |
| C35 | — | R1·R2 | horizon 35K |
| W128 | — | R1·R2 | width만 128로 증가 |

**s1 11 run·320K, s3 15 run·450K, 총 26 run·770K 새 update.**
R1/R2/R3는 261018101/261018102/261018103이며 기존 FOLLOW30 UBIL R1/R2가 중심점이다.
서버 간 donor가 다르므로 같은 설정이어도 같은 donor의 seed 반복으로 묶지 않는다.

이전 분석은 LR/일정 선별 후 Hard 튜닝을 권했지만,
이번에는 두 서버가 독립 수행할 고정 큐를 만들므로 Hard 두 축을 **C30·LR1e-4**에서 분리 검사한다.
중간 성능에 따른 자동 case 교체·하이퍼파라미터 재설정은 하지 않는다.
W128은 새 branch나 loss가 아니라 기존 U-Net의 폭 설정 변경이다. 입력·depth·학습 규칙은 유지한다.

## 30시간 계산과 완료 판정

서버별 30시간(108,000초)은 **FOLLOW30 정상 완료 확인 후 TUNE30 준비 시작부터** 계산한다.
profiling·CUDA preflight·calibration·학습·평가·저장과 종료 게시 여유를 포함한다.
네트워크 장애로 인한 Sheet 재시도는 GPU 학습 종료 후에도 계속될 수 있다.
기존 FOLLOW30의 남은 시간은 포함하지 않는다. 재시작 시 기존 경과 시간을 보존한다.
safety factor 1.08과 종료 여유 1,800초를 둔다.

| 서버 | 예상 학습·평가 wall | 8% + 30분 여유 적용 | 관측·가정 |
|---|---:|---:|---|
| s1 | 26.20h | 28.80h | 기존 UBIL C30 두 run의 실제 시작→완료 시간 |
| s3 | 24.95h | 27.45h | 기존 완료 시각 간격; W128만 1.30배 시간 가정 |

원자료와 계산은 [budget_evidence.json](../panda_tune30/budget_evidence.json)에 있다.
W128의 1.30배는 파라미터 증가를 참고한 계획 가정이지 GPU 속도 측정값이 아니다.
실행 admission 초기값은 s1/s3 .30/.20초 per update와 W128 1.31배로 더 보수적으로 둔다.
기존 W112 관측은 평가·저장을 포함한다. 같은 donor·geometry의 calibration 공유를 고려하며,
새 준비 비용을 run마다 측정된 비용인 것처럼 중복 합산하지 않는다.

남은 예산이 부족하면 마지막 seed 쌍을 시작하지 않고 명시적인 budget skip으로 남길 수 있다.
따라서 **등록 run 수와 실제 완료 run 수는 다를 수 있다.** 예상 속도 저하로 deadline에 중단된 run도
완료로 집계하지 않는다. 성능 기반 early stop은 하지 않으며, budget skip은 Sheet 게시 누락과 구분한다.

## s3 적용 절차

1. 이 변경을 포함한 Git revision을 s3에 반영한다. 로컬 data·checkpoint·credential은 유지하고 Git에 넣지 않는다.
2. `gspread/server.txt`가 `s3`인지 확인한다. 기존 학습/uploader Python·Docker 환경을 사용한다.
3. 아래 읽기 전용 확인과 CPU smoke를 실행한다.

```bash
python -m panda_tune30 catalog --server s3 --csv
python -m panda_tune30 check --server s3
python -m panda_tune30 smoke
```

host `check`는 donor SHA까지 검사한다. 경로를 변경하지 않고 원래 `/workspace`에서 수행해야 하는
donor tensor/config 전체 검사는 launcher의 컨테이너 검사 단계에서 강제된다.

4. FOLLOW30 `campaign_state.json`에서 **등록 12개 전부 정상 완료**를 확인한다.
   대표 checkpoint step을 종료 step으로 오인하지 않는다. launcher도 완료 증거를 확인한다.
5. GPU가 비고 기존 FOLLOW30가 완료됐을 때 실행한다.

```bash
bash scripts/start_panda_tune30_costed.sh s3
```

s1은 같은 절차에 서버 인자를 `s1`로 바꾸고 FOLLOW30 **10개 전부 완료**를 요구한다.
이 launcher는 진행 중인 FOLLOW30를 정지·skip·수정하지 않는다.
미완료나 실패를 만나면 새 실행을 거부하고 기존 실행 상태를 보존한다.
실제 GPU 준비에서는 cost profiling·실데이터 CUDA preflight·source/evaluator pin 검증을 수행한다.
준비 실패를 우회하거나 다른 donor·seed·checkpoint로 대체하지 않는다.

결과 디렉터리:

```text
work_dir/panda_tune30/PANDA_S13_TUNE_HARDONLY_30H_20261025_v1/s3/WV3
```

기존 코드와 predecessor work_dir는 읽기 전용이며 새 TUNE30 출력만 기록한다.
시작 후 pin이 달라졌다면 실행 중 원본을 임의 교체하지 않는다.

## Sheet·후속 분석 인계

기존 게시 규칙을 유지한다.

- 동일 checkpoint **HQNR ≥ .9585 AND ERGAS ≤ 2.040**인 결과 전부와 run별 대표 하나.
- 대표는 HQNR 충족 중 ERGAS 최저, 없으면 HQNR 최고. 대표 step과 최종 update를 별도 표시한다.
- Approach/설명에 scale1·L4·DIFF·UBIL·width·horizon·LR·α·γ를 남긴다.
- fixed64·같은 상태 RR/FR·checkpoint SHA·실측 비용·own-row 검증을 유지한다.
- 실제 측정 전 비용 가정, budget skip, 실패 run을 성공 성능으로 게시하지 않는다.

s3 인계 완료 후에는 대표 행뿐 아니라 15/20/25/30K·최종 평가와 campaign 상태를 함께 전달한다.
각 run의 case·replica·status·실제 update·budget skip 이유를 대조해 전체 완료 여부를 판단한다.
동일 seed·동일 step 및 HQNR 유지 구간을 우선 비교하고, ERGAS .001–.002 차이를 절대적 우열로 해석하지 않는다.
W128의 실측 비용은 parameter 개수와 별도로 보고한다.

이번 commit은 실행 구현과 등록 명세를 전달한다. **s3에서 실제 실행했다는 뜻은 아니다.**
새 학습의 목표 달성, W128 성능과 GPU 비용은 캠페인을 실행한 뒤 확인한다.
