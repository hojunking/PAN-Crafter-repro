# HARD36: α·γ 후속 실험, β=0

2026-10-26 사용자 요청에 따라 기존 TUNE30 큐를 유지하고 **후속 실험 시간을 추가**한다.
캠페인은 `PANDA_S13_ALPHA_GAMMA_HARDONLY_36H_20261026_v1`이다.
기존 TUNE30 코드·큐·checkpoint·30시간 시계는 수정하지 않는다.
새 결과는 `work_dir/panda_hard36/<campaign>/<server>/WV3`에 기록한다.

## 고정 조건

모든 케이스는 scale **1**, DIFF+UBIL, W112·D123, **30K**, peak LR **1e-4**,
warmup100→cosine0, batch48, AdamW WD=.01, FP32, fresh Student다.
서버별 기존 Mature30 validation-RRE 선택 Aligner/Teacher를 동결하여 재사용한다.
Student PAN warp는 Lanczos4, Teacher는 기존 bicubic, LRMS 확대는 interp23이다.
입력은 `[PANc, PANc−PAN, I_lrms]` 10ch이고 출력은 기존 8밴드 HRMS다.

**변경하는 것은 α·γ뿐이다. β=0을 강제**하며 Soft loss, rotation loss, TTA,
SWA·EMA·soup, 밴드 가중 loss, Stage1 재학습, LR·구조 변경을 추가하지 않는다.

## 추가 케이스

아래 각 설정을 **s1·s3 모두 R1/R2**로 실행한다. seed는 261018101/261018102다.
서버당 **6설정·12run·360K**, 두 서버 합계 **24run·720K**를 추가한다.

| 순서 | case | α | γ | 목적 |
|---|---|---:|---:|---|
| 1 | S100DIFFUBILC30AG000 | 0 | 0 | 두 추가 가중 항을 제거한 균일 GT L1 대조 |
| 2 | S100DIFFUBILC30G000 | .25 | 0 | 조건부 공간 가중 항 제거, Teacher 난도 강조만 유지 |
| 3 | S100DIFFUBILC30A000 | 0 | .50 | α 항 제거; γ·d_e·r_q는 유지 |
| 4 | S100DIFFUBILC30H050 | .125 | .25 | 두 계수 동시 축소의 결합 효과 |
| 5 | S100DIFFUBILC30A050 | .50 | .50 | α 단독 증가 |
| 6 | S100DIFFUBILC30G100 | .25 | 1.00 | γ 단독 증가 |

실행 순서 원본은 [s1 CSV](cases_s1.csv), [s3 CSV](cases_s3.csv)다.
기존 FOLLOW30의 `.25/.50` 중심점 R1/R2와 현재 TUNE30의 `.125/.50`, `.25/.25`
단독 축소 결과를 대조군으로 사용하며 동일 설정을 중복 실행하지 않는다.
H050은 이전 DIFF BASE C50의 H050과 달리 **UBIL C30**이므로 새 검증이다.

| α \ γ | 0 | .25 | .50 | 1.00 |
|---:|---|---|---|---|
| 0 | 새 AG000 | — | 새 A000 | — |
| .125 | — | 새 H050 | 기존 TUNE30 A0125 | — |
| .25 | 새 G000 | 기존 TUNE30 G025 | 기존 FOLLOW30 중심점 | 새 G100 |
| .50 | — | — | 새 A050 | — |

비교는 같은 서버·donor·seed에서 한다. 서버 간 donor가 다르므로 동일 조건 반복으로 합치지 않는다.
AG000은 두 항을 모두 제거하므로 단일 계수 변화라고 부르지 않는다.
기본/AG000/G000/A000 네 점은 각 항의 기여와 상호작용을 확인한다.
기본/A0125/G025/H050 네 점은 두 항 축소의 상호작용을 확인한다.
현재까지 이 새 설정들의 성능 이득은 검증되지 않았다.

## 0 계수의 정확한 의미

손실은 `mean_valid[(1+α*d_e+γ*d_e*r_q_bar)*mean_bands(abs(S−GT))]`다.
AG000은 **같은 유효 영역에서 균일 L1**이며 Aligner scale을 0으로 만드는 실험이 아니다.
A000에서도 γ 항에 d_e가 남으므로 Teacher guidance를 전부 제거한 것은 아니다.

γ=0은 기존 loss의 `panda_e` 경로로 계산하고, α=γ=0에서도 같은 경로에서 상수 가중치 1을 쓴다.
Teacher·calibration·입력·support 계산을 보존하여 픽셀 가중치만 비교한다.
이 비교는 Teacher 구성 자체를 삭제하는 코드상의 `gt_only` 경로와 구분한다.
γ=0에서 조건부 q lookup을 생략하므로 학습 시간 추정은 γ>0과 별도로 모은다.
추론은 모든 설정이 동일 A+S 구조이므로 배포 비용 측정은 공유한다.

## 추가 시간과 자동 연결

기존 UBIL W112의 학습·평가·저장 포함 관측 속도 s1 .294788864초/update,
s3 .191920082초/update를 사용하면 추가 예상은 **s1 약29.48h, s3 약19.19h**다.
γ=0 경로의 실제 속도는 아직 측정하지 않았으므로 동일 속도를 보수적으로 적용했다.
준비 비용과 변동 여유를 포함해 새 캠페인의 상한을 **서버별36h**로 둔다.
안전 계수1.08, 종료 여유30분, seed쌍 단위 admission과 checkpoint 재개 규칙을 유지한다.
기존 TUNE30 종료 대기 시간은 새36h에 포함하지 않고, 첫 비용 측정부터 시계를 시작한다.

대기기는 TUNE30가 정상 종료하거나 증거가 있는 budget skip으로 종료된 뒤에만 시작한다.
완료 run의 최종 checkpoint SHA를 검증하고, 기존 컨테이너의 정상 종료와 GPU 유휴를 확인한다.
실패·중간 중단·수동 skip은 자동으로 건너뛰지 않고 `BLOCKED`로 기록한다.
TUNE30의 budget skip은 새 실험 완료로 집계하지 않으며, 빠진 대조군이 있으면 그 비교는 미확인이다.
대기기는 기존 trainer/uploader를 정지하지 않는다. 학습이 시작되면 새 게시기가 별도로 작동한다.

## s3 인계 및 실행

이 변경이 포함된 Git revision을 반영하고 기존 s3 데이터·donor·Docker·인증 환경을 사용한다.
`gspread/server.txt`가 `s3`여야 한다. CPU 검사와 registry 확인:

```bash
python -m panda_hard36 catalog --server s3 --csv
python -m panda_hard36 smoke
```

TUNE30 뒤에 자동 연결하도록 등록한다. 기존 학습/uploader Python을 사용한다.

```bash
python -m panda_hard36.queue --server s3 --enqueue
```

s1은 서버 인자를 s1으로 바꾼다. 이미 대기 중이면 중복 등록하지 않는다.
상태는 `work_dir/panda_hard36_launchers/<server>/status.json`, 로그 위치는 `enqueue.json`에 있다.
TUNE30가 끝난 뒤 직접 시작할 때는 `bash scripts/start_panda_hard36_costed.sh s3`를 사용한다.
대기기가 등록된 상태에서 직접 중복 실행하지 않는다.

## Sheet와 검증

기존 규칙: 동일 checkpoint **HQNR≥.9585 AND ERGAS≤2.040** 통과 전부 + run별 대표 하나.
대표는 HQNR 충족 중 ERGAS 최소, 없으면 HQNR 최대다. β=0, α·γ, scale1·L4·DIFF·UBIL,
W112, C30, LR1e-4를 행과 설명에 남긴다. 기존 행을 덮어쓰지 않는다.
같은 step, 후반 구간, seed별 개선 방향을 함께 비교하고 작은 ERGAS 차이를 확정적 우열로 읽지 않는다.

CPU 검사는 실제 loss·Student update·동결 A/T·정확한 checkpoint 재개,
AG000의 L1 값/gradient 일치, Soft 활성화 거부, 게시 기준과 이전 run 보존을 확인한다.
실데이터 CUDA 사전 검사는 TUNE30가 종료되고 새 캠페인이 시작할 때 수행한다.
