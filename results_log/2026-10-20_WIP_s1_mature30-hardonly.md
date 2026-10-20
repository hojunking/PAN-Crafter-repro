# s1 Mature30 Hard-only 전환 — 2026-10-20

상태: **s1 실행 중 — 2026-10-20 19:08 KST 기준 FULL_NOSOFT 400 step 이상 확인**.
사용자 결정으로 다음 Stage2부터 Teacher Advantage 기반
Selective Soft를 제거한다. 기존 case/seed 순서와 정합 정책은 유지한다.
본 문서는 학습 실행 WIP이며, 앞선 기전 분석 및 v1 계획을 소급 수정하지 않는다.

## 실제 인계 상태

- 이전 마지막 case `G8_S1_BIC`는 50,000 update 완료. Band30은 `STOPPED_SAFE`.
- Mature30 v1은 2026-10-20 16:25 KST에 기동했으나, donor 검증 중
  `Original checkpoint/external official evaluation mismatch`로 종료했다.
  학습 case는 하나도 시작되지 않았다. 현 GPU를 사용하는 학습 프로세스도 없었다.
- 원인은 checkpoint 안의 `teacher.depth=(1,2,3)`와 JSON의 `[1,2,3]`을
  Python 객체로 직접 비교한 것. 지표·상태·가중치 차이가 아니다.
  JSON 직렬화 표현을 동일하게 만든 뒤 전체 값을 비교하도록 수정했다.
  실제 원본 35개 checkpoint의 외부/내부 평가 일치를 확인했고,
  원본 Aligner/Teacher 로드도 통과했다. 원본 파일은 수정하지 않았다.

## 새 설정

- 캠페인: `PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2`
- 컨테이너: `panda-mature30-hardonly-wv3-s1-v2`
- Stage2 loss: `mean((1 + .25*d_e + .5*d_e*r_q) * e_S)` (기존 MASK2 reduction).
- `beta=0`: advantage gate 및 Student–Teacher loss/gradient 계산 자체를 생략한다.
  `soft_unscaled`, `soft_weighted`, `a_T_mean`, `soft_active_fraction`은 0.
- Teacher forward는 `e_T`, `d_e`, calibration을 위해 유지한다.
  GT/Hard의 e·q 가중, 마스크, 업샘플러, 정합 배율, 시드, optimizer는 그대로다.
- Stage1은 기존 50K G8 후보의 validation RRE 선정 및 원본 A/T 재사용.
- 실행 순서: FULL R1 → HALF R1 → DOWN R1 → LOW R1 →
  조건부 HALF/DOWN R2 묶음 → 조건부 CAP085 R1.
- 원래 v1의 시작 시각을 승계한다. 마감은 **2026-10-21 22:25:55 KST**,
  마지막 30분 closeout 예약. 실패/전환 때문에 30h를 새로 부여하지 않는다.
- 구 결과/weight는 수정·삭제하지 않는다. 새 run ID에는 `NOSOFT`를 붙인다.

## 기록/Sheet

새 workbook의 `WV3-ablations`를 유지한다. Stage1 RRE 선정,
Stage2 공동 통과 step 또는 HQNR-best 및 FINAL50K 등 기존 게시 조건은 바꾸지 않는다.
`Approach`에 `NO-SOFT`, beta 열에 0, 새 method/campaign ID를 기록한다.
전체 원자료는 로컬 records/outbox/evaluation/train 로그에 계속 남긴다.
이 세션은 s1만 기동하며, 다른 서버를 원격 제어하지 않는다.

## 검증/진행 기록

- Mature30 테스트 **57개 통과**. 공유 Stage2 loss/calibration/mask/
  training-evaluation 테스트 **60개 통과**.
- Hard-only의 loss/gradient가 Hard 단독과 동일하고 Soft graph가 없음을 확인했다.
  기존 beta>0 loss 회귀 테스트도 유지한다.
- 새 캠페인의 beta>0 설정/잘못된 Sheet 표기를 거부한다.
  JSON tuple/list 동등성은 허용하되 metric 변조는 계속 거부한다.
- 실제 운영 Docker/CUDA에서 합성 1-step smoke 통과:
  `beta=0`, Soft loss/gradient=0, `loss == hard`.
  합성 smoke는 실험 결과나 Sheet 행으로 취급하지 않는다.
- 실행 스크립트 `bash -n` 및 `git diff --check` 통과.
- 기존 v1 컨테이너는 종료된 상태 그대로 보존했다. 새 컨테이너
  `panda-mature30-hardonly-wv3-s1-v2`를 기동했고,
  `PM30_WV3_S1_FULL_NOSOFT_R1_S261018101`이 19:06 KST에 시작했다.
- Stage1의 35개 후보를 validation으로 재선정한 결과 50K가 선택됐다:
  **RRE50=0.094862 px, RRE90=0.241326 px, Val-ERGAS=2.127971**.
  이 값들은 selection용 validation 지표이며 공식 RR/FR test 지표와 다르다.
- 실제 학습의 100/400-step 상세 로그에서 모두
  `soft_weighted=soft_unscaled=a_T_mean=soft_active_fraction=0`,
  `gradient.beta_soft_norm=0`, `loss == hard_weighted`를 확인했다.
  Teacher/Aligner gradient=0, Hard e/q 항은 0이 아닌 값으로 유지됐다.
  400-step의 loss와 Hard는 둘 다 `0.031111495103687048`이다.
- 호스트 publisher가 새 workbook `WV3-ablations`의 Stage1 선정 행을 게시하고
  readback 검증했다. 19:07 KST receipt는 `row_count=1`,
  `readback_verified=true`, `foreign_rows_untouched=true`.
  아직 Stage2 첫 공식 평가 이전이며 Stage2 결과 게시를 완료했다고 주장하지 않는다.
- 구 결과/weight와 원래 v1 시작 시각은 보존했다. 최초 전환 당시에는 commit/push 및
  다른 서버의 실행 변경을 하지 않았다.

## 후속 확인·s3 전달

- 2026-10-20 19:11 KST: 첫 Stage2 1K 공식 평가를 완료했고, Sheet 128행에
  `대표(미달) · 1,000`으로 게시됐다. HQNR=0.922986266,
  ERGAS=2.951423230, SCC=0.969573979. Stage1과 합쳐 2행의 readback을 확인했다.
  이는 매 평가 누적이 아니라 현재 HQNR-best fallback 1행이다.
- 이후 사용자의 커밋 요청에 따라 s1/s3 공용 Hard-only 변경과 검증을 커밋 대상으로
  묶는다. s3 원격 실행이나 push는 하지 않고, 실행은 s3 세션에 위임한다.
- [s3 실행·주의사항](../research_log/PANDA_Mature30_HardOnly_S3_Handoff_2026-10-20.md).

## cost 누락 수정 및 기존 행 보완

- 사용자가 cost 누락을 지적해 실제 셀을 재확인했다. 초기 업로더는 성능 지표는
  게시했지만 cost hook의 구현이 없어 P:V 7개 열은 비어 있었다.
  앞선 업로드 검증은 이 누락을 놓쳤다.
- `panda_mature30.cost`를 별도 sidecar 패키지로 구현했다.
  기존 trainer의 225개 source pin과 실험 설정은 변경하지 않았다.
- 정확한 GPU cost를 위해 **31,505 step**에서 SIGTERM 안전 checkpoint를 저장하고
  파일 SHA를 검증한 후, 학습이 GPU를 사용하지 않는 동안 동일 Docker 이미지에서
  Stage1 및 Stage2 5개 geometry를 측정했다. 기존 컨테이너를 다시 시작해
  같은 checkpoint에서 재개했고 31,693 step 이후의 실제 학습 진행을 확인했다.
  optimizer·stream·RNG를 포함한 전체 resume 경로를 사용했다.
- Sheet writer를 새 모듈이 로드되도록 재기동했다. 현재 publisher PID는
  `3092328`이며, 2026-10-20 21:32 KST receipt에서 기존 127/128행
  `OWN_ROWS_VERIFIED`, `readback_verified=true`를 확인했다.
- 현재 보이는 행의 실측 cost:

| 행 / 모델 | Infer ms | Mem MiB | Params M | FLOPs 열: MAC-eq G | Train h | Eval h | Wall h |
|---|---:|---:|---:|---:|---:|---:|---:|
| 127 / RRE-selected Stage1 50K | 15.6888 | 291.4194 | 2.7643 | 63.5038 | 2.3505 | 0.1938 | 2.6063 |
| 128 / FULL Stage2 HQNR-best 4K | 14.8818 | 291.4209 | 2.7643 | 63.5038 | 0.2775 | 0.0171 | 0.3074 |

- cost는 RR256×256 / batch1 / FP32 / 전체 native A+warp+T 또는 A+warp+S,
  새로운 동일 구조 weight로 측정한 architecture-level 값이다.
  실제 선택 checkpoint 자체의 latency 측정이라고 주장하지 않는다.
  FR512×512 실측도 각 profile에 별도로 보존한다.
- Wall은 run 시작부터 선택 평가 완료 시점까지이며 중간 저장·설정·정지 시간을
  포함할 수 있다. Train+Eval을 임의로 더해 Wall로 쓰지 않았다.
- 현재 두 행의 cost 표시만 소수점 4자리로 지정했고 원래 셀 값/로컬 JSON 정밀도는 유지했다.
- 6개 GPU profile 및 시간 근거는 캠페인 `cost_profiles/`에 저장한다.
  긴 35회 평가 근거는 `timing_evidence/`로 분리해 Sheet 셀 길이 한도를 지킨다.
- 기존 게시 규칙 및 성능 지표는 변경하지 않았다. 별도 cost 테스트를 포함해
  Mature30 **65개 테스트 통과**. 이후 신규 실행은 cost를 먼저 측정하는
  `scripts/start_panda_mature30_costed.sh`를 사용한다.
- 마지막 확인에서 33,167 step까지 진행, `resumed_from_step=31505`,
  trainer source pin 변경 0건이었다. 21:37 KST Sheet 재확인에서도 2행 모두
  `OWN_ROWS_VERIFIED`이며 다른 캠페인의 행은 변경하지 않았다.
