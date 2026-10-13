# EdgeRes48H-v2 구현·배포 검증

대상: `PANDA_Stage1_EdgeGuided_TargetSearch_S123_48H_2026-10-13_v2.md`.
이 문서는 구현 검증이며 48h 학습의 최종 성능 보고서가 아니다.

## 사용자 확정 사항

- workbook은 최신 `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE` 유지.
  s1/WV3-ablations, s2/QB-ablations, s3/GF2-ablations에만 업로드한다.
- 대표 선택 순서는 **HQNR 내림차순 → ERGAS 오름차순**. SCC는 필수 평가/출력한다.
- 저장 자격은 동일 A/T 상태의 full RR20·FR20에서 HQNR/ERGAS 목표 동시 충족이다.
  HQNR만 좋거나 서로 다른 step의 두 best를 합쳐 통과시키지 않는다.

## 구현

새 `edgeres48` 패키지로 기존 캠페인과 분리했다. 기존 scene/global-prior·FR loss 러너를
이름만 바꾸어 실행하지 않는다. 이전 미커밋 method 중 새 경로가 사용하는 공통 모듈과
관련 검증도 배포에 포함한다. 다른 실험 코드나 사용자가 이동한 과거 문서는 변경하지 않는다.

- RR64-only, provided stored LMS, P0/BICUBIC reconstruction, per-input hard EdgeMatch + residual CNN.
- radius4 / fixed R+5 / absolute margin16 / low-texture report-only / deterministic tie-break.
  값의 출처는 v2 운영 제안이며 최적값이나 독립 위상 검증으로 주장하지 않는다.
- 원배포 QB LMS 사용, reconstruction LRMS는 기존 msfix 유지. PAN/GT 행 대응을 검사한다.
- λabs·λε EF00–19, ET00–08, fresh H50/100/150, 서버별 R1/R2/R3 seed.
- effective48 = micro12 × 4, 실제 평균·ε RNG·1 optimizer step 검증.
- disposable 300-update 실측, ETA×1.2, confirmation 예약, closeout 최소 2h.
- 동일 snapshot 평가의 전후 hash, all20 scene 집합, 모든 metric 유한성 확인.
- 미달 weight/이미지 저장 금지. 통과 snapshot atomic save/load 검증과 전체 40장 raw/RGB export.
- LEAN edge/residual/total response, integer/fractional 분리, gradient scope,
  fixed-Teacher CURRENT/PRIOR_ONLY/ZERO numeric-only 대조, 실제 update·추론 비용 기록.
- 코드·문서·source/evaluator hash 및 method source snapshot 보존.
- 로컬 동일 payload → durable outbox → 새 fixed64 시트 → readback receipt.
  원정밀도는 보존하고 표시만 소수 4자리. 미측정 값은 공란이다.
- Checkpoint_SHA는 저장된 실제 파일만. 미달 logical model SHA는 BI JSON에 별도 기록.
- 큰 진단·마감 Pareto 기록은 숫자를 유지하며 분할. seed 통계는 같은 HQNR→ERGAS selector로 통일한다.

## 검증 결과

커밋 예정 index만 별도 디렉터리에 펼쳐, 작업 폴더의 미커밋 의존성 없이 검사했다.

| 검사 | 결과 |
|---|---|
| 새 캠페인 정책·러너·평가·진단·업로드·Docker mount 테스트 | 101 PASS |
| 배포에 포함된 method 공통/edge 회귀 테스트 | 140 PASS |
| 누적배치·optimizer·gradient isolation 테스트 | 7 PASS |
| Python compile / bash syntax | PASS |
| 실제 WV3 GPU, micro12×4, 2 updates | PASS |
| 실제 WV3 RR20 + FR20 전체 공식 metric | PASS, finite·same-state·20/20 확인 |
| 실제 QB GPU, 원배포 LMS, 2 updates 및 native RR/FR 추론 | PASS |
| 실제 GF2 GPU, 2 updates, 전체 LEAN 및 추론 비용 | PASS |
| 기존 Docker 환경, WV3 2 updates + RR20/FR20 + LEAN/cost | PASS |

총 248개 배포 범위 단위/회귀 테스트. 테스트의 의도적 synthetic exception은 실패 처리 검증이다.
실제 GPU는 RTX 4090이며 FP32/AMP off/TF32 off로 검사했다.
WV3 두 번째 update는 약 0.223초, RR20+FR20은 약 19.95초였다.
GF2 전체 고정 LEAN은 약 12.38초였다. 이 짧은 측정은 정식 ETA가 아니며,
실행 시 disposable 300-update와 실제 평가/진단으로 다시 측정한다.

smoke는 가중치·이미지를 저장하지 않았고 production Sheet에 업로드하지 않았다.
각 서버 장기 학습의 안정성·목표 성능·실제 Sheet 권한은 위 unit/smoke만으로 보장하지 않는다.
production 업로드 완료는 실제 receipt의 `READBACK_VERIFIED`로만 판단한다.

## 발견하고 보완한 실행 경계

1. Docker에서 H5의 저장소 밖 절대경로 symlink가 끊겼다. 등록된 실제 H5 파일을 읽기 전용으로 자동 mount한다.
2. 큰 Pareto 집계가 Sheets 45k 안전 제한을 넘을 수 있었다. recursive scalar 분할과 종료 상태 기록 순서를 보완했다.
3. 여러 모델 집계에 가짜 단일 SHA를 넣지 않도록 CAMPAIGN_AGGREGATE를 별도 구분했다.
4. 통과/미달 seed에서 다른 selector를 섞던 통계는 모두 BEST_OBSERVED_HQNR_THEN_ERGAS로 통일했다.
5. preflight 진단 PARTIAL_BUDGET을 PASS로 취급하지 않는다. 정식 실행의 부분 진단은 누락 사유를 기록한다.
6. 통과 weight 저장 후 정성 출력에 실패하면 QUAL_PENDING으로 남기고, 해당 저장 모델만 load해 재시도한다.

## 운영

```bash
bash scripts/run_edgeres48_docker.sh s1
bash scripts/run_edgeres48_docker.sh s2
bash scripts/run_edgeres48_docker.sh s3
```

각 명령은 **그 서버에서** 실행한다. git pull만으로 자동 시작되는 방식은 아니다.
서버 간 Teacher/lock 대기는 없고, 동일 서버의 중복 실행만 거부한다.
실행·결과 경로는 `work_dir/edgeres48/PANDA_S1_EDGERES_TEACHER48H_S123_20261013_v2/<server>/<dataset>`.
소스가 바뀌면 같은 캠페인을 조용히 이어가지 않는다.
조건부 EF16–19·추가 ET는 근거가 있을 때만 활성화하며, 미실행은 이유를 기록한다.

이 탐색은 benchmark metric을 선택에 사용하는 **test-aware 개발 실험**이다.
Scharr matching이나 높은 ε response는 물리적 정합 정확도의 독립 입증이 아니며,
JQM은 기존 SRF-substitute 변형으로 SIPSA 구현과 동등하다고 표시하지 않는다.
