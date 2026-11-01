# QB donor RRE 우선 큐 갱신과 s3 인계 — 2026-11-01

사용자가 검토 결과에 따른 큐 갱신과 commit을 승인했다.
실행 명세와 s3 명령은 [panda_qb_rre/README.md](../panda_qb_rre/README.md)에 정리했다.
기존 QB40 과학 코드·등록 파일·완료 기록은 변경하지 않았다.

## 근거

[원자료 재계산](assets/2026-11-01_qb-donor-rre-review/analysis.json)의 완료 donor 8개에서
검증 RRE50과 Student 공식 ERGAS의 Pearson 상관은 .854, 출력 이동을 제거한 진단값에서는 .878이다.
ε 반경 4px 케이스를 제외해도 각각 .879/.916이다. 다만 ε=.03·반경3의 donor는 두 개뿐이고,
s1/s3 평균 ERGAS 차이의 대부분이 Student R2에서 발생한다. RRE 감소를 목표 달성 보장으로
간주하거나 RRE≥.23의 새 중단 조건을 도입하지 않는다.

s3의 B96 네 결과는 모두 완료됐으며 LR1e-4/2e-4의 seed 평균 ERGAS는 3.571403/3.557144로,
기준 R1/R2 평균 3.525927보다 나쁘다. 미시작 B96 반복보다 A4/E.03 검증을 우선한다.

## 새 실행 순서

| 서버 | 경계 case | 이후 자동 실행 | 종료 뒤 |
|---|---|---|---|
| s1 | 현재 C60LR050 R1 60K 완료 | A4/E.03 Stage 1 50K → 기준 Student C30 R1/R2/R3 | 기존 WV3 TUNE30 → HARD36 |
| s3 | 적용 시점의 현재 QB40 case 완료 | 자체 A4/E.03 Stage 1 50K → 기준 Student C30 R1/R2/R3 | 정지 |

미시작 B96을 제외한다. C60 추가 반복·ε=.10 branch의 미시작 작업·W128H050은 이번 자동 큐에서
보류한다. A3/A6와 두 번째 Stage 1 seed는 A4 결과에 따른 후속 후보로 남긴다.
기존 QB40이 전부 완료됐다고 표시하지 않으며 미시작 목록은 전환 receipt에 별도로 남긴다.

Student는 LR1e-4, C30, B48, W112D123, DIFF/UBIL, α=.25/γ=.5/β=0, WD=.01, scale1이다.
Stage 1 seed는 s1=261016101, s3=261016301이고 Student seed는 양 서버 모두
261018101/102/103으로 맞춘다. 서버별 own A2/E.03 기준과 대표 결과 및 고정30K를 비교한다.

## s1 적용

2026-11-01 20:58–20:59 KST에 전환했다. 현재 C60LR050 R1을 **43,824 step**에서 complete-state
저장하고 동일 runtime/config/code pin으로 재개했다. `finish_current`는 이 case만 완료하며,
old QB40 runner가 다음 C60 R2나 B96을 시작하지 않는다. 새 host watcher가 완료와 Sheet
readback을 확인한 후 새 큐를 실행한다.

학습용 파일 463개의 원래 SHA가 모두 유지됨을 확인했다. checkpoint의 optimizer/RNG/data stream과
원래 60K cosine schedule을 그대로 복원하며 새 seed로 초기화하지 않는다.
실행 증거와 확인된 재개 step은 [전환 검증 기록](assets/2026-11-01_qb-rre-priority/verification.json)에 저장한다.

## 검증과 게시

CPU 테스트 15개 통과: 실제 Stage 1 loss/gradient의 QB40 일치, Student loss/gradient 일치,
실제 optimizer update 및 complete-state 재개, 현재 case 경계, 독립적인 서버 실행/마운트,
오류 구분, 실패 donor에 연결된 Student 3개 보류, 게시 namespace와 목표 기준을 검사했다.
s1 native Docker read-only check는 실제 QB train 17,139개 및 validation/RR/FR 접근을 통과했다.
새 큐의 CUDA cost/update/replay preflight는 현재 case가 끝난 뒤 실행한다.

QB-ablations, 동일 checkpoint의 HQNR≥.9315 AND ERGAS≤3.5 통과 전부 및 대표 한 개 규칙을 유지한다.
새 `PQRRE-` 행은 기존 `PQ40-` 행과 구분한다. 목표 통과 여부와 무관하게 모든 1K 평가 weights를
저장하고, QB Student 공식 8/9/10K HQNR이 모두 .90 미만인 기존 중단 규칙을 유지한다.

s3의 원격 실행은 이 세션에서 시작하지 않았다. 사용자의 push 이후 해당 서버 에이전트가
README의 `handoff plan/apply --server s3`를 실행한다. 추가 승인이나 s1 완료 대기를 요구하지 않는다.
