# QB Stage1 탐색 축소: s2·s3 실행 인계

현재 방법 유지: **RGBN · Lanczos-4 · 공통 Scharr 유효영역 · Teacher/Student
W128/D123/UBIL**. 과거 RGB weight를 가져오지 않고, 각 서버가 이미 학습한
현재 RGBN Stage1의 40K·48K·50K checkpoint만 사용한다.

| 서버 | Stage1 추가 학습 | 선택 | Stage2 |
|---|---:|---|---|
| s2 | 0 | 자체 UNIFIED RGBN R1,40/48/50K 중 validation RRE50→RRE90→Val_ERGAS→이른 step | LR1e-4,γ=.25,α=β=0,50K,seed261018101 |
| s3 | 0 | 자체 PARALLEL RGBN R1,동일 규칙 | 동일 설정1회 |

Stage1 계수는 기존 λ_spa=4,λ_ε=.03이다. Aligner scale1, paired Teacher 고정,
Student 입력 corrected PAN + interp23 I_lrms, zero-q 예외처리 유지.
신규 Stage1 sweep,10K/20K donor,λ_spa8,λ_ε.01/.10,추가 LR/γ 탐색은 등록하지 않았다.
향후 A2/E.03은 이번 결과를 검토한 뒤에만 별도로 결정한다.

근거: 과거 RGB A2/A4의 검증 선택 step은39~48K였고, 구방법에서 목표를 통과한
s2 Q2_W128 R1은 A4/E.03/D301 @48K를 사용했다. 구결과는 original-PAN HQNR이며
현재 corrected-PAN HQNR과 직접 비교하지 않는다. 이번40/48/50K 제한은 새로
합의한 선택 범위이며 과거의 전구간 validation 탐색을 동일 재현한 것은 아니다.

완료된 기존 Q1/P1과 donor·학습 설정·calibration 수치가 같으면 전체50K 결과를
검증해 재사용하며 새 반복으로 세지 않는다. 그 외에는 fresh Student를 실행한다.
실제 기존 모델 파일은 commit에 포함하지 않으며, 다른 서버에서 가져오지 않는다.

실행 방법과 검증·게시 규칙은
[패키지 README](../panda_qb_lateval/README.md)에 있다.
사용자 push 후 각 서버에서 전체 commit pull, `--check-only` 후 enqueue 실행.
런처가 해당 서버의 기존 QB 큐만 정상 종료하고 남은 케이스를 취소해 새 큐로 전환한다.
이전 결과와 weights는 보존한다. s1 WV3 실험에는 영향이 없다.

목표: 같은 checkpoint에서 **corrected HQNR≥0.9315, ERGAS≤3.5**.
각 Student50K 완주,1K 평가. 기존 QB-ablations에4자리,원본/보정 PAN FR 컬럼,
대표·통과·RR 극값·50K 결과 게시. 새로 선택한 Stage1 Teacher도 별도 게시한다.
기존 캠페인의 업로드 기준은 수정하지 않는다.

## 구현 검증

- production Docker `hojunqueen/pancrafter-env:torch2.4.0-cu118`에서 CPU 회귀 테스트
  **10개 통과**. 양 서버의 원본 Stage1 로더→실제 checkpoint→calibration→Student
  경로를 사용했다. 기존 Stage2와 출력/loss 일치, A/T 고정, 정확한 optimizer/RNG/stream
  재개, RGBN validation 집계, zero-q, 55×55 공통 마스크, FR67 컬럼을 검증했다.
- 다른 서버·변조된 checkpoint·후보 누락 거부, 기존 완료/실패 결과 보존 및 남은 큐
  취소 로직도 검증했다. 새 native CUDA preflight는 각 서버 실행 시 수행한다.
- 일반 host Python에서도 catalog와 launcher help 실행 확인. 학습용 라이브러리는
  Docker 환경에서 사용한다.
- source release: `979b9c110795a6c8905a3688ecadddb3fa4fa26e842f444fcb27ae5831e92f7f`
  (311 files). 기존 MAIN50/PARALLEL 및 s1 실행 소스는 수정하지 않았다.
- s2/s3의 실제 파일·GPU·Sheet 접속과 원격 학습 시작은 이 세션에서 확인하지 않았다.
  각 서버 에이전트가 pull 후 위 check/enqueue 명령을 실행한다. push는 사용자가 한다.
