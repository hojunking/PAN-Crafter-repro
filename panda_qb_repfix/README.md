# QB 대표값 게시 복구

V4의 HQNR gate 때문에 사라진 대표 수치를, 원본 평가/weight를 검증해 복구한다.
이 사용자 요청은 목표 미달 여부와 관계없이 대표 모델 수치를 게시하는 것이다.

HQNR 통과 후보 중 ERGAS 최소를 유지하고, 통과 후보가 전혀 없으면 원본 PAN HQNR
최고 후보를 `BEST_HQNR_FALLBACK`으로 게시한다. 같은 checkpoint의 corrected PAN
FR은 H:J 보조 열에 기록한다. 목표 통과 여부는 별도 메타데이터로 명확히 표시한다.

동일 Result_ID를 갱신하므로 빈 대표 행을 새 행으로 중복 추가하지 않는다.
이전 정책의 provenance도 검증하고, 타 캠페인 행을 수정하지 않으며,
예상하지 못한 행 삭제가 필요하면 중단한다. cloud snapshot 및 전체 평가 CSV를 보존한다.

기존 실험이 끝난 물리 s2에서 개별 실행 가능하다.

```bash
python -m panda_qb_repfix --campaign original
python -m panda_qb_repfix --campaign original --apply
python -m panda_qb_repfix --campaign c30 --apply
python -m panda_qb_repfix --campaign repeat --apply
```

학습을 시작하지 않는 복구 명령이다. production 의존성이 있는 환경에서 실행한다.
새 s2 donor-C launcher는 이 세 캠페인을 모두 먼저 복구한 뒤 5개 학습을 시작한다.
원본 INPUT3 파일을 수정하지 않는 별도 패키지이므로 다른 실행의 source pin을 유지한다.
