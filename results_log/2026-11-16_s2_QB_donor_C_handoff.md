# s2 QB 대표 게시 복구 → donor C 5개 실행 인계

상태: **s1 구현·CPU 검증 완료 / s2 실험 시작은 인계 후 확인 필요**.
사용자 확정: 기존 방식대로 s2 에이전트에 인계. Teacher 구조 및 α 증가 제한.

## s2 에이전트가 할 일

이 문서와 신규 `panda_qb_repfix`, `panda_qb_s2_donor_c`,
`config/panda_qb_s2_donor_c` 파일이 전달된 checkout에서 아래 명령을 실행한다.

```bash
python3 -m panda_qb_s2_donor_c.enqueue --check-only
python3 -m panda_qb_s2_donor_c.enqueue
docker logs --tail 80 panda-qb-s2-donor-c-v1
```

`--check-only`는 Sheet 변경/학습 없이 source와 게시 dry-run을 검증한다.
두 번째 명령은 **기존 s2 QB original/C30/R2 대표값 복구 → 검증된 readback →
실제 CUDA preflight → 신규 5개 학습**을 순서대로 수행한다.
컨테이너가 생성됐다는 사실과 실제 학습 시작은 구분해서 보고한다.

이 s1 세션에서는 s2 직접 접속 경로가 없어 s2 학습이나 live Sheet 복구를 실행하지 않았다.
local s1의 corrected-only 학습은 계속 진행 중이며 신규 패키지가 그 source를 바꾸지 않는다.

## 실험 5개

| 순서 | 같은 RGBN Stage1의 donor | Student seed | 검증 목적 |
|---|---:|---:|---|
| 1 | 48K | 261018101 | 중심 후보 |
| 2 | 40K | 261018101 | 같은 seed로 더 이른 donor 비교 |
| 3 | 50K | 261018101 | 같은 seed로 더 늦은 donor 비교 |
| 4 | 48K | 261018102 | 중심 후보 반복 |
| 5 | 48K | 261018103 | 중심 후보 반복 |

고정 donor run: `UNIFIED_QB_S2_S1_RGBN_FULL_R1_S261016301`.
기존 비교군은 donor32K C30 R1/R2. 모두 Student fresh30K이며 새 Stage1 학습은 없다.
48K 반복은 이번 결과를 보기 전에 지정했다.

고정: Teacher/Student W128·D123·UBIL, **α=0**, β=0, γ=.25,
LR1e-4, warmup100/cosine30K, batch48/micro12, RGBN, Lanczos4,
scale1, 공통 Scharr 유효영역, q=0의 rank0 처리.
입력 C는 학습·추론 모두 `[Pc, Pc−P, interp23 MS]`.

이 전략은 donor의 **학습된 가중치 시점**을 비교한다.
새 심층 구조를 도입하거나 Teacher의 폭·깊이를 키우는 실험이 아니다.
과거 Q2의 성공은 후보 우선순위 근거이며, 당시 RGB/W112 Teacher/α.25/γ.5 등과
달라 이번 RGBN48K의 성공을 보장하지 않는다.
기존 donor보다 나은지 먼저 확인하고, 같은 조건의 3 seed로 반복성을 판단한다.

## 대표 게시와 판정

이번 s2 C의 주평가: **일반 PAN HQNR≥.9315 AND RR ERGAS≤3.500**.
corrected PAN은 동일 출력의 보조 평가로 같이 업로드한다.
s1의 corrected-only 입력/학습 → corrected PAN 주평가와 구분한다.

- 평가가 하나라도 있으면 조건 미달이어도 대표 1행을 게시한다.
- HQNR 통과 후보가 있으면 ERGAS 최소, 없으면 **최고 HQNR fallback**.
- 일반 PAN D:F, corrected PAN H:J, 같은 checkpoint·같은 추론 출력.
- 모든 공동 목표 통과 checkpoint는 상세 행도 보존한다.
- 모든 평가점은 각 캠페인의 `publication/qb_representative/all_evaluations.csv`로 저장한다.

donor별 corrected 참조 영상이 다르므로 corrected HQNR 상승만으로 donor 개선을 주장하지 않는다.
같은 원본 PAN 기준의 대표와 30K endpoint, RR 변화, 전체 추세를 함께 비교한다.
학습 효과는 Aligner+Teacher checkpoint·calibration 전체의 효과로 해석한다.

## 인계 후 회신할 증거

1. 세 기존 캠페인의 `publication/qb_representative/last_poll.json`에서
   `readback_verified=true`, 복구 대표의 run/step/원본H/보정H/ERGAS.
2. 신규 root의 `preflight/verified.json`과 donor별 SHA/calibration.
3. `campaign_state.json`의 `runs` 중 `RUNNING` 행과 실제 증가한 step, 컨테이너 이름.
4. 완료 후 48K 3 seed의 공동 목표 통과 수·평균/표준편차와
   donor32K 기존 R1/R2 대비 같은 seed의 변화.

신규 root:
`work_dir/panda_qb_s2_donor_c/PANDA_QB_S2_DONOR_C_20261116_v1/s2/QB/`.
문제가 생기면 실제 오류와 마지막 step을 먼저 보존하고 `--resume`으로 이어간다.
기존 보류 donorgrid/lateval 큐는 재개하지 않는다.

## 전달·검증

코드는 기존 공유 학습 파일을 수정하지 않는 additive release다.
기준 checkout은 `62c184f370ab01bec9419d704c00e9ad1d8ea670`이며,
필요한 parent release 파일 402개가 이 commit과 일치함을 확인했다.
신규 파일 전달 후 `python3 -m panda_qb_s2_donor_c.release`로 전체 parent release까지 검증한다.
source pin을 맞추려고 release JSON을 s2에서 임의 재생성하지 않는다.

Git으로 신규 파일을 전달하지 않을 경우 아래 전송 묶음을 s2 저장소로 복사해서 푼다.
공유 parent source가 다른 경우 먼저 기존 저장소의 올바른 버전을 동기화한다.

```bash
tar --keep-old-files -xzf qb_s2_donor_c_20261116_v1.tar.gz
python3 -m panda_qb_s2_donor_c.release
python3 -m panda_qb_s2_donor_c.enqueue --check-only
python3 -m panda_qb_s2_donor_c.enqueue
```

[전송 묶음](assets/2026-11-16-qb-s2-donor-c/qb_s2_donor_c_20261116_v1.tar.gz),
[SHA256](assets/2026-11-16-qb-s2-donor-c/qb_s2_donor_c_20261116_v1.tar.gz.sha256),
[파일별 manifest](assets/2026-11-16-qb-s2-donor-c/handoff_manifest.json).
묶음에는 신규 패키지·설정·이 문서·CPU 검증 기록만 포함한다.

현재 s1에서 production Docker CPU 테스트 10개 통과:
fallback 정책·기존 V4 행 갱신·동일 출력 dual FR·타 행 보존·재게시 멱등성,
5개 config·구조/α 경계·완료 큐 검사·읽기 전용 마운트,
실제 직렬화 Stage1 donor의 C 입력/Student update/A·T 동결/재개 및 SHA 변조 거부.
시험용 Sheet adapter를 사용했으며 live Sheet 업로드 완료를 뜻하지 않는다.

[시험 로그](assets/2026-11-16-qb-s2-donor-c/tests.log),
[검증 기록](assets/2026-11-16-qb-s2-donor-c/verification.json).

상세 계약: [패키지 README](../panda_qb_s2_donor_c/README.md).
설정: [campaign.json](../config/panda_qb_s2_donor_c/campaign.json),
[cases_s2.csv](../config/panda_qb_s2_donor_c/cases_s2.csv).
