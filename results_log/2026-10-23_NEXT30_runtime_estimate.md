# NEXT30 전체 큐 예상 소요 시간

관측: 2026-10-23 17:50 KST 전후. 구현 commit `fa1e85b`의 등록 큐를 기준으로 계산했다. Stage1은 같은 서버의 기존 donor를 재사용한다.

**s1 약 43–50시간, s3 약 25–30시간. 두 서버가 지금부터 병행하면 전체 약 43–50시간(약 2일)을 예상한다.** 두 서버 실행시간 합계는 약 68–80시간이다. s3 인계 지연과 현재 bicubic run 마무리는 새 큐 시간에 별도로 더한다. 장애·재시도·추가 실험은 포함하지 않았다.

## 등록량과 계산

| 서버 | case 수(R1/R2 별도) | 총 update | RR20+FR20 평가 횟수 | 학습 시간 | 평가·저장 | 기준 합계 | 계획 범위 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 18 | 585,000 | 553 | 39.87 h | 3.19 h | 43.06 h | **43–50 h** |
| s3 | 17 | 535,000 | 511 | 23.18 h | 1.38 h | 24.57 h | **25–30 h** |

`시간 = update 수 × 초/update + 평가 횟수 × (평가 초 + 저장 초)`로 계산했다. horizon 30K/35K/50K의 평가 횟수는 각각 29/34/42회다. 실행 catalog를 읽어 case 수·update·평가 합계를 검산했다.

- **s1:** NEXT30 S020C30 R1의 최근 1,000-update p90 = 0.245365초/update, 평가 20.2839초, 저장 0.4567초. 4K checkpoint까지 완료한 것을 차감하면 기준 잔여 시간은 **42.77시간**이다. 현재 trace의 평균 속도도 약 0.244초/update다.
- **s3:** Sheet row183의 기존 SCHEDULE30 S020C30 R1은 local 16K까지 Train 0.674932시간, Eval 0.038529시간이다. 학습 **0.151860초/update**, 평가 **9.24699초/회**에 해당한다. row181의 25K fork도 새로 학습한 local 4K를 분모로 하면 0.152162초/update로 일치한다. 선택된 29K 전체를 분모로 쓰지 않았다.
- s3 NEXT30은 아직 직접 측정하지 않았다. 새 Lanczos4의 소폭 비용을 감안한 **0.156초/update**와 저장 **0.5초/회**를 계산 가정으로 사용했다. 저장 수치는 s3 실측이 아니다.
- 계획 범위에는 일반적인 준비·I/O·게시와 미측정 구조별 학습 속도 차이의 여유를 뒀다(s1 기준 약 15%, s3 약 20%). 통계적 신뢰구간이나 완료 보장은 아니다. down/up 변경의 추론 latency를 학습 시간 비율로 대입하지 않았다.

| 단일 run 길이 | s1 기준 | s3 기준 |
|---|---:|---:|
| 30K | 약 2.21 h | 약 1.38 h |
| 35K | 약 2.58 h | 약 1.61 h |
| 50K | 약 3.65 h | 약 2.28 h |

## 실행 상태와 인계

s1의 기존 두 fork는 완료됐다. 새 CUDA preflight는 2026-10-23 17:28:09 KST 통과했고 NEXT30 trainer는 17:28:15 KST에 실행됐다. 관측 시 첫 S020C30 R1은 RUNNING, 상태 파일의 저장 step은 4,000이다. 위 18개 전체가 `migration_effective_queue`에 등록돼 있다.

s3는 17:35:27 KST 게시 기록에서 기존 bicubic S020C30 R1 대표16K가 확인됐다. 대표 step은 live 진행 step이 아니다. 당시 이미 16K까지 진행됐으므로 30K 종료까지 남은 학습·평가 기준 시간은 최대 약 0.7시간이지만, 중단이나 대기 여부는 현지에서 확인해야 한다. 새 큐 시작은 [s3 인계서](2026-10-23_NEXT30_s3_handoff.md)를 따른다.

s3의 17개는 등록 상한이다. 인계 전에 기존 schedule의 대응 case가 더 시작됐다면 전환 규칙에 따라 해당 새 case를 생략할 수 있다. S020C30 R1/R2는 Lanczos4 대조군이라 항상 실행한다. 실제 목록은 s3의 전환 receipt와 `migration_effective_queue`로 확정한다. 30K case 하나를 생략하면 새 큐 기준 약 1.38시간이 줄어든다.

병행 완료 시각은 `max(s1 잔여 시간, s3 시작 지연 + 현재 run 잔여 시간 + s3 새 큐 시간)`으로 본다. s3가 곧 인계받고 장애 없이 진행하면 10월 25일 낮~저녁(KST)이 전체 완료 예상 구간이다.

## 근거

- [재계산 값과 s1 상태 snapshot](assets/2026-10-23_next30_eta_calculation.json)
- [s3 Sheet 원값 snapshot](assets/2026-10-23_next30_eta_sheet_snapshot.json): 17:43:41 KST 읽기, `UNFORMATTED_VALUE`, 정확한 범위 포함.
- [WV3-ablations 원본](https://docs.google.com/spreadsheets/d/1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE/edit#gid=261001010): rows181/183, Train(h)/Eval(h)/Updates/Selected_step.
- [전체 case와 순서](2026-10-23_NEXT30_current_and_queue.md), [s1 CSV](../panda_next30/cases_s1.csv), [s3 CSV](../panda_next30/cases_s3.csv).
