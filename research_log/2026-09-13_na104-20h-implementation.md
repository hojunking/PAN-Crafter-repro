# 2026-09-13 — NA104 20H 우선순위 구현 노트 (s2·s3 전달본)

계획: [`01_S2_20H_PRIORITY.md`](01_S2_20H_PRIORITY.md) · [`02_S3_20H_PRIORITY.md`](02_S3_20H_PRIORITY.md) (2026-09-13). 두 문서가 참조하는 상위 규약 `00_20H_CRITERIA_AND_RETIREMENT.md` 는 저장소에 **없다** — 두 문서의 §6 기준(같은 서버 N0 대비 3 seed 평균 Δ>0, 최소 2/3 양수, plateau/last 는 설명, ERGAS 보조) 으로 구현했다.
이 노트는 s1 에서 작성·검증한 코드 전달본이다. s2·s3 의 큐·프로세스는 바꾸지 않았다 — 각 서버에서 운영자가 `./tools/na104_20h_switch.sh` 를 실행할 때 바뀐다.

## 1. 계획 → 코드

| 계획 | 구현 |
|---|---|
| §1 후보 3 (Q36·Q12·CF01) + baseline Q00·대조 X02 | 큐 `config/queues/na104_20h_s2.txt`·`na104_20h_s3.txt` = P1 4벌(순서 고정). 나머지 NA104 큐(87/97) 는 이번 개발 목록에서 제외(폐기 목록 §7; 완료분 보존) |
| §2 P0 (현재 run 마무리·완료분 차감·common-grid·hash 고정·재선택) | `tools/na104_20h_switch.sh`: 감시자 cron 일시 해제 → 체인 runner 만 종료(학습 프로세스는 원 설정으로 끝까지) → 완료 대기 → export/업로드 → 20h 시계 시작(`work_dir/_na104_20h/ledger.json`) → smoke(P1 4 + CF01) → gate token → 큐 기동 → 감시자 재등록. `python tools/na104_20h.py report` 가 §회신 항목(현재 run·차감·큐 hash·core 결과·진입 여부·폐기·hash·다음 축) 을 `work_dir/_na104_20h/report_<srv>.json` 으로 |
| §2 common-grid 재선택 | `na104_20h.common_grid_pair`: 두 run 의 **실제 평가 update 교집합** 위에서 `pa/selector.BestSelector`(HQNR 1e-4 band → fSCC 1e-4 → 늦은 update) 를 각각 재생 → Δ. 원래 best(best_hqnr_meta) 는 병기·보존. eval 5/10 혼재를 이렇게 맞춘다 |
| §3 P1 필수 4 | config 존재 확인: `NA104_Q36_…_S777_v2`·`S2026_v2` 는 **새로 생성**(Q36 은 core10 밖이라 없었다), `Q00 S2026 v2`·`Q12 S2026 v2` 는 FINAL 편성분 그대로. Teacher/λ pilot 고정(T00 S2025 v1 / Q00 S1234 v1) 불변, baseline 은 같은 seed 의 Q00 |
| §4 CF01 (N0+GC-FIX, λ_C = Q36 재사용) | 새 case `CF01` (`tools/gen_na104_configs.py`): rec N0, stat GC/FIX window 5, `kd_weight 0.1` → `GT_L1 + λ_C·(mean|C_S−C_GT| + 0.1·mean|C_S−C_T|)`(FIX = fixed_kd criterion; Student 통계만 gradient — Teacher 통계는 no_grad), Teacher 는 학습에 필요하므로 `eval_only` 없음. **λ_C 재사용**: 새 키 `stat.lambda_from_run: NA104_Q36_W104_D122_WV3_N0_GCH_S1234_v1` → trainer 가 그 run 의 `calibration_resolved.json` 의 `lambda.lambda_V_used` 를 그대로 쓴다(재calibration 금지; 파일 없으면 gate `CALIBRATION_SOURCE_MISSING`, exit 4). config `NA104_CF01_W104_D122_WV3_N0_GCFIX_S{777,1234,2026}_v2`. 새 hard 재가중·adaptive gate 없음(TRI 없음) |
| §4 조건부 (서버별) | `tools/campaign_gate.py` gate `na104_20h`(token `work_dir/campaign_gates_enabled.txt`): s3 — Q36 3-seed 통과 → CF01 S777·S1234; 둘 다 Q36 대비 Δ(common grid)>0 → CF01 S2026 + `work_dir/_na104_20h/cf01_pilots_positive.json`(s2 전달용); 한쪽 비양성이면 종료(보류 큐 없음). s2 — `work_dir/_na104_20h/cf01_approved_by_s3.txt` 토큰(운영자가 s3 결과 확인 후 생성) + 자기 Q36 통과 → CF01 S777 → S1234 → S2026 |
| §5 X02 | Q12 3-seed 통과 → X02 S777·S2026 (`NA104_X02_…_S{777,2026}_v2` 새로 생성; Q12 와 같은 Teacher/λ_E/seed/eval) |
| §6 20h 예산 | `na104_20h.budget()`: switch 이후 whitelist run 의 실측(meta started/finished) 합 + 현재 run 경과; gate 는 `used + n×평균×1.1 ≤ 20` 일 때만 연다. 부족하면 P3(X02) → CF01 교차 순으로 자연히 닫힌다. 완료 시간 보장이 아니다 |
| §7 폐기·다음 축 | report 의 `retired`·`next_tuning_axes`(Q36 λ_C×{0.5,1,2}, Q12 λ_E×{0.5,1,2}, CF01 은 Q36 을 반복해서 넘을 때만 β_C) — 이번 20h 에는 계수·window·LR 변경 없음 |

## 2. gate (s1, 2026-09-13)

`tools/na104_20h_unit_tests.py` H01–H05: config 9벌 존재·큐 순서, CF01 spec(N0·GC/FIX·kd 0.1·λ from Q36·Teacher 학습·TRI 없음; Q36 과 mode/λ출처/eval_only 만 다름), λ 출처 중복 거부, common-grid 재생(eval 5/10 혼재 합성), 3-seed 기준(+0.002/−0.001/+0.004 → pass; seed 미완 → None), decide 닫힘, campaign gate 격리, switch 절차 — 통과.
**s1 에서 못 한 것**: CF01 의 실제 학습 forward(FIX 모드 + Teacher) smoke — Teacher·Q36 run 이 s2/s3 에만 있다. `na104_20h_switch.sh` 가 각 서버에서 P1 4 + CF01 을 smoke 한다(계획 §4 "먼저 smoke").

## 3. 운영자 절차 (s2·s3)

```
git pull
./tools/na104_20h_switch.sh          # 현재 run 은 끝까지 → 20h 큐(P1 4) 기동, gate token na104_20h
python tools/na104_20h.py report     # 언제든 회신 항목
# s3: CF01 pilot 둘 다 양성이면 work_dir/_na104_20h/cf01_pilots_positive.json 이 생긴다 → 그 내용을 s2 에 전달 →
# s2: touch work_dir/_na104_20h/cf01_approved_by_s3.txt (자기 Q36 도 통과해야 CF01 이 열린다)
```

큐가 끝나면 체인이 gate 를 다중 패스로 부르므로 CF01/X02 는 조건이 갖춰지는 대로 열린다. 조건이 닫히면 `[cases] DONE` 으로 끝난다(빈 시간을 폐기한 방법으로 채우지 않는다).

## 4. 약명 → 세팅

| 약명 | 세팅 |
|---|---|
| Q00 | N0 — KD·통계 없음 (독립 Student baseline) |
| Q36 | N0 + GC-H — GT covariance 통계 hard(window 5), Teacher 는 평가 전용 |
| Q12 | R3 + EDGE-H — GT-anchored adaptive KD(R3) + signed GT edge hard |
| CF01 | N0 + GC-FIX — GT_L1 + λ_C·(mean|C_S−C_GT| + 0.1·mean|C_S−C_T|), window 5, λ_C = Q36(S1234 v1) 이 쓴 값 |
| X02 | R1 + EDGE-H — Q12 의 output soft 제거 대조 |
