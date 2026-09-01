# KD(s1)·Mutual(s2) 캠페인 결과 — HQNR 는 어떤 변형으로도 움직이지 않았다 (2026-09-01)

명세·구현: [`research_log/2026-08-31_kd-mutual-implementation-report.md`](../research_log/2026-08-31_kd-mutual-implementation-report.md).
s1 KD 9건(테이블+게이트, 21h)·s2 mutual 4건(2-peer) **전부 무장애 완주**.
판정 기준: **HQNR(공식 12-19) → SCC**, 동급 band 0.011 / SCC 판정선 ±0.0005.
ERGAS·SAM 등은 참고 지표로 기록만 하고 판정에 쓰지 않는다.
Teacher = c6 계열(3.77M), Student = `R4_w96_d124_noattn`(w96·d124·attn0·11ch, 2.13M).

## 1. s1 — Teacher 준비와 KD 사다리

**Teacher (uncertainty head 부착 성공):**

| 실행 | HQNR | SCC | Calibration (θ-오차) |
|---|---:|---:|---|
| `T1_c6_unc` (c6+uncertainty) | 0.9522 | 0.9906 | **PASS** — Spearman 0.884, 5분위 MAE 완전 단조(8.6배) |
| `T2_c6_unc_sis` (+SiS) | 0.9498 | 0.9911 | **PASS** — 0.885, 완전 단조 |
| (참조) c6 | 0.9536 | 0.9908 | — |

셋 다 동급 band. θ 가 오차 순서를 강하게 예측해 uncertainty routing 의 전제 성립
— K2+ 의 teacher 는 HQNR 명목 우위인 T1 로 확정했다.

**KD 사다리 (Student, 전부 50K·HQNR 선택):**

| 실행 | soft 구성 | HQNR | SCC |
|---|---|---:|---:|
| `K0_R4_base` (기준선) | 없음 | **0.9562** | 0.9902 |
| `K1A_R4_fullKD` | full-output 모방 (T=c6) | 0.9551 | 0.9906 |
| `K1B_R4_specKD` | LR spectral 만 (T=c6) | 0.9550 | **0.9889** ← 유일한 악화 |
| `K1B_T1_specKD` | 〃 (T=T1) | 0.9522 | 0.9905 |
| `K2_R4_uknow` | + uncertainty routing (T=T1) | 0.9543 | 0.9905 |
| `K3_R4_uknow_gtvar` | + GT variance | 0.9552 | **0.9907** |
| `K4_R4_uknow_gtvar_sis` | + Student SiS | 0.9549 | 0.9907 |

K0 재기준은 기존 R4(0.9561)와 동일 — KD 코드 경로 등가 실증.

## 2. s2 — Mutual learning (2-peer, pair 평균 HQNR 선택)

| 실행 | 구성 | HQNR (peerA / peerB) | SCC |
|---|---|---|---:|
| `M0_R4R4_indep` | 독립 대조 (mutual 없음) | 0.9550 / 0.9520 | 0.9904 |
| `M1_R4R4_resmutual` | 양방향 residual mutual | 0.9553 / 0.9519 | 0.9905 |
| `M2_R4edge_R4sis_indep` | edge/SiS specialization 만 | 0.9548 / 0.9539 | 0.9903 |
| `M3_R4edge_R4sis_mutual` | component mutual | 0.9534 / 0.9530 | 0.9903 |

## 3. 판정

1. **KD·mutual 어느 변형도 HQNR 를 유의하게 바꾸지 못했다.** s1 사다리 전체
   (0.9522~0.9562)와 s2 네 구성(pair 평균 0.9532~0.9544)이 전부 동급 band 안이고,
   SCC 도 K1B 를 제외하면 판정선 안 — **"구분되지 않는다"가 공식 판정**이다.
   긍정적으로 읽으면: 어떤 변형도 Student 의 HQNR 특성을 훼손하지 않았다.
2. **유일하게 판별된 것은 K1B 의 SCC 악화(0.9889)와 그 원인 귀속**이다 — 같은
   spectral KD 를 T1 teacher 로 바꾸자(K1B_T1) 악화가 사라졌다. **spectral KD 는
   teacher 의 spectral 품질에 민감**하며, 대조군 쌍(K1B/K1B_T1)을 요구한 게이트
   설계가 이 교란을 정확히 분리했다.
3. **Mutual 은 재차 무효** — M1 ≈ M0(pair 평균 0.9536 vs 0.9535), M3 은 M2 보다
   명목 열위. 2026-08-20 no-go 판정과 일관되며, 명세 §35 결정 트리의 결론은
   "mutual 은 baseline/negative result 로 두고 KD 집중"이다.
4. 명목 서열에서는 **K3(uncertainty+GT variance)가 사다리 최상**(HQNR 0.9552 ·
   SCC 0.9907, K0 와 격차 최소)이고 K1B_T1→K2→K3 방향이 일관 개선이지만,
   증분이 판정선 미만이라 채택 주장은 하지 않는다.
5. 참고 지표(판정에 쓰지 않음): best epoch 의 `ergas_at_best` 는
   K0 2.1117 → K1A 2.0837 → K3 2.0776 → K4 2.0794 로 사다리를 따라 내려간다.
   D_λ/D_s 분해 등 세부는 시트(WV3-s1/s2) 참조.

## 4. 운영 기록

- 게이트 신설 항목 전부 실작동: calibration 자동 검사(서명 포함)·대조군 완비 조건·
  다중 패스 4회(K2→K3→K4 순차 개방)·"사다리 전부 완료" 종료 판정.
- s2 의 2-peer 학습·pair 평균 선택·peerA/B 이중 행 시트 업로드 실작동 확인.
- K5(feature KD)는 구현 결함으로 No-Go 보류 상태 유지.

## 5. 다음

1. **SE ablation 2건 즉시 실행** (선구현 완료 — `SE1_R4_btl_se`·`SE2_R4_dec_h2_se`).
2. KD 방향 재설계 논의 거리: HQNR 를 움직이려면 출력 모방이 아니라 **D_λ/D_s 를
   직접 겨냥하는 loss** 또는 Student 구조 개선(SE 등)이 필요하다는 것이 이번
   캠페인의 함의다. Mutual 갈래는 종료.
