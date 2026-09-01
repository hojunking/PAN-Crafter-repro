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

---

## 추기 (2026-09-01 12시) — D_λ/D_s 분해 분석과 MS-only 결과 병합

### A. 분해로 보면 증류는 실재했다 — 방법 승자 판정

"HQNR 무변동"(§3-1)을 HQNR = (1−D_λ)(1−D_s) 분해로 열면 성분 이동이 보인다:

| 실행 | D_λ↓ | D_s↓ | 해석 |
|---|---:|---:|---|
| (참조) `c6_c4d124` | 0.0213 | 0.0256 | teacher — spectral 우위 |
| `K0_R4_base` | 0.0235 | **0.0209** | student — spatial 우위 |
| `K2_R4_uknow` | **0.0211** | 0.0251 | teacher 의 spectral 일관성 완전 이식, spatial 강점 절반 상실 |
| **`K3_R4_uknow_gtvar`** | 0.0226 | 0.0228 | GT variance 가 균형 복원 |

**명세 §31.2 운영 목표 대조: `K3_R4_uknow_gtvar` 가 유일한 전항목 충족**이다 —
D_λ gap(K0−c6 = 0.0022) 41% 회복 ✓ · D_s 악화 +0.0019 ≤ 0.002 ✓ · HQNR 비회귀 ✓.
K2 는 D_λ 를 109% 회복하지만 D_s 악화가 한도(0.002)의 2배다. K4 는 K3 와 구분
불가라 단순한 K3 를 방법 후보로 확정한다. mutual 쪽도 같은 구조 — SiS peer 는
D_λ 최저(0.0202~3), edge peer 는 D_s 우위로 **specialization 이 구성비는 설계대로
움직였으나 곱은 불변**이었다.

Teacher 감사 보강(`tools/analyze_uncertainty.py`, T1): Top-10% θ 픽셀 오차 lift
**2.33×**, 10분위 완전 단조, edge/smooth 양 영역 Spearman 0.87(단순 edge 탐지기
아님), θ~GT분산 상관 0.52 — K3 에서 GT variance 가 θ 와 비중복 정보를 더한다는
사전 근거와 결과가 정합한다.

### B. s2 MS-only 2건 — clean single-task backbone 성립

같은 기간 s2 에서 완료 (계획: `research_log/s2_ms_only_ablation_and_uncertainty_audit.md`):

| 실행 | D_λ↓ | D_s↓ | HQNR | SCC |
|---|---:|---:|---:|---:|
| `MS1_R4_msonly` (PAN task 제거) | 0.0229 | 0.0257 | 0.9520 | 0.9905 |
| **`MS2_R4_plain_msonly`** (mode modulation 까지 제거) | 0.0217 | 0.0249 | 0.9539 | **0.9910** |

둘 다 같은 서버의 R4-dual 쌍(`M0_R4R4_indep` 두 peer 0.9550/0.9520)과 동급 band,
MS2 는 SCC 명목 최고. **PAN auxiliary·mode modulation 전부를 제거한 순수 residual
U-Net 이 HQNR·SCC 동급 + 학습 ~2배** — 계획 §13 의 "clean single-task backbone
전환 가능" 분기가 성립했다. §5-2 의 재설계 논의와 합치면 자연스러운 다음 캠페인은
**plain backbone 위에서 K3 레시피 재검증**(MARs 의존성 제거 확인)이다.
