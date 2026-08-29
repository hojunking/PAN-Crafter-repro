# [최종] Swin·CM3A 대조 및 압축 24h 실행 계획 (2026-08-29)

> **실행 기준 문서.** 세 문서의 통합 최종본이다 —
> [원안 8/28](2026-08-28_swin-24h-s1-s2-plan.md) →
> [검토 v2 + 검증 추기](2026-08-29_swin-24h-plan-v2.md) →
> [사용자 신계획 8/29](2026-08-29_swin-attention-compression-24h-plan.md).
> 이 문서와 앞선 문서가 다르면 이 문서를 따른다.
> 구현은 완료·검증됐다 ([구현 보고서](2026-08-29_swin-implementation-report.md), 검사 23/23).

## 0. 신계획에서 그대로 확정하는 것

- **판정 규칙 §3 전체** — HQNR band 0.011 → SCC → ERGAS 순서, SCC 단독 승자 선언 금지,
  **Quality winner / Efficiency Pareto 분리**, 축소 운영 게이트(HQNR drop ≤0.011 ·
  ERGAS ≤3% · latency/params ≥15% 감소).
- **c6-s2 재현 완료 반영** — s2 는 c6 를 다시 돌리지 않고 완료된 c6-s2
  (ERGAS 2.0716 · HQNR 0.9528)를 서버 내부 anchor 로 쓴다.
- CM3A 대조군은 **no-PAN-K/V (cm3a_pan_branch: false)** — c3b 계열을 d124 로 이식.
- SW8/SW10 삭제, SW4 정체 시 그 자체가 depth 포화 결론.
- LR-TinySwin 은 반증 한 벌만, 어떤 결과에서도 이번 큐에서 확대하지 않음.
- s1·s2 의 SW2 중복은 seed 검증이 아니라 서버별 local anchor.

ERGAS 보조판정의 유의선은 **0.23%** (두 단일 실행 차이의 3σ = 2σ 0.11% × √2)로 명시한다.
§3.3 의 "3%"는 품질 동급 정의가 아니라 축소 지속용 비용 예산이다 (신계획 서술 유지).

## 1. 통합에서 조정한 것 3가지

**① s1 에 순수 용량 대조군 `c6_d126` 추가 (구현 0줄, 2.6h).**
build 실측이 우연히 좋은 정렬을 준다 — **SW4 4.3054M · d126 4.3643M · CM3A 4.3921M
(±1.6%)**. 같은 예산에서 "표준 Swin attention vs 자체 CM3A attention vs 순수 conv
용량(btl ResBlock +2)"의 **매칭 3자 비교**가 성립한다. 이것 없이는 SW2/CM3A 의 승리가
"bottleneck 에 +0.27~0.62M 이 늘어서"라는 대안 가설을 가를 수 없다
(w112→w128 만으로 ERGAS 1.63% 가 움직이는 것이 확인된 저장소다).

**② LR 반증은 `LR_SW2_w128` (0.5907M) 로 변경.**
신계획의 w64(0.194M)는 이미 기각된 LR-Fuse L1(0.544M)의 1/3 용량이라 실패가
과잉결정된다 — 그 실패로는 "Swin 전역 attention 이면 다른가"(신계획 §9 의 질문)에
답할 수 없다. w128 은 L1 과 params +8.6% 로 매칭되어 **mechanism 만 다른 비교**가 된다.
비용 차이는 ~0.2h. 0.2M급 ultra-lite 지점 탐색은 w128 이 생존할 때만 다음 캠페인에서.

**③ s2 예비로 `N3_9_d124_noattn` 재현.**
c6-s2 가 환경을 검증했지만 **9ch 성립(Teacher-Student 입력 통일 결정)은 아직 s1 단일
근거**다. s2 확정+게이트가 ~14h 라 슬랙이 충분하므로, 게이트 종료 후 잔여 시간에 돌린다.

## 2. s1 큐 — attention 의 존재·종류·깊이·용량 등가 (확정 5 + 게이트 1)

전 case: c6 골격(w128·d124·nocrop·LN·50K·dual MARs·HQNR 선택). params 는 build 실측.

| # | config | 입력 | bottleneck | Params | 예상 | 확인 질문 |
|--:|---|---:|---|---:|---:|---|
| 1 | `SW2_add` | 11ch | Res×4 + Swin×2 | 4.0387M | 2.8h | Swin pair 자체가 유효한가 (s2 공통 anchor) |
| 2 | `CM3A_btl_nopan_d124` | 11ch | Res×4 + CM3A×1(noPANkv) | 4.3921M | 2.6h | 효과가 Swin 고유인가, btl attention 일반 효과인가 |
| 3 | `SW4_add` | 11ch | Res×4 + Swin×4 | 4.3054M | 3.1h | depth 확대 유효성 |
| 4 | `SW2_add_9ch` | 9ch | Res×4 + Swin×2 | 4.0364M | 2.8h | 입력×attention 상호작용 (N3 와 직접 비교) |
| 5 | `c6_d126` | 11ch | Res×6 (attention 없음) | 4.3643M | 2.6h | **용량 등가 대조** — attention 없이 params 만 늘리면? |
| g | [gate] `SW6_add` | 11ch | Res×4 + Swin×6 | 4.5722M | 3.4h | 포화점 |

확정 13.9h + SW6 3.4h = **최대 17.3h** (구현은 끝나 있어 전부 학습 시간).

**SW6 게이트** (신계획 §4.3 그대로): SW4 가 SW2 대비 HQNR +0.011 초과 개선, 또는 HQNR
동급이며 SCC·ERGAS 동방향 개선일 때만. 정체·지배·(ERGAS 악화∧latency 증가)면 중단.

**핵심 비교 4개** (신계획 §8 승계 + 1 추가):
c6↔SW2 (존재) · SW2↔CM3A (기전) · **SW4↔d126↔CM3A (용량 등가 3자)** ·
SW2↔SW4[↔SW6] (깊이) · N3↔SW2_9ch vs c6↔SW2 (입력 상호작용).

## 3. s2 큐 — Swin 유지 압축 (확정 4 + 게이트 2 + 예비 1)

anchor: 완료된 c6-s2. s2 는 `tools/campaign_start.sh --queue` 로 기동 (아래 §5).

| # | config | 구조 | Params | 예상 | 확인 질문 |
|--:|---|---|---:|---:|---|
| 1 | `SW2_add` (s1 과 동일 config) | d124 + Swin×2 | 4.0387M | ~3.0h | s2 에서 Swin 효과 |
| 2 | `SW2_d024` | d024 + Swin×2 | 3.2657M | ~2.0h | full-res 제거를 Swin 이 벌충하는가 |
| 3 | `SW2_d122` | d122(btl 4→2) + Swin×2 | 3.4463M | ~2.6h | 3M대 예산을 고해상도 vs btl 어디에 |
| 4 | `LR_SW2_w128` | LR-TinySwin w128·sw2 | 0.5907M | ~1.1h | LR 패러다임 최종 반증 (L1 과 용량 매칭) |
| g1 | [gate] `SW2_<승자>_w112` | quality parent 의 w112 | 2.5~3.1M | ~2.4h | hybrid 폭 축소 기울기 |
| g2 | [gate] `SW2_<승자>_w96` | 〃 w96 | build 시 | ~2.1h | w112 통과 시에만 |
| 예비 | `N3_9_d124_noattn` (기존 config) | 9ch 재현 | 3.7696M | ~3.0h | 9ch 성립의 서버 독립 확정 |

확정 ~8.7h + 게이트 ~4.5h + 예비 ~3h = **최대 ~16h**.

- **D024 vs BTL2 판정** (신계획 §5.2 그대로): HQNR band → SCC → ERGAS 로 quality
  parent 하나 선정. near-budget allocation 실험으로 기록 (3.27 vs 3.45M, exact 매칭 아님).
- **W112→W96 게이트** (신계획 §5.3 그대로): HQNR drop ≤0.011 ∧ ERGAS ≤3% ∧
  latency/params ≥15% 감소 ∧ 비지배. 러너의 **게이트 다중 패스** 덕에 2단 게이트
  (W112 결과 → W96)가 실제로 작동한다.
- **LR_SW2 판정** (신계획 §5.4 승계, w128 기준): HQNR <0.940 이면 LR-only 종료.
  생존해도 이번 큐 확대 없음.

## 4. 결과 기록·최종 결론 형식

신계획 §8(기록 항목)·§9(질문-결론 표) 그대로. 종료 시 **Quality winner 와
Efficiency Pareto winner 를 분리해 선언**하고, results_log 에 정식 보고서를 쓴다
(표의 실행명은 전체 config 이름, 약어는 문서 내 대응표 필수).

## 5. 실행 절차 (양 서버 공통)

```bash
# s1 (구현 커밋 push 후):
cat > /tmp/queue_s1.txt << 'Q'
SW2_add
CM3A_btl_nopan_d124
SW4_add
SW2_add_9ch
c6_d126
Q
./tools/campaign_start.sh --queue /tmp/queue_s1.txt --hours 24

# s2 (git pull 후):
cat > /tmp/queue_s2.txt << 'Q'
SW2_add
SW2_d024
SW2_d122
LR_SW2_w128
N3_9_d124_noattn
Q
./tools/campaign_start.sh --queue /tmp/queue_s2.txt --hours 24
```

- SW6·W112·W96 은 큐가 아니라 `campaign_gate.py` 가 연다 (다중 패스).
- 모든 신규 config 에 `expect_params_m` 을 기입한다 — 옵션 누락(cm3a_pan_branch 등)을
  smoke 가 학습 전에 잡는다.
- N3 재현은 s2 큐 마지막에 두어, 마감이 닥치면 자동으로 마감스킵된다 (예비 시맨틱).
- w112/w96 config 는 quality parent 확정 후 만들어 커밋한다 (게이트가 열리기 전까지
  config 부재는 exit 2 로 무해).

## 6. 예상 의사결정 (실행 전 가설 — 판정에는 쓰지 않는다)

- 보수적 예상: SW2 또는 CM3A 가 품질형 후보, D024 계열이 효율형 후보 (신계획 §9).
- d126 이 SW4·CM3A 와 동급이면 "attention 이 아니라 용량"이 결론이 되고, Swin 서사는
  기각된다 — 이 경우 Teacher 는 c6/N3 유지, KD 캠페인으로 직행한다.
