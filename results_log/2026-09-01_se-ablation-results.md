# SE ablation 결과 — 게이트는 살아 있으나 성능을 못 움직였다 (2026-09-01)

계획: [`research_log/se_ablation_two_experiments.md`](../research_log/se_ablation_two_experiments.md)
· s1, 2건(합계 4.0h), 기준선 = `R4_w96_d124_noattn`(w96·d124·attn0·11ch·2.13M).
공통: 50K·nocrop·LN·dual MARs·best 선택 = 공식 HQNR. 판정: HQNR(band 0.011) →
SCC(판정선 0.0005). 참고 지표는 시트(WV3-s1) 기록.

## 결과

| 실행 | 위치 | Params | HQNR | SCC | gate 진단 (`tools/analyze_se_gates.py`) |
|---|---|---:|---:|---:|---|
| (기준) `R4_w96_d124_noattn` | — | 2.1285M | 0.9561 | 0.9900 | — |
| (기준) `K0_R4_base` (재기준) | — | 2.1285M | 0.9562 | 0.9901 | — |
| `SE1_R4_btl_se` | bottleneck ResBlock ×4 | 2.1382M | 0.9543 | 0.9903 | 활성(±0.12, 포화 0%) · **mode-cos 0.994~0.9999** |
| `SE2_R4_dec_h2_se` | H/2 skip-fusion 뒤 ×1 | 2.1309M | 0.9533 | 0.9903 | 활성(±0.05, 포화 0%) · mode-cos 0.998 |

## 판정 (계획 §8·§9)

1. **두 위치 모두 기준선과 HQNR·SCC 로 구분되지 않는다** — 채택 조건 1·2(성능 개선)
   불충족.
2. **채택 조건 3(동급 + mode-dependent selection)도 불충족** — 게이트는 콘텐츠에
   따라 실제로 열리고 닫히지만(상수 수렴 아님 → 실패 조건은 미해당), MS/PAN mode
   간 cosine 이 0.994~0.9999 로 **mode 의존적 채널 선택은 형성되지 않았다.**
3. §9 결정표의 "둘 다 무효" 행 — **SE 계열 종료, loss/KD 연구에 집중.**
   SE1+SE2 결합 후속도 열지 않는다(전제 미충족).

## 해석 한 줄

R4 의 제한된 채널이 "중요도 재조정"으로 회복될 여지는 없었다 — 채널 부족이 아니라
용량 자체의 문제라는 기존 폭-곡선 결론(폭 축소 단조 유상)과 정합한다. SE 게이트가
mode 를 구분하지 않은 것은 MARs 의 mode 조건화(γβ)가 이미 그 역할을 흡수하고
있다는 방증이기도 하다 — MS-only(plain) 전환 논의와도 어긋나지 않는다.

## 운영

체인 2/2 무장애(마감 여유 2h), KD 게이트 재점검은 "전부 완료"로 무해 통과.
gate 진단 JSON 은 각 work_dir 의 `se_gates.json`.
