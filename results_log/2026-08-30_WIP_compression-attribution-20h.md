# [WIP] 진행 중인 실험 — 압축 귀속·9ch 통일 20h (2026-08-30)

> **진행 중 — s1 0/7(+게이트), s2 는 push·pull 후 기동.**
> 전부 끝나면 정식 문서로 대체하고 이 파일은 지운다 (`CONVENTION.md` §2).

계획: [`research_log/2026-08-30_compression-attribution-20h-plan.md`](../research_log/2026-08-30_compression-attribution-20h-plan.md)
· 직전 결과: [2026-08-30_swin-campaign-results.md](2026-08-30_swin-campaign-results.md)

**목표: KD Student 확정.** Efficiency winner `SW2_d122_w96`(1.95M)의 "우아한 압축"이
Swin 덕인지 d122 골격 덕인지 — d122 위 **입력{11,9ch}×Swin{유,무} 요인**을 w128·w96
두 지점에서 완성하고, CM3A 압축 짝과 w80 게이트 탐침까지.

s1 큐(≈14.9h+게이트): d122 → d122_w96 → SW2_d122_9ch → SW2_d122_w96_9ch →
d122_9ch → d122_w96_9ch → CM3A_d122 → [gate] SW2_d122_w80_9ch
s2 큐(≈10.5h+게이트): SW2_d122_w80 → d122_w96 → SW2_d122_w96_9ch → SW2_d122_9ch
→ CM3A_d122_w96 → [gate] SW2_d122_w80_9ch

전 config `expect_params_m` 기입·smoke 10/10 통과. 판정 규칙은 직전 캠페인 그대로.

### 진행 기록

- 2026-08-30: 캠페인 기동 (마감 +20h).
