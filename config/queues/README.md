# 서버별 캠페인 큐 (git 추적)
# 기동: ./tools/campaign_start.sh --queue config/queues/<파일> --hours <N>
# s1_kd: K2~K4 는 큐가 아니라 campaign_gate.py 가 연다
#        (T1 calibration PASS + 대조군 K1B·K1B_T1 종결 시 K2→K3→K4 순차)
# s2_mutual: M0→M1→M2→M3

## 2026-09-02 캠페인
- s1_teacher_arch.txt : 4-6M teacher 탐색 (계획 2026-09-01_s1-...). 큐는 6건,
  중간점(T04A/B)·winner dual(T05)·NAF fallback 은 campaign_gate.py 가 연다.
- s2_uncertainty_kd.txt : teacher 1건만. head calibration(tools/calibrate_head.py)과
  student 8건(PKD 2 + UKD 6), GT-var 2건은 전부 게이트가 순차 개방한다.

## 2026-09-02 재기동 — dual MARs 4-6M 탐색
- s1_teacher_arch_dual.txt : MS-only plain 이 4-6M 에서 후반 D_lambda 붕괴를 일으켜
  (같은 구조 dual 대비 plateau -0.0112) 같은 격자를 dual MARs 로 다시 돈다.
  기존 dual 3벌 재사용: R4_w96_d124_noattn(W96) · c6_c4d124(W128) ·
  S1_T05_W160_D124_DUAL(W160-d124). 이 큐의 5벌로 4-6M 격자가 완성된다.

## 2026-09-03 — 고해상도 배치 축 + 7M 상한 + 스케줄
- s1_teacher_placement.txt : 6벌(~19h). 근거는
  results_log/2026-09-03_teacher-arch-4to6m-results.md §3.
  ① 배치 축(d0/d1 을 처음 움직임) 4벌 — W144·d224 와 W144·d134 는 params 동일(5.5181M)
     한 짝이라 full-res vs H/2 를 순수 대조한다
  ② 7M 상한 1벌(W168·d124 6.48M) — params↔D_λ 추세(r=0.922) 반전 여부
  ③ 스케줄 1벌(W160·d124 25K) — 50K 판과 구조·시드 동일, 스케줄만 다름.
     이 비교는 plateau_report 의 **'plateau(후반)' 열**로 본다(공통 epoch 열이 아니라)
