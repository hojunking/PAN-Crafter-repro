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
