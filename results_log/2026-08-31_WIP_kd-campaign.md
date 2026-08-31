# [WIP] 진행 중인 실험 — KD(s1)·Mutual(s2) 캠페인 (2026-08-31)

> **진행 중 — s1 1/6(+게이트 K2~K4), s2 는 push·pull 후 기동.**
> 전부 끝나면 정식 문서로 대체하고 이 파일은 지운다 (`CONVENTION.md` §2).

명세·구현: [`research_log/2026-08-31_kd-mutual-implementation-report.md`](../research_log/2026-08-31_kd-mutual-implementation-report.md)
(checkpoint 선택 = 공식 HQNR 불변). Teacher = c6 계열(3.77M), Student = R4
(= `R4_w96_d124_noattn`, w96·d124·attn0·11ch, 2.13M — c6 대비 ERGAS +2.84% 가 회복 대상).

s1 큐: T1(c6+uncertainty) → T2(+SiS) → K0(R4 재기준) → K1A(full KD) → K1B(spectral
KD, T=c6) → K1B_T1(spectral KD, T=T1 — teacher 교란 분리) → [게이트] K2→K3→K4
(T1 calibration PASS ∧ 대조군 완료 시에만). s2 큐: M0→M1→M2→M3.

### 진행 기록

- 2026-08-31 16:55: **`T1_c6_unc` 완료 (1/6, 2.5h).** DLPan: ERGAS 2.0950(c6 +0.60%) ·
  **SAM 2.7937(c6 2.8178 보다 −0.85% 개선)** · SCC 0.9906 · Q2n 0.9211,
  HQNR 0.9522 @145. uncertainty NLL 학습은 teacher 품질을 거의 보존하며(ERGAS 소폭
  유상, SAM 은 오히려 개선 — 가중 L1 의 robust 효과로 해석) head 를 얹는 데 성공.
  **Calibration: Spearman(θ, |err|) = 0.884, 5분위 MAE 완전 단조(0.0050→0.0431,
  8.6배) → PASS** — θ 가 오차를 강하게 예측한다. **K2~K4 게이트의 calibration
  조건 충족** (대조군 K1B·K1B_T1 완료가 남은 조건). T2 는 16:57 개시.
- 2026-08-31 19:27: **`T2_c6_unc_sis` 완료 (2/6, 2.49h).** HQNR **0.9498** @185
  (T1 0.9522·c6 0.9536 과 동급 band, SCC 0.9911 명목 최고). Calibration 도
  **PASS**(Spearman 0.885, 5분위 완전 단조 — T1 과 동률). SiS 진단: center 비율
  0.64 / boundary 0.36 (r=1 의 9후보 중 중앙 선호 우세 — 경고 조건 아님).
  **K2+ 의 teacher 는 기본값 T1 유지** (HQNR 명목 우위 0.9522 vs 0.9498, calibration
  동률이라 교체 사유 없음). K0(R4 재기준) 19:27 개시.
- 2026-08-31 21:26: **`K0_R4_base` 완료 (3/6, 2.0h) — Student 재기준 성립.**
  HQNR **0.9562** @145 · SCC 0.9902. 기존 `R4_w96_d124_noattn`(HQNR 0.9561 @150)과
  사실상 동일 — **KDTrainer(variant k0)가 기존 학습 경로와 등가임이 실증**돼,
  이후 K1A~K4 의 차이는 순수하게 KD loss 몫으로 귀속할 수 있다.
  K1A(full-output KD) 21:27 개시.
- 2026-08-31 21:26: **`K0_R4_base` 완료 (3/6, 1.98h).** HQNR **0.9562** @145 ·
  SCC 0.9902. 원본 R4(HQNR 0.9561)와 사실상 동일 — **KD 코드 경로의 재기준
  측정이 원 실행을 완벽 재현**(코드 경로 동등성 확인 완료). 이후 모든 K-실행의
  local baseline 은 이 K0 이다. K1A(full-output KD) 21:27 개시.
- 2026-08-31 23:41: **`K1A_R4_fullKD` 완료 (4/6, 2.24h) — 첫 KD 결과, 긍정적.**
  HQNR **0.9551** @210 (K0 0.9562 와 동급 band — Student 특성 유지) ·
  SCC 0.9906 (K0 0.9902, 명목 개선). full-output KD 는 Student 의 HQNR 를
  해치지 않으면서 품질 신호를 개선 방향으로 움직였다.
  (참고 지표 — 판정에 쓰지 않음: ergas_at_best 2.0837, K0 2.1117 대비 큰 회복.)
  K1B(spectral-only KD, T=c6) 23:41 개시.
