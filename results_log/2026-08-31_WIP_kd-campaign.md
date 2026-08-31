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
