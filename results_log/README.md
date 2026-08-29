# 실험 로그 인덱스

작성 규약은 [CONVENTION.md](CONVENTION.md), 무엇을 위해 쓰는지는 [PURPOSE.md](PURPOSE.md).
옆 저장소 [`../../CANConv/results_log`](../../CANConv/results_log) 와 동일 규약이므로
두 저장소의 수치를 그대로 나란히 놓을 수 있다.

새 문서는 **맨 위에** 한 행 추가한다.

| 날짜 | 문서 | 요지 |
|---|---|---|
| 2026-08-29 | [WIP_swin-24h](2026-08-29_WIP_swin-24h.md) | **[진행 중] Swin·CM3A 대조 및 압축 24h.** s1: Swin 존재·기전·용량등가·깊이(5+게이트) · s2: Swin 유지 압축(4+게이트 3). 첫 결과 SW2_add — **c6 대비 ERGAS +2.85% 악화**, HQNR 동급. 대조군(CM3A·d126)이 원인 귀속 예정 |
| 2026-08-29 | [arch-search-24h-results](2026-08-29_arch-search-24h-results.md) | **24h 탐색 9/9 완주 (13.7h, 무장애).** ① **9ch 성립** — N3 이 c6 와 전 지표 동급, 입력을 논문 9ch 로 되돌릴 근거. ② **고해상도 인접 용량이 핵심 자원** — enc full-res 블록 1개가 +4.93%(A1→A2 짝), full-res 제거(R1)는 HQNR 무손실·ERGAS +2.28%·추론 2배. ③ **LR-Fuse 기각**(ERGAS 4.0+, 용량·구조 문제). 폭 곡선 단조 유상(w112 +1.63%, w96 +2.84%). 게이트 3개 자동 닫힘. **주력 R3 · 초경량 R6(1.69M, c8 실질 대체) · KD Student 1순위 R1** |
| 2026-08-28 | [lightweight-case-results](2026-08-28_lightweight-case-results.md) | **경량화 캠페인 11/11 완결 (HQNR 선택, 50K).** HQNR 폭 0.40% < 판정선 → **10벌 전부 동급**, 판별은 ERGAS·params 로. **MARs 재검증 — 논문 Table 16 재현 안 됨**(PAN mode 제거해도 HQNR 2위·ERGAS +1.4%, 학습 2배 가속 공짜). **c6(3.77M, attn0+d124)** 이 SCC 1위·ERGAS 2.0826 다크호스, d124 효과는 어텐션 유무에 조건부(§2.4). 위치 대결 btl>enc(명목). **(추기) c8(2.46M)에서 첫 전지표 하락 — 무손실 하한은 c6(3.77M)**. cron PATH 사고 기록 |
| 2026-08-25 | [architecture-tuning-review](2026-08-25_architecture-tuning-review.md) | **재구성본 위에서의 세팅 변형 검토.** 우리 구현 오류 1(GroupNorm — 논문은 Eq 5 에서 LN, 비용 **−1.34% = 12σ**, 구조 재구성 전체 −0.69% 보다 크다) · 논문이 열어둔 선택 4(mlp_ratio **구분 불가**로 정리, **depth 배분이 가장 근거 약함**, s_embed·잔차 기준선 미검증) · 논문 밖 시도 5(nocrop·11ch·용량확대·폭축소 진행 예정, AttnBlock/bottleneck 축소 취소). **시드 2σ = 0.11%** 로 A-1/A-2 의 RR 효과 판정 철회. **SCC 는 전 구성 0.9902 로 포화**, ERGAS 최고와 HQNR 최고가 갈린다 |
| 2026-08-25 | [paper-vs-code-divergences](2026-08-25_paper-vs-code-divergences.md) | **논문 서술 vs 배포 코드 불일치 13곳 + 추가 이슈.** 기존 코드베이스 | 재구성본 대조표. 구조 4(scale 3·Down/Up 2·AttnBlock 3·고해상도는 ResBlock only) · 블록내부 3(LayerNorm·γβ 직접학습·dropout 없음) · CM3A 4(K_pan/V_pan 결합·k=3 전역·key 별칭 버그·reset_parameters 미호출) · 입력 1(9ch) · 학습 1(crop 이 실은 scale jitter). **11곳 반영, 2곳 미반영**(결합 시 params 7.2122M 로 어긋남 / crop 은 논문대로 구현 불가). 판정 보류 4(잔차 보간법·depth 배분·mlp_ratio·s_embed 미기재). 추가 이슈로 F-1 데이터 결함, C·D·E 운용/지표/평가, FLOPs 미해결 |
| 2026-08-24 | [paper-faithful-rebuild](2026-08-24_paper-faithful-rebuild.md) | **논문 서술대로 재구현하니 params 가 맞았다 — 7.1707 M vs 논문 7.170 M (+0.01%).** 되돌린 것 셋 다 논문 본문: mode modulation 을 직접 학습 γ,β 로(Eq 6, 블록당 33,024→512), bottleneck 도 k=3(배포본만 k=1), 입력 9ch(PAN+LRMS). 구조는 Figure 3 의 **3-scale / Down·Up 2 / AttnBlock 3**. **FLOPs 79.03 G 는 여전히 미해결**(재구성본 161.9 G, 어텐션 빼도 125.9 G — '미집계' 가설 기각). **시드 폭 0.81% 로 기존 차이 대부분(−0.31%, +0.74%)을 삼킨다 → 앞으로 시드 3개 이상.** 기준선이 9.969 M→7.171 M 로 바뀌어 '−33% 무손실'의 상당 부분은 배포본 잉여였다. 경량화 축도 **width ≫ full-res depth** 로 이동 |
| 2026-08-24 | [wv3-summary-and-protocol](2026-08-24_wv3-summary-and-protocol.md) | **WV3 대표 결과표 + 프로토콜 정의.** RR 20장·FR 12–19 8장 전 지표에 **비용(params/FLOPs/time/학습시간) 통합**. `baseline`(배포본 그대로) vs `fixed`(논문 Eq 10/11·Sec 3.3 복원)의 정의와 **왜 둘 다 돌리는지**(ERGAS +0.74% 악화 / SAM −0.34% 개선, 둘 다 유의). **구조 실험만 25K 인 이유**(50K 는 5.94h → 스윕 71h vs 15h)와 그 편향 −3.6% 분리. 25K·50K 를 가로로 읽으면 안 된다 |
| 2026-08-24 | [WIP_running](2026-08-24_WIP_running.md) | **[진행 중]** 두 축 결합 8종(≈16:28) + 논문 격차 진단 2종(≈21:10). 시드 변동 폭을 얻어 ±0.4% 차이의 유의성을 판정하고, `crop`·잔차 기준선 후보를 검증한다 |
| 2026-08-24 | [reproduction-audit](2026-08-24_reproduction-audit.md) | **재현 감사.** ① **우리 수정은 무죄** — 기본값이 배포 원본과 **비트 동일**(출력 diff 0), C-1·D-2·E-1 전부 0.1% 미만. 단 **A-1/A-2 자체는 ERGAS +0.7% 악화·SAM −0.3% 개선**(둘 다 유의). ② **논문 격차도 측정 무죄** — CANConv 배포 가중치로 논문 행을 **6지표 0.5% 이내** 재현(PSNR 0.06%), 시드 2,025 까지 설정 일치 → **배포 코드 ≠ 논문 모델**(params 1.39×, FLOPs 2.1×)이 유력. **논문의 CANConv 대비 우위 재현 안 됨**(주장 −5.69% vs 재현 −0.34%, p=0.667). ③ 다중 지표 — ERGAS·SAM 만 판별력, Q8·SSIM·SCC 는 포화 |
| 2026-08-22 | [extended-ablation-and-kd-target](2026-08-22_extended-ablation-and-kd-target.md) | 확장 8종(15h). **6.694M / 8.4ms 가 Teacher 와 구분 불가**(p=0.114) — 파라미터 −33%, 추론 2.1× 를 무손실로. **PAN 브랜치는 빼면 오히려 나아진다**(최대 −0.88%). 8/21 의 '+119% 붕괴' 는 A3 조건부였음을 정정(A5 에선 +10.5%). 50K 에서 격차 절반(+0.88%→+0.41%). **KD 가 필요한 지점은 6M 아래**. |
| 2026-08-21 | [submodule-ablation](2026-08-21_submodule-ablation.md) | 단일 8종 + 조합 9종 (29h). **모든 서브모듈이 개별로 +1.5% 이내 제거 가능**, 조합은 가산 이하. **PAN 브랜치 전체 제거도 +1.07%** — CM3A 의 cross-modality 기여는 논문 주장(9.4%)보다 훨씬 작다. **MARs mode 조건화 2경로는 중복 — 둘 다 빼면 +119% 붕괴.** 최대 감축 −32% params / 2.0× 추론에 +0.88% |
| 2026-08-20 | [submodule-removal-analysis](2026-08-20_submodule-removal-analysis.md) | 서브모듈 전수 조사. **추론의 41.8% 가 H/2 CM3A** — 이미 무손실로 제거됨. 나머지는 전부 15% 이하라 비용 절감 여지 없음 → 앞으로는 **진단 목적** 실험. interpolate 중복 제거는 비트 동일 4.2% 공짜 |
| 2026-08-20 | [mutual-learning-go-no-go](2026-08-20_mutual-learning-go-no-go.md) | **양방향 mutual learning no-go**(단방향 T→S 증류는 별개·유효). 9쌍 전부 기준 미달(최선 상관 0.863 / 오라클 +5.6%, 기준 0.85·+15%). 용량비를 20배 범위로 훑어도 상관과 오라클이 함께 떨어진다. 학생 승리 영역에 구조 없음 |
| 2026-08-20 | [student-architecture-sweep](2026-08-20_student-architecture-sweep.md) | 12구성 스윕(15h). **CM3A 2개 제거는 무손실**(p=0.20)로 추론 1.7×. **width 축소는 가성비 최악**(26% 손실에 속도 이득 없음). full-res 는 용량에 둔감 — 2.51M 이 9.97M 과 HQNR 동일 |
| 2026-08-19 | [metric-and-dataset-audit](2026-08-19_metric-and-dataset-audit.md) | **두 논문 지표 전부 산출 가능, 데이터 결손 없음.** SCC 정의 오류 교정(0.878→0.990, 논문 0.988), SSIM 추가(0.9754 vs 0.976). PSNR·SSIM 은 DLPan 표준에 없어 구현 불확정 |
| 2026-08-19 | [review_v0.3_architecture_student_first](2026-08-19_review_v0.3_architecture_student_first.md) | v0.3 아키텍처 검토. **문서 채널 스펙이 논문의 1/4**(2.27M vs 9.115M), **OOM 위험 과장**(batch32에 1.8GB, 300K≈2.4h) → alternating 불필요. 비교 타깃은 Table 6 L1 행 GF2 HQNR 0.935 |
| 2026-08-19 | [wv3-four-runs-and-zeroshot](2026-08-19_wv3-four-runs-and-zeroshot.md) | WV3 4벌 완결. 재현 성립(ERGAS 2.164 / HQNR 0.948). **A-1/A-2 수정은 in-domain 중립~손해, WV2 zero-shot 에서 HQNR +2.0%**. 선택 편향 +0.14~0.21%. full-res 복구 0.377→0.949 |
| 2026-08-18 | [review_variance-regularized-mutual-overfitting](2026-08-18_review_variance-regularized-mutual-overfitting.md) | 신규 연구방향 검토. GT-variance 축은 근거 확인(corr 0.56), **mutual learning 축은 약함(오차 상관 0.94, 오라클 상한 +7.9%)** |
| 2026-08-14 | [wv3-baseline-vs-fixed](2026-08-14_wv3-baseline-vs-fixed.md) | WV3 reduced 재현 성립 — ERGAS 2.163 / Q8 0.9165 (논문 2.040 / 0.922). A-1/A-2 효과는 판정 보류. **배포 pan_h5.zip 의 WV3·QB full-res lpan 불일치 → full-res 평가 무효**, 재생성 레시피 확보 |

---

## 이 저장소 로그에만 있는 규칙

**모든 수치에 두 가지를 명시한다.**

1. **어느 실행인가** — `baseline`(배포본 그대로) / `fixed`(KNOWN_ISSUES A-1·A-2 적용).
   배포 코드가 논문과 다르게 동작하므로, 이것을 빠뜨리면 "재현 실패" 와 "코드 버그" 를
   구분할 수 없다.
2. **누가 잰 값인가** — `py`(학습 중 `metrics.csv`) / `matlab`(DLPan-Toolbox).
   둘은 정의가 다르다. 논문 Table 과 비교 가능한 것은 `matlab` 뿐이다.

표기 예: `ERGAS 2.31 (fixed, matlab)`

## 그림 자산

`assets/` 에 둔다. 학습 곡선은 다음으로 생성한다.

```bash
python ../tools/plot_metrics.py ../work_dir/wv3_baseline ../work_dir/wv3_fixed \
       --out assets/curve_wv3.png
```

위성영상은 라이선스 제약이 있으므로 **외부 서비스 업로드 금지**.
