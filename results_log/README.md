# 실험 로그 인덱스

작성 규약은 [CONVENTION.md](CONVENTION.md), 무엇을 위해 쓰는지는 [PURPOSE.md](PURPOSE.md).
옆 저장소 [`../../CANConv/results_log`](../../CANConv/results_log) 와 동일 규약이므로
두 저장소의 수치를 그대로 나란히 놓을 수 있다.

**하루에 한 문건 — 단, 서버별로 하나씩** ([CONVENTION.md](CONVENTION.md) §1).
같은 날 **같은 서버**의 문서가 둘 이상이면 하나로 합친다(맨 위에 요약, 아래에 원문 보존).
**서버가 다르면 합치지 않는다** — s1·s2·s3 가 각자 쓰므로 날짜는 당연히 겹친다.
서버 캠페인 결과는 파일명에 서버를 넣는다(`YYYY-MM-DD_s2_주제.md`), 교차 분석은 안 넣는다.
새 문서는 이 표 **맨 위에** 한 행 추가한다.

| 날짜 | 서버 | 문서 | 요지 |
|---|---|---|---|
| 2026-09-25 | s1·s4·s5 | [09-25 실험 보고서 — 감독 계수 실험 · 논문 최종 모델 원 해상도 예측 추출](2026-09-25_experiment-report.md) | **s4·s5**: 논문 WV3 최종 모델 기준으로 정답 가중 · Teacher 추종 · edge 감독 계수를 하나씩 낮춤·높임, seed 반복 98개 — 변형 6개 모두 차이 없음, 공동 목표 0/98. 09-23 주력 설정 민감도는 중단 후 교체. **s1**: 원 해상도 예측 20장 추가 추출 |
| 2026-09-24 | s1·s4 | [09-24 실험 보고서 — 논문 최종 모델 예측 영상 추출](2026-09-24_experiment-report.md) | 학습 없음. **s1**: WV3 · QB 최종 모델의 축소·원 해상도 예측과 정답 영상 추출(WV3 HQNR 0.9587 · ERGAS 2.056, QB 0.9251 · 3.561 — 다시 평가해 결과표와 일치). **s4**: GF2 최종 모델 예측 추출 |
| 2026-09-23 | s1–s5 | [09-23 실험 보고서 — 구성 요소 제거 반복 확장 · 주력 설정 민감도](2026-09-23_experiment-report.md) | **s1·s2·s3**: 제거 실험에 정합 모듈 초기값 출처 대조 추가, GF2 5회차 100개 — GF2 는 저·고주파 PAN 입력만 일관된 효과, 나머지 요소 차이 없음. **s4·s5**: 주력 설정 계수 변형 12개 모두 차이 없음, 공동 목표 0/43. 09-22 원 모델 재현은 중단 |
| 2026-09-22 | s3·s4·s5 | [09-22 실험 보고서 — GF2 추가 학습 세 방향 · PAN 혼합 증강 · PAN-Crafter 원 모델 재현](2026-09-22_experiment-report.md) | GF2: PAN 고주파 세기 증강만 HQNR 상승(+0.0047~+0.0106), 세 Teacher 에서 반복되고 증강 폭에 비례. 폭 두 배 모델 하나가 HQNR 0.9651 · ERGAS 0.5511 로 GF2 공동 목표 첫 동시 달성(seed 하나). PAN 혼합 실험은 26개 뒤 중단, 원 모델 재현으로 전환 — 재현 HQNR 은 QB 만 논문 수준 |
| 2026-09-21 | s1–s5 | [09-21 실험 보고서 — 구성 요소 제거 반복 · GF2 긴 학습 · GF2 평가 · 정합 분석](2026-09-21_experiment-report.md) | WV3 구성 요소 제거 9종 모두 차이 없음(저·고주파 PAN 입력만 일관된 방향), QB 는 회차 간 흔들림이 커 분리 불가. GF2 긴 학습 18개 공동 목표 0개(HQNR 최고 0.9560). 정합 모듈은 GF2 에서만 실제 어긋남 방향과 일치(코사인 +0.81, WV3 −0.40) |
| 2026-09-20 | s1–s5 | [09-20 실험 보고서 — QB·GF2 이식 40시간 · GF2 HQNR 격차 감사 · GF2 전 서버 20시간](2026-09-20_experiment-report.md) | WV3 탐색 종료(09-19 후속 실험 중단). QB: Student 27개 중 8개가 한 시점에서 논문 HQNR · ERGAS 동시 초과. GF2: 모두 논문 HQNR 미달(최고 0.9578), 부족분은 공간 왜곡 D_s. 격차 감사 — 평가·데이터 정상, 격차는 모델 구성 차이. GF2 20시간 다섯 축 일관된 개선 없음 |
| 2026-09-19 | s1–s5 | [09-19 실험 보고서 — 저·고주파 PAN 입력 후속 20시간 · 시점 교차 평가](2026-09-19_experiment-report.md) | 64개 중 공동 목표 0개(HQNR 하한 통과 12개, 그 안 ERGAS 최저 2.052). 후반 HQNR 하락은 U-Net 쪽 변화. 깊이 [1,2,2] 는 종료 시점 ERGAS 만 개선. s4·s5 에서 저·고주파 입력이 저주파 입력보다 12쌍 모두 높음(평균 +0.0028) |
| 2026-09-18 | s1–s5 | [09-18 실험 보고서 — 정합 모듈을 끈 추론 · 20시간 혼합 seed 재편 · 저·고주파 PAN 입력 + 새 Teacher](2026-09-18_experiment-report.md) | 정합 모듈을 끄면 평가한 64개 모두 악화(HQNR −0.0058). 20시간 재편 23개 모두 HQNR 하한 미달, Teacher 추종 강도 0.1 / 0.2 차이 없음. 저·고주파 PAN 입력은 6쌍 모두 종료 시점 ERGAS 개선. s1 분석 — ERGAS 격차는 입력 크기가 아닌 U-Net 학습 시점 문제 |
| 2026-09-17 | s1–s5 | [09-17 실험 보고서 — q 가중 복원 KD 튜닝 순서 조정 · 설정 고정 seed 반복 · 정합 모듈 방법 감사](2026-09-17_experiment-report.md) | 추가 14개 목표 미달, 대응 비교 전부 판정 기준 안. Teacher 추종 0.1 vs 0.2 엇갈려 0.1 유지. 고정 설정 새 seed 11개 HQNR 0.9550–0.9596, 공동 목표 0. 감사 — 정합 모듈은 방향은 배우나 크기 약 1/3, 정합이 강할수록 HQNR · ERGAS 악화, 현재 형태로 방법 기여 제안 불가 |
| 2026-09-16 | s1–s5 | [09-16 실험 보고서 — 정답 edge 감독 강도·시점 · q 가중 복원 KD 전 서버 튜닝](2026-09-16_experiment-report.md) | edge 감독의 강도 · 시점을 바꿔도 차이 없음. q 가중 복원 KD 67개 중 HQNR 0.9585 이상 2개, 공동 목표(ERGAS 2.040 미만 동시) 0개. 계수 · q 가중 모두 일관된 효과 없음 |
| 2026-09-15 | s1–s5 | [09-15 실험 보고서 — 기준 Teacher 정합 모듈 재사용 KD 파생 · q 선택 edge · 다중 데이터셋 표본 기전](2026-09-15_experiment-report.md) | KD 조합 효과가 서버 · seed 마다 다름. 작은 Student(파라미터 −27%)가 원래 크기와 같은 수준. q 선택 edge 는 기준 KD 조합과 차이 없음. q 와 복원 오차의 음의 상관이 네 위성 세트 모두에서 반복 |
| 2026-09-14 | s1–s5 | [09-14 실험 보고서 — 기준 Teacher 정합 모듈 재사용 KD 통합 · q 검증 · 사분면 검증](2026-09-14_experiment-report.md) | 목표 0.959 에 닿은 KD 모델 없음(KD 없는 대조 1개 0.9594, 재실행 0.9568). 적응형 KD + edge 는 서버 간 일관된 이득 없음. q 는 조각 수준 정합 품질의 표지가 아님(PAN 텍스처가 공통 요인) — q 기반 Teacher 신호 선택 채택 안 함. s1 통합 실험은 같은 날 밤 중단 |
| 2026-09-13 | s1·s2·s3 | [09-13 실험 보고서 — 보정량 일관성 가중치 정밀 확인 · 정합 모듈 없는 KD 20시간 우선순위](2026-09-13_experiment-report.md) | 가중치 1e-4 유지(3e-5 · 3e-4 모두 평균 낮음), seed 2025 모델이 이후 기준 Teacher. 정합 능력은 가중치와 함께 좋아지나 HQNR 은 1e-4 에서 정점. 정합 모듈 없는 KD 는 모델 92개에서 일관된 개선 없음 |
| 2026-09-12 | s1·s2·s3 | [09-12 실험 보고서 — 정합 모듈 실험 종합 판정 · 보정량 일관성 손실 가중치 탐색 · 정합 모듈 없는 KD 최종 편성](2026-09-12_experiment-report.md) | HQNR seed 판정 기준 0.0031 실측. PAN 을 옮긴 양이 클수록 HQNR 하락(상관 −0.91). 보정량 일관성 가중치 1e-4 가 정합 모듈 없음 대비 3 seed 모두 높음(평균 +0.0044). 정합 모듈 없는 KD 새 54개 중 판정 기준 이상 개선 0개 |
| 2026-09-11 | s1·s2·s3 | [09-11 실험 보고서 — 반응 학습된 정합 모듈 재사용 · 정합 모듈 없는 같은 크기 지식 증류](2026-09-11_experiment-report.md) | **s1**: 반응을 배운 정합 모듈을 가져와 원본 입력으로 복원 학습 — 반응을 유지한 설정은 모두 정합 모듈 없는 기준보다 HQNR 0.007–0.020 낮고, 이어 학습으로 반응이 사라져 이동량이 0.35 px 로 줄어야 기준 수준 회복. **s2·s3**: 정합 모듈 없는 폭 104·깊이 [1,2,2] 같은 크기 KD 서버당 14개 — 일관된 개선 없음(판정 기준을 넘은 두 건은 seed 하나의 경계값, 서버 간 반대), Teacher < KD 없는 Student. 확장·대조 묶음은 보류, 09-10 정답 방향 선택 KD 는 이 실험으로 대체 |
| 2026-09-10 | s1·s2·s3 | [09-10 실험 보고서 — 정합 모듈 반응 학습 · 정답 기준 지식 증류 · 모델 폭·깊이](2026-09-10_experiment-report.md) | **s1**: PAN 을 일부러 더 옮기며 보정량 일관성으로 학습하자 정합 모듈이 추가 이동을 거의 완전히 되돌리게 됨(반응 기울기 −0.98 · 잔여 오차 0.06 px). 그러나 원 해상도 이동량이 1.0–1.9 px 로 커져 학습 종료 HQNR 0.924–0.943 으로 하락(공간 왜곡 D_s 악화). WV3 학습 데이터의 PAN–MS 어긋남은 측정 오차 바닥 수준(0.17 vs 0.16 px). **s2**: Teacher 정합 모듈 재사용 + 적응형 KD 가 HQNR −0.0061, 통계 감독·정합 모듈 처리 방식 효과 없음. **s3**: 깊이 [1,2,3] ≈ [1,2,4](파라미터 −8%), 폭 112 > 96 (평균 +0.0025~+0.0050) |
| 2026-09-09 | s1 | [학습 후 정렬 분석 + 새 baseline 기동](2026-09-09_alignment-analysis-and-new-baseline.md) | **구조(edge)는 학습만으로 PAN 을 따라가지만 색(저주파)은 안 된다.** 출력 전역 shift WV3 1.79→0.55 · QB 2.78→0.18 px 인데(CANConv 도 동일 → 잔차 구조의 일반 성질), MTF 로 다시 흐리면 PAN 에서 여전히 1.2~1.6 px — **출력 안에서 구조와 색의 기하가 ~1px 어긋나 있다.** jitter 학습본도 CANConv 도 같다. **GF2 만 edge 도 절반**(1.62→1.04). 같은 날 **mainline 을 W96·D124 · MS+PAN 9ch · 단일 HRMS task 로 전환**(2.123M, WV3 3-seed 기동)하고 **best 선택 FR 세트를 논문 20장 전체로** 바꿨다 |
| 2026-09-08 | s1 | [지표 v2 + 다중 데이터셋 3-seed 기동](2026-09-08_metric-v2-and-multiset-3seed.md) | **시트가 논문과 비교 가능한지 코드 수준 검증 — 세 가지가 달랐고 전부 고쳤다.** ① **FR 테스트셋이 논문과 다른 장면**(논문 `.mat` 20장 vs 배포 H5 12-19, 겹침 6장) ② SCC 약 +0.004 · SSIM 약 +0.002 높은 옛 관례 ③ MTF 커널·경계·ddof·캐시·논문 기준행 오기. 고친 뒤 **EXP·CANConv anchor 가 논문 값과 평균·N−1 표준편차까지 일치.** 부수: **QB 학습셋 결함(F-3)** — `ms` 가 패치 67~70% 에서 `gt` 대비 LR 1px 어긋남 → QB 재시작 |
| 2026-09-07 | s1 | [alignment · shift-robust · 지표 v2](2026-09-07_alignment-shift-robust-and-metric-v2.md) | **정렬은 전부 실패, jitter 만 남았다.** MS 조건 입력 ±0.5px 무작위 jitter 가 후반 붕괴를 없앤다(final HQNR +0.005~0.006 · fSCC +0.01, 두 seed·두 backbone·두 커널 재현). **best 는 동급** — 얻는 것은 안정성. J3 blur 대조가 "smoothing 아님"을 확정. **PAN 을 흔들면 반드시 나빠진다**(G1·local·추론정렬 세 번 다 실패) → local 종료. **지표 v2**: 논문 비교 FR 은 PanCollection `.mat` 20장(배포 H5 와 다른 장면, 겹침 6장), SCC/SSIM 관례 교정, genMTF 충실 커널. 미세조정 2/4 에서 정지 |
| 2026-09-04 | s1 | [배치 축 + 판정밴드 무효](2026-09-04_placement-and-band-invalidation.md) | **판정선 0.011 이 공식 HQNR 에서 측정된 적이 없다** — 출처는 25K·proxy QNR. 실측 상한 **0.0027**(4배 좁다) → **과거 "구분되지 않는다" 판정 전부 소급 재검토.** 배치: 같은 params 를 full-res 에 쓰면 **−0.0022**(통제쌍), d0 가 params 보다 강한 예측자. 7M 확대도 실패(6.99M 이 14벌 최하위). 부수: `metrics.csv` 의 `d_lambda`/`d_s` 는 공식 분해가 아니다(순위 상관 ≈0) |
| 2026-09-03 | s1 | [Teacher 4–6M 탐색](2026-09-03_teacher-arch-4to6m.md) | **용량 확대 실패 (12건).** plateau 1위가 가장 작은 W96(2.13M)이고 4–6M 후보는 전부 아래. **best 와 plateau 가 정면 충돌한 첫 캠페인** — best 로만 봤다면 반대 결론. 부수: **MS-only plain 은 4–6M 에서 붕괴**(−0.012~−0.016), 원인은 PAN reconstruction task 제거(γβ 무관) → "MS-only 무손실" 은 **W96/W128 한정**으로 축소 |
| 2026-09-01 | s1·s2 | [KD · SE · MS-only 캠페인](2026-09-01_kd-se-msonly-campaigns.md) | **KD·SE·MS 12건 중 K0(Student 단독)를 넘은 것이 없다.** 원인은 기법이 아니라 전제 — **Teacher(3.77M)가 Student(2.13M)보다 HQNR 이 낮다**(T1 −0.0040, 판정선 초과). Teacher 를 ERGAS 로 고른 것이 어긋남의 출처. **U-Know 가중은 이득이 아니라 오염 방어**로 작동(K1B_T1→K2→K3 단조 회복). SE 두 위치 무효 → 계열 종료. **mutual 재차 무효** → 갈래 종료 |
| 2026-08-31 | s1 | [KD Teacher 준비 + Swin 종합](2026-08-31_kd-teacher-and-swin-bottleneck.md) | T1(c6+uncertainty)은 품질 손실 없이 신뢰도 맵 확보 — **θ calibration 압도적 통과**(10분위 완전 단조·꼬리 14.6배·risk–coverage oracle 대비 8%). T2 의 SiS 는 D_λ 를 사고 D_s 를 팔아 기각 → **K2+ teacher 는 T1**. **Bottleneck Swin 종합(26건)**: 11ch 기각, **효과 부호가 맥락에 따라 반전**(9ch·d122 에서만 유익), 비용은 공짜 |
| 2026-08-30 | s1·s2 | [Swin 캠페인 + 압축 귀속](2026-08-30_swin-campaign-and-compression-attribution.md) | **Swin·CM3A 13/13 완주.** Quality winner c6/N3 유지(9ch 서버 독립) · Efficiency winner SW2_d122_w96(1.95M). **입력×attention 상호작용 2.6%p**(Swin@btl 11ch 유해/9ch 무해) · CM3A>표준 Swin · LR-only 최종 기각. 압축 귀속 20h 는 KD Student 확정용 |
| 2026-08-29 | s1 | [arch-search-24h](2026-08-29_arch-search-24h-results.md) | **24h 탐색 9/9 완주.** ① **9ch 성립**(N3 이 c6 와 전 지표 동급) ② **고해상도 인접 용량이 핵심 자원** — full-res 제거(R1)는 HQNR 무손실·추론 2배 ③ **LR-Fuse 기각**. 주력 R3 · 초경량 R6 · KD Student 1순위 R1 |
| 2026-08-28 | s1 | [lightweight-case-results](2026-08-28_lightweight-case-results.md) | **경량화 11/11 완결.** **MARs 재검증 — 논문 Table 16 재현 안 됨**(PAN mode 제거해도 HQNR 2위, 학습 2배 가속 공짜). **c6(3.77M)** 이 다크호스. **c8(2.46M)에서 첫 전지표 하락 — 무손실 하한은 c6** |
| 2026-08-25 | s1 | [불일치 13곳 + 세팅 변형 검토](2026-08-25_divergences-and-tuning-review.md) | **논문 서술 vs 배포 코드 불일치 13곳**(구조 4·블록내부 3·CM3A 4·입력 1·학습 1), 11곳 반영. 재구성본 위 세팅 변형: 우리 구현 오류 1건이 **GroupNorm**(논문은 LN, 비용 −1.34% > 구조 재구성 전체 −0.69%). **depth 배분이 가장 근거 약함.** SCC 전 구성 포화 |
| 2026-08-24 | s1 | [논문 재구성 + 재현 감사 + 프로토콜](2026-08-24_paper-rebuild-and-reproduction-audit.md) | **논문대로 재구현하니 params 가 맞았다 — 7.1707 M vs 7.170 M (+0.01%).** **FLOPs 79.03 G 는 미해결.** 재현 감사: **우리 수정 무죄**(비트 동일) · **측정도 무죄**(CANConv 배포 가중치로 논문 행 6지표 0.5% 이내 재현) → **배포 코드 ≠ 논문 모델**. **논문의 CANConv 대비 우위는 재현 안 됨**(p=0.667). WV3 프로토콜(`baseline`/`fixed`, 25K·50K 가로 비교 금지) 정의 |
| 2026-08-22 | s1 | [extended-ablation-and-kd-target](2026-08-22_extended-ablation-and-kd-target.md) | 확장 8종(15h). **6.694M / 8.4ms 가 Teacher 와 구분 불가**(p=0.114). **PAN 브랜치는 빼면 오히려 나아진다.** **KD 가 필요한 지점은 6M 아래** |
| 2026-08-21 | s1 | [submodule-ablation](2026-08-21_submodule-ablation.md) | 단일 8종 + 조합 9종(29h). **모든 서브모듈이 개별로 +1.5% 이내 제거 가능**, 조합은 가산 이하. **PAN 브랜치 전체 제거도 +1.07%.** **MARs mode 조건화 2경로는 중복 — 둘 다 빼면 +119% 붕괴** |
| 2026-08-20 | s1 | [서브모듈 조사 + Student 스윕 + mutual no-go](2026-08-20_submodule-sweep-and-mutual-nogo.md) | **추론의 41.8% 가 H/2 CM3A** — 이미 무손실 제거됨, 나머지는 비용 절감 여지 없음 → 진단 목적으로 전환. **CM3A 2개 제거 무손실**(추론 1.7×) · **width 축소는 가성비 최악.** **양방향 mutual no-go**(9쌍 전부 기준 미달) — 단방향 T→S 는 별개 |
| 2026-08-19 | s1 | [지표·데이터 감사 + WV3 4벌](2026-08-19_metric-audit-and-wv3-four-runs.md) | **두 논문 지표 전부 산출 가능, 데이터 결손 없음.** SCC 정의 오류 교정(0.878→0.990). **재현 성립**(ERGAS 2.164 / HQNR 0.948), A-1/A-2 는 **WV2 zero-shot 에서 HQNR +2.0%**. v0.3 문서 채널 스펙이 논문의 1/4, OOM 위험 과장 |
| 2026-08-18 | — | [review_variance-regularized-mutual-overfitting](2026-08-18_review_variance-regularized-mutual-overfitting.md) | 신규 연구방향 검토. GT-variance 축은 근거 확인(corr 0.56), **mutual learning 축은 약함**(오차 상관 0.94) |
| 2026-08-14 | s1 | [wv3-baseline-vs-fixed](2026-08-14_wv3-baseline-vs-fixed.md) | WV3 reduced 재현 성립 — ERGAS 2.163 / Q8 0.9165. **배포 `pan_h5.zip` 의 WV3·QB full-res `lpan` 불일치 → full-res 평가 무효**, 재생성 레시피 확보 |

**진행 중**: [2026-09-06 WIP — UVS-KD s2 인계](2026-09-06_WIP_uvs-kd-s2.md)

---

## 판정 규약

**판정 지표는 HQNR(공식 FR 12-19), 보조는 SCC. 그게 전부다.**

- 판정값은 **best checkpoint HQNR**. plateau 는 **안정성 참고**로 병기하되 best 순위를
  뒤집는 근거로 앞세우지 않는다.
- **판정선 = 0.0027** (실측 상한). 종전 0.011 은
  [2026-09-04](2026-09-04_placement-and-band-invalidation.md) 에서 무효 판정됐다.
  진짜 값은 아직 미측정 — **공식 HQNR 시드 반복이 가장 시급한 실험이다.**
- HQNR 이 동급이고 SCC 도 포화면 **"구분되지 않는다"로 끝낸다.** ERGAS 로 내려가지 않는다.
- HQNR 해석은 **D_λ·D_s 분해**로 한다 (`HQNR = (1−D_λ)(1−D_s)`).
  분해는 `tools/eval_dlpan_fr.py --indices 12-19` 로만 — `metrics.csv` 의
  `d_lambda`/`d_s` 는 **proxy** 이고 공식과 순위 상관이 사실상 0 이다.
- ERGAS·SAM·PSNR·Q2n 은 **"참고 지표 — 판정에 쓰지 않음"** 으로 명시해 뒤에 둔다.
- **코드에도 적용한다** — 게이트·선택 도구에 ERGAS fallback 을 두지 않는다.

## 이 저장소 로그에만 있는 규칙

**모든 수치에 두 가지를 명시한다.**

1. **어느 실행인가** — `baseline`(배포본 그대로) / `fixed`(KNOWN_ISSUES A-1·A-2 적용).
2. **누가 잰 값인가** — `py`(학습 중 `metrics.csv`) / `matlab`(DLPan 프로토콜).
   논문 Table 과 비교 가능한 것은 `matlab` 뿐이다.

표기 예: `ERGAS 2.31 (fixed, matlab)`

**FR 은 어느 세트인가까지** — 논문 비교는 시트의 **`FR·paper mat20`** 열
(PanCollection `.mat` 20장). 문서 본문의 HQNR 12-19 는 **실행 간 판정용**이고
두 세트는 장면이 달라(겹침 6장) 섞으면 안 된다.

## 관통하는 발견

1. **`D_λ 를 사고 D_s 를 파는` 변경은 이 아키텍처에서 예외 없이 HQNR 순손실이다.**
   T2 SiS · KD 6벌 · SE1 · SE2 전부 같은 방향. R4 의 D_s 0.0209 는 전 실행 최저이고
   무엇을 더해도 올라갔다.
2. **경량화는 손실이 아니라 이득이다.** 2.13M 이 3.77M~7M 어느 구성보다 HQNR 이 높다.
   "경량화 손실 회복" 프레임은 폐기 대상이다.
3. **PAN 은 공간 기준이라 흔들면 안 된다.** 정렬은 MS 조건 쪽 강건성으로만 다룬다.
4. **판정 지표와 참고 지표가 이 구간에서 분리돼 있다** — HQNR 최하위가 ERGAS·SCC 1위인
   사례가 실재한다([2026-09-04](2026-09-04_placement-and-band-invalidation.md)).

## 그림 자산

`assets/` 에 둔다. 학습 곡선은 다음으로 생성한다.

```bash
python ../tools/plot_metrics.py ../work_dir/wv3_baseline ../work_dir/wv3_fixed \
       --out assets/curve_wv3.png
```

matplotlib 에 한글 글리프가 없다 — **그림 라벨은 ASCII 로 쓴다.**
위성영상은 라이선스 제약이 있으므로 **외부 서비스 업로드 금지**.

## 공통 진단 도구

```bash
python tools/uncertainty_diag.py <run>             # uncertainty 계열 필수 — 5종 + 4패널 그림
python tools/check_calibration.py work_dir/<run>   # 게이트 판정만
python tools/analyze_se_gates.py work_dir/<run>    # SE 게이트 (mode-cos·포화)
python tools/plateau_report.py <run> ...           # best vs plateau
python gspread/refile_sheet.py --dry-run           # 시트 범주 재정리
```

`uncertainty_diag.py` 가 내는 것 (uncertainty 를 다루는 실험은 **반드시** 포함):
① Spearman/Pearson(θ,|e|) ② 조건부 오차 순서 `E[e|Top10]>Top20>Top30>E[e]>Bot10`
③ 10분위 error curve ④ risk–coverage(+oracle/random, AURC, excess)
⑤ θ vs GT 국소분산 상관(+ GT 분산 자체의 오차 예측력 — θ 의 경쟁 가설)
