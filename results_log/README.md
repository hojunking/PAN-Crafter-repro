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
| 2026-09-12 | s1 | [정합 축 종합 판정 — PAN 을 옮기는 접근은 성립하지 않는다](2026-09-12_s1_alignment-axis-verdict.md) | **네 캠페인 17벌(BASE96·PA·PO10·NF16) 결론: 정합량 |Δ| 과 HQNR 이 단조 반비례.** W112·D123 안에서 r(|Δ|,HQNR) = **−0.912**, 기울기 −0.0169/px. 두 영역뿐이다 — **|Δ|<0.4 px 는 아무 효과 없고**(PA 3벌·NF16 P2), **|Δ|>1.4 px 는 옮긴 만큼 잃는다**(−0.018~−0.034, 11~22σ). 정합이 이득인 구간은 관측되지 않았다. **결정적: P1(frozen)과 P2(fine-tune)는 같은 donor 에서 출발하는데, 자유 학습한 P2 가 |Δ| 를 1.511→0.344 로 77% 줄이고서야 HQNR 이 무정합 수준으로 회복됐다 — 최적화의 답이 '옮기지 말라'였다.** **출력을 −Δ 로 되돌리면 더 나빠진다**(0.933→0.892, D_λ 3.3배 폭발) — 저주파는 MS 에, 고주파는 warp 된 PAN 에 묶여 전역 shift 로 못 고친다. 근본 원인은 HQNR 의 두 성분이 1.79px 어긋난 프레임(D_λ↔LRMS, D_s↔원 PAN)을 기준삼는 것. **aligner 컨셉은 유지**하고 방향은 [PAN_Aligner_Directions](../research_log/PAN_Aligner_Directions_2026-09-12.md) 로 정리. **부산물: HQNR 시드 판정선을 처음 실측 — 2σ = 0.0031**(3 경로 일치; 종전 0.011 은 3.5배 과대). PAN 을 옮기는 축 종료, MS 조건 입력 쪽만 남긴다 |
| 2026-09-11 | s1·s2·s3 | [시트 정리 + HQNR 두 방식 + NF16·NA104 WIP](2026-09-11_sheet-cleanup-and-hqnr-views.md) | §4 WIP: **NA104(W104·D122 no-align KD, s2·s3, 88 case) 구현 완료·기동 대기** — 정합 모듈 없는 동일 골격에서 KD·출력 통계·방향 gate 분해, 주 selector best_rr_val. §3 WIP: **NF16(N2 last aligner 재사용 + native fitting, P0–P4, s1 16 GPU-h) 구현 완료·기동 대기**.  WV3 본 탭에 현 접근(REF·BASE·PA·PO10·KDV)만 남기고 나머지(s1 137·s2 50·s3 11행)를 `WV3-<server>_v1` 탭으로 이전; HQNR↑(전체 프레임) vs HQNR(V64)↑(가장자리 64 px 제외 고정 영역)의 차이 정리 |
| 2026-09-10 | s1·s2·s3 | [A1–A3 PAN 전역 정합 s1 block](2026-09-10_pa-a1-a3-s1-results.md) | §10 **PO10 결과: corruption 학습으로 aligner 가 x 축 −0.5 만큼 반응하고 FR 보정이 센서 어긋남 규모(1.5 px)로 커지지만, 출력이 MS 프레임으로 옮겨가 논문 프로토콜 HQNR(원 PAN 참조 D_s)은 수렴점에서 0.948 → 0.93 으로 후퇴; 시트 best 는 초기 checkpoint(ERGAS 3.8–4.1)라 품질값 아님; N2(SG) 최선, N3 과보정.** §10.6 HQNR(V64) 정의. §9 WIP: **s2 W112·D123 KD·variance·aligner 재사용 캠페인 구현 완료**(Q00–Q08, `trainer: kdv`, 기동 `tools/kdv_prepare.sh`). §8 WIP: PO10. **aligner 는 방향은 맞지만 입력 무반응 상수(0.2 px).** 세 case 모두 Δ̂=(+0.18~+0.23, −0.04~−0.07) HR px 로 부호는 센서 어긋남과 일치하고 warp 한 PAN 이 MS 격자에 실제로 가까워진다(1.79→1.36~1.40). 그러나 입력을 ±1px 옮겨도 Δ̂ 가 안 움직인다(기울기 −0.02~−0.09) — **학습 분포에 변위 변동이 없는 것**이 원인(합성 e~U(−2,2)² 에서는 corr 0.997 로 배운다). **HQNR 로는 B0 와 구분되지 않는다**(A1 −0.0024 · A2 +0.0013 · A3 +0.0015, B0 3seed sd 0.0017). A2 의 이득은 정합이 아니라 edge loss 정규화일 가능성 |
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
