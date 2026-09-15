# PAN-Crafter 재현 + 경량화 연구

PAN-Crafter (ICCV 2025) 저자 배포 코드를 재현하고, **프루닝 + KD(상호학습)로
손실을 회복하는 새 방법**의 베이스라인을 만드는 저장소다.
원본은 `upstream` remote (KAIST-VICLab/PAN-Crafter), 이 저장소는 fork 가 아니라
clone 에 작업을 얹은 것이다.

## 먼저 읽을 것

| 문서 | |
|---|---|
| `SETUP.md` | 새 서버 셋업 (경로 치환, 데이터 배치, 형제 저장소) |
| `KNOWN_ISSUES.md` | 논문 불일치·잠재 버그·데이터 결함과 적용 현황. **코드를 고치기 전에 반드시 확인** |
| `INTRO.md` | 논문 요지와 코드 구조 |
| `DISK_CLEANUP.md` | `work_dir` 정리로 디스크 회수 (`tools/prune_workdir.py`). **캠페인 시작 전 `df -h` 확인** |
| `results_log/README.md` | 실험 결과 색인 (최신순). **수치를 인용할 때는 여기부터** |
| `results_log/CONVENTION.md` | 실험 문서 작성 규약 |

## 환경

- conda env `pancrafter`. python 은 `/home/knuvi/miniconda3/envs/pancrafter/bin/python`
- GPU 1장. Teacher 50K 학습 ≈ 5h, 25K ≈ 2.3h
- 새 서버·클라우드는 Docker 가 가장 빠르다 — `hojunqueen/pancrafter-env:latest`
  (환경만 담겨 있고 코드는 마운트한다. 코드가 바뀌어도 이미지 재빌드 불필요)
- **지표 구현은 저장소 안에 있다** (`tools/metrics/`). CANConv 를 clone 할 필요 없다
- full-resolution(D_λ/D_s/HQNR)만 DLPan-Toolbox 의 `wald_utilities.py` 를 런타임 import 한다.
  GPL-3.0 이라 편입하지 않고 외부에 둔다 — `export PANCRAFTER_DLPAN=/path/to/DLPan-Toolbox`
  reduced 지표(SAM/ERGAS/Q2n/PSNR/SSIM/SCC)는 이것 없이도 전부 동작한다
- 이식 확인: `python tools/verify_metrics.py` (6개 지표 상대오차 0 이면 정상)

## 실행

```bash
./tools/run.sh <config이름>                              # config/pancrafter_<이름>.yaml 또는 config/<이름>.yaml
setsid nohup ./tools/run.sh wv3 > /dev/null 2>&1 &       # SSH 끊겨도 유지 (PPID=1 로 확인)
```

실행 조건은 `work_dir/<실험>/meta/` 에 자동 스냅샷된다.

**장애 대비가 걸려 있다** — cron 이 15분마다 `tools/_watchdog.sh` 로 체인 생존을 확인하고,
죽어 있으면 재기동한다(재부팅 후 @reboot 포함). 체인은 완료분을 건너뛰고 이어 돈다.
학습 실패 시 최신 `epoch-*` 체크포인트에서 1회 재개 재시도하며, **exit 3(NaN 손실)은
재시도하지 않는다.** 캠페인이 끝나 로그에 `[cases] DONE` 이 찍히면 감시자는 멈춘다 —
새 캠페인은 새 로그로 시작할 것. cron 해제: `crontab -l | grep -v PANCRAFTER-WATCHDOG | crontab -`

## 반드시 지킬 규약

- **`results_log/` 의 기존 문서를 고치지 않는다** (`CONVENTION.md` §5). 각 문서는 그 시점의
  스냅샷이다. 결론이 바뀌면 **새 날짜 문서**를 쓰고 "무엇이 왜 바뀌었는지" 를 남긴다.
  예외는 오타·깨진 링크·계산 실수뿐이다.
- **30분 넘는 실험은 WIP 문서를 먼저 만든다** (§2). 사람이 기다리는 동안 볼 수 있어야 한다.
- **모든 수치에 두 가지를 명시한다** — 어느 실행인가(`baseline` 배포본 그대로 / `fixed`
  A-1·A-2 적용), 누가 쟀는가(`py` 학습 중 metrics.csv / `matlab` DLPan 프로토콜).
  **논문 Table 과 비교 가능한 것은 `matlab` 뿐이다.** FR 은 거기에 **어느 세트인가**까지 —
  논문 비교는 `.mat` 20장(`fr_mat20`), H5 12-19 는 선택용이라 논문 표에 넣지 않는다.
- 새 문서를 만들면 `results_log/README.md` 맨 위에 한 행 추가한다.
- **구글시트 업로드에는 서버 식별자를 반드시 붙인다** (`gspread/server.txt` 에 `s1`/`s2`).
  두 서버가 같은 config 를 돌리면 실행명이 같아져, suffix 가 없으면 상대 서버 값을
  덮어쓴다. 시트 위에서 서버 간 수치 혼용이 일어나는 것이다.

## 확정된 사실 (다시 파헤치지 말 것)

- **재현 성립.** WV3 reduced ERGAS **2.1633** (배포본 그대로, 50K). 논문 2.040 대비 +6.09%
- **그 격차는 우리 잘못이 아니다.** 평가기는 CANConv 배포 가중치로 논문 행을 6지표 0.5% 이내
  재현하고, 논문 명시 설정은 시드 2,025 까지 전부 일치한다.
  → `results_log/2026-08-24_paper-rebuild-and-reproduction-audit.md`
- **논문의 CANConv 대비 우위는 재현되지 않는다** (주장 −5.69% vs 재현 −0.34%, p=0.667).
- **배포 코드는 논문이 기술한 모델이 아니다 — 재구성으로 확인했다.**
  논문 본문·Figure 3 대로 다시 구현하니 params 가 **7.1707 M** 으로 논문 주장 7.170 M 과
  **+0.01%** 로 맞았다(배포 코드 9.969 M). 되돌린 것 셋 다 논문 본문이다 —
  mode modulation 을 Eq (6) 의 직접 학습 γ,β 로(블록당 33,024→512), bottleneck 도 k=3
  (배포본만 k=1), 입력 9ch. 구조는 **3-scale / Down·Up 2 / AttnBlock 3**.
  → `model/pancrafter_paper.py`, `results_log/2026-08-24_paper-rebuild-and-reproduction-audit.md`
- **FLOPs 79.03 G 는 미해결.** 재구성본도 161.9 G 이고, 어텐션을 전부 빼도 125.9 G 다.
  "어텐션 미집계" 가설은 기각했다(배포 구조에서 79.2 G 가 나온 것은 무관한 우연).
- **지표 선택**: ERGAS·SAM 만 판별력이 있다. Q8·SSIM·SCC 는 이 범위에서 포화(±0.15%)라
  판별 근거로 인용하면 안 된다. D_s·HQNR 은 축소하면 거의 항상 좋아지는 기전이 있어 단독 해석 금지.
- **양방향 mutual learning 은 no-go** (`2026-08-20_submodule-sweep-and-mutual-nogo.md`).
  단방향 T→S 증류는 별개이고 유효하다.
- **논문 비교 FR 세트는 PanCollection `.mat` 형식 20장이다 — 배포 H5 의 20장이 아니다** (KNOWN_ISSUES F-2,
  `2026-09-07_alignment-shift-robust-and-metric-v2.md`). H5 12-19 는 그중 6장만 겹친다. **시트의 FR 은
  `results/fr_mat20.json`(`tools/eval_fr_paperset.py`) = FR·paper mat20 열뿐이다** (2026-09-07 사용자 결정).
  **2026-09-09 부터 학습 중 best 선택도 논문 세트 20장 전체다** (`fr_select_indices` 기본 `0-19`, FR feeder 는
  `full_examples_mat20`). 12-19 같은 부분집합은 어디에도 쓰지 않는다 — 논문 프로토콜을 따른다. 그 전에 H5 12-19 로
  고른 ARCH run 은 `*_sel1219` 로 격리돼 시트·배치에서 빠진다. 새 서버·다른 서버 재측정은 `./tools/metric_v2_prepare.sh` 한 번.
- **JQM(Palubinskas 2015, `tools/metrics/jqm.py`)은 추가 지표다** — 두 논문이 보고하지 않으며 판정 기준이 아니다(HQNR→SCC 유지).
  규약은 **SIPSA-Net 보충자료**(QLR 밴드 균등평균, QHR 은 SRF 가중 intensity, v=0.5)를 따르되 SRF 가 없어 **NNLS 정규화 대체
  가중치**를 쓴다 → 인용할 때 "SRF 대체(회귀) JQM 변형" 이라고 적고 SIPSA 보고값과 같은 조건이라 하지 않는다. 입력 [0,R] 클립·
  볼록 가중으로 [0,1] 을 보장한다(합이 제한되지 않은 회귀 가중치는 1 을 넘긴다). 시트 FR·paper 의 JQM↑ 열.
- **시트·보고용 SCC 는 SCC.m(zero-padding), SSIM 은 Gaussian 11×11** (KNOWN_ISSUES D-7). 2026-09-07 이전
  문서의 SCC 는 약 +0.004, SSIM 은 약 +0.002 높은 옛 정의다. 학습 로그의 SCC 는 여전히 옛 정의(상대 비교용).
- **평가기는 MATLAB 소스를 파이썬으로 재구현한 것이지 MATLAB 실행이 아니다.** 비트 동일을 주장하지 않는다.
  근거는 anchor 두 개(EXP·CANConv 배포 가중치)가 논문 값과 평균·N−1 표준편차까지 맞는 것. MTF 커널은
  DLPan 파이썬 포트(정규화)가 아니라 `genMTF.m` 충실 재구현을 쓴다 — 2026-09-07 이전 HQNR 보다 ~3e-4 낮다.
  PSNR·SSIM 은 DLPan 프로토콜 밖이라 관례 추정이다.

## 판정 규칙 — 시드 오차가 대부분의 차이를 삼킨다

동일 구성을 시드만 바꿔 돌린 폭이 **0.81%** 다(2.2527 vs 2.2344). 이는 지금까지 인용해온
차이 대부분보다 크다.

| 비교 | 차이 | 시드 폭 대비 |
|---|---:|---|
| 선정 Student vs Teacher | −0.31% (p=0.114) | 0.4배 |
| 6.041 M 경계 | +0.74% (p=0.013) | 0.9배 |
| A-1/A-2 적용 효과 | +0.74% (p=0.0014) | 0.9배 |

**대응표본 t-검정은 같은 가중치를 20장에 적용한 것이라 시드 변동을 포착하지 못한다.**
p 값이 작아도 시드를 바꾸면 뒤집힐 수 있다.

- **0.8% 미만의 차이는 시드 3개 이상에서 방향이 일관될 때만 주장한다.**
- 단일 시드 대응표본 p 값만으로 구조 차이를 결론짓지 않는다.
- 기존 결론들도 시드 σ 가 확정되면 **소급 재판정 대상**이다.

## 진행 중 — 경량화 case 스크리닝 (HQNR 선택)

명세: `research_log/lightweight_case_specs_v1.md` · 실행: `tools/_run_cases.sh`

- **재현은 완결됐다.** `s1_A1`(11ch·nocrop·LN) 이 ERGAS **2.0351** 로 논문 2.040 을 넘었다.
  격차의 최대 원인은 배포 코드의 `crop`(실은 scale jitter, −3.63%)이었다.
- **best 선택 기준이 HQNR 로 바뀌었다** (`select_on: hqnr`). FR 검증 split 이 없어 FR
  테스트셋으로 고른다 — no-reference 라 GT 누출은 없지만 선택 편향은 있다. 산출물은
  `best_hqnr` / `reduced_best_hqnr.mat` 이고, 체인 완료 판정도 이 파일이다. 2026-09-09 부터 그 FR 테스트셋은
  **논문 세트(.mat 20장) 전체**다.
- 학습 로그에 매 eval epoch `[핵심] HQNR / SCC / ERGAS` 가 찍힌다.
- **지표 우선순위: HQNR > SCC > ERGAS.** best 선택은 HQNR(논문 세트 20장 전체; 2026-09-09 이전 run 은 H5 12-19), 동률이면
  SCC, 그 다음 ERGAS 로 가른다. 단 SCC 는 이 실험 범위에서 포화(0.9887~0.9914)라
  실질 tie-break 는 대부분 ERGAS 가 맡는다.
- **HQNR 시드 2σ ≈ 1.18%** (ERGAS 0.11%). HQNR 차이가 이보다 작으면 위 순위의
  다음 지표로 보조 판정.
- 실험 case 를 대화에서 W1/W2 처럼 부르더라도 **시트·config·문서·보고서에는 약명 단독으로
  쓰지 않는다.** 나중에 의미를 알 수 없다. 항상 서술형(c8_c4w96, "attn:0 w96 nocrop")으로
  남기고, 보고서 표에서 축약이 필요하면 **같은 문서 안에 약명→세팅 대응을 반드시 둔다.**
- s2 는 동일 config·동일 seed 로 같은 실험을 돌려 환경 변경을 검증한다 (명세 §5).

**경량화 축이 바뀐다.** 배포 코드에선 CM3A 제거가 공짜였지만 재구성본엔 그 여지가 없다
(AttnBlock 3개가 전부 mid/low 해상도에 있고, 무손실로 뺐던 것들이 애초에 없다).

```
배포 코드 :  CM3A 개수 -> PAN 브랜치 -> depth -> width
재구성본  :  width  >>  full-res depth  >  AttnBlock 개수  >  bottleneck
```

## 함정 (전부 한 번씩 당한 것)

- **`zero_module` 트랩**: 모델 초기 출력이 정확히 0 이라, 코드 동등성 테스트가 무엇을 넣어도
  통과한다. **0 인 파라미터를 난수화한 뒤** 비교해야 유효하다.
- **`best_full` 체크포인트는 퇴화본**이다 (D_s 기준 → epoch 5 선택, ERGAS 2.86).
  비교용 mat 은 반드시 이름으로 명시한다. 정렬로 고르면 이게 먼저 잡힌다.
- **배포 `pan_h5.zip` 의 WV3·QB full-res `lpan` 이 다른 장면이다** (F-1).
  `tools/repair_lpan.py` 로 복구하지 않으면 full-res 평가가 무효다.
- **PanCollection QB 학습·검증셋의 `ms` 는 2/3 패치가 `gt` 와 LR 1px 어긋나 있다** (F-3). `train_qb.h5` 로 학습한
  QB 결과는 무효 — `tools/repair_qb_ms.py` 가 만든 `*_msfix.h5` 를 쓴다. WV3·GF2·QB 테스트셋은 정상.
- **`pkill -f <패턴>` 이 자기 자신을 잡는다.** `ps -eo pid,args` 로 PID 를 골라 죽인다.
  `pgrep -f` 도 같다 — 확인 명령줄에 패턴 텍스트가 있으면 그 셸이 잡혀 "이미 실행 중" 이 거짓으로
  뜬다(`campaign_start.sh` 가 그렇게 한 번 거부됐다). `grep '[_]run_cases'` 처럼 bracket 으로 피한다.
- **`tail -f log | grep`** 은 마지막 10줄부터 시작해 아무것도 안 나온다.
  `tail -n +1 -f ... | grep --line-buffered` 를 쓴다.
- **`${1:?메시지}` 안에 `}` 를 넣지 않는다.** 파라미터 확장이 끊겨 인자가 오염된다.
- **`run.sh` 의 `trap ... EXIT` 는 실패해도 `finished_at.txt` 를 쓴다.**
  체인의 완료 판정은 `results/reduced_best_val.mat` 존재로 한다.
- matplotlib 에 한글 글리프가 없다. 그림 라벨은 ASCII 로 쓴다.
- 위성영상은 라이선스 제약이 있다. **외부 서비스에 업로드하지 않는다** (§6).

## 현재 진행 상황

`results_log/README.md` 최상단과 최신 `*_WIP_*.md` 를 보면 된다.
진행 중인 체인은 `ps -eo pid,ppid,args | grep _run_` 으로 확인한다.

2026-09-09 19:58 부터 s1 은 **새 baseline WV3 3-seed**(`config/queues/base_w96_d124_mspan_wv3_3seed.txt` 큐, 아래 결정 참조)를 돌린다. 그 전 캠페인은 중지 —
GF2 ×3 완료, QB S2025 중단(체크포인트 재개 가능), 6벌 미실행. 재개는 `campaign_start.sh --queue config/queues/arch_w168_multiset_3seed.txt`.

2026-09-08 부터: **아키텍처 고정(S1_T05_W168_D123_DUAL) 다중 데이터셋 3-seed** — WV3/QB/GF2 × seed 2025·1234·7777 +
WV2 zero-shot, 서버 3대가 같은 큐(`config/queues/arch_w168_multiset_3seed.txt`). 준비·기동은 `./tools/arch_multiset_prepare.sh`,
계획은 `research_log/2026-09-08_arch-w168-multiset-3seed-plan.md`. config 는 `tools/gen_arch_multiset_configs.py` 가 만든다.

2026-09-09 결정: **새 mainline 은 W96·D124 U-Net · MS+PAN 9ch · 단일 HRMS task(PAN 재구성·dual MARs·LPAN/HPAN 제거)** —
`research_log/PAN_research_baseline_W96_D124_2026-09-09.md`. 과거 W168·d123 dual 은 직접 대조군이 아니다. 첫 실행은 WV3 3-seed(s3):
config `BASE_W96_D124_MSPAN_WV3_S*`(`tools/gen_w96_d124_mspan_configs.py`), 기동 `./tools/base_w96_prepare.sh`,
준비 문서 `research_log/2026-09-09_w96-d124-mspan-wv3-3seed-launch.md`.

그 다음 캠페인(2026-09-09 결정): **A1–A3 PAN 앞단 전역 정합** — 명세 `research_log/PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md`,
구현 `pa/` + `train_pa.py`(trainer: pa), 검토·구현 노트 `research_log/2026-09-09_pa-a1-a3-implementation.md`. 서버-seed block: s1 2025 · s2 1234 · s3 7777,
각 서버 `./tools/pa_prepare.sh`(gate `tools/pa_unit_tests.py` 포함). `best_hqnr/` 는 best_raw(raw_original HQNR) 의 alias, `best_aligned/`·`last/` 별도.

2026-09-10 s1 결과(`results_log/2026-09-10_pa-a1-a3-s1-results.md`): aligner 는 방향은 맞지만 입력 무반응 상수(0.2 px). 후속 **PO10**(PAN 추가 변위 + offset consistency,
명세 `research_log/PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md`, 구현 `pa/offset.py` + `train_po.py`(trainer: po), 노트 `research_log/2026-09-10_po10-implementation.md`):
s1 seed 2025, 예산 ledger `work_dir/_po10_budget/ledger.json`. **2026-09-10 오후 변경**(`research_log/PAN_OffsetConsistency_ChangeNote_R100_to_R200_2026-09-10.md`):
R100(b=1.0, W96·D124)은 N1 만 기록으로 보존, 다음 실험부터 **R200(b=2.0, FR 입력 통계 참고) + 골격 W112·D123** — run `PO10_*_W112_D123_WV3_S2025_R200_FRSTAT`,
큐 `config/queues/po10_s1_r200_frstat_w112_d123.txt`, 생성 `tools/gen_po10_configs.py --radius 2.0 --width 112 --depth 1,2,3`. 골격이 달라 B0/A1/R100 과 직접 대응하지 않는다.
시트 FR·paper 에 **HQNR↑(전체 프레임, 논문 프로토콜)** 과 **HQNR(V64)↑(가장자리 64px 제외)** 두 열 — 비교표는 HQNR↑. evaluator 2026-09-10.5.

2026-09-10 저녁: **s2 캠페인 — W112·D123 GT-anchored adaptive KD · 출력 통계 variance · aligner 재사용** (계획 `research_log/PAN_S2_W112_KD_Variance_Plan_and_References_2026-09-10/`,
검토·구현 노트 `research_log/2026-09-10_s2-w112-kdv-implementation.md`). 구현 `kdv/` + `train_kdv.py`(trainer: kdv), config `config/S2W112D123_*.yaml`(`tools/gen_kdv_configs.py`, depth 1,2,3),
큐 `config/queues/kdv_s2.txt`(Q00 baseline → Q01 Teacher seed 2025 → Q02–Q08 Student seed 1234), 기동 `./tools/kdv_prepare.sh`(gate `tools/kdv_unit_tests.py` 포함), 시트 범주 ⑳ KDV.
donor aligner 는 `assets/donor_aligner/`(s1 PA_A1 seed 2025 의 aligner.*, strict load). 이름 규칙 `S2W112D123_<recipe>_<input>_<aligner>_<rec>_<stat>_<geomKD>_s<seed>_<ver>` —
약명은 구현 노트 §4 표로만 읽는다. **depth 는 사용자 결정으로 [1,2,3]**(계획 원안 [1,2,4]; s1 PO10 R200 과 같은 골격 2.6589 M). Teacher seed 2025 · Student 1234. **이동량 covariance KD G1–G5·G-STRUCT 는 구현됨**(`kdv/alignment_kd.py`, 출처 eq_closure/geo_curvature/struct, 구현 노트 §9) —
실행은 보류 큐 `config/queues/kdv_s2_geomkd.txt`(`gen_kdv_configs.py --geomkd`): PO10 N2/N3(반응하는 aligner) 와 Q07/Q08 결과 뒤에 기동한다. 현재 A1 donor 는 무반응이라 G-EQ precision 이 낮다.
**TRI-A/B/C**(addendum `research_log/PAN_S2_W112_D124_TGeo_ABC_Addendum_2026-09-10.md`: R3 soft 의 band/통계 성분 방향 gate, Teacher 출력의 correction 민감도 감쇠) 도 구현(`kdv/tri.py`, 구현 노트 §10) —
큐 `config/queues/kdv_s2_triabc.txt`(`--triabc`, 22 run, P1 → MASS/SHUFFLE/GV-WH 대조 → P2/P3), 본 큐 뒤 기동. 이름 토큰 `TRI_A*_B*_C*`. C-DIAG(EQ) 는 현 donor 에서 C-SENS 와 정보가 같다(등방 Σ).

2026-09-11 **시트 정리**: WV3 본 탭(WV3-s1/s2/s3(5090))에는 현 접근 범주만(REF · BASE_W*_MSPAN · PA · PO10 · KDV) 남긴다. 나머지 범주는 `WV3-<server>_v1` 탭 맨 아래로 옮겼다(`gspread/archive_to_v1.py`, 백업 `gspread/_sheet_backup/*.before_archive_2026-09-11.json`).
`gspread_upload.py --all` 은 옮긴 범주(`sheet_categories.ARCHIVED`)를 다시 올리지 않는다. 시트의 HQNR↑ = 전체 프레임(논문 프로토콜), HQNR(V64)↑ = 가장자리 64 px 제외 고정 영역 — `results_log/2026-09-11_sheet-cleanup-and-hqnr-views.md`.

2026-09-11 **NF16** (`research_log/PAN_N2_NativeFitting_16GPUh_W112_D123_2026-09-11.md`, 노트 `research_log/2026-09-11_nf16-implementation.md`): N2 R200 `last` aligner 재사용 + native fitting P0–P4, s1, 16 GPU-h.
구현은 KDV trainer 위 — 새 프로토콜 `I-AEQ`(복원은 매 update native, 홀수 update 에 P_ε 를 aligner 에만), `kdv.aligner_lr`, donor `expected_step`, 고정 donor 참조 view `aligned_fixed_v64`, 예산 gate(`kdv.budget`). config `NF16_P{0..4}_W112_D123_WV3_S{1234,7777}_N2LAST_v1`(`tools/gen_nf16_configs.py`),
큐 `config/queues/nf16_s1.txt`, gate `tools/nf16_unit_tests.py`, 기동 `./tools/nf16_prepare.sh`. 진단은 `po10_diag.py --ckpt last --native-reference`(참조 = 원 P / 고정 donor). 시트 범주 ㉑ NF16.

2026-09-11 **NA104** (`research_log/PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md`, 노트 `research_log/2026-09-11_na104-implementation.md`): **s2·s3 두 서버**의 새 캠페인 —
**정합 모듈이 전혀 없는** W104·depth[1,2,2](2.0989 M) 동일 골격에서 GT-anchored KD(REC N0/R0/R1/R2/R3) · 출력 통계(IV/GV/GC/SC/**M2** × H/T/FIX/WH/AD) · 방향 gate(TRI-A/B) · 필수 대조군(CTL) 을 87 case 로 분해한다.
구현은 기존 kdv trainer 위 — 새 키 `na_protocol`(NA-STRICT: 학습 forward 에 PAN warp 금지 / NA-TSENS: Teacher 입력 민감도 probe 만 별도 cohort), `rec.control`(hscale·rshuffle), `stat.transform/domain/windows/extra`, `select.primary`, `teacher.eval_only`.
config `NA104_*`(`tools/gen_na104_configs.py`), 큐 `config/queues/na104_s2.txt`·`na104_s3.txt`, gate `tools/na104_unit_tests.py`, 기동 `./tools/na104_prepare.sh`, 시트 범주 ㉒ NA104.
**주 selector 는 best_hqnr**(저장소 확정 지시; best_rr_val·last 는 보조), aligned view/selector 는 만들지 않는다(aligner 가 없으면 raw_valid 와 같다). 정합 축(PA/PO10/NF16/KDV) 과 직접 대응하지 않는 별개 골격이다.
**2026-09-12 FINAL 계획**(`research_log/01_S2_FINAL_EXPERIMENT_PLAN.md`·`02_S3_FINAL_EXPERIMENT_PLAN.md`, 노트 §10): 보류 case 전수 편성 + 신규 18 정의(X01–X12·PX·CX·LX, 통계 모드 HAD/WFIX/TMATCH) + core10×seed 777·2026 반복. 이름 규칙 기존 seed1234=v1·신규/추가 seed=v2, **Teacher(T00 S2025 v1/best_hqnr)·λ pilot(Q00 S1234 v1/last) 고정**. 큐 `na104_s2.txt`(87)·`na104_s3.txt`(97) + `na104_<srv>_stage_plan.json`(직접 대조). COSTMATCH 는 실측 후.

2026-09-12 **PALS24** (`research_log/PAN_P2_P3_LambdaSweep_MetricAware_24GPUh_Plan_2026-09-12_v2.md`, 노트 `research_log/2026-09-12_pals24-implementation.md`): s1 의 다음 캠페인 — NF16 P2(λ_off 0)/P3(λ_off 0.01) recipe 에서 **λ_off 만 0.0001/0.001/0.003** 으로 (seed 1234 탐색 A1–A3), 규칙(raw best HQNR → fSCC → 더 작은 λ)으로 λ* 를 한 번 고정한 뒤 seed 7777·2025 에서 **CTRL-P0 · L000(λ 0) · λ*** 를 대응 비교. 24 GPU-h 상한(`work_dir/_pals24_budget/ledger.json`, gate margin 1.1 · reserve 4.0 · block(P0·L000·λ*) 완결 검사 · 초과면 DEFERRED).
config `PALS24_<case>_W112_D123_WV3_S<seed>_N2LAST_R200_v1`(`tools/gen_pals24_configs.py`; 약명→세팅은 노트 §1 표), 큐 `config/queues/pals24_s1.txt`(A1–A3; stage 2 는 `tools/campaign_gate.py` 의 `pals24` gate 가 `work_dir/campaign_gates_enabled.txt` 로 연다 — 캠페인 뒤 이 파일을 지울 것), gate `tools/pals24_unit_tests.py` + `tools/pals24_metric_gate.py`(G-M0–G-M8, 재사용 registry), 기동 `./tools/pals24_prepare.sh`, 집계 `tools/pals24_report.py`, 시트 범주 ㉓ PALS24.
seed 1234 의 CTRL-P0/L000/L1E2 는 NF16 P0/P2/P3 를 재사용한다(다시 학습하지 않음; P0 는 eval_epoch 5 라 10 격자 재선택값 병기). 주 판정은 best_raw raw_original HQNR → fSCC(원 PAN 참조; RR SCC 아님), 판정선 0.0031 은 raw HQNR 에만 — aligned/V64/last 는 진단.

2026-09-13 **PALSV18** (`research_log/PAN_L1E4_Refinement_AlignmentValidation_S1_18GPUh_2026-09-13.md`, 노트 `research_log/2026-09-13_palsv18-implementation.md`): PALS24 결과(λ* 1e-4, 3 seed 양성 — `results_log/2026-09-13_s1_pals24-lambda-sweep.md`) 의 후속. 같은 recipe 에서 **λ_off 3e-5(L3E5)/3e-4(L3E4) × seed 1234·7777·2025** 6벌(50K) + 정합 검증 V0–V4. 18 GPU-h(`work_dir/_palsv18_budget/ledger.json`, seed pair 단위 gate, reserve 5.0).
config `PALSV18_<case>_…_v1`(`tools/gen_palsv18_configs.py`, PALS24 생성기 재사용), 큐 `config/queues/palsv18_s1.txt`(조건부 gate 없음), gate `tools/palsv18_unit_tests.py` + `tools/pals24_metric_gate.py --campaign palsv18`, 기동 `./tools/palsv18_prepare.sh`, 검증 `./tools/palsv18_validate.sh pre|post`(`tools/po10_diag.py --probe-set palsv18` + `tools/palsv18_validate.py` V2–V4), 집계 `tools/palsv18_report.py`, 시트 범주 ㉔ PALSV18. 대조군 P0/L000/L1E4 × 3 seed 는 NF16/PALS24 재사용(`work_dir/_palsv18_campaign/reuse_registry.json`). 판정선 안이면 L1E4 를 working reference 로 유지한다.

2026-09-13 **NA104 20H 우선순위**(`research_log/01_S2_20H_PRIORITY.md`·`02_S3_20H_PRIORITY.md`, 노트 `research_log/2026-09-13_na104-20h-implementation.md`): s2·s3 의 NA104 를 **Q36(N0+GC-H)·Q12(R3+EDGE-H)·CF01(N0+GC-FIX, λ_C = Q36 재사용)** 3 후보로 좁힌다. 큐 `config/queues/na104_20h_{s2,s3}.txt`(P1 4벌: Q36-777 → Q00-2026 → Q36-2026 → Q12-2026), 조건부 CF01/X02 는 `tools/campaign_gate.py` 의 `na104_20h` gate(3-seed common-grid 기준 + 20h 예산). 전환은 각 서버에서 `./tools/na104_20h_switch.sh`(현재 run 은 원 설정으로 마무리), 회신은 `python tools/na104_20h.py report`. 새 키 `stat.lambda_from_run`(다른 run 의 λ_V 를 그대로). s2 의 CF01 은 s3 pilot 확인 token(`work_dir/_na104_20h/cf01_approved_by_s3.txt`) 이 있어야 열린다.

2026-09-14 **PAKD50** (`research_log/PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md` + `PAN_Integrated_Method_Summary_2026-09-14.md`, 노트 `research_log/2026-09-14_pakd50-implementation.md`): **통합 캠페인** — Teacher T0(PALS24 L1E4 seed 2025 best_raw 의 A+U, `assets/pakd50/T0_run`) 의 aligner 를 Student 가 복사(J: 공동 적응 / F: frozen)하고 U-Net 은 새로 학습, backend N0/R1/Q12/X02 = 기존 `rec N0/R1/R3(+stat EDGE-H)`. 세 서버 병렬 50h(서버당 학습 46h + 감사 4h), seed s1 1234 · s2 777 · s3 2026, 목표 raw HQNR ≥ 0.959.
config `PAKD50_<case>_W112_D123_WV3_T0_S<seed>_FRESH50_v1`(`tools/gen_pakd50_configs.py`), 큐 `config/queues/pakd50_<srv>_stage1.txt` 는 **J0 만** — 나머지는 매 pass gate `pakd50` 가 우선순위 J0→JQ→F0→FQ→JR→FR→XJ 로 편성한다(`gen_pakd50_configs.schedule`; λE 없으면 τR-only 한 벌씩; s1 은 J0-1234 뒤 λE 고정 → `assets/pakd50/calibration_resolved.json` 사본으로 s2/s3 전달, pull 만으로 다음 pass 에 JQ 가 열린다). 공통 절대 시계 `assets/pakd50/campaign_clock.json`(학습 마감 09-16 11:22:31; trainer `kdv.budget.training_deadline`·gate admission·체인 마감이 같은 시각), 돌던 체인 재편성은 `tools/pakd50_requeue.sh`. 구현 감사 `research_log/PAN_Integrated_Implementation_Experiment_Audit_2026-09-14.md` 의 F01–F05 반영(노트 §6), F06–F10(exact resume·worktree·A gradient 분해·bin 자료원·package pin) 은 후속. τR/λE 는 `tools/pakd50_calibrate.py`(고정값 `assets/pakd50/calibration_resolved.json`), gate `tools/pakd50_unit_tests.py`, 기동 `./tools/pakd50_prepare.sh`. 후보 격자는 `GRID1010_50K_v1`(eval_epoch 5, 50 후보 전부 보존). C1/C2(routing P/JK0/JE0, D/LF, TCOPY/CONT) 는 미구현.
**s4**(배정 `research_log/PAN_S4_Integrated_Experiment_Cases_2026-09-14.md`, 노트 `research_log/2026-09-14_pakd50-s4-review-and-implementation.md`): seed 1234 서버 교차(J0/JQ, 독립 seed 아님) → AL0/ALQ(A LR 3e-6) → 진단으로 고른 Q12 scalar ≤2(`J_QA05/J_QB005/J_QB02/J_QE025/J_QE10`, λE 는 λE0 배율) → 결합 ≤1 → seed 3407 확인. 기본 묶음은 gate 가 `PRIORITY_BY_SERVER["s4"]` 로, 추가는 `work_dir/_pakd50/extra_priority.txt`(case id 또는 run 이름). T0 경로는 모든 서버에서 `assets/pakd50/T0_run`. 시트는 `gspread/server.txt`=s4 → `WV3-s4` 탭 자동, 범주 ㉕ PAKD50, X열 `통합실험`(PAKD50 / case / FRESH50).
**2026-09-14 22:40 s1 의 PAKD50 은 사용자 지시로 중단**(J0/JQ/F0/FQ-1234 완료, JR-1234 는 끝까지; s2–s5 는 계속). s1 은 **EQREC4-S1-v1**(계획 `research_log/PAN_S1_EQREC4_Alignment_Cue_Hypotheses_20h_2026-09-14.md`, 노트 `research_log/2026-09-14_eqrec4-implementation.md`): frozen checkpoint 의 sample 별 native error e 와 offset consistency q 의 4분면 atlas(D10) → 상대/절대/cross-modal 정합 검사(D20) → correction 치환·native proxy·landscape(D30) → stress·PAN 민감도·edge(D40) → gradient·one-step(D50) → hard/soft microtrial K10 → 조건부 gate pilot K20 → report. 구현 `tools/eqrec4/`(CLI `tools/eqrec4.py <stage>`, runner `tools/eqrec4_run.sh`, gate `tools/eqrec4_unit_tests.py`), 출력 `work_dir/_eqrec4_<server>_campaign/`(최종 `report_EQREC4.md`). **2026-09-15 s3 에서도 실행**(사용자 결정): registry checkpoint 는 s1 에서 `python tools/eqrec4_bundle.py pack` → `work_dir/_eqrec4_bundle/`(272 MB, git 밖) 을 옮긴 뒤 s3 에서 `./tools/eqrec4_prepare.sh`(verify/install → gate → G00 → 기동). run config 의 s1 절대경로는 `common.localize_cfg` 가 처리. 판정은 within-checkpoint·대응 개입·source-block bootstrap; L1E4 채택과 q gate 채택은 별개 결정. **결과(2026-09-15 03:00, `results_log/2026-09-15_s1_eqrec4-results.md`)**: q 는 patch 정합 품질의 표지가 아니다(H1 native·H2 3 seed 반대, FR scene 수준만 양) · learned correction > zero(3/3) · Student cue 로는 판정 불가, 5K pilot 은 모든 arm 에서 HQNR 하락 → **q_T Teacher-quality gate 채택 안 함**.
**s5**(배정 `research_log/PAN_S5_Timing_Routing_Experiment_Plan_2026-09-14.md`, 노트 `research_log/2026-09-14_pakd50-s5-review-and-implementation.md`): aligner 의 **업데이트 시점·loss 수신 경로** — seed 2026(s3 교차) 로 J0 → JQ → D0(0–4999 A 동결) → DQ → PQ(A 는 L0+LO 만) 뒤 조건부 ≤2(LF0/LFQ 25K 뒤 동결 · JK0/JE0 · DPQ · soft-off DX/PX) → seed 9091 확인. trainer 새 키 `kdv.aligner_schedule{freeze_until,freeze_from}`·`kdv.routing{qD,qK,qE}`(registry 가 미지원 조합 거부; 기본값이면 J 와 동일), 검사 K10–K12(실제 `_step` 의 gradient 수식 대조). 시트 `WV3-s5`.

2026-09-15 **PAKD50 재배정 (s2/s4/s5)** (`research_log/PAN_PAKD50_S2_S4_S5_Derived_Run_Allocation_2026-09-15.md`, 노트 `research_log/2026-09-15_pakd50-derived-allocation-implementation.md`): 기존 결과에서 파생된 **명시 순서** — s2(777) JR→XJ→J_R3_NOEDGE→J_N0_EDGE · s4(1234) F0→RC0→RCQ→JR→XJ→J_R3_NOEDGE · s5(2026) PQ→F0→LF0→LFQ→RC0→RCQ (16 run; 완료된 J0/JQ/F0/FQ/AL0/ALQ/D0/DQ 는 재편성 안 함, s1/s3 신규 배정 없음). 새 case: `J_R3_NOEDGE` = JQ 에서 GT edge 제거 · `J_N0_EDGE` = GT L1 + λE edge(Teacher 미사용) · `RC0/RCQ` = A trainable(LR 1e-5) 인데 **Student 단계 offset 연습 없음**(정책 RC: `I-NATIVE-TRANSFER`, radius 0, offset 0 — I-AEQ 에 offset 0 은 registry 가 거부) + N0/Q12.
slot 예약 `reservation_h = 1.10 × reference_train_h + 10/60`(기준값 = 같은 서버 완료 case 의 Sheet Train(h) 대용값, PQ·확인 seed 는 1.80 h 가예약; 실측 아님) 을 gate 편성 admission 과 trainer 예산 gate(서버 로컬 `work_dir/_pakd50/reservations.json` → `kdv.budget.projection_file`) 가 같이 쓴다. 전환은 각 서버에서 `./tools/pakd50_reallocate.sh`(runner 만 교체 → 현재 학습이 끝난 뒤 gate 가 새 순서; `--plan` dry-run 은 `gen_pakd50_configs.py --plan`), 확인 seed(s4 3407 / s5 9091, ≤3 run) 는 외부 WIN 뒤 `--confirm <WIN>`. 검사 K17–K20.

2026-09-15 **DCR12 (s1 → s2; 12:34 교체)** (`research_log/PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md`, 노트 `research_log/2026-09-15_dcr12-implementation.md`): PAKD50 Teacher/Student 에서 **offset consistency C × native 복원 R 의 네 사분면**(D01) → correction 치환 개입(D02) → Student A 의 task/offset/soft gradient 충돌(D03) → **B0/FQ ↔ B1/JK0 두 seed(1234·777) 같은 호스트 대응 학습** → 조건부 micro-update utility(D04, §9.2 gate) → 보고서. 구현 `tools/dcr12/`(CLI `tools/dcr12.py`, runner `tools/dcr12_run.sh`, gate `tools/dcr12_unit_tests.py` X01–X13), 출력 `work_dir/_dcr12_s1_campaign/`(§13 산출물 + `report.md`). B0-1234 는 s1 PAKD50 FQ 완료본 재사용, JK0-1234·FQ-777·JK0-777 은 큐 `config/queues/dcr12_s1.txt` 로 새 학습(v1 이름 그대로; s1 gate `pakd50` 는 비움). B2/B3/CMASS/B1-NOOFF 는 미구현(gate 통과 시 별도).
**2026-09-15 12:34 서버 교체(사용자 결정)**: s1 은 JK0 S1234 까지만 학습하고 멈췄다(12:05 완료; FQ/JK0 S777 은 시작 안 함) — 나머지는 **s2** 가 한다: s2 자기 PAKD50 FQ S777 완료본(B0) + 새 JK0 S777(큐 `config/queues/dcr12_s2.txt`), seed 1234 pair(FQ/JK0, s1 학습) 는 **bundle**(`work_dir/_dcr12_bundle/`, 127 MB, git 밖 — `python tools/dcr12_bundle.py pack/verify/install`; results .mat 은 sha 증거만, 완료 판정은 provenance) 로 옮긴다. s2 절차: `git pull` → `rsync -a s1:/home/knuvi/Desktop/song/PAN-Crafter/work_dir/_dcr12_bundle/ work_dir/_dcr12_bundle/` → `./tools/dcr12_prepare.sh`(자산·bundle install·gate X01–X15·마감 여유 검사 뒤 runner detached; **s2 의 chain 이 아직 돌면 끝난 뒤 자동 기동**, 출력 `work_dir/_dcr12_s2_campaign/`, gate 'pakd50' 는 runner 가 비움). bundle 을 나중에 넣으면 `./tools/dcr12_prepare.sh --post`. runner 는 서버 공용(`config/queues/dcr12_<server>.txt`); seed 간 학습 호스트가 다르면 REPORT 가 `SEED_HOST_COUPLED`(계획 §8.2 host A/B 배치) 를 붙인다. JK0 S777 config 의 `training_deadline`(09-16 11:22:31) 은 그대로라 그 뒤 기동은 노트 §5 의 처리. 노트 `research_log/2026-09-15_dcr12-implementation.md` §5.

2026-09-15 **PAKD50 s3 추가** (`research_log/PAN_PAKD50_Latest_Sheet_Analysis_and_S3_Experiments_2026-09-15.md`, 노트 `research_log/2026-09-15_pakd50-s3-additions-implementation.md`): s3(seed 2026) 명시 순서 **J_R3_NOEDGE → J_N0_EDGE → LF0 → LFQ → LFX**(새 case: LF 일정 + X02 = 25K 뒤 A 동결·soft 없음; control LF0), 예약 합 7.99 h; 기존 J0/JQ/JR/XJ/F0/FQ/FR 은 control 재사용. 확인 seed **4321** 은 후보별 묶음(§5 표, ≤3 run, reference JQ 1.34 h) — `./tools/pakd50_reallocate.sh --confirm <후보>`. 전환은 s3 에서 `./tools/pakd50_reallocate.sh`. 검사 K21. LF 계열의 처음 25K 는 대응 recipe 와 정의·RNG 가 같지만 GPU 학습은 run-to-run 재현이 아니다(s4 J0 v1/v2 0.0026 차).

2026-09-15 **PAKD50 s4 골격 이식** (`research_log/PAN_PAKD50_S4_W104D121_Architecture_Allocation_2026-09-15.md`, 노트 `research_log/2026-09-15_pakd50-s4-w104d121-implementation.md`): Student U 만 **W104·depth[1,2,1]**(backbone 1.9036 M = s3 GT-only 기록과 같은 template lineage), Teacher T0/A·τR·λE0 는 W112 그대로(branch `A104D121_T0FIX_E0_v1`, `kdv.teacher.bridge: true` 로 폭 불일치 검사를 명시적으로 푼다). s4 명시 순서 **NA0 → J0 → JQ → XJ → F0** `@W104_D121`(NA0 = A-ID/NOALIGN plain GT; 예약 합 7.75 h), 확인 seed 3407 = J0/WIN × 두 골격(≤4). generator 는 골격 인지형: 편성 항목·`--cases` 는 `case@arch`, run 이름 `PAKD50_<case>_<W…_D…>_…`, init hash namespace `unet@W104_D121`, 예약·실측 키 `case@arch`, 시트 X열 `PAKD50 / <case> / A104D121 / FRESH50`. 검사 K22(92 ALL OK).

2026-09-15 **SMEC12 (s1; 원안 s2 → 12:34 교체)** (`research_log/PAN_SMEC12_MultiDataset_SampleMechanism_ExperimentPlan_2026-09-15.md`, 노트 `research_log/2026-09-15_smec12-implementation.md`): "q 는 낮은데 복원이 어려운 sample" 의 특성·기전을 **WV3·QB·GF2(+WV2 zero-shot)** 에서 검증. (a) QB/GF2 는 W112·D123 모델이 없어 **준비 학습 10 run**(P0 → DON-N2 → L000 → L1E4 → L1E4-REP × 2 센서; `tools/gen_smec12_bootstrap.py`, 큐 `config/queues/smec12_<server>.txt`, 기동 `tools/smec12_prepare.sh` — 현재 chain 이 끝나면 자동). **s1 에서 12:34 기동**(WV3 lane 자산 PALS24/NF16/PO10 이 s1 에만 있어서; chain 마감 09-17 04:34, 10 run ≈ 23–26 h; 분석 runner 는 chain DONE 뒤 `work_dir/_smec12_s1_campaign/analyze_when_done.sh` 가 자동, WV3+WV2 검증 결과는 노트 §5). s2 는 SMEC12 를 돌리지 않는다(DCR12 로) — WV3 recipe(NF16 P0 / PO10 N2 R200 / PALS24 L000·L1E4) 의 공통 이식, 4-band backbone 2.6508 M, 센서별 init_dir, exact25K/50K state. (b) 분석 backbone `tools/smec12/`(A00 · D10/D11 · I20 · I23-B · I24-A · I25-A · X40 · REPORT; CLI `tools/smec12.py`, runner `tools/smec12_run.sh`, gate `tools/smec12_unit_tests.py`) 는 있는 자산만 증분 처리하고 미구현 stage(D12·I21·I22·I23-A/C·I24-B/C·I25-B·S30·C50·R60) 는 `pending_compute` 로 보고. 시트 범주 ㉖ SMEC12.

2026-09-15 **QEDGE9 (s5·s4)** (`research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md`, 노트 `research_log/2026-09-15_qedge9-implementation.md`): W104·D121 Student(T0 고정) 에서 **q 기반 GT edge gate** — 새 case `QE50`(Q12 hard/soft 그대로, GT edge 는 고정 T0 aligner 의 offset-consistency q(AXIS16 probe) < θq(train calibration 중앙값 0.327613) 인 patch 만: λE·Σ g_i E_i / B, 재정규화 없음) · `QEC`(모든 patch edge × c_E = Σ g E_pilot / Σ E_pilot, pilot = s4 W104 J0 S1234 exact50K — **s4 에서 산출**) · `QES`(gate 를 T0 e_roi32 decile × aug state stratum 안에서 permutation 51515). 새 캠페인 `QEDGE9_A104D121_20260915_v1` / branch `A104D121_T0FIX_QEDGE9_v1` — **PAKD50 의 50h·09-16 11:22 마감을 상속하지 않는다**(`kdv.budget`: 자체 ledger `work_dir/_qedge9_budget`, soft 9h, required=True 경고만, `time_policy`; gate admission 제외). s5(seed 2026·777, 시트상 옛 s5 묶음 전부 완료) J0→JQ→QE50 @W104 ×2 seed(예약 10.262 h) · **s1(seed 1234; 17:20 사용자 결정으로 s4 대신)** J0→JQ→QE50→QES→QEC @W104 **v2**(control 부터 새로; QEC pilot = s1 J0 v2 exact50K; 8.51 h; SMEC12 chain 뒤 자동, `./tools/qedge9_prepare_s1.sh`, 큐 `config/queues/qedge9_s1.txt`). s4 는 기존 E0 allocation 만. cue 자산 `assets/qedge9/cue_T0_AXIS16_v1.{json,npz}`(s1 이 `tools/qedge9_cue.py build` 로 0.8 min; git, hash 검증; 각 서버 `verify`) · feeder `return_meta`(index·rot; RNG 불변) · trainer `kdv.edge_gate`(registry: EDGE-H 위에서만, routing/TRI 와 결합 거부) · 전환: s5 `./tools/qedge9_switch.sh`(runner 를 죽이지 않는다 — 현재 run·후처리는 그 runner 가, 다음 gate pass 부터 새 순서; chain 없으면 기동; 마감 파일 제거 = soft; 대기자 `tools/qedge9_waiter.sh` 가 DONE-with-pending 을 다시 연다; gate K01–K33 117 검사 · cue verify · config 검사 · 완료 control 검증). **감사 대응(2026-09-15 `PAN_QEDGE9_Implementation_Audit_2026-09-15.md` F01–F08, 노트 §8)**: cue 자산 `asset_id`·내부 일관성·재개 대조, QEC pilot identity 강제·c_E 고정, `kdv.exact_resume`(`kdv/resume.py`: epoch 시작 RNG + batch skip → 재개 run 이 연속 실행과 같은 batch 열; e2e 20/20 동일), `verified_complete`(50K state·후보 격자·Teacher/데이터/init 동치), 실측 통합(PAKD50+QEDGE9 ledger), 옛 extra_priority 보존 분리. 시트 X열 `PAKD50 / <case> / A104D121 / QEDGE9 / FRESH50`, Notes 에 q_source/θq/gate/cue_sha(+cE/perm_seed). 판정은 seed 별 같은 서버의 QE50−JQ·QE50−J0, s4 QE50−QEC/QES.
