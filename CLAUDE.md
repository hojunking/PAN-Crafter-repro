# PAN-Crafter 재현 + 경량화 연구

PAN-Crafter (ICCV 2025) 저자 배포 코드를 재현하고, **경량화(골격 축소)와 정합 aligner 재사용·GT-anchored KD 로
손실을 회복하는 방법**의 베이스라인을 만드는 저장소다(초기 목표였던 상호학습 KD 는 2026-08-20 no-go).
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
- GPU 1장. 현 골격(W112·D123 kdv, 9ch) 50K ≈ 1.2–2.5 h(서버별: s1 1.9 h · s2 2.3 h · s4/s5 1.2–1.5 h). 원 배포 모델(9.97 M) 은 50K ≈ 5 h
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

큐 캠페인 기동·재개는 `./tools/campaign_start.sh --queue <큐파일> [--hours N(기본 24)] [--label 이름]` — 큐를 `work_dir/cases_queue.txt` 로 복사하고
이전 `cases_chain.log` 를 `cases_chain_<label>.log` 로 옮긴 뒤 `_run_cases.sh` 를 detached 로 띄운다(살아 있는 체인이 있으면 거부). chain 마감은
`work_dir/cases_deadline.txt`(ISO 시각) — **지난 마감이 남아 있으면 전 case 가 '마감 경과' 로 스킵되고 즉시 DONE 이 찍힌다**; 파일이 없으면 마감 없음
(QEDGE9·QEGX·EDGEBAL·QRECON24 switch/waiter 가 지우는 soft·무상한 정책). 돌고 있는 chain 의 큐 교체는 `work_dir/cases_queue_handover.txt`(runner 가 case 경계에서 적용) 로 한다 — chain 을 죽이지 않는다.. 사전 확인은 `python tools/gen_pakd50_configs.py --plan --server <srv>`(PAKD50 계열 편성 dry-run, config 생성 없음).

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
- **재현은 완결됐다.** 논문 충실 재구성본 `s1_A1`(11ch·nocrop·LN, 50K 단일 시드) 이 ERGAS **2.0351** 로 논문 2.040 을 넘었다(HQNR 0.9493;
  `research_log/lightweight_case_specs_v1.md` §기준). +6.09% 격차의 최대 원인은 배포 코드의 `crop`(실은 `cv2.resize` scale jitter, −3.63%;
  기전은 `results_log/2026-08-25_divergences-and-tuning-review.md`).
- **논문의 CANConv 대비 우위는 재현되지 않는다** (주장 −5.69% vs 재현 −0.34%, p=0.667).
- **배포 코드는 논문이 기술한 모델이 아니다 — 재구성으로 확인했다.**
  논문 본문·Figure 3 대로 다시 구현하니 params 가 **7.1707 M** 으로 논문 주장 7.170 M 과
  **+0.01%** 로 맞았다(배포 코드 9.969 M). 되돌린 것 셋 다 논문 본문이다 —
  mode modulation 을 Eq (6) 의 직접 학습 γ,β 로(블록당 33,024→512), bottleneck 도 k=3
  (배포본만 k=1), 입력 9ch. 구조는 **3-scale / Down·Up 2 / AttnBlock 3**.
  → `model/pancrafter_paper.py`, `results_log/2026-08-24_paper-rebuild-and-reproduction-audit.md`
- **FLOPs 79.03 G 는 미해결.** 재구성본도 161.9 G 이고, 어텐션을 전부 빼도 125.9 G 다.
  "어텐션 미집계" 가설은 기각했다(배포 구조에서 79.2 G 가 나온 것은 무관한 우연).
- **지표 선택(2026-08 reduced-resolution 아키텍처 비교 시절의 판정; 2026-09 이후는 아래 '판정·표기 규약' 이 우선 — best 선택·판정은 raw HQNR)**:
  당시엔 ERGAS·SAM 만 판별력이 있었고 Q8·SSIM·SCC 는 그 범위에서 포화(±0.15%)라 판별 근거로 인용하면 안 됐다. D_s·HQNR 은 축소하면 거의 항상
  좋아지는 기전이 있어 **단독 해석 금지** — 지금은 raw HQNR 을 주 판정으로 쓰되 D_λ/D_s·fSCC·ERGAS 를 함께 본다.
- **양방향 mutual learning 은 no-go** (`2026-08-20_submodule-sweep-and-mutual-nogo.md`).
  단방향 T→S 증류는 별개의 축이다(no-go 대상 아님) — 다만 09-01·09-14 결과에서 효과가 입증되진 않았고, 현재는 PAKD50(T0 aligner 재사용 + GT-anchored KD) 형태로만 진행한다.
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

## 판정 규칙 — 시드 오차가 대부분의 차이를 삼킨다 (reduced ERGAS 기준, 2026-08 W96 계열)

동일 구성을 시드만 바꿔 돌린 폭이 **0.81%** 다(2.2527 vs 2.2344). 이는 지금까지 인용해온
차이 대부분보다 크다.

| 비교 | 차이 | 시드 폭 대비 |
|---|---:|---|
| 선정 Student vs Teacher | −0.31% (p=0.114) | 0.4배 |
| 6.041 M 경계 | +0.74% (p=0.013) | 0.9배 |
| A-1/A-2 적용 효과 | +0.74% (p=0.0014) | 0.9배 |

**대응표본 t-검정은 같은 가중치를 20장에 적용한 것이라 시드 변동을 포착하지 못한다.**
p 값이 작아도 시드를 바꾸면 뒤집힐 수 있다.

- **0.8%(ERGAS) / 0.0031(raw HQNR, 2026-09-12 실측) 미만의 차이는 시드 3개 이상에서 방향이 일관될 때만 주장한다.**
- 단일 시드 대응표본 p 값만으로 구조 차이를 결론짓지 않는다.
- 기존 결론들도 시드 σ 가 확정되면 **소급 재판정 대상**이다.

## 판정·표기 규약 (경량화 스크리닝·정합 축에서 확정 — 지금도 유효)

- **best 선택 기준은 HQNR** (`select_on: hqnr`). FR 검증 split 이 없어 FR 테스트셋(논문 세트 .mat 20장 전체 — '확정된 사실' 참조)으로 고른다 —
  no-reference 라 GT 누출은 없지만 선택 편향은 있다. 산출물은 `best_hqnr/` / `results/reduced_best_hqnr.mat`(체인 완료 판정은 '함정' 절: reduced+full 둘 다).
- aligner 가 있는 trainer(pa/po/kdv)에서 `best_hqnr/` 는 **best_raw(raw_original HQNR 선택)** 의 alias 다(`kdv/registry.py` SELECTOR_ALIAS);
  `best_rr_val/`(검증 ERGAS)·`last/`·`best_aligned/` 는 별도 산출물이고 판정엔 쓰지 않는다. A-ID/NOALIGN 이면 aligned view 는 raw_valid 와 같아 만들지 않는다.
- 학습 로그에 매 eval epoch `[핵심] HQNR / SCC / ERGAS` 가 찍힌다.
- **지표 우선순위: HQNR > SCC > ERGAS.** 동률이면 SCC, 그 다음 ERGAS. 단 SCC 는 이 범위에서 포화(0.9887~0.9914)라
  실질 tie-break 는 대부분 ERGAS 가 맡는다.
- **HQNR 시드 판정선 2σ = 0.0031** (raw_original HQNR; 2026-09-12 실측 3 경로 일치, `results_log/2026-09-12_s1_alignment-axis-verdict.md`;
  종전 1.18%≈0.011 은 3.5배 과대로 무효 — `2026-09-04_placement-and-band-invalidation.md`). 판정선은 raw HQNR 에만 적용하고 aligned/V64/last 는 진단.
  PAKD50 계열의 작은 KD Δ 에는 threshold 가 아니라 provenance 로만 쓴다(계획 §6.1). 이보다 작은 차이는 SCC → ERGAS 순으로 보조 판정.
- **비교표·판정 view 는 HQNR↑(raw_original, 전체 프레임, `tools/eval_fr_paperset.py` EVAL_VERSION 2026-09-10.5)** 이고 HQNR(V64)↑(가장자리 64 px 제외)·aligned·last 는 진단이다.
  보조 지표 fSCC 는 원 PAN 참조(RR SCC 가 아니다).
- 실험 case 를 대화에서 W1/W2 처럼 부르더라도 **시트·config·문서·보고서에는 약명 단독으로 쓰지 않는다.** 항상 서술형으로
  남기고, 보고서 표에서 축약이 필요하면 **같은 문서 안에 약명→세팅 대응을 반드시 둔다.** kdv 계열 약명은 각 구현 노트의 표로만 읽는다.
- 경량화 축(재구성본): width ≫ full-res depth > AttnBlock 개수 > bottleneck — 배포 코드의 "CM3A 제거 공짜" 는 재구성본에 없다
  (`results_log/2026-08-29_arch-search-24h-results.md`, `2026-08-30_…`). 명세 `research_log/lightweight_case_specs_v1.md`, 실행 `tools/_run_cases.sh`.

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
  체인(`tools/_run_cases.sh`)의 완료 판정은 `results/reduced_best_hqnr.mat` **와** `results/full_best_hqnr.mat` 둘 다 존재로 한다 —
  `finished_at.txt` 로 판정하지 않는다(옛 runner 의 `reduced_best_val.mat` 도 아니다).
- matplotlib 에 한글 글리프가 없다. 그림 라벨은 ASCII 로 쓴다.
- 위성영상은 라이선스 제약이 있다. **외부 서비스에 업로드하지 않는다** (§6).

## 현재 진행 상황 (2026-09-18 기준)

확인은 `results_log/README.md` 맨 위 · `ps -eo pid,ppid,args | grep '[_]run_'` · `tail -f work_dir/cases_chain.log`. 캠페인별 상세는 아래 노트.
각 캠페인의 계획서는 `research_log/PAN_*_<날짜>.md`, 구현 노트는 `research_log/<날짜>_<캠페인>-implementation.md`, config 생성기는 `tools/gen_<캠페인>_configs.py`,
큐는 `config/queues/<캠페인>_<server>.txt`, 검사는 `tools/<캠페인>_unit_tests.py`, 기동은 `tools/<캠페인>_prepare.sh` 규칙이다.

| 서버 | 지금 | 그 다음 |
|---|---|---|
| s1 | QRECON24 ADJ-R1 12/14 완료 · 마지막 학습(G21@3407) 진행 중 → 끝나면 큐 비움 | **① NOA 전수 평가**(`./tools/noa_eval_switch.sh --dry-run` → 학습 끝난 뒤 본 실행 → `--upload`) → **② s1 전용 aligner 분석**(`tools/s1_aligner_analysis.py --assets`) → ③ `--release` 로 본 실험 복귀 |
| s2 | QRECON24 6 완료(S777) · ADJ-R1 6 run 편성 | **① 자기 서버 NOA 전수 평가**: pull → `./tools/noa_eval_switch.sh --dry-run` → `--hold`(진행 중 run 은 끝까지) → 본 실행 → `--upload` → **② `--release` 로 ADJ-R1 큐 복귀**. s1 분석·타 서버 완료를 기다리지 않는다 |
| s3 | QRECON24 10 완료(2026 G 3×3 + G22@4321) · ADJ-R1 10 run 편성 | **① 자기 서버 NOA 전수 평가** 같은 순서 → **② `--release` 로 복귀**(독립) |
| s4 | QRECON24 10 완료(1234 H 3×3 + H_ALPHA0) · ADJ-R1 9 run 편성 | **① 자기 서버 NOA 전수 평가** 같은 순서 → **② `--release` 로 복귀**(독립) |
| s5 | QRECON24 7 완료(L 계열) · ADJ-R1 9 run 편성 | **① 자기 서버 NOA 전수 평가** 같은 순서 → **② `--release` 로 복귀**(독립) |

**09-18: 전 서버가 자기 Student 전수 NOA 평가를 먼저 하고 각자 복귀한다 — 진입점은 `./tools/noa_eval_switch.sh` 하나, 서버용 지시서는 `research_log/2026-09-18_noa-eval-implementation.md` §8 이다(계획 원문 PAN_*.md 는 저장소에 두지 않으므로 서버에는 없다).**
(완료 현황은 ADJ-R1 계획 부록 A(09-17 native Sheet 39 run) 기준; 각 서버의 실제 프로세스 상태는 switch 가 확인한다 — Sheet 만 보고 kill 하지 않는다. 진행 중 run 은 끝까지, 큐 교체는 case 경계 인계.)

### 기반 — 지금 캠페인들의 공통 기준

- **골격·task**: W112·D123 U-Net · MS+PAN 9ch · 단일 HRMS task(PAN 재구성·dual MARs·LPAN/HPAN 제거). 2026-09-09 mainline 결정(원안 W96·D124,
  `research_log/PAN_research_baseline_W96_D124_2026-09-09.md`) → 09-10 PO10 R200 변경으로 W112·D123(2.6589 M). W104·D121(1.9036 M) 은 PAKD50 s4 골격 이식 branch **와 QEDGE9(s5·s1)** 의
  Student 골격(Teacher T0/A 는 W112 그대로, `kdv.teacher.bridge`). SMEC12 의 QB/GF2 준비 학습은 같은 W112·D123 을 4-band 로(2.6508 M).
  **골격이 다른 run 은 직접 대조군이 아니다** — W96·D124(BASE/PA/PO10 R100)·W168·D123 dual·W104·D122(NA104)·W104·D121 과 W112·D123 사이에서 절대 HQNR 을 빼지 않는다;
  대조는 같은 골격·같은 서버·같은 seed 의 대응 run 으로만.
- **trainer kdv** (`kdv/` + `train_kdv.py`, 노트 `research_log/2026-09-10_s2-w112-kdv-implementation.md`): aligner 정책 A-FR/A-FT/A-SC/A-ID · 입력 프로토콜 I-A/I-N/I-AEQ/I-NATIVE-TRANSFER ·
  GT-anchored adaptive KD(rec N0/R0/R1/R2/R3) · 출력 통계(EDGE-H 등) · `aligner_schedule`/`routing`(s5) · `edge_gate`(QEDGE9) · `edge_route`(QEGX) · `edge_schedule`/`edge_weight`(EDGEBAL) · `qrecon`(QRECON24: U/A 목적함수 분리) · `exact_resume`. registry(`kdv/registry.py`) 가 미지원 조합을 거부한다 —
  key 만 적혀 다른 실험이 조용히 도는 일이 없게. 이름 규칙·약명은 각 구현 노트 표. 정합 진단은 `tools/po10_diag.py`(`--ckpt last --native-reference`, `--probe-set palsv18`; |Δ|·EPE·입력 반응).
- **Teacher T0** = PALS24 L1E4 seed 2025 best_raw 의 A+U (`assets/pakd50/T0_run`, 모든 서버에서 같은 경로·sha 검사). τR 0.012463942170143127 · λE0 0.09075170336956798
  (`assets/pakd50/calibration_resolved.json`, `tools/pakd50_calibrate.py`) · 후보 격자 `GRID1010_50K_v1`(eval_epoch 5, 50 후보 전부 보존).
  git 자산(지우지 말 것): `assets/donor_aligner/`(PA_A1 S2025 aligner — kdv unit gate 가 읽는다) · `assets/pakd50/`(T0_run·calibration·clock·init_hashes) · `assets/qedge9/`(cue).
  prepare 스크립트는 pa/kdv/nf16/pals24/pakd50 unit test 를 전부 돌린다.
- **시트**: 탭 `WV3-<server>`(`gspread/server.txt`), 범주 ⑳ KDV ~ ㉖ SMEC12 는 `gspread/sheet_categories.py`, 옛 범주는 `WV3-<server>_v1` 탭. HQNR↑ = 전체 프레임(논문 프로토콜),
  HQNR(V64)↑ = 가장자리 64 px 제외(판정은 HQNR↑ — '판정·표기 규약'). PAKD50 계열 X열 `PAKD50 / <case> / [A104D121 /] [QEDGE9 | QEGX | EDGEBAL /] FRESH50`; QRECON24 는 `PAKD50 / QRC24 / <PROFILE> / A104D121 / FRESH50`.
  `gspread_upload.py --all` 은 `sheet_categories.ARCHIVED` 범주를 기본 제외(`--include-archived` 로 포함; 이전 도구 `gspread/archive_to_v1.py`, 백업 `gspread/_sheet_backup/`).
- **캠페인 gate**: `work_dir/campaign_gates_enabled.txt` 에 적은 gate 만 `tools/campaign_gate.py` 가 연다(기본 전부 닫힘) — 캠페인 뒤 비운다. 돌던 체인의 재편성은 runner 교체
  (`pakd50_requeue.sh`/`pakd50_reallocate.sh`) 또는 runner 를 두고 다음 gate pass 가 새 코드를 읽게 하는 방식(`qedge9_switch.sh`).
- **예산·시계**: PAKD50 branch 는 공통 절대 시계 `assets/pakd50/campaign_clock.json`(학습 마감 2026-09-16 11:22:31; trainer `kdv.budget.training_deadline`·gate admission·체인 마감이 같은 시각) + 50 h ledger.
  slot 예약 `reservation_h = 1.10 × reference_train_h + 10/60` → 서버 로컬 `work_dir/_pakd50/reservations.json`(= `kdv.budget.projection_file`) · `mandatory_runs.txt` ·
  선택 `extra_priority.txt`(case id 또는 run 이름; 없으면 `PRIORITY_BY_SERVER` 기본 묶음만). QEDGE9(soft 9 h)·QEGX·EDGEBAL·QRECON24(상한 없음) 는 이를 상속하지 않는다. **09-16 저녁부터 `PRIORITY_BY_SERVER` 는 다섯 서버 전부 QRECON24** — 직전 순서는 `PREVIOUS_PRIORITY_BY_SERVER`.
- **판정**: 각 캠페인 계획서의 판정 절(PAKD50 `PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md`, QEDGE9 `PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md` §10 등) 그대로 —
  공통 원칙: 같은 서버·같은 seed 안의 대응 차이를 먼저, 판정선 0.0031(raw HQNR), 서버 간 절대 HQNR 을 빼지 않는다. GPU 학습은 run-to-run 재현이 아니다(s4 J0 v1/v2 0.0026 차).

### 활성 캠페인

**PAKD50 통합 (2026-09-14~)** — 계획 `research_log/PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md` + `PAN_Integrated_Method_Summary_2026-09-14.md`, 노트 `2026-09-14_pakd50-implementation.md`.
T0 의 aligner 를 Student 가 복사(J: 공동 적응 / F: frozen)하고 U-Net 은 새로 학습, backend N0/R1/Q12/X02. seed s1 1234 · s2 777 · s3 2026 · s4 1234(교차) · s5 2026(교차); 목표 raw HQNR ≥ 0.959.
config `PAKD50_<case>_<W…_D…>_WV3_T0_S<seed>_FRESH50_v<n>`(`tools/gen_pakd50_configs.py`; 약명→세팅은 config 머리 주석·노트 표), 큐는 J0 만·나머지는 gate `pakd50` 가 서버별 명시 순서(`PRIORITY_BY_SERVER`)로 편성,
gate `tools/pakd50_unit_tests.py`(K01–K33), 기동 `pakd50_prepare.sh`. **s1 의 PAKD50 은 09-14 22:40 사용자 지시로 중단**(J0/JQ/F0/FQ/JR 완료).
- 재배정 09-15 (`PAN_PAKD50_S2_S4_S5_Derived_Run_Allocation_2026-09-15.md`, 노트 `2026-09-15_pakd50-derived-allocation-implementation.md`): 새 case J_R3_NOEDGE/J_N0_EDGE/RC0/RCQ, 예약식, 전환 `pakd50_reallocate.sh`(사전 `gen_pakd50_configs.py --plan`), 확인 seed s4 3407 / s5 9091 / s3 4321.
  남은 후속: PAKD50 감사 F07–F10(worktree·A gradient 분해·bin 자료원·package pin), TCOPY/CONT, DCR12 B2/B3/CMASS/B1-NOOFF — 각 구현 노트 §남긴 것.
- s3 추가 09-15 (`PAN_PAKD50_Latest_Sheet_Analysis_and_S3_Experiments_2026-09-15.md`, 노트 `2026-09-15_pakd50-s3-additions-implementation.md`): LFX(= LF 일정 + X02), 후보별 확인 묶음 표.
- s4 골격 이식 09-15 (`PAN_PAKD50_S4_W104D121_Architecture_Allocation_2026-09-15.md`, 노트 `2026-09-15_pakd50-s4-w104d121-implementation.md`): Student U 만 W104·D121(branch `A104D121_T0FIX_E0_v1`, `kdv.teacher.bridge`), 항목 `case@W104_D121`, init hash namespace `unet@W104_D121`.
- s5 (`PAN_S5_Timing_Routing_Experiment_Plan_2026-09-14.md`, 노트 `2026-09-14_pakd50-s5-review-and-implementation.md`): A 의 업데이트 시점(D/LF)·수신 경로(P/JK0/JE0) — `kdv.aligner_schedule`·`kdv.routing`. s4 초기 배정은 `2026-09-14_pakd50-s4-review-and-implementation.md`.

**DCR12 — offset consistency × reconstruction 사분면 검증 (s1 → s2)** — 계획 `PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md`, 노트 `2026-09-15_dcr12-implementation.md`.
T0/B0(FQ)/B1(JK0) 의 C×R 사분면(D01) → correction 개입(D02) → A gradient 충돌(D03) → 조건부 micro-update(D04) → 보고서. 구현 `tools/dcr12/`, runner `tools/dcr12_run.sh`(서버 공용, 큐 `config/queues/dcr12_<server>.txt`), gate `tools/dcr12_unit_tests.py` X01–X15.
09-15 12:34 서버 교체: s1 은 JK0 S1234 까지만(12:05 완료) → seed 1234 pair 는 bundle(`tools/dcr12_bundle.py`, 127 MB, git 밖; provenance 로 완료 판정) 로 s2 에 옮기고, s2 는 자기 FQ S777 + 새 JK0 S777 뒤 D01–D04·REPORT. seed 간 호스트가 다르면 REPORT 가 `SEED_HOST_COUPLED`. 절차는 노트 §5(bundle 을 늦게 넣으면 `dcr12_prepare.sh --post`).

**SMEC12 — 다중 데이터셋 sample 기전 검증 (s1)** — 계획 `PAN_SMEC12_MultiDataset_SampleMechanism_ExperimentPlan_2026-09-15.md`, 노트 `2026-09-15_smec12-implementation.md`.
"q 는 낮은데 복원이 어려운 sample" 의 특성·기전을 WV3·QB·GF2(+WV2 zero-shot) 에서. (a) QB/GF2 준비 학습 10 run(P0 → DON-N2 → L000 → L1E4 → L1E4-REP × 2 센서; `tools/gen_smec12_bootstrap.py`, 큐 `config/queues/smec12_<server>.txt`, 기동 `tools/smec12_prepare.sh`)
(b) 분석 backbone `tools/smec12/`(A00·D10/D11·I20·I23-B·I24-A·I25-A·X40·REPORT; runner `tools/smec12_run.sh`, gate `tools/smec12_unit_tests.py`) 는 있는 자산만 증분 처리, 미구현 stage 는 `pending_compute`. WV3 lane 자산이 s1 에만 있어 s1 에서(원안 s2 → 12:34 교체). WV3+WV2 검증 결과는 노트 §5.

**QEDGE9 — W104·D121 q-gated GT edge (s5·s1)** — 계획 `PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md`, 노트 `2026-09-15_qedge9-implementation.md`, 감사 `PAN_QEDGE9_Implementation_Audit_2026-09-15.md`(F01–F08 대응은 노트 §8).
case `QE50`(Q12 hard/soft 그대로, GT edge 는 고정 T0 aligner 의 q(AXIS16) < θq = 0.327613 인 patch 만: λE·Σ g_i E_i / B) · `QEC`(모든 patch edge × c_E, pilot = **s1 의 `PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v2`** exact50K — F02 로 identity 고정, s4 의 J0 v1 아님) · `QES`(gate 를 e_roi32 decile × aug state stratum 안에서 permutation 51515).
캠페인 `QEDGE9_A104D121_20260915_v1` / branch `A104D121_T0FIX_QEDGE9_v1`, **PAKD50 마감·50 h 미상속**(자체 ledger soft 9 h). cue 자산 `assets/qedge9/cue_T0_AXIS16_v1.{json,npz}`(`tools/qedge9_cue.py build/verify/pilot/status/stamp`, asset_id·내부 일관성·재개 대조; 상태 `work_dir/_qedge9/status.json`),
feeder `return_meta`, trainer `kdv.edge_gate`/`kdv.exact_resume`(`kdv/resume.py`). s5: J0→JQ→QE50 @W104 × seed 2026·777(`qedge9_switch.sh`; runner 를 죽이지 않는다, 대기자 `qedge9_waiter.sh`). s1(17:20 결정, s4 대신): J0→JQ→QE50→QES→QEC @W104 S1234 **v2**(큐 `config/queues/qedge9_s1.txt`, `qedge9_prepare_s1.sh`, SMEC12 뒤 자동). s4 는 QEDGE9 없음.

**QEGX — W104·D121 q-edge × soft · edge 수신 모듈 (s3·s4, 09-15 저녁)** — 계획 `PAN_QEGX_S3_S4_W104D121_Experiment_Plan_2026-09-15.md`(+ `S3_Run_Handoff_QEGX_2026-09-15.md`·`S4_Run_Handoff_QEGX_2026-09-15.md`), 노트 `2026-09-15_qegx-implementation.md`.
새 case: `QX50`(QE50 − output soft = R1 + gated edge; 'QE50 에서 β=0' 의 동치, XJ 아님) · `QE50_B005`(QE50 β 0.05; `J_QB005` 와 대응쌍) · `QEC3`(상수 c_E3, pilot = **s3 의 `PAKD50_J0_W104_D121_WV3_T0_S2026_FRESH50_v2`** exact50K → `work_dir/_qegx/qec3_cE.json`; s1 QEC 파일과 별도) · `LFQE50`(LF 일정 + QE50)
· **`QER50`/`QERS`** = `kdv.edge_route`(U 는 모든 patch 의 GT edge, A 는 q_T<θq / stratum 셔플 patch 의 edge 만 — total 은 JQ 와 같고 backward 뒤 A 의 .grad 에서 λE·mean((1−g)E_i) 를 뺀다; `edge_gate`·`routing` 과 결합 금지, registry 가 막는다). JQ(1/1)·JE0(1/0)·QE50(g/g)·QER50(1/g) 대응.
캠페인 `QEGX_A104D121_S3S4_20260915_v1` / branch `A104D121_T0FIX_QEGX_v1`, **시간 상한 없음**(자체 ledger `work_dir/_qegx_budget/`, `total 1000 h` + `required`(경고만), 절대 마감·9 h 미상속; gate admission 제외; 실패/NaN 자동 반복 없음).
s3 = 15 run **v2**(J0→JQ→QE50→XJ→QX50→J_R3_NOEDGE→QEC3→QES→J_QB005→QE50_B005 S2026 → J0/JQ/XJ/QE50/QX50 S4321; 예약 25.20 h) — s5 QEDGE9 의 J0/JQ/QE50@W104 S2026 v1 config 와 이름 충돌을 피한 것(정의는 같다). s4 = E0 5 완료 뒤 14 run v1(QE50→LF0→LFQ→LFQE50→LFX→JE0→QER50→QERS S1234 → J0/JQ/QE50/JE0/QER50/QERS S3407; 26.28 h).
큐 `config/queues/qegx_{s3,s4}.txt`, 전환 `tools/qegx_switch.sh`(`--dry-run`/`--pilot`(s3 c_E3)/`--refresh-controls`(s4)), 대기자 `tools/qegx_waiter.sh`(s3 는 J0 v2 exact50K 뒤 c_E3 자동), QEC3 pilot `tools/qedge9_cue.py pilot --branch qegx`. 검사 K34–K38(136 ALL OK). 시트 X열 `… / QEGX / FRESH50`, Notes `edge_U/edge_A/beta/freeze_from/cE_pilot`.
**함정**: bare `QE50@W104_D121` 항목은 QEDGE9 자동 규칙(9 h ledger) 으로 간다 — QEGX 큐/extra 에는 **run 이름만** 쓴다(switch 가 검사).

**QRECON24 — 확정 method 의 전 서버 튜닝 68 run (s1–s5, 09-16 저녁; 현재 큐)** — 계획 `PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md`, 노트 `2026-09-16_qrecon24-implementation.md`. 미실행 QETUNE24 86 run 은 superseded.
method `kdv.qrecon`(continuous_v1): **w_i = qref/(qref + q_T(i))**(raw q, qref 0.3276133416220546; threshold 아님; **09-16 저녁 사용자 결정으로 계획서의 분자 2 를 뺐다** — w(qref)=0.5, 실제 w 0.40–0.55, 평균 0.498; s1 첫 run G22 S1234 만 2 가 있던 코드로 시작, 노트 §9) 를 A 와 U 가 공유 — **U ← L_U = mean(H + K + λE·w^E·E)**, **A ← L_A = mean(w^A·H) 만**(soft·edge·offset 없음), 같은 forward 에서 parameter 집합별 `autograd.grad`(`_qrecon_backward`; 단일 total backward 금지, optimizer.step 한 번).
Student 는 T0 A 복사 + fresh U(W104·D121), native 입력(I-NATIVE-TRANSFER, jitter·offset 없음), λE **절대값**, A LR = rA × U LR. profile: G<ij>(λE **6e-4/2e-3/6e-3**(09-16 저녁: 분자 2 제거에 맞춘 2 배 환산; 계획서 3e-4/1e-3/3e-3) × rA .003/.01/.03; **G22 기준**) · A_UNIF/A_SHUF/A_FREEZE · E_UNIF/E_SHUF · ALL_UNIF(**uniform 가중 = s_q(qref) = 0.5**, 계획서의 1 아님) · H<ij>(α .5/1/1.5 × β .05/.1/.2) · H_ALPHA0/H_BETA0 · L100/L070/L050(U LR 1e-4/7e-5/5e-5). G22 = H22 = L100.
이름 `PAKD50_QRC24_<SRV>_<PROFILE>_W104_D121_WV3_T0_S<seed>_FRESH50_v1`(서버 토큰은 충돌 방지; X열 `PAKD50 / QRC24 / <PROFILE> / A104D121 / FRESH50`). 캠페인 `QRECON24_A104D121_S1S5_20260916_v1` / branch `A104D121_T0FIX_QRECON24_v1`, 상한 없음(24h 는 최소 운영구간; ledger `work_dir/_qrecon24_budget/`), 예약 1.20×R_s + 10/60.
큐 `config/queues/qrecon24_s{1..5}.txt`(사전 고정; 승자 대기 없음), 전환 `tools/qrecon24_switch.sh`(`--dry-run` / `--extend` §8.3), 대기자 `tools/qrecon24_waiter.sh`, **target selector** `tools/qrecon24_select.py <run> --official`(raw H ≥ 0.9585 후보 안에서 RR SCC→ERGAS→PSNR→SAM→Q8→SSIM; legacy best 보존; `results/qrecon24_target_selection.json`; 공식 RR 없는 적격 후보가 있으면 target 미확정). 검사 K44–K49(176 ALL OK).
감사 대응(09-16 저녁, `PAN_QRECON24_Implementation_Audit_2026-09-16.md` F01–F10, 노트 §8): epoch 끝 checkpoint 의 exact resume(다음 epoch 새로 시작; 불일치는 exit 4, 같은 id fresh 재실행 금지) · runner 가 **case 경계에서 `work_dir/cases_queue_handover.txt` 로 큐를 인계**(전환 스크립트가 chain 을 죽이지 않고 이 파일을 둔다) · `--extend` 는 활성 큐/mandatory/reservations 까지 · ledger `train_hours/postprocess_hours/setup_hours` · `verified_complete` 는 고유 50 격자·후보 checkpoint·T0/cue/init 필수 · s4/s5 control id 는 H22/L100.
**판정**: raw-original HQNR ≥ 0.9585 하한 + 같은 checkpoint 의 RR(논문 표시값 SCC .988 / ERGAS 2.040 / PSNR 37.956 / SAM 2.787 / Q8 .922 / SSIM .976). Asset board(seed 최고 자산) 와 Method board(profile 의 모든 seed 평균·σ·하한 통과 수) 를 분리. 같은 seed 의 다른 서버 실행은 host 반복이지 독립 seed 가 아니다.

**QRECON24 ADJ-R1 — 실행 순서·우선순위 조정 (s1–s5, 09-17; 현재 큐)** — 계획 `PAN_QRECON24_S1_S5_Queue_Adjustment_2026-09-17.md`, 노트 `2026-09-17_qrecon24-adjustment-r1-implementation.md`. 수식·gradient 경로·Teacher·q·λE 환산·uniform .5 는 불변; 바뀐 것은 편성과 metadata.
관측(§1): s1 G23 S1234 raw HQNR **.9591**(목표 초과) · s4 β .1→.2 가 α .5/1/1.5 에서 3/3 개선 · rA .03 은 s2 777 −.0031 / s3 2026 +.0003 이라 일괄 적용하지 않음 · 낮은 U LR 확대 보류.
신규 profile 3(§4; 새 loss 아님): **B20A03**(λE .002·rA .03·α 1·β .2 = G23 에서 β 만 / H23 에서 rA 만) · **A03_UNIF/A03_SHUF**(G23 에서 A 의 q 연결만 상수 .5 / stratum 셔플). 같은 서버·seed 의 A LR×β 2×2(G22 alias / G23 / H23 / B20A03) 를 s3 4321 · s4 1234 · s5 9091·1103 에 만든다(§6.1; I = H(B20A03)−H(G23)−H(H23)+H(G22)).
편성(§5·§9): 활성 = 등록 완료 39(원 순서; runner 가 건너뜀) + 남은 42 = **81 run**(s1 14 · s2 12 · s3 20 · s4 19 · s5 16). 추가 16 run 은 **`_FRESH50_v2` id**(부록 B; 편성 이력 표시), 유지 26 run 은 v1 id — **원계획 68 v1 config 는 바이트 불변**(K50). 보류 3(s4 H11@3407 · s5 L070/L050@1103) 은 `superseded_pending`(편성·mandatory 밖; 이미 시작/완료면 원 정의로 끝냄). generator `QRC24_ADJ_ORDER/QRC24_HELD/QRC24_QUEUES_20260916`, 큐 `--qrc24-queues`, v2 config `kdv.qrc24.queue_revision = QRC24_ADJ_R1_20260917`(+ `block_2x2`, 실제 id 의 `control_runs`), 시트 X열 `PAKD50 / QRC24 / <PROFILE> / A104D121 / ADJ-R1 / FRESH50`.
운영: 같은 `tools/qrecon24_switch.sh`(③-ADJ 보류 상태·버전 감사, ⑤ `queue_effective.txt`(미완만)·`cases_queue.txt` 영속 갱신, ⑦ selector backlog) · **공식 RR selector·버전 감사는 case 경계**(`tools/_upload.sh` → `tools/qrecon24_postrun.py --backlog`; 학습 중이면 건너뜀; GPU 시간은 ledger `select_<run>`) · `tools/qrecon24_version_audit.py`(§8.1: s1 완료 6 run 전부 `single_definition_factor1`; G22 S1234 의 옛 코드 시도는 ledger `#1` 로만 남음) · `tools/qrecon24_preserve.py`(§7.1: s1 G23 S1234 보존 완료). 24h 는 원 campaign 누적(다시 세지 않음). 판정은 계획 §12(같은 checkpoint raw H ≥ .9585 → RR SCC→ERGAS→PSNR→SAM→Q8→SSIM; 후보 최고치·seed 반복·host bridge·late6 분리).

**QRECON24 Narrow R2 — 마지막 β 비교 뒤 설정 하나 동결·seed 반복 (전 서버, 09-17 계획 / 09-18 구현)** — 계획 `PAN_QRC24_Narrow_R2_SeedLock_ERGAS_2026-09-17.md`(revision `QRC24_NARROW_R2_20260917`), 노트 `2026-09-18_noa-eval-implementation.md` §10.
**목표 2 순위가 SCC → ERGAS 로 바뀌었다.** 같은 checkpoint 에서 raw HQNR ≥ .9585 를 먼저 지키고, 그 집합 안에서 **공식 RR ERGAS 최소** → SCC → PSNR → SAM → Q8 → SSIM → HQNR → step. 새 selector `HQNR9585_ERGAS2040_v2`(`tools/qrecon24_select.py --selector`; 결과는 `results/qrecon24_target_selection_HQNR9585_ERGAS2040_v2.json` 로 따로 쓰고 v1·legacy best 는 보존). `joint_pass` 는 official RR 이 끝난 target 에서만 판정한다.
넓은 grid(λE·α·U LR·rA)는 **종료**한다. 남은 활성 축은 **β .1(G23) ↔ .2(B20A03)** 하나뿐이고, 기존 G23 의 짝만 채우는 **B20A03 4 run**(s1 1234/3407 · s2 777 · s3 2026; s3 4321 은 이미 있고 s4 는 G23·1234 host bridge 로 끝) 을 `_FRESH50_v3` 로 편성했다 — `qrc24_beta_close_items()`, 활성 큐 꼬리에 자동으로 붙는다.
**2026-09-18 사용자 결정으로 lock 게이팅을 없앴다** — 공동 목표 통과 seed 수를 확인하지 않고 각 서버가 사전 지정 **seed 4 개**(s1 41001/41006/41011/41016 · s2 41002/41007/41012/41017 · s3 41003… · s4 41004… · s5 41005…) 를 **바로** 돈다. 설정은 `qrc24_seed_profile()`(lock 파일이 있으면 그 profile, 없으면 **G23**) 하나로 전 서버 공통이고, 활성 큐 맨 앞에 seed 가 오고 B20A03 β-close 는 참고 비교로 뒤에 붙는다. `tools/qrc24_lock.py` 는 비교표·기록용으로만 남는다(아무것도 막지 않는다).
**시트 실행명은 2026-09-18 부터 짧게 쓴다** — 고정 method 를 접두어 하나로 포괄하고 뒤에 달라지는 것만: `QRC24 <PROFILE> S<seed>[ v<n>]`(예 `QRC24 G23 S41001`). 서버·골격(W104·D121)·프로토콜(FRESH50)은 탭과 고정 method 가 말한다. 긴 실행명은 Notes 의 `run=…` 에 남고, `sheet_categories.canonical_run_key()` 가 긴 표기와 짧은 표기를 같은 run 으로 묶어 이미 올라간 행을 중복 생성하지 않는다(기존 행의 이름은 덮어쓰지 않는다).
시트는 `gspread/target_upload.py` 가 target 열 묶음(`target_selector/step/HQNR_raw/ERGAS/SCC/PSNR/joint_pass/official_complete`, `recipe_lock_id`, `queue_revision`) 만 따로 쓴다 — legacy 열을 교체하지 않는다. 검사 K52(회귀 1–10).
**현재 상태(09-18 측정)**: 짝이 맞는 block 은 s3 4321 · s4 1234 둘뿐이고 두 후보 모두 H 하한 통과 seed 0 → 규칙 4 로 **G23 유지**가 잠정값이다. 공동 목표(H ≥ .9585 & E < 2.040)를 만족한 run 은 시트 전체 376 행 중 **0 건**이고, QRC24 90 행 중 ERGAS < 2.040 자체가 0 건이다(최저 2.0544, s3 G12·2026, H .9501). 이 사실을 숨기지 말고 보고한다.

**NOA 전수 Student 평가 — 추론 정합 경로 검증 (전 서버, 09-18)** — 계획 `PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md`(protocol `PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918`), 노트 `2026-09-18_noa-eval-implementation.md`. 학습 method(qrecon_continuous_v1)·Teacher·q cache·기존 checkpoint 를 바꾸지 않는다 — **추론 경로만** 비교한다.
모드: `A_ON`(기존) · **`A_BYPASS_RAW`(표시명 NOA) = aligner 호출 0회·warp 호출 0회**, 원 PAN·원 LRMS 를 같은 U-Net 에 직접 · s1 분석에만 `A_ZERO_WARP`(A 0회지만 sampler 사용 — NOA 와 다른 모드) · `A_CROP64_MED`(non-overlap 64 crop Δ 의 성분별 중앙값으로 전체 PAN 한 번 warp). 구현 `kdv/eval_modes.py`(CallCounter 가 0회를 실제로 센다, sampler 복원 보장).
**진입점은 `./tools/noa_eval_switch.sh`** (`--dry-run` → `--hold` → 본 실행 → `--upload` → `--release`; 각 서버가 자기 것만 한다). 내부 절차: `tools/eval_phase.py hold`(진행 중 학습은 그대로 끝나고 **새 학습만** 멈춘다 — smoke_cases rc2·waiter HOLD_EVAL·watchdog·switch·gate 다섯 경로가 `work_dir/_eval_phase/hold.json` 을 본다) → `tools/noa_eval.py --capture-cohort --sanity --all`(같은 checkpoint 의 A_ON/NOA RR·FR 쌍; RR 은 gspread_upload._rr, FR 은 논문 .mat20) → `gspread/noa_upload.py`(자기 탭 NOA 26 열만, 기존 열 보존) → `tools/eval_phase.py release`. s1 만 그 뒤 `tools/s1_aligner_analysis.py`(크기 패널·AXIS16 q_size·네 모드). **다른 서버 평가나 s1 분석을 기다리지 않는다.**
판정·주의: 같은 checkpoint·같은 모드에서 raw HQNR ≥ .9585 **그리고** RR ERGAS < 2.040(반올림 아님). ΔE = E_NOA − E_ON, ΔH = H_NOA − H_ON. NOA 가 좋아도 **학습·추론 정책을 자동으로 바꾸지 않는다**(계획 §10.5). q_size 는 사후 측정값이라 학습 q cache·q_ref 와 섞지 않는다. 검증: s1 G23 S1234 에서 paired A_ON 이 공식 저장값과 오차 0(legacy_on_check match), 한 run 두 모드 RR+FR 약 47 s. 검사 K51(V01–V18), gate 195 ALL OK.

**EDGEBAL — GT edge 를 얼마나·언제 (s2·s5, 09-16)** — 계획 `PAN_EDGEBAL_S2_S5_Experiment_Plan_2026-09-16.md`, 노트 `2026-09-16_edgebal-implementation.md`. q gate 를 더 복잡하게 만들기 전에 edge 강도·시간배분·완만한 cue 를 분리한다.
case `EB_*`(prefix; 전부 J 정책·R3 기본): 상수 배수 `EB_R3E025/050/075/100/200`(= r×λE0, `lam_mult`; 050 = J_QE025, 100 = JQ, 200 = J_QE10) · `EB_N0`(= J0) · `EB_R3E000`(= J_R3_NOEDGE) · `EB_N0E100`(= J_N0_EDGE) · `EB_R1E100`(= XJ) ·
**`EB_EDOWN`/`EB_EUP`** = `kdv.edge_schedule{before, after, switch 25000}`(w(t) 1→.5 / .5→1; 0-based update 순수 함수, A 는 계속 학습, optimizer 재시작 없음; 경계 기록 `edge_schedule_events.jsonl`) ·
**`EB_QFLOOR`/`EB_QFSHUF`/`EB_QFREV`** = `kdv.edge_weight{mode floor|floor_shuffle, low, high, asset}`(w_i = high + (low−high)·g_i; g 는 QEDGE9 cue 자산의 0/1 표 그대로 — `kdv/edge_gate.AffineEdgeWeight`; registry spec 키는 `edge_cue_weight`). schedule × weight 결합은 registry 가 거부(계획 §6).
캠페인 `EDGEBAL_A104D121_S2S5_20260916_v1` / branch `A104D121_T0FIX_EDGEBAL_v1`, 시간 상한 없음(자체 ledger `work_dir/_edgebal_budget/`, total 1000 h + required 경고만; gate admission 제외).
s2 = 12 run **v1**(EB_R3E000→R3E050→R3E025→R3E200→N0E100→R1E100 S777 → EB_N0/R3E100/R3E050 × S2026·S9091; 31.59 h; 대조 = s2 의 QEDGE9 v2 3 벌). s5 = QEDGE9 6 뒤 14 run **v2**(EB_R3E050→R3E075→EDOWN→EUP→QFLOOR→QFSHUF→QFREV→N0E100 S2026 → R3E075/EDOWN S777 → EB_N0/R3E100/R3E075/EDOWN S9091; 26.57 h).
큐 `config/queues/edgebal_{s2,s5}.txt`, 전환 `tools/edgebal_switch.sh`(`--dry-run`), 대기자 `tools/edgebal_waiter.sh`. 검사 K39–K43(152 ALL OK). 시트 X열 `… / EDGEBAL / FRESH50`, Notes `edge_mult/edge_schedule/edge_low/high/gt_hard_always/T0_sha/cue_asset_id/seed/release_sha/control_run_id`.
s5 의 W104 시간 기준값은 계획 §7.1 실측(J0 1.38 / JQ 1.57 / QE50 1.65) 으로 교체했다. 판정은 계획 §8(selected raw 우선, plateau_30Kplus 21 점·plateau_45_50 6 점은 별도 이름, 3 seed 대응 Δ).

**EQREC4 결과 (s1 완료 09-15 03:00; s3 반복은 사용자 결정)** — `results_log/2026-09-15_s1_eqrec4-results.md`: **q 는 patch 정합 품질의 표지가 아니다**(H1 native·H2 3 seed 반대, FR scene 수준만 양) · learned correction > zero(3/3) ·
Student cue 로는 판정 불가, 5K pilot 은 모든 arm 에서 HQNR 하락 → **q_T Teacher-quality gate 채택 안 함**(q 는 GT edge 선택에만 — QEDGE9). 구현 `tools/eqrec4/`, s3 는 `eqrec4_bundle.py` + `eqrec4_prepare.sh`.

### 지난 캠페인 (결론만 — 상세·수치는 `results_log/README.md`)

- 08-28~09-07 아키텍처 탐색·KD·SE·Swin·Teacher 4–6M: 9ch 성립, 고해상도 인접 용량이 핵심, 용량 확대·KD·SE 전부 K0(= Student 단독 학습 baseline, `results_log/2026-09-01_kd-se-msonly-campaigns.md`) 를 넘지 못함(`results_log/2026-08-29 … 09-04`).
- 09-08 W168 multiset 3-seed(중단, `config/queues/arch_w168_multiset_3seed.txt` 로 재개 가능) → 09-09 BASE W96·D124 3-seed → 09-09 PA A1–A3(aligner 는 방향은 맞지만 입력 무반응 상수 0.2 px, `pa/`) → 09-10 PO10 offset consistency(R100→R200 + W112·D123, aligner 가 x 축 −0.5 반응, `pa/offset.py`) → 09-11 NF16 native fitting →
  **09-12 정합 축 종합 판정: PAN 을 옮기는 접근은 성립하지 않는다**(BASE96·PA·PO10·NF16 17벌) → **09-13 PALS24: 작은 offset loss(λ* = 1e-4) 만 3 seed 양성** → 09-14 PALSV18: 3e-5/3e-4 는 넘지 못해 **L1E4 유지 = T0**.
- 09-10 KDV s2(W112·D123 GT-anchored KD·통계 variance, Q00–Q08; TRI-A/B/C·geomKD G1–G5·G-STRUCT 구현, 큐 `kdv_s2_geomkd.txt`/`kdv_s2_triabc.txt` 는 보류) · 09-11 NA104(W104·D122 no-align KD 87+ case, s2·s3; 20H 우선순위 Q36/Q12/CF01) → **09-14 92벌 무소득**, 09-14 PAKD50 으로 대체.
