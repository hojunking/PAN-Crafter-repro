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
  **논문 Table 과 비교 가능한 것은 `matlab` 뿐이다.**
- 새 문서를 만들면 `results_log/README.md` 맨 위에 한 행 추가한다.
- **구글시트 업로드에는 서버 식별자를 반드시 붙인다** (`gspread/server.txt` 에 `s1`/`s2`).
  두 서버가 같은 config 를 돌리면 실행명이 같아져, suffix 가 없으면 상대 서버 값을
  덮어쓴다. 시트 위에서 서버 간 수치 혼용이 일어나는 것이다.

## 확정된 사실 (다시 파헤치지 말 것)

- **재현 성립.** WV3 reduced ERGAS **2.1633** (배포본 그대로, 50K). 논문 2.040 대비 +6.09%
- **그 격차는 우리 잘못이 아니다.** 평가기는 CANConv 배포 가중치로 논문 행을 6지표 0.5% 이내
  재현하고, 논문 명시 설정은 시드 2,025 까지 전부 일치한다.
  → `results_log/2026-08-24_reproduction-audit.md`
- **논문의 CANConv 대비 우위는 재현되지 않는다** (주장 −5.69% vs 재현 −0.34%, p=0.667).
- **배포 코드는 논문이 기술한 모델이 아니다 — 재구성으로 확인했다.**
  논문 본문·Figure 3 대로 다시 구현하니 params 가 **7.1707 M** 으로 논문 주장 7.170 M 과
  **+0.01%** 로 맞았다(배포 코드 9.969 M). 되돌린 것 셋 다 논문 본문이다 —
  mode modulation 을 Eq (6) 의 직접 학습 γ,β 로(블록당 33,024→512), bottleneck 도 k=3
  (배포본만 k=1), 입력 9ch. 구조는 **3-scale / Down·Up 2 / AttnBlock 3**.
  → `model/pancrafter_paper.py`, `results_log/2026-08-24_paper-faithful-rebuild.md`
- **FLOPs 79.03 G 는 미해결.** 재구성본도 161.9 G 이고, 어텐션을 전부 빼도 125.9 G 다.
  "어텐션 미집계" 가설은 기각했다(배포 구조에서 79.2 G 가 나온 것은 무관한 우연).
- **지표 선택**: ERGAS·SAM 만 판별력이 있다. Q8·SSIM·SCC 는 이 범위에서 포화(±0.15%)라
  판별 근거로 인용하면 안 된다. D_s·HQNR 은 축소하면 거의 항상 좋아지는 기전이 있어 단독 해석 금지.
- **양방향 mutual learning 은 no-go** (`2026-08-20_mutual-learning-go-no-go.md`).
  단방향 T→S 증류는 별개이고 유효하다.

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
  `best_hqnr` / `reduced_best_hqnr.mat` 이고, 체인 완료 판정도 이 파일이다.
- 학습 로그에 매 eval epoch `[핵심] HQNR / SCC / ERGAS` 가 찍힌다.
- **지표 우선순위: HQNR > SCC > ERGAS.** best 선택은 HQNR(공식 12-19), 동률이면
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
- **`pkill -f <패턴>` 이 자기 자신을 잡는다.** `ps -eo pid,args` 로 PID 를 골라 죽인다.
- **`tail -f log | grep`** 은 마지막 10줄부터 시작해 아무것도 안 나온다.
  `tail -n +1 -f ... | grep --line-buffered` 를 쓴다.
- **`${1:?메시지}` 안에 `}` 를 넣지 않는다.** 파라미터 확장이 끊겨 인자가 오염된다.
- **`run.sh` 의 `trap ... EXIT` 는 실패해도 `finished_at.txt` 를 쓴다.**
  체인의 완료 판정은 `results/reduced_best_val.mat` 존재로 한다.
- matplotlib 에 한글 글리프가 없다. 그림 라벨은 ASCII 로 쓴다.
- 위성영상은 라이선스 제약이 있다. **외부 서비스에 업로드하지 않는다** (§6).

## 현재 진행 상황

`results_log/README.md` 최상단과 `2026-08-24_WIP_*.md` 를 보면 된다.
진행 중인 체인은 `ps -eo pid,ppid,args | grep _run_` 으로 확인한다.
