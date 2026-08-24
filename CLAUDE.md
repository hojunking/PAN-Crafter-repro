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
- 평가 도구는 형제 저장소 두 개가 필요하다 (없으면 학습만 가능)
  - `PANCRAFTER_CANCONV` — 지표 구현 재사용
  - `PANCRAFTER_DLPAN` — MTF/Q2n 공식 구현

## 실행

```bash
./tools/run.sh <config이름>                              # config/pancrafter_<이름>.yaml 또는 config/<이름>.yaml
setsid nohup ./tools/run.sh wv3 > /dev/null 2>&1 &       # SSH 끊겨도 유지 (PPID=1 로 확인)
```

실행 조건은 `work_dir/<실험>/meta/` 에 자동 스냅샷된다.

## 반드시 지킬 규약

- **`results_log/` 의 기존 문서를 고치지 않는다** (`CONVENTION.md` §5). 각 문서는 그 시점의
  스냅샷이다. 결론이 바뀌면 **새 날짜 문서**를 쓰고 "무엇이 왜 바뀌었는지" 를 남긴다.
  예외는 오타·깨진 링크·계산 실수뿐이다.
- **30분 넘는 실험은 WIP 문서를 먼저 만든다** (§2). 사람이 기다리는 동안 볼 수 있어야 한다.
- **모든 수치에 두 가지를 명시한다** — 어느 실행인가(`baseline` 배포본 그대로 / `fixed`
  A-1·A-2 적용), 누가 쟀는가(`py` 학습 중 metrics.csv / `matlab` DLPan 프로토콜).
  **논문 Table 과 비교 가능한 것은 `matlab` 뿐이다.**
- 새 문서를 만들면 `results_log/README.md` 맨 위에 한 행 추가한다.

## 확정된 사실 (다시 파헤치지 말 것)

- **재현 성립.** WV3 reduced ERGAS **2.1633** (배포본 그대로, 50K). 논문 2.040 대비 +6.09%
- **그 격차는 우리 잘못이 아니다.** 평가기는 CANConv 배포 가중치로 논문 행을 6지표 0.5% 이내
  재현하고, 논문 명시 설정은 시드 2,025 까지 전부 일치한다. 배포 코드가 논문 모델이 아닌 것이
  유력하다 (params 1.39×, FLOPs 2.1×). → `results_log/2026-08-24_reproduction-audit.md`
- **논문의 CANConv 대비 우위는 재현되지 않는다** (주장 −5.69% vs 재현 −0.34%, p=0.667).
  **비교 기준선은 논문 2.040 이 아니라 재현 Teacher 2.1633 이다.**
- **프루닝 최적점**: `x1_panbr_dec4` = 6.694M(−33%) / 8.3ms(2.2×) 가 Teacher 와 구분 불가
  (p=0.114). **6M 아래로 내려가야 실제 손실이 생기고, 거기가 KD 의 작업 대상이다.**
- **지표 선택**: ERGAS·SAM 만 판별력이 있다. Q8·SSIM·SCC 는 이 범위에서 포화(±0.15%)라
  판별 근거로 인용하면 안 된다. D_s·HQNR 은 축소하면 거의 항상 좋아지는 기전이 있어 단독 해석 금지.
- **양방향 mutual learning 은 no-go** (`2026-08-20_mutual-learning-go-no-go.md`).
  단방향 T→S 증류는 별개이고 유효하다.

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
