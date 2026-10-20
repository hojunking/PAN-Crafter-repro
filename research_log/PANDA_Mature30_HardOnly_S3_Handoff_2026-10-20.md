# Mature30 Hard-only — s3 전달 사항 (2026-10-20)

사용자 결정: 다음 Stage2부터 Teacher Advantage 기반 Selective Soft를 제거한다.
이번 변경은 s1/s3 Mature30 신규 캠페인에 적용된다. 구 캠페인의 결과·설정·weight를
덮어쓰거나, 이미 실행 중인 프로세스의 loss를 도중에 바꾸지 않는다.

## s3 실행

현재 case를 완료하고 구 큐의 후속 실행을 정상 종료한 뒤, GPU가 비어 있는 상태에서:

```bash
git pull --ff-only origin main
bash scripts/start_panda_mature30_costed.sh s3
```

- 먼저 s1의 이 커밋을 원격에 push해야 s3에서 받을 수 있다.
- **실행 중인 공유 작업 디렉터리에 바로 pull하지 않는다.** 코드 SHA 고정 검증으로
  구 실행이 중단될 수 있다. s3 세션에서 현재 case 완료와 구 controller 종료를 확인한다.
- 위 실행 명령 하나가 기존 로컬 Docker 이미지로 독립 cost 측정, preflight, 학습,
  호스트 Sheet publisher를 시작한다. 별도 smoke/승인 마커를 수동 생성할 필요는 없다.
- `gspread/server.txt`는 `s3`여야 하며, 기존 s3 데이터·평가 도구·Sheet 인증을 사용한다.
- 기존 warp30 Docker 이미지가 있으면 재사용한다. 자동 탐색 가능한 컨테이너가 없다면
  `PAN_MATURE30_IMAGE`로 s3에서 이미 사용하던 로컬 이미지를 지정한다.
  환경 교체나 이미지 pull은 자동으로 하지 않는다.
- 컨테이너: `panda-mature30-hardonly-wv3-s3-v2`.
- `git pull`만으로 실험이 시작되지는 않는다. 실행 명령을 명시적으로 호출한다.
- 구 컨테이너·weight는 자동 삭제하지 않는다. 이미 같은 v2 컨테이너가 있으면 중복
  실행을 거부하므로, 무조건 삭제하지 말고 해당 실행 상태를 확인한다.

## 실험 내용

- 캠페인: `PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2`
- s3는 **서버 자체 Stage1을 처음부터 50K 학습**한다. s1 weight 복사는 필요 없다.
- Stage1: 8밴드 평균 정렬 목표, interp23 base. validation
  `RRE50 → RRE90 → ERGAS → 빠른 step` 순으로 donor를 선택한다.
- Stage2: FULL R1 → HALF R1 → DOWN R1 → LOW R1.
- 잔여 시간으로 HALF/DOWN R2 묶음 → CAP085 R1 → CAP085 R2를 조건부 진행한다.
- 각 등록 run의 목표는 50K다. s3 신규 v2는 최초 기동부터 30h 벽시계 예산이며
  마지막 30분을 closeout에 사용한다. 재시작해도 같은 v2의 마감은 리셋하지 않는다.
  s1의 기존 시작 시각 승계는 s1 로컬 상태에만 적용한 조치다.
- 서버 간 완료 대기는 없다. 같은 서버의 중복 writer/controller 방지 lock만 유지한다.

Stage2 loss는 기존 MASK2 reduction으로 다음 Hard 항만 사용한다.

```text
L = reduce[(1 + 0.25*d_e + 0.5*d_e*r_q) * e_S]
alpha=0.25, beta=0, gamma=0.5
```

Soft 계수를 0으로 곱하는 것뿐 아니라 advantage gate와 Student–Teacher
Soft loss/gradient 계산을 생략한다. Teacher는 `e_T/d_e`와 calibration용으로
동결 유지한다. q 정의, GT 좌표, 정합 배율 정책, 업샘플러, seed, optimizer는 유지한다.
구 실험 재현을 위한 공용 beta>0 경로는 남지만, 신규 Mature30은 beta>0 설정을 거부한다.

## 평가·업로드

- 기존 새 workbook의 `WV3-ablations`, server=`s3`.
- Stage1: RRE로 선정된 모델의 공식 평가 결과 1행.
- Stage2 (**2026-10-21 사용자 변경**): run/seed/attempt별 **현재까지 HQNR 최고
  checkpoint 1행만 갱신**한다. 목표치 통과 여부는 선택에 영향을 주지 않는다.
  동점은 ERGAS가 낮은 순, 이어서 빠른 step 순이다. FINAL50K 별도 행은 게시하지 않는다.
  5K 고정이 아니며, 뒤의 평가에서 HQNR 최고값을 경신하면 모든 지표·cost를
  그 checkpoint 기준으로 함께 교체한다.
- **매 1,000 step 전체를 Sheet에 누적하지 않는다.** 평가 주기와 게시 조건은 별개다.
- RMSE·CC·JQM을 포함한 전체 평가 결과와 원래 정밀도는 로컬에 보존한다.
- run 이름은 `NOSOFT`, Approach는 `NO-SOFT`, beta 열은 `0`으로 구분한다.
- 결과 경로:
  `work_dir/panda_mature30/PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2/s3/WV3`
- publisher 로그: `work_dir/panda_mature30/publisher-s3-best-hqnr-v2.log`.

## 동봉 수정·검증

- checkpoint의 architecture tuple과 외부 JSON list를 동일 JSON 표현으로 비교해,
  데이터가 같아도 donor 검증에서 실패하던 문제를 수정했다.
  metric·모델 상태·원본 파일 SHA 검증은 그대로 유지한다.
- s1/s3 전체 등록 case의 beta=0, 기존 Hard 항 보존, Soft gradient=0,
  beta>0 신규 캠페인 거부, s3 Stage1/resume 및 게시 선별 규칙을 테스트한다.
- 커밋 전 재검증: Mature30 **57개**, 공유 loss/calibration/mask/training-evaluation
  **60개** 테스트 통과. 두 실행 스크립트의 `bash -n`도 통과했다.
- s1의 실제 Docker/CUDA 1-step smoke와 운영 학습 로그에서도 Soft=0,
  `loss == hard_weighted`를 확인했다. s3 GPU 실행은 s3 세션에서 확인해야 한다.

관련 s1 실행 기록:
[Hard-only 전환 WIP](../results_log/2026-10-20_WIP_s1_mature30-hardonly.md)

## 후속 수정: cost 열 누락

초기 Mature30은 `publication.project()`에 cost 연결용 hook만 있었고 측정·연결
모듈이 없어 cost 열이 비었다. `panda_mature30.cost` 패키지로 hook을 구현했다.
이 후속 수정은 위 Hard-only 최초 커밋과 별도이며, 새 코드를 전달받아야 사용할 수 있다.

- 신규 실행은 위의 **`start_panda_mature30_costed.sh`**를 사용한다.
  기존 `run_panda_mature30_docker.sh`만 실행하면 cost를 새로 생성하지 않는다.
  신규 캠페인의 30h 시작 시각은 cost 준비 시작 때 기록하고 재실행해도 리셋하지 않는다.
- 이미 같은 캠페인이 있는 경우 cost만 준비하는 명령은
  `bash scripts/prepare_panda_mature30_costs.sh s3`다.
  실행 중인 trainer를 자동 종료하지 않으며, 다른 GPU 작업이 있으면 측정을 거부한다.
  현재 case/큐의 종료·재개는 해당 서버 세션에서 처리한다.
- Stage1 A+T와 Stage2 A+S를 각각 측정한다. Stage2 학습용 Teacher는 cost에서 제외한다.
  DOWN의 0.5/0.375/0.25 및 CAP085까지 모든 실제 추론 경로를 사전 측정한다.
- 서버별 GPU 독립 측정이며 다른 서버의 latency·memory를 복사하지 않는다.
- Sheet 기본 cost는 **RR 256×256, batch 1, FP32**. FR 512×512도 측정하고
  로컬 profile/Sheet provenance에 함께 남긴다.
- `FLOPs(G)`는 기존 시트와 동일한 **부분 MAC-equivalent 규모**다.
  완전한 arithmetic FLOPs가 아니며 계수/제외 연산은 `Cost_scope`에 명시한다.
  `Mem(MB)` 역시 기존 규약대로 PyTorch peak allocated **MiB**다.
- Train/Eval/Wall은 선택된 step까지의 실측이다. 재사용 donor는 **원본 run**의
  학습·평가·wall 기록을 표시하며 신규 캠페인의 추가 학습 시간으로 이중 청구하지 않는다.
- 파라미터/학습 결과/optimizer를 변경하지 않는 sidecar다.
  기존 7개 cost 열을 채우고, 성능 지표와 선별 규칙은 그대로 유지한다.
- 후속 재검증: cost 테스트 포함 Mature30 **65개 통과**.

## 후속 변경: Stage2 HQNR-best 단일 행 (2026-10-21)

이미 같은 v2 학습이 진행 중이면, **이 후속 변경을 받은 뒤** 업로더만 교체한다.

```bash
bash scripts/start_panda_mature30_best_publisher.sh s3
```

- 별도 호스트 진입점은 `python -m panda_mature30.best_publication`이다.
  기존 `panda_mature30.publication`은 진행 중 학습의 소스 해시를 보존하기 위해
  수정하지 않았다. 구 진입점을 실행하면 구 게시 정책이므로 사용하지 않는다.
- 스크립트는 같은 서버·캠페인·명령행의 구 Sheet watcher만 확인 후 교체한다.
  trainer/container, optimizer, seed, budget, checkpoint에는 손대지 않는다.
- 신규 `start_panda_mature30_costed.sh`는 학습 기동 후 새 업로더로 전환한다.
- 기존 조건 통과/FINAL50K 추가 행은 원본 평가와 SHA가 확인된 **해당 서버의 행만**
  백업 후 정리한다. 행 삭제·정렬 없이 초과 셀만 비우므로 다른 서버 행 위치는 보존된다.
- Stage1 RRE 선택/표시는 그대로다. 모든 step별 원본 기록과 weight도 보존한다.
- source-pin 대상은 이 변경에서 수정하지 않지만, 다른 미적용 커밋까지 함께 pull할
  경우는 별개다. s3 세션에서 실행 중 admission의 소스 해시와 대조해야 한다.
- 검증: Mature30 **76개 테스트 통과**. 새 최고값의 동일 행 갱신, 목표치와 독립된
  선택, 5K 고정 없음, FINAL50K 추가 없음, Stage1 불변, 이전 행 안전 전환,
  다른 서버 보존, 비용의 selected-step 연결, Python 3.9 업로더 교체를 포함한다.
