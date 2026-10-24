# FOLLOW30 full-scale 큐 전환·s3 인계

2026-10-24 사용자 승인: 진행 중 캠페인에서 불필요한 case를 제외하고, 새 full-scale case를 연결해 Git으로 전달한다.
현재 실행 원본은 [FOLLOW30 README](../panda_follow30/README.md),
[s1 CSV](../panda_follow30/cases_s1.csv), [s3 CSV](../panda_follow30/cases_s3.csv)이다.

## 새 실험 범위

`PANDA_S13_FOLLOW_HARDONLY_20261024_v1`, **전 case Aligner scale 1.0**.
Student Lanczos4, 기존 frozen Aligner/Teacher·GT Hard를 유지한다.
승인된 DIFF 입력과 UBIL 구조 및 기존 LR/Hard/schedule만 비교한다.
SWA·EMA·soup·rotation loss·TTA·band-weighted loss는 없다. 단일 학습 checkpoint를 평가한다.

| case | 설정 | s1 | s3 |
|---|---|---|---|
| S100DIFFC50 | DIFF, 50K, LR1e-4 | R1·R2 | R1·R2 |
| S100DIFFLR150C50 | DIFF, 50K, LR1.5e-4 | — | R1·R2 |
| S100DIFFH050C50 | DIFF, 50K, LR1e-4, Hard α=.125·γ=.25 | — | R1·R2 |
| S100DIFFUBILC30 | DIFF+bilinear up, 30K, LR1e-4 | R1·R2 | R1·R2 |
| S100DIFFF15C30 | NEXT30 동일 seed DIFF 15K에서 추가15K, LR2e-5→1e-6 | R1·R2 | R1·R2 |
| S100DIFFC30 | DIFF 30K 세 번째 seed | R3 | R3 |
| S100C30 | corrected-only 30K 동일 seed 대조 | R3 | R3 |
| S100UBILC30 | corrected-only+bilinear up 30K | R1·R2 | — |

**s1 10개·310K 새 update, s3 12개·450K, 총 22개·760K.**
DIFF 18개, corrected-only 대조 4개. 서버 간 donor가 다르므로 seed 반복으로 합치지 않는다.

## 기존 NEXT30에서 남길 것과 제외할 것

정책은 서버마다 현재 상태를 읽어 결정한다. 과거 snapshot의 번호를 강제로 적용하지 않는다.

1. 이미 시작한 모든 run은 원래 horizon까지 완료한다. 완료 결과·checkpoint·Sheet 행을 삭제하지 않는다.
2. 아직 시작하지 않았어도 **S100C30·S100DIFFC30의 R1·R2는 유지**한다.
   DIFF R2의 15K checkpoint가 FOLLOW 분기의 정확한 부모이기 때문이다.
3. 그 밖의 **미착수 scale<1 case만** 기존 `pending_control.json` 프로토콜로 제외한다.
   시작 경합이 생긴 case는 유지하며 watcher가 pending 요청에서 제거한다.
4. NEXT30가 정상 종료하고, 완료 checkpoint와 승인된 skip 기록·필수 15K 부모를 검증한 뒤 FOLLOW30를 시작한다.
   failed/stopped run이나 불명확한 skip 기록을 성공으로 간주하지 않는다.

s1 확인 snapshot(2026-10-24 20:04 KST): 완료 10개, `S010C30 R2` 12K 진행 중. 미착수 7개 중 아래 **5개·190K**를 제외한다.

| 제외 case | 절약할 새 update | 이유 |
|---|---:|---|
| S020BASE R2 | 50K | 낮은 scale 50K 반복에서 full-scale 주력으로 전환 |
| S010BASE R2 | 50K | 같은 이유 |
| DAVG R2 | 30K | 낮은 scale의 downsample 확장 탐색 보류 |
| UBIL R2 | 30K | 새 full-scale UBIL 대응 실험으로 자원 이동 |
| DAVG_UBIL R2 | 30K | 낮은 scale의 결합 구조 확장 보류 |

`S010C30 R2`가 끝나면 NEXT30의 `S100C30 R2`와 `S100DIFFC30 R2`를 완료하고 새 s1 10개를 시작한다.
s3는 로컬 실행 상태를 이 세션에서 직접 보지 않았으므로 제외 개수를 단정하지 않는다.
같은 정책이 **s3에서 아직 시작하지 않은 case에만** 적용된다. 이미 끝난 낮은 scale 결과도 그대로 둔다.

### s1 적용 기록

2026-10-24 20:03 KST에 `start_panda_follow30_transition.sh s1` 실행을 완료했다.
워처 PID는 `3465161`, receipt 상태는 `WAITING_COMPLETE_PREVIOUS_QUEUE`이다.
위 5개 group의 제외 **요청을 등록한 상태**이며, NEXT30 trainer가 group 경계에서 소비할 예정이다.
현재 trainer에 중단 신호를 보내지 않았고(`trainer_signalled=false`), FOLLOW30 GPU 학습은 아직 시작하지 않았다.
기존 NEXT30 source pin 316개가 모두 일치함을 확인했다.

요청 ID: `20261024_follow30_full_scale_pending_cutover`.
적용 상태는 `work_dir/panda_follow30_launchers/s1/transition_receipt.json`,
소비 여부는 NEXT30의 `campaign_state.json` 내 `pending_amendments`에서 확인한다.

## s3 적용 절차

이 변경이 포함된 Git revision을 가져온 뒤 서버의 기존 학습/uploader Python 환경을 사용한다.
기존 환경·데이터·credential·NEXT30/Mature30 checkpoint가 필요하며 Git에는 넣지 않는다.
`gspread/server.txt`가 `s3`인지 확인한다. 기존 NEXT30 code pin 파일은 이 변경에서 수정하지 않는다.

읽기 전용 확인:

```bash
python -m panda_follow30 catalog --server s3 --csv
python -m panda_follow30 audit-sources --server s3
python -m panda_follow30.transition --server s3 --prepare-only
```

위 source audit의 `WAITING_REGISTERED_NEXT_PARENT`는 필요한 R2가 아직 학습 전이면 정상이다.
다른 seed/step의 checkpoint로 대체하지 않는다.

실제 제외 요청 및 자동 후속 전환 등록:

```bash
bash scripts/start_panda_follow30_transition.sh s3
```

이 명령은 현재 학습을 중단하지 않는다. 해당 서버의 미착수 low-scale만 제외하고, 유지한 full-scale이
끝나면 GPU가 비었는지 확인한 뒤 비용 측정·CUDA preflight·FOLLOW 학습·publisher를 연결한다.
기존 이미지와 로컬 자원을 사용한다. `PAN_HOST_PYTHON`은 필요할 때 기존 환경의 Python 절대경로로 지정한다.

관측 파일:

- `work_dir/panda_follow30_launchers/s3/transition_receipt.json`: 적용 요청·대기·전환 상태
- `work_dir/panda_follow30_launchers/s3/watcher.log`: watcher 로그
- `work_dir/panda_next30/PANDA_S13_NEXT_HARDONLY_20261023_v1/s3/WV3/pending_control.json`: 제외 요청
- 위 NEXT30 `campaign_state.json`의 `pending_amendments`: trainer가 실제로 소비한 skip 증거
- `work_dir/panda_follow30/PANDA_S13_FOLLOW_HARDONLY_20261024_v1/s3/WV3`: 새 결과

요청을 써 둔 상태와 trainer가 실제 skip을 소비한 상태를 구분한다. watcher는 같은 요청에 대해 재등록 가능하다.
Git commit만으로 s3 프로세스가 바뀌지는 않으므로 s3 에이전트가 위 명령을 실행해야 한다.

## 선택 근거와 논문 해석

동일 s1·R1·30K·Lanczos4 공식 평가:

| 구성 | HQNR | ERGAS |
|---|---:|---:|
| scale .1 / corrected-only | .956299842 | 2.067808856 |
| scale .2 / corrected-only | .955780486 | 2.059521402 |
| scale 1 / DIFF | .957647208 | 2.052012362 |
| scale 1 / corrected-only | .916942691 | 2.039417991 |

원본은 각 NEXT30 run `attempt_01/evaluation_0030000.json`이다.
s3의 2026-10-24 Sheet snapshot에서 DIFF R1 대표29K는 .958686915/2.049594457,
R2 대표15K는 .958685709/2.087480068이다. 서로 다른 대표 step은 동일 학습량 비교로 취급하지 않는다.
[결과 Sheet](https://docs.google.com/spreadsheets/d/1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE/edit).

DIFF는 낮은 scale과 경쟁할 수 있으나 목표 동시 통과가 확정된 구성은 아니다.
DIFF 입력은 원 PAN 정보도 보유한다. 기존 추론 probe에서 원 PAN만 주면 RR ERGAS가
2.0520→2.1268로 악화했고 HQNR는 거의 같았다. 1.5px PAN 어긋남 주입 시 DIFF ERGAS 4.5074,
corrected-only 2.2225였다. 입력 개입의 분포 변화도 있으므로 보정 기하만의 인과 효과로 단정하지 않는다.
**명목 scale 1과 실질적인 Aligner 활용은 별도 평가 항목**이다.

DIFFC50/C30 종료점 차이는 LR 경로+학습량, H050은 α·γ의 공동 변경이다.
한 seed의 ERGAS 경계값으로 우열을 확정하지 않고 대응 step·곡선·추가 seed를 함께 본다.

## 게시·검증

기존 **같은 checkpoint HQNR ≥ .9585 AND ERGAS ≤ 2.040** 공동 통과 전체와 run별 대표를 유지한다.
대표는 HQNR 충족 중 최소 ERGAS, 없으면 최고 HQNR. 하이퍼파라미터와 DIFF/UBIL/fork는 Approach에 표시한다.
평균 모델은 생성·게시하지 않는다. 제외한 미착수 case의 가짜 성능 행을 만들지 않는다.

최종 CPU smoke **28개 통과**, launcher **7개 shell syntax 검사 통과**.
선택적 skip·시작 경합·재등록·source pin 변경 거부·동일 seed 15K 부모 검증,
분기 학습의 전체 상태 재개와 Sheet 선택 규칙을 포함한다.
실제 s1 상태의 읽기 전용 전환 점검도 통과했다. 새 GPU preflight는 후속 캠페인 시작 때 실행한다.

학습 데이터·checkpoint·credential은 이번 commit에 포함하지 않는다. 기존 분석 파일의 미추적 자료나
다른 작업 변경도 포함하지 않는다. 실행 코드, 서버별 case CSV, launcher, CPU 테스트, 이 인계 문서만 전달한다.
