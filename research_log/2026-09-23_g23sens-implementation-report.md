# G23 SENS s4/s5 구현·검증

사용자 제공 `PANDA_G23_SENS_S45_UNLIMITED_2026-09-23`을 기준으로 구현했다.
제공 번들은 수정하지 않았으며 manifest SHA는
`759a54764ec3117cc0c7197b7079bc82778cfa115fa078a12c74ae500fca6243`이다.
기준 commit은 `52184d5578a2a10721e435f8fbf4a759497768e5`이고,
실행 시 구현 commit/content SHA를 별도로 고정한다.

## 구현

- 새 `g23sens/` 및 `tools/g23sens_*`만 추가. 기존 실험 코드/큐/결과/Sheet는 변경하지 않았다.
- s4: BASE, α×2, β×2, edge×2. s5: BASE, q_ref×2, τ×2, A LR×2.
  서버마다 7개를 완료한 뒤 새 seed로 반복한다. 다른 서버 완료·시간예산·성능 threshold는 없다.
- G23 원본 P0/W104D121 초기화, native 입력, T0 A 독립 복사, FP32·batch48·50K·AdamW·cosine 유지.
  원본 분리 gradient 메서드와 실제 gradient를 대조했다.
- qraw 고정/q-weight 재계산, tau 1회 배율, 실제 optimizer A LR 검증.
  같은 서버/cycle의 initial U/A 및 50K 전체 sample/augmentation stream을 고정한다.
- fullstate optimizer/scheduler/RNG/sampler 복구. 부분 optimizer 변경 후 예외 시 직전 정상
  checkpoint를 보존한다. 별도 attempt, 최대 2회 기술적 재시도, BASE 실패 중지,
  variant 발산 기록 후 진행, 평가만 재개, 업로드 outbox를 분리했다.
- 실제 train forward/gradient에 영향을 주지 않는 고정 probe를 0/1K/10K/25K/50K에 기록한다.
- 이번 계획의 EXACT50K primary와 native val-ERGAS minimum secondary를 별도 평가한다.
  같은 A/U checkpoint로 RR/FR 전체 평가 후 검증된 COMPLETE receipt를 발행한다.
- 전용 Sheet 탭 `SENS-G23-WV3-s4/s5`, RMSE/CC/JQM 포함, 원값 full precision·표시 4자리,
  날짜·학습시간·선택·attempt·해시·동일 서버/cycle BASE Δ 및 readback을 구현했다.
- Docker는 로컬 image content ID, Git frozen worktree, 선택 GPU UUID로 실행한다.
  알려진 PCREPRO는 현재 run의 학습/평가 완료를 확인한 후 인계한다. 기존 STOP_NOW_SAFE는
  보존하고, 알 수 없는 구형 runner나 미완료 active run은 강제 종료하지 않는다.

## 수행한 검증

1. 새 통합 CPU/offline 테스트 **81개 통과**.
2. 제공 case generator/receipt 테스트 **36개 통과**.
3. 재사용 PCREPRO 평가·학습 primitive 회귀 테스트 **21개 통과**.
4. 새 Python 27개 구문 검사 및 shell launcher 2개 `bash -n` 통과.
5. 실제 로컬 T0 패키지, cue 38,856개 뷰, calibration, 원본 H5/LP 8개 전체 읽기·SHA 검증 통과.
   train 9,714 / val 1,080 / RR 20 / FR 20이다.
6. 실제 T0·WV3·cue를 사용한 CPU forward/U-A gradient 검증 통과.
   U 1,903,624 params, A 105,330 params, Teacher gradient 없음.
7. 실제 원본 4-worker DataLoader의 index/rotation과 3 epoch 대조, 50K×48 전체 stream
   determinism 및 cursor 재개 검증. sandbox의 tensor IPC 제한 때문에 해당 sampler 테스트만
   NumPy metadata transport를 사용했으며 실제 GPU 실행으로 주장하지 않는다.
8. Docker mount 구성은 실제 로컬 자산 경로를 resolve하는 읽기 전용 검사까지 수행했다.

실행 명령과 세부 계약은 `g23sens/README.md`에 있다. 원본 자산 근거는
`2026-09-23_g23sens_asset_audit.json`, 수치 검증 근거는
`2026-09-23_g23sens-numerical-verification.json`에 기록했다.

## 미수행·해석 제한

- 실제 GPU smoke/50K 학습, 원격 s4/s5 자산 확인, 기존 실행 인계, 실시간 Sheet 쓰기는
  이번 구현 작업에서 수행하지 않았다. **학습을 시작했다고 보고하지 않는다.**
- 시작 명령에는 실제 CUDA smoke·split-gradient·resume 및 전체 native RR/FR 평가가 자동
  연결되어 있다. 실패를 무시하고 본 실험을 시작하지 않는다.
- 기존 파일과 측정 코드는 고정했지만 독립적인 논문 scene correspondence 검증은
  이번 작업 범위에서 다시 수행하지 않았다. PAPERSET_IDENTITY_UNVERIFIED로 표시한다.
- JQM은 SRF-substitute이며 SIPSA 동일 구현으로 주장하지 않는다.
- 고정 T0 조건부 Student-seed 민감도다. 다른 서버 BASE를 독립 seed로 중복 합산하거나,
  Teacher seed 일반화·다중 인자 상호작용으로 해석하지 않는다.

## 서버 실행

```bash
# push/pull 후 각각 해당 서버에서
bash tools/g23sens_docker_start.sh --server s4
bash tools/g23sens_docker_start.sh --server s5
```

Git pull 자체는 실행 명령이 아니다. 사용자 제공 자산이 없는 서버에서는 임의 대체/재생성하지
않고 필요한 경로와 해시 불일치를 보고한다. 무관한 기존 worktree 변경은 커밋에 포함하지 않는다.
