# s5 에이전트 실행 지시: PAN-Crafter QB·GF2 각 1회

사용자가 모델 확인과 실행 준비를 s1에서 완료하도록 했으며, **s5에서는 아래 확정 실험을 실행한다.** 구현 후보를 다시 선택하거나 학습 설정을 재설계할 단계가 아니다. 소스 ZIP을 옮기거나 별도 구현을 만들 필요 없이 이 저장소의 commit을 pull해서 사용한다.

## 실행할 실험

| 항목 | 확정값 |
|---|---|
| 구현 | `pcrepro.model.CorrectedPANCrafter`, `PC_PAPER_EQ11_C128_D224_LN_MARS_v1` |
| 순서 | **QB → GF2**, 각 1회 후 종료 |
| Seed·학습량 | **2025, fresh 50,000 updates** 각각 |
| 구조 | C128, 3-scale D224, LN, Eq11 raw-PAN query/joint K/V, 3×3 unfold, 7,129,476 parameters |
| 학습 | AdamW 1e-4/WD .01, warmup100+cosine, 48 pairs→MARs96, FP32/TF32 off |
| 데이터 | QB **원본** train/val, GF2 원본, 현재 RR20/FR MAT20 |
| 평가 입력·참조 | 모두 original PAN; QB/GF2 corrected PAN 결과는 생성하지 않음 |
| 선택 | primary exact50K, secondary validation ERGAS 최소 |
| 비용 | primary의 RR/FR batch1 time·memory·FLOPs(G)·Param(M) |

`CorrectedPANCrafter`의 Corrected는 attention 구현 수정을 뜻하며, PAN Aligner 보정 입력을 뜻하지 않는다. 배포본 `model/pancrafter.py`, 단독 `PANCrafterPaper`, PAN-Weave로 교체하지 않는다. 과거 무기한 실행기 `tools/pcrepro_start.sh`도 사용하지 않는다.

## 실행 명령

s5의 실제 PAN-Crafter 저장소에서 진행한다. 기존 학습용 Python 환경을 활성화하고 `gspread/server.txt`가 `s5`인지 확인한다. 파일이 없다면 실제 s5임을 확인한 뒤 로컬 marker를 생성한다. 서버 marker는 Git으로 배포하지 않는다.

```bash
git pull --ff-only origin main
python tools/pcrepro_qbgf2_s5.py start
```

`python`은 s5 GPU를 지원하며 기존 PAN-Crafter 의존성이 설치된 환경이어야 한다. 필요한 경우 그 환경의 Python 절대 경로로 호출한다. s1의 PyTorch 2.4/cu118을 s5에 강제로 설치하지 않는다. 기존 `DLPan-Toolbox`가 저장소의 형제 디렉터리에 있으면 자동으로 찾으며, 위치가 다르면 `PANCRAFTER_DLPAN`에 실제 경로를 지정한다.

이 한 명령이 다음을 처리한다.

1. s5 marker, GPU compute 점유, 필수 입력·평가 모듈을 검사한다. 다른 작업은 중단하지 않는다.
2. pull한 **commit의 detached worktree**를 자동 생성하고 `data`·`work_dir`만 연결한다. 인증정보를 복사하지 않는다. 학습 중 원 저장소에서 다른 commit을 pull해도 실행 코드는 고정된다.
3. QB/GF2 전체 데이터 SHA·finite·shape·FR MAT20 순서를 검증하고 고정 설정을 작성한다.
4. 연결 종료 후에도 유지되는 별도 세션으로 유한 runner를 시작한다. 실제 CUDA에서 batch48/MARs96 forward/backward·operator coverage를 통과한 뒤 학습한다.
5. QB50K와 후처리 → GF2 50K와 후처리 → 이관 결과 JSON 저장 후 종료한다.

같은 `start`를 실행 중에 다시 호출하면 기존 PID를 반환하며 중복 실행하지 않는다. 시작한 Python 환경과 데이터·모델 설정을 유지한다. `start`의 LAUNCHED는 runner 시작을 뜻하므로 아래 상태에서 **TRAINING과 update 증가까지 확인**한다.

```bash
python tools/pcrepro_qbgf2_s5.py status
tail -n 30 work_dir/_pcrepro_qbgf2_s5_20261118/logs/launcher.log
```

첫 단계 로그는 `logs/PCPAPER_QB_s5_S2025_F50K_20261118_v1.preflight.log`, 학습 로그는 같은 prefix의 `.train.log`다. 실제 optimizer update는 `status`의 `training.actual_updates`에 나타난다. s5 에이전트는 첫 update 진행을 확인하고, 이후 두 run의 완료/실패를 추적한다. 준비 단계만 성공한 상태를 학습 완료라고 보고하지 않는다.

필수 입력 50개는 `pcrepro/s5_assets.json`에 상대 경로·크기·SHA로 고정되어 있다. s5의 동일한 기존 자산을 재사용한다. 누락 시 서버의 데이터 위치를 확인해 연결한다. 실제 누락·hash 불일치·GPU 점유·OOM·지원하지 않는 CUDA가 있으면 해당 원인을 보고한다. batch·precision·seed·recipe를 자동으로 바꾸거나 다른 실험을 종료하지 않는다. 명시적 입력 경로를 수정할 필요가 있다면 고정 manifest 대신 데이터 위치의 symlink를 맞춘다.

## 정지·재개

```bash
python tools/pcrepro_qbgf2_s5.py pause
# runner 종료와 GPU 상태를 확인한 뒤 동일 Python 환경에서
python tools/pcrepro_qbgf2_s5.py resume
```

Pause는 optimizer update 경계에서 full state를 저장한다. Resume은 **같은 고정 worktree와 optimizer/scheduler/RNG/sampler 상태**로 계속한다. 이미 끝난 QB를 다시 학습하지 않는다. 실패한 후처리도 이어서 처리한다. 직접 `pkill`, 과거 캠페인 watchdog 변경, 기존 결과 삭제는 하지 않는다.

## 완료 판정과 전달할 결과

두 run ID:

```text
PCPAPER_QB_s5_S2025_F50K_20261118_v1
PCPAPER_GF2_s5_S2025_F50K_20261118_v1
```

다음 조건을 모두 확인한다.

- 운영 폴더 `work_dir/_pcrepro_qbgf2_s5_20261118/status.json`: `status=COMPLETE`, `runs=2`.
- 각 run의 `meta/training_status.json`: `actual_updates=50000`, `training_complete=true`.
- 각 run의 `official/summary.json`: `complete=true`, primary/secondary의 RR20·FR20 결과 존재.
- 각 run의 `official/profile.json`: 실제 GPU의 RR/FR time·memory·FLOPs·Param, 미지원 floating 연산 없음.
- `work_dir/_pcrepro_qbgf2_s5_20261118/additional_models_rows.json`: `complete=true`, QB/GF2 두 행.

Primary weight는 각 run의 `checkpoints/exact_50000.json` pointer가 가리키는 디렉터리의 `model.safetensors`다. Identity·두 선택점·지표·비용·원본 prediction 자료를 보존한다. Primary와 secondary를 별도 seed 실험처럼 중복 게시하지 않는다.

s5 에이전트는 완료 시 **commit, 실제 GPU/환경, 두 run ID, weight/summary/profile/이관 JSON 경로, 완료 여부**를 보고한다. s1 이관 후 시트 대상은 `pan-cvpr27-new`의 **`addtional-models`**다. 실행기는 업로드용 의미 필드 JSON까지만 만들며, 과거 `PC-Repro-s5`에 쓰지 않는다. 다른 GPU에서 측정한 latency를 s1 측정값으로 표기하지 않는다.

## 이미 완료된 s1 확인

모델/학습 설정은 로컬 논문 식(9)–(12), §4.1–4.2와 대조했고 당시 PCREPRO의 미기재 세부 선택을 유지했다. QB train17139/val1905, GF2 train19809/val2201 및 RR20/FR20 전체 자산을 검증했다. 실제 C128/D224 CPU forward/backward, prepared/native bit equality, 같은 예측에 대한 현재 RR/FR metric 코드 일치를 확인했다. 기존139 tests에 s5 Git 배포·중복 방지·실패 차단 검사를 추가했다.

구조와 평가의 자세한 검토 근거는 [검토 기록](../results_log/2026-11-18_pcrepro-qbgf2-s5-guide.md)에 있다. 실제 s5 GPU 검사와 50K 실행은 이 서버에서 수행할 남은 작업이다.
