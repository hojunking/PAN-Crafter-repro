# MIX20H 구현·실행 인계

대상: `PAN_AllServers_20H_MixedSeed_HQNR9585_ERGAS_Plan_2026-09-18.md`.

**상태: 구현 및 로컬 CPU 검증 완료. 캠페인은 아직 활성화하지 않았고, 새 학습·원격 배포·실제 Sheet 쓰기를 실행하지 않았다.**

**후속 지시 반영 — 실행 절차 간소화:** `bash tools/mix20h_start.sh` 한 번으로 로컬 서버의 config 준비, 이전 평가 hold 인계, 시간 설정, 학습·평가·업로드 runner를 기동한다. 수동 prepare·hold 해제·Git 커밋·GPU smoke 완료는 시작의 선행 조건이 아니다. 무결성 검사는 시작 명령 안에서 자동 수행한다.

## 1. 구현 범위

- `kdv/mix20h_plan.py`: 46개 실제 run ID의 명시 registry. 기본 36개 + 시간 예비 10개이며, 46개를 모두 필수 예약하지 않는다. 같은 서버·seed의 G23/B20A03를 계획의 AB/BA 순서대로 실행한다.
- `tools/gen_mix20h_configs.py`: 전용 v4 YAML 46개와 `config/queues/qrc24_mix20h_s1.txt` … `s5.txt` 생성·검증. 기존 generator의 `--mix20`도 이 경로로 연결했다. 과거 v1/v2/v3 YAML은 변경하지 않았다.
- `assets/mix20h/reference_recipe.json`: 기존 G23 S1234 실행에서 관측한 optimizer/precision/scheduler, 데이터·cue 해시를 고정했다. 학습 시작 시 설정 문자열뿐 아니라 실제 생성된 객체와 자산도 대조한다.
- `kdv/mix20h_runtime.py`, `train_kdv.py`, `main.py`: opt-in 학습 검증, 초기 U/A 전체 해시, 첫 16개 배치의 index/augmentation 일치, candidate identity, 정확 재개, 예산 정지 연결. 기존 `qrecon_continuous_v1` loss와 gradient routing은 유지했다.
- `tools/mix20h_runner.py`, `tools/mix20h_start.sh`, `tools/mix20h_switch.sh`: 자동 최초 시작 시각과 불변 +20h 마감, 다음 pair만 예약, 안전한 기존 큐 인계, 종료/미완료 상태와 보고서. 공통 T0 지정도 옵션으로 유지한다.
- `tools/mix20h_launch_config.py`: 누락된 config를 자동 생성하고, 미시작 config의 checkout 경로만 원본 백업 후 보정한다. 실험 계수 변경이나 이미 시작한 run의 config 교체는 허용하지 않는다.
- `tools/mix20h_postrun.py`: 50개 grid 검증 → 명시적 v2 선택 → exact50K 공식 평가 → 선택적 업로드. 평가 실패는 학습 재실행으로 이어지지 않는다.
- `gspread/mix20h_upload.py`: 실제 header와 서버 gid 확인, 기존 Legacy/NOA 값 보존, M20/Target/Exact50K 열 추가, full-precision read-back 확인.
- 기존 runner/watchdog/waiter/switch/upload 진입점은 M20 manifest가 존재할 때만 전용 경로로 전환한다. manifest 생성 전에는 기존 운용을 유지한다.

계획이 언급한 동봉 CSV는 현재 작업 트리에서 찾지 못했다. 따라서 §9.1이 허용한 **동등한 명시 tuple registry**로 MD의 46개 exact ID·순서를 구현하고 테스트했다. 동봉 CSV의 SHA256을 검증했다고 주장하지 않는다.

## 2. 선택·학습 불변 조건

- 새로운 campaign target는 `HQNR9585_ERGAS2040_v2`: A_ON raw-original HQNR ≥ 0.9585인 후보 중 공식 RR ERGAS 최소. 동률 순서와 ERGAS < 2.040의 엄격한 공동목표는 계획대로 적용한다.
- 기존 `best_hqnr`는 허용 band 안의 fSCC/step 동률 처리까지 그대로 유지한다. 이것은 엄밀한 raw H 최댓값과 다를 수 있으므로 보고서는 `legacy_best`와 CSV의 실제 `raw_max`를 따로 기록한다. **Target / legacy best / raw-max / exact50K를 구분**하며 H와 E를 다른 checkpoint에서 조합하지 않는다.
- 후보 grid는 1010, 2020, …, 49490, 50000의 50개. 누락·중복·비유한 값·checkpoint/config/data/evaluator 불일치 시 평가 완료로 인정하지 않는다.
- 모든 적격 후보의 공식 RR가 있어야 target를 확정한다. `no_eligible`는 정상 평가 종결이다. 미완료 RR, exact50K 미완료, 업로드 장애는 각각 별도 상태다.
- T0 고정 checkpoint, fresh W104D121 U 초기화, cloneT0 A 초기화, 50K/batch48/AdamW/cosine/warm100, 학습률과 q/τ/λE는 고정했다. β만 G23=0.1, B20A03=0.2다.
- Candidate에는 모델·평가 config·FR 데이터·evaluator 해시와 A_ON/precision을 기록한다. 실행 manifest에는 Git release/dirty 상태 외에 신규 미추적 소스까지 포함한 구현 파일별 해시도 기록한다.
- 이 평가 세트는 반복적인 모델 선택에 사용된다. 결과를 미사용 test set의 독립 성능으로 해석하지 않는다.

## 3. 바로 시작 — 추가 수동 준비 절차 없음

최신 코드를 받은 각 서버에서 같은 명령을 실행한다. `gspread/server.txt`의 s1–s5 식별자를 자동으로 읽으며 `s3(5090)`도 처리한다.

```bash
bash tools/mix20h_start.sh
```

이 명령은 config 생성/경로 보정 → 데이터·Teacher·cue 검사 → 이전 큐/hold 백업·인계 → background runner 기동을 수행한다. 시트 업로드도 기본 연결된다. `--no-upload`로 로컬 결과만 남길 수 있다. `--server sN`은 자동 식별 파일이 없는 신규 서버에서만 필요하다. 다른 서버의 로컬 식별자와 충돌하면 실행하지 않는다.

서버별 기본/예비 수는 s1 6+2, s2 6+2, s3 10+2, s4 10+2, s5 4+2다. 코드 배포 자체를 자동화하거나 다른 서버로 SSH하지는 않는다. 수동 Git 커밋 여부로 로컬 학습을 막지 않으며, 실제 실행 파일 해시는 기록한다.

최초 start는 인식된 이전 평가 hold를 `released_holds/`에 보관하고 인계하므로, 기존 ASV/NOA의 완료를 수동 선행 조건으로 요구하지 않는다. 과거 평가가 완료됐다고 거짓 기록하지 않는다. 실행 중인 evaluator/GPU 작업이 있으면 자동 대기하며 강제 종료하지 않는다. 준비 중 새로 교체된 hold나 M20 시작 후 새로 설정한 hold는 보존한다.

시간 정책도 후속 간소화 지시에 맞춰 변경했다. **기본은 각 서버에서 최초 start한 시각부터 20시간**이다. 원안의 모든 서버 공통 마감과 다르며 manifest에 `clock_policy=local_first_start_20h`를 명시한다. 반복 실행/재개는 기존 시각과 마감을 유지한다. 공통 마감을 원할 때만 아래 옵션으로 동일 T0를 전달한다.

```bash
# 선택 사항: 전 서버 공통 시간 창
bash tools/mix20h_start.sh --start-at '<COMMON_T0_ISO_WITH_UTC_OFFSET>'
# 선택 사항: 활성화하지 않고 동작·자산 확인만
bash tools/mix20h_start.sh --dry-run --check-assets
```

`mix20h_switch.sh`도 같은 간편 시작 경로다. `--launch`를 별도로 줄 필요가 없다. 기존 엄격한 prepare API는 `python tools/mix20h_runner.py prepare ...`로 남아 있으나 일상 실행에 필요하지 않다. 기존 watchdog는 재사용하고, resource 대기는 runner 자체가 처리하므로 새 cron 설치도 선행 조건이 아니다. `RUNNER_SUBMITTED`는 기동 요청 상태이며 실제 학습 중인지/자원 대기인지 `status`와 로그로 구분한다.

## 4. 인계·재시작·마감

- 실제 prepare에서 이전 queue/handover/deadline/hold/lock 및 로컬 실행 inventory를 `work_dir/_qrc24_mix20h/migration_manifest.json`과 `before/`에 보존한다. 이전 checkpoint나 init cache를 삭제하거나 새 seed로 재사용하지 않는다.
- 정상 실행 중인 기존 case는 중단하지 않는다. 이미 실행 중인 구형 shell loop에는 과거 시각의 **legacy barrier deadline**을 써서 다음 case로 넘어가지 않게 하고, 새 runner는 같은 chain lock을 기다린다. 실제 M20 deadline의 기준은 `plan_manifest.json`이다.
- 준비 도중 중단되면 동일 T0로 prepare를 다시 실행해 복구한다. activation state가 `ready`가 되기 전에는 새 학습을 허용하지 않는다.
- 학습 재시작은 동일 run의 완전한 Accelerate 상태만 허용한다. 모델·optimizer·scheduler·RNG·배치 generator·epoch 상태 중 일부만 존재하는 checkpoint는 배제한다. 평가가 완전히 commit된 candidate도 재개 대상으로 사용해 오래된 epoch checkpoint로 돌아가 grid를 중복 기록하는 문제를 막는다. 더 최신 grid 기록이 남은 상태에서 오래된 checkpoint로 재개하려는 경우에는 이력을 삭제하지 않고 거부한다. 무단 fresh 재시작은 차단한다.
- Grid 평가가 끝난 뒤 같은 candidate 경로에 post-evaluation RNG/optimizer 상태를 저장하고 7개 상태 파일의 SHA를 갖는 commit marker를 마지막에 쓴다. CSV는 기록됐지만 commit되지 않은 중간 장애는 자동 덮어쓰기하지 않고 명시적으로 막는다.
- 50K update는 끝났지만 마지막 평가/export가 중단된 경우, 정확한 step50000 상태로만 마무리를 재개한다. 이 경로는 optimizer update를 추가하지 않는다. 이미 commit된 마지막 평가는 중복 수행하지 않는다. 완전한 상태가 없으면 평가 pending으로 보존한다.
- 첫 두 개의 온전한 새 run 이후 보수적인 실측 end-to-end 시간으로 예약을 갱신한다. 재개 run의 짧은 잔여 시간은 전체 50K 실행시간의 표본으로 사용하지 않는다.
- deadline에는 새 run을 시작하지 않는다. 진행 중이면 optimizer update 경계에서 완전 resume checkpoint를 저장하고 exit 5/`stopped_budget`으로 정지한다. 이는 완료 50K run이 아니다.
- 최종 export 중 마감이면 현재 메모리에 올라온 best 모델을 50K resume state로 잘못 저장하지 않고, 이미 저장된 권위 있는 last 상태를 보존한다.
- 종료된 manifest를 유지하므로 이전 24h 자동 보충·41xxx·구형 extra가 되살아나지 않는다. 기간 연장은 이 runner의 자동 동작이 아니며 새 결정이 필요하다.
- 매 case 경계에서 새 eval hold와 GPU 사용자를 다시 확인한다. 실행 중 새 hold가 생겨도 다음 pairmate를 자동 기동하지 않는다.

```bash
python tools/mix20h_runner.py status
python tools/mix20h_runner.py report
# 같은 immutable window 안에서 runner 재연결. T0를 다시 설정하지 않는다.
python tools/mix20h_runner.py run --wait --upload
```

평가/업로드 재시도는 학습 없이 수행한다.

```bash
python tools/mix20h_postrun.py '<EXACT_RUN_ID>' --device cuda --upload
# 이미 평가가 완료된 로컬 결과의 업로드 내용 미리보기: 네트워크 접근 없음.
python gspread/mix20h_upload.py '<EXACT_RUN_ID>' --dry-run
```

Postrun은 실행 manifest의 deadline을 따른다. deadline 이후 필요한 GPU 평가를 새로 시작하지 않으며, 모든 hash-bound 평가 cache가 이미 완성됐다면 CPU 검증·업로드만 재시도할 수 있다. 직접 업로더도 로컬 평가 완료를 검증한다.

기존 s1 G23 S1234/s3 G23 S41003/s4 H31 S1234 자산, 과거 NOA 및 독립 ASV는 지우거나 재학습하지 않았다. 과거 자산의 v2 보완과 원격 NOA 완료 여부는 해당 서버 담당자의 별도 finite backlog다. 이번 신규 52xxx 학습 완료 조건과 혼합하거나 이미 처리됐다고 기록하지 않는다.

## 5. 검증 범위와 남은 실행 확인

최초 구현의 56개 검사에 간편 시작/경로 보정 검사를 추가했다. **간소화 후 통합 80개 테스트 통과**, 46개 생성 YAML 및 서버별 정적 큐의 exact 일치 통과, `git diff --check` 통과. Runtime activation manifest가 생성되지 않은 것도 확인했다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  python -m unittest tools.mix20h_config_tests tools.mix20h_runtime_tests \
  tools.mix20h_eval_tests tools.mix20h_runner_tests \
  tools.mix20h_launch_config_tests tools.mix20h_start_tests
```

검증에는 explicit ID/AB-BA/β-only 차이, 기존 config 불변, 실제 W104D121 optimizer 구성, β의 직접 gradient 분리, 실제 CPU checkpoint 저장·재개 후 batch/model/Adam/scheduler 일치, post-evaluation RNG/상태 복원, 50K 재개 시 추가 update 없음, 가짜 Sheet를 통한 header/read-back/기존 값 보존, 안전한 migration/예산/평가 재시도가 포함된다. 기존 PAKD50 단위검사도 통과했다.

실제 GPU smoke·첫 50K run·실제 Google Sheet 쓰기·원격 s2–s5 자산/프로세스 검증은 아직 수행하지 않았다. 그러므로 이 문서는 전 서버 `READY` 또는 학습 성공 증거가 아니다. 별도 smoke를 시작 장벽으로 강제하지 않으며, start 내부 검사와 run별 초기화/배치/평가 identity 확인은 자동으로 유지한다.
