# 구현 인계 — PANDA G23 SENS s4/s5

## 0. 작업 범위

`EXPERIMENT_PLAN_KR.md`와 `tools/casegen.py`가 이번 case 정의의 기준이다. 이 번들은 생성기까지 구현한 설계 번들이며, training runner/remote deployment/Sheet writer는 아직 연결하지 않았다. 계획만 읽고 실제 학습이 시작되었다고 보고하지 않는다.

현재 코드 기준은 `52184d5578a2a10721e435f8fbf4a759497768e5`다. runtime에서 새 코드가 필요하면 적용 diff와 최종 commit/content hash를 새로 고정한다. 기존 QRECON24, PAN-Crafter reproduction, ABLR2/ABLR2X queue를 손으로 재작성하지 말고, 새 campaign namespace와 별도 state 디렉터리를 사용한다.

## 1. 반드시 보존할 실험 정체성

WV3 / G23 / P0 / W104D121 / T0(step24240)다. PLH/W104D122, D123 Student, 새 Teacher, seed2025 PAN-Crafter, GF2 PANMIX와 혼합하지 않는다. `attn_locations=[]`와 `mode_modulation=False`를 검사한다. legacy `n_attn=3` 필드만 보고 CM3A3가 활성화되어 있다고 기록하면 안 된다.

고정 원값:

```text
q0 = 0.3276133416220546
tau0 = 0.012463942170143127
T0 checkpoint SHA256 = 16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32
```

원 config의 실제 수치다. 로컬 자산과 맞지 않으면 값을 임의로 고치지 말고 `BLOCKED_ASSET_MISMATCH`로 보고한다. T0 전체 checkpoint와 cue가 사용하는 Teacher-Aliger hash는 서로 다른 해시 대상일 수 있으므로 구분해서 기록한다. metadata/config/source SHA를 weight SHA처럼 비교하지 않는다.

## 2. config 연결

기준 config는
`config/PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S1234_FRESH50_v2.yaml`이다. s5도 동일한 계산 기준을 사용하며 이름의 S4는 reference file identity일 뿐 runtime 배정을 의미하지 않는다.

생성 JSON의 `backend_overrides`는 dotted-key의 **계산 필드 patch**다. 독립 학습 config로 main.py에 직접 넘기지 않는다. 먼저 복사한 base config에 적용하고, 경로·run identity·seed·초기화·selector·budget을 새 campaign 규칙으로 채운다.

| 개념 | 실제 계산 경로 | 주의 |
|---|---|---|
| α | `kdv.rec.alpha` | H를 U/A 양쪽이 사용 |
| β | `kdv.rec.kd_weight` | selective K만 변경 |
| λE | `kdv.stat.outer_weight` | absolute coefficient, 재차2배 금지 |
| q scale | `kdv.qrecon.q_ref=q0*scale` | strict registry에 임의 `q_ref_scale` key 추가 금지 |
| τ scale | `kdv.rec.tau=tau0`; `kdv.rec.tau_scale=scale` | **tau 자체에 scale을 먼저 곱하지 않는다** |
| rA | `kdv.aligner_lr=1e-4*rA` | U peak LR=1e-4 유지 |

`train_kdv.py`가 `tau_used=tau*rec_tau_scale`을 이미 계산한다. 명세의 `tau_R_used`는 검증용 기대값이지 `kdv.rec.tau`에 덮어쓸 값이 아니다. q raw table은 그대로, q-weight table은 q_ref_used에 맞게 재구성한다. stat.mode=H의 비활성 KD 설정과 실제 rec coefficient를 혼동하지 않는다.

기존 `kdv.qrc24`와 다른 explanatory metadata가 있다면 실제 적용값과 동기화한다. 다른 과거 run을 가리키는 `baseline_run`, `control_runs`, `block_2x2`, parent budgets, mandatory files 등은 새 local baseline 및 namespace로 교체한다. 기존 campaign임을 가장해 guard를 우회하지 않는다. 새 campaign을 명시적으로 등록하고 범위 검증을 둔다.

EXACT_50000은 새 report selection 규칙이다. trainer parser의 기존 enum을 확인하지 않고 `select.primary=EXACT_50000`을 주입하지 않는다. 정확한50K checkpoint를 반드시 저장하고 그 checkpoint를 직접 공식 evaluator에 넘기는 report 경로를 구현한다. 기존 best_hqnr 선택 로직이 유지되더라도 최종 sensitivity 비교와 run admission에 사용되지 않도록 분리한다. validation-selected를 같은 규칙으로 별도 기록한다.

## 3. runtime binding / preflight

`config/runtime_bindings.example.json`의 null은 로컬에서 확인해야 하는 값이다. 자동 fallback 금지. 경로를 추측해서 다른 Teacher나 다른 센서 데이터를 대체하지 않는다.

필수 확인:

1. 서버 identity, repo frozen commit+dirty diff, CUDA/GPU ID, 독점 run lock, 현재 기존 runner의 실제 상태.
2. WV3 train/val/RR/FR + LP 관련 파일, array layout/scene manifest/bit depth/normalization. reference config의 파일 경로를 로컬 절대경로로 resolve하되 내용은 해시로 고정.
3. T0 전체 checkpoint SHA, Aligner init hash, margin, Teacher eval/frozen 상태. cue JSON/NPZ SHA와 asset identity, 모든 `(index, actual rotation)` coverage 및 q≥0 유한성.
4. tau0 및 q0의 source provenance. 동일 Teacher/calibration이 아님에도 값을 그대로 사용하는 상황을 차단.
5. baseline 모델 signature, 계산용 override와 runtime loss/optimizer actual values 일치. nominal yaml 검사는 충분하지 않다.
6. 실제 GPU에서 baseline tiny smoke, 분리 gradient routing, official RR/FR evaluator, exact-resume 가능 여부. 이 번들의 CPU unit tests를 GPU test처럼 보고하지 않는다.

스모크는 별도 run/seed/directory에서 수행하고 본50K run을 오염시키지 않는다. smoke result는 sensitivity 결과로 업로드하지 않는다.

## 4. paired randomness와 초기화

각 서버·cycle마다 legacy G23 초기화 순서로 fresh U를 생성해 snapshot과 SHA를 고정한다. 같은 block의 각 arm은 이것을 독립 모델에 로드한다. A는 고정 T0에서 매번 새로 복사한다. 이전 arm의 Student/A optimizer state를 가져오지 않는다.

train indices/rotation/flip의 실제 sequence를 일치시켜야 한다. seed 숫자가 같다는 것만으로 충분하지 않다. calibration과 diagnostic의 RNG 소비를 training stream에서 분리하거나 완전히 복원한다. worker 수·sampler/drop_last·epoch→update 진행·feeder actual semantics를 통일한다. 시작 구간뿐 아니라 전체 stream hash 또는 reproducible stream manifest를 보존한다.

legacy feeder의 hflip/vflip 동작과 cue coverage를 sensitivity 중간에 교정하지 않는다. 버그수정이 필요하면 수정된 별도 reference baseline과 재생성 cue로 새 revision을 만든다.

## 5. server-local infinite runner 계약

가짜 `while True: main.py`를 반복하는 방식은 허용하지 않는다. state는 다음과 같이 지속한다.

```text
state: {campaign_id, server, cycle, position, active_run_id,
        active_case_spec_sha256, attempt, locked_bindings_sha256}
while STOP_AFTER_RUN is absent:
    validate code/data/T0/cue bindings and local GPU ownership
    case = next(iter_cases(server, cycle, position))
    materialize immutable case/config under unique run_id
    if COMPLETE receipt exists:
        recheck artifact/provenance hashes and skip idempotently
    elif exact resume is supported and checkpoint is valid:
        resume this SAME run (weights, optimizer, schedule, RNG, sampler cursor)
    else:
        train fresh with this block's common U/A and common stream
    save exact update50000 A/U checkpoint and validation-selected checkpoint
    evaluate same checkpoint for native RR and FR; write per-scene reports
    verify resolved coefficients, init/stream/cue/source identity and metrics
    write atomic COMPLETE receipt; queue Sheet outbox
    persist next_cursor(cycle, position) atomically
```

다른 서버의 완료를 기다리는 코드는 없어야 한다. s4/s5가 같은 seed 번호에 동시에 있어야 할 필요도 없다. 성공 metric threshold나 elapsed-hour로 종료하지 않는다. 낡은 `1000h budget` 같은 수치도 hard gate로 남기지 않는다.

재개 시 manifest/spec가 바뀌었으면 새 run이 아니라 기존 run을 덮어쓰는 시도를 거부한다. 결과가 좋지 않다는 이유로 retry하지 않는다. variant의 numerical divergence와 infra failure는 별도 상태로 남긴다. BASE failure는 해당 block을 pause한다. variant DIVERGED 이후에는 실패 기록을 남기고 다음 variant 진행 가능, 단 이를 정상50K receipt로 위장하지 않는다. `verify_receipt`는 정상 완료의 구조 검사만 담당한다.

## 6. 결과·Sheet·재현성

전용 탭은 `SENS-G23-WV3-s4`, `SENS-G23-WV3-s5`다. `result_columns.csv`의 단위/정의를 따른다. 생성기 case SHA, runtime config SHA, 실제 source commit/content SHA, Teacher/initial U/A/stream/q raw/q weight SHA를 보존한다. Runtime에서는 reference commit과 실제 implementation commit을 별도 필드로 기록한다.

업로드 key는 run_id + selection + attempt/provenance다. same checkpoint의 VAL alias는 중복 독립 run으로 만들지 않는다. 업로드 실패 시 outbox만 retry, 학습은 재수행하지 않는다. 비교 Δ는 해당 server/cycle BASE만 사용한다. source payload 값과 report 값을 readback해서 off-by-one이나 selection mixing이 없는지 확인한다.

## 7. 필수 acceptance tests

- 각 서버 7 cases, 총14; 각 variant에서 semantic knob 하나만 다름.
- qref half/double은 raw q hash 동일, w hash 변화. 원 q0에서 w=1/3·1/2·2/3.
- τR은 base×scale을 **한 번** 적용. 최종 criterion buffer가 기대값과 일치.
- A LR는1e-6/3e-6/6e-6, U LR는1e-4. Schedule 및 optimizer 실제값 확인.
- 동일 seed의 initial U/A/stream hashes 일치, 다른 cycle U snapshot은 새로 생성.
- 같은 forward에서 U/A gradient routing 분리, A의 직접 soft/edge/offset gradient 없음.
- primary EXACT50K, 모든 RR/FR metrics의 checkpoint hash 동일. 미완료·test-selected run은 정상 receipt 거부.
- process restart 후 duplicate training 없이 정확히 같은 cursor에서 복구.
- 한 서버 pause/업로드오류가 다른 서버 진행을 막지 않음.
- STOP_AFTER_RUN이 next admission만 막고 현재 run의 result integrity를 보존.
- old repro/GF2/ABLR2 queue·weights·Sheet를 덮어쓰거나 삭제하지 않음.

## 8. 배포 결과 보고

실제 수행한 항목만 구분해 보고한다: 파일 설치, 코드 통합, CPU tests, GPU smoke, asset preflight, old-run boundary handoff, 새 PID, 첫 BASE 학습, 공식평가, Sheet readback. 준비 단계가 끝났다는 이유만으로 '무한 cycle 가동'이라고 쓰지 않는다.
