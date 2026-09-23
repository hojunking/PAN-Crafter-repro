# PANDA G23 hyperparameter sensitivity — s4 / s5 무기한 반복

- Campaign: `PANDA_G23_SENS_WV3_S45_20260923_v1`
- 작성일: 2026-09-23
- 대상: s4와 s5, **WV3**, 독립 server-local 반복
- 산출물 상태: **실험 설계 / case 명세 / 생성기. 실제 서버 학습을 시작한 기록이 아니다.**
- 코드 확인 기준: `hojunking/PAN-Crafter-repro@52184d5578a2a10721e435f8fbf4a759497768e5`

## 1. 이번 실험의 범위

직전 합의한 **QRECON24 G23 / W104D121 / P0 / 고정 T0**를 중심으로 한 one-factor-at-a-time(OAT) sensitivity다. 각 축의 중심값에서 위·아래 한 점을 비교한다. 이는 최신 PLH/W104D122 PANDA 계열이나 PAN-Crafter reproduction의 sensitivity와 같은 실험이 아니다. 이 구분을 config, Sheet, figure caption에 남긴다. 이후 최종 PLH/W104D122로 기준을 바꿀 때는 별도 campaign revision과 해당 Teacher/calibration으로 새 비교군을 만든다. 기존 G23와 한 곡선에 섞지 않는다.

목표는 좋은 조합을 무작정 찾는 것이 아니라, 정해진 G23 주변에서 각 계수 변화가 RR/FR와 학습 동작에 주는 영향을 여러 Student seed에서 측정하는 것이다. 각 cycle 결과에 따라 다음 cycle의 값·범위·Teacher·seed를 자동 변경하지 않는다. U learning rate sweep, Teacher 재학습, architecture sweep, zero-weight ablation, PANMIX는 이번 범위 밖이다.

**무기한 반복의 단위는 50K run을 새로 시작하는 cycle이다. 한 Student를 무한히 이어 학습하지 않는다.**

## 2. 고정 기준선

아래 값은 기준 config [S1]에서 확인했다. `config/baseline_recipe.json`은 비교를 위한 정규화 명세이며 독립 실행용 main.py config는 아니다.

| 항목 | 고정값 / 계약 |
|---|---|
| 데이터셋 | WV3, train/val/RR/FR 및 LP 파일은 해시로 고정 |
| Student U | W104, depth [1,2,1], LN, attention locations=[], mode modulation=False |
| 입력 | P0: 정합 PAN + upsampled MS, 9채널. LPAN/HPAN feature 추가 없음 |
| Aligner | 고정 T0 Aligner의 독립적인 trainable copy, view margin 4 |
| Teacher | `PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1/best_hqnr`, step24240 |
| Teacher 파일 SHA256 | `16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32` |
| α / β / λE | 1.0 / 0.1 / **0.002(절대값)** |
| q reference 원값 | 0.3276133416220546 |
| reconstruction τR 원값 | 0.012463942170143127 |
| U / A peak LR | 1e-4 / 3e-6, rA=0.03 |
| 학습 | 각 run 정확히 50,000 optimizer updates, batch48, single MS task |
| Optimizer | AdamW, wd0.01, warmup100, cosine, U/A 동일 schedule 형태 |
| q 함수 | qref/(qref+qT), 분자 계수1, 평균 재정규화·clip·threshold 없음 |
| q 자산 | `assets/qedge9/cue_T0_AXIS16_v1.{json,npz}`의 frozen Teacher raw q |
| 재구성 | REC-R3, eps1e-6, 모든 픽셀 평균, soft-active 영역 수로 재정규화하지 않음 |
| Edge | 기존 GT signed Scharr EDGE-H, window config5, ramp0 |
| Student 별도 정합 loss | offset=0, geometry KD=0, synthetic shift radius=0 |
| Calibration | n_patches3072 / calibration seed1234는 cycle seed와 별개로 고정 |
| Feeder | crop=False, hflip=True, vflip=True, rot=True, return_meta=True; **기존 실제 동작 유지** |

Feeder flag가 실제로 어떤 확률적 동작을 하는지와 q table의 `(sample, rotation)` 대응은 동일한 계약이어야 한다. 이번 sensitivity 중에 flip 구현을 바꾸거나 crop/resize를 고치지 않는다. PAN-Crafter reproduction에서 별도 수정한 feeder를 가져오는 것도 금지한다. 정규화 및 label/MS/PAN 좌표계도 바꾸지 않는다.

원본 config에는 과거 `best_hqnr`, old baseline run, old campaign budget 메타데이터가 들어 있다. 새로운 campaign에서 이를 그대로 실행하는 것이 아니라, 아래 평가·반복 규칙으로 명시적으로 교체해야 한다. 계산용 값은 `backend_overrides`로 제공한다.

## 3. s4: loss coefficient sensitivity

각 행에서 표시된 한 축만 바꾸고 나머지는 §2에 고정한다.

| Case | 변경 축 | α | β | λE | qref 배율 | τR 배율 | rA |
|---|---|---:|---:|---:|---:|---:|---:|
| BASE | 없음 | 1.0 | 0.10 | 0.0020 | 1 | 1 | 0.03 |
| AL05 | α 낮음 | **0.5** | 0.10 | 0.0020 | 1 | 1 | 0.03 |
| AL15 | α 높음 | **1.5** | 0.10 | 0.0020 | 1 | 1 | 0.03 |
| BE005 | β 낮음 | 1.0 | **0.05** | 0.0020 | 1 | 1 | 0.03 |
| BE020 | β 높음 | 1.0 | **0.20** | 0.0020 | 1 | 1 | 0.03 |
| ED0006 | λE 낮음 | 1.0 | 0.10 | **0.0006** | 1 | 1 | 0.03 |
| ED006 | λE 높음 | 1.0 | 0.10 | **0.0060** | 1 | 1 | 0.03 |

직전 제안의 선택 사항이었던 λE 두 점도 포함했다. 이전 grid의 값을 다시 사용하되, 이번에는 rA=.03과 동일한 seed/초기화/평가 계약에 맞추므로 현재 G23 주변의 직접 비교가 된다. 추가적인 범위 확장은 없다.

## 4. s5: calibration / Aligner adaptation sensitivity

| Case | 변경 축 | qref 배율 | qref 실제값 | τR 배율 | τR 실제값 | rA | A peak LR |
|---|---|---:|---:|---:|---:|---:|---:|
| BASE | 없음 | 1 | 0.3276133416220546 | 1 | 0.012463942170143127 | 0.03 | 3e-6 |
| QR05 | qref 낮음 | **0.5** | 0.1638066708110273 | 1 | 0.012463942170143127 | 0.03 | 3e-6 |
| QR20 | qref 높음 | **2** | 0.6552266832441092 | 1 | 0.012463942170143127 | 0.03 | 3e-6 |
| TR05 | τR 낮음 | 1 | 0.3276133416220546 | **0.5** | 0.006231971085071564 | 0.03 | 3e-6 |
| TR20 | τR 높음 | 1 | 0.3276133416220546 | **2** | 0.024927884340286254 | 0.03 | 3e-6 |
| RA01 | A LR 낮음 | 1 | 0.3276133416220546 | 1 | 0.012463942170143127 | **0.01** | **1e-6** |
| RA06 | A LR 높음 | 1 | 0.3276133416220546 | 1 | 0.012463942170143127 | **0.06** | **6e-6** |

모든 행에서 α=1, β=.1, λE=.002, U LR=1e-4다. calibration 원값은 다시 추정하지 않고 **배율만** 바꾼다. 다른 Teacher의 median을 가져오지 않는다.

### qref 변화의 해석

raw q table은 그대로 두고 w만 다시 계산한다. q=q0인 표본에서 w는 각각 1/3, 1/2, 2/3이 된다. 이 변화는 sample 간 상대 가중치뿐 아니라 **A hard와 U edge의 전체 강도도 함께** 바꾼다. 따라서 결과를 순수한 sample ranking 효과만으로 해석하지 않는다. 평균 w로 나누거나 λE/rA를 보상하는 실험은 이번 표에 넣지 않는다.

### τR 변화의 해석

기존 R3는 `d=eT/(eT+τR)`와 `a=relu(eS-eT)/(eS+eps)`를 사용한다 [S3]. τR을 낮추면 같은 eT에서 d가 커져 hard 강조는 커지고, `(1-d)`에 의한 soft 허용량은 작아진다. 이 공동 변화가 이번 sensitivity 대상이다. α 또는 β를 반대로 조정하지 않는다.

### rA 변화의 해석

loss 계수를 추가하는 것이 아니라 **A optimizer의 learning rate만** 바꾼다. U LR와 schedule은 그대로다. A를 얼리거나 soft/edge gradient를 A에 연결하지 않는다. `.06`은 기존 `.03` 중심의 상측 확인점이지 새 기본값이 아니다.

## 5. Gradient 계약

원 구현 [S2, S3]의 표기를 기준으로, H는 adaptive hard, K는 selective soft, E는 GT edge다.

```text
d_T = e_T / (e_T + tau_R_used)
a_T = stop_gradient(relu(e_S - e_T) / (e_S + eps))
H_i = mean_pixels[(1 + alpha*d_T) * |S-GT|]
K_i = mean_pixels[beta*(1-d_T)*a_T * |S-T|]
w_i = stop_gradient(q_ref_used / (q_ref_used + q_T(i)))
L_U = mean_i[H_i + K_i + lambda_E*w_i*E_i]
L_A = mean_i[w_i*H_i]
U receives grad(L_U, U)
A receives grad(L_A, A)
```

Teacher/GT/gate는 detach한다. 같은 forward graph여도 단일 total.backward()를 사용하면 안 된다. A에는 soft·edge·offset gradient가 없어야 한다. α 및 τR 변형은 H를 통해 U와 A에 모두 영향을 주는 것이 정상이다. β/λE 변형도 학습이 진행되면 U 상태 변화에 따라 A에 간접 영향을 줄 수 있으므로, 경로 불변 검사와 결과의 동일성을 혼동하지 않는다.

## 6. 무기한 cycle와 seed

**서버당 7 runs / 두 서버의 같은 cycle 번호 합계14 runs.** 중복 baseline을 제외한 조건은13개다. 두 서버의 BASE는 동일 seed라도 두 개의 독립적인 seed 증거로 세지 않는다.

| Cycle | s4 seed | s5 seed | 목적 |
|---:|---:|---:|---|
| 0 | 1234 | 1234 | 기준 config와 연결되는 anchor block |
| 1 | 2000000 | 2000000 | 새 paired Student seed |
| 2 | 2000001 | 2000001 | 새 paired Student seed |
| 3 | 2000002 | 2000002 | 계속 반복 |

`cycle>=1: seed=2000000+cycle-1`. 각 서버가 자기 cycle을 진행하며 **서로의 완료를 기다리지 않는다**. 다른 서버보다 한 cycle 먼저 진행해도 된다. 동일 번호 cycle의 seed를 맞추는 것은 비교 편의를 위한 규칙이지 runtime lock이 아니다. 운영상 시간/run/cycle 상한은 없다. 기술적으로 seed 표현 범위를 소진하면 조용히 wrap하지 않고 명시적으로 멈춘다.

각 run은 새 U와 고정 T0에서 복사한 새 A로 시작한다. 이전 case/cycle의 Student U 또는 A를 이어받지 않는다. 한 cycle에서는 동일한 초기 U snapshot과 동일한 initial A 및 sample/augmentation sequence를 모든7개 case가 공유한다. Snapshot을 각 run의 독립 모델에 로드하므로 학습된 weight가 다음 case에 새지 않는다.

**매 cycle 새 BASE를 반드시 학습한다.** 이전 seed의 baseline이나 과거 test-aware best checkpoint를 현재 seed의 비교군으로 재사용하지 않는다. 별도 runtime에서 이미 완료한 동일 run을 재사용하는 경우는 동일 run_id/spec/source/data/init/stream/checkpoint 해시를 모두 확인한 exact-resume/idempotent skip뿐이다.

## 7. 실행 순서

Cycle0:

```text
s4: BASE -> AL05 -> AL15 -> BE005 -> BE020 -> ED0006 -> ED006 -> cycle1
s5: BASE -> QR05 -> QR20 -> TR05 -> TR20 -> RA01 -> RA06 -> cycle1
```

다음 cycle에서는 3개 축 묶음의 순서를 한 칸 회전하고, 홀수 cycle에서는 각 묶음의 high/low 순서를 뒤집는다. 예를 들면 cycle1의 s4는 BASE→BE020→BE005→ED006→ED0006→AL15→AL05다. 6-cycle 주기로 순서가 반복되며, 성능에 따른 우선순위 변경은 없다.

7개가 끝나면 seed를 바꿔 반복한다. 높은 성능, 낮은 성능, 논문 목표 통과 여부는 반복 중단이나 seed 재추첨의 조건이 아니다.

## 8. 평가와 비교

주 결과는 **EXACT_50000**, 보조는 동일 validation protocol의 **RR_VAL_ERGAS_MIN, 동률은 낮은 step**다. 모든 case에 같은 선택 규칙을 적용한다. best HQNR, TARGET, RAW_MAX, test oracle 선택점을 sensitivity curve에 사용하지 않는다. 학습 중 test 평가를 유지하더라도 로그용이며 제어 로직에 연결하지 않는다.

RR와 FR에는 반드시 같은 A/U checkpoint를 사용한다. 최소 지표는 HQNR, Dλ, Ds, ERGAS이며 SAM, PSNR, SSIM, SCC, Q8, RMSE, CC도 보존한다. Native RR/FR 전체 scene manifest를 사용하고, V64/masking/선택적 scene 제외를 섞지 않는다. 실제 scene 수 및 paper-set identity는 실측 manifest 상태 그대로 기록한다. 논문 데이터와의 동일성이 검증되지 않았으면 검증된 것으로 바꾸지 않는다. JQM을 기록할 때는 기존 SRF-substitute variant를 명시한다.

각 결과는 **같은 서버·같은 cycle BASE 대비**로 비교한다.

```text
delta_metric = variant_metric - local_BASE_metric
HQNR/SCC/PSNR/SSIM/Q8: positive is better
ERGAS/SAM/D_lambda/D_s/RMSE: negative is better
```

최초3개의 서로 다른 seed block이 완료되면 초기 경향을 요약하고 이후 누적한다. 3은 통계적 충분성을 보장하는 숫자도, 종료 조건도 아니다. Seed별 raw 값/paired Δ/평균/표준편차/실패 수를 함께 보고한다. s4/s5의 duplicate baseline은 두 seed처럼 합치지 않는다. Teacher는 하나이므로 결과의 범위는 **고정 T0 조건부 Student-seed sensitivity**다. Teacher-seed 일반화나 factor interaction을 입증했다고 쓰지 않는다.

## 9. 함께 남길 진단

고정 train probe에서 시작/1K/10K/25K/50K를 기준으로 다음을 기록한다. 별도 진단 RNG를 사용하거나 기존 RNG 상태를 복원해 training stream을 바꾸지 않는다.

- `d_T`, hard/soft weight 및 soft-positive fraction, raw/weighted H/K/E.
- `q_raw`, qref, w의 mean/min/max/분위수, q raw 및 w table SHA256.
- U에 대한 hard/KD/edge gradient norm과 실제 적용 λE, A hard gradient norm 및 실제 optimizer LR. Loss 크기 비율을 gradient 비중이라고 부르지 않는다.
- Aligner correction norm/분위수와 초기 Aligner 대비 변화. A로의 forbidden soft/edge/offset 경로 검사.

값이 작다는 이유로 β/λE를 자동 증폭하지 않는다. 진단은 해석용이다.

## 10. 운영·복구·종료

현재 s4/s5의 PAN-Crafter reproduction 또는 다른 작업을 이 번들이 직접 중단하지 않는다. 배포 담당자가 실제 PID/runner/자동 재기동 경로를 확인한 뒤 **현재 run이 끝나는 경계에서** 남은 기존 s4/s5 queue의 자동 재시작을 중지하고 이 campaign으로 넘긴다. 기존 결과와 checkpoint는 보존한다. s1/s2/s3는 변경하지 않는다.

서버별 campaign lock과 GPU 사용권을 확보한다. 다른 서버의 lock은 보지 않는다. `runtime_bindings`의 코드·데이터·T0·cue 해시와 경로를 검증한 뒤 한 run씩 수행한다. BASE 실패 시 그 block을 정지해 원인을 확인한다.

중단 run은 model/A/U, optimizer, scheduler, RNG, sampler/data cursor까지 맞는 exact resume를 우선한다. 이를 보장하지 못하면 같은 case의 fresh retry를 별도 attempt로 기록하며 원 실패를 지우지 않는다. 기술적 재시도는 최대2회 후 해당 서버를 PAUSED 상태로 둔다. 무기한 반복은 실패한 코드를 무기한 재시도한다는 뜻이 아니다.

수치 발산은 DIVERGED로 보존하고 seed를 재추첨하거나 LR/batch를 자동 변경하지 않는다. BASE가 정상인 block의 variant 발산은 유효한 실패 증거로 남긴 후 다음 variant를 진행할 수 있다. 기술 오류/평가 누락은 정상 비교로 집계하지 않는다. 업로드 오류는 학습을 재수행하지 않고 local outbox에서 재전송한다.

STOP_AFTER_RUN 요청이 있으면 현재 run 결과를 마무리한 뒤 다음 admission을 막는다. 급정지 경로는 해당 runner가 지원하는 안전 checkpoint 절차만 사용한다. 광범위한 `pkill python`, 다른 사용자의 프로세스 종료, 이전 실험 삭제는 금지한다.

누적 저장공간이 부족하면 PAUSED로 전환한다. 자동으로 batch/precision/평가법을 바꾸지 않는다. EXACT50K, VAL-selected, per-scene metrics, config/source hashes, init snapshot, diagnostics, completion receipt는 보존한다. 중간 후보 정리는 별도 명시적 retention 정책으로 해당 campaign의 완료·검증된 run만 대상으로 한다.

## 11. Sheet 구성

기존 `유의미한결과`나 PC-Repro/GF2 탭을 덮어쓰지 않는다. 구현 시 전용 탭 **`SENS-G23-WV3-s4` / `SENS-G23-WV3-s5`**를 만든다. Run 셀은 모델/계수 설정만 표시하고 server/cycle/seed/case_id/run_id/선택점은 별도 열에 둔다. `result_columns.csv`에 필수 필드를 제공했다.

모든 결과를 기록하며 좋은 run만 남기지 않는다. 원본값과 같은-cycle BASE 대비 Δ를 분리한다. 정확한50K와 VAL을 별도 행 또는 명확한 별도 필드로 기록하고 checkpoint가 같으면 alias로 표시한다. Train(h)는 training_seconds/3600, Teacher 비용을 매번 재과금하지 않는다. 실제 업로드 후 readback 확인을 남긴다.

## 12. 포함 파일과 사용법

- `case_templates.csv`: s4/s5의14개 반복 template.
- `config/`: baseline, case catalog, 무기한 운영 policy, runtime bindings 양식.
- `examples/first_3_cycles/`: 총42개 구체 JSON case와 `cases.csv`. **3-cycle로 종료하라는 뜻이 아니다.**
- `tools/casegen.py`: 임의 cycle 생성, lazy infinite iterator, cursor의 next-case 및 receipt 구조 검사.
- `IMPLEMENTATION_HANDOFF_KR.md`: 코드 연결·preflight·acceptance 계약.
- `server_handoffs/s4.md`, `s5.md`: 서버별 실행 담당자용 전달문.
- `sources/`: 기준 config 및 계산 코드의 출처/확인값.
- `tests/`, `build_validation.json`, `SHA256SUMS.txt`: 명세 생성기 검사 및 번들 무결성.

```bash
# 유한 미리보기만 출력한다. 학습을 시작하지 않는다.
python tools/casegen.py emit --server both --start-cycle 0 --cycles 3 --out /tmp/g23_sens_preview
# 서버 로컬 cursor가 다음에 실행해야 할 단일 case
python tools/casegen.py next --server s4 --cycle 4 --position 0
# 생성 명세 검증: 실제 GPU/데이터 검증과는 별개
python tools/casegen.py validate /tmp/g23_sens_preview/s4/SENS_G23_WV3_s4_C000000_BASE_S1234_F50K_v1.json
python -m unittest discover -s tests -v
```

실제 학습 launcher/Sheet uploader는 저장소의 trainer/evaluator에 연결해야 한다. 이 번들에 그런 서버 연결을 완료했다고 주장하는 자동 실행 명령이나 fabricated completion receipt는 없다.

## 13. 출처

소스의 표기·계산 방식과 이번에 제안한 운영 정책을 구분했다. Sweep 값은 직전 사용자 승인안, seed/반복/복구 정책은 이 계획의 설계 결정이다. 아래 자료는 구현과 baseline을 뒷받침한다.

- [S1] 기준 G23 config (W104D121, T0, α/β/λE/rA/qref/τR, optimizer/feeder):
  https://github.com/hojunking/PAN-Crafter-repro/blob/52184d5578a2a10721e435f8fbf4a759497768e5/config/PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S1234_FRESH50_v2.yaml
- [S2] q weight 및 raw q 자산/해시 검증:
  https://github.com/hojunking/PAN-Crafter-repro/blob/52184d5578a2a10721e435f8fbf4a759497768e5/kdv/qrecon.py
- [S3] R3 hard/soft, difficulty/advantage, detach 및 정규화:
  https://github.com/hojunking/PAN-Crafter-repro/blob/52184d5578a2a10721e435f8fbf4a759497768e5/kdv/losses_rec.py
- [S4] qrecon의 허용 계약 및 `rec.tau_scale`:
  https://github.com/hojunking/PAN-Crafter-repro/blob/52184d5578a2a10721e435f8fbf4a759497768e5/kdv/registry.py
- [S5] `tau_used=float(tau)*rec_tau_scale`, calibration 원값/사용값 별도 기록:
  https://github.com/hojunking/PAN-Crafter-repro/blob/52184d5578a2a10721e435f8fbf4a759497768e5/train_kdv.py
