# PANDA Sensor Focus queue handoff — 2026-10-29

## 1. 목적과 변경 범위

2026-10-29까지의 결과를 바탕으로 QB/GF2의 미시작 실험을 정리하고, 검증된 Stage 1 reference를 고정한 Student 실험을 우선한다. 사용자 요청에 따라 시간 제한을 두지 않는다. 기존 WV3 TUNE30/HARD36 등록 설정과 실행 순서는 유지하며, 센서 실험이 끝나면 미완료 항목을 이어서 실행한다.

이 문서의 s1 상태는 로컬 campaign state에 근거한다. s3에는 이 세션에서 직접 접속하지 않았으므로, 시트에 결과가 있다는 사실을 원격 큐 전체 완료로 해석하지 않는다. s3 에이전트는 실행 시 원격 로컬 상태를 읽어 완료·실행 중·미시작을 구분해야 한다.

관련 근거: [2026-10-29 분석 JSON](../results_log/assets/2026-10-29_sensor-results-analysis.json), [분석 그림](../results_log/assets/2026-10-29_sensor-results-analysis.png), [전체 등록 큐 CSV](../results_log/assets/2026-10-29_sensor-focus-full-queue.csv).

## 2. 과학적 판단

- QB 기준 A200_E030 R3 reference를 공유한 BASE R1/R2는 각각 HQNR/ERGAS `.942698/3.514885`, `.945384/3.509656`이다. LR120, C40, 반경 4px는 ERGAS를 개선하지 않았다. α=.125·γ=.25는 R1에서 `3.511504`로 소폭 개선됐으나 seed 차이보다 작은 효과이므로 반복 확인한다.
- QB에서 λ_ε=.01의 reference는 native RRE가 더 좋아도 Student HQNR이 8·9·10K 모두 .90 미만이었다. 작은 shift나 더 낮은 RRE만으로 Student 성능을 예측하지 않는다. 현재 진행 중인 λ_abs=1·λ_ε=.03 donor/Student는 완료하여 계수 효과를 확인한다.
- GF2의 검증된 SENSOR5 v2 donor를 사용한 LONG(60K, LR=1e-4)은 `.966801/.524202`로 가장 가깝다. 신규 LOSS9 계수 조합의 Student 결과보다 긴 스케줄의 개선폭이 크다. 이 donor를 재사용하여 반복·LR·width만 비교한다.
- WV3 W128은 두 Student seed 모두 목표를 충족했다. QB/GF2에서 W128은 검증할 가치가 있지만, 전이가 보장된다는 뜻은 아니다. 새 loss·SWA·회전 메서드를 추가하지 않는다.

## 3. 고정 조건 및 reference

### s1 / QB

- source campaign: `PANDA_S13_S1LOSS9_RGB_L4_STAGE12_20261026_v1`
- donor: `S1_A200_E030`, validation selector가 고른 **44K** checkpoint
- checkpoint SHA256: `7fa4491f0c0e7b8c5d559263c50fff18f84abd7a820852a0d24666928191ed34`
- selector receipt SHA256: `bea177873b2611822c60f9952c86f3a9f4982832142a7295c1053974e50e748b`
- Stage 1: λ_abs=2, λ_ε=.03, ε 분포 uniform-area disk, 반경 3px
- 신규 Student: 30K warmup cosine, warmup 100, min LR=0, LR=1e-4
- Student seed R1=`261018101`, R2=`261018102`; 동일 replica의 데이터/초기화 stream 정책 유지

### s3 / GF2

- source campaign: `PANDA_S13_SENSOR5_RGB_L4_STAGE12_20261026_v2`
- donor: `RGB_MATURE_U23`, validation selector가 고른 **42K** checkpoint
- Stage 1: λ_abs=2, λ_ε=.003, ε 분포 uniform-area disk, 반경 3px
- 원본 checkpoint: `work_dir/panda_sensor5/PANDA_S13_SENSOR5_RGB_L4_STAGE12_20261026_v2/s3/GF2/runs/PSN5R2_GF2_S3_RGB_MATURE_U23_NOSOFT_R1_S261016101/attempt_01/models/step_0042000.pt`
- 기존 GF2_LONG에 기록된 checkpoint SHA256: `937ac56b559b76a151df661bbb853119512d6f9cab283346c61186db27ff7478`
- 기존 GF2_LONG에 기록된 selector receipt SHA256: `52626a232557267d6852bc2b49bde8fc05aa6ccd5c1fc759a79fbe1101bedf95`
- 신규 Student: 60K warmup cosine, warmup 100, min LR=0

GF2의 과거 `SENSOR_REFINE GF2_C60`은 LOSS9 `S1_A200_E030`(λ_ε=.03)을 사용한다. 이번 C60은 SENSOR5 v2 `RGB_MATURE_U23`(λ_ε=.003)을 사용하므로 서로 같은 반복 실험이 아니다. 계수가 비슷한 LOSS9 `A200_E003 @42K`로도 대체하지 않는다. 원본 receipt·checkpoint·donor identity를 검증하고, 없거나 다르면 중단해야 한다.

### 공통 유지 사항

- Aligner scale=1, PAN의 전역 2D shift, RGB luminance reference, Teacher/Student Lanczos-4
- Student 입력 순서: `[corrected PAN, corrected PAN − original PAN, I_lrms]`
- QB/GF2: MS 4밴드, Student 입력 6채널, 출력 4밴드
- UBIL, depth=[1,2,3], β=0, 기존 GT Hard loss
- W128은 Student width만 변경한다. Teacher/Aligner와 source checkpoint는 변경하지 않는다.
- rotation consistency loss·TTA·SWA/EMA·밴드 가중 loss 없음
- 기존 학습 augmentation과 고정 데이터/평가 프로토콜 유지
- Stage 2는 각 설정마다 새 Student를 초기화한다. 이전 Student의 가중치를 재사용하지 않는다.

## 4. s1 전체 실행 순서

### 4.1 현재 진행 중인 실험 유지

2026-10-29 문서 작성 중 확인한 상태: `S1_A100_E030_R3` RUNNING, 27K. 최종 인계 시 실제 step은 더 진행될 수 있다.

| 순서 | case | replica | 목표 updates | 처리 |
|---:|---|---|---:|---|
| 1 | S1_A100_E030_R3 | R1 | 50K | 현재 Stage 1을 안전하게 이어서 완료 |
| 2 | QB_A100_E030_R3 | R1 | 30K | 해당 donor가 정상 선택되면 Student 실행 |

기존 QB BASE R2, LR120 R1, C40 R1, H050 R1, R4 donor/Student는 이미 완료됐다. E010 donor는 완료됐고 Student는 `EARLY_STOPPED_HQNR` 11K이다. 이들은 재실행하지 않으며 완료·중단 기록을 보존한다. A100 donor가 인증된 support failure로 종료되면 종속 Student만 보류하고 독립 신규 실험을 진행한다.

### 4.2 신규 QB 6회

공통 donor는 위의 **LOSS9 A200_E030 @44K**이다. 진행 중인 A100 donor로 신규 6회의 reference를 자동 교체하지 않는다.

| 순서 | case | replica | updates | width | LR | α | γ |
|---:|---|---|---:|---:|---:|---:|---:|
| 3 | QB_W128 | R1 | 30K | 128 | 1e-4 | .25 | .5 |
| 4 | QB_W128 | R2 | 30K | 128 | 1e-4 | .25 | .5 |
| 5 | QB_A0125 | R1 | 30K | 112 | 1e-4 | .125 | .5 |
| 6 | QB_A0125 | R2 | 30K | 112 | 1e-4 | .125 | .5 |
| 7 | QB_H050 | R2 | 30K | 112 | 1e-4 | .125 | .25 |
| 8 | QB_G000 | R1 | 30K | 112 | 1e-4 | .25 | 0 |

### 4.3 이후 WV3 TUNE30: 기존 미완료 9회

모든 case 이름의 공통 접두사는 `S100DIFFUBIL`이다. 아래 표에서는 생략한다. 기존 C30 R3와 C25 R1은 로컬 state에서 완료를 확인했다.

| 순서 | case 접미사 | replica | updates |
|---:|---|---|---:|
| 9 | C25 | R2 | 25K |
| 10–11 | C30LR080 | R1, R2 | 각 30K |
| 12–13 | C30LR120 | R1, R2 | 각 30K |
| 14–15 | C30G025 | R1, R2 | 각 30K |
| 16–17 | C30A0125 | R1, R2 | 각 30K |

### 4.4 이후 WV3 HARD36: 기존 12회

| 순서 | case 접미사 | replica | updates |
|---:|---|---|---:|
| 18–19 | C30AG000 | R1, R2 | 각 30K |
| 20–21 | C30G000 | R1, R2 | 각 30K |
| 22–23 | C30A000 | R1, R2 | 각 30K |
| 24–25 | C30H050 | R1, R2 | 각 30K |
| 26–27 | C30A050 | R1, R2 | 각 30K |
| 28–29 | C30G100 | R1, R2 | 각 30K |

따라서 s1은 현재 donor의 남은 부분과 종속 Student를 포함해 **최대 29개 항목**이다. support failure·QB early stop 등이 있으면 실제 수행량은 줄어든다. WV3는 설정과 등록 순서를 변경하지 않는다.

## 5. s3 전체 실행 순서와 제외 규칙

### 5.1 원격 상태에 따른 안전한 전환

먼저 원격 로컬 campaign state를 확인한다. 이미 시작한 Stage 1은 현재 실행을 안전하게 완료하고, 정상 donor 선택 후 해당 Student까지 완료한다. 이미 실행 중인 Student도 완료한다. QB의 .90 early-stop 규칙을 GF2에 적용하지 않는다. 인증된 support failure는 실패로 보존하며 종속 Student는 보류한다.

아래 항목은 **아직 시작하지 않은 경우에만** 새 우선순위로 대체한다. 완료·실행 중 결과를 삭제하거나 재초기화하지 않는다.

| 기존 블록 | 미시작 시 제외 대상 |
|---|---|
| LOSS9 | 남은 A200_E030, A050_E030 donor/Student pair 및 이전 정책으로 이미 제외된 미시작 pair |
| SENSOR_REFINE 기준 donor Student | GF2_BASE R2, GF2_LR150 R1, GF2_C60 R1, GF2_H050 R1 |
| SENSOR_REFINE fresh pair | S1_A200_E030_R4 + GF2_A200_E030_R4 |
| SENSOR_REFINE fresh pair | S1_A200_E010_R3 + GF2_A200_E010_R3 |
| SENSOR_REFINE fresh pair | S1_A100_E030_R3 + GF2_A100_E030_R3 |

2026-10-29 분석 시트에는 LOSS9 A020_E003, A200_E003, A050_E003, A200_E0003의 결과가 있었다. 나머지 항목의 미게시만으로 미실행이나 실패를 단정하지 않는다. 전환 controller가 실제 시작 여부를 판단하여 실행 중 pair를 우선 보존한다.

### 5.2 신규 GF2 5회

공통 donor는 **SENSOR5 v2 RGB_MATURE_U23 @42K**이다. 기존 GF2_LONG R1의 정확한 설정을 기준으로 사용한다.

| 신규 순서 | case | replica | updates | width | LR | α | γ |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | GF2_C60 | R2 | 60K | 112 | 1e-4 | .25 | .5 |
| 2 | GF2_C60LR150 | R1 | 60K | 112 | 1.5e-4 | .25 | .5 |
| 3 | GF2_C60LR150 | R2 | 60K | 112 | 1.5e-4 | .25 | .5 |
| 4 | GF2_C60W128 | R1 | 60K | 128 | 1e-4 | .25 | .5 |
| 5 | GF2_C60W128 | R2 | 60K | 128 | 1e-4 | .25 | .5 |

### 5.3 이후 WV3 TUNE30/HARD36 전체 등록 목록

실제 실행 대상은 아래 등록 목록에서 **s3 로컬 state상 미완료인 항목만**이다. 시트의 대표 checkpoint step은 전체 학습 완료 여부와 다르다. 이미 완료된 run은 재실행하지 않는다.

| 블록 | case 접미사 (`S100DIFFUBIL` 뒤) | replica | updates | 10/29 시트 결과 근거 |
|---|---|---|---:|---|
| TUNE30 | C30 | R3 | 30K | 게시 있음 |
| TUNE30 | C25 | R1, R2 | 각 25K | 게시 있음 |
| TUNE30 | C30LR080 | R1, R2 | 각 30K | 게시 있음 |
| TUNE30 | C35 | R1, R2 | 각 35K | 게시 있음 |
| TUNE30 | C30LR120 | R1, R2 | 각 30K | 게시 있음 |
| TUNE30 | C30G025 | R1, R2 | 각 30K | 게시 있음 |
| TUNE30 | C30A0125 | R1, R2 | 각 30K | 게시 있음 |
| TUNE30 | C30W128 | R1, R2 | 각 30K | 게시 있음 |
| HARD36 | C30AG000 | R1, R2 | 각 30K | 게시 있음 |
| HARD36 | C30G000 | R1, R2 | 각 30K | 게시 있음 |
| HARD36 | C30A000 | R1, R2 | 각 30K | 게시 있음 |
| HARD36 | C30H050 | R1, R2 | 각 30K | 게시 있음 |
| HARD36 | C30A050 | R1, R2 | 각 30K | 게시 있음 |
| HARD36 | C30G100 | R1, R2 | 각 30K | 분석 snapshot에 게시 근거 없음 |

s3의 WV3 등록은 TUNE30 15회 + HARD36 12회다. 이 27회가 모두 남았다는 의미가 아니며, 실제 잔여 횟수는 원격 state로 결정한다.

## 6. Sheet 및 중단 규칙

| 데이터셋 | 게시 탭 | HQNR 기준 | ERGAS 기준 |
|---|---|---:|---:|
| QB | QB-ablations | ≥ .9315 | ≤ 3.500 |
| GF2 | GF2-ablations | ≥ .9645 | ≤ .522 |
| WV3 | WV3-ablations | ≥ .9585 | ≤ 2.040 |

- 동일 checkpoint의 공식 RR20/FR20 결과를 사용한다.
- HQNR 기준을 충족하는 checkpoint가 있으면 그중 ERGAS 최소를 대표로 게시한다. 없으면 HQNR 최대를 대표로 게시한다.
- 두 목표를 동시에 충족한 checkpoint는 모두 게시하고 대표 게시도 유지한다.
- 행에 source donor/campaign·width·horizon·LR·α·γ·scale·입력·warp·seed를 식별할 수 있게 기록한다.
- GF2 과거 .548 기준으로 게시된 '조건 충족'과 현재 .522 기준을 혼동하지 않는다. 새 실행의 목표는 반드시 .522이다.
- **QB Student만** 8K·9K·10K의 HQNR이 모두 .90 미만이면 안전 중단한다. 세 공식 평가가 모두 존재하고 검증돼야 한다. Stage 1, GF2, WV3에는 적용하지 않는다.

## 7. 실행 인계

1. 사용자가 이번 commit을 원격에 push한다. 이 서버 에이전트는 push하지 않는다.
2. s3 에이전트는 현재 GPU 학습/launcher 상태와 Git 상태를 확인한 뒤 해당 commit을 pull한다. 기존 pinned source를 수정하거나 이미 실행 중인 원본 run을 재초기화하지 않는다.
3. 새 진입점으로 전환을 등록한다.

   ```bash
   bash scripts/enqueue_panda_sensor_focus.sh s3
   ```

4. controller의 전환 receipt에 실제 현재 run, 보존된 경계, 미시작 제외 목록, 고정 donor 출처가 기록됐는지 확인한다. source/receipt 불일치는 임의 우회하지 않는다.
5. 첫 신규 run의 config와 publisher에서 GF2-ablations, `.9645/.522`, 올바른 SENSOR5 v2 donor, β=0, scale=1을 확인한다.
6. 신규 센서 실험이 끝나면 기존 WV3 TUNE30→HARD36의 미완료 항목으로 이어진다.

새 코드·실행 전환·테스트·commit의 최종 검증 결과는 해당 commit과 launcher receipt를 기준으로 확인한다. 문서의 계획 표만으로 원격 실행 시작이나 전체 완료를 주장하지 않는다.

## 8. 구현 및 s1 적용 검증

- 2026-10-29 11:44 KST s1에 `enqueue_panda_sensor_focus.sh s1`을 적용했다. 새 controller는 `WAIT_RETAINED_REFINE`에서 현재 A100 Stage 1/Student의 종료를 기다린다.
- 기존 watcher만 교체했으며, 기존 학습 container를 중단하거나 재시작하지 않았다. 원본 runtime source pin과 admission의 일치를 재확인했다.
- 신규 core CPU 테스트 13개, 경계 실행·큐 제어·기존 source/WV3 불변성 테스트 24개가 통과했다. 실제 QB 44K checkpoint를 원래 container 안에서 CPU로 로드해 실제 입력의 frozen forward도 검증했다. 학습 가중치 변경이나 optimizer update는 없었다.
- W128 및 G000은 실제 CPU optimizer update와 checkpoint 재개 동등성까지 검사했다. GF2 실제 checkpoint는 s3에서만 접근 가능하므로, 로컬에서는 원본 GF2 Teacher 클래스와 합성 데이터의 학습·재개를 검증했고 s3 실행 시 원본 checkpoint/receipt SHA를 다시 검사한다.
- 새 QB early-stop은 10K 평가 파일이 저장된 직후 조건을 재확인한다. 기존 캠페인의 코드를 수정하지 않았으므로 현재 진행 중인 A100 Student의 기존 gate는 원래 구현을 유지한다.
- 적용 증거: [s1 전환 검증 JSON](../results_log/assets/2026-10-29_sensor-focus-deployment.json).
- 이 세션은 commit까지 수행하며 push는 사용자가 진행한다. s3는 pull 후 위 새 진입점을 실행해야 전환된다.
