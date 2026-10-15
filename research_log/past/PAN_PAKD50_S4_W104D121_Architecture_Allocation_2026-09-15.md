# s4 추가 실험 배정 — W104–D121 아키텍처 이식

**작성일:** 2026-09-15  
**배정:** s4 전용. s3의 W112–D123 손실·LF 실험은 유지한다. s1은 분석 전용이다.  
**분기:** `A104D121_T0FIX_E0_v1`  
**지위:** 제안·구현 인계. 이 문서는 학습 실행, Git push, Sheet 변경을 수행하지 않았다.

## 1. 결정 요약

s3에서 GT-only baseline으로 확인한 **width=104, depth=[1,2,1]**을 s4의 새 Student 후보로 시험한다. W104–D122가 아니다.

이번 첫 비교에서는 **Teacher T0 및 Teacher의 최종 aligner 자산을 고정**하고, Student U-Net만 W112–D123에서 W104–D121로 바꾼다. 새 Student는 독립 초기화한다. Teacher U-Net을 축소·복사하거나 Teacher도 동시에 새로 학습하지 않는다.

기본 5개는 `NA0 → J0 → JQ → XJ → F0`, 모두 seed1234이다. 작은 구조 안에서 no-align, joint, frozen 기준을 확보하고 Q12 및 output soft의 추가 가치를 본다. 유력하면 seed3407에서 두 구조 × 두 방법의 4개 확인 run을 수행한다.

**주의:** 첫 비교는 고정 T0에 조건부인 cross-capacity 실험이다. 현재 연구의 최종 동일 용량 Teacher–Student 주장을 그대로 적용하지 않는다. 이 단계에서 후보를 좁힌 뒤, 필요한 경우 별도 예산에서 W104–D121 Teacher/aligner와 동일 구조 Student를 준비한다.

## 2. 실제 관측과 이번 제안을 분리한다

### 2.1 s3의 W104–D121 기록

2026-09-15 live 조회: `pan-cvpr27 / WV3-s3(5090) / B28:W30`.

| Seed | Raw selected HQNR | ERGAS | Backbone params | Train(h), s3 |
|---:|---:|---:|---:|---:|
| 2025 | 0.9545 | 2.0772 | 1.9036M | 0.88 |
| 7777 | 0.9551 | 2.0892 | 1.9036M | 0.88 |
| 1234 | 0.9579 | 2.0686 | 1.9036M | 0.88 |
| 평균 | 0.955833 | 2.078333 | — | — |

Raw HQNR의 n−1 표본표준편차는 저장값에서 계산한 0.001815다. 계산 자리수가 원 측정 정밀도를 늘리는 것은 아니다.

위 결과는 **정합 없는 plain GT-only**이다. PAN aligner와 KD를 추가한 성능, s4에서의 재현성, 작은 모델의 KD 친화성은 아직 증명하지 않는다. seed1234는 과거 최고 seed이므로 이 seed 하나의 새 최고값으로 채택하지 않고 확인 seed를 분리한다.

Backbone 보고값 1.9036M에 reference aligner 0.105330M을 합하면 약 2.0089M이다. 현재 s4 W112–D123 통합 모델의 보고값 2.7642M보다 약 27.3% 작을 것으로 **추정**된다. 실제 params/FLOPs/추론시간은 새 wrapper에서 다시 측정한다. Parameter 감소율을 학습시간 감소율로 쓰지 않는다.

### 2.2 현재 s4 대조군과 완료 상태

Live 조회: `WV3-s4`, B3:AK198에서 PAKD50 행 검색. 기존 파생 묶음의 `F0/RC0/RCQ/JR/XJ/J_R3_NOEDGE`가 모두 Sheet에 있다. 마지막 `J_R3_NOEDGE`는 row19, raw 0.9524, Train(h)=1.38이다. Sheet 완료 행과 실제 실행 중 프로세스·local queue는 별도로 확인한다.

| 현재 W112–D123 case | Seed/version | Raw selected | Train(h) | Sheet 행 |
|---|---|---:|---:|---:|
| J0 | 1234/v1 | 0.9594 | 1.17 | 7 |
| JQ | 1234/v1 | 0.9558 | 1.39 | 8 |
| J0 | 1234/v2 | 0.9568 | 1.17 | 10 |
| JQ | 1234/v2 | 0.9541 | 1.40 | 11 |
| J0 | 7777/v1 | 0.9546 | 1.18 | 13 |
| F0 | 1234/v1 | 0.9553 | 1.14 | 14 |
| RC0 | 1234/v1 | 0.9551 | 1.16 | 15 |
| RCQ | 1234/v1 | 0.9554 | 1.37 | 16 |
| JR | 1234/v1 | 0.9551 | 1.37 | 17 |
| XJ | 1234/v1 | 0.9562 | 1.39 | 18 |
| J_R3_NOEDGE | 1234/v1 | 0.9524 | 1.38 | 19 |

XJ가 Q12 v1보다 0.0004 높고, edge 없는 R3는 0.9524다. 따라서 새 구조에서도 Q12만 실행하기보다 soft-off XJ를 함께 두는 편이 적절하다. 이 차이를 통계적 우위나 안정적인 late 개선으로 확대하지 않는다.

기존 v1/v2는 별도 자산이다. 비교할 W112 control의 버전·hash는 **새 결과를 보기 전에** 외부 분석에서 고정한다. 높은 값 또는 유리한 Δ가 나오는 control을 사후 선택하지 않는다. 초기 탐색의 구버전 비교는 참고로 두고, 최종 아키텍처 차이는 아래 확인 seed의 같은 release 비교를 우선한다.

## 3. 바꾸는 것과 고정하는 것

| 요소 | 이번 설정 |
|---|---|
| Student U-Net | W104–D121, 현재 plain `PANCrafterPaper` 계보. s3 성공 run의 model_args를 확인 |
| Student U 초기화 | 새 초기값. architecture+seed별 저장·strict load, 같은 architecture/seed의 모든 case가 동일 tensor |
| Teacher | T0 W112–D123의 A+U 한 쌍 frozen; 자신의 frozen aligner로 forward |
| Student aligner | NA0 없음; 나머지는 T0의 최종 aligner만 복사. J 계열 trainable, F0 frozen |
| 입출력 | PAN+MS 유효9ch, HRMS8 residual+기존 bicubic MS base 한 번 |
| Frame/sampler | 기존 M-frame, PAN만 warp, 기존 sampler/view/정규화 유지 |
| Loss | 기존 N0/Q12/X02만. 새 variance·feature·geometry 항 없음 |
| Train | 50K, batch48, U LR1e-4, A LR1e-5(J), AdamW/WD0.01, warmup100+cosine |
| J offset | λoff1e-4, 0-based 홀수 update, b2 HR 원판. Jittered PAN은 aligner만 사용 |
| Calibration | T0-fixed τR와 기존 λE0를 **의도적으로 고정 이식** |
| Evaluation | 동일 raw-original FR20 selector 및 GRID1010_50K_v1, exact50K, 고정 plateau |
| 추가 조정 | 이번 기본 묶음에서 LF/RC/AL/새 LR·계수·Teacher를 동시에 바꾸지 않음 |

Source config 조회에서는 `config/BASE_W104_D121_MSPAN_WV3_S1234.yaml`가 GitHub에 없었고 exact-name 검색도 비어 있었다. 따라서 s3의 실제 run `meta/config.yaml` 및 source revision에서 모델 설정을 전달받는다. 없는 normalization/decoder/hidden channel 세부를 논문 일반 U-Net 설명으로 대체하지 않는다.

모델 설정만 기존 성공 계보에서 확인하고, 학습·전처리·평가 schedule은 현재 PAKD 공통 규약을 사용한다. 그러므로 NA0의 성능이 이전 s3 BASE와 정확히 같아야 하는 완전 재현 실험이라고 부르지는 않는다.

## 4. 기본 실험 5개 — seed1234

아래 ID는 **계획상 run ID**다. 현재 generator가 바로 생성 가능한 CLI라고 보장하지 않는다.

| 순서 | Case | A 동작 | Backend | 핵심 비교 |
|---:|---|---|---|---|
| 1 | NA0 | 없음. PAN 원본 직접 입력, identity warp도 넣지 않음 | plain GT L1 | 같은 s4·현재 규약의 작은 backbone 기준 |
| 2 | J0 | T0 복사, 처음부터 joint + offset | plain GT L1 | J0−NA0: 정합 재사용·joint 묶음 |
| 3 | JQ | J0와 동일 | Q12 | JQ−J0: Q12 전체; JQ−XJ: output soft |
| 4 | XJ | J0와 동일 | X02=weighted hard+GT edge | XJ−J0: weighted hard+edge; soft 제거 대조 |
| 5 | F0 | T0 복사 후 계속 frozen | plain GT L1 | F0−NA0: 고정 정합의 이득; F0와 JQ/XJ 비교 |

```text
PAKD50_NA0_W104_D121_WV3_T0_S1234_FRESH50_v1
PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v1
PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v1
PAKD50_XJ_W104_D121_WV3_T0_S1234_FRESH50_v1
PAKD50_F0_W104_D121_WV3_T0_S1234_FRESH50_v1
```

NA0의 T0 표기는 parent campaign/eval-only reference와의 연결이다. Teacher output은 학습 loss에 쓰지 않으며, no-align 모델에 Teacher의 aligned PAN을 입력하지 않는다. NA0는 `A-ID/NOALIGN`의 실제 registry 규약에 맞춰 등록한다. F0는 `A-FR / I-NATIVE-TRANSFER / off0`, J는 `A-FT / I-AEQ / off1e-4`이다.

### 4.1 Loss의 역할

`L0`는 plain GT reconstruction, `LD`는 Teacher-error 기반 추가 hard, `LK`는 β를 이미 포함한 adaptive soft, `LEw=λE·LE`는 GT signed Scharr edge라 한다.

- NA0/F0: U에 L0. Aligner는 각각 없음/frozen.
- J0: U에 L0, A에 L0+LO.
- JQ: U에 L0+LD+LK+LEw, A에 같은 항+LO.
- XJ: U에 L0+LD+LEw, A에 같은 항+LO. **LK가 모델 전체에서 없음.**

Teacher·GT·gate는 detach한다. Student 출력·signed edge는 live다. Hard를 앞단 wrapper와 backend에서 두 번 계산하지 않는다.

### 4.2 읽어야 할 질문

작은 JQ가 큰 JQ보다 높아도 KD의 효과가 커졌다는 뜻은 아니다. 각 구조의 J0 대비 차이와 최종 절대 raw HQNR을 함께 본다.

- `H(JQ)-H(J0)`: Q12 전체 이득.
- `H(JQ)-H(XJ)`: 같은 weighted hard/edge에서 direct output soft의 추가 가치.
- `H(F0)-H(NA0)`: Teacher에서 얻은 정합을 고정 재사용하는 가치.
- `H(J0)-H(NA0)`: 정합+native joint+continued offset 묶음의 가치.
- `max(H(JQ),H(XJ))`와 F0/NA0의 비교: 더 단순한 대조군보다 최종 모델이 나은가?

Baseline이 강해서 ΔKD가 작더라도 최종 raw HQNR이 가장 높으면 실용 후보가 될 수 있다. 반대로 ΔKD가 커도 절대 HQNR이 낮으면 목표에 더 가까운 모델이 아니다.

## 5. Calibration: 첫 screen에서는 E0 고정

```text
tau_R     = 0.012463942170143127
lambda_E0 = 0.09075170336956798
alpha     = 1.0
beta      = 0.1
eps       = 1e-6
```

Teacher A/U, 입력·전처리가 동일하므로 Student 폭/깊이를 바꿨다는 이유만으로 Teacher-error scale τR를 새로 맞추지 않는다. **Teacher forward가 실제로 동일한지**를 사전 비교한다.

λE는 현재 W112 pilot에서 구한 E0를 Q12/XJ에 그대로 적용한다. 이는 **아키텍처만 바꾼 coefficient-transfer 비교**이다. W104–D121에서 새 gradient calibration을 했거나 해당 network parameter gradient 비가 5%라는 뜻이 아니다.

W104용 λE 재보정은 유력 후보를 좁힌 뒤의 별도 변형으로만 진행한다. Q12와 XJ는 반드시 같은 λE를 사용한다. 이번 계획에는 그 재보정 학습을 추가하지 않는다.

## 6. 확인 실험 — seed3407, 최대 4개

탐색 결과·외부 분석을 바탕으로 `WIN ∈ {JQ, XJ, F0}`을 하나 고정하고, 같은 release와 실제 모델 설정·계수·Teacher·selector를 잠근 뒤 실행한다. **s4의 기존 3407 확인 예약을 이 묶음으로 대체**하며, 이전 확인 큐와 중복 실행하지 않는다.

| 순서 | Architecture | Case | 목적 |
|---:|---|---|---|
| 6 | W104–D121 | J0 | 새 seed에서 작은 구조의 공동학습 기준 |
| 7 | W104–D121 | WIN | 작은 구조 후보의 반복 |
| 8 | W112–D123 | J0 | 같은 서버·seed·release의 원래 구조 기준 |
| 9 | W112–D123 | 같은 WIN | 아키텍처 차이와 방법 추가 이득의 비교 |

같은 정수 seed라도 두 아키텍처의 초기 weight는 shape가 다르므로 같을 수 없다. **구조 안에서 control과 treatment의 초기 tensor가 동일**해야 하고, data/augmentation/epsilon RNG는 모델 초기화의 RNG 소비와 분리해 비교 순서를 맞춘다.

WIN이 JQ 또는 XJ이면:

$$
I_s=[H(WIN)-H(J0)]_{104D121,s}-[H(WIN)-H(J0)]_{112D123,s}.
$$

동시에 `H(WIN_104)-H(WIN_112)`로 최종 아키텍처 차이를 기록한다. WIN이 F0이면 같은 차이는 **정합 업데이트 정책 차이**이며 KD 순증분이라고 부르지 않는다.

NA0만 좋고 정합 후보들이 모두 유리하지 않으면, 통합 후보를 억지로 WIN으로 승격하지 않는다. 이 경우 최대 4개 확인 묶음은 자동 실행하지 않고, no-align architecture 발견과 정합 결합의 한계를 보고한다. 확인 seed를 본 뒤 다시 α/β/LF를 맞추면 더 이상 untouched 확인으로 세지 않는다.

탐색1234+확인3407만 완료하면 신규 방법은 2-seed이다. 과거 s3의 GT-only 3seed, 다른 architecture 결과, 같은 seed의 서버 반복을 신규 통합 방법의 독립 seed에 합치지 않는다.

## 7. 무엇을 이번 큐에서 하지 않는가

s3는 W112–D123에서 soft/edge 분해와 LF/LFX를 담당하고, s4는 backbone 이식에 집중한다. s4의 첫 W104 run부터 LF·RC·작은 A LR·새 λE를 동시에 적용하지 않는다.

W104–D122, W112–D121, attention 추가, decoder 재배분의 전수 탐색도 열지 않는다. W104–D121의 결과가 유력한 뒤에야 폭과 깊이를 따로 분리하거나 s3에서 선정한 LF 정책을 한 묶음으로 옮긴다. 이런 추가 학습은 현재 9run 예산에 포함되지 않는다.

## 8. 실제 Sheet 시간 기반 예약

### 8.1 관측값과 대용값을 구분한다

현재 s4의 실제 W112–D123 시간은 J0 1.17–1.18h, JQ 1.39–1.40h, XJ1.39h, F0 1.14h이다. 반면 **s3의 W104–D121 0.88h는 no-align GT-only**이므로 s4의 aligned KD 실측 시간으로 사용할 수 없다.

이번에는 새 구조의 속도 이득을 미리 차감하지 않고 s4의 더 큰 기존 구조 시간을 계획 대용값으로 사용한다. 이는 보수적인 출발값이지 보장된 상한이 아니다. Teacher의 W112 forward, evaluator, I/O는 그대로이므로 parameter가 약27% 줄었다고 run도27% 짧아진다고 계산하지 않는다.

| Case | 대용 Train(h) | 출처/가정 |
|---|---:|---|
| NA0-104D121 | 1.18 | s4 큰 J0 관측값 대용. 새 no-align 시간 미측정 |
| J0-104D121 | 1.18 | s4 큰 J0 관측값 대용 |
| JQ-104D121 | 1.40 | s4 큰 JQ v2 관측값 대용 |
| XJ-104D121 | 1.39 | s4 큰 XJ 관측값 대용 |
| F0-104D121 | 1.14 | s4 큰 F0 관측값 대용 |
| **기본 5run 합** | **6.29** | 단일 s4 순차 합 |

### 8.2 예약식 검산

기존 식 `reservation_h = 1.10 × reference_train_h + 10/60`을 run당 한 번 적용한다. 10%는 실행변동, 10분은 기본 후처리·gspread·큐 전환의 가예약이다. 주기 평가가 포함된 run 시간을 기준으로 하므로 평가50회를 다시 더하지 않는다.

- 기본5개: `1.10×6.29 + 5×10/60 = 7.752333h` = **7시간45분08초**, 약7시간45분.
- 확인4개: 두 J0를1.18h씩, 두 WIN을1.40h씩 예약. 기준합 `2×1.18+2×1.40=5.16h`.
- 확인 예약: `1.10×5.16 + 4×10/60 = 6.342667h` = **6시간20분34초**, 약6시간21분.
- 최대9개 기준합: **11.45h**.
- 최대9개 예약합: **14.095h** = **14시간05분42초**, 약14시간06분.

이는 **한 서버의 순차 경과시간**이다. s3 예산을 여기에 합치지 않는다. 현재 실행 중인 job 잔여시간, source config 전달, 구현·Git 배포·smoke, 외부 분석의 WIN 결정 대기는 별도다.

첫 새 NA0/JQ의 측정이 확보되면 `(server, architecture, policy, backend)`별 실측으로 교체한다. 현재 코드의 `gate_hours = reservation_h / margin` 전달 규약을 유지해 trainer가 margin을 두 번 적용하지 않도록 한다. 새 profile에는 고정4h placeholder를 사용하지 않는다.

### 8.3 마감

기존 캠페인 시계를 유지한다.

```text
training_deadline = 2026-09-16 11:22:31 Asia/Seoul
final_deadline    = 2026-09-16 15:22:31 Asia/Seoul
```

현재 시각과 실제 슬롯 가용 시점으로 잔여시간을 계산한다. 마지막4h는 학습 마감 뒤에 이미 남아 있으므로 다시 차감하지 않는다. Architecture 분기를 추가했다고 50h 시계를 새로 시작하지 않는다.

확인 묶음은 비교에 필요한 전체 run을 마감 안에 마칠 수 있을 때만 admit한다. 부족하면 기본5개를 먼저 마치고, 확인은 축소한 결과 범위를 명시한다. 실험 도중 유리한 모델만 남기기 위해 case별 학습량을 다르게 줄이지 않는다.

## 9. 구현 인계: hidden_size 두 줄 수정만으로 끝내지 않는다

검토 Git snapshot은 `2d0fc25e54aa190688031f2feba0ad2a22db506c`이다. 기존 trainer에는 architecture-aware 기본 init_dir가 있지만 캠페인 generator·seed hash·run 이름은 별도 경로를 갖는다.

### 9.1 모델 로딩과 초기값

- Teacher는 T0 자체 config의 W112–D123으로 load한다. Student model_args를 Teacher loader에 전달하지 않는다.
- Student U-Net은 W104–D121로 새로 만든다. Teacher U state를 부분복사, channel slicing, `strict=False` 누락 허용으로 끼워 맞추지 않는다.
- Student A만 T0의 aligner 전체 state를 strict copy한다. 초기 c0와 aligner hash가 같아야 한다.
- 기본 `work_dir/_kdv_init_w104_d121/` 또는 동등한 별도 init_dir를 사용한다. 명시 config가 기존 w112 디렉터리를 덮어쓰지 않는지 검사한다.
- seed-only W112 기대 hash를 W104에 강제로 적용하거나 init 보호 전체를 끄지 않는다. `(architecture_signature, seed)`로 init hash namespace를 나눈다.
- Architecture signature에는 width/depth뿐 아니라 model class, decoder 설정, norm, 입출력·attention·task flags를 포함한다.

### 9.2 Run/queue/campaign 식별

`run_name`, `case_of`, queue 검색, config 생성, donor·pilot 찾기에서 `_W112_D123`을 고정 가정하는 부분을 architecture-aware하게 처리한다. `--width` 같은 새 CLI가 이미 있다고 가정하지 않는다.

기존 parent campaign의 deadline·Teacher·calibration binding은 유지하고 별도 `experiment_branch_id=A104D121_T0FIX_E0_v1`을 추가한다. Legacy campaign_id의 W112 문자열만 보고 Student를 W112로 로드하면 안 된다. 실제 architecture는 resolved model_args와 명시된 signature가 기준이다. 저장소의 기존 calibration_id 비교를 깨뜨리지 않도록 branch identity를 parent와 분리한다.

서버 예약도 case 이름만이 아니라 architecture까지 구분한다. W104 첫 실측을 W112 시간으로 덮어쓰거나 반대로 적용하지 않는다.

### 9.3 최소 smoke / 회귀 검사

| 검사 | 기대 결과 |
|---|---|
| 104D121 모델 구성 | hidden_size104, depth[1,2,1], s3 성공 model 구조의 나머지 설정 일치 |
| 입력/출력 | 기존 64/256/512 입력 경로에서 HRMS8 shape 및 residual base 한 번 |
| T0 forward | Student architecture만 바꿔도 Teacher 출력과 τR 입력조건은 동일 |
| A 재사용 | T0 A strict copy, 초기 state 및 native correction 일치 |
| NA0 | aligner·sampler 없음, GT-only. Teacher는 loss 미사용 |
| F0 | A state 불변, U gradient 존재 |
| J0/JQ/XJ | 지정한 native gradient 경로, offset은 A만, Teacher/gate gradient 없음 |
| Soft 제거 | JQ의 β=0이 같은 입력에서 X02 목적함수와 일치 |
| Init 재사용 | 같은 architecture/seed의 모든 case가 동일 U state; 다른 architecture의 hash 혼입 없음 |
| 112 회귀 | 기존 W112 defaults·run 경로·Teacher asset·init hash 보호 유지 |
| Resume | 중단 전후 RNG·optimizer·global schedule·grid가 기존 규약과 일치 |

이는 실행 전 요구사항이다. 문서 작성으로 이 CUDA 검사가 통과됐다고 보고하지 않는다.

## 10. gspread: WV3-s4를 그대로 사용

직접 Sheet를 수정하지 않는다. 기존 run 종료 후 `_upload.sh → gspread_upload.py` 경로에서 `WV3-s4`에 올린다.

- B열 run ID에 실제 `W104_D121`을 포함한다.
- C열은 PAKD50 범주를 유지한다.
- N열은 raw-original HQNR, O열은 기존 V64다. 값을 교체하지 않는다.
- Q열 params/FLOPs 등 비용은 새 Student+A의 실제 추론 모델에서 측정한다. Teacher params를 합하지 않는다.
- W열 Notes에는 `Student=W104D121; Teacher=T0/W112D123; A_source=T0; branch=A104D121_T0FIX_E0_v1; E0=fixed-transfer`를 넣는다.
- 기존 X열 `통합실험`에는 예를 들어 `PAKD50 / JQ / A104D121 / FRESH50`를 쓴다. NA0/J0/F0 대조군도 기록한다.
- 기존 B:W 배치와 예전 행은 유지한다. 새 아키텍처 때문에 새 spreadsheet나 별도 tab을 만들 필요는 없다.
- Aligned Student와 no-align NA0에서 비용 모델 wrapper의 A 포함 여부를 정확히 반영한다.

## 11. 판정과 후속 분기

공식 목표는 **selected raw-original HQNR0.959–0.960**이다. Exact50K와 고정 plateau를 별도로 함께 남기고 평가 기준을 사후 교체하지 않는다. ERGAS/SAM/fSCC 비용을 삭제하지 않고, seed·선택 시점의 변동도 같이 읽는다. W112에서 측정한 경험적0.0031을 새 아키텍처의 통계적 유의성 경계로 자동 적용하지 않는다.

| 관측 | 판정/다음 행동 |
|---|---|
| W104 JQ/XJ가 자기 J0와 F0/NA0보다 좋아짐 | WIN 고정 후 cross-architecture 확인4개 |
| JQ와 XJ의 우위 반전 | 해당 구조·seed에서의 soft 상호작용으로 기록. 높은 하나만 근거로 일반 법칙 금지 |
| F0만 유리 | 정합 고정 재사용/architecture 후보. KD 성공이라고 부르지 않음 |
| NA0가 모든 aligned case보다 좋음 | 작은 no-align 모델의 강점과 정합 결합의 비용을 보고. 통합 노벨티 유지를 위해 불리한 결과를 숨기지 않음 |
| Selected만 상승, last/plateau 비용 큼 | 높은 자산과 후반 안정화를 분리. s3 LF를 옮길 근거가 있는지는 별도 확인 |
| 확인seed에서 방향이 뒤집힘 | 탐색seed 의존성으로 보고; 새계수 탐색을 확인결과처럼 합치지 않음 |

최종 연구를 동일 용량 fitting으로 구성할 경우, 이 첫 T0-fixed 결과를 그대로 최종 method 검증으로 쓰지 않는다. 유력 architecture를 선정한 뒤 **같은104D121 Teacher/aligner를 준비하고104D121 Student와 대응 대조**를 별도 수행한다. 그 준비 비용과 새로운 τR/λE package는 현재9run에 포함하지 않았다.

## 12. 실행 순서와 인계 범위

1. 기존 s4 run이 실행 중이면 그 run 경계까지 기다리고 중복 큐를 막는다. 이미 완료된 old-case 묶음은 다시 넣지 않는다.
2. s3 성공 모델의 resolved model 설정과 현재 PAKD release를 받아 새 architecture binding을 구현·검사한다. s1에는 full training을 추가하지 않는다.
3. 기본5개를 같은 release·seed1234에서 실행한다. 첫 실제 런타임으로 예약값을 갱신한다.
4. 외부 분석이 고정한 WIN을 받은 뒤 seed3407 확인4개를 실행한다. 새결과를 보고 Teacher·계수·architecture를 다시 바꾸지 않는다.
5. 기존 gspread 경로로 실제 완료 결과만 업로드한다. 마지막 공통 감사 구간은 기존 운영을 유지한다.

## 근거 자료

- [S1] `pan-cvpr27 / WV3-s3(5090)`, 2026-09-15 live `B28:W30`. 동반 `source_snapshot.json`에 입력값 보존.
- [S2] `pan-cvpr27 / WV3-s4`, 같은 날짜 live PAKD50 rows. source snapshot에 run/row/HQNR/ERGAS/runtime 보존.
- [S3] `PAN_Architecture_Size_KD_HQNR_Analysis_2026-09-14.md` §3. baseline의 성능과 KD 순증분을 구분한 이전 분석.
- [S4] `PAN_Aligner_L1E4_TechnicalSpec_Evidence_KD_Handoff_2026-09-14.md` §§1,3,4,5,6. A 재사용·frame·gradient·공식 평가 규약.
- [S5] `PAN_KD_Q12_R1_Spec_and_Aligner_Interface_2026-09-14.md` §§3–7,9. Q12/X02, final-HRMS interface, calibration 및 Teacher 입력 고정.
- [S6] `hojunking/PAN-Crafter-repro @ 2d0fc25e54aa190688031f2feba0ad2a22db506c`: `tools/gen_pakd50_configs.py`; `train_kdv.py`의 init_dir 및 기대hash 검색 결과. 고정/신규 구현 경계를 위에서 구분했다.

동반 `case_registry.yaml`은 **계획용 registry이지 trainer config가 아니다**. `core_queue_plan.csv`는 계획상의 실행명이며 실제 실행을 시작하는 shell script가 아니다. `runtime_budget.json`은 시간 산식의 검산값이다.
