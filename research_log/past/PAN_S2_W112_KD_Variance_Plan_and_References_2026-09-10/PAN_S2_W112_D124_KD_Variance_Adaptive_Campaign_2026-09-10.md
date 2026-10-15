# s2 · W112–D124: GT 중심 KD·variance·aligner 재사용 통합 실험/구현 계획

**작성일:** 2026-09-10  
**캠페인 ID:** `S2_W112_KDV_20260910`  
**문서 버전:** `v1.0` — 실행하면서 분기·개정할 수 있는 우선순위형 계획  
**실행 서버:** **s2**. 서버의 실제 GPU·VRAM·CUDA·PyTorch·저장소 경로는 시작 시 확인한다.  
**시간 운영:** **약 24시간을 첫 검토 시점으로 삼되, 강제 종료하지 않는다.** 필요한 대조군 완주와 유력 방향 검증이 우선이며 24시간을 크게 넘어도 진행할 수 있다.  
**실행 상태:** 이 문서는 실험 계획과 참조 구현이다. s2 접속·실제 데이터 학습·W112 모델 통합·GPU 시간 측정은 수행하지 않았다.

> **이번 방향**
>
> 1. 복원 backbone은 **W112–D124**, 입력은 WV3 **PAN 1ch + MS 8ch = 9ch**로 변경한다.
> 2. Reconstruction은 **GT를 모든 위치에서 유지하는 연속 가중 hard/soft**를 계승한다. 이진 switch로 돌아가지 않는다.
> 3. Teacher aligner 재사용을 우선하지만 **frozen / Teacher 초기화 후 trainable / scratch**를 모두 실험 축으로 남긴다.
> 4. **출력 영상·gradient의 통계 variance**와 **정합 이동량 covariance**는 목적과 gradient 경로가 다른 별도 축이다.
> 5. 모든 현재 접근을 우선순위 목록에 남기되, 무의미한 전체 조합을 만들지 않는다. 구현·진단 → 대응 screening → 유력 구성 완주 → 확장 순으로 진행한다.
> 6. 중간 변경은 허용한다. 다만 checkpoint·설정·Teacher가 바뀐 시점을 기록하고, 동일 시작점의 continuation 대조를 남긴다.

---

## 0. 출처·결정·제안의 구분

### 0.1 출처 표기

이 문서의 `[S-KD]`, `[S-A]`, `[S-N]`, `[S-BASE]`는 말미의 첨부 문서 목록을 가리킨다. 해당 문서에 적힌 설정·결과와 **이번 캠페인에서 새로 제안한 선택**을 구분한다. 과거 문서에 기록된 성능은 W112/s2 실측값으로 옮기지 않는다.

**우선순위:** 이번 사용자 결정 → 충돌하지 않는 최근 합의 → 확인한 최신 실행 manifest → 첨부 설계 문서 → 이 문서의 신규 제안.

현재 자료에는 s2의 실제 W112 U-Net 코드, W112 Teacher checkpoint, 최신 resolved config, 실제 처리량이 없다. 이 값은 `REQUIRED`로 남긴다. 없는 checkpoint나 결과를 만들어 채우지 않는다.

### 0.2 이번 사용자 결정으로 바뀐 것

| 항목 | 기존 문서 | 이번 적용 |
|---|---|---|
| 주 서버 | s1 중심 또는 서버별 block | **s2에서 같은 환경의 대응 비교** |
| Backbone width | 96 | **112** |
| Depth | `[1,2,4]` | **유지** |
| 시간 | 10 GPUh 또는 기존 별도 예산 | **약 24h review, hard timeout 없음** |
| 중간 변경 | run 내 고정 원칙 위주 | **명시적인 phase/fork로 변경 허용** |
| Aligner 초기화 | 독립 초기화 또는 frozen 제안 | **Teacher 재사용 우선, frozen/trainable 미확정, scratch 포함** |
| Variance | 이동 covariance 또는 출력 통계가 번갈아 논의됨 | **두 축을 분리해서 모두 검증** |

### 0.3 유지하는 조건

| 항목 | 적용 |
|---|---|
| 센서 | 우선 WV3. 다른 센서는 이 캠페인의 후속 확장 |
| 모델 | MARs-free W112–D124 복원 U-Net |
| 입력 | 실제 checkpoint/config의 순서를 확인한 `[PAN, bicubic MS(8)]` 9ch |
| 출력 | 8ch HRMS residual. Wrapper에서 MS base를 정확히 한 번 더함 |
| 제거 유지 | PAN reconstruction, LPAN/HPAN 입력, dual MARs, attention/CM3A |
| 좌표계 | MS condition 원본, residual base, GT, 최종 출력은 M-frame |
| PAN 정합 | PAN **영상 1ch**만 sampling. Output inverse warp 없음 |
| GT 사용 | RR 학습 loss·통계 calibration에만 사용. 추론 입력에는 없음 |
| KD | 일단 frozen Teacher. EMA/mutual learning은 자동 도입하지 않음 |
| 추론 | 해당 run의 Student aligner(또는 frozen donor) + Student U-Net |
| MS 보간 | 기존 bicubic 경로·phase·intensity convention 유지 |

`nocrop`은 기존 데이터 증강/복원 영상 경로의 기준이다. N계열의 **aligner-only 내부 crop**까지 금지한다는 뜻이 아니다. 둘을 별도 config로 기록한다. [S-BASE §3; S-A §§2–4; S-N §5]

### 0.4 채택과 보류의 의미

`P2/P3`는 시간 부족을 이유로 버린 접근이 아니라 **선행 검증 뒤에 실행하는 순서**다. 모든 현재 접근에 상태를 남긴다:

`PENDING → IMPLEMENTED → UNIT_PASS → SMOKE_PASS → SCREENED → CONFIRMED / NEGATIVE / REVISE_REQUIRED`.

논리적으로 적용 불가능하면 `NOT_APPLICABLE`로 이유를 기록한다. 예: 같은 frozen aligner에 이동량 KD를 추가하는 경우. 새 uncertainty head는 현재 확정 범위가 아니므로 비교 이유가 생기기 전에는 `PARKED_SCOPE`다.

---

## 1. 전체 우선순위와 가장 먼저 확인할 질문

| 우선순위 | 작업 | 먼저 해야 하는 이유 | 완료 판정 |
|---|---|---|---|
| **P0** | s2 환경·W112 이행·Teacher 출처·sampling·gradient 확인 | 폭 변경이나 잘못된 checkpoint가 loss 실험을 무효화할 수 있음 | 실제 배치 smoke와 출처 manifest |
| **P1-a** | Frozen aligner 아래 reconstruction N0/R1/R3 | 기존 hard fitting과 output KD 효과 분리 | 같은 budget의 기준·대조 |
| **P1-b** | GT gradient variance / GT+Teacher adaptive 통계 | 이번에 우선하려는 variance의 직접 효과 | GT-only와 adaptive 비교 |
| **P1-c** | Teacher aligner frozen ↔ fine-tune ↔ scratch | 고정이 유리하다고 선결정하지 않음 | 같은 loss의 세 초기화/학습 정책 |
| **P2-a** | R0/R2, Teacher-only/fixed 통계, 영상 분산, 직접 edge | P1 개선의 원인을 분해 | 주요 대조군 screening |
| **P2-b** | Gradient 공분산·분광 공분산·복합 통계 | 공간 방향/밴드 관계의 추가 정보 확인 | 같은 reduction·calibration 원칙의 비교 |
| **P2-c** | Trainable aligner의 일반 mean/scalar/full covariance KD | 원래의 이동량 variance 접근 보존 | 신뢰도 산출 gate 및 matched KD 비교 |
| **P3-a** | Covariance diagonal/rotation/shuffle·구조 민감도 | 방향별 정보가 실제로 필요한지 확인 | 부정 대조 및 기전 분석 |
| **P3-b** | Student covariance head·분포 KD, GT-only 통계의 scratch 검증 | 더 복잡한 모델의 추가 이득 확인 | 별도 head 감독·복원 기여 확인 |
| **P3-c** | Seed 반복·동일 비용 장기 fitting·Teacher 초기화 | 작은 개선의 안정성 및 추가 학습 효과 분리 | 주요 쌍 최소 3 Student seeds |

**첫 주력 후보:** `A-FR + REC-R3 + STAT-GV-AD`.

이는 frozen aligner가 최종 정답이라는 뜻이 아니라, **loss 효과를 가장 먼저 분리하기 위한 시작 cell**이다. 같은 loss의 `A-FT`, `A-SC`를 뒤이어 비교한다. P1에서 정합 고정이 명백히 병목이면 P1-c를 앞당긴다.

---

## 2. W96 → W112 이행: 단순 이름 변경으로 처리하지 않는다

### 2.1 적용 범위

동일 구조 KD라는 기존 조건을 이어 **정식 주 비교의 Teacher와 Student 복원망을 모두 W112–D124**로 둔다. 이것은 이번 width 변경을 기존 동일 구조 원칙과 결합한 실행 해석이다.

- 기존 W96 U-Net의 `width` 설정만 112로 바꿔 로드하지 않는다.
- W96 convolution·normalization tensor는 W112와 shape가 다를 수 있다. `strict=False`로 조용히 누락시킨 뒤 pretrained W112라고 부르지 않는다.
- 채널 절단·반복·보간으로 가중치를 옮기는 실험은 별도 weight-morphing 방법이다. 이번 기본 경로에는 없다.
- 기존 aligner가 독립적인 `1/8 → 16/16 → 32/64 → 2` 구조라면 **U-Net width와 별개로 aligner만 가져올 수 있다**. 반드시 해당 모듈에 대해 key/shape를 strict 검사한다. [S-A §3]

### 2.2 반드시 출력할 architecture manifest

```text
backbone_class / source_file / commit
width=112, depth=[1,2,4]
각 encoder·bottleneck·decoder의 실제 width와 block 수
입력 9ch / 출력 8ch / MS base 합산 위치
normalization 종류·group 수·affine 설정
MARs 전용 mode parameter 존재 여부
PAN auxiliary forward 횟수=0
aligner module class / channel 명세 / source checkpoint
전체 params / trainable params / frozen params
```

문서만 보고 decoder 반복 수나 모든 stage가 `[112,224,448]`이라고 새로 가정하지 않는다. **현재 W96 코드의 채널 배율·skip 결합·decoder depth 해석을 보존하고, base width만 바뀌었는지 실제 모듈 트리로 확인**한다.

GN을 쓰는 layer가 있다면 `channels % num_groups == 0`을 검사한다. 예를 들어 112는 32로 나누어떨어지지 않는다. 이는 확인할 가능성이지, 기존 복원망이 GN32라는 주장이 아니다. 안 맞으면 임의로 group 수를 바꾸지 말고 새 normalization 결정과 대조 필요성을 기록한다. Aligner의 기존 GN4/GN8 구조는 width 변경과 별개다.

### 2.3 비용 예측의 제한

내부 채널이 폭에 비례하는 convolution의 계산/가중치 규모만 보면 `(112/96)^2 = 49/36 ≈ 1.361`의 증가가 예상되는 부분이 있다. **전체 params·메모리·속도가 36.1% 증가한다는 뜻은 아니다.** W112 실제 forward로 train 64², RR 256², FR 512²에서 측정한다.

s2를 이름만으로 RTX 4090이라고 단정하지 않는다. 24GB 환경으로 확인되면 §17의 메모리 정책을 적용한다.

---

## 3. Teacher 확보와 앞단 후보 계승

### 3.1 Teacher asset을 두 부분으로 관리한다

| Asset | 필요한 것 | 사용 |
|---|---|---|
| **Aligner donor** | `A_donor` weight + crop/normalization/warp manifest | Student 초기화·frozen 전처리 |
| **KD Teacher** | `A_T + F_T(W112)` + 정확한 forward/config | output target·오차 map·통계 target |

W96 캠페인의 aligner를 W112 Student에 가져올 수 있다고 해서 **W96 복원망이 W112 Teacher로 바뀌는 것은 아니다.** Donor 출처와 KD Teacher 출처를 별개로 적는다.

### 3.2 Teacher readiness 분기

1. **검증된 W112 Teacher가 이미 있다:** checkpoint와 source hash를 읽고 재평가한다. 같은 graph의 aligner를 donor로 사용한다.
2. **W96 Teacher/aligner만 있다:** 우선 독립 aligner만 strict하게 옮긴다. W112 Teacher 복원망은 새로 학습하거나 실제 W112 checkpoint를 확보한다. W112 Teacher 학습시간도 총 비용에 넣는다.
3. **당장 W112 Teacher가 없으나 loss 구현 선별이 필요하다:** `T96→S112_BRIDGE`를 별도 exploratory cohort로 허용할 수 있다. 이는 같은 용량 KD가 아니다. 정식 W112→W112 cell이 준비되면 비교를 다시 확인한다. 묵시적으로 실행하지 않는다.
4. **사용할 만한 aligner가 없다:** W112 no-align baseline과 scratch aligner supervised Teacher를 확보한다. GT-only 통계 실험은 Teacher 없이도 시작할 수 있다.

Teacher 결과가 중간에 도착해도 기존 run의 target을 덮어쓰지 않는다. `teacher_id`가 바뀌면 새 cohort/phase를 만들고 tau·통계 calibration을 다시 만든다.

### 3.3 Teacher recipe는 선택된 앞단의 실제 이력을 따른다

| Recipe | 기존 forward/loss | Student trainable일 때 계승할 감독 |
|---|---|---|
| **A1** | Native, recon | Recon만 |
| **A2** | Native, recon + bandwise Scharr edge | 같은 output edge 항 |
| **A3** | Native, recon + 보정 PAN–GT 구조 | 동일 geometry descriptor·mask·정규화 |
| **N1** | Native/corrupt 1:1, recon | 입력 프로토콜 유지, offset 계수 0 |
| **N2** | N1 + SG offset consistency | corrupted step의 SG 관계 |
| **N3** | N1 + no-SG offset consistency | 양쪽 gradient의 동일 관계 |

A2와 A3는 누적 조합이 아니고, N계열은 원래 A2/A3를 함께 넣지 않는다. **현재 어떤 recipe가 좋은지 미확정이므로 N2나 A3를 winner로 미리 정하지 않는다.** [S-A §5.4; S-N §0]

W112 Teacher가 새로 필요하면 donor가 유효한 경우 **donor frozen + W112 복원 supervised**로 먼저 확보하는 단순 경로가 가능하다. 이는 A/N 공동학습 Teacher와 다르므로 `T112_DONORFROZEN_REC`라는 새 recipe다. Joint adaptation이 필요하면 동일 추가 budget의 frozen continuation과 비교한다.

### 3.4 Teacher 적격성

출력 Teacher와 정합 Teacher의 적격성을 따로 기록한다.

- 출력: train/validation GT MAE·ERGAS, Teacher보다 나은 Student 영역 존재 여부, 경계·분광 왜곡.
- 정합: native 품질, 추가 변위 closure, response slope, MS 교체 의존성, learned/zero/wrong-sign 대조.
- Aligner correction이 상수에 가깝다는 것만으로 실패를 확정하지 않는다. 다만 알려진 추가 이동에도 반응하지 않으면 강한 alignment KD의 근거는 약하다.
- 기존 N 문서가 재인용한 s1 반응 기울기 `−0.02~−0.09`는 W112/s2 결과가 아니다. 최신 실제 로그로 확인한다. [S-N §1]

---

## 4. Aligner 아키텍처와 세 초기화 모드

### 4.1 초기 reference: 기존 dual-stem 유지

| Layer | 명세 |
|---|---|
| PAN stem | Conv3×3 `1→16`, GN4, SiLU |
| MS stem | Conv3×3 `8→16`, GN4, SiLU |
| Joint | concat32 → Conv3×3 `32→32` stride2, GN8, SiLU |
| Joint | Conv3×3 `32→64` stride2, GN8, SiLU |
| Residual | 64ch ResBlock 1개 |
| Regression | GAP → Linear64→32 → SiLU → Linear32→2 |

기존 문서의 reference count는 **105,330 parameters**다. 실제 s2 구현에서 재계산한다. U-Net을 W112로 바꾼다고 aligner까지 폭 112로 키우지 않는다. [S-A §3.3]

PAN과 MS를 따로 global pooling한 뒤 결합하지 않는다. 먼저 같은 격자의 공간 feature를 결합하고 joint convolution을 거친 뒤 GAP한다. Aligner용 z-score만 sample·band별로 적용하고, 원 PAN/MS의 방사값은 U-Net과 warp에 보존한다.

### 4.2 정책별 그래프

| ID | Student aligner 초기화 | 업데이트 | Teacher 상태 |
|---|---|---|---|
| **A-FR** | Teacher aligner 복사 또는 immutable 공유 | 없음 | `A_T,F_T` 모두 고정 |
| **A-FT** | `A_S ← deepcopy(A_T)` | 수행 | Teacher는 계속 고정 |
| **A-SC** | 독립 random initialization, 마지막 shift head만 0 | 수행 | Teacher는 계속 고정 |
| **A-ID** | aligner 없음, raw PAN | 없음 | no-align/no-KD 대조 또는 명시적 bridge |
| **A-FTW** | Teacher 복사 후 warm freeze→unfreeze | 단계적 | P3의 별도 phase 실험 |

A-FT의 초깃값이 Teacher와 같아서 이동 KD=0인 것은 정상이다. 이후 recon/auxiliary가 Student correction을 바꾸면서 KD가 작동한다.

**A-FT/A-SC에서 동일 Python module 객체를 Teacher와 공유하지 않는다.** Optimizer에 Teacher parameter가 들어가지 않았는지 parameter ID 교집합으로 검사한다.

### 4.3 복원 U-Net 초기화는 별도 축

초기 주 비교는 모든 Student `F_S(W112)`에 **같은 저장된 독립 초기값**을 사용한다. Aligner만 Teacher에서 가져와도 Student 출력이 Teacher와 같아지는 것은 아니다.

`F_S ← F_T` 복사는 P3의 continuation 실험이다. 이때 동일 초기 상태의 KD 없는 continuation을 붙인다. 처음 `e_S=e_T`이면 adaptive soft는 0이다. 이것을 코드 오류로 보지 않는다. [S-KD §10]

Scratch는 초기화 방식이며 KD 제거를 의미하지 않는다. `A-SC + REC-R3`, `A-SC + GT-only`, 전체 Student scratch를 서로 구분한다.

---

## 5. Forward 계약·입력 프로토콜·sampling

### 5.1 좌표와 기본식

$$
M=U_4(S),\quad \mu_i=A_i(P,M),\quad
\widetilde P_i(y,x)=P(y+\mu_{i,y},x+\mu_{i,x}),
$$
$$
\hat Y_i=M+F_i([\widetilde P_i,M]),\qquad i\in\{T,S\}.
$$

Correction은 `(dy,dx)`, 단위는 **현재 PAN HR pixel**이다. `grid_sample` grid는 마지막 축 `(x,y)`이다.

$$
g_x=2(x+0.5+\mu_x)/W-1,\quad g_y=2(y+0.5+\mu_y)/H-1.
$$

Sampler: `mode='bicubic'`, `padding_mode='border'`, `align_corners=False`, FP32. Learned shift가 0이어도 sampling을 건너뛰지 않는다. A-ID baseline의 sampler 부재는 명시적 아키텍처 대조이며, trainable head의 zero shortcut과 다르다. [S-A §4]

### 5.2 A-FR의 공유 입력

같은 view의 aligner가 frozen이면 correction과 정합 PAN을 한 번 계산하고 두 복원망에서 재사용할 수 있다.

```text
P,M → frozen A* → warp → P_aligned ─┬→ frozen F_T → Y_T
                                  └→ trainable F_S → Y_S
GT Y ─────────────────────────────────→ losses only
```

Frozen input tensor여도 Student의 convolution weight에는 gradient가 정상적으로 생긴다. U-Net forward 전체를 `no_grad`로 감싸지 않는다.

### 5.3 A-FT/A-SC의 Teacher 입력

Teacher는 **자기 frozen aligner가 만든 PAN**을 계속 본다. Student correction으로 PAN을 만든 뒤 frozen Teacher U-Net에 넣어 Teacher target을 변경하지 않는다.

두 모델의 raw PAN/MS, augmentation, corruption ε는 같지만 각자의 learned warp 결과는 다를 수 있다. 출력은 모두 같은 GT/M-frame에서 비교한다.

### 5.4 Native vs N계열 입력

| Protocol | 학습 | Aligner view | Warp 횟수 |
|---|---|---|---|
| **I-A** | 추가 PAN corruption 없음 | donor의 실제 full-view 규칙 | native 1회 |
| **I-N** | native/corrupt optimizer update 1:1 | donor의 고정 내부 crop | native 1회, corrupt 순차 2회 |
| **I-NATIVE-TRANSFER** | N donor를 쓰되 Student corruption 제거 | N donor crop은 유지 | 1회 |

I-NATIVE-TRANSFER는 별도 domain-transfer 대조다. Frozen으로 바뀌었다는 이유만으로 I-N에서 corruption을 자동 제거하지 않는다. 한 variance family에서는 모든 case가 같은 protocol을 사용한다.

N reference의 R=1 HR pixel은 train audit P90에서 얻은 **기존 임시 규모 proxy**다. 적용하려면 출처·단위·현재 s2 config를 확인한다. W112 변경과 radius 변경을 묶지 않는다. 학습 가능한 output을 `tanh×R`로 제한하지 않는다. [S-N §3]

### 5.5 N forward와 self-consistency

$$
P_\epsilon=\mathcal W(P,\epsilon),\quad
\mu_{i,0}=A_i^V(P,M),\quad
\mu_{i,\epsilon}=A_i^V(P_\epsilon,M),
$$
$$
\widetilde P_{i,\epsilon}=\mathcal W(P_\epsilon,\mu_{i,\epsilon}),\quad
\hat Y_{i,\epsilon}=M+F_i([\widetilde P_{i,\epsilon},M]).
$$

SG offset term:

$$L_{off}^{SG}=\operatorname{mean}|\mu_{S,\epsilon}+\epsilon-\operatorname{sg}(\mu_{S,0})|.$$

No-SG는 위 식에서 `sg`만 제거하고 clean call에도 graph를 둔다. N2의 clean reference는 **같은 Student의 현재 출력**이다. Frozen KD Teacher correction과 혼동하지 않는다.

초기값 0.01/5K ramp는 N 문서의 제안값이며 실제 s2 resolved 값이 확인되면 그 값을 계승한다. Offset은 알려진 상대 관계이지 절대 shift label이 아니다. [S-N §4]

`W(W(P,ε),c)`를 일부 run만 `W(P,ε+c)`로 최적화하지 않는다. 부화소 두 번 보간과 단일 보간은 같지 않다. Crop은 이미 만든 full `Pε`에서 수행하고 crop 후 z-score한다. R=1 reference margin=4이면 train aligner56², 복원 U-Net64²다. [S-N §§5–6]

### 5.6 Teacher view 기본값

첫 주력 KD는 **같은 raw 입력 view**다. Corrupted Student에는 같은 `Pε`를 받은 Teacher 출력을 target으로 준다. Native Teacher→corrupted Student는 `XVIEW`라는 P3 확장으로 별도 비교한다.

XVIEW에서 이동량까지 비교하면 `mu_S,ε + ε`와 `mu_T,0`를 비교해야 한다. 동일 corrupted view의 이동 KD는 `mu_S,ε−mu_T,ε`다.

---

## 6. Reconstruction hard/soft: 원안 그대로

### 6.1 정의

$$
e_T(p)=C^{-1}\sum_c|\hat Y_{T,c}(p)-Y_c(p)|,
$$
$$
e_S(p)=C^{-1}\sum_c|\hat Y_{S,c}(p)-Y_c(p)|,\quad
e_{ST}(p)=C^{-1}\sum_c|\hat Y_{S,c}(p)-\operatorname{sg}(\hat Y_{T,c}(p))|.
$$

$$
d_T=\frac{e_T}{e_T+\tau_R},\qquad
a_T=\frac{[e_S-e_T]_+}{e_S+\varepsilon_R}.
$$

$$
\boxed{L_R=\langle(1+\alpha_R d_T)e_S+\beta_R(1-d_T)a_Te_{ST}\rangle.}
$$

원안의 `lambda_R`를 이 문서에서는 다른 전체 계수와 혼동하지 않도록 `beta_R`로도 표기한다. **같은 reconstruction soft 계수**이며 참조 코드 인자는 `kd_weight`다. 초기안 `alpha_R=1`, `beta_R=0.1`, `epsilon_R=1e-6`은 약 [0,1] intensity 조건에서 사용한다. [S-KD §§3–4]

- GT는 모든 위치에서 남는다. Teacher가 틀리면 hard 증가·soft 감소.
- `e_S≤e_T`이면 soft=0. 둘 다 같은 오답이어도 GT gradient가 남는다.
- 둘 다 크게 틀리고 Teacher가 조금 낫다면 soft가 작은 양수일 수 있다.
- `d_T,a_T`, Teacher/GT target은 detach. 실제 `e_S,e_ST`는 live.
- 모든 픽셀/밴드 평균. `sum(soft_weight)` 또는 soft active pixel 수로 재정규화하지 않는다.

### 6.2 해석과 제한

`alpha_R≥0, 0≤beta_R<1`이면 hard 계수가 soft보다 크므로, GT와 다른 출력 성분에서 reconstruction의 local gradient 방향은 GT를 향한다. **통계·정합 loss까지 더한 전체 gradient의 성질은 아니다.** 밴드 평균 gate가 일부 밴드의 상대 열세를 가릴 수 있어 band별 진단도 남긴다.

Large Teacher error는 노이즈·표현 한계도 포함한다. 이를 반드시 복구 가능한 정보로 부르지 않는다. Overfitting 논지는 train fitting의 깊이와 validation/FR에서의 효용을 나누어 검증한다.

### 6.3 R-case registry

| ID | 참조 코드 mode | 수식 |
|---|---|---|
| **REC-N0** | `gt` | `<e_S>` |
| **REC-R0** | `fixed_kd` | `<e_S+beta_R e_ST>` |
| **REC-R1** | `hard_only` | `<(1+alpha_R d_T)e_S>` |
| **REC-R2** | `teacher_error` | `<(1+alpha_R d_T)e_S+beta_R(1-d_T)e_ST>` |
| **REC-R3** | `adaptive` | 승인한 전체식 |

먼저 N0/R1/R3, 이어 R0/R2. R3−R1은 output target의 추가 기여, R3−R2는 Student 우위 gate의 기여다. Teacher가 없는 N0 baseline은 Teacher forward를 불필요하게 실행하지 않아도 되지만 실제 wall-time 차이는 기록한다.

### 6.4 Calibration

$$\tau_R=\max\{\operatorname{median}_{train,p}e_T(p),\varepsilon_{scale}\}.$$

Native train으로 고정 Teacher의 통계를 만든다. I-N에서도 우선 같은 native τ를 사용하고 native/corrupt error histogram을 따로 기록한다. Corrupt에 맞춰 per-batch 재정규화하면 어려움의 기준 자체가 바뀐다.

Teacher checkpoint, intensity, upsampling, input-view 정책이 달라지면 재calibration한다. Student aligner만 fine-tune한다고 Teacher의 τ를 계속 바꿀 필요는 없다.

---

## 7. Variance를 세 종류로 분리한다

| 이름 | 대상 | 단위/의미 | Student에 작용하는 경로 |
|---|---|---|---|
| **출력 통계 variance/covariance** | `Y`, `Y_T`, `Y_S`의 공간·gradient·band 관계 | 영상의 국소 이차 통계 | Student 출력 → F_S, trainable이면 A_S도 |
| **정합 covariance Σ_T** | 동일 pair의 이동량 추정 | HR pixel², 방향별 신뢰도 근사 | 이동 KD → trainable A_S |
| **예측 uncertainty head** | 복원 오차를 설명하는 learned scale | 확률 모형에 따라 다름 | 별도 NLL/head가 있어야 함 |

GT 밝기 variance와 Teacher 이동 covariance를 같은 숫자로 맞추지 않는다. Aligner 입력 z-score의 분산도 위 두 실험 target과 다르다. 현재 mainline은 새 uncertainty head 없이 시작한다.

**GT 대 Teacher 통계만 비교한 scalar는 Student 학습 loss가 아니다.** 둘 다 고정이면 gradient가 없다. Student 통계를 비교하거나, GT–Teacher 차이를 Student의 가중치에 사용해야 한다.

---

## 8. 출력 통계 표현 S(Z): 정확한 구현 정의

공통으로 **최종 HRMS**에서 계산한다. Residual variance는 후속 대조다:

$$\operatorname{Var}(M+R)=\operatorname{Var}(M)+\operatorname{Var}(R)+2\operatorname{Cov}(M,R).$$

따라서 residual variance와 출력 variance를 동등하게 취급하지 않는다.

### 8.1 공간 window와 reduction

첫 설정은 **5×5, stride1, population variance(correction=0)**이다. 이는 이번 구현용 제안이며 최적값으로 확정된 것이 아니다. 후속 민감도는 3/5/7, 다중 창은 단일 창 효과 확인 뒤 수행한다.

$$\mathcal B_k(Z)(p)=k^{-2}\sum_{q\in\mathcal N_k(p)}Z(q),$$
$$\operatorname{Var}_k(Z)=\mathcal B_k(Z^2)-\mathcal B_k(Z)^2.$$

추가 정규화로 Teacher·Student·GT의 대비를 각각 1로 맞추지 않는다. 비교할 variance 차이 자체가 사라질 수 있다. 음수 roundoff만 `clamp_min(0)`하고 off-diagonal covariance의 음수는 보존한다. 안정성을 위해 영상별 공간 상수를 먼저 빼는 계산은 centered moment의 값을 바꾸지 않는다.

### 8.2 공통 유효 중심

Scharr radius1 + window radius `(k−1)/2`이므로 통계 중심의 margin:

$$m_V=1+(k-1)/2.$$

k5이면 margin3, train64²에서 **58×58**의 통계 중심을 사용한다. 모든 representation에 동일 중심을 적용하기 위해 image variance/spectral covariance도 먼저 고정 1px interior를 취한다. 이 통계 영역은 recon full-domain, aligner-only crop, metric ROI와 별개다.

출력 Scharr는 유효 convolution으로 계산한다. A2의 reflect-padding 후 1px 제외와 같은 내부 연산 결과인지 unit test로 확인한다. 통계 계산을 위해 GT나 출력 자체의 위치를 이동하지 않는다.

### 8.3 STAT-IV — 영상 분산

$$V_{Z,c}^{img}=\operatorname{Var}_k(Z_c).$$

표현 shape `[B,C,Hv,Wv]`. 국소 밝기/대비 변화량 대조군이다. 위치·물체 배치는 분산만으로 결정되지 않는다.

### 8.4 STAT-GV — 방향별 gradient variance: 첫 주력 표현

A2의 Scharr 계수 사용:

$$K_x=\frac1{32}\begin{bmatrix}-3&0&3\\-10&0&10\\-3&0&3\end{bmatrix},\quad K_y=K_x^T.$$

$$g_{Z,c,y}=K_y*Z_c,\quad g_{Z,c,x}=K_x*Z_c,$$
$$V_{Z,c,d}^{grad}=\operatorname{Var}_k(g_{Z,c,d}),\quad d\in\{y,x\}.$$

Band 평균으로 1ch를 만들지 않는다. 표현 shape `[B,2C,Hv,Wv]`; 참조 코드는 `[dy 전 밴드, dx 전 밴드]` 순서다. 분산은 부호 정보 일부를 잃으므로 signed Scharr direct-edge 대조를 둔다.

### 8.5 STAT-GC — gradient 공분산

$$g_{Z,c}=(g_{Z,c,y},g_{Z,c,x})^T,$$
$$C_{Z,c}^{grad}=\mathcal B_k(gg^T)-\mathcal B_k(g)\mathcal B_k(g)^T.$$

대각은 gradient variance, 비대각은 두 방향의 공동 변화다. **중심화 covariance**이며 구조 텐서 `E[ggᵀ]`와 다르다.

첫 구현은 각 band의 2×2 **모든 원소**를 `[cyy,cyx,cxy,cxx]`로 펼쳐 `[B,4C,Hv,Wv]`를 만든다. Loss는 원소 평균 L1이다. 대칭 off-diagonal이 두 번 등장하는 이 reduction을 명시한다. 이를 Frobenius norm이라고 부르지 않는다. GC와 GV의 계수는 동일 loss 숫자만 보고 정하지 말고 calibration한다.

### 8.6 STAT-SC — 분광 공분산

$$z_Z=(Z_1,\ldots,Z_C)^T,\qquad
C_Z^{spec}=\mathcal B_k(zz^T)-\mathcal B_k(z)\mathcal B_k(z)^T.$$

WV3는 8×8, `[B,64,Hv,Wv]` row-major representation. 공간창 안의 band 공동 변화를 비교한다. 단순 band별 variance 이상의 분광 정보가 있는지 보는 후속안이다.

전체 `B×C×C×H×W×k²` unfold를 만들지 않는다. 각 band row 또는 band pair를 나눠 local pooling하고 결과를 결합한다. PSD 여부는 수치 단위검사에서 보며 학습마다 전체 8×8 eigendecomposition을 할 필요는 없다.

### 8.7 범위 내 확장

| 확장 | 정의 | 우선순위 |
|---|---|---|
| 창 3/7·multi-window | 동일 표현을 다른 고정 창에서 비교 | P3 |
| Std / log-variance | `sqrt(v+ε)` 또는 `log(v+ε)`로 비교 | P3. 재calibration 필수 |
| Residual 통계 | `Y−M`, `Y_T−M`, `Y_S−M`에서 계산 | P3, 출력 통계와 별도 |
| 구조 텐서 이차 모멘트 | `E[ggᵀ]`를 비교, gradient mean 보존 | P3, covariance와 구분 |
| GV+SC 결합 | 두 표현을 각각 산출·가중합 | 각 단독 효과가 확인되면 P2/P3 |

모든 확장을 한 번에 켜지 않는다. Noise/ringing으로 variance를 맞추는지, mean bias와 edge 위치가 남는지 함께 확인한다.

---

## 9. 출력 통계의 hard/soft와 실험 case

표현을 `S_V(Z)`라 하고:

$$E_T^V(p)=\operatorname{mean}_j|S_V(\hat Y_T)_{pj}-S_V(Y)_{pj}|,$$
$$E_S^V(p)=\operatorname{mean}_j|S_V(\hat Y_S)_{pj}-S_V(Y)_{pj}|,$$
$$E_{ST}^V(p)=\operatorname{mean}_j|S_V(\hat Y_S)_{pj}-\operatorname{sg}(S_V(\hat Y_T)_{pj})|.$$

j는 band·방향 또는 행렬 원소다. 이 정의는 첨부 원안의 이동 covariance가 아니라 **최근 대화의 출력 통계 접근을 이번에 구체화한 것**이다.

### 9.1 통계 감독 모드

| Suffix | 통계 항 L_V | 의미 |
|---|---|---|
| **H** | `<E_S^V>` | GT 통계 감독 |
| **T** | `<E_ST^V>` | Teacher 통계만 증류. Pixel GT는 계속 유지 |
| **FIX** | `<E_S^V + beta_V E_ST^V>` | 고정 GT/Teacher 결합 |
| **WH** | `<(1+alpha_V d_T^V)E_S^V>` | Teacher의 통계 실패로 GT만 재가중 |
| **AD** | 아래 adaptive식 | Teacher 실패·Student 우위를 모두 반영 |

$$d_T^V=\frac{E_T^V}{E_T^V+\tau_V},\quad
a_T^V=\frac{[E_S^V-E_T^V]_+}{E_S^V+\varepsilon_V},$$
$$\boxed{L_V^{AD}=\langle(1+\alpha_Vd_T^V)E_S^V+\beta_V(1-d_T^V)a_T^V E_{ST}^V\rangle.}$$

`alpha_V=1, beta_V=0.1`을 첫 제안으로 둔다. `H/T/FIX/WH/AD`는 **통계 항만의 차이**이고 recon R3는 공통이다. Suffix T는 전체 학습에서 GT를 제거한다는 뜻이 아니다.

### 9.2 통계 calibration

`tau_V`는 표현·window·Teacher·intensity·ROI별 **별도 train 값**이다. `tau_R`를 재사용하지 않는다.

첫 규칙:

```text
e_scale = median(E_T^V on fixed native train calibration)
v_scale = median(abs(S_V(GT)) over strictly positive finite entries)
tau_V   = max(e_scale, 1e-3 * v_scale, 1e-12)
eps_V   = max(1e-3 * tau_V, 1e-12)
```

양의 통계가 전혀 없으면 해당 representation/cell의 target 신호가 없는 것으로 표시하고 통계 항을 비활성화한다. 위 하한은 [0,1] 단위용 신규 수치 규칙이며 정확한 센서 noise 모델이 아니다. 단위가 바뀌면 재검토한다. `tau_V`와 모든 gate는 detach다.

### 9.3 전체 가중치 λ_V

최종식은 `L_R + λ_V L_V`이다. Variance는 intensity² 단위이므로 A2 edge·offset의 0.01/0.1을 그대로 복사하지 않는다.

권장 pilot:

1. 고정 train calibration minibatch와 공통 Student pilot checkpoint에서 **optimizer step 없이** plain recon과 `STAT-*-H`의 출력 gradient RMS를 잰다.
2. 표시용 초기 목표비 `r_grad=0.05`로 `lambda_V = r_grad × median(g_rec/(g_V+eps))`를 계산한다.
3. 통계 gradient가 거의 0이면 거대한 λ를 적용하지 말고 `CALIBRATION_DEGENERATE`로 진단한다.
4. 같은 representation의 H/T/FIX/WH/AD에는 **같은 λ_V**를 사용한다. AD에만 유리하게 재조정하지 않는다.
5. 이후 필요하면 `×0.3 / ×1 / ×3`의 계수 대조를 별도 ID로 한다. Online auto-balancing은 기본에 넣지 않는다.

0.05는 검증된 최적값이 아닌 시작 규칙이다. λ와 실제 gradient ratio를 함께 기록한다. 서로 다른 representation은 같은 수치 λ를 강제하기보다 명시적인 동일 calibration 절차를 적용한다.

### 9.4 놓치지 않을 대조

- `GV-AD` vs `GV-WH`: Teacher 통계 target을 전달한 추가 효과.
- `GV-AD` vs `GV-FIX`: 통계 영역 adaptive 판단 효과.
- `GV-H` vs signed direct edge: variance 특유의 기여.
- `GV/GC/SC-H` vs `AD`: GT가 있으므로 Teacher 통계가 중복일 수 있음.
- `(V_S−V_T)−(V_G−V_T)`는 `V_S−V_G`와 같으므로 새 KD라 부르지 않는다.
- 통계 map을 shuffle한 부정 대조는 위치 정보의 기여를 확인한다.

---
## 10. 기존 앞단 감독: 무엇이 계속 학습되는가?

### 10.1 Recipe별 계승

- **A1:** 추가 hard geometry를 자동으로 넣지 않는다.
- **A2:** 같은 band의 signed Scharr edge를 비교한다. GV 대조에서 기존 edge가 이미 ON이면 동일 항을 두 번 더하지 않는다. `edge_inherited=true`로 표시하고 edge OFF 비교는 별도 family다.
- **A3:** **GT band 평균**과 PAN의 sign-invariant 구조 비교를 그대로 쓴다. 원 KD 개념 문서의 bandwise 구조 비교로 조용히 바꾸지 않는다.
- **N2/N3:** offset self-consistency를 계승한다. A3 geometry를 추가하려면 `AUX-N2+GEO`라는 별도 ablation이다.

### 10.2 A3 geometry의 재현 계약

[S-A §5.3]의 수식을 유지한다:

$$I_Y=C^{-1}\sum_cY_c.$$

원 PAN에서 구한 detached mean/std로 원 PAN과 warped PAN을 동일하게 z-score한다. GT 평균은 자체 통계로 z-score한다. warped PAN의 통계로 normalization을 다시 정의하지 않는다.

r=4 reference에서는 Gaussian sigma2, radius6, 13×13 separable kernel 후 Scharr:

$$g_P=\nabla G_r(Z_P(\widetilde P)),\qquad g_Y=\nabla G_r(Z(I_Y)).$$

Unwarped PAN/GT의 고정 gradient energy에서 `eta_P²,eta_Y²`를 구하고 detach:

$$Q(g;\eta^2)=\frac{(g_x^2,\sqrt2g_xg_y,g_y^2)}{g_x^2+g_y^2+\eta^2},$$
$$w_Y=\operatorname{sg}\left(\frac{\|g_Y\|^2}{\|g_Y\|^2+\eta_Y^2}\right),$$
$$L_{geo}=\operatorname{mean}_b\frac{\sum_{p\in V_g}w_Y(p)\|Q(g_P)-Q(g_Y)\|_2^2}{\sum_{p\in V_g}w_Y(p)+10^{-8}}.$$

고정 Vg의 reference margin11은 Gaussian·Scharr·warp support 및 안전 여유를 합친 기존 설계값이다. s2 실제 구현과 일치하는지 확인한다. 예측 correction으로 ROI를 줄여 어려운 영역을 제거하지 않는다. Support 실패는 조용히 sample drop하지 않고 failure로 기록한다. GT-weight가 수치적으로0인 경우 geometry만0, reconstruction은 유지한다.

이 representation은 PAN/MS의 공통 구조를 근사할 뿐 실제 spectral response를 정확히 일치시키지 않는다. `L_geo↓`를 절대 정합 성공으로 바로 해석하지 않는다.

### 10.3 Frozen일 때 제거할 항

A-FR에서는 다음 항이 Student에 학습 gradient를 주지 않는다:

$$\partial L_{geo}(\widetilde P^*,Y)/\partial\theta_S=0,\quad
\partial L_{off}(\mu^*_{\epsilon},\mu^*_0)/\partial\theta_S=0.$$

이동량 KD도 같은 입력/aligner에서는0이다. 따라서 **A-FR optimizer loss에서는 geo/off/mean-KD를 제외**하고 진단 로그만 남긴다. A2 output edge와 출력 variance는 F_S에 연결되므로 선택한 구성에 따라 계속 학습 가능하다.

---

## 11. 원래 이동량 covariance KD: trainable aligner에서 별도로 검증

### 11.1 전달 대상

$$d_\mu=\mu_S-\operatorname{sg}(\mu_T),\quad
L_G=\tfrac12\operatorname{mean}_b d_\mu^T\Pi_Td_\mu,\quad \Pi_T=\operatorname{sg}(\Sigma_T^{-1}).$$

원안의 의미는 **Teacher가 확실한 이동 방향을 더 강하게 전달**하는 것이다. 출력 gradient variance와 무관한 HR pixel²의 불확실성이다. A-FT 또는 A-SC에서만 학습 항으로 유효하다. [S-KD §5.2]

초기 계승에서 `mu_S=mu_T`여서 KD가0인 것은 정상이다. `mu_S`를 detach해 영구적으로 gradient가 끊기지 않았는지 확인한다.

### 11.2 Geometry 기반 covariance: G-GEO

GT 기반 구조 residual을 `r_raw(p,δ)`라 두고, 기존 A3 cost와 동등하게 weighted/normalized vector를 구성한다:

$$\mathbf r(\delta)=\operatorname{vec}\left(\sqrt{w_Y/(\sum w_Y+\epsilon)}\ r_{raw}(\delta)\right).$$

$$J_T=\left.\partial\mathbf r(\delta)/\partial\delta\right|_{\mu_T},\quad H_T=J_T^TJ_T.$$

원안 후보:

$$\Sigma_T^{geo}\approx s_T^2(H_T+\gamma I)^{-1}+\sigma_{min}^2I.$$

`r`의 normalization, residual scale `s_T²`, damping/floor를 사전에 고정한다. 예를 들어 `s_T²`를 weighted mean squared raw residual의 component 평균으로 정의할 수 있으나, **이는 calibration된 물리 likelihood가 아니라 본 descriptor 아래의 근사 규칙**이다. Loss의 전체 배율·pixel 수가 covariance를 임의로 바꾸지 않는지 size별 검사한다.

구현은 2개의 displacement 방향에 대한 Jacobian만 필요하다. 전체 모델 Jacobian을 만들지 않는다. Frozen Teacher로 얻은 `mu_T.detach()`를 별도 leaf로 복사하고 descriptor만 미분한다. Native graph와 Student graph를 보관한 채 J를 중복 계산하지 않는다.

초기 수치 확인에는 중심 유한차분 `h=0.05 HR px`를 신규 pilot 값으로 두고 h/2·2h와 비교할 수 있다. 이는 미분 간격이지 training jitter radius가 아니다. Autograd와 일치하지 않으면 G-GEO training을 시작하지 않는다. 모든 ±h에서 동일 고정 유효 support를 확인한다.

### 11.3 Rank와 precision 안전 규칙

1. H가 평탄하거나 rank-deficient하면 해당 방향의 precision을0으로 한다. Damping으로 생긴 정보량을 관측 확신으로 취급하지 않는다.
2. Relative/absolute eigenvalue 판정값은 train calibration manifest에 고정한다. 검증하지 않은 arbitrary floor로 모든 sample을 full-rank로 만들지 않는다.
3. 확률 KL에는 SPD가 필요하므로, rank 부족 sample은 mean-KD에서 유효 방향만 사용하거나 distribution-KD에서는 제외하고 그 비율을 기록한다.
4. `s_T²`가0에 가까워져 precision이 발산하지 않도록 하한·상한을 train pilot에서 정한다. 경계/수치 invalid는 GT loss에 영향을 주지 않는다.
5. Valid pair만으로 재평균해 dropout 비율이 달라도 같은 KD 크기를 강제하지 않는다. 기본은 batch 전체 평균에 invalid weight0, 유효 비율 별도 기록이다.
6. `inverse()` 결과를 그대로 곱하기보다 2×2 eigensystem/Cholesky solve를 사용한다. FP32 이상, 진단은 FP64와 교차검사한다.

잘못된 correspondence에서도 구조 cost가 날카로울 수 있다. Mean-KD가 유효하지 않은 Teacher에 covariance만 붙여 강하게 믿지 않는다. [S-KD §5.3; S-A §5.3(e)]

### 11.4 N계열 consistency 기반 통계: G-EQ

동일 pair에 K개의 알려진 probe ε를 적용한다. 초기 K=16은 신규 진단 제안이며 비용·안정성에 따라 version을 바꿀 수 있다. U-Net은 필요 없다.

$$r_T^{(k)}=\mu_{T,\epsilon_k}+\epsilon_k-\mu_{T,0},$$
$$b_T=K^{-1}\sum_kr_T^{(k)},\quad
C_T^{eq}=K^{-1}\sum_k(r_T^{(k)}-b_T)(r_T^{(k)}-b_T)^T,$$
$$Q_T^{eq}=C_T^{eq}+b_Tb_T^T=K^{-1}\sum_kr_T^{(k)}(r_T^{(k)})^T.$$

**`Cov(mu_T,ε)` 자체를 쓰지 않는다.** 완벽한 response도 ε의 분산을 갖고, 반응하지 않는 상수 aligner는 오히려 variance가0이 될 수 있다.

G-EQ는 추가 변위에 대한 **closure error의 방향별 이차 모멘트**다. 절대 정합 posterior covariance가 아니다. `mu_0=b, mu_ε=b−ε`는 b가 틀려도 closure가0이다. 따라서 G-GEO와 같은 이름/column으로 합치지 않고, 유효한 native Teacher 판단에 대해 신뢰도 weighting 후보로만 시험한다. [S-N §4.7]

Probe는 train calibration에서 정의하고 외부 FR test로 radius를 맞추지 않는다. K개 probe에 같은 interpolation/crop 단서가 존재하므로 MS 교체·kernel 변경·padding 진단을 함께 한다.

### 11.5 Geometry KD case registry

`Pi_T=Sigma_T⁻¹`, `k0=median_train(tr(Pi_T)/2)`를 고정한다.

| ID | Loss | 질문 |
|---|---|---|
| **G0** | 없음 | Aligner supervised/offset만으로 충분한가? |
| **G1** | `0.5 k0 ||d_mu||²` | Teacher 이동 중심 자체의 효과 |
| **G2** | `0.5 d_muᵀ[(tr Pi_T/2) I]d_mu` | sample별 scalar 신뢰도 |
| **G3** | `0.5 d_muᵀ Pi_T d_mu` | 방향별 full precision |
| **G4** | Pi_T 대각성분만 사용 | 비대각 방향 상관의 효과 |
| **G5** | Gaussian distribution-KD | Student covariance 자체의 학습 |
| **G-EQ** | closure 통계에서 만든 안전 precision | 상대 반응 신뢰도의 효과 |

G1/G2/G3는 같은 train calibration·유효 방향 규칙·lambda_GKD를 사용한다. G2는 G3와 trace가 같아서 방향성 비교에 적합하다. G4는 Sigma의 off-diagonal 제거가 아니라 **precision의 diagonal 유지**로 명세한다. 둘은 같은 연산이 아니다.

### 11.6 Student covariance head: G5

$$q_i=\mathcal N(\mu_i,\Sigma_i),\qquad
L_{dist}=D_{KL}(q_S\|\operatorname{sg}(q_T)),$$
$$L_{dist}=\tfrac12[d_\mu^T\Sigma_T^{-1}d_\mu+\operatorname{tr}(\Sigma_T^{-1}\Sigma_S)-2+\log(\det\Sigma_T/\det\Sigma_S)].$$

이 KL 방향에서 mean 오차를 Teacher precision으로 가중한다. 반대 방향을 같은 수식처럼 구현하지 않는다.

`Sigma=L Lᵀ+eps I`, L의 대각은 양수로 parameterize한다. 2D covariance에는 독립 값3개가 필요하다. 기존 delta head2개와 합쳐 U-Net 입력이11ch가 되는 것이 아니다. 입력9ch는 그대로다.

Teacher/Same-structure 조건을 유지하려면 두 aligner에 동일 head를 붙인다. Teacher head만 별도 고정 covariance target으로 calibration 학습하고 freeze하는 절차를 기록한다. 서로의 variance만 같게 한다고 유의미한 uncertainty가 생기지 않는다. G5는 head 학습이 복원에 기여하는지 G3와 비교한 뒤 채택한다.

### 11.7 GT 구조 민감도: G-STRUCT, 후속

$$J_Y=\operatorname{mean}_{p,c}\nabla Y_c(p)\nabla Y_c(p)^T,\qquad L_{struct\text{-}mu}=d_\mu^TJ_Yd_\mu.$$

J_Y는 방향별 영상 변화의 중요도이고 Sigma_T는 Teacher 대응 신뢰도다. 작은 이동 오차 m,Σ에 대해 `mᵀJ_Ym+tr(J_YΣ)`가 영상 MSE의 국소 근사로 등장할 수 있지만, 실제 U-Net 오류와 동일하지 않다. 이 축은 mean-KD의 중요도 weighting 대조이며 J_Y를 Sigma의 정답으로 맞추지 않는다.

---

## 12. 통합 loss와 gradient routing

선택한 aligner 정책 m과 recipe j에서:

$$
\boxed{L_S=L_R+\lambda_VL_V+L_{aux,j}+\lambda_{GKD}L_G+\lambda_{reg}\mathcal R(\mu_S).}
$$

항은 독립 flag로 켠다. 모든 현재 방법을 시험한다는 것은 모든 항을 동시에 켜라는 뜻이 아니다.

| 항 | F_S | A_S trainable | A_S frozen | Teacher |
|---|---|---|---|---|
| Reconstruction hard/soft | gradient | sampler를 통해 gradient | 없음 | target detach |
| 출력 variance/covariance | gradient | sampler를 통해 gradient | 없음 | target/gate detach |
| A2 output edge | gradient | gradient | 없음 | 없음 |
| A3 PAN–GT geometry | **없음** | 직접 gradient | 진단만 | 없음 |
| N2/N3 offset relation | **없음** | 직접 gradient | 진단만 | 없음 |
| Mean/covariance 이동 KD | **없음** | 직접 gradient | 동일 input이면0 | mean/precision detach |
| Teacher–GT 통계 차이만 | 없음 | 없음 | 없음 | 고정 통계 |

A-FR에서 `L_G`를 넣고 “covariance KD를 실행했다”고 표기하지 않는다. A-FT/A-SC에서 새로 geometry/off를 켜는 것은 frozen과 비교했을 때 **학습 가능한 자유도+해당 감독의 결합 변화**다. Pure freeze 비교가 필요하면 공통 recon/stat만 사용하는 추가 cell을 둔다.

N계열 초기 mean/covariance KD는 **native step에만 적용**하는 단순안으로 시작한다. 알려진 ε의 상대 반응은 Student offset loss가 맡는다. Corrupted correction까지 KD하는 것은 `G-CORR`/`G-XVIEW`로 별도 확장한다. 유효 적용빈도는 loss/report에 기록하고 native 적용 계수를 임의로2배 보상하지 않는다.

---

## 13. 실험 registry: 모든 현재 접근을 우선순위로 배치

### 13.1 이름 규칙

```text
S2W112_<recipe>_<input>_<aligner>_<rec>_<stat>_<geomKD>_s<seed>_<version>
예: S2W112_N2_IN_AFR_R3_GVAD_G0_s1234_v01
```

W112→W112면 teacher manifest에 T112를 기록한다. T96 bridge는 ID에도 `T96BRIDGE`를 넣는다. 현재 실행 recipe가 안 정해진 상태에서 N2를 확정값으로 기록하지 않는다.

### 13.2 우선 시작 queue

같은 `F_S init`, Teacher, input protocol, alpha/beta/tau 및 전체 training schedule을 공유한다.

| Queue | 구성 | 목적 | 우선순위 |
|---:|---|---|---|
| Q00 | W112 A-ID + REC-N0, no KD | 새 width의 독립 baseline | P0, 기존 적격 run 있으면 재사용 |
| Q01 | W112 Teacher 확보/검증 | 같은 구조 output target 및 donor | P0 |
| Q02 | A-FR + REC-N0 + STAT-OFF | Frozen aligner의 GT-only 기준 | P1 |
| Q03 | A-FR + REC-R1 + STAT-OFF | Teacher-error hard fitting | P1 |
| Q04 | A-FR + REC-R3 + STAT-OFF | Adaptive reconstruction 기준 | P1 |
| Q05 | A-FR + REC-R3 + GV-H | GT 구조 통계 추가 | P1 |
| Q06 | A-FR + REC-R3 + GV-AD | GT/Teacher adaptive 통계 | P1 |
| Q07 | A-FT + REC-R3 + STAT-* + G0 | Teacher aligner fine-tuning | P1, *는 Q04~06 중 고정 선택 |
| Q08 | A-SC + REC-R3 + 같은 STAT-* + G0 | Teacher 초기값의 제약 확인 | P1 |
| Q09 | A-FR/선택 정책 + REC-R0 + OFF | 고정 KD 대조 | P2 |
| Q10 | 같은 정책 + REC-R2 + OFF | Student 우위 gate 대조 | P2 |
| Q11 | 같은 정책 + R3 + GV-T | Teacher 통계만 | P2 |
| Q12 | 같은 정책 + R3 + GV-FIX | 고정 통계 결합 | P2 |
| Q13 | 같은 정책 + R3 + GV-WH | 통계 hard 재가중만 | P2 |
| Q14 | 같은 정책 + R3 + IV-H / IV-AD | 영상 분산 표현 | P2, 두 cell |
| Q15 | 같은 정책 + R3 + signed EDGE | 단순 edge 대조. inherited edge 중복 금지 | P2 |
| Q16 | 같은 정책 + R3 + GC-H / GC-AD | 방향별 gradient covariance | P2 |
| Q17 | 같은 정책 + R3 + SC-H / SC-AD | 분광 covariance | P2 |
| Q18 | 유력 GV/GC + SC 결합 | 상보성 검사 | P2, 단독 cell 뒤 |
| Q19 | A-FT 또는 A-SC + R3 + STAT-OFF + G1/G2/G3 | 원래 이동 covariance 접근 | P2, 세 대응 cell |
| Q20 | 같은 정책 + 선택 STAT-* + G3 | 출력 통계와 이동 KD 결합 | P2/P3 |
| Q21 | Q19의 G4/G-EQ/G-STRUCT | 방향·신뢰도 정의 비교 | P3 |
| Q22 | Q19의 G5 | Student covariance head | P3 |
| Q23 | Window/Std/log/residual/negative controls | 표현의 기전·민감도 | P3 |
| Q24 | 주요 대조·winner의 Student seed 반복 | 안정성 | P3 |
| Q25 | 장기 fitting/Teacher-copy/full scratch/no-KD | 학습비용·초기화·fallback 검증 | P3 |

**Q19를 실행할 수 없다는 이유로 Q05/Q06의 출력 variance를 같이 보류하지 않는다.** 이동 covariance 산출 gate 실패는 별도의 실패다. A-FR에서 통계가 유효하면 그 경로만으로도 연구를 계속할 수 있다.

Q07/Q08이 안 좋으면 R3 및 통계 loss를 없애고 같은 A-FT/A-SC에 plain GT를 붙인 대조를 추가해, 초기화 문제인지 복합 감독 문제인지 분리한다. 모든 새 접근이 나쁘면 W112 A-ID/독립 scratch GT baseline으로 돌아가 원인별 한 항씩 다시 켠다.

### 13.3 모든 조합을 곱하지 않는 규칙

- Frozen에서 값이 좋던 통계1개와 OFF를 A-FT/A-SC에 우선 교차한다.
- Geometry KD는 G0/G1/G2/G3를 동일한 trainable policy에서 비교한다. Policy마다 전수 반복은 유력 결과 후 수행한다.
- Dataset·width·normalization·Teacher를 동시에 바꾸지 않는다.
- Failed gate의 cell은 수치 결과를 만들어 채우지 않고 원인을 기록한 뒤 구현을 고쳐 재시도한다.
- 성능이 나쁜 접근도 최소 수학/unit·짧은 대응 screen까지는 기록하고, 무의미한 full-run 반복만 줄인다. 유망한 장기 fitting 가설은 짧은 prefix만으로 기각하지 않는다.

---

## 14. 약 24시간 운영안과 이후 계속 진행하는 기준

### 14.1 24시간은 checkpoint이지 종료 조건이 아니다

아래는 **예약 비중의 예시**이며 s2에서 실측한 실행시간이 아니다. GPU 시간, 사람의 구현시간, 서버 대기시간을 별도 기록한다. 단일 GPU 연속 실행이면 active wall-clock과 GPUh가 대략 비슷할 수 있지만 동일한 계측값은 아니다.

| 첫 review까지의 순서 | 대략적 시간 배치 | 내용 |
|---|---:|---|
| 환경·코드·unit·smoke | 0–2h | P0, W112 shape·sampler·Teacher 출처 |
| W112 baseline/Teacher 준비 | 2–7h | 기존 적격 checkpoint면 대폭 축소; 없으면 24h 이후까지 연장 가능 |
| Calibration·baseline KD | 7–10h | τ_R/τ_V, 속도·메모리, Q02/Q03/Q04 prefix |
| 첫 통계 대응 비교 | 10–17h | Q05/Q06, 필요하면 GV-WH/FIX를 앞당김 |
| Aligner 정책 비교 | 17–22h | Q07/Q08, 선택된 loss와 OFF 대응 |
| 중간 export·판정·다음 queue 갱신 | 22–24h | 결과·미결사항·예상 잔여시간 보고 |

Teacher 준비에 하루가 필요하면 준비를 생략하거나 미학습 Teacher로 KD를 시작하지 않는다. **24h 시점에 준비 상태와 누적 비용을 보고하고 계속 진행**한다. 모든 Q00–25가 24h에 끝난다는 주장이 아니다.

Teacher가 이미 준비되어 있으면 확보된 시간을 P2의 통계 대조·trainable covariance 준비로 이동한다. 큐의 순서는 정보를 얻는 효율에 따라 바꿀 수 있다.

### 14.2 Run당 learning budget

초기 후보 full budget은 최근 문서의 **50,000 optimizer updates**다. 실제 s2 W112 공통 기준이 다르면 최초 run 전 `N_FULL`로 확정한다. 24시간에 맞추려고 일부 case만 update를 줄이지 않는다.

추천 checkpoint:

```text
health: 200~1,000 update — numerical/gradient/throughput, 성능 결론 아님
review: 5,000 update — early fitting/gradient 확인
screen: 0.25 N_FULL (=12,500 if 50K)
second: 0.50 N_FULL (=25,000)
full  : N_FULL (=50,000)
extend: 1.5N / 2N — 동일 비용 continuation 대조 동반
```

Screen은 **처음부터 full-N scheduler를 사용한 prefix**다. 12.5K cosine을 끝낸 모델과 50K schedule의12.5K 지점을 섞지 않는다. 작은 이득/느린 수렴은 최소25K 대응 비교 후 해석한다.

### 14.3 Throughput 추정

실제 train module·microbatch·AMP로 warmup 후 최소200 update를 측정한다.

$$T_{run}\approx N_n t_n+N_c t_c+T_{eval}+T_{export}+T_{calib,amortized}.$$

- A-FR, A-FT, A-SC 각각 측정.
- R3/GV/SC/G-GEO 각각 비용 증가를 측정.
- Teacher no-grad 비용, covariance probe 비용, CPU dataload 대기를 별도 기록.
- 남은 run 예상시간은 새 측정에 따라 갱신한다. 초기24h 예약표를 실측 예측처럼 유지하지 않는다.

### 14.4 리뷰 순서 변경 예시

| 관찰 | 즉시 할 수 있는 다음 조정 |
|---|---|
| GV-H와 AD 모두 잘됨 | WH/FIX를 앞당겨 KD 고유 효과 확인 |
| GV 통계는 좋아지는데 RR/경계가 나빠짐 | signed edge·λ 감소·window 축소를 동일 parent 분기로 검사 |
| Aligner frozen에서 품질이 일찍 막힘 | FT/SC와 noalign 대조를 앞당김 |
| FT만 불안정 | aligner LR, offset/geometry gradient, support를 점검한 뒤 한 변수씩 조정 |
| G-GEO rank/calibration 실패 | covariance 구현 수리, G0/G1 및 출력 variance는 계속 |
| Teacher가 Student보다 대부분 나쁨 | output KD 자동 약화 로그 확인; hard-only, 새 Teacher 또는 no-KD 대조 |
| Teacher 교체 필요 | 새 T ID·calibration으로 cohort 분기. 기존 타깃 덮어쓰기 금지 |

---

## 15. 중간 변경을 허용하면서 비교를 남기는 방법

### 15.1 두 run 종류

**CONTROLLED:** 같은 초기값·정해진 schedule·고정 목적함수로 시작한 대응 실험. 최종 ablation의 근거.

**ADAPTIVE-PATH:** 중간에 loss·aligner 학습 상태·Teacher·입력 프로토콜을 바꾼 탐색 경로. 유력 설계를 찾는 데 사용하되 해당 checkpoint가 고정 구성으로 처음부터 학습된 것처럼 보고하지 않는다.

### 15.2 변경 절차

1. 현재 parent checkpoint에 model·optimizer·scheduler·GradScaler·RNG·data sampler·update index를 저장한다.
2. `phase_change.yaml`에 이전/이후 설정 diff, 이유, 근거 metrics, inherited step/cost를 적는다.
3. 동일 parent에서 기존 설정을 계속하는 **continuation control**과 새 설정의 child를 만든다. 시간이 부족하면 child를 먼저 보되 control을 `REQUIRED_PAIR`로 queue에 남긴다.
4. 바꾼 항 외의 loss, augmentation 순서, init/state는 유지한다.
5. Child가 유력하면 최종적으로 그 구성을 from-init 또는 동일한 사전선언 warm protocol로 재확인한다.

```yaml
phase_change:
  path_id: S2W112_PATH03
  parent_checkpoint: REQUIRED
  parent_step: 12500
  old_phase: AFR_R3_GVOFF
  new_phase: AFT_R3_GVAD
  change_axes: [aligner_trainability, statistic_loss]  # 여러 축이면 탐색 경로라고 명시
  optimizer_state_policy: preserve_existing_add_aligner_fresh
  scheduler_policy: inherit_full_N_global_step
  teacher_id_before: T112_v01
  teacher_id_after: T112_v01
  reason: REQUIRED_OBSERVATION
  matched_continuation: REQUIRED_CONTROL_ID
  formal_one_factor_ablation: false
```

두 항을 동시에 바꿔 유망 경로를 찾는 것은 가능하다. 단, 그 차이를 어느 하나의 효과라고 해석하지 않는다.

### 15.3 Optimizer 상태 규칙

- Frozen→trainable: Teacher가 아니라 Student aligner 복사본을 unfreeze. Optimizer에 새 parameter group을 추가한다. Momentum/Adam moments는 새 group에서만 새로 시작.
- 학습 LR은 현재 global scheduler 위치에 맞춘다. Unfreeze 시 최고 LR로 재시작하려면 그 restart 자체가 변경축이다.
- Scratch로 전환: Student aligner weight와 그 optimizer state만 초기화. 복원망은 이어갈 수 있지만 `PATH-ALIGNER-RESET`이라고 기록한다.
- F_S까지 scratch면 전체 Student state를 새로 만들며 새로운 run이다.
- Freeze할 때 파라미터를 optimizer에서 제거하거나 명확한 disable 정책을 둔다. 오래된 gradient로 한 번 더 update하지 않는다.
- Teacher를 바꾸면 τ·통계 target·covariance cache key도 바꾼다. 평균 gate의 불연속 변화를 로그로 남긴다.

### 15.4 최종 비교표에서 반드시 보일 항목

`from_init / warm_start / adaptive_path`, parent ID, parent step, 신규 step, **누적 training cost**, Teacher cost, total sample exposure를 모두 기록한다. 25K parent+25K child를 “새 방법25K”로만 표기하지 않는다.

---

## 16. 구현 모듈·인터페이스·training step

### 16.1 기존 코드와 새 구현 범위

현재 제공된 `pan_gt_anchored_kd.py`는 reconstruction criterion만 구현한다. **W112 U-Net·PAN aligner·covariance·trainer·server launcher는 아니다.** [S-KD §16]

이 문서와 함께 제공하는 `pan_s2_w112_stat_kd_reference.py`는 다음만 구현한다:

- image variance, gradient variance, gradient covariance, spectral covariance의 valid local map.
- 기존 reconstruction criterion을 통계 feature의 hard/soft에 재사용하는 wrapper.
- 수학/shape/gradient CPU unit test.

이 두 파일만으로 s2 full training이 실행되는 것은 아니다. 실제 repository adapter, data loader, optimizer, evaluator, config resolver, checkpoint manager는 실행자가 연결한다.

### 16.2 권장 코드 분리 — 경로는 제안

```text
experiments/s2_w112_kdv/
  model_adapter.py          # 실제 W112 residual U-Net과 aligner 연결
  teacher_assets.py         # donor/KD checkpoint 검증, strict load, freeze
  input_protocol.py         # I-A / I-N / 같은 raw view 생성
  losses_rec.py             # 제공 reconstruction criterion adapter
  losses_statistics.py      # local moment extractor + H/T/FIX/WH/AD
  losses_geometry.py        # A3 원식·N offset loss adapter
  covariance_geometry.py    # J/H/rank/precision, 아직 구현 대상
  covariance_equivariance.py# closure 통계, 아직 구현 대상
  calibration.py            # tau/λ/scale, train-only manifests
  evaluation_adapter.py     # raw/valid/aligned, 기존 evaluator 유지
  phase_manager.py          # parent/child, optimizer/RNG/manifest
  run_registry.py           # case 이름·허용 조합 검사
  tests/                    # s2 integration gates
```

이 경로가 기존 repository에 이미 존재한다고 가정하지 않는다. 실제 경로·함수명과 mapping을 `implementation_map.md`에 기록한다.

### 16.3 ForwardResult 계약

```python
# 실제 모델 API에 맞춰 작성할 adapter 계약
ForwardResult = {
    'pred': Tensor[B, 8, H, W],       # base가 이미 더해진 최종 HRMS
    'residual': Tensor[B, 8, H, W],
    'delta_yx': Tensor[B, 2],         # 현재 PAN HR px
    'pan_aligned': Tensor[B, 1, H, W],
    'support': SupportInfo,
    'native_reference_delta': Optional[Tensor[B, 2]],
}
```

9ch concat은 U-Net 직전에만 수행한다. `μ`, variance map, GT, LPAN/HPAN을 hidden condition channel로 추가하지 않는다.

### 16.4 Aligner 상태별 forward 의사코드

```python
# Interface sketch; 실제 wrapper/AMP/accumulation/eval API를 대신하지 않는다.
M = existing_ms_upsampler(lrms)                  # 기존 phase/단위
P_view, epsilon, corrupted = prepare_raw_view(P, protocol, update_index)

with torch.no_grad():
    T = teacher_forward(P_view, M)             # A_T + F_T 고정

if aligner_policy == 'A-FR':
    # exact checkpoint/view/normalization/warp가 같을 때만 공유
    delta_s = T['delta_yx'].detach()
    pan_s = T['pan_aligned'].detach()
else:
    delta_s = student_predict_delta(P_view, M) # live graph
    pan_s = warp_pan(P_view, delta_s)          # live sampler; delta=0도 실행

residual_s = student_unet(cat(pan_s, M))       # U-Net만 AMP 가능
pred_s = M.float() + residual_s.float()       # 이미 wrapper가 더했으면 다시 더하지 않음
rec = rec_criterion(pred_s, T['pred'], gt)
stat = output_statistics_loss(pred_s, T['pred'], gt, stat_cfg)
aux = inherited_auxiliary_loss(...)           # 정책과 recipe의 허용 항만
geom_kd = alignment_kd(...)                   # A-FR이면 계산/역전파하지 않음
loss = rec.loss + lambda_v * stat + aux + lambda_gkd * geom_kd
backward_and_optimizer_update(loss)
```

I-N에서 `P_view`는 이미 `W(P,epsilon)`이다. `teacher_forward`와 `warp_pan(P_view,delta_s)`는 learned 두 번째 warp를 수행한다. N2/N3의 native reference aligner 호출은 별도로 있고 F_S를 한 번 더 실행하지 않는다.

### 16.5 기존 criterion의 실제 호출

```python
from pan_gt_anchored_kd import GTAnchoredReconstructionKD
from pan_s2_w112_stat_kd_reference import statistic_loss

rec_criterion = GTAnchoredReconstructionKD(
    tau=tau_R, alpha=1.0, kd_weight=0.1, eps=1e-6, mode='adaptive'
).to(device)
stat_criterion = GTAnchoredReconstructionKD(
    tau=tau_V, alpha=1.0, kd_weight=0.1, eps=eps_V, mode='adaptive'
).to(device)

rec_result = rec_criterion(pred_s, pred_t, gt, return_maps=False)
stat_result = statistic_loss(
    pred_s, pred_t, gt, kind='grad_var', window=5, criterion=stat_criterion
)
loss = rec_result.loss + lambda_V * stat_result.loss
```

`stat_result`도 원소 평균 L1 기반이다. **Frobenius/Gaussian KL 구현이 아니다.** T-only 통계 mode는 `stat_map(S)−stat_map(T.detach())`의 absolute mean을 별도로 계산한다. 제공 criterion의 `fixed_kd`는 hard를 포함하므로 T-only라고 쓰지 않는다.

첫 reference는 항상 maps를 반환한다. 실제 장기 trainer에서는 logging 주기에만 maps를 복사/저장하도록 adapter를 추가해 I/O와 메모리를 제한한다. 수치 확인을 위한 `isfinite().all()` 동기화도 step마다 큰 tensor마다 반복하지 않도록 조정하되 NaN/Inf fail-fast는 유지한다.

### 16.6 Loss별 autograd 검사

- `teacher_pred.grad`, Teacher parameters grad: 없어야 한다.
- `gt`와 d/a gate: gradient 없어야 한다.
- F_S: recon/stat가 켜졌을 때 유효 gradient.
- A-FR: parameters checksum이 모든 update에서 불변.
- A-FT/A-SC: nonzero input에서 sampling→delta→aligner 경로 연결.
- A3 geo/off/GKD만 backward할 때 F_S gradient가 없다는 것을 확인.
- Zero-initialized final delta head의 첫 step 앞단 stem gradient가0일 수 있음. 두 번째 이후까지 검사.
- 최종 residual head도 zero initialization이면 첫 step PAN 민감도가 작을 수 있음. 변위 학습 실패를 첫 step 하나로 판단하지 않는다.

---
## 17. 학습 설정·메모리·속도·수치 안정성

### 17.1 s2 실제 설정을 먼저 읽는다

```text
hostname, GPU model/UUID/VRAM, driver, CUDA runtime, cuDNN
Python, torch, torchvision, 설치 패키지 snapshot
repo root / git commit / dirty diff / branch
train/val/test manifest와 SHA256
existing W112 또는 최신 W96 trainer의 optimizer·scheduler·batch·AMP
s2에서 실행 중인 다른 job과 메모리 점유
```

본 문서에서 s2 파일 경로·GPU 종류를 임의로 채우지 않는다. 같은 s2에서 주력 case를 순차 실행한다. 여러 GPU가 있어도 초기 대응 block의 GPU UUID를 맞추고, 병렬화는 환경 차이와 wall-time 정의를 기록한 뒤 한다.

### 17.2 Training profile

우선 `PROFILE-INHERIT`로 최신 실제 s2 설정을 가져온다. 값이 없을 때만 아래를 **새 캠페인 시작 제안**으로 채택하고 전체 cohort에 동일 적용한다:

| 항목 | 미확인 시 신규 제안값 | 주의 |
|---|---|---|
| Optimizer | AdamW | 기존 s2의 확인된 설정이라는 뜻은 아님 |
| Backbone LR | `1e-4` | W112 stability pilot에서 확인 |
| Weight decay | `0.01` | no-decay parameter group 관례가 있으면 기록 |
| Betas / eps | `(0.9,0.999)` / `1e-8` | source config 확인 우선 |
| Scheduler | linear warmup100 → cosine, full50K | optimizer update 기준 |
| Nominal effective batch | 48 | dual-MARs96 아님 |
| A-FT 초기 LR | backbone LR와 동일 | frozen-vs-trainable 비교 후 `×0.1` 별도 실험 |
| A-SC LR | backbone LR와 동일 | 초기화만 비교하는 FT/SC 주력안에서 LR 차이 만들지 않음 |
| Seed | 우선1234, 후속2025·777 | 기존 s2 대응 초기값이 있으면 manifest에 맞춤 |

Gradient clipping·AMP·optimizer state dtype·cosine minimum은 실제 config에서 확인한다. fallback을 채택할 때 null 값이 남으면 resolver가 실행 전에 해결하도록 한다. **문서 proposal과 실제 resolved config를 별도로 보관**한다.

### 17.3 FP32 영역과 AMP 영역

- Aligner forward, grid 생성, PAN sampling, geometry/offset, local moment 계산, covariance solve, gate 계산: **autocast disabled, FP32**.
- Teacher/Student 복원 U-Net: 검증된 기존 AMP 정책을 적용 가능.
- Teacher target과 Student output loss는 FP32. Teacher를 no-grad로 실행하되 Student U-Net은 반드시 graph 유지.
- Training 중 HRMS/PAN을 [0,1]로 clamp해 overshoot gradient를 없애지 않는다. Metric export의 clipping은 기존 evaluator 규칙 유지.
- AMP overflow로 optimizer step이 skip되면 scheduler/`optimizer_update_index`도 실제 update에 맞춘다. Attempt count는 별도 로그다.

API 세부는 s2 설치 version에서 확인한다. 문서 작성 환경의 PyTorch CPU 검사로 CUDA 재현성·AMP 동작까지 검증했다고 주장하지 않는다.

### 17.4 OOM 대응 순서

1. 실제 batch에서 Teacher no-grad와 Student backward의 peak allocated/reserved를 각각 기록한다.
2. Teacher 중간 feature가 필요 없으면 저장하지 않는다. Output과 correction 및 필요한 통계만 보관한다.
3. Aligner frozen이면 같은 raw view의 correction/PAN을 공유하되 Teacher forward 완전히 고정.
4. Microbatch를 줄이고 accumulation으로 **유효 batch와 sample 노출량을 보존**한다. 서로 다른 case에 유효 batch를 자동 변경하지 않는다.
5. Spectral covariance는 band-row/부분 representation 단위로 계산하고, 통계 map의 CPU 로깅은 낮은 주기로 제한한다.
6. 필요하면 activation checkpointing을 별도 검증 후 적용한다. Step time은 다시 측정한다. 동일 init/output/gradient 동치 확인 없이 특정 case만 최적화하지 않는다.
7. Dataset H/W를 줄이거나 band를 줄여 OOM을 해결하지 않는다. 이는 다른 실험이다.

Accumulation에서 microbatch 크기가 다르면 `loss_i × B_i/B_effective`로 전체 batch 평균을 재현한다. 동일 fixed image shape에서 전체 pixel 평균의 동치가 성립한다. Soft active count로 나누지 않는다. I-N은 같은 optimizer update 내 모든 microbatch에 같은 native/corrupt 종류를 쓴다.

### 17.5 수치·통계 위험

- `mean(z²)-mean(z)²`는 cancellation 가능성이 있다. FP32 이상과 공간 상수 제거를 이용하고 음수 diag roundoff만 보정한다.
- 실제로 완전히 일정한 Student 통계에서는 variance gradient가0일 수 있다. 거대한 λ로 해결하려 하지 말고 reconstruction을 유지한다.
- Covariance off-diagonal은 음수가 정상이다. `clamp_min(0)`를 행렬 전체에 적용하지 않는다.
- 통계 loss가 작다는 이유만으로 좋은 reconstruction이 아니다. Shift·ringing·noise의 반례를 tests/diagnostics로 남긴다.
- 대부분의 variance weight가0 또는 상한에 붙으면 representation·τ·단위를 점검한다.

### 17.6 Teacher cache

Online Teacher가 기본이다. Cache를 사용하려면 key에 다음을 포함한다:

`teacher weights hash, model/config hash, sample ID, augmentation transform, input intensity hash, upsampler/phase, crop/view, corruption ε, spatial shape, stat kind/window/version`.

Teacher가 rotation/flip-equivariant하다고 가정해 canonical prediction을 돌려서 대신하지 않는다. Covariance 좌표 변환은 수학적으로 `mu'=Qmu`, `Sigma'=QSigmaQᵀ`지만, 재추론 없이 Teacher network의 예측까지 동일하다는 보장은 없다. 캐시 근사 여부를 검사하고 별도 버전으로 둔다. [S-KD §14]

---

## 18. 평가·checkpoint: 기존 세 view를 보존한다

### 18.1 문서 간 차이를 명시적으로 처리한다

[S-KD §11]은 RR validation ERGAS 중심의 checkpoint 선택을 제안했고, 더 구체적인 [S-A §10; S-N §10]은 **raw/aligned HQNR dual-best와 last, FR mat20·metric v2 py**를 명시한다. 두 규칙은 동일하지 않다.

**이번 기본 계승안:** 실제 s2 evaluator/selector를 확인하고, 최근 앞단 실험의 세 view와 `best_raw`, `best_aligned`, `last`를 유지한다. 추가로 `best_rr_val`을 저장한다. Selector를 바꾸어 얻은 이득을 loss 개선으로 섞지 않는다.

FR mat20이 benchmark test라면, 그 점수로 checkpoint/방법을 계속 고른 결과는 **test-adaptive exploratory**라고 표시한다. 이를 독립 hold-out 성능이라고 부르지 않는다. 정식 확인에는 독립 validation selector 또는 사전 고정 final-N checkpoint와 최종 평가 절차를 별도로 사용한다. 과거 기록을 숨기거나 자동으로 새로운 holdout으로 재명명하지 않는다.

### 18.2 세 view

| View | PAN reference | 영역 | 의미 |
|---|---|---|---|
| **Raw-original** | 원래 P | 기존 Ω0 | 기존 benchmark와 주 비교 |
| **Raw-valid** | 원래 P | 사전 고정 V | 영역 변경 대조 |
| **Aligned-valid** | 해당 forward의 실제 P_aligned | 같은 V | 모델이 사용한 정합 reference의 진단 |

세 view는 **같은 HRMS prediction**을 사용한다. Raw-original은 aligner OFF inference가 아니다. Aligned PAN의 low-resolution reference도 같은 evaluator의 전체 처리 경로로 다시 만든다. 원 PAN의 cached low-PAN을 섞지 않는다.

`Dλ`가 PAN 비의존인 현재 evaluator라면 같은 output/MS/V의 Raw-valid와 Aligned-valid Dλ는 같아야 한다. 다르면 구현 의존성·cache·ROI를 조사한다. 바뀐 PAN reference의 HQNR은 논문 raw HQNR과 같은 지표 조건이 아니다. [S-A §10]

### 18.3 고정 ROI·support

- 원본 evaluator Ω0, 공통 V, filtering/window/block/decimation phase를 manifest로 고정한다.
- Predicted shift가 커졌다고 V를 줄이지 않는다.
- Source-coordinate bicubic support와 이후 필터/metric-window support를 전파한다.
- Support 실패 scene을 빼고 평균하지 않는다. Aligned selection은 INVALID, raw는 유효성 조건에 따라 별도 기록.
- N corrupted stress의 support는 두 warp를 모두 검사한다. 마지막 ε+c 좌표만 보지 않는다.
- `PAN==0`, all-ones border warp, clamp된 좌표만으로 valid mask를 만들지 않는다.
- s2 원본 metric code/config hash와 데이터 파일 hash를 모든 run에 저장한다.

### 18.4 Checkpoint 저장

```text
last.pt                  # 항상 정확한 현재 optimizer update와 상태
last_NFULL.pt            # 정확한 full-N 종료 checkpoint
best_raw.pt / meta.json
best_aligned.pt / meta.json
best_rr_val.pt / meta.json
phase_parent_stepXXXXX.pt
```

최근 dual-best 규칙은 running maximum HQNR, 1e-4 동률에서 해당 view fSCC, 다시 동률이면 더 늦은 checkpoint다. s2 실제 구현과 tie 규칙을 일치시킨다. Best label만 보고 Teacher를 골라오는 대신 run/step/view/hash를 정확히 선택한다. `best_aligned`를 raw best인 것처럼 로드하지 않는다. [S-N §10]

Native 지표로 선택하고 synthetic stress는 기능 진단으로 둔다. 평가 주기는 기존 설정을 유지하거나 신규 cohort 시작 전에 고정한다. 최초 full proposal은 RRval1K, FR checkpoint2.5K, exact final-N이지만 계산비가 크면 공통 주기로 수정 후 모든 matched run에 적용한다. 이는 새 제안값이지 과거 실측 설정이 아니다.

---

## 19. 기전 분석: 무엇이 개선되어야 하는가?

### 19.1 전체 품질

- RR: ERGAS, SAM, PSNR, SSIM, SCC, Q8. 같은 M-frame의 GT와 비교.
- FR: HQNR, Dλ, Ds, fSCC와 동일 crop 시각 결과. 지표 코드는 기존 것을 사용.
- Raw·aligned 점수 차이는 순수 정합 오차량이 아니다.
- 모듈/추론: params, total trainable/frozen, raw image sizes별 inference time, train peak memory.

### 19.2 Teacher 실패 영역의 fitting

고정 Teacher의 **평가용** error quantile로 영역을 정의해 모든 Student에 같은 mask를 적용한다:

`Q0–50 / Q50–90 / Q90–100`.

Pixel error bin과 통계 error bin을 별도로 만든다. 해당 mask는 사후 분석용이며 train calibration이나 test-driven hyperparameter tuning에 되먹이지 않는다.

$$\Delta e(q)=\mathbb E[e_{control}-e_{candidate}\mid q],\quad
WinRate(q)=\Pr(e_S<e_T\mid q).$$

Gate-positive 영역은 Student마다 변하므로 그것만 비교해 개선을 주장하지 않는다. 모든 run의 unweighted train/validation MAE도 보존한다.

### 19.3 Variance의 실제 효과

| 관찰 | 필요한 확인 |
|---|---|
| 통계 오차만 감소 | Pixel MAE/SAM/edge 위치도 개선되는지 |
| Gradient variance 증가 | GT 수준의 증가인지 noise/ringing인지 |
| GT variance-H ≈ AD | Teacher 통계 KD가 불필요한지 WH/FIX와 비교 |
| GC > GV | Off-diagonal 방향 정보가 기여하는지 |
| SC > GV | 분광 공동변화가 잔여 병목인지 |
| 가중 통계 위치 shuffle도 비슷함 | 지역 정보보다 전체 regularization 가능성 |

평탄/어두운 영역, 강한 edge, 반복 구조를 별도 집계한다. Sample panel은 개선 상위뿐 아니라 악화 상위와 미리 고정한 scene을 포함한다.

### 19.4 정합 기능

기본 probe:

$$q(\epsilon)=\mu_\epsilon-\mu_0\approx B\epsilon+b,\quad B\approx-I,\ b\approx0.$$

Native/train64, RR256, FR512에서 response slope·cross-axis·closure EPE를 기록한다. 단위는현재 HRpixel이며 크기가 바뀌었다고 correction에8을 곱하지 않는다.

Same-weight zero/learned/wrong-sign, MS-swap/constant, shuffled shift, interpolation kernel/padding 진단을 수행한다. 추가 변위 복원이 좋아진 것과 native absolute alignment가 정확한 것은 다르다. [S-N §10]

### 19.5 Overfitting 논지의 평가

이 문서는 사용자의 **특정 센서·분포에 더 깊게 fitting한다는 연구 목표**를 유지한다. 과거 [S-BASE]의 “overfitting 완화” 프레이밍으로 되돌리지 않는다.

단, 다음 결과를 분리한다:

1. 가중치 없는 train 오차와 Teacher 실패영역 오차가 감소했는가?
2. 같은 센서의 validation/RR·native FR 품질도 유지·개선됐는가?
3. 개선이 같은 추가 compute의 supervised continuation보다 큰가?
4. Teacher error map/statistics가 실제로 위치별 추가 정보를 제공하는가?

Train만 좋아지고 validation/FR이 나빠지면 “더 fitting됐다”는 관찰과 “실용적으로 좋아졌다”는 결론을 분리한다. 같은 architecture KD라서 extra training budget 대조가 특히 중요하다.

---

## 20. 단위검사·s2 integration gate

### 20.1 이 문서 작성 중 확인한 범위

문서 작성 컨테이너는 **PyTorch 2.10.0+cpu, CUDA unavailable**이다.

| 검사 | 결과 | 의미 |
|---|---|---|
| 기존 `pan_gt_anchored_kd.py` | **10/10 통과** | 손계산·gate detach·GT gradient·R-mode |
| 새 `pan_s2_w112_stat_kd_reference.py` | **12/12 통과** | 통계·shape·gradient·half 입력 승격 |
| 실제 W112 U-Net/aligner integration | **미실행** | s2에서 확인 필요 |
| PanCollection 실제 tensor 학습 | **미실행** | 성능 주장 없음 |
| CUDA/AMP/VRAM/throughput | **미실행** | 24h 소요 보장 없음 |

새 12개 검사는 상수 통계0, population variance explicit 비교, intensity 배율의 제곱 scaling, band별 brightness offset 불변성, GV/GC diagonal 동치, spectral covariance 대칭·PSD, Student-only gradient, target detach, exact-GT loss0, half FP32 계산, 오류 입력·공통 ROI를 확인한다.

### 20.2 실행자가 추가할 필수 gate

| Gate | 통과 조건 |
|---|---|
| M01 architecture | W112,D124,9→8,단일 HRMS,normalization divisibility,실제 param dump |
| M02 checkpoint | W96 F를 W112로 partial load하지 않음; donor strict compatibility |
| M03 teacher fixed | Teacher state hash 불변, optimizer parameter 교집합 없음 |
| M04 residual | MS base 정확히1회, GT/MS 위치불변 |
| W01 sampler | identity·impulse dy/dx·align_corners·nonzero delta gradient |
| W02 support | 고정 geometry/eval ROI, N 두 단계 support |
| W03 N SG/noSG | Native graph 유지, corrupt reference만 차단/해제 구분 |
| L01 rec | 파일의10 tests + 실제8band per-pixel gradient |
| L02 stats | gradientvar/cov 크기·단위·GT/Teacher detach·same centers |
| L03 routing | A-FR의 geo/off/GKD=학습효과 없음; FT/SC에서 live |
| L04 spectral | row chunking vs explicit small tensor 동치·음의 covariance 보존 |
| C01 covariance | delta-Jacobian autograd/finite-diff, rank/floor/precision cap |
| C02 distribution | KL 방향, SPD, 같은분포 loss0, head의 독립 target |
| E01 evaluator | raw/valid/aligned reference·ROI·mat20·aggregation·selector |
| R01 resume | 모델·optimizer·scheduler·scaler·모든 RNG·N parity 동치 |
| P01 actual smoke | 실제 배치에서 finite, OOM 없음,시간·memory 기록 |

Implementation gate 실패는 연구 가설의 실패가 아니다. 임의로 loss를0으로 바꾸고 계속 돌리지 않는다. 원인을 수정한 새 version으로 해당 matched pair를 재실행한다.

---

## 21. 필수 로그·폴더·실험 상태 관리

### 21.1 캠페인 폴더

```text
<ACTUAL_WORK_ROOT>/S2_W112_KDV_20260910/
  campaign_manifest.yaml
  environment_s2.json
  architecture_manifest.json
  source_hashes.json
  teacher_registry.json
  implementation_map.md
  queue.csv
  decisions.md
  review_24h.md
  calibration/<teacher_id>/<stat_or_geo_id>/
  initial_states/<seed>/
  runs/<run_id>/
    config_resolved.yaml
    init_and_teacher_hashes.json
    parent_and_phase.yaml
    train_log.jsonl
    gradient_diagnostics.jsonl
    checkpoint_metrics.csv
    scene_metrics.csv
    fitting_bins.csv
    covariance_diagnostics.jsonl
    memory_and_throughput.json
    phase_events.jsonl
    best_raw.pt / best_aligned.pt / best_rr_val.pt / last.pt
    predictions/ visualizations/
  reports/
    screen_comparison.md
    negative_results.md
    full_budget_comparison.md
    final_ablation_table.csv
    resource_ledger.csv
```

`ACTUAL_WORK_ROOT`는 s2에서 확인한다. `/mnt/data` 문서 생성 경로를 remote 실험 경로처럼 사용하지 않는다.

### 21.2 Train logging

일반 scalar는100update 간격 등의 공통 주기로, gradient 진단은1K에 수행하는 새 제안을 둔다. 실제 evaluator 주기에 맞춰 사전 확정한다.

```text
update / attempt / elapsed / active_gpu_hours / phase_id
raw_rec_mae / rec_hard / rec_soft / eT / eS / eST
dR/ aR / hard_weight / soft_weight / active_fraction / student_win_fraction
stat_kind / window / stat_H / stat_T / stat_total / tau_V / lambda_V
dV / aV / stat_student_win / flat_region_error / variance_ratio_to_GT
mu_T / mu_S / delta_drift / offset_closure / learned_support
geo_hard / mean_KD / cov_KD / rank / eigenvalues / precision_caps
norm_grad_F / norm_grad_A / dLoss_dDelta / gradient_cosines
peak_memory_allocated / reserved / update_time / dataload_time
actual_lr_F / actual_lr_A / amp_scale / skipped_step
```

Full maps는 고정한 몇 sample과 checkpoint에서만 저장한다. Scalar logging을 위해 GPU tensor를 매 step 수십 번 `.item()`해 속도를 왜곡하지 않는다.

### 21.3 Queue 상태

필수 column:

`run_id, priority, hypothesis, control_id, recipe, input_protocol, teacher_id, W_T, W_S, aligner_policy, rec, stat_kind, stat_mode, G_mode, seed, init_id, budget_updates, parent, status, last_step, elapsed, next_action`.

24h에는 완료·진행·미실행 접근과 이유, 주요 비교의 동일 budget 여부, 필요한 추가 비용을 보고한다. “시간 초과로 자동 중단” 상태는 기본값이 아니다.

---

## 22. 실제 실행용 config template

다음 YAML은 **새 adapter가 받아야 할 설정 계약**이다. 기존 repository에서 이 key가 이미 지원된다는 뜻은 아니다. Required/null은 resolver가 실제 값으로 채우고 검증해야 한다. 미확인 값을 채운 것으로 위장하지 않는다.

```yaml
campaign:
  id: S2_W112_KDV_20260910
  server: s2
  version: v01
  mode: prioritized_adaptive
  target_review_hours: 24
  hard_time_limit_hours: null
  allow_overrun: true
  continue_after_review: true
  run_kind: CONTROLLED                 # 또는 ADAPTIVE_PATH
  work_root: null                      # REQUIRED_ACTUAL_S2_PATH

model:
  backbone_class: null                  # REQUIRED_ACTUAL_CLASS
  backbone_source_hash: null
  width: 112
  depth: [1, 2, 4]
  ms_bands: 8
  in_channels: 9
  out_channels: 8
  input_order: [PAN, bicubic_MS]
  task: hrms_residual_only
  add_ms_base_exactly_once: true
  output_frame: M
  attention: false
  cm3a: false
  lpan_hpan_inputs: false
  pan_reconstruction: false
  output_inverse_warp: false

teacher:
  id: null                             # REQUIRED
  checkpoint: null                     # REQUIRED
  checksum: null                       # REQUIRED
  restoration_width: 112
  frozen: true
  same_raw_input_view: true
  bridge_t96_to_s112: false
  source_recipe: null                  # A1/A2/A3/N1/N2/N3/declared_new_recipe

student:
  restoration_init: saved_random
  restoration_init_path: null
  aligner_policy: A-FR                 # A-FR/A-FT/A-SC/A-ID/A-FTW
  aligner_donor_checkpoint: null
  teacher_student_shared_trainable_parameters: false

aligner:
  architecture: dualstem_global_105330_reference
  actual_source_hash: null
  output_order: [dy, dx]
  output_units: current_PAN_HR_pixel
  last_head_zero_init_if_scratch: true
  input_view_policy: inherit_donor
  crop_margin_hr: null                 # N reference는 4, 실제 donor 규칙 확인
  zscore_after_crop: true
  use_gt_in_forward: false

input:
  protocol: null                       # REQUIRED: I-A / I-N / I-NATIVE-TRANSFER
  ms_upsampler: inherit_verified_bicubic
  intensity_manifest: null
  random_scale_crop: false
  corruption:
    enabled: null                      # protocol과 일치시킬 것
    native_corrupt_ratio: [1, 1]
    distribution: uniform_disk_area
    radius_hr: null                     # I-N 기존 proxy 후보 1.0, 실제 manifest 확인
    scale_source: null
    independent_rng: true
    pass_epsilon_as_input: false

warp:
  convention: 'out[y,x]=source[y+dy,x+dx]'
  mode: bicubic
  padding_mode: border
  align_corners: false
  learned_zero_shortcut: false
  corrupt_two_raster_warps: true
  precision: float32

loss:
  reconstruction:
    mode: adaptive
    alpha: 1.0
    kd_weight: 0.1
    tau: null                          # train calibration 필수
    eps: 1.0e-6
    binary_switch: false
    detach_teacher_and_gate: true
    reduction: all_pixels_mean
  statistics:
    enabled: true
    kind: grad_var                    # image_var/grad_var/grad_cov/spectral_cov
    mode: adaptive                    # H/T/FIX/WH/AD를 adapter에서 mapping
    window: 5
    stride: 1
    centered: true
    correction: 0
    common_valid_margin: 3
    tensor_reduction: mean_absolute_all_components
    target: final_hrms
    alpha: 1.0
    kd_weight: 0.1
    tau: null
    eps: null
    outer_weight: null                # train pilot calibration
  inherited_auxiliary:
    recipe: null
    edge_weight: null
    geometry_weight: null
    offset_weight: null
    ramp_updates: null
    disable_aligner_only_terms_if_frozen: true
  alignment_kd:
    mode: G0
    covariance_source: none           # geo_curvature / eq_closure / learned
    apply_on: native_only
    outer_weight: null
    require_trainable_student_aligner: true
    keep_gt_reconstruction_when_invalid: true
    teacher_mean_detached: true
    teacher_precision_detached: true

training:
  profile: inherit_verified_s2
  fallback_profile_requires_resolution: true
  seed: 1234
  teacher_seed: null
  optimizer: null
  lr_backbone: null
  lr_aligner: null
  effective_batch: null
  microbatch: null
  accumulation: null
  full_updates: 50000
  screen_fractions: [0.25, 0.5, 1.0]
  screen_uses_full_run_scheduler: true
  teacher_no_grad: true
  covariance_separate_graph: true
  amp_backbone: null
  geometry_statistics_dtype: float32

selection:
  policy: inherit_verified_s2_dual_best
  save: [best_raw, best_aligned, best_rr_val, last, phase_parent]
  views: [raw_original, raw_valid, aligned_valid]
  expected_fr_reference: mat20_v2_py
  actual_metric_hash: null
  scene_and_split_manifest: null
  fixed_roi_manifest: null
  primary_exploratory: best_raw
  formal_report_also_fixed_final: true
  test_adaptive: null                  # 실제 selection split에 따라 true/false

phase_change:
  enabled: true
  checkpoint_before_change: true
  require_config_diff: true
  require_parent_and_cost_ledger: true
  paired_continuation_control: true
  overwrite_previous_run: false
```

Resolver는 `A-FR + alignment_kd≠G0`, `teacher width96 + bridge=false`, `input I-N + unknown radius`, `stat tau missing`, `shared trainable Teacher parameter` 등의 잘못된 조합을 error로 처리한다. 단순히 warning만 내고 진행하지 않는다.

---

## 23. 결과에 따른 채택·수정·scratch 전환

| 결과 | 다음 행동 | 해석 범위 |
|---|---|---|
| R3>R1 | output target KD 유지 | hard weighting 이상의 기여 |
| R3≈R1 | R1을 강한 기준으로 유지, KD 약함을 기록 | 불필요한 복잡도를 주장하지 않음 |
| GV-AD>GV-H/WH/FIX | adaptive 통계 후보를 확장 | GT·Teacher 통계 선택이 유효 |
| GV-H만 좋음 | GT 통계 fitting 중심으로 단순화 | Teacher variance KD의 우월성 미입증 |
| A-FT>A-FR | trainable aligner 채택 검토 | Student별 정합 수정이 유효할 수 있음 |
| A-SC>A-FT | scratch를 주 경로로 전환 | Teacher 초기값의 제약 가능성 |
| A-ID>모든 aligner | 정합 모듈을 뺀 baseline에서 R/stat 재시험 | 정합 효과를 전제로 KD를 해석하지 않음 |
| G3>G2>G1 | 방향별+sample별 신뢰도 유지 | covariance의 독립 효과 |
| G-EQ 좋지만 native alignment 불명확 | 반응 신뢰도라고만 해석 | 절대 uncertainty claim 금지 |
| G5 loss만 감소 | head 기여 보류 | 분산 예측 성공과 복원 개선 구분 |
| 모든 새 loss가 나쁨 | plain GT + scratch에서 한 항씩 재실험 | 강한 fitting 목표를 버리기보다 기전 재분해 |

**Scratch로 전환해도 원래 reconstruction hard/soft는 사용할 수 있다.** 다만 Teacher 정합이 오류의 원인이면 alignment KD도 끈 대조가 필요하다. 잘못된 Teacher를 새 초기값에서 다시 모방하는 것만으로 문제가 해결되지는 않는다.

Exploratory winner는 동일 시작점·full budget·후속 seeds로 확인한 뒤 정식 방법으로 고정한다. 24h 시점에는 “현재 가장 유력한 조합”과 “확인된 기여/아직 미확인인 기여”를 나누어 보고한다.

---

## 24. 문서·구현·실험의 출처와 확인 범위

### 24.1 내부 자료

- **[U-CURRENT]** 이번 사용자 결정: s2 실행, width112, 현 접근을 우선순위로 모두 검토, 약24h 목표지만 초과 허용, 중간 변경 허용, 상세 MD 요청.
- **[U-RECENT]** 최근 합의: Teacher aligner 재사용 우선, trainable 미정, 필요하면 scratch; 원래 이동 covariance와 새 출력 통계 variance를 구분.
- **[S-KD]** `PAN_W96_D124_GT_Anchored_KD_Experiment_Plan_2026-09-09.md`: §§3–4 continuous hard/soft, §5 alignment covariance, §§7–10 ablation/initialization, §§11–16 evaluation/implementation. 이번 W112/s2의 직접 성능 근거가 아니라 계승할 수식·검증 원칙.
- **[S-A]** `PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2 (1).md`: §§2–4 actual aligner reference/sampling, §5 A1/A2/A3와 A3 band-mean geometry, §6 support, §10 raw/aligned evaluation.
- **[S-N]** `PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md`: §0 N계열 차이, §§3–6 corruption/crop/sequential warp/relative offset, §§9–10 gradients/evaluation. s1의10h 제한은 이번 사용자 결정으로 승계하지 않음.
- **[S-BASE]** `PAN_research_baseline_W96_D124_2026-09-09.md`: 기존9ch·단일 HRMS·MARs/attention 제거·MS frame 원칙. width96 및 과거 연구 단계 표현은 이번 결정으로 대체.
- **[S-AUDIT]** `2026-09-03_alignment-audit-s2-detail.md`: RR/FR·phase·분광 하한·shift label 한계. Audit vector를 sample별 참 정합 label로 넣지 않음.
- **[C-REC]** `pan_gt_anchored_kd.py`: 실제 재실행한 reconstruction reference와10개 CPU tests.
- **[C-STAT]** `pan_s2_w112_stat_kd_reference.py`: 이번에 작성한 output-statistic reference와12개 CPU tests. W112 full network 구현은 아님.

### 24.2 외부 자료의 지위

첨부 설계에서 참조한 API/원리 자료의 주소를 아래에 보존한다. **이번 작성 시 웹 도구 재조회는 서비스 오류로 실패**했으므로 최신 문서 version을 다시 확인했다고 주장하지 않는다. 코드의 수학/CPU 동작은 위 명시한 로컬 환경에서 검사했다. s2의 설치 API·CUDA backward·AMP는 별도 gate다.

```text
PyTorch grid_sample
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
PyTorch autograd mechanics / no-grad / freezing
https://docs.pytorch.org/docs/stable/notes/autograd.html
PyTorch AMP examples / accumulation
https://docs.pytorch.org/docs/stable/notes/amp_examples.html
Ceres Solver — Covariance Estimation
https://ceres-solver.readthedocs.io/latest/nnls_covariance.html
```

이 문서는 신규성 조사를 새로 수행한 문헌 리뷰가 아니다. 학습 성능·alignment 정답·covariance calibration을 확보했다고 말하지 않는다. 새로운 window/λ calibration/priority/phase 규칙은 **실행을 위한 제안**이며, 기존 실험에서 검증된 설정으로 승격시키지 않는다.

### 24.3 읽은 주요 원본 파일의 SHA256

```text
PAN_W96_D124_GT_Anchored_KD_Experiment_Plan_2026-09-09.md
32d3907fcfa0a89e1fb5424a006c7596d88d8b728dd9120c0d64aaf197a3ac63

PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2 (1).md
4ea1f157151a1b8a25033afa175f9923e33d4ee385c6643c46c2c86d6d94d64b

PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md
334a13fdae48faa177d839d5b04e938b283083c97edc56b67c430152f70077f3

pan_gt_anchored_kd.py
36a69ad5516a635a86e895c8af291d3f8253fc49e6f3326d4862edaa3fc0f2a9
```

---

## 25. 실행자에게 전달할 최종 요약

**s2에서 W112–D124/9ch/단일 HRMS 기준으로 진행한다.** W96 복원망은 W112에 그대로 로드하지 않는다. 독립 aligner만 호환 확인 후 재사용할 수 있다. KD Teacher는 우선 W112로 맞추고 다른 폭의 bridge는 별도 표기한다.

**Reconstruction은 승인한 continuous hard/soft를 그대로 사용한다.** GT는 모든 위치에서 유지하며 Teacher error로 hard를 강화하고, Student가 더 좋아진 위치에서는 soft를 없앤다. 이진 switch·새 uncertainty head로 자동 변경하지 않는다.

**먼저 Teacher aligner frozen에서 통계 loss의 효과를 분리하되, frozen을 최종 결론으로 고정하지 않는다.** 같은 loss를 Teacher-copy trainable과 scratch aligner에서 이어 비교한다. 공유 Teacher 가중치는 업데이트하지 않는다.

**출력 GT/Teacher gradient variance와 이동량 covariance를 분리한다.** Frozen이면 출력 통계는 학습되지만 alignment-only loss는 Student에 신호가 없다. Trainable에서만 원래 mean/scalar/full covariance KD를 검증한다.

**약24시간에는 중간 review를 만들고, 필요한 비교가 남으면 계속 진행한다.** 중간 변경은 checkpoint·phase diff·누적 비용을 남기고 동일 parent의 continuation control을 붙인다. 유망 경로를 찾은 뒤 고정 구성의 대응 full-run으로 확인한다.

**핵심 첫 비교:** `REC-N0 / R1 / R3`, `GV-H / GV-AD`, `A-FR / A-FT / A-SC`. 이후 GV-WH/FIX/T, IV/GC/SC, mean/scalar/full alignment covariance, 분포 KD, 장기 fitting·seed 검증 순으로 확장한다.
