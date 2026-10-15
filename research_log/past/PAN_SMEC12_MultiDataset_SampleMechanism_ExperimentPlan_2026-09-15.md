# PAN SMEC12 — Recon–Consistency 불일치의 원인과 sample 특성 검증

**작성일:** 2026-09-15  
**문서 상태:** 신규 실험·분석 설계. 실행 결과가 아니다.  
**실행 대상:** s1을 포함한 가용 서버. 서버별 경로·자산·환경은 시작 시 탐색한다.  
**범위:** WV3·QB·GF2·WV2 전체의 표본 수집, sample 특성 분석, 정합·복원 반응 분리, 센서 간 재현, 조건부 대응 검증.  
**시간:** 자산이 준비된 경우 핵심 분석 약 12시간을 목표로 한다. 12시간은 강제 종료선이 아니다. 누락 모델의 준비 학습 및 필수 확인은 별도 합산하며, 시간을 맞추려고 학습량이나 데이터셋을 임의로 줄이지 않는다.  
**고정:** WV3의 L1E4 채택은 유지한다. 이 캠페인은 새 λ 최고점을 찾거나 q를 Teacher 신뢰도로 기정사실화하는 실험이 아니다.

> **중심 질문:** 왜 어떤 sample은 offset consistency는 좋지만 reconstruction은 어려운가? 그것은 구조·대비·spectral 차이·위치 민감도·입력 context 중 무엇과 연결되는가? 그 특성을 다른 장면과 센서에서도 식별할 수 있고, 특성에 맞춘 처리가 실제로 유리한가?
>
> **실행 원칙:** 필요한 baseline 또는 donor가 해당 서버에 없으면 **준비 학습을 자동으로 큐에 넣는다.** 모델이 없다는 이유로 그 센서의 분석을 조용히 생략하지 않는다. 단, 없는 학습 데이터나 GT를 만들어 넣지는 않는다.

---

## 문서 탐색

| 필요한 내용 | 바로가기 |
|---|---|
| 실행 요약·case 목록 | [시작 요약](#quickstart) · [전체 case](#cases) |
| 모델이 없는 서버 | [Baseline/donor 준비 학습](#bootstrap) · [데이터셋 계약](#datasets) |
| 네 집단과 데이터 특성 | [측정 정의](#definitions) · [Atlas/척도 분석](#atlas) |
| 실제 정합·복원 기전 | [같은 잔여 이동](#residual) · [Correction landscape](#landscape) |
| 센서 재현·대응 가능성 | [여러 센서](#crosssensor) · [q 추가 정보](#cuevalue) · [맞춤 대응](#treatment) |
| 운영·보고 | [12시간/준비 학습](#budget) · [구현](#implementation) · [판정](#verdicts) · [근거](#sources) |

---

<a id="quickstart"></a>

## 0. 실행자가 먼저 볼 요약

### 0.1 이번에 반드시 남길 답

1. EuCd(e↑·q↓)가 어떤 시각적·구조적·spectral 특성을 가지는지 원본 sample과 수치로 설명한다.
2. q–e 음의 상관이 대비·복원할 세부의 양을 통제해도 남는지 확인한다.
3. **변위를 읽는 능력**, **같은 잔여 변위에 대한 출력 민감도**, **보정의 실제 이득**을 각각 측정한다.
4. 구조적으로 선호되는 correction과 reconstruction이 선호하는 correction의 차이를 검사한다.
5. 위 관계가 같은 WV3 모델의 다른 seed뿐 아니라 QB·GF2의 해당 센서 학습 모델, WV2의 zero-shot 평가에서도 나타나는지 분리해서 보고한다.
6. q가 저렴한 영상 descriptor나 기존 error 정보보다 추가 정보를 주는지 확인한다.
7. 원인과 추가 정보가 확인된 경우에만 특성별 대응을 시험한다. q 기반 soft/hard gate를 먼저 만들고 결과를 끼워 맞추지 않는다.

### 0.2 실행 순서

```text
A00 자산·데이터·source audit
 ├─ 자산 있음: 재현 확인 후 사용
 └─ 자산 없음: B01 P0 → B02 donor → B03 L000/L1E4 → B04 L1E4 반복 학습
                     ↓
D10 전 센서 atlas → D11 대비·난이도 분해 → D12 구조/spectral 특성·blind gallery
                     ↓
I20 동일 잔여 이동 → I21 aligner 단서 개입 → I22 spectral/known-geometry 대조
                     ↓
I23 native 정합·correction landscape → I24 보간 분리 → I25 gradient 국소 개입
                     ↓
S30 context·checkpoint·seed 안정성 → X40 센서별/교차센서 재현
                     ↓
C50 q 추가 정보 검증 → 조건 충족 시 R60 맞춤 대응 pilot → 최종 판정표
```

### 0.3 하지 않을 것

- WV3의 q 임계값을 모든 센서에 그대로 적용하기.
- `q↓ ⇒ Teacher 정확`, `q↑ ⇒ hard`, `EuCd ⇒ soft`를 미리 정하기.
- 기존 8-band 모델에 4-band를 복제/zero-padding하여 정상적인 QB/GF2 모델로 취급하기.
- 학습 데이터가 없는 WV2의 공식 test를 학습셋으로 사용하기.
- q 또는 reconstruction으로 집단을 정의한 뒤, 그 정의 자체의 차이를 발견이라고 보고하기.
- 같은 checkpoint의 best 영상 품질과 last 정합 능력을 결합하기.
- 새로운 λ/width/local-flow/feature-warp를 동시에 탐색하기.
- 12시간에 맞추려고 누락 donor를 random weights로 대체하거나 센서 하나를 누락하기.

---

## 1. 출발 근거: 관측과 해석을 분리한다

### 1.1 자료가 실제로 보고한 관측

| 항목 | 보고 내용 | 이번 설계에서의 처리 |
|---|---|---|
| L1E4 native64의 q–e | 세 seed ρ = −0.115 / −0.122 / −0.149 | 약한 음의 관계의 재현부터 확인. 강한 결정 법칙으로 쓰지 않음 |
| H2 | q가 낮은 sample이 추가 PAN 이동에서 더 큰 절대 L1 증가 | 같은 잔여 이동에 대한 민감도와 분리 |
| EuCd | e=.0242, q=.311, stress d_e=.0413, PAN swap d_e=.0871 | 값의 집계 subset이 다름. 원시 sample ID를 맞춰 새로 연결 |
| EuCd correction 이득 | +.00167, 다른 집단보다 큼 | zero error·texture·대비를 통제하고 재검사 |
| EuCd Scharr energy | .0119, 다른 집단 .002–.0065 | 구조 복잡성인지 단순 대비인지 분리 |
| 모델 의존성 | 강한 λ의 L1E2에서는 q–e 부호가 반대 | 데이터 고유 특성과 학습 상태를 구별 |
| MS 내용 대조 | MS 상수화/교환 후 q·EPE 악화 | MS 사용의 근거. spectral 원인 확정은 아님 |
| q 기반 K20 | e/a/q 배분의 미세한 상대 L1 이득, 모든 arm의 source 대비 품질 하락 | q-gate를 현재 방법으로 채택하지 않음 |

근거: [R1] §2–§6. 네 집단은 전수 core, stress는 D 상세 subset이므로 기존 표의 열들을 동일 표본의 관측처럼 합산하지 않는다.

### 1.2 자료의 한계도 그대로 계승

- s3는 s1 checkpoint를 이식한 재현이다. 독립적인 센서/학습 반복이 아니다. [R2 §1]
- 실제 scene ID 대신 연속 index block을 사용한 한계가 있었다. 이번에는 source provenance 복원을 먼저 시도한다. [R1 §8]
- native 정합 proxy는 2차 독립 추정 검증이 부족했다. [R1 §6]
- `known_inverse`의 출력 복원은 CNN의 absolute offset 예측 정확도가 아니다.
- H3b 판정은 s1과 s3에서 다르다. **음의 상관 관측**과 **조건부 예측력 미확정**을 함께 남긴다. [R1 §2; R2 §2, §7]
- 기존 q-gate의 실패는 모든 특성 기반 대응의 실패가 아니다. 반대로 이번 특성 분석이 KD 성공을 보장하지도 않는다.

### 1.3 이번의 신규 가설

| ID | 가설 | 반증 또는 범위 축소에 해당하는 결과 |
|---|---|---|
| M1 대비/척도 | Aligner 표준화와 raw reconstruction 척도의 차이가 음의 상관에 기여 | 대비 통제 전후 관계가 거의 같고 통제 개입에서도 변화가 없음 |
| M2 구조 관측 가능성 | 두 방향의 구조 단서가 풍부해 q는 낮지만, 세부 복원량·위치 민감도는 큼 | 대비를 맞춘 corner/방향성·주파수 descriptor와 관계가 없음 |
| M3 출력 민감도 | q보다 `동일 잔여 이동에 대한 출력 변화`가 stress 손실을 잘 설명 | 동일 변위에서 집단 차이가 사라지거나 설명력 추가 없음 |
| M4 spectral 불일치 | 공통 구조는 이동 추정에 충분하지만 band별 복원에는 부족 | 기하 고정 spectral 개입에 q/absolute bias/오차 구조가 반응하지 않음 |
| M5 context/모델 | crop·GN/GAP·해상도·학습 상태에 따라 집단이 바뀜 | 동일 물리 support·다른 seed/context에서도 membership이 안정 |
| M6 데이터 재현 | 유사 특성–q–민감도 관계가 여러 센서에서 반복 | WV3에만 나타나거나 sensor-specific direction이 일관되게 반대 |
| M7 대응 가능성 | descriptor에 q를 추가하면 독립 outcome 예측과 맞춤 대응이 개선 | descriptor-only와 같거나, q 셔플로 결과가 유지됨 |

M1–M7은 새 가설이다. 연구자료가 이미 원인을 확정했다는 표현을 사용하지 않는다.

---

<a id="bootstrap"></a>

## 2. 고정 모델과 baseline 준비: 없으면 먼저 학습한다

### 2.1 공통 forward

센서의 MS band 수를 C라고 한다.

\[
M=U_{bicubic}(S),\quad c_0=A_\phi(P,M),\quad
\Phi(P,M;c)=M+F_\theta([\mathcal W(P,c),M]).
\]

\[
\widehat Y_0=\Phi(P,M;c_0),\qquad
\mathcal W(P,c)[y,x]=P[y+c_y,x+c_x].
\]

CNN 출력은 `(dy,dx)` 2개다. U-Net 입력은 **보정 PAN 1ch + 원래 MS Cch**다. MS·residual base·GT·output inverse는 움직이지 않는다.

| 항목 | 고정 내용 |
|---|---|
| Backbone | 기존 `fixed`, W112, depth=[1,2,3], attention/CM3A 없음 |
| 복원 출력 | C-channel residual + 그대로인 M |
| Aligner | dual stem 1→16 / C→16, concat32, stride2 32→32→64, 64 residual, GAP, MLP64→32→2 |
| Aligner view | margin4 crop 후 sample·band별 z-score; norm epsilon=1e-6 |
| Sampling | 기존 bicubic, border, align_corners=False, FP32 displacement/warp |
| Native training | reconstruction 매 update, Rec→A/U |
| L1E4 auxiliary | PAN 복사본 jitter, Off→A only, λ_off=1e-4, 홀수 update |
| Jitter | 반경 b=2 현재 HR pixel, 0 중심 원판 면적 uniform |
| 복원 학습 입력 | 추가 jitter 없는 PAN을 c0로 보정한 것. Jittered PAN은 U-Net에 미전달 |
| 추론 진단 | frozen A/U. 이것은 채택 method의 학습 동결을 뜻하지 않음 |

WV3 고정 명세는 [R3 §2–5]를 따른다. C=4로의 확장과 센서별 donor 준비는 **이번 신규 이식 설계**다. b=2와 λ=1e-4는 센서 간 비교를 위한 공통 recipe이며, 각 센서의 최적값이라는 주장이 아니다.

### 2.2 센서별 필수 모델 세트

| Asset | 초기화/학습 | 역할 |
|---|---|---|
| **P0** | Aligner 없음, 새 U-Net, recon-only 50K | 전체 method 대비 무정합 baseline |
| **DON-N2** | N2형 변위 사전학습 50K; exact50K aligner 보존 | L000/L1E4의 동일 donor |
| **L000** | DON-N2의 A + 새 U-Net, recon 공동 학습 50K | offset auxiliary 없는 대조 |
| **L1E4** | 같은 DON-N2의 A + L000과 동일 U-Net 초기값, 50K | 주 분석 모델 |
| **L1E4-REP** | 같은 donor + 다른 U-Net seed, 50K | 특성 관계의 학습 반복 확인 |

**기본 seed:** 2025 / **확인 seed:** 1234. WV3의 기존 7777도 유효 자산이 있으면 모두 분석한다. 신규 센서에서 세 번째 seed를 만들기 전에 전 센서의 1차 분석과 2-seed 관계 확인을 우선한다.

L000/P0를 1 seed만 준비한 신규 센서에서는 **λ/전체 방법의 seed-평균 성능 우월성까지 주장하지 않는다.** sample 기전의 확인 목적이다. 그 우월성을 주장하려면 이후 해당 대조군도 대응 seed를 추가한다.

### 2.3 자동 준비 규칙

1. 실행 서버의 검증된 registry를 읽는다. 서버 이름·절대 경로를 s1으로 하드코딩하지 않는다.
2. 기존 자산은 `dataset, band_order, model_class, source_split, donor_hash, update, A/U_hash, preprocess_hash`가 맞을 때만 사용한다.
3. 같은 자산의 검증된 bundle이 **이미 접근 가능**하면 hash 검사 후 재사용할 수 있다. 다른 서버에서 파일을 보내 주기만 기다리며 큐를 멈추지 않는다.
4. **유효 P0가 없으면 P0를 학습한다.**
5. **유효 DON-N2가 없고 새 L000/L1E4가 필요하면 donor를 먼저 학습한다.**
6. L000/L1E4가 없으면 donor 이후 해당 50K run을 만든다. Aligner만 준비됐다고 U-Net을 random인 채 분석하지 않는다.
7. 확인용 L1E4가 없으면 B04를 만든다. 모델 준비는 시간 때문에 자동 취소하지 않는다.
8. 기존 L1E4만 있고 donor가 사라졌다면 그 A/U의 frozen 진단은 가능하다. 다만 L000을 다른 donor로 만들어 같은 대응 비교라고 하지 않는다. 새 donor에서 L000/L1E4 대응 family를 별도로 만든다.
9. 모델이 재학습되면 새 run ID/hash를 부여한다. 원래 L1E4 수치를 재현한 것으로 덮어쓰지 않는다.

**모델 부재는 training dependency다. 데이터·필수 source code 부재는 실제 blocker다.** 데이터가 없을 때 FR test에서 임의 학습셋을 만들거나 과거 architecture로 대체하지 않는다. 의존성을 보고하되 다른 센서의 완료 가능한 작업은 계속한다.

### 2.4 B01 — P0 bootstrap

`B01_<sensor>_P0_W112_D123_S2025_<lineage>`

- 입력 `[P,M]`, output `M+F([P,M])`, L1 recon만.
- 원래 성공 config의 AdamW/weight decay/augmentation/intensity/precision 정책을 사용한다.
- U LR=1e-4, warmup100+cosine, 50K, 실효 batch48를 기본 의미로 고정한다.
- 같은 센서의 L000/L1E4에 사용할 U-Net 초기 state를 **학습 전 저장**한다.
- Source split은 §3에서 확정한다. 새 holdout 분할로 학습한 모델은 `_SOURCEHOLDOUT`로 표시한다.

### 2.5 B02 — DON-N2 bootstrap

`B02_<sensor>_DONN2_W112_D123_R200_S2025_<lineage>`

**과거 N2의 resolved config가 있으면 그것을 우선한다.** 과거 초기 A1 의존성이 명시되어 있으면 해당 A1도 준비 학습한 뒤 사용한다. 현재 문서에서 그 의존성이 실제 실행에 있었는지 추정해 채우지 않는다.

과거 초기 자산이 없으면 이번 표준 신규 계보 `_BOOTSCRATCH`를 사용한다:

- C-channel용 aligner와 W112-D123 U-Net을 새로 초기화. Aligner 최종 2D head는 0 초기화.
- 50K, Rec와 SG relative-offset auxiliary, native/corrupted update 1:1.
- even update: `c0=A(P,M)`, `Yhat=Phi(P,M;c0)`, recon.
- odd update: `P_eps=W(P,eps)`, `c_eps=A(P_eps,M)`, `Yhat=M+F([W(P_eps,c_eps),M])`.
- odd update의 auxiliary: `mean|c_eps+eps-sg(A(P,M))|`.
- auxiliary 계수: `.01 * min(1,(t+1)/5000)`. 두 번 sampling은 과거 N2형 사전학습 경로이며, 후단 L1E4와 구분한다.
- MS/base/GT는 그대로. 원본 offset의 pseudo-label이나 FR 중앙값을 GT로 쓰지 않는다.
- 신규 `_BOOTSCRATCH` 기본 LR은 A/U 모두 1e-4. 성공 N2 config와 다르면 그 차이를 명시한다.
- **donor는 exact50K의 aligner**다. 초기 best_raw로 대체하지 않는다.

이 새 donor가 기존 WV3의 exact donor와 같다는 주장은 금지한다. 반응/학습 곡선을 기록한다. q가 원하는 수준이 아니라는 이유로 좋은 seed가 나올 때까지 재시작하지 않는다. 구현 오류가 아니면 불리한 donor 결과도 그대로 남긴다.

### 2.6 B03/B04 — L000/L1E4 및 확인

`B03_<sensor>_{L000,L1E4}_W112_D123_S2025_<donor_id>`  
`B04_<sensor>_L1E4_W112_D123_S1234_<donor_id>`

- DON-N2에서 A 전체만 복사하고 U/optimizer/scheduler는 가져오지 않는다.
- L000/L1E4는 같은 seed의 동일 U 초기값·data 순서·augmentation을 사용한다.
- A LR=1e-5 / U LR=1e-4, 50K, warmup100+cosine, 실효 batch48.
- L000: native recon만. L1E4: native recon + odd update 1e-4·Off.
- N2와 달리 jittered PAN은 **U-Net에 넣지 않는다**.
- Train-time augmentation이 있는 경우 실제 증강된 입력에서 q와 recon을 계산한다. teacher output을 단순 회전해 cache 재사용하지 않는다.
- best_raw, exact25K, exact50K, checkpoint metrics를 보존한다. exact50K와 best_raw 모두 진단한다.

### 2.7 설정의 빈칸을 묵시적으로 채우지 않기

우선순위는 `검증된 성공 run resolved config → 해당 trainer의 명시적 config → 이번 새 bootstrap 설정`이다.

모든 과거 config가 없는 경우 새 bootstrap의 optimizer 선택은 AdamW `(betas=.9,.999, eps=1e-8, wd=.01)`, LR floor=0, FP32, 새 gradient clipping 없음으로 **명시적 신규 기본값**을 등록한다. 이것은 과거 run의 실제 값이라는 진술이 아니다. 공통 flip/rotation 및 norm/bias decay exclusion의 실제 설정도 manifest에 적는다. 기존 recipe가 있으면 새 기본값으로 덮지 않는다.

---

<a id="datasets"></a>

## 3. 여러 데이터셋을 전부 포함하는 방법

### 3.1 Coverage 계약

| 데이터셋 | MS 채널 | 학습 준비 | 필수 분석 |
|---|---:|---|---|
| **WV3** | 8 | 기존 자산 우선; 없으면 B01–B04 | native64 + RR + FR, 모든 핵심 case |
| **QB** | 4 | 해당 센서 모델·donor 없으면 B01–B04 | native64 + RR + FR, 동일한 핵심 case |
| **GF2** | 4 | 해당 센서 모델·donor 없으면 B01–B04 | native64 + RR + FR, 동일한 핵심 case |
| **WV2** | 8 | 기존 프로토콜상 학습셋 없음 | WV3 모델의 zero-shot RR/FR + sample 특성; test 학습 금지 |

이 데이터 구성은 [R6 §0]와 [R7 §4.1]의 자료에 따른다. 실행 시 실제 manifest로 band order·평가 장면 수·repaired/msfix 버전을 확인한다. 문서에 적힌 과거 파일 수와 현재 데이터 버전이 같다고 가정하지 않는다.

WV2에서 새로 학습 가능한 **별도 원본 training 자료가 실제로 존재**하면 source-disjoint 분할·권한·계보를 명시한 추가 sensor-trained cohort를 만들 수 있다. 공식 WV2 test의 일부를 떼어 학습하는 방법은 이 계획에 포함하지 않는다.

### 3.2 서버 분담은 센서별 완결 묶음

가용 서버를 확인한 뒤 다음 논리 lane을 배정한다. 실제 s2/s3 사용 가능 여부를 가정하지 않는다.

| Lane | 권고 작업 | 서버 내부 필수 대조 |
|---|---|---|
| 8-band lane | WV3, 같은 모델로 WV2 zero-shot | P0/L000/L1E4 재현, 2개 이상 L1E4 seed |
| 4-band lane A | QB 모델 준비·전체 분석 | 해당 QB의 P0/L000/L1E4 |
| 4-band lane B | GF2 모델 준비·전체 분석 | 해당 GF2의 P0/L000/L1E4 |

서버가 하나면 같은 큐를 순차 실행한다. 같은 weight를 다른 서버에서 평가한 것은 수치 재현이지 새 학습 반복이 아니다. 서로 다른 센서를 서로 다른 서버에 맡기면 hardware/implementation confound가 생길 수 있으므로 **각 lane의 reference packet을 coordinator에서 재실행**한다. 분석 관계는 센서 내부 비교로 얻고 절대 HQNR을 서버끼리 경쟁시키지 않는다.

### 3.3 자료 분할

**기존 자산의 replication cohort와 새 source-holdout cohort를 구분한다.**

- 기존 EQREC4 A/B/C/D 및 sample cache는 hash가 일치할 때만 복제한다. 모델이 이미 본 train patch에서의 분석은 `seen_in_pretraining=true`로 표시한다.
- 새 모델을 준비하는 센서는 model training 전에 원본 scene/strip 단위로 `FIT70 / CAL10 / DISC10 / CONF10`을 나눈다. 비율은 source group 경계 때문에 근사치다.
- 새 donor/P0/L000/L1E4의 학습은 FIT만 사용한다. CAL은 threshold·descriptor scale·estimator calibration, DISC는 기전 탐색, CONF는 잠근 확인용이다.
- 기존 donor가 새 CONF를 이미 봤다면 strict holdout이 아니다. `donor_seen_confirm`으로 기록하거나 새 split-donor를 학습한다.
- 기존 full-train 모델과 신규 FIT70 모델의 절대 수치를 하나의 paired 표로 섞지 않는다. 새 cohort는 모든 대응 model에 같은 split을 사용한다.
- source ID가 없으면 좌표·원파일·중복/인접 patch mapping을 복원한다. 안 되면 `source_group_unknown`으로 남기고 patch 수를 독립 표본 수로 간주하지 않는다. index32 block을 실제 scene으로 바꾸어 쓰지 않는다.

### 3.4 표본량과 선택 방식

**Cheap descriptor:** WV3/QB/GF2의 사용 가능한 원 train patch 전체를 streaming scan. WV2는 RR/FR 모든 장면과 사전 고정 tile을 scan한다.

**모델 atlas:** 센서별 최대 4,096개의 고유 native64 patch, 가능한 한 source를 균등 포함한다. 목표 CAL512 / DISC1536 / CONF1024 / FIT1024. source-disjoint 분할 비율과 가용량이 우선이며 부족한 cell을 중복 복제로 채우지 않는다.

**상세 intervention:** primary L1E4에서 네 집단당 최대32개 DISC + 32개 CONF, 총 최대256개/센서. source 균형을 맞추며, CONF sample은 case 결과를 보기 전에 ID를 고정한다. 같은 sample들을 모든 개입에 사용한다.

**시각 패널:** 네 집단당 무작위8, descriptor 중심4, 예외4개를 별도로 표시. 보기 좋은 patch만 선별하지 않는다. 예외는 `exploratory`로 표시하고 확인 통계 표본을 대체하지 않는다.

**RR/FR:** 실제 공식 평가 manifest 전체(기존 각20장)를 사용한다. model diagnostic은 장면당 최대16개의 겹치지 않는 64² tile을 deterministic하게 뽑는다. 이 tile은 native train64와 다른 split/scale 태그다. 작은 tile에서 HQNR을 계산해 공식 scene HQNR로 간주하지 않는다.

### 3.5 누락 센서 처리

`coverage.csv`에 각 센서의 `data_ready / models_ready / atlas_done / intervention_done / confirm_done`을 따로 기록한다. 모델 학습 중에는 descriptor scan을 진행할 수 있다. 최종 보고서에 센서가 누락되면 이유와 남은 training dependency를 기재하고 **전 센서 완료**라고 표시하지 않는다.

---

<a id="definitions"></a>

## 4. 측정량·네 집단·통계의 공통 정의

### 4.1 e와 q를 혼합하지 않는다

\[
e_i=\operatorname{mean}_{c,x}|\widehat Y_{0,i}-Y_i|,
\quad r_{i,k}=A(\mathcal W(P_i,\epsilon_k),M_i)+\epsilon_k-A(P_i,M_i).
\]
\[
q_{A,i}=\frac1{2K_A}\sum_k\|r_{i,k}\|_1,\qquad
EPE_{A,i}=\frac1{K_A}\sum_k\|r_{i,k}\|_2.
\]

`e`는 영상 intensity 단위, `q`는 HR pixel 단위다. `e/q`나 `λq`를 Teacher 신뢰도로 쓰지 않는다. 모든 평균 전에 sample·band·좌표 reduction을 명시한다.

| Probe | 반경(HR px) | 방향 | 사용 |
|---|---|---|---|
| A | .25 / .5 / 1 / 2 | 0/90/180/270도 | 주 q, 집단 정의 |
| B | 같은 반경 | 45/135/225/315도 | 독립 방향 확인·stress |
| S | .125 / .25 / .5 / 1 | A+B 8방향 | 동일 잔여 이동 민감도 |
| Identity | 0 | — | unit test, q 평균에 포함하지 않음 |

`epsilon=(r*sinθ,r*cosθ)`이다. A/B는 training disk와 다른 고정 probe 분포다. 축과 대각의 component-L1 기하가 다르므로 raw q_A=q_B를 기대하지 않는다.

추가 진단값:

- `q_const(bank)=mean ||epsilon||_1/2`, 같은 bank의 무반응 기준.
- `q_rel=q/q_const`, `EPE/r`의 반경별 값. 기존 training loss 변경이 아니다.
- `c_eps-c0 = B_resp epsilon + b_fit + residual`의 **전체 2×2 행렬**, 부호·비대각·적합 오차.
- 반경별 q, q_A/q_B rank 일치, 반복 실행 noise floor.

### 4.2 집단 label

CAL의 같은 `dataset × model_hash × scale`에서 `median(e)`와 `median(q_A)`를 정해 DISC/CONF에 고정 적용한다.

| Label | 조건 | 이름 |
|---|---|---|
| EdCd | e≤median e, q≤median q | 복원 오차 낮음·consistency 오차 낮음 |
| EdCu | e≤median e, q>median q | 복원 오차 낮음·consistency 오차 높음 |
| EuCd | e>median e, q≤median q | **복원 오차 높음·consistency 오차 낮음** |
| EuCu | e>median e, q>median q | 복원 오차 높음·consistency 오차 높음 |

신규 atlas의 primary group e는 **native full64 L1**이다. 개입 효과는 별도의 공통 ROI에서 계산하며, 같은 ROI의 e로 집단을 재정의한 sensitivity 표도 낸다. 기존 EQREC4가 다른 e/ROI를 사용했다면 `_legacy_group`으로 재현하고 신규 label과 섞지 않는다.

Primary label은 연속 e/q의 시각적 요약일 뿐이다. `q↓=좋은 Teacher`라는 별도 label을 만들지 않는다. 임계값 근처 ±CAL IQR의 10% 범위는 `boundary_near`로 표시하고, 제외한 보조 분석도 함께 낸다. 이 비율은 신규 sensitivity 설정이며 최적값이 아니다.

CAL의 q IQR이 재실행 q noise의 10배보다 작거나 q 값이 사실상 동일하면 `q_unresolved`로 표시한다. 이 10배는 신규 수치 식별성 검사이지 통계 유의성 기준이 아니다. 의미 없는 median split으로 네 집단을 강제로 만들지 않으며, e/descriptor 및 frozen intervention은 계속 분석한다.

센서별 label로 상대 집단을 비교하는 것과 WV3 threshold의 zero-shot 적용은 별도 분석이다. WV2에서 local median으로 만든 label은 **test-descriptive**이며 학습 policy에 사용하지 않는다. WV3에서 고정한 threshold를 적용한 결과도 병기한다.

### 4.3 세 가지 오차를 분리

| 종류 | 의미 | GT 필요 |
|---|---|---|
| Relative-offset q/EPE | 알려진 추가 이동에 대한 예측 변화 오차 | HRMS GT 불필요 |
| Absolute synthetic offset error | 생성 기하가 알려진 합성 pair의 total correction 오차 | 생성한 offset GT |
| Native geometric proxy | 실제 PAN–MS/GT 구조의 잔여 위치 차이 추정 | RR은 GT 사용 가능, FR은 관측 MS만 |

`known_inverse` 출력의 L1이 낮다는 것은 sampler 대조다. **CNN absolute offset error로 표기하지 않는다.**

### 4.4 ROI와 평가

- Native64 원 L1 full64와 **공통 intervention ROI L1**을 모두 보존한다.
- 기본 paired ROI: 64→margin16, 256→margin32, 512→margin96. CAL의 sampling/filter support 검사 후 manifest로 잠근다.
- 같은 sample의 before/after는 반드시 같은 ROI·band·intensity·export clip 규약을 사용한다.
- 큰 learned correction을 clamp하거나 불리한 sample만 제외하지 않는다. `invalid_support`를 별도 보고하고 모든 paired arm의 공동 유효 set에서 계산한다.
- ROI가 U-Net의 전체 receptive-field 경계 효과를 제거한다는 주장은 금지한다. S30 parent-context 검사를 수행한다.
- HQNR은 full-scene **원래 PAN 참조**를 유지한다. 장면별 `(1-Dλ_i)(1-Ds_i)` 계산 후 평균한다.
- stress 출력도 원래 P/MS와 같은 ROI로 평가한다. jittered reference/self-aligned 점수로 바꾸어 native와 빼지 않는다.
- FR에는 e가 없다. `1-HQNR`을 L1로 치환해 네 집단을 만들지 않는다.
- WV3 기존 선택은 raw HQNR→FR fSCC, tie 1e-4. 0.0031은 기존 WV3의 경험적 효과 판정선이며 다른 센서나 q/L1에 전용하지 않는다.
- 신규 센서의 1 seed에서 방법의 통계적 우월성을 선언하지 않는다. sample mechanism endpoint와 방법 성능 endpoint를 구분한다.

### 4.5 통계 기본 규칙

1. 연속값 관계를 먼저 본다. Spearman, 분위수별 곡선, 중간 구간의 비선형성을 보고 median 표만으로 결론 내리지 않는다.
2. 계층은 `sensor → original source → patch → probe → checkpoint`다. patch/probe/seed를 독립 scene 수로 세지 않는다.
3. 가능한 source-group bootstrap 2,000회. source가 불명확하면 CI를 확정 근거로 쓰지 않고 descriptive로 낮춘다.
4. 센서 내부 효과와 source 내부 효과를 먼저 보고, 센서별 동일 비중 요약을 추가한다. 모든 센서 patch를 한 상관계수로 섞지 않는다.
5. 확인 endpoint를 먼저 고정한다. 여러 descriptor 탐색의 p값은 exploratory로 표시하고 CONF의 반복 방향·효과 크기를 우선한다.
6. q–gain에서 `gain=e_zero-e_learned`의 공통 항을 명시한다. `e_learned`와 `e_zero`를 함께 입력하면 gain이 항등적으로 정해지므로 그것을 예측 성공으로 부르지 않는다.
7. q가 사용된 label(예: EuCd)을 q로 예측한 정확도는 q 추가 가치 검증이 아니다. C50의 독립 outcome을 사용한다.
8. 수치 재실행 noise와 source 간 통계 변동, 방법 간 seed 변동은 서로 다른 기준이다.

---

<a id="cases"></a>

## 5. Case 목록: 조작·고정·판정이 분명한 실험

| Case | 핵심 조작/분석 | 주요 outcome | 학습 |
|---|---|---|---|
| A00 | 자산·data·unit·기존 표 재현 | provenance/지원 가능성 | 없음 |
| B01–B04 | 누락 baseline/donor/sensor 모델 준비 | 정확한 local model family | 없을 때 50K |
| D10 | 전체 sample의 e/q·label·입력 특성 | atlas, prevalence, q–e 관계 | 없음 |
| D11 | 대비·기본 복원 난이도·band/edge 분해 | 정규화/조건부 관계 | 없음 |
| D12 | 구조/spectral descriptor·blind sample gallery | EuCd의 재현 가능한 특성 | 없음 |
| I20 | 같은 크기의 잔여 PAN 이동 강제 | 출력 민감도와 오차 증가 | 없음 |
| I21 | Aligner가 보는 구조/대비만 조작 | q·signed response 변화 | 없음 |
| I22 | 동일 기하에서 spectral 조합 변화 | relative/absolute offset, spectral 기여 | 없음 |
| I23 | Native geometry·correction landscape | c_geo 대 c_rec, 보정의 효과 | 없음 |
| I24 | single/sequential warp·filter 대조 | 위치 효과/보간 효과 분해 | 없음 |
| I25 | 두 loss gradient·작은 가중치 개입 | 실제 목적 충돌·국소 trade-off | snapshot 1-step |
| S30 | 동일 scene의 crop/context·seed/ckpt | 데이터 특성 대 모델 상태 | 없음 |
| X40 | 전 센서 반복·zero-shot/within 구분 | 센서 일반화 범위 | baseline 외 없음 |
| C50 | descriptor-only에 q 추가 | 독립 outcome의 추가 예측력 | 작은 통계 모델 |
| R60 | 확인된 기전에 맞춘 대응 vs 통제 | 집단×대응 interaction | 조건부 1K pilot |

**D10–X40는 전부 실행 대상이다.** 비용이 큰 세부는 아래의 고정 subset에서 실행한다. 시간 때문에 생략했다면 `pending`으로 남기고 `completed_no_effect`로 쓰지 않는다. R60만 과학적 진입 조건을 둔다.

---

## 6. A00 — 시작 검사와 기존 결과의 정확한 연결

### 검사

- 실제 서버 ID를 정규화하고 `server_id`, GPU, driver, torch, code commit을 기록한다. 과거 s3 이식의 경로·헤더 하드코딩 문제를 반복하지 않는다. [R2 §8]
- 기존 EQREC4 원시 `sample_metrics`, `interventions`, `probe_records`, `D30 landscape`, `D50`을 먼저 찾아 summary와 연결한다. 파일 존재를 확인하지 않고 완료라고 표시하지 않는다.
- report 항목의 sample set/ROI/ckpt를 확인한다. s1과 s3의 다른 H3b 판정·R arm 이름 차이는 별도 discrepancy log에 남긴다.
- Native sample의 M upsampling, GT crop, band order, radiometric scaling, loss reduction을 재현한다.
- c=0·known shift·sampling sign·dy/dx 순서, zero affine transform, identity probe를 검사한다.
- 반복 실행으로 q/e 수치 noise를 측정한다. 기존 q_A–q_B label 일치와 숫자를 새 source와 혼용하지 않는다.
- 기존 L1E4 primary의 official metric을 재현한다. 구현 오류와 하드웨어 허용 오차는 분리하고 hard threshold는 CAL에서 기록한다.
- G00가 못 찾은 baseline은 B queue에 넣는다. 부족한 자산을 제외한 센서만으로 모든 센서 결론을 내리지 않는다.

### 산출물

`asset_registry.json`, `data_registry.json`, `bootstrap_queue.json`, `reproduction.csv`, `discrepancies.md`, `coverage.csv`.

---

<a id="atlas"></a>

## 7. D10/D11 — 네 집단 atlas와 척도·난이도의 분해

### D10-A: 전체 atlas

모든 유효 model/sample에서 다음을 연결한다.

`e_full, e_roi, q_A, q_B, EPE_A/B, c0_y/x, correction_norm, quadrant, source_id, scene_id, sensor, scale, ckpt_hash`.

P0에는 q=0을 넣지 않는다. `q=NA`이며 데이터 특성과 출력 오류의 대조만 수행한다.

집단 수, CAL threshold, boundary-near 비율, source 구성, membership의 seed별 전이 행렬을 보고한다. q와 e의 상관이 소수 source 또는 큰 반경 r=2에 의해 주도되는지 leave-one-source-out 및 반경별 분석을 한다.

### D11-A: 절대 L1과 대비 정규화

Band c의 robust contrast를 `s_ic=P95(Y_ic)-P5(Y_ic)`로 정의한다. GT 기반 diagnostic이며 추론 입력 cue가 아니다. Input descriptor에는 같은 수식을 M/P에 별도로 계산한다.

\[
e_{contrast,i}=\frac1C\sum_c
\frac{\operatorname{mean}|\widehat Y_{ic}-Y_{ic}|}{\max(s_{ic},s_{floor,c})}.
\]

`s_floor,c`는 CAL의 양의 contrast P5와 실제 quantization step 중 큰 값. 둘 다 모르면 저장 scale의 1e-6을 **신규 fallback**으로 쓰고 flagged 비율을 낸다. Normalized error가 큰 저대비 patch를 유리하게 제거하지 않는다.

보고 순서:

1. raw e–q 상관.
2. contrast-normalized e–q.
3. 같은 source·비슷한 contrast의 matched sample에서 raw e–q.
4. intensity clipping/낮은 신호/포화 집단을 분리한 sensitivity.

**해석:** 정규화만으로 차이가 작아져도 원인이 확정된 것은 아니다. 분모와 group 구성이 바뀐다. matched raw 결과와 함께 판단한다.

### D11-B: 원래 복원해야 할 내용의 양

\[
e_{base,i}=\operatorname{mean}|M_i-Y_i|,\quad
g_{restore,i}=e_{base,i}-e_i.
\]

`relative_e=e/max(e_base, floor)`는 보조값으로만 쓰고 raw e/base/gain을 함께 낸다. `e_base≈0` sample은 별도 집계한다.

확인할 질문:

- EuCd는 e가 크지만 **MS base에서 더 많은 오류를 제거한** 집단인가?
- e_base가 같은 sample 안에서도 EuCd의 e가 높은가?
- 모델 출력을 보기 전의 `e_base`, GT high-frequency energy로 q–e 관계가 얼마나 설명되는가?

`e_base-e`와 e 사이의 상관은 공유 항이 있으므로 인과적 성공이라고 하지 않는다.

### D11-C: 오류가 어디에 집중되는가

- Band별 e, 최대 band error, band 간 error IQR.
- GT edge mask는 CAL에서 고정한 sensor별 threshold로 생성한다. 같은 sample의 모든 prediction에 같은 mask를 사용한다.
- edge area fraction, **edge pixel당 L1**, non-edge pixel당 L1, 전체 L1 중 edge 기여율을 구분한다.
- LP/HF 오차: source가 고정된 5×5 Gaussian σ=1, reflect kernel을 사용한 `L(I)`와 `H(I)=I-L(I)`의 band별 L1. 이는 진단 분해이며 input LPAN 재도입이 아니다.
- 강한 고립 edge에 대해 GT/출력의 동일 band intensity profile을 비교한다. edge 중심 오차·10–90% 폭·overshoot·double edge 여부를 기록한다.
- 자연 영상의 단일 경계 폭을 센서 PSF 참값으로 부르지 않는다.

**주 비교:** EuCd↔EuCu(복원 어려움 비슷)와 EuCd↔EdCd(q 비슷)를 별도로 한다. Match 후 탈락률과 공통 support를 보고한다.

---

## 8. D12 — 실제 sample은 어떤 특성을 가지는가?

### 8.1 사전 고정 descriptor bank

| 축 | 기본 descriptor | raw/normalized 및 주의 |
|---|---|---|
| 대비 | PAN/MS band IQR·P95−P5, standard deviation | 원래 intensity와 z-score 후 둘 다 |
| Edge | Scharr magnitude mean/P90, edge fraction | Scharr 계수·kernel norm·padding hash 고정 |
| 2D 구조 | `J=mean(g g^T)`의 고유값 λ1≥λ2, λ2/(λ1+floor) | 큰 gradient 하나와 양축 관측 가능성을 분리; 고유값은 새 모델의 정확도 이론값이 아님 |
| 방향성 | gradient orientation histogram·entropy·dominant orientation | magnitude-weighted, 8bin; low-energy는 NA |
| 반복 패턴 | centered autocorrelation의 비원점 peak/peak gap | lag2–16 HR px 범위, 동일 window·DC 제거 |
| 주파수 | 같은 window의 low/mid/high energy fraction | radial bins [0,.125),[.125,.25),[.25,.5] cycles/pixel; corner 영역 처리 명시 |
| 복원 과제 | e_base, GT HF energy, residual Y−M의 edge density | GT 기반 분석 전용 |
| Cross-modal | 해상도/phase를 맞춘 PAN–MS gradient similarity | 정합 전/후 분리. PAN raw와 매우 흐린 M을 바로 비교하지 않음 |
| Spectral | band 간 구조 유사도·edge disagreement, band별 e | band 수가 다르면 mean/분위수로 요약; band 의미는 metadata에서만 |
| 관측 이상 | saturation fraction, zero/nodata, low-signal fraction | 센서별 scaling/valid mask 확인 |

Descriptor는 primary raw P/M에서 계산하고, correction을 적용한 descriptor를 별도 접미사로 보관한다. q·e를 이용해 descriptor를 고른 뒤 같은 CONF에서 검증하지 않는다.

### 8.2 Blind sample gallery

각 sensor/집단에 고정 패널을 만든다.

`PAN | MS(RGB/개별 band) | GT | prediction | band error | edge error | learned–zero difference`.

- Sensor metadata로 RGB band를 결정한다. 없으면 `band index`로 표기하고 임의 RGB 의미를 부여하지 않는다.
- 같은 sample의 GT/prediction은 같은 stretch·crop을 사용한다. 별도 contrast stretch로 좋은 결과처럼 만들지 않는다.
- 1차 주석자는 q/e/집단과 method 이름을 숨긴 상태로 토지피복·경계·반복 패턴을 기술한다.
- 주석 예: 평탄한 수면/나지, 연속 수목 texture, 경작지·도로 경계, 밀집 인공구조, 그림자 경계, 혼합/불명확. 이 label은 영상 확인 후 기록하며 Scharr에서 자동 추론하지 않는다.
- 최소 일부는 두 사람이 독립 주석하고 불일치·unknown을 보존한다. 한 명이면 단독 탐색 주석으로 표시한다.
- 숫자 차이와 시각적 해석이 맞지 않는 sample도 최종 보고서에 넣는다.

### 8.3 D12 판정

`EuCd = 건물`처럼 하나의 의미 label로 환원하지 않는다. 예를 들어 **“같은 대비에서 방향이 다양한 경계·높은 HF가 더 많고, band별 edge error가 집중된다”**처럼 재현 가능한 특성으로 기술한다.

특성 차이는 standardized effect size·분포·source별 방향을 제시한다. 단순히 가장 큰 평균 descriptor를 원인으로 선언하지 않는다.

---

### 8.4 Matched 비교의 실행 규칙

- Match는 같은 sensor·checkpoint·scale·가능하면 같은 source 안에서 수행한다. 다른 source match는 별도 sensitivity로 표시한다.
- **EuCd↔EuCu**: reconstruction difficulty를 맞추기 위해 CAL scale로 표준화한 `e_base, contrast`를 우선 사용하고, native e까지 맞춘 보조 비교를 추가한다. q를 match 변수로 쓰지 않는다.
- **EuCd↔EdCd**: q를 비슷하게 맞춘 뒤 contrast/구조량의 차이와 e_base 차이를 본다. e를 맞춰 group 정의를 지워 버리지 않는다.
- Match에는 1:1 nearest neighbor without replacement를 사용한다. 연속 핵심 변수의 caliper는 CAL 표준편차의 .2로 사전 고정하고, .1/.3은 sensitivity로만 보고한다. 이 값은 신규 설계값이다.
- 먼저 contrast/e_base만 통제하고, 다음에 구조 descriptor를 추가한다. 처음부터 모든 가설 변수를 제거한 뒤 “특성 차이가 없다”고 결론 내리지 않는다.
- Match 후 covariate balance, unmatched 비율, source 수를 공개한다. 공통 support가 없으면 다른 집단을 억지로 만든 것이 아니라 `comparison_not_supported`로 남긴다.
- H1의 outcome이 e일 때 e를 공변량으로 사용하지 않는다. 위의 e-matching은 PAN 민감도/개입 이득의 비교에만 해당한다.

---

<a id="residual"></a>

## 9. I20 — 같은 잔여 PAN 이동을 주면 어느 sample이 더 민감한가?

### 목적

q는 `얼마나 변위가 남는가`를 측정한다. 여기서는 **같은 변위가 남았을 때 출력이 얼마나 바뀌는가**를 측정한다. 두 양을 섞지 않는다.

### I20-A: direct residual injection

Frozen native c0를 먼저 계산한다. 이후 A를 다시 호출하지 않고:

\[
\widehat Y_i(\delta)=\Phi(P_i,M_i;c_{0,i}+\delta).
\]

각 δ는 Probe S의 고정 벡터다. **원래 PAN에서 매번 한 번만 sampling**한다. 이미 보정된 PAN을 다시 shift하지 않는다.

\[
s_{out,i}(r)=E_{\|\delta\|=r}\operatorname{mean}|\widehat Y_i(\delta)-\widehat Y_i(0)|,
\]
\[
d_{GT,i}(\delta)=L_1(\widehat Y_i(\delta),Y_i)-L_1(\widehat Y_i(0),Y_i).
\]

- `s_out`은 출력 변화, `d_GT`는 signed 품질 변화다. 출력 변화=품질 악화라고 가정하지 않는다.
- 양·음/축·대각 방향을 모두 저장한다. 일부 방향은 native 오류를 줄일 수 있으므로 음수를 0으로 자르지 않는다.
- FR은 s_out 및 동일 참조의 ΔHQNR/fSCC만 산출한다. GT L1을 만들지 않는다.

### I20-B: 위치 Jacobian 근사

\[
J_x(h)=\frac{\widehat Y(+h e_x)-\widehat Y(-h e_x)}{2h},\quad
J_y(h)=\frac{\widehat Y(+h e_y)-\widehat Y(-h e_y)}{2h}.
\]

h=.125와 .25에서 안정성을 확인한다. 평균 L1·band·edge별 민감도, `J^T J/(CHW)`의 두 고유값과 방향을 기록한다. 이는 **현재 U-Net+sampling의 국소 민감도**이지 물리적 영상 noise covariance가 아니다.

**주 판정:** EuCd–EuCu의 s_out 차이가 source·contrast·e_base를 맞춰도 남는가? 같은 residual에서도 EuCd가 더 민감하면 H2의 반대 방향을 설명할 직접적인 기전 후보가 된다.

### I20-C: residual 크기와 방향을 분리

Probe B에서 얻은 실제 residual `r_i`를 사용해 `c0+r_i` single-warp 출력을 만들고, 같은 norm을 가진 방향 회전 residual도 평가한다. 작은 q라도 residual이 민감한 방향에 놓였는지 확인한다.

수식적 근사 `ΔY≈J r`의 예측 오차를 반경별로 보고한다. .125/.25에서 맞아도 1/2px 전체에 선형식을 외삽하지 않는다. `s_out/q`처럼 작은 분모의 비율 하나로 결론 내리지 않는다.

---

## 10. I21 — Aligner는 어떤 구조 단서 때문에 q가 낮은가?

### 목적과 고정 경로

A 전용 입력 view만 바꾸고, **U-Net에 들어가는 원래 PAN·MS·native correction은 별도 캐시에 고정**한다. 이 실험의 목적은 q를 만드는 단서이며, 복원 효과를 동시에 주장하지 않는다.

| Arm | A가 보는 입력 변화 | 확인할 것 |
|---|---|---|
| V0 | 원래 view | reference |
| V1 | PAN·MS에 공통 positive affine contrast ×.5/×1.5, mean 보존, clipping 없음 | z-score의 대비 불변성과 작은 variance/epsilon 영향 |
| V2 | PAN view에 isotropic Gaussian σ=.5/1만 적용 | 고주파 단서를 줄이면 q가 어떻게 바뀌는가 |
| V3 | PAN view를 dominant structure 방향/수직으로 anisotropic smoothing | 방향성 단서와 축별 response의 관계 |
| V4 | V2 후 mean·variance를 원 view와 맞춤 | blur 효과와 단순 대비 감소를 구분 |
| V5 | 해상도를 맞춘 PAN/MS 양쪽에 같은 σ=.5 smoothing | 한 modality만 변화시킨 것과 공통 bandwidth 감소의 차이 |

모든 filter는 parent에서 먼저 적용하고 aligner crop을 한다. Native와 jittered의 처리 순서를 동일하게 `I'=T(P), P'_eps=W(I',eps)`로 두며, transform과 warp 순서 차이는 별도 대조가 아니면 바꾸지 않는다.

V3의 orientation은 **원 입력 descriptor**에서 구하고 결과 q를 보고 정하지 않는다. Affine 시 clipping이 발생하면 정확한 대비 대조가 아니므로 clipped fraction을 기록하고 해당 해석을 제한한다.

### 판정

- 대비를 복원해도 구조 완화 후 q가 악화되고, anisotropic 방향에 맞춰 response가 달라지면 **구조 단서 사용**을 지지한다.
- V1만으로 q가 크게 변하면 z-score crop·variance floor·radiometric adapter를 먼저 확인한다.
- A view만 바꾸고 복원 경로를 고정했으므로 e가 같다는 것은 **구성상 보장**이다. 이를 새 robustness 성과라고 쓰지 않는다.
- 이 개입은 일부 OOD view를 포함한다. 실제 센서 변동의 빈도나 원인까지 증명하지 않는다.

---

## 11. I22 — Spectral 차이와 알려진 전체 기하의 2×N 대조

### I22-A: 구성상 정합된 합성 쌍

RR GT의 같은 raster Z를 사용한다. `M_syn=Z`, `P_syn=sum_c w_c Z_c`, 모든 w≥0·합1.

w는 사전 고정 세 종류: 전 band 균등 / 홀수 index 균등 / 짝수 index 균등. **홀수/짝수를 VIS/NIR라고 부르지 않는다.** 실제 band 의미는 metadata에 연결한다.

각 w에 대해 δ=0 및 Probe A/B의 {.5,1,2}px subset을 적용한다.

\[
E_{abs}=\|A(W(P_{syn},\delta),M_{syn})+\delta\|_2,
\]
\[
r_{rel}=A(W(P_{syn},\delta),M_{syn})+\delta-A(P_{syn},M_{syn}).
\]

**반드시 CNN 예측의 absolute error와 relative error를 둘 다 출력한다.** Ground-truth inverse를 sampler에 넣어 잘 복구되는 결과는 별도의 positive control이다.

2개 축:

| 변경 | 기하 | Spectral appearance |
|---|---|---|
| w 고정, δ 변경 | known 변화 | 고정 |
| δ 고정, w 변경 | 동일 | 변경 |
| w와 δ 모두 변경 | known 변화 | 변경 |

생성 쌍은 실제 sensor PSF/대역응답과 다르고 M_syn은 실제 upsampled LRMS보다 선명하다. 따라서 **calibration/OOD 기전 검사**로만 보고 native 모델 점수와 합치지 않는다. 같은 symmetric blur를 모든 Z band에 적용한 버전도 함께 검사해 bandwidth 영향을 확인한다.

### I22-B: native spectral mismatch descriptor

- 실제 sensor의 PAN을 MS 해상도로 내리는 **검증된 degradation/phase**를 사용한다.
- FIT/CAL에서만 비음수 band mixture w(합1)와 offset을 맞춰 LR PAN을 설명한다. 같은 w를 DISC/CONF에 고정한다.
- Native 구조 정합 전·후의 mixture residual, band별 gradient disagreement를 기록한다.
- 이 residual은 sensor response curve나 재질 GT가 아니다. geometry·PSF·radiometry 오차도 섞일 수 있다.
- Metadata가 허용하면 band family별 e를 기술하되 센서마다 같은 물리 band라고 자동 대응시키지 않는다.

### I22-C: spectral 가설을 지지하는 조합

1. EuCd의 특정 band/구조에서 reconstruction error가 집중되고,
2. contrast·구조량을 맞춰도 native spectral mismatch가 높고,
3. 생성 기하 고정 대조에서 spectral appearance만으로 A의 c0/absolute bias가 달라지는가?

세 결과가 함께 있어도 실제 센서의 모든 misalignment가 spectral 때문이라고 쓰지 않는다. **현재 모델이 spectral appearance에 의존하며 그 의존성이 해당 오류 구조와 관련된다**는 범위로 제한한다.

---

<a id="landscape"></a>

## 12. I23 — Native 위치와 reconstruction이 선호하는 위치는 같은가?

### I23-A: 독립 native 정합 proxy

- Primary: 기존 Scharr–MAD–ZNCC + subpixel 추정기.
- Secondary: **독립적인 위치값을 내는** 검증된 phase-correlation 또는 별도 census matcher. Primary 결과의 합격 여부만 검사하는 census gate는 독립 추정기가 아니다.
- 각 센서/scale에서 identity·known translation·spectrum-only zero geometry·blur control로 sign/unit/bias를 먼저 측정한다.
- PAN/MS의 bandwidth·phase를 맞춘다. RR은 GT↔PAN, FR은 관측 MS↔PAN을 비교한다.
- before/learned/zero/wrong-sign에 동일 설정을 쓴다. valid texture·peak margin·boundary hit·두 estimator 불일치를 기록한다.
- 두 방법 모두 신뢰할 수 있는 sample이 0이면 그대로 `n_both_accepted=0`으로 보고한다. 임계값을 풀어 성공표를 만들지 않는다.
- 64²의 native offset은 정보가 부족할 수 있으므로 parent RR/FR에서 위치를 읽은 결과와 patch 추정 결과를 분리한다. Parent correction을 patch의 참값으로 자동 전용하지 않는다.

이 검사는 물리적 정합 GT가 없는 native 자료에 대한 proxy다. q와 서로 독립적인 기하 검증을 제공하려는 목적이며 단독으로 완전 정합을 선언하지 않는다.

### I23-B: 같은 U-Net에서 correction 치환

모든 PAN 후보는 원래 P에서 한 번만 sampling한다.

`I-L learned`, `I-Z zero`, `I-W wrong sign`, `I-C CAL vector median`, `I-S source-matched shuffle`, `I-G external geometry`.

\[
g_{corr,i}=e_i(I\!Z)-e_i(I\!L).
\]

I-S는 10개 고정 derangement를 사용하고 실제 vector 변화량을 기록한다. I-C는 같은 scale의 CAL에서만 산출한다. Test prediction 중앙값은 별도 `test_informed` 설명용이며 학습 baseline으로 쓰지 않는다.

**기능적 개입:** 같은 U-Net에서 correction을 제거한 효과다. **전체 방법 효과:** 별도로 학습한 P0–L1E4 대응이다. 두 표를 구분한다.

### I23-C: correction landscape — 핵심 신규 분석

각 센서에서 DISC/CONF의 사전 고정 대표 sample에 대해 candidate bank를 만든다.

1. c0와 0 주위의 `(dy,dx)∈{−1,−.5,0,.5,1}²` grid 합집합.
2. 독립 c_geo가 유효하면 그 주위의 같은 grid를 합집합에 추가한다.
3. c0·0·−c0·CAL median 자체를 반드시 포함한다.
4. 발견된 objective 최소점 주위 ±.125/0의 3×3 정밀 grid를 **모든 objective에 동일하게 합집합**으로 적용한다.

원래 c0를 clamp하지 않는다. Candidate search의 안전 범위를 벗어난 점은 `search_boundary/invalid_support`로 기록하고 최적점 존재를 강제하지 않는다.

같은 후보 c마다:

- raw/full·paired ROI reconstruction L1
- band별 L1, LP/HF L1, edge 중심/폭
- independent geometry objectives
- PAN sampling의 gradient/주파수 변화

를 저장한다.

\[
c_{rec}^{*}=\arg\min_{c\in\mathcal C}L_1(\Phi(P,M;c),Y),
\quad c_{geo}^{*}=\arg\min_{c\in\mathcal C}E_{geo}(c).
\]

**주 outcome:** c_geo−c_rec 거리뿐 아니라 `e(c_geo)−e(c_rec)`, `e(c_geo)−e(c0)`, band별 최적점 disagreement를 함께 본다. 넓고 평평한 loss에서는 작은 vector 차이를 의미 있게 과장하지 않는다. Numeric noise·grid refinement 오차 안의 plateau는 `nonunique_optimum`으로 표시한다.

- q가 낮아도 c0에 공통 v를 더하면 q는 그대로인데 e가 바뀌는 대조를 포함한다.
- c_rec는 그 frozen U-Net과 유한 grid의 **GT-informed 진단 최적점**이다. 센서의 기하 참값도 deployable 방법도 아니다.
- 같은 GT로 c_rec를 고르고 e를 줄인 것은 구성상 유리하다. 연구 질문은 **그 최적점과 독립 기하/현재 c0의 차이가 어떤 특성·센서에서 반복되는가**다.
- CONF의 landscape는 고정된 분석 절차를 확인하는 데만 쓰며 학습 correction/gate target으로 되돌리지 않는다.

### I23-D: baseline 필요성의 추가 확인

P0·L000·L1E4의 **같은 sample**에서 landscape와 민감도를 일부 반복한다. Architecture가 같아도 가중치가 다르므로 model별 최적점 이동을 기록한다. “같은 위치가 언제나 최적”이라는 가정을 하지 않는다.

---

## 13. I24 — 위치 변화와 보간·평활화 효과를 분리

### I24-A: 두 번 sampling의 정확한 출력 분해

Probe B의 각 epsilon에 대해:

\[
Y_{seq}=M+F([W(W(P,\epsilon),c_\epsilon),M]),
\]
\[
Y_{single}=\Phi(P,M;\epsilon+c_\epsilon),\quad Y_0=\Phi(P,M;c_0).
\]

출력 벡터 차이는 정확히

\[
Y_{seq}-Y_0=(Y_{seq}-Y_{single})+(Y_{single}-Y_0)
\]

로 나뉜다. 첫 항은 같은 합성 좌표에서 sequential sampling이 만든 차이, 둘째는 single-warp 잔여 위치 변화다. **각 항의 L1 norm을 더하면 원래 출력 L1과 같아진다고 하지 않는다.** Signed GT error 차이는 망원합으로 분리할 수 있으나 각각 양수일 필요는 없다.

Known-inverse `c0−epsilon`에서도 sequential과 native single-warp를 비교한다. 축 방향 정수와 대각 subpixel은 따로 보고 `known_inverse≈0`를 모든 반경·방향에 일반화하지 않는다.

### I24-B: matched filtering control

- 같은 fractional displacement의 sampler impulse/edge 응답을 CAL에서 측정한다.
- 우선 zero-center symmetric filtering 대조를 만들어 gradient-energy ratio뿐 아니라 **edge center, edge width, frequency-band energy**를 맞춘다.
- 사전 목표: center error≤.05HRpx, gradient ratio 차≤2%, frequency bin별 energy ratio 차≤5%. 이는 신규 대조 품질 목표이지 성능 유의성 기준이 아니다.
- Gaussian σ 하나로 맞지 않으면 **unmatched_blur_control**로 보고하고 위치 효과가 입증됐다고 하지 않는다.
- 실제 sampler impulse의 전달함수에서 |H|>1 대역이 있는지도 검사한다. 그런 경우 단순한 감쇠 blur와 같다고 가정하지 않는다.

### I24-C: amplitude/phase 분해 — helper 검증 후 정밀 대조

가능하면 실제 discrete sampler impulse의 전달함수 H_c를 구해 `|H_c|`의 amplitude-only, `H_c/|H_c|`의 phase-only, full H_c를 비교한다. 영점 근방·padding·FFT periodic wrap·긴 impulse support의 처리를 명시한다.

**실행 조건:** full-H_c 구현이 같은 ROI의 실제 sampler를 재현하고, identity/impulse/edge 위치 검사를 통과해야 한다. 통과 전에는 amplitude/phase 결과를 native bicubic 대조로 쓰지 않는다. Phase-only는 반드시 이상적인 물리 shift와 같지 않으며 “sampler의 phase 성분”이라고 표기한다.

조건을 못 맞추면 이 case는 `control_not_identified`다. 이미 완료한 I24-A 및 edge 분석은 그대로 남긴다. 이 상태에서 smoothing 가능성을 배제하지 않는다.

---

## 14. I25 — Loss 크기 불일치와 gradient 충돌은 같은가?

### I25-A: gradient 측정

Frozen source에서 sample별, 같은 native forward에 대해:

\[
g_r=\nabla_\phi L_{rec},\quad
 g_o=\nabla_\phi L_{off}^{SG}.
\]

A stem/joint/head별 norm, weighted ratio `1e-4||g_o||/(||g_r||+floor)`, cosine을 저장한다. q가 큰 것을 gradient가 크다는 뜻으로, 음의 q–e 상관을 gradient 충돌로 번역하지 않는다. Near-zero gradient cosine은 NA로 둔다.

### I25-B: 되돌릴 수 있는 한 번의 parameter 개입

U-Net 고정, A snapshot을 매 arm 복구한다.

- no-op
- `−g_r`
- `−g_o`
- `−(g_r+1e-4 g_o)`

각 방향은 CAL에서 고정한 작은 step으로만 움직인다. 원래 optimizer update와 다른 **국소 개입**임을 표시한다. 단계 크기는 CAL의 native correction 변화 중앙값 약 .02HRpx를 목표로 한 번 정하고, 절반 step도 반복해 국소성 확인. 해당 방향의 gradient가 0이면 다른 step을 억지로 키우지 않는다.

추가로 같은 parameter-step norm의 대조를 제공한다. Functional step을 맞춘 분석과 현재 weighted objective의 실제 비율을 혼동하지 않는다.

개입 뒤 DISC/CONF에서 e, q_B, c0, PAN 민감도를 다시 계산한다. **q_after에서는 native와 jittered 둘 다 업데이트된 같은 A를 사용한다.** 이전 c0 target을 그대로 남겨 one-step loss 감소를 기능 향상처럼 보고하지 않는다.

SG 때문에 공통 output bias가 q의 값에서 상쇄되더라도 특정 update의 gradient가 0이라는 뜻은 아니다. [R3 §4.5]

### 판정

EuCd에서 offset 방향의 작은 개선이 recon을 체계적으로 악화시키면 목적 간 국소 trade-off의 근거다. 방향이 맞아도 전체 학습의 장기 효과를 증명하지 않는다. 개입 효과가 크기 선택에 따라 뒤집히면 `step_sensitive`로 남긴다.

---

## 15. S30 — 데이터 특성인가, crop/context·모델 상태인가?

### S30-A: 동일 scene의 nested context

실제 parent가 있는 RR256에서 64/128/256, FR512에서 64/128/256/512의 중심 정렬 crop을 만든다.

- M은 parent의 기존 bicubic 결과를 crop한다. 작은 MS를 별도로 upsample한 결과는 다른 phase/context arm으로 분리한다.
- 모든 context에서 **중앙 공통 32² ROI**의 출력 차이/GT 오차를 비교한다. 더 큰 영상을 새로 resize해 같은 입력인 것처럼 만들지 않는다.
- `local-context c`, `parent c를 고정 사용` 두 arm을 비교하여 Aligner context 효과와 U-Net context 효과를 분리한다.
- q는 각 context 입력의 A에서 계산한다. Local q와 parent q의 평균이 같다고 가정하지 않는다.
- FR은 q/context와 출력 변화만 비교하며 L1 GT를 만들지 않는다. 공식 HQNR은 전체 parent 출력에서만 계산한다.
- 원 train64에 parent가 없으면 이 검사를 수행한 것처럼 주변을 reflection으로 확장하지 않는다. `parent_unavailable`로 기록한다.

### S30-B: 단계와 seed의 membership

같은 sample의 primary/confirm seed, best_raw/exact25K/exact50K에서:

- q/e 연속 rank 변화
- model별 CAL threshold를 사용한 membership 전이
- reference model의 threshold를 고정했을 때의 전이
- 3/4 agreement 등 membership 안정성 요약

를 모두 기록한다. Model별 median 재정의는 절대 drift를 숨길 수 있으므로 raw q/e 변화가 필수다. 두 seed의 일치가 데이터 고유성의 증명은 아니지만, 한 모델 특이 현상인지 판단하는 데 도움이 된다.

### S30-C: Patch–scene 집계를 분리

가능한 parent RR scene에서 full-scene e, tile e 평균, full-scene q, tile q 평균을 각각 계산한다. q는 nonlinear network 함수이므로 `q(full)=mean q(tiles)`가 아니다.

FR의 q–HQNR 상관과 RR/native q–L1 상관은 **지표·해상도·집계 단위가 모두 다르다.**

같은 RR parent에서 q–L1의 tile 수준/scene 수준을 먼저 비교하고, 같은 FR parent에서 q의 full/tile 차이를 본다. 지표를 바꾼 효과를 단순한 집계 효과라고 단정하지 않는다.

---

<a id="crosssensor"></a>

## 16. X40 — 여러 데이터셋에서 무엇이 반복되는가?

### 필수 출력 행렬

| 관계 | WV3 in-domain | QB in-domain | GF2 in-domain | WV2 zero-shot |
|---|---|---|---|---|
| raw q–e | 측정 | 측정 | 측정 | RR/tile만, zero-shot 표시 |
| contrast/e_base 통제 q–e | 측정 | 측정 | 측정 | 위와 동일 |
| EuCd의 descriptor 차이 | 측정 | 측정 | 측정 | local-descriptive 및 WV3 threshold 둘 다 |
| 동일 residual의 출력 민감도 | 측정 | 측정 | 측정 | 측정 |
| q–correction 이득 | coupling 통제 | coupling 통제 | coupling 통제 | zero-shot coupling 통제 |
| spectral-only absolute bias | 합성 진단 | 합성 진단 | 합성 진단 | 합성 진단 |
| c_geo 대 c_rec | RR 중심 | RR 중심 | RR 중심 | RR zero-shot |
| q 추가 outcome 예측 | source holdout | source holdout | source holdout | frozen source policy 적용 |

### 비교 규칙

- 센서별 threshold·contrast floor·특성 분포·기본 e/q를 모두 공개한다. 센서별 rank/표준화 관계와 raw 관계를 병기한다.
- 4/8band는 error를 band 평균으로 집계하되, spectral coverage가 같아졌다고 주장하지 않는다.
- 0.5HRpx는 각 센서의 입력 픽셀 기준이다. 실제 GSD가 확인되면 미터 단위도 부록에 표시한다. 동일 지상 크기 비교를 임의 GSD로 생성하지 않는다.
- Data repair·MTF·sampling phase는 해당 데이터셋의 검증 버전을 유지한다. 센서 효과처럼 보이는 data-version 차이를 먼저 배제한다.
- WV3→WV2 weight는 band order/shape 호환을 확인한 뒤 사용한다. 8band라는 이유만으로 같은 spectral response라고 하지 않는다.
- QB/GF2에 센서 학습 모델이 없으면 B queue를 수행한다. 새로운 모델의 수렴이 불충분하면 그 상태를 함께 보고한다. 단일 50K recipe 결과를 해당 센서의 물리적 속성이라고 단정하지 않는다.
- 센서당 주 모델과 최소 1개의 확인 seed에서 효과 방향을 본다. WV3의 세 seed는 이미 개발에 사용됐다는 한계를 보존한다.

### 어떤 결과면 일반화했다고 할 수 있나?

**강한 근거:** 적어도 두 개의 센서 학습 조건에서 같은 특성–민감도 관계가 source 단위로 반복되고, 다른 seed에서도 유지된다. WV2 zero-shot의 같은 방향은 추가 근거이지 in-domain 반복의 대체가 아니다.

**조건부 근거:** sensor별 부호는 다르지만 특정 contrast/structure/spectral 범위에서 관계가 같다면 그 조건을 명시한다.

**미확인:** WV3만 양성, QB/GF2 자산 없음, 또는 source provenance가 불명확한데 pooled p값만 유의하면 일반화했다고 하지 않는다.

---

<a id="cuevalue"></a>

## 17. C50 — q가 필요한 cue인가, 저렴한 descriptor로 충분한가?

### 17.1 예측할 outcome

다음은 q를 정의에 포함하지 않는 독립 outcome이다.

1. I20의 `.25px 동일 잔여 이동에 대한 s_out` — primary.
2. I20의 `.5px 이동에 대한 signed d_GT 및 방향별 최대 손실` — secondary.
3. I23의 learned correction 이득 — coupling을 명시한 secondary.
4. 실제 처리 R60의 이후 holdout 이득 — 조건부 최종 검증.

EuCd label을 q로 맞히는 정확도는 primary outcome이 아니다. q_B 자체를 예측하는 것은 probe 일반화 검사일 뿐 Teacher 유용성의 증명이 아니다.

### 17.2 비교 모델

| ID | 설명 변수 | 용도 |
|---|---|---|
| C-D | 입력에서 계산한 contrast/edge/구조/spectral descriptor | 저렴한 데이터 특성 기준 |
| C-DQ | C-D + q_A 및 radius별 q | q의 추가 정보 |
| C-DE | C-D + native e/e_base 등 GT 기반 diagnostic | Error를 아는 학습 상황의 상한형 기준 |
| C-DEQ | C-DE + q_A | 기존 error 정보를 넘는지 |
| C-DQ-SHUF | 비슷한 descriptor/source 블록 안에서 q metadata만 셔플 | 실제 sample–q 대응의 필요성 |

q뿐 아니라 A의 c0를 포함한 보조 모델은 따로 표시한다. q-only를 mainline으로 삼지 않는다. 모든 후보의 변수 수·규제·fold 방식은 같은 원칙으로 맞춘다.

- 1차는 작은 regularized linear/spline model. 분포를 보고 큰 gating neural network를 추가하지 않는다.
- 모델 형식·규제는 DISC의 source-group cross-validation에서만 정하고 CONF에서는 한 번 평가한다.
- 전처리·변수 선택·threshold·scale도 fold 내부에서 fit한다.
- 부족한 source 수를 patch-level random split으로 보충하지 않는다.
- 평가: held-source MAE, rank correlation, source별 paired prediction improvement, calibration plot.
- Cross-sensor 모델은 한 센서를 통째로 제외한 fit/confirm을 별도로 한다. 제외 센서 결과로 가중치를 다시 맞추면 zero-shot 예측이 아니다.
- e를 outcome으로 예측할 때 e 자체를 covariate로 넣지 않는다. Gain 예측에 e_zero와 e_learned를 동시에 넣지 않는다.

### 17.3 q 사용의 정당화 조건

1. C-DQ가 C-D보다 CONF에서 나은가?
2. 같은 e/contrast/texture 수준에서도 추가 정보가 남는가?
3. 조건부 q 셔플에서 이득이 줄어드는가?
4. 두 seed·두 센서 이상에서 방향이 재현되는가?
5. q를 계산하는 추가 A forward 비용 대비 의미 있는 효과인가?

1–3이 없으면 q는 descriptor의 대체 또는 중복일 가능성이 크다. 그래도 sample 기전의 연구결과는 남지만, q 기반 대응을 꼭 도입할 이유는 부족하다.

---

<a id="treatment"></a>

## 18. R60 — 특성을 알았으니 맞춤 대응이 가능한가? 조건부 검증

### 18.1 이 단계는 자동 실행하지 않는다

D/I/X 분석에서 **어떤 부족이 남는지**가 확인되고 C50의 score가 새 source에서 그 부족을 구분할 때만 연다. 원하는 q 이득이 나올 때까지 treatment를 바꾸는 탐색은 하지 않는다.

다음 중 DISC 근거가 가장 분명한 한 가지를 고정한다. 두 개가 비슷하거나 증거가 부족하면 R60을 보류하고 원인 분석 결과로 끝낸다.

| 남은 문제 | 대응 후보 | 주의 |
|---|---|---|
| 경계/HF 재현 부족 | GT의 HF residual에 대한 추가 L1 | LP/HF filter는 loss 분석용. LPAN input을 되살리지 않음 |
| 특정 band의 복원 부족 | 지정 band의 추가 GT reconstruction | band 선정은 DISC/CAL에서 고정. CONF로 고르지 않음 |
| Teacher의 특정 성분이 Student보다 안정적으로 좋음 | 해당 성분의 output KD | Teacher가 그 성분을 잘한다는 확인이 없으면 실행하지 않음 |

공통 `base recon+1e-4 Off`는 그대로다. Extra는 우선 U-Net에만 직접 전달한다. 정합 반응이 이미 좋은 EuCd에서 λ_off를 더 키우는 실험은 하지 않는다.

### 18.2 같은 source에서 다섯 arm

| Arm | 추가 항 배분 |
|---|---|
| R60-BASE | 없음 — 같은 기간의 단순 추가 fitting |
| R60-UNI | 모든 sample에 같은 추가 항 |
| R60-D | 입력 descriptor만으로 예측한 필요도에 따라 배분 |
| R60-DQ | 동일 predictor에 q를 추가해 예측한 필요도로 배분 |
| R60-DQ-SHUF | descriptor/source가 유사한 block에서 q metadata만 조건부 셔플 |

No-update source도 함께 평가한다. `BASE`를 random gate라고 표기하지 않는다.

Weights는 FIT에서 사전 고정한 필요도 상위25%를 가중2, 나머지를1로 둔 후 **FIT 전체 평균1**로 정규화한다. UNI도 평균1이다. 상위 비율·tie 규칙은 CONF 전에 고정하며, D/DQ/SHUF의 선택 sample 수와 mass를 맞춘다. 이 값들은 신규 pilot 설정이지 최적값이 아니다.

q/필요도는 frozen source 모델의 metadata다. Source canonical cue를 사용할 경우 그 정책을 명시하고 S30에서 증강에 대한 membership 안정성을 먼저 확인한다. 실제 증강된 입력의 q가 필요하면 frozen A에서 재계산하거나 정확한 augmentation cache를 사용한다. CNN 출력을 단순 회전해 같은 것으로 취급하지 않는다.

### 18.3 추가 loss와 gradient

\[
L_{extra}=\gamma\operatorname{mean}_i w_i\,\ell_{treatment,i},\qquad \gamma=0.05.
\]

γ=.05는 신규 보수적 진단값이다. 다른 단위의 HF/band/KD 항의 수치 크기를 곧바로 같은 감독 강도로 보지 않고 CAL에서 gradient norm을 기록한다. 추가 항을 마음대로 키워 양성 결과를 만들지 않는다.

\[
g_\phi=\nabla_\phi(L_{rec}+m_t10^{-4}L_{off}),\qquad
g_\theta=\nabla_\theta(L_{rec}+L_{extra}).
\]

Teacher를 사용할 경우 같은 M-frame·band order·현재 augmentation의 output을 고정 target으로 제공한다. Aligner 전체를 detach해서 기본 recon 경로까지 끊지 않는다. Loss별 gradient 수신자 검사를 한다. 기존 graph에 extra를 단순 합산하면 A까지 흐를 수 있으므로 별도 routing이 필요하다. [R3 §11]

### 18.4 학습·평가

- 같은 source A/U에서 각 arm 독립 시작, 동일 fresh optimizer·RNG.
- 1,000 updates, A LR1e-6 / U LR1e-5, 고정 LR. 원 optimizer의 끝난 LR을 그대로 로드하지 않는다.
- 같은 FIT 전체에서 동일한 sample 순서와 원 recipe의 augmentation을 사용한다. 단일 작은 quadrant pool만 반복하여 native 분포를 바꾸지 않는다.
- 정확한 0/250/1000에서 기록, main endpoint는 exact1000. 중간 best로 결과를 바꾸지 않는다.
- source→BASE, source→candidate, candidate−BASE를 각각 보고한다. **모든 arm이 source보다 나쁘면 “개선”이 아니라 상대적인 손실 완화**다.
- 집단별 효과뿐 아니라 전체 CONF L1, band/edge, RR/FR raw HQNR/fSCC를 보고한다.
- 동일 intervention이 EuCd에서 더 유리한지, matched-difficulty control group과의 **treatment×group interaction**을 본다.
- DQ가 D/SHUF보다 낫지 않으면 q 필요성이 입증된 것이 아니다.

R60 기본은 **주 센서 1개에서 5arm×1K**다. 독립 센서·seed의 완전한 5arm 반복은 긍정 결과 확인 단계로 큐에 남긴다. 한 번의 pilot을 최종 KD recipe로 고정하지 않는다. 근거가 없으면 조건부 stage를 억지로 수행하지 않는다.

---

<a id="budget"></a>

## 19. 약 12시간 운영과 누락 모델 학습의 분리

### 19.1 자산 준비 후 warm-start 분석 목표

아래는 **추정 예약치**다. 실제 GPU 사용량·CPU estimator 시간·wall time을 구분한다. 기존 EQREC4가 s1에서 약3.7h였다는 기록은 참고일 뿐, 네 센서와 신규 landscape 비용을 보장하지 않는다. [R1 §7]

| 작업 묶음 | 목표 |
|---|---:|
| A00 재현·provenance·profile | 0.75h |
| D10/D11/D12 전 센서 atlas·특성표·패널 | 1.75h |
| I20 동일 residual·Jacobian 민감도 | 1.25h |
| I21/I22 구조·spectral 통제 | 1.25h |
| I23 native proxy·correction landscape | 1.50h |
| I24 보간 대조 | 0.75h |
| I25 gradient·국소 개입 | 0.50h |
| S30/X40/C50 context·센서·추가 정보 | 1.50h |
| R60 조건부 대응 pilot | 1.25h |
| 보고서·blind review·raw evidence 점검 | 0.75h |
| Buffer | 0.75h |
| **합계** | **12.00h** |

R60이 진입하지 못하면 그 시간을 독립 estimator 또는 sensor 확인에 사용하되, R60을 성공/실패로 가공하지 않는다. 사람의 원본 패널 검토·개발 시간까지 12 GPU-hours 안에 포함됐다고 주장하지 않는다.

### 19.2 모델이 없을 때의 총시간

\[
T_{total}=T_{asset/implementation}+\sum_{\text{missing runs}}T_{50K}(sensor,server)+T_{analysis}.
\]

데이터셋 1개에서 모두 없으면 기본적으로 **P0 + N2 donor + L000 + L1E4 + 확인 L1E4 = 50K 5회**다. 과거 recipe의 초기 A1 의존성이 있으면 그 시간도 추가다.

예를 들어 WV3는 준비됐고 QB/GF2는 전부 없다면, 신규 준비 학습은 **10회**다. 단일 GPU에서 이를 모두 학습하고 네 센서 정밀 분석까지 12시간 안에 끝난다고 약속하지 않는다. 실제 step time/evaluation overhead로 예측치를 갱신한다.

- **단일 서버:** 의존성을 순서대로 수행하고 분석까지 계속한다. 12시간은 첫 진행 보고 시점일 뿐이다.
- **여러 서버:** 센서별 lane을 병렬 수행한다. Wall time은 가장 긴 lane과 공통 집계의 합에 가깝지만, GPU-hour는 lane 사용량을 합산한다.
- 이미 검증된 자산이 많으면 12시간보다 일찍 끝낼 수 있다. 시간 소진을 위해 의미 없는 추가 run을 만들지 않는다.

### 19.3 12시간 진행 보고

모든 필수 stage를 `complete / pending_dependency / pending_compute / unsupported_data / invalid_control`로 표시한다.

시간이 밀릴 때:

1. **모든 센서의 sampling과 primary atlas를 우선 보장**한다.
2. 각 센서 primary L1E4의 상세 개입을 수행하고, 확인 seed는 핵심 I20/I23/X40부터 반복한다.
3. 같은 sensor의 주변 λ/추가 seed 전수 probing보다 아직 분석하지 않은 sensor를 우선한다.
4. 모델 준비가 끝나지 않았으면 해당 training을 계속하고 completed 모델의 frozen 분석을 병행한다.
5. 비교군의 50K를 20K로 줄여서 수렴한 baseline으로 표시하지 않는다. 미완료는 exact update와 함께 partial로 남긴다.
6. 필수 독립 estimator/대조가 실패했다면 실패 이유를 결과로 남긴다. 통과할 때까지 threshold를 완화하지 않는다.

---

<a id="implementation"></a>

## 20. 구현·이식 계약

### 20.1 최소 adapter

아래는 **필요한 의미 인터페이스**다. 현재 repository의 함수가 이미 존재한다는 뜻은 아니다. 기존 성공 code를 adapter로 연결하고 동치를 검증한다.

```text
load_asset(registry_id) -> model, resolved_config, model_hash
read_sample(sample_id) -> P, S, Y_or_None, source_meta
make_ms_base(S, sensor_recipe) -> M
predict_shift(P, M) -> c[B,2]  # margin4/z-score exactly once
reconstruct_at(P, M, c) -> Yhat  # original P sampled once
measure_offset_bank(P, M, bank) -> c0, c_eps, residuals
estimate_native_geometry(P, M, Y, sensor, estimator_id) -> delta, quality, unit
get_descriptor(P, M, Y, descriptor_version) -> feature_record
```

### 20.2 실행 manifest: 작성해야 할 설정 예

```yaml
campaign:
  id: SMEC12
  version: 1
  scope: sample_mechanism_not_lambda_search
  server_id: REQUIRED_RUNTIME_VALUE
  data_root: REQUIRED_RUNTIME_VALUE
  output_root: REQUIRED_RUNTIME_VALUE
  target_analysis_hours: 12
  hard_timeout: false

assets:
  on_missing_checkpoint: train_dependency
  reuse_policy: hash_and_recipe_match
  auto_wait_for_remote_transfer: false
  historical_assets_read_only: true
  paired_same_unet_init: true
  donor_stage_steps: 50000
  downstream_steps: 50000
  primary_seed: 2025
  confirm_seed: 1234
  available_extra_seeds: [7777]
  bootstrap_lineage_must_be_explicit: true

sensors:
  WV3: {bands: 8, role: in_domain}
  QB:  {bands: 4, role: in_domain}
  GF2: {bands: 4, role: in_domain}
  WV2: {bands: 8, role: zero_shot, train_on_official_test: false}

model:
  recipe: fixed
  width: 112
  depth: [1, 2, 3]
  pan_reconstruction: false
  lpan_hpan_input: false
  aligner_train_lambda: 0.0001
  jitter_disk_radius_hr: 2.0
  auxiliary_odd_updates: true
  native_reconstruction_only: true

sampling:
  manifest_seed: 271828
  group_key: original_scene_or_strip
  unknown_group_policy: descriptive_only
  cheap_descriptor_scope: all_available
  model_atlas_target_per_sensor: 4096
  detail_per_quadrant_per_partition: 32
  detail_partitions: [DISC, CONF]
  official_eval: full_verified_manifest

probes:
  bank_A_radii: [0.25, 0.5, 1.0, 2.0]
  bank_A_angles_deg: [0, 90, 180, 270]
  bank_B_angles_deg: [45, 135, 225, 315]
  sensitivity_radii: [0.125, 0.25, 0.5, 1.0]
  keep_signed_effects: true

stages:
  required: [A00, D10, D11, D12, I20, I21, I22, I23, I24, I25, S30, X40, C50]
  conditional: [R60]
  prepare_missing: [B01, B02, B03, B04]

report:
  hypothesis_defaults_to: insufficient
  require_sample_to_raw_evidence_join: true
  separate_legacy_and_source_holdout: true
  separate_sensor_and_server_effects: true
  raw_metrics_and_normalized_diagnostics_both: true
```

YAML은 config template이며 repository CLI가 이 key를 직접 받아들인다는 보장은 없다. `REQUIRED_RUNTIME_VALUE`를 실제 확인한 값으로 채운 뒤 실행한다. 없는 source 파일 경로를 문자열로만 만들어 통과시키지 않는다.

### 20.3 핵심 pseudocode: frozen 진단

```python
# Semantic pseudocode, not a replacement for the verified trainer.
model.eval()
freeze_all_parameters(model)
P, S, Y, meta = read_sample(sample_id)
M = existing_sensor_bicubic(S)
c0 = predict_shift(P, M)
y0 = reconstruct_at(P, M, c0)

q_records = []
for eps in BANK_A:
    p_eps = warp_original(P, eps)
    c_eps = predict_shift(p_eps, M)
    r = c_eps + eps - c0
    q_records.append(r)  # Preserve axes and eps, not only the mean.

sensitivity = []
for delta in BANK_S:
    # No second call to A; P is the original image on every iteration.
    yd = reconstruct_at(P, M, c0 + delta)
    out_change = l1_on_fixed_roi(yd, y0)
    signed_error_change = None if Y is None else (
        l1_on_fixed_roi(yd, Y) - l1_on_fixed_roi(y0, Y)
    )
    sensitivity.append((delta, out_change, signed_error_change))

save_joinable_records(meta, model_hash, q_records, sensitivity)
```

### 20.4 필수 단위 검사

- c0/c_eps 순서는 dy/dx, sampling grid는 x/y이다.
- Aligner crop·표준화는 한 번, MS base는 한 번 더한다.
- R60 extra=0에서 base gradient가 기존 L1E4와 같고, extra는 A의 직접 gradient를 바꾸지 않는다.
- Frozen inference 중 weights/buffers 변경 없음.
- I20의 P는 매번 original이고, I24의 sequential branch만 두 번 warp한다.
- Unit별 `r=0`, identity filter, spectral weight 동일, δ 동일 대조가 예상대로 동치다.
- P0의 없는 q를 0/최고 품질로 처리하지 않는다.
- 결측·낮은 texture·source_unknown을 자동으로 성공 sample에서만 제외하지 않는다.
- source/checkpoint hash가 다른 cache를 sample 이름만으로 재사용하지 않는다.

---

## 21. 반드시 남겨야 할 원시 산출물

파일 이름은 **신규 산출물 계약**이다. 기존 서버에 이미 있다는 뜻이 아니다.

```text
<campaign_root>/
  manifests/
    asset_registry.json
    data_registry.json
    source_split_manifest.csv
    sample_manifest.csv
    descriptor_spec.json
    probe_manifest.json
    roi_manifest.json
    bootstrap_queue.json
    coverage.csv
  models_prepared/
    <new_run_id>/...  # original historical assets are untouched
  raw/
    sample_metrics.parquet_or_csv
    sample_descriptors.parquet_or_csv
    probe_residuals.parquet_or_csv
    fixed_residual_response.parquet_or_csv
    geometry_estimates.parquet_or_csv
    correction_landscape.parquet_or_csv
    interpolation_controls.parquet_or_csv
    gradient_interventions.parquet_or_csv
    context_seed_membership.parquet_or_csv
    treatment_outcomes.parquet_or_csv
  figures/
    blind_input_gallery/
    labeled_error_gallery/
    sensor_faceted_plots/
    paired_intervention_plots/
  analysis/
    conditional_models.json
    matched_sample_pairs.csv
    cross_sensor_results.csv
    hypothesis_verdicts.csv
    discrepancies.md
  report_SMEC12.md
  ledger.jsonl
```

### 공통 join key

`sample_uid, parent_source_uid, sensor, split, native_scale, crop_box, model_uid, checkpoint_hash, checkpoint_step, ROI_hash, preprocess_hash, probe_hash, intervention_id`.

- 서로 다른 table에서 하나의 sample을 같은 의미로 연결할 수 있어야 한다.
- scalar/vector는 전수 저장하고, 큰 tensor는 사전 지정 대표 sample만 저장한다.
- Summary에 사용한 row ID/필터/aggregation code hash를 `verdict_evidence.csv`로 남긴다.
- 원본 image를 외부 업로드할 필요 없이 로컬 gallery로 검토한다.

### 최소 그림

1. 센서별 e–q scatter/hexbin와 continuous conditional curve.
2. 네 집단의 raw·contrast-normalized error, e_base, 구조·spectral descriptor 분포.
3. 동일 residual의 s_out/GT error 곡선 및 방향성.
4. c_geo·c_rec·c0의 loss landscape와 band별 disagreement.
5. context/seed별 membership transition.
6. C-D와 C-DQ의 held-source prediction difference.
7. R60이 실행되면 source/base/각 arm의 전체·집단별 변화.

---

<a id="verdicts"></a>

## 22. 최종 분석 지시: 어떤 결론을 써야 하는가?

### 22.1 네 집단별로 같은 질문에 답한다

| 집단 | 설명해야 할 핵심 |
|---|---|
| EdCd | 실제로 쉬운가, raw scale이 작아서 쉬워 보이는가? q와 출력 민감도는 별개인가? |
| EdCu | 어떤 PAN 정보는 필요하지만 작은 위치 차이에는 둔감한가? Aligner의 불일치가 뒷단에서 보상되는가? |
| **EuCd** | 왜 이동은 읽기 쉬우면서 복원이 어려운가? 세부량·방향성·spectral 차이·민감도 중 무엇이 남는가? |
| EuCu | 관측 가능한 구조가 부족한가, spectral/저신호 문제인가, 현재 모델의 학습 상태인가? |

보고서에는 각 집단의 **대표 가설 + 지지 관측 + 반대 관측 + 아직 구별 못한 대안**을 한 문단씩 쓴다. “어렵다”, “민감하다”만 반복하지 말고 무엇에 대한 민감도인지 명시한다.

### 22.2 단계별 허용 주장

| 증거 | 허용되는 표현 | 아직 금지되는 확대 |
|---|---|---|
| descriptor 차이 | 집단의 입력 특성이 다르다 | 그 특성이 원인이다 |
| matched source/contrast에서도 반복 | 단순 contrast/source 차이만으로는 설명되지 않는다 | 모든 confound를 제거했다 |
| view/known-geometry 개입에 반응 | 해당 모델이 그 단서를 사용한다 | 실제 센서 오정합의 물리 원인을 찾았다 |
| fixed residual에서 s_out 차이 | 같은 잔여 위치 변화에 대한 출력 민감도가 다르다 | q가 native uncertainty다 |
| q가 cheap descriptor를 넘는 예측력 | 제한된 조건에서 기능적 cue의 추가 정보가 있다 | KD를 반드시 개선한다 |
| targeted > uniform/descriptor/shuffle, CONF 개선 | 그 대응에서 q의 유용성이 확인됐다 | 모든 Teacher·센서·학습 단계에서 보편적이다 |

### 22.3 최종 판정표

각 M1–M7에 대해 `supported_within_condition / replicated_across_sensors / opposed / insufficient / invalid_control` 중 하나를 쓰고 다음을 붙인다.

- 정확한 sensor/model/checkpoint/sample count와 source count.
- 원시 값·대응 차이·효과 크기·가능한 불확실성.
- DISC에서 발견한 결과와 CONF에서 확인한 결과의 구분.
- 결측·control failure·분석 도중 바뀐 설정.
- 논문/방법에 반영해도 되는 범위.

### 22.4 결론의 세 갈래

**A. 구조/대비만으로 설명된다:** q가 저렴한 descriptor와 중복이면 굳이 q gate를 채택하지 않는다. 구조적 어려움을 다루는 방향을 따로 검토한다.

**B. 같은 구조에서도 q가 민감도·보정 필요성을 추가로 알려 준다:** Teacher의 정확도 점수보다 **PAN 위치 활용의 기능적 cue**로 정의할 근거가 생긴다. R60으로 실제 대응 이득을 확인한다.

**C. 관계가 model/context/sensor마다 다르다:** q를 보편적인 sample 속성으로 주장하지 않고, 조건별 상태 진단으로 한정한다. 실패한 센서와 반대 방향의 결과를 숨기지 않는다.

**모든 경우에 L1E4 채택과 q 기반 새 gate 채택은 별개다.** 이번 분석이 끝났다는 이유로 앞단 λ·출력 frame·Teacher 선택을 자동 변경하지 않는다.

---

<a id="sources"></a>

## 23. 근거 문서와 provenance

아래 파일들은 이번 설계의 근거다. 신규 case의 sampling 수·step·threshold·기본 bootstrap은 **이 문서의 새 실험 선택**이며 과거에 실행된 값으로 쓰지 않는다.

| ID | 자료 | 사용한 부분 |
|---|---|---|
| R1 | `2026-09-15_s1_eqrec4-results.md` | §2 H1–H4, §3 네 집단, §6 native proxy 한계, §7 실행 시간, §8 provenance |
| R2 | `2026-09-15_s3_eqrec4-results.md` | §1 동일 s1 자산 재현, §2 H3b 판정 차이, §7 한계, §8 서버 이식 수정 |
| R3 | `PAN_Aligner_L1E4_TechnicalSpec_Evidence_KD_Handoff_2026-09-14.md` | §2–5 구조·loss·donor·초기화, §6 metric, §11 gradient routing |
| R4 | `PAN_S1_EQREC4_Alignment_Cue_Hypotheses_20h_2026-09-14.md` | probe/ROI 정의, quadrant·gradient·conditional utility 구분 |
| R5 | `2026-09-10_pa-a1-a3-s1-results.md` | §7 4센서의 기존 audit, §8/10 N2-R200 전환 및 donor 결과 |
| R6 | `2026-09-03_alignment-audit-s2-detail.md` | §0 센서/데이터 구성, §8.5 spectral appearance·추정기 한계 |
| R7 | `pancrafter.pdf` | §4.1 WV3/QB/GF2 학습 및 WV2 zero-shot 데이터 역할. 현재 L1E4 architecture 근거로 대체하지 않음 |
| R8 | `2026-09-14_palsv18-lambda-confirmed-and-na104-null.md` | L1E4 고정, 기존 HQNR 판정선·검증 한계 |

### 원문 해석의 불일치 관리

- R1의 H3b Supported와 R2의 Insufficient를 묵시적으로 통합하지 않는다. 관측 상관과 조건부 예측/인과 주장으로 나누어 이번 C50/I23에서 재검사한다.
- R1/R2의 0.011 판정선 표현과 R8의 0.0031은 별도 출처로 남긴다. 본 캠페인에서 새 센서에 어느 값도 자동 이식하지 않는다.
- R1의 K20 상대 L1 효과 크기 표기와 R2의 R(무작위) 명칭은 실제 원시 config와 대조해야 한다. 이번 설계의 BASE는 추가 감독 없는 arm이다.
- R6의 초기 phase/repair 설명을 현재 복원 recipe에 그대로 적용하지 않는다. 데이터 version과 기존 검증된 upsampler/evaluator가 우선이다.
- 본 문서는 실제 원시 imagery/checkpoint를 새로 실행해서 원인을 찾은 보고서가 아니다. 실험자가 위 evidence chain을 생성하기 위한 명세다.

### 작성 시 참조한 로컬 원문 snapshot의 SHA-256

같은 파일명이라도 내용이 바뀌면 이 hash와 다를 수 있다. 이 값은 **문서 snapshot**의 hash이며 서버 checkpoint/hash가 아니다.

| ID | SHA-256 |
|---|---|
| R1 | `5895d8f73e3b21c32e32cee0345aa27654d06c9b20385883b1dc10d7b3f334bf` |
| R2 | `368e17372233cb9cedef8f6795cddb20f9b4d0dfd8aa66d83926d776dadf49f2` |
| R3 | `42e074a22e6f7c7fa255e8fc59c014b062d680d5a308081ee2512df850a351d2` |
| R4 | `2721172d91e498e75f7b0e19b84882e94ddf41c53aaa16d6fa91aa61f02f0e48` |
| R5 | `3fd15614bf37ebecd9a96af0d726302cedc13ae7d4696f30ff2559819e6a539e` |
| R6 | `197bfab70d240dc381b79b69c33a4262076e23158b5d46b962a3446ec1cab296` |
| R7 | `309e96d4b6b21a16c1ab96b3322dd3d49886848ed9462b1ae2b6a69595ae7ce5` |
| R8 | `3a1c60381bcfcf01ac318ad93d94c4f95e3ed844bd62dfd2d2551f05657b6af1` |

---

## 24. 실행 완료 체크리스트

- [ ] WV3/QB/GF2/WV2 모두 sampling coverage가 기록되었다.
- [ ] 누락 모델은 B dependency로 학습했거나 실제 진행 상태/데이터 blocker가 남아 있다.
- [ ] 4band/8band, donor lineage, model split, server identity가 정확하다.
- [ ] 모든 e/q/개입은 동일 checkpoint·sample·ROI로 연결된다.
- [ ] Raw e, contrast-normalized e, e_base, edge/band 오차가 구분되었다.
- [ ] EuCd의 blind sample 패널과 matched 비교가 있다.
- [ ] 같은 잔여 displacement의 민감도와 자연 residual 크기가 분리되었다.
- [ ] Known absolute CNN error와 known-inverse sampler 대조가 구분되었다.
- [ ] Spectral-only 검사·native mixture residual·band별 오류를 과장 없이 연결했다.
- [ ] c_geo/c_rec/c0의 차이 및 독립 estimator 품질이 원시값으로 있다.
- [ ] 보간 대조가 성립하지 않으면 미성립으로 기록했다.
- [ ] q–e 상관과 gradient 충돌을 분리했다.
- [ ] Context·seed·sensor 반복 또는 미완료 이유가 있다.
- [ ] q를 사용한 label을 q로 예측해 성공이라고 하지 않았다.
- [ ] CONF를 보고 threshold·loss·treatment를 다시 선택하지 않았다.
- [ ] R60이 있으면 no-update/source와 BASE를 포함하고, 전체 품질 손실도 보고했다.
- [ ] 시간 초과를 피하려고 미완료 분석을 complete로 표시하지 않았다.

**최종 산출물의 목표는 새 최고점이 아니라, “q가 작지만 reconstruction이 어려운 sample은 무엇이며, 왜 그런지, 그 특성을 알고 나면 어떤 대응이 실제로 유리해지는지”를 센서와 모델 조건을 명시해 설명하는 것이다.**
