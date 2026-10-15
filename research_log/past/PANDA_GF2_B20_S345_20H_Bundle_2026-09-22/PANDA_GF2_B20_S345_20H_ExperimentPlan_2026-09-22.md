# PANDA GF2 B20 — 학습 경로·supervision 강도·PAN 주파수 증강 20시간 실험

- **Campaign:** `PANDA_GF2_B20_S345_20260922_v1` (`GFB20`)
- **작성일:** 2026-09-22. 근거 cutoff는 직전 GF2 결과 검토 **2026-09-21 23:09 KST**다. 이후 live Sheet·서버 상태를 이번 작성에서 다시 조회하지 않았다.
- **대상:** GF2, **s3·s4·s5의 5090 서버만**. s1/WV3·s2/QB ablation은 보호한다.
- **예산:** 세 서버가 공유하는 실제 `t0`부터 **20시간의 wall-clock**. 서버별 20 GPU-hour, 병렬 합계 최대 60 GPU-hour에 해당하나 유휴·준비·평가 시간도 포함한다.
- **상태:** 실험 설계·case registry 작성. **실제 학습·원격 queue 변경·코드 배포·Sheet 쓰기를 수행한 문서가 아니다.** 기존 L100 runner에 바로 투입할 수 있는 config도 아니다.
- **핵심:** **새 Teacher를 학습하지 않는다.** 각 서버의 검증된 기존 Teacher·Student endpoint로 시작한다. 새로운 시도는 Student 후기 fitting, 기존 loss 계수의 큰 변경, train-only PAN 입력 분포 확장이다.

## 0. 실행 요약

| 서버 | 기본 실험 | 기본 학습 수 | 남는 시간의 조건부 확장 |
|---|---|---:|---|
| **s3** | 공간 수준이 다른 3개 로컬 endpoint 각각 **FT-BASE vs 후기 H010** | 6 × 추가20K | 같은 부모에서 별도 data stream 재검정 4개; 새 Student100K 2개에서 BASE/H010 후기 fork 4개 |
| **s4** | BASE100K endpoint 2개 각각 **FT-BASE vs K1 vs E100** | 6 × 추가20K | K1/E100 중 **한 조건만** fresh100K BASE와 2 seed 대응; FT 음성이면 E100 초기경로 검정 |
| **s5** | BASE100K endpoint 2개 각각 **NATIVE0 vs NAT_MIXCAL vs PANMIX** | 6 × 추가20K | NAT_MIXCAL vs PANMIX fresh100K 대응; FT와 초기부터 증강의 효과를 구분 |

**먼저 실행할 기본은 Student 18개, 추가 update 총 360K다.** 조건부 18개를 모두 포함한 **등록 상한은 36개**(s3 16, s4 10, s5 10)다. 새 Teacher 학습은 0개다. s5의 augmented LP/q-cache·mixed calibration 구축 1묶음은 별도 준비 작업이다. 최대 수는 전량 완료 약속이 아니며, 하단 시간 입장 규칙을 만족할 때만 확장한다.

이번 20시간은 무한 seed 반복이 아니다. 높은 점수가 한 번 나오면 멈추거나, 낮은 비교군을 그대로 두고 변경안만 재추첨하지 않는다. 같은 부모에서 갈라진 fork는 **독립적인 fresh 학습 반복으로 세지 않는다.**

## 1. 관측한 사실과 이번에 검정할 가설

직전 분석 [S1,S2]에서 같은 s3 R100/BASE/S100의 SS95001과 SS95002는 ERGAS 차이가 약 **0.0314%**지만 HQNR 차이는 **0.009297**이었다. H010은 두 pair에서 RR를 개선했지만 둘 다 Ds를 악화시켰고, E1은 두 pair 모두 H/E/Ds에서 불리했다. 반면 **11/11개의 100K run에서 자기 중간50K→최종100K의 H/E/Ds는 함께 개선**됐다. 따라서 이번 설계는 'GF2가 50K 이후 무너진다'는 전제를 사용하지 않는다.

| 근거 | 이번 가설 | 반증될 수 있는 결과 |
|---|---|---|
| 비슷한 RR 정확도와 크게 다른 FR 품질 | 이미 얻은 공간 mapping을 출발점으로 낮은 LR에서 더 나은 균형에 도달할 수 있는가? | 후기 fork에서도 RR만 개선하고 Ds가 계속 악화 |
| H010의 불리한 seed는 중간50K 이전부터 차이가 존재 | hard 추가 강조 감소를 **후기에만** 적용하면 초기부터 적용한 비용과 다른가? | FT-H010도 모든 부모에서 FT-BASE보다 공간 품질 저하 |
| 과거50K 진단의 작은 weighted soft/edge gradient | 10배 KD·50배 edge가 실효 supervision을 바꾸면 다른 mapping을 얻는가? | gradient가 유효하게 커져도 성능·안정성이 악화, 또는 soft gate가 거의 비활성 |
| RR–FR 관계 민감성·positive signed Ds 관측 | PAN detail amplitude의 train-only 변동이 native inference 일반화를 개선하는가? | native RR/FR 저하 또는 calibration 변경에서만 효과 발생 |

여기서 U가 모든 원인이라는 확증, q가 물리적 정합 정확도라는 해석, 큰 edge가 반드시 공간 점수를 높인다는 해석은 하지 않는다. 증강의 scale·계수·추가20K 등은 **본 계획이 새로 정한 검정값**이며 기존 실험에서 효과를 입증한 값이 아니다.

## 2. 그대로 유지하는 method 계약

| 항목 | 고정값 / 규칙 |
|---|---|
| Teacher / Student backbone | **P0/W112/D[1,2,3] → PLH/W104/D[1,2,2]** |
| PAN Aligner | 기존 `PANGlobalAligner`; PAN1 + bicubic-upsampled MS4; margin4; global `(dy,dx)` |
| Task / attention / modulation | 단일 HRMS; PAN 재구성·MARs·attention·mode modulation **OFF** |
| 밴드 / DN / patch | GF2 C4, maxDN1023, `2DN/1023−1`; PAN64/MS16/GT64; batch48 |
| MS frame | MS·GT·output 이동 없음; bicubic MS base를 한 번만 더함 |
| Warp | FP32, bicubic, border, align_corners=False; PAN/upLP를 하나의 grid로 warp; signed H=P−L |
| LP | σ1.98, k41, replicate, `[2::4,2::4]`, float64 생성/float32 cache; inference LP 규약 변경 없음 |
| Objective | 기존 adaptive hard·selective soft·q-edge·q-hard-A 모두 유지 |
| Gradient | **U ← hard+soft+q-edge; A ← q-hard만**. 두 parameter 집합을 별도로 미분 |
| Teacher | 각 Student의 원 frozen A/U와 reference 전체 유지; 임의 reference 교체 없음 |
| Precision / optimizer | FP32, AMP OFF; AdamW β=(.9,.999), eps1e−8, wd=.01; runtime/TF32/cuDNN 고정·기록 |
| 평가 | 기존 native RR20·FR20, Q4, DN1023, RR20:-21, full512 native PAN/original LMS; 공식 clamp 규약 그대로 |

`s_i`, `d_T`, `a_T`는 detached weight다. eS와 Student–Teacher discrepancy는 미분 가능하다. 수식 구현은 다음으로 고정한다 [S3,S4].

\[
e_T=\operatorname{mean}_b|Z_T-Y|,\quad e_S=\operatorname{mean}_b|Z_S-Y|,
\]
\[
d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\quad
 a_T=\operatorname{sg}\frac{[e_S-e_T]_+}{e_S+10^{-6}},\quad
s=\operatorname{sg}\frac{q_{\rm ref}}{q_{\rm ref}+q}.
\]
\[
\ell_H=\operatorname{mean}_p[(1+\alpha d_T)e_S],\quad
\ell_K=\operatorname{mean}_p[\beta(1-d_T)a_T\operatorname{mean}_b|Z_S-Z_T|],
\]
\[
L_U=\operatorname{mean}_i(\ell_{H,i}+\ell_{K,i}+\lambda_Es_i\ell_{E,i}),
\qquad L_A=\operatorname{mean}_i(s_i\ell_{H,i}).
\]

Edge는 signed Scharr(/32), reflect padding, 1px 경계 제외, x/y 각각 0.5 평균이다. H010은 **α 항만** 바꾼다. K1/E100도 새 loss를 추가하거나 gate를 제거하지 않는다. B/C의 큰 변경을 위해 A를 freeze하거나 soft/edge를 A로 직접 흘리는 우회는 금지한다.

**금지:** GF2 전용 backbone·PAN head 추가, MS/GT phase 변경, 평가 MTF 변경, 원본 FR PAN을 이동해 점수 계산, output blur/ensemble/test-time gain 선택, 성능이 좋은 다른 Teacher로 자동 대체. 기존 B02/DONN2·W168 DUAL 자산도 부모 후보가 아니다.

## 3. 사용할 부모 endpoint와 로컬 reference

| 부모 ID | 서버 | 원 run·seed | 선택 step | 원 reference | HQNR / ERGAS / Ds |
|---|---|---|---|---|---|
| P3HI | s3 | L100I1-S04 / S95002 | 100,000 | R3_100 | 0.955960/0.551502/0.023873 |
| P3LO | s3 | L100I1-S03 / S95001 | 100,000 | R3_100 | 0.946663/0.551329/0.033533 |
| P4A | s4 | L100I1-S07 / S95001 | 100,000 | R4_100 | 0.953483/0.552704/0.026773 |
| P4B | s4 | L100I1-S10 / S95002 | 100,000 | R4_100 | 0.954274/0.548244/0.026062 |
| P5A | s5 | L100I1-S11 / S95001 | 100,000 | R5_100 | 0.947716/0.550621/0.032101 |
| P5B | s5 | L100I1-S14 / S95002 | 100,000 | R5_100 | 0.951309/0.547760/0.028465 |
| P3OLD | s3 | QGBASE_S92001 / S92001 | 50,000 | R3_TA50 | 0.957673/0.556494/0.023503 |

모든 부모는 **exact endpoint**다. P3OLD는 45,450의 RAW 최고가 아닌 **exact50K**다. 성능이 좋았던 부모만 쓰지 않기 위해 s3에는 같은 R100의 낮은-H endpoint **P3LO**를 포함한다. s4/s5는 동일 reference의 BASE 두 seed를 모두 사용하며 H010/E1 부모를 섞지 않는다.

`GFB20_ParentAssets.json`에 full run ID, 보고된 model SHA, 원 reference identity와 출처 행을 담았다. 알 수 없는 원 경로/SHA는 `null`이다. **source 파일 경로를 추정해서 채우거나 manifest가 없는데 검증됐다고 표시하지 않는다.** 실행 전 원 endpoint의 identity와 실제 bytes를 읽어 채운다. R3_100 Teacher SHA는 원 Student reference manifest에서 해결한다. P3OLD의 원 Teacher run 이름도 원 manifest에서 해결한다.

### 3.1 재사용 단위

- 부모 Student의 **U와 A를 둘 다** 로드한다. FT 시작 시 A를 Teacher clone으로 다시 덮으면 실험이 달라진다.
- 부모마다 원 Teacher A/U, τR, qref, q-cache, LP/data identity, band order를 묶어서 사용한다.
- s3는 로컬 R3_100과 과거 로컬 R3_TA50, s4는 R4_100, s5는 R5_100만 사용한다. 새 서버 간 Teacher 전송·학습 대기는 없다.
- weights export의 포맷 변환이 필요하면 tensor별 hash 동등성을 입증한다. 포맷이 바뀐 byte SHA를 같은 값으로 꾸미지 않는다.
- P3OLD의 bytes 또는 reference를 찾지 못하면 A_OLD block을 `BLOCKED_MISSING_PARENT`로 남기고 A_HI/A_LO를 진행한다. 다른 모델을 몰래 대체하거나 새 Teacher를 학습하지 않는다.

## 4. 공통 FT20K 계약 — exact-resume과 구분

**이번 기본18개는 새로운 weights-initialized fine-tuning run**이다. 부모의 U/A 가중치는 유지하지만 **AdamW moment·step counter와 scheduler는 두 parameter 집합 모두 초기화**한다. 원 sampler/RNG를 이어받지 않고 명시한 새 paired stream을 만든다. 이것이 본 계획의 단일 fork 정책이다.

| 항목 | FT20K |
|---|---|
| 시작 | Student parent U+A weights, 원 frozen Teacher/reference |
| 추가 update | **20,000**, 사전 고정; 좋은 점수에서 조기 종료하지 않음 |
| Optimizer | U/A 모두 new AdamW, moment=0, step=0; 한 조건만 parent optimizer 유지 금지 |
| U / A peak LR | **1e−5 / 3e−7**, A trainable |
| Schedule | 추가 update 기준 warmup100 + cosine20K→0 |
| BASE loss | α1, β.1, λE.002 |
| Stream | block 내 같은 원 sample 순서·HV/ROT4 view. stream seed만 해당 표 사용 |
| Resume | **새 FT run 내부의 중단만** 그 run의 U/A+optimizer+scheduler+RNG+sampler 전체로 exact-resume |
| 기록 | `parent_step`, `added_updates`, `lifetime_student_updates`를 모두 보존 |

완료된 추가 update를 t라 할 때, 다음 update의 LR multiplier는 기존 인덱싱과 일치하게

\[
f_N(t)=\begin{cases}t/100,&0\le t<100,\\
\tfrac12[1+\cos(\pi(t-100)/(N-100))],&100\le t\le N.
\end{cases}
\]

로 둔다. f(0)=0, f(100)=1, f(N)=0을 unit test로 검증한다. N=20,000이며 종료 scheduler last_epoch도 N이다. 학습 시작의 zero-LR update를 빼서 19,999회로 세지 않는다. optimizer.step 호출 수가 총20,000이다.

P3OLD 기반은 **50K+20K=70K**, 나머지 기존 부모 기반은 **100K+20K=120K**의 누적 Student update다. 이를 fresh70K/fresh120K 또는 원래 100K의 exact-resume으로 부르지 않는다. `Train(h)`에는 새 구간, `Parent train(h)`에는 기존 비용, `Lineage train(h)`에는 합계를 별도 저장한다.

### 4.1 초기 동등성과 RNG

동일 부모 block의 t=0 model full/U/A tensor hash는 같아야 한다. 새 data stream seed는 profile명과 무관하며, 모델 가중치를 재초기화하는 seed가 아니다. DataLoader worker와 corruption/probe/gamma RNG는 서로 분리한다.

- source sample order: `stream_seed+300000`
- existing geometry-view draw: `stream_seed+400000`
- worker: `stream_seed+500000`
- gamma: `stream_seed+600000`, 별도 counter-based RNG

기존 BatchStream helper를 사용할 때 실제 RNG 매핑을 이 계약으로 고정한 전용 adapter와 test가 필요하다. 문자열 profile 차이가 data order를 바꾸면 실패다. C의 native arm도 gamma RNG를 동일하게 진행시키되 입력에는 적용하지 않는다. 모든 비교에서 first256 `(source_id,view_id)` hash를 대조하고, C에서는 `(source_id,view_id,gamma_id)`까지 기록한다.

## 5. 기본 실행 Case — 18개

| Case | 서버 | 부모 endpoint | Stream seed | Profile | α / β / λE | 추가 update | 목적 |
|---|---|---|---|---|---|---|---|
| A01 | s3 | P3HI | 96301 | BASE | 1 / 0.1 / 0.002 | 20,000 | 후기 hard 강조 대조 |
| A02 | s3 | P3HI | 96301 | H010 | 0.1 / 0.1 / 0.002 | 20,000 | 후기 hard 강조 대조 |
| A03 | s3 | P3OLD | 96302 | H010 | 0.1 / 0.1 / 0.002 | 20,000 | 후기 hard 강조 대조 |
| A04 | s3 | P3OLD | 96302 | BASE | 1 / 0.1 / 0.002 | 20,000 | 후기 hard 강조 대조 |
| A05 | s3 | P3LO | 96303 | BASE | 1 / 0.1 / 0.002 | 20,000 | 후기 hard 강조 대조 |
| A06 | s3 | P3LO | 96303 | H010 | 0.1 / 0.1 / 0.002 | 20,000 | 후기 hard 강조 대조 |
| B01 | s4 | P4B | 96401 | BASE | 1 / 0.1 / 0.002 | 20,000 | 실효 supervision 대조 |
| B02 | s4 | P4B | 96401 | K1 | 1 / 1 / 0.002 | 20,000 | 실효 supervision 대조 |
| B03 | s4 | P4B | 96401 | E100 | 1 / 0.1 / 0.1 | 20,000 | 실효 supervision 대조 |
| B04 | s4 | P4A | 96402 | E100 | 1 / 0.1 / 0.1 | 20,000 | 실효 supervision 대조 |
| B05 | s4 | P4A | 96402 | K1 | 1 / 1 / 0.002 | 20,000 | 실효 supervision 대조 |
| B06 | s4 | P4A | 96402 | BASE | 1 / 0.1 / 0.002 | 20,000 | 실효 supervision 대조 |
| C01 | s5 | P5B | 96501 | NATIVE0 | 1 / 0.1 / 0.002 | 20,000 | calibration / PAN 증강 분리 |
| C02 | s5 | P5B | 96501 | NAT_MIXCAL | 1 / 0.1 / 0.002 | 20,000 | calibration / PAN 증강 분리 |
| C03 | s5 | P5B | 96501 | PANMIX | 1 / 0.1 / 0.002 | 20,000 | calibration / PAN 증강 분리 |
| C04 | s5 | P5A | 96502 | PANMIX | 1 / 0.1 / 0.002 | 20,000 | calibration / PAN 증강 분리 |
| C05 | s5 | P5A | 96502 | NAT_MIXCAL | 1 / 0.1 / 0.002 | 20,000 | calibration / PAN 증강 분리 |
| C06 | s5 | P5A | 96502 | NATIVE0 | 1 / 0.1 / 0.002 | 20,000 | calibration / PAN 증강 분리 |

**A_HI/A_OLD/A_LO, B_PANEL1/2, C_PANEL1/2는 각각 하나의 입장 단위**다. 시작 전에 그 block의 모든 arm과 평가·보존 비용을 예약한다. BASE 하나만 끝내고 '대응 비교 완료'로 기록하지 않는다. Profile별 순서는 표·CSV에 고정했고 두 번째 부모에서는 일부 순서를 반대로 두었다.

## 6. s3 — 후기 hard fitting과 학습 경로

### 6.1 기본 비교

- A01↔A02: 최근 높은-H R100 endpoint에서 후기 α 감소.
- A03↔A04: 과거 R50/high-H endpoint에서 같은 개입. 서로 다른 reference이므로 앞 pair와의 절대값 우열을 α 효과로 해석하지 않는다.
- A05↔A06: 낮은-H R100 endpoint도 회복되는가, 아니면 좋은 시작점을 보존할 때만 가능한가?

H010의 α는 FT 첫 update부터 .1이다. 본 계획의 '후기'는 원 Student50K/100K 학습 뒤라는 뜻이지, 추가20K 안에서 다시 α를 뒤늦게 바꾸는 것을 뜻하지 않는다. β와 λE는 BASE다. U/A 두 hard 경로에 동일 α를 적용한다.

### 6.2 Stream 재검정

시간이 허용되면 A07–A10을 실행한다. 두 R100 부모에서 **별도의 새 stream seed**로 BASE/H010을 다시 비교한다. 같은 source parent를 공유하므로 이는 data-order sensitivity 검정이지 새4개 독립 training seed가 아니다. 기본 pair의 결과가 나쁘더라도 사전 정의한 재검정값을 바꾸지 않는다.

### 6.3 Fresh 전체 스케줄 재현

H010의 RR-preserving 또는 spatial 개선 신호가 반복되고 시간이 남으면 A11–A16을 실행한다. 로컬 R3_100을 유지한 채 **fresh Student100K BASE 두 seed**를 학습한다. 각 fresh exact100K에서 BASE20K/H01020K로 갈라져 전체 `BASE100K→FT20K` 경로를 검정한다.

이때 Teacher A는 fresh Student 시작 시에만 clone한다. 새로운100K의 두 fork에서는 그 Student A를 보존한다. Fresh seed는 **96611/96612**로 고정하며, 중간 HQNR를 보고 더 좋은 부모 seed로 갈아타지 않는다. A11/A14는 분기들의 공통 trunk로 비용을 한 번만 세되, 두 종단 arm의 총비용도 따로 보고한다.

## 7. s4 — K1·E100의 큰 supervision 대조

### 7.1 계수와 ramp

| Profile | α | β | λE | 변경 규모 |
|---|---:|---:|---:|---|
| BASE | 1 | .1 | .002 | 대조 |
| **K1** | 1 | **1.0** | .002 | soft KD 10배 |
| **E100** | 1 | .1 | **.1** | GT edge 50배 |

E100의 '100'은 λE=.100을 뜻한다. **100K 학습이나 100배를 뜻하지 않는다.** K1/E100은 처음1,000 update 동안 기존값에서 목표값으로 선형 ramp한다. t completed update에서 계수는 `w(t)=w_base+min(t/1000,1)*(w_target-w_base)`다. BASE는 상수다. Ramp까지 포함한 recipe를 비교하며, '고정 계수만의 효과'라고 축소해 설명하지 않는다.

두 계수를 동시에 강화하지 않고 A LR·α·Teacher·data는 고정한다. Soft gate aT=0인 곳에서 강한 β도 신호를 만들지 못한다. 이를 해결하려고 gate를 제거하거나 teacher target을 다른 것으로 바꾸지 않는다.

### 7.2 Gradient 진단은 실제100K에서 다시 한다

고정 train128 중24 patch를 선택하고 각 원 scalar loss가 만드는 U gradient를 **정확한 weighted term 기준**으로 계산한다. log에는 raw와 weighted norm 모두 남긴다.

- `||g_soft||/||g_hard||`, `||g_qedge||/||g_hard||`와 hard/soft·hard/edge cosine.
- aT>0 화소비율, `(1-dT)*aT`의 합·상위 분위, hard/soft/edge loss의 질량.
- U/A 실제 parameter-update norm, c 변화, border/interior 오차.

과거 .009·3e−4 비율은 **참고 가설**이다. 현재 계수 선택의 성공 근거나100K 실측으로 복사하지 않는다. 현재 finite gradient가 작다는 이유만으로 기본 K1 run을 없애지 않는다. 다만 t=0/5K/20K에서 soft/hard<1e−3이고 aT>0 화소비율도 계속1e−4 미만이면 `LOW_ACTIVE_MASS`를 기록한다. 이는 성숙한 부모에서의 후기 K1이 약하다는 진단이며, fresh Student 초기의 soft가 없다는 뜻은 아니다. 이 경우 단순 β 추가증가는 하지 않고 아래 고정 E100 fresh-path 검정을 우선한다. β를 자동으로10·100까지 올리지 않는다.

### 7.3 Fresh transfer는 후기 FT 결과의 동일 재현과 다르다

두 core 부모에서 공간 개선이 있는 **한 profile만** B_SELECTED로 잠근다. 후보 집합은 K1/E100뿐이며 결합형은 없다. B07–B10에서 fresh Student100K BASE와 그 profile을 seed96611/96612로 비교한다.

여기서는 강한 supervision을 **fresh 학습 초기부터** 같은1K ramp로 사용한다. 따라서 명칭은 `FRESH_TRANSFER`다. 후기 FT에서 좋았다는 것과 fresh 시작에서도 좋은지를 구분한다. U1e−4/A3e−6, warmup100/cosine100K는 양쪽에서 동일하다.

두 profile 모두 FT 개선 기준에 못 미쳐도, 낮은 LR의 성숙한 모델에서 실패한 것이 fresh 초기의 효과 부재를 뜻하지는 않는다. 따라서 **시간·무결성·발산 검사를 통과하면 첫 fresh pair를 실행하며, FT 음성일 때는 사전 지정한 E100을 사용**한다. 상태는 `FRESH_PATH_TEST_AFTER_NEGATIVE_FT`이지 `PROMOTED_SUCCESS`가 아니다. s4에서 첫 fresh pair를 고르는 것은 FT 성공에만 종속되지 않는다. 두 번째 fresh pair도 시간 입장 시 결정하며 첫 seed 점수에 따라 몰래 삭제하지 않는다.

K1/E100 두 조건을 모두 fresh로 늘리거나 좋은 seed가 나올 때까지 반복하지 않는다. 양쪽 모두 유망하면 정해진 규칙으로 하나를 택하고 두 번째 조건은 다음 campaign 후보로 보존한다.

## 8. s5 — train-only PAN 주파수 증강과 calibration 분리

### 8.1 세 arm이 필요한 이유

새 PAN view를 사용하면 q뿐 아니라 Teacher error 분포도 바뀔 수 있다. 증강 arm만 τ/qref까지 바꾸면 **입력 효과와 calibration 효과가 섞인다.** 이를 분리하기 위해 두 부모에서 다음3조건을 실행한다.

| Profile | 학습 PAN | τR / qref | q 값 | 해석 |
|---|---|---|---|---|
| **NATIVE0** | 원본만 | 부모의 native scale | 원 native q | 기존 recipe의 FT 대조 |
| **NAT_MIXCAL** | 원본만 | 아래 mixed scale | gamma1의 q | calibration scale만의 효과 |
| **PANMIX** | gamma .75/1/1.25 | **동일 mixed scale** | 해당 gamma의 q | NAT_MIXCAL 대비 입력 증강 효과 |

두 원 Student 부모는 같은 R5_100 Teacher를 쓰므로 **mixed cache/calibration은 한 번만 만들고 공유**한다. 새 Teacher를 증강 데이터로 재학습하지 않는다. 판정은 `PANMIX−NAT_MIXCAL`(입력), `NAT_MIXCAL−NATIVE0`(scale), `PANMIX−NATIVE0`(전체 recipe)를 모두 남긴다.

### 8.2 입력 생성의 정확한 정의

native PAN의 원 DN을 float64로 읽고, 중심 정렬된 separable Gaussian **σ=1.0 HR pixel, k=7, reflect padding**을 G로 둔다. 대칭 kernel은 좌표 −3,…,3에서 exp(−x²/2)를 합1로 정규화한다.

\[
P_\gamma=G(P)+\gamma[P-G(P)],\qquad
\gamma\in\{0.75,1.0,1.25\},\quad
\Pr(\gamma)=(0.25,0.50,0.25).
\]

- **gamma1은 수식 재계산 대신 원 PAN tensor를 직접 반환**하여 native identity를 보존한다.
- 새 global contrast·noise·shift를 동시에 더하지 않는다. MS/GT는 어떤 gamma에서도 같아야 한다.
- augmented PAN에서 **기존 LP σ1.98/k41/replicate/2::4** 레시피로 LPγ를 다시 만든다. gamma1은 기존 검증 LP cache를 사용한다.
- 원 source PAN64에서 gain과 LP를 생성한 뒤 원 augmentation의 **fixed-HV/ROT4 view를 PAN/MS/LP/GT에 동일하게 적용**한다. rotation 뒤에 LP를 다시 계산해서 decimation phase를 바꾸지 않는다.
- Teacher와 Student는 같은 Pγ/MS를 입력받는다. 각각 자기 Aligner로 보정하며 Student는 LPγ/Hγ를 사용한다.
- 입력 gain 생성에는 gradient가 필요하지 않다. 이후 Student warp의 c는 detached하지 않는다.
- no abs, no output blur, no per-image min–max, no rounding, no training clamp. γ>1의 overshoot는 분포 진단으로 기록한다. 범위를 보기 좋게 만들려고 임의 clip하지 않는다.
- 증강은 train에만 적용한다. **validation/RR/FR inference는 gamma1**, 기존 native PAN 및 LP, 기존 Student forward다.

G는 물리적 GF2 MTF 보정이라는 주장이 아니며, baseline의 LP filter도 교체하지 않는다. 64px patch 경계에서 reflect/G·replicate/LP 효과가 포함되므로 border 진단으로 부작용을 따로 본다.

### 8.3 q-cache와 mixed reference scale

원본 R5_100 Teacher A/U는 frozen이다. 원 native τ/qref/cache를 수정하지 않고 **새 별도 manifest**를 발행한다.

1. q-cache key는 `(source_id, geometry_view_id, gamma_id)`, shape은 GF2 N=19,809일 때 **[19,809,4,3]**다. 원 ID·view 정의와 Teacher SHA를 포함한다.
2. AXIS16 반경(.25,.5,1,2), 축4방향, mean_xy=L1/2 정의를 유지한다. **Warp(Pγ,ε)**를 probe로 쓰며 `gain(Warp(P,ε))`로 바꾸지 않는다.
3. q 생성과 parity verification은 batch16/FP32로 통일한다. gamma1 slice는 기존 cache와 동일 Teacher/runtime으로 검사한다. raw 수치 차이는 사전 tolerance와 원인을 기록하며 유리한 cache를 선택하지 않는다.
4. 같은 calibration3072 train base ID, seed1234를 사용한다. **모집단 가중치(.25,.50,.25)를 반영하려고 gamma token을 [.75,1,1,1.25]로 반복한 정확한 median**을 사용한다.
5. τmix는 geometry 무증강 calibration3072의 **전체 pixel·band-mean absolute error**를 gamma token별 pooling한 median, floor1e−6이다. patch median의 평균이나 RR ERGAS로 대체하지 않는다.
6. qref_mix는 calibration3072 × 4 geometry view × 4 gamma token의 q median이다. 전체 train median을 대신 쓰지 않는다. qref_mix가0/비정상이면 오류로 중단하며 임의 eps를 더하지 않는다.
7. Student의 dT/eT/aT는 그 update의 **같은 입력 view의 Teacher/Student/GT**에서 계산한다. augmented sample에 native Teacher prediction이나 native q를 붙이지 않는다.
8. tau_mix/qref_mix, native scales, view 확률, G/LP source·cache SHA, Teacher SHA, train/data SHA와 calibration ID를 한 manifest로 연결한다. Student 학습 도중 이 통계를 갱신하지 않는다.

이것은 **명시적 training-view/calibration protocol 확장**이다. backbone·loss 수식·gradient routing·native inference는 유지하지만, 기존 native-only 학습과 완전히 같은 protocol이라고 쓰지 않는다.

새 gamma의 q를 일부128 patch에서만 계산해 전체 cache처럼 사용하거나 nearest-ID q로 메우지 않는다. native slice 재사용은 원값 동등성이 검증될 때만 가능하다. 시간 부족은 cache 축소가 아니라 **조건부 fresh block 제외**로 해결한다.

### 8.4 Fresh transfer

PANMIX가 NAT_MIXCAL 대비 입력 효과를 보이면 C07–C10을 우선한다. **FT에서 개선되지 않아도 첫 fresh pair C07/C08은 시간·무결성·안전 조건이 맞으면 실행**한다. 후기20K 증강과 처음부터100K 증강은 다른 학습 경로이기 때문이다. 이 경우 `FRESH_PATH_TEST_AFTER_NEGATIVE_FT`로 남기고 효과가 입증됐다고 표현하지 않는다. 두 번째 pair는 전체 시간 예약으로 결정한다. fresh100K에서 **NAT_MIXCAL vs PANMIX**, seed96611/96612다. 같은 R5_100, 같은 mixed scale, 같은 U 초기값·A clone으로 시작한다. calibration 효과 자체의 fresh100K 일반화는 이번 두 조건만으로 재검정하지 못하므로 그 한계를 남긴다.

## 9. 조건부 등록 Case — 18개

| Case | 서버 | 단계 | 부모 / 초기화 | Profile | Model 또는 stream seed | update |
|---|---|---|---|---|---|---|
| A07 | s3 | STREAM_CHECK | P3HI | H010 | 96311 | 20,000 |
| A08 | s3 | STREAM_CHECK | P3HI | BASE | 96311 | 20,000 |
| A09 | s3 | STREAM_CHECK | P3LO | BASE | 96312 | 20,000 |
| A10 | s3 | STREAM_CHECK | P3LO | H010 | 96312 | 20,000 |
| A11 | s3 | FRESH_REPLAY | fresh U + local Teacher A | BASE | 96611 | 100,000 |
| A12 | s3 | FRESH_REPLAY | P3NEW1 | BASE | 96711 | 20,000 |
| A13 | s3 | FRESH_REPLAY | P3NEW1 | H010 | 96711 | 20,000 |
| A14 | s3 | FRESH_REPLAY | fresh U + local Teacher A | BASE | 96612 | 100,000 |
| A15 | s3 | FRESH_REPLAY | P3NEW2 | BASE | 96712 | 20,000 |
| A16 | s3 | FRESH_REPLAY | P3NEW2 | H010 | 96712 | 20,000 |
| B07 | s4 | FRESH_TRANSFER | fresh U + local Teacher A | BASE | 96611 | 100,000 |
| B08 | s4 | FRESH_TRANSFER | fresh U + local Teacher A | B_SELECTED | 96611 | 100,000 |
| B09 | s4 | FRESH_TRANSFER | fresh U + local Teacher A | B_SELECTED | 96612 | 100,000 |
| B10 | s4 | FRESH_TRANSFER | fresh U + local Teacher A | BASE | 96612 | 100,000 |
| C07 | s5 | FRESH_TRANSFER | fresh U + local Teacher A | NAT_MIXCAL | 96611 | 100,000 |
| C08 | s5 | FRESH_TRANSFER | fresh U + local Teacher A | PANMIX | 96611 | 100,000 |
| C09 | s5 | FRESH_TRANSFER | fresh U + local Teacher A | PANMIX | 96612 | 100,000 |
| C10 | s5 | FRESH_TRANSFER | fresh U + local Teacher A | NAT_MIXCAL | 96612 | 100,000 |

`B_SELECTED`는 실행값이 아니다. 두 core panel 분석 이후 고정된 우선순위/음성 시 E100 규칙으로 한 번만 bind하여 별도 decision JSON을 남긴다. bind되지 않은 template는 실행할 수 없다. 입력 source·seed·profile을 실행 이후 재명명하지 않는다.

### 9.1 승격 기준 — 운영용이지 통계적 유의성 판정이 아니다

Δ는 각 **로컬 같은 부모의 ALT−대응 대조**다. rE=E_ALT/E_CTL−1, 모든 지표는 같은 선택규칙의 동일 checkpoint다. 기준은 이번 계획의 운영값이며 과거 데이터로 검증한 보장선이 아니다.

**공동목표:** Exact endpoint에서 H>.964 및 E<.552. 보충 Table7/12의 .552를 운영 기준으로 유지하며 본문 .522는 strong goal로만 따로 남긴다 [S6]. 한 번 달성했다고 다른seed까지 성공으로 처리하지 않는다.

**SPATIAL_SIGNAL:** 두 core 부모에서 모두 ΔH>0, 평균ΔH≥.002, 두 부모 모두ΔDs<0, 평균rE≤+.005, 각 rE≤+.01, 각 ΔDλ≤+.001. VAL-selected에서도 한 부모라도 ΔH<−.001이면 `CHECKPOINT_SENSITIVE`로 표시하여 자동 fresh 승격하지 않는다.

**RR_PRESERVING_SIGNAL:** s3 후기 H010 전용 보조기준. 두 R100 부모의 평균 rE≤−.003, 각 ΔH≥−.001, 평균ΔDs≤+.0005. 이는 spatial breakthrough가 아니라 RR-preserving continuation 후보라는 뜻이다.

- s3: A_HI/A_LO에서 위 signal이 있으면 stream 재검정과 fresh replay를 우선한다. A_OLD만 성공하면 **reference-specific DEV**로 분리하고 R100 replay의 근거로 쓰지 않는다. A07–A10은 시간이 허용되면 불안정성 진단으로 수행할 수 있지만 실패를 숨기는 재추첨이 아니다.
- s4: K1/E100 중 SPATIAL_SIGNAL을 만족한 조건을 fresh transfer의 우선 후보로 한다. 둘 다 만족하면 **두 부모 중 더 작은 ΔH가 큰 조건**, 동률(1e−8 이내)이면 평균rE가 작은 조건, 다시 동률이면 K1로 잠근다. 둘 다 신호가 없으면 E100을 **초기경로 검정값**으로 잠그며 성공 승격으로 표시하지 않는다. train-only gate 진단을 같이 기록한다.
- s5: PANMIX−NAT_MIXCAL이 SPATIAL_SIGNAL을 만족하고, PANMIX−NATIVE0도 평균ΔH>0·각rE≤+.01이면 fresh transfer를 우선한다. NAT_MIXCAL만 좋아지면 augmentation 성공으로 보고하지 않는다. 신호가 없어도 첫 fresh pair의 시간·무결성·안전 입장은 허용하며, 후기와 초기경로의 음성/양성을 각각 기록한다.
- 새로운 fresh 결과는 이전 fork를 합쳐 좋은 평균을 만들지 않고 별도 표로 검정한다. fresh 첫 seed가 실패해도 이미 입장한 동일 pair는 모두 평가한다. 사전에 둘째 seed block이 입장했으면 첫 seed 순위 때문에 취소하지 않는다.

선택된 endpoint들의 test 점수를 이미 보았고 이 campaign의 승격에도 FR 결과가 사용되므로 전체는 **test-aware development**다. 새 seed 사용만으로 test set이 다시 untouched가 되지 않는다. 최종 논문 결과는 locked recipe·정한 seed 전체와 선택 경로를 보고하고 별도 holdout이 있으면 거기서 검증한다.

## 10. 20시간 운영·예약·축소

### 10.1 시계와 기존 작업 처리

새 GFB20은 기존 G20/L100과 별도 campaign이다. 처음 실제 준비·인계를 시작하는 순간 한 번 기록한 UTC t0를 세 서버에 공유한다. 문서를 읽거나 파일을 복사해 놓는 것만으로 clock을 생성하지 않는다. 실제로 진행 중인 작업의 drain·전환·부모 검증·cache·평가·전송 시간은 t0 이후라면 새20h에 포함한다.

기존 부모·평가 결과는 읽기 전용으로 보존한다. 진행 중 run을 kill하지 않고 안전한 case/checkpoint 경계에서 전환하며, 미완 L100은 그 campaign에 미완/보류로 남긴다. 기존 deadline을 새20h로 덮거나 부모 재학습 비용을 없던 것으로 하지 않는다. 서버 중 하나가 늦어져도 t0를 다시 시작하지 않는다.

| 시간 | 규칙 |
|---|---|
| t0~초기 구간 | source/runtime·parent/reference 검증; s5 새 cache; s3/s4는 다른 서버를 기다리지 않음 |
| 0~16h | core 우선; 결과와 예산 조건을 만족한 **전체 block**만 추가 입장 |
| **16h 이후** | 신규 block 금지. 이미 비용을 예약한 block의 다음 arm은 예약·완료 예상이 여전히 유효할 때만 진행 |
| **18h** | optimizer update 종료. 미완 상태는 fullstate 저장 후 PARTIAL_TIME_LIMIT; endpoint로 위장하지 않음 |
| **18~20h** | 평가 부채, parent/child 보존, 산출물 전송, Sheet readback, campaign 종결 |

### 10.2 초기 예약 추정

이전21:19 검토에 기록된100K Student optimizer 시간 약1.89~2.03h [S7]와 별도 evaluation/IO 여유를 바탕으로 아래 **보수적 입장 예산**을 둔다. 새로운 FT·augmentation 코드의 실제 측정치가 아니므로 완료를 보장하지 않는다. 20K에도 평가20회를 수행하며, 구현·환경·cache 비용은 실측으로 갱신한다.

| 서버 | 기본 6 FT | Stream 재검정 | Fresh / replay | 준비·진단·calibration | 18h 이전 예약합 | 최종 보존 포함 |
|---|---|---|---|---|---|---|
| s3 | 3.60h | 2.40h | 7.10h | 1.30h | 14.40h | 16.40h |
| s4 | 3.90h | 0.00h | 10.00h | 1.30h | 15.20h | 17.20h |
| s5 | 3.90h | 0.00h | 10.40h | 3.30h | 17.60h | 19.60h |

s5는 최대 확장 시 시간이 가장 빡빡하다. augmented cache가2h를 넘으면 기본 cache 범위를 줄이지 않고 **두 번째 fresh pair(C09/C10)**를 먼저 제외한다. 구현 검증이나 기존 작업 전환이 길어져도 같은 원칙이다.

입장 추정은 **max(표의 단위 예약 하한, 최근 동일 서버·동일 role/horizon/augmentation의 실제 총비용 추정×1.15)**를 사용한다. time/step만 쓰지 말고 eval·IO·diagnostic·calibration 비용과 기존 부채를 합산한다. 이 총량이18h까지 남은 시간보다 작고 최종2h 보존 공간을 남길 때만 block을 입장시킨다. 전체36개 완료를 맞추려고 후보 평가·q-cache·batch size를 줄이지 않는다.

**시간 부족 시 제외 순서:** 각 서버의 두 번째 fresh block → 첫 fresh block → s3 stream 재검정 → 아직 시작하지 않은 추가 부모 core block. 가능하면 최소 s3 A_HI, s4 B_PANEL1, s5 C_PANEL1의 **완결된 paired/triple 비교**를 확보한다. 시작한 block도 시간 상한에는 멈추되 어느 arm이 미완인지 명시한다.

다른 서버가 유휴라고 core 계수를 무단 이동하지 않는다. 추가시간을200K Teacher나 임의 seed 탐색으로 자동 소비하지 않는다. 시간·승격 기준에 못 미치면 부정 결과와 미실행 목록을 남기고 종료한다.

## 11. 평가·diagnostic·실패 보존

### 11.1 공식 선택점

- **FT20K:** 추가 step 1K,2K,…,20K의20개 후보. 원 parent(t=0)는 재평가 anchor이며 후보 경쟁에 포함하지 않는다.
- **Fresh100K:** 기존 고정50후보 `2020,4040,…,98980,100000`. 50K 등 진단 checkpoint를 후보 수에 끼워 넣지 않는다.
- **주 분석:** exact local endpoint와 RR-validation 최소ERGAS(동률은 더 이른 step). ALT와 CTL의 선택규칙이 같아야 한다.
- RAW 최고-H는 별도 test-aware 개발 진단. RAW의H와 최저RR의E를 합치지 않는다. 기존50후보 RAW_MAX와20후보 RAW_MAX를 동일 선택예산의 비교로 쓰지 않는다.
- train/val/RR/FR 각각의 sample 범위·normalization·crop·기존 clipping·sensor preset·source hash를 고정한다.
- HQNR는 장면별 곱을 계산한 뒤 평균한다. 평균Dλ와 평균Ds의 곱으로 공식 HQNR를 다시 만들지 않는다.

추가20K의 `local_step=20000`과 `lifetime_step=70000/120000`을 모두 기록하고, 이를 기존 Exact50K/Exact100K 열에 잘못 넣지 않는다. 기존 parent의 성능은 provenance이며 이번날 새 완료로 중복 등록하지 않는다.

### 11.2 고정 진단 표본과 비용 우선순위

train에서 seed1234로 고정128 base ID, 그중 첫24개 gradient probe를 쓴다. calibration3072와의 overlap은 기록한다. source에 저장된 동일 ID 집합이 있으면 검증 후 재사용한다. 새 선택기나 sampling class를 만들지 않는다.

FT t=0/1K/5K/10K/20K에서 아래 항목을 기록하되 무거운 gradient 전체 측정은0/5K/20K만 한다. RNG와 training/eval mode를 보존해 진단이 학습 stream을 바꾸지 않게 한다.

| 항목 | 해석 |
|---|---|
| c_T, c_S, c_S−c_parent, cosine/norm/공통 bias | Teacher→Student 보정 유지 및 FT 조정. **변위 GT로 부르지 않음** |
| weighted hard/soft/edge gradient norm·cosine | 실제 supervision 비중과 방향 |
| eT/eS·dT·aT·s, soft active mass | 어떤 reference 신호가 실제 활성인지 |
| 1px·4px border / interior error와 hard 기여 | warp/augmentation 경계 부작용. loss mask는 바꾸지 않음 |
| gamma별 PAN/LP/H 에너지·범위, q/tau 분포 | 증강이 의도한 입력 차이를 만들었는지 |
| signed Ds: 장면20×밴드4 operand | 높은H가 어떤 관계 변화에서 나왔는지; 평균·양수율·재구성 오차 |

s3 P3HI/P3LO와 s5 PANMIX의 t=0/end에서 가능하면 동일 입력·고정 c를 쓰는 sensitivity replay를 별도 진단으로 수행한다. **공식 추론에서 c를 교체하거나 gain을 고르는 용도가 아니다.** 독립 정합 추정기는 검증된 기존 구현·표본이 있을 때만 사용하고, 없으면 `ESTIMATOR_NOT_AVAILABLE`로 남긴다. 이를 새20h 구현의 필수 blocking task로 만들지 않는다.

P0 필수는 실제 model/reference/data 무결성·routing·native 평가·finite 여부다. P1의 무거운 원 추정기 분석이 지연되면 checkpoint를 보존하고 뒤로 미룬다. 단 B의 gradient-scale 진단과 C의 view/q 일치는 각 축의 해석·무결성에 필수다.

### 11.3 중단 규칙

- 잘못된 input/output shape, 다른 Teacher/q, native γ1 parity 실패, routing leakage, NaN/Inf는 해당 arm/block의 `BLOCKED_INTEGRITY` 또는 `FAILED_NUMERICAL`이다. 모듈을 끄거나 다른 cache로 대체하지 않는다.
- FT에서 full native validation ERGAS가 원 parent보다 **10% 이상 두 연속 평가점에서 악화**하면 `ABORTED_VAL_DIVERGENCE`로 종료할 수 있다. 안전 규칙은 BASE/ALT에 똑같이 적용하며 해당 실패를 제외한 평균을 만들지 않는다. FR HQNR를 조기종료 트리거로 쓰지 않는다.
- gradient clipping·AMP·batch 감소를 한 arm만 켜서 구제하지 않는다. 수정이 필요하면 모든 관련 arm에 새 revision을 적용해 다시 정의한다.
- 인프라 오류는 같은 fullstate/RNG로 최대2회 재시도한다. seed를 바꾼 재시도는 새로운 과학적 run이며 이 계획에 없다.
- 업로드만 실패했으면 학습을 다시 하지 않고 저장 결과의 업로드·readback만 재시도한다.
- 승리한 arm이 나와도 사전예약한 나머지 대조를 삭제하지 않는다. 음성·상충·미완 결과를 함께 저장한다.

## 12. 결과 기록과 인계

가능하면 별도 `GF2-B20-s3/s4/s5` 영역을 만들거나 기존 탭에 구분된 schema로 append한다. 이전 G20/L100 행과 header를 덮어쓰지 않는다. Sheet 쓰기 권한이 없으면 로컬 CSV/JSON을 원값으로 보존하고 `UPLOAD_PENDING`으로 둔다.

필수 열:
`campaign_id, case_id, server, stage, parent_run_id, parent_step, parent_model_sha, parent_teacher_sha, reference_id, native_or_mixcal_id, source_commit, data_sha, stream_seed, fresh_seed, init_mode, optimizer_reset, local_updates, lifetime_updates, alpha/beta/edge+ramp, LR/scheduler, gamma_view, tau_R, q_ref, q_cache_sha, EXACT_FINAL H/E/Dlambda/Ds, RR_VAL_SELECTED step/H/E, RAW_AUX, training_hours, evaluation_hours, io_hours, calibration_hours, wall_hours, parent_compute, status, readback_status`.

종결 산출물:
1. `completed_cases.csv`, `paired_effects.csv`, `failed_or_deferred_cases.csv`.
2. 각 source anchor 및 모든 end/best-validation checkpoint의 immutable manifest·SHA.
3. `runtime_binding.json`, `frozen_protocol.json`, `source_and_data_manifest.json`.
4. s5 `augmentation_manifest.json`, `mixed_calibration.json`, 전체q cache와 native slice parity.
5. `promotion_decision.json`, B_SELECTED 결정·근거·시각, admit/stop/부채 ledger.
6. 20h 요약: 공동목표 여부·시드별 paired 변화·학습경로/계수/입력 효과 구분·실제 비용.

## 13. 구현자 체크리스트 — 기존 L100을 그대로 실행하지 말 것

이 번들은 설계 registry이지 이미 지원되는 trainer config가 아니다. 기존 L100의 fresh50K/100K와 fixed profile guard를 몰래 우회하는 대신 **새 GFB20 namespace**를 구현한다. 아래 변경은 이 문서에서 설계한 사항이며 GPU에서 아직 검증하지 않았다.

- [ ] source snapshot을 먼저 잠그고 s1/s2 및 기존 L100 numerical path에 영향 없음 확인.
- [ ] FT가 부모 U와 A **둘 다** 보존, optimizer 둘 다 reset, new paired stream인지를 unit test.
- [ ] BASE/H010/K1/E100 수치식과 ramp0/1K/end, scheduler0/100/N 검증.
- [ ] grad(U)=LU, grad(A)=LA 및 Teacher frozen, soft/edge→A 직접gradient0 검증.
- [ ] 표36개/CORE18개 run ID·부모·seed·dependency·18h 입장 비용 검증.
- [ ] gamma1 bitwise native, symmetricG 상수입력 보존·좌표 이동 없음, augmented LP correspondence 검증.
- [ ] q key [ID,ROT,gamma], Warp(Pγ,ε) 순서, weighted median token복제, native/mixcal 두 control을 test.
- [ ] fresh replay/transfer와 FT의 lineage 및 shared-trunk 비용 구분.
- [ ] pause/restart fullstate의 optimizer·RNG·sampler·candidate 평가 parity 검증.
- [ ] parent 재평가와 source/runtime tolerance 검증, exact checkpoint SHA binding.
- [ ] 20시간 deadline, 16h admission cutoff, 18h optimizer cutoff, pair 부채·실패 보존 테스트.
- [ ] Sheet append/readback은 mock부터 검증. 같은 숫자의 표시 자릿수를 원값 변경으로 구현하지 않음.

부모 bytes와 구현이 준비되지 않은 경우 상태는 `READY_FOR_IMPLEMENTATION/TO_BIND_ASSETS`다. **문서 작성만으로 기동됐다고 보고하지 않는다.** 이 계획의 core가 끝나면 추가 방향은 실제 paired 결과로 정하며, 기법을 자동으로 계속 늘리지 않는다.

## 14. 근거 자료와 파일 목록

- **[S1]** `sources/GF2_L100_NearCompletion_Analysis_and_Bolder_Directions_20260921_2309KST.md`: §2–8, 기존 결과·세 가지 후속 가설. 해당 문서의 cutoff·증거 수준을 유지했다.
- **[S2]** `sources/GF2_L100_selection_metrics.csv`, `sources/GF2_L100_paired_deltas.csv`: reported endpoint·선택점·checkpoint SHA. 실제 로컬 bytes 확인은 별도다.
- **[S3]** 사용자 첨부 `PAN_Final_Method_qe_AlignmentAware_Fitting_2026-09-20_KR_v4_notation.md`: §4–5의 hard/soft/q/routing.
- **[S4]** `PANDA_Method_v4_Architecture_Conformance_Audit_2026-09-21.md`: calibration subset, edge 방향평균, advantage stop-gradient의 구현 세부. 전체train median으로 바꾸지 않았다.
- **[S5]** 기존 `PANDA_GF2_L100_S345_LOCALT_20H_ExperimentPlan_2026-09-21_v2.md`: 로컬reference·data·평가 계약. 이번 캠페인은 Teacher-first를 반복하는 새 L100이 아니다.
- **[S7]** `sources/GF2_L100_Results_and_Next_Direction_20260921_2119KST.md` §2: 서버별100K 학습시간. 새 FT/증강 시간은 이 자료의 직접 측정치가 아니다.
- **[S6]** PAN-Crafter 첨부본 p13 Table7 / p14 Table12: GF2 H=.964, E=.552, Ds=.017. 본문 Table2의 E=.522와 구분한다.

동봉: 본 MD, `GFB20_Cases_All36.csv`, `GFB20_Cases_Core18.csv`, `GFB20_DesignRegistry.json`, `GFB20_ParentAssets.json`, `GFB20_Budget.csv`, `GFB20_Queue_s3/s4/s5.json`, sources4개, `SHA256SUMS.txt`. Runtime path/clock/미확인SHA의 null은 실험을 시작할 때 검증해 채울 값이며 fabricated placeholder 경로가 아니다.
