# Student sampling 구조 실험 및 PAN warp 교체 검토 — 2026-10-23

현재 판단은 **bicubic 교체를 검증할 충분한 근거가 있지만, FFT로 바꾸면 목표 성능을 달성한다는 근거는 아직 없다**는 것이다. Student 구조 변경은 실행 큐에 추가한다. Warp 후보는 독립 구현과 CPU 수치 검증까지 진행하며, 현재 학습 코드에 바로 대입하지 않는다. 밴드 가중 loss와 rotation consistency loss는 제외한다.

## 1. 승인된 구조 실험

캠페인: `PANDA_S13_SAMPLING_HARDONLY_20261023_v1`, run prefix `PSA30`.

| Case | 두 down 위치 | 두 up 위치 | Student 파라미터 수 |
|---|---|---|---:|
| 기존 S020C30 control | conv3/stride2/pad1 | transposed conv2/stride2 | 2,658,888 |
| DAVG | average pool2/stride2 → conv3/stride1/pad1 | 기존 유지 | 2,658,888 |
| UBIL | 기존 유지 | bilinear ×2/align_corners=False → conv3/pad1 | 2,784,328 |
| DAVG_UBIL | DAVG | UBIL | 2,784,328 |

s1·s3 각각 `DAVG R1 → UBIL R1 → DAVG_UBIL R1 → DAVG R2 → UBIL R2 → DAVG_UBIL R2`, 총 6개다. 대조군은 **동일 서버의 현재 schedule 캠페인 S020C30 R1/R2**다. 대조군 가중치는 가져오지 않고 모든 Student를 새로 초기화한다. 공통 이름과 모양의 가중치는 기존 seed/role 초기화 값을 유지한다.

- PAN Aligner scale .2, Hard α=.25/γ=.5/β=0, peak LR=1e-4, warmup=100, cosine horizon=30K, floor=0.
- 기존 C4 입력 augmentation 유지. rotation loss, 회전 평균 추론, 새 밴드 loss 없음.
- Teacher/Aligner는 동일 서버의 기존 mature RRE 선택 checkpoint를 동결한다. Teacher 구조·warp·입력 경로 유지.
- Student PAN warp도 기존 bicubic 유지. 구조 효과와 warp 효과를 한 case에 섞지 않는다.
- GT·평가 기준은 원래 좌표의 RR20/FR20, 단일 forward다.
- 현재 schedule 큐의 정상 완료와 두 대조군 완료를 확인한 뒤 시작한다. 종료 신호나 추가 skip을 보내지 않는다. 총 wall-time 제한은 없다.
- Sheet는 full precision 동일 상태의 HQNR≥.9585 및 ERGAS≤2.040 통과 결과 전체와 대표 결과를 게시한다. 대표는 HQNR≥.9585 중 ERGAS 최소, 없으면 HQNR 최대다. 같은 checkpoint는 중복 게시하지 않는다.

이 구조가 엄밀한 C4 equivariance를 보장하지는 않는다. 학습 가능한 conv kernel은 방향 제약이 없고, bilinear는 세부 정보를 완화할 수도 있다. UBIL 계열의 파라미터 수는 +4.72%이므로 결과 해석과 비용 게시에 이를 포함한다. `4×4 conv/stride2/pad1`도 가능한 후속 후보지만 학습된 4×4 tap 자체의 회전 대칭을 보장하지 않으며 이번 세 case에는 포함하지 않는다.

실행과 s3 인계는 [Sampling30 README](../panda_sampling30/README.md), [s3 handoff](../results_log/2026-10-23_SAMPLING30_s3_handoff.md)를 따른다. 새 구조의 CPU 학습·resume 검증은 수행했으며, 실제 데이터 CUDA 검증과 GPU 비용 측정은 기존 큐 완료 후 자동 실행된다. 새 구조의 학습 성능은 아직 없다.

## 2. 제공된 warp 진단의 재검토

입력 증거는 사용자가 작성한 `results_log/2026-10-23_scale30-hardtune-interim.md` §7, `tools/warp_error_probe.py`, 원자료 `results_log/assets/2026-10-23_s1_warp_error_probe.json`이다. 이 파일들은 수정하지 않았다. 새 검토 결과는 [CPU audit JSON](../results_log/assets/2026-10-23_warp_candidate_audit.json)에 별도로 저장한다.

### 확인되는 내용

1. bicubic의 소수 위치 보간은 PAN의 고주파를 크게 바꾼다. 원자료에서 native ERGAS 2.048222 대비 왕복 시험은 최대 +16.0675%였다. 이 정도 차이는 warp 교체를 별도 실험할 충분한 이유다.
2. 같은 자료의 FFT 왕복 오차는 훨씬 작다. 따라서 **새 Student를 처음부터 새 warp로 학습하고 같은 warp로 평가**하는 실험의 우선순위가 높다.
3. frozen Teacher의 원래 입력 경로를 바꾸면 별도의 분포 변화까지 생긴다. Stage2 Student warp만 바꾸는 실험과 Stage1부터 바꾸는 실험은 분리해야 한다.

### 보정해야 할 해석

- 표의 이동량은 대각 벡터 크기다. 2px는 축별 1.4142px이고 정수 이동이 아니다. 1.5px도 축별 1.06066px로, bicubic의 ERGAS 증가는 0이 아니라 +0.06418%다.
- FFT의 해당 2px 왕복 ERGAS 증가는 **+0.08334%**다. 작지만 0 또는 항상 ≤.07%라고 쓰면 정확하지 않다.
- Keys bicubic(a=-.75)은 순수 저역통과만 수행하지 않는다. 반 픽셀에서 f=.10/.15의 진폭은 각각 1.019/1.029로 약간 증폭되고, 높은 주파수에서는 크게 감쇠한다. 음의 tap에 따른 ringing도 있다.
- 원 probe의 `hf_energy_retained`는 norm 비율이다. 엄밀히는 제곱 에너지 비율과 구분해야 한다.
- 원 probe는 shift를 Python 숫자로 변환한다. FFT 수식의 미분 가능성과 그 probe에서 Aligner gradient가 연결된다는 주장은 다르다. 새 tensor 구현은 image/shift gradcheck를 별도로 통과했다.
- 현재 FFT는 reflect-pad → 위상 곱 → inverse FFT의 실수부 → crop이다. 정수부를 별도 무손실 roll하는 구현은 아니다. crop 후 재-padding하므로 두 호출의 왕복은 원래 복소 FFT 연산의 정확한 역연산이 아니다.
- **왕복 안정성은 한 방향 이동의 실제 정확도와 다르다.** 실제 연속 신호를 아는 sine 시험, 경계 시험, 실제 PAN 성능을 따로 평가해야 한다.
- bicubic으로 학습한 Student의 추론만 FFT로 교체한 악화는 입력 분포 차이와 일관된다. 하지만 이것만으로 HALF/LOW의 HQNR 차이가 전적으로 기하 때문이라고 분리해 증명할 수는 없다. 기하·보간·학습 적응이 함께 바뀌었다.

실수 Nyquist 성분에는 추가 주의가 필요하다. 예를 들어 alternating signal을 반 픽셀 이동하면 실수 샘플에서 quadrature 정보가 사라질 수 있다. 복소 위상을 보존하면 다른 문제이며, 신호 prior를 추가하면 복원 가정도 달라진다. 따라서 "어떤 방법으로도 무조건 복원 불가"보다 "현재 실수 임계 샘플링만으로 Nyquist 위상을 유일하게 결정할 수 없다"가 정확하다. 2D FFT 후 `.real`을 취하는 방식은 Nyquist 교차점에서 축별 순차 실수 처리와도 다를 수 있어 C4 대칭·shift 합성까지 자동 보장하지 않는다.

## 3. FFT 외 추가 후보

아래 진폭은 **합성 신호, 1축 반 픽셀 이동의 CPU 수치 검증**이다. 1에 가까울수록 해당 주파수의 진폭을 유지한다. 학습된 PAN 모델의 HQNR/ERGAS 수치가 아니다.

| 방식 | f=.375 진폭 | f=.45 진폭 | step overshoot | 해석 |
|---|---:|---:|---:|---|
| bicubic | .628 | .271 | 9.38% | 현재 기준 |
| Lanczos-4, 8 tap | .901 | .451 | 11.89% | 유한 지지영역, 기존 제안 |
| Lanczos-6, 12 tap | 1.018 | .640 | 12.89% | 더 높은 고주파 보존, 일부 증폭 |
| Kaiser sinc, 12 tap, β=6 | .963 | .554 | 12.48% | window 파라미터로 평탄도/감쇠 조절 |
| quintic spline, prefilter | .910 | .538 | 11.99% | cubic spline보다 좋은 진폭 보존 후보 |
| FFT, reflect pad32 | 이상적 주기 복소 신호에서 1 | 같은 조건에서 1 | 13.66% | 실제 유한 영상은 경계·실수화 오차가 있음 |

Step overshoot는 같은 합성 단차에 대한 측정이며 영상 전체의 평균 ringing을 뜻하지 않는다. FFT 진폭 칸은 이상적인 수학적 성질이므로 유한 커널의 실제 측정값과 동일한 실험 결과로 해석하지 않는다. 실제 유한 sine 시험의 전체/내부 RMSE는 audit JSON에 있다. 예를 들어 f=.45에서 FFT pad16/32/64의 내부16 RMSE는 .0280/.0334/.0276으로, padding을 늘린다고 단조 개선하지도 않았다.

**Kaiser-windowed sinc**는 추가 검토 가치가 크다. 다만 window를 바꾸기만 하면 항상 좋아지지는 않는다. 같은 검증에서 Kaiser 8 tap/β6의 f=.45 진폭은 .386으로 Lanczos-4의 .451보다 낮았다. tap 수와 β를 함께 명시해야 한다. [원 연구](https://www.cg.tuwien.ac.at/research/vis/vismed/Windows/)도 Kaiser/Gaussian window의 형상 파라미터 선택을 다룬다. 이 연구는 PAN 성능 향상을 입증하는 자료는 아니다.

**Quintic spline(order=5)**은 추가 offline 기준점으로 측정했다. 계수 prefilter가 필요하고 그 단계는 국소 tap만의 연산이 아니다. SciPy 참조 구현을 바로 Aligner 학습의 gradient 경로에 넣을 수는 없으며 별도 미분 가능한 구현이 필요하다. [SciPy 공식 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html)는 0–5차 spline, prefilter, 서로 다른 `reflect`/`mirror` 경계 규약을 명시한다.

**DCT 기반 sinc 보간 또는 전체 even extension**은 FFT 경계 artifact가 지배적일 때의 후속 대안이다. [Yaroslavsky의 원 논문](https://opg.optica.org/ao/abstract.cfm?uri=ao-42-20-4166)은 전역·국소 DCT 기반 경계 처리 보간을 제시한다. DCT 계수에 FFT처럼 단순 위상만 곱하면 되는 방식은 아니며, 이번에는 구현하지 않았다. 현재 자료로 이 방식이 PAN에서 더 좋다고 순위를 확정하지 않는다.

Nearest/bilinear는 보간 손실의 하한·상한을 단순 비교하는 control로는 유용하지만 현재 고주파 보존 문제의 주 후보로 추천하지 않는다. Learned resampler/deformable conv는 새 학습 모듈 효과가 섞이므로 이 비교에서는 제외한다.

## 4. Warp 학습 실험을 추가하기 전 확정할 사항

독립 후보 모듈 `panda_warp_candidates`는 production에서 import하지 않는다. 구현된 후보는 FFT, Lanczos-4/6, Kaiser12/β6이며, 입력은 BCHW와 B×2 tensor shift다. 부호는 기존처럼 `output[y,x]=input[y+dy,x+dx]`다. 채널별 이동도 각 밴드를 batch로 묶으면 같은 연산을 적용할 수 있지만, 현재 학습 모델은 global PAN shift이고 bandwise Aligner로 바꾸지 않는다.

1. **Support/mask 규약:** 기존 `warp_support_mask`는 bicubic의 4 tap을 가정한다. Lanczos6/Kaiser12에는 12 tap에 맞춘 support가 필요하다. FFT는 전역 지지영역이므로 기존 mask로 "모든 참조 픽셀이 원영상 안"이라고 증명할 수 없다. 고정 loss ROI를 쓰더라도 그것은 선언된 학습 영역이지 FFT의 유한 support 증명이 아니다.
2. **경계·크기 규약:** RR train64, RR256, FR512에서 같은 reflect 정의와 padding 정책을 정한다. 단순 PyTorch reflect pad64는 64×64 train patch에 불가능하다. 먼저 모든 크기에 적용 가능한 pad16/32를 비교하고 더 큰 확장이 필요하면 반복 반사 등 별도 규약을 명시한다. 평가의 GT와 공식 RR/FR 영역을 유리하게 바꾸지 않는다.
3. **Nyquist 규약:** 현재 후보는 probe와 같은 2D 실수 projection이다. 후속 학습 후보에서는 Nyquist 행/열의 축별 real-compatible 처리와 현재 방식을 비교해 하나를 명시적으로 고정해야 한다. 실수화 손실을 숨겨 무손실이라고 하지 않는다.
4. **Teacher와 calibration:** frozen Teacher는 원래 full-u bicubic 경로를 유지한다. Student PAN/q 계산에 새 warp를 쓴다면 그 warp·mask로 train-only conditional-q calibration을 다시 계산하고 새 binding에 기록한다. 기존 bicubic q 분포를 그대로 이름만 바꾸지 않는다.
5. **새 학습과 추론 일치:** Student를 fresh로 학습한다. 보정·어긋남 주입을 함께 쓰는 후속 실험은 두 연산의 종류와 경계 정책도 일치시킨다. 최종 영상만 필요할 때는 global shift를 합성해 한 번 resample하는 방안도 비교할 수 있다. Aligner가 주입 영상을 관측하는 실험에서는 중간 영상 생성 자체가 필요하므로 무조건 한 번으로 대체할 수는 없다.
6. **선택 기준:** 왕복 오차뿐 아니라 알려진 sine의 한 방향 오차, overshoot, image/shift gradient, 경계 artifact, 실제 GPU 시간·메모리, 동일 상태 HQNR/ERGAS를 함께 본다. 시험 세트 지표를 보며 선택한 결과는 exploratory로 표기한다.

후속 case 제안은 `WBASE / WFFT32 / WLANCZOS6 / WKAISER12B6`이다. 기존 Student 구조, scale=.2, 30K schedule, Hard 계수, donor, seed를 고정하고 warp만 바꾼다. WBASE는 새 support/경계 실험 규약이 기존과 다르면 반드시 새로 실행한다. `WFFT16`은 pad 민감도가 실질적일 때 추가한다. 각각 R1에서 안정성과 양 지표를 확인하고 동일 규약으로 R2를 비교한다. **이 목록은 제안이며 학습 큐에는 아직 등록하지 않았다.** 현재 등록된 것은 위 구조 실험 6개/서버다.

Stage1부터 FFT를 넣는 것은 별도 후속 실험이다. 새 warp에서 Aligner+Teacher를 다시 학습하고 동일 val RRE 규칙으로 donor를 선택한 뒤 Stage2까지 이어야 한다. 기존 Teacher checkpoint의 추론 warp만 교체한 것을 Stage1 FFT 학습으로 표시하면 안 된다.

"Aligner가 정수 shift로 쏠린다"는 가설은 아직 미검증이다. raw `u`의 축별 소수부와 model-free offset을 같은 좌표·patch 집합에서 비교하고, phase별 정합 오차/Teacher loss를 함께 기록할 수 있다. `scale*u`만 보면 scale 자체가 0 근처로 모으는 효과가 섞인다. 관측 분포 차이는 단서이며, 원인을 확인하려면 warp만 바꾼 Stage1 재학습 대조가 필요하다.

## 5. 재현

```bash
OMP_NUM_THREADS=1 /home/knuvi/miniconda3/envs/pancrafter/bin/python -m unittest discover -s panda_sampling30/tests -v
OMP_NUM_THREADS=1 /home/knuvi/miniconda3/envs/pancrafter/bin/python -m unittest discover -s panda_warp_candidates/tests -v
OMP_NUM_THREADS=1 /home/knuvi/miniconda3/envs/pancrafter/bin/python tools/warp_candidate_audit.py --out /tmp/warp_candidate_audit.json
```

s3에서는 기존 환경의 Python을 사용한다. 원래 PAN probe JSON이 없는 checkout에서도 합성 수치 검증은 실행되며, 원자료 재검토 부분의 availability는 별도로 기록된다. 새 GPU 학습 결과는 아직 없고, 현재 문서의 warp 결과는 원자료 재계산과 CPU 합성 검증에 한정된다.
