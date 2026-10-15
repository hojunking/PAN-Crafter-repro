**PAN aligner의 입력 크기·pixel 단위·보정 필요성 검증 — 2026-09-18**

검증 시점 11:23 KST, HEAD `3a99515d4e4ff10525633915de2d353902883b05`. 현재 코드와 기존 N2/T0 probe CSV를 대조하고, CPU에서 좌표 변환 및 checkpoint의 가변 크기 forward를 확인했다. 신규 학습, 실영상 probe 재평가, GPU 추론, 기존 실험 결과 수정은 하지 않았다.

**인용된 설명의 “64→256이라는 이유로 c에 ×4를 할 필요가 없다”는 부분은 맞다. 현재 sampler에 필요한 변환도 이미 구현되어 있다. 다만 크기 일반화 전체가 검증됐다거나 synthetic shift 반응만으로 원래 PAN–MS의 절대 정합이 정확하다고 판단하는 것은 근거를 넘는다.**

| 확인할 주장 | 판정 | 근거·제한 |
|---|---|---|
| 256/512에서도 동일한 aligner 구조를 사용할 수 있다 | 확인 | convolution → GAP → 고정 차원 Linear. 실제 N2/T0/G23 checkpoint로 크기별 forward 통과 |
| 입력 한 변이 64→256이면 출력 c를 ×4 해야 한다 | 불필요 | c는 현재 PAN grid의 pixel 단위로 학습·사용됨. 입력 영역 크기가 늘었다고 pixel 단위가 바뀌지는 않음 |
| sampler는 현재 H,W로 normalized grid를 만들어야 한다 | 맞고, 이미 구현됨 | `gx=2(x+dx+0.5)/W−1`, `gy=2(y+dy+0.5)/H−1` |
| N2의 인용된 64/256/512 반응 수치 | 재현 | **N2 last**의 기존 CSV 회귀값과 일치. N2 best_hqnr의 결과는 아님 |
| 이로써 현재 모든 checkpoint의 크기 일반화·절대 정합 정확도가 확인됐다 | 미입증 | checkpoint·데이터 분포가 다르며, 추가 변위 반응은 native offset 편향을 식별하지 못함 |
| 기존 학습 모델에서 PAN warp를 유지해야 하는가 | 현재 비교에서는 유지가 타당 | 앞선 로컬 s1 13개 ON/NOA 비교에서 제거 시 RR·FR 모두 악화. 이는 별도의 경험적 근거 |

**구조와 좌표 변환**

[`PANGlobalAligner.forward`](../pa/aligner.py)는 PAN과 확대된 MS를 convolution으로 처리한 뒤 공간 평균 `x.mean(dim=(2,3))`을 적용한다. 결과는 크기와 관계없이 `[B,64]`이며 마지막 Linear가 `[B,2]=(dy,dx)`를 출력한다. 입력 전체를 고정 길이로 flatten하는 구조가 아니다.

현재 N2/T0/G23의 margin4 경로를 직접 확인한 결과는 다음과 같다. 각 checkpoint마다 아래 다섯 크기를 넣어 총 15회 확인했다. random 입력을 쓴 **shape 확인**이며 이 출력의 정합 정확도를 평가한 것은 아니다.

| 원 PAN H×W | Aligner 입력 view | 마지막 공간 feature | GAP | 출력 |
|---|---|---|---|---|
| 64×64 | 56×56 | 64×14×14 | 64 | dy,dx |
| 128×128 | 120×120 | 64×30×30 | 64 | dy,dx |
| 256×256 | 248×248 | 64×62×62 | 64 | dy,dx |
| 512×512 | 504×504 | 64×126×126 | 64 | dy,dx |
| 256×320 | 248×312 | 64×62×78 | 64 | dy,dx |

모든 출력은 finite였고, 검사 전후 aligner state도 동일했다. margin crop은 pixel을 다시 샘플링하지 않으므로 `64/56` 같은 크기 비율도 c에 곱하지 않는다. [`PAModel.predict_delta/forward`](../pa/model.py)는 이 c를 그대로 원 PAN의 `warp_pan`에 전달한다. MS에 적용하는 `bicubic ×4`는 LRMS→PAN 크기의 확대이고, c를 ×4 하는 연산이 아니다.

[`warp_pan`](../pa/warp.py)의 실제 식은 다음과 같다.

\[
\tilde P[y,x]=P[y+dy,x+dx],\quad
g_x=\frac{2(x+dx+0.5)}W-1,\quad
g_y=\frac{2(y+dy+0.5)}H-1.
\]

`align_corners=False`, bicubic, border padding과 일치하는 좌표식이다. 따라서 normalized grid에서의 **추가량**은 `2dx/W`, `2dy/H`다. c=(0.5,−0.3)일 때 다음 값이 된다.

| 입력 크기 | Δg_y | Δg_x | 실제 sampling 위치의 이동 |
|---|---:|---:|---|
| 64 | 0.015625 | −0.009375 | dy=0.5, dx=−0.3 px |
| 256 | 0.00390625 | −0.00234375 | dy=0.5, dx=−0.3 px |
| 512 | 0.001953125 | −0.001171875 | dy=0.5, dx=−0.3 px |

실제 `grid_sample`에 전달되는 grid를 CPU에서 포착하고 원 pixel 좌표로 역변환했다. 다섯 shape 전체에서 지정한 c와의 최대 오차는 **1.83×10⁻⁵ px 이하**였다(FP32). 별도로 impulse에 `(dy,dx)=(1,−2)`를 적용해 출력 peak가 `(y−1,x+2)`에 나타나는 것도 모두 확인했다. 이는 현재 코드가 sampling 위치 부호를 사용하는 것과 일치한다. 별도의 해상도 비율 보정이나 sampler 코드 수정이 필요하다는 증거는 없다.

여기서 **입력 영역 크기 변화와 영상 resize는 구분해야 한다.** 같은 pixel 간격으로 64 crop 대신 256 영역을 받는 경우 c에 자동 ×4를 하지 않는다. 반면 같은 장면 자체를 4배 resize하고 이전 grid에서 측정한 변위 숫자를 새 grid로 옮겨 쓴다면, 동일한 물리적 변위의 pixel 표현에는 배율 변환이 필요하다. 현재 forward는 다른 grid의 c를 가져오는 경로가 아니라 현재 입력에서 c를 직접 예측하는 경로다. RR·FR 간 pixel 단위가 각각 정의된다고 해서 같은 물리적 어긋남의 pixel 수까지 같다는 뜻은 아니다.

**인용된 N2 수치의 출처와 재계산**

대상은 `PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT/last`다. 원본 [`po10_diag_last.json`](../work_dir/PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT/results/po10_diag_last.json)과 `offset_response_{native64,rr256,fr512}.csv`에서 `cε−c0 ≈ Bε+b`를 다시 least-squares fitting했다. 저장된 B와 재계산 B의 차이는 세 scale 모두 0이었다.

| N2 last | sample 수 | B_yy | B_xx | 해석 |
|---|---:|---:|---:|---|
| native64 | 64 | −0.984238 | −0.987049 | 추가 이동에 거의 −1배 반응 |
| RR256 | 20 | −1.001495 | −0.993568 | 큰 입력에서도 반응 유지 |
| FR512 | 20 | −0.912575 | −0.952281 | 반응은 유지되나 완전히 동일하지 않음 |

따라서 인용된 반올림 수치 −0.98/−0.99, −1.00/−0.99, −0.91/−0.95는 맞다. FR에서는 대각 반응 크기가 이상적인 1보다 약 8.7%/4.8% 작고 cross-axis 반응도 있으므로 “모든 크기에서 완전히 같음”이라고 하면 안 된다. 그렇다고 이 차이가 ×4 같은 크기 배율 누락을 뜻하지도 않는다. probe 회귀 기울기의 역수를 native c에 곱해도 절대 정합이 맞는다는 보장은 없다.

중요한 조건은 **last checkpoint**라는 점이다. 같은 run의 [`po10_diag_best_hqnr.json`](../work_dir/PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT/results/po10_diag_best_hqnr.json)은 다음처럼 다르다.

| N2 best_hqnr | B_yy | B_xx |
|---|---:|---:|
| native64 | −0.069170 | −0.559511 |
| RR256 | −0.071171 | −0.488474 |
| FR512 | −0.040060 | −0.280464 |

즉 “N2라는 이름의 aligner면 반응이 거의 −1”이 아니라 특정 학습 상태의 결과다. 오래된 두 PO10 JSON에는 checkpoint hash가 없다. 그래서 hash가 있는 후속 PALSV18 진단도 교차 확인했다. N2 last hash `9d4cbf218cd8f674…`가 현재 파일과 일치했고, 그 진단의 대각 반응은 64 `(−0.984178,−0.984977)`, 256 `(−1.000501,−0.990898)`, 512 `(−0.909388,−0.941979)`였다. probe 구성 차이 때문에 숫자는 조금 다르지만 같은 관찰을 뒷받침한다. 오래된 PO10 결과에 당시 없던 hash를 소급하여 있다고 취급하지 않았다.

또한 이 결과는 동일 scene의 영역만 64→256→512로 바꾼 통제 실험이 아니다. [`tools/po10_diag.py:datasets`](../tools/po10_diag.py)는 native64를 validation split에서 고른 64 patch로, RR256/FR512를 각각의 test 파일로 구성한다. **sample·내용·전처리/공간 해상도 차이와 크기 차이가 섞여 있다.** 따라서 기존 결과는 각 domain에서의 작동 근거이지, 크기 효과만 완전히 분리한 근거는 아니다.

**현재 Teacher에서도 크기 변환 오류와 학습된 반응은 구분된다.**

T0인 `PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1/best_hqnr`의 PALSV18 CSV도 같은 방식으로 재계산했다. checkpoint hash `16b5cf78614be121…`도 일치했다. 아래는 N2와 같은 PALSV18 probe 집합의 global fit이다.

| 입력 | N2 last B_yy / B_xx | T0 selected B_yy / B_xx |
|---|---|---|
| native64 | −0.9842 / −0.9850 | −0.3273 / −0.3773 |
| RR256 | −1.0005 / −0.9909 | −0.3361 / −0.3676 |
| FR512 | −0.9094 / −0.9420 | −0.3428 / −0.3242 |

T0는 256에서만 반응이 약해지는 것이 아니라 64에서도 이미 약하다. “64에서 학습한 출력 단위를 256에 잘못 적용했다”는 설명보다, **해당 학습 상태의 변위 반응이 원래 약하다**는 관찰이 데이터에 맞는다. 이것을 sampler 배율 수정으로 해결할 근거는 없다. 이번 표는 fixed probe와 random probe를 함께 fit한 global B이며, 앞선 보고서의 fixed-probe scene 평균 B와 숫자가 조금 다른 것은 집계 정의 차이다.

현재 G23은 가변 크기 forward 자체는 통과했다. 그러나 앞선 분석에서 확인한 실영상 q_size는 G23/1234의 RR 세 scene pilot뿐이었다. 이 pilot은 원래 pixel scale을 유지한 같은 scene의 crop/full 비교라는 점에서 과거 domain별 probe와 다르지만, 모든 G23 seed·FR512의 정합 일반화까지 채우지는 못한다. N2의 좋은 반응을 미세조정된 T0나 Student에 그대로 전이할 수는 없다.

**synthetic shift로 확인되는 것과 확인되지 않는 것**

`cε−c0≈−ε`는 알려진 추가 변위에 대한 보상 반응을 검사한다. 그러나 어떤 aligner A에 대해 모든 출력에 같은 편향 b를 더한 A′=A+b를 생각하면,

\[
A'(W(P,\epsilon),M)-A'(P,M)
=A(W(P,\epsilon),M)-A(P,M).
\]

원래 c0가 b만큼 틀려도 이 검사는 동일한 결과를 낸다. 따라서 synthetic shift만으로 native absolute shift의 정확성을 확인한다는 인용 설명은 **“변위 반응의 일반화 확인”으로 한정해야 한다.** 절대 정합 정확도에는 독립적인 기하 기준이 추가로 필요하다. 이는 기존 checkpoint를 쓰느냐 새로 학습하느냐의 문제가 아니라, 무엇을 정답으로 확인하는가의 문제다.

**그래도 보정을 유지해야 하는가?**

현재 결과에서는 PAN warp를 유지하는 것이 타당하다. 그 이유는 “입력이 커져서 보정이 필요해졌다”가 아니라, **그 경로와 함께 학습한 현재 checkpoint에서 보정을 빼면 복원이 나빠졌기 때문**이다. [앞선 원본 재집계](PAN_Aligner_Three_Axes_Evidence_Analysis_2026-09-18.md)에서 로컬 s1 13개 checkpoint 모두 NOA로 RR ERGAS와 FR HQNR이 악화했다. 평균 NOA−ON은 ERGAS +0.079674, HQNR −0.005551이었다.

이에 따라 현재 필요한 조치는 c의 ×4나 추가 좌표 수정이 아니라 정상 A_ON 경로 유지다. 다만 이것은 모든 pansharpening 모델에 정합 CNN이 반드시 필요하다는 주장, 장면별 예측이 상수 보정보다 반드시 우수하다는 주장, 또는 native 기하가 정확하다는 주장과는 구분된다. no-align으로 재학습한 모델과의 비교도 같은 checkpoint의 NOA와 별개다.

**새 실험 없이 확인할 수 있는 범위**

- 구조와 sampler 단위: 코드·간단한 CPU 수치 검사로 이번에 확인했다.
- 인용된 N2 수치와 T0의 크기별 반응: 기존 raw CSV를 재집계해 확인했다. 실영상 추론도 다시 돌리지 않았다.
- 현재 Student 전체의 RR/FR 반응·crop-consensus 성능: 기존에 없는 조합은 기존 checkpoint로 추가 inference diagnostic이 필요하다. **새 학습은 필요 없지만, 새 평가 계산도 전혀 필요 없다는 뜻은 아니다.**
- native absolute alignment의 정확성: 추가 shift 반응만으로는 식별되지 않는다. 독립적인 기하 기준이 있어야 한다.

검증 코드와 원본 snapshot은 [분석 폴더](Aligner_Resolution_Check_2026-09-18/)에 저장했다. [verify.py](Aligner_Resolution_Check_2026-09-18/verify.py), [검증 결과 JSON](Aligner_Resolution_Check_2026-09-18/verification.json), [N2/T0 회귀 재계산 CSV](Aligner_Resolution_Check_2026-09-18/response_refits.csv), [입력 SHA 목록](Aligner_Resolution_Check_2026-09-18/input_manifest.json)을 함께 제공한다.
