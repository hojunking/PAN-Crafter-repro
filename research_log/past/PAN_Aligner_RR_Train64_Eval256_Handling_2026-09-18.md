**RR 평가에서 train64 aligner를 eval256에 적용하는 문제 — 2026-09-18**

현재 관심은 RR이다. 따라서 학습 RR patch64→평가 RR256의 입력 영역 변화와 RR→FR의 물리적 scale 변화는 분리해야 한다. 앞선 대화에서 두 상황을 함께 설명해, 학습 영상의 작은 배열 크기가 더 낮은 공간 해상도를 뜻하는 것처럼 읽힐 여지가 있었다.

**지금 우선할 처리는 출력 c에 256/64를 곱하는 것이 아니라, 같은 RR256 영상에서 full-aligner와 64-crop-aligner를 비교하는 것이다.** 후자는 이미 `A_CROP64_MED`로 구현돼 있다. 충분한 근거 없이 정상 추론을 바꾸거나 현재 학습을 중단할 이유는 없다.

직접 읽은 G23/1234의 config와 H5 shape는 다음과 같다.

| 용도 | sample 수 | PAN | 입력 MS | GT | loader resize |
|---|---:|---|---|---|---|
| train_wv3.h5 | 9714 | 64×64 | 16×16 | 64×64 | `crop=False`; crop-and-resize 분기 사용 안 함 |
| valid_wv3.h5 | 1080 | 64×64 | 16×16 | 64×64 | 없음 |
| RR test_wv3_multiExm1.h5 | 20 | 256×256 | 64×64 | 256×256 | 없음 |
| FR mat20, 이번 RR 질문의 별도 조건 | 20 | 512×512 | 128×128 | 없음 | 없음 |

N2 donor도 같은 학습/평가 파일과 `crop=False`를 사용한다. [`feeders/feeder.py`](../feeders/feeder.py)는 H5의 PAN 배열을 읽고, 현재 설정에서는 정규화·회전·반전만 적용한다. 256 PAN을 학습 때 64로 줄이고 평가 때만 원 크기로 복원하는 처리 경로는 없다. Aligner는 LRMS를 bicubic ×4로 PAN grid에 올린 MS와 PAN을 함께 받는다. LRMS→PAN의 ×4는 train과 RR test 모두 동일한 센서 간 비율이다.

공개 [DLPan-Toolbox의 데이터 구성 설명](https://github.com/liangjiandeng/DLPan-Toolbox)은 학습/검증 patch와 큰 평가 영역을 잘라 생성한다고 명시한다. [segImg_new.m](https://raw.githubusercontent.com/liangjiandeng/DLPan-Toolbox/main/03-Data-Simulation%28Matlab%29/segImg_new.m)은 PAN/GT/MS를 좌표로 자를 뿐, 작은 학습 patch를 만들기 위해 영상을 축소하지 않는다. [Demo_DataSimu_qb.m](https://raw.githubusercontent.com/liangjiandeng/DLPan-Toolbox/main/03-Data-Simulation%28Matlab%29/Demo_DataSimu_qb.m)은 이 원리를 보여 주는 QB 예제다. 이를 로컬 WV3의 정확한 원본 생성 이력이라고 주장하지는 않는다.

이 공개 프로토콜, 로컬 config/loader, 기존 RR 실험 기록을 함께 보면 **train64→RR256은 RR scale에서 작은 patch로 학습하고 더 큰 영역에 적용하는 설정으로 해석하는 것이 타당**하다. `256/64=4`만으로 pixel 간격이 4배 달라졌다고 추론할 수 없다. 단, 로컬 train/RR H5에는 GSD·원본 scene 좌표·생성 단계 metadata가 없다. 이 검사만으로 실제 지상 pixel 간격이나 원본 데이터의 모든 전처리 이력을 독립 인증한 것은 아니다.

그렇더라도 입력 영역에 따른 일반화 문제는 존재할 수 있다. [`pa/aligner.py`](../pa/aligner.py)는 각 입력 영역에서 z-normalization을 하고, spatial 차원을 포함하는 GroupNorm과 global average pooling을 사용한다. 64와 256에서는 통계를 계산하는 영상 내용·경계 비율·평균할 feature 구성이 달라진다. 동일 CNN이 forward 가능하다는 사실만으로 예측 불변성을 보장하지 않는다. 또한 전역 이동 하나로 서로 다른 지역의 변위를 모두 설명하지 못할 수도 있다.

**우선 적용할 진단: Aligner만 학습 때의 영역 크기로 제한한다.**

1. RR의 원 PAN256과 전체 MS를 한 번 확대한 M256을 유지한다.
2. 둘을 같은 위치의 64×64 crop 16개로 자른다. resize하지 않는다. 각 aligner에는 기존 margin4를 한 번 적용한다.
3. 동일 checkpoint의 aligner로 16개의 `(dy,dx)`를 예측한다.
4. 미리 고정한 규칙인 성분별 중앙값으로 한 이동량 `c_med`를 만든다. scene GT나 점수로 crop/집계법을 고르지 않는다.
5. `c_med`를 그대로 원 PAN256에 적용해 한 번 warp한다. crop과 full은 같은 grid이므로 ×4가 없다.
6. 복원 U-Net은 계속 전체 256을 처리한다. U-Net tiling이나 patch별 서로 다른 warp를 동시에 도입하지 않는다.

구현은 [`kdv/eval_modes.py:consensus_delta`](../kdv/eval_modes.py)와 `forward_mode(..., 'A_CROP64_MED')`에 이미 있다. 짝수 16개에서 두 중앙값의 평균을 사용하고, 전체 M을 만든 뒤 crop하며, 대표 이동량 하나로 전체 PAN을 warp한다.

이 방식은 **학습과 평가에서 A가 보는 영역 크기를 맞추는 후보**다. 원래 학습 patch와 평가 crop의 scene 분포·MS 확대 경계 조건까지 완전히 같아지는 것은 아니며, full 입력보다 반드시 좋은 방식도 아니다. 현재 전역 translation 가정을 유지한 채 입력 영역 효과를 비교할 수 있다는 장점이 있다.

같은 checkpoint·같은 RR20·같은 복원 U·같은 평가 convention에서 다음을 비교하면 판단이 가능하다.

| 비교 | 확인할 질문 |
|---|---|
| A_ON(full256) vs A_CROP64_MED | 학습 크기로 A 입력을 제한했을 때 RR ERGAS/SAM/PSNR이 개선되는가? |
| 각 방식의 알려진 추가 이동 반응 | 작은 crop에 비해 full 입력의 반응만 약해지는가? |
| native shift와 crop 간 분산 | 이동량이 지역별로 크게 다른가? 그 차이가 내용에 대한 추정 불안정인지 추가 확인이 필요한가? |
| NOA, 필요하면 ZERO | 보정 경로 제거 및 identity sampler 효과가 각각 어떠한가? |

q/반응 기울기만 좋아져도 복원이 좋아진다고 판정하지 않는다. 현재 A와 U가 함께 적응했기 때문에 동일 U의 RR metric을 같이 확인해야 한다. test20에서 결과를 본 뒤 최종 정책을 고른다면 그 세트가 선택에 사용됐음을 표시하고, 별도 validation/holdout으로 확인해야 한다.

기존 G23/1234의 RR 3-scene pilot에서는 full256 − crop64 median의 성분 차이 최대가 **0.009311 px**였다. 전체 q_size 평균은 64 crop 0.366085, full256 0.353471이었다. 이 작은 표본에서는 full 쪽의 큰 붕괴나 이동량의 4배 차이가 관찰되지 않는다. 두 방식의 absolute shift가 모두 정확하다거나 RR 복원 성능이 같다고 증명한 것은 아니다. [기존 재집계](Aligner_Three_Axes_Analysis_2026-09-18/full_vs_crop_by_scene.csv)

**결과에 따른 처리**

- crop64 대표 이동량이 여러 scene·checkpoint에서 RR 복원을 일관되게 개선하면, 검증된 추론 정책으로 채택할 근거가 생긴다. 새 학습 없이 적용 가능한 후보이나 현재 결과만으로 자동 전환하지 않는다.
- full과 crop이 비슷하면 입력 영역 mismatch를 주원인으로 삼기 어렵다. 그때는 변위 반응의 학습, native 정합 기준, A/U 상호 적응을 분석한다.
- 실제로 크기별 성능 문제가 확인되고 재학습이 필요하면, 원래 train source에서 같은 RR pixel scale의 64/128/256 영역을 구성해 mixed-size 학습하는 방법을 검토한다. 작은 64 영상을 단순 확대하거나 서로 다른 patch를 붙인 것이 원래 큰 영역의 학습을 대체하지는 않는다.
- RR256 전체를 실제로 64로 downsample한 별도 추론 경로를 택했다면, 그 **coarse grid에서 나온 이동량을 full grid로 옮길 때만** `c_full=4*c_coarse`가 맞다. 이는 원래 없던 추가 축소·정보 손실을 도입하므로 현재 crop64 대조군과 다른 실험이며 첫 번째 처방으로 추천하지 않는다.

새 평가나 학습은 이 분석 중 실행하지 않았다. 현재 질문에 대한 권장 조치는 기존 구현의 RR full-vs-crop64 비교를 완료하는 것이며, sampler 배율 수정은 아니다.
