# s2 W112–D124 실험 계획 및 loss 참조 구현

## 포함 파일

- `PAN_S2_W112_D124_KD_Variance_Adaptive_Campaign_2026-09-10.md`: 실험·구현·우선순위·약 24시간 review·중간 변경·평가 인계 문서.
- `pan_gt_anchored_kd.py`: 기존 continuous hard/soft reconstruction criterion. 원본 파일을 수정하지 않고 포함했다.
- `pan_s2_w112_stat_kd_reference.py`: 국소 영상/gradient 분산, gradient/분광 공분산과 통계 KD wrapper.
- `PAN_S2_W112_reference_checks_2026-09-10.json`: 작성 컨테이너에서 실행한 22개 CPU 단위검사 결과.

## 수학적 단위검사 실행

두 Python 파일을 같은 폴더에 두고 PyTorch가 설치된 환경에서 실행한다.

```bash
python pan_gt_anchored_kd.py
python pan_s2_w112_stat_kd_reference.py
```

작성 컨테이너에서는 PyTorch 2.10.0+cpu, CUDA unavailable 상태에서 10+12 검사가 통과했다. 설치 버전이 다른 s2에서 다시 확인한다.

## 범위

이 묶음은 실제 W112 U-Net, 정합기, PanCollection loader, trainer, covariance uncertainty 추정기, 원격 실행기 또는 평가기를 포함하지 않는다. 제공 통계는 영상의 공간적 centered moment이며, 이동량의 uncertainty가 아니다. 공분산 표현의 비교는 모든 원소 평균 L1이고 Gaussian KL이 아니다.

실제 repository의 모델·입력 순서·base 합산·crop·warp·AMP·checkpoint·평가기에 맞춰 plan의 adapter 계약을 구현해야 한다. W96 backbone 가중치를 W112에 조용히 partial load하지 않는다. Teacher/donor 출처, train-only calibration과 loss gradient gate를 먼저 확인한다.

실험 시간은 약 24시간 review를 권하며 강제 종료를 설정하지 않는다. 진행 중 변경은 parent checkpoint와 config diff, continuation 대조로 추적한다. 이 파일을 생성하면서 s2 실험을 시작한 것은 아니다.
