# GF2 논문 비교·평가·reference 감사 — 2026-09-21

## 1. 결론과 검증 범위

**현재 G20 GF2 평가에서 장면 누락, 잘못된 GT 대응, 다른 checkpoint의 RR/FR 혼합, masked HQNR 대체 등 실행 결과를 무효화할 오류는 발견하지 못했다. 그러나 PAN-Crafter/U-Know-DiffPAN 논문과 모든 평가·실험 조건이 완전히 동일하다고 판정할 수는 없다.**

- 데이터: RR 20장 / FR 20장. 보유한 PanCollection 원본 MAT와 모든 장면의 원본 배열이 정확히 일치한다.
- 핵심 지표: DLPan MATLAB의 정상 입력 계산 경로를 대조했고 현재/기존 로컬 평가기의 동일 출력 재계산도 일치했다. MATLAB 실행에 의한 독립 수치 검증은 하지 못했다.
- 비교표 주의: PSNR/SSIM 저자 최종 설정 미확인, RMSE/CC 정의 차이, JQM 대체 정의, checkpoint 선택 정책 차이가 남는다.
- reference: 현재 R1은 정상 검증됐다. 다만 생성 batch64 대 검증 batch16이라는 **실패 전력이 있는 구조가 그대로 남아 있다.** 현재 참조 자산의 손상을 뜻하지는 않지만 새 Teacher에서 재발할 수 있다.

검증은 개발 checkout, 실제 frozen runtime, s1의 현재 원본 데이터·reference·저장 평가 결과를 대상으로 했다. 다른 서버의 모든 실행 파일·시트 행까지 실시간 전수 확인했다는 뜻은 아니다. GPU 추론/학습 재실행, 환경 변경, 코드 수정, reference 재생성, 원격 시트 변경은 하지 않았다. 이 보고서만 추가했다.

현재 실행 소스 content SHA:
`56e80b5cd0d286def8293715747831cb4abd4b7ae9ebd02a01fddb1a8a543e69`.

실제 실행 소스 위치: `/home/knuvi/Desktop/song/PAN-Crafter-runtime-g20-s1-20260920`.
평가 6개 핵심 파일, 데이터 로더 및 reference 관련 핵심 파일의 개발본/실행본 내용 일치를 확인했다. Git HEAD만 보고 실행 코드를 추정하지 않았다.

## 2. 발견사항 — 중요도 순

| 우선순위 | 발견사항 | 영향 / 판정 |
| --- | --- | --- |
| 높음: 운영 재발 위험 | calibration 기본 batch64, online parity batch16; 실제 batch 차이로 실패한 전례 | 새 reference 발행이 다시 막힐 수 있음. 현재 R1 실패라는 뜻은 아님 |
| 높음: 비교표 주장 | JQM은 SRF-substitute 변형; PSNR/SSIM 저자 final driver 미확인 | 모든 metric을 동일 측정이라고 표기하면 안 됨 |
| 중간: 재현성 | calibration batch size가 manifest에 없음 | 동일 Teacher의 reference 재현 조건이 불완전 |
| 중간: 선택 정책 | 기본 시트값은 Exact50K. RAW_MAX/TARGET 등은 별도 열 | HQNR-best 결과와 혼동 금지; 저자 선택 정책과도 같지 않음 |
| 중간: 입력 차이 | LPAN은 자체 레시피 생성본, 저자 배포 LPAN과 불일치 | 원본 벤치마크는 동일하나 PAN-Crafter 완전 재현은 아님 |
| 중간: 기존 해석 | rounded anchor 일치, 기존 평가기 일치를 독립 MATLAB 검증처럼 해석 | 기존 “전부 일치/차이는 전부 recipe” 결론은 근거보다 강함 |
| 낮음: 소량 전처리 차이 | PAN-Crafter 저장 코드는 reference LMS까지 clip, 현재는 원본 유지 | 실제 CANConv FR20의 HQNR 평균 영향 약 6.14e-8 |
| 비교 조건 | Params/FLOPs/시간/메모리 측정 범위·하드웨어 차이 | 품질 지표와 별도로 비용 비교 조건 통일 필요 |

## 3. 확인한 1차 출처

2026-09-21 원격 HEAD와 보유 공식 소스 snapshot의 revision 일치를 확인했다.

- [PAN-Crafter 공식 저장소](https://github.com/KAIST-VICLab/PAN-Crafter): `e12fb5ef09051d19bc21f3fef6080a05cf7b223a`.
- [U-Know-DiffPAN 공식 저장소](https://github.com/KAIST-VICLab/U-Know-DiffPAN): `e3f7b9d079d0753ec16af2a44855e44fd3c2307b`.
- 로컬 DLPan-Toolbox: `a34f884af88967580e3ef4f61a89e12fdfe4cda5`의 MATLAB `Tools/Quality_Indices` 및 호출부.
- [PAN-Crafter 논문 및 보충자료 v2](https://arxiv.org/html/2505.23367v2); 로컬 `2505.23367v2.pdf`.
- [U-Know-DiffPAN CVPR 논문](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_U-Know-DiffPAN_An_Uncertainty-aware_Knowledge_Distillation_Diffusion_Framework_with_Details_Enhancement_CVPR_2025_paper.pdf); 로컬 `uknowdiff.pdf` 및 공개 소스.

PAN-Crafter README는 최종 지표가 DLPan MATLAB으로 측정됐다고 명시한다. U-Know 공개 Python metric은 training/validation용이며 MATLAB과 다르다고 직접 경고한다. 따라서 **공개 Python 학습 로그와 같지 않다는 이유만으로 현재 최종 평가를 오류라고 할 수 없고, 그 로그와 같다는 이유로 논문 표와 같다고 할 수도 없다.**

## 4. 데이터·장면 수·정규화

| split | 수 | PAN / GT 또는 LMS / LRMS | 확인 |
| --- | ---: | --- | --- |
| Train | 19,809 | 1×64×64 / 4×64×64 / 4×16×16 | 등록 데이터와 로더 일치 |
| Validation | 2,201 | 1×64×64 / 4×64×64 / 4×16×16 | 별도 검증 split; test20과 다름 |
| RR test | 20 | 1×256×256 / 4×256×256 / 4×64×64 | MAT 1…20의 GT/MS/LMS/PAN 모두 H5와 max diff 0 |
| FR test | 20 | 1×512×512 / 4×512×512 / 4×128×128 | MAT 1…20의 MS/LMS/PAN 모두 H5와 max diff 0; GT 없음 |

U-Know 논문 Table 1의 GF2 Train/Test는 19,809/20이며 4밴드·10bit·비율4이다. 현재 GF2에 예전 WV3의 19장 이슈를 이월할 근거는 없다.

세부 확인:

- RR/FR H5 실제 SHA가 등록 manifest와 일치. FR은 원본 MAT SHA 20개도 provenance와 일치했다.
- MAT의 숫자 순서 1…20 ↔ H5 row 0…19를 확인했다. 단순 파일 개수 검사만 한 것이 아니다.
- 원본 저장 배열은 float64, 모델 입력은 float32. 센서 명시값으로 `2×DN/1023−1` 정규화한다.
- 출력만 `[-1,1]` clip 후 `(y+1)×511.5`로 DN 복원한다. 저장 전 일반 정수 반올림은 하지 않는다. Q4 내부의 MATLAB uint16 변환은 별개다.
- 모델 입력과 평가용 원본 GT/PAN/MS/LMS는 임의 clipping하지 않는다. 실제 LMS overshoot는 RR 1065.43, FR 1042.83 DN까지 있다. FR PAN의 1023 DN 포화 화소 15개도 보존한다.
- train/val 모든 LRMS 패치의 정확한 중복은 split 내부와 split 사이 모두 0개였다. 좌표/scene ID가 없어 **지리적 중첩·인접 패치·같은 대장면에서 분할됐는지는 검증하지 못했다.**
- H5에 band-name/NoData metadata는 없다. B,G,R,NIR은 데이터 관례에 근거한 선언이며 독립적인 물리 분광 검증이 아니다. 별도 NoData 마스크는 적용하지 않는다.
- GF2 LRMS 생성 관계를 train8/val8/RR20에서 재계산하면 GNyq .3, replicate, phase(2,2)와 최대 차이 3.07e-12 DN이다. 학습 19,809장 전체의 MTF 관계를 전수 검증한 것은 아니다.

코드: [g20/data.py](/home/knuvi/Desktop/song/PAN-Crafter/g20/data.py:65), [추론 DN 복원](/home/knuvi/Desktop/song/PAN-Crafter/g20/evaluation.py:188).

## 5. RR 지표 정의

공통 입력은 GT와 같은 checkpoint의 SR 20×4×256×256, float64 계산이다. 경계는 Python `20:-21` → **215×215**. MATLAB `21:end-21`과 같은 인덱스다. “양쪽 21픽셀 제거”는 엄밀히 틀린 설명이지만 현재 인덱스는 원본과 맞는다.

| 지표 | 현재 구현 | 대조 판정 |
| --- | --- | --- |
| SAM | 화소별 4밴드 분광각, degree 평균 | MATLAB 정상 경로 일치; 작은 분모 가드·cosine clip 추가 |
| ERGAS | 밴드별 MSE / GT 밴드 평균², 밴드 평균 후 sqrt×25 | MATLAB 정상 경로 일치; prediction mean 분모 아님 |
| SCC | 각 밴드 2D Sobel; 내부 1px crop, zero padding correlation; 전체 gradient magnitude cosine | MATLAB 정상 경로 일치; Pearson CC나 3D Sobel 아님 |
| Q4 | 실제 4밴드 hypercomplex Q2n; block/stride32; uint16 반올림·포화 | MATLAB 정상 경로 대조; global Q4 proxy 아님 |
| PSNR | 장면 전체 HWC MSE, peak1023 | 관례 구현. 저자 최종 구현/peak 독립 확인 불가 |
| SSIM | 밴드별 Gaussian11, σ1.5, population covariance, K=.01/.03, 유효창 평균 | DLPan ssim.m 식과 수치 일치 확인. 논문이 그 설정을 사용했는지는 미확인 |
| RMSE | 장면 HWC 전체 MSE의 sqrt, DN 단위 | 추가 지표. mean-band RMSE 또는 공개 logger RMSE와 구별 필요 |
| CC | HWC 전체를 펼친 Pearson | 추가 지표. mean-band Pearson과 다름 |

Q4의 215×215 입력은 우/하단 대칭 연장 9px로 224×224, 총 49개의 32×32 블록이 된다. 밴드 누락이나 Q8 계산 후 이름만 Q4로 바꾸는 경로가 아니다. 표본 표준편차 `ddof=1`과 Cayley–Dickson 곱셈 순서를 MATLAB 소스와 대조했다.

각 scene의 지표를 계산한 뒤 20개 산술평균을 낸다. 표준편차는 scene 간 `ddof=1`이며 seed 간 표준편차가 아니다. 전체 장면을 합쳐 하나의 MSE나 CC를 구하는 방식도 아니다.

PSNR/SSIM은 `indexes_evaluation.m`의 호출 목록 밖이다. 현재 PSNR이 global MSE이고 SSIM이 Gaussian이라는 것은 명확하지만, PAN/U-Know 최종 테이블 생성 스크립트가 없으므로 완전 동일 판정은 불가하다. GF2 PSNR peak를 1024로 바꾸기만 해도 +0.00848646 dB가 달라진다.

실데이터의 RR cropped GT 밴드 평균 최솟값은 188.050254 DN이고 GT 영벡터 화소는 0개였다. 작은 분모 처리 차이가 현재 GT에서 활성화되는 상황은 아니다. 단, 완전 일치 SR의 PSNR=∞를 G20이 거부하는 등 극단 입력까지 MATLAB과 같은 것은 아니다.

코드: [RR 계산](/home/knuvi/Desktop/song/PAN-Crafter/g20/evaluation.py:88), [SCC·PSNR·SSIM](/home/knuvi/Desktop/song/PAN-Crafter/tools/eval_dlpan.py:24), [Q2n](/home/knuvi/Desktop/song/PAN-Crafter/tools/metrics/q2n.py:31).

## 6. FR HQNR, PAN shift, masking

**현재 공식 G20 FR는 native PAN + original LMS, 원본 512×512 전영역, 20장이다. masking=False이며 예측 shift로 평가 reference를 이동시키지 않는다.**

- A_ON은 모델의 PAN 입력 정렬을 활성화한다는 뜻이지, 평가 PAN을 같이 이동시키거나 유리한 영역만 평가한다는 뜻이 아니다.
- Dλ = 1 − Q4(original LMS, MTF(SR)). MTF는 GF2 `.3`×4, 41×41, replicate 경계다. DLPan MATLAB의 `otherwise` preset과 맞지만 측정된 GF2 물리 MTF라는 의미는 아니다.
- Ds의 high 항은 SR/native PAN의 비중첩 32×32 UQI 블록 평균이다.
- low 항은 original LMS와 `interp23tap(imresize_matlab(nativePAN,1/4),4)` 사이의 같은 UQI다. PAN 축소는 bicubic antialiasing이며 학습 LPAN 레시피나 sensor MTF 축소가 아니다.
- 각 밴드에서 블록 평균 Qhigh와 Qlow의 차에 절댓값을 취한 뒤 4밴드를 평균한다. 블록별 절댓값 평균이나 NCC가 아니다.
- HQNR은 장면별 `(1−Dλ)×(1−Ds)`를 먼저 구한 뒤 20장 평균이다. 평균 Dλ와 평균 Ds를 먼저 곱하지 않는다.
- signed Ds는 추가 진단이고 공식 Ds를 대체하지 않는다. 실제 20×4 Qhigh/Qlow로 공식 Ds를 다시 계산해 1e-12 내 일치를 검사한다.

따라서 현재 공식 HQNR에서 edge masking으로 점수를 올리거나, PAN shift에 따라 평가 reference까지 바꾸는 문제는 발견하지 못했다. 과거 다른 캠페인의 masked/V64 값은 이 공식 열과 혼합하면 안 된다.

수치상 남는 한계:

- MATLAB `fwind1`/`imresize`를 실행하여 같은 배열을 직접 대조한 결과는 없다. 포팅 소스 대조와 anchor는 실행 검증을 대신하지 못한다.
- 현재 MTF kernel 합은 0.9987399483, 최소값 약 −1.34e-12다. 원본 설계를 따르는 것이므로 임의 재정규화/음수 제거는 하지 않았다.
- FFT filter와 직접 correlation의 최대 차이는 1.48e-12. 이것은 Python 내 구현 일치이며 MATLAB golden 검증이 아니다.
- `_blockproc_uqi`의 0분모 반환값은 MATLAB의 NaN 처리와 다르지만 실제 native reference의 32블록에서 해당 퇴화 사례는 없었다.

코드: [FRMetrics](/home/knuvi/Desktop/song/PAN-Crafter/g20/evaluation.py:115), [MTF 및 Ds](/home/knuvi/Desktop/song/PAN-Crafter/tools/metrics/eval_fr.py:192).

## 7. JQM 및 공개 학습 logger의 차이

### JQM은 명시적으로 별도 변형

현재 JQM은 `SRF-substitute (NNLS-normalized), MTF41, phase2, global CMSC, v1=0.5; not SIPSA-equivalent`이다.

- 실제 sensor spectral response 대신 scene별 native LRMS/저역 PAN에서 NNLS weight를 추정하고 합1로 정규화한다.
- GF2 MS GNyq .3, PAN GNyq 기본 .15; 41탭, phase `[2::4]`.
- global CMSC/population std, 음수 correlation은0; 입력 및 저역 결과를 [0,1023] clip.
- QLR 밴드 균등평균과 QHR intensity 항을 각각 .5로 합친다.

현재 값이 누락되거나 임의 상수인 것은 아니다. 그러나 **SIPSA/원 JQM 논문 숫자와 같은 열에 무주석으로 직접 비교하면 안 된다.** PAN-Crafter/U-Know의 주요 비교표에는 이 JQM 열 자체가 없다. 모든 방법의 저장 출력을 동일한 변형으로 재평가하거나, 명확히 별도 진단 지표로 표시해야 한다.

코드: [variant 선언 및 호출](/home/knuvi/Desktop/song/PAN-Crafter/g20/evaluation.py:163), [JQM 구현](/home/knuvi/Desktop/song/PAN-Crafter/tools/metrics/jqm.py:35).

### 저자 Python logger와 최종 MATLAB 지표는 다르다

공개 PAN `utils.py`는 global Q4 proxy, HWC 전체 scipy Sobel, prediction-mean ERGAS, mean-band PSNR, 기본 skimage SSIM을 사용한다. HWC RMSE는 C로 나누지 않아 GF2에서 현재 RMSE의 2배가 된다.

같은 GF2 EXP scene0, 같은 215×215 crop으로 함수만 바꾼 결과:

| 지표 | 현재 최종 평가 | 공개 PAN 학습 함수 |
| --- | ---: | ---: |
| Q4 | .85934247 | .96416628 |
| SCC | .87048238 | .83042217 |
| ERGAS | 2.18893071 | 2.19182391 |
| PSNR | 30.82262967 | 30.88355086 |
| SSIM | .80367187 | .80496130 |
| RMSE DN | 29.42687540 | 58.85375080 |

같은 장면 flattened CC=.98100619, mean-band CC=.96561068이다. 이름이 같아도 정의를 생략할 수 없다.

U-Know Python logger도 기본 crop1, band PSNR/CC 평균, SAM 중간 반올림을 사용하며 FR은 sewar QNR이다. **그 QNR은 현재 Khan Dλ 기반 HQNR와 다른 지표다.** [저자 경고 코드](https://github.com/KAIST-VICLab/U-Know-DiffPAN/blob/e3f7b9d079d0753ec16af2a44855e44fd3c2307b/utils/metric.py#L105).

## 8. 입력 LPAN 및 reference clipping

### LPAN 입력

현재 LPAN은 Gaussian σ1.98, 41탭, replicate, phase2 레시피 생성본이다. 각 split 균등표본16개에서 cache와 현재 레시피는 정확히 일치했다. 저자 배포 LPAN 대비 차이는 다음과 같다.

| split | 표본 RMSE(DN) |
| --- | ---: |
| Train | .3143 |
| Validation | .3102 |
| RR | .3210 |
| FR | .3768 |

표본 내 FR 최대 절대 차이는 1.1656 DN. 이는 **모델 입력** 차이지 HQNR의 native PAN/LMS 평가 reference 차이가 아니다. 현재 CPU 감사로 이 LPAN 차이의 최종 HQNR 영향을 정량화하지 않았다.

### PAN 저자 export와 현재 평가 reference 차이

PAN 저자 `train.py`는 SR뿐 아니라 LMS/PAN/MS/GT 저장 배열도 clip한다. 현재 evaluator는 raw 원본 reference를 유지하며 U-Know의 test export는 원본 reference 배열을 유지한다. 따라서 양쪽 모두와 bitwise 같은 전처리라고 말할 수 없다.

실제 CANConv GF2 FR 저장 출력20장을 고정하고, 평가 reference만 raw 대 [0,1023] clipped로 바꿔 CPU 재계산했다. FR LMS의 범위 밖 값은 20,971,520개 원소 중 68개였다.

| 값 | raw reference | clipped reference | clipped−raw |
| --- | ---: | ---: | ---: |
| HQNR | .918871656115 | .918871717511 | +6.1396e-8 |
| Dλ | .019449440793 | .019449423977 | −1.6816e-8 |
| Ds | .062869273785 | .062869227146 | −4.6640e-8 |

장면별 HQNR 최대 절대 차이는 1.2718e-6. 이 특정 출력에 대해서 영향이 매우 작다는 증거이며, 모든 모델에 대한 상한은 아니다. 정규화↔역정규화 float32 round-trip 차이 전체까지 시험한 것은 아니다.

## 9. checkpoint 선택과 실제 업로드

[g20/postrun.py](/home/knuvi/Desktop/song/PAN-Crafter/g20/postrun.py:45)의 선택은 다음과 같다.

| 선택 | 규칙 | test-aware |
| --- | --- | --- |
| Exact50K | 고정50,000 update | 해당 run 안에서 test값으로 checkpoint 선택하지 않음 |
| RR_VAL_SELECTED | 별도 val2201 ERGAS 최소 | 별도 validation 기준 |
| RAW_MAX | FR test20 HQNR 최대, 동점 이른 step | 예 |
| TARGET | HQNR>.964 중 RR test20 ERGAS 최소, 다음 SCC/PSNR/step | 예 |
| E_MIN_DIAG50 | RR test20 ERGAS 최소 | 예 |

**현재 기본 Sheet metric 열은 Exact50K다. HQNR-best인 RAW_MAX는 별도 열이다.** 코드가 `main = reports['exact50k']`로 명확히 지정하며 Notes에도 기재한다. 각 선택 안에서 RR/FR은 같은 A/U checkpoint다. 본 감사에서 이 정책을 변경하지 않았다. [upload.py](/home/knuvi/Desktop/song/PAN-Crafter/g20/upload.py:148)

PAN 공개 코드는 RR ERGAS-best와 FR Ds-best를 별도로 저장한다. 최종 논문 표가 이 두 파일 중 무엇을 사용했는지까지 연결되는 driver는 없다. U-Know도 최종 표에 사용한 checkpoint 선택의 완전한 재현 경로가 공개돼 있지 않다. 따라서 선택 정책 동일 주장은 불가하다.

반복적인 architecture/seed screening에 같은 benchmark test를 사용했으므로, Exact50K라고 해도 전체 연구를 독립 held-out test라고 부를 수는 없다. 현재 report의 `independent_test=False`는 이를 적절히 표시한다.

2026-09-21 **09:18:50 KST** 로컬 저장 결과 스냅샷:

- s1 완료4run×50 checkpoint + 진행중1run×17 checkpoint = **217개 평가 기록** 점검.
- 모두 RR20/FR20, RR crop20:-21, FR full512/nativePAN/masking=False.
- RR8개 및 FR3개 metric의 유한값·20장 산술평균, 장면별 HQNR 곱, checkpoint update 일치를 확인했다.
- JQM은 전 기록에서 measured이며 장면 평균과 저장값도 일치했다. RMSE/CC도 실제 계산값이 존재한다.
- 완료4run의 로컬 upload receipt는 `readback_verified=true`, GF2-s1 16–19행이다. 이번 감사에서 원격 시트를 새로 읽은 것은 아니다.
- 저장/선택은 full precision, 시트 표시만4자리 형식이다. 반올림 후 checkpoint를 고르지 않는다.

## 10. 실제 저장 출력으로 기존 evaluator와 교차검증

CANConv 배포 GF2 RR/FR 저장 출력20장을 현재 G20 및 기존 업로드/평가 경로로 재계산했다. RR는 실제 `gspread_upload.py:_rr` 함수를 사용하고 파일 로더만 이미 읽은 실제 H5 배열로 대체했다. FR는 기존 `d_lambda_k/d_s` 조립 및 저장 reference JSON과 비교했다.

| 지표 | 현재 G20 재계산 | 기존 로컬 경로와 차이 |
| --- | ---: | ---: |
| ERGAS | .636197899288 | 0 |
| SCC | .991558145791 | 0 |
| SAM | .712806580998 | 0 |
| Q4 | .982416662500 | 0 |
| PSNR | 43.332202357283 | 0 |
| SSIM | .982068715981 | 0 |
| RMSE | 7.106682317441 | 0 |
| CC | .998013841010 | 0 |
| Dλ | .019449440793 | 0 |
| Ds | .062869273785 | 0 |
| HQNR | .918871656115 | 0 |
| JQM 변형 | .986278286433 | 0 |

FR 장면별 차이도0. 이것은 **현재/기존 로컬 평가 일관성**의 증거다. 두 경로가 공통 primitive를 사용하므로 독립 MATLAB 정답과 일치한다는 증거는 아니다.

추가 CPU 검사: 평가 테스트5개, reference 관련 테스트79개 통과. Q4/Q8 identity 215×215는 1±1e-16, UQI vector/loop 차이0, Gaussian SSIM 별도 식과 차이0. MATLAB/Octave 실행 파일은 없었다. mock을 사용하는 테스트를 MATLAB golden test라고 간주하지 않았다.

## 11. Teacher reference 생성·검증

구분해야 하는 reference는 세 종류다.

1. 평가용 원본 RR GT / FR PAN·LMS: §4–6의 benchmark reference.
2. LPAN 입력 cache: §8의 레시피 생성 입력.
3. KD Teacher R0/R1의 q-cache, τR, qref: **학습용** 불확실성/가중치 자산. 이를 평가 GT로 사용하지 않는다.

### 현재 자산 검사

| 항목 | R0 | R1 |
| --- | --- | --- |
| Teacher | 기존 s3 GF2_TA/91001 | 신규 s1 Teacher/91002 |
| τR | .005695626139640808 | .005706041119992733 |
| qref | .4086490869522095 | .417192280292511 |
| q cache | 19,809×4 float32 | 동일 |
| calibration subset | train3072, seed1234 | 동일 |
| 원 발행 q parity | PASS, max0 | PASS, max3.57628e-7 |
| s1 소비 | 승인된 R0 예외 | 원래 strict 검증 |

각 reference의 연결 아티팩트8개 실제 SHA를 manifest와 비교했다. q의 전체 shape/finite/nonnegative, calibration index3072의 재현성, subset q median, 저장 parity receipt와 cache 연결도 확인했다. R1 Student 두 seed는 이미 완료 및 upload 검증 상태다.

τR는 무증강 train subset에서 최종 출력의 band-mean L1을 pixel 전체로 모은 median이고, qref는 동일3072 ID×4 view median이다. q는 AXIS16 반경 .25/.5/1/2, 네 방향, 2K로 나누는 coordinate 평균이다. 학습 view와 cache는 고정HV+ROT4의 `(base ID, rotation)`으로 연결된다. train 이외 GT가 calibration에 섞인 증거는 없다.

### 실제로 남아 있는 재발 위험

생성 함수 기본 batch는64([calibration.py](/home/knuvi/Desktop/song/PAN-Crafter/g20/calibration.py:23)), 검증은16개 ID를 한 batch로 계산한다([reference_parity.py](/home/knuvi/Desktop/song/PAN-Crafter/qg40/reference_parity.py:53)).

9월20일 같은 GPU/환경/checkpoint에서 실제 결과:

- batch64 cache ↔ batch64 재계산: max0.
- batch64 cache ↔ batch16 검증: **1.6689300537109375e-5, 실패**.
- batch16 생성 ↔ batch16 검증: max0.

미발행 cache를 보존하고 batch16으로 재생성하여 당시 복구했다. [운영 기록](/home/knuvi/Desktop/song/PAN-Crafter/research_log/0920/2026-09-20_qg40-s2-start-gf2-recovery.md:15)

현재 기본값64가 남아 있으므로 신규 Teacher에서 재발 가능하다. manifest에 실제 batch size도 없다. GPU/runtime 차이만을 원인으로 설명하면 이 전례를 놓친다. parity 전에 immutable cache/calibration을 저장하므로 실패 후 다른 batch로 재시도할 때 미발행 파일 충돌도 고려해야 한다. 다만 **검증 전 consumable reference manifest를 발행하지 않는 보호장치**는 정상이다.

권고: 앞으로 생성할 reference에서 생성/검증 batch 정책과 실행 조건을 명시적으로 고정·기록하고 발행 전 수치검사를 완료할 것. 이미 소비 중인 R0/R1을 임의 재생성하거나 tolerance를 전역 완화하라는 뜻은 아니다.

### R0 서버간 수치 예외와의 구분

현재 s1/R0의 원 strict 실패는 보존돼 있다: q max2.0056963e-5, output max1.1372566e-4, delta max1.1584163e-4 HRpx. 같은4090 반복은 모두0이었다. TA HQNR replay 차이는1.3029851e-5로 원래1e-5 기준 밖이고 승인된2e-5 안이다.

R0 원 checkpoint/cache/τ/qref는 유지하며 승인된 예외 상태로 표시한다. 예외는 s1/R0/4090 범위이고 신규 R1–R4 전역 우회가 아니다. 소비 시 저장된 PASS만 믿지 않고 현재 local tensor를 다시 계산해 봉인된 local 값과 비교한다.

단, q16ID×4view와 출력2표본 검사는 전체train의 동일성 증명이 아니다. 표본 q-weight 차이 최대1.39475e-5를 전체 학습 영향 상한으로 해석해서도 안 된다. 이 parity/metric replay는 **같은 Python 구현의 환경 간 재현 검사**이지 MATLAB과의 독립 비교가 아니다.

## 12. 학습 조건·비용·논문 수치 인용 주의

현재 실험은 PAN-Crafter의 원 모델·MARs·crop을 그대로 재현하는 실험이 아니다. 50K/batch48은 같은 기본 예산이나 현재 W112D123 Teacher/W104D122 Student, global alignment, fixedHV/ROT4/nocrop/KD 및 자체 LP 입력으로 방법이 다르다. U-Know는 300K/batch32와 diffusion sampling25step 등 다른 학습·추론 예산을 사용한다. 이는 새로운 방법 비교 자체를 금지하지 않지만 동일 학습 recipe 재현이라는 설명은 불가하다.

비용은 [postrun.py](/home/knuvi/Desktop/song/PAN-Crafter/g20/postrun.py:78) 및 `official/profile.json`을 기준으로 확인했다.

- profile 입력은 train64가 아니라 **RR PAN256/MS64, batch1, C4**이다.
- Params는 배포 A+frontend+U 전체 파라미터. KD Teacher까지 포함한 학습 자원 비용은 아니다.
- FLOPs 표시는 실제로 THOP MACs + 수동 frontend FLOPs/2이며, functional normalization/grid 연산 누락 가능성을 명시한다. 저자 counting convention과 동일하다는 증거는 없다.
- latency는 FP32/3warmup/10회 CUDA sync 평균, offline LP 생성과 dataload 제외다.
- 메모리는 `max_memory_allocated()/2**20`이며 NVML 전체 사용량이나 reserved memory가 아니다.
- PAN 논문 비교는 WV3 C8/3090 조건이고 현재 s1은 GF2 C4/4090이다. 논문 시간·메모리와 그대로 순위를 매길 수 없다.
- 시트에 scope/convention은 올라가나 입력 shape/GPU/warmup/반복수까지 모두 별도 열로 올라가는 것은 아니다. 비용 표 작성 때 profile metadata를 함께 사용해야 한다.

PAN-Crafter arXiv v2에는 GF2 ERGAS가 본문 Table2 **.522**, 보충 Table7/12/14/15/16 **.552**로 내부 불일치가 있다. 어느 값을 인용했는지 표시하고 임의로 하나를 검증된 정답처럼 취급하지 말아야 한다. 현재 G20의 .552 goal/.522 strong goal은 이 구분을 반영한다.

기존 `results_log/2026-09-20_gf2-hqnr-gap-audit.md`의 “anchor 전부 일치”는 더 제한해서 해석해야 한다. GF2 EXP20 직접 계산 Q4=.8088935361, SAM=1.8232511394는 인용된 .812/1.820과 정확히 같지 않다. CANConv 재계산 ERGAS=.6361978993도 PAN 표의 .653과 같지 않다. 해당 논문 표와 동일 checkpoint임을 입증하지 않은 모델 출력은 evaluator golden 정답이 아니다. FR rounded 값이 .919/.019/.063에 맞는 것은 유용한 sanity check지만 전체 평가 차이를 배제하는 증거는 아니다.

## 13. 비교표 작성 전 필요한 조치

1. 현재 RR20/FR20/native full-frame 프로토콜을 명시하고, masked/다른 crop/학습 logger 값을 공식 열과 분리한다.
2. 가능한 baseline의 저장 출력을 **동일 evaluator**로 재계산한다. 논문 인용값과 재측정값은 행 또는 각주로 구별한다.
3. MATLAB golden 대조를 확보하기 전에는 “MATLAB과 bitwise 동일” 또는 미측정 오차 범위를 확정하지 않는다. 특히 PSNR/SSIM은 저자 final 설정을 추가 확보해야 한다.
4. JQM은 SRF-substitute임을 표기하고 원 JQM/SIPSA 값과 혼합하지 않는다. CC/RMSE의 global/밴드 평균 정의도 명시한다.
5. 기본 Exact50K와 별도 RAW_MAX/TARGET 선택을 혼동하지 않고, 여러 metric에 서로 다른 best checkpoint를 조합하지 않는다.
6. 신규 reference 생성의 batch-size 정책/metadata 문제를 별도 수정 대상으로 삼는다. 이미 검증된 실행 중 reference는 유지한다.

**현재 실험을 중단하거나 기존 결과를 폐기할 근거는 발견하지 못했다. 반면 “모든 논문과 모든 열이 온전히 같은 측정”이라는 인증은 아직 할 수 없다.**
