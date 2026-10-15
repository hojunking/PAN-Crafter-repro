# Native FR 학습 자료 조사 및 준비 구현

작성일: 2026-10-13 (Asia/Seoul)

## 1. 결론 — FR 평가 데이터는 있다

**WV3/QB/GF2의 공식 FR 평가 20장은 이미 로컬에 있다.** 부족한 것은 FR 데이터 전체가 아니라, 이 평가 20장과 분리된 **native FR 학습용 원본 및 기존 RR patch의 실제 source-scene/crop 연결**이다. 이를 단순히 “미등록”이라고만 설명하면 원천 자료와 등록 코드의 문제를 혼동하게 된다.

직접 H5 metadata를 재확인했다. 아래 세 파일에는 GT가 없다.

| 센서 | 보유 FR 수 | PAN | MS | LMS |
|---|---:|---|---|---|
| WV3 | 20 | `[20,1,512,512]` | `[20,8,128,128]` | `[20,8,512,512]` |
| QB | 20 | `[20,1,512,512]` | `[20,4,128,128]` | `[20,4,512,512]` |
| GF2 | 20 | `[20,1,512,512]` | `[20,4,128,128]` | `[20,4,512,512]` |

파일:

- [WV3 FR20](../data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5)
- [QB FR20](../data/PanCollection/QB/full_examples_mat20/test_qb_OrigScale_mat20.h5)
- [GF2 FR20](../data/PanCollection/GF2/full_examples_mat20/test_gf2_OrigScale_mat20.h5)

각 센서의 원본 `full_examples_mat` MAT도 20개씩 확인했다. 이 자료는 추론·HQNR 평가·학습 없는 입력 기반 shift 진단에 사용할 수 있다. 그러나 이를 ε consistency loss로 학습하면 GT를 쓰지 않더라도 **평가 입력을 학습에 이용하는 test-time/transductive protocol**이 된다. 기존 독립 테스트와 같다고 표기하거나 사용자 승인 없이 새 학습 자료로 전용하지 않는다.

현재 Teacher48 자체도 benchmark HQNR/ERGAS로 설정을 선택하는 test-aware 개발이라고 기록돼 있다. 그 기존 한계와 별개로, FR 입력을 gradient update에 직접 쓰는 것은 추가적인 프로토콜 변경이다. 현재 탐색을 완전히 독립적인 test 검증이었다고 소급 주장하지 않는다.

## 2. 기존 학습 자료와의 차이

| 센서 | train/val patch 수 | 저장 해상도 | 원본 scene/crop 정보 |
|---|---:|---|---|
| WV3 | 9,714 / 1,080 | PAN/LMS/GT64, MS16 | 확인한 H5에 없음 |
| QB | 17,139 / 1,905 | PAN/LMS/GT64, MS16 | msfix recipe만 있고 실제 scene 좌표는 없음 |
| GF2 | 19,809 / 2,201 | PAN/LMS/GT64, MS16 | 확인한 H5에 없음 |

native FR 학습쌍을 구성하려면 예를 들어 기존 GT64를 native MS64로 사용하는 FOV에 대해 **원본/native PAN256**이 필요하다. 현재 PAN64는 이미 축소된 관측이며, 확대해도 버려진 원본 관측이 복원되지 않는다. 별도 `_pan.h5`/`pan_h5.zip`도 확인했으나 그 내용은 LPAN16이다.

조사 범위: repository `data/PanCollection`, 그 symlink가 가리키는 CANConv dataset 경로, source 문서에 기록된 staging 경로, 로컬 별도 native bundle. 기록된 과거 staging 경로 `/home/knuvi/Desktop/song/datasets/Pansharpening/_staging/PanCollection`은 현재 없다. 모든 머신이나 다른 서버에 자료가 없다는 증명은 아니다.

실제 whole-scene 자료와 acquisition/geometry/split index가 있는 **WV2 private 5-scene bundle**도 찾았다. 위치는 `data/private/PANPV_S1_WV2_20261003_v3/extracted/practical_5scene_v3_20261002`다. 구현 패턴 참고에는 유용하지만 센서가 WV2이므로 WV3/QB/GF2 학습 자료로 대체 등록하지 않았다. 이 bundle의 Gaussian center1.5 RR recipe와 사전 정합도 public PanCollection에 이식하지 않았다.

## 3. 공식 공개 배포처 조사

PanCollection 공식 목록은 각 센서의 Training Dataset과 Testing ReducedData20/FullData20을 구분한다. 확인한 목록에서는 native whole-scene 학습 원본이나 acquisition→training crop mapping의 별도 배포 링크를 찾지 못했다. 이는 “어디에도 존재하지 않는다”는 단정이 아니다. [공식 README](https://github.com/liangjiandeng/PanCollection/blob/main/README.md), [공식 홈페이지](https://liangjiandeng.github.io/PanCollection.html).

공식 WV3 FullData Drive에서는 `fr1.mat`~`fr20.mat` 공개 목록을 확인했다. 세 센서 FullData 링크는 아래와 같다. 이미 보유한 평가 자료이므로 대용량 중복 다운로드는 하지 않았다.

- [WV3 FullData20](https://drive.google.com/drive/folders/16pGIqvwWfyQVvkk3s1xrwLpavqQd0Bv7)
- [QB FullData20](https://drive.google.com/drive/folders/1bTmEpPksnQjcyjKHJ-nORQwJkIqazml1)
- [GF2 FullData20](https://drive.google.com/drive/folders/1GMePYYHhQARDOGfeTv6cHqfxCck4XU5U)

PAN-Crafter와 U-Know-DiffPAN도 PanCollection training H5와 테스트 자료를 안내한다. PAN-Crafter의 추가 `_pan.h5`는 다운샘플 PAN 입력용이다. 두 README에서 별도 native 학습 source-scene 제공을 확인하지 못했다. [PAN-Crafter](https://github.com/KAIST-VICLab/PAN-Crafter), [U-Know-DiffPAN](https://github.com/KAIST-VICLab/U-Know-DiffPAN).

DLPan은 자체 GRSM 논문 자료는 저작권 사유로 제공하지 않으며 PanCollection 사용을 권장한다. 이 설명을 모든 PanCollection 원본의 비공개 사유로 확대 해석해서는 안 된다. [DLPan Dataset](https://github.com/liangjiandeng/DLPan-Toolbox#dataset).

원본이 다른 보유 경로에도 없다면 원 배포자에게 native PAN/MS와 training patch 생성 index/recipe를 요청하는 것이 다음 확보 경로다. 이메일·이슈를 실제 전송하지는 않았다. 다른 데이터셋 대체, FR20 학습 전용, FR loss 삭제 역시 이번 작업에서 임의로 수행하지 않았다.

## 4. 준비 도구 구현 완료 범위

별도 `python -m teacher48.prepare` 진입점을 추가했다. [실행·입력 계약 문서](../teacher48/PREPARATION.md).

1. `inventory`: H5 headers/작은 attrs, native 후보/축소 patch/평가 역할 구분, symlink 중복 방지.
2. `crop-native`: SHA-pinned 원본에서 PAN/MS를 그대로 잘라 FR H5와 실제 source 좌표 index 생성. PAN 보간·사전 shift·clip 없음.
3. LMS: 제공본을 그대로 읽거나 명시된 기존 `interp23tap`을 global-scene context와 함께 적용. 독립 patch 보간 때문에 생기는 가짜 경계를 방지한다.
4. `match-rr`: RR GT와 native MS의 all-band exact match를 8개 flip/rotation으로 탐색. 다중 후보는 모호한 것으로 남기며 유일 일치도 PAN degradation/phase 검증 전에는 승인하지 않는다.
5. `prior`: native PAN/LMS structural matching, census 교차 검사, weighted median/translation consensus 및 모든 confidence/gate 원인 기록. GT를 보지 않는다. 실패 prior를 0 또는 센서 평균으로 바꾸지 않는다.

LMS 보간 support가 held-out 영역을 참조하는 누수 가능성을 독립 검토에서 재현해 수정했다. crop 사각형뿐 아니라 16 LR-pixel halo에 해당하는 보수적 64 PAN-pixel support와 global wrap 영역까지 제외한다. 제공 LMS도 support/boundary 근거를 명시한다. Whole-scene prior CLI는 해당 acquisition에 held-out crop이 있으면 거부한다.

원본·catalog 변경 감지, H5 외부 링크/VDS/external-storage 거부, 명시적 메모리 상한, 기존 출력 덮어쓰기 거부도 추가했다. 준비 중 실패하면 부분 파일과 `FAILED.json`을 보존하고 성공 manifest를 생성하지 않는다.

생성물은 `STAGED_NOT_ADMITTED` 또는 `accepted=false`다. 아직 확인되지 않은 실제 acquisition·RR phase·near-duplicate 분리·prior acceptance를 소프트웨어가 사실인 것처럼 만들어 주지는 않는다. 입력 원본이 확인되면 JSON·hash·crop index는 준비 코드로 생성할 수 있으며, 사용자에게 모든 evidence JSON을 수작업으로 작성하라는 요구가 아니다.

## 5. 실제 실행 및 테스트

실제 실행은 **읽기 전용 inventory**만 했다. 결과: [JSON](2026-10-13_native-preparation-inventory.json).

- public PanCollection 41 H5 + 별도 private WV2 scene 5 H5 = 물리 파일 46개.
- 오류 0. native-scene 후보 5개는 모두 위 private WV2 자료다.
- metadata만 조사했으며 전체 영상 hash/near-duplicate 검사 완료를 주장하지 않는다.
- 실제 native FR training H5/accepted prior는 만들지 않았다. 발견하지 못한 원본을 기반으로 만들 수는 없다.

검증 환경: `/home/knuvi/miniconda3/envs/pancrafter/bin/python`, CPU 합성 자료·임시 파일·fake transport.

| 검증 | 결과 |
|---|---:|
| `unittest discover -s teacher48 -t . -q` | 176 PASS |
| `unittest discover -s panda_method -t . -q` | 69 PASS |
| `unittest ta2.test_model_loss -q` | 19 PASS |
| `compileall -q teacher48 panda_method` | PASS |

총 **264개 통과**, 이 중 새 준비 코드 관련 테스트는 **44개**다. 실제 native FR 학습이나 성능 향상을 검증했다는 뜻은 아니다. 실험·원격 Sheet 업로드·commit/push는 실행하지 않았다.
