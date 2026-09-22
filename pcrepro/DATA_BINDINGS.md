# PCREPRO local data binding

`bindings.example.json`은 검증된 데이터가 아니라 **빈 입력 양식**이다. 원본 설계의 동봉
data manifest가 이 저장소에 제공되지 않았으므로 경로, SHA, scene ID를 임의로 채우지 않았다.
`datasets.WV3` 등에는 완성한 개별 manifest의 로컬 경로 또는 JSON 객체를 넣는다.
미확보 데이터셋만 block하며, WV3/QB/GF2를 한꺼번에 확보할 필요는 없다.
WV2는 같은 서버·같은 cycle WV3 checkpoint의 평가만 한다.

## 읽기 전용 사전 확인

저장소 루트에서 사용 중인 Python 환경으로 실행한다. 아래 명령은 데이터나 queue를 바꾸지 않는다.

```bash
python -c 'from pcrepro.data import inventory_candidates; import json; print(json.dumps(inventory_candidates(), indent=2))'
sha256sum /actual/local/path/original_train.h5
python -c 'import h5py; f=h5py.File("/actual/local/path/original_train.h5","r"); print({k:(v.shape,str(v.dtype)) for k,v in f.items() if isinstance(v,h5py.Dataset)}); f.close()'
```

경로 예시는 존재를 가정하지 않는다. discovery 결과의 이름만으로 `paper set`이나 원본 여부를
판정하지 않는다. 실제 원본 출처와 순서 대응을 확인한 담당자가 manifest를 작성해야 한다.
완성한 개별 manifest를 검사하는 읽기 전용 명령은 다음과 같다.

```bash
python -c 'from pcrepro.data import validate_manifest; import json; print(json.dumps(validate_manifest("/actual/local/path/WV3.data_manifest.json"), indent=2))'
```

전용 CLI도 같은 검사기를 사용한다. 실제 로컬 binding을 작성한 뒤 dataset별로 검사한다.
아래 preflight는 GPU/runtime도 확인하고 전용 PCREPRO provenance를 저장하므로 CPU-only
읽기 확인은 위의 Python 명령을 사용한다. 기존 캠페인을 자동 중단하는 명령은 아니다.

```bash
python tools/pcrepro_runner.py inspect-data
python tools/pcrepro_runner.py preflight --server s3 --dataset WV3 --bindings /actual/local/path/bindings.json
```

manifest 안의 상대경로는 저장소 루트 기준이다. 다른 서버에는 각 서버의 실제 파일 위치를
사용하되 동일 원자료인지 파일 SHA와 scene 대응을 별도로 비교한다.

## 개별 manifest의 필수 내용

- `schema=PCREPRO_DATA_v1`, `dataset`, `bands`, `max_dn`, `units=DN`, 실제 물리적 `band_order`.
- WV3/WV2: 8 bands, DN2047. QB: 4 bands, DN2047. GF2: 4 bands, DN1023.
- WV3/QB/GF2의 `splits`는 정확히 `train,val,rr,fr`. WV2는 정확히 `rr,fr`만 둔다.
- 각 split은 `split`, `format`, `count`, 순서가 확정된 고유 `scene_ids`, `keys`, `shapes`를 갖는다.
  `count`는 파일의 실제 sample 수와 동일해야 한다. sample ID가 없는 H5라면 출처 파일과
  명시적 배열 index를 결합한 로컬 식별자임을 provenance에 기록한다. 저자 scene ID로 가장하지 않는다.
- H5: `format=h5`, `path`, 파일 전체 `sha256`, `layout=NCHW` 또는 `NHWC`.
  `keys`는 논리적 이름(`pan,ms,gt,lms`)을 실제 H5 key로 매핑한다.
- MAT 목록: `format=mat_list`, `layout=HWC` 또는 `CHW`, 순서가 확정된 `scenes`.
  각 entry는 `scene_id,path,sha256`; `scene_ids`와 같은 순서여야 한다.
- `shapes`는 파일 layout과 무관하게 sample 하나의 **CHW** 크기다. 가변 shape를 하나의
  선언으로 숨기지 않는다. PAN/MS 비율4, GT/LMS는 PAN과 같은 support가 필요하다.
- train/val에는 `pan,ms,gt`가 필요하고 `lms`는 선택사항이다. FR에는 `pan,ms,lms`가 필요하다.
  RR에는 `pan,ms,gt`가 필요하다. LPAN/HPAN sidecar key는 허용하지 않는다.
- validation은 native PAN64/MS16. train이 더 크면 짝지은 native64/16 crop만 적용한다.
  이미64/16이면 crop은 no-op이며 resize로 crop 분포를 만들지 않는다.
- QB train/val은 `origin=ORIGINAL_PANCOLLECTION`을 필수로 선언한다.
  `msfix` 경로, `derived_from`, `repair_applied`는 거부한다. 이름이 바뀐 보정본까지
  SHA만으로 원본이라고 증명할 수는 없으므로 원본 출처를 실제로 확인해야 한다.

정규화는 모든 입력에 `2*DN/max_dn-1`이다. 원자료를 clip/round/min-max하지 않는다.
paired augmentation 후 PAN의 원 DN을 보존하여 PAN-mode residual을 동일 PAN에서 계산한다.
원자료 LMS의 bicubic ringing도 바꾸지 않는다.

## Paper FR과 RR population

WV3/QB/GF2 FR은 명시적인 원본 paper MAT **20장**을 사용한다. 해당 목록에
`paper_identity: {"origin":"PANCOLLECTION_PAPER_MAT"}`를 둔다. 이는 제공자가 선언한
원본 출처와 실제 파일 SHA를 묶는 것이며, 외부 저자 목록을 자동 인증했다는 뜻은 아니다.

FR H5를 사용하려면 `paper_identity.mat_references`에 **모든20장**의 MAT 대응을 넣는다.
각 entry는 `source_index`(0부터 시작), 동일한 `scene_id`, MAT `path,sha256`, `layout`, `keys`다.
검사기는 PAN/MS/LMS의 전체 tensor를 `array_equal`로 대조한다. 일부 장면만 같은 H5를
전체 paper set으로 취급하지 않는다. `keys`가 빠지면 H5 key명을 MAT에서도 사용한다.

WV3 RR은 기존20장 H5를 우선하며 **19장 MAT만 확보돼도 20번째 장면을 버리지 않는다**.
19개의 `mat_references`가 정확히 대응하면 `WV3_RR20_WITH_19_MATCHED_MATS`와
`matched_scene_count=19, declared_scene_count=20, discarded_samples=0`을 남긴다.
모든20장 tensor가 논문 MAT로 검증됐다는 표현은 하지 않는다. 다른 명시적 RR population도
그대로 평가 수와 manifest에 기록하며 몰래19/20장으로 잘라 맞추지 않는다.

WV2는 paper identity가 불명확해도 명시적 RR/FR 입력을 계산할 수 있다. 이 경우
`PAPERSET_IDENTITY_UNVERIFIED`, `paper_comparable=false`로 분리한다. 미검증 H5를
논문 Table9의 동일 평가셋이라고 주장하지 않는다. FR20이 아닌 명시적 WV2 subset도 같은 규칙이다.

## 평가 provenance와 재개

RR은 모든 선언 장면에서 `20:-21` support, Q32, 센서별 Q4/Q8, 공식 Sobel-zero SCC,
global-MSE PSNR, Gaussian11 SSIM, SAM/ERGAS를 계산한다. FR은 full native PAN/LMS,
마스킹·정렬 없음, **장면별 HQNR의 평균**이다. RMSE/CC는 보조 RR, JQM은
`SRF-substitute` 보조 FR이며 SIPSA 보고값과 동일 조건이라고 주장하지 않는다.

평가 cursor에는 checkpoint SHA, 전체 data manifest SHA, metric 소스와 외부 DLPan
`wald_utilities.py` SHA, 수치 라이브러리 버전, 평가 규칙, 완료 scene ID와 결과 checksum이
묶인다. 같은 폴더에서 이 정체성이 달라지면 이전 scene을 재사용하지 않고 오류로 중단한다.
미완료 scene의 결과를 완성 지표로 업로드하지 않는다. 원값은 full precision으로 저장한다.
모든 metric은 같은 checkpoint에서 계산하며 test RR/FR을 training checkpoint 선택에 쓰지 않는다.
