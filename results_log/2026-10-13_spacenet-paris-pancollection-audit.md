# SpaceNet Paris와 PanCollection WV3 원본 일치성·활용성 검토

검사일: 2026-10-13. 학습, 기존 데이터 수정, 실험 등록, Sheets 업로드는 하지 않았다.

## 1. 결론

- **SpaceNet Paris에 WV3의 비-pansharpened PAN/MS 배포 영상이 실제로 있다.** 원본 TIFF 한 쌍과 XML/VRT를 다운로드하고 실제 파일을 검사했다.
- **현재 PanCollection WV3 training patch의 source-scene mapping을 이 자료로 복원할 근거는 없다.** 저자 논문은 원영상이 Paris가 아닌 Rio·Tripoli의 네 촬영 영상이라고 명시한다. 이번 Paris 한 타일과 로컬 전체 patch의 직접 DN 대조에서도 일치가 없었다.
- **새로운 외부 WV3 자료로는 활용 가능한 후보이다.** 촬영 ID와 좌표가 있어 새 patch의 출처를 명확하게 기록할 수 있다. 다만 현재 PanCollection-only 실험과는 별도 데이터 조건이며, 학습 준비가 완료됐다는 뜻은 아니다.
- 사용자가 확정한 **RR loss만 사용 / b도 RR 기준**을 그대로 유지할 수 있다. native PAN/MS를 확보하는 것과 FR loss를 사용하는 것은 별개이다.
- 기존 9,714개 training patch의 실제 scene mapping 미확인 문제는 여전히 남아 있다.

## 2. 공개 문헌의 원영상 출처

[PanCollection 저자 원논문](https://liangjiandeng.github.io/papers/2022/deng-jig2022.pdf)의 §2.1 및 Fig.3/5는 다음 WV3 원영상을 제시한다.

| 촬영 지역·월 | 원본 MS 크기 |
|---|---|
| Rio, 5월 | 2811×3408×8 |
| Rio, 1월 | 2811×3408×8 |
| Tripoli, 8월 | 1871×2020×8 |
| Tripoli, 3월 | 1871×2020×8 |

Fig.5의 train 9,714 / valid 1,080은 로컬 H5의 실제 행 수와 일치한다. 이는 문헌상 출처 연결이며, H5 각 행의 도시·촬영·crop 좌표를 복원한 것은 아니다. 해당 논문에서 정확한 연도·일·catalog ID까지 확인되지는 않았다.

[SpaceNet Paris 공식 페이지](https://spacenet.ai/paris/)는 Paris, catalog `1040010018805C00`, 촬영 시각 `2016-02-29T11:19:13Z`를 제시한다. `MS/`·`PAN/`와 이미 융합된 `PS-MS/`·`PS-RGB/`는 구분되어 있으며, 이번에는 전자만 다운로드했다.

따라서 동일한 WV3 센서라는 사실을 동일 촬영 영상이라는 뜻으로 해석하면 안 된다. 참고로 [SpaceNet Rio 공식 페이지](https://spacenet.ai/rio-de-janeiro/)의 배포 영상은 WV2로 설명되어 있으므로, 도시 이름만 맞춰 대체하는 것도 적절하지 않다.

## 3. 실제 다운로드 범위

저장 위치: `data/external_audit/spacenet_paris_20261013/`.

선택 타일: `R08C3`, product order `055649178040_01_P001`.

| 파일 | 크기(bytes) |
|---|---:|
| `MS/16FEB29111913-M2AS_R08C3-055649178040_01_P001.TIF` | 268,438,248 |
| `PAN/16FEB29111913-P2AS_R08C3-055649178040_01_P001.TIF` | 536,875,617 |
| PAN/MS XML 두 개와 MUL/PAN VRT 두 개 | 467,281 |
| 합계 | **805,781,146 (약 0.806 GB)** |

이는 Paris 전체 다운로드가 아니다. 공개 목록에서 MS 75개 TIFF 약 17.38 GB, PAN 75개 TIFF 약 34.60 GB를 확인했고, 그중 한 쌍만 검사했다. 이 75개 타일을 75개의 독립 촬영으로 취급해서는 안 된다.

전체 목록·원격 ETag·로컬 SHA256은 [download manifest](../data/external_audit/spacenet_paris_20261013/download_manifest.json)에 있다. S3 multipart ETag를 SHA256이나 단순 파일 MD5로 해석하지 않았다. SHA256은 내려받은 파일의 로컬 식별값이며, 별도의 공식 SHA256 대조를 수행했다는 뜻은 아니다.

- MS SHA256: `0c7b0a9b6c01d046a50bb5794a5a67b644e1944ef4ae6eb31445f07bce5a0256`
- PAN SHA256: `7e3fabbcd5517c091076c34459c77fa47c670738e4d20a9d4f0c5fdc3d54622f`

## 4. 로컬 PanCollection과 실제 픽셀 대조

MS 원본의 4096×4096 전체 영역에서 각 patch의 **동일 band 순서, 직접 DN, 정수 좌표 crop, 회전·반전 8종(D4)** 관계를 검색했다. 후보 검색은 8-band pixel vector hash로 가속하고, 후보가 있는 경우 최종 전체 crop에 `np.array_equal`을 적용한다. 정규화·허용오차·재표본화는 사용하지 않았다.

| 로컬 대상 | 대조 배열 | 검사 수 | 직접 일치 |
|---|---|---:|---:|
| WV3 train | GT, 8×64×64 | 9,714 | 0 |
| WV3 valid | GT, 8×64×64 | 1,080 | 0 |
| WV3 FR20 | MS, 8×128×128 | 20 | 0 |
| WV3 RR20 | GT, 8×256×256 | 20 | 0 |
| 합계 | | **10,834** | **0** |

원본 16,777,216개 pixel vector를 검색했다. query anchor hash 43,337개 중 source hit는 0이며, 후보 수 제한 초과와 미완료 행은 없다. 모든 입력은 uint16 범위 내 정수 DN이었다.

도구 단위 테스트 7개가 통과했다. 추가로 실제 다운로드 TIFF의 `(y=1001,x=2033,h=64,w=64)` crop을 D4 8종으로 변환한 양성 대조는 모두 원좌표 한 곳을 정확히 복구했다.

**한계:** 이 결과는 선택한 Paris 한 타일의 직접 DN crop 관계 부재만 검증한다. 다른 74개 MS 타일, band 순서 변경, radiometric 변환, 재표본화된 동일 지표 영상까지 배제하지 않는다. 전체 Paris 불일치의 픽셀 증명이라고 주장하지 않는다. 별도로 확인한 도시·촬영 출처 차이가 기존 corpus의 원본 후보로 보지 않는 주된 근거이다.

전체 행별 결과·파일 SHA·검사 한계: [exact-match JSON](2026-10-13_spacenet-paris-exact-match.json).

## 5. PAN/MS 쌍의 실제 기하·값 검증

최종 로컬 원본 전체를 대상으로 검사했다. [geometry JSON](2026-10-13_spacenet-paris-geometry-local.json).

| 항목 | 실측/메타데이터 |
|---|---|
| 센서·획득 | 양쪽 `WV03`, 동일 CATID `1040010018805C00` |
| PAN | 16384×16384×1, uint16 |
| MS | 4096×4096×8, uint16 |
| XML band 순서 | Coastal, Blue, Green, Yellow, Red, Red-edge, NIR1, NIR2 |
| PAN/MS 전체 DN | 각 band 최소 1, 최대 2047; 0 및 2047 초과 없음 |
| 융합 여부 | `PANSHARPENALGORITHM=None` |
| 투영 | WGS84 geographic, EPSG:4326 |
| 격자 비율 | x=4, y≈4.0000000024 |
| 선택 타일 외곽 footprint 차이 | 최대 약 0.00000965 PAN pixel |
| 영상 처리 수준 | Standard2A/LV2A, radiometric corrected, MTF resampling, Coarse DEM |

**첫 MS pixel 중심이 첫 PAN pixel 중심보다 우·하로 1.5 PAN pixel 떨어지는 것은 정상적인 4:1 중심 위상이다.** 두 TIFF 모두 PixelIsArea이며 tiepoint가 `(0.5,0.5)`다. 이를 외곽 좌표로 환산하면 선택 타일의 footprint는 일치한다. 이 1.5를 PAN Aligner가 제거할 misalignment로 간주해서는 안 된다.

반대로 지도상 격자 일치가 건물·edge 수준의 실제 PAN/MS 잔차 정합을 보증하지는 않는다. 이번에는 이미지 기반 residual shift, scene 내 위치별 shift 분포, RR 기준 b를 측정하지 않았다.

또한 여기서 '원본'은 **SpaceNet 배포의 비-pansharpened PAN/MS 영상**이라는 뜻이다. XML상 이미 보정·재표본화된 제품이며, 미보정 sensor raw telemetry라는 뜻은 아니다.

유효영역과 전체 모자이크 사용에는 추가 주의가 필요하다.

- TIFF/VRT에 명시적인 nodata 선언이 없었다. 값 1을 임의로 nodata로 처리하면 안 된다. 유효영역·구름/그림자 판정은 별도이다.
- 전체 모자이크 크기는 MS `18719×58105`, PAN `74875×232417`로 끝단이 정확히 4배가 아니다. 선택한 내부 타일의 정상 관계를 전체 경계까지 무조건 확장하면 안 된다.
- VRT의 상대 경로는 `MUL/...`, `PAN/...`이다. 현재 `metadata/` 위치와 `MS/` 저장 구조에서 그대로 열면 경로가 맞지 않는다. 이번 검사는 TIFF를 직접 열었으며 VRT는 참고 메타데이터로만 썼다.

## 6. RR-only method에 활용하려면

새로운 외부 WV3 데이터 조건에서 다음 준비는 가능하다. 아직 실행하지 않았다.

1. 실제 CATID를 source-scene ID로 유지하고, 원본 타일·전역 crop 좌표·band 순서·온라인/오프라인 augmentation 변환을 기록한다. 타일 ID와 촬영 ID는 별도 필드로 둔다.
2. native PAN/MS에서 명시한 MTF·decimation phase·경계 처리로 RR 자료를 만든다. 동일 FOV의 PAN256/MS64라면 RR 입력은 PAN64/MS16, GT는 원래 MS64가 되는 구조다. **FR loss를 추가할 필요는 없다.**
3. b는 확정한 RR 입력과 RR 좌표계에서 추정한다. 신뢰도 높은 여러 RR 영역의 근거를 공통 방향/좌표로 환산해 scene prior를 검증하고, texture 부족·공간적 불일치 시 임의 prior를 만들지 않는다.
4. 단일 촬영의 여러 타일을 독립 scene split으로 포장하지 않는다. 공간 분할에는 crop·필터 support 및 prior 추정 범위를 반영한 격리가 필요하고, 독립 acquisition 일반화 평가는 별도 촬영이 필요하다.
5. 현재 준비 loader는 명시적 H5/NPY 계약이다. GeoTIFF를 즉시 학습에 연결하지 말고, DN/좌표/phase/valid mask를 보존하는 변환과 검증이 필요하다.
6. PanCollection-only 결과와 외부 SpaceNet 사용 결과를 명확히 분리한다. 기존 train의 mapping을 이 CATID로 대체하거나, FR20을 training source로 전용하지 않는다.

[SpaceNet 2 공식 페이지](https://spacenet.ai/spacenet-buildings-dataset-v2/)에는 CC BY-SA 4.0과 인용 지침이 안내되어 있다. [전체 데이터 안내](https://spacenet.ai/datasets/)는 데이터셋별 조건이 다를 수 있음을 명시한다. 실제 원본/파생 데이터 재배포 전에는 해당 제품에 적용되는 조건을 확인하고 출처·처리 이력을 보존해야 한다.

## 7. 재현 도구와 수행 범위

```bash
/home/knuvi/miniconda3/envs/pancrafter/bin/python tools/download_spacenet_paris_audit.py
/home/knuvi/miniconda3/envs/pancrafter/bin/python -m unittest tools.test_audit_spacenet_exact_match -v
/home/knuvi/miniconda3/envs/pancrafter/bin/python tools/audit_spacenet_exact_match.py --tiff data/external_audit/spacenet_paris_20261013/AOIs/AOI_3_Paris/MS/16FEB29111913-M2AS_R08C3-055649178040_01_P001.TIF --output /tmp/spacenet_exact_recheck.json
/home/knuvi/miniconda3/envs/pancrafter/bin/python tools/audit_spacenet_geometry.py --local-root data/external_audit/spacenet_paris_20261013/AOIs/AOI_3_Paris --output /tmp/spacenet_geometry_recheck.json
```

검사 JSON 출력은 새 경로를 요구하며 기존 결과를 덮어쓰지 않는다. 다운로드 재실행은 기존 파일의 크기·로컬 SHA를 manifest와 대조한다. 진단 도구와 결과만 추가했으며 기존 학습 경로·config·prior·data manifest에는 연결하지 않았다. commit/push는 하지 않았다.
