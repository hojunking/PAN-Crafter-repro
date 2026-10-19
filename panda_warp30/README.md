# s3 / WV3 — fresh Stage 1 + interpolation / coordinate warp, 30h

이 캠페인은 기존 donor를 가져오지 않고 **Stage 1부터 새로 학습**한다. 실행은 s3에서 명시적으로 한 번 한다. `git pull`만으로 자동 시작하지 않으며 다른 프로세스·컨테이너를 중지하지 않는다.

```bash
bash scripts/run_panda_warp30_docker.sh s3
```

실행 스크립트는 s3의 기존 Docker 이미지와 등록된 WV3 자산을 사용한다. 이미지 자동 발견이 불가능한 경우에만 기존 이미지 이름을 `PAN_WARP30_IMAGE`로 지정한다. `gspread/server.txt=s3`, 기존 `gspread/account.json`, DLPan 평가 checkout, 기존 uploader Python이 필요하다. 새 패키지 설치나 서버 간 donor 전달은 없다. 사전 검증이 통과하면 같은 명령에서 학습과 업로더를 시작한다.

## 고정 조건

- 캠페인: `PANDA_S3_WARP_STAGE12_30H_20261019_v1`, s3 / WV3 전용.
- Stage 1: Aligner + W112/D123 Teacher, PAN→MS, RR 학습만. 기존 MASK2 reconstruction + `2 × Scharr absolute + 0.003 × epsilon consistency`, LR(T)=1e-4, LR(A)=1e-5.
- Stage 2: 같은 Aligner·Teacher 고정, W112/D123 Student fresh. 기존 conditional-q loss 그대로 `α=0.25, β=0.05, γ=0.5`, LR(S)=1e-4. 구조와 loss 계수는 case 간 동일하다.
- 각 run 50K optimizer updates, batch48=micro12×4, FP32/no TF32, 기존 flip/rot90·scheduler 유지. partial Stage 1은 donor가 될 수 없다.
- Stage 1 두 arm은 seed `261016101`과 초기 A/T·입력 순서·augment·epsilon 난수를 공유한다. Stage 2 R1은 `261018101`, 선택적 R2는 `261018102`; 각 seed 내 초기 S와 데이터 순서를 공유한다.
- 모델 선택은 **공식 FR20 HQNR 최대 → 동점 시 RR20 ERGAS 최소 → 먼저 평가된 update**. SCC는 출력·기록하지만 동점 기준은 아니다. HQNR/ERGAS 목표선은 주석이며 선택·업로드 제한이 아니다.
- Aligner의 기준 입력은 계속 제공 native LMS이다. 교체하는 것은 U-Net 입력 및 residual base의 LRMS 업샘플러이다. FR loss, GT 기반 FR 추론, 자동 ×4/¼ 변환은 없다.

## 등록 큐

| 순서 | case | 단계 / 변경 | donor | 우선순위 |
|---|---|---|---|---|
| 1 | S1_U23 | fresh PAN→MS, interp23tap base | 새로 학습 | 필수 |
| 2 | S1_BIC | 같은 fresh Stage 1, bicubic base | 새로 학습 | 필수 |
| 3 | U23_PAN | PAN→MS, full correction | U23 | 핵심 |
| 4 | BIC_PAN | PAN→MS, bicubic base | BIC | 핵심 |
| 5 | U23_MID | PAN↔MS 반반, 출력 반환 없음 | U23 | 핵심 |
| 6 | U23_MID_RETURN | 반반 입력 + 출력 MS 좌표로 역warp | U23 | 핵심 |
| 7 | U23_MS_RETURN | MS→PAN 입력 + 출력 MS 좌표로 역warp | U23 | 핵심 |
| 8 | U23_PAN_S050 | PAN→MS 보정량 0.5배 | U23 | 핵심 |
| 9 | U23_PAN_S000 | Student 보정량 0배 | U23 | 시간 여유 시 |
| 10 | U23_PAN_S025 | PAN→MS 보정량 0.25배 | U23 | 시간 여유 시 |
| 11–14 | U23_PAN / MID / MID_RETURN / MS_RETURN | 네 좌표 경로 R2 반복 | U23 | 네 벌 전체 시간 확보 시 |

등록된 실제 config/CSV는 다음으로 확인한다. 코드의 `plan.py`가 단일 원본이다.

```bash
python -m panda_warp30 catalog --server s3 --csv
python -m panda_warp30 check --server s3
python -m panda_warp30 smoke
```

`U23_PAN ↔ BIC_PAN`은 **Stage 1+2 전체 업샘플러 교체 효과** 비교이다. 서로 다른 Stage 1 donor를 사용하므로 동일 donor에서 업샘플러만 교체한 결과라고 해석하면 안 된다. 나머지 U23 좌표/배율 case는 동일한 선택 A/T checkpoint를 공유한다. R1은 단일 paired-seed 탐색이며 통계적 우월성 검증은 아니다.

## 좌표와 배율

`u=(dy,dx)`는 현재 PAN 격자 픽셀 단위 Aligner 출력, `d=s·u`이다. `W(I,u)(x)=I(x+u)`인 backward sampling이므로 보이는 영상 이동은 `−u`이다. `B`는 선택한 업샘플러로 생성한 HR base이다.

| frame | U-Net PAN | U-Net MS base | 학습 출력 | GT / 고정 Teacher 목표 |
|---|---|---|---|---|
| pan | W(P,d) | B | Y | 원래 MS 좌표 |
| mid | W(P,d/2) | W(B,−d/2) | Y_mid | W(GT,−d/2), W(T,−d/2) |
| mid_return | W(P,d/2) | W(B,−d/2) | W(Y_mid,+d/2) | 원래 MS 좌표 |
| ms_return | P | W(B,−d) | W(Y_pan,+d) | 원래 MS 좌표 |

반반 좌표 분담은 전체 보정량을 줄이는 것과 다르다. `MID(s=1)`의 두 영상 사이 상대 보정은 `u`; `PAN_S050`은 상대 보정 자체가 `0.5u`이다. RR와 FR에 같은 `s`를 사용하며 해상도에 따라 `u`를 자동 변환하지 않는다.

고정 Teacher는 항상 자신이 학습한 업샘플러와 **full u PAN→MS**로 예측한다. Student의 배율 실험 때문에 Teacher까지 약화시키지 않는다. 따라서 S000은 Student 입력 정합의 기여를 보는 대조군이지, Teacher를 포함한 전체 학습계에서 정합을 제거한 대조군은 아니다.

출력 역warp는 학습 autograd 내부에 있고 평가에도 동일하게 적용한다. `W(W(I,u),−u)`는 보간 때문에 정확한 항등함수가 아니다. MID_RETURN/MS_RETURN의 resampling 손실도 실제 방법 비용으로 남긴다. MID 무반환 출력은 중간 좌표이므로 원본 참조 평가와 좌표 차이가 있다. 평가 참조를 이동시켜 이 차이를 감추지 않는다.

## interp23tap과 fractional warp의 구분

- `interp23tap`은 LRMS ×4 업샘플링만 수행한다. 입력 LRMS flip/rotation **이후** 적용한다.
- DLPan 격자 위상 때문에 augmentation에 따른 정수 roll 보정이 필요하다. 공유 helper가 16가지 flip/rotation 조합을 처리하며, HR에서 추가 소수 warp를 하지 않는다.
- PAN/base/GT/Teacher/output의 임의 소수 shift는 기존 bicubic warp를 유지한다. 따라서 bicubic 변수를 모두 제거했다는 주장은 하지 않는다.
- Stage 1·Teacher·Student·추론이 같은 base helper를 사용한다. legacy constructor에 남은 `BICUBIC_MS` 문자열은 네트워크 구조 계약용이며 실제 forward base는 별도 명시되고 checkpoint에 고정된다.
- loss support는 실제 다단계 warp footprint의 교집합이다. 역warp에서 이동량을 상쇄해 경계를 유효하다고 처리하지 않는다. 다만 이 mask는 U-Net 전체 receptive field나 interp23tap circular 경계 영향까지 제거했다는 보장은 아니다.
- uncertainty/CDF는 각 geometry·배율의 GT/Teacher/q 좌표와 support로 RR train 전체를 다시 calibration한다. 다른 geometry cache를 재사용하지 않고 R2의 동일 geometry cache만 재사용한다.

## 30시간 운영

캠페인 프로세스 시작 시각을 저장한다. 본 실행의 자산 검증·profiling·calibration·학습·평가를 포함한 **누적 wall time 30h**이며 재실행·중단으로 deadline이 연장되지 않는다. Docker 시작 전 읽기 전용 사전 검사는 이 clock 바깥이다. 마지막 30분은 안전 저장용으로 남기며 29.5h에서 새 update/scene을 중단한다. 진행 중 원자 작업의 정확한 종료 시각을 OS 수준 강제 kill로 보장하는 방식은 아니다.

대략적 시간 prior는 Stage 1 한 벌 4h, Stage 2 한 벌 2.1h이다. 두 Stage 1+핵심 여섯 Stage 2는 prior상 20.6h이지만 서버 실측 보장은 아니다. 런타임이 update p90·40장 평가·checkpoint 저장 시간을 측정하고, 선택 case는 calibration 여유 + 20% safety를 더해 남은 시간에 맞을 때만 시작한다. R2는 네 벌을 하나의 그룹으로 admission한다. 예측보다 느리면 이 그룹도 중간 안전 중단될 수 있다.

**14벌 전부의 30h 완료를 약속하지 않는다.** 핵심 큐는 우선순위대로 진행하며 시간 초과는 `STOPPED_SAFE`, 핵심 완료 후 선택 case 미실행은 `FINISHED_CORE_OPTIONALS_BUDGET_SKIPPED`로 기록한다. partial run을 완료나 donor로 승격하지 않는다. GPU는 사전 idle 확인 후 사용하고, 다른 서버/기존 실험을 기다리는 cross-server lock은 없다. 같은 출력 폴더의 중복 controller만 로컬 nonblocking lock으로 거부한다.

동일 코드·데이터·환경에서 중단 재개 시 optimizer/RNG/stream/현재 update를 복원한다. resume-only 안전 checkpoint는 공식 평가 checkpoint로 업로드하지 않는다. 마지막 정규 평가 결과는 보존하지만 제한 시간에 off-grid 최종 평가를 추가로 보장하지 않는다. 데이터/code pin이 달라지면 혼합 재개를 거부하므로 실행 중 checkout을 변경하지 않는다.

## 평가·저장·시트

- 공식 평가: 원래 RR20/FR20 전체, 원본 참조, unmasked. 학습용 mask·midpoint target을 공식 참조로 사용하지 않는다. FR GT는 추론/metric에 사용하지 않는다.
- 로그 HQNR·SCC·ERGAS 출력. RMSE·CC·JQM을 포함한 모든 공식 metric과 scene별 결과를 JSON·로컬 records/outbox에 보존한다.
- **모든 정규 평가 checkpoint**를 현재 workbook `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE`의 `WV3-ablations`에 업로드한다. 성능 목표 미달 행도 삭제하지 않는다. 날짜·서버·seed·step·실측 비용·좌표 정책을 포함한 기존 64열 포맷과 표시 precision을 사용한다. raw JSON은 full precision이다.
- 로컬 경로: `work_dir/panda_warp30/PANDA_S3_WARP_STAGE12_30H_20261019_v1/s3/WV3/`. `runs/.../attempt_01`에는 config/train/evaluation/models, root에는 donor 선택 receipt·calibration·outbox·publication readback 기록이 남는다.
- Docker는 학습만, host uploader는 로컬 outbox 기반 전송·readback 확인을 담당한다. 네트워크 업로드 실패는 학습 결과를 지우지 않으며 같은 publisher로 재시도할 수 있다.

```bash
docker logs -f panda-warp30-wv3-s3-v1
```

이 문서의 검증은 CPU 단위/통합 테스트 및 stub launcher 검증이다. s3 GPU에서의 실제 속도·30h 완료량·클라우드 readback 성공은 실험 실행 후 기록으로 확인한다.
