# MethodPlan-v1.3 PAN Aligner 변경·검증

작성일: 2026-10-13 (Asia/Seoul)

기준: [MethodPlan-v1.3](../research_log/PANDA_Method_Implementation_Validation_Plan_2026-10-13_v1_3.md)

문서 SHA256: `122472a5c89b2b1e827bbb5c46d19d02821efdae264fb3b311ec2bd5f07f64c0`

## 결론

v1.3에서 요구한 head conditioning 변경을 구현하고 CPU 회귀 테스트 **220개를 통과**했다. 실제 학습·원격 Google Sheet 쓰기·commit/push는 이 작업에서 실행하지 않았다. 테스트 통과는 raw-prior head의 성능 향상이나 실제 데이터 admission 완료를 뜻하지 않는다.

## 구현 변경과 유지 항목

| 항목 | 반영 내용 |
|---|---|
| FC1 입력 | v1.2 `[GAP h; d_s]` → v1.3 `[GAP h; sg(b_s)]` |
| prior 의미 | current PAN grid의 raw signed `(dy,dx)`, 검증된 grid/phase 변환 및 실제 영상 증강 반영 |
| 금지한 변형 | unit normalization, magnitude clamp, 추가 scale/embedding, ε에 따른 prior 재추정/덧셈 없음 |
| head 구조 | FC66→32→SiLU→FC32→2 유지 |
| 출력 | `c = b_s + αd_s + βn_s`, `d_s=b_s/‖b_s‖`, `n_s=(-d_x,d_y)` 유지 |
| 초기화 | FC2 weight/bias 0, fresh 초기 correction `c=b_s` 유지 |
| 나머지 모델 | Aligner 공통 LN ResBlock, GN stem/joint, Teacher/Student U-Net 유지 |
| 학습 | 기존 loss·warp·ε 분포/반경·stop-gradient·Stage2 freeze 유지 |

실제 trainable parameter 수는 WV3 8-band **105,522**, QB/GF2 4-band **104,946**이다. v1.2와 head 차원이 같아서 parameter count도 동일하다. 새 revision marker는 int64 buffer이며 trainable parameter가 아니다. GPU throughput/VRAM 및 서버별 FLOPs는 이번 검증에서 측정하지 않았다.

핵심 코드:

- [geometry.py](../panda_method/geometry.py): raw signed prior concat, head 값 진단, incompatible state load 차단.
- [stages.py](../panda_method/stages.py): conditioning manifest 및 기존 native forward의 진단값 반환 옵션.
- [training.py](../teacher48/training.py): RR/FR별 prior dy/dx 범위 및 FC1 preactivation/SiLU min/max/RMS/finite 기록.
- [plan.py](../teacher48/plan.py), [runtime.py](../teacher48/runtime.py), [evaluation.py](../teacher48/evaluation.py), [reporting.py](../teacher48/reporting.py): 실행·평가·보존·업로드 identity에 v1.3 계약 반영.

RR/FR head 범위 통계는 기존 native forward를 재사용한다. 진단을 위한 별도 Aligner forward는 추가하지 않는다. microbatch 전체 범위는 extrema, RMS는 제곱 평균으로 집계한다. 비유한 head 값을 조용히 정규화하거나 대체하지 않고 실패시킨다.

## v1.2와의 경계

FC shape가 같다고 기존 학습 weight를 v1.3으로 묵시적으로 재사용하지 않는다.

- A state의 persistent `_conditioning_revision=13`을 검사한다. 구 marker 누락/불일치는 직접 로드·부모 모듈의 중첩 로드·`strict=False`에서도 A child weight를 복사하기 전에 거부한다.
- Stage1 checkpoint의 method/pipeline 계약, Stage2 calibration의 A state/pipeline hash를 확인한다. 초기 `c=b_s`가 우연히 같아도 구 calibration이 유효해지는 것은 아니다.
- 새 campaign은 `PANDA_S1_M13_TEACHER48H_S123_20261013_v1`, run prefix는 `S1M13_48H_`다. M12는 `parent_campaign_id`에 계획 출처로만 남긴다.
- 구 config, clock, outbox, recovery의 final-status 내부 평가 identity를 v1.3으로 재표기하지 못하도록 검사한다. marker/버전 문자열만 바꾸는 자동 migration은 제공하지 않는다.
- `config/teacher48/`의 세 catalog 및 세 baseline draft에 새 계약을 반영했다. **16 F / 9 T / 3 horizon, seed, loss 계수, LR/WD 배율, 서버 배치, 목표 gate, 평가 cadence, 48h 예산은 변경하지 않았다.**

## Google Sheet 주소

현재 Teacher48 업로더 workbook ID는 사용자 전달 URL과 이미 일치했다:
`1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE`.

| 서버 | 업로드 목적지 | sheetId |
|---|---|---:|
| s1 | WV3-ablations | 261001010 |
| s2 | QB-ablations | 261001011 |
| s3 | GF2-ablations | 261001012 |

URL의 `gid=261013013`은 [제공된 로컬 schema](../research_log/gspread_rules/PAN_FIXED64_V1.schema.json)상 WV2-main이다. 이는 workbook 링크로 참고했으며, 기존 사용자 지시인 데이터셋별 ablations 업로드를 WV2-main으로 바꾸지 않았다.

fixed64 목적지 검증, 전체 평가/per-scene/상태 기록, 로컬 outbox·원정밀도 payload·receipt 보존은 유지한다. 신규 method metadata는 기존 JSON 필드로 보내므로 열을 추가하지 않는다. 실제 원격 접근 권한·현재 탭 내용·헤더·업로드 readback은 이번 작업에서 검증하지 못했다. 위 확인은 로컬 schema와 코드 및 fake transport 테스트 기준이다.

## 검증 결과

환경: `/home/knuvi/miniconda3/envs/pancrafter/bin/python`, CPU 합성 tensor·임시 자산·fake transport.

| 명령 | 결과 |
|---|---:|
| `python -m unittest discover -s panda_method -t . -q` | 69 PASS |
| `python -m unittest discover -s teacher48 -t . -q` | 132 PASS |
| `python -m unittest ta2.test_model_loss -q` | 19 PASS |
| `python -m compileall -q panda_method teacher48` | PASS |

주요 값 기반 검증:

- FC1 hook에서 `(1,0)`과 `(3,0)`이 단위 방향으로 같아지지 않고 서로 다른 실제 값으로 전달됨.
- 음수 부호, `(dy,dx)` 순서, grid/phase 변환 및 flip/rotation이 head와 최종 correction에 동일하게 반영됨.
- RR/FR 각각 native·ε branch에 동일한 원래 prior를 전달하며 prior gradient가 없음.
- zero-init 시 prior 크기와 무관하게 `c=b_s`, 제어된 nonzero head에서는 magnitude conditioning 차이가 실제 출력에 작용함.
- Stage2 frozen A 및 student 배포 경로에도 raw signed prior가 전달됨.
- 구 checkpoint/calibration/config/clock/outbox/내부 평가 identity 혼합 거부.
- RR/FR 진단이 추가 forward 없이 기록되고 microbatch extrema가 평균값으로 축소되지 않음.
- 정적 catalog가 코드 생성값과 일치하며 URL의 WV2-main으로 잘못 라우팅하지 않음.

TA2 테스트에서 PyTorch 기존 autocast deprecation warning이 있었으나 실패는 없었다. 독립 검토에서도 이번 수치 변경에 대한 차단 오류는 발견되지 않았다.

## 실제 실행 전 남아 있는 조건

이번 변경은 head 구현 변경이지 데이터 원천 문제 해결이 아니다. [기존 자산 조사](2026-10-13_teacher48_existing_assets.json)에서 확인된 training-only native FR 자산·실제 source-scene/crop mapping·train/eval 분리 증거·승인된 prior/grid 변환 자료의 등록 부족은 그대로다. baseline draft의 미확정 항목도 임의로 채우지 않았다.

FR benchmark 20장을 학습으로 대체하거나 RR을 확대해 native FR로 선언하지 않는다. 이 자료와 baseline이 확정된 후 별도의 실제 admission/GPU smoke가 필요하다. 자세한 계약과 실행 방법은 [Teacher48 README](../teacher48/README.md)를 따른다.
