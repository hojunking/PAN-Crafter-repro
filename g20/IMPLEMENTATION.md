# G20 implementation and verification — 2026-09-20

## 구현 상태

요청한 MD/CSV를 SHA256으로 고정하고 별도 G20 campaign을 구현했다.
생성된 기본 config는 32개이며 registry는 조건부를 포함한 56개 슬롯이다.
기존 QG40/FH12/FH20R1 numerical source와 진행 중 runtime은 변경하지 않았다.
이 작업에서 시작·종료·hold 전환·cron 설치·Sheet 쓰기·commit/push는 하지 않았다.

| 계획 항목 | 구현 위치/검증 |
|---|---|
| 전 서버 GF2, fixed architecture, 32+24 슬롯 | `plan.py`, `config/g20/G20_Registry.json`; 미결정 슬롯 실행 금지 |
| C030/C100/C300, BASE/A1/A9/E1/E4/K05/K20 | `losses.py`, `training.py`; 한 축 변경, 전체 A LR schedule, 분리 U/A gradient |
| 동일 seed 초기값·sample/view·resume | `model.py`, `training.py`; full U/A hash, architecture signature, sampler/corruption/optimizer/fullstate |
| R0 pin 및 R1–R4 별도 provenance | `references.py`, `calibration.py`; τ/qref 실측, 전체 q cache, q16×4 online 검증 |
| 실제 TA/S92001 evaluator parity | `parity.py`; 원본 자동 탐색, 두 checkpoint 실제 재추론, ΔH≤1e-5·ΔE≤1e-4 |
| 원래 공식 RR/FR 및 signed Ds | `evaluation.py`; RR20/Q4, FR20/native PAN/full512, 실제 Q_high/Q_low 20×4와 Ds 재구성 |
| 50개 후보와 5개 선택점 | `postrun.py`; Exact50K/VAL 주 분석, RAW/TARGET/E_MIN test-aware 분리 |
| 16/18/20h·atomic block·paired 판정 | `policy.py`, `controller.py`; 전체 block+평가부채 예약, same-server 실측 wall time×안전계수 |
| 전환·중단·closeout | `transition.py`, `training.py`, `postrun.py`; 기존 block drain, 18h optimizer 종료, 20h까지 저장 후보 평가 |
| 조건부 확인·전이 | immutable screen/reference pin, s1 단일 global receipt, 원래 X* 서버의 2×2 difference-in-differences |
| 진단·Sheet·배포 | `diagnostics.py`, `upload.py`, `deployment.py`; metric 4자리 표시, full-precision 판단, protected header/readback |

## 검토 중 보완한 경계 조건

- 16h 이후에도 **이미 입장한 block**의 후속 case는 18h까지 완료할 수 있다.
- Calibration/평가의 종료 시각을 trainer의 18h와 분리하여 20h closeout을 허용한다.
- 정확한 50K candidate/fullstate 저장 후 상태 JSON 기록 전에 중단되면, 체크섬·
  source/data/config·model·scheduler·sampler/exposure를 검증해 완료 상태를 복원한다.
  이 복구에는 optimizer update가 없다.
- DataLoader 대기와 초기화 시간을 누락하지 않도록 train subprocess 전체 wall time을
  예약 추정에 반영한다. 재개/실패 시도 시간도 누적한다.
- 이전 마지막 admitted case가 실패/일시중단되면 GPU가 비어도 drain 완료로 취급하지 않는다.
- Upload retry로 달라지는 delivery 상태 JSON은 immutable screen 근거에서 제외한다.
  성능/초기값/config/checkpoint/fullstate 근거는 계속 검증한다.
- TARGET-only 공동목표는 `JOINT_SINGLE_RETEST`일 뿐, paired 개선/4-seed 성공을
  대신하지 않는다. 실패한 Exact50K/VAL guard도 함께 남긴다.
- 선택되지 않은 조건부 슬롯을 BASE로 채우거나, 다른 서버의 BASE를 local control로
  대체하거나, 미완 학습을 Exact50K로 표시하지 않는다.

## 검증

CPU-only 환경: 기존 `pancrafter` conda Python, `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1`.

- G20 unit/integration suite: 143개 통과.
- 기존 QG40 regression suite: 161개 통과.
- 기존 FH12/FH20R1 regression suite: 232개 통과.
- 합계 536개 테스트 통과.
- 추가 PA synthetic warp/aligner/loss checks 통과. 그 스크립트의 별도 외부
  evaluator gates는 데이터/환경이 없어 SKIP였으며 PASS로 계산하지 않았다.
- `build`: 기본 YAML 32개 + registry 56개 확인. `status`: 공통 clock 미설정,
  `DEFINED_NOT_DEPLOYED`, 실제 학습 미시작 확인.
- 테스트의 fake Sheet/temporary data를 실제 업로드·실데이터 parity로 간주하지 않는다.
  fault-injection 테스트의 의도된 P1 timeout 경고는 테스트 실패가 아니다.

## 실제 실행 전 남은 실측

현재 로컬 inventory에는 QB reference 두 개만 있으며, pin된 GF2_TA와 S92001
원본 자산은 없다. 따라서 실제 R0 q-cache/output/evaluator parity와 실제 GPU
학습/처리량은 이 작업에서 인증하지 않았다. 올바른 원본을 전달한 서버에서 P0를
실행해야 한다. 경로만 있거나 rounded Sheet 값만 있으면 PASS를 만들지 않는다.

실행·전달 명령은 [README.md](README.md), R0 및 parity 입력은
[REFERENCES.md](REFERENCES.md)에 정리했다. 새 공통 t0는 실제 준비/전환 시작 시에만
한 번 정하며, 기존 QG40 시간이나 서버별 독립 clock을 재사용하지 않는다.
