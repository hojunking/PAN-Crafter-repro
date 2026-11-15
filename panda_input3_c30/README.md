# INPUT3 fresh 30K — s1 WV3 / s2 QB / s3 GF2

2026-11-15 사용자 요청: 아직 시작하지 않은 INPUT3 case는 30K로 교체하고,
이미 시작했거나 완료한 case는 기존 결과를 보존하면서 새 30K 실행을 추가한다.

## 실행

사용자가 push한 뒤 각 서버에서 같은 commit을 pull하고 실행한다.
각 서버의 기존 고정 donor만 사용하며 Stage 1 재학습/weight 이동은 없다.

```bash
python3 -m panda_input3_c30.enqueue --server s1
python3 -m panda_input3_c30.enqueue --server s2
python3 -m panda_input3_c30.enqueue --server s3
```

자기 서버 명령 하나만 실행한다. `--check-only`는 donor·데이터·Sheet 사전 점검만 한다.
중단된 새 컨테이너를 점검하고 재개할 때는 `--resume`을 추가한다.
기존 `panda_input3` / `panda_input3_targets` launcher로 되돌아가지 않는다.
새 supervisor가 기존 실행의 종료와 C30 시작을 함께 관리한다.

## 전환 방식

- 실행 중인 기존 모델은 complete optimizer/RNG/stream checkpoint를 저장한 뒤
  큐 dispatcher 교체를 위해 잠깐 재시작한다. 기존 설정/40K 또는 50K endpoint까지 계속 학습한다.
- 서버 로컬 상태와 config/trace/weight 유무로 시작 여부를 보수적으로 판정한다.
  Sheet의 대표 step을 학습 진행 step으로 해석하지 않는다.
- 미시작 old case는 `REPLACED_BY_C30`으로 남기고 원래 horizon으로 실행하지 않는다.
- **미시작 case의 C30 교체 → 시작/완료 case의 C30 재실험** 순서이며,
  각 그룹 안에서는 원래 A/B/C 순서를 유지한다. 서버별 총 C30 실행은 항상 3개다.
- 이미 완료한 결과·weight·Sheet 행은 보존한다. 실패한 기존 run도 C30 재실험을 배정한다.
- 결정은 `work_dir/panda_input3_c30/PANDA_S123_INPUT3_C30_20261115_v1/<server>/<sensor>/cutover.json`
  에 고정하며 재실행해도 중복 추가하지 않는다. 기존 state 백업도 함께 저장한다.
- 이번 변경은 **30K를 추가 finetuning하는 것이 아니다**. C30은 같은 seed에서 0 step부터 새로 학습한다.

## 고정 조건

| 서버 | 데이터 | Stage 1 고정 donor | Student | peak LR | 새 schedule |
|---|---|---|---|---|---|
| s1 | WV3 | 기존 R1@27K | W112/D123/UBIL | 1.1e-4 | warmup100 + cosine30K, 최종 LR0 |
| s2 | QB | 기존 R1@32K | W128/D123/UBIL | 1.0e-4 | 동일 |
| s3 | GF2 | 기존 R2@42K | W128/D123/UBIL | 2.5e-4 | 동일 |

A = 원본 PAN, B = 원본 PAN + corrected PAN,
C = corrected PAN + (corrected PAN − 원본 PAN); 모두 interp23 MS를 함께 입력한다.
RGBN·Lanczos-4·공통 유효영역·scale1·α=β=0·γ=.25·q=0 예외처리·GT Hard를 유지한다.
원래 calibration 캐시를 그대로 재사용하며 Teacher/q/mask는 세 case 모두 동일하다.
seed261018101, batch48/micro12, AdamW 및 데이터 순서 설정도 동일하다.
원래 method/optimizer/evaluator 코드를 공유하고, C30 worker에서만 설정을 주입한다.
기존 실행의 release SHA와 checkpoint config는 변경하지 않는다.

## Sheet / 평가

매 1K RR20/FR20 평가, 양쪽 PAN 기준 FR metric과 weight를 로컬 보존한다.
`WV3-ablations`, `QB-ablations`, `GF2-ablations`에 `INPUT3-C30` 이름으로 별도 게시한다.
대표값은 corrected HQNR 하한 통과 후보 중 ERGAS 최소, 추가 step 행은 전체 목표 동시 통과만 게시한다.
소수점 4자리 표시, 판정은 반올림 전 수치이다.

| 데이터 | corrected HQNR ≥ | ERGAS ≤ | PSNR ≥ |
|---|---:|---:|---:|
| WV3 | .9585 | 2.040 | 없음 |
| QB | .9315 | 3.500 | 없음 |
| GF2 | .9645 | .522 | 45.08 |

HQNR만 통과한 대표값의 ERGAS/PSNR 미달 표시는 유지한다.
기존 40/50K 결과의 30K checkpoint와 새 C30 학습 결과를 같은 실험으로 합치지 않는다.

## s1 적용 기록

2026-11-15 17:36 KST 전환 시 A는 40K 완료, B는 30,277 step에서 완전한
재개 checkpoint를 저장했고, C는 0 step이었다. 실행 순서는 다음과 같다.

1. 기존 B를 같은 optimizer 상태로 40K까지 완료.
2. C-C30, 3. A-C30, 4. B-C30 — 모두 0→30K 새 학습.

Sheet의 C30 대기 행은 `WV3-ablations` 920–922행에 연속으로 생성됐다
(행 위치는 다른 서버의 행 추가/삭제에 따라 바뀔 수 있다).
s2/s3의 순서는 이 기록을 복사하지 않고 각 서버에서 launcher를 실행하는 시점에 확정한다.

검증: 전환/설정/게시 정책 테스트 6개 통과. 실제 CUDA에서 입력별 Teacher/q/mask 공통성,
B/C 초기 함수 일치, 모델·optimizer·stream·RNG 저장 경계의 정확 복원 통과.
실행 중인 원래 INPUT3 코드 및 reporting release는 수정하지 않았다.
