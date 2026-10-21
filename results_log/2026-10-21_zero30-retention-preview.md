# ZERO30 시작 시 이전 Mature30 weight 보관 사전 검토

2026-10-21 KST. 범위는 완료된 s1 캠페인
`PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2`의 Stage2 6개 run이다.
읽기 전용 검증이며 **weight 삭제·이동은 하지 않았다.** 새 학습 시작과 독립적으로 관리한다.

기존 확정 보관 정책의 목표 `.960 / 2.045`를 그대로 사용했다. 새 ZERO30의
탐색 목표 `.959 / 2.045`를 이전 run의 Balance 선정에 소급하지 않는다.

## 검증된 세 역할

기존 `checkpoint_retention.policy.preview`로 config, run/selection 상태,
동일 모델의 RR/FR 평가, checkpoint sidecar 및 실제 weight SHA를 대조했다.

| case / seed | BEST_HQNR | BEST_ERGAS | BEST_BALANCE | 평가 weight 용량 → 세 역할 용량 |
|---|---:|---:|---:|---:|
| DOWN R1 | 17K | 48K | 36K | 1.422 → 0.122 GiB |
| DOWN R2 | 4K | 50K | 34K | 1.422 → 0.122 GiB |
| HALF R1 | 5K | 48K | 24K | 1.421 → 0.122 GiB |
| HALF R2 | 5K | 50K | 18K | 1.421 → 0.122 GiB |
| LOW R1 | 19K | 48K | 32K | 1.421 → 0.122 GiB |

각 행은 같은 case·seed·attempt의 평가 35개 안에서만 선정했다.
표의 용량 감소는 보호 예외와 게시 의존성을 적용하기 전의 **가상 보관량**이며,
실제 확보한 디스크 용량이 아니다.

FULL R1은 평가 목록 밖에
`stop_0031505_1792499418579836789.pt` (43,567,980 bytes)가 있어 도구가 거부했다.
원래 31,505-step resume 자산에 대한 별도 의존성 검증 전까지 전체를 보호한다.

## 삭제/이동 보류 이유 및 보호 대상

- 기존 게시기의 원본 증거 검사와 재게시가 실제 `.pt` 파일 및 SHA를 요구한다.
  완료된 readback만으로 원본 weight를 지울 수 없다. 정리 이력 지원을 추가하지
  않은 채 세 역할 밖 파일을 삭제하거나 이동하면 재게시 검증이 깨진다.
- Stage1 원본 35개 후보와 RRE receipt는 이 검토의 삭제 범위가 아니다.
- ZERO30이 사용할 s1 donor는 `G8_S1_U23`의 RRE-selected 50K checkpoint이며,
  SHA는 `62f1c445b8500070429f46503a68fb972a9144a4321fc9da10ccd45891186b8d`이다.
  기존 원본 파일과 Mature30 선정 receipt를 모두 보호한다.
- 모든 `train.jsonl`, `evaluation_*.json`, config/state/selection, sidecar,
  records/outbox, 업로드·readback 증거를 계속 보존한다.

따라서 이번 갱신에서는 보관 역할과 예외만 기록한다. 안전한 재게시 호환성과
원본 의존성 검증을 완료하기 전까지 어느 weight도 삭제하지 않으며,
이 보류 때문에 ZERO30 시작을 막지 않는다.
