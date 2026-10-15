# ABLR2X 구현 및 검증 — 2026-09-23

대상: `PANDA_ABLR2X_S123_Continuous_C17_GF2_Bundle_2026-09-23`의 계획서,
18-component catalog, 19-relation graph, 기존190/추가110/전체300 CSV,
DesignRegistry·source manifest·checksum·생성기·18개 정적 테스트.

## 반영 내용

| 영역 | 구현 |
| --- | --- |
| 기존 s1/WV3·s2/QB | campaign/run ID·seed·기존190 config·기존 state/비용/임계값 보존. 기존 stage를 실제 조사해 C17만 append |
| C17 | C03와 동일한 Student W104/D122/PLH. matched TZERO exact50K Aligner만 독립 복제, trainable, U/A plain-GT, alpha/beta/edge=0 |
| 참조 차단 | C17 Teacher U 출력·uncertainty/q/cache/calibration 미사용. 정확한 같은 서버·seed·sensor endpoint만 허용 |
| 비교 근거 | 실제 초기 U·native sample/view stream·Teacher 공통 초기값/stream 검증. 원 C03 재사용 불가 시 별도 PAIR_REPAIR C03, 원 관측은 보존 |
| 누락분 처리 | 모든 등록 BOOT/REFRESH/VERIFY 조사. oldest-ready 우선, 기존 작업 사이 C17 최대2개, 재시작에도 제한 유지 |
| s3/GF2 | 신규100 run. C4/DN1023/Q4/GF2 MTF, train19809/val2201, native RR20/FR20. 기존 LP 레시피, QB msfix 미적용 |
| GF2 seed | 실제 학습 시작 이력 조사, 충돌 시 관측 전에 결정론적 mapping 고정. 원 CSV·원190 seed는 변경하지 않음 |
| 무기한 운영 | `--until-operator-stop` 별도 명시 승인. 기존72h lease 보존, 자동 갱신 없음, s4/s5 보호 |
| 소스 전환 | 원 실행 완료까지 원 frozen release 유지. CPU-only 대기 → 실제 old/new 수치 비교 → 검증된 bridge → 새 preflight |
| 통계 | legacy17 threshold 유지. core17/extended18 구분, A17 5쌍 필요, A17_SCRATCH 자동 queue 제외. source/data/coverage가 다른 누적 통계 분리 |
| 보고서 | 기존95 보고서 보존, 새100 보충 보고서를 별도 경로에 저장. repair는 추가 component/독립 반복으로 세지 않음 |
| Sheet | raw lane별 tab, 실제 GF2 C00–C17만 `ablations` marker 영역. 기존 영역 보호·표시 정밀도·readback·bounded outbox 재시도 |
| 안전성 | local 중복 실행 방지, max(100GiB, 다음 block 예상 쓰기×2), 자동 삭제 없음, 원 fullstate·queue·비용 보존 |

## 검증 결과

- 제공 bundle의 정적 테스트 **18개 통과**.
- 원 문서 SHA 계약 **5개**, 새 bundle SHA 계약 **18개** 모두 일치.
- 원 commit `6dde5ea`의 기존190개 `build_config` 출력과 현재 legacy190개가 **전부 정확히 동일**.
  고정 config-map digest: `724ca4dda88fb73a197d35d86f6afcd2f2d08af5d5b29f4c4c76c1161f75937c`.
- 전체 ABLR2 회귀 테스트 **201개 통과** (53.244초, CPU only).
- C17 Teacher 출력/q 접근 금지, U 초기값 동등성, Aligner 실제 gradient·갱신, Teacher 불변을 검증.
- GF2 DN1023 데이터 로딩 및 기존 GF2 RR metric 계산과의 일치, FR GF2 MTF/JQM variant를 검증.
- 기존95 상태·seed JSON/JSONL·누적 비용·고정 threshold byte 보존 및 C17 5개 추가를 검증.
- A17 RECHECK의 C17→C03 정의가 실행 시 C03→C17로 처리되며 불필요한 repair를 만들지 않음을 검증.
- 재시작 포함 C17 debt 최대2개 제한, STOP_AFTER_SWEEP의 다른 sweep debt 연기 동작을 검증.
- Sheet 테스트는 mock worksheet 사용. malformed outbox 한 건이 정상 후속 업로드를 막지 않는 것도 검증.
- shell 문법, CLI help, `git diff --check` 통과.

### 실제 원/새 소스 수치 비교

원 실행: `PAN-Crafter-runtime-ablr2-s1-6bb78e6d6118`.
동일한 별도 Python 프로세스에서 각 release 자신의 모듈을 import하여 비교했다.
증거: [runtime_parity_s1.json](ABLR2X_Implementation_Validation_2026-09-23/runtime_parity_s1.json).

비교 대상은 전체 폭 Teacher W112/D123, Student W104/D122이다.
합성 native64 H5/LP를 사용해 Teacher forward/consistency/backward/AdamW,
C03/C07 loss routing·gradient·2-step AdamW, native view·sampler·LR schedule,
validation 및 RR20 metric을 실제 계산했다. 원/새 출력 digest가 정확히 같고,
공유 핵심 model/loss/warp/RNG/평가 코드 **23개 SHA**도 동일했다.

- 최종 source manifest: **116 files**.
- 원/새 수치 digest: `19b0c3a562f5f215d97197f9e98953200c1edda1a9bab7b2ff5d819848e8a922`.
- 개발 검증 receipt는 캠페인에 설치하지 않았다. commit 이후 실행 시 source identity에 맞춰 자동으로 다시 측정한다.
- 이는 **CPU 합성 수치 동등성**이다. GPU 비트 동일성이나 실제 C03/C17 pairing을 대신 증명하지 않는다.
  실제 pairing은 각 run의 초기값·stream·Teacher endpoint 증거로 별도 검증한다.

## 실행과 남는 운영 조건

상세 명령: [ABLR2X_README.md](../ablr2/ABLR2X_README.md).

```bash
# commit → push → 각 서버 pull 이후, 해당 서버 번호만 지정
bash tools/ablr2_docker_start.sh --server s1 --until-operator-stop
```

기존 s1/s2는 현재 run 완료 뒤 전환한다. 원 lease가 먼저 만료되거나 기존에 미완료 fullstate가 남으면
그 원 release에서 명시적으로 복구해야 한다. 새 무기한 승인으로 원 작업의 시간을 자동 연장하지 않는다.

구형 GFP40/GFB20 runner는 exact run-boundary stop을 지원하지 않는다.
s3에서 해당 선행 실행이 있으면 기본 RUN 전환을 거부한다. 운영자가 계획의 긴급 SAFE 전환을 선택하는 경우만:

```bash
bash tools/ablr2_docker_start.sh --server s3 --until-operator-stop --boundary SAFE
```

SAFE는 원 controller의 안전 저장 후 종료·fullstate 보존·GPU idle을 검증한다. 강제 kill이나 cron 변경은 없다.
remote 데이터·GPU preflight 및 실제 Sheet 쓰기는 각 서버 시작 때 검증하며, 이번 구현 검증에서 수행하지 않았다.

## 기존 실행에 대한 영향

작업 중 읽기 전용 확인 시 s1은 원 캠페인31개 완료, P02/C10 학습 중이었다.
실제 학습 시작·중지, 원 campaign state/lease/control 변경, live Sheet 수정, commit/push는 수행하지 않았다.
기존에 사용자가 변경한 `g20/` 파일, watchdog, 그림 코드 및 `research_log/past` 문서 이동은 작업 범위에서 제외했다.

실제 사용 seed 이력은 읽기 전용으로61건/22종 확인했으며, GF2 예정10개 seed와의 충돌은 없었다.
실행 시 각 서버의 실제 이력을 다시 조사해 mapping receipt를 고정한다.
