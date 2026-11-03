# s3 인계: WV3 Stage 1 RGB ablation

사용자 지시: 실행 중인 Stage 1 ablation의 L_spa를 RGB 평균으로 변경하여 새로 실행하고,
Experiment에 `(RGB)`를 넣는다. 기존 각 case 3회 및 round-major 순서를 유지한다.

## 실행

사용자가 이 커밋을 push하면 s3 에이전트는 전체 커밋을 pull하고 아래를 실행한다.

```bash
python3 -m panda_wv3_ablation_rgb.enqueue --server s3 --image hojunqueen/pancrafter-env:latest --replace-legacy
docker logs -f panda-wv3-components-rgb-s3-v1
```

사전 점검 성공 후 기존 `panda-wv3-components-s3-v1`만 정상 종료하고 새 RGB 큐를 실행한다.
기존 결과와 weights를 삭제하지 않는다. 이전 mean8 weights를 RGB run으로 resume하지 않는다.
기존 컨테이너가 아닌 native 실행이면 해당 프로세스를 s3에서 식별해 정상 종료한 후 launcher를 실행한다.
다른 실험 프로세스는 이 전환기에서 정지하지 않는다.

## 큐

각 반복에서 A0 RAW → A1 SCHARR → A2 REC → A3 REC_SPA → A4 REC_EPS → A5 ALL 순서다.
R1 전체 후 R2 전체, R3 전체로 총 18 runs. 모든 run은 fresh 50K이다.
새 A5 ALL은 λ_spa=2, λ_ε=.003의 RGB Full 대조군이며, 과거 mean8 Ours를 대신한다.
Scharr baseline 참조와 validation RRE도 RGB 평균으로 통일했다.

- L_spa: GT RGB `[4,2,1]` 평균의 Scharr 응답과 보정 PAN 비교.
- Aligner 입력/Teacher 출력/복원 손실은 8밴드 전체 유지.
- s1의 기존 main 및 Stage 2 ablation 소스/큐는 변경하지 않음.
- `WV3-ablations`에 `WV3-ABL-S1-A3-REC-SPA R1 (RGB)` 형식으로 게시.
- Stage1 validation 대표 유지; weights는 평가 1K마다 저장.
- RGB test RRE50은 RR/FR 각 20장에 대해 추가 기록하며 donor 선택에는 사용하지 않음.

상세 설정·재개·게시 규칙: [패키지 README](../panda_wv3_ablation_rgb/README.md).
실제 서버 시작 여부는 s3 로그와 새 campaign_state.json으로 확인해야 한다.
