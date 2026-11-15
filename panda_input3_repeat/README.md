# INPUT3-C30 R2: 세 서버에 한 사이클 추가

2026-11-16 요청. 현재 A/B/C 입력 비교의 두 번째 Student seed를 실행한다.
기존 완료/실행 중 결과를 보존하고, 기존 C30 큐가 끝난 뒤 A→B→C를 각 30K 새로 학습한다.
Stage 1은 재학습하지 않으며 각 서버의 기존 고정 Aligner/Teacher를 그대로 사용한다.

| 서버 | 데이터 | donor | Student | peak LR | R2 |
|---|---|---|---|---:|---|
| s1 | WV3 | 기존 R1@27K | W112/D123/UBIL | 1.1e-4 | A→B→C 각 30K |
| s2 | QB | 기존 R1@32K | W128/D123/UBIL | 1.0e-4 | A→B→C 각 30K |
| s3 | GF2 | 기존 R2@42K | W128/D123/UBIL | 2.5e-4 | A→B→C 각 30K |

- A: 원본 PAN. B: 원본 PAN + corrected PAN. C: corrected PAN + (corrected−원본 PAN).
  세 경우 모두 interp23 MS를 함께 입력한다.
- 같은 데이터셋의 A/B/C는 Student seed **261018102**와 데이터/augmentation RNG를 공유한다.
  R1의 261018101과 다르며, 기존 weight를 추가 학습하는 방식이 아니다.
- RGBN, Lanczos-4, 공통 유효영역, scale1, α=β=0, γ=.25,
  `w=1+.25*d_e*r_q`, q=0 예외처리, frozen Teacher/q 및 기존 calibration 유지.
- batch48/micro12, AdamW WD.01, warmup100 + cosine30K→LR0 유지.
- 매1K RR/FR 평가와 weight 보관. 초기 HQNR에 따른 중단 없음.

## 실행

사용자 push 후 각 서버에서 pull하고 **자기 서버 명령 하나**를 실행한다.

```bash
git pull --ff-only
python3 -m panda_input3_repeat.enqueue --server s2
# s3: python3 -m panda_input3_repeat.enqueue --server s3
# s1: python3 -m panda_input3_repeat.enqueue --server s1
```

`--check-only`는 사전 점검, 중단된 R2 컨테이너 재개는 `--resume`.
기존 original-PAN C30 runner가 실행 중이면 건드리지 않고 종료를 기다린다.
기존 큐가 미완료인데 중단돼 있으면 기존 인증된 launcher로 먼저 재개한다.
현재 C30 cutover manifest와 기존 main donor가 있는 서버를 대상으로 한다.
새 컨테이너 이름은 `panda-input3-c30-r2-<server>-v1`.

## Sheet

각 데이터셋의 `*-ablations`에 `INPUT3-C30-… R2 (RGBN)`으로 별도 등록한다.
기존 PAN 기준 HQNR 하한을 충족한 checkpoint 중 ERGAS 최소를 대표로 게시한다.
대표의 ERGAS/PSNR 미달은 표시하며, 대표 외 상세 행은 모든 목표 동시 통과 시에만 게시한다.
HQNR 하한 미달이면 수치 없는 상태 행을 유지한다. 소수점4자리, 판정은 반올림 전 값이다.
기존 PAN FR D:F와 corrected PAN 보조 FR H:J를 같은 checkpoint 행에 함께 게시한다.

| 데이터 | 기존 PAN HQNR ≥ | ERGAS ≤ | PSNR ≥ |
|---|---:|---:|---:|
| WV3 | .9585 | 2.040 | — |
| QB | .9315 | 3.500 | — |
| GF2 | .9645 | .522 | 45.08 |

추가 캠페인: `PANDA_S123_INPUT3_C30_R2_20261116_v1`.
상태/결과: `work_dir/panda_input3_repeat/<campaign>/<server>/<dataset>/`.
기존 학습·평가·게시 코드를 공유하는 별도 wrapper이므로 원래 release/checkpoint pin은 바꾸지 않는다.
