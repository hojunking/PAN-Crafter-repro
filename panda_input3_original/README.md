# INPUT3 / C30: 기존 PAN 기준으로 모델 선정·Sheet 게시

2026-11-15 사용자 정정: corrected PAN이 아니라 **기존 원본 PAN으로 계산한 HQNR**을
목표 판정과 대표 모델 선정에 사용한다. 이 정책은 현재 s1 WV3 / s2 QB / s3 GF2의
INPUT3 및 INPUT3-C30 두 캠페인에 공통 적용한다.

## 서버 인계

사용자가 push한 뒤 각 서버에서 pull하고 자기 서버 명령 하나를 실행한다.
단순 pull만으로 실행 중인 Python 업로더가 바뀌지 않으므로 아래 전환이 필요하다.

```bash
python3 -m panda_input3_original.enqueue --server s1
python3 -m panda_input3_original.enqueue --server s2
python3 -m panda_input3_original.enqueue --server s3
```

`--check-only`: 코드/donor/Sheet 이행 사전 점검. 중단된 새 runner 재개는 `--resume`.
현재 INPUT3-C30 전환 manifest가 있는 서버용이다. 이전 launcher로 되돌아가지 않는다.
새 컨테이너 이름은 `panda-input3-original-pan-<server>-v1`이다.

기존 runner를 SIGTERM으로 정상 저장·종료하고, 두 캠페인의 기존 결과를 새 기준으로
다시 선정해 게시한 뒤 동일 optimizer·RNG·stream에서 학습을 재개한다.
진행 중인 기존 40/50K → C30 큐 순서, donor, 입력, loss, LR, seed는 변경하지 않는다.
기존 training source SHA / checkpoint / config / 평가 JSON을 수정하지 않는다.

## 기준

| 데이터 | **기존 PAN HQNR** ≥ | RR ERGAS ≤ | RR PSNR ≥ |
|---|---:|---:|---:|
| WV3 | .9585 | 2.040 | 없음 |
| QB | .9315 | 3.500 | 없음 |
| GF2 | .9645 | .522 | 45.08 |

- 대표값: 기존 PAN HQNR 하한 통과 후보 중 ERGAS 최소.
  동률이면 PSNR 높은 순 → 기존 PAN HQNR 높은 순 → 이른 step.
- 대표값의 ERGAS/PSNR이 미달이어도 게시하되 미달을 표시한다.
- 대표 외 checkpoint 행: 같은 checkpoint에서 모든 목표 동시 통과 시에만 게시한다.
- HQNR 하한을 통과한 checkpoint가 없으면 status 행만 게시한다. 최고 HQNR fallback은 없다.
- 4자리 표시, 판정은 저장된 반올림 전 값. FR 원본 지표 D:F, corrected 보조 지표 H:J.
- 두 지표를 바꿔치기하지 않는다. 기존 PAN의 `record.fr`와 corrected의
  `record.fr_corrected_pan`은 동일한 출력/weight에서 측정된 원래 값을 그대로 사용한다.
- Experiment/선정 설명은 `[기존 PAN 선정]`으로 갱신한다. 기존 config의 과거 selection
  표기는 보존하되 publication override가 `ORIGINAL_PAN`임을 note에 명시한다.
- 더 이상 게시 조건을 만족하지 않는 기존 상세 행은 증거·백업·외부 셀을 검사한 뒤
  행 자체를 삭제하여 중간 공백을 남기지 않는다. 다른 캠페인 및 로컬 weight는 보존한다.

## WV3 확인 결과

C30-C = corrected PAN + (corrected PAN − 원본 PAN), 새 cosine30K.
기존 PAN 기준으로 27K/29K/30K가 HQNR≥.9585, ERGAS≤2.040을 통과했다.
대표 29K: HQNR .9592798695, D_s .0295149437, D_lambda .0115544839,
ERGAS 2.0383333868, SAM 2.7638043058, PSNR 38.0220714573.
같은 모델의 corrected HQNR .9519161073은 보조값이며 선정에 사용하지 않는다.

## s1 적용 및 검증

- 기존/C30 두 namespace의 Sheet 이행 readback 통과. 기존 PAN D:F와 corrected H:J가
  같은 checkpoint에 연결되며 소수점 4자리 표시는 유지됐다.
- 적용 시 `WV3-ablations` 920행은 C30-C 29K 대표값, 923–925행은 27K/29K/30K 통과 결과다.
  원래 A·B 행도 기존 PAN 기준으로 다시 선정했다. 다른 캠페인의 행은 변경하지 않았다.
- 새 uploader 전환 시 진행 중 C30-B를 13,250 step에서 완전 저장했다.
- 원본/보정 기준 분리, 세 데이터셋 하한, 반올림 경계, 대표 예외, 기존 V3 행 검증,
  외부 행 보존, 동일 checkpoint의 두 FR 컬럼에 대한 테스트 5개 통과.
