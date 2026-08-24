# PAN-Crafter 배포 코드 — 논문 대비 차이 · 잠재 버그 · 개선안

대상: 저자 배포본(commit `e12fb5e`) vs 논문 arXiv 2505.23367v2 (ICCV 2025)
모든 항목은 `pancrafter` 환경에서 실행으로 확인했다.

## 요약 — 적용 현황

| ID | 구분 | 위치 | 상태 |
|---|---|---|---|
| [A-1](#a-1) | 논문 불일치 | `model/pancrafter.py` CM3A | **토글화** — `model_args.fix_key_alias` |
| [A-2](#a-2) | 논문 불일치 | `model/pancrafter.py` `reset_parameters` | **토글화** — `model_args.fix_local_attn` |
| [B-1](#b-1) | 잠재 버그 | `feeders/feeder.py:23` | 미적용 (문서화만) |
| [B-2](#b-2) | 잠재 버그 | `feeders/feeder.py:24` | 미적용 (문서화만) |
| [B-3](#b-3) | 잠재 버그 | `feeders/feeder.py:53` | 미적용 (문서화만) |
| [B-4](#b-4) | 잠재 버그 | `feeders/feeder.py:113` | 미적용 (문서화만) |
| [B-5](#b-5) | 잠재 버그 | `utils.py` `tensor2img` | 미적용 (문서화만) |
| [C-1](#c-1) | 운용 | `train.py` prepare | **적용** — optimizer·scheduler 등록 + `--resume` |
| [C-2](#c-2) | 운용 | `main.py` / `config/*.yaml` | **적용** — 가드 추가, `save_epoch` 25 |
| [C-3](#c-3) | 운용 | `train.py` 마지막 로그 | **적용** — 빈 report 재출력 방지 |
| [C-4](#c-4) | 운용 | `main.py` `--res` | **적용** — `str2bool` 연결 |
| [D-1](#d-1) | 지표 | `main.py` / `train.py` | **적용** — `metrics.csv` + `.mat` 태그화 + `export_mat.py` |
| [D-2](#d-2) | 지표 | `utils.py` `ERGAS_numpy` | **적용** — 참조 평균으로 정규화 |
| [D-3](#d-3) | 지표 | `utils.py` 로그 라벨 | **적용** — `Q4(first4)` |
| [D-4](#d-4) | 지표 | `utils.py` 로그 라벨 | **적용** — `ERGAS(vs PAN)`, `SCC(vs PAN)` |
| [D-5](#d-5) | 지표 | `utils.py` QNR | **적용** — 주석으로 HQNR 과 구분 |
| [D-6](#d-6) | 환경 | `utils.py` import | **적용** — `scipy.ndimage`, 미사용 import 제거 |
| [E-1](#e-1) | 평가 방법론 | `main.py` / `train.py` | **적용** — `select_on: test\|val` 스위치, `validate()` 추가 |
| [F-1](#f-1) | **배포 데이터 결함** | `pan_h5.zip` | **적용** — `tools/repair_lpan.py` 로 재생성 |

### A-1 / A-2 를 토글로 둔 이유

이 둘은 모델 동작 자체를 바꾼다. 고쳐 버리면 논문 Table 과 비교할 기준선이 사라지고,
그대로 두면 논문의 설계를 검증할 수 없다. 그래서 **config 로 켜고 끄게 만들어 두 벌을 돌린다.**

```yaml
model_args:
  fix_key_alias: False    # A-1
  fix_local_attn: False   # A-2
```

| config | `fix_key_alias` | `fix_local_attn` | 답하려는 질문 |
|---|---|---|---|
| `pancrafter_<ds>.yaml` | False | False | 논문 수치가 배포 코드로 재현되는가 |
| `pancrafter_<ds>_fixed.yaml` | True | True | 논문대로 고치면 더 좋아지는가 |

둘 다 False 면 배포본과 **완전히 동일한 출력**이다(회귀 확인 완료).
파라미터 총량은 9,968,808 개로 양쪽이 같아, 성능 차이를 모델 크기로 설명할 수 없다.

### D-2 적용 시 유의

ERGAS 정규화 기준을 바꿨으므로 **수정 전 로그의 ERGAS 값과 직접 비교할 수 없다.**
같은 스모크 조건에서 23.00 → 7.21 로 바뀌었다(표준 정의에 가까워진 것이다).

---

## A. 논문과 실제로 다르게 동작하는 부분

### A-1. CM3A 의 key 변수가 덮어써진다 {#a-1}

`model/pancrafter.py:72`

```python
q, k_pan, k_pan = self.q_norm(q.permute(0,1,3,4,2)), \
                  self.k_norm_pan(k_pan.permute(0,1,3,4,2)), \
                  self.k_norm_ms(k_ms.permute(0,1,3,4,2))
#     ^^^^^  ^^^^^   좌변에 k_pan 이 두 번, k_ms 가 없다
```

튜플 좌변이 `(q, k_pan, k_pan)` 이라 세 번째 값이 두 번째를 덮는다. 두 가지가 동시에 깨진다.

1. `self.k_pan` conv 출력이 통째로 버려지고 **PAN 브랜치가 MS key 로 attention 을 계산**한다.
2. `k_ms` 는 재할당된 적이 없어 다음 줄 `permute(0,1,4,2,3)` 의 왕복이 성립하지 않는다.
   `(B,N,C,H,W) → (B,N,W,C,H)` 가 되고, 이어지는 `reshape(-1, head_dim, H, W)` 는
   원소 수가 같아 **예외 없이 통과**하면서 내용만 뒤섞인다.

논문 Eq (10)/(11) 의 `x_pan = LocalAttn(Q, K_pan, V_pan)`, `x_ms = LocalAttn(Q, K_ms, V_ms)` 와 어긋난다.

**확인 결과**

```
k_pan(실제) == k_ms conv 출력 ?  True
k_pan(실제) == k_pan conv 출력 ? False
k_ms(실제) == k_ms conv 출력 ?   False
grad=None 파라미터 10개 / 611,584 개  (전체 9,968,808 의 6.13%)
  cond4.attn.k_pan / cond3_e.attn.k_pan / cond2_e.attn.k_pan
  cond3_d.attn.k_pan / cond2_d.attn.k_pan  (weight, bias)
```

`value`(`v_pan`, `v_ms`)는 정상이라 cross-modality 정보 자체는 흐른다.
망가진 것은 aggregation weight 를 만드는 key 쪽뿐이라 학습이 그대로 진행된다.

**변형안** — 좌변만 고치면 된다.

```python
q, k_pan, k_ms = self.q_norm(q.permute(0,1,3,4,2)), \
                 self.k_norm_pan(k_pan.permute(0,1,3,4,2)), \
                 self.k_norm_ms(k_ms.permute(0,1,3,4,2))
```

`qk_norm=False` 라 세 norm 은 모두 `Identity` 다. 따라서 실질 효과는
(a) 죽어 있던 `k_pan` 6.13% 가 살아나고 (b) `k_ms` 가 뒤섞이지 않는 것, 두 가지다.
파라미터 수는 그대로이므로 논문 Table 1 의 파라미터·메모리 수치와 충돌하지 않는다.

---

### A-2. `reset_parameters()` 가 한 번도 호출되지 않는다 {#a-2}

`model/pancrafter.py:54-60`

`dep_conv` 를 one-hot shift kernel 로 채우고 `requires_grad=False` 로 고정하는 함수인데
저장소 전체에 호출부가 없다(정의 1건뿐). 이 초기화가 있어야 `dep_conv` 가 k×k 이웃을
그대로 모아오는 연산이 되어, 논문 Sec 3.3 의 "LocalAttn computes attention scores within
the k × k local receptive field"(인용 [35], ACmix 방식)가 성립한다.

**확인 결과**

```
requires_grad: True          (shift kernel 이면 False 여야 함)
weight 가 0/1 shift kernel:  False
학습되는 dep_conv 파라미터:   7,200
```

3×3 수용영역은 유지되지만, softmax 가 고르는 9개 후보가 "3×3 이웃 9픽셀" 이 아니라
"3×3 패치의 학습된 선형결합 9개" 다.

**변형안** — `PANCrafter.initialize_weights()` 끝에 추가한다.

```python
from model.pancrafter import CMAAA   # 같은 파일이면 불필요
for m in self.modules():
    if isinstance(m, CMAAA):
        m.reset_parameters()
```

주의할 점이 둘 있다.

- `dep_conv` 는 `bias=True` 로 생성된다(`model/pancrafter.py:37`). `reset_parameters()` 는
  weight 만 고정하므로 bias 가 학습 가능한 채로 남아, 모아온 이웃값에 상수 offset 이 더해진다.
  순수한 이웃 gather 를 원하면 `bias=False` 로 바꾸거나 bias 도 0 으로 고정해야 한다.
- `requires_grad=False` 파라미터가 생기므로 optimizer 생성부(`train.py:60`)를
  `filter(lambda p: p.requires_grad, self.model.parameters())` 로 바꾸는 편이 명시적이다
  (AdamW 는 `grad=None` 이면 건너뛰므로 필수는 아니다).

---

### A-3. 표기만 다르고 동작은 문제없는 것 {#a-3}

수정 대상이 아니다. 논문을 읽고 코드를 볼 때 혼동하기 쉬운 지점이라 기록해 둔다.

- **입력 concat**: 논문 본문은 "`I_pan` 과 `I^lr_ms` 의 채널 concat" 이라고 쓰지만 코드는 4개를
  붙인다(`pan`, `↑4 ms`, `↑4 lpan`, `pan − ↑4 lpan`). Fig. 3 하단 캡션과는 일치하므로 본문이 축약된 것이다.
- **ResBlock 정규화**: 논문과 Fig. 3 은 LayerNorm, 코드는 `GroupNorm(32)`.
- **Eq (8) 의 α**: 코드는 `s_token` → `adaLN_modulation` 으로 mode 별 gate 를 만든다. 논문의 상수 α 를
  mode 조건부로 일반화한 형태로, 논문보다 표현력이 크다.
- **residual 의 `I^lr_ms`**: 데이터셋의 `lms` 가 아니라 `bicubic(ms, ×4)` 다.
  논문이 업샘플 필터를 명시하지 않아 불일치는 아니다. `lms` 는 로드만 되고 시각화·저장에만 쓰인다.

**검증했고 논문과 일치하는 것**: Eq (1)(2)(3)(4), MARs mode switching 과 실효 배치 96,
λ = `w_off` = 1.0, Eq (9)/(12) 의 mode 별 query conditioning, softmax 축(k×k 후보 축),
추론 시 MS mode 고정, 각 스케일의 해상도 정합(train 64² / reduced 256² / full 512² 전부).

---

## B. 지금 config 에선 안 터지지만 데이터셋을 늘리면 터지는 것

CANConv 쪽에서 이미 CAS500(scale 4095), Vantor(scale 16383) 를 다루고 있어 특히 관련 있다.

### B-1. `max_pixel` 을 feeder 인자로 넘기면 AttributeError {#b-1}

`feeders/feeder.py:23-30` — `if max_pixel == 0.:` 분기만 있고 `else` 대입이 없다.
`self.max_pixel` 이 아예 정의되지 않아 `np2tensor` 에서 터진다. 실행으로 확인했다.

**변형안**

```python
if max_pixel == 0.:
    ...  # 기존 추론 로직
else:
    self.max_pixel = max_pixel
```

### B-2. 미지원 센서명이면 `max_pixel=1.0` 으로 조용히 진행 {#b-2}

`feeders/feeder.py:24-30` — 경로 문자열에 `wv3`/`qb`/`wv2`/`gf2` 가 없으면
`print("Unsupported dataset.")` 만 하고 `self.max_pixel = 1.` 로 계속 간다.
`np2tensor` 가 `2x/max_pixel - 1` 이므로 정규화가 완전히 깨진 채 학습이 진행된다.

CAS500(4095), Vantor(16383) 를 그대로 붙이면 여기에 걸린다.

**변형안** — 조용한 진행을 막고, 센서 스케일을 표로 뺀다. B-1 수정과 함께 가야 한다.

```python
SENSOR_SCALE = {"wv3": 2047., "wv2": 2047., "qb": 2047.,
                "gf2": 1023., "cas500": 4095., "vantor": 16383.}

if max_pixel == 0.:
    hit = [v for k, v in SENSOR_SCALE.items() if k in dataroot]
    if not hit:
        raise ValueError(f"센서 스케일을 추론할 수 없다: {dataroot}. "
                         f"config 의 feeder_args 에 max_pixel 을 명시하라.")
    self.max_pixel = hit[0]
else:
    self.max_pixel = max_pixel
```

문자열 추론 자체를 없애고 config 의 각 `*_feeder_args` 에 `max_pixel` 을 명시하는 쪽이 더 안전하다.
그 경우 B-1 수정이 선행돼야 한다.

### B-3. split 토큰이 없으면 `self.split` 미정의 {#b-3}

`feeders/feeder.py:53-62` — 경로에 `train`/`valid`/`reduced`/`full` 이 없으면
`print("Unsupported file format.")` 만 하고 `self.split` 을 만들지 않아 `__getitem__` 에서 AttributeError.

**변형안** — `else: raise ValueError(...)` 로 바꾸거나, `split` 을 feeder 인자로 받아 config 에 명시한다.
후자가 낫다. 경로 문자열 의존이 사라지면 디렉터리 이름을 자유롭게 쓸 수 있다.

```python
def __init__(self, dataroot, split=None, max_pixel=0., ...):
    ...
    self.split = split or self._infer_split(dataroot)
```

### B-4. `augment_without_gt` 에서 `lpan` 만 crop 되지 않는다 {#b-4}

`feeders/feeder.py:113-130` — crop 블록에서 `lms`/`ms`/`pan` 은 슬라이싱되는데 `lpan` 은 빠져 있다.
전체 영역을 그대로 `cv2.resize` 해 넣으므로 다른 modality 와 정합이 깨진다.

`has_gt=False` 이면서 `split=='train'` 일 때만 도달한다. 배포된 config 에서 GT 없는 split 은
`test_full` 뿐이라 현재는 dead code 다. **full-resolution 데이터로 파인튜닝하면 바로 걸린다.**

**변형안** — `augment()` 와 동일하게 한 줄 추가한다.

```python
ms   = ms[ms_y:ms_y + ms_p, ms_x:ms_x + ms_p, :]
lpan = lpan[ms_y:ms_y + ms_p, ms_x:ms_x + ms_p, :]   # <-- 누락된 줄
pan  = pan[pan_y:pan_y + pan_p, pan_x:pan_x + pan_p, :]
```

### B-5. `tensor2img` 가 batch=1 을 가정 {#b-5}

`utils.py:67-73` — `tensor.squeeze(0)` 후 `np.transpose(img_np, (1,2,0))`.
`test_batch_size > 1` 이면 `ValueError: axes don't match array`. 실행으로 확인했다.

**변형안** — 최소한 계약을 명시하고, 원한다면 배치를 지원한다.

```python
def tensor2img(tensor, max_pixel):
    assert tensor.shape[0] == 1, "tensor2img 는 batch=1 만 지원한다 (test_batch_size 를 1 로 두라)"
    ...
```

---

## C. 50k iteration 실행 전에 손봐야 할 것

### C-1. 중간 재개가 불가능하다 {#c-1}

`train.py:71` 에서 **model 만** `accelerator.prepare` 된다.

```python
self.model = self.accelerator.prepare(self.model)
```

optimizer 와 lr_scheduler 가 accelerator 에 등록되지 않아 `save_state()` 가 남기는 것은
model weight 와 RNG state 뿐이다. `main.py` 는 `last_epoch = 0` 하드코딩이고 `load_state` 호출도 없다.
WV3 기준 약 5~6시간 실행 도중 죽으면 처음부터다. **재현 실행 전에 이것만은 적용할 가치가 있다.**

**변형안 (1) — optimizer/scheduler 를 등록해 checkpoint 를 완전하게 만든다**

`train.py:71`

```python
self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
    self.model, self.optimizer, self.lr_scheduler)
```

이것만으로 `save_state()` 가 optimizer·scheduler state 를 함께 저장한다.
모델 수학은 건드리지 않으므로 재현 기준선을 해치지 않는다.

**변형안 (2) — resume 경로를 추가한다**

`main.py`

```python
parser.add_argument('--resume', type=str, default=None,
                    help='이어서 학습할 checkpoint 디렉터리 (예: work_dir/.../checkpoint-20000)')
```

```python
last_epoch, global_step = 0, 0
if args.resume:
    trainer.accelerator.load_state(args.resume)
    global_step = int(os.path.basename(args.resume).split('-')[-1])
    last_epoch = global_step // len(data_loader['train'])
```

한계: dataloader 는 `prepare` 되지 않으므로 배치 순서까지 정확히 복원되지는 않는다.
"정확한 재개" 가 아니라 "근사 재개" 이며, 이 사실을 로그에 남겨야 한다.
엄밀한 재현이 목적이라면 재개하지 말고 처음부터 다시 돌리는 편이 낫다.

**변형안 (3) — 재개를 포기하고 저장 간격만 촘촘히**
`save_iter` 를 10000 → 2500 으로 낮추면 최악의 손실이 약 15분으로 줄고,
사후 평가용 체크포인트도 늘어난다(D-1 과 함께 쓰면 유용). 디스크는 체크포인트당 약 40MB 다.

### C-2. `save_epoch: 500` 이 한 번도 발동하지 않는다 {#c-2}

총 epoch 이 WV3 248 / QB 141 / GF2 122 이므로 `(epoch+1) % 500 == 0` 이 성립할 수 없다.
`trainer.save_checkpoint()` 는 사실상 비활성이고 `save_iter: 10000` 만 동작한다.

추가로 `main.py:41` 의 `--save-epoch` 기본값이 `0` 이라, config 에서 이 키를 빼면
`(epoch+1) % 0` 으로 **ZeroDivisionError** 가 난다. 실행으로 확인했다.

**변형안** — main.py 에 가드를 넣고, config 값을 실제 총 epoch 에 맞춘다.

```python
if args.save_epoch > 0 and (epoch + 1) % args.save_epoch == 0:
    trainer.save_checkpoint(epoch + 1)
```

```yaml
save_epoch: 25    # WV3 248 epoch 기준 약 10회 저장
```

### C-3. 마지막 학습 로그가 `nan` 으로 찍힌다 {#c-3}

`train.py:170-188` — `global_step % log_iter == 0` 에서 `report.__init__()` 로 리셋한 직후
`global_step >= num_iter` 블록이 같은 report 로 다시 출력한다. `num_examples` 가 0 이라 0/0 이 된다.
`num_iter=50000`, `log_iter=100` 이면 마지막 스텝에서 반드시 발생한다. 실측으로 확인했다.

```
Iter[5/5]  Total Loss: nan  Loss MS: nan  Loss PAN: nan
```

**변형안** — 비어 있으면 재출력하지 않는다.

```python
if global_step >= self.args.num_iter:
    if report.num_examples > 0:
        ...  # 기존 로그 출력
    save_path = os.path.join(self.args.work_dir, 'lastest')
    ...
```

### C-4. `--res` 가 `type=bool` 이라 항상 True {#c-4}

`main.py:61` — `parser.add_argument('--res', type=bool, ...)`.
`bool("False") == True` 이므로 `--res False` 로 residual 을 끌 수 없다.
`main.py:91` 에 `str2bool` 함수가 **정의되어 있으나 어디서도 쓰이지 않는다.**

**변형안** — 이미 있는 함수를 연결한다.

```python
parser.add_argument('--res', type=str2bool, default=True, help='residual connection')
```

---

## D. 학습 중 지표는 논문 수치와 다른 구현이다

`test_log.txt` 값으로 논문 Table 1/2 와 직접 비교하면 안 된다.
논문 수치는 DLPan-Toolbox(MATLAB) 기준이다. 파이썬 지표는 **best 체크포인트 선택에만** 쓰인다.

### D-1. best 선택 기준이 파이썬 자체 구현이다 {#d-1}

`main.py:167-180` — reduced 는 파이썬 `ERGAS`, full 은 파이썬 `D_s` 로 best 를 고른다.
아래 D-2~D-5 의 편차가 그대로 선택에 반영되므로, **선택된 best 가 MATLAB 기준 best 와 다를 수 있다.**

게다가 `.mat` 파일명이 고정이라(`train.py:290,333,380,423`) best 가 갱신될 때마다 덮어쓴다.
최종적으로 남는 것은 "파이썬 ERGAS 기준 마지막 best" 하나뿐이고, 다른 후보는 사후 재평가할 수 없다.

**변형안 (1) — 후보를 남기고 사후에 MATLAB 으로 고른다 (권장)**

`.mat` 파일명에 epoch 을 넣어 덮어쓰기를 없앤다.

```python
savemat(f'{path}/reduced_ep{epoch:04d}.mat', d)
```

학습이 끝난 뒤 전부 DLPan-Toolbox 에 넣어 평가하고 best 를 고른다.
파이썬 지표는 "저장 여부를 결정하는 필터" 로만 쓰는 셈이 된다.
20장 × 8밴드 × 256² 기준 파일 하나가 약 100MB 이므로 저장 주기는 조절해야 한다.

**변형안 (2) — 파이썬 지표를 표준 정의로 교체한다**
D-2 를 고치면 파이썬 ERGAS 가 표준에 가까워져 선택 신뢰도가 올라간다.
다만 MATLAB 과 완전히 일치시키기는 어렵고, 재현 기준선과의 비교 가능성이 떨어진다.

### D-2. `ERGAS` 가 참조가 아니라 예측의 밴드 평균으로 나눈다 {#d-2}

`utils.py:171-177`

```python
summed += RMSE(x_true[:,:,i], x_pred[:,:,i])**2 / np.mean(x_pred[:,:,i])**2
#                                                          ^^^^^^ 표준은 참조(x_true)
```

표준 ERGAS 는 참조 영상 i번째 밴드의 평균 μ_i 로 나눈다. 예측 평균을 쓰면
예측이 전체적으로 어두울수록 ERGAS 가 커지는 방향으로 편향된다.

**변형안**

```python
summed += RMSE_numpy(x_true[:,:,i], x_pred[:,:,i])**2 / np.mean(x_true[:,:,i])**2
```

### D-3. 로그의 `Q4` 는 8밴드에서도 앞 4밴드만 쓴다 {#d-3}

`utils.py:140-141` — `if x_true.shape[2] > 4: x_true, x_pred = x_true[:,:,:4], x_pred[:,:,:4]`.
논문 Table 1 의 **Q8 이 아니다.** WV3 로그의 `Q4` 를 Q8 로 인용하면 틀린다.

**변형안** — 가장 싼 해결은 라벨을 바로잡는 것이다(`utils.py:341`).

```python
str += f'SAM: {self.sam:.6f}\tQ4(first4): {self.q4:.6f}\tERGAS: {self.ergas:.6f}'
```

진짜 Q8 이 필요하면 Q2ⁿ 구현이 따로 있어야 한다. best 선택에는 쓰이지 않으므로 급하지 않다.

### D-4. full-res 의 `ERGAS`/`SCC` 는 PAN 을 참조로 삼는 비표준 진단값 {#d-4}

`utils.py:180-188, 110-120, 262-263` — full-resolution 에는 GT 가 없어 PAN 을 참조로 쓴다.
`ERGAS_full_numpy` 의 `cv2.resize` 는 입력과 출력 크기가 같아 no-op 이고,
`x_true.shape[2] == 1` 이라 루프가 1회만 돌아 **예측의 0번 밴드만** PAN 과 비교한다.

**변형안** — 값을 고치기보다 오해를 막는 쪽이 낫다. 라벨에 참조가 PAN 임을 명시하거나
(`ERGAS(vs PAN)`, `SCC(vs PAN)`), full 로그에서 아예 빼고 D_λ·D_s·QNR 만 남긴다.

### D-5. 논문은 HQNR, 코드는 QNR {#d-5}

`utils.py:231-235` — 코드는 `QNR = (1 - D_λ)(1 - D_s)` 를 계산한다.
논문 Table 1/2/3 이 보고하는 **HQNR 은 D_λ 를 다른 방식으로 계산하는 별개 지표**다.
최종 수치는 어차피 MATLAB 으로 내므로, 로그 라벨을 `QNR` 로 유지하되
논문 HQNR 과 같은 것으로 읽지 않도록 주석을 다는 선에서 충분하다.

### D-6. `scipy.ndimage.filters` 는 deprecated {#d-6}

`utils.py:22` — `from scipy.ndimage.filters import sobel, convolve`.
현재 pin(scipy 1.13.1)에서는 `DeprecationWarning` 만 뜨지만 SciPy 2.0 에서 제거된다.
`convolve` 는 import 만 되고 쓰이지 않는다(`pearsonr` 도 마찬가지).

**변형안**

```python
from scipy.ndimage import sobel
```

---

## E. 평가 방법론 — 검증셋을 쓰지 않고 테스트셋으로 best 를 고른다

### E-1. 검증셋이 로드만 되고 쓰이지 않는다 {#e-1}

`main.py` 가 `data_loader['val']` 을 만들고 `train.py` 가 `self.val_data_loader` 로 받지만
**어디서도 참조되지 않는다.** `Trainer.save_val()` 도 정의만 있고 호출부가 없다.
`valid_wv3.h5` 1,080장(620MB)이 매 실행마다 RAM 에 올라갔다 그대로 버려진다.

그 결과 best 체크포인트를 **논문이 결과를 보고하는 그 테스트셋 20장으로** 고른다.
5 epoch 마다 평가해 그때까지 최고면 덮어쓰므로, 실질적으로 **테스트셋 49회 평가 중
최고를 뽑는 것**과 같다. 테스트 수치가 낙관적으로 부풀려진다.

1차 실행에서 편향의 크기는 작았다(best ERGAS 2.1532 vs 마지막 2.1567, 0.16%).
곡선이 매끄러워 어느 지점을 잡아도 비슷하기 때문이다. 그러나 **두 실행을 비교할 때는
문제가 커진다** — 양쪽 모두 같은 20장에서 best-of-49 를 뽑은 값이기 때문이다.

**변형안 (적용됨)** — `select_on` 스위치를 넣어 두 동작을 모두 보존했다.

```yaml
select_on: test   # 배포본 동작 (기본값). 테스트셋으로 고른다
select_on: val    # 검증셋으로 고르고, 테스트 지표는 곡선 기록용으로만 남긴다
```

`select_on: val` 이면 `Trainer.validate()` 가 검증셋 전체로 reduced 지표를 내고
ERGAS 기준 best 를 `best_val/` 에 저장한다. 테스트 지표는 `metrics.csv` 에 계속 기록되지만
**선택에 쓰이지 않는다.** 검증 1회 비용은 약 20초(1080장), 50k 학습 기준 총 16분이다.

full-resolution 용 별도 선택(`best_full`)은 `select_on: val` 에서 만들지 않는다.
D_s 단독 선택이 "PAN 디테일을 주입하지 않을수록 좋다" 는 방향이라 미학습 모델을 고르기
때문이다(1차 실행에서 epoch 5 가 뽑혔다). 하나의 체크포인트를 골라 두 프로토콜로 평가한다.

---

## F. 배포 데이터 결함 — full-resolution lpan 불일치

### F-1. `pan_h5.zip` 의 WV3·QB full-res `lpan` 이 다른 장면이다 {#f-1}

저자 배포 `pan_h5.zip` 의 `test_{wv3,qb}_OrigScale_multiExm1_pan.h5` 는 짝이 되는 PAN 과
**상관계수 0.011 / −0.001** 이다. 순열 문제도 아니고(최적 매칭도 0.37/0.28) 다른 센서·WV2
와 대조해도 맞는 원본이 없다. 잡음은 아니며(인접픽셀 상관 0.88) 값 범위도 센서와 일치한다.
GF2 만 정확히 일치한다(1.000). 근거 그림: `results_log/assets/lpan_mismatch.png`.

**이것은 지표 계산 문제가 아니라 데이터 파일 자체의 문제다.** `lpan` 은 모델의 *입력* 이고
평가식에는 등장하지 않는다. 엉뚱한 장면이 입력으로 들어가 출력이 망가진 것이다.

FR 테스트셋 장면 불일치와 같은 뿌리일 가능성이 높다
(`results_log/2026-08-19_metric-and-dataset-audit.md` ④). lpan 이 어긋난 WV3·QB 가
논문 EXP 기준선과도 어긋나고, lpan 이 맞는 GF2 는 둘 다 정상이다.
저자의 `pan_h5.zip` 이 교체 전 FR 테스트셋으로 만들어졌다면 두 현상이 함께 설명된다.
PanCollection README 가 "Dec. 11, 2022: we updated full-resolution test examples that contain
more different image scenes" 라고 FR 셋 교체를 명시하고 있어, 버전 차이 자체는 문서화된 사실이다.

**중요**: `lpan` 은 PanCollection 이 배포하는 파일이 아니다. PanCollection h5 의 키는
`gt/lms/ms/pan` 뿐이며 `lpan` 은 없다. `*_pan.h5` 는 PAN-Crafter 저자가 자기 저장소에
커밋한 `pan_h5/pan_h5.zip`(2025-07-18 "Code Release")에서만 나온다.
따라서 **PanCollection 을 다시 받아도 이 문제는 해결되지 않는다.**

### 재생성본의 타당성 검증

저자 원본이 **정상인** WV3 reduced 셋에서 원본과 재생성본을 같은 모델에 각각 넣어 비교했다.

| | ERGAS↓ | SSIM↑ | SCC↑ | Q2n↑ |
|---|---:|---:|---:|---:|
| 저자 원본 `lpan` | 2.1633 | 0.9754 | 0.9900 | 0.9165 |
| 재생성 `lpan` | 2.1635 | 0.9754 | 0.9900 | 0.9164 |

출력 영상 자체의 평균 차이는 화소값 0~2047 기준 **0.13 (상대 0.040%)**, `lpan` 자체의
RMSE 는 0.50 / 상관 0.9999997 이다. **재생성본은 원본과 기능적으로 동등하다.**

`lpan` 은 입력 concat(`↑4 lpan`, `pan − ↑4 lpan`)과 모든 AttnBlock 의 value 경로에 들어가므로,
full-res 추론에서 이 값이 엉뚱하면 논문이 핵심이라 말한 고주파 주입 채널이 통째로 오염된다.

| WV3 full-res (20장) | D_λ↓ | D_s↓ | HQNR↑ |
|---|---:|---:|---:|
| 손상된 배포본 `lpan` | 0.5489 | 0.1645 | **0.3768** |
| 재생성한 `lpan` | 0.0849 | 0.0827 | **0.8446** |

**변형안 (적용됨)** — `tools/repair_lpan.py`. 정상인 파일 10개에서 생성 필터를 역추정했다.

```
Gaussian(sigma=1.98, N=41, BORDER_REPLICATE) 후 [2::4, 2::4] 데시메이션
```

12개 배포 파일 중 10개를 RMSE 0.23~0.53(값 범위 0~2047, 상관 0.9999997)으로 재현하고,
어긋나는 2개가 정확히 문제의 파일이다. 원본은 건드리지 않고
`full_examples_h5_repaired/` 에 새로 만든다. 같은 레시피로 WV2 의 `lpan` 도 생성했다
(`tools/setup_wv2.py`) — 다만 WV2 는 대조할 배포본이 없어 **검증 불가**다.

---

## G. VRAM 사용량 — 논문과 모순되지 않으며 버그와도 무관하다

논문 Sec 4.2 는 RTX 3090(24GB) 1장, Table 1 의 `Memory 1.711 GB` 는
"measured on a 256 × 256 × 8 HRMS target at reduced-resolution" 즉 **추론** 수치다.
학습 메모리가 아니다.

**단계별 실측** (`torch.cuda.max_memory_allocated`, fp32)

| 단계 | peak allocated |
|---|---|
| 학습 B=96(48×2), PAN 64² | **16.51 GB** |
| 추론 reduced, B=1, PAN 256² | 0.86 GB |
| 추론 full, B=1, PAN 512² | 3.29 GB |

추론 0.86 GB 는 논문의 1.711 GB 와 같은 자릿수다(논문 쪽은 allocator reserve 기준으로 보인다).
`nvidia-smi` 로 관측한 19.8 GB 는 allocated 16.5 GB + allocator reserve + CUDA context 로,
정상 범위다. **24GB 카드에서 OOM 나지 않는다.** 3090 에서 돌렸다는 논문 서술과 일관된다.

**메모리를 먹는 곳은 CM3A 의 ka² 확장이다.** 학습 배치에 정확히 비례한다.

| batch | peak |
|---|---|
| 96 | 16.51 GB |
| 64 | 11.03 GB |
| 32 | 5.55 GB |
| 16 | 2.80 GB |

`cond2_e` 한 블록이 만드는 텐서(B=96, 32×32, fp32):

```
k_pan / v_pan / k_ms / v_ms   각 (96,8,16,9,32,32) = 0.42 GB  x4 = 1.69 GB
torch.stack 로 만든 k, v      각 (96,2,8,16,9,32,32) = 0.84 GB x2 = 1.69 GB
q*k 곱, attn*v 곱             각 0.84 GB              x2 = 1.69 GB
--------------------------------------------------------------
블록 하나당 약 5.1 GB, cond2_e + cond2_d 로 약 10.1 GB
```

여기에 `cond3_e`/`cond3_d`(16², 1/4 크기)와 `cond4` 가 더해져 16.5 GB 가 된다.
`dep_conv` 가 k·v 를 ka²=9 배로 펼치고, `torch.stack` 이 PAN·MS 를 합치며 한 번 더 복제하는 구조다.

**A-1·A-2 에서 파생된 것이 아니다.** A-2(shift kernel 고정)를 적용해 직접 비교했다.

```
배포본 그대로                        : 16.51 GB
A-2 적용 (dep_conv shift kernel 고정) : 16.52 GB   차이 -0.01 GB
```

A-1 이 버리는 `k_pan` 출력도 즉시 해제되는 일시 할당이라 영향이 없다.

**줄이고 싶다면** (재현이 목적이면 건드리지 말 것)

- `--mixed-precision bf16` — main.py 에 이미 인자가 있고 `train.py:50-54` 가 이를 반영한다.
  중간 텐서가 절반이 되어 8~9 GB 대로 떨어질 것으로 보인다. 다만 논문 설정과 달라진다.
- `torch.stack` 대신 PAN·MS 브랜치를 따로 계산하면 복제분 1.69 GB/블록을 아낄 수 있다.
  attention 수학은 동일하다.
- batch 를 줄이면 선형으로 줄지만 실효 배치 96 이 논문 설정이므로 재현이 깨진다.
