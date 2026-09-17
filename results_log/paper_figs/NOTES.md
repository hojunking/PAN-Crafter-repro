# 논문 그림 초안 — 자료 출처와 주의 (2026-09-15)

생성기 `tools/paper_figs.py` (`python tools/paper_figs.py`). PNG 미리보기만 낸다.
라벨은 전부 ASCII — matplotlib 에 한글 글리프가 없다.

---

## fig4_plane_corners.png · fig4_plane_corners_grad.png

`results_log/assets/0915_eqrec4_S5_plane.png` 의 **S5a 와 같은 5분위 격자**에서 **네 모서리 칸**만
뽑아 4장씩(2×2 묶음) 가로로 나열한 것. 두 판은 내용이 같고, `_grad` 판만 각 patch 오른쪽 아래에
**PAN Scharr gradient** 를 겹쳐 놓았다. 자료는 `work_dir/_eqrec4_s1_campaign/`
(`L1E4_S2025_best_raw`, native64, n=4,096).

### 네 범주의 이름

`q_A` 는 offset consistency **잔차**라 **낮을수록 일관적(consistent)** 이고, `e` 는 native 복원
오차라 **높을수록 어렵다(hard)**. 그래서 이름을 `consistent/inconsistent × easy/hard` 로 붙였다.

라벨은 **두 줄**이다 — 위에 작게 축 표기(`q low / e low`), 아래에 굵게 이름. 수치를 뺐어도
정의가 그림 안에 남게 하려는 것이다 (2026-09-15 사용자 선택).

| 위치 | 축 표기 (윗줄) | 이름 (아랫줄) | 옛 코드 | 중앙 e | 중앙 q_A | 칸 평균 texture | n |
|---|---|---|---|---:|---:|---:|---:|
| 1 | q low / e low | Consistent - low recon error | EdCd | 0.0077 | 0.3083 | 0.0032 | 50 |
| 2 | q low / e high | Consistent - high recon error | EuCd | 0.0303 | 0.3080 | 0.0148 | 170 |
| 3 | q high / e low | Inconsistent - low recon error | EdCu | 0.0061 | 0.3565 | 0.0008 | 289 |
| 4 | q high / e high | Inconsistent - high recon error | EuCu | 0.0319 | 0.3563 | 0.0044 | 242 |

e 축은 `easy / hard` 가 아니라 **`low / high recon error`** 로 쓴다 (2026-09-17 사용자 지정) —
e 가 복원 오차 자체라 난이도로 바꿔 부르면 원인(난이도)을 결과(오차)로 대신하게 된다.

**옛 4분면 코드(EuCd …)는 그림에 쓰지 않는다** — 읽는 사람이 해독할 수 없다.
S1·S5a·EQREC4 문서와 맞춰 볼 때만 이 표로 대응한다.

배치는 **q 낮은 쌍 → q 높은 쌍**, 각 쌍 안에서 **error 낮은 쪽 → 높은 쪽** 이다 (사용자 지정).
축별 배율(칸 평균 texture): error 축(q 고정) ×4.6–5.8, q 축(error 고정) ÷3.4–4.3.
e 가 올라가면 texture 도 올라가고 q 가 올라가면 texture 는 내려간다 — 두 축이 정반대로 밀기
때문에 두 라벨이 어긋난다(2026-09-15 EQREC4 보완 §A.5 의 "하나의 대각축").

### 표시 규약

- **수치는 그림에 넣지 않는다** (사용자 지정). 위 표가 대신한다.
- patch 는 **patch 별 percentile stretch** — S5a 와 같은 표시다. 대비가 낮은 patch 도 꽉 차 보이므로
  **PAN 회색 영상만으로 texture 를 비교하면 안 된다.**
- 그래서 `_grad` 판을 같이 둔다. gradient inset 은 `pan_scharr_energy` 와 **같은 연산자**
  (`pa.losses.scharr`) 의 크기 맵이고, **16 장 공통 스케일**(전체 분포 p99.5 로 나눔)이라
  밝기를 그대로 비교할 수 있다. q 낮은 두 묶음이 밝고 q 높은 두 묶음이 어둡다 — 숫자 없이
  같은 사실을 보여 준다. inset 은 patch 변의 46 % 다.
  **patch 별로 정규화하면 안 된다** — 그러면 집단 차이가 그림에서 사라진다.
- `_grad` 판 오른쪽의 **colorbar 는 실측 눈금**이다. 0 … **0.3867** = 보이는 16 장 crop 의
  \|∇P\| p99.5. PAN 은 `[-1, 1]` 로 정규화된 값이고 Scharr 커널은 `pa/losses.py` 의 `/32` 정규화판이라,
  이 단위가 `pan_scharr_energy`(= 평균 \|∇P\|²) 와 같은 단위다.
- 묶음 안 4장은 간격 없이 붙이고 **검은 얇은 선(0.8 pt)** 으로만 나눈다. 묶음 사이만 띄운다.
- 칸마다 **칸 중앙에 가까운 순 4장, source block 이 겹치지 않게** 고른다(S5a 의 대표 선택 규칙 +
  S1 atlas 의 block 분리). 극단 cherry-pick 이 아니다.
- 그림의 patch 는 **가로 4 : 세로 3 중앙 크롭**(64×64 → 행 8:56, `CROP43`)이다. e 를 재는 ROI
  `[16:48]²` 은 크롭 안에 그대로 남는다. ROI 표시선은 그림에서 뺐다.
- **16 장은 각 칸의 표본 4장이다.** 집단 주장은 위 표의 칸 평균·n 으로 한다.

## fig4_patches/

위 16 장의 원본. 그림을 다시 쓰거나 확대할 때 이걸 쓴다.

| | |
|---|---|
| `NN_<옛코드>_<슬롯>_id<sample_id>.png` | 표시용(그림과 같은 patch 별 stretch), 라벨·테두리 없음. **크롭하지 않은 64×64** |
| 같은 이름 `.npy` | 원 DN 배열 64×64 float32, `[-1, 1]` 정규화 (`tools/eqrec4/common.load_patches` 그대로) |

그림에 쓴 4:3 장면은 이 원본의 `[8:56, :]` 이다 — 원본은 자르지 않고 남긴다.
| `index.csv` | order · name · axes · group · sample_id · source_group_id · e · q_A · texture · file |

### 검토했다가 쓰지 않은 이름

- **장면 성격**(`Structured / Smooth`): gradient 그림과 바로 연결되지만 **q 축을 texture 로 규정해 버린다.**
  q 는 정의상 offset consistency 잔차이고, texture 는 그것과 함께 움직이는 별개 양이다.
- **예측자 틀**(`true/false confident`, `Reliable / Deceptive …`): "q 가 난이도를 예측한다" 를 전제하는데
  EQREC4 H1 native 는 그 방향을 **3 seed 전부 기각**했다. 이름이 기각된 가설을 되살린다.
