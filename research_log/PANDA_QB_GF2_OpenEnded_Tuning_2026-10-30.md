# QB·GF2 후속 튜닝 큐 — 시간 제한 없음 (2026-10-30)

## 1. 사용자 요청과 운영 원칙

GF2는 PSNR을 포함한 새 목표, QB는 현재 결과를 기준으로 추가 튜닝한다.
서버별 전체 캠페인의 시간 제한은 없다. 개별 run의 종료 step은 아래 설정대로 유지한다.
**사용자가 중간 분석을 요청할 때 결과를 검토하고 남은 case의 유지·제외·추가·우선순위를 갱신한다.**
성적을 보고 임의로 남은 실험을 자동 삭제하거나, 한 번 목표를 달성했다고 큐 전체를 종료하지 않는다.

기존에 명시적으로 승인된 QB 조기 중단 규칙만 유지한다:
**공식 8K·9K·10K HQNR이 모두 .90 미만이면 checkpoint를 안전하게 저장하고 다음 case로 진행한다.**
해당 run은 `EARLY_STOPPED_HQNR`으로 별도 집계하며 30K/25K 완료로 표기하지 않는다.
GF2에는 이 규칙을 적용하지 않는다.

현재 WV3 실험과 기존 큐는 유지한다. 새 s1/QB와 s3/GF2 큐는 각 서버의 기존 focus/WV3
controller가 `COMPLETED`가 된 뒤 실행한다. s1은 새 대기 launcher를 등록하고,
s3는 사용자의 Git push 후 해당 서버 에이전트가 pull하여 등록한다.

## 2. 목표와 공통 method

| 서버 | 데이터셋 | 같은 checkpoint에서 동시에 만족할 목표 |
|---|---|---|
| s1 | QB | HQNR ≥ .9315, ERGAS ≤ 3.5 |
| s3 | GF2 | HQNR ≥ .9645, ERGAS ≤ .522, PSNR ≥ 45.08 dB |

- 기존 Stage1의 정확히 같은 frozen Teacher/Aligner를 재사용한다. 이번 추가 Stage1 학습은 0개다.
- PAN Aligner scale=1, RGB 정합 기준, Lanczos-4 warp.
- Student 입력 `[P_corrected, P_corrected−P_original, I_lrms]`, 두 데이터셋 모두 6채널→4밴드 HRMS.
- UBIL, depth [1,2,3], interp23 MS base, 기존 GT Hard loss, β=0.
- AdamW weight decay=.01, batch48/micro12, warmup100, cosine minLR=0, 1K 간격 공식 평가.
- rotation loss/TTA/SWA/EMA/밴드 가중 loss 없음. loss 종류나 optimizer 종류는 추가하지 않는다.
- R1/R2는 기존 seed `261018101` / `261018102` 및 대응하는 원본 정수 stream seed를 유지한다.

## 3. s1/QB — 낮은 LR·짧은 cosine과 기존 Hard 가중 분리

기준 donor는 `panda_s1loss9/S1_A200_E030 @44K`이다.
λ_abs=2, λ_ε=.03, ε radius=3 RR px.
checkpoint SHA256: `7fa4491f0c0e7b8c5d559263c50fff18f84abd7a820852a0d24666928191ed34`.

현재 대표 최선은 BASE R2 @26K, **HQNR .945383746 / ERGAS 3.509656381**이다.
다만 이 checkpoint 하나를 모든 실험의 seed 불일치 대조군으로 삼지 않는다.
각 R1은 원본 QB_A200_E030 R1, R2는 QB_BASE R2와 비교한다. 같은 step 및 25–30K 평균도 확인한다.
C25는 대조군과 25K 또는 공통 20–25K 구간을 비교한다.

| 실행 순서 | Case | Seed | Step | LR | α | γ | 가설/역할 |
|---:|---|---|---:|---:|---:|---:|---|
| 1–2 | QB2_C30LR080 | R1, R2 | 30K | 8e−5 | .25 | .5 | 초기/중기 LR을 완만하게 낮추기 |
| 3–4 | QB2_C30LR050 | R1, R2 | 30K | 5e−5 | .25 | .5 | 낮은 LR의 범위를 넓혀 확인 |
| 5–6 | QB2_C25 | R1, R2 | 25K | 1e−4 | .25 | .5 | cosine 감쇠를 앞당기기 |
| 7–8 | QB2_C30AG000 | R1, R2 | 30K | 1e−4 | 0 | 0 | 기존 Hard 추가 가중 전체의 필요성 확인 |
| 9–10 | QB2_C30G025 | R1, R2 | 30K | 1e−4 | .25 | .25 | α 고정, conditional-q 가중만 절반 |

모두 W112, β=0. **5개 설정×2 seeds=10 run, 총 290K updates**이다.

### 선택 근거

동일 donor·R1 초기화·표본 순서로 비교한 기존 LR1.2e−4는 30K ERGAS를 기본보다
**+.029236** 악화시켰고 RR 20장 중 17장에서 악화됐다.
40K horizon의 같은 30K 평가 역시 **+.033822**, 20장 중 17장에서 악화됐다.
따라서 높은 LR과 긴 horizon의 추가 확대 대신 반대 방향을 먼저 검증한다.
낮은 LR과 짧은 horizon의 개선은 아직 확인되지 않은 가설이다.

| 설정 | 두 seed 대표 ERGAS 평균 | 동일 30K ERGAS 평균 | 25–30K ERGAS 평균 |
|---|---:|---:|---:|
| 기본 | 3.512271 | **3.512657** | **3.513936** |
| W128 | 3.511640 | 3.514135 | 3.514385 |
| H050 | 3.511838 | 3.513278 | 3.514394 |
| A0125 | 3.515179 | 3.516854 | 3.517243 |

대표 step 선택에서 보인 W128/H050의 매우 작은 차이가 고정 step에서는 반복되지 않았다.
이번에는 폭 확대, A0125 반복, H050 반복, LR 증가, 40K 이상 연장, 새로운 Stage1 donor 탐색을 추가하지 않는다.
Weight decay 변경 역시 과적합·train/validation 차이에 대한 근거가 없어 이번 큐에서 제외한다.

### AG000은 Aligner 제거 실험이 아니다

현재 가중은 `w_h = 1 + α d_e + γ d_e r_q`이다.
AG000은 α=γ=0으로 `w_h=1`이 되는 기존 masked GT L1 대조군이다.
**PAN Aligner scale1과 corrected PAN·차영상 입력은 그대로 남는다.**
Teacher의 추가 손실 가중만 없어지며, 새로운 loss를 도입하지 않는다.
기존 `panda_e` 경로를 사용해 frozen donor/calibration 등 나머지 조건을 보존한다.
G025는 앞서 α·γ를 함께 낮춘 H050과 달리 γ만 바꾸므로 conditional-q의 영향을 분리한다.

## 4. s3/GF2 — 기존에 준비한 8개 유지

현재 최선 `GF2_C60LR150 R1 @47K`는 **.969704044 / .518119552 / 45.035145591**이다.
동일 seed의 LR100보다 LR150이 PSNR을 R1 +.097368, R2 +.021525 dB 개선했다.
새 목표까지는 .044854409 dB가 남았다. R2의 H/E 동시 통과는 1/60개이므로 HQNR 안정성도 함께 본다.

| 실행 순서 | Case | Seed | Step | LR | α | γ | 폭 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1–2 | GF2P_C60LR200 | R1, R2 | 60K | 2e−4 | .25 | .5 | 112 |
| 3–4 | GF2P_C50LR150 | R1, R2 | 50K | 1.5e−4 | .25 | .5 | 112 |
| 5–6 | GF2P_C60LR150A0125 | R1, R2 | 60K | 1.5e−4 | .125 | .5 | 112 |
| 7–8 | GF2P_C60LR150W128 | R1, R2 | 60K | 1.5e−4 | .25 | .5 | 128 |

**4개 설정×2 seeds=8 run, 총 460K updates**. 기준은 각 seed의 기존 C60LR150이다.
donor `RGB_MATURE_U23 @42K`, λ_abs=2, λ_ε=.003, ε radius3을 고정한다.
실행 세부사항은 [GF2 인계 문서](PANDA_GF2_PSNR4508_Handoff_2026-10-30.md)를 따른다.

## 5. 실행과 Sheet

사용자가 Git push한 후 s3 에이전트가 pull한다. 아래 명령의 Python은 기존 host publisher에서 사용하는
torch 포함 환경으로 실행한다. s1에 이미 같은 대기 프로세스가 있다면 중복으로 시작하지 않는다.

```bash
# s1/QB
python -m unittest discover -s panda_qb_refit/tests -v
python -m panda_qb_refit catalog
mkdir -p work_dir/panda_qb_refit/launcher_s1
nohup python -u -m panda_qb_refit.enqueue --wait > work_dir/panda_qb_refit/launcher_s1/controller.log 2>&1 < /dev/null &

# s3/GF2
python -m unittest discover -s panda_gf2_psnr/tests -v
python -m panda_gf2_psnr catalog
mkdir -p work_dir/panda_gf2_psnr/launcher_s3
nohup python -u -m panda_gf2_psnr.enqueue --wait > work_dir/panda_gf2_psnr/launcher_s3/controller.log 2>&1 < /dev/null &
```

각 launcher는 기존 큐 완료, 원본 source/donor, 유휴 GPU를 확인한 뒤 cost 측정과 학습을 시작한다.
데이터/성능의 합성 검증과 실제 s3 GPU preflight는 구분한다.
실행 중인 기존 학습 프로세스를 종료하지 않는다.

- QB는 `QB-ablations`, GF2는 `GF2-ablations`에 게시한다.
- 두 데이터셋 모두 기존 H/E 통과 step 전체와 대표를 유지한다.
- 대표는 HQNR 목표 충족 중 ERGAS 최소, 없으면 HQNR 최고 1개다.
- GF2는 PSNR까지 포함한 3목표 통과 여부를 추가 표시한다.
- case, seed, LR, α·γ, 폭, 종료 step과 접근 방식이 표시된다.
- 조기 종료 및 실제 완료 step을 보존한다. 종료된 run을 full-horizon 완료로 바꾸지 않는다.

## 6. 중간 분석을 요청받았을 때의 갱신 절차

1. 원본 평가와 Sheet를 대조하고 같은 checkpoint의 지표를 확인한다.
2. 같은 seed·같은 step, 후반 구간 평균, 목표 통과 구간을 함께 비교한다.
   seed에 민감한 ERGAS의 미세 차이만으로 우열을 확정하지 않는다.
3. 요청에 따라 아직 시작하지 않은 case의 유지·제외·추가와 순서를 결정하고 변경 사유를 기록한다.
4. 이미 시작한 run의 config/seed/step/metric을 소급 변경하지 않는다.
   실행 중인 campaign은 source/config pin이 있으므로 registry 파일을 그대로 편집하지 않는다.
5. 대기 중이면 기존 대기 launcher를 안전하게 종료·갱신·재등록한다.
   학습 중이면 안전한 run 경계 또는 완전한 checkpoint 지점에서 명시적으로 인계하고,
   새 버전 campaign에 남은 변경 case를 등록한다. 기존 완료/중단 근거와 Sheet 행은 보존한다.

시간 제한을 없애는 것은 위 검토 없이 영구히 실험을 생성한다는 뜻이 아니다.
현재 등록한 **QB 10개 + GF2 8개**를 실행하고, 다음 변경은 사용자의 요청 시 분석을 거쳐 결정한다.

근거: [최신 분석](../results_log/2026-10-30_sensor-results-and-gf2-psnr.md),
[QB 고정 step·scene별 비교](../results_log/assets/2026-10-30_qb-next-evidence.json).
전체 신규 큐: [18개 run CSV](../results_log/assets/2026-10-30_qb-gf2-followup-queue.csv).
