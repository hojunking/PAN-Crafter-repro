#!/usr/bin/env bash
# 25K 실험 — 현재 최고 세팅(paper_ln: LayerNorm, mlp 4.0, depth (2,2,4)) 기준.
#   setsid nohup ./tools/_run_p25.sh > work_dir/p25_chain.log 2>&1 < /dev/null &
#
# 1단계  기준선 시드 3벌 (7.8h)  — 25K 대조군 + 시드 표준편차 sigma 확정.
#        sigma 를 모르면 아래 탐색 결과를 해석할 수 없다.
# 2단계  구조 탐색 5종 x 시드 3벌 (39h) — 정보가치 순. 구성별로 3벌을 몰아서 끝내므로
#        중간에 멈춰도 완결된 구성은 판정할 수 있다.
#
#   d124  full-res depth 2->1   6.578M  FLOPs 가성비 최고 후보
#   w96   width 128->96         4.060M  유일하게 params·FLOPs 둘 다 -43%
#   a2    AttnBlock 3->2        6.234M
#   d322  depth (3,2,2)         7.171M  params 동일 — 순수 배분 효과
#   d222  bottleneck 4->2       6.578M  싼 자리를 줄여도 무의미한지 확인
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ORDER=(p25_s2025 p25_s1234 p25_s7777
       p25_d124 p25_d124_s1234 p25_d124_s7777
       p25_w96  p25_w96_s1234  p25_w96_s7777
       p25_a2   p25_a2_s1234   p25_a2_s7777
       p25_d322 p25_d322_s1234 p25_d322_s7777
       p25_d222 p25_d222_s1234 p25_d222_s7777)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter

# pgrep -f 는 이 문자열을 담은 다른 셸 명령까지 잡는다. 실제 학습 프로세스만 세도록
# 실행 파일이 python 인 것으로 좁힌다.
running(){ pgrep -x -f "python -u main.py --config .*" > /dev/null 2>&1 \
           || ps -eo comm,args | awk '$1 ~ /^python/ && /main\.py --config/ {f=1} END{exit !f}'; }
while running; do
    echo "[p25] 다른 학습 진행 중 — 대기 $(date -Iseconds)"; sleep 300
done
echo "[p25] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 47h"

sync(){  # 결과를 구글시트로 올린다 (실패해도 체인은 계속)
    python gspread/gspread_upload.py "p25_*" > work_dir/p25_gspread.log 2>&1 \
      || echo "[p25] gspread 업로드 실패 (work_dir/p25_gspread.log 참고)"
}
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    [ -f "work_dir/${TAG}/results/reduced_best_val.mat" ] && { echo "[p25] ($i/${#ORDER[@]}) $TAG 완료됨 — 건너뜀"; continue; }
    echo "[p25] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "$TAG"; then
        echo "[p25] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
        python tools/eval_dlpan.py "work_dir/$TAG/results/reduced_best_val.mat" --preset wv3 2>/dev/null \
          | grep "$TAG" || true
    else
        echo "[p25] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"
    fi
    sync
done
sync
echo "[p25] DONE $(date -Iseconds)"
