#!/usr/bin/env bash
# PAN-Crafter 환경을 한 번에 준비한다 — clone, docker pull, 데이터 내려받기·재배치·복구까지.
#
#   ./tools/bootstrap.sh                     # 저장소 안에서 실행 (clone 생략)
#   ./tools/bootstrap.sh --repo <git-url>    # 빈 디렉터리에서 실행 (clone 부터)
#
# 옵션
#   --repo URL        clone 할 저장소. 이미 저장소 안이면 무시된다
#   --dir PATH        clone 위치 (기본: ./PAN-Crafter)
#   --sensors LIST    배치할 센서 (기본: wv3). 예) --sensors wv3,qb,gf2
#   --skip-docker     이미지 pull 생략
#   --skip-data       데이터 내려받기 생략
#   --keep-archive    받은 zip 을 지우지 않는다
#   --data-id ID      Google Drive 파일 id (기본값 내장)
#
# 디스크: zip 12 GB + 압축해제 19 GB + 이미지 9.4 GB. 여유 35 GB 이상 권장.
set -euo pipefail

IMAGE="hojunqueen/pancrafter-env:latest"
DATA_ID="1Q5HHIIAi1SD83jlJMACM7oBuxyHeEURt"
REPO_URL=""; TARGET_DIR="PAN-Crafter"; SENSORS="wv3"
SKIP_DOCKER=0; SKIP_DATA=0; KEEP_ARCHIVE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO_URL="$2"; shift 2;;
    --dir) TARGET_DIR="$2"; shift 2;;
    --sensors) SENSORS="$2"; shift 2;;
    --data-id) DATA_ID="$2"; shift 2;;
    --skip-docker) SKIP_DOCKER=1; shift;;
    --skip-data) SKIP_DATA=1; shift;;
    --keep-archive) KEEP_ARCHIVE=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 1;;
  esac
done

say(){ printf '\n\033[1m[bootstrap]\033[0m %s\n' "$*"; }
die(){ printf '\033[31m[bootstrap] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. 사전 점검
say "사전 점검"
command -v git    >/dev/null || die "git 이 없다."
[ "$SKIP_DOCKER" -eq 1 ] || command -v docker >/dev/null || die "docker 가 없다. --skip-docker 로 건너뛸 수 있다."
if command -v nvidia-smi >/dev/null; then
  echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  echo "  GPU: nvidia-smi 없음 — 학습은 불가하고 지표 계산만 된다"
fi
AVAIL=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "  여유 공간: ${AVAIL} GB"
[ "$SKIP_DATA" -eq 1 ] || [ "$AVAIL" -ge 35 ] || echo "  !! 35 GB 이상을 권장한다 (zip 12 + 해제 19 + 이미지 9.4)"

# ---------------------------------------------------------------- 1. 저장소
if [ -f "tools/bootstrap.sh" ] && [ -f "main.py" ]; then
  REPO="$(pwd)"; say "저장소: 현재 위치를 쓴다 ($REPO)"
elif [ -n "$REPO_URL" ]; then
  say "저장소 clone: $REPO_URL -> $TARGET_DIR"
  [ -d "$TARGET_DIR/.git" ] || git clone "$REPO_URL" "$TARGET_DIR"
  REPO="$(cd "$TARGET_DIR" && pwd)"; cd "$REPO"
else
  die "저장소 안이 아니다. --repo <git-url> 을 주거나 저장소 안에서 실행할 것."
fi

# ---------------------------------------------------------------- 2. 이미지
if [ "$SKIP_DOCKER" -eq 0 ]; then
  say "Docker 이미지 pull: $IMAGE"
  docker pull "$IMAGE"
fi
# 컨테이너에서 현재 사용자 권한으로 명령을 돌린다 (마운트 파일이 root 소유가 되지 않게)
# -i 를 반드시 붙인다. heredoc 으로 넘기는 파이썬 스크립트가 stdin 으로 들어가기 때문이다.
DRUN(){ docker run --rm -i -v "$REPO":/workspace -w /workspace "$IMAGE" "$@"; }
DROOT(){ docker run --rm -i -u 0:0 -v "$REPO":/workspace -w /workspace "$IMAGE" "$@"; }

# ------------------------------------------------------- 2.5 config 경로 정리
# config/*.yaml 에는 원 개발 환경의 절대경로가 박혀 있다. 컨테이너 기준(/workspace)으로 맞춘다.
say "config 절대경로를 컨테이너 기준으로 치환"
DROOT ./tools/setup_paths.sh --apply
DROOT chown -R "$(id -u):$(id -g)" /workspace/config

# ---------------------------------------------------------------- 3. 데이터
if [ "$SKIP_DATA" -eq 0 ]; then
  ARCHIVE="$REPO/data/_download/PanCollection.zip"
  mkdir -p "$(dirname "$ARCHIVE")"
  if [ -s "$ARCHIVE" ]; then
    say "데이터 아카이브가 이미 있다 — 내려받기 생략 ($(du -h "$ARCHIVE" | cut -f1))"
  else
    say "데이터 내려받기 (약 12 GB). gdown 은 컨테이너 안에 설치한다"
    DROOT sh -c "pip install --quiet --no-cache-dir gdown && \
                 gdown '$DATA_ID' -O '/workspace/data/_download/PanCollection.zip'"
    [ -s "$ARCHIVE" ] || die "내려받기 실패."
  fi

  say "압축 해제 + 재배치 (센서: $SENSORS)"
  # 아카이브 이름과 코드가 기대하는 배치가 다르므로 매핑해서 푼다.
  #   training_data/train_wv3_9714.h5 -> WV3/train_wv3.h5
  #   test_data/test_wv3_multiExm1.h5 -> WV3/reduced_examples_h5/test_wv3_multiExm1.h5
  DROOT python - "$SENSORS" <<'PY'
import os, re, sys, zipfile
sensors = [s.strip().lower() for s in sys.argv[1].split(",") if s.strip()]
Z = "/workspace/data/_download/PanCollection.zip"
ROOT = "/workspace/data/PanCollection"
z = zipfile.ZipFile(Z)
names = z.namelist()

def target(n, s):
    b = os.path.basename(n)
    if not b.endswith(".h5"):
        return None
    S = s.upper()
    if re.fullmatch(rf"train_{s}(_\d+)?\.h5", b):  return f"{ROOT}/{S}/train_{s}.h5"
    if re.fullmatch(rf"valid_{s}(_\d+)?\.h5", b):  return f"{ROOT}/{S}/valid_{s}.h5"
    if b == f"test_{s}_multiExm1.h5":              return f"{ROOT}/{S}/reduced_examples_h5/{b}"
    if b == f"test_{s}_OrigScale_multiExm1.h5":    return f"{ROOT}/{S}/full_examples_h5/{b}"
    return None

n_done = 0
for s in sensors:
    for n in names:
        t = target(n, s)
        if not t:
            continue
        os.makedirs(os.path.dirname(t), exist_ok=True)
        if os.path.exists(t) and os.path.getsize(t) == z.getinfo(n).file_size:
            print(f"  = {os.path.relpath(t, '/workspace')} (이미 있음)"); n_done += 1; continue
        with z.open(n) as src, open(t, "wb") as dst:
            while True:
                chunk = src.read(1 << 24)
                if not chunk: break
                dst.write(chunk)
        print(f"  + {os.path.relpath(t, '/workspace')}  ({os.path.getsize(t)/2**20:.0f} MB)")
        n_done += 1
if n_done == 0:
    raise SystemExit(f"!! 배치된 파일이 없다. 센서명을 확인할 것: {sensors}")
print(f"  총 {n_done}개")
PY

  say "저자 배포 lpan 풀기 (pan_h5.zip)"
  DROOT python - "$SENSORS" <<'PY'
import os, sys, zipfile
sensors = [s.strip().lower() for s in sys.argv[1].split(",") if s.strip()]
ROOT = "/workspace/data/PanCollection"
z = zipfile.ZipFile("/workspace/pan_h5/pan_h5.zip")
for s in sensors:
    S = s.upper()
    plan = {f"train_{s}_pan.h5":              f"{ROOT}/{S}/train_{s}_pan.h5",
            f"valid_{s}_pan.h5":              f"{ROOT}/{S}/valid_{s}_pan.h5",
            f"test_{s}_multiExm1_pan.h5":     f"{ROOT}/{S}/reduced_examples_h5/test_{s}_multiExm1_pan.h5",
            f"test_{s}_OrigScale_multiExm1_pan.h5":
                                              f"{ROOT}/{S}/full_examples_h5/test_{s}_OrigScale_multiExm1_pan.h5"}
    for src, dst in plan.items():
        if src not in z.namelist():
            print(f"  - {src} (아카이브에 없음)"); continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with z.open(src) as f, open(dst, "wb") as o: o.write(f.read())
        print(f"  + {os.path.relpath(dst, '/workspace')}")
PY

  # 컨테이너가 root 로 쓴 파일의 소유권을 되돌린다
  DROOT chown -R "$(id -u):$(id -g)" /workspace/data

  say "full-resolution lpan 복구 (KNOWN_ISSUES F-1)"
  echo "  배포 pan_h5.zip 의 WV3·QB full-res lpan 은 다른 장면이다. 그대로 쓰면 full-res 평가가 무효다."
  DROOT python tools/repair_lpan.py || echo "  !! 복구 실패 — full-res 평가 전에 수동 확인이 필요하다"
  DROOT chown -R --no-dereference "$(id -u):$(id -g)" /workspace/data

  [ "$KEEP_ARCHIVE" -eq 1 ] || { say "아카이브 삭제 (--keep-archive 로 보존 가능)"; rm -f "$ARCHIVE"; rmdir "$(dirname "$ARCHIVE")" 2>/dev/null || true; }
fi

# ---------------------------------------------------------------- 4. 검증
say "검증"
DRUN python tools/verify_metrics.py || die "지표 구현 검증 실패."
if [ "$SKIP_DATA" -eq 0 ]; then
  # 배치한 센서의 config 만 점검한다. 전체를 돌리면 받지 않은 센서 때문에 실패로 보인다.
  CFGS=""
  for s in $(echo "$SENSORS" | tr ',' ' '); do
    for f in config/pancrafter_${s}.yaml config/paper_wv3.yaml; do
      [ "$s" = "wv3" ] || [ "$f" != "config/paper_wv3.yaml" ] || continue
      [ -f "$f" ] && CFGS="$CFGS $f"
    done
  done
  # shellcheck disable=SC2086
  DRUN python tools/check_data.py $CFGS || echo "  !! 데이터 점검에서 경고가 나왔다. 위 출력을 확인할 것."
fi

say "완료"
cat <<EOF
  저장소   : $REPO
  이미지   : $IMAGE
  데이터   : $REPO/data/PanCollection

  학습 실행:
    docker run --gpus all -it --rm -v "$REPO":/workspace $IMAGE ./tools/run.sh wv3

  SSH 가 끊겨도 유지하려면:
    docker run --gpus all -d -v "$REPO":/workspace $IMAGE ./tools/run.sh wv3
EOF
