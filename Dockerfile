# PAN-Crafter 재현/경량화 실행 환경.
#
#   docker build -t pancrafter .
#   docker run --gpus all -it --rm \
#       -v /path/to/data:/workspace/data \
#       -v /path/to/work_dir:/workspace/work_dir \
#       pancrafter ./tools/run.sh wv3
#
# 데이터(data/)와 산출물(work_dir/)은 **이미지에 넣지 않고 마운트한다** —
# 데이터는 PanCollection 배포 조건이 있고, work_dir 은 수십 GB 로 커진다.
#
# 이 저장소에는 커스텀 CUDA 확장이 없으므로 빌드 중 컴파일이 없다.

FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# git 은 DLPan-Toolbox clone 용, libgl/libglib 은 opencv 런타임 의존이다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch/torchvision 은 베이스 이미지에 이미 있으므로 제외하고 설치한다.
COPY requirements.txt .
RUN grep -vE "^(torch|torchvision)==" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && rm /tmp/req.txt

# full-resolution 지표(D_lambda/D_s/HQNR)가 쓰는 MTF/interp23tap 구현.
# GPL-3.0 이라 저장소에는 편입하지 않고 별도 프로그램으로 둔다(단순 병치).
# reduced 지표는 이것 없이도 동작하므로, 불필요하면 --build-arg WITH_DLPAN=0.
ARG WITH_DLPAN=1
RUN if [ "$WITH_DLPAN" = "1" ]; then \
        git clone --depth 1 https://github.com/liangjiandeng/DLPan-Toolbox.git /opt/DLPan-Toolbox; \
    fi
ENV PANCRAFTER_DLPAN=/opt/DLPan-Toolbox

COPY . /workspace

# config 의 절대경로를 컨테이너 기준으로 맞춘다.
RUN ./tools/setup_paths.sh --apply || true

# 지표 구현이 정상 이식됐는지 빌드 시점에 확인한다.
RUN python tools/verify_metrics.py

CMD ["/bin/bash"]
