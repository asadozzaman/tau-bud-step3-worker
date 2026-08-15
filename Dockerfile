FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models \
    CUTIE_REPO=/opt/Cutie \
    OUTPUT_S3_PREFIX=bud-results \
    SAVE_OUTPUT_VIDEOS=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

ARG CUTIE_REPO_URL=https://github.com/hkchengrex/Cutie.git
ARG CUTIE_REF=main
RUN git clone --depth 1 "${CUTIE_REPO_URL}" /opt/Cutie \
    && cd /opt/Cutie \
    && git fetch --depth 1 origin "${CUTIE_REF}" || true \
    && git checkout "${CUTIE_REF}" || true \
    && python -m pip install -e /opt/Cutie \
    && python /opt/Cutie/cutie/utils/download_models.py

COPY Step_3_BudSpur_Combined_v3.py /app/Step_3_BudSpur_Combined_v3.py
COPY handler.py /app/handler.py
COPY examples /app/examples
COPY models /models

CMD ["python", "-u", "/app/handler.py"]
