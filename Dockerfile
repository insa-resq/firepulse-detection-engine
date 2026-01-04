FROM python:3.12-slim AS build

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .

RUN pip install --no-cache-dir ".[detection]" --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless==4.12.0.88

FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    gdal-bin && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV YOLO_CONFIG_DIR="/tmp"

COPY model/v4/weights/best.pt ./model/v4/weights/best.pt
COPY src/common ./src/common
COPY src/detection ./src/detection

CMD ["uvicorn", "src.detection.app:app", "--host", "0.0.0.0", "--port", "8000"]
