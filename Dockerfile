FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir ".[production,detection]"

COPY src/common ./src/common
COPY src/detection ./src/detection

ARG BEST_MODEL_VERSION

COPY model/${BEST_MODEL_VERSION}/weights/best.pt ./model/${BEST_MODEL_VERSION}/weights/best.pt

CMD ["uvicorn", "src.detection.app:app", "--host", "0.0.0.0", "--port", "8000"]
