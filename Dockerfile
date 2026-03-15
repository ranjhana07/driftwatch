FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server]"

COPY ui ./ui
COPY deploy/alert_rules.yaml ./deploy/alert_rules.yaml

ENV DRIFTWATCH_HOST=0.0.0.0 \
    DRIFTWATCH_LOCAL_DIR=/data/store

EXPOSE 8000 9102

CMD ["driftwatch", "serve"]
