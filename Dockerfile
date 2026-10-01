FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-research.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --require-hashes -r requirements-research.lock

COPY . .
RUN pip install --no-cache-dir --no-deps .

ARG USER_ID=1000
RUN useradd -m -u "${USER_ID}" interfaceml \
    && mkdir -p /workspace \
    && chown interfaceml:interfaceml /workspace
USER interfaceml

HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import json, urllib.request; data = json.load(urllib.request.urlopen('http://localhost:5000/api/health')); assert data['status'] == 'ok' and data['core_available']"

EXPOSE 5000 8888

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "interfaceml.web.app:app"]
