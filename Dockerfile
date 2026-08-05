# Single shared image for producer/scorer(api)/monitor — they differ only by
# the command docker-compose runs, so one image keeps builds and deps in sync.
#
# Multi-stage: gcc and libpq-dev are needed to build psycopg2, but shipping a
# compiler in the runtime image is attack surface for no benefit. The builder
# produces a venv; the runtime stage copies that and nothing else.
#
# TODO(pin): `-slim-bookworm` pins the Debian release but still floats the
# patch version. Digest-pinning is stronger — replace with
# `python:3.11-slim-bookworm@sha256:<digest>` from `docker inspect` after a
# `docker pull`, and refresh it deliberately.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libpq-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt


FROM ${PYTHON_IMAGE} AS runtime

# libpq5 is the runtime half of libpq-dev: psycopg2 needs the shared library,
# not the headers or the compiler it was built against. curl is for the
# healthcheck below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --chown=appuser:appuser common ./common
COPY --chown=appuser:appuser producer ./producer
COPY --chown=appuser:appuser scorer ./scorer
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser monitor ./monitor
COPY --chown=appuser:appuser training ./training
COPY --chown=appuser:appuser scripts ./scripts

# Non-root. Nothing here needs a privileged port, and the only writes are to
# /reports, which compose mounts.
USER appuser

EXPOSE 8000

# Compose overrides this for producer/monitor. Having a real default means
# `docker run` on the image alone does something sensible.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
