# Single shared image for producer/scorer(api)/monitor — they differ only by
# the command docker-compose runs, so one image keeps builds/deps in sync.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY producer ./producer
COPY scorer ./scorer
COPY api ./api
COPY monitor ./monitor
COPY training ./training

ENV PYTHONUNBUFFERED=1
