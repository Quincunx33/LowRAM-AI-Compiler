FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOWRAM_HOST=0.0.0.0 \
    LOWRAM_PORT=8000 \
    LOWRAM_MAX_CONTEXT=256

WORKDIR /app
COPY pyproject.toml README.md ./
COPY lowram_ai ./lowram_ai
COPY native ./native

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ cmake make \
    && cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build native/build -j2 \
    && pip install --no-cache-dir --no-deps .[server] \
    && apt-get purge -y g++ cmake make \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache

EXPOSE 8000
ENTRYPOINT ["lowram-ai-server"]
