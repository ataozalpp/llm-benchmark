FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
COPY migrations ./migrations
COPY configs ./configs
COPY data/fixtures ./data/fixtures

# Dense and hybrid RAG runs need the optional CPU embedding stack:
#   docker compose build --build-arg INSTALL_EMBEDDINGS=true
ARG INSTALL_EMBEDDINGS=false
RUN if [ "$INSTALL_EMBEDDINGS" = "true" ]; then         python -m pip install --no-cache-dir             --extra-index-url https://download.pytorch.org/whl/cpu ".[embeddings]";     else         python -m pip install --no-cache-dir .;     fi

RUN addgroup --system benchmark \
    && adduser --system --ingroup benchmark benchmark \
    && mkdir -p /app/runtime /app/outputs \
    && chown -R benchmark:benchmark /app

USER benchmark

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "llm_benchmark.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]