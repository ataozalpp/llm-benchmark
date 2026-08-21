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

RUN python -m pip install --no-cache-dir .

RUN addgroup --system benchmark \
    && adduser --system --ingroup benchmark benchmark \
    && mkdir -p /app/runtime /app/outputs \
    && chown -R benchmark:benchmark /app

USER benchmark

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "llm_benchmark.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]