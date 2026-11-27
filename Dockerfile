FROM python:3.12-trixie

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_NO_DEV=1 \
    UV_FROZEN=1

# Dependencies first (cached layer, TensorFlow only reinstalls if the lock changes)
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
RUN uv sync --no-install-project

# Then the project itself
COPY README.md README.md
COPY src ./src
COPY api ./api
COPY models ./models

RUN uv sync

CMD uv run uvicorn api.fast:app --host 0.0.0.0 --port 8080
