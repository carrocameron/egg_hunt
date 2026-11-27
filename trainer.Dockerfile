FROM python:3.12-slim

# Copy uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install minimal system dependencies required by OpenCV (used by ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files and install packages into .venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy your training script
COPY src/ ./

# Run using the virtual environment
CMD ["uv", "run", "src/egg_hunt/train.py"]
