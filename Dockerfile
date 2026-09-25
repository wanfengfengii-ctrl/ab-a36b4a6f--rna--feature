# syntax=docker/dockerfile:1

# Pure-backend image for the RNA folding adjudication API and the one-shot
# acceptance service.  No frontend assets are built or served.
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Install the API runtime and the HTTP client used by the acceptance runner.
COPY requirements.txt requirements-acceptance.txt ./
RUN pip install --no-cache-dir -r requirements-acceptance.txt

# Application and acceptance sources.
COPY app ./app
COPY acceptance ./acceptance

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

# Default target is the API; docker compose overrides this for acceptance.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
