# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for PyMuPDF and friends
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (lean deployment set)
COPY requirements.deploy.txt /app/requirements.deploy.txt
RUN pip install --upgrade pip && pip install -r requirements.deploy.txt

# Copy app
COPY . /app

# Cloud Run sets $PORT. Keep default 8000 for local runs.
ENV PORT=8000

# Start the API (Cloud Run will pass PORT)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]