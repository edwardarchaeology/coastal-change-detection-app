# Minimal Dockerfile for Shiny for Python app
# Uses slim Python base; installs requirements and runs the Shiny app
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Avoid Python writing .pyc files and buffer issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for rasterio / geospatial libs are not installed here; if you use rasterio/rtree/gdal in production
# you'll likely need a larger base image or add the relevant apt packages. Keep this file minimal; extend as needed.

# Copy dependency manifests
COPY requirements.txt .
# If you maintain a separate requirements_shiny.txt, copy that as well (optional)
COPY requirements_shiny.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt --no-cache-dir || true \
    && pip install -r requirements_shiny.txt --no-cache-dir || true

# Copy app files
COPY . /app

# Expose port used by Shiny/uvicorn
EXPOSE 8000

# Use uv run wrapper if available; fallback to shiny run
# Using uv run ensures Uvicorn is used when present; configure as you prefer in production.
CMD ["/bin/sh", "-c", "if command -v uv >/dev/null 2>&1; then uv run shiny run app_folium.py --host 0.0.0.0 --port 8000; else python -m shiny run app_folium.py --host 0.0.0.0 --port 8000; fi"]
