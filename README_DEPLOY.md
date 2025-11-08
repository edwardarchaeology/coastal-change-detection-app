Deployment notes — Coastal Change Detection App

This repository contains a Shiny for Python app built around `app_folium.py`.
The following files were added to help with deployment:

- `Dockerfile` — builds a container image for the app (minimal base; extend if you need system GIS libs)
- `docker-compose.yml` — quick local compose file to run the container
- `Procfile` — for PaaS (Heroku/Render) style deploys
- `.dockerignore` — keep build context small
- `start_local.bat` — Windows helper to start the app locally

Quick local run (no Docker):
1. Create and activate a virtualenv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements_shiny.txt
```

2. Start the app (use `uv` wrapper if you have it):

```powershell
# via helper
start_local.bat

# or directly
uv run shiny run app_folium.py --host 127.0.0.1 --port 8000
# or fallback
python -m shiny run app_folium.py --host 127.0.0.1 --port 8000
```

Docker (build + run):

```bash
# Build
docker build -t coastal-change-app:latest .
# Run
docker run -p 8000:8000 --rm coastal-change-app:latest
```

Docker-compose:

```bash
docker compose up --build
```

Notes and production considerations:
- If your app uses rasterio, GDAL, or other geospatial compiled libraries, the slim image may be insufficient.
  For production you may need to install system packages (libgdal-dev, gdal-bin, build-essential) or use an image that already bundles them.
- For a production-grade ASGI server with concurrency, prefer running under Uvicorn/Gunicorn with workers; the `Procfile` uses the `uv` wrapper to ensure Uvicorn is used when available.
- Keep secrets out of the repo. Use environment variables or secrets management for API keys.

If you want, I can:
- Add a fuller Dockerfile that installs system geospatial dependencies (GDAL/rasterio) for production
- Add a small systemd service definition or a GitHub Actions CI workflow to build and push images
