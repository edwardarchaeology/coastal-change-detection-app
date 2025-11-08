@echo off
REM Start the Shiny app locally using uv if available, otherwise shiny run
where uv >nul 2>&1
if %ERRORLEVEL% == 0 (
  uv run shiny run app_folium.py --host 127.0.0.1 --port 8000
) else (
  python -m shiny run app_folium.py --host 127.0.0.1 --port 8000
)
