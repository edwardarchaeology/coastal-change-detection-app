# Archive Directory

This directory contains deprecated/experimental versions of the app that are no longer used in production.

## Files

### Deprecated App Versions
- `app.py` - Original Streamlit version (replaced by Shiny)
- `app_simple.py` - Simplified Streamlit prototype
- `app_minimal.py` - Minimal proof-of-concept
- `app_shiny.py` - Early Shiny migration attempt
- `shiny_app.py` - Another Shiny iteration

**Current production app**: `../app_folium.py` (Shiny for Python with Folium mapping)

### Test Files
- `test_simple.py` - Basic unit tests
- `test.html` - HTML output test
- `main.py` - Entry point placeholder
- `diagnostic.py` - Debugging utilities

## Why Archived?

These files were part of the development process but are no longer actively maintained. They're kept for:
- Historical reference
- Code examples for specific features
- Rollback capability if needed

## Migration History

The app went through several iterations:
1. **Streamlit + Folium** (`app.py`) - Original version
2. **Streamlit simplified** (`app_simple.py`, `app_minimal.py`) - Prototyping
3. **Shiny early attempts** (`app_shiny.py`, `shiny_app.py`) - Migration experiments
4. **Shiny + Folium final** (`app_folium.py`) - Current production version ✅

See `../docs/FOLIUM_MIGRATION.md` and `../docs/MIGRATION_COMPLETE.md` for detailed migration notes.

## Usage

**Do not use these files for deployment.** They may have outdated dependencies, incomplete features, or bugs.

For the production-ready app, use:
```bash
cd ..
uv run shiny run app_folium.py
```
