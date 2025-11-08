# README: Troubleshooting Blank Page in Shiny App

## Issue
The Shiny app runs (`uv run shiny run app_shiny.py --reload`) but shows a blank page in the browser.

## Diagnosis Steps

### 1. Check if the server is running
✓ The terminal shows: "INFO: Application startup complete"
✓ The server is running on http://127.0.0.1:8000

### 2. Check browser developer console
1. Open http://127.0.0.1:8000 in your browser
2. Press F12 to open Developer Tools
3. Click the "Console" tab
4. Look for errors (they will be in red)

### Common Errors and Solutions:

#### Error: "Failed to load resource" or "404 Not Found" for widget files
**Cause:** ipywidgets/ipyleaflet JavaScript files not loading
**Solution:** This is a known issue with shinywidgets + ipyleaflet

Try running the diagnostic app first:
```bash
uv run shiny run diagnostic.py
```

#### Error: "Widget model not found" or "Widget ... not found in registry"
**Cause:** Jupyter widget infrastructure not initialized
**Solution:** Need to ensure widget extensions are properly loaded

#### Blank page with NO errors in console
**Cause:** The app might be working but rendering outside viewport
**Solution:** Try scrolling or zooming out (Ctrl + -)

### 3. Test with minimal app
```bash
uv run shiny run app_minimal.py
```

If this works but app_shiny.py doesn't, the issue is specific to ipyleaflet.

### 4. Known ipyleaflet + Shiny Issues

The combination of ipyleaflet + shinywidgets can have compatibility issues:
- ipyleaflet requires Jupyter widget protocol
- Shiny uses a different widget system
- shinywidgets bridges them, but not always perfectly

### 5. Alternative Solution: Use a different map library

Instead of ipyleaflet, consider:
- **folium** (more Shiny-friendly, but less interactive)
- **pydeck** (better Shiny support)  
- **plotly** with mapbox (works well in Shiny)

### 6. Check widget versions
```bash
uv pip list | findstr widget
uv pip list | findstr ipyleaflet
uv pip list | findstr shiny
```

Required versions:
- shiny >= 0.6.0
- shinywidgets >= 0.3.0  
- ipyleaflet >= 0.17.0
- ipywidgets >= 8.0.0

### 7. Browser Cache
Sometimes the issue is cached JavaScript. Try:
- Hard refresh: Ctrl + F5
- Open in incognito/private window
- Clear browser cache

### 8. Check Terminal Output
When you access http://127.0.0.1:8000, you should see:
```
Server function called
Rendering main map widget...
Map created at (29.24, -90.06), zoom=13
Map widget fully configured
```

If you DON'T see these messages, the server function isn't being called.

## Quick Fix to Try

Add this to the top of app_shiny.py (after imports):

```python
# Force widget manager initialization
from ipywidgets import Widget
from shinywidgets import register_widget

# This ensures widget comm infrastructure is ready
```

## Report Back

Please check:
1. What do you see in the browser console (F12 → Console tab)?
2. Do you see the debug messages in the terminal when you load the page?
3. Does the diagnostic.py app work?
4. Does app_minimal.py show content?
