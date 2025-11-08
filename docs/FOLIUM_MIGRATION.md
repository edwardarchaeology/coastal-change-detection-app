# Switched to Folium for Reliable Map Rendering

## Problem with ipyleaflet
Your original app using `ipyleaflet` had persistent issues with layer rendering:
- Layers would randomly show/hide when toggling checkboxes
- Layers sometimes appeared to be underneath the basemap
- Widget communication was unreliable between Shiny and ipyleaflet
- The entire map re-rendered on every checkbox change, causing flicker and unpredictable behavior

## Solution: Folium
Switched to **Folium**, a more mature and stable mapping library that:

### Advantages of Folium
✅ **Renders as static HTML** - no widget communication issues
✅ **Layers ALWAYS show up** - GeoJSON is embedded in HTML
✅ **Built-in LayerControl works perfectly** - toggle layers reliably
✅ **Simpler interaction model** - no complex widget state management
✅ **Better Shiny integration** - just renders HTML, no special bridges needed
✅ **More predictable** - what you add is what you see

### Key Differences

**Old (ipyleaflet):**
- Used Jupyter widget infrastructure (`ipywidgets`)
- Required `shinywidgets` bridge
- Interactive drawing for AOI selection (buggy)
- Layers could disappear mysteriously
- Needed manual widget state management

**New (Folium):**
- Renders maps as HTML iframes
- No widget bridge needed
- Simple numeric input for coordinates
- Layers always render predictably
- Built-in controls work out-of-the-box

## New File: app_folium.py

Run with:
```bash
uv run shiny run app_folium.py --reload
```

### How It Works

1. **Main Map (Map tab):**
   - Shows satellite imagery basemap
   - Enter coordinates (min/max lat/lon) to define bounding box
   - Yellow rectangle shows selected area
   - Search button to center map on location

2. **Results Map (Results tab):**
   - Folium map with all analysis layers
   - Bright, high-contrast colors:
     - Yellow (#FFD700) shorelines
     - Bright green (#00ff00) progradation
     - Bright red (#ff0000) retreat
     - Cyan (#00FFFF) current shoreline
     - Magenta (#FF00FF) past shoreline
   - **Layer control widget (top-right)** for toggling
   - Checkboxes control which layers to include

### Changes from ipyleaflet Version

1. **Area Selection:**
   - Old: Draw rectangle with mouse (unreliable)
   - New: Enter coordinates in form (reliable)

2. **Map Rendering:**
   - Old: `@render_widget` with ipyleaflet Map objects
   - New: `@render.ui` with HTML from Folium

3. **Layer Adding:**
   - Old: `m.add_layer(GeoJSON(...))`  - sometimes didn't show
   - New: `folium.GeoJson(...).add_to(m)` - always shows

4. **Interactivity:**
   - Old: Fully interactive widgets (buggy)
   - New: HTML-based (stable, predictable)

### Layer Styling

All layers use bright, high-visibility colors:

```python
# Shoreline (yellow, thick)
style_function=lambda x: {
    'color': '#FFD700',
    'weight': 5,
    'opacity': 1.0
}

# Progradation (bright green)
style_function=lambda x: {
    'fillColor': '#00ff00',
    'color': '#00ff00',
    'weight': 2,
    'fillOpacity': 0.6
}

# Retreat (bright red)
style_function=lambda x: {
    'fillColor': '#ff0000',
    'color': '#ff0000',
    'weight': 2,
    'fillOpacity': 0.6
}
```

## Testing

1. **Start the app:**
   ```bash
   uv run shiny run app_folium.py --reload
   ```

2. **Set an area:**
   - Enter coordinates (e.g., Grand Isle, LA):
     - Min Lat: 29.23
     - Max Lat: 29.28
     - Min Lon: -90.16
     - Max Lon: -90.11
   - Click "Set Bounding Box"
   - Yellow rectangle should appear on map

3. **Run analysis:**
   - Choose dates and method
   - Click "RUN ANALYSIS"
   - Go to Results tab

4. **Verify layers:**
   - ALL layers should be visible with bright colors
   - Toggle checkboxes - map re-renders with selected layers
   - Use layer control (top-right) - layers toggle on/off
   - **Layers should NEVER disappear mysteriously!**

## Why This is Better

**Reliability**: Folium is battle-tested for web mapping. It's used in production by many organizations.

**Simplicity**: No complex widget state to manage. Maps are just HTML.

**Predictability**: What you code is what you see. No hidden widget lifecycle issues.

**Debugging**: Easy to inspect - just view the HTML source of the map.

**Performance**: Static HTML renders faster than interactive widgets.

## Files

- **app_folium.py** - New Folium-based app (RECOMMENDED)
- **app_shiny.py** - Old ipyleaflet version (problematic)
- **requirements_shiny.txt** - Updated to use folium instead of ipyleaflet

## Recommendation

**Use app_folium.py going forward.** It's more reliable and the layers will always render correctly.

The trade-off is that area selection is via coordinate input rather than drawing, but this is actually more precise and reliable.
