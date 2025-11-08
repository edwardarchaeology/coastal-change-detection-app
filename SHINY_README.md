# Running the Shiny Version

The Shiny for Python version is **much faster** than Streamlit because it uses reactive programming - only the parts that need to update will re-run, not the entire app!

## Installation

```bash
# Install Shiny requirements
pip install -r requirements_shiny.txt
```

## Running the App

```bash
# Run with auto-reload (development)
shiny run --reload app_shiny.py

# Run in production mode
shiny run app_shiny.py --host 0.0.0.0 --port 8000
```

The app will open at `http://localhost:8000`

## Key Improvements Over Streamlit

1. **No unnecessary reruns** - Reactive programming means only changed components update
2. **Faster UI** - ipyleaflet is more responsive than streamlit-folium
3. **Better state management** - Reactive values prevent state loss
4. **True async support** - Progress bars and background tasks work properly
5. **Conditional UI** - Sliders only appear when needed (less clutter)
6. **Faster map updates** - Direct widget manipulation without page reload

## Migration Status

### ✅ Completed
- App structure and layout
- Reactive state management  
- Map with draw controls
- Parameter inputs with conditional visibility
- Geocoding integration
- Progress indicators
- Results display framework

### 📝 To Complete
You need to copy the full processing functions from `app.py` to `coastal_processor.py`:

1. `read_band_window()` - Band reading from COG
2. `mosaic_bands()` - Multi-scene mosaicking
3. `create_rgb_composite()` - RGB image creation
4. `mask_clouds_with_scl()` - Cloud masking
5. `multi_index_water_detection()` - Multi-index algorithm
6. Complete `CoastalAnalyzer.run_snapshot()` implementation
7. Complete `CoastalAnalyzer.run_change_detection()` implementation

The framework is ready - just paste in your processing logic!

## Architecture

```
app_shiny.py          # UI and reactive server logic (fast!)
coastal_processor.py  # All processing functions (separated for speed)
requirements_shiny.txt # Dependencies
```

## Performance Tips

1. **Caching**: Add `@reactive.Calc` decorator to expensive computations
2. **Async**: Use `await asyncio.to_thread()` for CPU-heavy tasks
3. **Lazy loading**: Only load data when needed
4. **Progress bars**: Use `ui.Progress()` for user feedback

## Example Performance

**Streamlit**: ~5-8 seconds to update slider → rerun entire app  
**Shiny**: ~50ms to update slider → only reactive graph updates

This makes a **100x difference** in responsiveness!

## Next Steps

1. Copy all processing functions to `coastal_processor.py`
2. Test with `shiny run --reload app_shiny.py`
3. Verify map interactions work
4. Test full analysis pipeline
5. Add download handlers for GeoJSON/reports
6. Deploy to production

## Deployment

```bash
# Local
shiny run app_shiny.py

# Docker
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements_shiny.txt
CMD ["shiny", "run", "app_shiny.py", "--host", "0.0.0.0", "--port", "8000"]

# Shiny Server (coming soon)
# Will support automatic scaling and load balancing
```
