# Coastal Change Monitor — Shiny for Python
# Fast reactive app for coastal erosion/accretion detection
# Much faster than Streamlit - no unnecessary reruns!

from shiny import App, ui, render, reactive, req
from shinywidgets import output_widget, render_widget
from ipyleaflet import (
    Map, TileLayer, GeoJSON, DrawControl, 
    FullScreenControl, ScaleControl, LayersControl
)
from datetime import date, timedelta
import json
import asyncio
from pathlib import Path
import sys

# Debug logging
print("=" * 50, file=sys.stderr)
print("Shiny app starting...", file=sys.stderr)
print(f"Python version: {sys.version}", file=sys.stderr)

# Verify critical imports
try:
    from ipywidgets import Widget
    print("✓ ipywidgets imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"✗ ipywidgets import failed: {e}", file=sys.stderr)

try:
    import ipyleaflet
    print(f"✓ ipyleaflet version: {ipyleaflet.__version__}", file=sys.stderr)
except Exception as e:
    print(f"✗ ipyleaflet check failed: {e}", file=sys.stderr)

print("=" * 50, file=sys.stderr)

# Import all processing functions (keeping them modular)
from coastal_processor import (
    CoastalAnalyzer,
    geocode_location,
    format_results
)

# Constants
DEFAULT_CENTER = (29.24, -90.06)
DEFAULT_ZOOM = 13

app_ui = ui.page_navbar(
    ui.nav_panel("🌊 Coastal Monitor",
        ui.h2("Coastal Change Detection App", style="padding: 20px;"),
        ui.p("If you can see this text, the app is loading correctly!", style="padding: 0 20px; background-color: #e7f3ff; border: 2px solid #2196F3; border-radius: 5px;"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("📍 Location"),
                ui.input_text("search_location", "", placeholder="Search: Grand Isle, Louisiana"),
                ui.input_action_button("btn_geocode", "🔎 Search", class_="btn-sm btn-primary w-100"),
                ui.hr(),
                
                ui.h4("📅 Time Period"),
                ui.input_date("date_start", "Start", value=date.today() - timedelta(days=30)),
                ui.input_date("date_end", "End", value=date.today()),
                ui.input_slider("cloud_max", "Max cloud %", 0, 100, 30, step=5),
                
                ui.input_select("provider", "Provider",
                              choices={"mpc": "Microsoft Planetary Computer", "e84": "Element84"},
                              selected="mpc"),
                ui.hr(),
                
                ui.h4("🔬 Detection"),
                ui.input_select("method", "Algorithm",
                              choices={
                                  "fixed": "Fixed Threshold (fast)",
                                  "otsu": "Otsu Auto (recommended)",
                                  "adaptive": "Adaptive Local",
                                  "multi": "Multi-Index (best quality)"
                              },
                              selected="otsu"),
                
                ui.panel_conditional(
                    "input.method === 'fixed'",
                    ui.input_slider("threshold", "NDWI threshold", -0.3, 0.3, 0.0, step=0.05),
                ),
                
                ui.panel_conditional(
                    "input.method === 'multi'",
                    ui.input_slider("consensus", "Votes required", 1, 3, 2),
                ),
                
                ui.input_checkbox("refine", "Refine mask (morphology)", value=False),
                ui.panel_conditional("input.refine",
                    ui.input_slider("kernel", "Refinement", 2, 7, 3)
                ),
                
                ui.input_checkbox("smooth", "Smooth shoreline", value=False),
                ui.panel_conditional("input.smooth",
                    ui.input_slider("tolerance", "Smoothing (m)", 0.0, 10.0, 2.0, step=0.5)
                ),
                ui.hr(),
                
                ui.h4("📊 Analysis Mode"),
                ui.input_radio_buttons("analysis_mode", "",
                                     choices={"snapshot": "📸 Snapshot", "change": "🔄 Change Detection"},
                                     selected="snapshot"),
                
                ui.panel_conditional(
                    "input.analysis_mode === 'change'",
                    ui.input_slider("offset_days", "Historical offset (days)", 30, 730, 365, step=30),
                    ui.output_text_verbatim("date_info", placeholder=True),
                ),
                ui.hr(),
                
                ui.input_action_button("btn_run", "▶️  RUN ANALYSIS", class_="btn-success btn-lg w-100"),
                ui.div(
                    ui.output_text("aoi_status"),
                    style="margin-top: 10px; padding: 5px; text-align: center; font-size: 0.9em;"
                ),
                
                width=320
            ),
            
            ui.navset_card_tab(
                ui.nav_panel("🗺️ Map",
                    output_widget("map", height="700px"),
                    ui.div(
                        ui.p("👆 Use the draw tool (square icon in top left) to draw a rectangle around your area of interest", 
                             class_="text-muted text-center", 
                             style="font-weight: bold; font-size: 1.1em; margin: 10px;"),
                        ui.p("Then click the RUN ANALYSIS button in the sidebar", 
                             class_="text-muted text-center"),
                        style="margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;"
                    ),
                ),
                
                ui.nav_panel("📊 Results",
                    ui.output_ui("results_ui"),
                ),
                
                ui.nav_panel("ℹ️ Help",
                    ui.markdown("""
                    ## Quick Start Guide
                    
                    ### 1. Select Area
                    - Use the **draw tool** (rectangle) on the map
                    - Draw around your coastal area of interest
                    - Keep it small (< 50km²) for faster processing
                    
                    ### 2. Choose Dates
                    - **Snapshot mode**: Shows water/shoreline for selected period
                    - **Change mode**: Compares two time periods
                    
                    ### 3. Select Detection Method
                    - **Fixed**: Fast, manual threshold (adjust slider)
                    - **Otsu**: Auto-threshold, recommended for most cases
                    - **Adaptive**: Best for mixed water conditions
                    - **Multi-Index**: Slowest but most accurate (turbid water)
                    
                    ### 4. Run Analysis
                    - Click **RUN ANALYSIS** button
                    - Processing takes 30-60 seconds typically
                    - View results in the Results tab
                    
                    ### 5. Explore Results
                    - Use the **layer control** (📊 icon in top-right) to toggle layers
                    - Click layer names to show/hide:
                      - Shorelines (current/past)
                      - Water bodies
                      - Change areas (progradation/retreat)
                    - Use fullscreen button for better viewing
                    - Download results as GeoJSON for use in GIS software
                    
                    ### Understanding Results
                    
                    **Snapshot Mode:**
                    - 🟡 Yellow lines = detected shoreline (bright and clear)
                    - 💧 Blue areas = water bodies (semi-transparent)
                    - Use checkboxes in Results tab to toggle layers
                    - Use layer control (top-right icon) to toggle in map
                    
                    **Change Detection:**
                    - 🟢 Bright Green areas = Progradation (land gained/built)
                    - 🔴 Bright Red areas = Retreat/Erosion (land lost)
                    - 🔵 Cyan line (solid) = Current shoreline
                    - 🟣 Magenta line (dashed) = Past shoreline
                    - 💧 Blue/Pink areas = Current/Past water (optional, lower opacity)
                    
                    **Layer Visibility Tips:**
                    - Layers are drawn on TOP of the satellite imagery
                    - Brighter colors are easier to see over dark water
                    - Shorelines are thick (5px) for visibility
                    - Use the layer control in top-right to toggle individual layers
                    - Checkboxes control which layers are included when map re-renders
                    
                    ### Tips
                    - Start with Otsu Auto for best results
                    - Lower cloud % for clearer imagery
                    - Use 6-12 month gaps for change detection
                    - Sentinel-2 resolution = 10m pixels (~20-30m minimum detectable change)
                    - Toggle layers to focus on specific features
                    
                    ### Troubleshooting
                    - **No data found**: Try larger date range or higher cloud %
                    - **Noisy results**: Increase refinement strength
                    - **Missing water**: Try Multi-Index method or lower threshold
                    - **Slow processing**: Make area smaller, use Fixed method
                    """),
                ),
            ),
        ),
    ),
    
    title="Coastal Change Monitor",
    id="navbar",
    footer=ui.div(
        ui.p("Powered by Sentinel-2 L2A • Built with Shiny for Python", class_="text-muted text-center"),
        style="padding: 10px;"
    )
)


def server(input, output, session):
    # Debug: Log when server function is called
    print("Server function called", file=sys.stderr)
    
    # Reactive values (state management)
    aoi_bounds = reactive.Value(None)
    result_data = reactive.Value(None)
    map_widget = reactive.Value(None)
    
    @render_widget
    def map():
        """Create interactive map with draw controls"""
        print("Rendering main map widget...", file=sys.stderr)
        try:
            m = Map(
                center=DEFAULT_CENTER,
                zoom=DEFAULT_ZOOM,
                scroll_wheel_zoom=True,
                layout={'height': '700px', 'width': '100%'}
            )
            print(f"Map created at {DEFAULT_CENTER}, zoom={DEFAULT_ZOOM}", file=sys.stderr)
            
            # Add satellite basemap
            basemap = TileLayer(
                url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attribution='Esri',
                name='Satellite'
            )
            m.add_layer(basemap)
            
            # Draw control for selecting AOI
            draw = DrawControl(
                rectangle={'shapeOptions': {'color': '#0066cc', 'weight': 2}},
                polyline={},
                polygon={},
                circle={},
                marker={},
                circlemarker={}
            )
            
            def handle_draw(target, action, geo_json):
                """Handle rectangle drawing events"""
                print(f"DEBUG: Draw event - action={action}, type={geo_json.get('geometry', {}).get('type')}")  # Debug
                if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
                    coords = geo_json['geometry']['coordinates'][0]
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    
                    bounds = {
                        'min_lon': min(lons),
                        'max_lon': max(lons),
                        'min_lat': min(lats),
                        'max_lat': max(lats)
                    }
                    print(f"DEBUG: Setting bounds: {bounds}")  # Debug
                    aoi_bounds.set(bounds)
                    ui.notification_show("✓ Area selected", type="message", duration=2)
            
            draw.on_draw(handle_draw)
            m.add_control(draw)
            
            # Full screen control
            m.add_control(FullScreenControl())
            
            map_widget.set(m)
            print("Map widget fully configured", file=sys.stderr)
            return m
            
        except Exception as e:
            print(f"ERROR creating map: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Return a simple fallback map
            return Map(center=DEFAULT_CENTER, zoom=DEFAULT_ZOOM)
    
    @output
    @render.text
    def date_info():
        """Show historical date calculation"""
        if input.analysis_mode() == "change":
            start = input.date_start()
            end = input.date_end()
            offset = input.offset_days()
            
            hist_start = start - timedelta(days=offset)
            hist_end = end - timedelta(days=offset)
            
            return f"Current: {start} to {end}\nHistorical: {hist_start} to {hist_end}"
        return ""
    
    @output
    @render.text
    def aoi_status():
        """Show whether AOI is selected"""
        bounds = aoi_bounds.get()
        if bounds:
            area_deg = (bounds['max_lon'] - bounds['min_lon']) * (bounds['max_lat'] - bounds['min_lat'])
            return f"✓ AOI selected (~{area_deg:.4f} deg²)"
        return "⚠️ No area selected - draw a rectangle on the map"
    
    @render_widget
    def results_map():
        """Create results map with analysis layers"""
        result = result_data.get()
        if not result:
            # Empty map if no results
            m = Map(center=DEFAULT_CENTER, zoom=DEFAULT_ZOOM)
            return m
        
        mode = result.get('mode')
        data = result.get('data', {})
        
        # Access checkbox inputs to make this function reactive to them
        # This ensures the map re-renders when checkboxes change
        if mode == 'snapshot':
            show_water = input.layer_water()
            show_shoreline = input.layer_shoreline()
        else:  # change mode
            show_progradation = input.layer_progradation()
            show_retreat = input.layer_retreat()
            show_current_water = input.layer_current_water()
            show_past_water = input.layer_past_water()
            show_current_shore = input.layer_current_shore()
            show_past_shore = input.layer_past_shore()
        
        # Debug output
        print(f"DEBUG results_map: mode={mode}")
        print(f"DEBUG results_map: data keys={list(data.keys())}")
        if mode == 'change':
            print(f"DEBUG change mode: has progradation={('progradation' in data)}")
            print(f"DEBUG change mode: has retreat={('retreat' in data)}")
            if 'now' in data:
                print(f"DEBUG change mode: now keys={list(data['now'].keys())}")
            if 'then' in data:
                print(f"DEBUG change mode: then keys={list(data['then'].keys())}")
        
        # Get bounds from AOI
        bounds = aoi_bounds.get()
        if bounds:
            center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
            center_lon = (bounds['min_lon'] + bounds['max_lon']) / 2
            
            # Calculate zoom level based on bounds size
            # Rough approximation: smaller bounds = higher zoom
            lat_diff = bounds['max_lat'] - bounds['min_lat']
            lon_diff = bounds['max_lon'] - bounds['min_lon']
            max_diff = max(lat_diff, lon_diff)
            
            # Estimate zoom (rough heuristic)
            if max_diff < 0.01:  # ~1km
                zoom = 16
            elif max_diff < 0.05:  # ~5km
                zoom = 14
            elif max_diff < 0.1:  # ~10km
                zoom = 13
            elif max_diff < 0.2:  # ~20km
                zoom = 12
            else:
                zoom = 11
        else:
            center_lat, center_lon = DEFAULT_CENTER
            zoom = DEFAULT_ZOOM
        
        # Create map
        m = Map(
            center=(center_lat, center_lon),
            zoom=zoom,
            scroll_wheel_zoom=True,
            layout={'height': '600px', 'width': '100%'}
        )
        
        # Add satellite basemap FIRST with explicit base=True
        basemap = TileLayer(
            url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attribution='Esri',
            name='Satellite Imagery',
            base=True
        )
        m.add_layer(basemap)
        
        print(f"DEBUG: Basemap added. Mode={mode}", file=sys.stderr)
        
        # Add ALL data layers (always), checkboxes just control which ones we add
        if mode == 'snapshot':
            # Add water polygons if checkbox is checked
            if show_water and 'water' in data and data['water'].get('features'):
                water_layer = GeoJSON(
                    data=data['water'],
                    style={'color': '#3399ff', 'fillColor': '#3399ff', 'fillOpacity': 0.5, 'weight': 2},
                    name='Water Bodies'
                )
                m.add_layer(water_layer)
                print(f"DEBUG: Added water layer with {len(data['water']['features'])} features", file=sys.stderr)
            
            # Add shoreline if checkbox is checked
            if show_shoreline and 'shoreline' in data and data['shoreline'].get('features'):
                shoreline_layer = GeoJSON(
                    data=data['shoreline'],
                    style={'color': '#FFD700', 'weight': 4, 'opacity': 1.0},  # Bright yellow, thick line
                    name='Shoreline'
                )
                m.add_layer(shoreline_layer)
                print(f"DEBUG: Added shoreline layer with {len(data['shoreline']['features'])} features", file=sys.stderr)
        
        elif mode == 'change':
            # Progradation areas (green) - land gained - ADD FIRST (bottom layer)
            if show_progradation and 'progradation' in data and data['progradation'].get('features'):
                print(f"DEBUG: Adding {len(data['progradation']['features'])} progradation features", file=sys.stderr)
                prog_layer = GeoJSON(
                    data=data['progradation'],
                    style={'color': '#00ff00', 'fillColor': '#00ff00', 'fillOpacity': 0.6, 'weight': 2},
                    name='Land Gained (Progradation)'
                )
                m.add_layer(prog_layer)
            
            # Retreat areas (red) - land lost
            if show_retreat and 'retreat' in data and data['retreat'].get('features'):
                print(f"DEBUG: Adding {len(data['retreat']['features'])} retreat features", file=sys.stderr)
                retreat_layer = GeoJSON(
                    data=data['retreat'],
                    style={'color': '#ff0000', 'fillColor': '#ff0000', 'fillOpacity': 0.6, 'weight': 2},
                    name='Land Lost (Retreat)'
                )
                m.add_layer(retreat_layer)
            
            # Current period water
            if show_current_water and 'now' in data and 'water' in data['now'] and data['now']['water'].get('features'):
                print(f"DEBUG: Adding {len(data['now']['water']['features'])} current water features", file=sys.stderr)
                current_water_layer = GeoJSON(
                    data=data['now']['water'],
                    style={'color': '#0099ff', 'fillColor': '#0099ff', 'fillOpacity': 0.4, 'weight': 2},
                    name='Current Water'
                )
                m.add_layer(current_water_layer)
            
            # Past period water
            if show_past_water and 'then' in data and 'water' in data['then'] and data['then']['water'].get('features'):
                print(f"DEBUG: Adding {len(data['then']['water']['features'])} past water features", file=sys.stderr)
                past_water_layer = GeoJSON(
                    data=data['then']['water'],
                    style={'color': '#ff6666', 'fillColor': '#ff6666', 'fillOpacity': 0.4, 'weight': 2},
                    name='Past Water'
                )
                m.add_layer(past_water_layer)
            
            # Current shoreline (bright cyan) - ADD LAST (top layer)
            if show_current_shore and 'now' in data and 'shoreline' in data['now']:
                feats = data['now']['shoreline'].get('features', [])
                print(f"DEBUG: Adding {len(feats)} current shoreline features", file=sys.stderr)
                current_shore_layer = GeoJSON(
                    data=data['now']['shoreline'],
                    style={'color': '#00FFFF', 'weight': 5, 'opacity': 1.0},  # Bright cyan, thick
                    name='Current Shoreline'
                )
                m.add_layer(current_shore_layer)
            
            # Past shoreline (bright magenta)
            if show_past_shore and 'then' in data and 'shoreline' in data['then']:
                feats = data['then']['shoreline'].get('features', [])
                print(f"DEBUG: Adding {len(feats)} past shoreline features", file=sys.stderr)
                past_shore_layer = GeoJSON(
                    data=data['then']['shoreline'],
                    style={'color': '#FF00FF', 'weight': 5, 'opacity': 1.0, 'dashArray': '10, 5'},  # Bright magenta, dashed
                    name='Past Shoreline'
                )
                m.add_layer(past_shore_layer)
        
        # Add controls
        # Scale control
        scale = ScaleControl(position='bottomleft')
        m.add_control(scale)
        
        # Fullscreen control
        fullscreen = FullScreenControl(position='topleft')
        m.add_control(fullscreen)
        
        # Add layer control to toggle layers in the map
        layers_control = LayersControl(position='topright')
        m.add_control(layers_control)
        
        print(f"DEBUG: Map rendering complete", file=sys.stderr)
        print(f"DEBUG: Total layers: {len(m.layers)}", file=sys.stderr)
        print(f"DEBUG: Layer names: {[getattr(layer, 'name', 'unnamed') for layer in m.layers]}", file=sys.stderr)
        print(f"DEBUG: Total controls: {len(m.controls)}", file=sys.stderr)
        
        return m
    
    @reactive.Effect
    @reactive.event(input.btn_geocode)
    async def handle_geocode():
        """Geocode and update map"""
        query = input.search_location()
        if not query.strip():
            return
        
        with ui.Progress(min=0, max=100) as p:
            p.set(message="Searching...", value=50)
            
            try:
                result = await asyncio.to_thread(geocode_location, query)
                if result:
                    lat, lon = result
                    # Update map center
                    m = map_widget.get()
                    if m:
                        m.center = (lat, lon)
                        m.zoom = 13
                    
                    ui.notification_show(f"✓ Found: {query}", type="message")
                else:
                    ui.notification_show("Location not found", type="warning")
            except Exception as e:
                ui.notification_show(f"Geocoding error: {str(e)}", type="error")
    
    @reactive.Effect
    @reactive.event(input.btn_run)
    async def run_analysis():
        """Main analysis execution"""
        # Validate inputs
        if not aoi_bounds.get():
            ui.notification_show("⚠️ Please draw a rectangle on the map first to select your area of interest", 
                               type="warning", duration=5)
            return
        
        bounds = aoi_bounds.get()
        
        # Check bounds size
        lon_diff = bounds['max_lon'] - bounds['min_lon']
        lat_diff = bounds['max_lat'] - bounds['min_lat']
        
        if lon_diff < 0.01 or lat_diff < 0.01:
            ui.notification_show("Area too small! Draw a larger rectangle.", type="error")
            return
        
        if lon_diff > 1.0 or lat_diff > 1.0:
            ui.notification_show("Area too large! Processing may be slow.", type="warning")
        
        # Map method names to codes
        method_map = {
            'fixed': 'fixed',
            'otsu': 'otsu',
            'adaptive': 'adaptive',
            'multi': 'multi-index'
        }
        method_code = method_map.get(input.method(), 'fixed')
        
        # Build parameters
        params = {
            'bbox': bounds,
            'start_date': input.date_start(),
            'end_date': input.date_end(),
            'cloud_max': input.cloud_max(),
            'provider': input.provider(),
            'method': input.analysis_mode(),  # 'snapshot' or 'change'
            'mode': input.analysis_mode(),
            'detection_method': method_code,
            'ndwi_threshold': input.threshold() if input.method() == 'fixed' else 0.0,
            'consensus_votes': input.consensus() if input.method() == 'multi' else 2,
            'apply_morphology': input.refine(),
            'morph_kernel': input.kernel() if input.refine() else 3,
            'smooth_tolerance': input.tolerance() if input.smooth() else 0.0,
            'offset_days': input.offset_days() if input.analysis_mode() == 'change' else 365
        }
        
        # Run analysis
        with ui.Progress(min=0, max=100) as p:
            p.set(message="🔍 Searching for scenes...", value=10)
            
            try:
                analyzer = CoastalAnalyzer(params)
                
                p.set(message="📡 Downloading satellite data...", value=30)
                await asyncio.sleep(0.1)  # Allow UI update
                
                p.set(message="🧮 Computing water indices...", value=50)
                result = await asyncio.to_thread(analyzer.run)
                
                p.set(message="🗺️ Creating vectors...", value=80)
                await asyncio.sleep(0.1)
                
                p.set(message="✓ Complete!", value=100)
                
                # Store result with mode information
                result_with_mode = {
                    'mode': input.analysis_mode(),  # 'snapshot' or 'change'
                    'data': result
                }
                result_data.set(result_with_mode)
                
                # Results are shown in a separate map in the Results tab
                # add_result_layers(result)  # Optional: could also add to main map
                
                ui.notification_show("✓ Analysis complete! Check Results tab", type="message", duration=5)
                
            except Exception as e:
                ui.notification_show(f"❌ Analysis failed: {str(e)}", type="error", duration=10)
                print(f"Error: {e}")  # Log for debugging
    
    def add_result_layers(result):
        """Add GeoJSON layers to map"""
        m = map_widget.get()
        if not m or not result:
            return
        
        # Clear existing layers (keep basemap and draw control)
        # ... add GeoJSON layers for water, shoreline, changes
        # This would be implemented based on result structure
    
    @output
    @render.ui
    def results_ui():
        """Render results panel"""
        result = result_data.get()
        
        if result is None:
            return ui.div(
                ui.tags.div(
                    ui.tags.i(class_="fa fa-info-circle fa-3x text-muted"),
                    ui.h4("No results yet", class_="text-muted mt-3"),
                    ui.p("Run an analysis to see results here"),
                    class_="text-center py-5"
                )
            )
        
        # Render based on result type
        if result['mode'] == 'snapshot':
            return render_snapshot_results(result)
        else:
            return render_change_results(result)
    
    def render_snapshot_results(result):
        """Render snapshot analysis results"""
        data = result['data']
        
        return ui.div(
            ui.h3("📸 Snapshot Analysis Results"),
            ui.hr(),
            
            ui.row(
                ui.column(6,
                    ui.card(
                        ui.card_header("Scene Information"),
                        ui.p(f"Scene ID: {data.get('item_id', 'N/A')}"),
                        ui.p(f"Cloud cover: {data.get('cloud_cover', 0):.1f}%"),
                        ui.p(f"Detection method: {data.get('method', 'N/A')}"),
                    )
                ),
                ui.column(6,
                    ui.card(
                        ui.card_header("Detected Features"),
                        ui.p(f"Water polygons: {len(data.get('water', {}).get('features', []))}"),
                        ui.p(f"Shoreline segments: {len(data.get('shoreline', {}).get('features', []))}"),
                        ui.p(f"Total water area: {data.get('water_area_km2', 0):.2f} km²"),
                    )
                ),
            ),
            
            ui.hr(),
            ui.h4("Results Map"),
            
            ui.card(
                ui.card_header("Layer Visibility"),
                ui.row(
                    ui.column(6, ui.input_checkbox("layer_water", "💧 Water Bodies", value=True)),
                    ui.column(6, ui.input_checkbox("layer_shoreline", "🌊 Shoreline", value=True)),
                ),
                class_="mb-2"
            ),
            
            output_widget("results_map", height="600px"),
            
            ui.hr(),
            ui.download_button("btn_download_geojson", "⬇️ Download GeoJSON", class_="btn-primary"),
        )
    
    def render_change_results(result):
        """Render change detection results"""
        data = result['data']
        
        return ui.div(
            ui.h3("🔄 Change Detection Results"),
            ui.hr(),
            
            ui.row(
                ui.column(4,
                    ui.value_box(
                        "Progradation",
                        f"{data.get('progradation_area_km2', 0):.3f} km²",
                        "Land gained",
                        theme="success"
                    )
                ),
                ui.column(4,
                    ui.value_box(
                        "Retreat",
                        f"{data.get('retreat_area_km2', 0):.3f} km²",
                        "Land lost",
                        theme="danger"
                    )
                ),
                ui.column(4,
                    ui.value_box(
                        "Net Change",
                        f"{data.get('net_change_km2', 0):+.3f} km²",
                        "Overall",
                        theme="info"
                    )
                ),
            ),
            
            ui.hr(),
            ui.h4("Period Comparison"),
            ui.row(
                ui.column(6,
                    ui.p(f"**Current:** {data.get('current_period', 'N/A')}"),
                    ui.p(f"Cloud cover: {data.get('current_cloud', 0):.1f}%"),
                ),
                ui.column(6,
                    ui.p(f"**Historical:** {data.get('historical_period', 'N/A')}"),
                    ui.p(f"Cloud cover: {data.get('historical_cloud', 0):.1f}%"),
                ),
            ),
            
            ui.hr(),
            ui.h4("Change Detection Map"),
            
            ui.card(
                ui.card_header("Layer Visibility"),
                ui.row(
                    ui.column(4, ui.input_checkbox("layer_progradation", "🟢 Land Gained", value=True)),
                    ui.column(4, ui.input_checkbox("layer_retreat", "� Land Lost", value=True)),
                    ui.column(4, ui.input_checkbox("layer_current_shore", "🌊 Current Shoreline", value=True)),
                ),
                ui.row(
                    ui.column(4, ui.input_checkbox("layer_past_shore", "🌊 Past Shoreline", value=True)),
                    ui.column(4, ui.input_checkbox("layer_current_water", "💧 Current Water", value=False)),
                    ui.column(4, ui.input_checkbox("layer_past_water", "💧 Past Water", value=False)),
                ),
                class_="mb-2"
            ),
            
            output_widget("results_map", height="650px"),
            
            ui.hr(),
            ui.row(
                ui.column(6,
                    ui.download_button("btn_download_change", "⬇️ Download Change Data", class_="btn-primary"),
                ),
                ui.column(6,
                    ui.download_button("btn_download_report", "📄 Download Report", class_="btn-secondary"),
                ),
            ),
        )


app = App(app_ui, server)
