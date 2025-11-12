# Coastal Change Monitor — Shiny for Python (Folium Version)
# Using Folium for more reliable map rendering

from shiny import App, ui, render, reactive, req
import folium
from folium import plugins
from datetime import date, timedelta
import json
import asyncio
from pathlib import Path
import sys

# Import all processing functions
from coastal_processor import (
    CoastalAnalyzer,
    geocode_location,
    format_results
)

# Debug logging
print("=" * 50, file=sys.stderr)
print("Shiny app with Folium starting...", file=sys.stderr)
print(f"Python version: {sys.version}", file=sys.stderr)
print("=" * 50, file=sys.stderr)

# Constants
DEFAULT_CENTER = (29.24, -90.06)
DEFAULT_ZOOM = 13

app_ui = ui.page_navbar(
    ui.nav_panel("🌊 Coastal Monitor",
        ui.h2("Coastal Change Detection App", style="padding: 20px;"),
        
        ui.navset_card_tab(
            ui.nav_panel("🗺️ Map",
                # Map section
                ui.row(
                    ui.column(7,
                        ui.output_ui("map_ui"),
                    ),
                    ui.column(5,
                        ui.card(
                            ui.card_header("🗺️ Map Controls"),
                            ui.div(
                                ui.input_select("basemap", "Basemap Style",
                                              choices={
                                                  "satellite": "🛰️ Satellite Imagery",
                                                  "grayscale": "⚪ Grayscale (High Contrast)"
                                              },
                                              selected="satellite"),
                                ui.hr(),
                                ui.h6("Define Area of Interest"),
                                
                                # Set Bounding Box button at the top
                                ui.input_action_button("btn_set_bbox", "✓ Set Bounding Box", class_="btn-success w-100 mb-3 btn-lg fw-bold"),
                                ui.p("👆 CLICK THIS after drawing rectangle or entering coordinates", class_="small text-success text-center mb-3 fw-bold"),
                                
                                ui.div(
                                    ui.div(
                                        ui.p("⚠️ After drawing a rectangle on the map:", class_="small fw-bold text-warning mb-1"),
                                        ui.p("Click the 'Set Bounding Box' button above to confirm your selection", 
                                             class_="small text-warning mb-2"),
                                        class_="alert alert-warning py-2 px-2 mb-2"
                                    ),
                                    ui.p("📍 Option 1: Draw on map", class_="small fw-bold mb-1"),
                                    ui.p("Use the rectangle tool (◻️) in top-left of map, draw your area, then click 'Set Bounding Box' above", 
                                         class_="small text-muted mb-2"),
                                    ui.p("📝 Option 2: Enter coordinates manually", class_="small fw-bold mb-1"),
                                    ui.p("Expand 'Manual Coordinate Entry' below to enter lat/lon values", 
                                         class_="small text-muted mb-2"),
                                ),
                                
                                # Collapsible manual coordinate entry
                                ui.accordion(
                                    ui.accordion_panel(
                                        "📝 Manual Coordinate Entry",
                                        ui.p("Enter lat/lon values:", class_="small text-muted mb-2"),
                                        ui.input_numeric("min_lat", "Min Latitude (South)", value=29.25, step=0.001),
                                        ui.input_numeric("max_lat", "Max Latitude (North)", value=29.27, step=0.001),
                                        ui.input_numeric("min_lon", "Min Longitude (West)", value=-90.15, step=0.001),
                                        ui.input_numeric("max_lon", "Max Longitude (East)", value=-90.12, step=0.001),
                                    ),
                                    id="coord_accordion",
                                    open=False,
                                    class_="mb-3"
                                ),
                                
                                ui.input_action_button("btn_clear_bbox", "Clear Bounding Box", class_="btn-secondary w-100 btn-sm"),
                                style="height: 400px; overflow-y: auto;"
                            ),
                        ),
                    ),
                ),
                
                ui.hr(),
                
                # Analysis controls below map
                ui.card(
                    ui.card_header("⚙️ Analysis Configuration"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("📍 Location"),
                            ui.input_text("search_location", "", placeholder="Search: Grand Isle, Louisiana"),
                            ui.input_action_button("btn_geocode", "🔎 Search", class_="btn-sm btn-primary w-100"),
                        ),
                        
                        ui.card(
                            ui.card_header("📅 Time Period"),
                            ui.input_date("date_start", "Start", value=date.today() - timedelta(days=30)),
                            ui.input_date("date_end", "End", value=date.today()),
                            ui.input_slider("cloud_max", "Max cloud %", 0, 100, 30, step=5),
                            ui.input_slider("max_scenes", "Max scenes to process", 
                                          min=1, max=20, value=10, step=1),
                            ui.tags.small("⚡ Fewer scenes = faster processing", class_="text-muted"),
                        ),
                        
                        ui.card(
                            ui.card_header("🔬 Detection Method"),
                            ui.input_select("method", "Algorithm",
                                          choices={
                                              "fixed": "Fixed Threshold",
                                              "otsu": "Otsu Auto ⭐",
                                              "adaptive": "Adaptive Local",
                                              "multi": "Multi-Index (best)"
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
                        ),
                        
                        ui.card(
                            ui.card_header("⚙️ Analysis Mode"),
                            ui.input_select("provider", "Data Provider",
                                          choices={"mpc": "Microsoft Planetary", "e84": "Element84"},
                                          selected="mpc"),
                            
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
                        ),
                        
                        col_widths=[3, 3, 3, 3]
                    ),
                ),
            ),
            
            ui.nav_panel("📊 Results",
                ui.output_ui("results_ui"),
                value="results_tab"  # ID for programmatic access
            ),
            
            ui.nav_panel("ℹ️ Help",
                ui.markdown("""
                ## Quick Start Guide
                
                ### 1. Select Your Area
                
                **🎨 DRAW METHOD:**
                1. Click the **rectangle tool** (⬜) in the top-left corner of the map
                2. Click and drag on the map to draw your area
                3. ⚠️ **IMPORTANT**: Click the green **"✓ Set Bounding Box"** button to confirm
                4. Yellow box appears - now you're ready to analyze!
                
                **⌨️ MANUAL METHOD:**
                1. Enter a location in the **Search box** (e.g., "Grand Isle, Louisiana")
                2. Click **🔎 Search** - map centers on that location
                3. Adjust the **coordinate values** in the right panel
                4. Click **✓ Set Bounding Box** - yellow rectangle appears
                
                **Coordinate Tips:**
                - Latitude: Positive = North, range typically 25-50° for USA
                - Longitude: Negative = West, range typically -125 to -65° for USA  
                - Keep area small: 0.02° × 0.03° ≈ 2km × 3km processes fastest
                - The sidebar shows area size after you set the bounding box

                
                ### 2. Choose Dates
                - **Snapshot mode**: Shows water/shoreline for selected period
                - **Change mode**: Compares two time periods (historical offset)
                
                ### 3. Select Detection Method
                - **Otsu**: Recommended! Auto-threshold works for most cases
                - **Fixed**: Manual threshold (for experts)
                - **Adaptive**: Best for varied water conditions
                - **Multi-Index**: Most accurate for turbid/muddy water
                
                ### 4. Run Analysis
                - Ensure your yellow bounding box is visible
                - Click **▶️ RUN ANALYSIS** button
                - Progress bar shows: searching → downloading → computing → vectorizing
                - Takes 30-90 seconds typically
                
                ### 5. Explore Results
                - Switch to **Results tab**
                - **Layers render reliably with bright colors!**
                - Use **layer control** (top-right icon) to toggle visibility
                - Toggle checkboxes to show/hide specific layers
                
                ### Understanding Results
                
                **Snapshot Mode:**
                - 🟡 **Yellow lines** = detected shoreline (thick, visible!)
                - 💙 **Blue areas** = water bodies (semi-transparent)
                
                **Change Detection:**
                - 🟢 **Bright Green** = Land gained (progradation)
                - 🔴 **Bright Red** = Land lost (retreat/erosion)
                - 🔵 **Cyan** (solid line) = Current shoreline
                - 🟣 **Magenta** (dashed line) = Past shoreline
                - 💙 **Blue** = Current water (optional layer)
                - 🩷 **Pink** = Past water (optional layer)
                
                ### Pro Tips
                - Start with Otsu + 20% cloud max
                - Use 6-12 month gaps for change detection
                - Smaller areas = faster processing
                - Sentinel-2 resolution = 10m/pixel (minimum detectable ~20-30m)
                - Save good coordinate sets for reuse
                - Yellow box must be visible before running analysis!
                """),
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
    print("Server function called", file=sys.stderr)
    
    # Reactive values
    aoi_bounds = reactive.Value(None)
    result_data = reactive.Value(None)
    map_center = reactive.Value(DEFAULT_CENTER)
    
    @output
    @render.ui
    def map_ui():
        """Render Folium map as HTML"""
        # Get basemap selection
        basemap_choice = input.basemap() if hasattr(input, 'basemap') else 'satellite'
        
        # Define basemap tiles
        if basemap_choice == 'grayscale':
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
            attr = 'Esri Grayscale'
        else:  # satellite
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
            attr = 'Esri Satellite'
        
        # Get current bounds to determine map center
        bounds = aoi_bounds.get()
        if bounds:
            # Center map on the bounding box
            center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
            center_lon = (bounds['min_lon'] + bounds['max_lon']) / 2
            center_location = (center_lat, center_lon)
        else:
            center_location = map_center.get()
        
        # Create map
        m = folium.Map(
            location=center_location,
            zoom_start=DEFAULT_ZOOM,
            tiles=tiles,
            attr=attr
        )
        
        # If AOI is set, draw it
        bounds = aoi_bounds.get()
        if bounds:
            sw = [bounds['min_lat'], bounds['min_lon']]
            ne = [bounds['max_lat'], bounds['max_lon']]
            folium.Rectangle(
                bounds=[sw, ne],
                color='#FFD700',
                fill=True,
                fillColor='#FFD700',
                fillOpacity=0.3,
                weight=4,
                popup='Area of Interest'
            ).add_to(m)
        
        # Add Draw control for interactive drawing
        draw = plugins.Draw(
            export=False,  # Disable export to prevent JSON popup
            position='topleft',
            draw_options={
                'polyline': False,
                'polygon': False,
                'circle': False,
                'marker': False,
                'circlemarker': False,
                'rectangle': {
                    'shapeOptions': {
                        'color': '#00FFFF',
                        'weight': 3,
                        'fillOpacity': 0.3
                    }
                }
            },
            edit_options={'edit': False, 'remove': True}
        )
        draw.add_to(m)
        
        # Add fullscreen
        plugins.Fullscreen(position='topleft').add_to(m)
        
        # Add scale
        plugins.MeasureControl(position='bottomleft').add_to(m)
        
        # Get HTML
        map_html = m._repr_html_()
        
        # Add custom CSS to hide the export popup/modal
        custom_css = """
        <style>
        /* Hide Leaflet Draw export modal/popup */
        .leaflet-draw-actions-bottom,
        .leaflet-draw-toolbar-button-export,
        .leaflet-draw-section:has(.leaflet-draw-edit-export),
        [data-action="export"],
        button[title*="xport"],
        .leaflet-popup-pane .leaflet-popup:has(pre) {
            display: none !important;
        }
        
        /* Also hide any alert/modal dialogs that might be GeoJSON popups */
        div[role="dialog"]:has(pre),
        div.modal:has(pre) {
            display: none !important;
        }
        </style>
        """
        
        # Add custom JavaScript to capture draw events and update Shiny inputs
        custom_js = """
        <script>
        (function() {
            let checkInterval;
            let checkCount = 0;
            const maxChecks = 100;
            
            function setupDrawListener() {
                checkCount++;
                
                // Find all iframes (Folium renders in iframe)
                const iframes = document.querySelectorAll('iframe');
                
                for (let iframe of iframes) {
                    try {
                        const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                        const mapDivs = iframeDoc.querySelectorAll('.folium-map');
                        
                        for (let mapDiv of mapDivs) {
                            if (mapDiv._leaflet_id && iframeDoc.defaultView.L) {
                                const L = iframeDoc.defaultView.L;
                                
                                // Find the map instance
                                for (let key in iframeDoc.defaultView) {
                                    const obj = iframeDoc.defaultView[key];
                                    if (obj && obj._container === mapDiv) {
                                        const map = obj;
                                        
                                        // Attach draw:created event
                                        map.on('draw:created', function(e) {
                                            const layer = e.layer;
                                            const bounds = layer.getBounds();
                                            
                                            const coords = {
                                                min_lat: bounds.getSouth().toFixed(6),
                                                max_lat: bounds.getNorth().toFixed(6),
                                                min_lon: bounds.getWest().toFixed(6),
                                                max_lon: bounds.getEast().toFixed(6)
                                            };
                                            
                                            console.log('Rectangle drawn:', coords);
                                            
                                            // Update Shiny inputs
                                            if (window.Shiny) {
                                                Shiny.setInputValue('min_lat', parseFloat(coords.min_lat));
                                                Shiny.setInputValue('max_lat', parseFloat(coords.max_lat));
                                                Shiny.setInputValue('min_lon', parseFloat(coords.min_lon));
                                                Shiny.setInputValue('max_lon', parseFloat(coords.max_lon));
                                                Shiny.setInputValue('auto_set_bbox', Date.now(), {priority: 'event'});
                                            }
                                            
                                            // Remove the drawn layer
                                            map.removeLayer(layer);
                                        });
                                        
                                        console.log('Successfully attached draw listener to Folium map');
                                        clearInterval(checkInterval);
                                        return;
                                    }
                                }
                            }
                        }
                    } catch (e) {
                        // Cross-origin or access error, continue
                    }
                }
                
                // Stop after max attempts
                if (checkCount >= maxChecks) {
                    clearInterval(checkInterval);
                    console.warn('Could not attach draw listener to map');
                }
            }
            
            // Check every 100ms for the map
            checkInterval = setInterval(setupDrawListener, 100);
            setupDrawListener(); // Try immediately
        })();
        
        // Additional script to prevent and close any GeoJSON export popups
        (function() {
            // Override alert to prevent GeoJSON popups
            const originalAlert = window.alert;
            window.alert = function(msg) {
                // Only block alerts that look like GeoJSON (contain "type":"Feature" or coordinates)
                if (msg && (msg.includes('"type":"Feature"') || msg.includes('"coordinates"'))) {
                    console.log('Blocked GeoJSON export popup');
                    return;
                }
                originalAlert.apply(window, arguments);
            };
            
            // Listen for iframes and do the same inside them
            document.addEventListener('DOMContentLoaded', function() {
                const iframes = document.querySelectorAll('iframe');
                iframes.forEach(function(iframe) {
                    try {
                        const iframeWindow = iframe.contentWindow;
                        if (iframeWindow) {
                            const originalIframeAlert = iframeWindow.alert;
                            iframeWindow.alert = function(msg) {
                                if (msg && (msg.includes('"type":"Feature"') || msg.includes('"coordinates"'))) {
                                    console.log('Blocked GeoJSON export popup in iframe');
                                    return;
                                }
                                originalIframeAlert.apply(iframeWindow, arguments);
                            };
                        }
                    } catch (e) {
                        // Cross-origin, ignore
                    }
                });
            });
        })();
        </script>
        """
        
        return ui.HTML(f'<div style="height: 400px; width: 100%;">{custom_css}{map_html}{custom_js}</div>')
    
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
        return "⚠️ Enter coordinates and click 'Set Bounding Box'"
    
    @reactive.Effect
    @reactive.event(input.btn_set_bbox)
    def set_bbox():
        """Set bounding box from input coordinates"""
        try:
            bounds = {
                'min_lon': input.min_lon(),
                'max_lon': input.max_lon(),
                'min_lat': input.min_lat(),
                'max_lat': input.max_lat()
            }
            
            # Validate
            if bounds['min_lon'] >= bounds['max_lon']:
                ui.notification_show("❌ Min longitude must be less than max longitude", type="error", duration=3)
                return
            if bounds['min_lat'] >= bounds['max_lat']:
                ui.notification_show("❌ Min latitude must be less than max latitude", type="error", duration=3)
                return
            
            aoi_bounds.set(bounds)
            
            # Calculate and show area
            area_deg = (bounds['max_lon'] - bounds['min_lon']) * (bounds['max_lat'] - bounds['min_lat'])
            ui.notification_show(f"✅ Area of interest set! ({area_deg:.6f}° area)", 
                               type="message", duration=3)
            print(f"Bounding box set from coordinates: {bounds}", file=sys.stderr)
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error", duration=3)
    
    @reactive.Effect
    @reactive.event(input.btn_clear_bbox)
    def clear_bbox():
        """Clear bounding box"""
        aoi_bounds.set(None)
        print("Bounding box cleared", file=sys.stderr)
    
    @reactive.Effect
    @reactive.event(input.auto_set_bbox)
    def auto_set_bbox_from_draw():
        """Automatically set bounding box when rectangle is drawn on map"""
        try:
            # Get the coordinates from the input fields (which were updated by JavaScript)
            min_lat = input.min_lat()
            max_lat = input.max_lat()
            min_lon = input.min_lon()
            max_lon = input.max_lon()
            
            if all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
                bounds = {
                    'min_lat': float(min_lat),
                    'max_lat': float(max_lat),
                    'min_lon': float(min_lon),
                    'max_lon': float(max_lon)
                }
                aoi_bounds.set(bounds)
                area = (bounds['max_lon'] - bounds['min_lon']) * (bounds['max_lat'] - bounds['min_lat'])
                print(f"✓ Bounding box auto-set from drawing: {bounds} (area: {area:.6f} deg²)", file=sys.stderr)
                
                # Show success notification
                ui.notification_show(f"✅ Area selected: {area:.6f}° × {area:.6f}°", 
                                   type="message", duration=2)
        except Exception as e:
            print(f"Error auto-setting bbox: {e}", file=sys.stderr)
            ui.notification_show(f"⚠️ Error setting area: {str(e)}", type="warning", duration=3)
    
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
                    map_center.set((lat, lon))
                    print(f"Location found: {query} at ({lat}, {lon})", file=sys.stderr)
                else:
                    ui.notification_show("Location not found", type="warning", duration=3)
            except Exception as e:
                ui.notification_show(f"Search error: {str(e)}", type="error", duration=3)
    
    @reactive.Effect
    @reactive.event(input.btn_run)
    async def run_analysis():
        """Main analysis execution"""
        # Validate inputs
        if not aoi_bounds.get():
            ui.notification_show(
                "⚠️ Please define your area of interest:\n" +
                "• Draw a rectangle on the map, OR\n" +
                "• Enter coordinates and click 'Set Bounding Box'", 
                type="warning", duration=7)
            return
        
        bounds = aoi_bounds.get()
        
        # Check bounds size
        lon_diff = bounds['max_lon'] - bounds['min_lon']
        lat_diff = bounds['max_lat'] - bounds['min_lat']
        
        if lon_diff < 0.01 or lat_diff < 0.01:
            ui.notification_show("Area too small! Select a larger area.", type="error")
            return
        
        if lon_diff > 1.0 or lat_diff > 1.0:
            ui.notification_show("Area too large! Processing may be slow.", type="warning")
        
        # Map method names
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
            'method': input.analysis_mode(),
            'mode': input.analysis_mode(),
            'detection_method': method_code,
            'ndwi_threshold': input.threshold() if input.method() == 'fixed' else 0.0,
            'consensus_votes': input.consensus() if input.method() == 'multi' else 2,
            'apply_morphology': input.refine(),
            'morph_kernel': input.kernel() if input.refine() else 3,
            'smooth_tolerance': input.tolerance() if input.smooth() else 0.0,
            'offset_days': input.offset_days() if input.analysis_mode() == 'change' else 365,
            'max_scenes': input.max_scenes()
        }
        
        # Run analysis
        mode_name = "change detection" if input.analysis_mode() == "change" else "snapshot"
        max_sc = input.max_scenes()
        
        with ui.Progress(min=0, max=100) as p:
            p.set(message=f"Step 1/9: 🔍 Searching STAC catalog for Sentinel-2 scenes...", value=3)
            await asyncio.sleep(0.1)
            
            try:
                p.set(message=f"Step 2/9: 📋 Filtering best {max_sc} scenes (cloud cover ≤ {input.cloud_max()}%)...", value=8)
                analyzer = CoastalAnalyzer(params)
                await asyncio.sleep(0.1)
                
                p.set(message=f"Step 3/9: 📡 Downloading Red band from {max_sc} scenes... (SLOWEST STEP)", value=15)
                await asyncio.sleep(0.2)
                
                p.set(message=f"Step 4/9: 📡 Downloading Green band from {max_sc} scenes...", value=28)
                await asyncio.sleep(0.2)
                
                p.set(message=f"Step 5/9: 📡 Downloading Blue band from {max_sc} scenes...", value=41)
                await asyncio.sleep(0.2)
                
                p.set(message=f"Step 6/9: 📡 Downloading NIR band from {max_sc} scenes...", value=54)
                await asyncio.sleep(0.2)
                
                p.set(message=f"Step 7/9: 🧮 Computing NDWI = (Green - NIR) / (Green + NIR) for all pixels...", value=67)
                
                # Run the actual analysis
                result = await asyncio.to_thread(analyzer.run)
                
                p.set(message=f"Step 8/9: 🌊 Applying {input.method()} thresholding & vectorizing to GeoJSON...", value=85)
                await asyncio.sleep(0.1)
                
                p.set(message=f"Step 9/9: 🖼️ Creating RGB composite images...", value=95)
                await asyncio.sleep(0.1)
                
                p.set(message="✅ Complete! Switch to Results tab to view.", value=100)
                
                # Store result with mode
                result_with_mode = {
                    'mode': input.analysis_mode(),
                    'data': result,
                    'bounds': bounds
                }
                result_data.set(result_with_mode)
                
                # Flash the Results tab green until the user clicks it
                ui.insert_ui(
                    ui.tags.script("""
                        (function() {
                            // Find the Results tab link
                            const tabs = document.querySelectorAll('[role="tab"]');
                            let resultsTab = null;
                            tabs.forEach(tab => {
                                if (tab.textContent.includes('Results')) {
                                    resultsTab = tab;
                                }
                            });

                            if (!resultsTab) return;

                            // If the tab is already active, nothing to do
                            if (resultsTab.classList.contains('active')) return;

                            // Inject flash CSS once
                            if (!document.getElementById('results-flash-style')) {
                                const style = document.createElement('style');
                                style.id = 'results-flash-style';
                                style.textContent = `
                                    @keyframes resultsFlashGreen {
                                        0% { background-color: transparent; }
                                        50% { background-color: #90EE90; }
                                        100% { background-color: transparent; }
                                    }
                                    .results-flash {
                                        animation: resultsFlashGreen 1s infinite;
                                    }
                                `;
                                document.head.appendChild(style);
                            }

                            // Avoid adding the class multiple times
                            if (!resultsTab.classList.contains('results-flash')) {
                                resultsTab.classList.add('results-flash');

                                // Remove the flash when the tab is clicked
                                const stopFlash = () => {
                                    resultsTab.classList.remove('results-flash');
                                };

                                resultsTab.addEventListener('click', function() {
                                    stopFlash();
                                }, { once: true });

                                // Also stop if the tab becomes active via another UI action
                                const observer = new MutationObserver(function(mutations) {
                                    for (const m of mutations) {
                                        if (m.attributeName === 'class' && resultsTab.classList.contains('active')) {
                                            stopFlash();
                                            observer.disconnect();
                                            break;
                                        }
                                    }
                                });
                                observer.observe(resultsTab, { attributes: true, attributeFilter: ['class'] });
                            }
                        })();
                    """),
                    selector="body",
                    where="beforeEnd"
                )
                
                # No notification - just log to console
                print("✓ Analysis complete! Results available in Results tab", file=sys.stderr)
                
            except Exception as e:
                ui.notification_show(f"❌ Analysis failed: {str(e)}", type="error", duration=8)
                print(f"Error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
    
    @output
    @render.ui
    def results_ui():
        """Render results panel"""
        result = result_data.get()
        
        # Access basemap inputs to make this function reactive to them
        # This ensures the UI re-renders when basemap selection changes
        try:
            _ = input.results_basemap_snapshot()
        except:
            pass
        try:
            _ = input.results_basemap_change()
        except:
            pass
        
        if result is None:
            return ui.div(
                ui.tags.div(
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
        bounds = result.get('bounds', {})
        
        # Calculate center
        center_lat = (bounds.get('min_lat', 29.24) + bounds.get('max_lat', 29.24)) / 2
        center_lon = (bounds.get('min_lon', -90.06) + bounds.get('max_lon', -90.06)) / 2
        
        # Get basemap selection
        basemap_choice = 'satellite'
        try:
            basemap_choice = input.results_basemap_snapshot()
        except:
            pass
        
        # Define basemap tiles
        if basemap_choice == 'grayscale':
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
            attr = 'Esri Grayscale'
        else:  # satellite
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
            attr = 'Esri Satellite'
        
        # Create Folium map
        m = folium.Map(
            location=(center_lat, center_lon),
            zoom_start=14,
            tiles=tiles,
            attr=attr
        )
        
        # Fit map to bounding box
        if bounds:
            sw = [bounds['min_lat'], bounds['min_lon']]
            ne = [bounds['max_lat'], bounds['max_lon']]
            m.fit_bounds([sw, ne], padding=[30, 30])
        
        # Add RGB composite layer as raster overlay
        if 'rgb_image' in data and data['rgb_image']:
            try:
                folium.raster_layers.ImageOverlay(
                    image=data['rgb_image'],  # Already in data:image/png;base64,... format
                    bounds=[[bounds['min_lat'], bounds['min_lon']], 
                            [bounds['max_lat'], bounds['max_lon']]],
                    opacity=0.7,
                    name='🛰️ RGB Satellite Image',
                    interactive=False,
                    cross_origin=False,
                    show=False  # Hidden by default, user can toggle via layer control
                ).add_to(m)
                print("✓ Added RGB composite overlay to map", file=sys.stderr)
            except Exception as e:
                print(f"✗ Error adding RGB overlay: {e}", file=sys.stderr)
        
        # Add water layer
        if 'water' in data and data['water'].get('features'):
            folium.GeoJson(
                data['water'],
                name='💧 Water Bodies',
                style_function=lambda x: {
                    'fillColor': '#3399ff',
                    'color': '#3399ff',
                    'weight': 2,
                    'fillOpacity': 0.5
                },
                show=True
            ).add_to(m)
            print(f"Added water layer: {len(data['water']['features'])} features", file=sys.stderr)
        
        # Add shoreline layer
        if 'shoreline' in data and data['shoreline'].get('features'):
            folium.GeoJson(
                data['shoreline'],
                name='🌊 Shoreline',
                style_function=lambda x: {
                    'color': '#FFD700',
                    'weight': 5,
                    'opacity': 1.0
                },
                show=True
            ).add_to(m)
            print(f"Added shoreline layer: {len(data['shoreline']['features'])} features", file=sys.stderr)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        
        map_html = m._repr_html_()
        
        return ui.div(
            ui.h3("📸 Snapshot Analysis Results"),
            ui.hr(),
            
            # Add basemap selector for results
            ui.card(
                ui.card_header("🗺️ Map Display Options"),
                ui.row(
                    ui.column(6,
                        ui.input_select("results_basemap_snapshot", "Basemap Style",
                                      choices={
                                          "satellite": "🛰️ Satellite Imagery",
                                          "grayscale": "⚪ Grayscale (High Contrast)"
                                      },
                                      selected=basemap_choice),
                    ),
                    ui.column(6,
                        ui.p("Use grayscale to see colored layers more clearly", class_="text-muted small mt-2"),
                    ),
                ),
                class_="mb-3"
            ),
            
            ui.row(
                ui.column(6,
                    ui.card(
                        ui.card_header("📡 Data Source"),
                        ui.HTML(f"<p><strong>Satellite scenes used:</strong> {data.get('item_id', 'N/A')}</p>"),
                        ui.HTML(f"<p><strong>Average cloud cover:</strong> {data.get('cloud_cover', 0):.1f}%</p>"),
                        ui.HTML(f"<p><strong>Detection method:</strong> {data.get('method', 'N/A').title()}</p>"),
                        ui.tags.small("Multiple scenes provide more reliable results by combining observations.", class_="text-muted")
                    )
                ),
                ui.column(6,
                    ui.card(
                        ui.card_header("🗺️ Detected Features"),
                        ui.HTML(f"<p><strong>Water polygons:</strong> {len(data.get('water', {}).get('features', []))}</p>"),
                        ui.HTML(f"<p><strong>Shoreline segments:</strong> {len(data.get('shoreline', {}).get('features', []))}</p>"),
                        ui.HTML(f"<p><strong>Total water area:</strong> {data.get('water_area_km2', 0):.2f} km²</p>"),
                    )
                ),
            ),
            
            ui.hr(),
            ui.h4("Results Map"),
            ui.p("Use the layer control (📋 icon) in the top-right of the map to toggle layer visibility.", 
                 class_="text-muted small"),
            
            ui.HTML(map_html),
        )
    
    def render_change_results(result):
        """Render change detection results"""
        data = result['data']
        bounds = result.get('bounds', {})
        
        # Calculate center
        center_lat = (bounds.get('min_lat', 29.24) + bounds.get('max_lat', 29.24)) / 2
        center_lon = (bounds.get('min_lon', -90.06) + bounds.get('max_lon', -90.06)) / 2
        
        # Get basemap selection
        basemap_choice = 'satellite'
        try:
            basemap_choice = input.results_basemap_change()
        except:
            pass
        
        # Define basemap tiles
        if basemap_choice == 'grayscale':
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
            attr = 'Esri Grayscale'
        else:  # satellite
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
            attr = 'Esri Satellite'
        
        # Create Folium map
        m = folium.Map(
            location=(center_lat, center_lon),
            zoom_start=14,
            tiles=tiles,
            attr=attr
        )
        
        # Fit map to bounding box
        if bounds:
            sw = [bounds['min_lat'], bounds['min_lon']]
            ne = [bounds['max_lat'], bounds['max_lon']]
            m.fit_bounds([sw, ne], padding=[30, 30])
        
        # Add RGB composite overlays for both time periods
        if 'now' in data and 'rgb_image' in data['now'] and data['now']['rgb_image']:
            try:
                folium.raster_layers.ImageOverlay(
                    image=data['now']['rgb_image'],
                    bounds=[[bounds['min_lat'], bounds['min_lon']], 
                            [bounds['max_lat'], bounds['max_lon']]],
                    opacity=0.7,
                    name='🛰️ Current Period RGB',
                    interactive=False,
                    cross_origin=False,
                    show=False  # Hidden by default
                ).add_to(m)
                print("✓ Added current period RGB overlay to map", file=sys.stderr)
            except Exception as e:
                print(f"✗ Error adding current RGB overlay: {e}", file=sys.stderr)
        
        if 'then' in data and 'rgb_image' in data['then'] and data['then']['rgb_image']:
            try:
                folium.raster_layers.ImageOverlay(
                    image=data['then']['rgb_image'],
                    bounds=[[bounds['min_lat'], bounds['min_lon']], 
                            [bounds['max_lat'], bounds['max_lon']]],
                    opacity=0.7,
                    name='🛰️ Historical Period RGB',
                    interactive=False,
                    cross_origin=False,
                    show=False  # Hidden by default
                ).add_to(m)
                print("✓ Added historical period RGB overlay to map", file=sys.stderr)
            except Exception as e:
                print(f"✗ Error adding historical RGB overlay: {e}", file=sys.stderr)
        
        # Add progradation layer
        if 'progradation' in data and data['progradation'].get('features'):
            folium.GeoJson(
                data['progradation'],
                name='🟢 Land Gained (Progradation)',
                style_function=lambda x: {
                    'fillColor': '#00ff00',
                    'color': '#00ff00',
                    'weight': 2,
                    'fillOpacity': 0.6
                },
                show=True
            ).add_to(m)
        
        # Add retreat layer
        if 'retreat' in data and data['retreat'].get('features'):
            folium.GeoJson(
                data['retreat'],
                name='🔴 Land Lost (Retreat)',
                style_function=lambda x: {
                    'fillColor': '#ff0000',
                    'color': '#ff0000',
                    'weight': 2,
                    'fillOpacity': 0.6
                },
                show=True
            ).add_to(m)
        
        # Add current water
        if 'now' in data and 'water' in data['now'] and data['now']['water'].get('features'):
            folium.GeoJson(
                data['now']['water'],
                name='💙 Current Water',
                style_function=lambda x: {
                    'fillColor': '#0099ff',
                    'color': '#0099ff',
                    'weight': 2,
                    'fillOpacity': 0.4
                },
                show=False
            ).add_to(m)
        
        # Add past water
        if 'then' in data and 'water' in data['then'] and data['then']['water'].get('features'):
            folium.GeoJson(
                data['then']['water'],
                name='🩷 Past Water',
                style_function=lambda x: {
                    'fillColor': '#ff6666',
                    'color': '#ff6666',
                    'weight': 2,
                    'fillOpacity': 0.4
                },
                show=False
            ).add_to(m)
        
        # Add current shoreline
        if 'now' in data and 'shoreline' in data['now'] and data['now']['shoreline'].get('features'):
            folium.GeoJson(
                data['now']['shoreline'],
                name='🔵 Current Shoreline',
                style_function=lambda x: {
                    'color': '#00FFFF',
                    'weight': 5,
                    'opacity': 1.0
                },
                show=True
            ).add_to(m)
        
        # Add past shoreline
        if 'then' in data and 'shoreline' in data['then'] and data['then']['shoreline'].get('features'):
            folium.GeoJson(
                data['then']['shoreline'],
                name='🟣 Past Shoreline',
                style_function=lambda x: {
                    'color': '#FF00FF',
                    'weight': 5,
                    'opacity': 1.0,
                    'dashArray': '10, 5'
                },
                show=True
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        
        map_html = m._repr_html_()
        
        return ui.div(
            ui.h3("🔄 Change Detection Results"),
            ui.hr(),
            
            # Add basemap selector for results
            ui.card(
                ui.card_header("🗺️ Map Display Options"),
                ui.row(
                    ui.column(6,
                        ui.input_select("results_basemap_change", "Basemap Style",
                                      choices={
                                          "satellite": "🛰️ Satellite Imagery",
                                          "grayscale": "⚪ Grayscale (High Contrast)"
                                      },
                                      selected=basemap_choice),
                    ),
                    ui.column(6,
                        ui.p("Use grayscale to see change layers (green/red) more clearly", class_="text-muted small mt-2"),
                    ),
                ),
                class_="mb-3"
            ),
            
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
            
            # Add scene information
            ui.card(
                ui.card_header("📡 Data Sources"),
                ui.row(
                    ui.column(6,
                        ui.HTML(f"<p><strong>Current period:</strong> {data.get('now', {}).get('item_id', 'N/A')}</p>"),
                        ui.HTML(f"<p class='mb-0'><strong>Cloud cover:</strong> {data.get('now', {}).get('cloud_cover', 0):.1f}%</p>"),
                    ),
                    ui.column(6,
                        ui.HTML(f"<p><strong>Historical period:</strong> {data.get('then', {}).get('item_id', 'N/A')}</p>"),
                        ui.HTML(f"<p class='mb-0'><strong>Cloud cover:</strong> {data.get('then', {}).get('cloud_cover', 0):.1f}%</p>"),
                    ),
                ),
                ui.tags.hr(),
                ui.tags.small("Multiple scenes per period improve accuracy by combining observations and avoiding cloud cover.", 
                             class_="text-muted"),
                class_="mb-3"
            ),
            
            ui.hr(),
            ui.h4("Change Detection Map"),
            ui.p("Use the layer control (📋 icon) in the top-right of the map to toggle layer visibility.", 
                 class_="text-muted small"),
            
            ui.HTML(map_html),
        )


app = App(app_ui, server)
