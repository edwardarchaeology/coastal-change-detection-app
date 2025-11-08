# Coastal Change Monitor — Shiny for Python version
# Much faster than Streamlit due to reactive programming model

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import ipyleaflet as L
from datetime import date, timedelta
import json
import numpy as np
from typing import Dict, Optional, Tuple, List
import requests

# Import all the core processing functions from the original app
import rasterio
from rasterio.session import AWSSession
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.crs import CRS
from shapely.geometry import shape, mapping, LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from pystac_client import Client

try:
    import planetary_computer as pc
    MPC_AVAILABLE = True
except:
    MPC_AVAILABLE = False

# Copy all the helper functions from app.py
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
E84_STAC = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
SCL_BAD = {0, 1, 3, 7, 8, 9, 10, 11}

# [Include all your processing functions here - compute_ndwi, ndwi_to_watermask, etc.]
# For brevity, I'll show the structure and you can paste them in

def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = green + nir
    denom[denom == 0] = np.nan
    ndwi = (green - nir) / denom
    return np.clip(ndwi, -1.0, 1.0)

def compute_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    denom = green + swir
    denom[denom == 0] = np.nan
    mndwi = (green - swir) / denom
    return np.clip(mndwi, -1.0, 1.0)

def compute_awei(green: np.ndarray, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    awei = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    return awei

# ... [include all other helper functions]

# UI Definition
app_ui = ui.page_fluid(
    ui.h2("🌊 Coastal Change Monitor — Real Sentinel-2"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Search"),
            ui.input_text("location", "Find a place", value="Grand Isle, Louisiana"),
            ui.input_action_button("geocode", "🔎 Geocode"),
            ui.hr(),
            
            ui.h3("Acquisition"),
            ui.input_date("start_date", "Start date", value=date.today() - timedelta(days=30)),
            ui.input_date("end_date", "End date", value=date.today()),
            ui.input_slider("cloud_cover", "Max cloud cover (%)", 0, 100, 30, step=5),
            ui.input_select("provider", "Data provider", 
                          choices={"mpc": "Microsoft Planetary Computer", "e84": "Element84 Earth Search"}),
            ui.input_select("basemap", "Basemap",
                          choices={"satellite": "Satellite", "light": "Light", "terrain": "Terrain"}),
            ui.hr(),
            
            ui.h3("Detection Settings"),
            ui.input_select("detection_method", "Detection Method",
                          choices={
                              "fixed": "NDWI + Fixed Threshold",
                              "otsu": "NDWI + Otsu (Auto)",
                              "adaptive": "NDWI + Adaptive",
                              "multi": "Multi-Index Consensus"
                          }),
            ui.panel_conditional(
                "input.detection_method === 'fixed'",
                ui.input_slider("ndwi_threshold", "NDWI Water Threshold", -0.3, 0.3, 0.0, step=0.05)
            ),
            ui.panel_conditional(
                "input.detection_method === 'multi'",
                ui.input_slider("consensus_votes", "Consensus Threshold", 1, 3, 2)
            ),
            
            ui.h4("Post-processing"),
            ui.input_checkbox("apply_morphology", "Refine water mask", value=True),
            ui.panel_conditional(
                "input.apply_morphology",
                ui.input_slider("morph_kernel", "Refinement strength", 2, 7, 3)
            ),
            ui.input_checkbox("smooth_shoreline", "Smooth shoreline", value=True),
            ui.panel_conditional(
                "input.smooth_shoreline",
                ui.input_slider("smooth_tolerance", "Smoothing tolerance (m)", 0.0, 10.0, 2.0, step=0.5)
            ),
            ui.hr(),
            
            ui.input_checkbox("compare_mode", "Compare with earlier window", value=False),
            ui.panel_conditional(
                "input.compare_mode",
                ui.input_slider("back_days", "Earlier window offset (days)", 30, 730, 365, step=15)
            ),
            
            width=300
        ),
        
        # Main panel
        ui.navset_card_tab(
            ui.nav_panel("Map",
                output_widget("map"),
                ui.input_action_button("run_snapshot", "📸 Snapshot: Current Water/Shoreline", class_="btn-primary"),
                ui.input_action_button("run_change", "🔄 Change Detection", class_="btn-success"),
            ),
            ui.nav_panel("Results",
                ui.output_ui("results_panel")
            ),
            ui.nav_panel("Help",
                ui.markdown("""
                ### How to Use This App:
                
                **Single Window Analysis:**
                - Pan/zoom the map to your area of interest
                - Select date range and parameters
                - Click "Snapshot" to analyze current shoreline
                
                **Change Detection:**
                1. Enable "Compare with earlier window"
                2. Choose how far back to compare
                3. Click "Change Detection" button
                4. View erosion (red) and accretion (green) areas
                
                **Tips:**
                - Keep AOI small for faster processing
                - Lower cloud cover % for clearer results
                - Sentinel-2 has 10m resolution (~20-30m reliable detection)
                """)
            )
        )
    )
)

# Server logic
def server(input, output, session):
    # Reactive values
    current_bounds = reactive.Value(None)
    analysis_result = reactive.Value(None)
    
    @render_widget
    def map():
        """Create and return the leaflet map"""
        m = L.Map(
            center=(29.24, -90.06),
            zoom=13,
            scroll_wheel_zoom=True
        )
        
        # Add basemap
        if input.basemap() == "satellite":
            L.TileLayer(
                url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attribution='Esri'
            ).add_to(m)
        elif input.basemap() == "light":
            L.TileLayer(
                url='https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png',
                attribution='CartoDB'
            ).add_to(m)
        else:
            L.TileLayer(
                url='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
                attribution='Stamen'
            ).add_to(m)
        
        # Add draw control to capture bounds
        draw_control = L.DrawControl(
            rectangle={'shapeOptions': {'color': '#0066cc'}},
            polyline={},
            polygon={},
            circle={},
            marker={},
            circlemarker={}
        )
        draw_control.add_to(m)
        
        # Store bounds when rectangle is drawn
        def handle_draw(self, action, geo_json):
            if action == 'created':
                coords = geo_json['geometry']['coordinates'][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                current_bounds.set({
                    'min_lon': min(lons),
                    'max_lon': max(lons),
                    'min_lat': min(lats),
                    'max_lat': max(lats)
                })
        
        draw_control.on_draw(handle_draw)
        
        return m
    
    @reactive.Effect
    @reactive.event(input.geocode)
    def handle_geocode():
        """Geocode location and update map center"""
        query = input.location()
        if not query:
            return
        
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "coastal-monitor-shiny/1.0"}
            )
            data = r.json()
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                # Update map center (you'll need to implement map.center update)
                ui.notification_show(f"Found: {query}", type="message")
        except Exception as e:
            ui.notification_show(f"Geocoding failed: {e}", type="error")
    
    @reactive.Effect
    @reactive.event(input.run_snapshot)
    async def run_snapshot_analysis():
        """Run single window analysis"""
        if current_bounds.get() is None:
            ui.notification_show("Please draw a rectangle on the map to set your area of interest", type="warning")
            return
        
        with ui.Progress(min=0, max=100) as p:
            p.set(message="Processing Sentinel-2 data...", value=10)
            
            try:
                bbox = current_bounds.get()
                
                # Get parameters
                use_mpc = input.provider() == "mpc"
                detection = input.detection_method()
                
                p.set(value=30, message="Searching for scenes...")
                
                # Run analysis (you'll need to paste in all the processing functions)
                # result = run_once(bbox, input.start_date(), input.end_date(), ...)
                
                p.set(value=100, message="Complete!")
                
                # Store result
                # analysis_result.set({"mode": "single", "data": result})
                
                ui.notification_show("Analysis complete!", type="message")
                
            except Exception as e:
                ui.notification_show(f"Analysis failed: {e}", type="error")
    
    @reactive.Effect
    @reactive.event(input.run_change)
    async def run_change_analysis():
        """Run change detection analysis"""
        if not input.compare_mode():
            ui.notification_show("Enable 'Compare with earlier window' first", type="warning")
            return
        
        if current_bounds.get() is None:
            ui.notification_show("Please draw a rectangle on the map", type="warning")
            return
        
        with ui.Progress(min=0, max=100) as p:
            p.set(message="Processing change detection...", value=10)
            
            try:
                bbox = current_bounds.get()
                
                p.set(value=30, message="Analyzing current period...")
                # Run current period analysis
                
                p.set(value=60, message="Analyzing historical period...")
                # Run historical period analysis
                
                p.set(value=90, message="Computing changes...")
                # Compute differences
                
                p.set(value=100, message="Complete!")
                
                ui.notification_show("Change detection complete!", type="message")
                
            except Exception as e:
                ui.notification_show(f"Change detection failed: {e}", type="error")
    
    @output
    @render.ui
    def results_panel():
        """Render results panel"""
        result = analysis_result.get()
        
        if result is None:
            return ui.div(
                ui.p("No results yet. Run an analysis from the Map tab."),
                class_="text-muted"
            )
        
        if result["mode"] == "single":
            data = result["data"]
            return ui.div(
                ui.h3("Single Window Analysis Results"),
                ui.p(f"Scene: {data.get('item_id', 'N/A')}"),
                ui.p(f"Cloud cover: {data.get('cloud_cover', 0):.1f}%"),
                # Add more result display here
            )
        else:
            # Change detection results
            return ui.div(
                ui.h3("Change Detection Results"),
                # Add change detection display here
            )

app = App(app_ui, server)
