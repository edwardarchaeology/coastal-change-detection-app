# Coastal Change Monitor — REAL DATA interactive app
# - Search a place (Nominatim / OSM)
# - Pan/zoom the map (Leaflet via streamlit-folium)
# - Click "Use current view → Run analysis" to:
#     * query Sentinel-2 L2A via STAC
#     * read COGs (B03, B08 [+ SCL if present])
#     * compute NDWI -> water mask -> shoreline (vector)
#     * (optional) compare against an earlier date to show change polygons
#
# Notes:
# - Default provider: Microsoft Planetary Computer (MPC) STAC (no API key needed)
#   and links are signed automatically.
# - Fallback provider: Element84 Earth Search (public HTTPS COGs).
# - Keep your AOI small (think city/island scale). Big bboxes = big reads.
# - Works on Windows; if you need a proxy, set HTTP(S)_PROXY env vars.

import io
import json
import math
import os
from datetime import date, timedelta
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
import folium

import rasterio
from rasterio.session import AWSSession
from rasterio.windows import from_bounds
from rasterio.features import shapes
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling as WarpResampling
from rasterio.crs import CRS

from shapely.geometry import shape, mapping, LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely import wkt

# STAC clients
from pystac_client import Client

# Planetary Computer signing (safe to import even if you don't use MPC)
try:
    import planetary_computer as pc
    MPC_AVAILABLE = True
except Exception:
    MPC_AVAILABLE = False


# ---------------------------- Config ----------------------------
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
E84_STAC = "https://earth-search.aws.element84.com/v1"  # fallback

S2_COLLECTION = "sentinel-2-l2a"
DEFAULT_CLOUD = 30  # %
DEFAULT_MAX_ITEMS = 12

# SCL (Scene Classification Layer) "cloudy" classes to mask out
# 0: No data, 1: Saturated/defective, 2: Dark area, 3: Cloud shadows,
# 4: Vegetation, 5: Bare, 6: Water, 7: Unclassified, 8: Cloud medium prob,
# 9: Cloud high prob, 10: Thin cirrus, 11: Snow/ice
# Note: We include 7 (unclassified) as it often contains thin clouds/haze
SCL_BAD = {0, 1, 3, 7, 8, 9, 10, 11}

# ------------------------ Streamlit UI --------------------------
st.set_page_config(page_title="Coastal Change Monitor — Real Sentinel-2", layout="wide")
st.title("🌊 Coastal Change Monitor — Real Sentinel-2 (NDWI)")

st.markdown("""
### How to Use This App:

**Single Window Analysis** (Current water/shoreline):
- Shows water mask (blue polygons) and shoreline (blue lines) for a specific time period
- Useful for current coastline mapping

**Change Detection** (Coastal erosion/accretion):
1. ✅ Check **"Compare with earlier window"** in the sidebar
2. Select your current time period (Start/End date)
3. Choose how far back to compare (e.g., 365 days = 1 year ago)
4. Click **"Run REAL analysis + earlier window (change)"**
5. Results show:
   - 🟢 **Green areas** = Progradation (water gained/land built)
   - 🔴 **Red areas** = Retreat (water lost/erosion)
   - Blue/Red lines = Current vs. historical shorelines

**Tips:**
- Keep view zoomed to your area of interest for faster processing
- Lower cloud cover % for clearer results
- Larger time gaps (6-12 months) show more visible changes
""")

st.divider()

with st.sidebar:
    st.header("Search")
    q = st.text_input("Find a place (city, island, landmark…)", value="Grand Isle, Louisiana")
    c1, c2 = st.columns([1,1])
    with c1:
        do_geocode = st.button("🔎 Geocode")
    with c2:
        reset_view = st.button("↺ Reset")

    st.divider()
    st.header("Acquisition")
    today = date.today()
    start = st.date_input("Start date", value=today - timedelta(days=30))
    end   = st.date_input("End date", value=today)
    cloud = st.slider("Max cloud cover (%)", 0, 100, DEFAULT_CLOUD, 5)

    provider = st.selectbox("Data provider", ["Microsoft Planetary Computer (default)", "Element84 Earth Search"])
    basemap_choice = st.selectbox("Basemap", ["Satellite (Esri WorldImagery)", "CartoDB Positron (light)", "Stamen Terrain"], index=0)
    
    st.divider()
    st.header("Detection Settings")
    
    # Algorithm selection
    detection_method = st.selectbox(
        "Detection Method",
        ["NDWI + Fixed Threshold", "NDWI + Otsu (Auto)", "NDWI + Adaptive", "Multi-Index Consensus"],
        help="""
        **Fixed:** Fast, manual tuning, works in most conditions
        **Otsu:** Auto-threshold per scene, better for variable turbidity
        **Adaptive:** Local thresholding, best for complex mixed scenes
        **Multi-Index:** Uses NDWI+MNDWI+AWEI consensus (slower, most robust for challenging conditions)
        """
    )
    
    # Show threshold slider only for fixed method
    if detection_method == "NDWI + Fixed Threshold":
        ndwi_threshold = st.slider(
            "NDWI Water Threshold", 
            min_value=-0.3, 
            max_value=0.3, 
            value=0.0, 
            step=0.05,
            help="Lower values detect more water (useful for shallow/turbid water near shore). Higher values are more conservative."
        )
    else:
        ndwi_threshold = 0.0  # Default, not used for other methods
    
    # Consensus votes slider for multi-index method
    if detection_method == "Multi-Index Consensus":
        consensus_votes = st.slider(
            "Consensus Threshold",
            min_value=1,
            max_value=3,
            value=2,
            help="Number of indices (NDWI, MNDWI, AWEI) that must agree a pixel is water. Higher = more conservative."
        )
    else:
        consensus_votes = 2  # Default value for other methods
    
    # Post-processing options
    st.subheader("Post-processing")
    apply_morphology = st.checkbox(
        "Refine water mask (morphology)",
        value=True,
        help="Remove noise and fill small gaps using mathematical morphology"
    )
    
    if apply_morphology:
        morph_kernel_size = st.slider(
            "Refinement strength",
            min_value=2,
            max_value=7,
            value=3,
            step=1,
            help="Larger values = more aggressive smoothing (removes more noise but may lose detail)"
        )
    else:
        morph_kernel_size = 3
    
    smooth_shoreline = st.checkbox(
        "Smooth shoreline vectors",
        value=True,
        help="Simplify shoreline using Douglas-Peucker algorithm for cleaner results"
    )
    
    if smooth_shoreline:
        smooth_tolerance = st.slider(
            "Smoothing tolerance (meters)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Larger values = smoother lines (but may lose small features)"
        )
    else:
        smooth_tolerance = 0.0
    
    st.caption("💡 **Tip:** Start with default settings. If results are noisy, increase refinement strength.")
    st.caption("📏 **Resolution:** Sentinel-2 has 10m pixel resolution. Changes smaller than ~20-30m may not be reliably detected.")
    
    st.divider()
    compare_mode = st.checkbox("Compare with earlier window (change polygons)", value=False)
    if compare_mode:
        back_days = st.slider("Earlier window offset (days before current window start)", 30, 730, 365, 15)
        # Calculate window duration
        window_duration = (end - start).days
        # Historical window: same duration, offset back in time
        historical_start = start - timedelta(days=back_days)
        historical_end = historical_start + timedelta(days=window_duration)
        st.caption(f"📅 Current period: {start.strftime('%Y/%m/%d')} to {end.strftime('%Y/%m/%d')} ({window_duration} days)")
        st.caption(f"📅 Historical period: {historical_start.strftime('%Y/%m/%d')} to {historical_end.strftime('%Y/%m/%d')} ({window_duration} days, offset by {back_days} days)")
        st.caption("Both windows have the same duration - only the time offset differs")

    st.caption("Tip: Keep the map view tight around the coast you care about for faster runs.")

# Session state for map & results
if "center" not in st.session_state:
    st.session_state.center = {"lat": 29.24, "lon": -90.06, "zoom": 13}
if "bounds" not in st.session_state:
    st.session_state.bounds = None
if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------- Helpers (geocode, bbox) ----------------
def geocode_osm(query: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) via Nominatim."""
    if not query.strip():
        return None
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "coastal-monitor-real/1.0"}
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None

def bounds_from_leaflet(b: Dict) -> Dict:
    """Leaflet bounds → bbox dict (lon/lat)."""
    sw = b["_southWest"]; ne = b["_northEast"]
    min_lon, min_lat = sw["lng"], sw["lat"]
    max_lon, max_lat = ne["lng"], ne["lat"]
    if min_lon > max_lon: min_lon, max_lon = max_lon, min_lon
    if min_lat > max_lat: min_lat, max_lat = max_lat, min_lat
    return {"min_lon": min_lon, "min_lat": min_lat, "max_lon": max_lon, "max_lat": max_lat}

def bbox_to_wkt(b: Dict) -> str:
    return f"POLYGON(({b['min_lon']} {b['min_lat']}, {b['max_lon']} {b['min_lat']}, {b['max_lon']} {b['max_lat']}, {b['min_lon']} {b['max_lat']}, {b['min_lon']} {b['min_lat']}))"

# ---------------------- STAC search & assets -------------------
def stac_search(bbox: Dict, start_date: date, end_date: date, max_cc: int, use_mpc=True):
    """Search S2 L2A with cloud cover filter; return items (most recent first)."""
    endpoint = MPC_STAC if use_mpc else E84_STAC
    client = Client.open(endpoint, modifier=pc.sign_inplace if (use_mpc and MPC_AVAILABLE) else None)
    items = client.search(
        collections=[S2_COLLECTION],
        bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
        datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        query={"eo:cloud_cover": {"lte": max_cc}},
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
        max_items=DEFAULT_MAX_ITEMS,
    ).get_all_items()
    return items

def pick_best_items_for_mosaic(items, bbox: Dict, max_scenes: int = 10) -> List[dict]:
    """
    Select multiple items that together cover the bbox, prioritizing least cloudy recent scenes.
    Returns a list of items sorted by cloud cover (best first).
    """
    from shapely.geometry import box as create_box
    
    # Create bbox geometry
    bbox_geom = create_box(bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"])
    
    # Group items by date (same day acquisitions)
    from collections import defaultdict
    items_by_date = defaultdict(list)
    for item in items:
        date_key = item.datetime.date() if hasattr(item.datetime, 'date') else item.datetime
        items_by_date[date_key].append(item)
    
    # For each date, find items that cover the bbox
    best_set = None
    best_date = None
    best_coverage = 0
    best_cloud = 100
    
    for date_key in sorted(items_by_date.keys(), reverse=True):  # Most recent first
        date_items = items_by_date[date_key]
        
        # Filter items that intersect with bbox and sort by cloud cover
        relevant_items = []
        for item in date_items:
            if item.geometry:
                item_geom = shape(item.geometry)
                if item_geom.intersects(bbox_geom):
                    cc = item.properties.get("eo:cloud_cover", 100.0)
                    relevant_items.append((cc, item))
        
        if not relevant_items:
            continue
            
        # Sort by cloud cover
        relevant_items.sort(key=lambda x: x[0])
        items_for_date = [item for cc, item in relevant_items[:max_scenes]]
        
        # Calculate coverage
        coverage_geoms = [shape(item.geometry) for item in items_for_date if item.geometry]
        if coverage_geoms:
            coverage_union = unary_union(coverage_geoms)
            coverage_pct = coverage_union.intersection(bbox_geom).area / bbox_geom.area
            avg_cloud = sum(item.properties.get("eo:cloud_cover", 100.0) for item in items_for_date) / len(items_for_date)
            
            # Prefer good coverage with low clouds
            if coverage_pct > 0.8 or (coverage_pct > best_coverage and avg_cloud <= best_cloud + 10):
                best_set = items_for_date
                best_date = date_key
                best_coverage = coverage_pct
                best_cloud = avg_cloud
                
                if coverage_pct > 0.95:  # Good enough
                    break
    
    if best_set:
        st.write(f"Using {len(best_set)} scene(s) from {best_date} (coverage: {best_coverage*100:.1f}%, avg cloud: {best_cloud:.1f}%)")
        return best_set
    
    return []

def pick_best_item(items) -> Optional[dict]:
    """Choose least-cloudy most recent item."""
    best = None
    for it in items:
        cc = it.properties.get("eo:cloud_cover", 100.0)
        if best is None:
            best = (cc, it)
        else:
            if (cc < best[0]) or (cc == best[0] and it.datetime > best[1].datetime):
                best = (cc, it)
    return None if best is None else best[1]

def get_asset_url(item, name: str, use_mpc=True) -> Optional[str]:
    """Return HTTPS URL for an asset (B03, B08, SCL) — signed if MPC."""
    if name not in item.assets:
        return None
    asset = item.assets[name]
    if use_mpc and MPC_AVAILABLE:
        # signing already happened via sign_inplace, but asset.href is fine
        return asset.href
    href = asset.href
    # Element84 usually gives HTTPS COGs; if s3://, try to construct HTTP (rare)
    if href.startswith("s3://"):
        # Try public S3 HTTP for sentinel-cogs
        href = href.replace("s3://", "https://").replace(".s3.", ".s3.")
    return href

# ---------------------- Raster reading + NDWI ------------------
def read_band_window(url: str, bbox: Dict, dst_res=10) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    """
    Read a rectangular window from a COG URL, reprojected to EPSG:4326.
    Uses rasterio.warp.reproject for robust CRS transformation.
    """
    with rasterio.Env(AWSSession() if url.startswith("s3://") else None):
        with rasterio.open(url) as src:
            # Calculate the transform and dimensions for the output in EPSG:4326
            dst_crs = CRS.from_epsg(4326)
            
            # Use the bbox to calculate bounds
            left, bottom, right, top = bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]
            
            # Calculate approximate output dimensions based on desired resolution
            # For EPSG:4326, degrees per meter varies by latitude, so we estimate
            width = int((right - left) / (dst_res / 111320))  # rough conversion
            height = int((top - bottom) / (dst_res / 111320))
            
            # Ensure minimum size
            if width < 10:
                width = 10
            if height < 10:
                height = 10
            
            # Create output array
            transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
            
            # Prepare destination array
            dst_array = np.empty((height, width), dtype=np.float32)
            
            # Reproject
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=WarpResampling.bilinear
            )
            
            # Check if data is valid
            if dst_array.size == 0 or dst_array.shape[0] == 0 or dst_array.shape[1] == 0:
                raise RuntimeError(f"Failed to read data from Sentinel-2 scene. "
                                 f"The area may not have data coverage. Bbox: {bbox}")
            
            meta = {
                "driver": "GTiff",
                "dtype": "float32",
                "nodata": None,
                "width": width,
                "height": height,
                "count": 1,
                "crs": dst_crs,
                "transform": transform
            }
            
            return dst_array, transform, meta

def mosaic_bands(items: List[dict], band_name: str, bbox: Dict, use_mpc=True) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    """
    Read and mosaic multiple Sentinel-2 scenes for a specific band.
    Returns a merged array covering the bbox.
    """
    if len(items) == 1:
        # Single scene - use normal read
        url = get_asset_url(items[0], band_name, use_mpc=use_mpc)
        if url is None:
            raise RuntimeError(f"Scene missing required band {band_name}")
        return read_band_window(url, bbox)
    
    # Multiple scenes - mosaic them
    st.write(f"Mosaicking {len(items)} scenes for band {band_name}...")
    
    # Read all scenes
    arrays = []
    transforms = []
    for idx, item in enumerate(items):
        url = get_asset_url(item, band_name, use_mpc=use_mpc)
        if url is None:
            st.warning(f"Scene {idx+1} missing band {band_name}, skipping")
            continue
        try:
            arr, trans, meta = read_band_window(url, bbox)
            arrays.append(arr)
            transforms.append(trans)
        except Exception as e:
            st.warning(f"Could not read scene {idx+1} for {band_name}: {e}")
            continue
    
    if not arrays:
        raise RuntimeError(f"No valid data for band {band_name}")
    
    if len(arrays) == 1:
        return arrays[0], transforms[0], meta
    
    # Mosaic: average where there's overlap, use available data where not
    # All arrays should have same shape since they're reprojected to same bbox
    mosaic = np.full_like(arrays[0], np.nan, dtype=np.float32)
    counts = np.zeros_like(arrays[0], dtype=np.int32)
    
    for arr in arrays:
        valid = ~np.isnan(arr) & (arr != 0)
        mosaic = np.where(valid & np.isnan(mosaic), arr, mosaic)
        mosaic = np.where(valid & ~np.isnan(mosaic), (mosaic * counts + arr) / (counts + 1), mosaic)
        counts = np.where(valid, counts + 1, counts)
    
    return mosaic, transforms[0], meta

def create_rgb_composite(items: List[dict], bbox: Dict, use_mpc=True) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    """
    Create a true-color RGB composite from Sentinel-2 B04 (Red), B03 (Green), B02 (Blue).
    Returns a 3-band array (height, width, 3) with values scaled to 0-255 for display.
    """
    st.write(f"Creating RGB composite from {len(items)} scene(s)...")
    
    # Read R, G, B bands
    r_arr, r_transform, meta = mosaic_bands(items, "B04", bbox, use_mpc=use_mpc)  # Red
    g_arr, g_transform, _ = mosaic_bands(items, "B03", bbox, use_mpc=use_mpc)     # Green
    b_arr, b_transform, _ = mosaic_bands(items, "B02", bbox, use_mpc=use_mpc)     # Blue
    
    # Stack into RGB
    rgb = np.stack([r_arr, g_arr, b_arr], axis=-1)
    
    # Enhance contrast and scale to 0-255
    # Sentinel-2 L2A is in reflectance (0-10000), we'll scale and enhance
    rgb = np.clip(rgb / 3000.0, 0, 1)  # Adjust divisor for brightness
    rgb = np.power(rgb, 0.8)  # Gamma correction for better visual contrast
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb, r_transform, meta

def rgb_to_base64_png(rgb_array: np.ndarray) -> str:
    """Convert RGB array to base64-encoded PNG for folium ImageOverlay."""
    from PIL import Image
    import base64
    
    # Handle NaN values
    mask = np.any(np.isnan(rgb_array), axis=-1)
    rgb_clean = rgb_array.copy()
    rgb_clean[mask] = 0  # Set NaN pixels to black
    
    # Create PIL image
    img = Image.fromarray(rgb_clean, mode='RGB')
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    b64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Water Index."""
    denom = green + nir
    denom[denom == 0] = np.nan
    ndwi = (green - nir) / denom
    # Clip to sane range
    ndwi = np.clip(ndwi, -1.0, 1.0)
    return ndwi

def compute_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Compute Modified Normalized Difference Water Index (better for turbid/coastal water)."""
    denom = green + swir
    denom[denom == 0] = np.nan
    mndwi = (green - swir) / denom
    return np.clip(mndwi, -1.0, 1.0)

def compute_awei(green: np.ndarray, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Compute Automated Water Extraction Index (suppresses non-water pixels)."""
    awei = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    return awei

def mask_clouds_with_scl(scl: np.ndarray) -> np.ndarray:
    """
    Build a boolean mask where True = good (not cloudy).
    If no SCL provided, return all True.
    SCL should be read at same transform/shape as bands.
    
    This function applies morphological dilation to expand the cloud mask
    by a few pixels to catch cloud edges and reduce false positives.
    """
    if scl is None:
        return None
    
    # Identify bad pixels (clouds, shadows, etc.)
    bad = np.isin(scl.astype(np.uint8), list(SCL_BAD))
    
    # Dilate the bad mask by 2-3 pixels to catch cloud edges
    from scipy.ndimage import binary_dilation
    structure = np.ones((5, 5), dtype=bool)  # 5x5 kernel = ~2 pixel buffer
    bad_dilated = binary_dilation(bad, structure=structure, iterations=1)
    
    good = ~bad_dilated
    return good

def ndwi_to_watermask(ndwi: np.ndarray, thresh: float = 0.0, method: str = "fixed") -> np.ndarray:
    """
    Convert NDWI to binary water mask using different methods.
    
    Args:
        ndwi: NDWI array (may contain NaN for clouds/invalid pixels)
        thresh: Threshold value (used for 'fixed' method)
        method: 'fixed', 'otsu', or 'adaptive'
    """
    if method == "otsu":
        # Otsu auto-thresholding
        from skimage.filters import threshold_otsu
        valid_ndwi = ndwi[~np.isnan(ndwi)]
        if len(valid_ndwi) < 100:
            # Fallback to fixed if insufficient data
            mask = (ndwi >= thresh).astype(np.uint8)
            return mask
        try:
            otsu_thresh = threshold_otsu(valid_ndwi)
            mask = (ndwi >= otsu_thresh).astype(np.uint8)
            return mask
        except:
            mask = (ndwi >= thresh).astype(np.uint8)
            return mask
    elif method == "adaptive":
        # Adaptive local thresholding (for complex scenes)
        from skimage.filters import threshold_local
        # Replace NaN with low value for processing
        ndwi_filled = np.nan_to_num(ndwi, nan=-1.0)
        try:
            local_thresh = threshold_local(ndwi_filled, block_size=35, method='gaussian')
            mask = (ndwi_filled > local_thresh).astype(np.uint8)
            # But exclude the filled NaN areas from final mask
            mask[np.isnan(ndwi)] = 0
            return mask
        except:
            mask = (ndwi >= thresh).astype(np.uint8)
            return mask
    else:  # fixed threshold
        mask = (ndwi >= thresh).astype(np.uint8)
        return mask

def refine_water_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Clean water mask using morphological operations.
    Opening removes small noise, closing fills small gaps.
    """
    from scipy.ndimage import binary_opening, binary_closing
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    # Opening removes small isolated pixels
    mask_clean = binary_opening(mask.astype(bool), structure=kernel)
    # Closing fills small holes
    mask_clean = binary_closing(mask_clean, structure=kernel)
    return mask_clean.astype(np.uint8)

def multi_index_water_detection(b02: np.ndarray, b03: np.ndarray, b08: np.ndarray, 
                                b11: np.ndarray, b12: np.ndarray, 
                                consensus_threshold: int = 2) -> np.ndarray:
    """
    Combine multiple water indices for robust detection.
    Returns water mask where at least `consensus_threshold` indices agree.
    
    Args:
        b02: Blue band (10m)
        b03: Green band (10m)
        b08: NIR band (10m)
        b11: SWIR1 band (20m, will be resampled)
        b12: SWIR2 band (20m, will be resampled)
        consensus_threshold: Number of indices that must agree (1-3)
    """
    # NDWI
    ndwi = compute_ndwi(b03, b08)
    water_ndwi = (ndwi > 0.0).astype(int)
    
    # MNDWI (better for turbid water)
    mndwi = compute_mndwi(b03, b11)
    water_mndwi = (mndwi > 0.0).astype(int)
    
    # AWEI (suppresses non-water pixels)
    awei = compute_awei(b03, b08, b11, b12)
    water_awei = (awei > 0.0).astype(int)
    
    # Consensus voting
    vote_count = water_ndwi + water_mndwi + water_awei
    consensus = (vote_count >= consensus_threshold).astype(np.uint8)
    
    return consensus

def vectorize_mask(mask: np.ndarray, transform, bbox: Dict = None) -> List[Polygon]:
    """
    Convert raster mask to vector polygons.
    Filters out only the largest boundary-extending polygon (ocean).
    Keeps all islands regardless of boundary touching.
    
    Args:
        mask: Binary raster mask (0/1)
        transform: Rasterio affine transform
        bbox: Bounding box dict with min_lon, max_lon, min_lat, max_lat (optional)
    """
    polys = []
    for geom, val in shapes(mask, mask=None, transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        if g.area <= 0:
            continue
        polys.append(g)
    if not polys:
        return []
    # Merge small pieces
    merged = unary_union(polys)
    if isinstance(merged, (Polygon, MultiPolygon)):
        geoms = [merged] if isinstance(merged, Polygon) else list(merged.geoms)
        # Filter very small noise polygons
        min_area = 1e-7  # Very small threshold - only filter obvious noise
        geoms = [p for p in geoms if p.area > min_area]
        
        # Smart boundary filtering: 
        # - Keep the LARGEST polygon (main ocean) even if it touches edges
        # - Keep ALL other polygons (islands) regardless of boundary
        if bbox and len(geoms) > 1:
            # Create a boundary box slightly inset
            buffer = 1e-6  # Tiny buffer in degrees
            boundary_box = box(
                bbox['min_lon'] + buffer,
                bbox['min_lat'] + buffer,
                bbox['max_lon'] - buffer,
                bbox['max_lat'] - buffer
            )
            
            # Sort by area (largest first)
            geoms_sorted = sorted(geoms, key=lambda g: g.area, reverse=True)
            
            # Strategy: Keep largest polygon + all fully-contained polygons + medium-sized edge polygons
            filtered_geoms = []
            
            # Always keep the largest polygon (main water body/ocean)
            filtered_geoms.append(geoms_sorted[0])
            
            # For other polygons, use smart filtering
            for g in geoms_sorted[1:]:
                # Check if polygon is fully contained (island completely inside bbox)
                if boundary_box.contains(g):
                    # Fully contained island - definitely keep it
                    filtered_geoms.append(g)
                elif g.intersects(boundary_box):
                    # Touches boundary - could be real island or edge artifact
                    # Keep if it's not too small (real islands have meaningful size)
                    # Use relative size: if > 1% of largest polygon, likely real
                    if g.area > geoms_sorted[0].area * 0.01:
                        filtered_geoms.append(g)
                    # OR if it's moderately sized in absolute terms
                    elif g.area > 1e-6:  # ~0.01 km² at mid-latitudes
                        filtered_geoms.append(g)
            
            return filtered_geoms
        
        return geoms
    return []

def shoreline_from_polys(polys: List[Polygon], smooth_tolerance: float = 0.0) -> List[LineString]:
    """
    Extract exterior boundaries as shoreline lines.
    
    Args:
        polys: List of water polygons
        smooth_tolerance: Douglas-Peucker smoothing tolerance (0 = no smoothing)
    """
    lines = []
    for p in polys:
        line = LineString(p.exterior.coords)
        if smooth_tolerance > 0:
            line = line.simplify(smooth_tolerance, preserve_topology=True)
        lines.append(line)
    return lines

def run_once(bbox: Dict, start_date: date, end_date: date, max_cc: int, use_mpc=True,
             ndwi_thresh: float = 0.0, external_cloud_mask: np.ndarray = None,
             detection_method: str = "NDWI + Fixed Threshold",
             apply_morphology: bool = True, morph_kernel_size: int = 3,
             smooth_tolerance: float = 2.0, consensus_votes: int = 2) -> Dict:
    """
    Execute one run: find best scenes to cover bbox, compute NDWI water mask + shoreline.
    Returns dict with {item_id, cc, shoreline_geojson, water_poly_geojson, cloud_mask}.
    
    Args:
        external_cloud_mask: Optional combined cloud mask to use (e.g., union of both periods in change detection)
    """
    items = stac_search(bbox, start_date, end_date, max_cc, use_mpc=use_mpc)
    if not items:
        raise RuntimeError("No Sentinel-2 items found for that time window and AOI.")
    
    # Find best set of scenes that cover the bbox
    best_items = pick_best_items_for_mosaic(items, bbox, max_scenes=10)
    if not best_items:
        # Fallback to single best item
        st.warning("Could not find multiple scenes to cover area, using single best scene")
        best_items = [pick_best_item(items)]
        if best_items[0] is None:
            raise RuntimeError("No suitable scene found (cloud thresholds too strict?).")
    
    # Create scene ID summary
    scene_ids = [item.id for item in best_items]
    scene_summary = f"{len(best_items)} scene(s): " + ", ".join([s[:30] + "..." for s in scene_ids[:3]])
    if len(scene_ids) > 3:
        scene_summary += f" (+{len(scene_ids)-3} more)"
    
    avg_cloud = sum(item.properties.get("eo:cloud_cover", 0) for item in best_items) / len(best_items)
    
    # Log processing details in expander
    with st.expander("📋 Processing Details", expanded=False):
        st.write(f"Using {scene_summary}  — avg cloud cover: {avg_cloud:.2f}%")

    # Mosaic the bands
    g_arr, g_transform, _ = mosaic_bands(best_items, "B03", bbox, use_mpc=use_mpc)  # Green 10m
    n_arr, n_transform, _ = mosaic_bands(best_items, "B08", bbox, use_mpc=use_mpc)  # NIR 10m

    # Try to mosaic SCL (cloud mask) - optional, won't fail if missing
    scl_arr = None
    try:
        scl_arr, scl_transform, _ = mosaic_bands(best_items, "SCL", bbox, use_mpc=use_mpc)
        # Reproject SCL to match shape (already warped to EPSG:4326; just resample with nearest)
        if scl_arr.shape != g_arr.shape:
            if g_arr.shape[0] == 0 or g_arr.shape[1] == 0:
                raise RuntimeError(f"Cannot resize SCL array: target shape is invalid {g_arr.shape}")
            # Use scipy or numpy for resizing instead of rasterio.warp.resize
            from scipy.ndimage import zoom
            zoom_factors = (g_arr.shape[0] / scl_arr.shape[0], g_arr.shape[1] / scl_arr.shape[1])
            scl_arr = zoom(scl_arr, zoom_factors, order=0)  # order=0 for nearest neighbor
    except Exception as e:
        st.warning(f"Could not read SCL (cloud mask) data: {e}. Proceeding without cloud masking.")
        scl_arr = None

    # Compute NDWI
    ndwi = compute_ndwi(g_arr, n_arr)
    
    # Apply cloud masking
    if external_cloud_mask is not None:
        # Use the externally provided combined mask (for change detection)
        good_mask = external_cloud_mask
    else:
        # Generate mask from this scene's SCL data
        good_mask = mask_clouds_with_scl(scl_arr)
    
    if good_mask is not None:
        ndwi = np.where(good_mask, ndwi, np.nan)

    # Water detection based on selected method
    if detection_method == "Multi-Index Consensus":
        # Need SWIR bands (B11, B12) and Blue (B02) for multi-index
        try:
            b02_arr, _, _ = mosaic_bands(best_items, "B02", bbox, use_mpc=use_mpc)  # Blue 10m
            b11_arr, _, _ = mosaic_bands(best_items, "B11", bbox, use_mpc=use_mpc)  # SWIR1 20m
            b12_arr, _, _ = mosaic_bands(best_items, "B12", bbox, use_mpc=use_mpc)  # SWIR2 20m
            
            # Resize SWIR to match Green/NIR shape if needed
            if b11_arr.shape != g_arr.shape:
                from scipy.ndimage import zoom
                zoom_factors = (g_arr.shape[0] / b11_arr.shape[0], g_arr.shape[1] / b11_arr.shape[1])
                b11_arr = zoom(b11_arr, zoom_factors, order=1)
                b12_arr = zoom(b12_arr, zoom_factors, order=1)
            
            # Apply cloud mask to all bands before multi-index detection
            if good_mask is not None:
                b02_arr = np.where(good_mask, b02_arr, np.nan)
                g_arr_masked = np.where(good_mask, g_arr, np.nan)
                n_arr_masked = np.where(good_mask, n_arr, np.nan)
                b11_arr = np.where(good_mask, b11_arr, np.nan)
                b12_arr = np.where(good_mask, b12_arr, np.nan)
            else:
                g_arr_masked = g_arr
                n_arr_masked = n_arr
            
            water_mask = multi_index_water_detection(b02_arr, g_arr_masked, n_arr_masked, b11_arr, b12_arr, consensus_votes)
        except Exception as e:
            st.warning(f"Could not use multi-index method (missing bands?): {e}. Falling back to NDWI.")
            # Keep NaN as NaN - don't classify cloudy pixels as water
            water_mask = ndwi_to_watermask(ndwi, thresh=ndwi_thresh, method="fixed")
            # Set cloudy pixels to 0 (not water) in mask
            water_mask[np.isnan(ndwi)] = 0
    else:
        # Map method names to internal codes
        method_map = {
            "NDWI + Fixed Threshold": "fixed",
            "NDWI + Otsu (Auto)": "otsu",
            "NDWI + Adaptive": "adaptive"
        }
        method_code = method_map.get(detection_method, "fixed")
        # Keep NaN as NaN for thresholding, but exclude from final mask
        water_mask = ndwi_to_watermask(ndwi, thresh=ndwi_thresh, method=method_code)
        # Set cloudy/invalid pixels to 0 (not water) in mask
        water_mask[np.isnan(ndwi)] = 0
    
    # Apply morphological refinement if requested
    if apply_morphology:
        water_mask = refine_water_mask(water_mask, kernel_size=morph_kernel_size)

    # Vectorize (pass bbox to filter edge-touching polygons)
    polys = vectorize_mask(water_mask, g_transform, bbox=bbox)
    shores = shoreline_from_polys(polys, smooth_tolerance=smooth_tolerance)

    # Build GeoJSON outputs
    poly_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(p), "properties": {"type": "water"}} for p in polys
    ]}
    line_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(l), "properties": {"type": "shoreline"}} for l in shores
    ]}
    
    # Create RGB composite for visualization
    rgb_composite, rgb_transform, _ = create_rgb_composite(best_items, bbox, use_mpc=use_mpc)
    rgb_base64 = rgb_to_base64_png(rgb_composite)
    
    # Calculate bounds for image overlay [[south, west], [north, east]]
    bounds = [[bbox["min_lat"], bbox["min_lon"]], [bbox["max_lat"], bbox["max_lon"]]]

    return {
        "item_id": scene_summary,
        "cloud_cover": avg_cloud,
        "shoreline": line_fc,
        "water": poly_fc,
        "rgb_image": rgb_base64,
        "rgb_bounds": bounds,
        "cloud_mask": good_mask,  # Return the cloud mask for potential reuse
    }

def run_change(bbox: Dict, start_date: date, end_date: date, back_days: int, max_cc: int, use_mpc=True,
               ndwi_thresh: float = 0.0, detection_method: str = "NDWI + Fixed Threshold",
               apply_morphology: bool = True, morph_kernel_size: int = 3,
               smooth_tolerance: float = 2.0, consensus_votes: int = 2) -> Dict:
    """
    Compute water masks for current window and an earlier window, return change polygons.
    
    This function first runs both periods separately to get their cloud masks,
    then combines the masks (union) and re-runs with the combined mask to ensure
    any pixel that is cloudy in EITHER period is masked out in BOTH periods.
    This prevents false change detection from clouds.
    """
    # First pass: get cloud masks from both periods
    with st.expander("📋 Processing Details", expanded=False):
        st.write("🔍 First pass: detecting clouds in both periods...")
    now_run_initial = run_once(bbox, start_date, end_date, max_cc, use_mpc=use_mpc, 
                               ndwi_thresh=ndwi_thresh, detection_method=detection_method,
                               apply_morphology=apply_morphology, morph_kernel_size=morph_kernel_size,
                               smooth_tolerance=smooth_tolerance, consensus_votes=consensus_votes)
    
    # Calculate historical window: same duration as current window, offset back in time
    window_duration = (end_date - start_date).days
    earlier_start = start_date - timedelta(days=back_days)
    earlier_end = earlier_start + timedelta(days=window_duration)
    
    then_run_initial = run_once(bbox, earlier_start, earlier_end, max_cc, use_mpc=use_mpc, 
                                ndwi_thresh=ndwi_thresh, detection_method=detection_method,
                                apply_morphology=apply_morphology, morph_kernel_size=morph_kernel_size,
                                smooth_tolerance=smooth_tolerance, consensus_votes=consensus_votes)
    
    # Combine cloud masks: a pixel is good only if good in BOTH periods
    now_mask = now_run_initial.get("cloud_mask")
    then_mask = then_run_initial.get("cloud_mask")
    
    combined_mask = None
    if now_mask is not None and then_mask is not None:
        # Ensure same shape
        if now_mask.shape == then_mask.shape:
            combined_mask = now_mask & then_mask  # Logical AND: good only if good in both
            masked_pixels = np.sum(~combined_mask)
            total_pixels = combined_mask.size
            with st.expander("📋 Processing Details", expanded=False):
                st.write(f"✓ Combined cloud mask: {masked_pixels}/{total_pixels} pixels masked ({100*masked_pixels/total_pixels:.1f}%)")
        else:
            with st.expander("📋 Processing Details", expanded=False):
                st.warning("Cloud mask shapes don't match between periods, using individual masks")
    
    # Second pass: re-run with combined cloud mask if available
    if combined_mask is not None:
        with st.expander("📋 Processing Details", expanded=False):
            st.write("🔄 Second pass: re-analyzing with combined cloud mask...")
        now_run = run_once(bbox, start_date, end_date, max_cc, use_mpc=use_mpc, 
                          ndwi_thresh=ndwi_thresh, external_cloud_mask=combined_mask,
                          detection_method=detection_method, apply_morphology=apply_morphology,
                          morph_kernel_size=morph_kernel_size, smooth_tolerance=smooth_tolerance,
                          consensus_votes=consensus_votes)
        then_run = run_once(bbox, earlier_start, earlier_end, max_cc, use_mpc=use_mpc, 
                           ndwi_thresh=ndwi_thresh, external_cloud_mask=combined_mask,
                           detection_method=detection_method, apply_morphology=apply_morphology,
                           morph_kernel_size=morph_kernel_size, smooth_tolerance=smooth_tolerance,
                           consensus_votes=consensus_votes)
    else:
        # Use initial results if no combined mask
        now_run = now_run_initial
        then_run = then_run_initial

    # Water polygons: compare current vs earlier to detect coastal changes
    # - More water now than before = land lost = RETREAT/EROSION
    # - Less water now than before = land gained = PROGRADATION/ACCRETION
    now_polys = [shape(f["geometry"]) for f in now_run["water"]["features"]]
    then_polys = [shape(f["geometry"]) for f in then_run["water"]["features"]]
    now_union = unary_union(now_polys) if now_polys else None
    then_union = unary_union(then_polys) if then_polys else None

    prog = []  # Progradation: land gained (where water DISAPPEARED)
    ret  = []  # Retreat: land lost (where water APPEARED)
    if now_union and then_union:
        # Water that appeared = land lost = RETREAT
        new_water = now_union.difference(then_union)
        # Water that disappeared = land gained = PROGRADATION
        lost_water = then_union.difference(now_union)
        
        if isinstance(new_water, (Polygon, MultiPolygon)):
            ret = [new_water] if isinstance(new_water, Polygon) else list(new_water.geoms)
        if isinstance(lost_water, (Polygon, MultiPolygon)):
            prog = [lost_water] if isinstance(lost_water, Polygon) else list(lost_water.geoms)

    prog_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(p), "properties": {"change": "progradation"}} for p in prog
    ]}
    ret_fc  = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(p), "properties": {"change": "retreat"}} for p in ret
    ]}

    return {"now": now_run, "then": then_run, "progradation": prog_fc, "retreat": ret_fc}

# ---------------------- Map init & capture ---------------------
if do_geocode:
    pt = geocode_osm(q)
    if pt:
        st.session_state.center = {"lat": pt[0], "lon": pt[1], "zoom": 12}
    else:
        st.warning("No geocode result.")

if reset_view:
    st.session_state.center = {"lat": 29.24, "lon": -90.06, "zoom": 13}
    st.session_state.bounds = None
    st.session_state.result = None

m = folium.Map(
    location=[st.session_state.center["lat"], st.session_state.center["lon"]],
    zoom_start=st.session_state.center["zoom"],
    # Choose tiles based on sidebar selection
    tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" if basemap_choice.startswith("Satellite") else ("CartoDB Positron" if basemap_choice.startswith("CartoDB") else "Stamen Terrain")),
    attr=("Esri WorldImagery" if basemap_choice.startswith("Satellite") else None),
    control_scale=True
)
folium.Marker(
    [st.session_state.center["lat"], st.session_state.center["lon"]],
    tooltip=folium.Tooltip("Pan/zoom to set Area of Interest", permanent=False),
    icon=folium.Icon(color="blue", icon="info-sign")
).add_to(m)
map_state = st_folium(m, height=560, width="100%", returned_objects=["bounds", "zoom", "center"])

if map_state and map_state.get("bounds"):
    st.session_state.bounds = map_state["bounds"]
    # Also capture the current zoom level
    if map_state.get("zoom"):
        st.session_state.zoom = map_state["zoom"]
    if map_state.get("center"):
        st.session_state.map_center = map_state["center"]

st.caption("Pan/zoom to set AOI, choose dates & cloud limit, then run.")

# ---------------------- Run buttons ----------------------------
col_run1, col_run2 = st.columns([1,1])
with col_run1:
    run_now = st.button("📸 Snapshot: Current Water/Shoreline", help="Analyze water and shoreline for the selected time period", use_container_width=True)
with col_run2:
    run_cmp = st.button("🔄 Change Detection: Compare Two Periods", help="Compare coastline changes between current and earlier period", use_container_width=True, type="primary")

def ensure_bbox() -> Optional[Dict]:
    if not st.session_state.bounds:
        st.warning("No AOI yet — pan/zoom the map to set bounds.")
        return None
    bbox = bounds_from_leaflet(st.session_state.bounds)
    
    # Check if bbox is too small
    lon_diff = abs(bbox['max_lon'] - bbox['min_lon'])
    lat_diff = abs(bbox['max_lat'] - bbox['min_lat'])
    min_size = 0.01  # about 1km at equator
    if lon_diff < min_size or lat_diff < min_size:
        st.error(f"Selected area is too small ({lon_diff:.6f}° x {lat_diff:.6f}°). Please zoom out and select a larger area (at least {min_size}° x {min_size}°).")
        return None
    
    st.info(f"AOI bbox: lon [{bbox['min_lon']:.4f} → {bbox['max_lon']:.4f}], lat [{bbox['min_lat']:.4f} → {bbox['max_lat']:.4f}]")
    return bbox

use_mpc = (provider.startswith("Microsoft"))

# ---------------------- Execute --------------------------------
if run_now:
    bbox = ensure_bbox()
    if bbox:
        with st.status("🔄 Processing Sentinel-2 data...", expanded=False) as status:
            try:
                res = run_once(bbox, start, end, cloud, use_mpc=use_mpc, 
                              ndwi_thresh=ndwi_threshold, detection_method=detection_method,
                              apply_morphology=apply_morphology, morph_kernel_size=morph_kernel_size,
                              smooth_tolerance=smooth_tolerance, consensus_votes=consensus_votes)
                st.session_state.result = {"mode": "single", "data": res}
                status.update(label="✅ Analysis complete!", state="complete")
            except Exception as e:
                status.update(label="❌ Analysis failed", state="error")
                st.error(f"Run failed: {e}")

if run_cmp:
    if not compare_mode:
        st.warning("Enable 'Compare with earlier window' in the sidebar first.")
    else:
        bbox = ensure_bbox()
        if bbox:
            with st.status("🔄 Processing change detection...", expanded=False) as status:
                try:
                    res = run_change(bbox, start, end, back_days, cloud, use_mpc=use_mpc, 
                                   ndwi_thresh=ndwi_threshold, detection_method=detection_method,
                                   apply_morphology=apply_morphology, morph_kernel_size=morph_kernel_size,
                                   smooth_tolerance=smooth_tolerance, consensus_votes=consensus_votes)
                    st.session_state.result = {"mode": "compare", "data": res}
                    status.update(label="✅ Change detection complete!", state="complete")
                except Exception as e:
                    status.update(label="❌ Analysis failed", state="error")
                    st.error(f"Run failed: {e}")

# ---------------------- Display results ------------------------
def add_geojson_layer(m: folium.Map, fc: dict, name: str, color: str, fill: bool = False, opacity: float = 0.9):
    style_fn = (lambda f: {"color": color, "weight": 2, "opacity": opacity}) if not fill else \
               (lambda f: {"color": color, "weight": 1, "fillColor": color, "fillOpacity": 0.35, "opacity": 0.7})
    gj = folium.GeoJson(fc, name=name, style_function=style_fn)
    gj.add_to(m)

def result_map(fc_dicts: List[Tuple[str, dict, str, bool]], 
               rgb_image: str = None, rgb_bounds: list = None,
               rgb_image_current: str = None, rgb_bounds_current: list = None,
               rgb_image_past: str = None, rgb_bounds_past: list = None) -> folium.Map:
    """
    Create result map with vector layers and optional RGB imagery.
    Can display multiple RGB composites (current, past, or single).
    """
    # Calculate center from the actual bounds that were selected
    if st.session_state.bounds:
        bounds = st.session_state.bounds
        lat = (bounds["_southWest"]["lat"] + bounds["_northEast"]["lat"]) / 2
        lon = (bounds["_southWest"]["lng"] + bounds["_northEast"]["lng"]) / 2
        # Use the captured zoom from the map interaction
        zoom = st.session_state.get("zoom", st.session_state.center.get("zoom", 10))
    else:
        # Fallback to initial center
        lat = st.session_state.center.get("lat", 0)
        lon = st.session_state.center.get("lon", 0)
        zoom = st.session_state.center.get("zoom", 10)
    
    # Create map with center/zoom
    m2 = folium.Map(
        location=[lat, lon], 
        zoom_start=zoom,
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" if basemap_choice.startswith("Satellite") else ("CartoDB Positron" if basemap_choice.startswith("CartoDB") else "Stamen Terrain")),
        attr=("Esri WorldImagery" if basemap_choice.startswith("Satellite") else None),
        control_scale=True,
        max_bounds=True  # Prevent infinite world repeating
    )
    
    # Apply fit_bounds to ensure exact match with selection area
    if st.session_state.bounds:
        bounds = st.session_state.bounds
        m2.fit_bounds([
            [bounds["_southWest"]["lat"], bounds["_southWest"]["lng"]],
            [bounds["_northEast"]["lat"], bounds["_northEast"]["lng"]]
        ])
    
    # Add Sentinel-2 RGB image overlays
    # For comparison mode, add both current and past imagery
    if rgb_image_current and rgb_bounds_current:
        folium.raster_layers.ImageOverlay(
            image=rgb_image_current,
            bounds=rgb_bounds_current,
            opacity=0.7,
            name="🛰️ Current Period RGB",
            interactive=False,
            cross_origin=False,
            show=True  # Show by default
        ).add_to(m2)
    
    if rgb_image_past and rgb_bounds_past:
        folium.raster_layers.ImageOverlay(
            image=rgb_image_past,
            bounds=rgb_bounds_past,
            opacity=0.7,
            name="🛰️ Past Period RGB",
            interactive=False,
            cross_origin=False,
            show=False  # Hidden by default, user can toggle
        ).add_to(m2)
    
    # For single window mode, use the simple rgb_image parameter
    if rgb_image and rgb_bounds and not (rgb_image_current or rgb_image_past):
        folium.raster_layers.ImageOverlay(
            image=rgb_image,
            bounds=rgb_bounds,
            opacity=0.8,
            name="🛰️ Sentinel-2 RGB",
            interactive=False,
            cross_origin=False,
        ).add_to(m2)
    
    for name, fc, color, fill in fc_dicts:
        if fc and len(fc.get("features", [])) > 0:
            add_geojson_layer(m2, fc, name, color, fill)
    folium.LayerControl(collapsed=False).add_to(m2)
    return m2

if st.session_state.result:
    mode = st.session_state.result["mode"]
    data = st.session_state.result["data"]

    if mode == "single":
        st.divider()
        st.subheader("📊 Results: Single Window Analysis")
        st.info("ℹ️ **What you're seeing:** Water detection and shoreline for the selected time period. "
                "To see **coastline changes over time**, enable 'Compare with earlier window' in the sidebar and run the comparison analysis.")
        
        # Display detection method and parameters
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Detection Method:** {detection_method}")
            st.write(f"**Scene:** `{data['item_id']}`")
        with col2:
            st.write(f"**Cloud cover:** {data['cloud_cover']:.1f}%")
            # Count features
            water_count = len(data["water"].get("features", []))
            shore_count = len(data["shoreline"].get("features", []))
            st.write(f"**Detected:** {water_count} water polygon(s), {shore_count} shoreline segment(s)")
        
        # Calculate total water area
        water_area_deg2 = sum(shape(f["geometry"]).area for f in data["water"].get("features", []))
        lat_avg = (st.session_state.bounds['_southWest']['lat'] + st.session_state.bounds['_northEast']['lat']) / 2 if st.session_state.bounds else 30
        deg_to_km2 = (111.32 * np.cos(np.radians(lat_avg)) * 111.32)
        water_area_km2 = water_area_deg2 * deg_to_km2
        st.metric("💧 Total Water Area", f"{water_area_km2:.2f} km²")
        
        st.markdown("""
        **Legend:**
        - 🛰️ **Satellite image** = Actual Sentinel-2 RGB composite (what the satellite sees)
        - 🔵 **Blue lines** = Detected shoreline (water/land boundary)
        - 💧 **Light blue areas** = Water mask (detected water bodies)
        
        *Tip: Use the layer control (top right) to toggle layers on/off*
        """)
        
        mres = result_map([
            ("Shoreline", data["shoreline"], "#0066cc", False),
            ("Water mask", data["water"], "#3399ff", True),
        ], rgb_image=data.get("rgb_image"), rgb_bounds=data.get("rgb_bounds"))
        st_folium(mres, height=560, width="100%", key="single_result", returned_objects=[])

        # downloads
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download shoreline GeoJSON", data=json.dumps(data["shoreline"]), file_name="shoreline.geojson", mime="application/geo+json")
        with c2:
            st.download_button("⬇️ Download water polygons GeoJSON", data=json.dumps(data["water"]), file_name="water.geojson", mime="application/geo+json")

    elif mode == "compare":
        st.divider()
        st.subheader("📊 Results: Coastal Change Detection")
        now = data["now"]; then = data["then"]
        
        st.success("✅ **You're now viewing coastline changes!**")
        
        # Display detection method and parameters
        st.write(f"**Detection Method:** {detection_method}")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Current period scene(s):** `{now['item_id']}`")
            st.write(f"Cloud cover: {now['cloud_cover']:.1f}%")
        with col2:
            st.write(f"**Earlier period scene(s):** `{then['item_id']}`")
            st.write(f"Cloud cover: {then['cloud_cover']:.1f}%")
        
        st.markdown("""
        ### How to Interpret the Results:
        
        **🛰️ Satellite Imagery:**
        - Toggle between "Current Period RGB" and "Past Period RGB" in the layer control
        - Compare the actual satellite photos to see visual changes over time
        - Both imagery layers are included - current period visible by default
        
        **🟢 Green areas (Progradation):** 
        - Water that appeared in the recent period but wasn't there before
        - Indicates land building, sediment deposition, or coastal accretion
        - Positive change for land
        
        **🔴 Red areas (Retreat/Erosion):**
        - Water that wasn't there before but appeared in the recent period
        - Indicates coastal erosion, land loss, or submergence
        - Negative change for land
        
        **Shorelines:**
        - 🔵 **Blue line** = Current shoreline position
        - 🔴 **Dark red line** = Past shoreline position
        - Compare the two to see how the coastline has moved
        
        *Tip: Use layer control (top right) to show/hide different layers and toggle between current/past satellite imagery*
        
        ---
        """)
        
        # Single combined map with all data
        st.write("**Change detection with current and past satellite imagery**")
        st.caption(f"Current period: {now['item_id']} | Past period: {then['item_id']}")
        
        mres = result_map([
            ("Current shoreline", now["shoreline"], "#0066cc", False),
            ("Past shoreline", then["shoreline"], "#b30000", False),
            ("🟢 Progradation (land gained)", data["progradation"], "#2ca02c", True),
            ("🔴 Retreat (land lost)", data["retreat"], "#d62728", True),
        ], 
        rgb_image_current=now.get("rgb_image"), 
        rgb_bounds_current=now.get("rgb_bounds"),
        rgb_image_past=then.get("rgb_image"),
        rgb_bounds_past=then.get("rgb_bounds"))
        
        st_folium(mres, height=650, width="100%", key="change_overview", returned_objects=[])

        # quick area stats (deg^2 rough; for proper area, reproject to local UTM)
        def rough_area_deg2(fc):
            total = 0.0
            for f in fc.get("features", []):
                total += shape(f["geometry"]).area
            return total
        a_prog = rough_area_deg2(data["progradation"])
        a_ret  = rough_area_deg2(data["retreat"])
        
        # Convert to approximate km²
        # At latitude ~30°, 1 degree ≈ 111 km, so deg² ≈ 12321 km²
        lat_avg = (st.session_state.bounds['_southWest']['lat'] + st.session_state.bounds['_northEast']['lat']) / 2 if st.session_state.bounds else 30
        deg_to_km2 = (111.32 * np.cos(np.radians(lat_avg)) * 111.32)
        
        st.write(f"**Area change (approximate):**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 Progradation", f"{a_prog * deg_to_km2:.3f} km²", delta="Land gained", delta_color="normal")
        with col2:
            st.metric("🔴 Retreat", f"{a_ret * deg_to_km2:.3f} km²", delta="Land lost", delta_color="inverse")
        with col3:
            net = (a_prog - a_ret) * deg_to_km2
            if net >= 0:
                st.metric("Net Change", f"{net:+.3f} km²", delta="Net gain", delta_color="normal")
            else:
                st.metric("Net Change", f"{net:+.3f} km²", delta="Net loss", delta_color="inverse")
        
        st.caption(f"_Note: Areas in deg²: progradation {a_prog:.6f}, retreat {a_ret:.6f}. For precise measurements, reproject to local UTM coordinate system._")

        # downloads
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("⬇️ Current shoreline", data=json.dumps(now["shoreline"]), file_name="current_shoreline.geojson", mime="application/geo+json")
        with c2:
            st.download_button("⬇️ Past shoreline", data=json.dumps(then["shoreline"]), file_name="past_shoreline.geojson", mime="application/geo+json")
        with c3:
            st.download_button("⬇️ Progradation polygons", data=json.dumps(data["progradation"]), file_name="progradation.geojson", mime="application/geo+json")
        with c4:
            st.download_button("⬇️ Retreat polygons", data=json.dumps(data["retreat"]), file_name="retreat.geojson", mime="application/geo+json")
else:
    st.info("Use the search box, pan/zoom to your AOI, then run one of the REAL analysis buttons.")
