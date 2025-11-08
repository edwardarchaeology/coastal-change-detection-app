# coastal_processor.py
# Core processing functions for coastal analysis
# Separated from UI for better performance and testing

import io
import sys
import base64
import numpy as np
import requests
from typing import Dict, Optional, Tuple, List
from datetime import date, datetime, timedelta
from collections import defaultdict

import rasterio
from rasterio.session import AWSSession
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.crs import CRS
from rasterio.features import shapes

from shapely.geometry import shape, mapping, LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union

from pystac_client import Client
from scipy.ndimage import zoom, binary_opening, binary_closing, binary_dilation
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from PIL import Image

try:
    import planetary_computer as pc
    MPC_AVAILABLE = True
except:
    MPC_AVAILABLE = False

# Constants
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
E84_STAC = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
DEFAULT_MAX_ITEMS = 12
SCL_BAD = {0, 1, 3, 7, 8, 9, 10, 11}


def geocode_location(query: str) -> Optional[Tuple[float, float]]:
    """Geocode a location using Nominatim"""
    if not query.strip():
        return None
    
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "coastal-monitor-shiny/1.0"}
        )
        r.raise_for_status()
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


# Copy all the processing functions from app.py here
# I'll include the key ones:

def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Water Index"""
    denom = green + nir
    denom[denom == 0] = np.nan
    ndwi = (green - nir) / denom
    return np.clip(ndwi, -1.0, 1.0)


def compute_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Compute Modified NDWI for turbid water"""
    denom = green + swir
    denom[denom == 0] = np.nan
    mndwi = (green - swir) / denom
    return np.clip(mndwi, -1.0, 1.0)


def compute_awei(green: np.ndarray, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Compute Automated Water Extraction Index"""
    awei = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    return awei


def ndwi_to_watermask(ndwi: np.ndarray, thresh: float = 0.0, method: str = "fixed") -> np.ndarray:
    """Convert NDWI to binary water mask"""
    if method == "otsu":
        from skimage.filters import threshold_otsu
        valid_ndwi = ndwi[~np.isnan(ndwi)]
        if len(valid_ndwi) < 100:
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
        from skimage.filters import threshold_local
        ndwi_filled = np.nan_to_num(ndwi, nan=-1.0)
        try:
            local_thresh = threshold_local(ndwi_filled, block_size=35, method='gaussian')
            mask = (ndwi_filled > local_thresh).astype(np.uint8)
            mask[np.isnan(ndwi)] = 0
            return mask
        except:
            mask = (ndwi >= thresh).astype(np.uint8)
            return mask
    else:  # fixed
        mask = (ndwi >= thresh).astype(np.uint8)
        return mask


def refine_water_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Clean water mask with morphological operations"""
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    mask_clean = binary_opening(mask.astype(bool), structure=kernel)
    mask_clean = binary_closing(mask_clean, structure=kernel)
    return mask_clean.astype(np.uint8)


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
    """Extract shoreline from water polygons"""
    lines = []
    for p in polys:
        line = LineString(p.exterior.coords)
        if smooth_tolerance > 0:
            line = line.simplify(smooth_tolerance, preserve_topology=True)
        lines.append(line)
    return lines


def read_band_window(url: str, bbox: Dict, dst_res=10) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    """Read a rectangular window from a COG URL, reprojected to EPSG:4326"""
    with rasterio.Env(AWSSession() if url.startswith("s3://") else None):
        with rasterio.open(url) as src:
            dst_crs = CRS.from_epsg(4326)
            left, bottom, right, top = bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]
            
            width = int((right - left) / (dst_res / 111320))
            height = int((top - bottom) / (dst_res / 111320))
            
            if width < 10:
                width = 10
            if height < 10:
                height = 10
            
            transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
            dst_array = np.empty((height, width), dtype=np.float32)
            
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=WarpResampling.bilinear
            )
            
            if dst_array.size == 0 or dst_array.shape[0] == 0 or dst_array.shape[1] == 0:
                raise RuntimeError(f"Failed to read data from Sentinel-2 scene. Bbox: {bbox}")
            
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
    """Read and mosaic multiple Sentinel-2 scenes for a specific band"""
    def get_asset_url(item, name: str, use_mpc=True) -> Optional[str]:
        if name not in item.assets:
            return None
        asset = item.assets[name]
        if use_mpc and MPC_AVAILABLE:
            return asset.href
        href = asset.href
        if href.startswith("s3://"):
            href = href.replace("s3://", "https://").replace(".s3.", ".s3.")
        return href
    
    if len(items) == 1:
        url = get_asset_url(items[0], band_name, use_mpc=use_mpc)
        if url is None:
            raise RuntimeError(f"Scene missing required band {band_name}")
        return read_band_window(url, bbox)
    
    # Multiple scenes - mosaic them
    arrays = []
    transforms = []
    for idx, item in enumerate(items):
        url = get_asset_url(item, band_name, use_mpc=use_mpc)
        if url is None:
            continue
        try:
            arr, trans, meta = read_band_window(url, bbox)
            arrays.append(arr)
            transforms.append(trans)
        except Exception:
            continue
    
    if not arrays:
        raise RuntimeError(f"No valid data for band {band_name}")
    
    if len(arrays) == 1:
        return arrays[0], transforms[0], meta
    
    # Mosaic: average where there's overlap
    mosaic = np.full_like(arrays[0], np.nan, dtype=np.float32)
    counts = np.zeros_like(arrays[0], dtype=np.int32)
    
    for arr in arrays:
        valid = ~np.isnan(arr) & (arr != 0)
        mosaic = np.where(valid & np.isnan(mosaic), arr, mosaic)
        mosaic = np.where(valid & ~np.isnan(mosaic), (mosaic * counts + arr) / (counts + 1), mosaic)
        counts = np.where(valid, counts + 1, counts)
    
    return mosaic, transforms[0], meta


def create_rgb_composite(items: List[dict], bbox: Dict, use_mpc=True) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    """Create a true-color RGB composite from Sentinel-2"""
    r_arr, r_transform, meta = mosaic_bands(items, "B04", bbox, use_mpc=use_mpc)
    g_arr, g_transform, _ = mosaic_bands(items, "B03", bbox, use_mpc=use_mpc)
    b_arr, b_transform, _ = mosaic_bands(items, "B02", bbox, use_mpc=use_mpc)
    
    rgb = np.stack([r_arr, g_arr, b_arr], axis=-1)
    rgb = np.clip(rgb / 3000.0, 0, 1)
    rgb = np.power(rgb, 0.8)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb, r_transform, meta


def rgb_to_base64_png(rgb_array: np.ndarray) -> str:
    """Convert RGB array to base64-encoded PNG"""
    mask = np.any(np.isnan(rgb_array), axis=-1)
    rgb_clean = rgb_array.copy()
    rgb_clean[mask] = 0
    
    img = Image.fromarray(rgb_clean, mode='RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    b64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def mask_clouds_with_scl(scl: np.ndarray) -> np.ndarray:
    """Build a boolean mask where True = good (not cloudy)"""
    if scl is None:
        return None
    
    bad = np.isin(scl.astype(np.uint8), list(SCL_BAD))
    structure = np.ones((5, 5), dtype=bool)
    bad_dilated = binary_dilation(bad, structure=structure, iterations=1)
    good = ~bad_dilated
    return good


def multi_index_water_detection(b02: np.ndarray, b03: np.ndarray, b08: np.ndarray, 
                                b11: np.ndarray, b12: np.ndarray, 
                                consensus_threshold: int = 2) -> np.ndarray:
    """Combine multiple water indices for robust detection"""
    ndwi = compute_ndwi(b03, b08)
    water_ndwi = (ndwi > 0.0).astype(int)
    
    mndwi = compute_mndwi(b03, b11)
    water_mndwi = (mndwi > 0.0).astype(int)
    
    awei = compute_awei(b03, b08, b11, b12)
    water_awei = (awei > 0.0).astype(int)
    
    vote_count = water_ndwi + water_mndwi + water_awei
    consensus = (vote_count >= consensus_threshold).astype(np.uint8)
    
    return consensus


def stac_search(bbox: Dict, start_date: str, end_date: str, max_items=12, use_mpc=True) -> List[dict]:
    """Search for Sentinel-2 L2A scenes via STAC API"""
    left, bottom, right, top = bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]
    
    catalog_url = MPC_STAC if use_mpc else E84_STAC
    catalog = Client.open(catalog_url)
    
    if use_mpc and MPC_AVAILABLE:
        catalog.modifier = pc.sign_inplace
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[left, bottom, right, top],
        datetime=f"{start_date}/{end_date}",
        limit=max_items
    )
    
    items = list(search.items())
    return items


def pick_best_items_for_mosaic(items: List[dict], max_items=3) -> List[dict]:
    """Pick the best scenes from a list, avoiding clouds"""
    scored = []
    
    for item in items:
        score = 0.0
        props = item.properties
        
        cloud_cover = props.get("eo:cloud_cover", 100.0)
        score += (100.0 - cloud_cover)
        
        date_str = props.get("datetime", "")
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                days_old = (datetime.now(dt.tzinfo) - dt).days
                score -= days_old * 0.1
            except:
                pass
        
        scored.append((score, item))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    best = [item for score, item in scored[:max_items]]
    
    return best


def pick_best_item(items: List[dict]) -> Optional[dict]:
    """Pick the single best scene from a list"""
    if not items:
        return None
    
    best = pick_best_items_for_mosaic(items, max_items=1)
    return best[0] if best else None


def get_asset_url(item, name: str, use_mpc=True) -> Optional[str]:
    """Get the URL for a specific asset from a STAC item"""
    if name not in item.assets:
        return None
    asset = item.assets[name]
    if use_mpc and MPC_AVAILABLE:
        return asset.href
    href = asset.href
    if href.startswith("s3://"):
        href = href.replace("s3://", "https://").replace(".s3.", ".s3.")
    return href


class CoastalAnalyzer:
    """Main analyzer class for coastal change detection"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.bbox = params['bbox']
        self.start_date = params['start_date']
        self.end_date = params['end_date']
        self.cloud_max = params['cloud_max']
        self.max_scenes = params.get('max_scenes', 10)  # Add max_scenes parameter
        self.use_mpc = params['provider'] == 'mpc'
        self.method = params['method']
        self.mode = params['mode']
    
    def run(self) -> Dict:
        """Execute analysis"""
        if self.mode == 'snapshot':
            return self.run_snapshot()
        else:
            return self.run_change_detection()
    
    def run_snapshot(self) -> Dict:
        """Run single window analysis"""
        # Search for scenes
        items = stac_search(self.bbox, self.start_date.isoformat(), self.end_date.isoformat(), 
                           max_items=DEFAULT_MAX_ITEMS, use_mpc=self.use_mpc)
        
        if not items:
            raise RuntimeError("No Sentinel-2 items found for that time window and AOI.")
        
        # Find best items for mosaic - use max_scenes parameter
        best_items = pick_best_items_for_mosaic(items, max_items=self.max_scenes)
        print(f"Using {len(best_items)} scenes for analysis (max_scenes={self.max_scenes})", file=sys.stderr)
        if not best_items:
            best_items = [pick_best_item(items)]
            if best_items[0] is None:
                raise RuntimeError("No suitable scene found (cloud thresholds too strict?).")
        
        # Extract parameters
        detection_method = self.params.get('detection_method', 'fixed')
        ndwi_thresh = self.params.get('ndwi_threshold', 0.0)
        apply_morphology = self.params.get('apply_morphology', True)
        morph_kernel_size = self.params.get('morph_kernel', 3)
        smooth_tolerance = self.params.get('smooth_tolerance', 2.0)
        consensus_votes = self.params.get('consensus_votes', 2)
        external_cloud_mask = self.params.get('external_cloud_mask', None)
        
        # Create scene summary
        scene_ids = [item.id for item in best_items]
        scene_summary = f"{len(best_items)} scene(s)"
        avg_cloud = sum(item.properties.get("eo:cloud_cover", 0) for item in best_items) / len(best_items)
        
        # Mosaic bands
        g_arr, g_transform, _ = mosaic_bands(best_items, "B03", self.bbox, use_mpc=self.use_mpc)
        n_arr, n_transform, _ = mosaic_bands(best_items, "B08", self.bbox, use_mpc=self.use_mpc)
        
        # Try to get cloud mask
        scl_arr = None
        try:
            scl_arr, scl_transform, _ = mosaic_bands(best_items, "SCL", self.bbox, use_mpc=self.use_mpc)
            if scl_arr.shape != g_arr.shape:
                from scipy.ndimage import zoom
                zoom_factors = (g_arr.shape[0] / scl_arr.shape[0], g_arr.shape[1] / scl_arr.shape[1])
                scl_arr = zoom(scl_arr, zoom_factors, order=0)
        except Exception:
            scl_arr = None
        
        # Compute NDWI
        ndwi = compute_ndwi(g_arr, n_arr)
        
        # Apply cloud masking
        if external_cloud_mask is not None:
            good_mask = external_cloud_mask
        else:
            good_mask = mask_clouds_with_scl(scl_arr)
        
        if good_mask is not None:
            ndwi = np.where(good_mask, ndwi, np.nan)
        
        # Water detection
        if detection_method == "multi-index":
            try:
                b02_arr, _, _ = mosaic_bands(best_items, "B02", self.bbox, use_mpc=self.use_mpc)
                b11_arr, _, _ = mosaic_bands(best_items, "B11", self.bbox, use_mpc=self.use_mpc)
                b12_arr, _, _ = mosaic_bands(best_items, "B12", self.bbox, use_mpc=self.use_mpc)
                
                if b11_arr.shape != g_arr.shape:
                    from scipy.ndimage import zoom
                    zoom_factors = (g_arr.shape[0] / b11_arr.shape[0], g_arr.shape[1] / b11_arr.shape[1])
                    b11_arr = zoom(b11_arr, zoom_factors, order=1)
                    b12_arr = zoom(b12_arr, zoom_factors, order=1)
                
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
            except Exception:
                water_mask = ndwi_to_watermask(ndwi, thresh=ndwi_thresh, method="fixed")
                water_mask[np.isnan(ndwi)] = 0
        else:
            water_mask = ndwi_to_watermask(ndwi, thresh=ndwi_thresh, method=detection_method)
            water_mask[np.isnan(ndwi)] = 0
        
        # Morphological refinement
        if apply_morphology:
            water_mask = refine_water_mask(water_mask, kernel_size=morph_kernel_size)
        
        # Vectorize (pass bbox to filter edge-touching polygons)
        polys = vectorize_mask(water_mask, g_transform, bbox=self.bbox)
        shores = shoreline_from_polys(polys, smooth_tolerance=smooth_tolerance)
        
        # Build GeoJSON
        poly_fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(p), "properties": {"type": "water"}} for p in polys
        ]}
        line_fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(l), "properties": {"type": "shoreline"}} for l in shores
        ]}
        
        # Create RGB composite
        rgb_composite, rgb_transform, _ = create_rgb_composite(best_items, self.bbox, use_mpc=self.use_mpc)
        rgb_base64 = rgb_to_base64_png(rgb_composite)
        bounds = [[self.bbox["min_lat"], self.bbox["min_lon"]], [self.bbox["max_lat"], self.bbox["max_lon"]]]
        
        # Calculate water area
        water_area_km2 = sum(p.area * 111.32 * 111.32 for p in polys)  # Convert deg² to km²
        
        return {
            "item_id": scene_summary,
            "cloud_cover": avg_cloud,
            "shoreline": line_fc,
            "water": poly_fc,
            "rgb_image": rgb_base64,
            "rgb_bounds": bounds,
            "cloud_mask": good_mask,
            "water_area_km2": water_area_km2,
        }
    
    def run_change_detection(self) -> Dict:
        """Run change detection analysis"""
        # Get parameters
        back_days = self.params.get('offset_days', 365)
        
        # First pass: get cloud masks from both periods
        now_run_initial = self.run_snapshot()
        
        # Calculate historical window
        window_duration = (self.end_date - self.start_date).days
        earlier_start = self.start_date - timedelta(days=back_days)
        earlier_end = earlier_start + timedelta(days=window_duration)
        
        # Create historical analyzer
        hist_params = self.params.copy()
        hist_params['start_date'] = earlier_start
        hist_params['end_date'] = earlier_end
        hist_analyzer = CoastalAnalyzer(hist_params)
        then_run_initial = hist_analyzer.run_snapshot()
        
        # Combine cloud masks
        now_mask = now_run_initial.get("cloud_mask")
        then_mask = then_run_initial.get("cloud_mask")
        
        combined_mask = None
        if now_mask is not None and then_mask is not None:
            if now_mask.shape == then_mask.shape:
                combined_mask = now_mask & then_mask
        
        # Second pass with combined mask
        if combined_mask is not None:
            # Re-run current period
            params_now = self.params.copy()
            params_now['external_cloud_mask'] = combined_mask
            analyzer_now = CoastalAnalyzer(params_now)
            now_run = analyzer_now.run_snapshot()
            
            # Re-run historical period
            params_then = hist_params.copy()
            params_then['external_cloud_mask'] = combined_mask
            analyzer_then = CoastalAnalyzer(params_then)
            then_run = analyzer_then.run_snapshot()
        else:
            now_run = now_run_initial
            then_run = then_run_initial
        
        # Compute change polygons
        now_polys = [shape(f["geometry"]) for f in now_run["water"]["features"]]
        then_polys = [shape(f["geometry"]) for f in then_run["water"]["features"]]
        now_union = unary_union(now_polys) if now_polys else None
        then_union = unary_union(then_polys) if then_polys else None
        
        prog = []  # Progradation: water disappeared = land gained
        ret = []   # Retreat: water appeared = land lost
        
        if now_union and then_union:
            new_water = now_union.difference(then_union)
            lost_water = then_union.difference(now_union)
            
            if isinstance(new_water, (Polygon, MultiPolygon)):
                ret = [new_water] if isinstance(new_water, Polygon) else list(new_water.geoms)
            if isinstance(lost_water, (Polygon, MultiPolygon)):
                prog = [lost_water] if isinstance(lost_water, Polygon) else list(lost_water.geoms)
        
        prog_fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(p), "properties": {"change": "progradation"}} for p in prog
        ]}
        ret_fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(p), "properties": {"change": "retreat"}} for p in ret
        ]}
        
        # Calculate areas
        prog_area_km2 = sum(p.area * 111.32 * 111.32 for p in prog)
        ret_area_km2 = sum(p.area * 111.32 * 111.32 for p in ret)
        net_change_km2 = prog_area_km2 - ret_area_km2
        
        return {
            "now": now_run,
            "then": then_run,
            "progradation": prog_fc,
            "retreat": ret_fc,
            "progradation_area_km2": prog_area_km2,
            "retreat_area_km2": ret_area_km2,
            "net_change_km2": net_change_km2,
            "current_period": f"{self.start_date} to {self.end_date}",
            "historical_period": f"{earlier_start} to {earlier_end}",
            "current_cloud": now_run.get("cloud_cover", 0),
            "historical_cloud": then_run.get("cloud_cover", 0),
        }
    
    def search_scenes(self, start: date, end: date):
        """Search STAC for Sentinel-2 scenes"""
        endpoint = MPC_STAC if self.use_mpc else E84_STAC
        client = Client.open(endpoint, modifier=pc.sign_inplace if (self.use_mpc and MPC_AVAILABLE) else None)
        
        items = client.search(
            collections=[S2_COLLECTION],
            bbox=[self.bbox["min_lon"], self.bbox["min_lat"], self.bbox["max_lon"], self.bbox["max_lat"]],
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            query={"eo:cloud_cover": {"lte": self.cloud_max}},
            sortby=[{"field": "properties.datetime", "direction": "desc"}],
            max_items=12,
        ).get_all_items()
        
        return items
    
    def pick_best_item(self, items):
        """Select best scene"""
        best = None
        for it in items:
            cc = it.properties.get("eo:cloud_cover", 100.0)
            if best is None:
                best = (cc, it)
            else:
                if (cc < best[0]) or (cc == best[0] and it.datetime > best[1].datetime):
                    best = (cc, it)
        return None if best is None else best[1]


def format_results(result: Dict) -> str:
    """Format results as markdown/HTML"""
    # Helper to format results for display
    return "Results formatted"
