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

import rasterio
from rasterio.session import AWSSession
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.crs import CRS
from rasterio.features import shapes

from shapely.geometry import shape, mapping, LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union

from pystac_client import Client
from scipy.ndimage import zoom, binary_closing, binary_dilation
from PIL import Image

try:
    import planetary_computer as pc
    MPC_AVAILABLE = True
except Exception:
    MPC_AVAILABLE = False

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
E84_STAC = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
DEFAULT_MAX_ITEMS = 12

# Sentinel-2 SCL values to treat as "bad"
# 0: No data, 1: Saturated/defective, 3: Cloud shadows, 7: Medium prob. clouds,
# 8: High prob. clouds, 9: Thin cirrus, 10: Snow/ice, 11: Cloud (combined)
SCL_BAD = {0, 1, 3, 7, 8, 9, 10, 11}

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def geocode_location(query: str) -> Optional[Tuple[float, float]]:
    """Geocode a location using Nominatim."""
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

# ---------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------
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
    # Automated Water Extraction Index (AWEI_sh)
    return 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)

def ndwi_to_watermask(ndwi: np.ndarray, thresh: float = 0.0, method: str = "fixed") -> np.ndarray:
    """Convert NDWI (or any index) to binary mask by thresholding."""
    if method == "otsu":
        from skimage.filters import threshold_otsu
        valid = ndwi[~np.isnan(ndwi)]
        if valid.size < 100:
            return (ndwi >= thresh).astype(np.uint8)
        try:
            t = threshold_otsu(valid)
            return (ndwi >= t).astype(np.uint8)
        except Exception:
            return (ndwi >= thresh).astype(np.uint8)
    elif method == "adaptive":
        from skimage.filters import threshold_local
        nd = np.nan_to_num(ndwi, nan=-1.0)
        try:
            tloc = threshold_local(nd, block_size=35, method="gaussian")
            mask = (nd > tloc).astype(np.uint8)
            mask[np.isnan(ndwi)] = 0
            return mask
        except Exception:
            return (ndwi >= thresh).astype(np.uint8)
    else:
        return (ndwi >= thresh).astype(np.uint8)

# ---------------------------------------------------------------------
# Morphology / vectorization / shoreline
# ---------------------------------------------------------------------
def refine_water_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Clean water mask with morphological operations.
    Prefer closing-only to preserve narrow coastal features at 10 m.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    mask_clean = binary_closing(mask.astype(bool), structure=kernel)
    return mask_clean.astype(np.uint8)

def vectorize_mask(mask: np.ndarray, transform, bbox: Dict = None) -> List[Polygon]:
    """
    Convert raster mask to vector polygons and keep coastal features.
    Filters only obvious noise; keeps islands and the main ocean polygon.
    """
    polys = []
    for geom, val in shapes(mask, mask=None, transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        if g.area > 0:
            polys.append(g)

    if not polys:
        return []

    merged = unary_union(polys)
    if isinstance(merged, Polygon):
        geoms = [merged]
    elif isinstance(merged, MultiPolygon):
        geoms = list(merged.geoms)
    else:
        return []

    # Keep small shoreline bits (≈0.001 km² at mid-latitudes)
    min_area = 1e-8
    geoms = [p for p in geoms if p.area > min_area]

    if bbox and len(geoms) > 1:
        # Slight inset to decide "fully inside" vs edge-touching
        boundary_box = box(
            bbox['min_lon'] + 1e-6,
            bbox['min_lat'] + 1e-6,
            bbox['max_lon'] - 1e-6,
            bbox['max_lat'] - 1e-6
        )
        geoms_sorted = sorted(geoms, key=lambda g: g.area, reverse=True)
        filtered = [geoms_sorted[0]]  # always keep the largest (ocean)
        for g in geoms_sorted[1:]:
            if boundary_box.contains(g):
                filtered.append(g)
            elif g.intersects(boundary_box):
                if g.area > geoms_sorted[0].area * 0.01 or g.area > 1e-6:
                    filtered.append(g)
        return filtered

    return geoms

def _meters_to_degrees(meters: float, lat_deg: float) -> float:
    lat_deg_per_m = 1.0 / 111_320.0
    lon_deg_per_m = 1.0 / (111_320.0 * max(np.cos(np.deg2rad(lat_deg)), 1e-6))
    return meters * max(lat_deg_per_m, lon_deg_per_m)  # isotropic simplify

def extract_coastline(polys: List[Polygon], bbox: Dict,
                      smooth_tolerance_m: float = 0.0,
                      clip_aoi_edge_within_m: float = 30.0) -> List[LineString]:
    """
    Coastline = exterior of the *largest* water polygon (ocean), with any parts
    that coincide with the AOI boundary removed, then (optionally) smoothed.
    """
    if not polys:
        return []

    merged = unary_union(polys)
    geoms = [merged] if isinstance(merged, Polygon) else list(merged.geoms)

    # 1) Pick the 'ocean' polygon
    ocean = max(geoms, key=lambda g: g.area)

    # 2) Build AOI box + thin strip around its edges to clip away box-hugging lines
    mid_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2.0
    eps_deg = _meters_to_degrees(clip_aoi_edge_within_m, mid_lat)
    aoi = box(bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"])
    aoi_edge_strip = aoi.boundary.buffer(eps_deg)

    # 3) Coastline is the ocean exterior minus anything glued to AOI edges
    coast = LineString(ocean.exterior.coords)
    coast = coast.difference(aoi_edge_strip)

    # 4) Optional light smoothing (meters input)
    if smooth_tolerance_m and smooth_tolerance_m > 0:
        tol_deg = _meters_to_degrees(smooth_tolerance_m, mid_lat)
        if not coast.is_empty:
            coast = coast.simplify(tol_deg, preserve_topology=True)

    # 5) Return as list (API-compatible)
    if coast.is_empty:
        return []
    if coast.geom_type == "MultiLineString":
        return [ls for ls in coast.geoms if not ls.is_empty]
    return [coast]

# ---------------------------------------------------------------------
# IO helpers (read/mosaic/resample)  — NOW SUPPORTS REFERENCE GRID
# ---------------------------------------------------------------------
def read_band_window(
    url: str, bbox: Dict, dst_res=10,
    reference_transform: Optional[rasterio.Affine] = None,
    reference_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    """Read a window from a COG URL, reprojected to EPSG:4326 (optionally onto a reference grid)."""
    with rasterio.Env(AWSSession() if url.startswith("s3://") else None):
        with rasterio.open(url) as src:
            dst_crs = CRS.from_epsg(4326)

            if reference_transform is not None and reference_shape is not None:
                transform = reference_transform
                height, width = reference_shape
            else:
                left, bottom, right, top = bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]
                # crude degrees-per-meter conversion at equator; fine for window dims
                width = max(10, int((right - left) / (dst_res / 111320)))
                height = max(10, int((top - bottom) / (dst_res / 111320)))
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

def mosaic_bands(
    items: List[dict], band_name: str, bbox: Dict, use_mpc=True,
    reference_transform: Optional[rasterio.Affine] = None,
    reference_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    """Read and mosaic multiple Sentinel-2 scenes for a specific band (honors reference grid)."""
    def _asset_url(item, name: str, use_mpc=True) -> Optional[str]:
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
        url = _asset_url(items[0], band_name, use_mpc=use_mpc)
        if url is None:
            raise RuntimeError(f"Scene missing required band {band_name}")
        return read_band_window(
            url, bbox,
            reference_transform=reference_transform,
            reference_shape=reference_shape
        )

    arrays, transforms, meta = [], [], None
    for item in items:
        url = _asset_url(item, band_name, use_mpc=use_mpc)
        if url is None:
            continue
        try:
            arr, trans, meta = read_band_window(
                url, bbox,
                reference_transform=reference_transform,
                reference_shape=reference_shape
            )
            arrays.append(arr)
            transforms.append(trans)
        except Exception:
            continue

    if not arrays:
        raise RuntimeError(f"No valid data for band {band_name}")

    if len(arrays) == 1:
        return arrays[0], transforms[0], meta

    # Simple average mosaic where overlapping
    mosaic = np.full_like(arrays[0], np.nan, dtype=np.float32)
    counts = np.zeros_like(arrays[0], dtype=np.int32)
    for arr in arrays:
        valid = ~np.isnan(arr) & (arr != 0)
        mosaic = np.where(valid & np.isnan(mosaic), arr, mosaic)
        mosaic = np.where(valid & ~np.isnan(mosaic), (mosaic * counts + arr) / (counts + 1), mosaic)
        counts = np.where(valid, counts + 1, counts)

    return mosaic, transforms[0], meta

def create_rgb_composite(
    items: List[dict], bbox: Dict, use_mpc=True,
    reference_transform: Optional[rasterio.Affine] = None,
    reference_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    r, tr, meta = mosaic_bands(items, "B04", bbox, use_mpc=use_mpc,
                               reference_transform=reference_transform, reference_shape=reference_shape)
    g, _, _   = mosaic_bands(items, "B03", bbox, use_mpc=use_mpc,
                               reference_transform=reference_transform, reference_shape=reference_shape)
    b, _, _   = mosaic_bands(items, "B02", bbox, use_mpc=use_mpc,
                               reference_transform=reference_transform, reference_shape=reference_shape)
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb / 3000.0, 0, 1)
    rgb = np.power(rgb, 0.8)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb, tr, meta

def rgb_to_base64_png(rgb_array: np.ndarray) -> str:
    mask = np.any(np.isnan(rgb_array), axis=-1)
    rgb_clean = rgb_array.copy()
    rgb_clean[mask] = 0
    img = Image.fromarray(rgb_clean, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

# ---------------------------------------------------------------------
# Cloud masking
# ---------------------------------------------------------------------
def mask_clouds_with_scl(scl: np.ndarray, dilation: int = 1) -> np.ndarray:
    """
    True = good (not cloudy).
    Keep dilation small so we don't shave off beaches (~1 pixel ≈ 10 m).
    """
    if scl is None:
        return None
    bad = np.isin(scl.astype(np.uint8), list(SCL_BAD))
    if dilation and dilation > 0:
        structure = np.ones((3, 3), dtype=bool)
        bad = binary_dilation(bad, structure=structure, iterations=int(dilation))
    return ~bad

# ---------------------------------------------------------------------
# Multi-index voting (adaptive thresholds)
# ---------------------------------------------------------------------
def multi_index_water_detection(b02: np.ndarray, b03: np.ndarray, b08: np.ndarray,
                                b11: np.ndarray, b12: np.ndarray,
                                consensus_threshold: int = 2) -> np.ndarray:
    """
    Robust land/water via per-index Otsu thresholds + majority vote.
    Works better on bright sand, surf, and glinty/turbid water than fixed >0 cuts.
    """
    from skimage.filters import threshold_otsu

    ndwi  = compute_ndwi(b03, b08)
    mndwi = compute_mndwi(b03, b11)
    awei  = compute_awei(b03, b08, b11, b12)

    def _bin_by_otsu(arr, fallback_quantile=0.50):
        valid = arr[~np.isnan(arr)]
        if valid.size < 100:
            thr = np.quantile(valid, fallback_quantile) if valid.size > 0 else 0.0
        else:
            try:
                thr = threshold_otsu(valid)
            except Exception:
                thr = np.quantile(valid, fallback_quantile)
        return (arr > thr).astype(np.uint8)

    w1 = _bin_by_otsu(ndwi,  0.55)  # NDWI skews higher for water
    w2 = _bin_by_otsu(mndwi, 0.50)
    w3 = _bin_by_otsu(awei,  0.50)

    vote = w1 + w2 + w3
    return (vote >= consensus_threshold).astype(np.uint8)

# ---------------------------------------------------------------------
# STAC search helpers
# ---------------------------------------------------------------------
def stac_search(bbox: Dict, start_date: str, end_date: str, max_items=12, use_mpc=True) -> List[dict]:
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
    return list(search.items())

def pick_best_items_for_mosaic(items: List[dict], max_items=3) -> List[dict]:
    scored = []
    for it in items:
        score = 0.0
        cc = it.properties.get("eo:cloud_cover", 100.0)
        score += (100.0 - cc)
        date_str = it.properties.get("datetime", "")
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                days = (datetime.now(dt.tzinfo) - dt).days
                score -= days * 0.1
            except Exception:
                pass
        scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:max_items]]

def pick_best_item(items: List[dict]) -> Optional[dict]:
    best = pick_best_items_for_mosaic(items, max_items=1)
    return best[0] if best else None

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

# ---------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------
class CoastalAnalyzer:
    """Main analyzer class for coastal change detection."""

    def __init__(self, params: Dict):
        self.params = params
        self.bbox = params['bbox']
        self.start_date = params['start_date']
        self.end_date = params['end_date']
        self.cloud_max = params['cloud_max']
        self.max_scenes = params.get('max_scenes', 10)
        self.use_mpc = params['provider'] == 'mpc'
        self.method = params['method']
        self.mode = params['mode']

    def run(self) -> Dict:
        return self.run_snapshot() if self.mode == 'snapshot' else self.run_change_detection()

    def run_snapshot(self) -> Dict:
        # Search scenes
        items = stac_search(self.bbox, self.start_date.isoformat(), self.end_date.isoformat(),
                            max_items=DEFAULT_MAX_ITEMS, use_mpc=self.use_mpc)
        if not items:
            raise RuntimeError("No Sentinel-2 items found for that time window and AOI.")

        best_items = pick_best_items_for_mosaic(items, max_items=self.max_scenes)
        print(f"Using {len(best_items)} scenes for analysis (max_scenes={self.max_scenes})", file=sys.stderr)
        if not best_items:
            best_items = [pick_best_item(items)]
            if best_items[0] is None:
                raise RuntimeError("No suitable scene found (cloud thresholds too strict?).")

        # Params
        detection_method   = self.params.get('detection_method', 'fixed')
        ndwi_thresh        = self.params.get('ndwi_threshold', 0.0)
        apply_morphology   = self.params.get('apply_morphology', True)
        morph_kernel_size  = self.params.get('morph_kernel', 3)
        smooth_tolerance   = self.params.get('smooth_tolerance', 2.0)
        consensus_votes    = self.params.get('consensus_votes', 2)
        external_cloud_mask = self.params.get('external_cloud_mask', None)

        # Optional reference grid (for change detection locking)
        ref_T = self.params.get('reference_transform')
        ref_S = self.params.get('reference_shape')

        # Scene summary
        scene_summary = f"{len(best_items)} scene(s)"
        avg_cloud = sum(it.properties.get("eo:cloud_cover", 0.0) for it in best_items) / len(best_items)

        # Mosaic bands (10 m grid for B03/B08), honoring reference grid
        g_arr, g_transform, _ = mosaic_bands(best_items, "B03", self.bbox, use_mpc=self.use_mpc,
                                             reference_transform=ref_T, reference_shape=ref_S)
        n_arr, _, _ = mosaic_bands(best_items, "B08", self.bbox, use_mpc=self.use_mpc,
                                   reference_transform=g_transform, reference_shape=g_arr.shape)

        # SCL cloud mask (resample to 10 m grid; also onto reference grid)
        scl_arr = None
        try:
            scl_arr, _, _ = mosaic_bands(best_items, "SCL", self.bbox, use_mpc=self.use_mpc,
                                         reference_transform=g_transform, reference_shape=g_arr.shape)
        except Exception:
            scl_arr = None

        # NDWI base
        ndwi = compute_ndwi(g_arr, n_arr)

        # Cloud masking
        if external_cloud_mask is not None:
            good_mask = external_cloud_mask
        else:
            cloud_dilation = int(self.params.get('cloud_dilation', 1))
            good_mask = mask_clouds_with_scl(scl_arr, dilation=cloud_dilation)  # gentle ~10 m

        if good_mask is not None:
            ndwi = np.where(good_mask, ndwi, np.nan)

        # Water detection
        if detection_method == "multi-index":
            try:
                b02_arr, _, _ = mosaic_bands(best_items, "B02", self.bbox, use_mpc=self.use_mpc,
                                             reference_transform=g_transform, reference_shape=g_arr.shape)
                b11_arr, _, _ = mosaic_bands(best_items, "B11", self.bbox, use_mpc=self.use_mpc,
                                             reference_transform=g_transform, reference_shape=g_arr.shape)  # 20 m → 10 m later
                b12_arr, _, _ = mosaic_bands(best_items, "B12", self.bbox, use_mpc=self.use_mpc,
                                             reference_transform=g_transform, reference_shape=g_arr.shape)  # 20 m → 10 m later

                # (already on 10 m grid due to reference grid; keep for safety if source scaled)
                if b11_arr.shape != g_arr.shape:
                    zf = (g_arr.shape[0] / b11_arr.shape[0], g_arr.shape[1] / b11_arr.shape[1])
                    b11_arr = zoom(b11_arr, zf, order=1)
                if b12_arr.shape != g_arr.shape:
                    zf = (g_arr.shape[0] / b12_arr.shape[0], g_arr.shape[1] / b12_arr.shape[1])
                    b12_arr = zoom(b12_arr, zf, order=1)

                # Apply cloud mask consistently
                if good_mask is not None:
                    b02_arr = np.where(good_mask, b02_arr, np.nan)
                    g_arr_masked = np.where(good_mask, g_arr, np.nan)
                    n_arr_masked = np.where(good_mask, n_arr, np.nan)
                    b11_arr = np.where(good_mask, b11_arr, np.nan)
                    b12_arr = np.where(good_mask, b12_arr, np.nan)
                else:
                    g_arr_masked = g_arr
                    n_arr_masked = n_arr

                water_mask = multi_index_water_detection(
                    b02_arr, g_arr_masked, n_arr_masked, b11_arr, b12_arr,
                    consensus_threshold=consensus_votes
                )
            except Exception:
                water_mask = ndwi_to_watermask(ndwi, thresh=ndwi_thresh, method="fixed")
                water_mask[np.isnan(ndwi)] = 0
        else:
            water_mask = ndwi_to_watermask(ndwi, thresh=ndwi_thresh, method=detection_method)
            water_mask[np.isnan(ndwi)] = 0

        # Morphological refinement
        if apply_morphology:
            water_mask = refine_water_mask(water_mask, kernel_size=morph_kernel_size)

        # Vectorize & shoreline
        polys = vectorize_mask(water_mask, g_transform, bbox=self.bbox)
        shores = extract_coastline(polys, bbox=self.bbox,
                                   smooth_tolerance_m=smooth_tolerance,
                                   clip_aoi_edge_within_m=30.0)

        # GeoJSON outputs
        poly_fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(p), "properties": {"type": "water"}} for p in polys
        ]}
        line_fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(l), "properties": {"type": "shoreline"}} for l in shores
        ]}

        # RGB composite (on same grid)
        rgb_composite, _, _ = create_rgb_composite(best_items, self.bbox, use_mpc=self.use_mpc,
                                                   reference_transform=g_transform, reference_shape=g_arr.shape)
        rgb_base64 = rgb_to_base64_png(rgb_composite)
        bounds = [[self.bbox["min_lat"], self.bbox["min_lon"]],
                  [self.bbox["max_lat"], self.bbox["max_lon"]]]

        # Areas (approx; deg² → km²)
        water_area_km2 = sum(p.area * 111.32 * 111.32 for p in polys)

        return {
            "item_id": scene_summary,
            "cloud_cover": avg_cloud,
            "shoreline": line_fc,
            "water": poly_fc,
            "rgb_image": rgb_base64,
            "rgb_bounds": bounds,
            "cloud_mask": good_mask,
            "water_area_km2": water_area_km2,
            # expose grid so change-detection can reuse it
            "grid_transform": g_transform,
            "grid_shape": g_arr.shape,
        }

    def run_change_detection(self) -> Dict:
        back_days = self.params.get('offset_days', 365)

        # --- CURRENT snapshot (independent) ---
        now_snap = self.run_snapshot()

        # --- HISTORICAL snapshot on the SAME GRID ---
        window_days   = (self.end_date - self.start_date).days
        earlier_start = self.start_date - timedelta(days=back_days)
        earlier_end   = earlier_start + timedelta(days=window_days)

        hist_params = self.params.copy()
        hist_params['start_date'] = earlier_start
        hist_params['end_date']   = earlier_end
        # lock grid to the current snapshot grid
        hist_params['reference_transform'] = now_snap.get("grid_transform")
        hist_params['reference_shape']     = now_snap.get("grid_shape")

        then_snap = CoastalAnalyzer(hist_params).run_snapshot()

        # Save the exact snapshot shorelines to reuse in the change view
        shoreline_now_snapshot  = now_snap.get("shoreline")
        shoreline_then_snapshot = then_snap.get("shoreline")

        # --- Build combined clear-sky mask and rerun FOR CHANGE ONLY ---
        now_mask  = now_snap.get("cloud_mask")
        then_mask = then_snap.get("cloud_mask")
        combined  = None
        if (now_mask is not None and then_mask is not None
                and now_mask.shape == then_mask.shape):
            combined = now_mask & then_mask

        if combined is not None:
            p_now  = self.params.copy()
            p_now['external_cloud_mask']  = combined
            p_now['reference_transform']  = now_snap.get("grid_transform")
            p_now['reference_shape']      = now_snap.get("grid_shape")

            p_then = hist_params.copy()
            p_then['external_cloud_mask'] = combined
            # (hist params already include the reference grid)

            now_run  = CoastalAnalyzer(p_now).run_snapshot()
            then_run = CoastalAnalyzer(p_then).run_snapshot()
        else:
            now_run, then_run = now_snap, then_snap

        # Overwrite the nested shorelines with the snapshot coastlines so they match
        if shoreline_now_snapshot is not None:
            now_run['shoreline'] = shoreline_now_snapshot
        if shoreline_then_snapshot is not None:
            then_run['shoreline'] = shoreline_then_snapshot

        # Change polygons
        now_polys  = [shape(f["geometry"]) for f in now_run["water"]["features"]]
        then_polys = [shape(f["geometry"]) for f in then_run["water"]["features"]]
        now_union  = unary_union(now_polys) if now_polys else None
        then_union = unary_union(then_polys) if then_polys else None

        prog, ret = [], []  # progradation (land gained), retreat (land lost)
        if now_union and then_union:
            new_water  = now_union.difference(then_union)
            lost_water = then_union.difference(now_union)
            if isinstance(new_water, (Polygon, MultiPolygon)):
                ret  = [new_water] if isinstance(new_water, Polygon) else list(new_water.geoms)
            if isinstance(lost_water, (Polygon, MultiPolygon)):
                prog = [lost_water] if isinstance(lost_water, Polygon) else list(lost_water.geoms)

        prog_fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(p), "properties": {"change": "progradation"}} for p in prog
        ]}
        ret_fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(p), "properties": {"change": "retreat"}} for p in ret
        ]}

        # Areas (km²)
        to_km2 = lambda geoms: sum(p.area * 111.32 * 111.32 for p in geoms)
        prog_area_km2 = to_km2(prog)
        ret_area_km2  = to_km2(ret)
        net_change_km2 = prog_area_km2 - ret_area_km2

        return {
            "now": now_run,
            "then": then_run,
            # these are identical to snapshot-mode coastlines for each period
            "shoreline_now":  shoreline_now_snapshot,
            "shoreline_then": shoreline_then_snapshot,
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

def format_results(result: Dict) -> str:
    """Format results as markdown/HTML (placeholder)."""
    return "Results formatted"
