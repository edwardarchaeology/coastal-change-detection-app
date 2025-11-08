# Coastal Change Monitor - Real Sentinel-2 Application

Advanced coastal change detection using real Sentinel-2 satellite imagery with multiple detection algorithms and post-processing options.

## 🌊 Features

### Multi-Algorithm Detection
- **NDWI + Fixed Threshold**: Fast, manual control, works well in clear conditions
- **NDWI + Otsu Auto**: Adaptive per-scene thresholding for variable conditions
- **NDWI + Adaptive**: Local thresholding for complex mixed scenes
- **Multi-Index Consensus**: Combines NDWI + MNDWI + AWEI for robust detection

### Advanced Post-Processing
- **Morphological refinement**: Remove noise and fill gaps in water masks
- **Shoreline smoothing**: Douglas-Peucker simplification for cleaner results
- **Cloud masking**: Scene Classification Layer (SCL) with dilation
- **Two-pass change detection**: Union mask prevents false changes from clouds

### Real Satellite Data
- **Sentinel-2 L2A**: 10m RGB/NIR, 20m SWIR bands
- **Multi-scene mosaicking**: Automatically combines scenes for complete coverage
- **RGB composites**: True-color satellite imagery overlays
- **Change detection**: Compare two time periods with erosion/accretion polygons

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open browser to `http://localhost:8501`

### First Analysis

1. **Search location**: Enter a coastal area (e.g., "Grand Isle, Louisiana")
2. **Set time period**: Choose start/end dates (default: last 30 days)
3. **Select detection method**: Start with "NDWI + Otsu (Auto)"
4. **Click "Snapshot"**: Analyze current shoreline

### Change Detection

1. Enable **"Compare with earlier window"**
2. Set **offset** (e.g., 365 days for yearly comparison)
3. Click **"Change Detection"**
4. View green (land gain) and red (land loss) areas

## 📖 Documentation

- **[QUICK_START.md](QUICK_START.md)**: User guide with recommended settings by scenario
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)**: Technical details of algorithm enhancements
- **In-app tooltips**: Hover over options for detailed explanations

## 🎯 Detection Methods Explained

### When to Use Each Method

| Situation | Best Method | Why |
|-----------|-------------|-----|
| Clear water, simple beach | Fixed Threshold | Fast, reliable |
| Variable turbidity | Otsu Auto | Adapts automatically |
| Very turbid/shallow water | Multi-Index | Uses SWIR bands |
| Complex mixed scenes | Adaptive | Local thresholding |

### Algorithm Comparison

- **Fixed**: ⚡⚡⚡ Fast, manual tuning
- **Otsu**: ⚡⚡ Medium speed, automatic
- **Adaptive**: ⚡ Slower, handles variation
- **Multi-Index**: ⚡ Slowest, most robust

## 🔧 Configuration

### Detection Settings (Sidebar)

**Detection Method**: Choose algorithm  
**NDWI Threshold**: -0.3 to 0.3 (Fixed method only)  
**Consensus Votes**: 1-3 indices must agree (Multi-Index only)  
**Refine water mask**: Enable morphological post-processing  
**Refinement strength**: 2-7 pixel kernel size  
**Smooth shoreline**: Enable Douglas-Peucker simplification  
**Smoothing tolerance**: 0-10 meters  

### Acquisition Settings

**Start/End dates**: Time period to analyze  
**Max cloud cover**: 0-100% threshold  
**Data provider**: Microsoft Planetary Computer (default) or Element84  
**Basemap**: Satellite, light, or terrain background  

## 📊 Understanding Results

### Single Window Analysis

- **Blue lines**: Detected shoreline (water/land boundary)
- **Light blue areas**: Water mask (all detected water)
- **Satellite image**: True-color RGB composite
- **Metrics**: Water area, scene info, cloud cover

### Change Detection

- **🟢 Green areas**: Progradation (land gained, water disappeared)
- **🔴 Red areas**: Retreat/erosion (land lost, water appeared)
- **Blue line**: Current shoreline position
- **Red line**: Historical shoreline position
- **Net change**: Total land gain/loss in km²

## 💡 Tips for Best Results

1. **Keep AOI small**: Zoom to your specific area of interest
2. **Use appropriate method**: See QUICK_START.md decision tree
3. **Check cloud cover**: Lower threshold (20-30%) for clearer scenes
4. **Similar tidal states**: Compare same tidal conditions
5. **Longer time gaps**: 6-12 months show clearer trends
6. **Inspect RGB images**: Always visually verify satellite photos
7. **Adjust post-processing**: Increase if results are noisy
8. **Download data**: Save GeoJSON for GIS analysis

## ⚠️ Limitations

- **Resolution**: 10m pixels limit sub-20m change detection
- **Tidal effects**: Shoreline position varies with tide (not corrected)
- **Wave run-up**: 5-20m surf zone adds uncertainty
- **Geolocation**: ±5-10m sensor positioning error
- **Turbidity**: Very murky water may not be detected (use Multi-Index)

## 📚 References

- **CoastSat**: https://github.com/kvos/CoastSat
- **Sentinel-2**: https://sentinel.esa.int/web/sentinel/missions/sentinel-2

---

**Version**: 2.0.0 (November 6, 2025)  
**Major Update**: Multi-algorithm detection, morphological refinement, enhanced UI
