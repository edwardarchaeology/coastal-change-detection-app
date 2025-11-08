# Coastal Change Detection App - Algorithm Improvements

## Summary of Enhancements (November 6, 2025)

This document describes the improvements made to increase accuracy and add functionality to the coastal change detection application.

---

## 🎯 Key Improvements

### 1. **Multiple Detection Algorithms**

Users can now choose from 4 detection methods:

#### **NDWI + Fixed Threshold** (Original - Fast)
- Simple, fast, works in most conditions
- Manual threshold adjustment (-0.3 to 0.3)
- Best for: Clear water, simple coastlines

#### **NDWI + Otsu (Auto)** (NEW - Adaptive)
- Automatically finds optimal threshold per scene
- Adapts to local water turbidity and lighting conditions
- Best for: Variable water clarity, seasonal changes

#### **NDWI + Adaptive** (NEW - Complex Scenes)
- Local thresholding using Gaussian windows
- Handles varying water conditions within same scene
- Best for: Complex scenes with turbid and clear water

#### **Multi-Index Consensus** (NEW - Most Robust)
- Combines NDWI, MNDWI, and AWEI indices
- Pixel identified as water only if 2+ indices agree
- Uses additional SWIR bands (B11, B12) at 20m resolution
- Best for: Challenging conditions, turbid/shallow coastal water

---

### 2. **Morphological Refinement**

**Water Mask Post-Processing:**
- **Binary Opening**: Removes small isolated noise pixels
- **Binary Closing**: Fills small gaps in water bodies
- Adjustable kernel size (2-7 pixels) for refinement strength
- Default: 3x3 kernel (enabled by default)

**Benefits:**
- Cleaner water masks with less speckle
- More natural-looking boundaries
- Reduces false positives from image noise

---

### 3. **Shoreline Smoothing**

**Douglas-Peucker Simplification:**
- Smooths jagged polygon boundaries
- Adjustable tolerance (0-10 meters)
- Default: 2 meters
- Preserves topology while reducing vertex count

**Benefits:**
- More natural-looking shorelines
- Reduces noise from pixel-level artifacts
- Smaller file sizes for GeoJSON exports

---

### 4. **Enhanced Water Index Suite**

#### **MNDWI (Modified NDWI)**
```
MNDWI = (Green - SWIR1) / (Green + SWIR1)
```
- Better water discrimination in turbid/shallow coastal areas
- SWIR absorbed more strongly by water than NIR
- Uses Sentinel-2 B11 (20m resolution)

#### **AWEI (Automated Water Extraction Index)**
```
AWEI = 4 × (Green - SWIR1) - (0.25 × NIR + 2.75 × SWIR2)
```
- Suppresses non-water pixels (buildings, bare soil)
- More robust in urban coastal environments
- Uses B03, B08, B11, B12

---

### 5. **Improved User Interface**

#### **Detection Settings Panel**
- Algorithm dropdown with descriptions
- Conditional parameter display (only show relevant options)
- Post-processing controls (morphology, smoothing)
- Real-time parameter tooltips

#### **Results Display Enhancements**
- Detection method shown in results
- Water area calculation (km²)
- Two-column layout for better information density
- Enhanced metrics display

---

## 📊 Technical Details

### Algorithm Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Fixed Threshold | ⚡⚡⚡ Fast | ⭐⭐ Good | Clear water, simple scenes |
| Otsu Auto | ⚡⚡ Medium | ⭐⭐⭐ Better | Variable turbidity |
| Adaptive | ⚡ Slower | ⭐⭐⭐ Better | Complex mixed scenes |
| Multi-Index | ⚡ Slowest | ⭐⭐⭐⭐ Best | Challenging conditions |

### Resolution and Accuracy

**Sentinel-2 L2A Specifications:**
- RGB/NIR bands (B02/B03/B04/B08): 10m resolution
- SWIR bands (B11/B12): 20m resolution
- Geolocation accuracy: ±5-10m

**Reliable Detection Thresholds:**
- **Excellent**: Changes >30m (3+ pixels)
- **Good**: Changes 20-30m (2-3 pixels)
- **Uncertain**: Changes <20m (1-2 pixels)

**Factors Affecting Accuracy:**
- Tidal variation (±1-2m vertical = 10-40m horizontal on gentle slopes)
- Wave run-up (5-20m depending on conditions)
- Cloud/shadow contamination
- Water turbidity and suspended sediment

---

## 🔧 Implementation Details

### New Dependencies
- **scikit-image >= 0.22**: Otsu thresholding, adaptive thresholding, edge detection

### New Functions

```python
# Water index calculations
compute_mndwi(green, swir) -> ndarray
compute_awei(green, nir, swir1, swir2) -> ndarray

# Detection methods
ndwi_to_watermask(ndwi, thresh, method) -> ndarray
  # method: 'fixed', 'otsu', 'adaptive'

multi_index_water_detection(b02, b03, b08, b11, b12, consensus_threshold) -> ndarray

# Post-processing
refine_water_mask(mask, kernel_size) -> ndarray
shoreline_from_polys(polys, smooth_tolerance) -> List[LineString]
```

### Modified Functions

```python
run_once(..., detection_method, apply_morphology, morph_kernel_size, 
         smooth_tolerance, consensus_votes)
         
run_change(..., detection_method, apply_morphology, morph_kernel_size,
           smooth_tolerance, consensus_votes)
```

---

## 📖 Usage Recommendations

### For Most Users (Default Settings):
1. **Detection Method**: "NDWI + Otsu (Auto)"
2. **Refine water mask**: ✅ Enabled (kernel size: 3)
3. **Smooth shoreline**: ✅ Enabled (tolerance: 2.0m)

### For Challenging Conditions (Turbid/Shallow Water):
1. **Detection Method**: "Multi-Index Consensus"
2. **Consensus Threshold**: 2 votes
3. **Refine water mask**: ✅ Enabled (kernel size: 4-5)
4. **Smooth shoreline**: ✅ Enabled (tolerance: 2.5m)

### For Maximum Detail (Clear Water, High Confidence):
1. **Detection Method**: "NDWI + Fixed Threshold" 
2. **Threshold**: -0.05 to 0.0
3. **Refine water mask**: ✅ Enabled (kernel size: 2)
4. **Smooth shoreline**: Optional (tolerance: 1.0m)

### For Noisy Scenes:
1. **Detection Method**: "NDWI + Otsu (Auto)"
2. **Refine water mask**: ✅ Enabled (kernel size: 5-7)
3. **Smooth shoreline**: ✅ Enabled (tolerance: 3.0-5.0m)

---

## 🔬 Algorithm Performance

### Computational Overhead

**Single Window Analysis:**
- Fixed Threshold: ~5-10 seconds (baseline)
- Otsu Auto: +1-2 seconds
- Adaptive: +3-5 seconds
- Multi-Index: +10-15 seconds (reads 3 extra bands)

**Change Detection:**
- 2x single window time (two periods analyzed)
- Plus cloud mask combination overhead (~1 second)

### Memory Usage

**Band Reading:**
- 2 bands (G, NIR): ~40 MB for 1000x1000 pixels
- 5 bands (G, NIR, SWIR1, SWIR2, B): ~100 MB
- Mosaic of 3 scenes: ~3x memory

---

## 🎓 References

### Scientific Papers

1. **Otsu Thresholding**:
   - Otsu, N. (1979). "A threshold selection method from gray-level histograms"
   - IEEE Transactions on Systems, Man, and Cybernetics

2. **MNDWI**:
   - Xu, H. (2006). "Modification of normalised difference water index (NDWI)"
   - International Journal of Remote Sensing

3. **AWEI**:
   - Feyisa et al. (2014). "Automated Water Extraction Index"
   - Remote Sensing

4. **CoastSat Framework**:
   - Vos et al. (2019). "CoastSat: A Google Earth Engine-enabled Python toolkit"
   - Environmental Modelling & Software

### Similar Projects

- **CoastSat**: https://github.com/kvos/CoastSat
- **Satellite-derived shorelines**: http://coastsat.space/

---

## 🚀 Future Enhancements

### Potential Additions (Not Yet Implemented):

1. **Sub-pixel Edge Detection**
   - Canny edge detection on NDWI gradient
   - Sub-pixel shoreline refinement

2. **Time-Series Analysis**
   - Multi-temporal change tracking
   - Seasonal trend analysis
   - Erosion/accretion rate calculation

3. **Machine Learning Classification**
   - Random Forest water/land classifier
   - Training on user-provided ground truth

4. **Higher Resolution Data Integration**
   - PlanetScope (3m) support for detailed analysis
   - Commercial imagery APIs (0.5m resolution)

5. **Uncertainty Quantification**
   - Confidence intervals for change areas
   - Geolocation error propagation
   - Tidal uncertainty mapping

---

## 📝 Version History

- **v2.0.0** (Nov 6, 2025): Major algorithm improvements
  - Added 3 new detection methods
  - Morphological refinement
  - Shoreline smoothing
  - Multi-index consensus
  - Enhanced UI with method selection

- **v1.0.0** (Previous): Initial release
  - NDWI with fixed threshold
  - Basic cloud masking
  - RGB composite visualization
  - Change detection framework

---

## 💡 Tips for Best Results

1. **Keep AOI small**: Large areas slow processing and increase memory usage
2. **Lower cloud threshold**: Use 20-30% for clearer results
3. **Compare similar tidal states**: Large tidal differences create false changes
4. **Use longer time gaps**: 6-12 month gaps show clearer trends than 1-2 months
5. **Try different methods**: If one method fails, another might work better
6. **Adjust post-processing**: Increase morphology/smoothing if results are noisy
7. **Check satellite imagery**: Always visually inspect the RGB composites
8. **Consider seasons**: Seasonal vegetation, ice, or water clarity affects detection

---

## ⚠️ Known Limitations

1. **Resolution**: 10m pixels limit detection of changes <20-30m
2. **Tidal effects**: Shoreline position varies with tide (not corrected)
3. **Wave effects**: Breaking waves and surf zone confusion
4. **Turbidity**: Very turbid water may not be detected (try Multi-Index)
5. **Shadows**: Mountain/cloud shadows can create false water signals
6. **Ice/Snow**: May be misclassified as water (check winter scenes manually)
7. **Shallow water**: Mixed pixels at water edge reduce accuracy
8. **Geolocation**: ±5-10m sensor positioning error

---

## 📧 Support

For issues or questions:
- Check IMPROVEMENTS.md (this file)
- Review app tooltips and help text
- Experiment with different detection methods
- Try adjusting post-processing parameters

**Remember**: Algorithm choice matters! Each method has strengths for different conditions.
