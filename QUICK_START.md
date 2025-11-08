# Quick Start Guide - Coastal Change Detection

## 🚀 Quick Detection Method Selection

### Simple Decision Tree:

```
Is your coastline...

├─ Clear water, simple beach?
│  └─ Use: "NDWI + Fixed Threshold" (0.0)
│     ⚡ Fastest, works great for most beaches
│
├─ Variable water clarity (sometimes clear, sometimes murky)?
│  └─ Use: "NDWI + Otsu (Auto)"
│     ⚡ Adapts automatically, good default choice
│
├─ Very turbid, shallow, or has mixed conditions?
│  └─ Use: "Multi-Index Consensus"
│     ⚡ Most robust, uses multiple water indices
│
└─ Complex scene with both clear AND turbid water?
   └─ Use: "NDWI + Adaptive"
      ⚡ Local thresholding handles variation
```

---

## ⚙️ Recommended Settings by Use Case

### 📊 Standard Beach Monitoring
```
Detection Method: NDWI + Otsu (Auto)
Refine water mask: ✅ (size: 3)
Smooth shoreline: ✅ (2.0 m)
Cloud cover: 30%
Time range: 30-90 days
```

### 🌊 Erosion/Accretion Analysis
```
Detection Method: NDWI + Otsu (Auto)
Refine water mask: ✅ (size: 4)
Smooth shoreline: ✅ (2.5 m)
Cloud cover: 20%
Compare mode: ✅ (12 months offset)
```

### 🏝️ Island/Reef Mapping
```
Detection Method: Multi-Index Consensus (2 votes)
Refine water mask: ✅ (size: 3)
Smooth shoreline: ✅ (2.0 m)
Cloud cover: 20%
```

### 🏞️ Estuaries/River Mouths (Turbid Water)
```
Detection Method: Multi-Index Consensus (2 votes)
Refine water mask: ✅ (size: 4-5)
Smooth shoreline: ✅ (3.0 m)
Cloud cover: 25%
Threshold: Try lowering if NDWI Fixed used
```

### 🏖️ Sandy Beach Changes
```
Detection Method: NDWI + Fixed Threshold (-0.05)
Refine water mask: ✅ (size: 3)
Smooth shoreline: ✅ (2.0 m)
Cloud cover: 25%
Compare mode: ✅ (6-12 months)
```

### 🌴 Tropical Coastlines (Clear Water)
```
Detection Method: NDWI + Fixed Threshold (0.0)
Refine water mask: ✅ (size: 2-3)
Smooth shoreline: ✅ (1.5-2.0 m)
Cloud cover: 30%
```

---

## 🎯 Parameter Quick Reference

### Detection Methods

| Method | Speed | When to Use |
|--------|-------|-------------|
| **Fixed Threshold** | ⚡⚡⚡ | Simple, clear conditions |
| **Otsu Auto** | ⚡⚡ | Default choice, adapts to scene |
| **Adaptive** | ⚡ | Complex mixed conditions |
| **Multi-Index** | ⚡ | Challenging/turbid water |

### Refinement Strength (Morphology Kernel)

| Size | Effect | When to Use |
|------|--------|-------------|
| **2-3** | Gentle | Clean scenes, preserve detail |
| **4-5** | Moderate | Standard noisy scenes |
| **6-7** | Strong | Very noisy, remove artifacts |

### Smoothing Tolerance

| Value | Effect | When to Use |
|-------|--------|-------------|
| **0-1m** | Minimal | Preserve small features |
| **2-3m** | Moderate | Default, balanced |
| **4-10m** | Strong | Remove all small details |

---

## 🔍 Troubleshooting Guide

### Problem: Shoreline looks too jagged/noisy
**Solution:**
1. Increase morphology kernel size (4-5)
2. Increase smoothing tolerance (3-5m)
3. Try Otsu Auto method

### Problem: Missing water in shallow areas
**Solution:**
1. Try Multi-Index Consensus
2. Lower fixed threshold to -0.1 or -0.15
3. Reduce morphology kernel size (2-3)

### Problem: Too much "water" detected (false positives)
**Solution:**
1. Try Multi-Index with 3-vote consensus
2. Increase fixed threshold to 0.1
3. Increase morphology kernel size (5-6)

### Problem: Results vary too much between scenes
**Solution:**
1. Use Otsu Auto instead of Fixed
2. Enable morphology refinement
3. Increase smoothing tolerance
4. Use longer time windows (more scenes averaged)

### Problem: Shorelines look identical (no change detected)
**Solution:**
1. Check RGB imagery - is there actual change?
2. Try Multi-Index method
3. Lower threshold (Fixed: -0.15)
4. Use longer time offset (12+ months)
5. Verify tidal states are similar

### Problem: Too slow, processing takes forever
**Solution:**
1. Zoom in to smaller area
2. Use Fixed Threshold (fastest)
3. Reduce max cloud cover (fewer scenes)
4. Disable morphology/smoothing temporarily

---

## 📏 Understanding Resolution Limits

### What Can Be Detected?

| Change Size | Reliability | Example |
|-------------|-------------|---------|
| **>50m** | ✅ Excellent | Beach erosion, new sand bars |
| **30-50m** | ✅ Good | Small dune changes, shoreline shifts |
| **20-30m** | ⚠️ Possible | Large building, major wave cut |
| **10-20m** | ⚠️ Uncertain | House-scale, small structures |
| **<10m** | ❌ Unreliable | Sub-pixel, individual trees |

### Reality Check:
- **Sentinel-2 pixel**: 10m × 10m = 100m² area
- **Geolocation error**: ±5-10m (half to one pixel)
- **Tidal variation**: ±1-2m vertical, 10-40m horizontal on gentle slopes
- **Wave run-up**: 5-20m depending on conditions

**Bottom Line**: Changes smaller than 20-30m (2-3 pixels) are at the edge of reliable detection.

---

## 🎨 Reading the Results

### Single Window Analysis

**Water Mask (Light Blue)**: All detected water areas
**Shoreline (Blue Line)**: Boundary between water and land
**RGB Composite**: Actual satellite photo

### Change Detection

**Green Areas**: Land gained (progradation/accretion)
- Water disappeared between past and current
- Sand deposition, beach building
- Positive for land

**Red Areas**: Land lost (retreat/erosion)
- New water appeared
- Beach erosion, submergence
- Negative for land

**Blue Line**: Current shoreline position
**Red Line**: Past shoreline position

---

## 💡 Pro Tips

1. **Start with Otsu**: It's the best default for most situations

2. **Compare similar seasons**: Winter vs winter, summer vs summer

3. **Use longer time gaps**: 6-12 months shows clearer trends than 1-2 months

4. **Check the satellite images**: Always visually inspect RGB composites

5. **Experiment with methods**: Different algorithms work better for different coasts

6. **Keep it small**: Process small areas for faster results

7. **Download data**: Save GeoJSON for use in QGIS/ArcGIS

8. **Multiple dates**: Run analysis for several time periods to see trends

9. **Document settings**: Note which method worked best for your area

10. **Tide matters**: Similar tidal states give more accurate change detection

---

## 📚 Learn More

- **Full documentation**: See `IMPROVEMENTS.md`
- **CoastSat project**: https://github.com/kvos/CoastSat
- **Sentinel-2 info**: https://sentinel.esa.int/web/sentinel/missions/sentinel-2

---

## 🆘 Common Questions

**Q: Why are my results noisy?**
A: Increase morphology kernel (4-5) and smoothing (3-5m)

**Q: Why is processing slow?**
A: Zoom in to smaller area, use Fixed Threshold method

**Q: Which method is best?**
A: Start with "Otsu Auto" - it works well for most coastlines

**Q: Can I detect a single house that fell into the ocean?**
A: Probably not reliably - houses are ~10-15m (1-1.5 pixels). Try zooming way in and using Multi-Index method, but expect high uncertainty.

**Q: How do I get more accurate results?**
A: Use Multi-Index Consensus with refinement enabled. But remember - 10m resolution has physical limits.

**Q: Should I smooth the shoreline?**
A: Yes, 2-3m smoothing removes pixel noise while preserving real features

**Q: What if clouds cover my area?**
A: Lower the cloud threshold or try different dates. The app automatically picks least-cloudy scenes.

---

**Remember**: Experiment with settings! Each coastline is unique, and different methods work better in different conditions.
