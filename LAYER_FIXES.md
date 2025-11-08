# Map Layer Visibility Fixes

## Problem
Map layers (water bodies, shorelines, change areas) were not showing up reliably when toggling checkboxes. Layers would sometimes appear, but often remained hidden or seemed to be underneath the basemap.

## Root Cause
The issue was caused by several factors:

1. **Color visibility**: Original colors (dark blue #0066cc, dark green #2ca02c) were too dark to see clearly over satellite imagery, especially over water
2. **Layer opacity**: Low opacity values made layers nearly invisible
3. **Line thickness**: Thin lines (2-3px) were hard to see
4. **No layer names**: Layers didn't have names, making debugging difficult
5. **Missing LayersControl**: No in-map control to toggle layers

## Solutions Applied

### 1. Brighter, High-Contrast Colors

**Snapshot Mode:**
- Shoreline: Changed from `#0066cc` (dark blue) to `#FFD700` (bright yellow/gold)
- Water bodies: Increased opacity from 0.35 to 0.5, kept blue `#3399ff`

**Change Detection Mode:**
- Progradation (land gained): Changed from `#2ca02c` (dark green) to `#00ff00` (bright green)
- Retreat (land lost): Changed from `#d62728` (dark red) to `#ff0000` (bright red)
- Current shoreline: Changed from `#0066cc` (dark blue) to `#00FFFF` (cyan)
- Past shoreline: Changed from `#8B0000` (dark red) to `#FF00FF` (magenta)
- Water bodies: Increased opacity from 0.2 to 0.4

### 2. Increased Line Thickness
- All shorelines now use `weight: 5` (instead of 2-3)
- Change areas use `weight: 2` (kept, as these are polygons)
- Water bodies use `weight: 2` (instead of 1)

### 3. Increased Opacity
- All layers now have higher fillOpacity (0.4-0.6 instead of 0.2-0.35)
- All lines now have `opacity: 1.0` (fully opaque)

### 4. Added Layer Names
Every layer now has a descriptive `name` property:
- "Water Bodies"
- "Shoreline"  
- "Land Gained (Progradation)"
- "Land Lost (Retreat)"
- "Current Shoreline"
- "Past Shoreline"
- "Current Water"
- "Past Water"

### 5. Added LayersControl
A LayersControl widget is now added to the top-right of the results map, allowing users to toggle layers on/off directly in the map interface.

### 6. Basemap Configuration
The basemap now has `base=True` explicitly set to ensure it stays at the bottom of the layer stack.

### 7. Better Debug Logging
Added comprehensive debug logging to track:
- Which layers are being added
- How many features each layer contains
- Total layer count
- Layer names

## How It Works Now

### Layer Rendering Process
1. Map is created with center and zoom
2. Basemap (satellite imagery) is added FIRST as base layer
3. Data layers are added conditionally based on checkbox state
4. Each layer is given a bright, contrasting color
5. LayersControl is added for in-map toggling
6. Scale and fullscreen controls are added

### User Interaction
Users now have TWO ways to control layer visibility:

1. **Checkboxes in Results tab**: Control which layers are included when map re-renders
2. **LayersControl (map icon in top-right)**: Toggle individual layers on/off without re-rendering

### Why Brighter Colors?
- Satellite imagery is dark over water bodies
- Dark blues, greens, and reds blend into the imagery
- Bright neon colors (cyan, magenta, yellow, bright green/red) stand out clearly
- High opacity ensures features are visible even over varied backgrounds

## Color Legend (Updated)

### Snapshot Mode
- 🟡 **Yellow** (#FFD700) = Shoreline (thick line)
- 💙 **Blue** (#3399ff, 50% opacity) = Water bodies

### Change Detection Mode
- 🟢 **Bright Green** (#00ff00, 60% opacity) = Land gained (progradation)
- 🔴 **Bright Red** (#ff0000, 60% opacity) = Land lost (retreat)
- 🔵 **Cyan** (#00FFFF, solid line) = Current shoreline
- 🟣 **Magenta** (#FF00FF, dashed line) = Past shoreline
- 💙 **Blue** (#0099ff, 40% opacity) = Current water (optional)
- 🩷 **Pink** (#ff6666, 40% opacity) = Past water (optional)

## Testing
After applying these fixes:
1. Reload the page (Ctrl+F5)
2. Run an analysis
3. Go to Results tab
4. Toggle checkboxes - you should see layers appearing/disappearing
5. Use the LayersControl icon (top-right of map) to toggle layers interactively

## Notes
- The bright colors are intentionally "loud" to ensure visibility over satellite imagery
- Lines are thick (5px) to be visible even when zoomed out
- If you prefer different colors, adjust the `style` dict in the GeoJSON layer creation
- The terminal debug output shows exactly which layers are being added (check for "DEBUG: Adding X features")
