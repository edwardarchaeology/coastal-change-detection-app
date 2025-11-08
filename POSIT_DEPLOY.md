# Deploying to Posit Connect Cloud

This guide covers deploying the Coastal Change Monitor app to **Posit Connect Cloud**.

## 🚀 Quick Deploy

### Prerequisites
- Posit Connect Cloud account ([sign up here](https://connect.posit.cloud/))
- Git repository (already set up)
- Python 3.12+ environment

### Deployment Steps

#### Option 1: Direct from GitHub (Recommended)

1. **Connect your GitHub repository to Posit Connect Cloud:**
   - Log in to [Posit Connect Cloud](https://connect.posit.cloud/)
   - Click **"Publish"** → **"New Content"**
   - Select **"Python - Shiny"**
   - Choose **"Import from Git"**
   - Authenticate with GitHub and select this repository
   - Branch: `main`

2. **Configure the deployment:**
   - **Entry point**: `app_folium.py` (auto-detected from manifest.json)
   - **Python version**: 3.12.x (specified in manifest.json)
   - **Requirements**: `requirements.txt` (auto-detected)

3. **Deploy:**
   - Click **"Deploy"**
   - Posit Connect will automatically install dependencies and start the app
   - Your app will be live at: `https://connect.posit.cloud/content/YOUR-CONTENT-ID/`

#### Option 2: Using `rsconnect-python` CLI

1. **Install the deployment tool:**
   ```bash
   pip install rsconnect-python
   ```

2. **Add your Posit Connect Cloud server:**
   ```bash
   rsconnect add --account YOUR_ACCOUNT --name posit-cloud --server connect.posit.cloud --api-key YOUR_API_KEY
   ```

3. **Deploy from your local machine:**
   ```bash
   rsconnect deploy shiny . --name posit-cloud --entrypoint app_folium.py --title "Coastal Change Monitor"
   ```

### Configuration Notes

**Environment Variables** (if needed):
- Set environment variables in the Posit Connect Cloud UI under **"Vars"** tab
- No API keys are required for basic functionality (uses public Sentinel-2 data)

**Python Version:**
- The app requires Python 3.12+ (specified in `manifest.json`)
- Posit Connect Cloud will automatically use the correct version

**Dependencies:**
- All dependencies are in `requirements.txt`
- Posit Connect will install them automatically during deployment
- Note: Geospatial libraries (rasterio, shapely) may take 2-3 minutes to install

**Memory & Resources:**
- Processing satellite imagery can be memory-intensive
- If you encounter memory issues, consider:
  - Reducing `max_scenes` slider (fewer scenes = less memory)
  - Using smaller bounding boxes
  - Upgrading to a higher-tier Posit Connect plan

### Post-Deployment

**Access Control:**
- Set who can view your app in the **"Access"** tab
- Options: Anyone, Logged in users, Specific collaborators

**Logs & Monitoring:**
- View logs in the **"Logs"** tab
- Monitor performance in the **"Info"** tab

**Updates:**
- Push changes to your GitHub repository
- Posit Connect Cloud will auto-redeploy (if configured)
- Or manually redeploy from the UI

### Troubleshooting

**Deployment fails during dependency installation:**
- Check the logs for specific package errors
- Ensure `requirements.txt` versions are compatible
- Try pinning specific versions if needed

**App crashes with memory errors:**
- Reduce processing parameters (max_scenes, bbox size)
- Contact Posit support for resource allocation

**Geospatial libraries fail to install:**
- Posit Connect Cloud includes system libraries for rasterio/GDAL
- If issues persist, file a support ticket

**Map doesn't load or shows errors:**
- Check browser console for JavaScript errors
- Verify Folium version compatibility (≥0.15.0)

### Support

- **Posit Connect Cloud Docs**: https://docs.posit.co/connect-cloud/
- **Shiny for Python Docs**: https://shiny.posit.co/py/
- **App Issues**: File an issue in this GitHub repository

### Local Testing Before Deploy

Test locally with the same environment:

```bash
# Using uv (recommended)
uv run shiny run app_folium.py --host 127.0.0.1 --port 8000

# Or with pip/venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m shiny run app_folium.py --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and verify all features work before deploying.

---

## 📂 Repository Structure (Post-Cleanup)

```
coastline_app/
├── app_folium.py          # Main Shiny app (entry point)
├── coastal_processor.py   # Core processing logic
├── requirements.txt       # Python dependencies
├── manifest.json          # Posit Connect deployment manifest
├── README.md              # Main documentation
├── POSIT_DEPLOY.md        # This file
├── QUICK_START.md         # User guide
├── pyproject.toml         # Project metadata
├── .python-version        # Python version
├── archive/               # Deprecated/old app versions
├── docs/                  # Migration & development docs
└── .venv/                 # Virtual environment (local only)
```

**Key Files for Deployment:**
- `app_folium.py` - Entry point
- `coastal_processor.py` - Backend logic
- `requirements.txt` - All dependencies
- `manifest.json` - Posit Connect configuration

---

**Ready to deploy!** Follow the steps above and your app will be live on Posit Connect Cloud.
