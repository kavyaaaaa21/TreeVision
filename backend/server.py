"""
server.py — FastAPI application.
 
Wires together preprocess → predict → geo and exposes:
  GET  /                       serve index.html
  GET  /api/status             model + library health check
  POST /api/predict            upload image → get detections + features
  POST /api/validate           submit a manual correction
  POST /api/export/csv         download results as CSV
  POST /api/export/gpkg        download GeoPackage (georeferenced only)
 
Run locally:
  uvicorn server:app --reload --port 8000
Then open: http://localhost:8000
"""
 
import os
import io
from pathlib import Path
from datetime import datetime
 
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
 
import predict
import preprocess
import geo
from species import CONF_MIN, CONF_AUTO


# ── Dynamic species registry (rebuilt after model loads) ──────────────────────

def _generate_colors(names: list[str]) -> dict[str, str]:
    """
    Auto-assign visually distinct HSL hex colors to every species.
    Uses golden-angle hue stepping for maximum perceptual separation.
    """
    import math
    golden_angle = 137.508  # degrees
    colors = {}
    for i, name in enumerate(names):
        hue = int((i * golden_angle) % 360)
        # Vary saturation and lightness slightly for more richness
        sat = 70 + (i % 3) * 8          # 70–86%
        lit = 50 + (i % 5) * 4          # 50–66%
        # Convert HSL → hex
        h, s, l = hue / 360, sat / 100, lit / 100
        def hsl2rgb(h, s, l):
            if s == 0:
                return l, l, l
            def hue2rgb(p, q, t):
                if t < 0: t += 1
                if t > 1: t -= 1
                if t < 1/6: return p + (q-p)*6*t
                if t < 1/2: return q
                if t < 2/3: return p + (q-p)*(2/3-t)*6
                return p
            q = l * (1+s) if l < 0.5 else l+s-l*s
            p = 2*l - q
            return hue2rgb(p,q,h+1/3), hue2rgb(p,q,h), hue2rgb(p,q,h-1/3)
        r, g, b = hsl2rgb(h, s, l)
        colors[name] = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    return colors


_DYNAMIC_SPECIES: list[str] = []
_DYNAMIC_COLORS: dict[str, str] = {}
 
 
# ── Paths ─────────────────────────────────────────────────────────────────────
 
BASE_DIR     = Path(__file__).parent
WEIGHTS_PATH = Path(os.getenv("WEIGHTS_PATH", BASE_DIR / "weights" / "TreeVision_best.pt"))
DATA_DIR     = Path(os.getenv("DATA_DIR",     BASE_DIR / "data"))
OUTPUTS_DIR  = DATA_DIR / "outputs"
STATIC_DIR   = BASE_DIR / "static"
 
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
 
 
# ── App setup ─────────────────────────────────────────────────────────────────
 
app = FastAPI(
    title="TreeVision API",
    version="1.0.0",
    description="Tree species detection and crown segmentation dashboard",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
 
 
# ── Startup ───────────────────────────────────────────────────────────────────
 
@app.on_event("startup")
async def startup():
    global _DYNAMIC_SPECIES, _DYNAMIC_COLORS
    print(f"\n{'='*50}")
    print(f"  TreeVision — starting up")
    print(f"{'='*50}")
    print(f"  Weights : {WEIGHTS_PATH}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Outputs : {OUTPUTS_DIR}")
    loaded = predict.load_model(WEIGHTS_PATH)
    if not loaded:
        print("\n  ⚠️  Server running WITHOUT model.")
        print(f"  ⚠️  Place TreeVision_best.pt in: {WEIGHTS_PATH.parent}\n")
    else:
        # Build dynamic species list directly from the model's class names
        model_classes = predict.get_model_classes()   # {0: 'Mango', 1: 'Neem', ...}
        _DYNAMIC_SPECIES = [model_classes[i] for i in sorted(model_classes)]
        # Always include 'Other' for uncertain / overlapping detections
        if 'Other' not in _DYNAMIC_SPECIES:
            _DYNAMIC_SPECIES.append('Other')
        _DYNAMIC_COLORS  = _generate_colors(_DYNAMIC_SPECIES)
        _DYNAMIC_COLORS['Other'] = '#94A3B8'   # fixed neutral grey for Other
        print(f"  Species : {len(_DYNAMIC_SPECIES)} classes loaded from weights (incl. Other)")
    print(f"{'='*50}\n")


 
 
# ── Routes ────────────────────────────────────────────────────────────────────
 
@app.get("/", include_in_schema=False)
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"status": "TreeVision API running — frontend not found in static/"})
 
 
@app.get("/api/status")
def status():
    return {
        "model_loaded":        predict.is_loaded(),
        "weights_path":        str(WEIGHTS_PATH),
        "weights_exist":       WEIGHTS_PATH.exists(),
        "conf_min":            CONF_MIN,
        "conf_auto":           CONF_AUTO,
        "species":             _DYNAMIC_SPECIES,
        "species_colors":      _DYNAMIC_COLORS,
        "rasterio_available":  preprocess.RASTERIO_AVAILABLE,
        "shapely_available":   geo.SHAPELY_AVAILABLE,
        "geopandas_available": geo.GEOPANDAS_AVAILABLE,
    }
 
 
@app.post("/api/annotated-image/by-name")
async def annotated_by_name(body: dict):
    """
    Return a JPEG with YOLO's native bounding-box annotations.
    Body: { "filename": "patch_5.jpg", "conf_min": 0.25 }
    Frontend swaps the raw image for this annotated one after prediction.
    """
    filename = body.get("filename", "")
    conf_min = float(body.get("conf_min", CONF_MIN))

    if not filename:
        raise HTTPException(400, detail="filename required")

    img_path = IMAGES_DIR / filename
    if not img_path.exists():
        raise HTTPException(404, detail=f"Not found: {filename}")

    if not predict.is_loaded():
        raise HTTPException(503, detail="Model not loaded")

    raw = img_path.read_bytes()
    img = preprocess.load_image(raw, filename)

    try:
        jpeg_bytes = predict.render_annotated_image(
            img.rgb, filename, conf_min,
            iou_threshold=0.85, max_detections=1000,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Annotation failed: {e}")

    return StreamingResponse(io.BytesIO(jpeg_bytes), media_type="image/jpeg")


@app.post("/api/annotated-image/upload")
async def annotated_upload(
    file: UploadFile = File(...),
    conf_min: float = CONF_MIN,
):
    """Same as above but for a freshly uploaded file."""
    if not predict.is_loaded():
        raise HTTPException(503, detail="Model not loaded")

    data = await file.read()
    if not data:
        raise HTTPException(400, detail="Empty file")

    img = preprocess.load_image(data, file.filename)

    try:
        jpeg_bytes = predict.render_annotated_image(
            img.rgb, file.filename, conf_min,
            iou_threshold=0.85, max_detections=1000,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Annotation failed: {e}")

    return StreamingResponse(io.BytesIO(jpeg_bytes), media_type="image/jpeg")



@app.post("/api/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    conf_min: float  = CONF_MIN,
):

    if not predict.is_loaded():
        raise HTTPException(503, detail=(
            "Model not loaded. Place TreeVision_best.pt in weights/ and restart."
        ))
 
    data = await file.read()
    if not data:
        raise HTTPException(400, detail="Uploaded file is empty.")
 
    try:
        img = preprocess.load_image(data, file.filename)
    except Exception as e:
        raise HTTPException(422, detail=f"Could not read image: {e}")
 
    try:
        detections = predict.run_prediction(
            img.rgb, file.filename, conf_min,
            iou_threshold=0.85, max_detections=1000,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Inference failed: {e}")
 
    features = geo.detections_to_features(
        detections, transform=img.transform, crs=img.crs,
    )
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = Path(file.filename).stem
    csv_path  = OUTPUTS_DIR / f"{stem}_{timestamp}.csv"
    gpkg_path = OUTPUTS_DIR / f"{stem}_{timestamp}.gpkg"
 
    geo.export_csv(features, csv_path)
    geo.export_geopackage(features, gpkg_path)
 
    return {
        "filename":      file.filename,
        "image_size":    [img.width, img.height],
        "georeferenced": img.georeferenced,
        "summary":       predict.summarise(detections),
        "features":      features,
        "saved_csv":     str(csv_path),
        "saved_gpkg":    str(gpkg_path) if img.georeferenced else None,
    }
 
 
class ValidationPayload(BaseModel):
    id:                str
    corrected_species: str
    original_species:  str
    confidence:        float
    filename:          str
 
 
@app.post("/api/validate")
def validate_endpoint(payload: ValidationPayload):
    # Accept any species in the dynamic list OR 'Other'
    valid_species = set(_DYNAMIC_SPECIES) | {'Other'}
    if payload.corrected_species not in valid_species:
        raise HTTPException(400, detail=f"Unknown species: {payload.corrected_species}")
 
    corrections_path = OUTPUTS_DIR / "corrections.csv"
    import csv
    write_header = not corrections_path.exists()
    with open(corrections_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp","id","filename",
                             "original_species","corrected_species","confidence"])
        writer.writerow([
            datetime.now().isoformat(), payload.id, payload.filename,
            payload.original_species, payload.corrected_species, payload.confidence,
        ])
 
    return {
        "id":                payload.id,
        "corrected_species": payload.corrected_species,
        "status":            "MANUALLY_VERIFIED",
        "saved_to":          str(corrections_path),
    }
 
 
# ── Image library routes ──────────────────────────────────────────────────────
 
IMAGES_DIR = DATA_DIR / "images"
IMAGE_EXTS  = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
 
 
@app.get("/api/images")
def list_images():
    """
    Return metadata for every image file found in data/images/.
    The frontend uses this to build the image gallery.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(IMAGES_DIR.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            stat = p.stat()
            files.append({
                "name":      p.name,
                "stem":      p.stem,
                "ext":       p.suffix.lower(),
                "size_mb":   round(stat.st_size / 1_048_576, 2),
                "is_tiff":   p.suffix.lower() in {".tif", ".tiff"},
                "thumb_url": f"/api/images/{p.name}/thumb",
            })
    return {"images": files, "count": len(files), "directory": str(IMAGES_DIR)}
 
 
@app.get("/api/images/{filename}/thumb")
def image_thumb(filename: str):
    """
    Return a small JPEG thumbnail of any image in data/images/.
    TIFFs are converted on the fly; JPEGs are resized.
    Max size: 280×180 px — good enough for the gallery cards.
    """
    img_path = IMAGES_DIR / filename
    if not img_path.exists() or img_path.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(404, detail=f"Image not found: {filename}")
 
    try:
        raw = img_path.read_bytes()
        img_data = preprocess.load_image(raw, filename)
 
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img_data.rgb)
        pil.thumbnail((280, 180), PILImage.LANCZOS)
 
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
 
    except Exception as e:
        raise HTTPException(500, detail=f"Could not generate thumbnail: {e}")


@app.get("/api/images/{filename}/full")
def image_full(filename: str):
    """
    Serve the full-resolution image from data/images/.
    Used by the frontend detection overlay view.
    """
    img_path = IMAGES_DIR / filename
    if not img_path.exists() or img_path.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(404, detail=f"Image not found: {filename}")

    # For TIFFs, convert to JPEG on the fly (browsers can't display raw TIFF)
    if img_path.suffix.lower() in {".tif", ".tiff"}:
        try:
            raw = img_path.read_bytes()
            img_data = preprocess.load_image(raw, filename)
            from PIL import Image as PILImage
            pil = PILImage.fromarray(img_data.rgb)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(500, detail=f"Could not convert TIFF: {e}")

    # For JPEG/PNG serve directly
    from fastapi.responses import FileResponse
    media = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(str(img_path), media_type=media)


 
 
@app.post("/api/predict-by-name")
async def predict_by_name(body: dict):
    """
    Run prediction on a file already on disk in data/images/.
    Body: { "filename": "tile_row1.tif", "conf_min": 0.35 }
 
    This is the route called when the user clicks an image in the gallery,
    so they don't need to re-upload files that are already on the server.
    """
    filename = body.get("filename", "")
    conf_min = float(body.get("conf_min", CONF_MIN))
 
    if not filename:
        raise HTTPException(400, detail="filename is required.")
 
    img_path = IMAGES_DIR / filename
    if not img_path.exists():
        raise HTTPException(404, detail=f"File not found in data/images/: {filename}")
 
    if not predict.is_loaded():
        raise HTTPException(503, detail="Model not loaded.")
 
    raw = img_path.read_bytes()
 
    try:
        img = preprocess.load_image(raw, filename)
    except Exception as e:
        raise HTTPException(422, detail=f"Could not read image: {e}")
 
    try:
        detections = predict.run_prediction(
            img.rgb, filename, conf_min,
            iou_threshold=0.85, max_detections=1000,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Inference failed: {e}")
 
    features  = geo.detections_to_features(detections, transform=img.transform, crs=img.crs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = Path(filename).stem
    csv_path  = OUTPUTS_DIR / f"{stem}_{timestamp}.csv"
    gpkg_path = OUTPUTS_DIR / f"{stem}_{timestamp}.gpkg"
 
    geo.export_csv(features, csv_path)
    geo.export_geopackage(features, gpkg_path)
 
    return {
        "filename":      filename,
        "image_size":    [img.width, img.height],
        "georeferenced": img.georeferenced,
        "summary":       predict.summarise(detections),
        "features":      features,
        "saved_csv":     str(csv_path),
        "saved_gpkg":    str(gpkg_path) if img.georeferenced else None,
    }
 
 
class ExportPayload(BaseModel):
    features: list[dict]
    filename: str = "export"
 
 
@app.post("/api/export/csv")
def export_csv_endpoint(payload: ExportPayload):
    if not payload.features:
        raise HTTPException(400, detail="No features to export.")
    buffer = io.StringIO()
    import csv
    fieldnames = ["id","species","confidence","status",
                  "crown_area_px","circularity","center_lat","center_lon"]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for f in payload.features:
        center = f.get("center_latlon") or [0.0, 0.0]
        writer.writerow({**f, "center_lat": center[0], "center_lon": center[1]})
    buffer.seek(0)
    fname = f"{payload.filename}_detections.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )
 