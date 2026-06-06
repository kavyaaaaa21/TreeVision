"""
geo.py — Geospatial processing pipeline.

Takes raw Detection objects (pixel-space bboxes) and:
  1. Converts bounding boxes → Shapely polygons
  2. If the image is georeferenced, reprojects polygon vertices to lat/lon
  3. Computes crown metrics: area, perimeter, circularity
  4. Exports a GeoPackage (.gpkg) for QGIS / GIS tools
  5. Returns GeoJSON-compatible feature dicts for the Leaflet map

Crown metrics explained:
  - crown_area_px:   pixel area of the bounding box
  - perimeter_px:    bbox perimeter in pixels
  - circularity:     4π × area / perimeter²
                     → 1.0 = perfect circle, lower = more elongated
                     Used to color-code crowns on the map
"""

import math
from pathlib import Path
from dataclasses import asdict
from typing import Optional

from species import Detection, STATUS_AUTO

# Optional geo dependencies — gracefully degrade if missing
try:
    from shapely.geometry import Polygon, mapping
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import rasterio.transform
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# ── Crown metrics ─────────────────────────────────────────────────────────────

def compute_metrics(bbox: list[int]) -> dict:
    """
    Compute crown shape metrics from a bounding box [x1, y1, x2, y2].

    Returns dict with:
        crown_area_px, perimeter_px, circularity, width_px, height_px
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    area = w * h
    perim = 2 * (w + h)
    circ = (4 * math.pi * area / (perim ** 2)) if perim > 0 else 0.0

    return {
        "crown_area_px":  area,
        "perimeter_px":   round(perim, 1),
        "circularity":    round(circ, 4),
        "width_px":       w,
        "height_px":      h,
    }


# ── Pixel polygon → geo polygon ───────────────────────────────────────────────

def pixel_polygon_to_latlon(
    polygon_px: list[list[int]],
    transform,
    crs,
) -> Optional[list[list[float]]]:
    """
    Convert a list of pixel [x, y] coords to [[lat, lon], ...].

    Returns None if the image is not georeferenced or conversion fails.
    """
    if transform is None or crs is None or not RASTERIO_AVAILABLE:
        return None

    try:
        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        latlon_coords = []

        for px, py in polygon_px:
            proj_x, proj_y = rasterio.transform.xy(transform, py, px)
            lon, lat = to_wgs84.transform(proj_x, proj_y)
            latlon_coords.append([round(lat, 8), round(lon, 8)])

        return latlon_coords

    except Exception as e:
        print(f"[geo] Pixel→latlon conversion failed: {e}")
        return None


# ── Detection → GeoJSON feature ───────────────────────────────────────────────

def detection_to_feature(
    det: Detection,
    transform=None,
    crs=None,
) -> dict:
    """
    Convert a Detection into a GeoJSON-compatible feature dict
    ready to send to the frontend Leaflet map.

    If the image is georeferenced, polygon coords are real lat/lon.
    Otherwise, they remain in pixel space (Leaflet handles both).
    """
    metrics = compute_metrics(det.bbox)

    # Try to get real geo coords
    latlon_polygon = pixel_polygon_to_latlon(
        det.crown_polygon, transform, crs
    )

    # Crown center in geo coords
    cx, cy = det.center
    latlon_center = pixel_polygon_to_latlon([[cx, cy]], transform, crs)
    latlon_center = latlon_center[0] if latlon_center else None

    feature = {
        "id":               det.id,
        "species":          det.species,
        "class_id":         det.class_id,
        "confidence":       det.confidence,
        "status":           det.status,
        "color":            det.color,

        # Pixel-space (always present)
        "bbox":             det.bbox,
        "center_px":        det.center,
        "crown_polygon_px": det.crown_polygon,

        # Geo-space (present only if georeferenced)
        "georeferenced":    latlon_polygon is not None,
        "crown_polygon_latlon": latlon_polygon,
        "center_latlon":    latlon_center,

        # Metrics
        **metrics,
    }
    return feature


# ── Batch conversion ──────────────────────────────────────────────────────────

def detections_to_features(
    detections: list[Detection],
    transform=None,
    crs=None,
) -> list[dict]:
    """Convert a list of Detections to a list of GeoJSON features."""
    return [
        detection_to_feature(d, transform, crs)
        for d in detections
    ]


# ── GeoPackage export ─────────────────────────────────────────────────────────

def export_geopackage(
    features: list[dict],
    output_path: str | Path,
) -> bool:
    """
    Export crown polygons to a GeoPackage (.gpkg) file.
    Only works when features have latlon polygon coords (georeferenced images).

    Returns True on success, False if geopandas is missing or no geo data.
    """
    if not GEOPANDAS_AVAILABLE or not SHAPELY_AVAILABLE:
        print("[geo] geopandas or shapely not installed — skipping GeoPackage export")
        return False

    geo_features = [f for f in features if f.get("crown_polygon_latlon")]
    if not geo_features:
        print("[geo] No georeferenced features to export")
        return False

    rows = []
    for f in geo_features:
        coords = [(lon, lat) for lat, lon in f["crown_polygon_latlon"]]
        if len(coords) < 3:
            continue

        poly = make_valid(Polygon(coords))
        if poly.is_empty:
            continue

        rows.append({
            "geometry":    poly,
            "id":          f["id"],
            "species":     f["species"],
            "confidence":  f["confidence"],
            "status":      f["status"],
            "crown_area":  f["crown_area_px"],
            "circularity": f["circularity"],
        })

    if not rows:
        print("[geo] All polygons invalid after make_valid — nothing exported")
        return False

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(output_path), driver="GPKG")
    print(f"[geo] ✅ GeoPackage saved → {output_path}  ({len(rows)} crowns)")
    return True


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(features: list[dict], output_path: str | Path) -> bool:
    """
    Export detection results to CSV.
    Works with or without georeferencing.
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not features:
        return False

    fieldnames = [
        "id", "species", "confidence", "status",
        "crown_area_px", "circularity", "width_px", "height_px",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "center_lat", "center_lon",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for feat in features:
            x1, y1, x2, y2 = feat["bbox"]
            center = feat.get("center_latlon") or [0.0, 0.0]
            writer.writerow({
                "id":           feat["id"],
                "species":      feat["species"],
                "confidence":   feat["confidence"],
                "status":       feat["status"],
                "crown_area_px": feat["crown_area_px"],
                "circularity":  feat["circularity"],
                "width_px":     feat["width_px"],
                "height_px":    feat["height_px"],
                "bbox_x1":      x1,
                "bbox_y1":      y1,
                "bbox_x2":      x2,
                "bbox_y2":      y2,
                "center_lat":   center[0],
                "center_lon":   center[1],
            })

    print(f"[geo] ✅ CSV saved → {output_path}  ({len(features)} rows)")
    return True