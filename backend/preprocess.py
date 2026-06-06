"""
preprocess.py — Image loading and preprocessing pipeline.

Handles:
  - Standard JPEG/PNG via Pillow
  - GeoTIFF (1-band, 3-band, 4-band with alpha) via rasterio
  - Returns a normalised uint8 RGB numpy array ready for YOLO
  - Also returns the rasterio transform + CRS if georeferenced
    (used downstream by geo.py to convert pixel coords → lat/lon)
"""

import io
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from PIL import Image

try:
    import rasterio
    import rasterio.transform
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ImageData:
    """Everything downstream modules need from a loaded image."""
    rgb:          np.ndarray          # shape (H, W, 3), dtype uint8
    width:        int
    height:       int
    filename:     str
    georeferenced: bool = False
    transform:    Optional[object] = None   # rasterio Affine transform
    crs:          Optional[object] = None   # rasterio CRS object
    original_bands: int = 3


# ── Main entry point ──────────────────────────────────────────────────────────

def load_image(data: bytes, filename: str) -> ImageData:
    """
    Load image bytes into a normalised RGB numpy array.

    Tries rasterio first for .tif/.tiff files (preserves geo metadata),
    falls back to Pillow for everything else.

    Args:
        data:     raw bytes from the uploaded file
        filename: original filename (used to pick the right loader)

    Returns:
        ImageData with .rgb ready for model inference
    """
    fname = filename.lower()
    is_tiff = fname.endswith((".tif", ".tiff"))

    if is_tiff and RASTERIO_AVAILABLE:
        return _load_tiff(data, filename)

    return _load_pil(data, filename)


# ── TIFF loader ───────────────────────────────────────────────────────────────

def _load_tiff(data: bytes, filename: str) -> ImageData:
    """
    Load a GeoTIFF preserving geospatial metadata.

    Handles:
      - 1-band grayscale (DSM/DTM elevation models)
      - 3-band RGB
      - 4-band RGBA (typical drone orthomosaic output)
    """
    import tempfile, os

    # rasterio needs a file path, not bytes — write to a temp file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path) as src:
            bands      = src.read()          # shape (C, H, W)
            transform  = src.transform
            crs        = src.crs
            n_bands    = bands.shape[0]
            h, w       = bands.shape[1], bands.shape[2]
            georef     = _is_georeferenced(transform)

            rgb = _bands_to_rgb(bands, n_bands)

    finally:
        os.unlink(tmp_path)

    return ImageData(
        rgb=rgb,
        width=w,
        height=h,
        filename=filename,
        georeferenced=georef,
        transform=transform if georef else None,
        crs=crs if georef else None,
        original_bands=n_bands,
    )


def _bands_to_rgb(bands: np.ndarray, n_bands: int) -> np.ndarray:
    """Convert any band configuration to uint8 RGB."""
    if n_bands >= 3:
        # Use first 3 bands (R, G, B) — drop alpha if present
        rgb = np.transpose(bands[:3], (1, 2, 0)).astype(np.float32)
    elif n_bands == 1:
        # Grayscale (elevation model) → duplicate across 3 channels
        gray = bands[0].astype(np.float32)
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        # 2-band edge case — pad with zeros
        r = bands[0].astype(np.float32)
        g = bands[1].astype(np.float32)
        b = np.zeros_like(r)
        rgb = np.stack([r, g, b], axis=-1)

    return _normalize_to_uint8(rgb)


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Stretch values to 0-255 uint8.
    Works for float, uint16 (drone sensors), and already-uint8 data.
    """
    if CV2_AVAILABLE:
        return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Pure numpy fallback
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - mn) / (mx - mn) * 255.0
    return normalized.astype(np.uint8)


def _is_georeferenced(transform) -> bool:
    """
    Return True if the rasterio transform contains real GPS coordinates.
    A default identity transform means the file has no georeference.
    """
    if transform is None:
        return False
    # rasterio's default identity transform has c=0, f=0
    # A georeferenced file will have non-zero top-left coordinates
    return not (transform.c == 0.0 and transform.f == 0.0)


# ── PIL / JPEG loader ─────────────────────────────────────────────────────────

def _load_pil(data: bytes, filename: str) -> ImageData:
    """Load any PIL-supported format (JPEG, PNG, BMP, WebP)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    h, w = rgb.shape[:2]

    return ImageData(
        rgb=rgb,
        width=w,
        height=h,
        filename=filename,
        georeferenced=False,
        original_bands=3,
    )


# ── Utility: pixel → geo coordinate ──────────────────────────────────────────

def pixel_to_latlon(
    px: float, py: float,
    transform,
    crs
) -> tuple[float, float]:
    """
    Convert pixel (col, row) coordinates to (latitude, longitude).

    Args:
        px, py:    pixel column and row
        transform: rasterio Affine transform from ImageData
        crs:       rasterio CRS from ImageData

    Returns:
        (lat, lon) as floats, or (0.0, 0.0) if conversion fails
    """
    if transform is None or crs is None:
        return 0.0, 0.0

    try:
        from pyproj import Transformer

        # Pixel → projected coordinates using the affine transform
        proj_x, proj_y = rasterio.transform.xy(transform, py, px)

        # Project to WGS84 (lat/lon)
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat  = to_wgs84.transform(proj_x, proj_y)
        return round(lat, 8), round(lon, 8)

    except Exception:
        return 0.0, 0.0