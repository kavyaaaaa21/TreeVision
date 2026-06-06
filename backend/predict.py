"""
predict.py — YOLO inference engine.

Loads TreeVision_best.pt once at startup and exposes a single
run_prediction() function that every API route calls.

Key behaviours:
  - NMS IoU set to 0.85 so OVERLAPPING tree crowns are NOT suppressed
  - max_det=1000 to detect dense canopy patches fully
  - Detections below SPECIES_CERTAINTY_THRESHOLD are labelled "Other"
    (a box is still drawn — the tree IS there, species is just uncertain)
  - agnostic_nms=False so different-species overlapping boxes are kept
"""

import uuid
from pathlib import Path

import numpy as np

from species import Detection, CONF_MIN, get_status, SPECIES_ALIASES, SPECIES_COLORS

# ── Model state (module-level singleton) ──────────────────────────────────────

_model = None   # ultralytics YOLO instance, loaded once

# Species certainty: if the top-class confidence is below this value
# the tree is real but species ID is unreliable → label "Other"
SPECIES_CERTAINTY_THRESHOLD = 0.45

OTHER_COLOR  = '#94A3B8'   # neutral grey for uncertain detections
OTHER_LABEL  = 'Other'


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(weights_path: str | Path) -> bool:
    """
    Load the YOLO model from weights_path into the module singleton.
    Call this once at server startup. Returns True on success.
    """
    global _model
    weights_path = Path(weights_path)

    if not weights_path.exists():
        print(f"[predict] ❌ Weights file not found: {weights_path}")
        print(f"[predict]    Place TreeVision_best.pt in the weights/ folder.")
        return False

    try:
        from ultralytics import YOLO
        _model = YOLO(str(weights_path))
        _build_color_palette()
        names = list(_model.names.values()) if hasattr(_model, 'names') else []
        print(f"[predict] ✅ Model loaded — {weights_path.name}")
        print(f"[predict]    {len(names)} classes: {names}")
        return True

    except Exception as e:
        print(f"[predict] ❌ Failed to load model: {e}")
        return False


def is_loaded() -> bool:
    return _model is not None


def get_model_classes() -> dict:
    """Return {class_id: name} dict from the loaded model."""
    if _model is not None and hasattr(_model, 'names'):
        return dict(_model.names)
    return {}


# ── Per-species color palette (built once after model loads) ──────────────────

_MODEL_COLORS: dict[str, str] = {}

def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL (0-1 each) → hex color string."""
    if s == 0:
        v = int(l * 255)
        return f'#{v:02x}{v:02x}{v:02x}'

    def _chan(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    r = int(_chan(p, q, h + 1/3) * 255)
    g = int(_chan(p, q, h      ) * 255)
    b = int(_chan(p, q, h - 1/3) * 255)
    return f'#{r:02x}{g:02x}{b:02x}'


def _build_color_palette():
    """Assign a visually distinct color to every model class."""
    global _MODEL_COLORS
    if _model is None or not hasattr(_model, 'names'):
        return

    names = [_model.names[i] for i in sorted(_model.names)]
    golden = 137.508  # golden-angle hue stepping — maximises perceptual distance

    for idx, name in enumerate(names):
        hue = (idx * golden) % 360
        sat = 0.70 + (idx % 3) * 0.08    # 70 – 86 %
        lit = 0.52 + (idx % 5) * 0.04    # 52 – 68 %
        _MODEL_COLORS[name] = _hsl_to_hex(hue / 360, sat, lit)

    _MODEL_COLORS[OTHER_LABEL] = OTHER_COLOR   # always register "Other"

    # Override colors for aliased species so they get their defined color
    # (e.g. Coconut → teal #14B8A6) rather than the auto-generated proxy color.
    for alias_target in set(SPECIES_ALIASES.values()):
        if alias_target in SPECIES_COLORS:
            _MODEL_COLORS[alias_target] = SPECIES_COLORS[alias_target]


# ── Inference ─────────────────────────────────────────────────────────────────

def run_prediction(
    rgb: np.ndarray,
    filename: str,
    conf_min: float = CONF_MIN,
    iou_threshold: float = 0.85,
    max_detections: int = 1000,
) -> list[Detection]:
    """
    Run YOLO inference on a uint8 RGB numpy array.

    Args:
        rgb:            (H, W, 3) uint8 image from preprocess.load_image()
        filename:       original filename (for traceability)
        conf_min:       minimum confidence to include ANY detection
        iou_threshold:  NMS IoU cutoff — higher = more overlapping boxes kept
                        Default 0.85 keeps overlapping/touching tree crowns
        max_detections: upper cap on total detections per image

    Returns:
        List of Detection objects, sorted by confidence descending.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call predict.load_model(path) at startup.")

    results = _model(
        rgb,
        conf=conf_min,
        iou=iou_threshold,       # high → keep overlapping tree boxes
        max_det=max_detections,   # allow dense canopy patches
        agnostic_nms=False,       # class-aware NMS: different species can overlap
        verbose=False,
    )

    if not results or len(results) == 0:
        return []

    detections: list[Detection] = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id     = int(box.cls[0].item())
        confidence = round(float(box.conf[0].item()), 4)

        # Determine species label:
        #   • Confidence ≥ SPECIES_CERTAINTY_THRESHOLD → trust the model class
        #   • Confidence below that → tree is real but species unclear → "Other"
        if confidence >= SPECIES_CERTAINTY_THRESHOLD:
            if hasattr(_model, 'names') and cls_id in _model.names:
                species = _model.names[cls_id]
            else:
                species = OTHER_LABEL
        else:
            species = OTHER_LABEL

        # Apply alias map (e.g. "Date palm" / "Drumstick" → "Coconut").
        # SPECIES_ALIASES is defined in species.py — add/remove entries there,
        # no code change needed here.
        species = SPECIES_ALIASES.get(species, species)

        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)

        # Bounding-box polygon (5 pts, closed clockwise) — used by frontend SVG
        crown_polygon = [
            [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]
        ]

        color = _MODEL_COLORS.get(species, OTHER_COLOR)

        det = Detection(
            id            = str(uuid.uuid4())[:8],
            bbox          = [x1, y1, x2, y2],
            center        = [round(cx, 1), round(cy, 1)],
            species       = species,
            class_id      = cls_id,
            confidence    = confidence,
            status        = get_status(confidence),
            crown_polygon = crown_polygon,
            crown_area_px = area,
            color         = color,
        )
        detections.append(det)

    # Sort: highest confidence first for display priority
    detections.sort(key=lambda d: d.confidence, reverse=True)
    print(f"[predict] {len(detections)} detections in {filename} "
          f"(conf≥{conf_min}, iou≤{iou_threshold})")
    return detections



def render_annotated_image(
    rgb: np.ndarray,
    filename: str,
    conf_min: float = CONF_MIN,
    iou_threshold: float = 0.85,
    max_detections: int = 1000,
) -> bytes:
    """
    Run inference and return a JPEG image with YOLO's own box renderer.
    Boxes are drawn by Ultralytics internally → perfect pixel alignment.
    Returns raw JPEG bytes.
    """
    if _model is None:
        raise RuntimeError("Model not loaded.")

    results = _model(
        rgb,
        conf=conf_min,
        iou=iou_threshold,
        max_det=max_detections,
        agnostic_nms=False,
        verbose=False,
    )

    if not results:
        from PIL import Image as PILImage
        import io
        buf = io.BytesIO()
        PILImage.fromarray(rgb).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    # Use YOLO's built-in plot() — draws species labels + confidence + colored boxes
    annotated_bgr = results[0].plot(
        conf=True,
        line_width=2,
        font_size=12,
    )

    # plot() returns BGR numpy array — convert to RGB then save as JPEG
    from PIL import Image as PILImage
    import io
    annotated_rgb = annotated_bgr[..., ::-1]   # BGR → RGB
    buf = io.BytesIO()
    PILImage.fromarray(annotated_rgb).save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return buf.getvalue()




# ── Summary helpers ───────────────────────────────────────────────────────────

def summarise(detections: list[Detection]) -> dict:
    """Build a summary dict from a list of detections."""
    species_counts: dict[str, int] = {}
    auto = review = 0

    for d in detections:
        species_counts[d.species] = species_counts.get(d.species, 0) + 1
        if d.status == "AUTO_ACCEPTED":
            auto += 1
        else:
            review += 1

    avg_conf = (
        round(sum(d.confidence for d in detections) / len(detections), 3)
        if detections else 0.0
    )

    return {
        "total":           len(detections),
        "auto_accepted":   auto,
        "review_required": review,
        "avg_confidence":  avg_conf,
        "species_counts":  species_counts,
    }
