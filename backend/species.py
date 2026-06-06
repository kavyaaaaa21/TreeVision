"""
species.py — Single source of truth for all species-related config.
Every other module imports from here. Never hardcode class names or
colors anywhere else.
"""

from dataclasses import dataclass


# ── Confidence thresholds ──────────────────────────────────────────────────────

CONF_MIN       = 0.10   # below this → detection ignored entirely
CONF_AUTO      = 0.70   # at or above this → AUTO_ACCEPTED
                         # between CONF_MIN and CONF_AUTO → REVIEW_REQUIRED


# ── Detection status labels ────────────────────────────────────────────────────

STATUS_AUTO     = "AUTO_ACCEPTED"
STATUS_REVIEW   = "REVIEW_REQUIRED"
STATUS_VERIFIED = "MANUALLY_VERIFIED"


# ── Class mapping (must match the order your model was trained with) ───────────
#    Key   = integer class ID from YOLO output
#    Value = human-readable species name

CLASS_NAMES: dict[int, str] = {
    0: "Mango",
    1: "Neem",
    2: "Coconut",
    3: "Bamboo",
    4: "Teak",
    5: "Banana",
    6: "Guava",
    7: "Palm",
    8: "Peepal",
    9: "Other",
}

# Reverse lookup: name → id  (useful for validation corrections)
CLASS_IDS: dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}

# Sorted list of all valid species names (used to populate dropdowns)
ALL_SPECIES: list[str] = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]


# ── Species alias map ─────────────────────────────────────────────────────────
#    Maps model class names → display names for species the model was NOT
#    explicitly trained on, but reliably misidentifies from a known proxy class.
#
#    How to extend:
#      Add  "<model_class_name>": "<desired_display_name>"  to this dict.
#      predict.py applies these AFTER inference with a single .get() call.
#
#    Current aliases:
#      "Date palm"  → "Coconut"  — both Arecaceae palms; aerial crown shape
#                                  (radial frond star) is indistinguishable
#      "Drumstick"  → "Coconut"  — model's top-scoring proxy for dense
#                                  coconut plantation canopy in orthomosaics

SPECIES_ALIASES: dict[str, str] = {
    "Date palm": "Coconut",
    "Drumstick": "Coconut",
}


# ── Per-species colors (hex) ───────────────────────────────────────────────────
#    Used by the frontend map and legend. Keep visually distinct.

SPECIES_COLORS: dict[str, str] = {
    "Mango":   "#F59E0B",
    "Neem":    "#10B981",
    "Coconut": "#14B8A6",   # teal — distinct from all 33 model classes
    "Bamboo":  "#06B6D4",
    "Teak":    "#EF4444",
    "Banana":  "#F97316",
    "Guava":   "#EC4899",
    "Palm":    "#84CC16",
    "Peepal":  "#6366F1",
    "Other":   "#94A3B8",
}


# ── Helper dataclass for a single detection ───────────────────────────────────

@dataclass
class Detection:
    """Represents one tree detection from the YOLO model."""
    id:            str            # short uuid
    bbox:          list[int]      # [x1, y1, x2, y2] in pixel coords
    center:        list[float]    # [cx, cy]
    species:       str
    class_id:      int
    confidence:    float
    status:        str            # AUTO_ACCEPTED | REVIEW_REQUIRED
    crown_polygon: list[list[int]]# [[x,y], ...] pixel coords
    crown_area_px: int
    color:         str            # hex from SPECIES_COLORS
    lat:           float = 0.0    # filled by geo.py if georeferenced
    lon:           float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_status(confidence: float) -> str:
    """Return status string based on confidence value."""
    if confidence >= CONF_AUTO:
        return STATUS_AUTO
    return STATUS_REVIEW


def get_color(species: str) -> str:
    """Return hex color for a species, falling back to Other."""
    return SPECIES_COLORS.get(species, SPECIES_COLORS["Other"])


def class_name(class_id: int) -> str:
    """Return species name for a class id, falling back to Other."""
    return CLASS_NAMES.get(class_id, "Other")