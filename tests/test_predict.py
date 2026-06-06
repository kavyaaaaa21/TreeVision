"""
tests/test_predict.py — Smoke tests for species.py and predict.py

Run with:
    pytest tests/ -v

These tests do NOT require the model weights to pass — they cover
all logic that doesn't need ultralytics loaded.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from species import (
    CLASS_NAMES, CLASS_IDS, ALL_SPECIES,
    SPECIES_COLORS, CONF_MIN, CONF_AUTO,
    get_status, get_color, class_name,
    STATUS_AUTO, STATUS_REVIEW, Detection,
)


# ── species.py tests ──────────────────────────────────────────────────────────

class TestSpeciesConfig:

    def test_class_names_has_10_entries(self):
        assert len(CLASS_NAMES) == 10

    def test_class_ids_is_reverse_of_class_names(self):
        for k, v in CLASS_NAMES.items():
            assert CLASS_IDS[v] == k

    def test_all_species_sorted(self):
        assert ALL_SPECIES == sorted(ALL_SPECIES)

    def test_every_species_has_color(self):
        for sp in CLASS_NAMES.values():
            assert sp in SPECIES_COLORS, f"Missing color for {sp}"

    def test_colors_are_hex(self):
        for sp, color in SPECIES_COLORS.items():
            assert color.startswith("#"), f"{sp} color not hex: {color}"
            assert len(color) == 7, f"{sp} color wrong length: {color}"

    def test_conf_thresholds_order(self):
        assert 0 < CONF_MIN < CONF_AUTO < 1

    def test_get_status_auto(self):
        assert get_status(0.95) == STATUS_AUTO
        assert get_status(0.80) == STATUS_AUTO

    def test_get_status_review(self):
        assert get_status(0.79) == STATUS_REVIEW
        assert get_status(0.35) == STATUS_REVIEW

    def test_get_color_known_species(self):
        assert get_color("Mango") == SPECIES_COLORS["Mango"]

    def test_get_color_unknown_falls_back(self):
        assert get_color("Unicorn") == SPECIES_COLORS["Other"]

    def test_class_name_known(self):
        assert class_name(0) == "Mango"

    def test_class_name_unknown_returns_other(self):
        assert class_name(999) == "Other"


# ── predict.py tests (no model required) ─────────────────────────────────────

class TestPredictModule:

    def test_is_loaded_false_before_load(self):
        import predict
        predict._model = None
        assert predict.is_loaded() is False

    def test_run_prediction_raises_without_model(self):
        import predict
        import numpy as np
        predict._model = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            predict.run_prediction(np.zeros((100, 100, 3), dtype="uint8"), "test.jpg")

    def test_summarise_empty(self):
        import predict
        result = predict.summarise([])
        assert result["total"] == 0
        assert result["avg_confidence"] == 0.0
        assert result["species_counts"] == {}

    def test_summarise_counts(self):
        import predict
        dets = [
            Detection("a1", [0,0,10,10], [5,5], "Mango",  0, 0.95, STATUS_AUTO,   [[0,0],[10,0],[10,10],[0,10]], 100, "#F59E0B"),
            Detection("a2", [0,0,10,10], [5,5], "Mango",  0, 0.90, STATUS_AUTO,   [[0,0],[10,0],[10,10],[0,10]], 100, "#F59E0B"),
            Detection("a3", [0,0,10,10], [5,5], "Neem",   1, 0.60, STATUS_REVIEW, [[0,0],[10,0],[10,10],[0,10]], 100, "#10B981"),
        ]
        s = predict.summarise(dets)
        assert s["total"]           == 3
        assert s["auto_accepted"]   == 2
        assert s["review_required"] == 1
        assert s["species_counts"]["Mango"] == 2
        assert s["species_counts"]["Neem"]  == 1
        assert abs(s["avg_confidence"] - (0.95 + 0.90 + 0.60) / 3) < 0.001