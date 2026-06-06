"""
tests/test_geo.py — Unit tests for geo.py

Tests crown metric computation and feature building.
Does not require rasterio or geopandas to run the metric tests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pytest
from geo import compute_metrics, detection_to_feature, export_csv
from species import Detection, STATUS_AUTO


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_square_bbox(self):
        m = compute_metrics([0, 0, 100, 100])
        assert m["crown_area_px"]  == 10000
        assert m["perimeter_px"]   == 400.0
        assert m["width_px"]       == 100
        assert m["height_px"]      == 100
        # Square circularity: 4π×10000/400² ≈ 0.785
        assert abs(m["circularity"] - (4 * math.pi * 10000 / (400 ** 2))) < 0.001

    def test_tall_bbox(self):
        m = compute_metrics([0, 0, 10, 100])
        assert m["width_px"]  == 10
        assert m["height_px"] == 100
        # Circularity should be < square (more elongated)
        sq = compute_metrics([0, 0, 50, 50])
        assert m["circularity"] < sq["circularity"]

    def test_zero_area_bbox(self):
        m = compute_metrics([5, 5, 5, 5])
        assert m["crown_area_px"] == 0
        assert m["circularity"]   == 0.0   # no divide by zero

    def test_circularity_max_is_circle_approximated_by_square(self):
        # Perfect circle → circularity = 1.0
        # A square approximates it at ~0.785
        m = compute_metrics([0, 0, 100, 100])
        assert m["circularity"] < 1.0


# ── detection_to_feature ──────────────────────────────────────────────────────

class TestDetectionToFeature:

    def _make_det(self):
        return Detection(
            id="ab12cd34",
            bbox=[10, 20, 60, 80],
            center=[35.0, 50.0],
            species="Mango",
            class_id=0,
            confidence=0.92,
            status=STATUS_AUTO,
            crown_polygon=[[10,20],[60,20],[60,80],[10,80],[10,20]],
            crown_area_px=3000,
            color="#F59E0B",
        )

    def test_feature_has_required_keys(self):
        feat = detection_to_feature(self._make_det())
        for key in ["id","species","confidence","status","color",
                    "bbox","crown_area_px","circularity","georeferenced"]:
            assert key in feat, f"Missing key: {key}"

    def test_feature_not_georeferenced_without_transform(self):
        feat = detection_to_feature(self._make_det(), transform=None, crs=None)
        assert feat["georeferenced"] is False
        assert feat["crown_polygon_latlon"] is None
        assert feat["center_latlon"] is None

    def test_feature_metrics_match_bbox(self):
        det  = self._make_det()
        feat = detection_to_feature(det)
        expected = compute_metrics(det.bbox)
        assert feat["crown_area_px"] == expected["crown_area_px"]
        assert feat["circularity"]   == expected["circularity"]


# ── export_csv ────────────────────────────────────────────────────────────────

class TestExportCSV:

    def _make_feature(self, id="x1"):
        return {
            "id": id, "species": "Neem", "confidence": 0.88,
            "status": "AUTO_ACCEPTED",
            "crown_area_px": 500, "circularity": 0.72,
            "width_px": 25, "height_px": 20,
            "bbox": [5, 10, 30, 30],
            "center_latlon": [18.52, 73.85],
        }

    def test_csv_created(self, tmp_path):
        out = tmp_path / "test_out.csv"
        result = export_csv([self._make_feature()], out)
        assert result is True
        assert out.exists()

    def test_csv_has_header(self, tmp_path):
        out = tmp_path / "test_out.csv"
        export_csv([self._make_feature()], out)
        content = out.read_text()
        assert "species" in content
        assert "confidence" in content
        assert "circularity" in content

    def test_csv_empty_features_returns_false(self, tmp_path):
        out = tmp_path / "empty.csv"
        result = export_csv([], out)
        assert result is False

    def test_csv_multiple_rows(self, tmp_path):
        out = tmp_path / "multi.csv"
        feats = [self._make_feature("a1"), self._make_feature("a2"), self._make_feature("a3")]
        export_csv(feats, out)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 4   # 1 header + 3 data rows