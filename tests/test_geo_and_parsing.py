"""Unit tests for geometry helpers, number parsing and the labelling rule."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amr_access.geo import (  # noqa: E402
    distance_to_polygons,
    inside_even_odd,
    local_xy_m,
    parse_multipolygon_wkt,
    parse_point,
    polygon_centroid_area,
)
from amr_access.labels import RULE_TABLE, rule_label  # noqa: E402

spec = importlib.util.spec_from_file_location("bba", ROOT / "pipeline/a_build_records/01_build_building_attributes.py")
bba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bba)

# a ~100 m x 100 m square near Midtown with a 20 m x 20 m hole in the middle
LON0, LAT0 = -73.985, 40.75
DLON, DLAT = 100 / (111_320 * np.cos(np.radians(LAT0))), 100 / 110_950


def square(x0, y0, dx, dy):
    pts = [(x0, y0), (x0 + dx, y0), (x0 + dx, y0 + dy), (x0, y0 + dy), (x0, y0)]
    return "(" + ", ".join(f"{x} {y}" for x, y in pts) + ")"


OUTER = square(LON0, LAT0, DLON, DLAT)
HOLE = square(LON0 + 0.4 * DLON, LAT0 + 0.4 * DLAT, 0.2 * DLON, 0.2 * DLAT)
WKT = f"MULTIPOLYGON (({OUTER}, {HOLE}))"


def test_parse_point_formats():
    assert parse_point("{'type': 'Point', 'coordinates': [-73.9, 40.7]}") == (-73.9, 40.7)
    assert parse_point("POINT (-73.9 40.7)") == (-73.9, 40.7)


def test_polygon_area_with_hole():
    polys = parse_multipolygon_wkt(WKT)
    assert len(polys) == 1 and len(polys[0]) == 2
    lon, lat, area = polygon_centroid_area(polys)
    assert area == pytest.approx(100 * 100 - 20 * 20, rel=0.01)
    assert lon == pytest.approx(LON0 + DLON / 2, abs=1e-6)
    assert lat == pytest.approx(LAT0 + DLAT / 2, abs=1e-6)


def test_even_odd_and_distance():
    from matplotlib.path import Path as MPath
    polys = parse_multipolygon_wkt(WKT)
    rings = [MPath(np.column_stack(local_xy_m(r[:, 0], r[:, 1]))) for r in polys[0]]
    centre = np.column_stack(local_xy_m([LON0 + DLON / 2], [LAT0 + DLAT / 2]))
    inside_body = np.column_stack(local_xy_m([LON0 + 0.1 * DLON], [LAT0 + 0.1 * DLAT]))
    assert not inside_even_odd(rings, centre)[0]          # in the hole
    assert inside_even_odd(rings, inside_body)[0]          # in the ring body
    d = distance_to_polygons(np.array([LON0 + DLON / 2, LON0 + 0.1 * DLON, LON0 - 0.5 * DLON]),
                             np.array([LAT0 + DLAT / 2, LAT0 + 0.1 * DLAT, LAT0 + DLAT / 2]), [WKT])
    assert d[0] == pytest.approx(10.0, abs=0.3)            # centre of hole -> 10 m to hole edge
    assert d[1] == 0.0
    assert d[2] == pytest.approx(50.0, abs=0.3)


def test_number_format_detection_and_parsing():
    it = pd.DataFrame({"height_roof": ["1.396", "13,06", "27"], "construction_year": ["1.950", "2.001", "1.900"]})
    assert bba.detect_number_format(it, ["construction_year", "height_roof"]) == "it"
    assert bba.parse_number(it["height_roof"], "it", "h").tolist() == [1396.0, 13.06, 27.0]
    en = pd.DataFrame({"numfloors": ["2.0000000", "1.67"]})
    assert bba.detect_number_format(en, ["numfloors"]) == "en"
    grouped = pd.DataFrame({"bldgarea": ["7,520", "277,073", "1,234,567"], "builtfar": ["0.99", "1.5", "2"]})
    assert bba.detect_number_format(grouped, ["bldgarea", "builtfar"]) == "en_grouped"
    assert bba.parse_number(grouped["bldgarea"], "en_grouped", "a").tolist() == [7520.0, 277073.0, 1234567.0]
    with pytest.raises(ValueError):
        bba.detect_number_format(pd.DataFrame({"x": ["7,520"]}), ["x"])   # ambiguous on its own
    with pytest.raises(ValueError):
        bba.parse_number(pd.Series(["1,5"]), "en", "x")


def test_rule_table():
    t = pd.DataFrame(RULE_TABLE, columns=["s", "g", "r", "acc"])
    got = rule_label(t["s"], t["g"], t["r"], pd.Series([True] * len(t)))
    assert got.tolist() == t["acc"].astype(int).tolist()
    got_unusable = rule_label(t["s"], t["g"], t["r"], pd.Series([False] * len(t)))
    assert got_unusable.isna().all()
