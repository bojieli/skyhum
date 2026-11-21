import json

import numpy as np
import pytest

from skyhum.errors import UsageError
from skyhum.io.regions import displaced, load_regions, parse_place


def test_named_and_alias():
    a = parse_place("Los Angeles basin")
    b = parse_place("la basin")
    assert a.kind == b.kind == "named" and a.bbox == b.bbox
    assert a.contains(34.05, -118.30) and not a.contains(37.77, -122.42)


def test_every_named_region_is_within_the_find_cap():
    for reg in load_regions():
        assert parse_place(reg["name"]).extent_km() <= 100.0, reg["name"]
        assert reg["country"] != "CN"


def test_circle_units():
    a = parse_place("37.77,-122.42,r=5km")
    b = parse_place("37.77,-122.42,r=500m")
    assert a.circle == (37.77, -122.42, 5.0) and b.circle[2] == 0.5
    assert a.extent_km() == 10.0
    assert a.contains(37.77, -122.42) and not a.contains(37.9, -122.42)


def test_circle_needs_radius_and_valid_coords():
    with pytest.raises(UsageError, match="radius"):
        parse_place("37.77,-122.42")
    with pytest.raises(UsageError):
        parse_place("-122.42,37.77,r=1km")          # longitude first: latitude out of range
    with pytest.raises(UsageError):
        parse_place("37.77,-122.42,r=0km")


def test_bbox():
    a = parse_place("bbox:-122.6,37.2,-121.7,38.0")
    assert a.kind == "bbox" and a.contains(37.5, -122.0)
    with pytest.raises(UsageError):
        parse_place("bbox:-121.7,37.2,-122.6,38.0")
    with pytest.raises(UsageError):
        parse_place("bbox:1,2,3")


def test_geojson_polygon_with_hole(tmp_path):
    outer = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    hole = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6], [0.4, 0.4]]
    p = tmp_path / "a.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [outer, hole]}}]}))
    a = parse_place(f"@{p}")
    assert a.kind == "geojson"
    assert a.contains(0.2, 0.2) and not a.contains(0.5, 0.5) and not a.contains(1.5, 0.5)
    lat, lon = a.grid(10.0)
    assert lat.size > 0 and a.contains(lat, lon).all()


def test_geojson_errors(tmp_path):
    p = tmp_path / "pt.geojson"
    p.write_text(json.dumps({"type": "Point", "coordinates": [0, 0]}))
    with pytest.raises(UsageError):
        parse_place(f"@{p}")
    with pytest.raises(UsageError):
        parse_place(f"@{tmp_path / 'missing.geojson'}")


def test_unknown_place():
    with pytest.raises(UsageError, match="unknown place"):
        parse_place("Atlantis")


def test_displaced_keeps_shape():
    a = parse_place("34.05,-118.30,r=1.5km")
    b = displaced(a, 20.0)
    assert b.circle[2] == 1.5
    from skyhum.io.geo import haversine_km
    assert abs(haversine_km(a.circle[0], a.circle[1], b.circle[0], b.circle[1]) - 20.0) < 0.2


def test_grid_inside_area():
    a = parse_place("34.05,-118.30,r=2km")
    lat, lon = a.grid(0.5)
    assert 40 < lat.size < 70 and np.all(a.contains(lat, lon))
