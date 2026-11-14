"""Lightweight geometry helpers.

The published records only need point geometries, point-in-polygon tests and
short distances, so these helpers avoid a hard dependency on GeoPandas/Shapely.
When Shapely is installed it is used for parsing WKT polygons; otherwise only
point inputs are supported.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

EARTH_RADIUS_M = 6_371_008.8

_POINT_WKT = re.compile(r"POINT\s*\(\s*([-0-9.eE]+)\s+([-0-9.eE]+)\s*\)", re.IGNORECASE)


def parse_point(value) -> tuple[float, float]:
    """Parse a Socrata point (dict-like string) or WKT POINT into (lon, lat)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (np.nan, np.nan)
    text = str(value).strip()
    if text.startswith("{"):
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            obj = json.loads(text)
        lon, lat = obj["coordinates"][:2]
        return float(lon), float(lat)
    match = _POINT_WKT.search(text)
    if match:
        return float(match.group(1)), float(match.group(2))
    try:  # polygon WKT: fall back to Shapely centroid in a projected CRS
        from shapely import wkt  # type: ignore
        geom = wkt.loads(text)
        c = geom.centroid
        return float(c.x), float(c.y)
    except Exception:  # noqa: BLE001
        return (np.nan, np.nan)


_NUM_PAIR = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_multipolygon_wkt(text: str) -> list[list[np.ndarray]]:
    """Parse (MULTI)POLYGON WKT into a list of polygons, each a list of rings (lon, lat arrays)."""
    if not isinstance(text, str) or "POLYGON" not in text.upper():
        return []
    body = text[text.index("("):]
    polys = []
    # split polygons on ')), ((' and rings on '), ('
    for poly_txt in re.split(r"\)\s*\)\s*,\s*\(\s*\(", body):
        rings = []
        for ring_txt in re.split(r"\)\s*,\s*\(", poly_txt):
            pts = np.array(_NUM_PAIR.findall(ring_txt), dtype=float)
            if len(pts) >= 3:
                rings.append(pts)
        if rings:
            polys.append(rings)
    return polys


def polygon_centroid_area(polys: list[list[np.ndarray]]) -> tuple[float, float, float]:
    """Area-weighted centroid (lon, lat) and planar area (m^2) of a multipolygon.

    Holes are subtracted. Computed in a local equirectangular projection, which
    is accurate to <0.1 % for building-sized features in New York City.
    """
    if not polys:
        return (np.nan, np.nan, np.nan)
    lat0 = float(np.mean(polys[0][0][:, 1]))
    lon0 = float(np.mean(polys[0][0][:, 0]))
    kx = math.radians(1) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    ky = math.radians(1) * EARTH_RADIUS_M
    a_tot = cx_tot = cy_tot = 0.0
    for rings in polys:
        for k, ring in enumerate(rings):
            x = (ring[:, 0] - lon0) * kx
            y = (ring[:, 1] - lat0) * ky
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cross = x * y1 - x1 * y
            a = 0.5 * cross.sum()
            if a == 0:
                continue
            cx = ((x + x1) * cross).sum() / (6 * a)
            cy = ((y + y1) * cross).sum() / (6 * a)
            sign = 1.0 if k == 0 else -1.0          # exterior ring first, then holes
            a_abs = abs(a) * sign
            a_tot += a_abs
            cx_tot += cx * a_abs
            cy_tot += cy * a_abs
    if a_tot == 0:
        return (np.nan, np.nan, np.nan)
    return (lon0 + cx_tot / a_tot / kx, lat0 + cy_tot / a_tot / ky, a_tot)


def distance_to_polygons(lon, lat, polygons_wkt, max_piece_m: float = 2.0, k: int = 24) -> np.ndarray:
    """Planar distance (m) from each point to the nearest polygon; 0 if the point lies inside one.

    Polygon boundaries are split into pieces no longer than ``max_piece_m``; the
    ``k`` pieces with the nearest midpoints are evaluated exactly (point-to-segment
    distance). Equivalent to a nearest-feature join in a projected CRS, without
    requiring GeoPandas.
    """
    from scipy.spatial import cKDTree

    ax_list, ay_list, bx_list, by_list, owner = [], [], [], [], []
    ring_paths: list[list[MplPath]] = []
    for pid, text in enumerate(polygons_wkt):
        polys = parse_multipolygon_wkt(text)
        rp = []
        for rings in polys:
            for ring in rings:
                x, y = local_xy_m(ring[:, 0], ring[:, 1])
                rp.append(MplPath(np.column_stack([x, y])))
                x0, y0, x1, y1 = x[:-1], y[:-1], x[1:], y[1:]
                seg_len = np.hypot(x1 - x0, y1 - y0)
                n_piece = np.maximum(1, np.ceil(seg_len / max_piece_m).astype(int))
                for xa, ya, xb, yb, n in zip(x0, y0, x1, y1, n_piece):
                    t = np.linspace(0, 1, n + 1)
                    px, py = xa + (xb - xa) * t, ya + (yb - ya) * t
                    ax_list.append(px[:-1]); ay_list.append(py[:-1]); bx_list.append(px[1:]); by_list.append(py[1:])
                    owner.append(np.full(n, pid))
        ring_paths.append(rp)
    ax, ay = np.concatenate(ax_list), np.concatenate(ay_list)
    bx, by = np.concatenate(bx_list), np.concatenate(by_list)
    owner = np.concatenate(owner)
    tree = cKDTree(np.column_stack([(ax + bx) / 2, (ay + by) / 2]))

    px, py = local_xy_m(lon, lat)
    out = np.full(len(px), np.nan)
    ok = np.isfinite(px) & np.isfinite(py)
    _, idx = tree.query(np.column_stack([px[ok], py[ok]]), k=k)
    qx, qy = px[ok][:, None], py[ok][:, None]
    sx, sy, ex, ey = ax[idx], ay[idx], bx[idx], by[idx]
    dx, dy = ex - sx, ey - sy
    denom = np.where(dx * dx + dy * dy == 0, 1, dx * dx + dy * dy)
    t = np.clip(((qx - sx) * dx + (qy - sy) * dy) / denom, 0, 1)
    d = np.hypot(qx - (sx + t * dx), qy - (sy + t * dy))
    best = d.min(axis=1)
    # zero distance when the point lies inside one of the nearest polygons
    # (even-odd rule over rings, so that holes are handled correctly)
    rows = np.flatnonzero(ok)
    for j, (r, cand) in enumerate(zip(rows, owner[idx])):
        for pid in np.unique(cand):
            if inside_even_odd(ring_paths[pid], np.array([[px[r], py[r]]]))[0]:
                best[j] = 0.0
                break
    out[ok] = best
    return out


def inside_even_odd(rings: list[MplPath], pts: np.ndarray) -> np.ndarray:
    """Point-in-polygon with holes: inside if the point is inside an odd number of rings."""
    count = np.zeros(len(pts), dtype=int)
    for ring in rings:
        count += ring.contains_points(pts)
    return count % 2 == 1


def parse_multilinestring_wkt(text: str) -> list[np.ndarray]:
    """Parse (MULTI)LINESTRING WKT into a list of (lon, lat) vertex arrays."""
    if not isinstance(text, str) or "LINESTRING" not in text.upper():
        return []
    body = text[text.index("("):]
    lines = []
    for part in re.split(r"\)\s*,\s*\(", body):
        pts = np.array(_NUM_PAIR.findall(part), dtype=float)
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def distance_to_lines(lon, lat, lines_wkt, max_piece_m: float = 2.0, k: int = 24) -> np.ndarray:
    """Planar distance (m) from each point to the nearest (multi)linestring."""
    from scipy.spatial import cKDTree

    ax_l, ay_l, bx_l, by_l = [], [], [], []
    for text in lines_wkt:
        for line in parse_multilinestring_wkt(text):
            x, y = local_xy_m(line[:, 0], line[:, 1])
            for xa, ya, xb, yb in zip(x[:-1], y[:-1], x[1:], y[1:]):
                n = max(1, int(np.ceil(np.hypot(xb - xa, yb - ya) / max_piece_m)))
                t = np.linspace(0, 1, n + 1)
                px, py = xa + (xb - xa) * t, ya + (yb - ya) * t
                ax_l.append(px[:-1]); ay_l.append(py[:-1]); bx_l.append(px[1:]); by_l.append(py[1:])
    ax, ay, bx, by = map(np.concatenate, (ax_l, ay_l, bx_l, by_l))
    tree = cKDTree(np.column_stack([(ax + bx) / 2, (ay + by) / 2]))
    px, py = local_xy_m(lon, lat)
    out = np.full(len(px), np.nan)
    ok = np.isfinite(px) & np.isfinite(py)
    _, idx = tree.query(np.column_stack([px[ok], py[ok]]), k=k)
    qx, qy = px[ok][:, None], py[ok][:, None]
    sx, sy, dx, dy = ax[idx], ay[idx], bx[idx] - ax[idx], by[idx] - ay[idx]
    denom = np.where(dx * dx + dy * dy == 0, 1, dx * dx + dy * dy)
    t = np.clip(((qx - sx) * dx + (qy - sy) * dy) / denom, 0, 1)
    out[ok] = np.hypot(qx - (sx + t * dx), qy - (sy + t * dy)).min(axis=1)
    return out


def parse_points(series: pd.Series) -> pd.DataFrame:
    coords = [parse_point(v) for v in series]
    return pd.DataFrame(coords, columns=["lon", "lat"], index=series.index)


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def local_xy_m(lon, lat, lon0: float = -73.97, lat0: float = 40.78):
    """Equirectangular projection around Manhattan (error < 0.1 % over 30 km)."""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_M
    return x, y


def load_polygons(geojson_path: Path, name_field: str, code_field: str | None = None):
    """Load (Multi)Polygon features as a list of (name, code, [MplPath...], bbox)."""
    path = Path(geojson_path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for feat in data["features"]:
        geom = feat["geometry"]
        props = feat["properties"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        paths = []
        xs, ys = [], []
        for poly in polys:
            for ring in poly:  # one simple path per ring; holes resolved by even-odd parity
                ring = np.asarray(ring, dtype=float)[:, :2]
                paths.append(MplPath(ring))
                xs.extend(ring[:, 0]); ys.extend(ring[:, 1])
        bbox = (min(xs), min(ys), max(xs), max(ys))
        out.append((props.get(name_field), props.get(code_field) if code_field else None, paths, bbox, props))
    return out


def points_in_polygons(lon: np.ndarray, lat: np.ndarray, polygons) -> tuple[np.ndarray, np.ndarray]:
    """Return (name, code) of the first polygon containing each point (object arrays, None if none)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    names = np.full(lon.shape, None, dtype=object)
    codes = np.full(lon.shape, None, dtype=object)
    pts = np.column_stack([lon, lat])
    valid = np.isfinite(lon) & np.isfinite(lat)
    for name, code, paths, (x0, y0, x1, y1), _ in polygons:
        cand = valid & (names == None) & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)  # noqa: E711
        if not cand.any():
            continue
        idx = np.flatnonzero(cand)
        inside = inside_even_odd(paths, pts[idx])
        names[idx[inside]] = name
        codes[idx[inside]] = code
    return names, codes


def nearest_neighbour(src_lon, src_lat, dst_lon, dst_lat, max_dist_m: float = np.inf):
    """Index of the nearest destination point and its distance (m) for each source point.

    Returns -1 / NaN where no destination lies within ``max_dist_m`` or the
    source coordinates are missing.
    """
    from scipy.spatial import cKDTree

    sx, sy = local_xy_m(src_lon, src_lat)
    dx, dy = local_xy_m(dst_lon, dst_lat)
    dst_ok = np.isfinite(dx) & np.isfinite(dy)
    dst_index = np.flatnonzero(dst_ok)
    tree = cKDTree(np.column_stack([dx[dst_ok], dy[dst_ok]]))
    src = np.column_stack([sx, sy])
    src_ok = np.isfinite(sx) & np.isfinite(sy)
    best_idx = np.full(len(sx), -1, dtype=int)
    best_d = np.full(len(sx), np.nan)
    if src_ok.any():
        d, k = tree.query(src[src_ok], distance_upper_bound=max_dist_m)
        hit = np.isfinite(d)
        rows = np.flatnonzero(src_ok)[hit]
        best_idx[rows] = dst_index[k[hit]]
        best_d[rows] = d[hit]
    return best_idx, best_d
