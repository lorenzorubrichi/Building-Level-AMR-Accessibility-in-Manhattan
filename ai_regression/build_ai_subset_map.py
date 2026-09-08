from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


BASE_DIR = Path(__file__).resolve().parent
PRIMARY_INPUT_CSV = BASE_DIR / "complete_last_meter_dataset_ai_subset.csv"
FALLBACK_INPUT_CSV = BASE_DIR / "complete_last_meter_dataset_ai_subset_updated.csv"
OUTPUT_HTML = BASE_DIR / "complete_last_meter_dataset_ai_subset_map.html"
OUTPUT_DATA_JS = BASE_DIR / "complete_last_meter_dataset_ai_subset_map.data.js"


def resolve_input_csv() -> Path:
    if PRIMARY_INPUT_CSV.exists() and FALLBACK_INPUT_CSV.exists():
        return FALLBACK_INPUT_CSV if FALLBACK_INPUT_CSV.stat().st_mtime > PRIMARY_INPUT_CSV.stat().st_mtime else PRIMARY_INPUT_CSV
    if PRIMARY_INPUT_CSV.exists():
        return PRIMARY_INPUT_CSV
    return FALLBACK_INPUT_CSV


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def pick_lat_lon(row: dict[str, str]) -> tuple[float | None, float | None]:
    centroid_lat = to_float(row.get("centroid_lat"))
    centroid_lon = to_float(row.get("centroid_lon"))
    address_lat = to_float(row.get("address_lat"))
    address_lon = to_float(row.get("address_lon"))
    if centroid_lat is not None and centroid_lon is not None:
        return centroid_lat, centroid_lon
    return address_lat, address_lon


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    input_csv = resolve_input_csv()
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            lat, lon = pick_lat_lon(raw)
            if lat is None or lon is None:
                continue

            row = {
                "bin": str(raw.get("bin", "")).strip(),
                "borough": str(raw.get("borough", "")).strip(),
                "neighborhood": str(raw.get("neighborhood", "")).strip(),
                "full_address": str(raw.get("full_address", "")).strip(),
                "house_number": str(raw.get("house_number", "")).strip(),
                "street_name": str(raw.get("street_name", "")).strip(),
                "full_street_name": str(raw.get("full_street_name", "")).strip(),
                "zipcode": str(raw.get("zipcode", "")).strip(),
                "lat": lat,
                "lon": lon,
                "car_time": to_float(raw.get("car_last_meter_mean_s")),
                "amr_time": to_float(raw.get("amr_last_meter_mean_s")),
                "usable": to_bool(raw.get("image_usable")),
                "stairs": to_bool(raw.get("stairs_present")),
                "gate": to_bool(raw.get("gate_present")),
                "ramp": to_bool(raw.get("ramp_present")),
                "amr_reach": to_bool(raw.get("amr_can_reach_door")),
                "ai_access_barrier_mean": to_float(raw.get("ai_access_barrier_mean")),
                "floors_norm": to_float(raw.get("a1_Floors_norm")),
                "road_delivery_norm": to_float(raw.get("a2_RoadToDeliveryDistance_norm")),
                "shape_penalty_norm": to_float(raw.get("ShapePenalty_norm")),
                "building_type_penalty_norm": to_float(raw.get("BuildingTypePenalty_norm")),
                "population_norm": to_float(raw.get("b1_Population_norm")),
                "ped_presence_norm": to_float(raw.get("b2_PedestrianPresence_norm")),
                "urban_activity_norm": to_float(raw.get("b3_UrbanActivity_norm")),
                "curb_crowding_penalty": to_float(raw.get("CurbCrowdingPenalty")),
                "num_floors": to_float(raw.get("raw_numfloors")),
            }
            rows.append(row)
    return rows


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def build_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    boroughs = sorted({str(row["borough"]) for row in rows if row["borough"]})
    neighborhoods = sorted({str(row["neighborhood"]) for row in rows if row["neighborhood"]})
    car_values = sorted([float(row["car_time"]) for row in rows if row["car_time"] is not None])
    amr_values = sorted([float(row["amr_time"]) for row in rows if row["amr_time"] is not None])

    usable_count = sum(1 for row in rows if row["usable"] is True)
    amr_reach_count = sum(1 for row in rows if row["amr_reach"] is True)
    stairs_count = sum(1 for row in rows if row["stairs"] is True)
    gate_count = sum(1 for row in rows if row["gate"] is True)
    ramp_count = sum(1 for row in rows if row["ramp"] is True)

    return {
        "count": len(rows),
        "boroughs": boroughs,
        "neighborhoods": neighborhoods,
        "car_breaks": [percentile(car_values, p) for p in (0.2, 0.4, 0.6, 0.8)],
        "amr_breaks": [percentile(amr_values, p) for p in (0.2, 0.4, 0.6, 0.8)],
        "car_avg": mean(car_values) if car_values else None,
        "amr_avg": mean(amr_values) if amr_values else None,
        "usable_share": usable_count / len(rows) if rows else 0,
        "amr_reach_share": amr_reach_count / len(rows) if rows else 0,
        "stairs_share": stairs_count / len(rows) if rows else 0,
        "gate_share": gate_count / len(rows) if rows else 0,
        "ramp_share": ramp_count / len(rows) if rows else 0,
    }


def write_data_js(rows: list[dict[str, object]], stats: dict[str, object]) -> None:
    js = (
        "window.AI_SUBSET_MAP_DATA = "
        + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        + ";\nwindow.AI_SUBSET_MAP_STATS = "
        + json.dumps(stats, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )
    OUTPUT_DATA_JS.write_text(js, encoding="utf-8")


def write_html() -> None:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Subset Last-Meter Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    :root {
      --bg: #0b0f14;
      --panel: #121821;
      --line: #243041;
      --text: #f5f7fb;
      --muted: #9fb0c3;
      --blue: #2563eb;
      --panel-soft: #171f2a;
      --input-bg: #0f141c;
      --accent-text: #8cc8ff;
    }
    body.light-theme {
      --bg: #f3f6fb;
      --panel: #ffffff;
      --line: #d9e1ee;
      --text: #172033;
      --muted: #5d687b;
      --blue: #2563eb;
      --panel-soft: #ffffff;
      --input-bg: #ffffff;
      --accent-text: #4d97d8;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; font-family: Arial, sans-serif; color: var(--text); background: var(--bg); }
    .app { display: grid; grid-template-columns: 340px 1fr; height: 100vh; }
    .sidebar { overflow: auto; padding: 16px; background: var(--panel); border-right: 1px solid var(--line); }
    .main { display: grid; grid-template-rows: auto 1fr; gap: 14px; padding: 16px; min-width: 0; }
    .topbar { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
    .stat { background: var(--panel-soft); border: 1px solid var(--line); border-radius: 10px; padding: 12px; box-shadow: inset 0 1px 0 rgba(255,255,255,.02); }
    .stat-label { font-size: 12px; color: var(--accent-text); text-transform: uppercase; margin-bottom: 8px; }
    .stat-value { font-size: 24px; font-weight: 700; }
    .panel { min-width: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,.22); }
    .panel-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
    .panel-title { font-size: 16px; font-weight: 700; }
    .note { font-size: 12px; color: var(--muted); }
    #map { height: calc(100vh - 120px); min-height: 560px; }
    h1 { margin: 0 0 6px; font-size: 22px; }
    .subtitle { color: var(--muted); font-size: 13px; line-height: 1.4; margin-bottom: 14px; }
    .section { padding: 12px; border: 1px solid var(--line); border-radius: 10px; margin-bottom: 12px; background: var(--panel-soft); }
    .section-title { font-size: 12px; color: var(--accent-text); text-transform: uppercase; margin-bottom: 10px; font-weight: 700; }
    label { display: block; font-size: 13px; margin-bottom: 6px; }
    input[type="text"], select { width: 100%; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; font-size: 13px; background: var(--input-bg); color: var(--text); margin-bottom: 8px; }
    input[type="text"]::placeholder { color: var(--muted); }
    .button-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    button { padding: 9px 12px; border: 1px solid var(--line); border-radius: 9px; background: var(--input-bg); color: var(--text); cursor: pointer; font-size: 13px; }
    button.primary { background: var(--blue); border-color: var(--blue); color: #ffffff; font-weight: 700; }
    .legend { min-width: 210px; max-width: 260px; padding: 10px 12px; background: rgba(18,24,33,.96); color: var(--text); border: 1px solid var(--line); border-radius: 10px; line-height: 1.45; font-size: 12px; box-shadow: 0 6px 18px rgba(0,0,0,.28); }
    .legend-title { font-weight: 700; margin-bottom: 8px; white-space: normal; }
    .legend-row { display: grid; grid-template-columns: 14px 1fr; align-items: start; gap: 8px; margin-top: 6px; }
    .legend-label { white-space: normal; word-break: break-word; }
    .swatch { width: 14px; height: 14px; border-radius: 999px; border: 1px solid rgba(0,0,0,.12); margin-top: 1px; }
    .leaflet-popup-content { min-width: 320px; color: var(--text); }
    .popup-grid { display: grid; grid-template-columns: auto 1fr; gap: 6px 10px; font-size: 12px; }
    .popup-key { color: var(--muted); }
    .popup-section { grid-column: 1 / -1; margin-top: 6px; font-weight: 700; color: var(--text); }
    .leaflet-popup-content-wrapper, .leaflet-popup-tip { background: #121821; color: var(--text); }
    .leaflet-container a.leaflet-popup-close-button { color: #d5dee9; }
    body.light-theme .leaflet-popup-content-wrapper,
    body.light-theme .leaflet-popup-tip { background: #ffffff; color: var(--text); }
    body.light-theme .leaflet-container a.leaflet-popup-close-button { color: #425066; }
    body.light-theme .legend { background: rgba(255,255,255,.97); color: var(--text); box-shadow: 0 6px 18px rgba(17,24,39,.10); }
    .theme-toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: var(--input-bg);
      color: var(--text);
      cursor: pointer;
      font-size: 13px;
    }
    .theme-icons {
      display: inline-flex;
      gap: 6px;
      font-size: 15px;
      line-height: 1;
    }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 1fr; height: auto; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .topbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      #map { height: 70vh; min-height: 520px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:6px;">
        <h1 style="margin:0;">AI + Last-Meter Map</h1>
        <button id="themeToggle" class="theme-toggle" title="Switch between dark and light theme">
          <span class="theme-icons"><span>☀</span><span>☾</span></span>
        </button>
      </div>
      <div class="subtitle">Interactive view of the ~7000 buildings with Street View AI features and AI-informed modeled car / AMR last-meter times.</div>

      <div class="section">
        <div class="section-title">Color mode</div>
        <label title="Modeled car last-meter delivery time in seconds."><input type="radio" name="colorMode" value="car_time" checked> Car last-meter time</label>
        <label title="Modeled AMR last-meter delivery time in seconds."><input type="radio" name="colorMode" value="amr_time"> AMR last-meter time</label>
        <label title="Whether the Street View image was considered usable for AI interpretation."><input type="radio" name="colorMode" value="usable"> Image usable</label>
        <label title="Whether the AI judged that an AMR could plausibly reach the building door."><input type="radio" name="colorMode" value="amr_reach"> AMR can reach door</label>
      </div>

      <div class="section">
        <div class="section-title">Search</div>
        <input id="searchText" type="text" placeholder="Search BIN, neighborhood, or address" />
      </div>

      <div class="section">
        <div class="section-title">Filters</div>
        <label for="boroughSelect">Borough</label>
        <select id="boroughSelect"></select>

        <label for="neighborhoodSelect">Neighborhood</label>
        <select id="neighborhoodSelect"></select>

        <label for="usableSelect">Image usable</label>
        <select id="usableSelect">
          <option value="">All</option>
          <option value="true">True</option>
          <option value="false">False</option>
        </select>

        <label for="reachSelect">AMR can reach door</label>
        <select id="reachSelect">
          <option value="">All</option>
          <option value="true">True</option>
          <option value="false">False</option>
        </select>

        <div class="button-row">
          <button class="primary" id="applyBtn">Apply filters</button>
          <button id="resetBtn">Reset</button>
          <button id="downloadBtn">Download CSV</button>
        </div>
      </div>

      <div class="section">
        <div class="section-title">How to read it</div>
        <div class="note">
          Click a point for BIN, borough, neighborhood, car/AMR times, and AI-derived access flags.<br><br>
          Use the color mode to switch between modeled times and binary AI fields.
        </div>
      </div>
    </aside>

    <main class="main">
      <section class="topbar">
        <div class="stat"><div class="stat-label">Visible buildings</div><div class="stat-value" id="visibleCount">-</div></div>
        <div class="stat"><div class="stat-label">Avg car time (s)</div><div class="stat-value" id="avgCar">-</div></div>
        <div class="stat"><div class="stat-label">Avg AMR time (s)</div><div class="stat-value" id="avgAmr">-</div></div>
        <div class="stat"><div class="stat-label">Usable image share</div><div class="stat-value" id="usableShare">-</div></div>
        <div class="stat"><div class="stat-label">AMR reach share</div><div class="stat-value" id="reachShare">-</div></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div class="panel-title">Map</div>
          <div class="note" id="mapNote">Colored by car last-meter time.</div>
        </div>
        <div id="map"></div>
      </section>
    </main>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="./complete_last_meter_dataset_ai_subset_map.data.js"></script>
  <script>
    const rows = window.AI_SUBSET_MAP_DATA || [];
    const stats = window.AI_SUBSET_MAP_STATS || {};
    const colorPalette = ["#2dc937", "#99c140", "#e7b416", "#db7b2b", "#cc3232"];
    const boolColors = { true: "#1f9d55", false: "#cf4446", null: "#98a2b3" };

    const state = {
      filtered: rows,
      colorMode: "car_time",
      theme: "dark",
      map: null,
      layer: null,
      legend: null,
      renderer: null,
      markers: [],
      tileLayer: null,
    };
    const DARK_TILES = {
      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      options: { attribution: "&copy; OpenStreetMap contributors &copy; CARTO", subdomains: "abcd", maxZoom: 20 }
    };
    const LIGHT_TILES = {
      url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      options: { attribution: "&copy; OpenStreetMap contributors &copy; CARTO", subdomains: "abcd", maxZoom: 20 }
    };

    function fmt(value, digits = 1) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return Number(value).toFixed(digits);
    }

    function formatLegendValue(mode, value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      if (mode === "car_time" || mode === "amr_time") return Number(value).toFixed(0);
      return Number(value).toFixed(2);
    }

    function pct(value) {
      return value === null || value === undefined ? "-" : (value * 100).toFixed(1) + "%";
    }

    function csvEscape(value) {
      if (value === null || value === undefined) return "";
      const text = String(value);
      if (text.includes('"') || text.includes(',') || text.includes('\\n')) {
        return '"' + text.replace(/"/g, '""') + '"';
      }
      return text;
    }

    function applyTheme() {
      const light = state.theme === "light";
      document.body.classList.toggle("light-theme", light);
      const toggle = document.getElementById("themeToggle");
      toggle.title = light ? "Switch to dark theme" : "Switch to light theme";
      toggle.setAttribute("aria-label", toggle.title);
      if (state.tileLayer) {
        state.map.removeLayer(state.tileLayer);
      }
      const tileConfig = light ? LIGHT_TILES : DARK_TILES;
      state.tileLayer = L.tileLayer(tileConfig.url, tileConfig.options).addTo(state.map);
    }

    function fillSelect(select, values, placeholder) {
      select.innerHTML = "";
      const allOpt = document.createElement("option");
      allOpt.value = "";
      allOpt.textContent = placeholder;
      select.appendChild(allOpt);
      values.forEach((value) => {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
      });
    }

    function populateBoroughs() {
      fillSelect(document.getElementById("boroughSelect"), stats.boroughs || [], "All boroughs");
    }

    function populateNeighborhoods() {
      const borough = document.getElementById("boroughSelect").value;
      const neighborhoods = [...new Set(
        rows
          .filter((row) => !borough || row.borough === borough)
          .map((row) => row.neighborhood)
          .filter((value) => value)
      )].sort();
      fillSelect(document.getElementById("neighborhoodSelect"), neighborhoods, "All neighborhoods");
    }

    function rowMatches(row) {
      const borough = document.getElementById("boroughSelect").value;
      const neighborhood = document.getElementById("neighborhoodSelect").value;
      const usable = document.getElementById("usableSelect").value;
      const reach = document.getElementById("reachSelect").value;
      const search = document.getElementById("searchText").value.trim().toLowerCase();

      if (borough && row.borough !== borough) return false;
      if (neighborhood && row.neighborhood !== neighborhood) return false;
      if (usable && String(row.usable) !== usable) return false;
      if (reach && String(row.amr_reach) !== reach) return false;
      if (search) {
        const hay = [
          row.bin,
          row.borough,
          row.neighborhood,
          row.full_address,
          row.street_name,
          row.full_street_name,
          row.zipcode
        ].join(" ").toLowerCase();
        if (!hay.includes(search)) return false;
      }
      return true;
    }

    function quantileBreaks(values) {
      if (!values.length) return [];
      const sorted = [...values].sort((a, b) => a - b);
      const q = (p) => {
        if (sorted.length === 1) return sorted[0];
        const idx = (sorted.length - 1) * p;
        const lo = Math.floor(idx);
        const hi = Math.min(lo + 1, sorted.length - 1);
        const frac = idx - lo;
        return sorted[lo] * (1 - frac) + sorted[hi] * frac;
      };
      return [q(0.2), q(0.4), q(0.6), q(0.8)];
    }

    function getBreaks(mode) {
      if (mode === "amr_time") {
        const values = state.filtered
          .filter((row) => row.amr_reach !== false)
          .map((row) => row.amr_time)
          .filter((value) => value !== null && value !== undefined && !Number.isNaN(value));
        return quantileBreaks(values);
      }
      return stats.car_breaks || [];
    }

    function colorForRow(row) {
      if (state.colorMode === "usable") return boolColors[String(row.usable)] || boolColors.null;
      if (state.colorMode === "amr_reach") return boolColors[String(row.amr_reach)] || boolColors.null;
      if (state.colorMode === "amr_time" && row.amr_reach === false) return null;

      const value =
        state.colorMode === "amr_time" ? row.amr_time :
        state.colorMode === "car_time" ? row.car_time :
        row[state.colorMode];
      const breaks = getBreaks(state.colorMode);
      if (value === null || value === undefined) return boolColors.null;
      if (breaks.length < 4) return colorPalette[2];
      if (value <= breaks[0]) return colorPalette[0];
      if (value <= breaks[1]) return colorPalette[1];
      if (value <= breaks[2]) return colorPalette[2];
      if (value <= breaks[3]) return colorPalette[3];
      return colorPalette[4];
    }

    function updateMarkerColors() {
      state.markers.forEach((marker) => {
        const fill = colorForRow(marker._row);
        if (fill === null) {
          marker.setStyle({
            fillColor: "#000000",
            fillOpacity: 0,
            opacity: 0,
            stroke: false,
            weight: 0,
          });
        } else {
          marker.setStyle({
            fillColor: fill,
            fillOpacity: 0.9,
            opacity: 1,
            stroke: true,
            weight: 0.5,
            color: "rgba(255,255,255,0.18)",
          });
        }
      });
    }

    function popupFeatureRow(label, description, value) {
      return `<div class="popup-key"><span title="${description}">${label}</span></div><div>${value}</div>`;
    }

    function popupHtml(row) {
      const showAmr = row.amr_reach !== false;
      const amrTimeRow = showAmr
        ? `<div class="popup-key">AMR time</div><div>${fmt(row.amr_time, 1)} s</div>`
        : ``;
      const amrFeaturesBlock = showAmr
        ? `
          ${popupFeatureRow("Population intensity", "Normalized neighborhood population proxy used in the AMR model.", fmt(row.population_norm, 3))}
          ${popupFeatureRow("Pedestrian presence", "Normalized pedestrian corridor / presence proxy used in the AMR model.", fmt(row.ped_presence_norm, 3))}
          ${popupFeatureRow("Urban activity", "Normalized urban activity proxy derived from land use.", fmt(row.urban_activity_norm, 3))}
        `
        : "";
      return `
        <div style="font-weight:700;margin-bottom:8px;">BIN ${row.bin}</div>
        <div class="popup-grid">
          <div class="popup-key">Borough</div><div>${row.borough || "-"}</div>
          <div class="popup-key">Neighborhood</div><div>${row.neighborhood || "-"}</div>
          <div class="popup-key">Address</div><div>${row.full_address || "-"}</div>
          <div class="popup-key">ZIP code</div><div>${row.zipcode || "-"}</div>
          <div class="popup-key">Number of floors</div><div>${fmt(row.num_floors, 0)}</div>
          <div class="popup-section">Last meter estimated times</div>
          <div class="popup-key">Car time</div><div>${fmt(row.car_time, 1)} s</div>
          ${amrTimeRow}
          <div class="popup-section">AI derived features</div>
          <div class="popup-key">Image usable</div><div>${row.usable}</div>
          <div class="popup-key">Stairs</div><div>${row.stairs}</div>
          <div class="popup-key">Gate</div><div>${row.gate}</div>
          <div class="popup-key">Ramp</div><div>${row.ramp}</div>
          <div class="popup-key">AI barrier mean</div><div>${fmt(row.ai_access_barrier_mean, 3)}</div>
          <div class="popup-key">AMR reach door</div><div>${row.amr_reach}</div>
          <div class="popup-section">Normalized features</div>
          ${popupFeatureRow("Normalized floors", "Min-max normalized number of floors.", fmt(row.floors_norm, 3))}
          ${popupFeatureRow("Normalized road-to-delivery distance", "Min-max normalized proxy of how far the delivery point is from the street context.", fmt(row.road_delivery_norm, 3))}
          ${popupFeatureRow("Shape penalty", "Normalized building elongation penalty derived from footprint geometry.", fmt(row.shape_penalty_norm, 3))}
          ${popupFeatureRow("Building type penalty", "Rule-based difficulty score derived from land-use category.", fmt(row.building_type_penalty_norm, 3))}
          ${amrFeaturesBlock}
          ${popupFeatureRow("Curb crowding penalty", "Normalized curbside difficulty proxy used in the car model.", fmt(row.curb_crowding_penalty, 3))}
        </div>
      `;
    }

    function updateSummary() {
      const subset = state.filtered;
      document.getElementById("visibleCount").textContent = subset.length;

      const carVals = subset.map(r => r.car_time).filter(v => v !== null && v !== undefined);
      const amrVals = subset.map(r => r.amr_time).filter(v => v !== null && v !== undefined);
      const usableVals = subset.map(r => r.usable).filter(v => v !== null);
      const reachVals = subset.map(r => r.amr_reach).filter(v => v !== null);

      const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
      const shareTrue = (arr) => arr.length ? arr.filter(Boolean).length / arr.length : null;

      document.getElementById("avgCar").textContent = fmt(avg(carVals), 1);
      document.getElementById("avgAmr").textContent = fmt(avg(amrVals), 1);
      document.getElementById("usableShare").textContent = pct(shareTrue(usableVals));
      document.getElementById("reachShare").textContent = pct(shareTrue(reachVals));
    }

    function modeTitle(mode) {
      const titles = {
        car_time: "Car last-meter time (s)",
        amr_time: "AMR last-meter time (s)",
        usable: "Image usable",
        amr_reach: "AMR can reach door",
      };
      return titles[mode] || mode;
    }

    function buildLegend() {
      if (state.legend) state.legend.remove();
      const legend = L.control({ position: "bottomright" });
      legend.onAdd = function() {
        const div = L.DomUtil.create("div", "legend");
        if (state.colorMode === "usable" || state.colorMode === "amr_reach") {
          const title = modeTitle(state.colorMode);
          div.innerHTML = `<div class="legend-title">${title}</div>`;
          [["True", boolColors.true], ["False", boolColors.false], ["Missing", boolColors.null]].forEach(([label, color]) => {
            div.innerHTML += `<div class="legend-row"><span class="swatch" style="background:${color}"></span><span class="legend-label">${label}</span></div>`;
          });
        } else {
          const title = modeTitle(state.colorMode);
          const breaks = getBreaks(state.colorMode);
          div.innerHTML = `<div class="legend-title">${title}</div>`;
          const labels = [
            `<= ${formatLegendValue(state.colorMode, breaks[0])}`,
            `${formatLegendValue(state.colorMode, breaks[0])} to ${formatLegendValue(state.colorMode, breaks[1])}`,
            `${formatLegendValue(state.colorMode, breaks[1])} to ${formatLegendValue(state.colorMode, breaks[2])}`,
            `${formatLegendValue(state.colorMode, breaks[2])} to ${formatLegendValue(state.colorMode, breaks[3])}`,
            `> ${formatLegendValue(state.colorMode, breaks[3])}`
          ];
          labels.forEach((label, idx) => {
            div.innerHTML += `<div class="legend-row"><span class="swatch" style="background:${colorPalette[idx]}"></span><span class="legend-label">${label}</span></div>`;
          });
        }
        return div;
      };
      legend.addTo(state.map);
      state.legend = legend;
    }

    function renderMap(refit = false) {
      if (state.layer) state.layer.remove();
      const markers = state.filtered.map((row) => {
        const fill = colorForRow(row);
        const marker = L.circleMarker([row.lat, row.lon], {
          renderer: state.renderer,
          radius: 4,
          weight: fill === null ? 0 : 0.5,
          color: "rgba(255,255,255,0.18)",
          fillColor: fill || "#000000",
          fillOpacity: fill === null ? 0 : 0.9,
          opacity: fill === null ? 0 : 1,
          stroke: fill !== null,
        }).bindPopup(popupHtml(row));
        marker._row = row;
        return marker;
      });
      state.markers = markers;
      state.layer = L.layerGroup(markers).addTo(state.map);
      if (refit && markers.length) {
        const group = L.featureGroup(markers);
        state.map.fitBounds(group.getBounds().pad(0.08));
      }
      buildLegend();
      updateSummary();
      document.getElementById("mapNote").textContent =
        state.colorMode === "usable" ? "Green = usable image, red = unusable." :
        state.colorMode === "amr_reach" ? "Green = AMR likely can reach the door, red = likely cannot." :
        `Colored by ${modeTitle(state.colorMode).toLowerCase()}.`;
    }

    function applyFilters() {
      state.filtered = rows.filter(rowMatches);
      renderMap(true);
    }

    function resetFilters() {
      document.getElementById("boroughSelect").value = "";
      populateNeighborhoods();
      document.getElementById("neighborhoodSelect").value = "";
      document.getElementById("usableSelect").value = "";
      document.getElementById("reachSelect").value = "";
      document.getElementById("searchText").value = "";
      state.filtered = rows.slice();
      renderMap(true);
    }

    function downloadFilteredCsv() {
      const columns = [
        ["bin", "BIN"],
        ["borough", "borough"],
        ["neighborhood", "neighborhood"],
        ["full_address", "full_address"],
        ["zipcode", "zipcode"],
        ["num_floors", "num_floors"],
        ["amr_time", "amr_time_s"],
        ["car_time", "car_time_s"],
        ["usable", "image_usable"],
        ["stairs", "stairs_present"],
        ["gate", "gate_present"],
        ["ramp", "ramp_present"],
        ["amr_reach", "amr_can_reach_door"],
      ];
      const lines = [];
      lines.push(columns.map(([, label]) => csvEscape(label)).join(","));
      state.filtered.forEach((row) => {
        lines.push(columns.map(([key]) => csvEscape(row[key])).join(","));
      });
      const blob = new Blob([lines.join("\\n")], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "ai_subset_filtered_export.csv";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }

    function initMap() {
      state.map = L.map("map", { preferCanvas: true, zoomSnap: 0.25 }).setView([40.75, -73.97], 12);
      state.renderer = L.canvas({ padding: 0.5 });
      applyTheme();
      renderMap(true);
    }

    document.querySelectorAll("input[name='colorMode']").forEach((input) => {
      input.addEventListener("change", () => {
        state.colorMode = input.value;
        updateMarkerColors();
        buildLegend();
        document.getElementById("mapNote").textContent =
          state.colorMode === "usable" ? "Green = usable image, red = unusable." :
          state.colorMode === "amr_reach" ? "Green = AMR likely can reach the door, red = likely cannot." :
          `Colored by ${modeTitle(state.colorMode).toLowerCase()}.`;
      });
    });
    document.getElementById("boroughSelect").addEventListener("change", () => {
      populateNeighborhoods();
    });
    document.getElementById("applyBtn").addEventListener("click", applyFilters);
    document.getElementById("resetBtn").addEventListener("click", resetFilters);
    document.getElementById("downloadBtn").addEventListener("click", downloadFilteredCsv);
    document.getElementById("themeToggle").addEventListener("click", () => {
      state.theme = state.theme === "dark" ? "light" : "dark";
      applyTheme();
    });
    document.getElementById("searchText").addEventListener("keydown", (event) => {
      if (event.key === "Enter") applyFilters();
    });

    populateBoroughs();
    populateNeighborhoods();
    initMap();
  </script>
</body>
</html>
"""
    OUTPUT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    rows = load_rows()
    if not rows:
        raise RuntimeError(f"No mappable rows found in {resolve_input_csv()}")
    stats = build_stats(rows)
    write_data_js(rows, stats)
    write_html()
    print(f"Saved map HTML: {OUTPUT_HTML}")
    print(f"Saved map data JS: {OUTPUT_DATA_JS}")
    print(f"Source CSV used: {resolve_input_csv()}")
    print(f"Rows mapped: {len(rows)}")


if __name__ == "__main__":
    main()
