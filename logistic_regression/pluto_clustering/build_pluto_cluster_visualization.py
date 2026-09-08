"""Build a focused Plotly HTML explorer for PLUTO accessibility summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

CLUSTERED_CSV = OUTPUT_DIR / "pluto_clustered_ai_buildings.csv"
CLUSTER_SUMMARY_CSV = OUTPUT_DIR / "pluto_cluster_summary.csv"
LANDUSE_SUMMARY_CSV = OUTPUT_DIR / "landuse_accessibility_summary.csv"
BLDGCLASS_GROUP_SUMMARY_CSV = OUTPUT_DIR / "bldgclass_group_accessibility_summary.csv"
HTML_OUTPUT = OUTPUT_DIR / "pluto_cluster_explorer.html"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def to_records(df: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(df.where(pd.notna(df), None).to_json(orient="records"))


def build_heatmap(clustered: pd.DataFrame) -> dict[str, object]:
    work = clustered.copy()
    work["landuse_axis"] = work["landuse_clean"].astype(str) + " - " + work["landuse_label"].astype(str)
    work["bldgclass_axis"] = work["bldgclass_group"].astype(str) + " - " + work["bldgclass_group_label"].astype(str)

    grouped = (
        work.groupby(["landuse_axis", "bldgclass_axis"], dropna=False)["target_amr_accessible"]
        .agg(rows="count", accessible="sum", accessible_rate="mean")
        .reset_index()
    )
    grouped["not_accessible"] = grouped["rows"] - grouped["accessible"]

    landuse_order = (
        work[["landuse_axis", "landuse_clean"]]
        .drop_duplicates()
        .assign(sort_key=lambda df: pd.to_numeric(df["landuse_clean"], errors="coerce"))
        .sort_values(["sort_key", "landuse_axis"], na_position="last")["landuse_axis"]
        .tolist()
    )
    bldgclass_order = (
        work[["bldgclass_axis", "bldgclass_group"]]
        .drop_duplicates()
        .sort_values(["bldgclass_group", "bldgclass_axis"])["bldgclass_axis"]
        .tolist()
    )

    rate = grouped.pivot(index="bldgclass_axis", columns="landuse_axis", values="accessible_rate").reindex(
        index=bldgclass_order,
        columns=landuse_order,
    )
    rows = grouped.pivot(index="bldgclass_axis", columns="landuse_axis", values="rows").reindex(
        index=bldgclass_order,
        columns=landuse_order,
    )
    accessible = grouped.pivot(index="bldgclass_axis", columns="landuse_axis", values="accessible").reindex(
        index=bldgclass_order,
        columns=landuse_order,
    )
    not_accessible = grouped.pivot(index="bldgclass_axis", columns="landuse_axis", values="not_accessible").reindex(
        index=bldgclass_order,
        columns=landuse_order,
    )

    customdata = []
    for row_label in bldgclass_order:
        row_data = []
        for column_label in landuse_order:
            row_data.append(
                [
                    None if pd.isna(rows.loc[row_label, column_label]) else int(rows.loc[row_label, column_label]),
                    None if pd.isna(accessible.loc[row_label, column_label]) else int(accessible.loc[row_label, column_label]),
                    None if pd.isna(not_accessible.loc[row_label, column_label]) else int(not_accessible.loc[row_label, column_label]),
                ]
            )
        customdata.append(row_data)

    return {
        "x": landuse_order,
        "y": bldgclass_order,
        "z": [[None if pd.isna(value) else float(value) * 100 for value in row] for row in rate.to_numpy()],
        "customdata": customdata,
    }


def build_html() -> str:
    clustered = read_csv(CLUSTERED_CSV)
    cluster_summary = read_csv(CLUSTER_SUMMARY_CSV)
    landuse_summary = read_csv(LANDUSE_SUMMARY_CSV)
    bldgclass_group_summary = read_csv(BLDGCLASS_GROUP_SUMMARY_CSV)
    heatmap = build_heatmap(clustered)

    payload = {
        "clusters": to_records(cluster_summary),
        "landuse": to_records(landuse_summary),
        "bldgclassGroups": to_records(bldgclass_group_summary),
        "heatmap": heatmap,
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PLUTO AMR Accessibility Explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f7f9;
      color: #17202a;
    }}
    header {{
      padding: 22px 28px;
      background: #1f2933;
      color: white;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 1.8rem;
    }}
    header p {{
      margin: 0;
      color: #d8dee7;
      max-width: 980px;
    }}
    main {{
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }}
    section {{
      min-height: 430px;
      background: white;
      border: 1px solid #d7dde5;
      border-radius: 8px;
      box-shadow: 0 8px 22px rgba(23, 32, 42, 0.08);
      padding: 10px;
    }}
    .wide {{
      grid-column: 1 / -1;
      min-height: 680px;
    }}
    @media (max-width: 1100px) {{
      main {{
        grid-template-columns: 1fr;
      }}
      .wide {{
        grid-column: auto;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>PLUTO AMR Accessibility Explorer</h1>
    <p>Focused interactive charts showing AMR accessibility by PLUTO cluster, land use, and building class group.</p>
  </header>
  <main>
    <section id="clusterRate"></section>
    <section id="landuseRate"></section>
    <section id="bldgclassRate"></section>
    <section class="wide" id="landuseBldgclassHeatmap"></section>
  </main>
  <script>
    const data = {json.dumps(payload)};

    function pct(values) {{
      return values.map(v => Number(v) * 100);
    }}

    function layout(title, xTitle, yTitle) {{
      return {{
        title: {{ text: title, font: {{ size: 16 }} }},
        margin: {{ l: 64, r: 24, t: 56, b: 96 }},
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        xaxis: {{ title: xTitle, gridcolor: '#eef1f5', automargin: true }},
        yaxis: {{ title: yTitle, gridcolor: '#eef1f5', automargin: true }}
      }};
    }}

    const clusters = data.clusters.slice().sort((a, b) => Number(a.pluto_cluster) - Number(b.pluto_cluster));
    Plotly.newPlot('clusterRate', [{{
      type: 'bar',
      x: clusters.map(d => `Cluster ${{d.pluto_cluster}}`),
      y: pct(clusters.map(d => d.accessible_rate)),
      marker: {{ color: pct(clusters.map(d => d.accessible_rate)), colorscale: 'Viridis', cmin: 0, cmax: 100 }},
      customdata: clusters.map(d => [d.rows, d.accessible, d.not_accessible, d.top_landuse, d.top_bldgclass_group]),
      hovertemplate:
        '<b>%{{x}}</b><br>' +
        'Accessible rate: %{{y:.1f}}%<br>' +
        'Rows: %{{customdata[0]}}<br>' +
        'Accessible: %{{customdata[1]}}<br>' +
        'Not accessible: %{{customdata[2]}}<br>' +
        'Top land use: %{{customdata[3]}}<br>' +
        'Top building group: %{{customdata[4]}}<extra></extra>'
    }}], layout('AMR Accessibility by Cluster', 'PLUTO cluster', 'AMR accessible rate (%)'), {{ responsive: true }});

    const landuse = data.landuse
      .filter(d => d.landuse_clean !== null && d.landuse_clean !== undefined && String(d.landuse_clean).toLowerCase() !== 'null')
      .sort((a, b) => Number(a.landuse_clean) - Number(b.landuse_clean));
    Plotly.newPlot('landuseRate', [{{
      type: 'bar',
      x: landuse.map(d => `${{d.landuse_clean}} - ${{d.landuse_label}}`),
      y: pct(landuse.map(d => d.accessible_rate)),
      marker: {{ color: pct(landuse.map(d => d.accessible_rate)), colorscale: 'Greens', cmin: 0, cmax: 100 }},
      customdata: landuse.map(d => [d.rows, d.accessible, d.not_accessible]),
      hovertemplate:
        '<b>%{{x}}</b><br>' +
        'Accessible rate: %{{y:.1f}}%<br>' +
        'Rows: %{{customdata[0]}}<br>' +
        'Accessible: %{{customdata[1]}}<br>' +
        'Not accessible: %{{customdata[2]}}<extra></extra>'
    }}], layout('AMR Accessibility by Land Use', 'PLUTO land use', 'AMR accessible rate (%)'), {{ responsive: true }});

    const bGroups = data.bldgclassGroups
      .slice()
      .sort((a, b) => String(a.bldgclass_group).localeCompare(String(b.bldgclass_group)));
    Plotly.newPlot('bldgclassRate', [{{
      type: 'bar',
      x: bGroups.map(d => `${{d.bldgclass_group}} - ${{d.bldgclass_group_label}}`),
      y: pct(bGroups.map(d => d.accessible_rate)),
      marker: {{ color: pct(bGroups.map(d => d.accessible_rate)), colorscale: 'Blues', cmin: 0, cmax: 100 }},
      customdata: bGroups.map(d => [d.rows, d.accessible, d.not_accessible]),
      hovertemplate:
        '<b>%{{x}}</b><br>' +
        'Accessible rate: %{{y:.1f}}%<br>' +
        'Rows: %{{customdata[0]}}<br>' +
        'Accessible: %{{customdata[1]}}<br>' +
        'Not accessible: %{{customdata[2]}}<extra></extra>'
    }}], layout('AMR Accessibility by Building Class Group', 'Building class group', 'AMR accessible rate (%)'), {{ responsive: true }});

    Plotly.newPlot('landuseBldgclassHeatmap', [{{
      type: 'heatmap',
      x: data.heatmap.x,
      y: data.heatmap.y,
      z: data.heatmap.z,
      customdata: data.heatmap.customdata,
      colorscale: 'YlGnBu',
      zmin: 0,
      zmax: 100,
      colorbar: {{ title: 'Accessible %' }},
      hoverongaps: false,
      hovertemplate:
        '<b>Land use:</b> %{{x}}<br>' +
        '<b>Building class:</b> %{{y}}<br>' +
        'Accessible rate: %{{z:.1f}}%<br>' +
        'Rows: %{{customdata[0]}}<br>' +
        'Accessible: %{{customdata[1]}}<br>' +
        'Not accessible: %{{customdata[2]}}<extra></extra>'
    }}], {{
      title: {{ text: 'AMR Accessibility Heatmap: Land Use x Building Class Group', font: {{ size: 17 }} }},
      margin: {{ l: 190, r: 40, t: 70, b: 170 }},
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      xaxis: {{ title: 'PLUTO land use', automargin: true }},
      yaxis: {{ title: 'Building class group', automargin: true }}
    }}, {{ responsive: true }});
  </script>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUTPUT.write_text(build_html(), encoding="utf-8")
    print(f"Saved focused interactive explorer: {HTML_OUTPUT}")


if __name__ == "__main__":
    main()
