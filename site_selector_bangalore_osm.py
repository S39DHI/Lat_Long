# delhi_fullscreen_smoothed_noscroll_with_controls.py
import streamlit as st
st.set_page_config(layout="wide", page_title="Delhi Fullscreen Smoothed (NoScroll)", initial_sidebar_state="collapsed")

import os
import json
import math
import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point
from streamlit.components.v1 import html as st_html
from sklearn.neighbors import KernelDensity
from shapely.ops import unary_union

# ---------------- CONFIG ----------------
PURPLE_BORDER = "#7B1FA2"
FILL_COLOR = "#f5e9fb"
DEFAULT_AREA = "Delhi, India"
SMOOTH_ITERS = 3

# ---------------- Chaikin smoothing (same as before) ----------------
def chaikin_smooth(geom, iterations=3):
    def _cut_ring(coords):
        if len(coords) < 3:
            return coords
        new_coords = []
        n = len(coords)
        for i in range(n - 1):
            p0 = coords[i]
            p1 = coords[(i + 1) % (n - 1)]
            Q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            R = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_coords.append(Q)
            new_coords.append(R)
        new_coords.append(new_coords[0])
        return new_coords

    def smooth_polygon(poly):
        ext_coords = list(poly.exterior.coords)
        for _ in range(iterations):
            ext_coords = _cut_ring(ext_coords)
        interiors = []
        for interior in poly.interiors:
            i_coords = list(interior.coords)
            for _ in range(iterations):
                i_coords = _cut_ring(i_coords)
            interiors.append(i_coords)
        try:
            return Polygon(ext_coords, interiors)
        except Exception:
            return poly

    if geom is None:
        return geom
    if isinstance(geom, Polygon):
        return smooth_polygon(geom)
    if isinstance(geom, MultiPolygon):
        parts = []
        for p in geom.geoms:
            parts.append(smooth_polygon(p))
        return MultiPolygon(parts)
    return geom

# ---------------- Load + smooth Delhi (cached) ----------------
@st.cache_data(show_spinner=False)
def load_and_smooth_delhi(smooth_iters=3):
    try:
        gdf = ox.geocode_to_gdf(DEFAULT_AREA)
        gdf = gdf.to_crs(epsg=4326)
    except Exception:
        return {"type": "FeatureCollection", "features": []}
    gj = json.loads(gdf.to_json())
    for feat in gj.get("features", []):
        try:
            geom = shape(feat["geometry"])
            smooth_geom = chaikin_smooth(geom, iterations=smooth_iters)
            feat["geometry"] = mapping(smooth_geom)
        except Exception:
            continue
    return gj

delhi_geojson = load_and_smooth_delhi(SMOOTH_ITERS)
delhi_geojson_str = json.dumps(delhi_geojson)

# ---------------- helper: map category name -> OSM tags ----------------
CATEGORY_TO_TAGS = {
    "shops": {"shop": True},
    "banks": {"amenity": "bank"},
    "hospitals": {"amenity": "hospital"},
    "schools": {"amenity": "school"},
    "clinics": {"amenity": "clinic"},
    "bus_stops": {"highway": "bus_stop"},
    "theatres": {"amenity": "theatre"},
    "restaurants": {"amenity": "restaurant"},
    "pharmacies": {"amenity": "pharmacy"},
    "parking": {"amenity": "parking"},
}

# ---------------- candidate compute function ----------------
@st.cache_data(show_spinner=True)
def compute_candidates(place=DEFAULT_AREA,
                       selected_categories=None,
                       bandwidth=800,
                       grid_size=250,
                       competitor_radius=500,
                       top_n=12,
                       proj_crs="EPSG:3857"):
    """
    Fetch selected categories from OSM, compute KDE peaks, score and return GeoJSON
    """
    ox.config(use_cache=True, log_console=False)
    if not selected_categories:
        # default to shops + some amenities if none selected
        selected_categories = ["shops", "banks", "hospitals", "schools", "bus_stops"]

    # geocode area
    try:
        area_gdf = ox.geocode_to_gdf(place)
    except Exception as e:
        return {"type":"FeatureCollection","features":[]}

    # build list of tag dicts to fetch (one API call per category)
    tag_list = []
    for cat in selected_categories:
        tag = CATEGORY_TO_TAGS.get(cat)
        if tag:
            tag_list.append(tag)

    poi_frames = []
    for t in tag_list:
        try:
            g = ox.geometries_from_place(place, tags=t)
            if not g.empty:
                poi_frames.append(g)
        except Exception:
            continue

    if not poi_frames:
        return {"type":"FeatureCollection","features":[]}

    # concat and keep only geometry
    pois = gpd.GeoDataFrame(pd.concat(poi_frames, ignore_index=True), crs=poi_frames[0].crs)
    # ensure area is WGS84 then project everything to metric CRS for distances
    area_gdf = area_gdf.to_crs(epsg=4326).to_crs(proj_crs)
    pois = pois.to_crs(proj_crs)

    # reduce to points (centroid) for polygons/lines
    pois['geometry'] = pois.geometry.centroid
    # clip to area polygon
    area_poly = area_gdf.geometry.unary_union
    pois = pois[pois.geometry.within(area_poly)].reset_index(drop=True)

    if len(pois) < 3:
        return {"type":"FeatureCollection","features":[]}

    # KDE
    coords = np.vstack([pois.geometry.x.values, pois.geometry.y.values]).T
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(coords)

    # grid
    minx, miny, maxx, maxy = area_gdf.geometry.total_bounds
    xv = np.arange(minx, maxx, grid_size)
    yv = np.arange(miny, maxy, grid_size)
    if len(xv) == 0 or len(yv) == 0:
        return {"type":"FeatureCollection","features":[]}
    xx, yy = np.meshgrid(xv, yv)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    zs = np.exp(kde.score_samples(grid_coords))
    Z = zs.reshape(len(yv), len(xv))

    # find local maxima
    from scipy import ndimage
    neighborhood = ndimage.maximum_filter(Z, size=3)
    local_max = (Z == neighborhood) & (Z > np.percentile(Z, 75))
    ys_idx, xs_idx = np.where(local_max)

    candidates = []
    for yi, xi in zip(ys_idx, xs_idx):
        gx = xv[xi]
        gy = yv[yi]
        score = float(Z[yi, xi])
        pt = Point(gx, gy)
        buffer = pt.buffer(competitor_radius)
        competitor_count = int(pois[pois.geometry.within(buffer)].shape[0])
        candidates.append({"x": gx, "y": gy, "score": score, "competitors": competitor_count})

    if not candidates:
        return {"type":"FeatureCollection","features":[]}

    # normalize and rank
    scores = np.array([c['score'] for c in candidates], dtype=float)
    comps = np.array([c['competitors'] for c in candidates], dtype=float)
    score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    comp_norm = 1 - (comps - comps.min()) / (comps.max() - comps.min() + 1e-9)
    final = 0.7 * score_norm + 0.3 * comp_norm
    for i, c in enumerate(candidates):
        c['final'] = float(final[i])

    candidates = sorted(candidates, key=lambda d: d['final'], reverse=True)[:top_n]

    # convert to WGS84 geojson features
    cand_gdf = gpd.GeoDataFrame(candidates, geometry=[Point(c['x'], c['y']) for c in candidates], crs=proj_crs)
    cand_wgs = cand_gdf.to_crs(epsg=4326)
    features = []
    for _, r in cand_wgs.iterrows():
        feat = {
            "type": "Feature",
            "properties": {
                "final": float(r['final']),
                "competitors": int(r['competitors'])
            },
            "geometry": mapping(r.geometry)
        }
        features.append(feat)
    return {"type": "FeatureCollection", "features": features}

# ---------------- Streamlit layout ----------------
# We'll use two columns so left column acts like the transparent control panel
left_col, right_col = st.columns([0.35, 0.65])

with left_col:
    # style the left column to look like your translucent overlay
    st.markdown(
        f"""
        <div style="
           background: rgba(255,255,255,0.10);
           border-radius: 20px;
           padding: 16px;
           box-shadow: 0 10px 30px rgba(0,0,0,0.35);
           color: #fff;
        ">
        <h3 style="color:{PURPLE_BORDER}; margin:0 0 6px 0;">Controls</h3>
        <p style="font-size:13px; margin:4px 0 12px 0; color:#e6e6e6;">Pick categories and tune the algorithm.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # sliders & multi-select
    kde_bandwidth = st.slider("KDE bandwidth (m)", min_value=200, max_value=2000, value=800, step=100)
    grid_size = st.slider("Grid size (m)", min_value=100, max_value=500, value=250, step=50)
    competitor_radius = st.slider("Competitor radius (m)", min_value=200, max_value=1500, value=500, step=50)

    # categories multiselect (human-friendly labels -> internal keys)
    friendly = {
        "Shops": "shops",
        "Banks": "banks",
        "Hospitals": "hospitals",
        "Schools": "schools",
        "Clinics": "clinics",
        "Bus stops": "bus_stops",
        "Theatres": "theatres",
        "Restaurants": "restaurants",
        "Pharmacies": "pharmacies",
        "Parking": "parking"
    }
    chosen = st.multiselect("Select categories (OSM types)", options=list(friendly.keys()),
                            default=["Shops", "Banks", "Schools"])

    # map friendly names to internal keys
    selected_categories = [friendly[name] for name in chosen]

    # Find button
    if st.button("Find best locations"):
        with st.spinner("Computing candidates..."):
            cand_geojson = compute_candidates(place=DEFAULT_AREA,
                                              selected_categories=selected_categories,
                                              bandwidth=kde_bandwidth,
                                              grid_size=grid_size,
                                              competitor_radius=competitor_radius,
                                              top_n=12)
            st.session_state.candidates_geojson = cand_geojson
            st.success(f"Found {len(cand_geojson.get('features', []))} candidate points")

    # show table + csv if present
    if "candidates_geojson" in st.session_state and st.session_state.candidates_geojson.get("features"):
        feats = st.session_state.candidates_geojson["features"]
        rows = []
        for f in feats:
            geom = f.get("geometry")
            lon, lat = (geom["coordinates"][0], geom["coordinates"][1]) if geom else (None, None)
            rows.append({"lat": lat, "lon": lon, "score": f["properties"]["final"], "competitors": f["properties"]["competitors"]})
        df = pd.DataFrame(rows)
        st.markdown("**Candidate locations**")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download candidates (CSV)", data=csv, file_name="candidates.csv", mime="text/csv")

with right_col:
    # Build the HTML map doc (same as your previous iframe-based map) and inject candidates
    candidates_geojson_str = json.dumps(st.session_state.get("candidates_geojson", {"type":"FeatureCollection","features":[]}))

    html_doc = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Delhi Fullscreen Smoothed (NoScroll)</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>
        html, body {{ margin:0; padding:0; height:100%; background:#111; font-family: "Segoe UI", Roboto, Arial, sans-serif; }}
        #map {{ width:100%; height:80vh; border-radius:12px; overflow:hidden; }}
        .overlay-card {{ position:absolute; left:22px; top:22px; z-index:6000; width:320px; background: rgba(255,255,255,0.12); padding:14px; border-radius:18px; backdrop-filter: blur(8px); color:#111; }}
        .right-card {{ position:absolute; right:22px; top:22px; z-index:6000; width:360px; background: rgba(255,255,255,0.12); padding:14px; border-radius:18px; backdrop-filter: blur(8px); color:#111; }}
        .candidate-marker-label {{ font-size:12px; font-weight:700; color: #fff; padding:6px 8px; border-radius:8px; background:{PURPLE_BORDER}; }}
      </style>
    </head>
    <body>
      <div id="map"></div>

      <div class="overlay-card">
        <h3 style="color:{PURPLE_BORDER}; margin:0 0 6px 0;">Delhi Map (Smoothed)</h3>
        <div style="font-size:13px; color:#222;">
          <p style="margin:6px 0;">• Purple border shows the smoothed Delhi administrative area (OSM).</p>
          <p style="margin:6px 0;">• Use the left panel to pick categories and compute candidate locations.</p>
        </div>
      </div>

      <div class="right-card">
        <h3 style="color:{PURPLE_BORDER}; margin:0 0 6px 0;">Clicked Location</h3>
        <div id="info" style="font-size:13px; color:#111;">
          <p>Click the map to view coordinates and address here.</p>
        </div>
      </div>

    <script>
      const delhiGeo = {delhi_geojson_str};
      const candidatesGeo = {candidates_geojson_str};

      const map = L.map('map', {{ zoomControl:true, attributionControl:true }});
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
      }}).addTo(map);

      // draw Delhi polygon
      if (delhiGeo.features && delhiGeo.features.length > 0) {{
        const layer = L.geoJSON(delhiGeo, {{
          style: function(feature) {{
            return {{ color: '{PURPLE_BORDER}', weight: 3, fillColor: '{FILL_COLOR}', fillOpacity: 0.08 }};
          }}
        }}).addTo(map);
        map.fitBounds(layer.getBounds().pad(0.05));
      }} else {{
        map.setView([28.6139,77.2090], 11);
      }}

      // draw candidate markers
      let candidateLayer = null;
      function drawCandidates() {{
        if (candidateLayer) {{ try {{ map.removeLayer(candidateLayer); }} catch(e){{}} candidateLayer = null; }}
        if (!candidatesGeo || !candidatesGeo.features || candidatesGeo.features.length === 0) return;
        candidateLayer = L.geoJSON(candidatesGeo, {{
          pointToLayer: function(feature, latlng) {{
            const score = feature.properties && feature.properties.final ? feature.properties.final.toFixed(3) : '';
            const comp = feature.properties && feature.properties.competitors ? feature.properties.competitors : '';
            const label = L.divIcon({{
              html: `<div class="candidate-marker-label">${{(score)}}<div style="font-size:10px;">c:${{comp}}</div></div>`,
              className: '',
              iconSize: [56, 30],
              iconAnchor: [28, 15]
            }});
            return L.marker(latlng, {{ icon: label }});
          }},
          onEachFeature: function(feature, layer) {{
            const body = `<b>Score:</b> ${{feature.properties.final.toFixed(3)}}<br><b>Competitors:</b> ${{feature.properties.competitors}}`;
            layer.bindPopup(body);
          }}
        }}).addTo(map);
      }}
      drawCandidates();

      // reverse geocode
      async function reverseGeocode(lat, lon) {{
        try {{
          const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${{lat}}&lon=${{lon}}&addressdetails=1`;
          const resp = await fetch(url, {{ headers: {{ "Accept": "application/json" }} }});
          if (!resp.ok) return {{ ok:false, error: resp.status + " " + resp.statusText }};
          const data = await resp.json();
          return {{ ok:true, data }};
        }} catch (err) {{
          return {{ ok:false, error: err.toString() }};
        }}
      }}

      function setInfo(lat, lon, payload) {{
        const infoDiv = document.getElementById('info');
        let html = `<p style="font-size:13px;"><b>Latitude:</b> ${{lat.toFixed(6)}}</p>`;
        html += `<p style="font-size:13px;"><b>Longitude:</b> ${{lon.toFixed(6)}}</p><hr style="border:none;border-top:1px solid rgba(0,0,0,0.06);">`;
        if (!payload.ok) {{
          html += `<p style="color:#900;">Address unavailable (${{payload.error||'error'}})</p>`;
        }} else {{
          const d = payload.data;
          const addr = d.address || {{}};
          html += `<p style="font-size:13px;"><b>Address:</b><br>${{d.display_name || ''}}</p>`;
          html += `<p style="font-size:12px;color:#333;"><b>City:</b> ${{addr.city || addr.town || addr.village || '-'}}<br><b>State:</b> ${{addr.state || '-'}}<br><b>Postcode:</b> ${{addr.postcode || '-'}}</p>`;
        }}
        infoDiv.innerHTML = html;
      }}

      let clickMarker = null;
      map.on('click', async function(e) {{
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;
        if (!clickMarker) {{
          clickMarker = L.circleMarker([lat, lon], {{ radius:6, color: '{PURPLE_BORDER}', fillColor: '{PURPLE_BORDER}', fillOpacity:1 }}).addTo(map);
        }} else {{
          clickMarker.setLatLng([lat, lon]);
        }}
        const infoDiv = document.getElementById('info');
        infoDiv.innerHTML = `<p style="font-size:13px;"><b>Latitude:</b> ${{lat.toFixed(6)}}</p><p style="font-size:13px;"><b>Longitude:</b> ${{lon.toFixed(6)}}</p><p style="font-size:12px;">Resolving address...</p>`;
        const res = await reverseGeocode(lat, lon);
        setInfo(lat, lon, res);
      }});
    </script>
    </body>
    </html>
    """

    # embed HTML
    st_html(html_doc, height=820, scrolling=False)
