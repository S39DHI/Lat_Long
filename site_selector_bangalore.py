# state_district_working_app.py
import streamlit as st
st.set_page_config(layout="wide", page_title="India State/District Selector", initial_sidebar_state="collapsed")

import json
import os
import urllib.parse
import requests
import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point
from streamlit.components.v1 import html as st_html
from sklearn.neighbors import KernelDensity

# ---------------- CONFIG ----------------
PURPLE_BORDER = "#7B1FA2"
FILL_COLOR = "#f5e9fb"
SMOOTH_ITERS = 3

# ---------------- smoothing routine (Chaikin) ----------------
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

# ---------------- load & smooth generic area (state/district) ----------------
@st.cache_data(show_spinner=False)
def load_and_smooth_area(area_name, smooth_iters=3):
    try:
        q = area_name if "India" in area_name else f"{area_name}, India"
        gdf = ox.geocode_to_gdf(q)
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

# ---------------- safe district lookup ----------------
@st.cache_data(show_spinner=True)
def get_districts_for_state(state_name):
    try:
        local_path = os.path.join(os.path.dirname(__file__), "india_districts.json")
    except Exception:
        local_path = "india_districts.json"
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            names = data.get(state_name) or data.get(state_name.title()) or data.get(state_name.upper()) or []
            names = sorted(list({n.strip() for n in names if isinstance(n, str) and n.strip()}), key=lambda s: s.lower())
            return names
        except Exception:
            pass

    try:
        overpass = "https://overpass-api.de/api/interpreter"
        q = f"""
        [out:json][timeout:12];
        area["name"="{state_name}"]["boundary"="administrative"]["admin_level"~"[24]"];
        relation(area)[admin_level=6][boundary=administrative];
        out tags;
        """
        resp = requests.post(overpass, data={"data": q}, timeout=12)
        if resp.status_code == 200:
            j = resp.json()
            names = []
            for el in j.get("elements", []):
                tags = el.get("tags", {})
                n = tags.get("name")
                if isinstance(n, str) and n.strip():
                    names.append(n.strip())
            names = sorted(list(set(names)), key=lambda s: s.lower())
            return names
    except Exception:
        return []

    return []

# ---------------- helper: OSM tags for categories ----------------
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

# ---------------- candidate computation ----------------
@st.cache_data(show_spinner=True)
def compute_candidates(place= "Delhi, India",
                       selected_categories=None,
                       bandwidth=800,
                       grid_size=250,
                       competitor_radius=500,
                       top_n=12,
                       proj_crs="EPSG:3857"):
    ox.config(use_cache=True, log_console=False)
    if not selected_categories:
        selected_categories = ["shops", "banks", "hospitals", "schools", "bus_stops"]

    try:
        area_gdf = ox.geocode_to_gdf(place)
    except Exception:
        return {"type":"FeatureCollection","features":[]}

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

    pois = gpd.GeoDataFrame(pd.concat(poi_frames, ignore_index=True), crs=poi_frames[0].crs)
    area_gdf = area_gdf.to_crs(epsg=4326).to_crs(proj_crs)
    pois = pois.to_crs(proj_crs)

    pois['geometry'] = pois.geometry.centroid
    area_poly = area_gdf.geometry.unary_union
    pois = pois[pois.geometry.within(area_poly)].reset_index(drop=True)

    if len(pois) < 3:
        return {"type":"FeatureCollection","features":[]}

    coords = np.vstack([pois.geometry.x.values, pois.geometry.y.values]).T
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(coords)

    minx, miny, maxx, maxy = area_gdf.geometry.total_bounds
    xv = np.arange(minx, maxx, grid_size)
    yv = np.arange(miny, maxy, grid_size)
    if len(xv) == 0 or len(yv) == 0:
        return {"type":"FeatureCollection","features":[]}
    xx, yy = np.meshgrid(xv, yv)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    zs = np.exp(kde.score_samples(grid_coords))
    Z = zs.reshape(len(yv), len(xv))

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

    scores = np.array([c['score'] for c in candidates], dtype=float)
    comps = np.array([c['competitors'] for c in candidates], dtype=float)
    score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    comp_norm = 1 - (comps - comps.min()) / (comps.max() - comps.min() + 1e-9)
    final = 0.7 * score_norm + 0.3 * comp_norm
    for i, c in enumerate(candidates):
        c['final'] = float(final[i])

    candidates = sorted(candidates, key=lambda d: d['final'], reverse=True)[:top_n]

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

# ---------------- UI & main layout ----------------
params = st.experimental_get_query_params()
pre_lat = params.get("lat", [""])[0]
pre_lon = params.get("lon", [""])[0]
pre_r = params.get("r", ["10"])[0]
pre_state = params.get("state", [""])[0]
pre_district = params.get("district", [""])[0]

INDIA_STATES_AND_UTS = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana",
    "Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur",
    "Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana",
    "Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
    "Andaman and Nicobar Islands","Chandigarh","Dadra and Nagar Haveli and Daman and Diu","Lakshadweep",
    "Delhi","Jammu and Kashmir","Ladakh","Puducherry"
]

left_col, right_col = st.columns([0.35, 0.65])

with left_col:
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
        <p style="font-size:13px; margin:4px 0 12px 0; color:#e6e6e6;">
          Select state -> district dropdown will appear (best-effort). Click map to fill fields (won't compute until you press "Find best locations").
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    initial_state_index = 0
    if pre_state and pre_state in INDIA_STATES_AND_UTS:
        initial_state_index = INDIA_STATES_AND_UTS.index(pre_state)
    selected_state = st.selectbox("Select State / Union Territory", INDIA_STATES_AND_UTS, index=initial_state_index)

    with st.spinner("Loading districts for selected state..."):
        district_list = get_districts_for_state(selected_state)

    if pre_district and pre_district not in district_list and pre_district.strip():
        district_list = [pre_district] + district_list

    if district_list:
        opts = [""] + district_list
        pre_index = 0
        if pre_district and pre_district in district_list:
            pre_index = opts.index(pre_district)
        district_choice = st.selectbox("District (auto-filled when you click map)", opts, index=pre_index)
    else:
        district_choice = st.text_input("District (auto-filled when you click map)", value=pre_district or "")

    kde_bandwidth = st.slider("KDE bandwidth (m)", min_value=200, max_value=2000, value=800, step=100)
    grid_size = st.slider("Grid size (m)", min_value=100, max_value=500, value=250, step=50)
    competitor_radius = st.slider("Competitor radius (m)", min_value=200, max_value=1500, value=500, step=50)

    friendly = {
        "Shops": "shops", "Banks": "banks", "Hospitals": "hospitals", "Schools": "schools",
        "Clinics": "clinics", "Bus stops": "bus_stops", "Theatres": "theatres",
        "Restaurants": "restaurants", "Pharmacies": "pharmacies", "Parking": "parking"
    }
    chosen = st.multiselect("Select categories (OSM types)", options=list(friendly.keys()), default=["Shops","Banks","Schools"])
    selected_categories = [friendly[n] for n in chosen]

    st.markdown("**Center (auto-filled when you click inside selected district/state)**")
    center_lat = st.text_input("Center latitude (e.g. 28.6139)", value=pre_lat or "")
    center_lon = st.text_input("Center longitude (e.g. 77.2090)", value=pre_lon or "")
    search_radius_km = st.slider("Search radius around center (km)", min_value=1, max_value=200, value=int(pre_r) if pre_r else 10)

    if st.button("Find best locations"):
        if district_choice and district_choice.strip():
            place_to_use = f"{district_choice}, {selected_state}, India"
        else:
            place_to_use = f"{selected_state}, India"
        with st.spinner("Computing candidates..."):
            cand_geojson = compute_candidates(place=place_to_use,
                                              selected_categories=selected_categories,
                                              bandwidth=kde_bandwidth,
                                              grid_size=grid_size,
                                              competitor_radius=competitor_radius,
                                              top_n=12)
            st.session_state.candidates_geojson = cand_geojson
            st.success(f"Found {len(cand_geojson.get('features', []))} candidate points")

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
    if district_choice and isinstance(district_choice, str) and district_choice.strip():
        area_query = f"{district_choice}, {selected_state}, India"
        title = f"{district_choice} — {selected_state}"
    else:
        area_query = f"{selected_state}, India"
        title = selected_state

    area_geojson = load_and_smooth_area(area_query, smooth_iters=SMOOTH_ITERS)
    candidates_geojson_str = json.dumps(st.session_state.get("candidates_geojson", {"type":"FeatureCollection","features":[]}))
    area_geojson_str = json.dumps(area_geojson)
    pre_lat_js = json.dumps(pre_lat if pre_lat else None)
    pre_lon_js = json.dumps(pre_lon if pre_lon else None)

    html_doc = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>{title} Map</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/@turf/turf@6/turf.min.js"></script>
      <style>
        html, body {{ margin:0; padding:0; height:100%; background:#111; font-family: "Segoe UI", Roboto, Arial, sans-serif; }}
        #map {{ width:100%; height:80vh; border-radius:12px; overflow:hidden; }}
        .right-card {{ position:absolute; right:22px; top:22px; z-index:6000; width:360px; background: rgba(255,255,255,0.92); padding:14px; border-radius:12px; color:#111; box-shadow: 0 6px 20px rgba(0,0,0,0.3); }}
        .candidate-marker-label {{ font-size:12px; font-weight:700; color: #fff; padding:6px 8px; border-radius:8px; background:{PURPLE_BORDER}; }}
      </style>
    </head>
    <body>
      <div id="map"></div>
      <div class="right-card">
        <h3 style="color:{PURPLE_BORDER}; margin:0 0 6px 0;">Clicked Location</h3>
        <div id="info">Click the map to view coordinates and address here.</div>
        <hr/>
        <label for="radiusRange">Radius (km): <span id="radiusVal">10</span></label>
        <input id="radiusRange" type="range" min="1" max="200" value="10" style="width:100%; margin-top:6px;">
        <div style="margin-top:8px; display:flex; gap:8px;">
          <button id="copyCoords" style="flex:1;padding:8px;border-radius:8px;border:none;background:{PURPLE_BORDER};color:#fff;">Copy lat,lon</button>
          <button id="centerArea" style="flex:1;padding:8px;border-radius:8px;border:1px solid #ccc;background:#fff;color:#111;">Center {title}</button>
        </div>
      </div>

    <script>
      const areaGeo = {area_geojson_str};
      const candidatesGeo = {candidates_geojson_str};
      const preLat = {pre_lat_js};
      const preLon = {pre_lon_js};

      const map = L.map('map', {{ zoomControl:true, attributionControl:true }});
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
      }}).addTo(map);

      let areaLayer = null;
      if (areaGeo.features && areaGeo.features.length > 0) {{
        areaLayer = L.geoJSON(areaGeo, {{
          style: function(feature) {{
            return {{ color: '{PURPLE_BORDER}', weight: 3, fillColor: '{FILL_COLOR}', fillOpacity: 0.06 }};
          }}
        }}).addTo(map);
        map.fitBounds(areaLayer.getBounds().pad(0.05));
      }} else {{
        map.setView([22.0,78.0], 5);
      }}

      let candidateLayer = null;
      function drawCandidates() {{
        if (candidateLayer) {{ try {{ map.removeLayer(candidateLayer); }} catch(e){{}} candidateLayer = null; }}
        if (!candidatesGeo || !candidatesGeo.features || candidatesGeo.features.length === 0) return;
        candidateLayer = L.geoJSON(candidatesGeo, {{
          pointToLayer: function(feature, latlng) {{
            const score = feature.properties && feature.properties.final ? feature.properties.final.toFixed(3) : '';
            const comp = feature.properties && feature.properties.competitors ? feature.properties.competitors : '';
            const label = L.divIcon({{
              html: `<div class="candidate-marker-label">${{score}}<div style="font-size:10px;">c:${{comp}}</div></div>`,
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

      async function reverseGeocode(lat, lon) {{
        try {{
          const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${{lat}}&lon=${{lon}}&addressdetails=1`;
          const resp = await fetch(url);
          if (!resp.ok) return {{ ok:false }};
          const data = await resp.json();
          return {{ ok:true, data }};
        }} catch (err) {{
          return {{ ok:false }};
        }}
      }}

      function setInfo(lat, lon, payload) {{
        const info = document.getElementById('info');
        let html = `<p><b>Latitude:</b> ${{lat.toFixed(6)}}</p><p><b>Longitude:</b> ${{lon.toFixed(6)}}</p>`;
        if (!payload || !payload.ok) {{
          html += `<p style="color:#900;">Address unavailable</p>`;
        }} else {{
          html += `<p style="font-size:13px;color:#444;">${{payload.data.display_name||''}}</p>`;
        }}
        info.innerHTML = html;
      }}

      const clickGroup = L.featureGroup().addTo(map);

      const radiusRange = document.getElementById('radiusRange');
      const radiusVal = document.getElementById('radiusVal');
      radiusVal.innerText = radiusRange.value;
      radiusRange.addEventListener('input', function() {{
        radiusVal.innerText = this.value;
        clickGroup.eachLayer(function(layer) {{
          if (layer instanceof L.Circle) {{
            layer.setRadius(parseFloat(radiusRange.value) * 1000);
          }}
        }});
      }});

      function removeOldClickLayers() {{
        const toRemove = [];
        map.eachLayer(function(layer) {{
          try {{
            if (layer && layer.options && layer.options._is_click_marker === true) {{
              toRemove.push(layer);
            }}
          }} catch(e){{/* ignore */}}
        }});
        toRemove.forEach(l => {{
          try {{ map.removeLayer(l); }} catch(e){{/* ignore */}}
        }});
      }}

      function placeClickMarker(lat, lon, radius_km) {{
        removeOldClickLayers();
        clickGroup.clearLayers();
        const marker = L.circleMarker([lat, lon], {{ radius:6, color: '{PURPLE_BORDER}', fillColor: '{PURPLE_BORDER}', fillOpacity:1 }});
        const circle = L.circle([lat, lon], {{ radius: (radius_km || parseFloat(radiusRange.value))*1000, color: '{PURPLE_BORDER}', weight:2, fillOpacity:0.06 }});
        try {{ marker.options._is_click_marker = true; circle.options._is_click_marker = true; }} catch(e){{/* ignore */}}
        clickGroup.addLayer(circle);
        clickGroup.addLayer(marker);
      }}

      if (preLat !== null && preLon !== null) {{
        try {{
          const lat = parseFloat(preLat);
          const lon = parseFloat(preLon);
          if (!isNaN(lat) && !isNaN(lon)) {{
            placeClickMarker(lat, lon, parseFloat(radiusRange.value));
            map.setView([lat, lon], Math.max(12, map.getZoom()));
          }}
        }} catch(e){{/* ignore */}}
      }}

      map.on('click', async function(e) {{
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;

        placeClickMarker(lat, lon, parseFloat(radiusRange.value));

        document.getElementById('info').innerHTML = `<p>Resolving address...</p>`;
        const res = await reverseGeocode(lat, lon);
        setInfo(lat, lon, res);

        let insideArea = false;
        try {{
          if (areaGeo.features && areaGeo.features.length > 0) {{
            const pt = turf.point([lon, lat]);
            for (let i=0;i<areaGeo.features.length;i++) {{
              if (turf.booleanPointInPolygon(pt, areaGeo.features[i])) {{ insideArea = true; break; }}
            }}
          }}
        }} catch (err) {{ insideArea = false; }}

        let stateName = "";
        let districtName = "";
        if (res && res.ok && res.data && res.data.address) {{
          const a = res.data.address;
          stateName = a.state || a.region || "";
          districtName = a.district || a.county || a.city_district || a.suburb || a.state_district || "";
        }}

        const q = `?lat=${{lat}}&lon=${{lon}}&r=${{parseFloat(radiusRange.value)}}&state=${{encodeURIComponent(stateName)}}&district=${{encodeURIComponent(districtName)}}`;
        try {{
          history.replaceState(null, '', q);
        }} catch(e) {{ /* ignore */ }}

        const text = lat.toFixed(6) + ',' + lon.toFixed(6);
        try {{
          await navigator.clipboard.writeText(text);
          if (insideArea) {{
            alert('Coordinates and district/state saved to URL (no reload).\\nCopied: ' + text + '\\nPaste into the left panel or press \"Find best locations\" to compute.');
          }} else {{
            alert('Clicked point copied: ' + text + '\\nPaste into the left panel center latitude/longitude if you want to compute there.');
          }}
        }} catch(e) {{
          if (insideArea) {{
            document.getElementById('info').innerHTML += `<p style="color:#0a0;">URL updated (no reload). Paste coords: ${{text}}</p>`;
          }} else {{
            document.getElementById('info').innerHTML += `<p style="color:#900;">Could not copy automatically. Paste coords: ${{text}}</p>`;
          }}
        }}
      }});

      document.getElementById('copyCoords').addEventListener('click', async function() {{
        let latlng = null;
        clickGroup.eachLayer(function(layer) {{
          if (layer instanceof L.CircleMarker) {{
            latlng = layer.getLatLng();
          }}
        }});
        if (!latlng) return alert('Click on the map first to pick a point.');
        const text = latlng.lat.toFixed(6) + ',' + latlng.lng.toFixed(6);
        try {{ await navigator.clipboard.writeText(text); alert('Copied: ' + text + '\\nNow paste into the left panel center latitude/longitude fields.'); }} catch(e) {{ alert('Could not copy to clipboard — please copy manually: ' + text); }}
      }});

      document.getElementById('centerArea').addEventListener('click', function() {{
        if (areaLayer) map.fitBounds(areaLayer.getBounds().pad(0.05));
      }});
    </script>
    </body>
    </html>
    """

    st_html(html_doc, height=820, scrolling=False)
