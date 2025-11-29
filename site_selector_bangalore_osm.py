# delhi_fullscreen_clientside_fixed2.py
import streamlit as st
st.set_page_config(layout="wide", page_title="Delhi Fullscreen Map (fixed 2)", initial_sidebar_state="collapsed")

import osmnx as ox
import json
from streamlit.components.v1 import html as st_html

# ---------- constants ----------
PURPLE_BORDER = "#7B1FA2"
FILL_COLOR = "#f5e9fb"
DEFAULT_AREA = "Delhi, India"

# ---------- fetch Delhi geojson ----------
@st.cache_data(show_spinner=False)
def load_delhi_geojson():
    try:
        g = ox.geocode_to_gdf(DEFAULT_AREA)
        g = g.to_crs(epsg=4326)
        return json.loads(g.to_json())
    except Exception:
        return {"type": "FeatureCollection", "features": []}

delhi_geojson = load_delhi_geojson()
delhi_geojson_str = json.dumps(delhi_geojson)

# ---------- HTML (no integrity attributes, ensure map.invalidateSize) ----------
html_doc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Delhi Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <!-- Leaflet CSS/JS (no integrity attributes) -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      background: #111;
      font-family: "Segoe UI", Roboto, Arial, sans-serif;
    }}
    #map {{
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      z-index: 0;
      background: #222;
    }}
    .overlay {{
      position: fixed;
      z-index: 99999;
      padding: 16px;
      border-radius: 12px;
      background: rgba(255,255,255,0.18);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      box-shadow: 0 8px 26px rgba(0,0,0,0.18);
      color: #111;
    }}
    .left-panel {{ left: 20px; top: 20px; width: 320px; }}
    .right-panel {{ right: 20px; top: 20px; width: 380px; max-width: 42%; overflow-wrap: break-word; }}
    h2.overlay-title {{ margin: 0 0 8px 0; color: {PURPLE_BORDER}; font-size: 20px; }}
    p.small {{ margin: 6px 0; font-size: 13px; color:#222; }}
    @media (max-width: 900px) {{
      .left-panel, .right-panel {{ width: 42vw; left: 12px; right: 12px; }}
      .right-panel {{ top: auto; bottom: 18px; }}
    }}
    .leaflet-control-attribution {{ font-size: 11px !important; }}
  </style>
</head>
<body>
  <div id="map"></div>

  <div class="overlay left-panel">
    <h2 class="overlay-title">Delhi Map</h2>
    <div style="font-size:13px; line-height:1.35;">
      <p class="small">• Purple border shows Delhi administrative area (OSM).</p>
      <p class="small">• Click anywhere on the map to show coordinates &amp; address on the right.</p>
      <p class="small" style="font-size:12px; color:#444;">Note: Reverse geocoding uses public Nominatim (rate-limited).</p>
    </div>
  </div>

  <div class="overlay right-panel">
    <h2 class="overlay-title">Clicked Location</h2>
    <div id="info" style="font-size:13px; color:#111;">
      <p>Click on the map to view coordinates and address here.</p>
    </div>
  </div>

<script>
  const delhiGeo = {delhi_geojson_str};

  // create map
  const map = L.map('map', {{ zoomControl: true, attributionControl: true }});

  // OSM tiles
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  // draw Delhi boundary if available
  if (delhiGeo.features && delhiGeo.features.length > 0) {{
    const gLayer = L.geoJSON(delhiGeo, {{
      style: function(feature) {{
        return {{ color: '{PURPLE_BORDER}', weight: 3, fillColor: '{FILL_COLOR}', fillOpacity: 0.08 }};
      }}
    }}).addTo(map);
    map.fitBounds(gLayer.getBounds().pad(0.05));
  }} else {{
    map.setView([28.6139, 77.2090], 11);
  }}

  // ensure proper layout after render and on resize
  function safeInvalidate() {{
    setTimeout(function() {{
      try {{ map.invalidateSize(); }} catch(e){{ console.warn("invalidateSize failed", e); }}
    }}, 250);
  }}
  window.addEventListener('load', safeInvalidate);
  window.addEventListener('resize', safeInvalidate);
  safeInvalidate();

  // reverse geocode via public nominatim
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

  function updateInfo(lat, lon, payload) {{
    const infoDiv = document.getElementById('info');
    let html = `<p class="small"><b>Latitude:</b> ${{lat.toFixed(6)}}</p>`;
    html += `<p class="small"><b>Longitude:</b> ${{lon.toFixed(6)}}</p>`;
    html += '<hr style="border:none;border-top:1px solid rgba(0,0,0,0.06);">';
    if (!payload.ok) {{
      html += `<p class="small" style="color:#900;">Address unavailable (${{payload.error||'error'}})</p>`;
    }} else {{
      const d = payload.data;
      const addr = d.address || {{}};
      html += `<p class="small"><b>Address:</b><br>${{d.display_name || ''}}</p>`;
      html += `<p class="small" style="font-size:12px; color:#333;"><b>City:</b> ${{addr.city || addr.town || addr.village || '-'}}<br><b>State:</b> ${{addr.state || '-'}}<br><b>Postcode:</b> ${{addr.postcode || '-'}}</p>`;
    }}
    infoDiv.innerHTML = html;
  }}

  // marker for clicked location
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
    infoDiv.innerHTML = `<p class="small"><b>Latitude:</b> ${{lat.toFixed(6)}}</p><p class="small"><b>Longitude:</b> ${{lon.toFixed(6)}}</p><p class="small">Resolving address...</p>`;

    const res = await reverseGeocode(lat, lon);
    updateInfo(lat, lon, res);
  }});
</script>
</body>
</html>
"""

# ---------- embed HTML into Streamlit ----------
# Use a large height so the iframe size is generous; the map inside the iframe is 100vh and will fill it.
st_html(html_doc, height=1100, scrolling=True)
