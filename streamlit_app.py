# streamlit_app.py
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
from shapely.geometry import Point, box
import folium
from streamlit_folium import st_folium
from shapely.ops import unary_union

st.set_page_config(layout="wide", page_title="SiteSelector - School/Hospital Location Tool")

st.title("SiteSelector — Candidate Site Ranking for School / Hospital 🗺️")

# ---- Sidebar inputs ----
st.sidebar.header("Study area input")
uploaded_shp = st.sidebar.file_uploader("Upload study area (GeoJSON / Shapefile zip)", type=["geojson","zip","shp"], accept_multiple_files=False)
use_bbox = st.sidebar.checkbox("Or enter bounding box (minx,miny,maxx,maxy)", value=False)
bbox_input = None
if use_bbox:
    bbox_input = st.sidebar.text_input("BBox (comma separated)", "77.55,12.90,77.70,13.05")  # example for Bengaluru
st.sidebar.markdown("---")
st.sidebar.header("Site & analysis settings")
service_km = st.sidebar.number_input("Service radius (km)", value=2.0, min_value=0.1, step=0.1)
grid_res_m = st.sidebar.number_input("Grid resolution (meters)", value=250, min_value=50, step=50)
top_n = st.sidebar.number_input("Top N results to show", value=10, min_value=1, step=1)
facility_type = st.sidebar.selectbox("Facility type", ["school","hospital"])
st.sidebar.markdown("---")
st.sidebar.write("If you have candidate points (CSV with lon,lat), upload below")
cand_upload = st.sidebar.file_uploader("Upload candidate CSV (lon,lat,name optional)", type=["csv"], accept_multiple_files=False)

# ---- Load or create study area ----
@st.cache_data
def load_area_from_bbox(bbox_txt):
    minx,miny,maxx,maxy = [float(x.strip()) for x in bbox_txt.split(",")]
    g = gpd.GeoDataFrame({"id":[1]}, geometry=[box(minx,miny,maxx,maxy)], crs="EPSG:4326")
    return g

@st.cache_data
def load_osm_for_bbox(minx,miny,maxx,maxy):
    north, south, east, west = maxy, miny, maxx, minx
    # using ox.geometries_from_bbox to pull amenities and buildings (may take time)
    tags = {"amenity": True, "building": True, "shop": True}
    try:
        gdf = ox.geometries_from_bbox(north, south, east, west, tags)
    except Exception as e:
        st.error(f"OSM fetch failed: {e}")
        return None
    return gdf

study_area = None
if uploaded_shp:
    try:
        # note: handle shapefile zip or geojson simply by geopandas.read_file
        if uploaded_shp.type == "application/zip":
            import zipfile, tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".zip")
            tmp.write(uploaded_shp.getvalue())
            tmp.flush()
            with zipfile.ZipFile(tmp.name, "r") as z:
                # geopandas can read from zip path
                g = gpd.read_file(f"zip://{tmp.name}")
        else:
            # geojson or shp direct
            tmp = uploaded_shp
            g = gpd.read_file(tmp)
        study_area = g.to_crs("EPSG:4326")
        st.sidebar.success("Study area loaded from file.")
    except Exception as e:
        st.sidebar.error(f"Failed to load study area: {e}")

if study_area is None and use_bbox:
    try:
        study_area = load_area_from_bbox(bbox_input)
        st.sidebar.success("BBox study area created.")
    except Exception as e:
        st.sidebar.error("Failed to parse bbox. Use minx,miny,maxx,maxy.")

if study_area is None:
    st.warning("No study area provided yet. The app will generate a demo bbox.")
    # demo bbox (approx central Bengaluru)
    study_area = load_area_from_bbox("77.55,12.90,77.70,13.05")

st.write("### Study area preview")
st.write(f"CRS: {study_area.crs}")
m = folium.Map(location=[study_area.geometry.centroid.y.mean(), study_area.geometry.centroid.x.mean()], zoom_start=12)
folium.GeoJson(study_area.__geo_interface__, name="study_area").add_to(m)
st_folium(m, width=900, height=400)

# ---- Create candidate grid or load user candidates ----
@st.cache_data
def create_grid(area_gdf, res_m=250):
    # project to mercator for meters
    area = area_gdf.to_crs(epsg=3857)
    minx,miny,maxx,maxy = area.total_bounds
    xs = np.arange(minx, maxx, res_m)
    ys = np.arange(miny, maxy, res_m)
    pts = []
    for x in xs:
        for y in ys:
            pts.append(Point(x + res_m/2, y + res_m/2))
    g = gpd.GeoDataFrame(geometry=pts, crs="EPSG:3857")
    # clip to area
    mask = g.within(unary_union(area.geometry))
    g = g[mask].copy()
    g = g.to_crs(epsg=4326)
    g["id"] = range(len(g))
    return g

if cand_upload:
    try:
        df = pd.read_csv(cand_upload)
        if {'lon','lat'}.issubset(set(df.columns)):
            g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
            candidates = g
            st.success("Candidate points loaded from CSV.")
        else:
            st.error("CSV must contain 'lon' and 'lat' columns.")
            candidates = create_grid(study_area, grid_res_m)
    except Exception as e:
        st.error(f"Error reading candidate CSV: {e}")
        candidates = create_grid(study_area, grid_res_m)
else:
    candidates = create_grid(study_area, grid_res_m)
    st.info(f"Generated grid candidates: {len(candidates)} points")

st.write("Sample of candidate points")
st.dataframe(candidates.head())

# ---- Fetch / derive simple indicators from OSM ----
st.write("### Computing indicators (using OSM amenities where possible)...")
minx, miny, maxx, maxy = study_area.total_bounds
osm_g = None
with st.spinner("Fetching OSM data for bbox (may take ~10-30s)..."):
    try:
        osm_g = load_osm_for_bbox(minx, miny, maxx, maxy)
    except Exception:
        osm_g = None

# Quick helper: extract amenity points of interest relevant to facility_type
def extract_pois(osm_gdf, key="amenity"):
    if osm_gdf is None or osm_gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    # convert buildings/amenities to points where needed
    pts = osm_gdf[osm_gdf.geometry.type=="Point"].copy()
    # keep if amenity present or shop or building centroids
    return pts.to_crs("EPSG:4326")

pois = extract_pois(osm_g)
st.write(f"OSM objects found: {len(pois)} (points, buildings)")

# indicator computations
@st.cache_data
def compute_indicators(candidates_gdf, study_area_gdf, pois_gdf, service_km=2.0):
    cand = candidates_gdf.copy()
    cand = cand.to_crs(epsg=3857)
    pois_p = pois_gdf.to_crs(epsg=3857) if (pois_gdf is not None and not pois_gdf.empty) else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3857")
    # simple: count POIs in radius (proxy for population/activity). Use buffer.
    r_m = int(service_km * 1000)
    cand["buffer"] = cand.geometry.buffer(r_m)
    counts = []
    nearest_dist = []
    for idx,row in cand.iterrows():
        buf = row["buffer"]
        if not pois_p.empty:
            pts_in = pois_p[pois_p.geometry.within(buf)]
            counts.append(len(pts_in))
            # compute nearest distance to any POI (meters)
            if len(pts_in)>0:
                d = pts_in.geometry.distance(row.geometry).min()
                nearest_dist.append(float(d))
            else:
                nearest_dist.append(float(r_m)*1.5)
        else:
            counts.append(0)
            nearest_dist.append(float(r_m)*1.5)
    cand["poi_count_service"] = counts
    cand["dist_to_poi_m"] = nearest_dist
    # distance to study area centroid as proxy for center
    centroid = study_area_gdf.to_crs(epsg=3857).geometry.centroid.unary_union
    cand["dist_to_center_m"] = cand.geometry.distance(centroid)
    # placeholder: land suitability (random) — ideally from land use data
    cand["land_suitability"] = np.random.rand(len(cand))
    # drop buffer and reproject back
    cand = cand.drop(columns=["buffer"])
    return cand.to_crs(epsg=4326)

indicators = compute_indicators(candidates, study_area, pois, service_km)
st.success("Indicators computed (quick proxies).")

# ---- Weight UI ----
st.write("### Weight your priorities")
col1, col2 = st.columns(2)
with col1:
    w_pop = st.slider("Population/Activity in service area (higher better)", 0.0, 1.0, 0.4)
    w_comp = st.slider("Distance from competitors (higher better)", 0.0, 1.0, 0.2)
with col2:
    w_access = st.slider("Access to amenities/POIs (lower distance better)", 0.0, 1.0, 0.2)
    w_land = st.slider("Land suitability (site-specific)", 0.0, 1.0, 0.2)

# normalize weights to sum=1
weights = np.array([w_pop, w_comp, w_access, w_land])
if weights.sum() == 0:
    weights = np.array([0.25,0.25,0.25,0.25])
weights = weights / weights.sum()

# ---- compute normalized score ----
def normalize_series(s):
    if s.max() == s.min():
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

df = indicators.copy()
# indicators chosen:
# pop/activity proxy: poi_count_service  -> higher better
# competitor proxy: dist_to_poi_m (if many POIs of same type near => small distance => negative) -> higher better
# access: dist_to_center_m or dist_to_poi_m as inverse -> lower better
# land: land_suitability -> higher better
df["pop_norm"] = normalize_series(df["poi_count_service"])
df["comp_norm"] = normalize_series(df["dist_to_poi_m"])  # bigger distance means fewer nearby (better)
df["access_norm"] = 1 - normalize_series(df["dist_to_center_m"])  # closer to center better
df["land_norm"] = normalize_series(df["land_suitability"])

# combine
df["score"] = (weights[0]*df["pop_norm"] +
               weights[1]*df["comp_norm"] +
               weights[2]*df["access_norm"] +
               weights[3]*df["land_norm"])

df = df.sort_values("score", ascending=False)

st.write(f"Showing top {top_n} candidate sites")
st.dataframe(df[["id","score","poi_count_service","dist_to_poi_m","dist_to_center_m"]].head(int(top_n)))

# ---- Map top results ----
top = df.head(int(top_n)).to_crs(epsg=4326)

m2 = folium.Map(location=[study_area.geometry.centroid.y.mean(), study_area.geometry.centroid.x.mean()], zoom_start=12)
folium.GeoJson(study_area.__geo_interface__, name="Study Area").add_to(m2)
# add candidate points
for _,r in df.head(int(top_n)).iterrows():
    folium.CircleMarker(location=[r.geometry.y, r.geometry.x],
                        radius=6,
                        popup=folium.Popup(f"ID: {r['id']}<br>Score: {r['score']:.3f}<br>POIcount: {r['poi_count_service']}"),
                        color="blue", fill=True).add_to(m2)
st.write("Map of top candidates")
st_folium(m2, width=900, height=500)

st.write("### Notes & next steps")
st.markdown("""
- This demo uses quick proxies from OSM. Replace `land_suitability` and population proxies with real data (census, building occupancy, land parcels).
- For **network travel times**, replace Euclidean buffer counts with network isochrones (use `osmnx` `graph` and `ox.distance.nearest_nodes` / `ox.shortest_path_length`).
- For **explainability**, display per-candidate normalized indicator breakdown (radar or bar).
- To move to ML, collect historical labeled sites and train a classifier (RF/XGBoost) using these indicators and spatial cross-validation.
""")
