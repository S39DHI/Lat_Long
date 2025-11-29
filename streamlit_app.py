import json
p = "/mnt/data/India.geojson"   # change path if different
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

# top-level info
print("top-level type:", data.get("type"))
features = data.get("features")
if features is None:
    print("No 'features' key found. GeoJSON may be a single Feature or geometry.")
else:
    print("feature count:", len(features))
    if len(features) > 0:
        # print first 3 feature summaries
        for i, feat in enumerate(features[:3]):
            geom_type = feat.get("geometry", {}).get("type")
            props = feat.get("properties", {})
            print(f"Feature {i}: geom_type={geom_type}, prop_keys={list(props.keys())[:8]}")
# If it was not a FeatureCollection, inspect the top-level geometry
if features is None:
    if "geometry" in data:
        print("top-level geometry type:", data["geometry"].get("type"))
    else:
        print("Unknown structure — show top-level keys:", list(data.keys()))
