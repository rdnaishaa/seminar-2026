from pathlib import Path
import os

import folium
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CORRIDORS_PATH = ROOT / "corridors.csv"

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "forecasting_train_preprocessed.csv"
)

FIXED_EDGES_PATH = (
    ROOT
    / "outputs"
    / "results"
    / "fixed_graph_edges.csv"
)

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "figures"
    / "graph_satellite.html"
)

ENV_FILE = ROOT / ".env"


# ============================================================
# LOAD ENV
# ============================================================

load_dotenv(
    dotenv_path=ENV_FILE,
    override=False
)

TOMTOM_API_KEY = os.getenv(
    "TOMTOM_API_KEY"
)

if not TOMTOM_API_KEY:
    raise RuntimeError(
        "TOMTOM_API_KEY tidak ditemukan. "
        "Pastikan .env berisi TOMTOM_API_KEY."
    )


# ============================================================
# LOAD FINAL 14 CORRIDORS
# ============================================================

def load_top14():

    if not CORRIDORS_PATH.exists():
        raise FileNotFoundError(
            f"corridors.csv tidak ditemukan:\n"
            f"{CORRIDORS_PATH}"
        )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"forecasting_train_preprocessed.csv "
            f"tidak ditemukan:\n"
            f"{TRAIN_PATH}"
        )

    corridors = pd.read_csv(
        CORRIDORS_PATH
    )

    train = pd.read_csv(
        TRAIN_PATH
    )

    top14 = sorted(
        train["corridor_file"]
        .dropna()
        .astype(str)
        .unique()
    )

    corridors["station_id"] = (
        corridors["station_id"]
        .astype(str)
    )

    selected = corridors[
        corridors["station_id"].isin(top14)
    ].copy()

    selected = (
        selected
        .set_index("station_id")
        .loc[top14]
        .reset_index()
    )

    if len(selected) != 14:
        raise ValueError(
            f"Expected 14 corridors, "
            f"found {len(selected)}"
        )

    return selected


# ============================================================
# TOMTOM ROUTING
# ============================================================

def get_route_geometry(
    start_lat,
    start_lon,
    end_lat,
    end_lon
):

    url = (
        "https://api.tomtom.com/routing/1/"
        f"calculateRoute/"
        f"{start_lat},{start_lon}:"
        f"{end_lat},{end_lon}/json"
    )

    response = requests.get(
        url,
        params={
            "key": TOMTOM_API_KEY,
            "traffic": "false",
            "routeType": "fastest",
            "travelMode": "car",
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if (
        "routes" not in data
        or len(data["routes"]) == 0
    ):
        raise ValueError(
            "Tidak ada route yang ditemukan"
        )

    route = data["routes"][0]

    points = []

    for leg in route["legs"]:

        for point in leg["points"]:

            points.append(
                [
                    point["latitude"],
                    point["longitude"],
                ]
            )

    summary = route.get(
        "summary",
        {}
    )

    distance_km = (
        summary.get(
            "lengthInMeters",
            0
        )
        / 1000
    )

    travel_time_min = (
        summary.get(
            "travelTimeInSeconds",
            0
        )
        / 60
    )

    return (
        points,
        distance_km,
        travel_time_min
    )


# ============================================================
# CREATE MAP
# ============================================================

def create_map():

    corridors = load_top14()

    if not FIXED_EDGES_PATH.exists():

        raise FileNotFoundError(
            f"fixed_graph_edges.csv "
            f"tidak ditemukan:\n"
            f"{FIXED_EDGES_PATH}"
        )

    fixed_edges = pd.read_csv(
        FIXED_EDGES_PATH
    )

    print()
    print(
        "========================================="
    )
    print(
        "BUILDING SATELLITE FIXED GRAPH MAP"
    )
    print(
        "========================================="
    )

    print(
        f"Corridors : {len(corridors)}"
    )

    print(
        f"Edges     : {len(fixed_edges)}"
    )

    print()


    # ========================================================
    # MAP CENTER
    # ========================================================

    center_lat = (
        corridors["lat"].mean()
    )

    center_lon = (
        corridors["lon"].mean()
    )

    traffic_map = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=10,
        tiles=None,
        control_scale=True
    )


    # ========================================================
    # SATELLITE BASEMAP
    # ========================================================

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/"
            "{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Satellite",
        overlay=False,
        control=True,
        show=True
    ).add_to(
        traffic_map
    )


    # ========================================================
    # STREET MAP
    # ========================================================

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street Map",
        overlay=False,
        control=True,
        show=False
    ).add_to(
        traffic_map
    )


    # ========================================================
    # CORRIDOR LOOKUP
    # ========================================================

    corridor_lookup = (
        corridors
        .set_index("station_id")
        .to_dict("index")
    )


    # ========================================================
    # FIXED GRAPH LAYER
    # ========================================================

    fixed_layer = folium.FeatureGroup(
        name="Fixed Graph 3-NN",
        show=True
    )


    print(
        "Fetching road geometry..."
    )

    print()


    for index, edge in enumerate(
        fixed_edges.itertuples(),
        start=1
    ):

        source = str(
            edge.source
        )

        target = str(
            edge.target
        )


        if (
            source not in corridor_lookup
            or target not in corridor_lookup
        ):

            print(
                f"[SKIP] {source} -> {target}"
            )

            continue


        source_data = (
            corridor_lookup[source]
        )

        target_data = (
            corridor_lookup[target]
        )


        source_coord = [
            float(
                source_data["lat"]
            ),
            float(
                source_data["lon"]
            )
        ]

        target_coord = [
            float(
                target_data["lat"]
            ),
            float(
                target_data["lon"]
            )
        ]


        source_name = (
            source_data["name"]
        )

        target_name = (
            target_data["name"]
        )


        print(
            f"[{index}/{len(fixed_edges)}] "
            f"{source_name} -> "
            f"{target_name}"
        )


        # ====================================================
        # GET ROAD GEOMETRY
        # ====================================================

        try:

            (
                route_points,
                route_distance_km,
                route_time_min
            ) = get_route_geometry(
                source_coord[0],
                source_coord[1],
                target_coord[0],
                target_coord[1]
            )

            print(
                f"    OK | "
                f"{route_distance_km:.2f} km | "
                f"{route_time_min:.1f} min | "
                f"{len(route_points)} points"
            )


        except Exception as exc:

            print(
                f"    ROUTE FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                "    fallback -> straight line"
            )

            route_points = [
                source_coord,
                target_coord
            ]

            route_distance_km = None
            route_time_min = None


        # ====================================================
        # ORIGINAL HAVERSINE DISTANCE
        # ====================================================

        original_distance = getattr(
            edge,
            "distance_km",
            None
        )


        # ====================================================
        # POPUP
        # ====================================================

        popup_text = (
            "<b>Fixed Graph Edge</b>"
            "<br><br>"
            f"<b>Source:</b> "
            f"{source_name}"
            "<br>"
            f"<b>Target:</b> "
            f"{target_name}"
        )


        if original_distance is not None:

            try:

                popup_text += (
                    "<br>"
                    f"<b>Haversine:</b> "
                    f"{float(original_distance):.2f} km"
                )

            except Exception:
                pass


        if route_distance_km is not None:

            popup_text += (
                "<br>"
                f"<b>Road distance:</b> "
                f"{route_distance_km:.2f} km"
            )


        if route_time_min is not None:

            popup_text += (
                "<br>"
                f"<b>Route time:</b> "
                f"{route_time_min:.1f} min"
            )


        popup_text += (
            "<br><br>"
            "<i>"
            "Fixed edge ditentukan oleh "
            "3-Nearest Neighbors berdasarkan "
            "Haversine distance. "
            "Bentuk garis mengikuti road geometry "
            "hanya untuk visualisasi."
            "</i>"
        )


        # ====================================================
        # DRAW ROUTE
        # ====================================================

        folium.PolyLine(
            locations=route_points,
            weight=4,
            opacity=0.85,
            popup=folium.Popup(
                popup_text,
                max_width=400
            ),
            tooltip=(
                f"{source_name} ↔ "
                f"{target_name}"
            )
        ).add_to(
            fixed_layer
        )


    fixed_layer.add_to(
        traffic_map
    )


    # ========================================================
    # NODE LAYER
    # ========================================================

    node_layer = folium.FeatureGroup(
        name="14 TomTom Corridors",
        show=True
    )


    for corridor in corridors.itertuples():

        popup_html = (
            f"<b>{corridor.name}</b>"
            "<br><br>"
            f"<b>Station ID:</b> "
            f"{corridor.station_id}"
            "<br>"
            f"<b>Latitude:</b> "
            f"{corridor.lat}"
            "<br>"
            f"<b>Longitude:</b> "
            f"{corridor.lon}"
        )


        folium.CircleMarker(
            location=[
                corridor.lat,
                corridor.lon
            ],
            radius=7,
            fill=True,
            fill_opacity=0.9,
            weight=2,
            tooltip=(
                corridor.name
            ),
            popup=folium.Popup(
                popup_html,
                max_width=300
            )
        ).add_to(
            node_layer
        )


    node_layer.add_to(
        traffic_map
    )


    # ========================================================
    # FIT MAP TO ALL CORRIDORS
    # ========================================================

    bounds = [
        [
            corridors["lat"].min(),
            corridors["lon"].min()
        ],
        [
            corridors["lat"].max(),
            corridors["lon"].max()
        ]
    ]

    traffic_map.fit_bounds(
        bounds,
        padding=[
            30,
            30
        ]
    )


    # ========================================================
    # LAYER CONTROL
    # ========================================================

    folium.LayerControl(
        collapsed=False
    ).add_to(
        traffic_map
    )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    traffic_map.save(
        OUTPUT_PATH
    )


    print()
    print(
        "========================================="
    )

    print(
        "SATELLITE GRAPH CREATED"
    )

    print(
        "========================================="
    )

    print(
        f"Nodes  : {len(corridors)}"
    )

    print(
        f"Edges  : {len(fixed_edges)}"
    )

    print(
        f"Output : {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_map()