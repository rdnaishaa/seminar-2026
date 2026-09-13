import folium


def build_route_map(
    corridors,
    origin,
    destination,
    route_points=None,
):

    # ========================================================
    # MAP CENTER
    # ========================================================

    center_lat = corridors["lat"].mean()
    center_lon = corridors["lon"].mean()

    traffic_map = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=10,
        tiles=None,
    )

    # ========================================================
    # SATELLITE
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
        show=False,
    ).add_to(
        traffic_map
    )

    # ========================================================
    # ALL FINAL CORRIDORS
    # ========================================================

    corridor_group = folium.FeatureGroup(
        name="Monitoring Corridors",
        show=True,
    )

    for corridor in corridors.itertuples():

        folium.CircleMarker(
            location=[
                corridor.lat,
                corridor.lon,
            ],
            radius=5,
            fill=True,
            fill_opacity=0.8,
            tooltip=corridor.display_name,
            popup=(
                f"<b>{corridor.display_name}</b>"
                "<br>"
                f"Station ID: {corridor.station_id}"
            ),
        ).add_to(
            corridor_group
        )

    corridor_group.add_to(
        traffic_map
    )

    # ========================================================
    # ORIGIN
    # ========================================================

    folium.Marker(
        location=[
            origin["lat"],
            origin["lon"],
        ],
        tooltip=(
            f"Asal: {origin['display_name']}"
        ),
        popup=(
            "<b>ASAL</b><br>"
            f"{origin['display_name']}"
        ),
    ).add_to(
        traffic_map
    )

    # ========================================================
    # DESTINATION
    # ========================================================

    folium.Marker(
        location=[
            destination["lat"],
            destination["lon"],
        ],
        tooltip=(
            f"Tujuan: {destination['display_name']}"
        ),
        popup=(
            "<b>TUJUAN</b><br>"
            f"{destination['display_name']}"
        ),
    ).add_to(
        traffic_map
    )

    # ========================================================
    # ROUTE
    # ========================================================

    if route_points:

        folium.PolyLine(
            locations=route_points,
            weight=6,
            opacity=0.9,
            tooltip=(
                f"{origin['display_name']} → "
                f"{destination['display_name']}"
            ),
        ).add_to(
            traffic_map
        )

        traffic_map.fit_bounds(
            route_points
        )

    # ========================================================
    # LAYER CONTROL
    # ========================================================

    folium.LayerControl().add_to(
        traffic_map
    )

    return traffic_map