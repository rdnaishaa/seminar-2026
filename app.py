import math

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.data_loader import (
    load_final64,
    load_history,
)

from src.routing import (
    get_route_geometry,
)

from src.map_builder import (
    build_route_map,
)

from src.historical_analysis import (
    get_latest_status,
    hourly_profile,
    get_best_worst_hour,
    get_corridor_summary,
)

from src.route_history import (
    get_route_hourly_profile,
    get_route_departure_summary,
)


WIB = ZoneInfo("Asia/Jakarta")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Historical Traffic Explorer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* -------- Fonts (Google's own: Roboto + Google Sans fallback) -------- */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&family=Google+Sans:wght@400;500;700&display=swap');

    :root {
        --gm-blue: #1a73e8;
        --gm-blue-dark: #1967d2;
        --gm-blue-tint: #e8f0fe;
        --gm-green: #188038;
        --gm-green-tint: #e6f4ea;
        --gm-yellow: #ea8600;
        --gm-yellow-tint: #fef7e0;
        --gm-red: #d93025;
        --gm-red-tint: #fce8e6;
        --gm-text: #202124;
        --gm-text-muted: #5f6368;
        --gm-border: #dadce0;
        --gm-surface: #ffffff;
        --gm-app-bg: #f1f3f4;
        --gm-radius-sm: 8px;
        --gm-radius-md: 12px;
        --gm-radius-lg: 24px;
        --gm-shadow-1: 0 1px 2px rgba(60,64,67,0.30), 0 1px 3px 1px rgba(60,64,67,0.15);
        --gm-shadow-2: 0 1px 3px rgba(60,64,67,0.30), 0 4px 8px 3px rgba(60,64,67,0.15);
    }

    html, body, [class*="css"] {
        font-family: 'Google Sans', 'Roboto', Arial, sans-serif;
        color: var(--gm-text) !important;
    }

    /* App background = Google Maps grey */
    [data-testid="stAppViewContainer"] {
        background-color: var(--gm-app-bg);
    }
    [data-testid="stHeader"] {
        background-color: transparent;
    }

    /* -------- Force text color everywhere (fixes invisible text when
       the underlying Streamlit theme is still set to dark) -------- */
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] h6,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input,
    div[data-testid="stDateInput"] input,
    div[data-testid="stTimeInput"] input,
    div[data-baseweb="popover"] * {
        color: var(--gm-text) !important;
    }

    /* Widget backgrounds should stay white/light regardless of theme */
    div[data-baseweb="select"] > div,
    div[data-baseweb="popover"] div[role="listbox"],
    div[data-testid="stDateInput"] input,
    div[data-testid="stTimeInput"] input {
        background-color: var(--gm-surface) !important;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    h1, h2, h3, h4 {
        font-family: 'Google Sans', 'Roboto', sans-serif;
        font-weight: 500;
        color: var(--gm-text);
        letter-spacing: 0;
    }

    p, span, div, label {
        font-family: 'Roboto', sans-serif;
    }

    hr {
        border-color: var(--gm-border) !important;
    }

    /* -------- Search-bar style hero (mimics the GMaps search box) -------- */
    .tc-hero {
        background: var(--gm-surface);
        border-radius: var(--gm-radius-lg);
        padding: 14px 24px;
        margin-bottom: 18px;
        box-shadow: var(--gm-shadow-1);
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .tc-hero-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--gm-blue-tint);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }
    .tc-hero h1 {
        margin: 0;
        font-size: 1.35rem;
        font-weight: 500;
        color: var(--gm-text);
    }
    .tc-hero p {
        margin: 2px 0 0 0;
        color: var(--gm-text-muted);
        font-size: 0.88rem;
        max-width: 640px;
    }

    /* -------- Metrics (place-info chips) -------- */
    div[data-testid="stMetric"] {
        background: var(--gm-surface);
        border: 1px solid var(--gm-border);
        border-radius: var(--gm-radius-md);
        padding: 14px 16px;
        box-shadow: var(--gm-shadow-1);
        transition: box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: var(--gm-shadow-2);
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--gm-text-muted) !important;
        letter-spacing: 0.01em;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        font-family: 'Google Sans', sans-serif;
        font-weight: 500;
        color: var(--gm-text) !important;
    }

    /* -------- Badges (Google Maps "Open" / "Busy" pill style) -------- */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    .badge-green, .badge-green * {
        background-color: var(--gm-green-tint) !important;
        color: var(--gm-green) !important;
    }
    .badge-yellow, .badge-yellow * {
        background-color: var(--gm-yellow-tint) !important;
        color: var(--gm-yellow) !important;
    }
    .badge-red, .badge-red * {
        background-color: var(--gm-red-tint) !important;
        color: var(--gm-red) !important;
    }

    /* -------- Location card (Google Maps "place card") -------- */
    .location-card {
        background: var(--gm-surface);
        border: 1px solid var(--gm-border);
        border-radius: var(--gm-radius-md);
        padding: 20px 22px;
        height: 100%;
        box-shadow: var(--gm-shadow-1);
    }
    .location-card h4 {
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1rem;
        font-weight: 500;
        color: var(--gm-text);
    }

    /* -------- Tabs (styled like GMaps top filter chips) -------- */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: none !important;
        background: var(--gm-surface);
        padding: 6px;
        border-radius: 999px;
        box-shadow: var(--gm-shadow-1);
        width: fit-content;
    }
    button[data-baseweb="tab"],
    button[data-baseweb="tab"] * {
        font-weight: 500;
        font-size: 0.88rem;
        border-radius: 999px !important;
        padding: 8px 16px !important;
        color: var(--gm-text-muted) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] * {
        background-color: var(--gm-blue-tint) !important;
        color: var(--gm-blue) !important;
    }
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }
    div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* -------- Sidebar (mimics the GMaps left search panel) -------- */
    section[data-testid="stSidebar"] {
        background-color: var(--gm-surface);
        border-right: 1px solid var(--gm-border);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 1rem;
        font-weight: 500;
        color: var(--gm-text);
    }

    /* Selects / inputs styled like GMaps rounded search fields */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    div[data-testid="stDateInput"] input,
    div[data-testid="stTimeInput"] input {
        border-radius: 999px !important;
        border: 1px solid var(--gm-border) !important;
        background-color: var(--gm-app-bg) !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
        border-color: var(--gm-blue) !important;
    }

    /* -------- Buttons styled like Google's blue "Directions" button -------- */
    .stButton > button, .stDownloadButton > button,
    .stButton > button *, .stDownloadButton > button * {
        background-color: var(--gm-blue) !important;
        color: #fff !important;
        border: none;
        border-radius: 999px;
        font-weight: 500;
        padding: 8px 20px;
        box-shadow: var(--gm-shadow-1);
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    .stButton > button:hover *, .stDownloadButton > button:hover * {
        background-color: var(--gm-blue-dark) !important;
        color: #fff !important;
    }

    /* -------- Progress bar (GMaps blue) -------- */
    div[data-testid="stProgress"] > div > div {
        background-color: var(--gm-blue);
        border-radius: 999px;
    }
    div[data-testid="stProgress"] > div {
        background-color: #e8eaed;
        border-radius: 999px;
    }

    /* -------- Alerts / info boxes -------- */
    div[data-testid="stAlert"] {
        border-radius: var(--gm-radius-md);
        border: 1px solid var(--gm-border);
        box-shadow: var(--gm-shadow-1);
    }

    /* -------- Captions -------- */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--gm-text-muted) !important;
        font-size: 0.82rem;
    }

    /* -------- Map container: rounded corners + shadow like an embedded GMaps frame -------- */
    iframe {
        border-radius: var(--gm-radius-md);
    }
    div[data-testid="stIFrame"] {
        border-radius: var(--gm-radius-md);
        overflow: hidden;
        box-shadow: var(--gm-shadow-2);
        border: 1px solid var(--gm-border);
    }

    /* -------- Expander / containers -------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: var(--gm-radius-md) !important;
    }

    /* -------- Subheaders get a small blue accent bar -------- */
    h3 {
        position: relative;
        padding-left: 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_corridors():
    return load_final64()


@st.cache_data(ttl=300)
def get_history():
    return load_history()


corridors = get_corridors()
history = get_history()


# ============================================================
# HELPERS
# ============================================================

def congestion_badge(ratio: float):
    """Return (emoji label, css class) based on congestion ratio."""
    if ratio < 0.20:
        return "🟢 Lancar", "badge-green"
    elif ratio < 0.40:
        return "🟡 Ramai Lancar", "badge-yellow"
    else:
        return "🔴 Padat", "badge-red"


def render_location_status(label, row):
    """Render a status card for a single monitoring location."""
    current_speed = float(row["current_speed"])
    free_flow_speed = float(row["free_flow_speed"])
    congestion_ratio = float(row["congestion_ratio"])

    badge_text, badge_class = congestion_badge(congestion_ratio)

    st.markdown(
        f"""
        <div class="location-card">
            <h4>📍 {label}</h4>
            <span class="badge {badge_class}">{badge_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.metric("Kecepatan saat ini", f"{current_speed:.0f} km/jam")

    with m_col2:
        st.metric("Kecepatan kondisi lancar", f"{free_flow_speed:.0f} km/jam")

    progress_value = max(0.0, min(1.0, current_speed / free_flow_speed if free_flow_speed else 0))
    st.progress(progress_value)

    st.caption(
        f"Kecepatan saat ini sekitar **{congestion_ratio * 100:.1f}% lebih rendah** "
        "dibanding kondisi arus lalu lintas yang lebih bebas."
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two coordinates, in kilometers."""
    radius = 6371.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


@st.cache_data
def build_corridor_graph(corridors_df, k=3):
    """
    Build the corridor graph the same way as the Fixed ST-GNN graph:
    each node (corridor) is connected to its k geographically nearest
    neighbors, using Haversine distance.
    """
    nodes = corridors_df[["station_id", "display_name", "lat", "lon"]].copy()
    nodes["lat"] = nodes["lat"].astype(float)
    nodes["lon"] = nodes["lon"].astype(float)

    edges = []

    for _, row_a in nodes.iterrows():

        distances = []

        for _, row_b in nodes.iterrows():
            if row_a["station_id"] == row_b["station_id"]:
                continue

            d = haversine_km(
                row_a["lat"], row_a["lon"],
                row_b["lat"], row_b["lon"],
            )
            distances.append((d, row_b["station_id"]))

        distances.sort(key=lambda item: item[0])

        for d, neighbor_id in distances[:k]:
            edges.append(
                {
                    "a_id": min(row_a["station_id"], neighbor_id),
                    "b_id": max(row_a["station_id"], neighbor_id),
                    "distance_km": d,
                }
            )

    edges_df = pd.DataFrame(edges).drop_duplicates(subset=["a_id", "b_id"])
    return edges_df.reset_index(drop=True)


def build_corridor_graph_map(corridors_df, edges_df):
    """Render the corridor graph (nodes + k-NN edges) on a real map."""
    nodes = corridors_df[["station_id", "display_name", "lat", "lon"]].copy()
    nodes["lat"] = nodes["lat"].astype(float)
    nodes["lon"] = nodes["lon"].astype(float)

    center_lat = nodes["lat"].mean()
    center_lon = nodes["lon"].mean()

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="cartodbpositron",
    )

    coord_lookup = {
        row["station_id"]: (row["lat"], row["lon"], row["display_name"])
        for _, row in nodes.iterrows()
    }

    for _, edge in edges_df.iterrows():
        a = coord_lookup.get(edge["a_id"])
        b = coord_lookup.get(edge["b_id"])

        if a is None or b is None:
            continue

        folium.PolyLine(
            locations=[(a[0], a[1]), (b[0], b[1])],
            color="#1a73e8",
            weight=2,
            opacity=0.55,
            tooltip=f"{a[2]} ↔ {b[2]} · {edge['distance_km']:.1f} km",
        ).add_to(fmap)

    for station_id, (lat, lon, label) in coord_lookup.items():
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color="#1967d2",
            fill=True,
            fill_color="#4285f4",
            fill_opacity=0.9,
            tooltip=label,
        ).add_to(fmap)

    return fmap


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="tc-hero">
        <div class="tc-hero-icon">🚦</div>
        <div>
            <h1>Historical Traffic Explorer</h1>
            <p>
                Lihat rute perjalanan, kondisi lalu lintas pada lokasi yang dipantau,
                dan pola kecepatan berdasarkan data historis TomTom.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — TRIP PLANNER
# ============================================================

with st.sidebar:
    st.header("🧭 Rencana Perjalanan")

    origin_label = st.selectbox(
        "📍 Lokasi awal",
        corridors["display_name"].tolist(),
        index=4,
    )

    destination_label = st.selectbox(
        "🏁 Tujuan",
        corridors["display_name"].tolist(),
        index=0,
    )

    st.divider()

    st.caption("🕒 Waktu keberangkatan")

    now_wib = datetime.now(WIB)

    selected_date = st.date_input(
        "Tanggal",
        value=now_wib.date(),
        min_value=now_wib.date(),
        max_value=now_wib.date() + timedelta(days=7),
    )

    selected_time = st.time_input(
        "Jam",
        value=now_wib.time().replace(microsecond=0),
    )

    st.divider()

    st.caption("🚗 Mode perjalanan: **Mobil**")

    if origin_label == destination_label:
        st.warning("Lokasi awal dan tujuan masih sama. Pilih tujuan yang berbeda.")


# Build depart_at (WIB) from the sidebar date + time pickers
selected_datetime = datetime.combine(
    selected_date,
    selected_time,
)

selected_datetime = selected_datetime.replace(
    tzinfo=WIB
)

depart_at = selected_datetime.isoformat()


origin = corridors[
    corridors["display_name"] == origin_label
].iloc[0]

destination = corridors[
    corridors["display_name"] == destination_label
].iloc[0]


# ============================================================
# GET ROUTE
# ============================================================

route_points = None
distance_km = None
travel_time_min = None
route_error = None

if origin_label != destination_label:

    with st.spinner("Menghitung rute perjalanan..."):
        try:
            try:
                route = get_route_geometry(
                    float(origin["lat"]),
                    float(origin["lon"]),
                    float(destination["lat"]),
                    float(destination["lon"]),
                    depart_at=depart_at,
                )
            except TypeError:
                # Fallback for routing modules that don't support
                # depart_at yet.
                route = get_route_geometry(
                    float(origin["lat"]),
                    float(origin["lon"]),
                    float(destination["lat"]),
                    float(destination["lon"]),
                )

            route_points = route["points"]
            distance_km = route["distance_km"]
            travel_time_min = route["travel_time_min"]

        except Exception as exc:
            route_error = str(exc)


# ============================================================
# TABS
# ============================================================

tab_overview, tab_graph, tab_status, tab_history, tab_route_history, tab_about = st.tabs(
    [
        "🗺️ Rute & Peta",
        "🕸️ Graph Koridor",
        "📡 Kondisi Real-time",
        "📈 Pola Historis",
        "🔍 Riwayat Rute",
        "ℹ️ Tentang",
    ]
)


# ------------------------------------------------------------
# TAB 1 — ROUTE + MAP
# ------------------------------------------------------------

with tab_overview:

    if route_error:
        st.error(f"Rute belum bisa ditampilkan: {route_error}")

    if distance_km is not None:

        st.subheader("Ringkasan Perjalanan")

        route_col1, route_col2, route_col3 = st.columns(3)

        with route_col1:
            st.metric("📏 Jarak perjalanan", f"{distance_km:.1f} km")

        with route_col2:
            st.metric("⏱️ Perkiraan waktu tempuh", f"{travel_time_min:.0f} menit")

        with route_col3:
            st.metric(
                "🕒 Berangkat",
                selected_datetime.strftime("%d %b, %H:%M"),
                help="Waktu keberangkatan yang dipilih di sidebar (WIB).",
            )

        st.caption(
            "Perkiraan waktu di atas berasal dari rute kendaraan mobil "
            "yang diberikan oleh layanan routing TomTom."
        )

        st.write("")

    st.subheader("Peta Perjalanan")

    traffic_map = build_route_map(
        corridors,
        origin,
        destination,
        route_points,
    )

    st_folium(
        traffic_map,
        width=None,
        height=600,
    )


# ------------------------------------------------------------
# TAB 1b — CORRIDOR GRAPH (FIXED ST-GNN STYLE)
# ------------------------------------------------------------

with tab_graph:

    st.subheader("🕸️ Graph 64 Koridor")

    st.write(
        "Setiap koridor direpresentasikan sebagai node, dengan edge yang "
        "dibentuk dari kedekatan geografis (**Haversine Distance**) ke "
        "**k tetangga terdekat** — pendekatan yang sama seperti Fixed Graph "
        "pada model ST-GNN, ditampilkan di atas peta asli berdasarkan "
        "koordinat sebenarnya."
    )

    k_neighbors = st.slider(
        "Jumlah tetangga terdekat (k)",
        min_value=1,
        max_value=6,
        value=4,
        help="k=4 merupakan konfigurasi final Fixed ST-GNN pada penelitian.",
    )

    edges_df = build_corridor_graph(corridors, k=k_neighbors)
    graph_map = build_corridor_graph_map(corridors, edges_df)

    st_folium(
        graph_map,
        width=None,
        height=600,
    )

    st.caption(
        f"🔗 {len(edges_df)} edge unik dari {len(corridors)} koridor pada k={k_neighbors}. "
        "Garis biru menunjukkan hubungan geografis antar-koridor; semakin besar k, "
        "semakin padat graph yang terbentuk. Ini adalah visualisasi struktur graph, "
        "bukan rute jalan sebenarnya."
    )


# ------------------------------------------------------------
# TAB 2 — REAL-TIME STATUS
# ------------------------------------------------------------

with tab_status:

    st.subheader("Kondisi Lalu Lintas di Lokasi Pemantauan")

    if history.empty:
        st.warning("Data lalu lintas belum tersedia.")

    else:

        latest_status = latest = get_latest_status(history)
        latest_timestamp = latest_status["obs_time_utc"].max()

        if latest_timestamp is not None:
            latest_timestamp_wib = latest_timestamp.tz_convert(WIB)
            st.caption(
                "🔄 Data terakhir yang tersedia: "
                f"{latest_timestamp_wib.strftime('%d %B %Y, %H:%M')} WIB"
            )

        origin_latest = latest_status[
            latest_status["station_id"] == origin["station_id"]
        ]

        destination_latest = latest_status[
            latest_status["station_id"] == destination["station_id"]
        ]

        st.write("")

        status_col1, status_col2 = st.columns(2)

        with status_col1:
            if not origin_latest.empty:
                render_location_status(origin_label, origin_latest.iloc[0])
            else:
                st.info(f"📍 {origin_label}: belum ada data untuk lokasi ini.")

        with status_col2:
            if not destination_latest.empty:
                render_location_status(destination_label, destination_latest.iloc[0])
            else:
                st.info(f"📍 {destination_label}: belum ada data untuk lokasi ini.")


# ------------------------------------------------------------
# TAB 3 — HISTORICAL PATTERN
# ------------------------------------------------------------

with tab_history:

    st.subheader("Pola Lalu Lintas Berdasarkan Data Historis")

    if history.empty:
        st.info("Belum ada data historis yang dapat dianalisis.")

    else:

        profile = hourly_profile(history, origin["station_id"])
        total_timestamps = history["obs_time_utc"].nunique()

        if total_timestamps < 3:
            st.info(
                "Data historis masih terlalu sedikit untuk "
                "menyimpulkan pola lalu lintas berdasarkan jam."
            )

        else:

            st.write(
                f"Bagian ini menunjukkan pola kecepatan pada **{origin_label}** "
                "berdasarkan data historis yang tersedia."
            )

            if not profile.empty:

                st.markdown("#### Rata-rata Kecepatan Berdasarkan Jam")

                chart_data = (
                    profile[["hour_wib", "avg_speed"]]
                    .rename(columns={"avg_speed": "Rata-rata kecepatan (km/jam)"})
                    .set_index("hour_wib")
                )

                st.line_chart(chart_data)

                st.caption(
                    "Sumbu waktu menggunakan WIB. "
                    "Grafik dihitung dari observasi historis yang tersedia, "
                    "bukan berarti data tersedia terus-menerus tanpa jeda."
                )

            result = get_best_worst_hour(history, origin["station_id"])

            if result:

                st.write("")

                best_col, worst_col = st.columns(2)

                with best_col:
                    st.metric(
                        "🟢 Waktu yang biasanya lebih lancar",
                        f"{result['best_hour']:02d}:00 WIB",
                        f"Rata-rata {result['best_speed']:.1f} km/jam",
                    )

                with worst_col:
                    st.metric(
                        "🔴 Waktu yang biasanya lebih padat",
                        f"{result['worst_hour']:02d}:00 WIB",
                        f"Rata-rata {result['worst_speed']:.1f} km/jam",
                    )

                st.caption(
                    "Jam di atas merupakan hasil ringkasan dari data historis "
                    "pada lokasi yang dipilih. Informasi ini tidak mewakili "
                    "seluruh kondisi jalan di sepanjang rute perjalanan."
                )


# ------------------------------------------------------------
# TAB — ROUTE HISTORY
# ------------------------------------------------------------

with tab_route_history:

    st.subheader("🔍 Riwayat Rute")

    st.write(
        f"Ringkasan historis untuk **{origin_label} → {destination_label}** "
        f"berdasarkan jam keberangkatan yang dipilih."
    )

    if history.empty:

        st.info(
            "Belum ada data historis yang dapat dianalisis."
        )

    else:

        departure_hour = (
            selected_datetime.hour
        )

        summary = (
            get_route_departure_summary(
                history,
                origin["station_id"],
                destination["station_id"],
                departure_hour,
            )
        )

        st.markdown(
            f"#### Kondisi historis sekitar pukul {departure_hour:02d}:00 WIB"
        )

        if summary.empty:

            st.info(
                "Belum ada data historis yang tersedia "
                "untuk jam keberangkatan tersebut."
            )

        else:

            col1, col2 = st.columns(2)

            origin_summary = summary[
                summary["station_id"]
                == origin["station_id"]
            ]

            destination_summary = summary[
                summary["station_id"]
                == destination["station_id"]
            ]

            with col1:

                st.markdown(
                    f"##### 📍 {origin_label}"
                )

                if not origin_summary.empty:

                    row = (
                        origin_summary.iloc[0]
                    )

                    st.metric(
                        "Rata-rata kecepatan",
                        f"{row['avg_speed']:.1f} km/jam",
                    )

                    st.metric(
                        "Rata-rata kondisi lancar",
                        f"{row['avg_free_flow_speed']:.1f} km/jam",
                    )

                    st.metric(
                        "Rata-rata congestion ratio",
                        f"{row['avg_congestion_ratio'] * 100:.1f}%",
                    )

                else:

                    st.info(
                        "Data tidak tersedia."
                    )

            with col2:

                st.markdown(
                    f"##### 🏁 {destination_label}"
                )

                if not destination_summary.empty:

                    row = (
                        destination_summary.iloc[0]
                    )

                    st.metric(
                        "Rata-rata kecepatan",
                        f"{row['avg_speed']:.1f} km/jam",
                    )

                    st.metric(
                        "Rata-rata kondisi lancar",
                        f"{row['avg_free_flow_speed']:.1f} km/jam",
                    )

                    st.metric(
                        "Rata-rata congestion ratio",
                        f"{row['avg_congestion_ratio'] * 100:.1f}%",
                    )

                else:

                    st.info(
                        "Data tidak tersedia."
                    )

        st.write("")

        st.markdown(
            "#### Perbandingan Pola Historis per Jam"
        )

        profile = (
            get_route_hourly_profile(
                history,
                origin["station_id"],
                destination["station_id"],
            )
        )

        if profile.empty:

            st.info(
                "Data historis belum cukup untuk membuat grafik."
            )

        else:

            chart_data = profile[
                [
                    "station_id",
                    "hour_wib",
                    "avg_speed",
                ]
            ].copy()

            name_map = {
                origin["station_id"]: origin_label,
                destination["station_id"]: destination_label,
            }

            chart_data["Lokasi"] = (
                chart_data["station_id"]
                .map(name_map)
            )

            pivot = (
                chart_data
                .pivot(
                    index="hour_wib",
                    columns="Lokasi",
                    values="avg_speed",
                )
                .sort_index()
            )

            st.line_chart(
                pivot
            )

            st.caption(
                "Grafik membandingkan pola kecepatan historis "
                "pada lokasi asal dan tujuan berdasarkan jam WIB. "
                "Bagian ini tidak merepresentasikan seluruh ruas jalan "
                "di sepanjang rute."
            )

with tab_about:

    st.subheader("Tentang Informasi di Halaman Ini")

    st.write(
        "Data lalu lintas berasal dari lokasi pemantauan pada 64 koridor "
        "yang digunakan dalam penelitian. Rute perjalanan ditampilkan "
        "menggunakan mode kendaraan mobil."
    )

    st.write(
        "Pola historis digunakan untuk melihat kecenderungan kondisi lalu lintas "
        "pada lokasi yang dipilih. Karena pemantauan hanya dilakukan pada titik "
        "atau koridor tertentu, informasi ini tidak menunjukkan kondisi setiap "
        "ruas jalan yang dilewati sepanjang perjalanan."
    )