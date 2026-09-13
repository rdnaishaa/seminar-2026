# TomTom Traffic Data Extraction — Guide

A minimal, reproducible pipeline for collecting road-traffic data from the
**TomTom Traffic API** with Python, illustrated with real data from a dense **96-corridor network across
Jabodetabek** (Jakarta + Bogor, Depok, Tangerang, Bekasi). Everything here works with a **free** TomTom developer account;
no key or secret is included in this repository — you bring your own.

```
TomTom_Extract/
├── README.md                  ← this guide
├── corridors.csv              ← 96 road points to monitor (edit for your city)
├── tomtom_extract.py          ← the collector (run on a schedule)
├── generate_eda.py            ← rebuilds EDA.md + figures/ from collected data
├── EDA.md                     ← sample exploratory analysis of the data
├── figures/                   ← plots used by EDA.md (incl. road map)
├── sample_data/
│   ├── sample_flow_response.json   ← one raw API response (what you get back)
│   ├── tomtom_sample.csv           ← 4 weeks of hourly data, 96 corridors (one file)
│   └── corridor_geometry.csv       ← road polylines (returned by the API)
└── full_data/                 ← complete collected history, one CSV per corridor
    ├── tt_sudirman.csv        ←   (96 files, 2026-07-23 → present; columns:
    ├── tt_thamrin.csv         ←    obs_time_utc, name, city, speeds, travel
    └── ...                    ←    times, congestion_ratio, closure, confidence)
```

## 1. Get a (free) API key

1. Create an account at **https://developer.tomtom.com** (free, no credit
   card).
2. In the dashboard: **My apps → Create app** — give it any name, and select
   the **Traffic API** product.
3. Copy the generated key.

The free tier allows **2,500 requests/day**; the 96-corridor list here at
hourly cadence uses 96 × 24 = 2,304 requests/day (92%) — trim the list or
slow the cadence before adding more.

Keep the key out of your code and repository. This project reads it from an
environment variable only:

```bash
export TOMTOM_API_KEY="paste-your-key-here"    # never commit this
```

## 2. What the API returns

We use the **Traffic Flow Segment Data** endpoint: give it one `lat,lon`
point, and it returns live conditions for the nearest road segment:

```
GET https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json
    ?point={lat},{lon}&unit=KMPH&key={YOUR_KEY}
```

Response fields (see `sample_data/sample_flow_response.json` for a real one):

| field | meaning |
|---|---|
| `currentSpeed` / `freeFlowSpeed` | live vs uncongested speed (km/h) |
| `currentTravelTime` / `freeFlowTravelTime` | segment travel time (s) |
| `confidence` | 0–1, probe-vehicle coverage quality |
| `roadClosure` | boolean |
| `frc` | functional road class (FRC0 motorway … FRC6 local) |
| `coordinates` | the road-segment polyline (usable for mapping!) |

**Important:** the API is **real-time only** — there is no free historical
endpoint. A time-series is built by polling on a schedule and accumulating.

## 3. Collect

```bash
pip install requests pandas
export TOMTOM_API_KEY="..."
python tomtom_extract.py
```

Each run appends one row per corridor to `collected/tomtom_flow.csv` and
(optionally) stores the raw JSON under `collected/raw/YYYY/MM/DD/` for
provenance. Schedule it hourly, e.g. with cron:

```cron
0 * * * * cd /path/to/TomTom_Extract && TOMTOM_API_KEY=$(cat ~/.tomtom_key) python3 tomtom_extract.py >> collect.log 2>&1
```

Derived metric added by the collector:
`congestion_ratio = 1 − currentSpeed / freeFlowSpeed`
(0 = free flow; 0.5 = traffic at half the free-flow speed).

### Adapting to your own city

Edit `corridors.csv` — one row per road point
(`station_id,name,city,lat,lon`).
Pick points *on* major roads (the API snaps to the nearest segment; verify
the returned polyline looks right). Practical tips learned collecting Jakarta:

- Query by **coordinates on the carriageway**, not intersections — snapping
  at junctions can flip between crossing roads.
- Space requests ~1.5 s apart (the script does) to stay well inside rate
  limits.
- The response has **no server timestamp** — stamp rows with your own UTC
  collection time (the script does).

## 4. Sample data

`sample_data/tomtom_sample.csv` holds ~14,000 real hourly observations
(96 Jabodetabek corridors, 4 weeks, collected with exactly this pipeline —
wide format, one row per corridor-hour), and `full_data/` has the complete
per-corridor history. Corridors were added in stages, so early dates cover
fewer of them. Use it to test analysis code before
your own collection accumulates.

## 5. Exploratory analysis

See **[EDA.md](EDA.md)** — corridor summary tables, the Jakarta twin-peak
diurnal congestion cycle, corridor ranking, an hour×corridor heatmap, and the
96-corridor network over a real basemap, drawn from the API's own segment
polylines:

![road map preview](figures/road_map.png)

Regenerate everything from the sample (or your own) data with:

```bash
pip install pandas numpy matplotlib pillow requests
python generate_eda.py
```

## 6. Licensing & attribution

- **Traffic data** © TomTom. Collected data is subject to the
  [TomTom developer terms](https://developer.tomtom.com/terms-and-conditions) —
  review them before redistributing datasets; the small sample here is
  included for illustration/education.
- **Basemap tiles** in the map figure: © OpenStreetMap contributors © CARTO
  (free tier, attribution required — keep the caption).
- Code in this folder: use freely (MIT-style; attribution appreciated).
