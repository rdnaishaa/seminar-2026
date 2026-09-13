import requests

from src.config import TOMTOM_API_KEY


def get_route_geometry(
    start_lat,
    start_lon,
    end_lat,
    end_lon,
    depart_at=None,
):

    if not TOMTOM_API_KEY:
        raise RuntimeError(
            "TOMTOM_API_KEY tidak ditemukan."
        )

    url = (
        "https://api.tomtom.com/routing/1/"
        f"calculateRoute/"
        f"{start_lat},{start_lon}:"
        f"{end_lat},{end_lon}/json"
    )

    params = {
        "key": TOMTOM_API_KEY,
        "traffic": "true",
        "routeType": "fastest",
        "travelMode": "car",
    }

    if depart_at is not None:
        params["departAt"] = depart_at

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("routes"):
        raise ValueError(
            "Route tidak ditemukan."
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

    summary = route["summary"]

    distance_km = (
        summary["lengthInMeters"]
        / 1000
    )

    travel_time_min = (
        summary["travelTimeInSeconds"]
        / 60
    )

    return {
        "points": points,
        "distance_km": distance_km,
        "travel_time_min": travel_time_min,
    }