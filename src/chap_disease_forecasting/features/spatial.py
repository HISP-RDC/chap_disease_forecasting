"""Neighbour graph between locations, built from the GeoJSON CHAP sends.

CHAP Core passes the dataset polygons to chapkit services at train and predict
time. Each location's ``k`` nearest neighbours (great-circle distance between
polygon centroids) feed the spatial features: outbreaks spread between nearby
health zones, which a per-location model cannot see. Nearest neighbours are
used rather than shared borders because health-zone polygons often have gaps,
overlaps or islands that make adjacency unreliable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_EARTH_RADIUS_KM = 6371.0088


def _ring_centroid(ring: Sequence[Sequence[float]]) -> tuple[float, float, float] | None:
    """Area-weighted centroid (lon, lat) and absolute area of one linear ring."""

    points = np.asarray([point[:2] for point in ring], dtype=float)
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    x, y = points[:, 0], points[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return float(x.mean()), float(y.mean()), 0.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return float(cx), float(cy), abs(float(area))


def geometry_centroid(geometry: Mapping[str, Any]) -> tuple[float, float] | None:
    """Centroid of a GeoJSON Point, Polygon or MultiPolygon (exterior rings)."""

    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not coordinates:
        return None
    if kind == "Point":
        return float(coordinates[0]), float(coordinates[1])
    if kind == "Polygon":
        polygons = [coordinates]
    elif kind == "MultiPolygon":
        polygons = list(coordinates)
    else:
        return None
    parts: list[tuple[float, float, float]] = []
    for polygon in polygons:
        part = _ring_centroid(polygon[0]) if polygon else None
        if part is not None:
            parts.append(part)
    if not parts:
        return None
    total = sum(area for _, _, area in parts)
    if total <= 0:
        return (
            float(np.mean([lon for lon, _, _ in parts])),
            float(np.mean([lat for _, lat, _ in parts])),
        )
    return (
        sum(lon * area for lon, _, area in parts) / total,
        sum(lat * area for _, lat, area in parts) / total,
    )


def _feature_location(feature: Mapping[str, Any]) -> str | None:
    properties = feature.get("properties") or {}
    for value in (feature.get("id"), properties.get("id"), properties.get("location")):
        if value is not None and str(value).strip():
            return str(value)
    return None


def location_centroids(geo: Any) -> dict[str, tuple[float, float]]:
    """Map location identifier to centroid for a FeatureCollection (dict or model)."""

    if geo is None:
        return {}
    if hasattr(geo, "model_dump"):
        geo = geo.model_dump(mode="json")
    features = geo.get("features") if isinstance(geo, Mapping) else None
    centroids: dict[str, tuple[float, float]] = {}
    for feature in features or []:
        if not isinstance(feature, Mapping):
            continue
        location = _feature_location(feature)
        geometry = feature.get("geometry")
        if location is None or not isinstance(geometry, Mapping):
            continue
        centroid = geometry_centroid(geometry)
        if centroid is not None and all(math.isfinite(value) for value in centroid):
            centroids[location] = centroid
    return centroids


def nearest_neighbours(
    centroids: Mapping[str, tuple[float, float]],
    k: int,
    locations: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Each location's ``k`` nearest other locations by great-circle distance."""

    names = [name for name in (locations or sorted(centroids)) if name in centroids]
    if k <= 0 or len(names) < 2:
        return {}
    k = min(k, len(names) - 1)
    lon = np.radians([centroids[name][0] for name in names])
    lat = np.radians([centroids[name][1] for name in names])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    cos_product = np.cos(lat[:, None]) * np.cos(lat[None, :])
    haversine = np.sin(dlat / 2.0) ** 2 + cos_product * np.sin(dlon / 2.0) ** 2
    distance = 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    np.fill_diagonal(distance, np.inf)
    order = np.argsort(distance, axis=1, kind="stable")[:, :k]
    return {name: [names[j] for j in order[i]] for i, name in enumerate(names)}


def neighbour_graph(geo: Any, k: int, locations: Sequence[str]) -> dict[str, list[str]]:
    """Neighbour lists for the given locations, or an empty graph without geometry."""

    return nearest_neighbours(location_centroids(geo), k, [str(name) for name in locations])
