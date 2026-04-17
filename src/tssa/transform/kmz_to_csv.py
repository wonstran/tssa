#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert KMZ placemark point coordinates into a CSV table.

The converter reads a KMZ archive, extracts the first available KML document
(preferring ``doc.kml``), parses point placemarks, and writes rows containing
name, latitude, longitude, altitude, and description.
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
from pyproj import Transformer

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def _to_float(x):
    """Safely convert a value to ``float``.

    Args:
        x: Value to convert.

    Returns:
        Parsed float value, or ``None`` when conversion fails.
    """
    try:
        return float(x)
    except Exception:
        return None


def _to_utm(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 lat/lon to UTM coordinates.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        Tuple of (utm_x, utm_y) in meters for the appropriate UTM zone.
    """
    zone_number = int((lon + 180) / 6) + 1
    epsg_code = 32600 + zone_number if lat >= 0 else 32700 + zone_number

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)

    return utm_x, utm_y


def kmz_to_csv(kmz_path: str, csv_path: str, verbose: bool = True) -> pd.DataFrame:
    """Convert KMZ point placemarks to a CSV file.

    Reads KML content from a KMZ archive, extracts only ``Placemark`` entries
    containing ``Point`` coordinates, and writes them to a CSV with columns:
    ``name``, ``lat``, ``lon``, ``alt``, and ``description``.

    Args:
        kmz_path: Input path to a ``.kmz`` file.
        csv_path: Output path for the generated ``.csv`` file.
        verbose: If ``True``, print progress messages.
    Raises:
        FileNotFoundError: If ``kmz_path`` does not exist.
        RuntimeError: If no ``.kml`` file exists inside the KMZ archive.
        zipfile.BadZipFile: If the input is not a valid ZIP/KMZ archive.
        xml.etree.ElementTree.ParseError: If KML XML parsing fails.
    """
    kmz_path = Path(kmz_path)
    csv_path = Path(csv_path)

    if not kmz_path.exists():
        raise FileNotFoundError(kmz_path)

    # 1) read doc.kml from kmz(zip)
    with zipfile.ZipFile(kmz_path, "r") as zf:
        # Prefer doc.kml, otherwise first *.kml
        kml_name = None
        names = zf.namelist()
        if "doc.kml" in names:
            kml_name = "doc.kml"
        else:
            for n in names:
                if n.lower().endswith(".kml"):
                    kml_name = n
                    break
        if not kml_name:
            raise RuntimeError("No .kml found inside the KMZ.")

        kml_bytes = zf.read(kml_name)

    # 2) parse KML
    root = ET.fromstring(kml_bytes)

    rows = []
    # Placemark can contain Point/LineString/Polygon; we extract Point coordinates only
    for pm in root.findall(".//kml:Placemark", KML_NS):
        name = pm.findtext("kml:name", default="", namespaces=KML_NS).strip()
        desc = pm.findtext("kml:description", default="", namespaces=KML_NS).strip()

        # Find Point coordinates
        coord_text = pm.findtext(".//kml:Point/kml:coordinates", default="", namespaces=KML_NS).strip()
        if not coord_text:
            continue

        # coordinates format: "lon,lat,alt" (alt optional), may have spaces/newlines
        # Some KML may have multiple coords; Point usually single coord
        coord_text = re.sub(r"\s+", " ", coord_text).strip()
        first = coord_text.split(" ")[0]
        parts = first.split(",")

        lon = _to_float(parts[0]) if len(parts) >= 1 else None
        lat = _to_float(parts[1]) if len(parts) >= 2 else None
        alt = _to_float(parts[2]) if len(parts) >= 3 else 0.0  # default alt=0 if missing

        x, y = _to_utm(lat, lon)
        rows.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "x": x,
            "y": y,
            "description": desc,
        })

    # 3) write CSV
    df = pd.DataFrame(rows, columns=["name", "lat", "lon", "alt", "x", "y", "description"])
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"Saved {len(rows)} point(s) to: {csv_path}")

    return df


if __name__ == "__main__":
    kmz_file = "/mnt/e/videos/tmc/before/001/View_North/homography/north.kmz"
    csv_file = "/mnt/e/videos/tmc/before/001/View_North/homography/north.csv"
    kmz_to_csv(kmz_file, csv_file)
