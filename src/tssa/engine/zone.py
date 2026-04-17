"""
This script scans the track position and determine if the track belongs to a pre-define zone
last modified: 2021-09-30
created by: wonstran
"""

import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tssa.io.definition import track_fields


def match_zone_by_point(
    tracks: pd.DataFrame,
    zone: Polygon,
    ref_point: str = "cc",
    ref_offset: tuple = (0, 0),
    zone_field: str = "in_zone",
    output_ref_xy: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan the position of a point of tracks and determine if the track belongs to a pre-define zone\n
    Inputs:\n
        tracks - a DataFrame of tracks
        zone - a polygon of the zone
        ref_point - the reference point of the track, default is 'cc'\n
                    'cc' - center of the bounding box
                    'tc' - top center of the bounding box
                    'bc' - bottom center of the bounding box
                    'cl' - center left of the bounding box
                    'cr' - center right of the bounding box
                    'tl' - top left of the bounding box
                    'tr' - top right of the bounding box
                    'bl' - bottom left of the bounding box
                    'br' - bottom right of the bounding box
        ref_offset - the offset of the reference point, default is (0,0)
        zone_field - the field name of the zone, default is 'zone'
        output_ref_xy - if True, output the reference point x and y
        verbose - if True, show the progress bar
    Returns:
        A DataFrame of tracks with the zone field
    """

    if tracks.columns.tolist() == list(range(tracks.shape[1])):
        tracks.columns = track_fields

    if verbose:
        tqdm.pandas(desc="Generating geometries for reference points")
        if ref_point == "cc":
            geo = tracks.progress_apply(
                lambda track: Point([
                    (track["x"] + track["w"] / 2 + ref_offset[0]),
                    (track["y"] + track["h"] / 2 + ref_offset[1]),
                ]),
                axis=1,
            )
        elif ref_point == "tc":
            geo = tracks.progress_apply(
                lambda track: Point([(track["x"] + track["w"] / 2 + ref_offset[0]), (track["y"] + ref_offset[1])]),
                axis=1,
            )
        elif ref_point == "bc":
            geo = tracks.progress_apply(
                lambda track: Point([
                    [(track["x"] + track["w"] / 2 + ref_offset[0]), (track["y"] + track["h"] + ref_offset[1])]
                ]),
                axis=1,
            )
        elif ref_point == "cl":
            geo = tracks.progress_apply(
                lambda track: Point([[track["x"] + ref_offset[0], (track["y"] + track["h"] / 2 + ref_offset[1])]]),
                axis=1,
            )
        elif ref_point == "cr":
            geo = tracks.progress_apply(
                lambda track: Point([
                    (track["x"] + track["w"] + ref_offset[0]),
                    (track["y"] + track["h"] / 2 + ref_offset[1]),
                ]),
                axis=1,
            )
        elif ref_point == "tl":
            geo = tracks.progress_apply(
                lambda track: Point([track["x"] + ref_offset[0], track["y"] + ref_offset[1]]), axis=1
            )
        elif ref_point == "tr":
            geo = tracks.progress_apply(
                lambda track: Point([(track["x"] + track["w"] + ref_offset[0]), track["y"] + ref_offset[1]]), axis=1
            )
        elif ref_point == "bl":
            geo = tracks.progress_apply(
                lambda track: Point([track["x"] + ref_offset[0], (track["y"] + tracks["h"] + ref_offset[1])]), axis=1
            )
        elif ref_point == "br":
            geo = tracks.progress_apply(
                lambda track: Point([
                    (track["x"] + track["w"] + ref_offset[0]),
                    (track["y"] + track["h"] + ref_offset[1]),
                ]),
                axis=1,
            )
        else:
            geo = tracks.progress_apply(
                lambda track: Point([
                    (track["x"] + track["w"] / 2 + ref_offset[0]),
                    (track["y"] + track["h"] + ref_offset[1]),
                ]),
                axis=1,
            )
    else:
        if ref_point == "cc":
            geo = tracks.apply(
                lambda track: Point([
                    (track["x"] + track["w"] / 2 + ref_offset[0]),
                    (track["y"] + track["h"] / 2 + ref_offset[1]),
                ]),
                axis=1,
            )
        elif ref_point == "tc":
            geo = tracks.apply(
                lambda track: Point([(track["x"] + track["w"] / 2 + ref_offset[0]), (track["y"] + ref_offset[1])]),
                axis=1,
            )
        elif ref_point == "bc":
            geo = tracks.apply(
                lambda track: Point([
                    [(track["x"] + track["w"] / 2 + ref_offset[0]), (track["y"] + track["h"] + ref_offset[1])]
                ]),
                axis=1,
            )
        elif ref_point == "cl":
            geo = tracks.apply(
                lambda track: Point([[track["x"] + ref_offset[0], (track["y"] + track["h"] / 2 + ref_offset[1])]]),
                axis=1,
            )
        elif ref_point == "cr":
            geo = tracks.apply(
                lambda track: Point([
                    (track["x"] + track["w"] + ref_offset[0]),
                    (track["y"] + track["h"] / 2 + ref_offset[1]),
                ]),
                axis=1,
            )
        elif ref_point == "tl":
            geo = tracks.apply(lambda track: Point([track["x"] + ref_offset[0], track["y"] + ref_offset[1]]), axis=1)
        elif ref_point == "tr":
            geo = tracks.apply(
                lambda track: Point([track["x"] + track["w"] + ref_offset[0], track["y"] + ref_offset[1]]), axis=1
            )
        elif ref_point == "bl":
            geo = tracks.apply(
                lambda track: Point([track["x"] + ref_offset[0], (track["y"] + tracks["h"] + ref_offset[1])]), axis=1
            )
        elif ref_point == "br":
            geo = tracks.apply(
                lambda track: Point([
                    track["x"] + track["w"] + ref_offset[0],
                    (track["y"] + track["h"] + ref_offset[1]),
                ]),
                axis=1,
            )
        else:
            geo = tracks.apply(
                lambda track: Point([
                    track["x"] + track["w"] / 2 + ref_offset[0],
                    (track["y"] + track["h"] + ref_offset[1]),
                ]),
                axis=1,
            )

    geo_tracks = gpd.GeoDataFrame(tracks, geometry=geo)
    geo_tracks[zone_field] = False
    geo_tracks.loc[(geo_tracks.geometry.within(zone)), zone_field] = True
    if output_ref_xy:
        geo_tracks[["ref_x", "ref_y"]] = geo_tracks.geometry.apply(lambda p: [int(p.x), int(p.y)]).apply(pd.Series)
    return pd.DataFrame(geo_tracks.drop(columns="geometry"))


def match_zones_by_point(
    tracks: pd.DataFrame,
    zones: list[Polygon],
    ref_point: str = "cc",
    ref_offset: tuple = (0, 0),
    zone_field: str = "zone",
    output_ref_xy: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan the position of a point of tracks and determine if the track belongs to pre-define zones\n
    Inputs:\n
        tracks - a DataFrame of tracks
        zones - a list of polygons
        ref_point - the reference point of the track, default is 'cc'\n
                    'cc' - center of the bounding box
                    'tc' - top center of the bounding box
                    'bc' - bottom center of the bounding box
                    'cl' - center left of the bounding box
                    'cr' - center right of the bounding box
                    'tl' - top left of the bounding box
                    'tr' - top right of the bounding box
                    'bl' - bottom left of the bounding box
                    'br' - bottom right of the bounding box
        ref_offset - the offset of the reference point, default is (0,0)
        zone_field - the field name of the zone, default is 'zone'
        output_ref_xy - if True, output the reference point x and y
        verbose - if True, show the progress bar
    Returns:\n
        A DataFrame of tracks with the zone field
    """

    if tracks.columns.tolist() == list(range(tracks.shape[1])):
        tracks.columns = track_fields
    tracks[zone_field] = -1

    if verbose:
        tqdm.pandas(desc="Generating geometries for reference points")
        if ref_point == "cc":
            geo = tracks.progress_apply(
                lambda track: Point([
                    (track["x"] + track["w"] / 2 + ref_offset[0]),
                    (track["y"] + track["h"] / 2 + ref_offset[1]),
                ]),
                axis=1,
            )
        elif ref_point == "tc":
            geo = tracks.progress_apply(
                lambda track: Point([(track["x"] + track["w"] / 2 + ref_offset[0]), (track["y"] + ref_offset[1])]),
                axis=1,
            )
        elif ref_point == "bc":
            geo = tracks.progress_apply(
                lambda track: Point([
                    [(track["x"] + track["w"] / 2 + ref_offset[0]), (track["y"] + track["h"] + ref_offset[1])]
                ]),
                axis=1,
            )
        elif ref_point == "cl":
            geo = tracks.progress_apply(
                lambda track: Point([[track["x"] + ref_offset[0], (track["y"] + track["h"] / 2 + ref_offset[1])]]),
                axis=1,
            )
        elif ref_point == "cr":
            geo = tracks.progress_apply(
                lambda track: Point([
                    (track["x"] + track["w"] + ref_offset[0]),
                    (track["y"] + track["h"] / 2 + ref_offset[1]),
                ]),
                axis=1,
            )
        elif ref_point == "tl":
            geo = tracks.progress_apply(
                lambda track: Point([track["x"] + ref_offset[0], track["y"] + ref_offset[1]]), axis=1
            )
        elif ref_point == "tr":
            geo = tracks.progress_apply(
                lambda track: Point([(track["x"] + track["w"] + ref_offset[0]), (track["y"] + ref_offset[1])]), axis=1
            )
        elif ref_point == "bl":
            geo = tracks.progress_apply(
                lambda track: Point([track["x"] + ref_offset[0], (track["y"] + tracks["h"] + ref_offset[1])]), axis=1
            )
        elif ref_point == "br":
            geo = tracks.progress_apply(
                lambda track: Point([
                    (track["x"] + track["w"] + ref_offset[0]),
                    (track["y"] + track["h"] + ref_offset[1]),
                ]),
                axis=1,
            )
        else:
            geo = tracks.progress_apply(
                lambda track: Point([
                    (track["x"] + track["w"] / 2 + ref_offset[0]),
                    (track["y"] + track["h"] + ref_offset[1]),
                ]),
                axis=1,
            )
    else:
        if ref_point == "cc":
            geo = tracks.apply(
                lambda track: Point([
                    (track["x"] + track["w"] / 2 + ref_offset[0]),
                    (track["y"] + track["h"] / 2 + ref_offset[1]),
                ]),
                axis=1,
            )
        elif ref_point == "tc":
            geo = tracks.apply(
                lambda track: Point([(track["x"] + track["w"] / 2 + ref_offset[0]), (track["y"] + ref_offset[1])]),
                axis=1,
            )
        elif ref_point == "bc":
            geo = tracks.apply(
                lambda track: Point([
                    [(track["x"] + track["w"] / 2 + ref_offset[0]), (track["y"] + track["h"] + ref_offset[1])]
                ]),
                axis=1,
            )
        elif ref_point == "cl":
            geo = tracks.apply(
                lambda track: Point([[track["x"] + ref_offset[0], (track["y"] + track["h"] / 2 + ref_offset[1])]]),
                axis=1,
            )
        elif ref_point == "cr":
            geo = tracks.apply(
                lambda track: Point([
                    (track["x"] + track["w"] + ref_offset[0]),
                    (track["y"] + track["h"] / 2 + ref_offset[1]),
                ]),
                axis=1,
            )
        elif ref_point == "tl":
            geo = tracks.apply(lambda track: Point([track["x"] + ref_offset[0], track["y"] + ref_offset[1]]), axis=1)
        elif ref_point == "tr":
            geo = tracks.apply(
                lambda track: Point([track["x"] + track["w"] + ref_offset[0], track["y"] + ref_offset[1]]), axis=1
            )
        elif ref_point == "bl":
            geo = tracks.apply(
                lambda track: Point([track["x"] + ref_offset[0], (track["y"] + tracks["h"] + ref_offset[1])]), axis=1
            )
        elif ref_point == "br":
            geo = tracks.apply(
                lambda track: Point([
                    track["x"] + track["w"] + ref_offset[0],
                    (track["y"] + track["h"] + ref_offset[1]),
                ]),
                axis=1,
            )
        else:
            geo = tracks.apply(
                lambda track: Point([
                    track["x"] + track["w"] / 2 + ref_offset[0],
                    (track["y"] + track["h"] + ref_offset[1]),
                ]),
                axis=1,
            )

    geo_tracks = gpd.GeoDataFrame(tracks, geometry=geo)
    if verbose:
        pbar = tqdm(total=len(zones), unit=" zone", desc="Matching zones ")
    for i in range(len(zones)):
        geo_tracks.loc[(geo_tracks.geometry.within(zones[i])), zone_field] = i
        if verbose:
            pbar.update()
    if verbose:
        pbar.close()
    if output_ref_xy:
        geo_tracks[["ref_x", "ref_y"]] = geo_tracks.geometry.apply(lambda p: [p.x, p.y]).apply(pd.Series)

    return pd.DataFrame(geo_tracks.drop(columns="geometry"))
