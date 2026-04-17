# pylint: disable=too-many-arguments
"""Count tracks that pass a reference line for post-analysis."""

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tssa.engine.geometry import point_relative_to_polyline
from tssa.io.definition import track_fields


class Count:
    """Count tracks that pass a reference line."""

    def __init__(self, line: LineString) -> None:
        """Initialize a counter with a reference line.

        Args:
            line: Reference line used to detect crossing tracks.
        """
        self.line = line

    def count_tracks_by_line(
        self,
        tracks: pd.DataFrame = None,
        track_file: str = None,
        with_header: bool = False,
        video_index: int = None,
        video_tot: int = None,
    ) -> pd.DataFrame:
        """Count tracks that intersect the reference line.

        Args:
            tracks: Track data as a DataFrame. If ``None``, data is loaded from
                ``track_file``.
            track_file: Path to a track text/CSV file used when ``tracks`` is
                ``None``.
            with_header: Whether ``track_file`` already includes a header row.
            video_index: Optional current video index for progress display.
            video_tot: Optional total number of videos for progress display.

        Returns:
            DataFrame with columns ``count``, ``track_id``, ``frame``, and
            ``direction``.
        """
        # Load tracks
        if tracks is None:
            if with_header:
                tracks = pd.read_csv(track_file)
            else:
                tracks = pd.read_csv(track_file, header=None)
                if len(tracks.columns) != len(track_fields):
                    raise Exception(
                        "The number of data fields in tracks is bigger \
                                    or smaller than pre-defined!"
                    )
                tracks.columns = track_fields
        track_ids = tracks["track_id"].unique()

        # Create a GeoDataFrame of tracks
        geo = tracks.apply(
            lambda track: Polygon([
                (track["x"], track["y"]),
                (track["x"] + track["w"], track["y"]),
                (track["x"] + track["w"], track["y"] + track["h"]),
                (track["x"], track["y"] + track["h"]),
            ]),
            axis=1,
        )
        geo_tracks = gpd.GeoDataFrame(tracks, geometry=geo)

        # Check if bboxes intersect with the line
        geo_tracks["intersected"] = self.line.intersects(geo_tracks.geometry).values.tolist()

        # Interate through all tracks
        pbar = tqdm(total=len(track_ids), unit=" tracks")
        if video_index and video_tot:
            pbar.set_description_str(f"Counting {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Counting")
        intersected_tracks = []
        intersected_frames = []
        intersected_direct = []

        for track_id in track_ids:
            track_frames = geo_tracks.loc[(geo_tracks["track_id"] == track_id)].copy()
            if len(track_frames) > 0:
                track_intersected_frames = track_frames.loc[track_frames["intersected"]].copy()
                if len(track_intersected_frames) > 0:
                    track_frames.sort_values(by="frame", inplace=True)
                    if track_id not in intersected_tracks:
                        intersected_tracks.append(track_id)
                        intersected_frames.append(
                            track_intersected_frames["frame"].values[int(len(track_intersected_frames) / 2)]
                        )
                        # center point of the first frame
                        point_c_end = Point(
                            np.array((
                                track_frames.iloc[-1]["x"] + track_frames.iloc[-1]["w"] / 2,
                                track_frames.iloc[-1]["y"] + track_frames.iloc[-1]["h"] / 2,
                            ))
                        )
                        intersected_direct.append(point_relative_to_polyline(point_c_end, self.line))
            pbar.update()
        pbar.close()

        results = pd.DataFrame({
            "track_id": intersected_tracks,
            "frame": intersected_frames,
            "direction": intersected_direct,
        })
        results.sort_values(by="frame", inplace=True)
        results = results.reset_index(drop=True)
        results = results.index.to_frame(name="count").join(results)
        results["count"] = results["count"] + 1
        return results


if __name__ == "__main__":
    pass
