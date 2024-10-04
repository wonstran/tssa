# pylint: disable=too-many-arguments
'''
This script is used to count the number of tracks that passing a reference line for post-analysis
last modified: 2021-09-30
'''
import os
import sys
from shapely.geometry import Polygon, LineString
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.spatial_calculation import point_relative_to_line

class Count:
    '''
    Count tracks that passing a reference line for post-analysis
    Methods:
        __init__(): initialize the class
        count_tracks_by_line(): 
            count tracks that passing a reference line for post-analysis
    '''
    def __init__(self, line:LineString) -> None:
        '''
        Initialize the class of Count
        Inputs:
            line - the reference line, LineString(pointA, pointB)
        '''
        self.track_fields = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'r1', 'r2', 'r3', 'r4']
        self.line = line
        self.line_end_a = np.array(self.line.coords[0])
        self.line_end_b = np.array(self.line.coords[1])

    def count_tracks_by_line(self,
                             tracks:pd.DataFrame=None,
                             track_file:str=None,
                             video_index:int=None,
                             video_tot:int=None) -> pd.DataFrame:
        '''
        Count tracks that passing a reference line for post-analysis
            Inputs:
                tracks - a DataFrame of tracks, if None (default), read track_file
                track_file -  a txt file contains tracks, 
                video_index - the index of video for processing
                video_tot - the total number of videos
            Return:
                A DataFrame of track_id, frame, direction, count
        '''
        # Load tracks
        if tracks is None:
            tracks = pd.read_csv(track_file, header=None)
        if len(tracks.columns) != len(self.track_fields):
            raise Exception("The number of data fields in tracks is bigger \
                            or smaller than pre-defined!")
        tracks.columns = self.track_fields
        track_ids = tracks['track_id'].unique()

        # Create a GeoDataFrame of tracks
        geo = tracks.apply(lambda track: Polygon([(track['x'], track['y']),
                                                (track['x'] + track['w'], track['y']),
                                                (track['x'] + track['w'], track['y'] + track['h']),
                                                (track['x'], track['y'] + track['h'])]),
                                                axis =1)
        geo_tracks = gpd.GeoDataFrame(tracks, geometry=geo)

        # Check if bboxes intersect with the line
        geo_tracks['intersected'] = self.line.intersects(geo_tracks.geometry).values.tolist()

        # Interate through all tracks
        pbar = tqdm(total=len(track_ids), unit=' tracks')
        if video_index and video_tot:
            pbar.set_description_str(f"Counting {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Counting")
        intersected_tracks = []
        intersected_frames = []
        intersected_direct = []

        for track_id in track_ids:
            track_frames = geo_tracks.loc[(geo_tracks['track_id']==track_id)].copy()
            if len(track_frames)>0:
                track_intersected_frames = track_frames.loc[(track_frames['intersected']==True)].copy()
                if len(track_intersected_frames) > 0:
                    track_frames.sort_values(by='frame', inplace=True)
                    if not track_id in intersected_tracks:
                        intersected_tracks.append(track_id)
                        intersected_frames.append(track_intersected_frames['frame'].values[
                                                        int(len(track_intersected_frames)/2)
                                                        ])
                        # center point of the first frame
                        point_c_end = np.array((track_frames.iloc[-1]['x'] + track_frames.iloc[-1]['w']/2,
                                            track_frames.iloc[-1]['y'] + track_frames.iloc[-1]['h']/2))
                        intersected_direct.append(
                            point_relative_to_line(point_c_end, self.line_end_a, self.line_end_b)
                            )
            pbar.update()
        pbar.close()

        results = pd.DataFrame(
            {
                'track_id': intersected_tracks,
                'frame': intersected_frames,
                'direction': intersected_direct,
            }
        )
        results.sort_values(by='frame', inplace=True)
        results = results.reset_index(drop=True)
        results = results.index.to_frame(name='count').join(results)
        return results

if __name__ == "__main__":
    pass
