"""
This module contains the Speed class, which is used to calculate the speed of an object.
Updates:
    2024-10-13
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union
from shapely.geometry import LineString, Polygon
from dnt.track import Tracker
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tssa.engine.line import match_lines_by_point, \
                             match_lines_by_bbox
from tssa.engine.zone import match_zone_by_point
from tssa.transform.homography import HomographyTransform

FT_PER_M = 3.28084

class SpeedMeterbyLine:
    """
    Class for speed estimation by lines
    """
    def __init__(self,
                 speed_lines:list[LineString],
                 line_distances_in_ft:Union[float, list[float]],
                 fps:int=30,
                 ref_point:str='bbox',
                 ref_offset:tuple=(0,0),
                 ref_buffer:int=5,
                 speed_unit:str='mph',
                 speed_field:str='speed'
                 )->None:
        """
        Initialize the class of SpeedMeter\n
        Inputs:
            speed_lines - a list of LineString
            line_distances - distances of between lines in feet
                            could be a consistent distance or a list of distances
            fps - frame per second, default is 30
            ref_point - the reference point of the track, default is 'bbox'
                        'bbox' - bounding box
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
            ref_buffer - the buffer of ref_point for matching, default is 5 pixels
            speed_unit - the unit of the speed, default is 'mph',
                        'km/h' - kilometer per hour
            speed_field - the field name of the speed, default is 'speed'
        """
        self.speed_lines = speed_lines
        if isinstance(line_distances_in_ft, list):
            if len(speed_lines) != (len(line_distances_in_ft)+1):
                raise ValueError('The number of speed lines should be " + \
                                    "one more than the number of distances')
            self.distances = line_distances_in_ft
        else:
            self.distances = float(line_distances_in_ft)

        self.fps = fps
        self.ref_point = ref_point
        self.ref_offset = ref_offset
        self.ref_buffer = ref_buffer
        self.speed_unit = speed_unit
        self.speed_field = speed_field

    def detect(self,
                 tracks:pd.DataFrame,
                 out_file:str=None,
                 frame_infill:bool=True,
                 at_frame:bool=False,
                 verbose:bool=True)->pd.DataFrame:
        """
        Estimate the speed of an object by the lines
        Inputs:
            tracks - a DataFrame of tracks
            frame_infill - if True, infill the missing frames
            at_frame - if True, return the speed at each frame
                       if False, return the speed at the first frame of each line
            verbose - if True, show the progress bar
        Returns:
            A DataFrame of tracks with the speed estimation
        """
        if frame_infill:
            tracks = Tracker.infill_frames(tracks, verbose=verbose)

        if self.ref_point == 'bbox':
            matched_tracks = match_lines_by_bbox(tracks=tracks,
                                        lines=self.speed_lines,
                                        output_ref_xy=True,
                                        verbose=verbose)
        else:
            matched_tracks = match_lines_by_point(tracks=tracks,
                                        lines=self.speed_lines,
                                        ref_point=self.ref_point,
                                        ref_offset=self.ref_offset,
                                        ref_buffer=self.ref_buffer,
                                        output_ref_xy=True,
                                        verbose=verbose)

        matched_tracks = matched_tracks.loc[matched_tracks['line']>-1].copy()
        first_indices = matched_tracks.groupby(['track_id', 'line'])['frame'].idxmin()
        first_intersect_frames = matched_tracks.loc[first_indices].sort_values(
                                                                by=['track_id','line'])
        track_ids = first_intersect_frames['track_id'].unique()

        results = []
        if verbose:
            pbar = tqdm(total=len(track_ids), unit='track', desc='Estimating speed ')

        for track_id in track_ids:
            selected = first_intersect_frames.loc[first_intersect_frames['track_id']
                                                  ==track_id].copy()
            selected['frame_diff'] = selected['frame'].diff()
            selected['time_diff'] = selected['frame_diff'] / (self.fps if self.fps > 0 else 1.e-17)

            if isinstance(self.distances, float):
                selected['line_diff'] = selected['line'].diff()
                selected['distance'] = self.distances * selected['line_diff']
            else:
                selected['distance'] = self._cal_distance(selected)
            selected.dropna(inplace=True)
            selected['speed_fps'] = selected['distance']/selected['time_diff']
            if self.speed_unit == 'mph':
                selected[self.speed_field] = selected['speed_fps'].apply(lambda x: int(x*0.681818))
                selected['speed_unit'] = 'mph'
            elif self.speed_unit == 'kph':
                selected[self.speed_field] = selected['speed_fps'].apply(lambda x: int(x*1.09728))
                selected['speed_unit'] = 'kph'
            else:
                selected[self.speed_field] = selected['speed_fps']
                selected['speed_unit'] = 'fps'
            results.append(selected)
            if verbose:
                pbar.update()

        speed_tracks = pd.concat(results)
        speed_tracks.drop(columns=['speed_fps'], inplace=True)

        if at_frame:
            # merge into tracks
            frame_results = []
            for track_id in track_ids:           
                selected = speed_tracks.loc[speed_tracks['track_id']==track_id] \
                                    .sort_values(by='line').copy()

                frame_ids = selected['frame'].unique()
                for j in range(len(frame_ids)):
                    print(j)
                    input('*')
                    beg_frame = frame_ids[j]
                    end_frame = frame_ids[j+1]-1
                    matched_track_frames = tracks.loc[(tracks['track_id']==track_id) &
                                                    (tracks['frame']>=beg_frame) &
                                                (tracks['frame']<=end_frame)].copy()
                    matched_track_frames['line'] = int(selected.loc[selected['frame']
                                                        ==beg_frame]['line'].values[0])
                    matched_track_frames[self.speed_field] = \
                                                    int(selected.loc[selected['frame']
                                                        ==beg_frame][self.speed_field].values[0])
                    frame_results.append(matched_track_frames)

            if len(frame_results) > 0:
                speed_tracks = pd.concat(frame_results)

        if out_file:
            speed_tracks.to_csv(out_file, index=False)

        return speed_tracks

    def _cal_distance(self, tracks:pd.DataFrame)->list[float]:
        """
        calculate the distance between the speed lines
        Inputs:
            tracks - a DataFrame of tracks
        Returns:
            A DataFrame of tracks with the distance between the speed lines
        """
        lines = tracks['line'].unique()
        results = [np.nan]
        for i in range(len(lines)-1):
            line_1 = lines[i]
            line_2 = lines[i+1]
            if line_2 > line_1:
                distance = sum(self.distances[line_1:line_2])
            else:
                distance = 0
            results.append(distance)

        return results


class SpeedMeterByTransformation():
    """
    Class for speed estimation tranforming image pixels to real-world coordinates
    """
    def __init__(self,
                 source:np.ndarray,
                 destination:np.ndarray,
                 fps:int=30,
                 ref_point:str='bbox',
                 ref_offset:tuple=(0,0),
                 rolling_window:int=None,
                 speed_unit:str='mph',
                 speed_field:str='speed',
                 in_zone:bool=True,
                 distance_unit:str='ft')->None:

        """
        Initialize the class of SpeedMeter\n
        Inputs:\n
            source - the source coordinates on image (4+ points), also used as the speed zone
            destination - the matching real-world coordinates, in distance_unit
            fps - frame per second, default is 30
            ref_point - the reference point of the track, default is 'bbox'
                        'bbox' - bounding box
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
            rolling_window - the window size for rolling average, default is fps frames
            speed_unit - the unit of the speed, default is 'mph',
                        'km/h' - kilometer per hour
            speed_field - the field name of the speed, default is 'speed'
            in_zone - if True, only calculate the speed in the zone
            distance_unit - the unit of the destination coordinates, default is 'ft'
                        'ft' - feet (e.g. State Plane)
                        'm' - meters (e.g. UTM from kmz_to_csv)
        """
        if distance_unit not in ('ft', 'm'):
            raise ValueError("distance_unit must be 'ft' or 'm'.")
        self.transformer = HomographyTransform(source, destination)
        self.distance_unit = distance_unit
        self.speed_zone = Polygon(source)
        self.fps = fps
        self.ref_point = ref_point
        self.ref_offset = ref_offset
        self.speed_unit = speed_unit
        self.speed_field = speed_field
        if rolling_window:
            self.rolling_window = rolling_window
        else:
            self.rolling_window = fps
        self.in_zone = in_zone

    def detect(self,
               tracks:pd.DataFrame,
               out_file:str=None,
               frame_infill:bool=False,
               verbose:bool=True)->pd.DataFrame:
        """
        Estimate the speed of an object by tranformed coordinates\n
        Inputs:\n
            tracks - a DataFrame of tracks
            frame_infill - if True, infill the missing frames
            verbose - if True, show the progress bar
        Returns:\n
            A DataFrame of tracks with the speed estimation
        """
        if frame_infill:
            tracks = Tracker.infill_frames(tracks, verbose=verbose)

        
        matched_tracks = match_zone_by_point(tracks,
                                             self.speed_zone,
                                             ref_point=self.ref_point,
                                             ref_offset=self.ref_offset,
                                             output_ref_xy=True,
                                             verbose=verbose)
        if self.in_zone:
            matched_tracks = matched_tracks.loc[matched_tracks['in_zone']==True].copy()

        anchor_xy = self.transformer.image_to_world(
            matched_tracks[['ref_x', 'ref_y']].to_numpy(dtype=np.float64))
        matched_tracks['anchor_x'] = anchor_xy[:, 0]
        matched_tracks['anchor_y'] = anchor_xy[:, 1]

        track_ids = matched_tracks['track_id'].unique()
        results = []
        if verbose:
            pbar = tqdm(total=len(track_ids), unit='track', desc='Estimating speed ')
        for track_id in track_ids:
            track_seq = matched_tracks.loc[matched_tracks['track_id']==track_id] \
                                            .copy().sort_values(by='frame')
            track_seq['frame_diff'] = track_seq['frame'].diff()
            track_seq['x_diff'] = track_seq['anchor_x'].diff()
            track_seq['y_diff'] = track_seq['anchor_y'].diff()
            track_seq.dropna(inplace=True)
            track_seq['distance'] = np.sqrt(track_seq['x_diff']**2 + track_seq['y_diff']**2)
            if self.distance_unit == 'm':
                track_seq['distance'] = track_seq['distance']*FT_PER_M
            track_seq['time_diff'] = track_seq['frame_diff']/(self.fps if self.fps > 0 else 1.e-17)
            track_seq['speed_fps'] = track_seq['distance']/track_seq['time_diff']
            track_seq['rolling_speed'] = track_seq['speed_fps'].rolling(window=self.rolling_window,
                                            min_periods=1).mean().interpolate(method='nearest')
            if self.speed_unit == 'mph':
                track_seq[self.speed_field] = track_seq['rolling_speed'].apply(
                                                            lambda x: int(x*0.681818))
                track_seq['speed_unit'] = 'mph'
            elif self.speed_unit == 'km/h':
                track_seq[self.speed_field] = track_seq['rolling_speed'].apply(
                                                            lambda x: int(x*1.09728))
                track_seq['speed_unit'] = 'km/h'
            else:
                track_seq[self.speed_field] = track_seq['rolling_speed']
                track_seq['speed_unit'] = 'fps'

            results.append(track_seq)
            if verbose:
                pbar.update()

        if verbose:
            pbar.close()

        speed_tracks = pd.concat(results)
        if out_file:
            speed_tracks.to_csv(out_file, index=False)

        return speed_tracks
