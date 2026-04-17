'''
This module contains the Lane class, which is used to identify a object's travel lane (path).
'''
import os
import sys
from typing import Union
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tssa.engine.zone import match_zones_by_point, match_zone_by_point
from tssa.io.track_file import read_track, write_track_header

class LaneDetector:
    '''
    Detect the travel lane of an object
    Methods:
        __init__(): initialize the class
        detect_lane(): detect the lane of an object
    '''
    def __init__(self, lanes:list[Polygon],
                 lane_names:list[str]=None,
                 ref_point:str='cc',
                 ref_offset:Union[tuple, list[tuple]]=(0,0)) -> None:
        '''
        Initialize the class of LaneDetector\n
        Inputs:\n
            lanes - a list of Polygons, each represents a lane
            ref_point - the reference point of the lane, default is 'cc' (center of the lane)
                        'cc' - center of the bounding box
                        'tc' - top center of the bounding box
                        'bc' - bottom center of the bounding box
                        'cl' - center left of the bounding box
                        'cr' - center right of the bounding box
                        'tl' - top left of the bounding box
                        'tr' - top right of the bounding box
                        'bl' - bottom left of the bounding box
                        'br' - bottom right of the bounding box
            lane_names - a list of lane names, default is None
                         \tif None, the lane ID is the index of the lane in lanes
            red_offsets - a list of tuples, each tuple is the offset (x, y) 
                            \tof the reference point of the lane
                         if single tuple, or the list lenth less than the number of lanes,
                            \tthe same offset is applied to all lanes
        '''
        self.lanes = lanes

        if lanes is None:
            raise Exception("The lanes is not defined!")

        if lane_names is None:
            self.lane_names = list(range(len(lanes)))
        elif len(lane_names) == len(lanes):
            self.lane_names = lane_names
        else:
            self.lane_names = list(range(len(lanes)))

        self.ref_point = ref_point

        if isinstance(ref_offset, tuple):
            self.ref_offset = ref_offset
            self.ref_offsets = None
        elif isinstance(ref_offset, list):
            if (len(ref_offset)==1) or (len(ref_offset)!=len(lanes)):
                self.ref_offset = ref_offset[0]
                self.ref_offsets = None
            else:
                self.ref_offset = None
                self.ref_offsets = ref_offset

    def _replace_lane_names(self, tracks:pd.DataFrame) -> pd.DataFrame:
        '''
        Replace the lane ID with lane names\n
        Inputs:\n
            tracks - a DataFrame of tracks
        Return:\n
            tracks - a DataFrame of tracks with lane names
        '''
        if self.lane_names:
            tracks['lane'] = tracks['lane'].apply(lambda lane_id:
                                                  self.lane_names[lane_id]
                                                  if lane_id>=0 else lane_id)
        return tracks

    def detect(self,
               tracks:pd.DataFrame=None,
               track_file:str=None,
               mask_unmatched:bool=False,
               out_file:str=None,
               output_ref:bool=True) -> pd.DataFrame:
        '''
        Detect the lane of an object\n
        Inputs:\n
            tracks - a DataFrame of tracks
            track_file - a txt file contains tracks
                        if tracks is None, read the track_file
            mask_unmatched - mask the unmatched tracks, default is False
            out_file - a txt file to save the tracks with lane information
            output_ref - output the reference point of the lane, default is True
        Return:\n
            tracks - a DataFrame of tracks with lane information
        '''
        if (tracks is None) and (not track_file is None):
            tracks = read_track(track_file)
        else:
            raise Exception("The tracks is not defined!")

        if self.ref_offset:
            results = match_zones_by_point(tracks,
                                           self.lanes,
                                           self.ref_point,
                                           self.ref_offset,
                                           'lane',
                                           output_ref_xy=output_ref)
        elif self.ref_offsets:
            coll_tracks = []
            pbar = tqdm(total=len(self.lanes), unit='lane')
            for i in range(len(self.lanes)):
                pbar.set_description(f"Scanning lane {i+1}")
                result_zone = match_zone_by_point(tracks,
                                              self.lanes[i],
                                              self.ref_point,
                                              self.ref_offsets[i],
                                              output_ref_xy=output_ref,
                                              verbose=False)
                matched = result_zone.loc[result_zone['in_zone']==True].copy()
                matched.drop(columns=['in_zone'], inplace=True)
                matched['lane'] = i
                coll_tracks.append(matched)
                pbar.update(1)
            pbar.close()
            results = pd.concat(coll_tracks).sort_values(by=['frame', 'track_id'])
            results.reset_index(drop=True, inplace=True)
        else:
            results = match_zones_by_point(tracks,
                                           self.lanes,
                                           (0,0),
                                           'lane',
                                           output_ref_xy=output_ref,
                                           verbose=False)
        results = self._replace_lane_names(results)

        if mask_unmatched:
            results = results.loc[results['lane']>=0].copy()
        if out_file:
            write_track_header(results, out_file)

        return results
