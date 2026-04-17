"""
This module is used to analyze the interaction between two tracks.
    Author: Zhenyu Wang
    Date: 2023/10/20
    Email: wonstran@hotmail.com
    Updates:
        2025/03/19: Add the function of nearmiss based on PET
        2023/10/20: Initial version
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
from dnt.track import Tracker
from dnt.label.labeler2 import Labeler
from tssa.core.interact import InteractionAnalyzer
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, box
from shapely import intersection, distance, intersects
from shapelysmooth import taubin_smooth, chaikin_smooth, catmull_rom_smooth
from tqdm import tqdm
import bisect
import random

class NearmissAnalyzer(InteractionAnalyzer):
    def __init__(self,
                 thresholds:list[float]=[1.5, 3.5, 5],
                 values:list[str]=['high', 'medium', 'low', 'safe'],
                 event_field_name:str='event',
                 fps:int=30,
                 ref_point='bc', 
                 ref_offset:tuple=(0,0),
                 filter_buffer:int=0) -> None:
        '''
        Parameters:
            thresholds: the thresholds for nearmiss, default is [1.5, 3] seconds 
            values: the values for nearmiss, default is ['crtical', 'nearmiss', 'safe']
                        critical - < 1.5s
                        nearmiss - 1.5s ~ 3s
                        safe - > 3s
            fps: frames per sencond, default is 30
            ref_point: the reference point for lane recognition
                        br (buttom-right, default), bl (bottom-left), bc (bottom-center)
                        tl (top-left), tr (top-right), tc (top-center)
                        cc (center-center), cl (center-left), cr (center-right)
            ref_offset: the offset (x, y) for reference point, default is (0, 0)
            filter_buffer: the buffer between two track frames, default is 0 means tracks will be paired only if two tracks are overlapped in time
        '''
        super().__init__(fps=fps, ref_point=ref_point, ref_offset=ref_offset, filter_buffer=filter_buffer)
        self.frame_thresholds = self.__gen_frame_thresholds(thresholds)
        self.values = values
        self.event_field_name = event_field_name
        
    def scan(self,
             tracks_p:pd.DataFrame,
             tracks_y:pd.DataFrame,
             name_p:str='',
             name_y:str=''):
        '''
        Parameters:
            tracks_p: tracks with priority
            tracks_y: tracks should yield to tracks_p
            name_p: the name of tracks_p
            name_y: the name of tracks_y
        '''

        # Add field names to tracks
        tracks_p = InteractionAnalyzer.add_field_names(tracks_p)
        tracks_y = InteractionAnalyzer.add_field_names(tracks_y)

        # Generate reference points
        tracks_p = self.gen_ref(tracks_p, info=name_p)
        tracks_y = self.gen_ref(tracks_y, info=name_y)

        # Generate trajectories
        trajectories_p = self.gen_trajectory(tracks_p, info=name_p)
        trajectories_y = self.gen_trajectory(tracks_y, info=name_y)

        # Scan intersections    
        intersect_pairs = self.scan_intersections(trajectories_p, trajectories_y)

        tqdm.pandas(desc='Scanning nearmiss ', unit='frames') 
        intersect_pairs[self.event_field_name] = intersect_pairs['frame_gap'].progress_apply(lambda x: 
                                                                                             self.__get_nearmiss(abs(x), 
                                                                                                                 self.frame_thresholds, 
                                                                                                                 self.values))
        return pd.DataFrame(intersect_pairs.drop(columns=['geometry']))
    
    def __get_nearmiss(self, frame_gap:int, thresholds:list[float], values:list[str]):
        index = bisect.bisect(thresholds, frame_gap) - 1

        if values is None:
            return index
        else:
            return values[index]
    
    def __gen_frame_thresholds(self, thresholds:list[float]):
        frame_thresholds = [int(x*self.fps) for x in thresholds]
        # Insert 0 at the beginning
        frame_thresholds.insert(0, 0)
        # Remove duplicates by converting to a set, then back to a list
        unique_thresholds = list(set(frame_thresholds))
        # Sort the list
        unique_thresholds.sort()
        return unique_thresholds
    
    @staticmethod
    def draw_clips(pairs:pd.DataFrame,
                   tracks_p:pd.DataFrame,
                   tracks_y:pd.DataFrame,
                   input_video:str,
                   out_path:str,
                   method:str='all',
                   event_list:list=None,
                   size:int=1,
                   thick:int=2,
                   padding:int=0,
                   show_track:bool=False,
                   show_desc:bool=False,
                   prefix:bool=False,
                   event_field_name:str='event',
                   colors:list=[(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0)],
                   values:list[str]=['high', 'medium', 'low', 'safe'],
                   draw_method:str='opecv',
                   encoder:str='libx264',
                   pix_fmt:str='yuv420p',
                   verbose:bool=True):
        '''
        Parameters:
        - yields: the dataframe of stop events
        - tracks_p: the dataframe of tracks with priority
        - tracks_y: the dataframe of tracks should yield
        - input_video: the raw video file, if None, generate the label files only
        - out_path: the folder for outputing track clips
        - method: 'all' (default) - all tracks,  'include' - include specific event types, 'exclude' - exclude specific event types
        - event_list: the list of events types, if method == 'include' or 'exclude'
        - size: font size, default is 1
        - thick: line thinckness, defualt is 2
        - padding: add addtional frames at beggining and ending, default is 0
        - show_track: show track id, default is False
        - show_desc: show event desc, default is Falase
        - verbose: if show progressing bar, default is True
        
        Return:
        - A list of dataframes for labels, ['frame','type','coords','color','size','thick','desc']
        '''
        
        tracks_p = InteractionAnalyzer.add_field_names(tracks_p)
        tracks_y = InteractionAnalyzer.add_field_names(tracks_y)

        tracks_p_grouped = tracks_p.groupby('track')
        tracks_y_grouped = tracks_y.groupby('track')

        if method == 'all':
            events = pairs.copy()
        elif method == 'include':
            events = pairs[pairs[event_field_name].isin(event_list)].copy()
        elif method == 'exclude':
            events = pairs[~pairs[event_field_name].isin(event_list)].copy()        

        pbar = tqdm(total= len(events) , unit='events', desc="Generating labels ...")        
        labeler = Labeler(method=draw_method, encoder=encoder, pix_fmt=pix_fmt)
        for _, event in events.iterrows():
            result = []
            track_p_id = event['track_p']
            track_y_id = event['track_y']

            if colors is None:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = colors[values.index(event[event_field_name])]
            
            frames_p = tracks_p_grouped.get_group(track_p_id)            
            frames_y = tracks_y_grouped.get_group(track_y_id)
            frames = pd.concat([frames_p, frames_y])
            for _, frame in frames.iterrows():
                label_text = ''
                if show_track:
                    label_text += '| T:' + str(int(frame['track']))
                
                if show_desc:
                    label_text += ' | ' + str(event[event_field_name])

                result.append([frame['frame'], 'bbox', [(frame['x'], frame['y']), (frame['x']+frame['w'], frame['y']+frame['h'])], 
                            color, size, thick, label_text])
    
            df = pd.DataFrame(result, columns=['frame','type','coords','color','size','thick','desc'])

            if input_video:
                if prefix:
                    basename = os.path.splitext(os.path.basename(input_video))[0]
                    file_name = os.path.join(out_path, basename+'_'+str(track_p_id)+'-'+str(track_y_id)+'_'+str(event[event_field_name])+'.mp4')
                else:
                    file_name = os.path.join(out_path, str(track_p_id)+'-'+str(track_y_id)+'_'+str(event[event_field_name])+'.mp4')
                min_frame = df['frame'].min() - padding
                max_frame = df['frame'].max() + padding
                labeler.draw(input_video=input_video, output_video=file_name, draws=df, 
                                start_frame=min_frame, end_frame=max_frame, verbose=False)

            if verbose:
                pbar.update()      
        
        pbar.close()