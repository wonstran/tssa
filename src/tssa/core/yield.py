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
from dnt.label.labeler2 import Labeler
from tssa.core.interact import InteractionAnalyzer
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, box
from shapely import intersection, distance, intersects
from shapelysmooth import taubin_smooth, chaikin_smooth, catmull_rom_smooth
from tqdm import tqdm
import bisect

class YieldAnalyzer(InteractionAnalyzer):
    def __init__(self,
                 waiting_dist_p:int=600,
                 waiting_dist_y:int=300, 
                 leading_p:bool=False,
                 leading_axis_p:str='y',
                 leading_y:bool=True,
                 leading_axis_y:str='x',
                 yield_gap:int=10,
                 fps:int=30,
                 ref_point='bc', 
                 ref_offset:tuple=(0,0),
                 filter_buffer:int=0,
                 p_zone:Polygon=None,
                 y_zone:Polygon=None) -> None:
        '''
        Parameters:
            threshold: the hyperparameter to determine if a yield event (frame difference <=yield_gap*fps), default is 3 seconds
            fps: frames per sencond, default is 30
            ref_point: the reference point for lane recognition
                        br (buttom-right, default), bl (bottom-left), bc (bottom-center)
                        tl (top-left), tr (top-right), tc (top-center)
                        cc (center-center), cl (center-left), cr (center-right)
            ref_offset: the offset (x, y) for reference point, default is (0, 0)
            filter_buffer: the buffer between two track frames, default is 0 means tracks will be paired only if two tracks are overlapped in time
        '''
        super().__init__(fps=fps, ref_point=ref_point, ref_offset=ref_offset, filter_buffer=filter_buffer)
        self.waiting_dist_p = waiting_dist_p
        self.waiting_dist_y = waiting_dist_y
        self.leading_p = leading_p
        self.leading_axis_p = leading_axis_p
        self.leading_y = leading_y
        self.leading_axis_y = leading_axis_y
        self.yield_gap = yield_gap
        self.p_zone = p_zone
        self.y_zone = y_zone

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

        # scan yield events
        if (self.p_zone is not None) and (self.y_zone is not None):
            yield_pairs = self.scan_yield_events_byzone(intersect_pairs, tracks_p, tracks_y)
        else:
            yield_pairs = self.scan_yield_events(intersect_pairs, tracks_p, tracks_y)
        
        return yield_pairs

    def scan_yield_events(self, pairs:gpd.GeoDataFrame, tracks_p:gpd.GeoDataFrame, tracks_y:gpd.GeoDataFrame, 
                          video_index:int=None, video_tot:int=None)->pd.DataFrame:

        pbar = tqdm(total=len(pairs), unit=' pair')
        if video_index and video_tot:
            pbar.desc = 'Scan events {} of {}'.format(video_index, video_tot)
        else:
            pbar.desc = 'Scan events'
        for index, pair in pairs.iterrows():
            
            if pair['frame_p'] < pair['frame_y']:

                selected = tracks_y[(tracks_y['track']==pair['track_y']) & (tracks_y['frame']<=pair['frame_y'])].copy() # find the frames for track_y before reaching conflict point
                selected['dif'] = selected['frame'].apply(lambda f: abs(f - pair['frame_p'])) # calculate the frame difference between each before frame and the conflict frame
                selected = selected[selected['dif']<=self.yield_gap]        # the frame difference should <= self.yield_gap
                closest = selected[selected['dif']==selected['dif'].min()]  # find the closest frame
                if len(closest)>0:
                    closest_point = Point(closest['x'].iloc[0]+closest['w'].iloc[0]/2, 
                                          closest['y'].iloc[0]+closest['h'].iloc[0]) # find the cloest point
                    dist = distance(closest_point, pair.geometry)   # calculate the distance
                    pairs.at[index, 'dist_y'] = dist
                    inwaiting = 1 if dist<=self.waiting_dist_y else 0

                    if self.leading_y:
                        isleading = 1 if self.__is_leading(pair['track_y'], closest['frame'].iloc[0], closest['x'].iloc[0], 
                                      closest['y'].iloc[0], pair.geometry.x, pair.geometry.y, tracks_y, self.leading_axis_y) else 0
                    else:
                        isleading = 1
                
                    if (inwaiting == 1) and (isleading == 1):
                        pairs.at[index, 'event'] = 'yield'

                    pairs.at[index, 'closest_frame'] = closest['frame'].iloc[0]

            elif pair['frame_p'] >= pair['frame_y']:
                selected = tracks_p[(tracks_p['track']==pair['track_p']) & (tracks_p['frame']<=pair['frame_p'])].copy()
                selected['dif'] = selected['frame'].apply(lambda f: abs(f - pair['frame_y']))
                selected = selected[selected['dif']<=self.yield_gap] 
                closest = selected[selected['dif']==selected['dif'].min()]
                if len(closest)>0:
                    closest_point = Point(closest['x'].iloc[0]+closest['w'].iloc[0]/2, 
                                          closest['y'].iloc[0]+closest['h'].iloc[0])
                    dist = distance(closest_point, pair.geometry)
                    inwaiting = 1 if dist<=self.waiting_dist_p else 0
                    pairs.at[index, 'dist_p'] = dist
                    if self.leading_p:
                        isleading = 1 if self.__is_leading(pair['track_p'], closest['frame'].iloc[0], closest['x'].iloc[0], 
                                      closest['y'].iloc[0], pair.geometry.x, pair.geometry.y, tracks_p, self.leading_axis_p) else 0
                    else:
                        isleading = 1
                
                    if (inwaiting == 1) and (isleading == 1):
                        pairs.at[index, 'event'] = 'not_yield'

                    pairs.at[index, 'closest_frame'] = closest['frame'].iloc[0]

            pbar.update()
        
        pbar.close()

        return self.__export(pairs)
    
    def scan_yield_events_byzone(self, pairs:gpd.GeoDataFrame, tracks_p:gpd.GeoDataFrame, tracks_y:gpd.GeoDataFrame, 
                                 video_index:int=None, video_tot:int=None)->pd.DataFrame:

        pbar = tqdm(total=len(pairs), unit=' pair')
        if video_index and video_tot:
            pbar.desc = 'Scan yield events {} of {}'.format(video_index, video_tot)
        else:
            pbar.desc = 'Scan yield events'
        for index, pair in pairs.iterrows():
            
            if pair['frame_p'] < pair['frame_y']:

                selected = tracks_y[(tracks_y['track']==pair['track_y']) & (tracks_y['frame']<=pair['frame_y'])].copy() # find the frames for track_y before reaching conflict point
                selected['dif'] = selected['frame'].apply(lambda f: abs(f - pair['frame_p'])) # calculate the frame difference between each before frame and the conflict frame
                selected = selected[selected['dif']<=self.yield_gap]        # the frame difference should <= self.yield_gap
                closest = selected[selected['dif']==selected['dif'].min()]  # find the closest frame
                if len(closest)>0:
                    x, y, w, h = closest['x'].iloc[0], closest['y'].iloc[0], closest['w'].iloc[0], closest['h'].iloc[0]
                    closest_bbox = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
                    
                    inwaiting = 1 if intersects(closest_bbox, self.y_zone) else 0
                    pairs.at[index, 'dist_y'] = -1
                    if self.leading_y:
                        isleading = 1 if self.__is_leading(pair['track_y'], closest['frame'].iloc[0], closest['x'].iloc[0], 
                                      closest['y'].iloc[0], pair.geometry.x, pair.geometry.y, tracks_y, self.leading_axis_y) else 0
                    else:
                        isleading = 1
                
                    if (inwaiting == 1) and (isleading == 1):
                        pairs.at[index, 'event'] = 'yield'

                    pairs.at[index, 'closest_frame'] = closest['frame'].iloc[0]

            elif pair['frame_p'] >= pair['frame_y']:
                selected = tracks_p[(tracks_p['track']==pair['track_p']) & (tracks_p['frame']<=pair['frame_p'])].copy()
                selected['dif'] = selected['frame'].apply(lambda f: abs(f - pair['frame_y']))
                selected = selected[selected['dif']<=self.yield_gap] 
                closest = selected[selected['dif']==selected['dif'].min()]
                if len(closest)>0:
                    x, y, w, h = closest['x'].iloc[0], closest['y'].iloc[0], closest['w'].iloc[0], closest['h'].iloc[0]
                    closest_bbox = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
                            
                    inwaiting = 1 if intersects(closest_bbox, self.p_zone) else 0
                    pairs.at[index, 'dist_p'] = -1
                    if self.leading_p:
                        isleading = 1 if self.__is_leading(pair['track_p'], closest['frame'].iloc[0], closest['x'].iloc[0], 
                                      closest['y'].iloc[0], pair.geometry.x, pair.geometry.y, tracks_p, self.leading_axis_p) else 0
                    else:
                        isleading = 1
                
                    if (inwaiting == 1) and (isleading == 1):
                        pairs.at[index, 'event'] = 'not_yield'

                    pairs.at[index, 'closest_frame'] = closest['frame'].iloc[0]

            pbar.update()
        
        pbar.close()

        return self.__export(pairs)
    
                             
    def __export(self, pairs:gpd.GeoDataFrame):
        pairs['int_x'] = pairs.geometry.x
        pairs['int_y'] = pairs.geometry.y
        return pd.DataFrame(pairs.drop(columns=['geometry']))

    def __is_leading(self, track_id:int, frame_close:int, close_x:int, close_y:int, int_x:int, int_y:int, 
                          tracks:gpd.GeoDataFrame, leading_axis:str):
        
        min_x = min(close_x, int_x)
        max_x = max(close_x, int_x)
        min_y = min(close_y, int_y)
        max_y = max(close_y, int_y)
        if leading_axis == 'x':
            leadings = tracks[(tracks['track']!=track_id) & (tracks['frame']==frame_close) & 
                              (tracks['ref_x']>=min_x) & (tracks['ref_x']<=max_x)]
        elif leading_axis == 'y':
            leadings = tracks[(tracks['track']!=track_id) & (tracks['frame']==frame_close) & 
                              (tracks['ref_y']>=min_y) & (tracks['ref_y']<=max_y)]
        elif leading_axis == 'xy':
            leadings = tracks[(tracks['track']!=track_id) & (tracks['frame']==frame_close) & 
                              (tracks['ref_x']>=min_x) & (tracks['ref_x']<=max_x) &
                              (tracks['ref_y']>=min_y) & (tracks['ref_y']<=max_y)]
        
        if len(leadings) > 0:
            return False
        else:
            return True 

    def __get_waiting_trj_y(self, track_id:int, frame_id_p:int, frame_id_y:int, intersect_point:Point, trajectories:gpd.GeoDataFrame) -> any:
        
        result = -1
        if frame_id_p < frame_id_y:
            selected = trajectories[trajectories['track']==track_id]
            df = pd.DataFrame(np.vstack((selected['frames'].tolist(), selected['ref_x'].tolist(), selected['ref_y'].tolist())).T, 
                              columns=['frame', 'x', 'y'])
            df = df.loc[df['frame']<=frame_id_y]
            df['dif'] = df['frame'].apply(lambda f: abs((f - frame_id_p)))
            closest = df[df['dif']==df['dif'].min()]
            closest_point = Point(closest['x'].ioc[0], closest['y'].ioc[0])
            dist = distance(closest, intersect_point)

        return result
    
    @staticmethod
    def draw_clips(yields:pd.DataFrame,
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
                   verbose:bool=True,
                   video_index:int=None,
                   video_tot:int=None):
        '''
        Parameters:
        - yields: the dataframe of stop events
        - tracks_p: the dataframe of tracks with priority
        - tracks_y: the dataframe of tracks should yield
        - input_video: the raw video file, if None, generate the label files only
        - out_path: the folder for outputing track clips
        - method: 'all' (default) - all tracks,  'specify' - specify events, 
                    'yield' - yield events, 'not_yield' - non-yield events, 'none' - non interaction events
        - random_number: the number of track ids if method == 'random'
        - event_list: the list of events [(track_p_id, track_y_id)] if method == 'specify'
        - size: font size, default is 1
        - thick: line thinckness, defualt is 2
        - padding: add addtional frames at beggining and ending, default is 0
        - show_track: show track id, default is False
        - show_desc: show event desc, default is Falase
        - verbose: if show progressing bar, default is True
        - video_idex: the index of the video in processing
        - video_tot: the total number of videos

        Return:
        - A list of dataframes for labels, ['frame','type','coords','color','size','thick','desc']
        '''
        
        tracks_p = InteractionAnalyzer.add_field_names(tracks_p)
        tracks_y = InteractionAnalyzer.add_field_names(tracks_y)

        tracks_p_grouped = tracks_p.groupby('track')
        tracks_y_grouped = tracks_y.groupby('track')

        if method == 'all':
            events = yields
        elif method == 'specify':
            events = []
            for p, y in event_list:
                events.append(yields[(yields['track_p']==p) & (yields['track_y']==y)])
            events = pd.concat(events)
        elif method == 'yield':
            events = yields[yields['event']=='yield'].copy()
        elif method == 'not_yield':
            events = yields[yields['event']=='not_yield'].copy()
        elif method == 'event':
            events = yields[yields['event'].isin(['yield', 'not_yield'])].copy()
        elif method == 'none':
            events = yields[~yields['event'].isin(['yield', 'not_yield'])].copy()        

        pbar = tqdm(total= len(events) , unit='event')        
        if video_index and video_tot:
            pbar.set_description_str("Generating labeles for yielding events {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Generating labels for yielding events")

        labeler = Labeler()
        for index, event in events.iterrows():
            result = []
            track_p_id = event['track_p']
            track_y_id = event['track_y']
            int_x = int(event['int_x'])
            int_y = int(event['int_y'])

            if event['event'] == 'yield':
                desc = 'yield'
                color = (0, 255, 0)
            elif event['event'] == 'not_yield':
                desc = 'not yield'
                color = (0, 0, 255)
            else:
                desc = 'no interaction'
                color = (255, 0, 0)
            
            frames_p = tracks_p_grouped.get_group(track_p_id)            
            frames_y = tracks_y_grouped.get_group(track_y_id)
            frames = pd.concat([frames_p, frames_y])
            for index2, frame in frames.iterrows():
                label_text = ''
                if show_track:
                    label_text+='| T:'+str(int(frame['track']))
                
                if show_desc:
                    label_text+= ' | '+desc

                result.append([frame['frame'], 'bbox', [(frame['x'], frame['y']), (frame['x']+frame['w'], frame['y']+frame['h'])], 
                            color, size, thick, label_text])
                result.append([frame['frame'], 'box', [(int_x - 5, int_y - 5), (int_x + 5, int_y + 5)], 
                            (255, 255, 0), size, thick, ''])

            df = pd.DataFrame(result, columns=['frame','type','coords','color','size','thick','desc'])
                
            if out_path:
                file_name = os.path.join(out_path, str(track_p_id)+'-'+str(track_y_id)+'_label.csv')
                df.to_csv(file_name, index=False)

            if input_video:
                file_name = os.path.join(out_path, str(track_p_id)+'-'+str(track_y_id)+'_label.mp4')
                min_frame = df['frame'].min() - padding
                max_frame = df['frame'].max() + padding
                labeler.draw(input_video=input_video, output_video=file_name, draws=df, 
                                start_frame=min_frame, end_frame=max_frame, verbose=False)

            if verbose:
                pbar.update()      
        
        pbar.close()

        