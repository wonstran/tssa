'''
This script is used to match tracks with pre-defined lines
last modified: 2024-10-13
created by: wonstran
'''
import os
import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tssa.io.definition import track_fields
from tssa.engine.geometry import generate_ref_points, generate_bbox

def match_line_by_point(tracks:pd.DataFrame,
                        line:Polygon,
                        ref_point:str='cc',
                        ref_offset:tuple=(0,0),
                        ref_buffer:int=5,
                        match_field:str='on_line',
                        output_ref_xy:bool=False,
                        verbose:bool=True)->pd.DataFrame:
    '''
    Scan the position of a point of tracks and determine if the track belongs to a pre-define zone
    Inputs:
        tracks - a DataFrame of tracks
        line - a linestring
        ref_point - the reference point of the track, default is 'cc'
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
        match_field - the field name of the match status, default is 'on_line'
        output_ref_xy - if True, output the reference point x and y
        verbose - if True, show the progress bar
    Returns:
        A DataFrame of tracks with the matched status
    '''
    tracks.columns = track_fields

    geo_tracks = generate_ref_points(tracks, ref_point, ref_offset, ref_buffer, verbose)
    geo_tracks[match_field] = False
    geo_tracks.loc[(geo_tracks.geometry.intersects(line)), match_field] = True

    if output_ref_xy:
        geo_tracks[['ref_x', 'ref_y']] = geo_tracks.geometry.apply(
                                            lambda p: [int(p.centroid.x),
                                                       int(p.centroid.y)]).apply(pd.Series)
    return pd.DataFrame(geo_tracks.drop(columns='geometry'))

def match_lines_by_point(tracks:pd.DataFrame,
                        lines:list[LineString],
                        ref_point:str='cc',
                        ref_offset:tuple=(0,0),
                        ref_buffer:int=5,
                        line_field:str='line',
                        output_ref_xy:bool=False,
                        verbose:bool=True)->pd.DataFrame:
    '''
    Scan the position of a point of tracks and determine if the track belongs to pre-define zones
    Inputs:
        tracks - a DataFrame of tracks
        zones - a list of polygons
        ref_point - the reference point of the track, default is 'cc'
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
        line_field - the field name of the match status, default is 'line'
        output_ref_xy - if True, output the reference point x and y
        verbose - if True, show the progress bar
    Returns:
        A DataFrame of tracks with the zone field
    '''

    tracks.columns = track_fields
    tracks[line_field] = -1

    geo_tracks = generate_ref_points(tracks, ref_point, ref_offset, ref_buffer, verbose)
    if verbose:
        pbar = tqdm(total=len(lines), unit=' line', desc='Matching lines ')
    for i in range(len(lines)):
        geo_tracks.loc[(geo_tracks.geometry.intersects(lines[i])), line_field] = i
        if verbose:
            pbar.update()
    if verbose:
        pbar.close()
    if output_ref_xy:
        geo_tracks[['ref_x', 'ref_y']] = geo_tracks.geometry.apply(
                                            lambda p: [int(p.centroid.x), 
                                                       int(p.centroid.y)]).apply(pd.Series)
    return pd.DataFrame(geo_tracks.drop(columns='geometry'))

def match_lines_by_bbox(tracks:pd.DataFrame,
                        lines:list[LineString],
                        line_field:str='line',
                        output_ref_xy:bool=False,
                        verbose:bool=True)->pd.DataFrame:
    '''
    Scan the position of a point of tracks and determine if the track belongs to pre-define zones
    Inputs:
        tracks - a DataFrame of tracks
        zones - a list of polygons
        line_field - the field name of the match status, default is 'line'
        output_ref_xy - if True, output the reference point x and y
        verbose - if True, show the progress bar
    Returns:
        A DataFrame of tracks with the zone field
    '''

    tracks.columns = track_fields
    tracks[line_field] = -1

    geo_tracks = generate_bbox(tracks, verbose)
    if verbose:
        pbar = tqdm(total=len(lines), unit=' line', desc='Matching lines ')
    for i in range(len(lines)):
        geo_tracks.loc[(geo_tracks.geometry.intersects(lines[i])), line_field] = i
        if verbose:
            pbar.update()
    if verbose:
        pbar.close()
    if output_ref_xy:
        geo_tracks[['ref_x', 'ref_y']] = geo_tracks.geometry.apply(
                                            lambda p: [int(p.centroid.x), 
                                                       int(p.centroid.y)]).apply(pd.Series)
    return pd.DataFrame(geo_tracks.drop(columns='geometry'))
