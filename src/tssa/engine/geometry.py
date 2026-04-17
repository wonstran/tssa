'''
This script is used to calculate spatial relationships
updates: 
    2021-09-30
'''
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely import Point, LineString, Polygon

def point_relative_to_line(point:np.array, line_start:np.array, line_end:np.array) -> str:
    """
    Determines the position of a point relative to a line.

    Args:
        point (numpy.ndarray): The point in question (shape: (2,))
        line_start (numpy.ndarray): The starting point of the line (shape: (2,))
        line_end (numpy.ndarray): The ending point of the line (shape: (2,))

    Returns:
        str: 'left-up', 'right-down', or 'on'
    """

    # Calculate the cross product of the vectors from the line start to the point and the line end
    cross_product = np.cross(line_end - line_start, point - line_start)

    result = 'on'
    if cross_product > 0:
        result = 'right-down'
    elif cross_product < 0:
        result = 'left-up'

    return result

'''
# will be removed -----------------------------------------------------

def point_relative_to_polyline(point:Point, polyline:LineString) -> int:
    """
    Determines the relative position of a point to a polyline.\n
    Inputs:\n
        point (shapely.geometry.Point): The point in question
        polyline (shapely.geometry.LineString): The polyline
    Returns:\n
        'left-up' if the point is to the left or up of the polyline,
        'right-down' if the point is to the right-down of the polyline,
        'on' if the point lies on the polyline.
    """

    # Project the point onto the polyline
    projected_point = polyline.interpolate(polyline.project(point))
    # Create a line segment from the start of the polyline to the projected point
    line_segment = LineString([polyline.coords[0], projected_point.coords[0]])
    # Check if the point is on the polyline
    if point.distance(polyline) < 1e-8:
        return 'on'

    # Calculate the cross product of the line segment and the vector from the projected point to the point
    cross_product = line_segment.crosses(LineString([projected_point.coords[0], point.coords[0]]))
    # Determine the direction based on the cross product
    if cross_product > 0:
        return 'left-up'
    else:
        return 'right-down'
'''

def point_relative_to_polyline(point:Point, line:LineString, min_distance:float=1e-8) -> str:
    """
    Determines the relative position of a point to a polyline.\n
    Inputs:\n
        point (shapely.geometry.Point): The point in question
        polyline (shapely.geometry.LineString): The polyline
    Returns:\n
        'left-up' if the point is to the left or up of the polyline,
        'right-down' if the point is to the right-down of the polyline,
        'on' if the point lies on the polyline.
    """

    relative_position = "on"
    
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[1]
    x, y = point.x, point.y

    # Vector AB (segment), and vector AP (from start of segment to point)
    dx1, dy1 = x2 - x1, -(y2 - y1)
    dx2, dy2 = x - x1, -(y - y1)

    # Cross product: AB × AP
    cross = dx1 * dy2 - dy1 * dx2

    # Distance from point to segment (used to determine closest segment)
    segment = LineString([(x1, y1), (x2, y2)])
    dist = point.distance(segment)

    if dist < min_distance:
        relative_position = "on"
    else:
        if cross > 0:
            relative_position = "left-up"
        elif cross < 0:
            relative_position = "right-down"
    
    return relative_position

def generate_ref_points(tracks:pd.DataFrame,
                        ref_point:str='cc',
                        ref_offset:tuple=(0,0),
                        ref_buffer:int=5,
                        verbose:bool=True)->gpd.GeoDataFrame:
    """
    Generate reference points for tracks\n
    Args:\n
        tracks - a DataFrame of tracks
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
        verbose - if True, show the progress bar
    Returns:\n
        A GeoDataFrame of tracks with reference points
    """
    if verbose:
        tqdm.pandas(desc='Generating geometries for reference points')
        if ref_point == 'cc':
            geo = tracks.progress_apply(lambda track: Point([(track['x']
                                                              + track['w']/2
                                                              + ref_offset[0]),
                                                             (track['y']
                                                              + track['h']/2
                                                              + ref_offset[1])
                                                            ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'tc':
            geo = tracks.progress_apply(lambda track: Point([(track['x']
                                                              + track['w']/2
                                                              + ref_offset[0]),
                                                             (track['y']
                                                              + ref_offset[1])
                                                            ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'bc':
            geo = tracks.progress_apply(lambda track: Point([[(track['x']
                                                               + track['w']/2
                                                               + ref_offset[0]),
                                                              (track['y']
                                                               + track['h']
                                                               + ref_offset[1])
                                                              ]]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'cl':
            geo = tracks.progress_apply(lambda track: Point([[(track['x']
                                                               + ref_offset[0]),
                                                              (track['y']
                                                               + track['h']/2 
                                                               + ref_offset[1])
                                                             ]]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'cr':
            geo = tracks.progress_apply(lambda track: Point([(track['x']
                                                              + track['w']
                                                              + ref_offset[0]),
                                                             (track['y']
                                                              + track['h']/2 
                                                              + ref_offset[1])
                                                            ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'tl':
            geo = tracks.progress_apply(lambda track: Point([(track['x']
                                                             + ref_offset[0]),
                                                            (track['y']
                                                             + ref_offset[1])
                                                            ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'tr':
            geo = tracks.progress_apply(lambda track: Point([(track['x']
                                                              + track['w']
                                                              + ref_offset[0]),
                                                             (track['y'] + ref_offset[1])
                                                            ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'bl':
            geo = tracks.progress_apply(lambda track: Point([(track['x']
                                                              + ref_offset[0]),
                                                             (track['y']
                                                              + tracks['h']
                                                              + ref_offset[1])
                                                            ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'br':
            geo = tracks.progress_apply(lambda track: Point([(track['x']
                                                              + track['w']
                                                              + ref_offset[0]),
                                                             (track['y']
                                                              + track['h']
                                                              + ref_offset[1])
                                                            ]).buffer(ref_buffer),
                                        axis=1)
        else:
            geo = tracks.progress_apply(lambda track: Point([(track['x'] + track['w']/2 +
                                                              ref_offset[0]),
                                                            (track['y'] + track['h'] +
                                                            ref_offset[1])]).buffer(ref_buffer),
                                                            axis=1)
    else:
        if ref_point == 'cc':
            geo = tracks.apply(lambda track: Point([(track['x']
                                                     + track['w']/2
                                                     + ref_offset[0]),
                                                    (track['y']
                                                     + track['h']/2
                                                     + ref_offset[1])
                                                    ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'tc':
            geo = tracks.apply(lambda track: Point([(track['x']
                                                     + track['w']/2
                                                     + ref_offset[0]),
                                                    (track['y']
                                                     + ref_offset[1])
                                                   ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'bc':
            geo = tracks.apply(lambda track: Point([[(track['x']
                                                      + track['w']/2
                                                      + ref_offset[0]),
                                                     (track['y']
                                                      + track['h']
                                                      + ref_offset[1])]
                                                    ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'cl':
            geo = tracks.apply(lambda track: Point([[(track['x']
                                                      + ref_offset[0]),
                                                     (track['y']
                                                      + track['h']/2
                                                      + ref_offset[1])]
                                                    ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'cr':
            geo = tracks.apply(lambda track: Point([(track['x']
                                                     + track['w']
                                                     + ref_offset[0]),
                                                    (track['y']
                                                     + track['h']/2
                                                     + ref_offset[1])
                                                    ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'tl':
            geo = tracks.apply(lambda track: Point([(track['x']
                                                     + ref_offset[0]),
                                                    (track['y']
                                                     + ref_offset[1])
                                                    ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'tr':
            geo = tracks.apply(lambda track: Point([(track['x']
                                                     + track['w']
                                                     + ref_offset[0]),
                                                    (track['y']
                                                     + ref_offset[1])
                                                    ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'bl':
            geo = tracks.apply(lambda track: Point([(track['x']
                                                     + ref_offset[0]),
                                                    (track['y']
                                                     + tracks['h']
                                                     + ref_offset[1])
                                                    ]).buffer(ref_buffer),
                                        axis=1)
        elif ref_point == 'br':
            geo = tracks.apply(lambda track: Point([(track['x']
                                                     + track['w']
                                                     + ref_offset[0]),
                                                    (track['y']
                                                     + track['h']
                                                     + ref_offset[1])
                                                    ]).buffer(ref_buffer),
                                        axis=1)
        else:
            geo = tracks.apply(lambda track: Point([(track['x']
                                                     + track['w']/2
                                                     + ref_offset[0]),
                                                    (track['y']
                                                     + track['h']
                                                     + ref_offset[1])
                                                    ]).buffer(ref_buffer),
                                        axis=1)
    geo_tracks = gpd.GeoDataFrame(tracks, geometry=geo)
    return geo_tracks

def generate_bbox(tracks:pd.DataFrame,
                  verbose:bool=True)->gpd.GeoDataFrame:
    """
    Generate reference points for tracks\n
    Args:\n
        tracks - a DataFrame of tracks
        verbose - if True, show the progress bar
    Returns:\n
        A GeoDataFrame of tracks with bbox geometries
    """
    if verbose:
        tqdm.pandas(desc='Generating geometries for reference points')
        geo = tracks.progress_apply(lambda track: Polygon([(track['x'], track['y']),
                                                           (track['x'] + track['w'], 
                                                            track['y']),
                                                           (track['x'] + track['w'],
                                                            track['y'] + track['h']),
                                                           (track['x'], track['y'] + track['h'])
                                                           ]),
                                        axis=1)
    geo_tracks = gpd.GeoDataFrame(tracks, geometry=geo)
    return geo_tracks