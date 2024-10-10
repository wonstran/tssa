'''
This file contains functions to read/write tracks from/to files
'''
import pandas as pd
from .definition import track_fields

def read_track(track_file:str) -> pd.DataFrame:
    '''
    Read tracks from a txt file
    Inputs:
        track_file - a txt file contains tracks 
    Return:
        tracks - a DataFrame of tracks
    '''
    tracks = pd.read_csv(track_file, header=None)
    if len(tracks.columns) == len(track_fields):
        tracks.columns = track_fields
    else:
        raise Exception("The number of data fields in tracks is smaller than pre-defined!")
    return tracks

def write_track_header(tracks:pd.DataFrame, track_file:str) -> None:
    '''
    Write tracks to a csv file
    Inputs:
        tracks - a DataFrame of tracks
        track_file - a csv file contains tracks 
    '''
    tracks.to_csv(track_file, index=False)

if __name__ == '__main__':
    track_file = '/mnt/d/videos/sample/tracks/traffic_track.txt'
    tracks = read_track(track_file)
    print(tracks)