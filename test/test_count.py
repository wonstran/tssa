import os
import sys
import pandas as pd
from shapely.geometry import LineString
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tssa.count.count import Count

track_file ='/mnt/d/videos/ped2stage/05032024/wb_20240416/tracks/wb_20240416_cross_ped_track.txt'
tracks = pd.read_csv(track_file)
ped_line = LineString([(9, 174), (858, 174)])
ped_count_file = 'ped_count.csv'

counter = Count(line=ped_line)
ped_count = counter.count_tracks_by_line(tracks=tracks)
ped_count.to_csv(ped_count_file, index=False)
