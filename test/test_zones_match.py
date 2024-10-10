import os
import sys
import pandas as pd
from shapely.geometry import Polygon
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tssa.core.lane import LaneDetector
from dnt.label.labeler2 import Labeler
from label_zone import label_track_lane

zones = [Polygon([(352,45), (393,41), (409,191), (380,294), 
                 (177,652), (0,617), (208,374), (328,188)]
                ),
        Polygon([(393,41), (409,191), (380,294), (177,652),
                  (426,686), (505,349), (507,243), (486,202),
                  (458,96), (432,35)]
                ),
        Polygon([(432,35), (474,34), (616,206), (654, 261),
                 (705, 427), (742, 690), (528, 690), (556, 359),
                 (548, 254), (526, 203), (511, 172), (457, 97)]
                ),
        Polygon([(1192,313), (1277,393), (1158,504), (1111, 600),
                 (1094,700), (772,701), (877,511), (1028, 394)])
        ]
track_file = '/mnt/d/videos/sample/tracks/traffic_track.txt'
out_file = '/mnt/d/videos/sample/lanes/traffic_tracks_zones.csv'

lane_detector = LaneDetector(lanes=zones, ref_point='bc')
tracks = lane_detector.detect(track_file=track_file, out_file=out_file, mask_unmatched=True)

input_video = '/mnt/d/videos/sample/traffic.mp4'
out_video = '/mnt/d/videos/sample/labels/traffic_zones.mp4'
colors = [[255, 0, 0], [0, 255, 0], [130, 0, 75], [0, 127, 255]]
label_track_lane(input_video, out_file, zones, colors, out_video)

print('ok')