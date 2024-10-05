import os
import sys
import pandas as pd
from shapely.geometry import Polygon
from dnt.label.labeler2 import Labeler
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tssa.engine.zone import match_zones_by_point

zones = [Polygon([(352,45), (393,41), (409,191), (380,294), 
                 (177,652), (0,617), (208,374), (328,188)]
                ),
        Polygon([(393,41), (409,191), (380,294), (177,652),
                  (426,686), (505,349), (507,243), (486,202),
                  (458,96), (432,35)]
                )
        ]
tracks = pd.read_csv('/mnt/d/videos/sample/tracks/traffic_track.txt', header=None)

matched_tracks = match_zones_by_point(tracks, zones, ref_point='bc')
matched_tracks.to_csv('/mnt/d/videos/sample/lanes/traffic_tracks_zones.csv', index=False)


labeler = Labeler()
label_tracks = matched_tracks.loc[matched_tracks['zone']==1].copy()
label_tracks.drop(columns=['zone'], inplace=True)
labeler.draw_tracks(input_video='/mnt/d/videos/sample/traffic.mp4', 
                    tracks=label_tracks, 
                    output_video='/mnt/d/videos/sample/labels/traffic_zone_2.mp4')

print('ok')