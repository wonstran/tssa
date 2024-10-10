import os
import sys
import numpy as np
from shapely.geometry import Polygon, LineString
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tssa.core.lane import LaneDetector
from dnt.label.labeler2 import Labeler
from label_zone import label_track_lane

lanes = [
    Polygon(np.array([[403, 716], [477, 678], [1253, 1016], [1123, 1078]])),
    Polygon(np.array([[481, 677], [535, 652], [1359, 957], [1253, 1016], [1253, 1016]])),
    Polygon(np.array([[538, 651], [577, 634], [1430, 909], [1363, 955]])),
    Polygon(np.array([[575, 634], [627, 610], [1530, 836], [1430, 907]])),
    Polygon(np.array([[660, 593], [1541, 817], [1610, 765], [689, 569]]))]

ref_line = LineString(np.array([[730, 1003], [1237, 732]]))

track_file = '/mnt/d/videos/sample/tracks/0180-cctv_track.txt'
out_file = '/mnt/d/videos/sample/lanes/0180-cctv_track_zones.csv'

lane_detector = LaneDetector(lanes=lanes, ref_point='br', ref_offset=(-3,0))
tracks = lane_detector.detect(track_file=track_file, out_file=out_file, mask_unmatched=True)

input_video = '/mnt/d/videos/sample/0180-cctv.mp4'
out_video = '/mnt/d/videos/sample/labels/1080-cctv_zones.mp4'
colors = [[255, 0, 0], [0, 255, 0], [130, 0, 75], [0, 127, 255], [128, 0, 128]]
label_track_lane(input_video, out_file, lanes, colors, out_video)

print('ok')