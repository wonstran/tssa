from dnt.detect import Detector
from dnt.track import Tracker, Config
from dnt.label.labeler2 import Labeler

input_video = '/mnt/d/videos/sample/0180-cctv.mp4'
det_file = '/mnt/d/videos/sample/dets/0180-cctv_det.txt'
track_file = '/mnt/d/videos/sample/tracks/0180-cctv_track.txt'
output_video = '/mnt/d/videos/sample/labels/0180-cctv_track.mp4'

detector = Detector(weights='l')
detector.detect(input_video, det_file)

cfg = Config.get_cfg_dsort()
cfg['max_age'] = 30
cfg['nn_budget'] = 100
cfg['nms_max_overlap'] = 1
print(cfg)
tracker = Tracker(cfg)
tracker.track(det_file, track_file, video_file=input_video)

labeler = Labeler()
labeler.draw_tracks(input_video=input_video, track_file = track_file, output_video=output_video)