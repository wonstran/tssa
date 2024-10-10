from shapely.geometry import Polygon
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

def draw(input_video:str,
         output_video:str,
         draws:pd.DataFrame = None,
         draw_file:str = None,
         start_frame:int=None,
         end_frame:int=None,
         video_index:int=None,
         video_tot:int=None,
         verbose:bool=True):
        '''
        Draw labled objects on a video\n
        Inputs:\n
                draws: a DataFrame contains labeling information, if None, read label_file
                label_file: a txt file with a header ['frame','type','coords','color','size','thick','alpha','desc']
                input_video: raw video
                output_video: labeled video
                start_frame: the first frame to label
                end_frame:  the last frame to label
                video_index: display video index in a batch processing
                video_tot: display total video number in a batch processing
        '''
        compress_message = False
        if draws is not None:
            data = draws
        else:
            data = pd.read_csv(draw_file, 
                               dtype={'frame':int,
                                      'type':str,
                                      'size':float,
                                      'desc':str,
                                      'thick':int,
                                      'aplha':float},
                               converters={'coords': lambda x:list(eval(x)),
                                           'color': lambda x:eval(x)})

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

        tot_frames = end_frame - start_frame + 1       
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        if verbose:
            pbar = tqdm(total=tot_frames, unit=" frames")
            if compress_message:
                pbar.set_description_str("Labeling")
            else:
                if video_index and video_tot:
                    pbar.set_description_str("Labeling {} of {}".format(video_index, video_tot))
                else:
                    pbar.set_description_str("Labeling {} ".format(input_video))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if (not ret) or (pos_frame>end_frame):
                break
            
            elements = data.loc[data['frame']==pos_frame]

            for index, element in elements.iterrows():
                if element['type'] == 'text':
                    label_txt = element['desc']
                    color = element['color']
                    size = element['size']
                    thick = element['thick']
                    cv2.putText(frame, label_txt, tuple(map(int, element['coords'][0])), 0, size, color, thick)
                
                elif element['type'] == 'line':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.line(frame, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, thick)

                elif element['type'] == 'box':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.rectangle(frame, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, thick)
                
                elif element['type'] == 'bbox':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    label_txt = element['desc']
                    size = element['size'] 
                    cv2.rectangle(frame, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, thick)
                    cv2.putText(frame, str(label_txt), (int(coords[0][0]), int(coords[0][1]-int(10*size))), 
                            cv2.FONT_HERSHEY_SIMPLEX, size, color, thick)
                    
                elif element['type'] == 'circle':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    label_txt = element['desc']
                    radius = int(element['size']) 
                    cv2.circle(frame, tuple(map(int, coords[0])), radius=radius, color=color,  thickness=thick)
                
                elif element['type'] == 'polygon':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.polylines(frame, [np.array(coords)], isClosed=True, color=color, thickness=thick)

                elif element['type'] == 'polylines':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.polylines(frame, [np.array(coords)], isClosed=False, color=color, thickness=thick)

                elif element['type'] == 'filled_polygon':
                    coords = element['coords']
                    color = element['color']
                    alpha = element['alpha']
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [np.array(coords)], color)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            writer.write(frame)
            if verbose:
                pbar.update()

        if verbose:
            pbar.close()
        cap.release()
        writer.release()      

def label_track_lane(input_video:str,
                     track_file:str,
                     lanes:list[Polygon],
                     colors:list[list[int]],
                     output_video:str,
                     label_ref_point:bool=True,
                     label_lanes:bool=True):
        
    tracks = pd.read_csv(track_file)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")

    pbar = tqdm(total=len(tracks), unit=" frames", desc="Generating labels ...")
    results = []
    for index, frame in tracks.iterrows():
        final_color = colors[int(frame['lane'])]
        label_str = '#'+str(int(frame['track_id']))
        results.append([frame['frame'],
                        'bbox',
                        [(frame['x'], frame['y']),
                         (frame['x'] + frame['w'], 
                          frame['y'] + frame['h'])],
                        final_color, 
                        0.5,
                        1,
                        1.0,
                        label_str])
        if label_ref_point:
            results.append([frame['frame'],
                        'circle',
                        [(frame['ref_x'], frame['ref_y'])],
                        [0, 0, 255], 
                        3,
                        0,
                        1.0,
                        ""])
        if label_lanes:
            for i in range(len(lanes)):
                lane = lanes[i]
                '''results.append([frame['frame'], 
                            'filled_polygon',
                            np.array(list(lane.exterior.coords), np.int32).reshape((-1, 1, 2)),
                            [218, 177, 218],
                            1,
                            1,
                            0.1, 
                            ""])'''
                results.append([frame['frame'], 
                            'text',
                            [(int(lane.centroid.x), int(lane.centroid.y))], 
                            colors[i],
                            1,
                            2,
                            1,
                            str(i+1)])
                results.append([frame['frame'], 
                            'polygon',
                            np.array(list(lane.exterior.coords), np.int32).reshape((-1, 1, 2)),
                            [0, 0, 0],
                            1,
                            1,
                            1,
                            ""])
        pbar.update()
    pbar.close()

    df = pd.DataFrame(results, columns=['frame','type','coords','color','size','thick','alpha','desc'])
    df.sort_values(by='frame', inplace=True)
    df.to_csv(output_video.replace('.mp4', '.csv'), index=False)
    draw(input_video = input_video,
         output_video = output_video,
         draws = df)

