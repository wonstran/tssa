"""
This module contains the annotation class that is used to display results on videos.
"""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import magic

class Annotation:
    """
    A class used to display results on videos
    """
    def __init__(self, input_media:str):
        self.input_video = input_media
        cap = cv2.VideoCapture(self.input_media)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.draws = pd.DataFrame(columns=['frame',
                                          'type',
                                          'coords',
                                          'size',
                                          'desc',
                                          'color',
                                          'thick',
                                          'alpha'])
        
    def load_draws(self, draw_file:str):
        """
        Load the draws from a csv file\n
        Inputs:\n
            draw_file: a csv file containing the draws
        """
        self.draws = pd.read_csv(draw_file,
                                dtype={'frame':int,
                                    'type':str,
                                    'size':float,
                                    'desc':str,
                                    'thick':int,
                                    'aplha':float},
                                converters={'coords': lambda x:list(eval(x)),
                                            'color': lambda x:eval(x)})

    def draw(self,
             output_media:str,
             start_frame:int=None,
             end_frame:int=None,
             verbose:bool=True):
        '''
        Draw objects on a video\n
        Inputs:\n
            output_media: file to save the output video
            start_frame: the first frame to label
            end_frame:  the last frame to label
            verbose: if True, display progress bar
        '''
        if self.draws is not None:
            cap = cv2.VideoCapture(self.input_video)
            if not cap.isOpened():
                raise IOError("Couldn't open webcam or video")

            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = int(self.frame_num)-1

            tot_frames = end_frame - start_frame + 1
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            writer = cv2.VideoWriter(output_media,
                                     fourcc,
                                     self.fps,
                                     (self.width, self.height)
                                    )

        if verbose:
            pbar = tqdm(total=tot_frames,
                        unit=" frames",
                        desc="Generating annotations ...")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if (not ret) or (pos_frame>end_frame):
                break

            elements = self.draws.loc[self.draws['frame']==pos_frame]
            for _, element in elements.iterrows():
                if element['type'] == 'text':
                    label_txt = element['desc']
                    color = element['color']
                    size = element['size']
                    thick = element['thick']
                    cv2.putText(frame, label_txt,
                                tuple(map(int, element['coords'][0])),
                                0,
                                size,
                                color,
                                thick)

                elif element['type'] == 'line':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.line(frame,
                             tuple(map(int, coords[0])),
                             tuple(map(int, coords[1])),
                             color,
                             thick)

                elif element['type'] == 'box':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.rectangle(frame,
                                  tuple(map(int, coords[0])),
                                  tuple(map(int, coords[1])),
                                  color,
                                  thick)

                elif element['type'] == 'bbox':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    label_txt = element['desc']
                    size = element['size']
                    cv2.rectangle(frame,
                                  tuple(map(int, coords[0])),
                                  tuple(map(int, coords[1])),
                                  color,
                                  thick)
                    cv2.putText(frame,
                                str(label_txt),
                                (int(coords[0][0]),
                                 int(coords[0][1]-int(10*size))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                size,
                                color,
                                thick)

                elif element['type'] == 'circle':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    label_txt = element['desc']
                    radius = int(element['size'])
                    cv2.circle(frame,
                               coords,
                               radius=radius,
                               color=color,
                               thickness=thick)

                elif element['type'] == 'polygon':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.polylines(frame,
                                  [np.array(coords)],
                                  isClosed=True,
                                  color=color,
                                  thickness=thick)

                elif element['type'] == 'polylines':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.polylines(frame,
                                  [np.array(coords)],
                                  isClosed=False,
                                  color=color,
                                  thickness=thick)

                elif element['type'] == 'filled_polygon':
                    coords = element['coords']
                    color = element['color']
                    alpha = element['alpha']
                    overlay = frame.copy()
                    cv2.fillPoly(overlay,
                                 [np.array(coords)],
                                 color)
                    frame = cv2.addWeighted(overlay,
                                            alpha,
                                            frame,
                                            1 - alpha,
                                            0)
            writer.write(frame)
            if verbose:
                pbar.update()

        if verbose:
            pbar.close()
        cap.release()
        writer.release()

    def add_circle(self,
                   frame:int,
                   coords:tuple,
                   color:tuple,
                   radius:float,
                   thick:int=1
                   ):
        '''
        Add a circle to the draws\n
        Inputs:\n
            frame: the frame number
            coords: the center of the circle, (x,y)
            radius: the radius of the circle
            color: the color of the circle
            thick: the thickness of the circle, default is 1
        '''
        self.draws = self.draws.append({'frame':frame,
                                        'type':'circle',
                                        'coords':coords,
                                        'size':radius,
                                        'color':color,
                                        'thick':thick,
                                        'alpha':1,
                                        'desc':''},
                                       ignore_index=True)
    
    def add_filled_circle(self,
                   frame:int,
                   coords:tuple,
                   color:tuple,
                   radius:float,
                   thick:int=1
                   ):
        '''
        Add a circle to the draws\n
        Inputs:\n
            frame: the frame number
            coords: the center of the circle, (x,y)
            radius: the radius of the circle
            color: the color of the circle
        '''
        self.draws = self.draws.append({'frame':frame,
                                        'type':'circle',
                                        'coords':coords,
                                        'size':radius,
                                        'color':color,
                                        'thick':-1,
                                        'alpha':1,
                                        'desc':''},
                                       ignore_index=True)
    
    @staticmethod    
    def check_file_type_by_content(file_name:str):
        mime = magic.Magic(mime=True)      
        mime_type = mime.from_file(file_name)
        
        if mime_type.startswith('image'):
            return "image"
        elif mime_type.startswith('video'):
            return "video"
        else:
            return "unknown"

if __name__ == "__main__":
    pass
