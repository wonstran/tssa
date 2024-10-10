from shapely.geometry import Polygon
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

def draw(input_image:str,
         output_image:str,
         zones:list[Polygon],
         selected:list[int]=None,
         fill:bool=False,
         label:bool=False):
        
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        img = cv2.imread(input_image)
        if selected is None:
            selected = range(len(zones))
        for i in selected:
            zone = zones[i]
            final_color = colors[i]

            cv2.polylines(img, 
                          [np.array(list(zone.exterior.coords), np.int32).reshape(-1, 1, 2)],
                           isClosed=True, 
                           color=(0,0,0), 
                           thickness=1)
            if fill:
                overlay = img.copy()
                cv2.fillPoly(overlay, 
                             [np.array(list(zone.exterior.coords), np.int32).reshape(-1, 1, 2)],
                             color=final_color)
                alpha = 0.2
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            
            if label:
                cv2.putText(img, str(i+1), (int(zone.centroid.x), int(zone.centroid.y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        cv2.imwrite(output_image, img)

if __name__ == '__main__':
    input_image = '/mnt/d/videos/sample/frames/0180-cctv.png'
    output_image = '/mnt/d/videos/sample/frames/0180-cctv_lanes.png'
    '''lanes = [
        Polygon([(403, 716), (477, 678), (1253, 1016), (1123, 1078)]),
        Polygon([(481, 677), (535, 652), (1359, 957), (1253, 1016), (1253, 1016)]),
        Polygon([(538, 651), (577, 634), (1430, 909), (1363, 955)]),
        Polygon([(575, 634), (627, 610), (1530, 836), (1430, 907)])
        ]
    '''
    lanes = [
    Polygon(np.array([[403, 716], [477, 678], [1253, 1016], [1123, 1078]])),
    Polygon(np.array([[481, 677], [535, 652], [1359, 957], [1253, 1016], [1253, 1016]])),
    Polygon(np.array([[538, 651], [577, 634], [1430, 909], [1363, 955]])),
    Polygon(np.array([[575, 634], [627, 610], [1530, 836], [1430, 907]]))]

    draw(input_image, output_image, lanes, fill=True, label=True)
    print('ok')