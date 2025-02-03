import cv2
import numpy as np
from ultralytics import YOLO
import torch 
import torch.nn as nn
import logging
import gc
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.special import binom

# YOLO verbose False 적용
logging.getLogger("ultralytics").setLevel(logging.ERROR)



"""
| AI Server |
VideoProcessor
Conversion
Centroid
GetIntersection
"""

"""
To get 'slope coef',
you should import pursuit and use VideoProcessor object cuz it returns slope parameters.
"""
class Conversion:
    def __init__(self, w_res, h_res, inch):
        self.__w_res = w_res
        self.__h_res = h_res
        self.__inch = inch

        self.__PPI = np.sqrt(np.power(self.__w_res, 2)+np.power(self.__h_res, 2))/self.__inch
        
        self.x = 0
        self.y = 0

    def p2cm(self):
        return  2.54 / self.__PPI

class VideoProcessor:
    def __init__(self, video_path, model):
        self.video_path = video_path
        self.model = model

        self.unit = Conversion(1920, 1080, 16.1)
        self.center = Centroid()
        self.car_position = np.array([320, 940])
        self.lookahead_distance = 60

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        

        if not cap.isOpened():
            print("Error: Couldn't open the video file.")
            exit()

        plt.ion()
        fig, ax = plt.subplots(figsize=(6,6))

        ax.set_xlim(0, 640)
        ax.set_ylim(1000, 0)

        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            
            ret, frame = cap.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            if not ret:
                print("Error: Couldn't read a frame.")
                break
            
            frame_resized = cv2.resize(frame, (640, 640))
            
            results = self.model.predict(frame_resized)
            
            for seg_result in results:
                    
                    classes = seg_result.boxes.cls.cpu().numpy() #cls_id = {0: center, 1: right, 2: left, 7: safety zone}
                    masks = seg_result.masks.xy if seg_result.masks else None

                    if masks != None:
                        prev_masks = masks
                        """ classes = [cls_id[0], cls_id[1], cls_id[2], ... ]
                            masks = [mask[0], mask[1], mask[2], ... ]
                                    mask[0] = [[20.2, 428.5], [167.0, 39.4], [420.7, 350.9], ...] (polygon)"""

                        for cls_id, mask in zip(classes, masks):
                            if cls_id == 0:
                                    # print(f"Class: {self.model.names[int(cls_id)]}")
                                    self.destination = mask
                                
                    elif prev_masks:
                                for cls_id, mask in zip(classes, prev_masks):
                                    if cls_id == 0:
                                        self.destination = mask


                    self.center.get_centroid(self.destination)                    

                    annotated_frame = results[0].plot(boxes=False)
                    
            processor = PurePursuit(self.destination)
            bezier_points = processor.get_bezier_points(self.car_position, (self.center.centroid_x, self.center.centroid_y))
            bezier_path = processor.bezier_curve(bezier_points)
            lookahead_point = processor.find_lookahead_point(bezier_path, self.car_position, self.lookahead_distance)
            
            ax.clear()
            polygon = Polygon(self.destination, closed=True, edgecolor='r', facecolor='none', linewidth=1, label="Lane Edge")
            ax.add_patch(polygon)
            ax.plot(bezier_path[:, 0], bezier_path[:, 1], 'g-', label="Bezier Curve")
            ax.scatter(*zip(*bezier_points), color='blue', label="Control Points")
            ax.scatter(*lookahead_point, color='purple', s=100, label="Lookahead Point")
            print(f"Lookahead Point: {lookahead_point}")                

            ax.invert_yaxis()
            ax.set_xlim(0, 640)
            ax.set_ylim(1000, 0)
            ax.set_aspect('equal')
            ax.legend()
            
            plt.draw()
            plt.pause(0.1)
            cv2.imshow('center_pursuit', annotated_frame)
            
            delay = int(50/ fps) # original fps
            
            del frame
            gc.collect()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

class PurePursuit():
    def __init__(self, lane_polygon):
        self.lane_polygon = lane_polygon

    def get_bezier_points(self, car_position, centroid):
        
        trans_polygon = self.lane_polygon.copy()
        dest = self.find_nearest_value(trans_polygon[:, 0], centroid[0])
        target_y = trans_polygon[trans_polygon[:, 0]==dest][0][1]

        dist = centroid[1] - target_y

        if dist > 0:
            trans_polygon[:, 1] += int(dist)
        else:
            trans_polygon[:, 1] -= int(dist)

        sort_index = np.argsort(trans_polygon[:, 1])
        y_max = trans_polygon[sort_index[0]]

        """ car_position ~ centroid"""
        mid_control1 = (car_position[0]-(car_position[0] - centroid[0]) / 3, 1000 - ((car_position[1] - centroid[1]) * 5 / 10))
        mid_control2 = (car_position[0]-(car_position[0] - centroid[0]) * 2 / 3, 1000- ((car_position[1] - centroid[1]) * 8 / 10))

        """ centroid ~ y_max"""
        mid_control3 = (centroid[0]-(centroid[0] - y_max[0]) / 3, centroid[1] - ((centroid[1] - y_max[1]) * 5 / 10))
        mid_control4 = (centroid[0]-(centroid[0] - y_max[0]) * 2 / 3, centroid[1]- ((centroid[1] - y_max[1]) * 8 / 10))
        
        return (car_position, mid_control1, mid_control2, mid_control3, mid_control4, y_max)


    def bezier_curve(self, bezier_points, num_points=100):
        n = len(bezier_points) - 1
        t_values = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))
    
        for i in range(n + 1):
            bernstein_poly = binom(n, i) * (t_values ** i) * ((1 - t_values) ** (n - i))
            curve += np.outer(bernstein_poly, bezier_points[i])
        
        return curve

    def find_lookahead_point(self, curve, current_pos, lookahead_distance):
        distances = np.linalg.norm(curve - current_pos, axis=1)
        idx = np.argmin(np.abs(distances - lookahead_distance))
        return curve[idx]

    def find_nearest_value(self, arr, value):
        idx = np.argmin(np.abs(arr - value))
        return arr[idx]


class Centroid():
    def __init__(self):
        self.centroid_x, self.centroid_y = 0, 0

    def get_centroid(self, polygon):
        area = 0
        self.centroid_x = 0
        self.centroid_y = 0
        n = len(polygon)

        for i in range(n):
            j = (i + 1) % n
            factor = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
            area += factor
            self.centroid_x += (polygon[i][0] + polygon[j][0]) * factor
            self.centroid_y += (polygon[i][1] + polygon[j][1]) * factor
        area /= 2.0
        if area != 0:
            self.centroid_x /= (6 * area)
            self.centroid_y /= (6 * area)


if __name__ == "__main__":
    video_path = '/home/ms/Downloads/video_output5.mp4' #'/home/ms/ws/git_ws/ComputerVision/src/ellipse_intersection/video_output6.mp4'
    checkpoint_path = '/home/ms/Downloads/best.pt'

    model = YOLO(checkpoint_path, verbose=False)

    processor = VideoProcessor(video_path, model)
    processor.process_video()