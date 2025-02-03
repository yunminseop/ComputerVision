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
import math

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
    def __init__(self, video_path, seg_model, det_model):
        self.video_path = video_path

        self.seg_model = seg_model
        self.det_model = det_model

        self.unit = Conversion(1920, 1080, 16.1)
        self.center = Centroid()
        self.car_position = np.array([320, 940])
        self.lookahead_distance = 60


        self.frame_skip = 5  # 5배속
        self.frame_count = 0

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
            
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                continue

            frame_resized = cv2.resize(frame, (640, 640))
            
            seg_results = self.seg_model.predict(frame_resized)
            det_results = self.det_model.predict(frame_resized)

            if det_results:
                    for det_result in det_results:
                    
                        classes = det_result.boxes.cls.cpu().numpy()
                        boxes = det_result.boxes.xyxy.cpu().numpy()  # (x_min, y_min, x_max, y_max)
                        
                        for cls_id in classes:
                            print(f"Object: {self.det_model.names[int(cls_id)]}")

            for seg_result in seg_results:
                    
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

                    annotated_frame = seg_results[0].plot(boxes=False)
                    
            processor = PurePursuit(self.destination, self.lookahead_distance)
            lookahead_distance, self.bezier_points = processor.get_bezier_points(self.car_position, (self.center.centroid_x, self.center.centroid_y))
            bezier_path = processor.bezier_curve(self.bezier_points)
            lookahead_point = processor.find_lookahead_point(bezier_path, self.car_position, lookahead_distance)
            slope = self.get_slope(lookahead_point)
            print(f"slope: {slope:.3f} deg")

            ax.clear()
            polygon = Polygon(self.destination, closed=True, edgecolor='r', facecolor='none', linewidth=1, label="Lane Edge")
            ax.add_patch(polygon)
            ax.plot(bezier_path[:, 0], bezier_path[:, 1], 'g-', label="Bezier Curve")
            ax.scatter(*zip(*self.bezier_points), color='blue', label="Control Points")
            ax.scatter(*lookahead_point, color='purple', s=100, label="Lookahead Point")
            # print(f"Lookahead Point: {lookahead_point}")                

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

    def get_slope(self, lookahead_point):
        x1, y1 = self.car_position
        x2, y2 = lookahead_point

        delta_y = y1 - y2 
        delta_x = x1 - x2

        desired_angle = math.atan2(delta_x, delta_y)

        vehicle_angle = 0 

        steering_angle = desired_angle - vehicle_angle

        steering_angle = math.degrees((steering_angle + math.pi) % (2 * math.pi) - math.pi)

        return steering_angle

class PurePursuit():
    def __init__(self, lane_polygon, lookahead_distance):
        self.lane_polygon = lane_polygon
        self.lookahead_distance = lookahead_distance

    def get_bezier_points(self, car_position, centroid):
        
        trans_polygon = self.lane_polygon.copy()
        dest = self.find_nearest_value(trans_polygon[:, 0], centroid[0])
        target_y = trans_polygon[trans_polygon[:, 0]==dest][0][1]

        dist = centroid[1] - target_y

        if dist > 0:
            trans_polygon[:, 1] += int(dist)
        else:
            trans_polygon[:, 1] -= int(dist)

        if 0 <= np.abs(dist) <= 100:
            """ dist가 100이면 lah_d+=50
                dist가 50이면 lah_d+=100
                즉, 무게중심과 edge간 거리가 가까워질수록 lah_d는 비례증가"""
            self.lookahead_distance += (100 - np.abs(dist))
        

        sort_index = np.argsort(trans_polygon[:, 1])
        y_max = trans_polygon[sort_index[0]]

        if dist < 100:
            """ car_position ~ centroid"""
            mid_control1 = (car_position[0]-(car_position[0] - centroid[0]) / 3, 1000 - ((car_position[1] - centroid[1]) * 5 / 10))
            mid_control2 = (car_position[0]-(car_position[0] - centroid[0]) * 2 / 3, 1000- ((car_position[1] - centroid[1]) * 8 / 10))

            """ centroid ~ y_max"""
            mid_control3 = (centroid[0]-(centroid[0] - y_max[0]) / 3, centroid[1] - ((centroid[1] - y_max[1]) * 5 / 10))
            mid_control4 = (centroid[0]-(centroid[0] - y_max[0]) * 2 / 3, centroid[1]- ((centroid[1] - y_max[1]) * 8 / 10))
            
            return (self.lookahead_distance, (car_position, mid_control1, mid_control2, mid_control3, mid_control4, y_max))
        else:
            """ car_position ~ centroid"""
            mid_control1 = (car_position[0]-(car_position[0] - centroid[0]) / 3, 1000 - ((car_position[1] - centroid[1]) * 5 / 10))
            mid_control2 = (car_position[0]-(car_position[0] - centroid[0]) * 2 / 3, 1000- ((car_position[1] - centroid[1]) * 8 / 10))
            return (self.lookahead_distance, (car_position, mid_control1, mid_control2, centroid)) 


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
    video_path = './src/ellipse_intersection/video_output6.mp4' #'/home/ms/ws/git_ws/ComputerVision/src/ellipse_intersection/video_output6.mp4'
    seg_path = '/home/ms/Downloads/best.pt'
    det_path = '/home/ms/Downloads/best_det.pt'

    seg_model = YOLO(seg_path, verbose=False)
    det_model = YOLO(det_path, verbose=False)

    processor = VideoProcessor(video_path, seg_model, det_model)
    processor.process_video()