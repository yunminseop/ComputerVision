import cv2
import numpy as np

class Conversion:
    def __init__(self, w_res, h_res, inch):
        self.__w_res = w_res
        self.__h_res = h_res
        self.__inch = inch

        self.__PPI = np.sqrt(np.pow(self.__w_res, 2)+np.pow(self.__h_res, 2))/self.__inch

    def p2cm(self):
        return  2.54 / self.__PPI

                        
class GetIntersection:
    def __init__(self, ellipse_center, ellipse_axes):
        self.h, self.k = ellipse_center  # 타원의 중심 (h, k)
        self.a, self.b = ellipse_axes   # 타원의 반지름 (a, b)
        self.fixed_point = (320, 640)  # 반드시 지나는 점 (x0, y0)

    def set_dynamic_line(self, target_point):
        """주어진 좌표와 (320, 640)을 지나는 직선 설정."""
        x0, y0 = self.fixed_point
        x1, y1 = target_point

        # 기울기 계산 (m)
        self.m = (y1 - y0) / (x1 - x0) if x1 != x0 else float('inf')  # x1 == x0일 경우 수직선
        
        # y절편 계산 (c)
        self.c = y0 - self.m * x0

    def change_derivative(self):
        self.m = 2
        print(self.m)

    def calculate_intersection(self):
        """타원과 직선의 교점을 계산합니다."""
        A = (1 / self.a**2) + (self.m**2 / self.b**2)
        B = (2 * self.m * (self.c - self.k) / self.b**2) - (2 * self.h / self.a**2)
        C = ((self.h**2) / self.a**2) + ((self.c - self.k)**2 / self.b**2) - 1

        # 판별식
        discriminant = B**2 - 4 * A * C
        # print(f"discriminant = {discriminant}")

        if discriminant < 0:
            return None  # 교점 없음

        x1 = (-B + np.sqrt(discriminant)) / (2 * A)
        x2 = (-B - np.sqrt(discriminant)) / (2 * A)

        y1 = self.m * x1 + self.c
        y2 = self.m * x2 + self.c

        if x1 > 0 and y1 > 0 or x1 < 0 and y1 > 0 :
            self.upper_point = (x1, y1)
        elif x2 > 0 and y2 > 0 or x2 < 0 and y2 > 0:
            self.upper_point = (x2, y2)

        
        return [[x1, y1], [x2, y2]]

    def filter_valid_points(self, points, width, height):
        """유효한 화면 안의 교점만 필터링"""
        valid_points = [
            (int(x), int(y)) for x, y in points
            if 0 <= x <= width and 0 <= y <= height
        ]
        # print(valid_points)
        return valid_points


## main ##
video_path = './data/video_output6.mp4'

unit = Conversion(1920, 1080, 16.1)

ellipse_center = (320, 640)  # 타원의 중심
ellipse_axes = (320, 100)    # 타원의 반지름 (a, b)

cap = cv2.VideoCapture(video_path)
standard_point = (320, 540)
pursuit_point = (500, 540)

if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

intersection_finder = GetIntersection(ellipse_center, ellipse_axes)

while True:
    target_point_x = np.random.randint(640)
    target_point_y = np.random.randint(640)
    dynamic_point = (target_point_x, target_point_y)
    intersection_finder.set_dynamic_line(dynamic_point)

    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    if not ret:
        print("Error: Couldn't read a frame.")
        break
    
    frame_resized = cv2.resize(frame, (640, 640))
    
    cv2.ellipse(frame_resized, ellipse_center, ellipse_axes, 0, 0, 360, (0, 0, 0), 2)

    cv2.circle(frame_resized, standard_point, 5, (0, 0, 255), -1, cv2.LINE_AA)

    points = intersection_finder.calculate_intersection()
    if points:
        valid_points = intersection_finder.filter_valid_points(points, 640, 640)
        for point in valid_points:
            cv2.circle(frame_resized, point, 5, (0, 255, 0), -1, cv2.LINE_AA)  # 교점에 원 표시
            # print(f"x: {(640-point[0])*(unit.p2cm()):.2f}, y: {(640-point[1])*unit.p2cm():.2f}")
        
        if (640-point[0])*(unit.p2cm()) < 320*unit.p2cm():
            x = -(320 - point[0])*unit.p2cm()
        else:
            x = (point[0] - 320)*unit.p2cm()

        y = (640-point[1])*unit.p2cm()

        print(f"x: {x:.2f}, y: {y:.2f}")
    slope = np.arctan(x/(0.396*y+6.3)) if point[1] != 0 else np.arctan(x)
            
    slope = np.degrees(slope)
    print(f"조향각: {slope:.2f}'")
    
    cv2.imshow('center_pursuit', frame_resized)
    
    delay = int(1000 / fps) # original fps
    
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# 영상 파일 해제
cap.release()

# 모든 윈도우 창 종료
cv2.destroyAllWindows()