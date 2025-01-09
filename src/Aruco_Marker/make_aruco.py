import cv2
import cv2.aruco as aruco

# ArUco 사전 선택 (DICT_6X6_250은 6x6 크기의 마커를 250개 생성할 수 있음)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# ID 42의 ArUco 마커 생성
marker_id = 50
marker_size = 200  # 마커 이미지 크기 (픽셀 단위)

# 마커 생성
marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# 마커 이미지 보기
cv2.imshow("aruco Marker", marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 파일로 저장
cv2.imwrite("./data/aruco_marker_50.png", marker_image)