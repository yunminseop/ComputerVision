import cv2
import cv2.aruco as aruco
import numpy as np

# aruco 사전 선택
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# 마커의 실제 크기 (unit: m)
marker_length = 0.1  # 10cm

# 카메라 행렬 및 왜곡 계수
# camera_calibration.py 활용하여 구하기
camera_matrix = np.array([[721, 0, 311],
                        [0, 716, 207],
                        [0, 0, 1]], dtype=np.float32)

# 왜곡 계수
dist_coeffs = np.array([[-7.08957266e-01,  4.57196343e+00,  3.76320952e-02,  5.78087141e-03, -2.83497356e+01]])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # aruco 마커 감지
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # 감지된 마커의 자세 추정
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            # 축 그리기
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            # 자세 출력 (회전 벡터와 변환 벡터)
            print("회전 벡터 (rvec, nx3):", rvec)
            print("변환 벡터 (tvec, nx3):", tvec)

    cv2.imshow("aruco Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
