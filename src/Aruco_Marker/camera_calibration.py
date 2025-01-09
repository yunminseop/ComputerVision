import cv2
import numpy as np

# 체스보드 크기 설정 (체스보드의 내부 코너 수)
chessboard_size = (9, 6)  # 9x6 체스보드의 코너 수

# 체스보드 패턴의 3D 좌표 준비
# 체스보드의 각 코너는 3D 공간에서 (0, 0, 0), (1, 0, 0), (2, 0, 0), ... 형태로 가정
obj_p = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
obj_p[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)

# 객체 포인트와 이미지 포인트 리스트 준비
obj_points = []  # 실제 3D 좌표
img_points = []  # 이미지에서 찾은 2D 좌표

# 카메라 캡처
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size)

    if ret:
        # 코너 찾았을 때, 2D 좌표를 이미지 포인트에 저장
        img_points.append(corners)

        # 3D 좌표를 객체 포인트에 저장
        obj_points.append(obj_p)

        # 인식된 코너를 이미지에 표시
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

    cv2.imshow('Chessboard', frame)
    print(f"img_points: {img_points}")
    print(f"obj_points: {obj_points}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 카메라 캘리브레이션 수행
# opencv 실행 프로그램 종료 후 프로그램 강제종료 X, 카메라 행렬 계산 중임.
try:
    print("try to calibrate...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None)
except:
    print("calibration failed")

# 카메라 매트릭스와 왜곡 계수 출력
print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)
