import cv2
import cv2.aruco as aruco

# aruco 사전 선택
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# 비디오 캡처 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 흑백 변환 (aruco 감지는 흑백 이미지를 사용)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # aruco 마커 감지
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # 감지된 마커 그리기
    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # aruco 마커 ID 출력
        for i, marker_id in enumerate(ids.flatten()):
            print(f"Detected Marker ID: {marker_id}")

    # 영상 출력
    cv2.imshow("aruco Marker Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
