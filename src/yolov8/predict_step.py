import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

video_path = '/home/ms/Videos/Screencasts/gangbuk.webm' 

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("open failed")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.7, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow('yolov8 lane detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

