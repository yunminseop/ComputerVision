from ultralytics import YOLO


model = YOLO('yolov8n.pt')

model.train(
    data='/home/ms/my_yolo_data/data.yaml',
    epochs=50,                
    imgsz=640,               
    batch=16,
    conf_thres=0.7,
)
