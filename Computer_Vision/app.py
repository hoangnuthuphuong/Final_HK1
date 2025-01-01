from ultralytics import YOLO


model = YOLO('yolov8n.pt')

result = model.train(data='datasets/data.yaml', epochs=10, imgsz=640, lr0=0.005)
