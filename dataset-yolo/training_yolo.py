from ultralytics import YOLO

model = YOLO("yolo11n.pt")  

results = model.train(
    data="data.yaml",  # path to dataset YAML file
    epochs=100,                     # number of training epochs
    imgsz=640                       # training image size
)       