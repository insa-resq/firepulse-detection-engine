from ultralytics import YOLO

model = YOLO("yolo11n.pt")  

results = model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    device=0,
    batch=8,
    patience=10,
    save=True,
    project="runs/detect",
    name="train"
)       