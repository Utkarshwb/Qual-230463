from ultralytics import YOLO
# Load a model
model = YOLO("yolov8n.pt")
#use the model
results = model.train(data="config.yaml", epochs=3)