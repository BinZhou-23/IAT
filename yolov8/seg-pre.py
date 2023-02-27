from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n-seg.pt")  # load an official model
model = YOLO("E:/yolov8/runs/segment/train3/weights/best.pt")  # load a custom model

# Predict with the model
results = model("E:/yolov8/datasets/test.v1/test/images/")  # predict on an image