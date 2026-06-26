from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

# Export to TFLite with INT8 quantization
model.export(format="tflite", imgsz=320, int8=True)