from ultralytics import YOLO

# Load the Triton Server model
model = YOLO(f"http://localhost:8000/yolov8n", task="detect")

# Run inference on the server
results = model(r"C:\Users\autumn\Desktop\yolo\coco128\images\train2017")