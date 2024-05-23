from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model

# Export the model
# ONNX ===> imgsz, half, dynamic, simplify, opset, batch
# TensorRT ===> imgsz, half, dynamic, simplify, workspace, int8, batch
model.export(format='onnx',
             imgsz=(640, 640),
             half=True,
             dynamic=False,
             simplify=True,
             batch=1)

# model.export(format='engine',
#              imgsz=(640, 1024),
#              half=False,
#              dynamic=False,
#              simplify=True,
#              workspace=4.0,
#              int8=True,
#              batch=1)


# docker run --gpus all --name tensorrt -itd --network=host -v D:/dev/code/GitHub/yolov8/tensorrt_triton:/workspace/tensorrt_triton -p 9000:9000 --name tensorrt nvcr.io/nvidia/tensorrt:23.09-py3
# docker exec -it tensorrt /bin/bash
# pip install tensorflow onnx keras2onnx tf2onnx scikit-learn scikit-image torch torchvision jupyter notebook

# --explicitBatch
# ./trtexec --onnx=/workspace/tensorrt_triton/yolov8n.onnx --saveEngine=/workspace/tensorrt_triton/model.plan --useCudaGraph --fp16

# docker run --rm --gpus all --name tritonserver -p 8000:8000 -p 8001:8001 -p 8002:8002 -v D:/autumn/Documents/JetBrainsProjects/PyCharm/yolov8/models:/models nvcr.io/nvidia/tritonserver:23.09-py3 tritonserver --model-repository=/models