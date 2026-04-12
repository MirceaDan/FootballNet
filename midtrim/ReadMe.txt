Requirements:
1. Detection (2D)
    detect ball within image (pixels)

2. Depth / 3D
    transform pixels in real coordinates  (x,y,z)

3. Tracking + Trajectory
    bind points to time

4. Top-view map
    ground projection

Input: .avi video
Output: annotated .avi video

Strategy:
For 1:
    use yolo11n.onnx from https://github.com/ultralytics/assets/releases
    ONNX inference C++

For 2:
𝑍=𝑓⋅𝐷/𝑑 
where:
Z = distance to ball
𝑓 = focal length (from camera)
𝐷 = real ball diameter (~0.22 m)
𝑑 = diameter in pixels (from within bounding box)

Then 3D coordinates:
𝑋=(𝑢−𝑐𝑥)⋅𝑍/𝑓,
𝑌=(𝑣−𝑐𝑦)⋅𝑍/𝑓
Where:
(u, v) = pixel
(cx, cy) = image center
Output: X, Y, Z per frame

For 3:
    Kalman Filter

For 4:
    ignore height -> proiect straight to ground
Simplified:
Top-view = (X, Z)
X → lateral
Z → distance to camera
So:
    map_x = X
    map_y = Z

assumed a focal length of 800 pixels and measured a 2008 footbal at 0.22m
static camera perspective set from player 1 on a rectangular plane with ball moving back & forth
visual depiciton of scene:
        ---------------------------
        |                         |
player1 |       o                 | player 2
        |                         | 
camera  ---------------------------

Prerequisits:
dataset: https://www.kaggle.com/datasets/mdkabinhasan/sports-ball-dataset
dataset structure simplified:
# dataset/
#    train/
#    labels/

Inference code dependencies:
Opencv

how to setup Opencv for C++ Linux:
sudo apt get update
sudo apt install python3-Opencv

Build commands:
Release:
cmake -S . -B build
Debug:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
this should create the build folder and place there all the build related files
and then 
cmake --build build

expected build output:
mircea@raspberrypi:~/Desktop/FootballNet/midtrim $ cmake --build build
[ 50%] Building CXX object CMakeFiles/footballnet.dir/inference.cpp.o
[100%] Linking CXX executable footballnet
[100%] Built target footballnet

Observations:
If the model is incorrect one might run into this issue:
[ERROR:0@2.079] global onnx_importer.cpp:1035 handleNode DNN/ONNX: ERROR during processing node with 2 inputs and 1 outputs: [Concat]:(onnx_node!/model.12/Concat) from domain='ai.onnx'
terminate called after throwing an instance of 'cv::Exception'
  what():  OpenCV(4.10.0) ./modules/dnn/src/onnx/onnx_importer.cpp:1057: error: (-2:Unspecified error) in function 'handleNode'
> Node [Concat@ai.onnx]:(onnx_node!/model.12/Concat) parse error: OpenCV(4.10.0) ./modules/dnn/src/layers/concat_layer.cpp:108: error: (-201:Incorrect size of input array) Inconsistent shape for ConcatLayer in function 'getMemoryShapes'
> 
Aborted

This is because the yolo model might've been exported with dynamic input shapes or OpenCV ARM does not allow for dynamic shapes in which case the onnx model has to be reexported with the flag set to false
if your ultralytics env is broken (highly likely as latest release is unstable) then go to:
https://colab.research.google.com
run these commands:
!pip install ultralytics

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)

and then downlaod the updated model:
from google.colab import files
files.download("yolo11n.onnx")
