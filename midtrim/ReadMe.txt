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
    train Pytorch YoloV5 python
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