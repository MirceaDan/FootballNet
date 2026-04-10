basic requirements:
1. Detection (2D)
    detect ball within image (pixels)

2. Depth / 3D
    transform pixels in real coordinates  (x,y,z)

3. Tracking + Trajectory
    bind points to time

4. Top-view map
    ground projection

Input: .avi video
Output: annotated output_rgb.avi video, trajectory.csv

Strategy:
Distance (Z): 
Used the similarity relation of traingles within the projected perspective: Z = f * Wreal/wimage
where f of focal distanve
Wreal is ball diameter (0.22m)
wimage is detected iamge width in pixels

Position X (3D): Computed through: X = (u - c_x) * Z/f
where u is center in pixels
cx camera optical center

Kalman Filter: used for Trajectory stabilisation. 

Top-View Map: a representation of the plan X-Z. Camera located at base of image (bottom center), while the ball moves "in depth" on the Z axis.

assumed a focal length of 800 pixels and measured a 2008 footbal at 0.22m
static camera perspective set from player 1 on a rectangular plane with ball moving back & forth
visual depiciton of scene:
        ---------------------------
        |                         |
player1 |       o                 | player 2
        |                         | 
camera  ---------------------------

Prerequisits:
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

how to run:
[from within entrytrim folder]  ./build/footballnet 
