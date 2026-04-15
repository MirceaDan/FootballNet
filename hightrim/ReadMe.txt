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
Output: annotated .avi video & trajectory .csv

Strategy:
For 1: 
    Detection (Hybrid: NN + Motion + Filtering) 
    check moca_bg_det.pdf for mathematical formulation

Model outputs:
P_hat → attention heatmap (where ball likely is)
M → motion map (what is moving)
xyz_pred → coarse 3D estimate

In C++:
Combine signals:
fusion = P_hat * M

Extract max location → (u, v)
Apply:
thresholding
temporal gating (reject large jumps)
EMA smoothing

Final result = stable 2D ball position

For 2:
Depth is taken from model: Z = xyz_pred[2]
Then recompute X,Y using projection:
X = (u - cx) * Z / f
Y = (v - cy) * Z / f
Where:

(u, v) = detected pixel
(cx, cy) = image center
f = focal length

For 3:
Combination of:
Kalman Filter (state: position + velocity)
Prediction fallback (when detection missing)
Outlier rejection (max jump constraint)
EMA smoothing (remove jitter)

For 4:
Ignore height (Y), keep ground plane (X & Y)

visual depiciton of scene:
        ---------------------------
        |                         |
player1 |       o                 | player 2
        |                         | 
camera  ---------------------------

Prerequisits:
rgb.avi is the input data needed to run

Inference code dependencies:
Torch 2.5.1
Opencv

how to generate libtorch for aarch64 (brace for 10 hour compilation)
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.5.1
#pytorch commit: a8d6afb511a69687bbb2b7e88a3cf67917e1697e
git submodule sync
git submodule update --init --recursive

export BUILD_SHARED_LIBS=ON
export USE_NNPACK=OFF
export USE_QNNPACK=ON
export USE_PYTORCH_QNNPACK=ON
export USE_XNNPACK=ON
export BUILD_MOBILE_AUTOGRAD=OFF
export BUILD_TEST=OFF
export BUILD_BINARY=OFF
export BUILD_PYTHON=OFF
export USE_OPENMP=ON
export USE_MKLDNN=OFF
export USE_FBGEMM=OFF
export USE_CUDA=OFF
export USE_ROCM=OFF
export USE_METAL=OFF

cmake -S . -B build \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/libtorch

cmake --build build -j1
cmake --install build

#go to /opt
cp -r libtorch /home/mircea/Desktop/FootballNet/midtrim/

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
cd hightrim
./build/footballnet
should output output_moca.avi and trajectory.csv
